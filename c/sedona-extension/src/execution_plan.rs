// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::{
    any::Any,
    ffi::{c_int, c_void},
    fmt::{Debug, Display, Formatter},
    ptr::null_mut,
    sync::Arc,
    time::Duration,
};

use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use arrow_schema::{ffi::FFI_ArrowSchema, Schema, SchemaRef};
use datafusion_common::{exec_err, Result, Statistics};
use datafusion_execution::TaskContext;
use datafusion_physical_plan::{
    execution_plan::{Boundedness, CardinalityEffect, EmissionType},
    metrics::{CustomMetricValue, Metric, MetricValue, MetricsSet},
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
};
use sedona_common::{sedona_internal_datafusion_err, sedona_internal_err};
use serde::{Deserialize, Serialize};
use tokio::runtime::Runtime;

use crate::extension::{SedonaCError, SedonaCExecutionPlan, SedonaCExecutionPlanArgs};
use crate::set_ffi_error;
use crate::streaming::{ffi_stream_to_sendable, CancelChecker, StreamingRecordBatchReader};
use crate::utils::{cstr_from_ptr_or_empty, get_plan_property, get_plan_string_property, ERRNO_OK};

/// Wrapper around an [ExecutionPlan] that can be exported across FFI.
///
/// Holds an `Arc<Runtime>` to ensure the runtime stays alive for the lifetime
/// of the exported plan.
pub struct ExportedExecutionPlan {
    plan: Arc<dyn ExecutionPlan>,
    task_context: Arc<TaskContext>,
    runtime: Arc<Runtime>,
}

impl Debug for ExportedExecutionPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExportedExecutionPlan")
            .field("plan", &self.plan)
            .finish()
    }
}

impl ExportedExecutionPlan {
    /// Create a new ExportedExecutionPlan from an ExecutionPlan.
    ///
    /// Takes an `Arc<Runtime>` to ensure the runtime stays alive for the lifetime
    /// of the exported plan, preventing "Worker thread terminated" errors.
    pub fn new(
        plan: Arc<dyn ExecutionPlan>,
        task_context: Arc<TaskContext>,
        runtime: Arc<Runtime>,
    ) -> Self {
        Self {
            plan,
            task_context,
            runtime,
        }
    }

    fn schema(&self) -> SchemaRef {
        self.plan.schema()
    }

    fn get_property(&self, property: &str) -> Result<String> {
        match property {
            "plan_properties" => {
                let props = PlanPropertiesArgs::from_plan(self.plan.as_ref());
                serde_json::to_string(&props).map_err(|e| {
                    sedona_internal_datafusion_err!("Failed to serialize plan properties: {}", e)
                })
            }
            "debug_string" => Ok(format!("{:?}", self.plan)),
            "display_default" => {
                use std::fmt::Write;
                let mut s = String::new();
                let _ = write!(
                    s,
                    "{}",
                    DisplayAsWrapper(&self.plan, DisplayFormatType::Default)
                );
                Ok(s)
            }
            "display_verbose" => {
                use std::fmt::Write;
                let mut s = String::new();
                let _ = write!(
                    s,
                    "{}",
                    DisplayAsWrapper(&self.plan, DisplayFormatType::Verbose)
                );
                Ok(s)
            }
            "display_tree_render" => {
                use std::fmt::Write;
                let mut s = String::new();
                let _ = write!(
                    s,
                    "{}",
                    DisplayAsWrapper(&self.plan, DisplayFormatType::TreeRender)
                );
                Ok(s)
            }
            "name" => Ok(self.plan.name().to_string()),
            "cardinality_effect" => {
                let effect = match self.plan.cardinality_effect() {
                    CardinalityEffect::Unknown => "Unknown",
                    CardinalityEffect::Equal => "Equal",
                    CardinalityEffect::LowerEqual => "LowerEqual",
                    CardinalityEffect::GreaterEqual => "GreaterEqual",
                };
                Ok(effect.to_string())
            }
            "maintains_input_order" => {
                let order = self.plan.maintains_input_order();
                serde_json::to_string(&order).map_err(|e| {
                    sedona_internal_datafusion_err!(
                        "Failed to serialize maintains_input_order: {}",
                        e
                    )
                })
            }
            "metrics" => {
                // Serialize metrics as JSON with aggregated values
                if let Some(metrics) = self.plan.metrics() {
                    let serialized = SerializedMetrics::from_metrics_set(&metrics);
                    serde_json::to_string(&serialized).map_err(|e| {
                        sedona_internal_datafusion_err!("Failed to serialize metrics: {}", e)
                    })
                } else {
                    Ok("null".to_string())
                }
            }
            _ => exec_err!("Unknown property: {}", property),
        }
    }

    fn execute(&self, partition: usize) -> Result<SendableRecordBatchStream> {
        // Enter the runtime context so that plan.execute() can spawn tasks
        let _guard = self.runtime.enter();
        self.plan.execute(partition, self.task_context.clone())
    }
}

impl From<ExportedExecutionPlan> for SedonaCExecutionPlan {
    fn from(value: ExportedExecutionPlan) -> Self {
        let boxed = Box::new(value);
        Self {
            get_schema: Some(c_exec_plan_get_schema),
            get_property_schema: Some(c_exec_plan_get_property_schema),
            get_property: Some(c_exec_plan_get_property),
            with_property: None,
            execute: Some(c_exec_plan_execute),
            execute_async: None,
            reserved: null_mut(),
            release: Some(c_exec_plan_release),
            private_data: Box::into_raw(boxed) as *mut c_void,
        }
    }
}

unsafe extern "C" fn c_exec_plan_get_schema(
    self_: *const SedonaCExecutionPlan,
    out: *mut FFI_ArrowSchema,
    err: *mut SedonaCError,
) -> c_int {
    debug_assert!(!self_.is_null(), "self pointer is null");
    debug_assert!(!out.is_null(), "out pointer is null");
    let self_ref = &*self_;
    debug_assert!(!self_ref.private_data.is_null(), "private_data is null");
    let plan = &*(self_ref.private_data as *const ExportedExecutionPlan);

    let schema = plan.schema();
    match FFI_ArrowSchema::try_from(schema.as_ref()) {
        Ok(ffi_schema) => {
            std::ptr::write(out, ffi_schema);
            ERRNO_OK
        }
        Err(e) => {
            set_ffi_error!(err, "Failed to convert schema to FFI: {}", e);
            libc::EINVAL
        }
    }
}

unsafe extern "C" fn c_exec_plan_get_property_schema(
    _self_: *const SedonaCExecutionPlan,
    _property: *const std::ffi::c_char,
    out: *mut FFI_ArrowSchema,
    err: *mut SedonaCError,
) -> c_int {
    debug_assert!(!out.is_null(), "out pointer is null");
    // All properties are returned as Utf8 strings (including JSON)
    use arrow_schema::{DataType, Field};
    let field = Field::new("value", DataType::Utf8, false);
    match FFI_ArrowSchema::try_from(&field) {
        Ok(ffi_schema) => {
            std::ptr::write(out, ffi_schema);
            ERRNO_OK
        }
        Err(e) => {
            set_ffi_error!(err, "Failed to convert field to FFI schema: {}", e);
            libc::EINVAL
        }
    }
}

unsafe extern "C" fn c_exec_plan_get_property(
    self_: *const SedonaCExecutionPlan,
    property: *const std::ffi::c_char,
    _args: *mut SedonaCExecutionPlanArgs,
    out: *mut arrow_array::ffi::FFI_ArrowArray,
    err: *mut SedonaCError,
) -> c_int {
    debug_assert!(!self_.is_null(), "self pointer is null");
    debug_assert!(!out.is_null(), "out pointer is null");
    let self_ref = &*self_;
    debug_assert!(!self_ref.private_data.is_null(), "private_data is null");
    let plan = &*(self_ref.private_data as *const ExportedExecutionPlan);
    let property_str = cstr_from_ptr_or_empty(property);

    match plan.get_property(&property_str) {
        Ok(value) => {
            // Return the string as a single-element string array
            use arrow_array::{builder::StringBuilder, Array};
            let mut builder = StringBuilder::new();
            builder.append_value(&value);
            let array = builder.finish();
            let ffi_array = arrow_array::ffi::FFI_ArrowArray::new(&array.to_data());
            std::ptr::write(out, ffi_array);
            ERRNO_OK
        }
        Err(e) => {
            set_ffi_error!(err, "{}", e);
            libc::EINVAL
        }
    }
}

unsafe extern "C" fn c_exec_plan_execute(
    self_: *const SedonaCExecutionPlan,
    args: *mut SedonaCExecutionPlanArgs,
    out: *mut FFI_ArrowArrayStream,
    err: *mut SedonaCError,
) -> c_int {
    debug_assert!(!self_.is_null(), "self pointer is null");
    debug_assert!(!args.is_null(), "args pointer is null");
    debug_assert!(!out.is_null(), "out pointer is null");
    let self_ref = &*self_;
    let args_ref = &*args;
    debug_assert!(!self_ref.private_data.is_null(), "private_data is null");
    let plan = &*(self_ref.private_data as *const ExportedExecutionPlan);

    let args_slice = if args_ref.args.is_null() || args_ref.args_len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(args_ref.args, args_ref.args_len)
    };

    let execute_args: ExecuteArgs = match serde_json::from_slice(args_slice) {
        Ok(a) => a,
        Err(e) => {
            set_ffi_error!(err, "Failed to parse execute args: {}", e);
            return libc::EINVAL;
        }
    };

    match plan.execute(execute_args.partition) {
        Ok(stream) => {
            // Create a streaming reader that polls the stream lazily
            let reader = StreamingRecordBatchReader::new(stream, plan.runtime.clone());
            let ffi_stream = FFI_ArrowArrayStream::new(Box::new(reader));
            std::ptr::write(out, ffi_stream);
            ERRNO_OK
        }
        Err(e) => {
            set_ffi_error!(err, "{}", e);
            libc::EINVAL
        }
    }
}

unsafe extern "C" fn c_exec_plan_release(self_: *mut SedonaCExecutionPlan) {
    debug_assert!(!self_.is_null(), "self pointer is null");
    let self_ref = &mut *self_;
    if !self_ref.private_data.is_null() {
        let _ = Box::from_raw(self_ref.private_data as *mut ExportedExecutionPlan);
        self_ref.private_data = null_mut();
    }
    self_ref.release = None;
}

/// An [ExecutionPlan] that wraps an imported [SedonaCExecutionPlan].
///
/// This allows plans to be imported from across an FFI boundary and executed
/// within DataFusion.
pub struct ImportedSedonaCExec {
    inner: SedonaCExecutionPlan,
    schema: SchemaRef,
    properties: PlanProperties,
    supports_limit_pushdown: bool,
    name: String,
    /// Stored as Arc so it can be cloned for each partition execution.
    cancel_checker: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
    /// Interval for periodic cancellation checking during stream consumption.
    check_interval: Option<Duration>,
}

impl Debug for ImportedSedonaCExec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Try to get debug string from the FFI plan
        if let Ok(debug_str) = self.get_debug_string() {
            f.debug_struct("ImportedSedonaCExec")
                .field("inner", &debug_str)
                .finish()
        } else {
            f.debug_struct("ImportedSedonaCExec").finish()
        }
    }
}

impl ImportedSedonaCExec {
    /// Create a new ImportedSedonaCExec from a SedonaCExecutionPlan.
    ///
    /// This will query the plan for its schema and properties.
    pub fn try_new(inner: SedonaCExecutionPlan) -> Result<Self> {
        // Refuse to import a structure without a valid release callback
        if inner.release.is_none() {
            return sedona_internal_err!("SedonaCExecutionPlan does not have a release callback");
        }

        // Get schema
        let Some(get_schema) = inner.get_schema else {
            return sedona_internal_err!("SedonaCExecutionPlan does not have get_schema");
        };

        let mut ffi_schema = FFI_ArrowSchema::empty();
        let mut err = SedonaCError::default();
        let code = unsafe { get_schema(&inner, &mut ffi_schema, &mut err) };
        if code != ERRNO_OK {
            return sedona_internal_err!("Failed to get schema: {}", err);
        }
        let schema = Arc::new(Schema::try_from(&ffi_schema)?);

        // Get plan properties
        let props_args = PlanPropertiesArgs::from_ffi_plan(&inner)?;
        let supports_limit_pushdown = props_args.supports_limit_pushdown;
        let properties = props_args.into_plan_properties(schema.clone());

        // Get the inner plan's name
        let inner_name = get_plan_string_property(&inner, "name").unwrap_or_default();
        let name = format!("ImportedSedonaCExec<{}>", inner_name);

        Ok(Self {
            inner,
            schema,
            properties,
            supports_limit_pushdown,
            name,
            cancel_checker: None,
            check_interval: None,
        })
    }

    /// Set a cancellation checker for this execution plan.
    ///
    /// The checker is called periodically (controlled by `with_check_interval`)
    /// before batches are read from the FFI stream.
    /// If it returns `true`, the stream yields a cancellation error.
    pub fn with_cancel_checker<F>(mut self, cancel_checker: F) -> Self
    where
        F: Fn() -> bool + Send + Sync + 'static,
    {
        self.cancel_checker = Some(Arc::new(cancel_checker));
        self
    }

    /// Set the interval for periodic cancellation checking.
    ///
    /// When set, the cancel checker will only be called when this interval
    /// has elapsed since the last check, reducing overhead for fast streams.
    /// If not set, the checker is called before every batch.
    pub fn with_check_interval(mut self, interval: Duration) -> Self {
        self.check_interval = Some(interval);
        self
    }

    fn get_debug_string(&self) -> Result<String> {
        get_plan_string_property(&self.inner, "debug_string")
    }
}

impl DisplayAs for ImportedSedonaCExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let property = match t {
            DisplayFormatType::Default => "display_default",
            DisplayFormatType::Verbose => "display_verbose",
            DisplayFormatType::TreeRender => "display_tree_render",
        };

        // Always show the wrapper name, with the inner plan's display info
        if let Ok(display_str) = get_plan_string_property(&self.inner, property) {
            write!(f, "ImportedSedonaCExec: {}", display_str)
        } else {
            write!(f, "ImportedSedonaCExec")
        }
    }
}

impl ExecutionPlan for ImportedSedonaCExec {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn partition_statistics(&self, _partition: Option<usize>) -> Result<Statistics> {
        Ok(Statistics::new_unknown(&self.schema))
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        get_plan_string_property(&self.inner, "cardinality_effect")
            .ok()
            .and_then(|s| match s.as_str() {
                "Equal" => Some(CardinalityEffect::Equal),
                "LowerEqual" => Some(CardinalityEffect::LowerEqual),
                "GreaterEqual" => Some(CardinalityEffect::GreaterEqual),
                _ => None,
            })
            .unwrap_or(CardinalityEffect::Unknown)
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        // ImportedSedonaCExec has no children (it's a leaf node), so we return
        // an empty Vec. The inner plan's maintains_input_order doesn't apply
        // since we don't expose the inner plan's children.
        vec![]
    }

    fn metrics(&self) -> Option<MetricsSet> {
        get_plan_property::<Option<SerializedMetrics>, ()>(&self.inner, "metrics", None)
            .ok()
            .flatten()
            .map(|sm| sm.into_metrics_set())
    }

    fn supports_limit_pushdown(&self) -> bool {
        self.supports_limit_pushdown
    }

    fn statistics(&self) -> Result<Statistics> {
        Ok(Statistics::new_unknown(&self.schema))
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let Some(execute) = self.inner.execute else {
            return sedona_internal_err!("SedonaCExecutionPlan does not have execute");
        };

        let args = ExecuteArgs { partition };
        let args_bytes = serde_json::to_vec(&args).map_err(|e| {
            sedona_internal_datafusion_err!("Failed to serialize execute args: {}", e)
        })?;

        let mut ffi_args = SedonaCExecutionPlanArgs {
            args: args_bytes.as_ptr(),
            args_len: args_bytes.len(),
            exec_plans: std::ptr::null(),
            num_exec_plans: 0,
            exprs: std::ptr::null(),
            num_exprs: 0,
            reserved: null_mut(),
        };

        let mut ffi_stream = FFI_ArrowArrayStream::empty();
        let mut err = SedonaCError::default();

        let code = unsafe { execute(&self.inner, &mut ffi_args, &mut ffi_stream, &mut err) };

        if code != ERRNO_OK {
            return exec_err!("Failed to execute plan: {}", err);
        }

        // Convert FFI stream to SendableRecordBatchStream
        // Clone the Arc and wrap in a new Box for this execution
        let cancel_checker: Option<CancelChecker> = self.cancel_checker.as_ref().map(|c| {
            let c = c.clone();
            Box::new(move || c()) as CancelChecker
        });
        unsafe { ffi_stream_to_sendable(&mut ffi_stream, cancel_checker, self.check_interval) }
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return exec_err!("ImportedSedonaCExec does not support children");
        }
        Ok(self)
    }
}

/// Arguments for executing a partition of an execution plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecuteArgs {
    pub partition: usize,
}

/// Metrics serialized for FFI transfer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedMetrics {
    pub display: String,
}

impl SerializedMetrics {
    /// Create from a MetricsSet by capturing its display string.
    pub fn from_metrics_set(set: &MetricsSet) -> Self {
        Self {
            display: set.to_string(),
        }
    }

    /// Convert back to a MetricsSet with a custom metric containing the display string.
    pub fn into_metrics_set(self) -> MetricsSet {
        let mut set = MetricsSet::new();
        let custom = ImportedMetrics::new(self.display);
        let metric = Metric::new(
            MetricValue::Custom {
                name: "imported_metrics".into(),
                value: Arc::new(custom),
            },
            None,
        );
        set.push(Arc::new(metric));
        set
    }
}

/// Custom metric value that holds imported metrics display string.
#[derive(Debug, Clone)]
pub struct ImportedMetrics {
    display: String,
}

impl ImportedMetrics {
    pub fn new(display: String) -> Self {
        Self { display }
    }
}

impl Display for ImportedMetrics {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display)
    }
}

impl CustomMetricValue for ImportedMetrics {
    fn new_empty(&self) -> Arc<dyn CustomMetricValue> {
        Arc::new(Self {
            display: String::new(),
        })
    }

    fn aggregate(&self, other: Arc<dyn CustomMetricValue>) {
        // No aggregation for imported metrics - they're read-only snapshots
        let _ = other;
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn is_eq(&self, other: &Arc<dyn CustomMetricValue>) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|o| o.display == self.display)
    }
}

/// Properties of an execution plan serialized across FFI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanPropertiesArgs {
    pub num_partitions: usize,
    pub supports_limit_pushdown: bool,
    pub emission_type: String,
    pub boundedness: String,
}

impl PlanPropertiesArgs {
    /// Create from an ExecutionPlan.
    pub fn from_plan(plan: &dyn ExecutionPlan) -> Self {
        let plan_props = plan.properties();
        let emission_type = match plan_props.emission_type {
            EmissionType::Incremental => "Incremental",
            EmissionType::Final => "Final",
            EmissionType::Both => "Both",
        };
        let boundedness = match plan_props.boundedness {
            Boundedness::Bounded => "Bounded",
            Boundedness::Unbounded {
                requires_infinite_memory: false,
            } => "Unbounded",
            Boundedness::Unbounded {
                requires_infinite_memory: true,
            } => "UnboundedInfiniteMemory",
        };
        Self {
            num_partitions: plan_props.output_partitioning().partition_count(),
            supports_limit_pushdown: plan.supports_limit_pushdown(),
            emission_type: emission_type.to_string(),
            boundedness: boundedness.to_string(),
        }
    }

    /// Extract plan properties from a SedonaCExecutionPlan via FFI.
    pub fn from_ffi_plan(plan: &SedonaCExecutionPlan) -> Result<Self> {
        get_plan_property::<Self, ()>(plan, "plan_properties", None)
    }

    /// Convert to DataFusion PlanProperties.
    pub fn into_plan_properties(self, schema: SchemaRef) -> PlanProperties {
        let emission_type = match self.emission_type.as_str() {
            "Final" => EmissionType::Final,
            "Both" => EmissionType::Both,
            _ => EmissionType::Incremental,
        };
        let boundedness = match self.boundedness.as_str() {
            "Unbounded" => Boundedness::Unbounded {
                requires_infinite_memory: false,
            },
            "UnboundedInfiniteMemory" => Boundedness::Unbounded {
                requires_infinite_memory: true,
            },
            _ => Boundedness::Bounded,
        };
        PlanProperties::new(
            datafusion_physical_expr::EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(self.num_partitions),
            emission_type,
            boundedness,
        )
    }
}

/// Helper wrapper to format an ExecutionPlan with a specific DisplayFormatType.
struct DisplayAsWrapper<'a>(&'a Arc<dyn ExecutionPlan>, DisplayFormatType);

impl std::fmt::Display for DisplayAsWrapper<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt_as(self.1, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field};
    use datafusion_common::assert_batches_eq;
    use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
    use futures::{stream, StreamExt};
    use std::fmt::Formatter;

    /// A dummy ExecutionPlan with fixed, predictable values for testing FFI roundtrip.
    #[derive(Debug)]
    struct DummyExec {
        schema: SchemaRef,
        properties: PlanProperties,
        limit_pushdown: bool,
    }

    impl DummyExec {
        fn new() -> Self {
            Self::with_properties(EmissionType::Incremental, Boundedness::Bounded, true)
        }

        fn with_properties(
            emission_type: EmissionType,
            boundedness: Boundedness,
            supports_limit_pushdown: bool,
        ) -> Self {
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int32, false),
                Field::new("value", DataType::Int32, false),
            ]));
            let properties = PlanProperties::new(
                datafusion_physical_expr::EquivalenceProperties::new(schema.clone()),
                Partitioning::UnknownPartitioning(3),
                emission_type,
                boundedness,
            );
            Self {
                schema,
                properties,
                limit_pushdown: supports_limit_pushdown,
            }
        }
    }

    impl DisplayAs for DummyExec {
        fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
            match t {
                DisplayFormatType::Default => write!(f, "DummyExec: default format"),
                DisplayFormatType::Verbose => write!(f, "DummyExec: verbose format with schema"),
                DisplayFormatType::TreeRender => write!(f, "DummyExec: tree render format"),
            }
        }
    }

    impl ExecutionPlan for DummyExec {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn name(&self) -> &str {
            "DummyExec"
        }

        fn properties(&self) -> &PlanProperties {
            &self.properties
        }

        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            vec![]
        }

        fn with_new_children(
            self: Arc<Self>,
            children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            if !children.is_empty() {
                return exec_err!("DummyExec does not support children");
            }
            Ok(self)
        }

        fn supports_limit_pushdown(&self) -> bool {
            self.limit_pushdown
        }

        fn execute(
            &self,
            partition: usize,
            _context: Arc<TaskContext>,
        ) -> Result<SendableRecordBatchStream> {
            // Return a batch with partition-specific data
            let ids = Int32Array::from(vec![
                partition as i32 * 10 + 1,
                partition as i32 * 10 + 2,
                partition as i32 * 10 + 3,
            ]);
            let values = Int32Array::from(vec![100, 200, 300]);
            let batch =
                RecordBatch::try_new(self.schema.clone(), vec![Arc::new(ids), Arc::new(values)])?;

            Ok(Box::pin(RecordBatchStreamAdapter::new(
                self.schema.clone(),
                stream::iter(vec![Ok(batch)]),
            )))
        }
    }

    /// Create a test runtime wrapped in Arc.
    fn test_runtime() -> Arc<Runtime> {
        Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap(),
        )
    }

    /// Helper to set up an imported plan from a DummyExec through FFI roundtrip.
    /// Returns the runtime to keep it alive for the duration of the test.
    fn setup_imported_plan() -> (ImportedSedonaCExec, Arc<TaskContext>, Arc<Runtime>) {
        let dummy = Arc::new(DummyExec::new());
        let runtime = test_runtime();
        let task_ctx = Arc::new(TaskContext::default());

        let exported = ExportedExecutionPlan::new(dummy, task_ctx.clone(), runtime.clone());
        let ffi_plan: SedonaCExecutionPlan = exported.into();
        let imported = ImportedSedonaCExec::try_new(ffi_plan).unwrap();

        (imported, task_ctx, runtime)
    }

    fn setup_imported_plan_with(
        emission_type: EmissionType,
        boundedness: Boundedness,
        supports_limit_pushdown: bool,
    ) -> (ImportedSedonaCExec, Arc<TaskContext>, Arc<Runtime>) {
        let dummy = Arc::new(DummyExec::with_properties(
            emission_type,
            boundedness,
            supports_limit_pushdown,
        ));
        let runtime = test_runtime();
        let task_ctx = Arc::new(TaskContext::default());

        let exported = ExportedExecutionPlan::new(dummy, task_ctx.clone(), runtime.clone());
        let ffi_plan: SedonaCExecutionPlan = exported.into();
        let imported = ImportedSedonaCExec::try_new(ffi_plan).unwrap();

        (imported, task_ctx, runtime)
    }

    #[test]
    fn test_execution_plan_roundtrip_schema() {
        let (imported, _, _runtime) = setup_imported_plan();

        // Verify schema matches
        assert_eq!(imported.schema().fields().len(), 2);
        assert_eq!(imported.schema().field(0).name(), "id");
        assert_eq!(imported.schema().field(1).name(), "value");
    }

    #[test]
    fn test_execution_plan_roundtrip_name() {
        let (imported, _, _runtime) = setup_imported_plan();
        assert_eq!(imported.name(), "ImportedSedonaCExec<DummyExec>");
    }

    #[test]
    fn test_execution_plan_roundtrip_properties() {
        // Test all EmissionType variants
        for (emission_type, expected_str) in [
            (EmissionType::Incremental, "Incremental"),
            (EmissionType::Final, "Final"),
            (EmissionType::Both, "Both"),
        ] {
            let (imported, _, _runtime) =
                setup_imported_plan_with(emission_type, Boundedness::Bounded, true);
            let props = PlanPropertiesArgs::from_plan(&imported);
            assert_eq!(
                props.emission_type, expected_str,
                "emission_type mismatch for {:?}",
                emission_type
            );
        }

        // Test all Boundedness variants
        for (boundedness, expected_str) in [
            (Boundedness::Bounded, "Bounded"),
            (
                Boundedness::Unbounded {
                    requires_infinite_memory: false,
                },
                "Unbounded",
            ),
            (
                Boundedness::Unbounded {
                    requires_infinite_memory: true,
                },
                "UnboundedInfiniteMemory",
            ),
        ] {
            let (imported, _, _runtime) =
                setup_imported_plan_with(EmissionType::Incremental, boundedness, true);
            let props = PlanPropertiesArgs::from_plan(&imported);
            assert_eq!(
                props.boundedness, expected_str,
                "boundedness mismatch for {:?}",
                boundedness
            );
        }

        // Test supports_limit_pushdown
        let (imported_with, _, _runtime) =
            setup_imported_plan_with(EmissionType::Incremental, Boundedness::Bounded, true);
        assert!(imported_with.supports_limit_pushdown());

        let (imported_without, _, _runtime) =
            setup_imported_plan_with(EmissionType::Incremental, Boundedness::Bounded, false);
        assert!(!imported_without.supports_limit_pushdown());

        // Test partition count
        let (imported, _, _runtime) = setup_imported_plan();
        assert_eq!(
            imported
                .properties()
                .output_partitioning()
                .partition_count(),
            3
        );
    }

    #[test]
    fn test_execution_plan_roundtrip_display_as() {
        let (imported, _, _runtime) = setup_imported_plan();

        // Helper struct to format DisplayAs with a specific format type
        struct DisplayAsFormat<'a, T: DisplayAs>(&'a T, DisplayFormatType);
        impl<T: DisplayAs> std::fmt::Display for DisplayAsFormat<'_, T> {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                self.0.fmt_as(self.1, f)
            }
        }

        // ImportedSedonaCExec shows itself with the inner plan's display
        assert_eq!(
            format!("{}", DisplayAsFormat(&imported, DisplayFormatType::Default)),
            "ImportedSedonaCExec: DummyExec: default format"
        );
        assert_eq!(
            format!("{}", DisplayAsFormat(&imported, DisplayFormatType::Verbose)),
            "ImportedSedonaCExec: DummyExec: verbose format with schema"
        );
        assert_eq!(
            format!(
                "{}",
                DisplayAsFormat(&imported, DisplayFormatType::TreeRender)
            ),
            "ImportedSedonaCExec: DummyExec: tree render format"
        );
    }

    #[test]
    fn test_execution_plan_roundtrip_debug_string() {
        let (imported, _, _runtime) = setup_imported_plan();

        let debug_str = imported.get_debug_string().unwrap();
        assert!(
            debug_str.contains("DummyExec"),
            "debug_string should contain 'DummyExec', got: {}",
            debug_str
        );
    }

    #[test]
    fn test_execution_plan_roundtrip_execute() {
        let (imported, task_ctx, runtime) = setup_imported_plan();

        runtime.block_on(async {
            // Execute partition 0
            let stream = imported.execute(0, task_ctx.clone()).unwrap();
            let batches: Vec<RecordBatch> = stream
                .collect::<Vec<_>>()
                .await
                .into_iter()
                .collect::<Result<Vec<_>>>()
                .unwrap();

            let expected = [
                "+----+-------+",
                "| id | value |",
                "+----+-------+",
                "| 1  | 100   |",
                "| 2  | 200   |",
                "| 3  | 300   |",
                "+----+-------+",
            ];
            assert_batches_eq!(expected, &batches);

            // Execute partition 1
            let stream2 = imported.execute(1, task_ctx).unwrap();
            let batches2: Vec<RecordBatch> = stream2
                .collect::<Vec<_>>()
                .await
                .into_iter()
                .collect::<Result<Vec<_>>>()
                .unwrap();

            let expected2 = [
                "+----+-------+",
                "| id | value |",
                "+----+-------+",
                "| 11 | 100   |",
                "| 12 | 200   |",
                "| 13 | 300   |",
                "+----+-------+",
            ];
            assert_batches_eq!(expected2, &batches2);
        });
    }
}
