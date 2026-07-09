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
    fmt::Debug,
    ptr::null_mut,
    sync::Arc,
    time::Duration,
};

use arrow_array::ffi::FFI_ArrowArray;
use arrow_schema::{ffi::FFI_ArrowSchema, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion_catalog::{Session, TableProvider};
use datafusion_common::{exec_err, Result, Statistics};
use datafusion_expr::{Expr, TableType};
use datafusion_physical_plan::ExecutionPlan;
use sedona_common::{sedona_internal_datafusion_err, sedona_internal_err};
use serde::{Deserialize, Serialize};
use tokio::runtime::Runtime;

use crate::execution_plan::{ExportedExecutionPlan, ImportedSedonaCExec};
use crate::extension::{
    SedonaCError, SedonaCExecutionPlan, SedonaCExecutionPlanArgs, SedonaCTableProvider,
};
use crate::set_ffi_error;
use crate::utils::{cstr_from_ptr_or_empty, get_table_provider_string_property, ERRNO_OK};

/// A TableProvider wrapper that can be exported across FFI.
///
/// This wraps an inner TableProvider and exposes it via the SedonaCTableProvider
/// FFI interface.
pub struct ExportedTableProvider {
    inner: Arc<dyn TableProvider>,
    session: Arc<dyn Session>,
    /// We hold Arc<Runtime> instead of just Handle to keep the runtime alive
    /// as long as this provider exists.
    runtime: Arc<Runtime>,
}

impl Debug for ExportedTableProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExportedTableProvider")
            .field("inner", &self.inner)
            .finish()
    }
}

impl ExportedTableProvider {
    /// Create a new ExportedTableProvider from a TableProvider.
    ///
    /// The session is used during scan operations and must support physical planning
    /// if the inner TableProvider requires it (e.g., for Views).
    ///
    /// Takes an `Arc<Runtime>` to ensure the runtime stays alive for the lifetime
    /// of the provider, preventing "Worker thread terminated" errors.
    pub fn new(
        inner: Arc<dyn TableProvider>,
        session: Arc<dyn Session>,
        runtime: Arc<Runtime>,
    ) -> Self {
        Self {
            inner,
            session,
            runtime,
        }
    }

    fn scan(
        &self,
        projection: Option<Vec<usize>>,
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let inner = self.inner.clone();
        let session = self.session.clone();
        let runtime = self.runtime.clone();

        std::thread::spawn(move || {
            let projection_ref = projection.as_ref();
            runtime
                .handle()
                .block_on(inner.scan(session.as_ref(), projection_ref, &[], limit))
        })
        .join()
        .map_err(|e| sedona_internal_datafusion_err!("Scan thread panicked {e:?}"))?
    }

    fn get_property(&self, property: &str) -> Result<String> {
        match property {
            "table_type" => {
                let table_type = self.inner.table_type();
                let type_str = match table_type {
                    TableType::Base => "Base",
                    TableType::View => "View",
                    TableType::Temporary => "Temporary",
                };
                Ok(type_str.to_string())
            }
            _ => exec_err!("Unknown property: {}", property),
        }
    }
}

impl From<ExportedTableProvider> for SedonaCTableProvider {
    fn from(value: ExportedTableProvider) -> Self {
        let boxed = Box::new(value);
        Self {
            get_schema: Some(c_table_provider_get_schema),
            get_property_schema: Some(c_table_provider_get_property_schema),
            get_property: Some(c_table_provider_get_property),
            scan: Some(c_table_provider_scan),
            insert: None,
            update: None,
            delete_rows: None,
            reserved: null_mut(),
            release: Some(c_table_provider_release),
            private_data: Box::into_raw(boxed) as *mut c_void,
        }
    }
}

unsafe extern "C" fn c_table_provider_get_schema(
    self_: *const SedonaCTableProvider,
    out: *mut FFI_ArrowSchema,
    err: *mut SedonaCError,
) -> c_int {
    debug_assert!(!self_.is_null(), "self pointer is null");
    debug_assert!(!out.is_null(), "out pointer is null");
    let self_ref = &*self_;
    debug_assert!(!self_ref.private_data.is_null(), "private_data is null");
    let provider = &*(self_ref.private_data as *const ExportedTableProvider);

    let schema = provider.inner.schema();
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

unsafe extern "C" fn c_table_provider_get_property_schema(
    _self_: *const SedonaCTableProvider,
    _property: *const std::ffi::c_char,
    out: *mut FFI_ArrowSchema,
    err: *mut SedonaCError,
) -> c_int {
    debug_assert!(!out.is_null(), "out pointer is null");
    // All properties are returned as Utf8 strings
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

unsafe extern "C" fn c_table_provider_get_property(
    self_: *const SedonaCTableProvider,
    property: *const std::ffi::c_char,
    _args: *mut SedonaCExecutionPlanArgs,
    out: *mut FFI_ArrowArray,
    err: *mut SedonaCError,
) -> c_int {
    debug_assert!(!self_.is_null(), "self pointer is null");
    debug_assert!(!out.is_null(), "out pointer is null");
    let self_ref = &*self_;
    debug_assert!(!self_ref.private_data.is_null(), "private_data is null");
    let provider = &*(self_ref.private_data as *const ExportedTableProvider);
    let property_str = cstr_from_ptr_or_empty(property);

    match provider.get_property(&property_str) {
        Ok(value) => {
            // Return the string as a single-element string array
            use arrow_array::{builder::StringBuilder, Array};
            let mut builder = StringBuilder::new();
            builder.append_value(&value);
            let array = builder.finish();
            let ffi_array = FFI_ArrowArray::new(&array.to_data());
            std::ptr::write(out, ffi_array);
            ERRNO_OK
        }
        Err(e) => {
            set_ffi_error!(err, "{}", e);
            libc::EINVAL
        }
    }
}

unsafe extern "C" fn c_table_provider_scan(
    self_: *const SedonaCTableProvider,
    args: *mut SedonaCExecutionPlanArgs,
    out: *mut SedonaCExecutionPlan,
    err: *mut SedonaCError,
) -> c_int {
    debug_assert!(!self_.is_null(), "self pointer is null");
    debug_assert!(!args.is_null(), "args pointer is null");
    debug_assert!(!out.is_null(), "out pointer is null");
    let self_ref = &*self_;
    let args_ref = &*args;
    debug_assert!(!self_ref.private_data.is_null(), "private_data is null");
    let provider = &*(self_ref.private_data as *const ExportedTableProvider);

    // Parse scan args
    let args_slice = if args_ref.args.is_null() || args_ref.args_len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(args_ref.args, args_ref.args_len)
    };

    let scan_args: ScanArgs = if args_slice.is_empty() {
        ScanArgs {
            projection: None,
            limit: None,
        }
    } else {
        match serde_json::from_slice(args_slice) {
            Ok(a) => a,
            Err(e) => {
                set_ffi_error!(err, "Failed to parse scan args: {}", e);
                return libc::EINVAL;
            }
        }
    };

    match provider.scan(scan_args.projection, scan_args.limit) {
        Ok(plan) => {
            let task_ctx = provider.session.task_ctx();
            let exported = ExportedExecutionPlan::new(plan, task_ctx, provider.runtime.clone());
            let ffi_plan: SedonaCExecutionPlan = exported.into();
            std::ptr::write(out, ffi_plan);
            ERRNO_OK
        }
        Err(e) => {
            set_ffi_error!(err, "{}", e);
            libc::EINVAL
        }
    }
}

unsafe extern "C" fn c_table_provider_release(self_: *mut SedonaCTableProvider) {
    debug_assert!(!self_.is_null(), "self pointer is null");
    let self_ref = &mut *self_;
    if !self_ref.private_data.is_null() {
        let _ = Box::from_raw(self_ref.private_data as *mut ExportedTableProvider);
        self_ref.private_data = null_mut();
    }
    self_ref.release = None;
}

/// A TableProvider that wraps an imported SedonaCTableProvider.
///
/// This allows table providers from across an FFI boundary to be used
/// within DataFusion.
pub struct ImportedTableProvider {
    inner: SedonaCTableProvider,
    schema: SchemaRef,
    table_type: TableType,
    /// Stored as Arc so it can be cloned for each scan execution.
    cancel_checker: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
    /// Interval for periodic cancellation checking during stream consumption.
    check_interval: Option<Duration>,
}

impl Debug for ImportedTableProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImportedTableProvider").finish()
    }
}

impl ImportedTableProvider {
    /// Create a new ImportedTableProvider from a SedonaCTableProvider.
    pub fn try_new(inner: SedonaCTableProvider) -> Result<Self> {
        // Refuse to import a structure without a valid release callback
        if inner.release.is_none() {
            return sedona_internal_err!("SedonaCTableProvider does not have a release callback");
        }

        // Get schema via Arrow C Data Interface
        let Some(get_schema) = inner.get_schema else {
            return sedona_internal_err!("SedonaCTableProvider does not have get_schema");
        };

        let mut ffi_schema = FFI_ArrowSchema::empty();
        let mut err = SedonaCError::default();
        let code = unsafe { get_schema(&inner, &mut ffi_schema, &mut err) };
        if code != ERRNO_OK {
            return sedona_internal_err!("Failed to get schema: {}", err);
        }
        let schema = Arc::new(Schema::try_from(&ffi_schema)?);

        // Get table type via get_property
        let table_type = Self::get_table_type(&inner)?;

        Ok(Self {
            inner,
            schema,
            table_type,
            cancel_checker: None,
            check_interval: None,
        })
    }

    /// Set a cancellation checker for this table provider.
    ///
    /// The checker is called periodically (controlled by `with_check_interval`)
    /// before batches are read when scanning.
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

    fn get_table_type(provider: &SedonaCTableProvider) -> Result<TableType> {
        let type_str = match get_table_provider_string_property(provider, "table_type") {
            Ok(s) => s,
            Err(_) => return Ok(TableType::Base), // Default to Base if property fetch fails
        };

        match type_str.as_str() {
            "Base" => Ok(TableType::Base),
            "View" => Ok(TableType::View),
            "Temporary" => Ok(TableType::Temporary),
            _ => Ok(TableType::Base), // Default to Base for unknown types
        }
    }
}

#[async_trait]
impl TableProvider for ImportedTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        self.table_type
    }

    fn statistics(&self) -> Option<Statistics> {
        None
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let Some(scan) = self.inner.scan else {
            return sedona_internal_err!("SedonaCTableProvider does not have scan");
        };

        let args = ScanArgs {
            projection: projection.cloned(),
            limit,
        };
        let args_bytes = serde_json::to_vec(&args)
            .map_err(|e| sedona_internal_datafusion_err!("Failed to serialize scan args: {}", e))?;

        let mut ffi_args = SedonaCExecutionPlanArgs {
            args: args_bytes.as_ptr(),
            args_len: args_bytes.len(),
            exec_plans: std::ptr::null(),
            num_exec_plans: 0,
            exprs: std::ptr::null(),
            num_exprs: 0,
            reserved: null_mut(),
        };

        let mut ffi_plan = SedonaCExecutionPlan::default();
        let mut err = SedonaCError::default();

        let code = unsafe { scan(&self.inner, &mut ffi_args, &mut ffi_plan, &mut err) };

        if code != ERRNO_OK {
            return exec_err!("Failed to scan table: {}", err);
        }

        let mut exec = ImportedSedonaCExec::try_new(ffi_plan)?;

        // Pipe through the cancel checker and interval if configured
        if let Some(ref checker) = self.cancel_checker {
            let checker = checker.clone();
            exec = exec.with_cancel_checker(move || checker());
        }
        if let Some(interval) = self.check_interval {
            exec = exec.with_check_interval(interval);
        }

        Ok(Arc::new(exec))
    }
}

/// Arguments for a scan operation, serialized as JSON across FFI.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScanArgs {
    /// Column indices to project, or None for all columns.
    pub projection: Option<Vec<usize>>,
    /// Maximum number of rows to return, or None for unlimited.
    pub limit: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Float64Array, Int32Array, Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::prelude::SessionContext;
    use datafusion_catalog::Session;
    use datafusion_common::assert_batches_eq;
    use datafusion_expr::Expr;
    use datafusion_physical_plan::ExecutionPlan;

    /// Create a SessionContext with a test table containing 5 numeric columns and multiple batches.
    async fn create_test_context() -> Result<SessionContext> {
        let ctx = SessionContext::new();

        // Create schema with 5 numeric columns
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value_a", DataType::Int64, false),
            Field::new("value_b", DataType::Float64, false),
            Field::new("value_c", DataType::Int32, false),
            Field::new("value_d", DataType::Int64, false),
        ]));

        // Create 5 batches with different data
        let mut batches = Vec::new();
        for batch_num in 0..5 {
            let offset = batch_num * 10;
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from(vec![
                        offset + 1,
                        offset + 2,
                        offset + 3,
                        offset + 4,
                        offset + 5,
                    ])),
                    Arc::new(Int64Array::from(vec![
                        (offset + 1) as i64 * 100,
                        (offset + 2) as i64 * 100,
                        (offset + 3) as i64 * 100,
                        (offset + 4) as i64 * 100,
                        (offset + 5) as i64 * 100,
                    ])),
                    Arc::new(Float64Array::from(vec![
                        (offset + 1) as f64 * 1.5,
                        (offset + 2) as f64 * 1.5,
                        (offset + 3) as f64 * 1.5,
                        (offset + 4) as f64 * 1.5,
                        (offset + 5) as f64 * 1.5,
                    ])),
                    Arc::new(Int32Array::from(vec![
                        (offset + 1) * 2,
                        (offset + 2) * 2,
                        (offset + 3) * 2,
                        (offset + 4) * 2,
                        (offset + 5) * 2,
                    ])),
                    Arc::new(Int64Array::from(vec![
                        (offset + 1) as i64 * 1000,
                        (offset + 2) as i64 * 1000,
                        (offset + 3) as i64 * 1000,
                        (offset + 4) as i64 * 1000,
                        (offset + 5) as i64 * 1000,
                    ])),
                ],
            )?;
            batches.push(batch);
        }

        // Use a MemTable with all batches
        let provider = datafusion::datasource::MemTable::try_new(schema, vec![batches])?;
        ctx.register_table("test_data", Arc::new(provider))?;

        Ok(ctx)
    }

    /// Helper to test FFI table provider roundtrip with a SQL query.
    ///
    /// This handles the common pattern of:
    /// 1. Creating a test context with test_data table
    /// 2. Exporting the table provider through FFI
    /// 3. Importing it and registering in a new context
    /// 4. Running the SQL query and asserting results
    fn test_roundtrip_query(sql: &str, expected: &[&str]) -> Result<()> {
        let runtime = test_runtime();
        runtime.block_on(async {
            let ctx = create_test_context().await?;

            // Get the table provider from the context
            let table = ctx.table_provider("test_data").await?;

            // Export the table provider
            let session = Arc::new(ctx.state());
            let exported = ExportedTableProvider::new(table, session, runtime.clone());
            let ffi_provider: SedonaCTableProvider = exported.into();

            // Import the table provider
            let imported = ImportedTableProvider::try_new(ffi_provider)?;

            // Create a new context and register the imported table
            let ctx2 = SessionContext::new();
            ctx2.register_table("imported_data", Arc::new(imported))?;

            // Query and verify
            let result = ctx2.sql(sql).await?.collect().await?;
            assert_batches_eq!(expected, &result);
            Ok(())
        })
    }

    #[test]
    fn test_roundtrip_simple_select() {
        test_roundtrip_query(
            "SELECT id, value_a FROM imported_data ORDER BY id LIMIT 5",
            &[
                "+----+---------+",
                "| id | value_a |",
                "+----+---------+",
                "| 1  | 100     |",
                "| 2  | 200     |",
                "| 3  | 300     |",
                "| 4  | 400     |",
                "| 5  | 500     |",
                "+----+---------+",
            ],
        )
        .unwrap();
    }

    #[test]
    fn test_roundtrip_projection() {
        test_roundtrip_query(
            "SELECT value_b, value_d FROM imported_data ORDER BY value_b LIMIT 3",
            &[
                "+---------+---------+",
                "| value_b | value_d |",
                "+---------+---------+",
                "| 1.5     | 1000    |",
                "| 3.0     | 2000    |",
                "| 4.5     | 3000    |",
                "+---------+---------+",
            ],
        )
        .unwrap();
    }

    #[test]
    fn test_roundtrip_filter() {
        test_roundtrip_query(
            "SELECT id, value_a FROM imported_data WHERE id > 20 ORDER BY id LIMIT 5",
            &[
                "+----+---------+",
                "| id | value_a |",
                "+----+---------+",
                "| 21 | 2100    |",
                "| 22 | 2200    |",
                "| 23 | 2300    |",
                "| 24 | 2400    |",
                "| 25 | 2500    |",
                "+----+---------+",
            ],
        )
        .unwrap();
    }

    #[test]
    fn test_roundtrip_sort() {
        test_roundtrip_query(
            "SELECT id, value_c FROM imported_data ORDER BY id DESC LIMIT 5",
            &[
                "+----+---------+",
                "| id | value_c |",
                "+----+---------+",
                "| 45 | 90      |",
                "| 44 | 88      |",
                "| 43 | 86      |",
                "| 42 | 84      |",
                "| 41 | 82      |",
                "+----+---------+",
            ],
        )
        .unwrap();
    }

    #[test]
    fn test_roundtrip_limit() {
        test_roundtrip_query(
            "SELECT id FROM imported_data ORDER BY id LIMIT 3",
            &[
                "+----+", "| id |", "+----+", "| 1  |", "| 2  |", "| 3  |", "+----+",
            ],
        )
        .unwrap();
    }

    #[test]
    fn test_roundtrip_all_columns() {
        test_roundtrip_query(
            "SELECT * FROM imported_data ORDER BY id LIMIT 2",
            &[
                "+----+---------+---------+---------+---------+",
                "| id | value_a | value_b | value_c | value_d |",
                "+----+---------+---------+---------+---------+",
                "| 1  | 100     | 1.5     | 2       | 1000    |",
                "| 2  | 200     | 3.0     | 4       | 2000    |",
                "+----+---------+---------+---------+---------+",
            ],
        )
        .unwrap();
    }

    /// A dummy TableProvider with configurable table_type for testing FFI roundtrip.
    #[derive(Debug)]
    struct DummyTableProvider {
        schema: SchemaRef,
        table_type: TableType,
    }

    impl DummyTableProvider {
        fn with_table_type(table_type: TableType) -> Self {
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int32, false),
                Field::new("value", DataType::Int32, false),
            ]));
            Self { schema, table_type }
        }
    }

    #[async_trait]
    impl TableProvider for DummyTableProvider {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn schema(&self) -> SchemaRef {
            self.schema.clone()
        }

        fn table_type(&self) -> TableType {
            self.table_type
        }

        async fn scan(
            &self,
            _state: &dyn Session,
            _projection: Option<&Vec<usize>>,
            _filters: &[Expr],
            _limit: Option<usize>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            exec_err!("DummyTableProvider does not support scan")
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

    /// Helper to set up an imported table provider from a DummyTableProvider through FFI roundtrip.
    /// Returns the runtime to keep it alive for the duration of the test.
    fn setup_imported_provider_with(
        table_type: TableType,
    ) -> (ImportedTableProvider, Arc<Runtime>) {
        let dummy = Arc::new(DummyTableProvider::with_table_type(table_type));
        let ctx = SessionContext::new();
        let runtime = test_runtime();
        let session = Arc::new(ctx.state());
        let exported = ExportedTableProvider::new(dummy, session, runtime.clone());
        let ffi_provider: SedonaCTableProvider = exported.into();
        let imported =
            ImportedTableProvider::try_new(ffi_provider).expect("Failed to import table provider");
        (imported, runtime)
    }

    #[test]
    fn test_table_provider_roundtrip_schema() {
        let (imported, _runtime) = setup_imported_provider_with(TableType::Base);

        // Check schema roundtrip
        let schema = imported.schema();
        assert_eq!(schema.fields().len(), 2);
        assert_eq!(schema.field(0).name(), "id");
        assert_eq!(schema.field(0).data_type(), &DataType::Int32);
        assert_eq!(schema.field(1).name(), "value");
        assert_eq!(schema.field(1).data_type(), &DataType::Int32);
    }

    #[test]
    fn test_table_provider_roundtrip_table_type() {
        for table_type in [TableType::Base, TableType::View, TableType::Temporary] {
            let (imported, _runtime) = setup_imported_provider_with(table_type);
            assert_eq!(imported.table_type(), table_type);
        }
    }
}
