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

//! Byte-bounded batches for raster materialization.
//!
//! `datafusion.execution.batch_size` counts rows, but a row holding a
//! raster can carry megabytes of pixels, so the batch DataFusion hands
//! `RS_EnsureLoaded` can be gigabytes once loaded. DataFusion runs async
//! UDFs in an `AsyncFuncExec` that re-coalesces its input to exactly
//! `batch_size` rows before evaluating, so nothing upstream can shrink that
//! batch, and the UDF's `ideal_batch_size` only chunks the *invocation* —
//! the results are concatenated back into one array.
//!
//! [`RasterBatchBudgetRule`] therefore replaces every `AsyncFuncExec` that
//! carries an `rs_ensureloaded` call with an [`EnsureLoadedExec`]: same
//! expressions, same output schema, but each input batch is sliced so that
//! the *estimated* bytes of the rasters about to be materialized stay
//! within `sedona.raster.max_batch_bytes`, and each slice is evaluated and
//! emitted as its own batch. The estimate is metadata-only
//! ([`sedona_raster::size`]), so it is available before any loading
//! happens and is identical for InDb and OutDb bands. A single row larger
//! than the budget still goes through, alone.
//!
//! Slices are contiguous row ranges of the input batch (`RecordBatch::slice`
//! is zero-copy), so output order is the input order.
//!
//! That bounds the bytes *created* at the materialization point. Operators
//! that *merge* batches downstream still work by row count and would
//! rebuild an oversized batch from these slices. Since DataFusion 54 the
//! only such operator the planner emits on a linear pipeline is `FilterExec`
//! (its internal coalescer — the separate `CoalesceBatchesExec` is no
//! longer inserted), so the rule turns every `FilterExec` whose schema
//! carries a raster column into a pass-through (see
//! [`PASSTHROUGH_BATCH_SIZE`]). Memory cost is a property of the bytes in a
//! batch, not of the operator: a filter on an id column never reads the
//! raster, but its coalescer would still buffer 8192 rows of pixels. The
//! check is by schema only — a stream whose raster column is still OutDb
//! (cheap references) is included too, which costs nothing beyond skipping
//! the merge of small filtered remainders. `RepartitionExec` reads the
//! session `batch_size` at execute time and is not per-node configurable,
//! so it remains a known gap; with the default partitioning preference the
//! planner places it below this node, over OutDb references.
//!
//! Longer term the merging half belongs upstream: if arrow's
//! `BatchCoalescer` (and DataFusion's `LimitedBatchCoalescer` on top of it)
//! accepted a byte target next to the row target, every merger would become
//! byte-aware on its own and only the materialization points would need
//! Sedona-owned operators. See
//! <https://github.com/apache/datafusion/issues/23385>, which tracks the
//! coalescers' lack of memory awareness.

use std::fmt;
use std::sync::Arc;

use arrow_array::{Array, RecordBatch, StructArray};
use arrow_schema::Schema;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion_common::{Result, internal_err};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_expr::async_scalar_function::AsyncFuncExpr;
use datafusion_physical_expr::{PhysicalExpr, ScalarFunctionExpr};
use datafusion_physical_plan::async_func::AsyncFuncExec;
use datafusion_physical_plan::execution_plan::CardinalityEffect;
use datafusion_physical_plan::filter::FilterExec;
use datafusion_physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures::{StreamExt, stream};
use sedona_common::option::{DEFAULT_RASTER_MAX_BATCH_BYTES, SedonaOptions};
use sedona_common::sedona_internal_datafusion_err;
use sedona_raster::array::RasterStructArray;
use sedona_raster::size::estimated_row_bytes;
use sedona_schema::datatypes::SedonaType;

/// Name of the async UDF whose raster argument sizes the slices. Kept in
/// sync with `sedona_raster_functions::rs_ensure_loaded` (this crate can't
/// depend on it); `ensure_loaded.rs` resolves the same name.
const ENSURE_LOADED_NAME: &str = "rs_ensureloaded";

/// Batch size that turns `FilterExec`'s row-count coalescer into a
/// pass-through.
///
/// `FilterExec` drives a `LimitedBatchCoalescer` whose "biggest coalesce
/// batch size" is
/// `target / 2`: a batch larger than that bypasses coalescing whenever
/// nothing is buffered. With a target of 1 the threshold is 0, so every
/// non-empty batch is forwarded as-is and nothing ever accumulates —
/// byte-bounded slices produced upstream keep their boundaries.
pub const PASSTHROUGH_BATCH_SIZE: usize = 1;

/// Physical optimizer rule that keeps raster batches byte-bounded:
///
/// - swaps DataFusion's `AsyncFuncExec` for an [`EnsureLoadedExec`]
///   wherever the async expressions include `rs_ensureloaded`;
/// - sets the batch size of every `FilterExec` whose output schema carries
///   a raster column to [`PASSTHROUGH_BATCH_SIZE`], so it forwards batches
///   instead of merging them back up to `datafusion.execution.batch_size`
///   rows (its `fetch` and projection are preserved).
///
/// Runs after DataFusion's own physical rules (it is appended via
/// `SessionStateBuilder::with_physical_optimizer_rule`), so it sees the
/// final placement of the async node. Plans without raster columns are
/// untouched.
#[derive(Debug, Default)]
pub struct RasterBatchBudgetRule;

impl PhysicalOptimizerRule for RasterBatchBudgetRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        plan.transform_up(|node| {
            match (
                node.downcast_ref::<AsyncFuncExec>(),
                node.downcast_ref::<FilterExec>(),
            ) {
                (Some(async_exec), _)
                    if async_exec.async_exprs().iter().any(|e| is_ensure_loaded(e)) =>
                {
                    Ok(Transformed::yes(Arc::new(
                        EnsureLoadedExec::from_async_func_exec(async_exec)?,
                    )))
                }
                (_, Some(filter))
                    if filter.batch_size() != PASSTHROUGH_BATCH_SIZE
                        && has_raster_column(&node.schema()) =>
                {
                    Ok(Transformed::yes(Arc::new(
                        filter.with_batch_size(PASSTHROUGH_BATCH_SIZE)?,
                    )))
                }
                _ => Ok(Transformed::no(node)),
            }
        })
        .data()
    }

    fn name(&self) -> &str {
        "sedona.raster_batch_budget"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// Whether any field of `schema` is a Sedona raster: by the
/// `sedona.raster` extension metadata, or structurally by the raster
/// storage type, since DataFusion drops field metadata on async UDF
/// outputs (the `__async_fn_N` columns; see `sd_restore_metadata`).
fn has_raster_column(schema: &Schema) -> bool {
    schema.fields().iter().any(|field| {
        matches!(
            SedonaType::from_storage_field(field),
            Ok(SedonaType::Raster)
        ) || field.data_type() == SedonaType::Raster.storage_type()
    })
}

fn scalar_function(expr: &AsyncFuncExpr) -> Option<&ScalarFunctionExpr> {
    expr.func.downcast_ref::<ScalarFunctionExpr>()
}

fn is_ensure_loaded(expr: &AsyncFuncExpr) -> bool {
    scalar_function(expr).is_some_and(|f| f.fun().name() == ENSURE_LOADED_NAME)
}

/// The raster argument of an `rs_ensureloaded` call.
fn raster_arg(expr: &AsyncFuncExpr) -> Result<&Arc<dyn PhysicalExpr>> {
    scalar_function(expr)
        .and_then(|f| f.args().first())
        .ok_or_else(|| {
            sedona_internal_datafusion_err!(
                "EnsureLoadedExec: {} has no raster argument to size by",
                expr.name()
            )
        })
}

/// Contiguous `[start, end)` row ranges whose summed `estimates` stay within
/// `budget`. A row that alone exceeds the budget gets a range of its own. An
/// empty input yields one empty range so an empty batch still flows through.
fn slice_ranges(estimates: &[u64], budget: u64) -> Vec<(usize, usize)> {
    if estimates.is_empty() {
        return vec![(0, 0)];
    }
    let mut ranges = Vec::new();
    let mut start = 0;
    let mut acc = 0u64;
    for (idx, &bytes) in estimates.iter().enumerate() {
        if idx > start && acc.saturating_add(bytes) > budget {
            ranges.push((start, idx));
            start = idx;
            acc = 0;
        }
        acc = acc.saturating_add(bytes);
    }
    ranges.push((start, estimates.len()));
    ranges
}

/// Byte budget for one slice, from the session's
/// `sedona.raster.max_batch_bytes`. It applies **per batch, per partition**:
/// each partition's stream is sliced independently, so the loaded pixels in
/// flight across a query are roughly `target_partitions × (batches a
/// pipeline holds, typically 2–3) × budget`.
///
/// The rule never consults the memory limit itself. The value read here is
/// whatever the session holds: the 256 MiB crate default, the lower default
/// `SedonaContext` derives at construction when a memory limit is configured
/// (`(memory_limit / target_partitions) / 8`, floored at 16 MiB, capped at
/// 256 MiB — worked examples on the option's docs in `sedona-common`), or an
/// explicit `SET`. `0` disables slicing. A session without the extension (a
/// bare DataFusion context) gets the crate default.
fn budget_bytes(context: &TaskContext) -> u64 {
    let bytes = context
        .session_config()
        .options()
        .extensions
        .get::<SedonaOptions>()
        .map(|opts| opts.raster.max_batch_bytes)
        .unwrap_or(DEFAULT_RASTER_MAX_BATCH_BYTES);
    if bytes == 0 { u64::MAX } else { bytes as u64 }
}

/// DataFusion's `AsyncFuncExec`, re-batched by estimated raster bytes. See
/// the module docs.
#[derive(Debug)]
pub struct EnsureLoadedExec {
    async_exprs: Vec<Arc<AsyncFuncExpr>>,
    input: Arc<dyn ExecutionPlan>,
    /// Indices into `async_exprs` of the `rs_ensureloaded` calls whose
    /// raster arguments size the slices (summed per row when several).
    sized_by: Vec<usize>,
    cache: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl EnsureLoadedExec {
    /// Build from an `AsyncFuncExec`, inheriting its schema and plan
    /// properties verbatim — including the `__async_fn_N` output column
    /// names DataFusion's planner projects on top of this node.
    pub fn from_async_func_exec(exec: &AsyncFuncExec) -> Result<Self> {
        let async_exprs = exec.async_exprs().to_vec();
        let sized_by: Vec<usize> = async_exprs
            .iter()
            .enumerate()
            .filter(|(_, e)| is_ensure_loaded(e))
            .map(|(idx, _)| idx)
            .collect();
        if sized_by.is_empty() {
            return internal_err!(
                "EnsureLoadedExec requires at least one {ENSURE_LOADED_NAME} expression"
            );
        }
        for &idx in &sized_by {
            raster_arg(&async_exprs[idx])?;
        }
        Ok(Self {
            async_exprs,
            input: Arc::clone(exec.input()),
            sized_by,
            cache: Arc::clone(exec.properties()),
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    pub fn try_new(
        async_exprs: Vec<Arc<AsyncFuncExpr>>,
        input: Arc<dyn ExecutionPlan>,
    ) -> Result<Self> {
        Self::from_async_func_exec(&AsyncFuncExec::try_new(async_exprs, input)?)
    }

    pub fn async_exprs(&self) -> &[Arc<AsyncFuncExpr>] {
        &self.async_exprs
    }

    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// Per-row estimated bytes of everything `rs_ensureloaded` will
    /// materialize from `batch`: the sum over the sized expressions of the
    /// metadata-only estimate of their raster argument.
    fn estimate_rows(
        async_exprs: &[Arc<AsyncFuncExpr>],
        sized_by: &[usize],
        batch: &RecordBatch,
    ) -> Result<Vec<u64>> {
        let mut totals = vec![0u64; batch.num_rows()];
        for &idx in sized_by {
            let array = raster_arg(&async_exprs[idx])?
                .evaluate(batch)?
                .into_array(batch.num_rows())?;
            let struct_array = array
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| {
                    sedona_internal_datafusion_err!(
                        "EnsureLoadedExec: expected a Raster (Struct) argument, got {:?}",
                        array.data_type()
                    )
                })?;
            let rasters = RasterStructArray::try_new(struct_array)?;
            for (total, bytes) in totals.iter_mut().zip(estimated_row_bytes(&rasters)?) {
                *total = total.saturating_add(bytes);
            }
        }
        Ok(totals)
    }
}

impl DisplayAs for EnsureLoadedExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        let exprs = self
            .async_exprs
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "EnsureLoadedExec: async_expr=[{exprs}]")
            }
            DisplayFormatType::TreeRender => {
                writeln!(f, "format=async_expr")?;
                writeln!(f, "async_expr={exprs}")
            }
        }
    }
}

impl ExecutionPlan for EnsureLoadedExec {
    fn name(&self) -> &str {
        "EnsureLoadedExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return internal_err!(
                "EnsureLoadedExec expects exactly 1 child, got {}",
                children.len()
            );
        }
        Ok(Arc::new(Self::try_new(
            self.async_exprs.clone(),
            children.remove(0),
        )?))
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    // `benefits_from_input_partitioning` is deliberately left at the default
    // (true), matching DataFusion's `AsyncFuncExec`: the planner then puts any
    // round-robin repartition *below* this node, over cheap OutDb references,
    // so loads run in parallel and no repartition coalescer sits above the
    // byte-bounded output re-merging it.

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::Equal
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(partition, Arc::clone(&context))?;
        let budget = budget_bytes(&context);
        let config_options = Arc::clone(context.session_config().options());
        let async_exprs = Arc::new(self.async_exprs.clone());
        let sized_by = Arc::new(self.sized_by.clone());
        let schema = self.schema();
        let baseline = BaselineMetrics::new(&self.metrics, partition);

        // One input batch fans out into as many output batches as the
        // budget requires; each slice is evaluated in turn so at most one
        // slice's worth of loaded pixels is in flight per partition.
        let output = input.flat_map(move |batch| {
            let batch = match batch {
                Ok(batch) => batch,
                Err(e) => return stream::once(async move { Err(e) }).boxed(),
            };
            let ranges = match Self::estimate_rows(&async_exprs, &sized_by, &batch) {
                Ok(estimates) => slice_ranges(&estimates, budget),
                Err(e) => return stream::once(async move { Err(e) }).boxed(),
            };
            let slices: Vec<RecordBatch> = ranges
                .into_iter()
                .map(|(start, end)| batch.slice(start, end - start))
                .collect();

            let async_exprs = Arc::clone(&async_exprs);
            let schema = Arc::clone(&schema);
            let config_options = Arc::clone(&config_options);
            let baseline = baseline.clone();
            stream::iter(slices)
                .then(move |slice| {
                    let async_exprs = Arc::clone(&async_exprs);
                    let schema = Arc::clone(&schema);
                    let config_options = Arc::clone(&config_options);
                    let baseline = baseline.clone();
                    async move {
                        let mut columns = slice.columns().to_vec();
                        for expr in async_exprs.iter() {
                            let value = expr
                                .invoke_with_args(&slice, Arc::clone(&config_options))
                                .await?;
                            columns.push(value.to_array(slice.num_rows())?);
                        }
                        let out = RecordBatch::try_new(schema, columns)?;
                        baseline.record_output(out.num_rows());
                        Ok(out)
                    }
                })
                .boxed()
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            output,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use arrow_array::{ArrayRef, Int32Array};
    use arrow_schema::{DataType, Field, Schema, SchemaRef};
    use async_trait::async_trait;
    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::prelude::SessionConfig;
    use datafusion_expr::async_udf::{AsyncScalarUDF, AsyncScalarUDFImpl};
    use datafusion_expr::{
        ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
    };
    use datafusion_physical_expr::expressions::col;
    use datafusion_physical_expr::expressions::lit;
    use datafusion_physical_plan::common::collect;
    use sedona_common::option::RasterOptions;
    use sedona_raster::builder::{RasterBuilder, StartBandArgs};
    use sedona_schema::raster::BandDataType;

    /// Stand-in for `RS_EnsureLoaded`: same name, identity on its argument,
    /// records the row count of every invocation.
    #[derive(Debug)]
    struct MockEnsureLoaded {
        signature: Signature,
        calls: Arc<Mutex<Vec<usize>>>,
    }

    impl MockEnsureLoaded {
        fn new(calls: Arc<Mutex<Vec<usize>>>) -> Self {
            Self {
                signature: Signature::any(1, Volatility::Stable),
                calls,
            }
        }
    }

    // DataFusion dedups UDFs by equality/hash; identity by name is enough here.
    impl PartialEq for MockEnsureLoaded {
        fn eq(&self, _other: &Self) -> bool {
            true
        }
    }
    impl Eq for MockEnsureLoaded {}
    impl std::hash::Hash for MockEnsureLoaded {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            ENSURE_LOADED_NAME.hash(state);
        }
    }

    impl ScalarUDFImpl for MockEnsureLoaded {
        fn name(&self) -> &str {
            ENSURE_LOADED_NAME
        }
        fn signature(&self) -> &Signature {
            &self.signature
        }
        fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
            Ok(arg_types[0].clone())
        }
        fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
            internal_err!("async only")
        }
    }

    #[async_trait]
    impl AsyncScalarUDFImpl for MockEnsureLoaded {
        async fn invoke_async_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
            self.calls.lock().unwrap().push(args.number_rows);
            Ok(ColumnarValue::Array(
                args.args[0].to_array(args.number_rows)?,
            ))
        }
    }

    /// Async UDF that is *not* rs_ensureloaded; the rule must leave it alone.
    #[derive(Debug)]
    struct OtherAsync {
        signature: Signature,
    }

    impl PartialEq for OtherAsync {
        fn eq(&self, _other: &Self) -> bool {
            true
        }
    }
    impl Eq for OtherAsync {}
    impl std::hash::Hash for OtherAsync {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            "other_async".hash(state);
        }
    }

    impl ScalarUDFImpl for OtherAsync {
        fn name(&self) -> &str {
            "other_async"
        }
        fn signature(&self) -> &Signature {
            &self.signature
        }
        fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
            Ok(DataType::Int32)
        }
        fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
            internal_err!("async only")
        }
    }

    #[async_trait]
    impl AsyncScalarUDFImpl for OtherAsync {
        async fn invoke_async_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
            // One value per row: DataFusion expands a scalar async result to a
            // length-1 array, so async UDFs must return arrays.
            let rows = args.number_rows;
            Ok(ColumnarValue::Array(Arc::new(Int32Array::from(vec![
                rows as i32;
                rows
            ]))))
        }
    }

    /// One OutDb `[side, side]` UInt8 band per row, so row `i` estimates to
    /// `sides[i]²` bytes without anything to load; `None` is a null row.
    fn raster_batch(sides: &[Option<i64>]) -> (SchemaRef, RecordBatch) {
        let mut b = RasterBuilder::new(sides.len());
        for side in sides {
            let Some(side) = side else {
                b.append_null().unwrap();
                continue;
            };
            b.start_raster_nd(
                &[0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
                &["y", "x"],
                &[*side, *side],
                None,
            )
            .unwrap();
            b.start_band(StartBandArgs {
                outdb_uri: Some("mock://tile"),
                outdb_format: Some("mock"),
                ..StartBandArgs::new(&["y", "x"], &[*side, *side], BandDataType::UInt8)
            })
            .unwrap();
            b.band_data_writer().append_value([0u8; 0]);
            b.finish_band().unwrap();
            b.finish_raster().unwrap();
        }
        let rasters: ArrayRef = Arc::new(b.finish().unwrap());
        let raster_field = SedonaType::Raster.to_storage_field("rast", true).unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("rast", rasters.data_type().clone(), true)
                .with_metadata(raster_field.metadata().clone()),
        ]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![rasters]).unwrap();
        (schema, batch)
    }

    fn async_expr(
        udf: Arc<dyn AsyncScalarUDFImpl>,
        name: &str,
        schema: &Schema,
    ) -> Arc<AsyncFuncExpr> {
        let udf = Arc::new(AsyncScalarUDF::new(udf).into_scalar_udf());
        let func = Arc::new(
            ScalarFunctionExpr::try_new(
                udf,
                vec![col("rast", schema).unwrap()],
                schema,
                Arc::new(ConfigOptions::default()),
            )
            .unwrap(),
        );
        Arc::new(AsyncFuncExpr::try_new(name, func, schema).unwrap())
    }

    fn task_context(max_batch_bytes: usize) -> Arc<TaskContext> {
        let config = SessionConfig::new().with_option_extension(SedonaOptions {
            raster: RasterOptions { max_batch_bytes },
            ..Default::default()
        });
        SessionStateBuilder::new()
            .with_config(config)
            .build()
            .task_ctx()
    }

    /// Build `AsyncFuncExec(rs_ensureloaded(rast))` over `batch`, run the
    /// rule, execute with `max_batch_bytes`, and return the output batches
    /// plus the row counts the mock saw.
    async fn run(
        sides: &[Option<i64>],
        max_batch_bytes: usize,
    ) -> (Vec<RecordBatch>, Vec<usize>, SchemaRef) {
        let (schema, batch) = raster_batch(sides);
        let input =
            MemorySourceConfig::try_new_exec(&[vec![batch]], Arc::clone(&schema), None).unwrap();
        let calls = Arc::new(Mutex::new(Vec::new()));
        let expr = async_expr(
            Arc::new(MockEnsureLoaded::new(Arc::clone(&calls))),
            "__async_fn_0",
            &schema,
        );
        let async_exec = Arc::new(AsyncFuncExec::try_new(vec![expr], input).unwrap());
        let expected_schema = async_exec.schema();

        let plan = RasterBatchBudgetRule
            .optimize(async_exec, &ConfigOptions::default())
            .unwrap();
        assert!(
            plan.downcast_ref::<EnsureLoadedExec>().is_some(),
            "rule must replace AsyncFuncExec, got {}",
            plan.name()
        );
        assert_eq!(
            plan.schema(),
            expected_schema,
            "schema must match AsyncFuncExec's"
        );

        let stream = plan.execute(0, task_context(max_batch_bytes)).unwrap();
        let batches = collect(stream).await.unwrap();
        let calls = calls.lock().unwrap().clone();
        (batches, calls, expected_schema)
    }

    fn row_counts(batches: &[RecordBatch]) -> Vec<usize> {
        batches.iter().map(RecordBatch::num_rows).collect()
    }

    #[test]
    fn slice_ranges_packs_rows_up_to_the_budget() {
        assert_eq!(
            slice_ranges(&[4, 4, 4, 4, 4], 8),
            vec![(0, 2), (2, 4), (4, 5)]
        );
        assert_eq!(slice_ranges(&[1, 1, 1], 100), vec![(0, 3)]);
        // A row over budget goes alone, without dragging its neighbours in.
        assert_eq!(
            slice_ranges(&[1, 50, 1, 1], 8),
            vec![(0, 1), (1, 2), (2, 4)]
        );
        assert_eq!(slice_ranges(&[50], 8), vec![(0, 1)]);
        assert_eq!(slice_ranges(&[], 8), vec![(0, 0)]);
        // Zero-byte rows (nulls) never force a split.
        assert_eq!(slice_ranges(&[0, 0, 8, 0], 8), vec![(0, 4)]);
    }

    #[tokio::test]
    async fn uniform_rasters_are_sliced_to_the_budget() {
        // Six 256-byte rasters under a 600-byte budget → 2 per slice.
        let (batches, calls, _) = run(&[Some(16); 6], 600).await;
        assert_eq!(row_counts(&batches), vec![2, 2, 2]);
        assert_eq!(calls, vec![2, 2, 2]);
    }

    #[tokio::test]
    async fn oversized_rows_go_alone_and_nulls_cost_nothing() {
        // 100, 1024, null, 100, 100 bytes under a 256-byte budget.
        let (batches, calls, _) = run(&[Some(10), Some(32), None, Some(10), Some(10)], 256).await;
        assert_eq!(row_counts(&batches), vec![1, 1, 3]);
        assert_eq!(calls, vec![1, 1, 3]);
        // Output rows are the input rows in order, and the null survives.
        let out: Vec<usize> = batches.iter().map(|b| b.column(1).null_count()).collect();
        assert_eq!(out, vec![0, 0, 1]);
    }

    #[tokio::test]
    async fn zero_budget_disables_slicing() {
        let (batches, calls, _) = run(&[Some(16); 6], 0).await;
        assert_eq!(row_counts(&batches), vec![6]);
        assert_eq!(calls, vec![6]);
    }

    #[tokio::test]
    async fn output_appends_the_async_column_after_the_input_columns() {
        let (batches, _, schema) = run(&[Some(4), Some(4)], 1024).await;
        assert_eq!(schema.fields().len(), 2);
        assert_eq!(schema.field(1).name(), "__async_fn_0");
        let batch = &batches[0];
        assert_eq!(batch.schema(), schema);
        // Identity mock: the appended column equals the raster input.
        assert_eq!(batch.column(0).as_ref(), batch.column(1).as_ref());
    }

    #[tokio::test]
    async fn empty_batch_flows_through() {
        let (batches, calls, _) = run(&[], 1024).await;
        assert_eq!(row_counts(&batches), vec![0]);
        assert_eq!(calls, vec![0]);
    }

    #[tokio::test]
    async fn rule_ignores_async_funcs_without_ensure_loaded() {
        let (schema, batch) = raster_batch(&[Some(4)]);
        let input =
            MemorySourceConfig::try_new_exec(&[vec![batch]], Arc::clone(&schema), None).unwrap();
        let expr = async_expr(
            Arc::new(OtherAsync {
                signature: Signature::any(1, Volatility::Stable),
            }),
            "__async_fn_0",
            &schema,
        );
        let async_exec: Arc<dyn ExecutionPlan> =
            Arc::new(AsyncFuncExec::try_new(vec![expr], input).unwrap());
        let plan = RasterBatchBudgetRule
            .optimize(Arc::clone(&async_exec), &ConfigOptions::default())
            .unwrap();
        assert!(plan.downcast_ref::<AsyncFuncExec>().is_some());
    }

    #[tokio::test]
    async fn mixed_async_funcs_are_sized_by_the_ensure_loaded_argument_only() {
        let (schema, batch) = raster_batch(&[Some(16); 4]);
        let input =
            MemorySourceConfig::try_new_exec(&[vec![batch]], Arc::clone(&schema), None).unwrap();
        let calls = Arc::new(Mutex::new(Vec::new()));
        let exprs = vec![
            async_expr(
                Arc::new(OtherAsync {
                    signature: Signature::any(1, Volatility::Stable),
                }),
                "__async_fn_0",
                &schema,
            ),
            async_expr(
                Arc::new(MockEnsureLoaded::new(Arc::clone(&calls))),
                "__async_fn_1",
                &schema,
            ),
        ];
        let async_exec = Arc::new(AsyncFuncExec::try_new(exprs, input).unwrap());
        let plan = RasterBatchBudgetRule
            .optimize(async_exec, &ConfigOptions::default())
            .unwrap();
        let batches = collect(plan.execute(0, task_context(512)).unwrap())
            .await
            .unwrap();
        assert_eq!(row_counts(&batches), vec![2, 2]);
        assert_eq!(*calls.lock().unwrap(), vec![2, 2]);
        // The other async column is evaluated per slice too (scalar → 2 rows).
        assert_eq!(batches[0].column(1).len(), 2);
    }

    /// `AsyncFuncExec(rs_ensureloaded(rast))` over six 256-byte rasters, and
    /// its schema, for wrapping in the merger under test.
    fn six_raster_async_exec() -> (Arc<AsyncFuncExec>, SchemaRef, Arc<Mutex<Vec<usize>>>) {
        let (schema, batch) = raster_batch(&[Some(16); 6]);
        let input =
            MemorySourceConfig::try_new_exec(&[vec![batch]], Arc::clone(&schema), None).unwrap();
        let calls = Arc::new(Mutex::new(Vec::new()));
        let expr = async_expr(
            Arc::new(MockEnsureLoaded::new(Arc::clone(&calls))),
            "__async_fn_0",
            &schema,
        );
        let exec = Arc::new(AsyncFuncExec::try_new(vec![expr], input).unwrap());
        (exec, schema, calls)
    }

    async fn run_plan(plan: Arc<dyn ExecutionPlan>, max_batch_bytes: usize) -> Vec<usize> {
        let batches = collect(plan.execute(0, task_context(max_batch_bytes)).unwrap())
            .await
            .unwrap();
        row_counts(&batches)
    }

    #[tokio::test]
    async fn filter_over_raster_stream_forwards_slices_instead_of_merging_them() {
        // Baseline: a stock FilterExec above a byte-bounded exec merges the
        // three 2-row slices back into one 6-row batch.
        let (async_exec, _, _) = six_raster_async_exec();
        let budgeted: Arc<dyn ExecutionPlan> =
            Arc::new(EnsureLoadedExec::from_async_func_exec(&async_exec).unwrap());
        let stock = Arc::new(FilterExec::try_new(lit(true), budgeted).unwrap());
        assert_eq!(run_plan(stock, 600).await, vec![6]);

        // With the rule, the filter passes each slice through.
        let (async_exec, _, calls) = six_raster_async_exec();
        let filtered: Arc<dyn ExecutionPlan> =
            Arc::new(FilterExec::try_new(lit(true), async_exec as Arc<dyn ExecutionPlan>).unwrap());
        let plan = RasterBatchBudgetRule
            .optimize(filtered, &ConfigOptions::default())
            .unwrap();
        let filter = plan.downcast_ref::<FilterExec>().expect("filter kept");
        assert_eq!(filter.batch_size(), PASSTHROUGH_BATCH_SIZE);
        assert!(filter.input().downcast_ref::<EnsureLoadedExec>().is_some());
        assert_eq!(run_plan(plan, 600).await, vec![2, 2, 2]);
        assert_eq!(*calls.lock().unwrap(), vec![2, 2, 2]);
    }

    #[tokio::test]
    async fn filters_without_raster_columns_are_left_alone() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap();
        let filter: Arc<dyn ExecutionPlan> =
            Arc::new(FilterExec::try_new(lit(true), input).unwrap());

        let plan = RasterBatchBudgetRule
            .optimize(Arc::clone(&filter), &ConfigOptions::default())
            .unwrap();
        assert!(
            Arc::ptr_eq(&plan, &filter),
            "vector-only plan must be untouched"
        );
        assert_eq!(
            plan.downcast_ref::<FilterExec>().unwrap().batch_size(),
            8192
        );
    }

    #[test]
    fn has_raster_column_recognises_rasters_with_and_without_extension_metadata() {
        // A scan column carries the extension metadata.
        let (raster_schema, _) = raster_batch(&[Some(2)]);
        assert!(has_raster_column(&raster_schema));
        // An async UDF output has the raster storage type but no metadata.
        let (async_exec, _, _) = six_raster_async_exec();
        let bare = Schema::new(vec![
            async_exec
                .schema()
                .field(1)
                .clone()
                .with_metadata(Default::default()),
        ]);
        assert!(bare.field(0).metadata().is_empty());
        assert!(has_raster_column(&bare));
        let plain = Schema::new(vec![Field::new("id", DataType::Int32, false)]);
        assert!(!has_raster_column(&plain));
    }
}
