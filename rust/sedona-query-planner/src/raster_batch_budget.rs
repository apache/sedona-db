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
//!
//! The rule also decides *where* rasters are loaded. The planner usually
//! feeds this node through a round-robin repartition, which deals whole
//! batches, and a raster catalog is often tiny (one Parquet row group, a
//! couple of batches), so all the loading lands on a couple of partitions
//! while the rest idle. The rule therefore hash-partitions the input by
//! raster source (see `partition_by_raster_source`): hashing is per row,
//! so even a two-batch catalog spreads over every partition, and the rows
//! of one raster share a partition, so the chunk cache loads each raster
//! once instead of once per partition that happens to hold some of its rows.

use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use arrow_array::{Array, ArrayRef, RecordBatch, StructArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion_common::{Result, internal_err};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_expr::ColumnarValue;
use datafusion_physical_expr::async_scalar_function::AsyncFuncExpr;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::{PhysicalExpr, ScalarFunctionExpr};
use datafusion_physical_plan::async_func::AsyncFuncExec;
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::execution_plan::CardinalityEffect;
use datafusion_physical_plan::filter::FilterExec;
use datafusion_physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    PlanProperties,
};
use futures::{StreamExt, stream};
use sedona_common::option::{DEFAULT_RASTER_MAX_BATCH_BYTES, SedonaOptions};
use sedona_common::sedona_internal_datafusion_err;
use sedona_raster::array::RasterStructArray;
use sedona_raster::size::estimated_row_bytes;
use sedona_raster::traits::outdb_source;
use sedona_schema::datatypes::SedonaType;

/// Name prefix of the columns that carry an evaluated raster argument (see
/// `EnsureLoadedExec::hoist_raster_args`).
const ENSURE_LOADED_ARG_PREFIX: &str = "__ensure_loaded_arg_";

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
///   wherever the async expressions include `rs_ensureloaded`, and
///   hash-partitions its input by raster source where that is safe (see
///   `partition_by_raster_source`);
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
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        plan.transform_up(|node| {
            match (
                node.downcast_ref::<AsyncFuncExec>(),
                node.downcast_ref::<FilterExec>(),
            ) {
                (Some(async_exec), _)
                    if async_exec.async_exprs().iter().any(|e| is_ensure_loaded(e)) =>
                {
                    let exec = EnsureLoadedExec::from_async_func_exec(async_exec)?;
                    Ok(Transformed::yes(partition_by_raster_source(
                        exec,
                        config.execution.target_partitions,
                    )?))
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
        return Vec::new();
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

/// Where a sized call reads its raster argument from at execute time.
#[derive(Debug, Clone, Copy)]
enum RasterArgSource {
    /// The argument is input column `index`; evaluating it is a clone.
    Input(usize),
    /// The argument is an expression; it is evaluated once per batch into
    /// the `k`-th column appended after the input columns.
    Appended(usize),
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
    /// Where each `sized_by` call's raster argument is read from at
    /// execute time: an input column as is, or a column appended to the
    /// batch by evaluating the argument once. See [`Self::hoist_raster_args`].
    sized_args: Vec<RasterArgSource>,
    /// The input schema followed by one field per appended argument.
    extended_schema: SchemaRef,
    /// `async_exprs` with each sized call's raster argument replaced by a
    /// reference to its appended column; what `execute` actually invokes.
    hoisted: Vec<Arc<AsyncFuncExpr>>,
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
        let input_schema = exec.input().schema();
        let (sized_args, extended_schema, hoisted) =
            Self::hoist_raster_args(&async_exprs, &sized_by, &input_schema)?;
        Ok(Self {
            async_exprs,
            input: Arc::clone(exec.input()),
            sized_by,
            sized_args,
            extended_schema,
            hoisted,
            cache: Self::compute_properties(exec),
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    /// `exec`'s plan properties, except over a raster-source repartition.
    ///
    /// `AsyncFuncExec` passes its input's partitioning through, so it would
    /// advertise `Hash([raster_source(..)], n)`. That is not a distribution
    /// any other plan can match: rows without an out-DB file draw their key
    /// from a running counter (see [`RasterSourceKeyExpr`]), so evaluating
    /// the same expression elsewhere would not reproduce it. The spread is a
    /// load-balancing device, not a guarantee, so this node reports
    /// `UnknownPartitioning` over that many partitions instead, and no parent
    /// (nor a later rewrite of the plan) can rely on it.
    fn compute_properties(exec: &AsyncFuncExec) -> Arc<PlanProperties> {
        match exec.input().output_partitioning() {
            Partitioning::Hash(keys, n)
                if keys
                    .iter()
                    .any(|key| key.downcast_ref::<RasterSourceKeyExpr>().is_some()) =>
            {
                Arc::new(
                    exec.properties()
                        .as_ref()
                        .clone()
                        .with_partitioning(Partitioning::UnknownPartitioning(*n)),
                )
            }
            _ => Arc::clone(exec.properties()),
        }
    }

    /// Decide where each sized call's raster argument is read from, and
    /// rewrite the calls whose argument is not already an input column so
    /// they read it from a column appended to the batch.
    ///
    /// This is the one place expression arguments are turned into columns.
    /// The exec appends them per batch at execute time; the raster-source
    /// spread (`hash_by_raster_source`) calls it for the keyed call alone
    /// and evaluates the same appended field in a `ProjectionExec` below the
    /// repartition instead, so the exec then finds a plain column there.
    /// Appended columns are named `__ensure_loaded_arg_<position>` but always
    /// referenced by position, so a user column of the same name cannot be
    /// mistaken for one.
    ///
    /// The argument has to be evaluated once for sizing anyway. Invoking the
    /// original expression per slice would evaluate it again (`AsyncFuncExpr`
    /// evaluates its children on every chunk), which for a nested argument
    /// such as `RS_FromPath(path)` opens every file for metadata twice, and
    /// for a volatile argument could load a different raster from the one
    /// that was sized. Reading the sized array back through a column costs
    /// a clone. A plain column argument, the common case, already is that
    /// clone and is left untouched so its slices carry nothing extra.
    fn hoist_raster_args(
        async_exprs: &[Arc<AsyncFuncExpr>],
        sized_by: &[usize],
        input_schema: &Schema,
    ) -> Result<(Vec<RasterArgSource>, SchemaRef, Vec<Arc<AsyncFuncExpr>>)> {
        let mut fields: Vec<Arc<Field>> = input_schema.fields().to_vec();
        let mut sized_args = Vec::with_capacity(sized_by.len());
        let mut columns_by_expr: Vec<Option<Column>> = vec![None; async_exprs.len()];
        for &idx in sized_by {
            let arg = raster_arg(&async_exprs[idx])?;
            if let Some(column) = arg.downcast_ref::<Column>() {
                sized_args.push(RasterArgSource::Input(column.index()));
                continue;
            }
            let appended = fields.len() - input_schema.fields().len();
            let name = format!("{ENSURE_LOADED_ARG_PREFIX}{}", fields.len());
            let field = arg
                .return_field(input_schema)?
                .as_ref()
                .clone()
                .with_name(&name);
            columns_by_expr[idx] = Some(Column::new(&name, fields.len()));
            sized_args.push(RasterArgSource::Appended(appended));
            fields.push(Arc::new(field));
        }
        let extended_schema = Arc::new(Schema::new_with_metadata(
            fields,
            input_schema.metadata().clone(),
        ));

        let hoisted = async_exprs
            .iter()
            .zip(columns_by_expr)
            .map(|(expr, column)| match column {
                None => Ok(Arc::clone(expr)),
                Some(column) => {
                    let mut children: Vec<Arc<dyn PhysicalExpr>> =
                        expr.func.children().into_iter().cloned().collect();
                    children[0] = Arc::new(column);
                    let func = Arc::clone(&expr.func).with_new_children(children)?;
                    Ok(Arc::new(AsyncFuncExpr::try_new(
                        expr.name(),
                        func,
                        &extended_schema,
                    )?))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((sized_args, extended_schema, hoisted))
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

    /// `batch` plus one column per appended raster argument, evaluated
    /// once. Sizing reads these columns and the hoisted expressions read
    /// them again per slice, so the loader always sees the array that was
    /// sized. A batch whose sized arguments are all input columns comes back
    /// as is.
    fn extend_with_raster_args(
        async_exprs: &[Arc<AsyncFuncExpr>],
        sized_by: &[usize],
        sized_args: &[RasterArgSource],
        extended_schema: &SchemaRef,
        batch: &RecordBatch,
    ) -> Result<RecordBatch> {
        if sized_args
            .iter()
            .all(|source| matches!(source, RasterArgSource::Input(_)))
        {
            return Ok(batch.clone());
        }
        let mut columns = batch.columns().to_vec();
        for (&idx, source) in sized_by.iter().zip(sized_args) {
            if let RasterArgSource::Appended(_) = source {
                let array = raster_arg(&async_exprs[idx])?
                    .evaluate(batch)?
                    .into_array(batch.num_rows())?;
                columns.push(array);
            }
        }
        Ok(RecordBatch::try_new(Arc::clone(extended_schema), columns)?)
    }

    /// The evaluated raster argument of each sized call, read from the
    /// extended batch.
    fn raster_args(
        sized_args: &[RasterArgSource],
        input_columns: usize,
        extended: &RecordBatch,
    ) -> Vec<ArrayRef> {
        sized_args
            .iter()
            .map(|source| match *source {
                RasterArgSource::Input(idx) => Arc::clone(extended.column(idx)),
                RasterArgSource::Appended(k) => Arc::clone(extended.column(input_columns + k)),
            })
            .collect()
    }

    /// Per-row estimated bytes of everything `rs_ensureloaded` will
    /// materialize: the sum over the sized calls' evaluated raster
    /// arguments of the metadata-only estimate.
    fn estimate_rows(raster_args: &[ArrayRef], num_rows: usize) -> Result<Vec<u64>> {
        let mut totals = vec![0u64; num_rows];
        for array in raster_args {
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
        let sized_args = Arc::new(self.sized_args.clone());
        let hoisted = Arc::new(self.hoisted.clone());
        let extended_schema = Arc::clone(&self.extended_schema);
        let schema = self.schema();
        let baseline = BaselineMetrics::new(&self.metrics, partition);

        // One input batch fans out into as many output batches as the
        // budget requires; each slice is evaluated in turn so at most one
        // slice's worth of loaded pixels is in flight per partition. The
        // sized raster arguments are evaluated once here, appended to the
        // batch, and read back by column from every slice (see `hoisted`).
        let output = input.flat_map(move |batch| {
            let batch = match batch {
                Ok(batch) => batch,
                Err(e) => return stream::once(async move { Err(e) }).boxed(),
            };
            // An empty batch has nothing to size or load. Invoking on it
            // would fail: `AsyncFuncExpr` chunks by `ideal_batch_size` and
            // concatenates the chunk results, and zero rows make zero chunks.
            if batch.num_rows() == 0 {
                return stream::empty().boxed();
            }
            let extended = match Self::extend_with_raster_args(
                &async_exprs,
                &sized_by,
                &sized_args,
                &extended_schema,
                &batch,
            ) {
                Ok(extended) => extended,
                Err(e) => return stream::once(async move { Err(e) }).boxed(),
            };
            let input_columns = batch.num_columns();
            let raster_args = Self::raster_args(&sized_args, input_columns, &extended);
            let ranges = match Self::estimate_rows(&raster_args, batch.num_rows()) {
                Ok(estimates) => slice_ranges(&estimates, budget),
                Err(e) => return stream::once(async move { Err(e) }).boxed(),
            };
            let slices: Vec<RecordBatch> = ranges
                .into_iter()
                .map(|(start, end)| extended.slice(start, end - start))
                .collect();

            let hoisted = Arc::clone(&hoisted);
            let schema = Arc::clone(&schema);
            let config_options = Arc::clone(&config_options);
            let baseline = baseline.clone();
            stream::iter(slices)
                .then(move |slice| {
                    let hoisted = Arc::clone(&hoisted);
                    let schema = Arc::clone(&schema);
                    let config_options = Arc::clone(&config_options);
                    let baseline = baseline.clone();
                    async move {
                        let mut columns = slice.columns()[..input_columns].to_vec();
                        for expr in hoisted.iter() {
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

/// Hash-partition `exec`'s input by the raster source of its first sized
/// call's raster argument (see [`RasterSourceKeyExpr`]), returning the plan
/// that replaces `exec`.
///
/// Why hash rather than the round-robin the planner already put below this
/// node: round-robin deals whole batches, and a raster catalog is often one
/// or two batches, so only one or two partitions ever load anything. A hash
/// is computed per row, so it spreads any number of batches over every
/// partition, and it keeps all rows of one raster in one partition. That
/// locality is what keeps a load per raster: the chunk cache then sees each
/// raster's rows in order on one stream instead of several partitions
/// missing on the same raster at once.
///
/// The key is the *first* sized call's argument. When several
/// `rs_ensureloaded` calls share a node they usually read rasters from the
/// same catalog row, and one key is all a partitioning can follow.
///
/// `exec` is returned unchanged when repartitioning could only cost, or
/// could break a requirement of a parent. The rule runs after
/// `EnforceDistribution`, `EnforceSorting` and `SanityCheckPlan`, so
/// whatever this changes is not checked again:
///
/// - a single target partition: nothing to spread over;
/// - an input with an ordering: a parent may rely on it, and this node
///   otherwise maintains it;
/// - an input already hash partitioned (for example the probe side of a
///   partitioned join): a parent may require exactly that distribution, so
///   it keeps the spread it already has.
///
/// The planner's own repartitioning is replaced rather than stacked on: a
/// round-robin (`RepartitionExec(RoundRobinBatch)` without
/// `preserve_order`) or a merge (`CoalescePartitionsExec` without a fetch)
/// directly below this node is dropped and its input hashed instead, so the
/// rows cross one repartition, not two (see [`spread_source`]).
///
/// A single input partition is spread too, then merged back with a
/// `CoalescePartitionsExec` so the output is still one partition. The
/// planner leaves an input that small when its statistics say it fits in
/// one batch (a catalog of a few thousand rasters does) or when a parent
/// requires a single partition (a `CollectLeft` build side, a global
/// limit); either way the loads are the expensive part and run in parallel,
/// and whatever is above still sees one partition.
///
/// The spread has `max(target_partitions, input partitions)` partitions, so
/// an input wider than the target (a `UnionExec`, say) keeps its width.
///
/// The rule does not try to detect an input that is already well spread:
/// the plan cannot tell how many batches a partition will deliver, and the
/// cost of the repartition is one hash of rows that are still metadata-only
/// OutDb references (about 0.9 s for 6M joined rows, a few milliseconds for
/// a ten-thousand-raster catalog), while the locality it adds is worth
/// having either way.
fn partition_by_raster_source(
    exec: EnsureLoadedExec,
    target_partitions: usize,
) -> Result<Arc<dyn ExecutionPlan>> {
    let input = Arc::clone(exec.input());
    if target_partitions <= 1
        || input.output_ordering().is_some()
        || matches!(input.output_partitioning(), Partitioning::Hash(_, _))
    {
        return Ok(Arc::new(exec));
    }

    let input_partitions = input.output_partitioning().partition_count();
    let source = spread_source(&input);
    let partitions = target_partitions
        .max(input_partitions)
        .max(source.output_partitioning().partition_count());
    let spread = hash_by_raster_source(exec, source, partitions)?;
    if input_partitions == 1 {
        Ok(Arc::new(CoalescePartitionsExec::new(spread)))
    } else {
        Ok(spread)
    }
}

/// The plan to hash for [`partition_by_raster_source`]: `input` itself, or
/// the input of a round-robin or merge directly below the exec. Hashing
/// redistributes every row anyway, so a round-robin under it is a wasted
/// hop, and a merge under it is undone by the hash (the merge back to one
/// partition happens above the exec instead).
fn spread_source(input: &Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    if let Some(repartition) = input.downcast_ref::<RepartitionExec>()
        && matches!(repartition.partitioning(), Partitioning::RoundRobinBatch(_))
        && !repartition.preserve_order()
    {
        return Arc::clone(repartition.input());
    }
    if let Some(merge) = input.downcast_ref::<CoalescePartitionsExec>()
        && merge.fetch().is_none()
    {
        return Arc::clone(merge.input());
    }
    Arc::clone(input)
}

/// The repartition itself, for [`partition_by_raster_source`]: `source`
/// (which has `exec`'s input schema) hashed into `partitions` by the keyed
/// call's raster argument, with `exec` rebuilt on top.
///
/// When that argument is an expression rather than a column (for example
/// `rs_ensureloaded(RS_FromPath(path))`), it is evaluated into a column by a
/// `ProjectionExec` below the repartition, the hash and the call both read
/// that column, and a projection above drops it again so the output schema
/// is unchanged. Hashing the expression itself would evaluate it a second
/// time, and for `RS_FromPath` that repeats every file open. The column is
/// the one [`EnsureLoadedExec::hoist_raster_args`] would append inside the
/// exec, so both paths share one rewrite. Projecting means the argument is
/// evaluated before the repartition, on the input's partitions (for a
/// single-partition input, on that one partition), which is the price of
/// evaluating it once. Other calls keep their arguments and are evaluated
/// inside the exec, after the spread. An argument shared by several calls
/// has already been extracted into a column by DataFusion's common
/// subexpression elimination, so it takes the plain column path.
fn hash_by_raster_source(
    exec: EnsureLoadedExec,
    source: Arc<dyn ExecutionPlan>,
    partitions: usize,
) -> Result<Arc<dyn ExecutionPlan>> {
    let key_call = exec.sized_by[0];
    let source_schema = source.schema();
    let (key_source, extended_schema, hoisted) =
        EnsureLoadedExec::hoist_raster_args(&exec.async_exprs, &[key_call], &source_schema)?;
    let hash = |input: Arc<dyn ExecutionPlan>, key_column: usize| {
        let key = Column::new(input.schema().field(key_column).name(), key_column);
        RepartitionExec::try_new(
            input,
            Partitioning::Hash(
                vec![Arc::new(RasterSourceKeyExpr::new(Arc::new(key)))],
                partitions,
            ),
        )
    };

    let input_columns = source_schema.fields().len();
    let key_column = match key_source[0] {
        RasterArgSource::Input(idx) => {
            let repartition = hash(source, idx)?;
            return Ok(Arc::new(EnsureLoadedExec::try_new(
                exec.async_exprs,
                Arc::new(repartition),
            )?));
        }
        RasterArgSource::Appended(k) => input_columns + k,
    };

    let column = |idx: usize, schema: &Schema| -> (Arc<dyn PhysicalExpr>, String) {
        let name = schema.field(idx).name();
        (Arc::new(Column::new(name, idx)), name.to_string())
    };
    let key_arg = Arc::clone(raster_arg(&exec.async_exprs[key_call])?);
    let projected = Arc::new(ProjectionExec::try_new(
        (0..input_columns)
            .map(|idx| column(idx, &source_schema))
            .chain([(
                key_arg,
                extended_schema.field(key_column).name().to_string(),
            )]),
        source,
    )?);
    let repartition = Arc::new(hash(projected, key_column)?);
    let loaded = Arc::new(EnsureLoadedExec::try_new(hoisted, repartition)?);

    // Drop the projected argument: the input columns, then the async outputs
    // one position further right than they are in the loaded schema.
    let loaded_schema = loaded.schema();
    let output = (0..input_columns)
        .chain(key_column + 1..loaded_schema.fields().len())
        .map(|idx| column(idx, &loaded_schema));
    Ok(Arc::new(ProjectionExec::try_new(output, loaded)?))
}

/// Per-row partitioning key: which file a raster reads its pixels from.
///
/// Evaluates its child (a raster) and returns, per row, a `UInt64` hash of
/// the first band's `outdb_uri` reduced to its source with
/// [`outdb_source`] (a trailing `#band=N` stripped), so every band and every
/// row reading the same file shares a key and therefore a partition. Rows
/// with no out-DB URI (in-db rasters, null rasters, rasters without bands)
/// have no file to keep together; they get values from a running counter
/// instead, so they spread like a round-robin rather than all hashing to one
/// partition.
///
/// That counter makes the expression non-deterministic: the same row can
/// get a different key on another evaluation. It therefore reports itself
/// volatile, and an [`EnsureLoadedExec`] above the repartition advertises
/// `UnknownPartitioning` rather than this hash (see
/// `EnsureLoadedExec::compute_properties`). Hashing fallback rows
/// deterministically was the alternative, but there is nothing cheap to
/// hash: an in-db raster's identity is its pixels, and a position in the
/// batch is no more reproducible than the counter. Equality and `Hash`
/// ignore the counter, which only matters for comparing plans.
#[derive(Debug)]
pub struct RasterSourceKeyExpr {
    arg: Arc<dyn PhysicalExpr>,
    fallback: Arc<AtomicU64>,
}

impl RasterSourceKeyExpr {
    pub fn new(arg: Arc<dyn PhysicalExpr>) -> Self {
        Self {
            arg,
            fallback: Arc::new(AtomicU64::new(0)),
        }
    }

    fn keys(&self, rasters: &RasterStructArray) -> Result<UInt64Array> {
        let mut out = Vec::with_capacity(rasters.len());
        let mut without_source = 0u64;
        // Raster rows arrive in runs (a join emits a raster's matches
        // together), so reuse the previous hash when the source repeats.
        // The cache must stay keyed on the *stripped* source: then the
        // bands of one file (`a.tif#band=1`, `a.tif#band=2`) share a run
        // as well as a key.
        let mut last: Option<(&str, u64)> = None;
        for i in 0..rasters.len() {
            let key = match rasters.band_outdb_uri(i, 0).map(outdb_source) {
                Some(source) => match last {
                    Some((prev, hash)) if prev == source => Some(hash),
                    _ => {
                        let mut hasher = DefaultHasher::new();
                        source.hash(&mut hasher);
                        let hash = hasher.finish();
                        last = Some((source, hash));
                        Some(hash)
                    }
                },
                None => {
                    without_source += 1;
                    None
                }
            };
            out.push(key);
        }
        // One atomic add per batch for all rows without a source.
        let mut next = self
            .fallback
            .fetch_add(without_source, AtomicOrdering::Relaxed);
        Ok(out
            .into_iter()
            .map(|key| {
                key.unwrap_or_else(|| {
                    let key = next;
                    next = next.wrapping_add(1);
                    key
                })
            })
            .collect())
    }
}

impl fmt::Display for RasterSourceKeyExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "raster_source({})", self.arg)
    }
}

impl PartialEq for RasterSourceKeyExpr {
    fn eq(&self, other: &Self) -> bool {
        self.arg.eq(&other.arg)
    }
}

impl Eq for RasterSourceKeyExpr {}

impl Hash for RasterSourceKeyExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.arg.hash(state);
    }
}

impl PhysicalExpr for RasterSourceKeyExpr {
    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(DataType::UInt64)
    }

    fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
        Ok(false)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let array = self.arg.evaluate(batch)?.into_array(batch.num_rows())?;
        let rasters = array
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| {
                sedona_internal_datafusion_err!(
                    "raster_source: expected a Raster (Struct) argument, got {:?}",
                    array.data_type()
                )
            })?;
        let rasters = RasterStructArray::try_new(rasters)?;
        Ok(ColumnarValue::Array(Arc::new(self.keys(&rasters)?)))
    }

    /// Fallback keys come from a counter; see the type docs.
    fn is_volatile_node(&self) -> bool {
        true
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.arg]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        if children.len() != 1 {
            return internal_err!("raster_source expects 1 child, got {}", children.len());
        }
        Ok(Arc::new(Self {
            arg: children.remove(0),
            fallback: Arc::clone(&self.fallback),
        }))
    }

    fn fmt_sql(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "raster_source(")?;
        self.arg.fmt_sql(f)?;
        write!(f, ")")
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
    use datafusion::datasource::source::DataSourceExec;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::prelude::SessionConfig;
    use datafusion_expr::async_udf::{AsyncScalarUDF, AsyncScalarUDFImpl};
    use datafusion_expr::{
        ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
    };
    use datafusion_physical_expr::expressions::col;
    use datafusion_physical_expr::expressions::lit;
    use datafusion_physical_expr::{LexOrdering, PhysicalSortExpr};
    use datafusion_physical_plan::common::collect;
    use datafusion_physical_plan::sorts::sort::SortExec;
    use datafusion_physical_plan::union::UnionExec;
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
        /// The production value: the exec must behave with the chunked
        /// invocation path, including on zero-row input.
        fn ideal_batch_size(&self) -> Option<usize> {
            Some(1024)
        }

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
            raster: RasterOptions {
                max_batch_bytes,
                ..Default::default()
            },
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

        // One target partition: the test observes a single stream's slices.
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 1;
        let plan = RasterBatchBudgetRule.optimize(async_exec, &config).unwrap();
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
        assert_eq!(slice_ranges(&[], 8), Vec::<(usize, usize)>::new());
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
    async fn empty_batches_emit_nothing_and_never_invoke() {
        // The mock chunks by `ideal_batch_size` like the real UDF, so an
        // invocation on zero rows would fail to concatenate zero chunks.
        let (batches, calls, _) = run(&[], 1024).await;
        assert!(
            row_counts(&batches).is_empty(),
            "{:?}",
            row_counts(&batches)
        );
        assert!(calls.is_empty());
    }

    #[tokio::test]
    async fn raster_argument_is_evaluated_once() {
        // A nested argument such as `RS_FromPath(path)` must be evaluated
        // once per batch, for sizing, and that array reused for the load;
        // a second evaluation would repeat its I/O and, if volatile, load a
        // different raster from the one that was sized.
        use std::sync::atomic::{AtomicUsize, Ordering};

        let (schema, batch) = raster_batch(&[Some(4), Some(4), Some(4)]);
        let calls = Arc::new(AtomicUsize::new(0));
        let observed_calls = Arc::clone(&calls);
        let raster_type = schema.field(0).data_type().clone();
        let counted_udf = datafusion_expr::create_udf(
            "counted_raster",
            vec![raster_type.clone()],
            raster_type,
            Volatility::Volatile,
            Arc::new(move |args| {
                observed_calls.fetch_add(1, Ordering::SeqCst);
                Ok(args[0].clone())
            }),
        );
        let counted_arg = Arc::new(
            ScalarFunctionExpr::try_new(
                Arc::new(counted_udf),
                vec![col("rast", &schema).unwrap()],
                &schema,
                Arc::new(ConfigOptions::default()),
            )
            .unwrap(),
        );
        let expr = async_expr(
            Arc::new(MockEnsureLoaded::new(Arc::new(Mutex::new(Vec::new())))),
            "__async_fn_0",
            &schema,
        );
        let nested = Arc::clone(&expr.func)
            .with_new_children(vec![counted_arg])
            .unwrap();
        let expr = Arc::new(AsyncFuncExpr::try_new("__async_fn_0", nested, &schema).unwrap());
        let input =
            MemorySourceConfig::try_new_exec(&[vec![batch]], Arc::clone(&schema), None).unwrap();
        let stock: Arc<dyn ExecutionPlan> =
            Arc::new(AsyncFuncExec::try_new(vec![expr], input).unwrap());
        // One target partition: the test observes a single stream's slices.
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 1;
        let bounded = RasterBatchBudgetRule
            .optimize(Arc::clone(&stock), &config)
            .unwrap();

        // Three 16-byte rows under a 16-byte budget: the bounded plan cuts
        // three slices, and the argument must still be evaluated once.
        for (name, plan, rows) in [
            ("AsyncFuncExec", stock, vec![3]),
            ("EnsureLoadedExec", bounded, vec![1, 1, 1]),
        ] {
            calls.store(0, Ordering::SeqCst);
            let batches = collect(plan.execute(0, task_context(16)).unwrap())
                .await
                .unwrap();
            assert_eq!(row_counts(&batches), rows, "{name}");
            assert_eq!(calls.load(Ordering::SeqCst), 1, "{name}");
        }
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
        // One target partition: the test observes a single stream's slices.
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 1;
        let plan = RasterBatchBudgetRule.optimize(async_exec, &config).unwrap();
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
        // One target partition: the test observes a single stream's slices.
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 1;
        let plan = RasterBatchBudgetRule.optimize(filtered, &config).unwrap();
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

    /// `id` (the row number) and `rast`: one OutDb `[2, 2]` UInt8 band per
    /// row, reading `uris[id]`.
    fn outdb_raster_batch(uris: &[&str]) -> (SchemaRef, RecordBatch) {
        let mut b = RasterBuilder::new(uris.len());
        for uri in uris {
            b.start_raster_nd(&[0.0, 1.0, 0.0, 0.0, 0.0, -1.0], &["y", "x"], &[2, 2], None)
                .unwrap();
            b.start_band(StartBandArgs {
                outdb_uri: Some(uri),
                outdb_format: Some("mock"),
                ..StartBandArgs::new(&["y", "x"], &[2, 2], BandDataType::UInt8)
            })
            .unwrap();
            b.band_data_writer().append_value([0u8; 0]);
            b.finish_band().unwrap();
            b.finish_raster().unwrap();
        }
        let rasters: ArrayRef = Arc::new(b.finish().unwrap());
        let raster_field = SedonaType::Raster.to_storage_field("rast", true).unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("rast", rasters.data_type().clone(), true)
                .with_metadata(raster_field.metadata().clone()),
        ]));
        let ids = Arc::new(Int32Array::from_iter_values(0..uris.len() as i32));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![ids, rasters]).unwrap();
        (schema, batch)
    }

    #[test]
    fn raster_source_key_groups_rows_by_file_and_spreads_rows_without_one() {
        // Rows 0-3 and 6-9 are OutDb, rows 4-5 InDb; then a null raster and
        // a raster without bands.
        let mut b = RasterBuilder::new(12);
        for uri in [
            Some("s3://b/a.tif#band=1"),
            Some("s3://b/a.tif#band=2"),
            Some("s3://b/b.tif"),
            Some("s3://b/a.tif"),
            None,
            None,
            Some("s3://b/x.zarr#var=a"),
            Some("s3://b/x.zarr#var=b"),
            Some("s3://b/p.tif#anchor#band=1"),
            Some("s3://b/p.tif#anchor"),
        ] {
            b.start_raster_nd(&[0.0, 1.0, 0.0, 0.0, 0.0, -1.0], &["y", "x"], &[2, 2], None)
                .unwrap();
            b.start_band(StartBandArgs {
                outdb_uri: uri,
                outdb_format: uri.map(|_| "mock"),
                ..StartBandArgs::new(&["y", "x"], &[2, 2], BandDataType::UInt8)
            })
            .unwrap();
            if uri.is_some() {
                b.band_data_writer().append_value([0u8; 0]);
            } else {
                b.band_data_writer().append_value([0u8; 4]);
            }
            b.finish_band().unwrap();
            b.finish_raster().unwrap();
        }
        b.append_null().unwrap();
        b.start_raster_nd(&[0.0, 1.0, 0.0, 0.0, 0.0, -1.0], &["y", "x"], &[2, 2], None)
            .unwrap();
        b.finish_raster().unwrap();
        let rasters: ArrayRef = Arc::new(b.finish().unwrap());
        let schema = Schema::new(vec![Field::new("rast", rasters.data_type().clone(), true)]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![rasters]).unwrap();
        let key = RasterSourceKeyExpr::new(col("rast", &schema).unwrap());
        assert!(key.is_volatile_node());
        let keys = key.evaluate(&batch).unwrap().into_array(12).unwrap();
        let keys = keys
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .values()
            .to_vec();

        // Every band and every row of one file share a key; other files differ.
        assert_eq!(keys[0], keys[1]);
        assert_eq!(keys[0], keys[3]);
        assert_ne!(keys[0], keys[2]);
        // Only a trailing `#band=N` is stripped: other fragments name
        // different sources, and an earlier `#anchor` stays.
        assert_ne!(keys[6], keys[7]);
        assert_eq!(keys[8], keys[9]);
        // Rows without a file draw consecutive counter values, one batch at
        // a time.
        let mut fallback = vec![keys[4], keys[5], keys[10], keys[11]];
        fallback.sort();
        assert_eq!(fallback, (fallback[0]..fallback[0] + 4).collect::<Vec<_>>());
        // A sliced batch reads its own rows, and an empty one yields no keys.
        let sliced = key
            .evaluate(&batch.slice(2, 2))
            .unwrap()
            .into_array(2)
            .unwrap();
        assert_eq!(
            sliced
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .values()
                .to_vec(),
            keys[2..4].to_vec()
        );
        let empty = key
            .evaluate(&batch.slice(0, 0))
            .unwrap()
            .into_array(0)
            .unwrap();
        assert_eq!(empty.len(), 0);
    }

    #[tokio::test]
    async fn ensure_loaded_input_is_hash_partitioned_by_raster_source() {
        // Sixteen rasters, four rows each, interleaved over two input
        // partitions of one batch each: a whole-batch round-robin could only
        // ever keep two partitions busy.
        let uris: Vec<String> = (0..64)
            .map(|i| format!("s3://b/r{}.tif#band={}", i % 16, i / 16 + 1))
            .collect();
        let uris: Vec<&str> = uris.iter().map(String::as_str).collect();
        let (schema, first) = outdb_raster_batch(&uris[..32]);
        let (_, second) = outdb_raster_batch(&uris[32..]);
        let input = MemorySourceConfig::try_new_exec(
            &[vec![first], vec![second]],
            Arc::clone(&schema),
            None,
        )
        .unwrap();
        let calls = Arc::new(Mutex::new(Vec::new()));
        let expr = async_expr(
            Arc::new(MockEnsureLoaded::new(Arc::clone(&calls))),
            "__async_fn_0",
            &schema,
        );
        let async_exec = Arc::new(AsyncFuncExec::try_new(vec![expr], input).unwrap());
        let expected_schema = async_exec.schema();
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 4;
        let plan = RasterBatchBudgetRule.optimize(async_exec, &config).unwrap();

        let exec = plan.downcast_ref::<EnsureLoadedExec>().unwrap();
        let repartition = exec.input().downcast_ref::<RepartitionExec>().unwrap();
        let Partitioning::Hash(keys, 4) = repartition.partitioning() else {
            panic!("expected a 4-way hash, got {}", repartition.partitioning());
        };
        assert_eq!(keys[0].to_string(), "raster_source(rast@1)");
        assert_eq!(plan.schema(), expected_schema);
        // The counter-drawn keys of rows without a file are not reproducible,
        // so the hash is not advertised to parents.
        assert!(matches!(
            plan.output_partitioning(),
            Partitioning::UnknownPartitioning(4)
        ));
        assert!(plan.output_ordering().is_none());

        let key = RasterSourceKeyExpr::new(col("rast", &schema).unwrap());
        let ctx = task_context(DEFAULT_RASTER_MAX_BATCH_BYTES);
        let mut partition_of = std::collections::HashMap::new();
        let mut rows = 0;
        let mut busy = 0;
        for partition in 0..4 {
            let batches = collect(plan.execute(partition, Arc::clone(&ctx)).unwrap())
                .await
                .unwrap();
            let mut here = 0;
            for batch in &batches {
                let keys = key
                    .evaluate(batch)
                    .unwrap()
                    .into_array(batch.num_rows())
                    .unwrap();
                for &k in keys
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap()
                    .values()
                {
                    // Every row of a raster, from either input partition,
                    // lands in the same output partition.
                    assert_eq!(*partition_of.entry(k).or_insert(partition), partition);
                    here += 1;
                }
            }
            rows += here;
            busy += usize::from(here > 0);
        }
        assert_eq!(rows, 64);
        assert_eq!(partition_of.len(), 16);
        assert!(busy > 2, "only {busy} of 4 partitions received rows");
        assert_eq!(calls.lock().unwrap().iter().sum::<usize>(), 64);
    }

    #[test]
    fn raster_source_partitioning_is_skipped_where_it_could_break_a_parent() {
        let (schema, batch) = outdb_raster_batch(&["s3://b/a.tif", "s3://b/b.tif"]);
        let two_partitions = MemorySourceConfig::try_new_exec(
            &[vec![batch.clone()], vec![batch]],
            Arc::clone(&schema),
            None,
        )
        .unwrap();
        let by_id =
            LexOrdering::new([PhysicalSortExpr::new_default(col("id", &schema).unwrap())]).unwrap();
        let cases: Vec<(&str, Arc<dyn ExecutionPlan>, usize)> = vec![
            ("one target partition", two_partitions.clone(), 1),
            (
                "ordered input",
                Arc::new(
                    SortExec::new(by_id, two_partitions.clone()).with_preserve_partitioning(true),
                ),
                4,
            ),
            (
                "hash-partitioned input",
                Arc::new(
                    RepartitionExec::try_new(
                        two_partitions.clone(),
                        Partitioning::Hash(vec![col("id", &schema).unwrap()], 4),
                    )
                    .unwrap(),
                ),
                4,
            ),
        ];
        for (name, input, target_partitions) in cases {
            let expr = async_expr(
                Arc::new(MockEnsureLoaded::new(Arc::new(Mutex::new(Vec::new())))),
                "__async_fn_0",
                &schema,
            );
            let async_exec =
                Arc::new(AsyncFuncExec::try_new(vec![expr], Arc::clone(&input)).unwrap());
            let mut config = ConfigOptions::default();
            config.execution.target_partitions = target_partitions;
            let plan = RasterBatchBudgetRule.optimize(async_exec, &config).unwrap();
            let exec = plan
                .downcast_ref::<EnsureLoadedExec>()
                .unwrap_or_else(|| panic!("{name}: got {}", plan.name()));
            assert!(Arc::ptr_eq(exec.input(), &input), "{name}");
        }
    }

    #[tokio::test]
    async fn single_input_partition_is_spread_and_merged_back() {
        // The planner leaves a small input (or one under a single-partition
        // requirement) on one partition; the loads still spread, and the
        // output is one partition again.
        let uris: Vec<String> = (0..16).map(|i| format!("s3://b/r{i}.tif")).collect();
        let uris: Vec<&str> = uris.iter().map(String::as_str).collect();
        let (schema, batch) = outdb_raster_batch(&uris);
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
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 4;
        let plan = RasterBatchBudgetRule.optimize(async_exec, &config).unwrap();

        assert_eq!(plan.schema(), expected_schema);
        assert_eq!(plan.output_partitioning().partition_count(), 1);
        let merge = plan.downcast_ref::<CoalescePartitionsExec>().unwrap();
        let exec = merge.input().downcast_ref::<EnsureLoadedExec>().unwrap();
        let repartition = exec.input().downcast_ref::<RepartitionExec>().unwrap();
        assert!(matches!(
            repartition.partitioning(),
            Partitioning::Hash(_, 4)
        ));

        let batches = collect(
            plan.execute(0, task_context(DEFAULT_RASTER_MAX_BATCH_BYTES))
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(row_counts(&batches).iter().sum::<usize>(), 16);
        // One invocation per busy partition, not one for the whole input.
        assert!(calls.lock().unwrap().len() > 1);
    }

    #[tokio::test]
    async fn expression_raster_argument_is_projected_below_the_repartition() {
        // Hashing `RS_FromPath(path)`-style arguments directly would evaluate
        // them twice (key, then load). The rule projects the argument into a
        // column below the repartition instead, so it is evaluated once per
        // input batch, and drops that column above the exec.
        use std::sync::atomic::{AtomicUsize, Ordering};

        let uris: Vec<String> = (0..16).map(|i| format!("s3://b/r{i}.tif")).collect();
        let uris: Vec<&str> = uris.iter().map(String::as_str).collect();
        let (schema, first) = outdb_raster_batch(&uris[..8]);
        let (_, second) = outdb_raster_batch(&uris[8..]);
        let input = MemorySourceConfig::try_new_exec(
            &[vec![first], vec![second]],
            Arc::clone(&schema),
            None,
        )
        .unwrap();
        let evaluations = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&evaluations);
        let raster_type = schema.field(1).data_type().clone();
        let counted_udf = datafusion_expr::create_udf(
            "counted_raster",
            vec![raster_type.clone()],
            raster_type,
            Volatility::Volatile,
            Arc::new(move |args| {
                observed.fetch_add(1, Ordering::SeqCst);
                Ok(args[0].clone())
            }),
        );
        let counted_arg = Arc::new(
            ScalarFunctionExpr::try_new(
                Arc::new(counted_udf),
                vec![col("rast", &schema).unwrap()],
                &schema,
                Arc::new(ConfigOptions::default()),
            )
            .unwrap(),
        );
        let calls = Arc::new(Mutex::new(Vec::new()));
        let expr = async_expr(
            Arc::new(MockEnsureLoaded::new(Arc::clone(&calls))),
            "__async_fn_0",
            &schema,
        );
        let nested = Arc::clone(&expr.func)
            .with_new_children(vec![counted_arg])
            .unwrap();
        let expr = Arc::new(AsyncFuncExpr::try_new("__async_fn_0", nested, &schema).unwrap());
        let async_exec = Arc::new(AsyncFuncExec::try_new(vec![expr], input).unwrap());
        let expected_schema = async_exec.schema();
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 4;
        let plan = RasterBatchBudgetRule.optimize(async_exec, &config).unwrap();

        // Projection (drop the argument) > EnsureLoadedExec > hash
        // RepartitionExec > Projection (evaluate the argument).
        assert_eq!(plan.schema(), expected_schema);
        let top = plan.downcast_ref::<ProjectionExec>().unwrap();
        let exec = top.input().downcast_ref::<EnsureLoadedExec>().unwrap();
        assert_eq!(
            exec.async_exprs()[0].func.to_string(),
            "rs_ensureloaded(__ensure_loaded_arg_2@2)"
        );
        let repartition = exec.input().downcast_ref::<RepartitionExec>().unwrap();
        let Partitioning::Hash(keys, 4) = repartition.partitioning() else {
            panic!("expected a 4-way hash, got {}", repartition.partitioning());
        };
        assert_eq!(
            keys[0].to_string(),
            "raster_source(__ensure_loaded_arg_2@2)"
        );
        let below = repartition
            .input()
            .downcast_ref::<ProjectionExec>()
            .unwrap();
        assert!(matches!(
            plan.output_partitioning(),
            Partitioning::UnknownPartitioning(4)
        ));
        // The raster column keeps its extension metadata through both
        // projections.
        let raster_metadata = SedonaType::Raster
            .to_storage_field("rast", true)
            .unwrap()
            .metadata()
            .clone();
        assert!(!raster_metadata.is_empty());
        assert_eq!(below.schema().field(1).metadata(), &raster_metadata);
        assert_eq!(plan.schema().field(1).metadata(), &raster_metadata);

        let ctx = task_context(DEFAULT_RASTER_MAX_BATCH_BYTES);
        let mut rows = 0;
        for partition in 0..4 {
            for batch in collect(plan.execute(partition, Arc::clone(&ctx)).unwrap())
                .await
                .unwrap()
            {
                assert_eq!(batch.schema(), expected_schema);
                // Identity mock: the loaded column is the input raster.
                assert_eq!(batch.column(1).as_ref(), batch.column(2).as_ref());
                rows += batch.num_rows();
            }
        }
        assert_eq!(rows, 16);
        assert_eq!(calls.lock().unwrap().iter().sum::<usize>(), 16);
        // Once per input batch, below the repartition; never again above it.
        assert_eq!(evaluations.load(Ordering::SeqCst), 2);
    }

    /// `counted_raster(rast)`: an identity raster expression that counts its
    /// evaluations, standing in for an argument such as `RS_FromPath(path)`.
    fn counted_raster_arg(
        schema: &Schema,
        evaluations: Arc<std::sync::atomic::AtomicUsize>,
    ) -> Arc<dyn PhysicalExpr> {
        let raster_type = schema.field_with_name("rast").unwrap().data_type().clone();
        let counted_udf = datafusion_expr::create_udf(
            "counted_raster",
            vec![raster_type.clone()],
            raster_type,
            Volatility::Volatile,
            Arc::new(move |args| {
                evaluations.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(args[0].clone())
            }),
        );
        Arc::new(
            ScalarFunctionExpr::try_new(
                Arc::new(counted_udf),
                vec![col("rast", schema).unwrap()],
                schema,
                Arc::new(ConfigOptions::default()),
            )
            .unwrap(),
        )
    }

    /// `rs_ensureloaded(arg)` named `name`, recording invocations in `calls`.
    fn ensure_loaded_of(
        arg: Arc<dyn PhysicalExpr>,
        name: &str,
        schema: &Schema,
        calls: Arc<Mutex<Vec<usize>>>,
    ) -> Arc<AsyncFuncExpr> {
        let expr = async_expr(Arc::new(MockEnsureLoaded::new(calls)), name, schema);
        let func = Arc::clone(&expr.func).with_new_children(vec![arg]).unwrap();
        Arc::new(AsyncFuncExpr::try_new(name, func, schema).unwrap())
    }

    #[tokio::test]
    async fn expression_argument_over_a_single_input_partition_is_spread_and_merged_back() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let uris: Vec<String> = (0..16).map(|i| format!("s3://b/r{i}.tif")).collect();
        let uris: Vec<&str> = uris.iter().map(String::as_str).collect();
        let (schema, batch) = outdb_raster_batch(&uris);
        let input =
            MemorySourceConfig::try_new_exec(&[vec![batch]], Arc::clone(&schema), None).unwrap();
        let evaluations = Arc::new(AtomicUsize::new(0));
        let calls = Arc::new(Mutex::new(Vec::new()));
        let expr = ensure_loaded_of(
            counted_raster_arg(&schema, Arc::clone(&evaluations)),
            "__async_fn_0",
            &schema,
            Arc::clone(&calls),
        );
        let async_exec = Arc::new(AsyncFuncExec::try_new(vec![expr], input).unwrap());
        let expected_schema = async_exec.schema();
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 4;
        let plan = RasterBatchBudgetRule.optimize(async_exec, &config).unwrap();

        // Coalesce > Projection (drop) > EnsureLoadedExec > hash
        // RepartitionExec > Projection (evaluate) > input.
        assert_eq!(plan.schema(), expected_schema);
        assert_eq!(plan.output_partitioning().partition_count(), 1);
        let merge = plan.downcast_ref::<CoalescePartitionsExec>().unwrap();
        let top = merge.input().downcast_ref::<ProjectionExec>().unwrap();
        let exec = top.input().downcast_ref::<EnsureLoadedExec>().unwrap();
        let repartition = exec.input().downcast_ref::<RepartitionExec>().unwrap();
        assert!(matches!(
            repartition.partitioning(),
            Partitioning::Hash(_, 4)
        ));
        let below = repartition
            .input()
            .downcast_ref::<ProjectionExec>()
            .unwrap();
        assert!(below.input().downcast_ref::<DataSourceExec>().is_some());

        let batches = collect(
            plan.execute(0, task_context(DEFAULT_RASTER_MAX_BATCH_BYTES))
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(row_counts(&batches).iter().sum::<usize>(), 16);
        for batch in &batches {
            assert_eq!(batch.column(1).as_ref(), batch.column(2).as_ref());
        }
        assert!(calls.lock().unwrap().len() > 1);
        // Evaluated once, on the single input partition, before the spread.
        assert_eq!(evaluations.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn spread_keys_on_the_first_call_when_it_is_an_expression_and_the_second_a_column() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let uris: Vec<String> = (0..16).map(|i| format!("s3://b/r{i}.tif")).collect();
        let uris: Vec<&str> = uris.iter().map(String::as_str).collect();
        let (schema, first) = outdb_raster_batch(&uris[..8]);
        let (_, second) = outdb_raster_batch(&uris[8..]);
        let input = MemorySourceConfig::try_new_exec(
            &[vec![first], vec![second]],
            Arc::clone(&schema),
            None,
        )
        .unwrap();
        let evaluations = Arc::new(AtomicUsize::new(0));
        let calls = Arc::new(Mutex::new(Vec::new()));
        let exprs = vec![
            ensure_loaded_of(
                counted_raster_arg(&schema, Arc::clone(&evaluations)),
                "__async_fn_0",
                &schema,
                Arc::clone(&calls),
            ),
            ensure_loaded_of(
                col("rast", &schema).unwrap(),
                "__async_fn_1",
                &schema,
                Arc::clone(&calls),
            ),
        ];
        let async_exec = Arc::new(AsyncFuncExec::try_new(exprs, input).unwrap());
        let expected_schema = async_exec.schema();
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 4;
        let plan = RasterBatchBudgetRule.optimize(async_exec, &config).unwrap();

        assert_eq!(plan.schema(), expected_schema);
        let top = plan.downcast_ref::<ProjectionExec>().unwrap();
        let exec = top.input().downcast_ref::<EnsureLoadedExec>().unwrap();
        let calls_shown: Vec<String> = exec
            .async_exprs()
            .iter()
            .map(|e| e.func.to_string())
            .collect();
        assert_eq!(
            calls_shown,
            vec![
                "rs_ensureloaded(__ensure_loaded_arg_2@2)",
                "rs_ensureloaded(rast@1)"
            ]
        );
        let repartition = exec.input().downcast_ref::<RepartitionExec>().unwrap();
        let Partitioning::Hash(keys, 4) = repartition.partitioning() else {
            panic!("expected a 4-way hash, got {}", repartition.partitioning());
        };
        assert_eq!(
            keys[0].to_string(),
            "raster_source(__ensure_loaded_arg_2@2)"
        );

        let ctx = task_context(DEFAULT_RASTER_MAX_BATCH_BYTES);
        let mut rows = 0;
        for partition in 0..4 {
            for batch in collect(plan.execute(partition, Arc::clone(&ctx)).unwrap())
                .await
                .unwrap()
            {
                assert_eq!(batch.schema(), expected_schema);
                assert_eq!(batch.column(1).as_ref(), batch.column(2).as_ref());
                assert_eq!(batch.column(1).as_ref(), batch.column(3).as_ref());
                rows += batch.num_rows();
            }
        }
        assert_eq!(rows, 16);
        assert_eq!(calls.lock().unwrap().iter().sum::<usize>(), 32);
        assert_eq!(evaluations.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn parents_above_the_spread_are_rebuilt_and_filters_still_pass_through() {
        use std::sync::atomic::AtomicUsize;

        let uris: Vec<String> = (0..16).map(|i| format!("s3://b/r{i}.tif")).collect();
        let uris: Vec<&str> = uris.iter().map(String::as_str).collect();
        let (schema, first) = outdb_raster_batch(&uris[..8]);
        let (_, second) = outdb_raster_batch(&uris[8..]);
        let input = MemorySourceConfig::try_new_exec(
            &[vec![first], vec![second]],
            Arc::clone(&schema),
            None,
        )
        .unwrap();
        let expr = ensure_loaded_of(
            counted_raster_arg(&schema, Arc::new(AtomicUsize::new(0))),
            "__async_fn_0",
            &schema,
            Arc::new(Mutex::new(Vec::new())),
        );
        let async_exec: Arc<dyn ExecutionPlan> =
            Arc::new(AsyncFuncExec::try_new(vec![expr], input).unwrap());
        let filter: Arc<dyn ExecutionPlan> =
            Arc::new(FilterExec::try_new(lit(true), async_exec).unwrap());
        let filter_schema = filter.schema();
        let projection: Arc<dyn ExecutionPlan> = Arc::new(
            ProjectionExec::try_new(
                [
                    (col("id", &filter_schema).unwrap(), "id".to_string()),
                    (
                        col("__async_fn_0", &filter_schema).unwrap(),
                        "loaded".to_string(),
                    ),
                ],
                filter,
            )
            .unwrap(),
        );
        let expected_schema = projection.schema();
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 4;
        let plan = RasterBatchBudgetRule.optimize(projection, &config).unwrap();

        // Projection > Filter > Projection (drop) > EnsureLoadedExec > hash.
        assert_eq!(plan.schema(), expected_schema);
        let top = plan.downcast_ref::<ProjectionExec>().unwrap();
        let filter = top.input().downcast_ref::<FilterExec>().unwrap();
        assert_eq!(filter.batch_size(), PASSTHROUGH_BATCH_SIZE);
        let drop = filter.input().downcast_ref::<ProjectionExec>().unwrap();
        let exec = drop.input().downcast_ref::<EnsureLoadedExec>().unwrap();
        assert!(exec.input().downcast_ref::<RepartitionExec>().is_some());
        assert_eq!(plan.output_partitioning().partition_count(), 4);

        // Four-byte rasters under a four-byte budget: one row per slice, and
        // the filter forwards every slice instead of merging them.
        let ctx = task_context(4);
        let mut sizes = Vec::new();
        for partition in 0..4 {
            let batches = collect(plan.execute(partition, Arc::clone(&ctx)).unwrap())
                .await
                .unwrap();
            sizes.extend(row_counts(&batches));
        }
        assert_eq!(sizes, vec![1; 16]);
    }

    #[test]
    fn planner_repartitioning_below_the_exec_is_replaced_not_stacked() {
        let (schema, batch) = outdb_raster_batch(&["s3://b/a.tif", "s3://b/b.tif"]);
        let one_partition: Arc<dyn ExecutionPlan> =
            MemorySourceConfig::try_new_exec(&[vec![batch.clone()]], Arc::clone(&schema), None)
                .unwrap();
        let two_partitions: Arc<dyn ExecutionPlan> = MemorySourceConfig::try_new_exec(
            &[vec![batch.clone()], vec![batch]],
            Arc::clone(&schema),
            None,
        )
        .unwrap();
        let round_robin: Arc<dyn ExecutionPlan> = Arc::new(
            RepartitionExec::try_new(Arc::clone(&one_partition), Partitioning::RoundRobinBatch(4))
                .unwrap(),
        );
        let merge: Arc<dyn ExecutionPlan> =
            Arc::new(CoalescePartitionsExec::new(Arc::clone(&two_partitions)));
        let limited_merge: Arc<dyn ExecutionPlan> =
            Arc::new(CoalescePartitionsExec::new(Arc::clone(&two_partitions)).with_fetch(Some(1)));
        let wide_union: Arc<dyn ExecutionPlan> = UnionExec::try_new(vec![
            Arc::clone(&two_partitions),
            Arc::clone(&two_partitions),
            Arc::clone(&two_partitions),
        ])
        .unwrap();
        // (case, input, plan hashed below the exec, partitions, merged back)
        let cases = vec![
            ("round-robin", round_robin, one_partition, 4, false),
            ("merge", merge, Arc::clone(&two_partitions), 4, true),
            // A merge with a fetch limits rows; it stays.
            (
                "merge with fetch",
                Arc::clone(&limited_merge),
                limited_merge,
                4,
                true,
            ),
            // Six input partitions over a target of four keep all six.
            ("wide union", Arc::clone(&wide_union), wide_union, 6, false),
        ];
        for (name, input, hashed, partitions, merged) in cases {
            let expr = async_expr(
                Arc::new(MockEnsureLoaded::new(Arc::new(Mutex::new(Vec::new())))),
                "__async_fn_0",
                &schema,
            );
            let async_exec = Arc::new(AsyncFuncExec::try_new(vec![expr], input).unwrap());
            let expected_partitions = async_exec
                .properties()
                .output_partitioning()
                .partition_count();
            let mut config = ConfigOptions::default();
            config.execution.target_partitions = 4;
            let plan = RasterBatchBudgetRule.optimize(async_exec, &config).unwrap();

            let exec = if merged {
                let merge = plan
                    .downcast_ref::<CoalescePartitionsExec>()
                    .unwrap_or_else(|| panic!("{name}: got {}", plan.name()));
                merge.input().downcast_ref::<EnsureLoadedExec>()
            } else {
                assert_eq!(
                    plan.output_partitioning().partition_count(),
                    partitions,
                    "{name}"
                );
                plan.downcast_ref::<EnsureLoadedExec>()
            }
            .unwrap_or_else(|| panic!("{name}: no EnsureLoadedExec"));
            if merged {
                assert_eq!(expected_partitions, 1, "{name}");
            }
            let repartition = exec.input().downcast_ref::<RepartitionExec>().unwrap();
            assert!(
                matches!(repartition.partitioning(), Partitioning::Hash(_, n) if *n == partitions),
                "{name}: {}",
                repartition.partitioning()
            );
            assert!(Arc::ptr_eq(repartition.input(), &hashed), "{name}");
        }
    }

    #[tokio::test]
    async fn a_user_column_named_like_the_hoisted_argument_is_not_read_as_it() {
        use std::sync::atomic::AtomicUsize;

        // The argument is appended at position 2, so it is named
        // `__ensure_loaded_arg_2`; the user's Int32 column of that name sits
        // at position 0. Columns are bound by position, so the loader still
        // reads the raster.
        let uris: Vec<String> = (0..8).map(|i| format!("s3://b/r{i}.tif")).collect();
        let uris: Vec<&str> = uris.iter().map(String::as_str).collect();
        let (outdb_schema, outdb) = outdb_raster_batch(&uris);
        let schema = Arc::new(Schema::new(vec![
            Field::new("__ensure_loaded_arg_2", DataType::Int32, false),
            outdb_schema.field(1).clone(),
        ]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), outdb.columns().to_vec()).unwrap();
        for target_partitions in [1, 4] {
            let input =
                MemorySourceConfig::try_new_exec(&[vec![batch.clone()]], Arc::clone(&schema), None)
                    .unwrap();
            let expr = ensure_loaded_of(
                counted_raster_arg(&schema, Arc::new(AtomicUsize::new(0))),
                "__async_fn_0",
                &schema,
                Arc::new(Mutex::new(Vec::new())),
            );
            let async_exec = Arc::new(AsyncFuncExec::try_new(vec![expr], input).unwrap());
            let expected_schema = async_exec.schema();
            let mut config = ConfigOptions::default();
            config.execution.target_partitions = target_partitions;
            let plan = RasterBatchBudgetRule.optimize(async_exec, &config).unwrap();
            assert_eq!(plan.schema(), expected_schema);
            let batches = collect(
                plan.execute(0, task_context(DEFAULT_RASTER_MAX_BATCH_BYTES))
                    .unwrap(),
            )
            .await
            .unwrap();
            assert_eq!(row_counts(&batches).iter().sum::<usize>(), 8);
            for batch in &batches {
                assert_eq!(batch.column(1).as_ref(), batch.column(2).as_ref());
            }
        }
    }
}
