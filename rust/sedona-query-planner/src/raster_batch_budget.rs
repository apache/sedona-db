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
//! raster can carry megabytes of pixels, so a batch of a few thousand OutDb
//! references is gigabytes once `RS_EnsureLoaded` has loaded it.
//! `RS_EnsureLoaded` is an ordinary scalar UDF: whichever node's expression
//! calls it (a `ProjectionExec`, `FilterExec` or `AggregateExec`, once the
//! `ensure_loaded` logical rule has wrapped the raster arguments of pixel
//! functions) evaluates it over each input batch in full.
//!
//! [`RasterBatchBudgetRule`] therefore puts a [`RasterBatchSplitExec`] under
//! every such node. It re-slices each input batch so that the *estimated*
//! bytes of the rasters the node is about to load stay within
//! `sedona.raster.max_batch_bytes`, and the node then evaluates each slice
//! as its own batch, so at most one slice's worth of loaded pixels is in
//! flight per partition. The estimate is metadata-only
//! ([`sedona_raster::size`]), so it is available before any loading happens
//! and is identical for InDb and OutDb bands. A single row larger than the
//! budget still goes through, alone. Slices are contiguous row ranges of
//! the input batch (`RecordBatch::slice` is zero-copy), so output order is
//! the input order.
//!
//! The slices are sized by the raster argument of every `rs_ensureloaded`
//! call in the node, summed per row when there are several. An argument
//! that is an input column is read as is. An argument that is an expression
//! (`rs_ensureloaded(RS_FromPath(path))`) is evaluated once per batch into a
//! column appended after the input's by a `ProjectionExec` under the
//! splitter, and the call is rewritten to read that column: evaluating it a
//! second time for the load would repeat its I/O (`RS_FromPath` opens the
//! file for its metadata) and, for a volatile argument, load a different
//! raster from the one that was sized. A `FilterExec` whose input grew this
//! way gets a projection back to its original columns.
//!
//! Hoisting evaluates the argument for every row of the batch, so it is only
//! done where the call itself is evaluated for every row. A call in a `CASE`
//! branch (anything after the first `WHEN` condition) or on the right of
//! `AND`/`OR` runs only for the rows that reach it; an expression argument
//! there stays in place and is not sized, since hoisting it would evaluate it
//! (open the file, say) for rows the node never touches. An argument that
//! itself contains an `rs_ensureloaded` call is not sized either: the inner
//! call loads it, so the outer one is a pass-through. Calls in other nodes (a
//! join filter, say) are evaluated unsliced.
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
//! planner places it below the loading node, over OutDb references.
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
use arrow_schema::{Field, Schema};
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{Transformed, TransformedResult, TreeNode, TreeNodeRecursion};
use datafusion_common::{Result, internal_err};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_expr::Operator;
use datafusion_physical_expr::expressions::{BinaryExpr, CaseExpr, Column};
use datafusion_physical_expr::{PhysicalExpr, ScalarFunctionExpr};
use datafusion_physical_plan::aggregates::{AggregateExec, AggregateInputMode, PhysicalGroupBy};
use datafusion_physical_plan::execution_plan::CardinalityEffect;
use datafusion_physical_plan::filter::{FilterExec, FilterExecBuilder};
use datafusion_physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use datafusion_physical_plan::projection::{ProjectionExec, ProjectionExpr};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};
use futures::{StreamExt, stream};
use sedona_common::option::{DEFAULT_RASTER_MAX_BATCH_BYTES, SedonaOptions};
use sedona_common::sedona_internal_datafusion_err;
use sedona_raster::array::RasterStructArray;
use sedona_raster::size::estimated_row_bytes;
use sedona_schema::datatypes::SedonaType;

/// Name of the UDF whose raster argument sizes the slices. Kept in sync
/// with `sedona_raster_functions::rs_ensure_loaded` (this crate can't
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
/// - puts a [`RasterBatchSplitExec`] under every `ProjectionExec`,
///   `FilterExec` and `AggregateExec` whose expressions call
///   `rs_ensureloaded`, sized by the calls' raster arguments;
/// - sets the batch size of every `FilterExec` whose output schema carries
///   a raster column to [`PASSTHROUGH_BATCH_SIZE`], so it forwards batches
///   instead of merging them back up to `datafusion.execution.batch_size`
///   rows (its `fetch` and projection are preserved).
///
/// Runs after DataFusion's own physical rules (it is appended via
/// `SessionStateBuilder::with_physical_optimizer_rule`), so it sees the
/// final placement of the loading nodes. Plans without raster columns are
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
            split_under_loading_node(node)?.transform_data(forward_raster_filter_batches)
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

/// Make a `FilterExec` over a raster-bearing stream forward its input
/// batches instead of merging them.
fn forward_raster_filter_batches(
    node: Arc<dyn ExecutionPlan>,
) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
    match node.downcast_ref::<FilterExec>() {
        Some(filter)
            if filter.batch_size() != PASSTHROUGH_BATCH_SIZE
                && has_raster_column(&node.schema()) =>
        {
            Ok(Transformed::yes(Arc::new(
                filter.with_batch_size(PASSTHROUGH_BATCH_SIZE)?,
            )))
        }
        _ => Ok(Transformed::no(node)),
    }
}

/// Whether `field` is a Sedona raster: by its `sedona.raster` extension
/// metadata, or structurally by the raster storage type. A raster can reach
/// the planner as a plain struct (an array built in Python without the
/// extension type, say), and `rs_ensureloaded` accepts it either way.
fn is_raster_field(field: &Field) -> bool {
    matches!(
        SedonaType::from_storage_field(field),
        Ok(SedonaType::Raster)
    ) || field.data_type() == SedonaType::Raster.storage_type()
}

/// Whether any field of `schema` is a Sedona raster.
fn has_raster_column(schema: &Schema) -> bool {
    schema.fields().iter().any(|field| is_raster_field(field))
}

/// The raster argument of `expr` if it is an `rs_ensureloaded` call.
fn ensure_loaded_arg(expr: &Arc<dyn PhysicalExpr>) -> Option<&Arc<dyn PhysicalExpr>> {
    expr.downcast_ref::<ScalarFunctionExpr>()
        .filter(|f| f.fun().name() == ENSURE_LOADED_NAME)
        .and_then(|f| f.args().first())
}

/// Whether `expr` is, or contains, an `rs_ensureloaded` call.
fn calls_ensure_loaded(expr: &Arc<dyn PhysicalExpr>) -> bool {
    let mut found = false;
    expr.apply(|e| {
        if ensure_loaded_arg(e).is_some() {
            found = true;
            Ok(TreeNodeRecursion::Stop)
        } else {
            Ok(TreeNodeRecursion::Continue)
        }
    })
    .expect("the visitor is infallible");
    found
}

/// What sizes a node's slices: the raster arguments of its
/// `rs_ensureloaded` calls (see the module docs for which ones count).
#[derive(Debug, Default)]
struct SizedArgs {
    /// Input columns, read as is.
    columns: Vec<usize>,
    /// Expressions, each evaluated once per batch into a column appended
    /// after the input's. Identical expressions share one column, volatile
    /// ones included (`RS_FromPath` is declared volatile to stay out of
    /// constant folding, and two calls over the same path should open the
    /// file once); a genuinely nondeterministic argument written twice in
    /// one node therefore yields the same raster twice.
    exprs: Vec<Arc<dyn PhysicalExpr>>,
}

impl SizedArgs {
    fn collect(exprs: &[Arc<dyn PhysicalExpr>]) -> Self {
        let mut sized = SizedArgs::default();
        for expr in exprs {
            sized.visit(expr, false);
        }
        sized
    }

    /// Record the sized arguments of the `rs_ensureloaded` calls in `expr`.
    /// `conditional` says whether `expr` is evaluated for only some rows of
    /// the batch (a `CASE` branch, the right side of `AND`/`OR`): a column
    /// argument there is still sized, since reading a column has no cost,
    /// but an expression argument is left in place.
    fn visit(&mut self, expr: &Arc<dyn PhysicalExpr>, conditional: bool) {
        if let Some(arg) = ensure_loaded_arg(expr) {
            if calls_ensure_loaded(arg) {
                // The inner call loads the raster; size by its arguments.
                self.visit(arg, conditional);
            } else if let Some(column) = arg.downcast_ref::<Column>() {
                if !self.columns.contains(&column.index()) {
                    self.columns.push(column.index());
                }
            } else if !conditional
                && !self
                    .exprs
                    .iter()
                    .any(|known| known.as_ref() == arg.as_ref())
            {
                self.exprs.push(Arc::clone(arg));
            }
            return;
        }
        if let Some(case) = expr.downcast_ref::<CaseExpr>() {
            if let Some(base) = case.expr() {
                self.visit(base, conditional);
            }
            for (idx, (when, then)) in case.when_then_expr().iter().enumerate() {
                self.visit(when, conditional || idx > 0);
                self.visit(then, true);
            }
            if let Some(otherwise) = case.else_expr() {
                self.visit(otherwise, true);
            }
            return;
        }
        if let Some(binary) = expr.downcast_ref::<BinaryExpr>()
            && matches!(binary.op(), Operator::And | Operator::Or)
        {
            self.visit(binary.left(), conditional);
            self.visit(binary.right(), true);
            return;
        }
        for child in expr.children() {
            self.visit(child, conditional);
        }
    }

    fn is_empty(&self) -> bool {
        self.columns.is_empty() && self.exprs.is_empty()
    }
}

/// A raster argument expression evaluated once per batch under the split,
/// and the column the node reads it from instead.
struct HoistedArg {
    expr: Arc<dyn PhysicalExpr>,
    column: Arc<dyn PhysicalExpr>,
}

/// The subtree the rule puts under a loading node.
struct SplitInput {
    /// The [`RasterBatchSplitExec`], over a `ProjectionExec` that appends the
    /// hoisted argument expressions when there are any.
    plan: Arc<dyn ExecutionPlan>,
    hoisted: Vec<HoistedArg>,
}

fn split_input(input: &Arc<dyn ExecutionPlan>, sized: &SizedArgs) -> Result<SplitInput> {
    if sized.exprs.is_empty() {
        return Ok(SplitInput {
            plan: Arc::new(RasterBatchSplitExec::try_new(
                Arc::clone(input),
                sized.columns.clone(),
            )?),
            hoisted: Vec::new(),
        });
    }
    let input_schema = input.schema();
    let input_columns = input_schema.fields().len();
    let mut projection: Vec<ProjectionExpr> = input_schema
        .fields()
        .iter()
        .enumerate()
        .map(|(idx, field)| {
            ProjectionExpr::new(
                Arc::new(Column::new(field.name(), idx)) as Arc<dyn PhysicalExpr>,
                field.name().clone(),
            )
        })
        .collect();
    let mut sized_columns = sized.columns.clone();
    let mut hoisted = Vec::with_capacity(sized.exprs.len());
    for (k, expr) in sized.exprs.iter().enumerate() {
        let name = format!("__ensure_loaded_arg_{k}");
        let index = input_columns + k;
        projection.push(ProjectionExpr::new(Arc::clone(expr), name.clone()));
        hoisted.push(HoistedArg {
            expr: Arc::clone(expr),
            column: Arc::new(Column::new(&name, index)),
        });
        sized_columns.push(index);
    }
    let args = Arc::new(ProjectionExec::try_new(projection, Arc::clone(input))?);
    Ok(SplitInput {
        plan: Arc::new(RasterBatchSplitExec::try_new(args, sized_columns)?),
        hoisted,
    })
}

/// `expr` with every `rs_ensureloaded` call whose argument was hoisted
/// reading the hoisted column instead.
fn read_hoisted_args(
    expr: Arc<dyn PhysicalExpr>,
    hoisted: &[HoistedArg],
) -> Result<Arc<dyn PhysicalExpr>> {
    if hoisted.is_empty() {
        return Ok(expr);
    }
    expr.transform_up(|e| {
        let column = ensure_loaded_arg(&e).and_then(|arg| {
            hoisted
                .iter()
                .find(|hoisted| hoisted.expr.as_ref() == arg.as_ref())
                .map(|hoisted| Arc::clone(&hoisted.column))
        });
        let Some(column) = column else {
            return Ok(Transformed::no(e));
        };
        let mut children: Vec<Arc<dyn PhysicalExpr>> = e.children().into_iter().cloned().collect();
        children[0] = column;
        Ok(Transformed::yes(e.with_new_children(children)?))
    })
    .data()
}

/// Put a [`RasterBatchSplitExec`] under `node` if its expressions call
/// `rs_ensureloaded`.
fn split_under_loading_node(
    node: Arc<dyn ExecutionPlan>,
) -> Result<Transformed<Arc<dyn ExecutionPlan>>> {
    // Already split (the rule ran on this plan before); never stack a second.
    if node
        .children()
        .first()
        .is_some_and(|input| input.downcast_ref::<RasterBatchSplitExec>().is_some())
    {
        return Ok(Transformed::no(node));
    }

    if let Some(projection) = node.downcast_ref::<ProjectionExec>() {
        let exprs: Vec<Arc<dyn PhysicalExpr>> = projection
            .expr()
            .iter()
            .map(|p| Arc::clone(&p.expr))
            .collect();
        let sized = SizedArgs::collect(&exprs);
        if sized.is_empty() {
            return Ok(Transformed::no(node));
        }
        let split = split_input(projection.input(), &sized)?;
        let exprs = projection
            .expr()
            .iter()
            .map(|p| {
                Ok(ProjectionExpr::new(
                    read_hoisted_args(Arc::clone(&p.expr), &split.hoisted)?,
                    p.alias.clone(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        return Ok(Transformed::yes(Arc::new(ProjectionExec::try_new(
            exprs, split.plan,
        )?)));
    }

    if let Some(filter) = node.downcast_ref::<FilterExec>() {
        let sized = SizedArgs::collect(std::slice::from_ref(filter.predicate()));
        if sized.is_empty() {
            return Ok(Transformed::no(node));
        }
        let split = split_input(filter.input(), &sized)?;
        let predicate = read_hoisted_args(Arc::clone(filter.predicate()), &split.hoisted)?;
        // The filter's output is its input's columns (or the projection of
        // them it already had); when the input grew by hoisted arguments,
        // project back to the original columns.
        let projection: Option<Vec<usize>> = match filter.projection() {
            Some(projection) => Some(projection.to_vec()),
            None if split.hoisted.is_empty() => None,
            None => Some((0..filter.input().schema().fields().len()).collect()),
        };
        let filter = FilterExecBuilder::new(predicate, split.plan)
            .with_default_selectivity(filter.default_selectivity())
            .with_batch_size(filter.batch_size())
            .with_fetch(filter.fetch())
            .apply_projection(projection)?
            .build()?;
        return Ok(Transformed::yes(Arc::new(filter)));
    }

    if let Some(aggregate) = node.downcast_ref::<AggregateExec>() {
        // Only a first-stage aggregate evaluates its expressions over raw
        // input. A `Final`/`FinalPartitioned` stage carries the same
        // expressions (bound to the raw input schema) but reads partial
        // state, so there is nothing to split under it.
        if aggregate.mode().input_mode() != AggregateInputMode::Raw {
            return Ok(Transformed::no(node));
        }
        let group_by = aggregate.group_expr();
        let mut exprs: Vec<Arc<dyn PhysicalExpr>> = group_by.input_exprs();
        exprs.extend(group_by.null_expr().iter().map(|(e, _)| Arc::clone(e)));
        for aggr in aggregate.aggr_expr() {
            exprs.extend(aggr.expressions());
            exprs.extend(aggr.order_bys().iter().map(|s| Arc::clone(&s.expr)));
        }
        exprs.extend(aggregate.filter_expr().iter().flatten().cloned());
        let sized = SizedArgs::collect(&exprs);
        if sized.is_empty() {
            return Ok(Transformed::no(node));
        }
        let split = split_input(aggregate.input(), &sized)?;
        let rewrite =
            |expr: &Arc<dyn PhysicalExpr>| read_hoisted_args(Arc::clone(expr), &split.hoisted);
        let rewrite_named = |pairs: &[(Arc<dyn PhysicalExpr>, String)]| {
            pairs
                .iter()
                .map(|(expr, name)| Ok((rewrite(expr)?, name.clone())))
                .collect::<Result<Vec<_>>>()
        };
        let mut aggr_exprs = Vec::with_capacity(aggregate.aggr_expr().len());
        for aggr in aggregate.aggr_expr() {
            let args = aggr
                .expressions()
                .iter()
                .map(rewrite)
                .collect::<Result<Vec<_>>>()?;
            let order_bys = aggr
                .order_bys()
                .iter()
                .map(|s| rewrite(&s.expr))
                .collect::<Result<Vec<_>>>()?;
            // `with_new_expressions` only refuses a different number of
            // arguments or order-bys, and the rewrite keeps both.
            let Some(rewritten) = aggr.with_new_expressions(args, order_bys) else {
                return internal_err!(
                    "RasterBatchBudgetRule: aggregate {} rejected its rewritten arguments",
                    aggr.name()
                );
            };
            aggr_exprs.push(Arc::new(rewritten));
        }
        let filter_expr = aggregate
            .filter_expr()
            .iter()
            .map(|f| f.as_ref().map(rewrite).transpose())
            .collect::<Result<Vec<_>>>()?;
        let group_by = PhysicalGroupBy::new(
            rewrite_named(group_by.expr())?,
            rewrite_named(group_by.null_expr())?,
            group_by.groups().to_vec(),
            !group_by.is_single(),
        );
        let input_schema = if split.hoisted.is_empty() {
            aggregate.input_schema()
        } else {
            split.plan.schema()
        };
        let aggregate = AggregateExec::try_new(
            *aggregate.mode(),
            group_by,
            aggr_exprs,
            filter_expr,
            split.plan,
            input_schema,
        )?
        .with_limit_options(aggregate.limit_options());
        return Ok(Transformed::yes(Arc::new(aggregate)));
    }

    Ok(Transformed::no(node))
}

/// Contiguous `[start, end)` row ranges whose summed `estimates` stay within
/// `budget`. A row that alone exceeds the budget gets a range of its own. An
/// empty input yields no ranges.
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

/// Per-row estimated bytes of everything the node above will load: the sum
/// over the sized columns of the metadata-only estimate.
fn estimate_rows(batch: &RecordBatch, sized_columns: &[usize]) -> Result<Vec<u64>> {
    let mut totals = vec![0u64; batch.num_rows()];
    for &idx in sized_columns {
        let struct_array = batch
            .column(idx)
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| {
                sedona_internal_datafusion_err!(
                    "RasterBatchSplitExec: expected a Raster (Struct) column at {idx}, got {:?}",
                    batch.column(idx).data_type()
                )
            })?;
        let rasters = RasterStructArray::try_new(struct_array)?;
        for (total, bytes) in totals.iter_mut().zip(estimated_row_bytes(&rasters)?) {
            *total = total.saturating_add(bytes);
        }
    }
    Ok(totals)
}

/// Re-slices its input's batches by the estimated bytes of the rasters in
/// `sized_columns`, so the node above loads at most
/// `sedona.raster.max_batch_bytes` of pixels per batch. The schema, order and
/// partitioning are the input's. See the module docs.
#[derive(Debug)]
pub struct RasterBatchSplitExec {
    input: Arc<dyn ExecutionPlan>,
    /// Input columns whose per-row estimated bytes, summed, size the slices.
    sized_columns: Vec<usize>,
    cache: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl RasterBatchSplitExec {
    pub fn try_new(input: Arc<dyn ExecutionPlan>, sized_columns: Vec<usize>) -> Result<Self> {
        if sized_columns.is_empty() {
            return internal_err!("RasterBatchSplitExec requires at least one column to size by");
        }
        let schema = input.schema();
        for &idx in &sized_columns {
            match schema.fields().get(idx) {
                Some(field) if is_raster_field(field) => {}
                Some(field) => {
                    return internal_err!(
                        "RasterBatchSplitExec: column {idx} ({}) is not a raster",
                        field.name()
                    );
                }
                None => {
                    return internal_err!(
                        "RasterBatchSplitExec: column {idx} is out of range for {} input columns",
                        schema.fields().len()
                    );
                }
            }
        }
        Ok(Self {
            cache: Arc::clone(input.properties()),
            input,
            sized_columns,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    pub fn sized_columns(&self) -> &[usize] {
        &self.sized_columns
    }
}

impl DisplayAs for RasterBatchSplitExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        let schema = self.input.schema();
        let sized_by = self
            .sized_columns
            .iter()
            .map(|&idx| format!("{}@{idx}", schema.field(idx).name()))
            .collect::<Vec<_>>()
            .join(", ");
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "RasterBatchSplitExec: sized_by=[{sized_by}]")
            }
            DisplayFormatType::TreeRender => {
                writeln!(f, "sized_by={sized_by}")
            }
        }
    }
}

impl ExecutionPlan for RasterBatchSplitExec {
    fn name(&self) -> &str {
        "RasterBatchSplitExec"
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
                "RasterBatchSplitExec expects exactly 1 child, got {}",
                children.len()
            );
        }
        Ok(Arc::new(Self::try_new(
            children.remove(0),
            self.sized_columns.clone(),
        )?))
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    // The rule runs after `EnforceDistribution`, so no repartition is ever
    // placed relative to this node: whatever the planner put under the
    // loading node (a round-robin over cheap OutDb references, usually) ends
    // up under the split.

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
        let sized_columns = self.sized_columns.clone();
        let baseline = BaselineMetrics::new(&self.metrics, partition);

        // One input batch fans out into as many output batches as the
        // budget requires.
        let output = input.flat_map(move |batch| {
            let batch = match batch {
                Ok(batch) => batch,
                Err(e) => return stream::once(async move { Err(e) }).boxed(),
            };
            // An empty batch has nothing to size or load, and nothing
            // downstream can tell it was dropped.
            if batch.num_rows() == 0 {
                return stream::empty().boxed();
            }
            let ranges = match estimate_rows(&batch, &sized_columns) {
                Ok(estimates) => slice_ranges(&estimates, budget),
                Err(e) => return stream::once(async move { Err(e) }).boxed(),
            };
            let slices: Vec<Result<RecordBatch>> = ranges
                .into_iter()
                .map(|(start, end)| {
                    let slice = batch.slice(start, end - start);
                    baseline.record_output(slice.num_rows());
                    Ok(slice)
                })
                .collect();
            stream::iter(slices).boxed()
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
    use std::hash::{Hash, Hasher};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow_array::{ArrayRef, Int32Array};
    use arrow_schema::{DataType, Field, FieldRef, Schema, SchemaRef};
    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::functions_aggregate::count::count_udaf;
    use datafusion::prelude::SessionConfig;
    use datafusion_expr::{
        ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
        Volatility,
    };
    use datafusion_physical_expr::aggregate::AggregateExprBuilder;
    use datafusion_physical_expr::expressions::{binary, col, is_not_null, is_null, lit};
    use datafusion_physical_plan::aggregates::AggregateMode;
    use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
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
    impl Hash for MockEnsureLoaded {
        fn hash<H: Hasher>(&self, state: &mut H) {
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
        fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
            Ok(Arc::clone(&args.arg_fields[0]))
        }
        fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
            self.calls.lock().unwrap().push(args.number_rows);
            Ok(ColumnarValue::Array(
                args.args[0].to_array(args.number_rows)?,
            ))
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

    /// `rs_ensureloaded(arg)` as the mock, recording into `calls`.
    fn ensure_loaded_call(
        calls: &Arc<Mutex<Vec<usize>>>,
        arg: Arc<dyn PhysicalExpr>,
        schema: &Schema,
    ) -> Arc<dyn PhysicalExpr> {
        let udf = Arc::new(ScalarUDF::new_from_impl(MockEnsureLoaded::new(Arc::clone(
            calls,
        ))));
        Arc::new(
            ScalarFunctionExpr::try_new(udf, vec![arg], schema, Arc::new(ConfigOptions::default()))
                .unwrap(),
        )
    }

    /// A volatile raster → raster identity UDF that counts its invocations
    /// and keeps the raster extension type on its output, standing in for an
    /// argument such as `RS_FromPath(path)` whose evaluation has a cost.
    #[derive(Debug)]
    struct CountedRaster {
        signature: Signature,
        evaluations: Arc<AtomicUsize>,
    }

    impl PartialEq for CountedRaster {
        fn eq(&self, _other: &Self) -> bool {
            true
        }
    }
    impl Eq for CountedRaster {}
    impl Hash for CountedRaster {
        fn hash<H: Hasher>(&self, state: &mut H) {
            "counted_raster".hash(state);
        }
    }

    impl ScalarUDFImpl for CountedRaster {
        fn name(&self) -> &str {
            "counted_raster"
        }
        fn signature(&self) -> &Signature {
            &self.signature
        }
        fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
            Ok(arg_types[0].clone())
        }
        fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
            Ok(Arc::clone(&args.arg_fields[0]))
        }
        fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
            self.evaluations.fetch_add(1, Ordering::SeqCst);
            Ok(args.args[0].clone())
        }
    }

    fn counted_raster_call(
        evaluations: &Arc<AtomicUsize>,
        arg: Arc<dyn PhysicalExpr>,
        schema: &Schema,
    ) -> Arc<dyn PhysicalExpr> {
        let udf = Arc::new(ScalarUDF::new_from_impl(CountedRaster {
            signature: Signature::any(1, Volatility::Volatile),
            evaluations: Arc::clone(evaluations),
        }));
        Arc::new(
            ScalarFunctionExpr::try_new(udf, vec![arg], schema, Arc::new(ConfigOptions::default()))
                .unwrap(),
        )
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

    /// `ProjectionExec(rs_ensureloaded(rast) AS loaded)` over one batch of
    /// `sides`, with the rule applied; returns the plan and the mock's log.
    fn budgeted_projection(
        sides: &[Option<i64>],
    ) -> (Arc<dyn ExecutionPlan>, Arc<Mutex<Vec<usize>>>, SchemaRef) {
        let (schema, batch) = raster_batch(sides);
        let input =
            MemorySourceConfig::try_new_exec(&[vec![batch]], Arc::clone(&schema), None).unwrap();
        let calls = Arc::new(Mutex::new(Vec::new()));
        let projection: Arc<dyn ExecutionPlan> = Arc::new(
            ProjectionExec::try_new(
                vec![ProjectionExpr::new(
                    ensure_loaded_call(&calls, col("rast", &schema).unwrap(), &schema),
                    "loaded",
                )],
                input,
            )
            .unwrap(),
        );
        let expected_schema = projection.schema();
        let plan = RasterBatchBudgetRule
            .optimize(projection, &ConfigOptions::default())
            .unwrap();
        assert_eq!(plan.schema(), expected_schema, "schema must be unchanged");
        (plan, calls, expected_schema)
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
    async fn projection_over_a_raster_column_is_sliced_to_the_budget() {
        // Six 256-byte rasters under a 600-byte budget → 2 per slice.
        let (plan, calls, schema) = budgeted_projection(&[Some(16); 6]);
        let projection = plan
            .downcast_ref::<ProjectionExec>()
            .expect("projection kept");
        let split = projection
            .input()
            .downcast_ref::<RasterBatchSplitExec>()
            .expect("split under the projection");
        assert_eq!(split.sized_columns(), &[0]);
        assert_eq!(
            split.input().name(),
            "DataSourceExec",
            "a column argument needs no projection below the split"
        );

        let batches = collect(plan.execute(0, task_context(600)).unwrap())
            .await
            .unwrap();
        let rows: Vec<usize> = batches.iter().map(RecordBatch::num_rows).collect();
        assert_eq!(rows, vec![2, 2, 2]);
        assert_eq!(*calls.lock().unwrap(), vec![2, 2, 2]);
        assert_eq!(batches[0].schema(), schema);
        assert_eq!(schema.field(0).name(), "loaded");
        assert!(is_raster_field(schema.field(0)));
    }

    #[tokio::test]
    async fn oversized_rows_go_alone_and_nulls_cost_nothing() {
        // 100, 1024, null, 100, 100 bytes under a 256-byte budget.
        let (plan, calls, _) = budgeted_projection(&[Some(10), Some(32), None, Some(10), Some(10)]);
        let batches = collect(plan.execute(0, task_context(256)).unwrap())
            .await
            .unwrap();
        let rows: Vec<usize> = batches.iter().map(RecordBatch::num_rows).collect();
        assert_eq!(rows, vec![1, 1, 3]);
        assert_eq!(*calls.lock().unwrap(), vec![1, 1, 3]);
        // Output rows are the input rows in order, and the null survives.
        let nulls: Vec<usize> = batches.iter().map(|b| b.column(0).null_count()).collect();
        assert_eq!(nulls, vec![0, 0, 1]);
    }

    #[tokio::test]
    async fn zero_budget_disables_slicing() {
        let (plan, calls, _) = budgeted_projection(&[Some(16); 6]);
        let batches = collect(plan.execute(0, task_context(0)).unwrap())
            .await
            .unwrap();
        let rows: Vec<usize> = batches.iter().map(RecordBatch::num_rows).collect();
        assert_eq!(rows, vec![6]);
        assert_eq!(*calls.lock().unwrap(), vec![6]);
    }

    #[tokio::test]
    async fn empty_batches_emit_nothing_and_never_invoke() {
        let (plan, calls, _) = budgeted_projection(&[]);
        let batches = collect(plan.execute(0, task_context(1024)).unwrap())
            .await
            .unwrap();
        assert!(batches.is_empty(), "{batches:?}");
        assert!(calls.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn raster_argument_expression_is_evaluated_once_per_batch() {
        // `rs_ensureloaded(counted_raster(rast))`: the argument is evaluated
        // once per batch, for sizing, into a column the call then reads; a
        // second evaluation would repeat its I/O and, being volatile, could
        // hand the loader a different raster from the one that was sized.
        let (schema, batch) = raster_batch(&[Some(4), Some(4), Some(4)]);
        let input =
            MemorySourceConfig::try_new_exec(&[vec![batch]], Arc::clone(&schema), None).unwrap();
        let evaluations = Arc::new(AtomicUsize::new(0));
        let calls = Arc::new(Mutex::new(Vec::new()));
        let projection: Arc<dyn ExecutionPlan> = Arc::new(
            ProjectionExec::try_new(
                vec![ProjectionExpr::new(
                    ensure_loaded_call(
                        &calls,
                        counted_raster_call(&evaluations, col("rast", &schema).unwrap(), &schema),
                        &schema,
                    ),
                    "loaded",
                )],
                input,
            )
            .unwrap(),
        );
        let expected_schema = projection.schema();
        let plan = RasterBatchBudgetRule
            .optimize(projection, &ConfigOptions::default())
            .unwrap();
        assert_eq!(plan.schema(), expected_schema);

        // Projection ← Split ← Projection(rast, __ensure_loaded_arg_0) ← scan
        let projection = plan
            .downcast_ref::<ProjectionExec>()
            .expect("projection kept");
        let split = projection
            .input()
            .downcast_ref::<RasterBatchSplitExec>()
            .expect("split under the projection");
        assert_eq!(split.sized_columns(), &[1]);
        let args = split
            .input()
            .downcast_ref::<ProjectionExec>()
            .expect("argument projection under the split");
        assert_eq!(
            args.schema()
                .fields()
                .iter()
                .map(|f| f.name().as_str())
                .collect::<Vec<_>>(),
            vec!["rast", "__ensure_loaded_arg_0"]
        );
        assert!(is_raster_field(args.schema().field(1)));
        // The call reads the appended column, not the expression.
        assert_eq!(
            projection.expr()[0].expr.to_string(),
            "rs_ensureloaded(__ensure_loaded_arg_0@1)"
        );

        // Three 16-byte rows under a 16-byte budget: three slices, and the
        // argument is still evaluated once.
        let batches = collect(plan.execute(0, task_context(16)).unwrap())
            .await
            .unwrap();
        let rows: Vec<usize> = batches.iter().map(RecordBatch::num_rows).collect();
        assert_eq!(rows, vec![1, 1, 1]);
        assert_eq!(*calls.lock().unwrap(), vec![1, 1, 1]);
        assert_eq!(evaluations.load(Ordering::SeqCst), 1);
        assert_eq!(batches[0].schema(), expected_schema);
    }

    #[tokio::test]
    async fn nested_loader_is_sized_by_the_inner_call_only() {
        // `rs_ensureloaded(counted_raster(rs_ensureloaded(rast)))`: the inner
        // call loads `rast`, so the outer one is a pass-through and must not
        // be sized (that would evaluate the inner call for sizing) nor hoisted.
        let (schema, batch) = raster_batch(&[Some(16); 6]);
        let input =
            MemorySourceConfig::try_new_exec(&[vec![batch]], Arc::clone(&schema), None).unwrap();
        let evaluations = Arc::new(AtomicUsize::new(0));
        let calls = Arc::new(Mutex::new(Vec::new()));
        let inner = ensure_loaded_call(&calls, col("rast", &schema).unwrap(), &schema);
        let outer = ensure_loaded_call(
            &calls,
            counted_raster_call(&evaluations, inner, &schema),
            &schema,
        );
        let projection: Arc<dyn ExecutionPlan> = Arc::new(
            ProjectionExec::try_new(vec![ProjectionExpr::new(outer, "loaded")], input).unwrap(),
        );
        let plan = RasterBatchBudgetRule
            .optimize(projection, &ConfigOptions::default())
            .unwrap();
        let split = plan
            .downcast_ref::<ProjectionExec>()
            .unwrap()
            .input()
            .downcast_ref::<RasterBatchSplitExec>()
            .expect("split under the projection");
        assert_eq!(split.sized_columns(), &[0]);
        assert_eq!(split.input().name(), "DataSourceExec", "nothing hoisted");

        let batches = collect(plan.execute(0, task_context(600)).unwrap())
            .await
            .unwrap();
        let rows: Vec<usize> = batches.iter().map(RecordBatch::num_rows).collect();
        assert_eq!(rows, vec![2, 2, 2]);
        // Both calls run once per slice, inner then outer.
        assert_eq!(*calls.lock().unwrap(), vec![2, 2, 2, 2, 2, 2]);
        assert_eq!(evaluations.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn filter_calling_the_loader_is_split_and_forwards_the_slices() {
        let (schema, batch) = raster_batch(&[Some(16); 6]);
        let input =
            MemorySourceConfig::try_new_exec(&[vec![batch]], Arc::clone(&schema), None).unwrap();
        let calls = Arc::new(Mutex::new(Vec::new()));
        let filter: Arc<dyn ExecutionPlan> = Arc::new(
            FilterExec::try_new(
                is_not_null(ensure_loaded_call(
                    &calls,
                    col("rast", &schema).unwrap(),
                    &schema,
                ))
                .unwrap(),
                input,
            )
            .unwrap(),
        );
        let plan = RasterBatchBudgetRule
            .optimize(filter, &ConfigOptions::default())
            .unwrap();
        assert_eq!(plan.schema(), schema);
        let filter = plan.downcast_ref::<FilterExec>().expect("filter kept");
        assert_eq!(filter.batch_size(), PASSTHROUGH_BATCH_SIZE);
        assert!(filter.projection().is_none());
        let split = filter
            .input()
            .downcast_ref::<RasterBatchSplitExec>()
            .expect("split under the filter");
        assert_eq!(split.sized_columns(), &[0]);

        let batches = collect(plan.execute(0, task_context(600)).unwrap())
            .await
            .unwrap();
        let rows: Vec<usize> = batches.iter().map(RecordBatch::num_rows).collect();
        assert_eq!(rows, vec![2, 2, 2]);
        assert_eq!(*calls.lock().unwrap(), vec![2, 2, 2]);
    }

    #[tokio::test]
    async fn filter_with_a_hoisted_argument_keeps_its_schema() {
        let (schema, batch) = raster_batch(&[Some(16); 6]);
        let input =
            MemorySourceConfig::try_new_exec(&[vec![batch]], Arc::clone(&schema), None).unwrap();
        let evaluations = Arc::new(AtomicUsize::new(0));
        let calls = Arc::new(Mutex::new(Vec::new()));
        let filter: Arc<dyn ExecutionPlan> = Arc::new(
            FilterExec::try_new(
                is_not_null(ensure_loaded_call(
                    &calls,
                    counted_raster_call(&evaluations, col("rast", &schema).unwrap(), &schema),
                    &schema,
                ))
                .unwrap(),
                input,
            )
            .unwrap(),
        );
        let plan = RasterBatchBudgetRule
            .optimize(filter, &ConfigOptions::default())
            .unwrap();
        // The filter's input grew by the hoisted argument; its output did not.
        assert_eq!(plan.schema(), schema);
        let filter = plan.downcast_ref::<FilterExec>().expect("filter kept");
        assert_eq!(filter.projection().as_deref(), Some(&[0][..]));
        assert_eq!(filter.input().schema().fields().len(), 2);
        assert_eq!(
            filter.predicate().to_string(),
            "rs_ensureloaded(__ensure_loaded_arg_0@1) IS NOT NULL"
        );

        let batches = collect(plan.execute(0, task_context(600)).unwrap())
            .await
            .unwrap();
        let rows: Vec<usize> = batches.iter().map(RecordBatch::num_rows).collect();
        assert_eq!(rows, vec![2, 2, 2]);
        assert_eq!(batches[0].schema(), schema);
        assert_eq!(*calls.lock().unwrap(), vec![2, 2, 2]);
        assert_eq!(evaluations.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn aggregate_calling_the_loader_is_split() {
        // `count(rs_ensureloaded(counted_raster(rast)))`, partial mode: the
        // argument is hoisted under the split and the aggregate reads it from
        // the appended column.
        let (schema, batch) = raster_batch(&[Some(16); 6]);
        let input =
            MemorySourceConfig::try_new_exec(&[vec![batch]], Arc::clone(&schema), None).unwrap();
        let evaluations = Arc::new(AtomicUsize::new(0));
        let calls = Arc::new(Mutex::new(Vec::new()));
        let count = AggregateExprBuilder::new(
            count_udaf(),
            vec![ensure_loaded_call(
                &calls,
                counted_raster_call(&evaluations, col("rast", &schema).unwrap(), &schema),
                &schema,
            )],
        )
        .schema(Arc::clone(&schema))
        .alias("c")
        .build()
        .unwrap();
        let aggregate: Arc<dyn ExecutionPlan> = Arc::new(
            AggregateExec::try_new(
                AggregateMode::Partial,
                PhysicalGroupBy::new_single(vec![]),
                vec![Arc::new(count)],
                vec![None],
                input,
                Arc::clone(&schema),
            )
            .unwrap(),
        );
        let expected_schema = aggregate.schema();
        let plan = RasterBatchBudgetRule
            .optimize(aggregate, &ConfigOptions::default())
            .unwrap();
        assert_eq!(plan.schema(), expected_schema);
        let aggregate = plan
            .downcast_ref::<AggregateExec>()
            .expect("aggregate kept");
        let split = aggregate
            .input()
            .downcast_ref::<RasterBatchSplitExec>()
            .expect("split under the aggregate");
        assert_eq!(split.sized_columns(), &[1]);
        assert_eq!(aggregate.input_schema().fields().len(), 2);
        assert_eq!(
            aggregate.aggr_expr()[0].expressions()[0].to_string(),
            "rs_ensureloaded(__ensure_loaded_arg_0@1)"
        );

        let batches = collect(plan.execute(0, task_context(600)).unwrap())
            .await
            .unwrap();
        let counts: Vec<i64> = batches
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<arrow_array::Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();
        assert_eq!(counts, vec![6]);
        assert_eq!(*calls.lock().unwrap(), vec![2, 2, 2]);
        assert_eq!(evaluations.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn filter_over_raster_stream_forwards_slices_instead_of_merging_them() {
        // Baseline: a stock FilterExec above a budgeted projection merges the
        // three 2-row slices back into one 6-row batch.
        let (budgeted, _, _) = budgeted_projection(&[Some(16); 6]);
        let stock = Arc::new(FilterExec::try_new(lit(true), budgeted).unwrap());
        let batches = collect(stock.execute(0, task_context(600)).unwrap())
            .await
            .unwrap();
        let rows: Vec<usize> = batches.iter().map(RecordBatch::num_rows).collect();
        assert_eq!(rows, vec![6]);

        // With the rule, the filter passes each slice through.
        let (budgeted, calls, _) = budgeted_projection(&[Some(16); 6]);
        let filtered: Arc<dyn ExecutionPlan> =
            Arc::new(FilterExec::try_new(lit(true), budgeted).unwrap());
        let plan = RasterBatchBudgetRule
            .optimize(filtered, &ConfigOptions::default())
            .unwrap();
        let filter = plan.downcast_ref::<FilterExec>().expect("filter kept");
        assert_eq!(filter.batch_size(), PASSTHROUGH_BATCH_SIZE);
        let batches = collect(plan.execute(0, task_context(600)).unwrap())
            .await
            .unwrap();
        let rows: Vec<usize> = batches.iter().map(RecordBatch::num_rows).collect();
        assert_eq!(rows, vec![2, 2, 2]);
        assert_eq!(*calls.lock().unwrap(), vec![2, 2, 2]);
    }

    #[tokio::test]
    async fn nodes_without_loader_calls_are_left_alone() {
        let (schema, batch) = raster_batch(&[Some(4)]);
        let input =
            MemorySourceConfig::try_new_exec(&[vec![batch]], Arc::clone(&schema), None).unwrap();
        let projection: Arc<dyn ExecutionPlan> = Arc::new(
            ProjectionExec::try_new(
                vec![ProjectionExpr::new(col("rast", &schema).unwrap(), "rast")],
                input,
            )
            .unwrap(),
        );
        let plan = RasterBatchBudgetRule
            .optimize(Arc::clone(&projection), &ConfigOptions::default())
            .unwrap();
        assert!(
            Arc::ptr_eq(&plan, &projection),
            "projection must be untouched"
        );
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
        // A raster built without it (a plain struct from Python) counts too.
        let bare = Schema::new(vec![
            raster_schema
                .field(0)
                .clone()
                .with_metadata(Default::default()),
        ]);
        assert!(bare.field(0).metadata().is_empty());
        assert!(has_raster_column(&bare));
        let plain = Schema::new(vec![Field::new("id", DataType::Int32, false)]);
        assert!(!has_raster_column(&plain));
    }

    #[tokio::test]
    async fn expression_arguments_in_conditional_positions_are_not_hoisted() {
        // Hoisting an argument under the split evaluates it for every row of
        // the batch. A loader in a CASE branch or on the right of AND/OR only
        // runs for the rows that reach it, so its expression argument stays
        // in place (and unsized); in the first WHEN or on the left it is
        // hoisted as usual.
        let (schema, batch) = raster_batch(&[Some(16); 6]);
        let evaluations = Arc::new(AtomicUsize::new(0));
        let calls = Arc::new(Mutex::new(Vec::new()));
        let loaded = || {
            ensure_loaded_call(
                &calls,
                counted_raster_call(&evaluations, col("rast", &schema).unwrap(), &schema),
                &schema,
            )
        };
        let project = |expr: Arc<dyn PhysicalExpr>| -> Arc<dyn ExecutionPlan> {
            let input =
                MemorySourceConfig::try_new_exec(&[vec![batch.clone()]], Arc::clone(&schema), None)
                    .unwrap();
            Arc::new(ProjectionExec::try_new(vec![ProjectionExpr::new(expr, "v")], input).unwrap())
        };

        // CASE WHEN rast IS NULL THEN loaded ELSE rast END: no row is null, so
        // the branch, and the argument, must never run.
        let in_branch: Arc<dyn PhysicalExpr> = Arc::new(
            CaseExpr::try_new(
                None,
                vec![(is_null(col("rast", &schema).unwrap()).unwrap(), loaded())],
                Some(col("rast", &schema).unwrap()),
            )
            .unwrap(),
        );
        let projection = project(Arc::clone(&in_branch));
        let plan = RasterBatchBudgetRule
            .optimize(Arc::clone(&projection), &ConfigOptions::default())
            .unwrap();
        assert!(
            Arc::ptr_eq(&plan, &projection),
            "{in_branch}: must be left alone"
        );
        let batches = collect(plan.execute(0, task_context(600)).unwrap())
            .await
            .unwrap();
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 6);
        assert_eq!(evaluations.load(Ordering::SeqCst), 0);
        assert!(calls.lock().unwrap().is_empty());

        let shapes: Vec<(Arc<dyn PhysicalExpr>, bool)> = vec![
            // The first WHEN condition is evaluated for every row.
            (
                Arc::new(
                    CaseExpr::try_new(
                        None,
                        vec![(
                            is_not_null(loaded()).unwrap(),
                            col("rast", &schema).unwrap(),
                        )],
                        None,
                    )
                    .unwrap(),
                ),
                true,
            ),
            // A later WHEN condition is not.
            (
                Arc::new(
                    CaseExpr::try_new(
                        None,
                        vec![
                            (
                                is_null(col("rast", &schema).unwrap()).unwrap(),
                                col("rast", &schema).unwrap(),
                            ),
                            (
                                is_not_null(loaded()).unwrap(),
                                col("rast", &schema).unwrap(),
                            ),
                        ],
                        None,
                    )
                    .unwrap(),
                ),
                false,
            ),
            // The right side of AND may be skipped; the left side never is.
            (
                binary(
                    is_null(col("rast", &schema).unwrap()).unwrap(),
                    Operator::And,
                    is_not_null(loaded()).unwrap(),
                    &schema,
                )
                .unwrap(),
                false,
            ),
            (
                binary(
                    is_not_null(loaded()).unwrap(),
                    Operator::Or,
                    is_null(col("rast", &schema).unwrap()).unwrap(),
                    &schema,
                )
                .unwrap(),
                true,
            ),
        ];
        for (expr, hoisted) in shapes {
            let projection = project(Arc::clone(&expr));
            let plan = RasterBatchBudgetRule
                .optimize(Arc::clone(&projection), &ConfigOptions::default())
                .unwrap();
            if hoisted {
                let split = plan
                    .downcast_ref::<ProjectionExec>()
                    .unwrap()
                    .input()
                    .downcast_ref::<RasterBatchSplitExec>()
                    .unwrap_or_else(|| panic!("{expr}: must be split"));
                assert!(
                    split.input().downcast_ref::<ProjectionExec>().is_some(),
                    "{expr}: the argument must be hoisted"
                );
            } else {
                assert!(
                    Arc::ptr_eq(&plan, &projection),
                    "{expr}: must be left alone"
                );
            }
        }
    }

    #[tokio::test]
    async fn only_the_first_aggregate_stage_is_split() {
        // Final ← CoalescePartitions ← Partial ← scan, as the planner builds
        // `SELECT count(rs_ensureloaded(rast))` over several partitions.
        // Both stages carry the same expressions, bound to the raw schema;
        // only the partial stage reads rows, so only it gets a split. (The
        // final stage reads partial state, where the raster column does not
        // even exist.)
        let (schema, batch) = raster_batch(&[Some(16); 6]);
        let input =
            MemorySourceConfig::try_new_exec(&[vec![batch]], Arc::clone(&schema), None).unwrap();
        let calls = Arc::new(Mutex::new(Vec::new()));
        let count = Arc::new(
            AggregateExprBuilder::new(
                count_udaf(),
                vec![ensure_loaded_call(
                    &calls,
                    col("rast", &schema).unwrap(),
                    &schema,
                )],
            )
            .schema(Arc::clone(&schema))
            .alias("c")
            .build()
            .unwrap(),
        );
        let partial = Arc::new(
            AggregateExec::try_new(
                AggregateMode::Partial,
                PhysicalGroupBy::new_single(vec![]),
                vec![Arc::clone(&count)],
                vec![None],
                input,
                Arc::clone(&schema),
            )
            .unwrap(),
        );
        let merged = Arc::new(CoalescePartitionsExec::new(partial));
        let final_stage: Arc<dyn ExecutionPlan> = Arc::new(
            AggregateExec::try_new(
                AggregateMode::Final,
                PhysicalGroupBy::new_single(vec![]),
                vec![count],
                vec![None],
                merged,
                Arc::clone(&schema),
            )
            .unwrap(),
        );
        let expected_schema = final_stage.schema();

        let plan = RasterBatchBudgetRule
            .optimize(final_stage, &ConfigOptions::default())
            .unwrap();
        assert_eq!(plan.schema(), expected_schema);
        let final_stage = plan.downcast_ref::<AggregateExec>().expect("final kept");
        assert_eq!(*final_stage.mode(), AggregateMode::Final);
        let partial = final_stage
            .input()
            .downcast_ref::<CoalescePartitionsExec>()
            .expect("merge kept")
            .input()
            .downcast_ref::<AggregateExec>()
            .expect("partial kept");
        assert_eq!(*partial.mode(), AggregateMode::Partial);
        assert!(
            partial
                .input()
                .downcast_ref::<RasterBatchSplitExec>()
                .is_some(),
            "the split goes under the partial stage"
        );

        let batches = collect(plan.execute(0, task_context(600)).unwrap())
            .await
            .unwrap();
        let counts: Vec<i64> = batches
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<arrow_array::Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();
        assert_eq!(counts, vec![6]);
        assert_eq!(*calls.lock().unwrap(), vec![2, 2, 2]);
    }

    #[tokio::test]
    async fn filter_keeps_its_projection_and_fetch_around_a_hoisted_argument() {
        // A filter that already projects its output (only `id`) and stops
        // after three rows, with an expression argument hoisted under it.
        let (raster_schema, raster_batch) = raster_batch(&[Some(16); 6]);
        let schema = Arc::new(Schema::new(vec![
            raster_schema.field(0).clone(),
            Field::new("id", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::clone(raster_batch.column(0)),
                Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6])),
            ],
        )
        .unwrap();
        let input =
            MemorySourceConfig::try_new_exec(&[vec![batch]], Arc::clone(&schema), None).unwrap();
        let evaluations = Arc::new(AtomicUsize::new(0));
        let calls = Arc::new(Mutex::new(Vec::new()));
        let filter: Arc<dyn ExecutionPlan> = Arc::new(
            FilterExecBuilder::new(
                is_not_null(ensure_loaded_call(
                    &calls,
                    counted_raster_call(&evaluations, col("rast", &schema).unwrap(), &schema),
                    &schema,
                ))
                .unwrap(),
                input,
            )
            .apply_projection(Some(vec![1]))
            .unwrap()
            .with_fetch(Some(3))
            .build()
            .unwrap(),
        );
        let expected_schema = filter.schema();
        assert_eq!(expected_schema.fields().len(), 1);

        let plan = RasterBatchBudgetRule
            .optimize(filter, &ConfigOptions::default())
            .unwrap();
        assert_eq!(plan.schema(), expected_schema);
        let filter = plan.downcast_ref::<FilterExec>().expect("filter kept");
        assert_eq!(filter.projection().as_deref(), Some(&[1][..]));
        assert_eq!(filter.fetch(), Some(3));
        assert_eq!(filter.input().schema().fields().len(), 3);

        let batches = collect(plan.execute(0, task_context(600)).unwrap())
            .await
            .unwrap();
        let ids: Vec<i32> = batches
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect();
        assert_eq!(ids, vec![1, 2, 3]);
        assert_eq!(evaluations.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn slices_are_sized_by_every_loader_argument() {
        // Two loader calls over two raster columns: each row costs the sum
        // of both estimates, 512 bytes, so a 600-byte budget takes one row
        // per slice.
        let (raster_schema, a) = raster_batch(&[Some(16); 6]);
        let (_, b) = raster_batch(&[Some(16); 6]);
        let schema = Arc::new(Schema::new(vec![
            raster_schema.field(0).clone().with_name("a"),
            raster_schema.field(0).clone().with_name("b"),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::clone(a.column(0)), Arc::clone(b.column(0))],
        )
        .unwrap();
        let input =
            MemorySourceConfig::try_new_exec(&[vec![batch]], Arc::clone(&schema), None).unwrap();
        let calls = Arc::new(Mutex::new(Vec::new()));
        let projection: Arc<dyn ExecutionPlan> = Arc::new(
            ProjectionExec::try_new(
                vec![
                    ProjectionExpr::new(
                        ensure_loaded_call(&calls, col("a", &schema).unwrap(), &schema),
                        "la",
                    ),
                    ProjectionExpr::new(
                        ensure_loaded_call(&calls, col("b", &schema).unwrap(), &schema),
                        "lb",
                    ),
                ],
                input,
            )
            .unwrap(),
        );
        let plan = RasterBatchBudgetRule
            .optimize(projection, &ConfigOptions::default())
            .unwrap();
        let split = plan
            .downcast_ref::<ProjectionExec>()
            .unwrap()
            .input()
            .downcast_ref::<RasterBatchSplitExec>()
            .unwrap();
        assert_eq!(split.sized_columns(), &[0, 1]);

        let batches = collect(plan.execute(0, task_context(600)).unwrap())
            .await
            .unwrap();
        let rows: Vec<usize> = batches.iter().map(RecordBatch::num_rows).collect();
        assert_eq!(rows, vec![1; 6]);
        assert_eq!(*calls.lock().unwrap(), vec![1; 12]);
    }

    #[tokio::test]
    async fn calls_sharing_an_argument_read_one_hoisted_column() {
        // The same expression argument under an unconditional call and under
        // a CASE branch: hoisted once (the unconditional call needs it for
        // every row anyway) and both calls read the column.
        let (schema, batch) = raster_batch(&[Some(16); 6]);
        let input =
            MemorySourceConfig::try_new_exec(&[vec![batch]], Arc::clone(&schema), None).unwrap();
        let evaluations = Arc::new(AtomicUsize::new(0));
        let calls = Arc::new(Mutex::new(Vec::new()));
        let loaded = || {
            ensure_loaded_call(
                &calls,
                counted_raster_call(&evaluations, col("rast", &schema).unwrap(), &schema),
                &schema,
            )
        };
        let in_branch: Arc<dyn PhysicalExpr> = Arc::new(
            CaseExpr::try_new(
                None,
                vec![(is_null(col("rast", &schema).unwrap()).unwrap(), loaded())],
                Some(col("rast", &schema).unwrap()),
            )
            .unwrap(),
        );
        let projection: Arc<dyn ExecutionPlan> = Arc::new(
            ProjectionExec::try_new(
                vec![
                    ProjectionExpr::new(loaded(), "loaded"),
                    ProjectionExpr::new(in_branch, "maybe"),
                ],
                input,
            )
            .unwrap(),
        );
        let plan = RasterBatchBudgetRule
            .optimize(projection, &ConfigOptions::default())
            .unwrap();
        let projection = plan.downcast_ref::<ProjectionExec>().unwrap();
        let split = projection
            .input()
            .downcast_ref::<RasterBatchSplitExec>()
            .unwrap();
        assert_eq!(split.sized_columns(), &[1]);
        assert_eq!(
            split.input().schema().fields().len(),
            2,
            "one hoisted column"
        );
        for p in projection.expr() {
            assert!(
                p.expr
                    .to_string()
                    .contains("rs_ensureloaded(__ensure_loaded_arg_0@1)"),
                "{}",
                p.expr
            );
        }

        let batches = collect(plan.execute(0, task_context(600)).unwrap())
            .await
            .unwrap();
        let rows: Vec<usize> = batches.iter().map(RecordBatch::num_rows).collect();
        assert_eq!(rows, vec![2, 2, 2]);
        assert_eq!(evaluations.load(Ordering::SeqCst), 1);
        // The branch is never taken, so only the unconditional call runs.
        assert_eq!(*calls.lock().unwrap(), vec![2, 2, 2]);
    }
}
