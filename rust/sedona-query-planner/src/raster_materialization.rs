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

//! Physical planning of raster materialization: byte-bounded load batches,
//! pass-through filters over raster streams, and loads spread by raster
//! source.
//!
//! `datafusion.execution.batch_size` counts rows, but a row holding a
//! raster can carry megabytes of pixels, so a batch of a few thousand OutDb
//! references is gigabytes once `RS_EnsureLoaded` has loaded it.
//! `RS_EnsureLoaded` is an ordinary scalar UDF: whichever node's expression
//! calls it (a `ProjectionExec`, `FilterExec` or `AggregateExec`, once the
//! `ensure_loaded` logical rule has wrapped the raster arguments of pixel
//! functions) evaluates it over each input batch in full.
//!
//! [`RasterMaterializationRule`] therefore puts a [`RasterBatchSplitExec`] under
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
//!
//! The rule also decides *where* rasters are loaded. The planner usually
//! feeds the loading node through a round-robin repartition, which deals
//! whole batches, and a raster catalog is often tiny (one Parquet row
//! group, a couple of batches), so all the loading lands on a couple of
//! partitions while the rest idle. The rule therefore hash-partitions the
//! split's input by raster source (see [`raster_source_spread`] and
//! [`RasterSourceKeyExpr`]): hashing is per row, so even a two-batch
//! catalog spreads over every partition, and the rows of one raster share
//! a partition, so the chunk cache loads each raster once instead of once
//! per partition that happens to hold some of its rows.

use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use arrow_array::{Array, RecordBatch, StructArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{Transformed, TransformedResult, TreeNode, TreeNodeRecursion};
use datafusion_common::{Result, internal_err};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_expr::{ColumnarValue, Operator};
use datafusion_physical_expr::expressions::{BinaryExpr, CaseExpr, Column};
use datafusion_physical_expr::{PhysicalExpr, ScalarFunctionExpr};
use datafusion_physical_plan::aggregates::{
    AggregateExec, AggregateInputMode, AggregateMode, PhysicalGroupBy,
};
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::execution_plan::CardinalityEffect;
use datafusion_physical_plan::filter::{FilterExec, FilterExecBuilder};
use datafusion_physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use datafusion_physical_plan::projection::{ProjectionExec, ProjectionExpr};
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

/// Physical optimizer rule that plans raster materialization:
///
/// - puts a [`RasterBatchSplitExec`] under every `ProjectionExec`,
///   `FilterExec` and `AggregateExec` whose expressions call
///   `rs_ensureloaded`, sized by the calls' raster arguments, and
///   hash-partitions the split's input by raster source where that is safe
///   (see [`raster_source_spread`]);
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
pub struct RasterMaterializationRule;

impl PhysicalOptimizerRule for RasterMaterializationRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let target_partitions = config.execution.target_partitions;
        plan.transform_up(|node| {
            // The pass-through batch size first: a loading filter is rebuilt
            // below with the batch size it has at that point.
            forward_raster_filter_batches(node)?
                .transform_data(|node| split_under_loading_node(node, target_partitions))
        })
        .data()
    }

    fn name(&self) -> &str {
        "sedona.raster_materialization"
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
    /// The first sized argument, which keys the raster-source spread. When
    /// several `rs_ensureloaded` calls share a node they usually read
    /// rasters from the same catalog row, and one key is all a
    /// partitioning can follow.
    key: Option<SizedKey>,
}

/// Where a sized argument is read from (see [`SizedArgs`]).
#[derive(Debug, Clone, Copy)]
enum SizedKey {
    /// An input column.
    Column(usize),
    /// The `k`-th appended expression column.
    Expr(usize),
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
                self.key.get_or_insert(SizedKey::Column(column.index()));
            } else if !conditional {
                let k = self
                    .exprs
                    .iter()
                    .position(|known| known.as_ref() == arg.as_ref())
                    .unwrap_or_else(|| {
                        self.exprs.push(Arc::clone(arg));
                        self.exprs.len() - 1
                    });
                self.key.get_or_insert(SizedKey::Expr(k));
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
    /// The [`RasterBatchSplitExec`], over the raster-source repartition when
    /// the input is spread, over a `ProjectionExec` that appends the hoisted
    /// argument expressions when there are any.
    plan: Arc<dyn ExecutionPlan>,
    hoisted: Vec<HoistedArg>,
    /// The input was a single partition that is now spread, so the rebuilt
    /// node is merged back to one partition (see [`merged_back`]).
    merge_back: bool,
}

/// The spread of a loading node's `input` by raster source: the plan to
/// hash (`input` itself, or what the planner's own repartitioning below the
/// node reads, see [`spread_source`]) and the number of partitions, or
/// `None` when the spread is skipped.
///
/// Why hash rather than the round-robin the planner already put below the
/// node: round-robin deals whole batches, and a raster catalog is often one
/// or two batches, so only one or two partitions ever load anything. A hash
/// is computed per row, so it spreads any number of batches over every
/// partition, and it keeps all rows of one raster in one partition. That
/// locality is what keeps a load per raster: the chunk cache then sees each
/// raster's rows in order on one stream instead of several partitions
/// missing on the same raster at once.
///
/// The spread is skipped when repartitioning could only cost, or could
/// break a requirement of a parent. The rule runs after
/// `EnforceDistribution`, `EnforceSorting` and `SanityCheckPlan`, so
/// whatever it changes is not checked again:
///
/// - a single target partition: nothing to spread over;
/// - an input with an ordering: a parent may rely on it, and the node
///   otherwise maintains it;
/// - an input already hash partitioned (for example the probe side of a
///   partitioned join): a parent may require exactly that distribution, so
///   it keeps the spread it already has.
///
/// The node can rule it out too (see [`split_under_loading_node`]): a
/// `FilterExec` with a fetch, and an aggregate that is not a `Partial` one.
///
/// The planner's own repartitioning is replaced rather than stacked on: a
/// round-robin (`RepartitionExec(RoundRobinBatch)` without
/// `preserve_order`) or a merge (`CoalescePartitionsExec` without a fetch)
/// directly below the node is dropped and its input hashed instead, so the
/// rows cross one repartition, not two.
///
/// A single input partition is spread too, then the node is merged back
/// with a `CoalescePartitionsExec` so its output is still one partition.
/// The planner leaves an input that small when its statistics say it fits
/// in one batch (a catalog of a few thousand rasters does) or when a parent
/// requires a single partition (a `CollectLeft` build side, a global
/// limit); either way the loads are the expensive part and run in parallel,
/// and whatever is above still sees one partition.
///
/// The spread has `max(target_partitions, input partitions)` partitions, so
/// an input wider than the target (a `UnionExec`, say) keeps its width.
///
/// The repartition decouples the producer below from the loads: a fast
/// producer (a spatial join) fills the repartition's channels while the
/// loads drain them slowly, so rows queue there, still as metadata-only
/// references. Under a memory limit the repartition spills them; without one
/// they cost resident memory (about 2 GB on SpatialBench Raster Q5 at SF1).
/// Nor does the spread look at a `LIMIT` above, so a limited query may pull
/// more input than it needs.
///
/// The rule does not try to detect an input that is already well spread:
/// the plan cannot tell how many batches a partition will deliver, and the
/// cost of the repartition is one hash of rows that are still metadata-only
/// OutDb references (about 0.9 s for 6M joined rows, a few milliseconds for
/// a ten-thousand-raster catalog), while the locality it adds is worth
/// having either way.
fn raster_source_spread(
    input: &Arc<dyn ExecutionPlan>,
    target_partitions: usize,
) -> Option<(Arc<dyn ExecutionPlan>, usize)> {
    if target_partitions <= 1
        || input.output_ordering().is_some()
        || matches!(input.output_partitioning(), Partitioning::Hash(_, _))
    {
        return None;
    }
    let source = spread_source(input);
    let partitions = target_partitions
        .max(input.output_partitioning().partition_count())
        .max(source.output_partitioning().partition_count());
    Some((source, partitions))
}

/// The plan to hash for [`raster_source_spread`]: `input` itself, or the
/// input of a round-robin or merge directly below the loading node. Hashing
/// redistributes every row anyway, so a round-robin under it is a wasted
/// hop, and a merge under it is undone by the hash (the merge back to one
/// partition happens above the node instead).
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

/// The subtree under a loading node whose input is `input` and whose
/// slices are sized by `sized`. `spread_target` is the session's target
/// partitions when the node allows its input to be spread by raster source
/// (see [`raster_source_spread`]), `None` when it does not.
///
/// From the top: the [`RasterBatchSplitExec`], the raster-source
/// `RepartitionExec` when the input is spread, the `ProjectionExec` that
/// appends the hoisted argument expressions when there are any, then the
/// input (or what the planner's own repartitioning below the node read).
/// The hash and the sizing both read a hoisted argument from its appended
/// column, so it is evaluated exactly once per batch, on the input's
/// partitions (for a single-partition input, on that one partition): the
/// price of evaluating it once rather than again after the spread, which
/// for `RS_FromPath` would repeat every file open.
fn split_input(
    input: &Arc<dyn ExecutionPlan>,
    sized: &SizedArgs,
    spread_target: Option<usize>,
) -> Result<SplitInput> {
    let spread = spread_target.and_then(|target| raster_source_spread(input, target));
    let source = match &spread {
        Some((source, _)) => Arc::clone(source),
        None => Arc::clone(input),
    };
    let source_schema = source.schema();
    let input_columns = source_schema.fields().len();
    let mut sized_columns = sized.columns.clone();
    let mut hoisted = Vec::with_capacity(sized.exprs.len());
    let args = if sized.exprs.is_empty() {
        source
    } else {
        let mut projection: Vec<ProjectionExpr> = source_schema
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
        Arc::new(ProjectionExec::try_new(projection, source)?) as Arc<dyn ExecutionPlan>
    };
    let below_split = match spread {
        Some((_, partitions)) => {
            let key_column = match sized.key {
                Some(SizedKey::Column(idx)) => idx,
                Some(SizedKey::Expr(k)) => input_columns + k,
                None => return internal_err!("sized arguments without a key"),
            };
            let key = Column::new(args.schema().field(key_column).name(), key_column);
            Arc::new(RepartitionExec::try_new(
                args,
                Partitioning::Hash(
                    vec![Arc::new(RasterSourceKeyExpr::new(Arc::new(key)))],
                    partitions,
                ),
            )?)
        }
        None => args,
    };
    Ok(SplitInput {
        plan: Arc::new(RasterBatchSplitExec::try_new(below_split, sized_columns)?),
        hoisted,
        merge_back: spread.is_some() && input.output_partitioning().partition_count() == 1,
    })
}

/// The rebuilt loading `node` as the rule leaves it: merged back to one
/// partition when `merge` says its single input partition was spread.
fn merged_back(node: Arc<dyn ExecutionPlan>, merge: bool) -> Arc<dyn ExecutionPlan> {
    if merge {
        Arc::new(CoalescePartitionsExec::new(node))
    } else {
        node
    }
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
/// `rs_ensureloaded`, and spread its input by raster source over
/// `target_partitions` where the node allows it (see [`split_input`]).
fn split_under_loading_node(
    node: Arc<dyn ExecutionPlan>,
    target_partitions: usize,
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
        let split = split_input(projection.input(), &sized, Some(target_partitions))?;
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
        let projection = Arc::new(ProjectionExec::try_new(exprs, split.plan)?);
        return Ok(Transformed::yes(merged_back(projection, split.merge_back)));
    }

    if let Some(filter) = node.downcast_ref::<FilterExec>() {
        let sized = SizedArgs::collect(std::slice::from_ref(filter.predicate()));
        if sized.is_empty() {
            return Ok(Transformed::no(node));
        }
        // A filter with a fetch is not spread: over a single partition the
        // fetch is the query's limit, which a spread merged back would
        // multiply by the partitions.
        let spread_target = filter.fetch().is_none().then_some(target_partitions);
        let split = split_input(filter.input(), &sized, spread_target)?;
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
        return Ok(Transformed::yes(merged_back(
            Arc::new(filter),
            split.merge_back,
        )));
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
        // Only a `Partial` stage runs on any number of partitions, so only
        // its input is spread; a `Single` stage over a single partition
        // must stay one, loads and all, and a `SinglePartitioned` one has
        // a hash-partitioned input. A limit on the stage is a per-partition
        // limit and is left alone like a filter's fetch.
        let spread_target = (*aggregate.mode() == AggregateMode::Partial
            && aggregate.limit_options().is_none())
        .then_some(target_partitions);
        let split = split_input(aggregate.input(), &sized, spread_target)?;
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
                    "RasterMaterializationRule: aggregate {} rejected its rewritten arguments",
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
        return Ok(Transformed::yes(merged_back(
            Arc::new(aggregate),
            split.merge_back,
        )));
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
/// partitioning are the input's, except over a raster-source spread (see
/// [`RasterBatchSplitExec::try_new`]). See the module docs.
#[derive(Debug)]
pub struct RasterBatchSplitExec {
    input: Arc<dyn ExecutionPlan>,
    /// Input columns whose per-row estimated bytes, summed, size the slices.
    sized_columns: Vec<usize>,
    cache: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl RasterBatchSplitExec {
    /// Over a raster-source spread (a `RepartitionExec` hashed by a
    /// [`RasterSourceKeyExpr`]) the split reports
    /// `Partitioning::UnknownPartitioning` over that many partitions instead
    /// of passing `Hash([raster_source(..)], n)` up. That hash is not a
    /// distribution any other plan can match: rows without an out-DB file
    /// draw their key from a running counter, so evaluating the same
    /// expression elsewhere would not reproduce it. The spread is a
    /// load-balancing device, not a guarantee, and no parent (nor a later
    /// rewrite of the plan) can rely on it.
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
        let cache = match input.output_partitioning() {
            Partitioning::Hash(keys, n)
                if keys
                    .iter()
                    .any(|key| key.downcast_ref::<RasterSourceKeyExpr>().is_some()) =>
            {
                Arc::new(
                    input
                        .properties()
                        .as_ref()
                        .clone()
                        .with_partitioning(Partitioning::UnknownPartitioning(*n)),
                )
            }
            _ => Arc::clone(input.properties()),
        };
        Ok(Self {
            cache,
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
/// volatile, and a [`RasterBatchSplitExec`] above the repartition advertises
/// `UnknownPartitioning` rather than this hash (see
/// [`RasterBatchSplitExec::try_new`]). Hashing fallback rows
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
    use std::collections::HashMap;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow_array::{ArrayRef, Int32Array, Int64Array};
    use arrow_schema::{DataType, Field, FieldRef, Schema, SchemaRef};
    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::datasource::source::DataSourceExec;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::functions_aggregate::count::count_udaf;
    use datafusion::prelude::SessionConfig;
    use datafusion_expr::{
        ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
        Volatility,
    };
    use datafusion_physical_expr::aggregate::AggregateExprBuilder;
    use datafusion_physical_expr::expressions::{binary, col, is_not_null, is_null, lit};
    use datafusion_physical_expr::{LexOrdering, PhysicalSortExpr};
    use datafusion_physical_plan::aggregates::AggregateMode;
    use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
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
        // One target partition: the test observes a single stream's slices.
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 1;
        let plan = RasterMaterializationRule
            .optimize(projection, &config)
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
        // One target partition: the test observes a single stream's slices.
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 1;
        let plan = RasterMaterializationRule
            .optimize(projection, &config)
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
        // One target partition: the test observes a single stream's slices.
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 1;
        let plan = RasterMaterializationRule
            .optimize(projection, &config)
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
        // One target partition: the test observes a single stream's slices.
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 1;
        let plan = RasterMaterializationRule.optimize(filter, &config).unwrap();
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
        // One target partition: the test observes a single stream's slices.
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 1;
        let plan = RasterMaterializationRule.optimize(filter, &config).unwrap();
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
        // One target partition: the test observes a single stream's slices.
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 1;
        let plan = RasterMaterializationRule
            .optimize(aggregate, &config)
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
                    .downcast_ref::<Int64Array>()
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
        // One target partition: the test observes a single stream's slices.
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 1;
        let plan = RasterMaterializationRule
            .optimize(filtered, &config)
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
        // One target partition: the test observes a single stream's slices.
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 1;
        let plan = RasterMaterializationRule
            .optimize(Arc::clone(&projection), &config)
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

        // One target partition: the test observes a single stream's slices.
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 1;
        let plan = RasterMaterializationRule
            .optimize(Arc::clone(&filter), &config)
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
        // One target partition: the test observes a single stream's slices.
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 1;
        let plan = RasterMaterializationRule
            .optimize(Arc::clone(&projection), &config)
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
            let plan = RasterMaterializationRule
                .optimize(Arc::clone(&projection), &config)
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
        // One target partition: the test observes a single stream's slices.
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 1;

        let plan = RasterMaterializationRule
            .optimize(final_stage, &config)
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

        let plan = RasterMaterializationRule
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
        // One target partition: the test observes a single stream's slices.
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 1;
        let plan = RasterMaterializationRule
            .optimize(projection, &config)
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
        // One target partition: the test observes a single stream's slices.
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 1;
        let plan = RasterMaterializationRule
            .optimize(projection, &config)
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

    /// `ProjectionExec(id, rast, call AS loaded)` over `input`: the shape the
    /// planner gives `SELECT id, rast, RS_EnsureLoaded(..) AS loaded`.
    fn loading_projection(
        call: Arc<dyn PhysicalExpr>,
        input: Arc<dyn ExecutionPlan>,
    ) -> Arc<dyn ExecutionPlan> {
        let schema = input.schema();
        Arc::new(
            ProjectionExec::try_new(
                vec![
                    ProjectionExpr::new(col("id", &schema).unwrap(), "id"),
                    ProjectionExpr::new(col("rast", &schema).unwrap(), "rast"),
                    ProjectionExpr::new(call, "loaded"),
                ],
                input,
            )
            .unwrap(),
        )
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
    async fn loading_node_input_is_hash_partitioned_by_raster_source() {
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
        let projection = loading_projection(
            ensure_loaded_call(&calls, col("rast", &schema).unwrap(), &schema),
            input,
        );
        let expected_schema = projection.schema();
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 4;
        let plan = RasterMaterializationRule
            .optimize(projection, &config)
            .unwrap();

        // Projection ← Split ← hash RepartitionExec ← scan
        assert_eq!(plan.schema(), expected_schema);
        let split = plan
            .downcast_ref::<ProjectionExec>()
            .expect("projection kept")
            .input()
            .downcast_ref::<RasterBatchSplitExec>()
            .expect("split under the projection");
        assert_eq!(split.sized_columns(), &[1]);
        let repartition = split.input().downcast_ref::<RepartitionExec>().unwrap();
        let Partitioning::Hash(keys, 4) = repartition.partitioning() else {
            panic!("expected a 4-way hash, got {}", repartition.partitioning());
        };
        assert_eq!(keys[0].to_string(), "raster_source(rast@1)");
        assert!(
            repartition
                .input()
                .downcast_ref::<DataSourceExec>()
                .is_some()
        );
        // The counter-drawn keys of rows without a file are not reproducible,
        // so the hash is not advertised to parents.
        assert!(matches!(
            split.properties().output_partitioning(),
            Partitioning::UnknownPartitioning(4)
        ));
        assert!(matches!(
            plan.output_partitioning(),
            Partitioning::UnknownPartitioning(4)
        ));
        assert!(plan.output_ordering().is_none());

        let key = RasterSourceKeyExpr::new(col("rast", &schema).unwrap());
        let ctx = task_context(DEFAULT_RASTER_MAX_BATCH_BYTES);
        let mut partition_of = HashMap::new();
        let mut rows = 0;
        let mut busy = 0;
        for partition in 0..4 {
            let batches = collect(plan.execute(partition, Arc::clone(&ctx)).unwrap())
                .await
                .unwrap();
            let mut here = 0;
            for batch in &batches {
                assert_eq!(batch.schema(), expected_schema);
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
    fn raster_source_spread_is_skipped_where_it_could_break_a_parent() {
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
            let projection = loading_projection(
                ensure_loaded_call(
                    &Arc::new(Mutex::new(Vec::new())),
                    col("rast", &schema).unwrap(),
                    &schema,
                ),
                Arc::clone(&input),
            );
            let mut config = ConfigOptions::default();
            config.execution.target_partitions = target_partitions;
            let plan = RasterMaterializationRule
                .optimize(projection, &config)
                .unwrap();
            // The split still goes in, directly over the input.
            let split = plan
                .downcast_ref::<ProjectionExec>()
                .unwrap_or_else(|| panic!("{name}: got {}", plan.name()))
                .input()
                .downcast_ref::<RasterBatchSplitExec>()
                .unwrap_or_else(|| panic!("{name}: no split"));
            assert!(Arc::ptr_eq(split.input(), &input), "{name}");
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
        let projection = loading_projection(
            ensure_loaded_call(&calls, col("rast", &schema).unwrap(), &schema),
            input,
        );
        let expected_schema = projection.schema();
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 4;
        let plan = RasterMaterializationRule
            .optimize(projection, &config)
            .unwrap();

        // Coalesce ← Projection ← Split ← hash RepartitionExec ← scan
        assert_eq!(plan.schema(), expected_schema);
        assert_eq!(plan.output_partitioning().partition_count(), 1);
        let merge = plan.downcast_ref::<CoalescePartitionsExec>().unwrap();
        let split = merge
            .input()
            .downcast_ref::<ProjectionExec>()
            .unwrap()
            .input()
            .downcast_ref::<RasterBatchSplitExec>()
            .unwrap();
        let repartition = split.input().downcast_ref::<RepartitionExec>().unwrap();
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
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 16);
        // One invocation per busy partition, not one for the whole input.
        assert!(calls.lock().unwrap().len() > 1);
    }

    #[tokio::test]
    async fn expression_raster_argument_is_evaluated_once_below_the_repartition() {
        // Hashing `RS_FromPath(path)`-style arguments directly would evaluate
        // them twice (key, then load). The argument is hoisted into a column
        // under the repartition instead, so it is evaluated once per input
        // batch and both the hash and the sizing read the column.
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
        let projection = loading_projection(
            ensure_loaded_call(
                &calls,
                counted_raster_call(&evaluations, col("rast", &schema).unwrap(), &schema),
                &schema,
            ),
            input,
        );
        let expected_schema = projection.schema();
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 4;
        let plan = RasterMaterializationRule
            .optimize(projection, &config)
            .unwrap();

        // Projection ← Split ← hash RepartitionExec ← Projection (evaluate
        // the argument) ← scan
        assert_eq!(plan.schema(), expected_schema);
        let top = plan.downcast_ref::<ProjectionExec>().unwrap();
        assert_eq!(
            top.expr()[2].expr.to_string(),
            "rs_ensureloaded(__ensure_loaded_arg_0@2)"
        );
        let split = top.input().downcast_ref::<RasterBatchSplitExec>().unwrap();
        assert_eq!(split.sized_columns(), &[2]);
        let repartition = split.input().downcast_ref::<RepartitionExec>().unwrap();
        let Partitioning::Hash(keys, 4) = repartition.partitioning() else {
            panic!("expected a 4-way hash, got {}", repartition.partitioning());
        };
        assert_eq!(
            keys[0].to_string(),
            "raster_source(__ensure_loaded_arg_0@2)"
        );
        let below = repartition
            .input()
            .downcast_ref::<ProjectionExec>()
            .unwrap();
        assert_eq!(
            below
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().as_str())
                .collect::<Vec<_>>(),
            vec!["id", "rast", "__ensure_loaded_arg_0"]
        );
        assert!(below.input().downcast_ref::<DataSourceExec>().is_some());
        assert!(matches!(
            plan.output_partitioning(),
            Partitioning::UnknownPartitioning(4)
        ));
        // The raster columns keep their extension metadata through both
        // projections.
        let raster_metadata = SedonaType::Raster
            .to_storage_field("rast", true)
            .unwrap()
            .metadata()
            .clone();
        assert!(!raster_metadata.is_empty());
        assert_eq!(below.schema().field(1).metadata(), &raster_metadata);
        assert_eq!(below.schema().field(2).metadata(), &raster_metadata);
        assert_eq!(plan.schema().field(1).metadata(), &raster_metadata);
        assert_eq!(plan.schema().field(2).metadata(), &raster_metadata);

        let ctx = task_context(DEFAULT_RASTER_MAX_BATCH_BYTES);
        let mut rows = 0;
        for partition in 0..4 {
            for batch in collect(plan.execute(partition, Arc::clone(&ctx)).unwrap())
                .await
                .unwrap()
            {
                assert_eq!(batch.schema(), expected_schema);
                // Identity mocks: the loaded column is the input raster.
                assert_eq!(batch.column(1).as_ref(), batch.column(2).as_ref());
                rows += batch.num_rows();
            }
        }
        assert_eq!(rows, 16);
        assert_eq!(calls.lock().unwrap().iter().sum::<usize>(), 16);
        // Once per input batch, below the repartition; never again above it.
        assert_eq!(evaluations.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn expression_argument_over_a_single_input_partition_is_spread_and_merged_back() {
        let uris: Vec<String> = (0..16).map(|i| format!("s3://b/r{i}.tif")).collect();
        let uris: Vec<&str> = uris.iter().map(String::as_str).collect();
        let (schema, batch) = outdb_raster_batch(&uris);
        let input =
            MemorySourceConfig::try_new_exec(&[vec![batch]], Arc::clone(&schema), None).unwrap();
        let evaluations = Arc::new(AtomicUsize::new(0));
        let calls = Arc::new(Mutex::new(Vec::new()));
        let projection = loading_projection(
            ensure_loaded_call(
                &calls,
                counted_raster_call(&evaluations, col("rast", &schema).unwrap(), &schema),
                &schema,
            ),
            input,
        );
        let expected_schema = projection.schema();
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 4;
        let plan = RasterMaterializationRule
            .optimize(projection, &config)
            .unwrap();

        // Coalesce ← Projection ← Split ← hash RepartitionExec ← Projection
        // (evaluate the argument) ← scan
        assert_eq!(plan.schema(), expected_schema);
        assert_eq!(plan.output_partitioning().partition_count(), 1);
        let merge = plan.downcast_ref::<CoalescePartitionsExec>().unwrap();
        let split = merge
            .input()
            .downcast_ref::<ProjectionExec>()
            .unwrap()
            .input()
            .downcast_ref::<RasterBatchSplitExec>()
            .unwrap();
        let repartition = split.input().downcast_ref::<RepartitionExec>().unwrap();
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
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 16);
        for batch in &batches {
            assert_eq!(batch.column(1).as_ref(), batch.column(2).as_ref());
        }
        assert!(calls.lock().unwrap().len() > 1);
        // Evaluated once, on the single input partition, before the spread.
        assert_eq!(evaluations.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn spread_keys_on_the_first_call_when_it_is_an_expression_and_the_second_a_column() {
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
        let projection: Arc<dyn ExecutionPlan> = Arc::new(
            ProjectionExec::try_new(
                vec![
                    ProjectionExpr::new(col("id", &schema).unwrap(), "id"),
                    ProjectionExpr::new(col("rast", &schema).unwrap(), "rast"),
                    ProjectionExpr::new(
                        ensure_loaded_call(
                            &calls,
                            counted_raster_call(
                                &evaluations,
                                col("rast", &schema).unwrap(),
                                &schema,
                            ),
                            &schema,
                        ),
                        "a",
                    ),
                    ProjectionExpr::new(
                        ensure_loaded_call(&calls, col("rast", &schema).unwrap(), &schema),
                        "b",
                    ),
                ],
                input,
            )
            .unwrap(),
        );
        let expected_schema = projection.schema();
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 4;
        let plan = RasterMaterializationRule
            .optimize(projection, &config)
            .unwrap();

        assert_eq!(plan.schema(), expected_schema);
        let top = plan.downcast_ref::<ProjectionExec>().unwrap();
        let calls_shown: Vec<String> = top.expr()[2..].iter().map(|p| p.expr.to_string()).collect();
        assert_eq!(
            calls_shown,
            vec![
                "rs_ensureloaded(__ensure_loaded_arg_0@2)",
                "rs_ensureloaded(rast@1)"
            ]
        );
        let split = top.input().downcast_ref::<RasterBatchSplitExec>().unwrap();
        assert_eq!(split.sized_columns(), &[1, 2]);
        let repartition = split.input().downcast_ref::<RepartitionExec>().unwrap();
        let Partitioning::Hash(keys, 4) = repartition.partitioning() else {
            panic!("expected a 4-way hash, got {}", repartition.partitioning());
        };
        assert_eq!(
            keys[0].to_string(),
            "raster_source(__ensure_loaded_arg_0@2)"
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
        let loading = loading_projection(
            ensure_loaded_call(
                &Arc::new(Mutex::new(Vec::new())),
                counted_raster_call(
                    &Arc::new(AtomicUsize::new(0)),
                    col("rast", &schema).unwrap(),
                    &schema,
                ),
                &schema,
            ),
            input,
        );
        let filter: Arc<dyn ExecutionPlan> =
            Arc::new(FilterExec::try_new(lit(true), loading).unwrap());
        let filter_schema = filter.schema();
        let projection: Arc<dyn ExecutionPlan> = Arc::new(
            ProjectionExec::try_new(
                vec![
                    ProjectionExpr::new(col("id", &filter_schema).unwrap(), "id"),
                    ProjectionExpr::new(col("loaded", &filter_schema).unwrap(), "loaded"),
                ],
                filter,
            )
            .unwrap(),
        );
        let expected_schema = projection.schema();
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 4;
        let plan = RasterMaterializationRule
            .optimize(projection, &config)
            .unwrap();

        // Projection ← Filter ← Projection (load) ← Split ← hash.
        assert_eq!(plan.schema(), expected_schema);
        let top = plan.downcast_ref::<ProjectionExec>().unwrap();
        let filter = top.input().downcast_ref::<FilterExec>().unwrap();
        assert_eq!(filter.batch_size(), PASSTHROUGH_BATCH_SIZE);
        let split = filter
            .input()
            .downcast_ref::<ProjectionExec>()
            .unwrap()
            .input()
            .downcast_ref::<RasterBatchSplitExec>()
            .unwrap();
        assert!(split.input().downcast_ref::<RepartitionExec>().is_some());
        assert_eq!(plan.output_partitioning().partition_count(), 4);

        // Four-byte rasters under a four-byte budget: one row per slice, and
        // the filter forwards every slice instead of merging them.
        let ctx = task_context(4);
        let mut sizes = Vec::new();
        for partition in 0..4 {
            let batches = collect(plan.execute(partition, Arc::clone(&ctx)).unwrap())
                .await
                .unwrap();
            sizes.extend(batches.iter().map(RecordBatch::num_rows));
        }
        assert_eq!(sizes, vec![1; 16]);
    }

    #[test]
    fn planner_repartitioning_below_the_node_is_replaced_not_stacked() {
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
        // (case, input, plan hashed below the split, partitions, merged back)
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
            let projection = loading_projection(
                ensure_loaded_call(
                    &Arc::new(Mutex::new(Vec::new())),
                    col("rast", &schema).unwrap(),
                    &schema,
                ),
                input,
            );
            let mut config = ConfigOptions::default();
            config.execution.target_partitions = 4;
            let plan = RasterMaterializationRule
                .optimize(projection, &config)
                .unwrap();

            let loading = if merged {
                assert_eq!(plan.output_partitioning().partition_count(), 1, "{name}");
                plan.downcast_ref::<CoalescePartitionsExec>()
                    .unwrap_or_else(|| panic!("{name}: got {}", plan.name()))
                    .input()
            } else {
                assert_eq!(
                    plan.output_partitioning().partition_count(),
                    partitions,
                    "{name}"
                );
                &plan
            };
            let split = loading
                .downcast_ref::<ProjectionExec>()
                .unwrap_or_else(|| panic!("{name}: got {}", loading.name()))
                .input()
                .downcast_ref::<RasterBatchSplitExec>()
                .unwrap_or_else(|| panic!("{name}: no split"));
            let repartition = split.input().downcast_ref::<RepartitionExec>().unwrap();
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
        // The hoisted argument is appended at position 2 as
        // `__ensure_loaded_arg_0`; the user's Int32 column of that name sits
        // at position 0. Columns are bound by position, so the loader and the
        // hash still read the raster.
        let uris: Vec<String> = (0..8).map(|i| format!("s3://b/r{i}.tif")).collect();
        let uris: Vec<&str> = uris.iter().map(String::as_str).collect();
        let (outdb_schema, outdb) = outdb_raster_batch(&uris);
        let schema = Arc::new(Schema::new(vec![
            Field::new("__ensure_loaded_arg_0", DataType::Int32, false),
            outdb_schema.field(1).clone(),
        ]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), outdb.columns().to_vec()).unwrap();
        for target_partitions in [1, 4] {
            let input =
                MemorySourceConfig::try_new_exec(&[vec![batch.clone()]], Arc::clone(&schema), None)
                    .unwrap();
            let projection: Arc<dyn ExecutionPlan> = Arc::new(
                ProjectionExec::try_new(
                    vec![
                        ProjectionExpr::new(col("__ensure_loaded_arg_0", &schema).unwrap(), "n"),
                        ProjectionExpr::new(col("rast", &schema).unwrap(), "rast"),
                        ProjectionExpr::new(
                            ensure_loaded_call(
                                &Arc::new(Mutex::new(Vec::new())),
                                counted_raster_call(
                                    &Arc::new(AtomicUsize::new(0)),
                                    col("rast", &schema).unwrap(),
                                    &schema,
                                ),
                                &schema,
                            ),
                            "loaded",
                        ),
                    ],
                    input,
                )
                .unwrap(),
            );
            let expected_schema = projection.schema();
            let mut config = ConfigOptions::default();
            config.execution.target_partitions = target_partitions;
            let plan = RasterMaterializationRule
                .optimize(projection, &config)
                .unwrap();
            assert_eq!(plan.schema(), expected_schema);
            let batches = collect(
                plan.execute(0, task_context(DEFAULT_RASTER_MAX_BATCH_BYTES))
                    .unwrap(),
            )
            .await
            .unwrap();
            assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 8);
            for batch in &batches {
                assert_eq!(batch.column(1).as_ref(), batch.column(2).as_ref());
            }
        }
    }

    #[tokio::test]
    async fn only_partial_aggregates_and_unlimited_filters_are_spread() {
        // Over a single input partition: a `Partial` stage is spread and
        // merged back, while a `Single` stage must stay on one partition
        // (its result is the one aggregate) and a filter with a fetch is the
        // query's limit. Both still get the split.
        let uris: Vec<String> = (0..16).map(|i| format!("s3://b/r{i}.tif")).collect();
        let uris: Vec<&str> = uris.iter().map(String::as_str).collect();
        let (schema, batch) = outdb_raster_batch(&uris);
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut config = ConfigOptions::default();
        config.execution.target_partitions = 4;

        for (mode, spread) in [
            (AggregateMode::Partial, true),
            (AggregateMode::Single, false),
        ] {
            let input =
                MemorySourceConfig::try_new_exec(&[vec![batch.clone()]], Arc::clone(&schema), None)
                    .unwrap();
            let count = AggregateExprBuilder::new(
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
            .unwrap();
            let aggregate: Arc<dyn ExecutionPlan> = Arc::new(
                AggregateExec::try_new(
                    mode,
                    PhysicalGroupBy::new_single(vec![]),
                    vec![Arc::new(count)],
                    vec![None],
                    input,
                    Arc::clone(&schema),
                )
                .unwrap(),
            );
            let expected_schema = aggregate.schema();
            let plan = RasterMaterializationRule
                .optimize(aggregate, &config)
                .unwrap();
            assert_eq!(plan.schema(), expected_schema, "{mode:?}");
            assert_eq!(plan.output_partitioning().partition_count(), 1, "{mode:?}");
            let aggregate = if spread {
                plan.downcast_ref::<CoalescePartitionsExec>()
                    .unwrap_or_else(|| panic!("{mode:?}: got {}", plan.name()))
                    .input()
            } else {
                &plan
            }
            .downcast_ref::<AggregateExec>()
            .unwrap_or_else(|| panic!("{mode:?}: no aggregate"));
            let split = aggregate
                .input()
                .downcast_ref::<RasterBatchSplitExec>()
                .unwrap_or_else(|| panic!("{mode:?}: no split"));
            assert_eq!(
                split.input().downcast_ref::<RepartitionExec>().is_some(),
                spread,
                "{mode:?}"
            );

            let batches = collect(
                plan.execute(0, task_context(DEFAULT_RASTER_MAX_BATCH_BYTES))
                    .unwrap(),
            )
            .await
            .unwrap();
            let counts: Vec<i64> = batches
                .iter()
                .flat_map(|b| {
                    b.column(0)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                        .values()
                        .to_vec()
                })
                .collect();
            // A partial stage emits one state per busy partition, a single
            // stage the one result.
            assert_eq!(counts.iter().sum::<i64>(), 16, "{mode:?}");
            assert_eq!(counts.len() > 1, spread, "{mode:?}");
        }

        let input =
            MemorySourceConfig::try_new_exec(&[vec![batch.clone()]], Arc::clone(&schema), None)
                .unwrap();
        let filter: Arc<dyn ExecutionPlan> = Arc::new(
            FilterExecBuilder::new(
                is_not_null(ensure_loaded_call(
                    &calls,
                    col("rast", &schema).unwrap(),
                    &schema,
                ))
                .unwrap(),
                input,
            )
            .with_fetch(Some(3))
            .build()
            .unwrap(),
        );
        let plan = RasterMaterializationRule.optimize(filter, &config).unwrap();
        let filter = plan.downcast_ref::<FilterExec>().expect("filter kept");
        assert_eq!(filter.fetch(), Some(3));
        let split = filter
            .input()
            .downcast_ref::<RasterBatchSplitExec>()
            .expect("split under the filter");
        assert!(split.input().downcast_ref::<DataSourceExec>().is_some());
        let batches = collect(
            plan.execute(0, task_context(DEFAULT_RASTER_MAX_BATCH_BYTES))
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
    }
}
