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
use std::sync::Arc;

use crate::planner::logical_plan_node::SpatialJoinPlanNode;
use crate::planner::spatial_expr_utils::{
    collect_spatial_predicate_names, find_knn_query_side, KNNJoinQuerySide,
};
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::optimizer::{ApplyOrder, Optimizer, OptimizerConfig, OptimizerRule};
use datafusion_common::tree_node::Transformed;
use datafusion_common::{NullEquality, Result};
use datafusion_expr::logical_plan::Extension;
use datafusion_expr::utils::{conjunction, split_conjunction};
use datafusion_expr::{BinaryExpr, Expr, Operator};
use datafusion_expr::{Filter, Join, JoinType, LogicalPlan};
use sedona_common::option::SedonaOptions;
use sedona_common::{sedona_internal_datafusion_err, sedona_internal_err};

/// Register the logical spatial join optimizer rules.
///
/// This inserts rules at specific positions relative to DataFusion's built-in `PushDownFilter`
/// rule to ensure correct semantics for KNN joins:
///
/// - `MergeSpatialFilterIntoJoin` and `KnnJoinEarlyRewrite` are inserted *before*
///   `PushDownFilter` so that KNN joins are converted to `SpatialJoinPlanNode` extension nodes
///   before filter pushdown runs. Extension nodes naturally block filter pushdown via
///   `prevent_predicate_push_down_columns()`, preventing incorrect pushdown to the build side
///   of KNN joins.
///
/// - `KnnQuerySideFilterPushdown` is inserted *before* `PushDownFilter` (but after
///   `KnnJoinEarlyRewrite`) to selectively push query-side-only filters below the
///   `SpatialJoinPlanNode` extension node. This allows the subsequent `PushDownFilter` to push
///   those filters further down into scan nodes in the same optimizer pass. DataFusion's
///   built-in `PushDownFilter` cannot push through extension nodes because they block all
///   pushdown by default, and the built-in rule pushes the same predicate to ALL children
///   (which would fail for a filter referencing only one side's columns).
///
/// - `SpatialJoinLogicalRewrite` is appended at the end so that non-KNN spatial joins still
///   benefit from filter pushdown before being converted to extension nodes.
pub(crate) fn register_spatial_join_logical_optimizer(
    mut session_state_builder: SessionStateBuilder,
) -> Result<SessionStateBuilder> {
    let optimizer = session_state_builder
        .optimizer()
        .get_or_insert_with(Optimizer::new);

    // Find PushDownFilter position by name
    let push_down_pos = optimizer
        .rules
        .iter()
        .position(|r| r.name() == "push_down_filter")
        .ok_or_else(|| {
            sedona_internal_datafusion_err!(
                "PushDownFilter rule not found in default optimizer rules"
            )
        })?;

    // Insert KNN-specific rules BEFORE PushDownFilter. Each `insert` at the same
    // index pushes earlier insertions forward, so we insert in reverse order.
    //
    // Effective order:
    //   1. MergeSpatialFilterIntoJoin — merges spatial predicates into Join(filter=...)
    //   2. KnnJoinEarlyRewrite       — converts KNN joins to SpatialJoinPlanNode
    //   3. KnnQuerySideFilterPushdown — pushes query-side filters below the extension node
    //   4. PushDownFilter (built-in)  — pushes filters further into scan nodes
    optimizer
        .rules
        .insert(push_down_pos, Arc::new(KnnQuerySideFilterPushdown));
    optimizer
        .rules
        .insert(push_down_pos, Arc::new(KnnJoinEarlyRewrite));
    optimizer
        .rules
        .insert(push_down_pos, Arc::new(MergeSpatialFilterIntoJoin));

    // Append SpatialJoinLogicalRewrite at the end so non-KNN joins benefit from filter pushdown.
    optimizer.rules.push(Arc::new(SpatialJoinLogicalRewrite));

    Ok(session_state_builder)
}

/// Early optimizer rule that converts KNN joins to `SpatialJoinPlanNode` extension nodes
/// *before* DataFusion's `PushDownFilter` runs.
///
/// This prevents `PushDownFilter` from pushing filters to the build (object) side of KNN joins,
/// which would change which objects are the K nearest neighbors and produce incorrect results.
///
/// Extension nodes naturally block filter pushdown because their default
/// `prevent_predicate_push_down_columns()` returns all columns.
///
/// Handles two patterns that DataFusion's SQL planner creates:
///
/// 1. `Join(filter=ST_KNN(...))` — when the ON clause has only the spatial predicate
/// 2. `Filter(ST_KNN(...), Join(on=[...]))` — when the ON clause also has equi-join conditions,
///    the SQL planner separates equi-conditions into `on` and the spatial predicate into a Filter
#[derive(Default, Debug)]
struct KnnJoinEarlyRewrite;

impl OptimizerRule for KnnJoinEarlyRewrite {
    fn name(&self) -> &str {
        "knn_join_early_rewrite"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let options = config.options();
        let Some(ext) = options.extensions.get::<SedonaOptions>() else {
            return Ok(Transformed::no(plan));
        };
        if !ext.spatial_join.enable {
            return Ok(Transformed::no(plan));
        }

        // Join(filter=ST_KNN(...))
        if let LogicalPlan::Join(join) = &plan {
            if let Some(filter) = join.filter.as_ref() {
                let names = collect_spatial_predicate_names(filter);
                if names.contains("st_knn") {
                    return rewrite_join_to_spatial_join_plan_node(join);
                }
            }
        }

        Ok(Transformed::no(plan))
    }
}

/// Logical optimizer rule that converts non-KNN spatial joins to `SpatialJoinPlanNode`.
///
/// This rule runs *after* `PushDownFilter` so that non-KNN spatial joins benefit from
/// filter pushdown before being converted to extension nodes.
///
/// KNN joins are skipped here because they are already handled by [[KnnJoinEarlyRewrite]].
#[derive(Default, Debug)]
struct SpatialJoinLogicalRewrite;

impl OptimizerRule for SpatialJoinLogicalRewrite {
    fn name(&self) -> &str {
        "spatial_join_logical_rewrite"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let options = config.options();
        let Some(ext) = options.extensions.get::<SedonaOptions>() else {
            return Ok(Transformed::no(plan));
        };
        if !ext.spatial_join.enable {
            return Ok(Transformed::no(plan));
        }

        let LogicalPlan::Join(join) = &plan else {
            return Ok(Transformed::no(plan));
        };

        // only rewrite joins that already have a spatial predicate in `filter`.
        let Some(filter) = join.filter.as_ref() else {
            return Ok(Transformed::no(plan));
        };

        let spatial_predicate_names = collect_spatial_predicate_names(filter);
        if spatial_predicate_names.is_empty() {
            return Ok(Transformed::no(plan));
        }

        if spatial_predicate_names.contains("st_knn") {
            // KNN joins should have already been rewritten by KnnJoinEarlyRewrite, so we shouldn't
            // see them here.
            return sedona_internal_err!(
                "Found KNN predicate in SpatialJoinLogicalRewrite, which should have been handled by KnnJoinEarlyRewrite");
        }

        // Join with with equi-join condition should be planned as a regular HashJoin
        // or SortMergeJoin.
        if !join.on.is_empty() {
            return Ok(Transformed::no(plan));
        }

        rewrite_join_to_spatial_join_plan_node(join)
    }
}

/// Shared helper: convert a `Join` node (with spatial predicate in `filter`) to a
/// `SpatialJoinPlanNode`, folding any equi-join `on` conditions into the filter expression.
fn rewrite_join_to_spatial_join_plan_node(join: &Join) -> Result<Transformed<LogicalPlan>> {
    let filter = join
        .filter
        .as_ref()
        .ok_or_else(|| {
            datafusion_common::DataFusionError::Internal(
                "join filter must be present for spatial join rewrite".to_string(),
            )
        })?
        .clone();

    let eq_op = if join.null_equality == NullEquality::NullEqualsNothing {
        Operator::Eq
    } else {
        Operator::IsNotDistinctFrom
    };
    let filter = join.on.iter().fold(filter, |acc, (l, r)| {
        let eq_expr = Expr::BinaryExpr(BinaryExpr::new(
            Box::new(l.clone()),
            eq_op,
            Box::new(r.clone()),
        ));
        Expr::and(acc, eq_expr)
    });

    let schema = Arc::clone(&join.schema);
    let node = SpatialJoinPlanNode {
        left: join.left.as_ref().clone(),
        right: join.right.as_ref().clone(),
        join_type: join.join_type,
        filter,
        schema,
        join_constraint: join.join_constraint,
        null_equality: join.null_equality,
    };

    Ok(Transformed::yes(LogicalPlan::Extension(Extension {
        node: Arc::new(node),
    })))
}

/// Logical optimizer rule that enables spatial join planning.
///
/// This rule turns eligible `Filter(Join(filter=...))` nodes into a `Join(filter=...)` node,
/// so that the spatial join can be rewritten later by [SpatialJoinLogicalRewrite].
#[derive(Debug, Default)]
struct MergeSpatialFilterIntoJoin;

impl OptimizerRule for MergeSpatialFilterIntoJoin {
    fn name(&self) -> &str {
        "merge_spatial_filter_into_join"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    /// Try to rewrite the plan containing a spatial Filter on top of a cross join without on or filter
    /// to a theta-join with filter. For instance, the following query plan:
    ///
    /// ```text
    /// Filter: st_intersects(l.geom, _scalar_sq_1.geom)
    ///   Left Join (no on, no filter):
    ///     TableScan: l projection=[id, geom]
    ///     SubqueryAlias: __scalar_sq_1
    ///       Projection: r.geom
    ///         Filter: r.id = Int32(1)
    ///           TableScan: r projection=[id, geom]
    /// ```
    ///
    /// will be rewritten to
    ///
    /// ```text
    /// Inner Join: Filter: st_intersects(l.geom, _scalar_sq_1.geom)
    ///   TableScan: l projection=[id, geom]
    ///   SubqueryAlias: __scalar_sq_1
    ///     Projection: r.geom
    ///       Filter: r.id = Int32(1)
    ///         TableScan: r projection=[id, geom]
    /// ```
    ///
    /// This is for enabling this logical join operator to be converted to a [SpatialJoinPlanNode]
    /// by [SpatialJoinLogicalRewrite], so that it could subsequently be optimized to a SpatialJoin
    /// physical node.
    fn rewrite(
        &self,
        plan: LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let options = config.options();
        let Some(extension) = options.extensions.get::<SedonaOptions>() else {
            return Ok(Transformed::no(plan));
        };
        if !extension.spatial_join.enable {
            return Ok(Transformed::no(plan));
        }

        let LogicalPlan::Filter(Filter {
            predicate, input, ..
        }) = &plan
        else {
            return Ok(Transformed::no(plan));
        };

        let spatial_predicates = collect_spatial_predicate_names(predicate);
        if spatial_predicates.is_empty() {
            return Ok(Transformed::no(plan));
        }

        let LogicalPlan::Join(Join {
            ref left,
            ref right,
            ref on,
            ref filter,
            join_type,
            ref join_constraint,
            ref null_equality,
            ..
        }) = input.as_ref()
        else {
            return Ok(Transformed::no(plan));
        };

        // Check if this is a suitable join for rewriting
        let is_equi_join = !on.is_empty() && !spatial_predicates.contains("st_knn");
        if !matches!(
            join_type,
            JoinType::Inner | JoinType::Left | JoinType::Right
        ) || is_equi_join
        {
            return Ok(Transformed::no(plan));
        }

        let new_filter = match filter {
            Some(existing_filter) => Expr::and(predicate.clone(), existing_filter.clone()),
            None => predicate.clone(),
        };

        let rewritten_plan = Join::try_new(
            Arc::clone(left),
            Arc::clone(right),
            on.clone(),
            Some(new_filter),
            JoinType::Inner,
            *join_constraint,
            *null_equality,
        )?;

        Ok(Transformed::yes(LogicalPlan::Join(rewritten_plan)))
    }
}

/// Optimizer rule that pushes query-side-only filters below the `SpatialJoinPlanNode` extension
/// node for KNN joins.
///
/// This rule handles the pattern:
/// ```text
/// Filter(predicate)
///   Extension(SpatialJoinPlanNode { filter contains ST_KNN, ... })
/// ```
///
/// For each conjunct in `predicate`, if ALL referenced columns exist in the query-side child's
/// schema, the conjunct is pushed down as a `Filter` wrapping the query-side child. Remaining
/// conjuncts stay above the extension node.
#[derive(Default, Debug)]
struct KnnQuerySideFilterPushdown;

impl OptimizerRule for KnnQuerySideFilterPushdown {
    fn name(&self) -> &str {
        "knn_query_side_filter_pushdown"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        // Match: Filter(predicate, Extension(SpatialJoinPlanNode))
        let LogicalPlan::Filter(ref filter) = plan else {
            return Ok(Transformed::no(plan));
        };

        let LogicalPlan::Extension(ref ext) = *filter.input else {
            return Ok(Transformed::no(plan));
        };

        let Some(spatial_join) = ext.node.as_any().downcast_ref::<SpatialJoinPlanNode>() else {
            return Ok(Transformed::no(plan));
        };

        // The SpatialJoinPlanNode should be a KNN join (i.e. its filter should contain ST_KNN),
        // otherwise we shouldn't be seeing it here. This SpatialJoinPlanNode should be produced
        // by KnnJoinEarlyRewrite.
        let spatial_predicate_names = collect_spatial_predicate_names(&spatial_join.filter);
        if !spatial_predicate_names.contains("st_knn") {
            return Ok(Transformed::no(plan));
        }

        // Find which child is the query side (first argument of ST_KNN)
        let query_side = find_knn_query_side(
            &spatial_join.filter,
            spatial_join.left.schema(),
            spatial_join.right.schema(),
        );
        let Some(query_side) = query_side else {
            return Ok(Transformed::no(plan));
        };

        let query_child_schema = match query_side {
            KNNJoinQuerySide::Left => spatial_join.left.schema(),
            KNNJoinQuerySide::Right => spatial_join.right.schema(),
        };

        // Split the filter predicate into conjuncts
        let conjuncts = split_conjunction(&filter.predicate);

        let mut push_down: Vec<Expr> = Vec::new();
        let mut keep_above: Vec<Expr> = Vec::new();

        for conjunct in conjuncts {
            let col_refs = conjunct.column_refs();
            if !col_refs.is_empty()
                && col_refs
                    .iter()
                    .all(|col| query_child_schema.has_column(col))
            {
                push_down.push(conjunct.clone());
            } else {
                keep_above.push(conjunct.clone());
            }
        }

        // If nothing to push down, return unchanged
        if push_down.is_empty() {
            return Ok(Transformed::no(plan));
        }

        let pushed_predicate =
            conjunction(push_down).expect("push_down is non-empty, conjunction should return Some");

        // Wrap the query-side child in a Filter node
        let (new_left, new_right) = match query_side {
            KNNJoinQuerySide::Left => {
                let new_left = LogicalPlan::Filter(Filter::try_new(
                    pushed_predicate,
                    Arc::new(spatial_join.left.clone()),
                )?);
                (new_left, spatial_join.right.clone())
            }
            KNNJoinQuerySide::Right => {
                let new_right = LogicalPlan::Filter(Filter::try_new(
                    pushed_predicate,
                    Arc::new(spatial_join.right.clone()),
                )?);
                (spatial_join.left.clone(), new_right)
            }
        };

        // Rebuild the SpatialJoinPlanNode with the new children
        let new_spatial_join = SpatialJoinPlanNode {
            left: new_left,
            right: new_right,
            join_type: spatial_join.join_type,
            filter: spatial_join.filter.clone(),
            schema: Arc::clone(&spatial_join.schema),
            join_constraint: spatial_join.join_constraint,
            null_equality: spatial_join.null_equality,
        };

        let new_extension = LogicalPlan::Extension(Extension {
            node: Arc::new(new_spatial_join),
        });

        // If there are remaining predicates, wrap in a Filter
        let result = if let Some(remaining) = conjunction(keep_above) {
            LogicalPlan::Filter(Filter::try_new(remaining, Arc::new(new_extension))?)
        } else {
            new_extension
        };

        Ok(Transformed::yes(result))
    }
}
