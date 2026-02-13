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
use crate::planner::spatial_expr_utils::collect_spatial_predicate_names;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::optimizer::{ApplyOrder, Optimizer, OptimizerConfig, OptimizerRule};
use datafusion_common::tree_node::Transformed;
use datafusion_common::NullEquality;
use datafusion_common::Result;
use datafusion_expr::logical_plan::Extension;
use datafusion_expr::{BinaryExpr, Expr, Operator};
use datafusion_expr::{Filter, Join, JoinType, LogicalPlan};
use sedona_common::option::SedonaOptions;

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
/// - `SpatialJoinLogicalRewrite` is appended at the end so that non-KNN spatial joins still
///   benefit from filter pushdown before being converted to extension nodes.
pub(crate) fn register_spatial_join_logical_optimizer(
    mut session_state_builder: SessionStateBuilder,
) -> SessionStateBuilder {
    let optimizer = session_state_builder
        .optimizer()
        .get_or_insert_with(Optimizer::new);

    // Find PushDownFilter position by name
    let push_down_pos = optimizer
        .rules
        .iter()
        .position(|r| r.name() == "push_down_filter")
        .expect("PushDownFilter rule not found in default optimizer rules");

    // Insert KNN-specific rules BEFORE PushDownFilter.
    // MergeSpatialFilterIntoJoin must come first because it creates the Join(filter=...)
    // nodes that KnnJoinEarlyRewrite then converts to SpatialJoinPlanNode.
    optimizer
        .rules
        .insert(push_down_pos, Arc::new(KnnJoinEarlyRewrite));
    optimizer
        .rules
        .insert(push_down_pos, Arc::new(MergeSpatialFilterIntoJoin));

    // Append SpatialJoinLogicalRewrite at the end so non-KNN joins benefit from filter pushdown.
    optimizer.rules.push(Arc::new(SpatialJoinLogicalRewrite));

    session_state_builder
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
/// KNN joins are skipped here because they are already handled by `KnnJoinEarlyRewrite`.
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

        // Join with with equi-join condition and spatial join condition. Only handle it
        // when the join condition contains ST_KNN. KNN join is not a regular join and
        // ST_KNN is also not a regular predicate. It must be handled by our spatial join exec.
        if !join.on.is_empty() && !spatial_predicate_names.contains("st_knn") {
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
        .expect("join filter must be present")
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
