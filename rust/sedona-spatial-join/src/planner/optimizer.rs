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
use crate::planner::spatial_expr_utils::is_spatial_predicate;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};
use datafusion_common::tree_node::Transformed;
use datafusion_common::NullEquality;
use datafusion_common::Result;
use datafusion_expr::logical_plan::Extension;
use datafusion_expr::{BinaryExpr, Expr, Operator};
use datafusion_expr::{Filter, Join, JoinType, LogicalPlan};
use sedona_common::option::SedonaOptions;

/// Register only the logical spatial join optimizer rule.
///
/// This enables building `Join(filter=...)` from patterns like `Filter(CrossJoin)`.
/// It intentionally does not register any physical plan rewrite rules.
pub(crate) fn register_spatial_join_logical_optimizer(
    session_state_builder: SessionStateBuilder,
) -> SessionStateBuilder {
    session_state_builder
        .with_optimizer_rule(Arc::new(MergeSpatialProjectionIntoJoin))
        .with_optimizer_rule(Arc::new(SpatialJoinLogicalRewrite))
}
/// Logical optimizer rule that enables spatial join planning.
///
/// This rule turns eligible `Join(filter=...)` nodes into a `SpatialJoinPlanNode` extension.
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

        // Build new filter expression including equi-join conditions
        let filter = filter.clone();
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
}

/// Logical optimizer rule that enables spatial join planning.
///
/// This rule turns eligible `Filter(Join(filter=...))` nodes into a `Join(filter=...)` node,
/// so that the spatial join can be rewritten later by [SpatialJoinLogicalRewrite].
#[derive(Debug, Default)]
struct MergeSpatialProjectionIntoJoin;

impl OptimizerRule for MergeSpatialProjectionIntoJoin {
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
        if !is_spatial_predicate(predicate) {
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
        if !matches!(
            join_type,
            JoinType::Inner | JoinType::Left | JoinType::Right
        ) || !on.is_empty()
            || filter.is_some()
        {
            return Ok(Transformed::no(plan));
        }

        let rewritten_plan = Join::try_new(
            Arc::clone(left),
            Arc::clone(right),
            on.clone(),
            Some(predicate.clone()),
            JoinType::Inner,
            *join_constraint,
            *null_equality,
        )?;

        Ok(Transformed::yes(LogicalPlan::Join(rewritten_plan)))
    }
}
