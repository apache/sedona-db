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

use std::cmp::Ordering;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use datafusion::execution::context::QueryPlanner;
use datafusion::execution::session_state::{SessionState, SessionStateBuilder};
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner};
use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_common::{plan_err, DFSchemaRef, NullEquality, Result};
use datafusion_expr::logical_plan::{
    Extension, Join as LogicalJoin, UserDefinedLogicalNode, UserDefinedLogicalNodeCore,
};
use datafusion_expr::{BinaryExpr, Expr, JoinConstraint, JoinType, LogicalPlan, Operator};
use datafusion_physical_plan::joins::NestedLoopJoinExec;

use crate::exec::SpatialJoinExec;
use crate::planner::spatial_expr_utils::{
    collect_spatial_predicate_names, is_spatial_predicate_supported, transform_join_filter,
};
use sedona_common::option::SedonaOptions;

pub struct SedonaSpatialQueryPlanner;

impl fmt::Debug for SedonaSpatialQueryPlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SedonaSpatialQueryPlanner").finish()
    }
}

#[async_trait]
impl QueryPlanner for SedonaSpatialQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session_state: &SessionState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let physical_planner =
            DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(SpatialJoinPlanner {})]);
        physical_planner
            .create_physical_plan(logical_plan, session_state)
            .await
    }
}

/// Register logical rewrite and a query planner that can produce [`SpatialJoinExec`]
/// directly from a rewritten extension node.
pub fn register_spatial_join_planner(builder: SessionStateBuilder) -> SessionStateBuilder {
    builder
        .with_optimizer_rule(Arc::new(SpatialJoinLogicalRewrite::default()))
        .with_query_planner(Arc::new(SedonaSpatialQueryPlanner))
}

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

        // v1: only rewrite joins that already have a spatial predicate in `filter`.
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

#[derive(PartialEq, Eq, Hash)]
struct SpatialJoinPlanNode {
    left: LogicalPlan,
    right: LogicalPlan,
    join_type: JoinType,
    filter: Expr,
    schema: DFSchemaRef,
    join_constraint: JoinConstraint,
    null_equality: NullEquality,
}

// Manual implementation needed because of `schema` field. Comparison excludes this field.
// See https://github.com/apache/datafusion/blob/52.1.0/datafusion/expr/src/logical_plan/plan.rs#L3886
impl PartialOrd for SpatialJoinPlanNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        #[derive(PartialEq, PartialOrd)]
        struct ComparableJoin<'a> {
            pub left: &'a LogicalPlan,
            pub right: &'a LogicalPlan,
            pub filter: &'a Expr,
            pub join_type: &'a JoinType,
            pub join_constraint: &'a JoinConstraint,
            pub null_equality: &'a NullEquality,
        }
        let comparable_self = ComparableJoin {
            left: &self.left,
            right: &self.right,
            filter: &self.filter,
            join_type: &self.join_type,
            join_constraint: &self.join_constraint,
            null_equality: &self.null_equality,
        };
        let comparable_other = ComparableJoin {
            left: &other.left,
            right: &other.right,
            filter: &other.filter,
            join_type: &other.join_type,
            join_constraint: &other.join_constraint,
            null_equality: &self.null_equality,
        };
        comparable_self
            .partial_cmp(&comparable_other)
            // TODO (https://github.com/apache/datafusion/issues/17477) avoid recomparing all fields
            .filter(|cmp| *cmp != Ordering::Equal || self == other)
    }
}

impl fmt::Debug for SpatialJoinPlanNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        UserDefinedLogicalNodeCore::fmt_for_explain(self, f)
    }
}

impl UserDefinedLogicalNodeCore for SpatialJoinPlanNode {
    fn name(&self) -> &str {
        "SpatialJoin"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.left, &self.right]
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    fn expressions(&self) -> Vec<Expr> {
        vec![self.filter.clone()]
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "SpatialJoin: join_type={:?}, filter={}",
            self.join_type, self.filter
        )
    }

    fn with_exprs_and_inputs(
        &self,
        mut exprs: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> Result<Self> {
        if exprs.len() != 1 {
            return plan_err!("SpatialJoinPlanNode expects 1 expr");
        }
        if inputs.len() != 2 {
            return plan_err!("SpatialJoinPlanNode expects 2 inputs");
        }
        Ok(Self {
            left: inputs.swap_remove(0),
            right: inputs.swap_remove(0),
            join_type: self.join_type,
            filter: exprs.swap_remove(0),
            schema: Arc::clone(&self.schema),
            join_constraint: self.join_constraint,
            null_equality: self.null_equality,
        })
    }
}

struct SpatialJoinPlanner;

#[async_trait]
impl ExtensionPlanner for SpatialJoinPlanner {
    async fn plan_extension(
        &self,
        planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        session_state: &SessionState,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let Some(ext) = session_state
            .config_options()
            .extensions
            .get::<SedonaOptions>()
        else {
            return Ok(None);
        };

        if !ext.spatial_join.enable {
            return Ok(None);
        }

        let Some(spatial_node) = node.as_any().downcast_ref::<SpatialJoinPlanNode>() else {
            return Ok(None);
        };
        if logical_inputs.len() != 2 || physical_inputs.len() != 2 {
            return plan_err!("SpatialJoinPlanNode expects 2 inputs");
        }

        // Delegate join planning to DataFusion using the *real* logical inputs.
        //
        // This avoids re-implementing DataFusion's NestedLoopJoinExec / JoinFilter construction,
        // and also avoids having to swap children into a template plan (which can be incorrect if
        // DataFusion reorders join sides).
        let df_join = LogicalJoin::try_new(
            Arc::new(spatial_node.left.clone()),
            Arc::new(spatial_node.right.clone()),
            vec![],
            Some(spatial_node.filter.clone()),
            spatial_node.join_type,
            JoinConstraint::On,
            spatial_node.null_equality,
        )?;
        let df_join_plan = LogicalPlan::Join(df_join);

        let planned_join = planner
            .create_physical_plan(&df_join_plan, session_state)
            .await?;

        let transformed = planned_join
            .transform_up(|plan| {
                let Some(nlj) = plan.as_any().downcast_ref::<NestedLoopJoinExec>() else {
                    return Ok(Transformed::no(plan));
                };
                let Some(join_filter) = nlj.filter() else {
                    return Ok(Transformed::no(plan));
                };
                let Some((spatial_predicate, remainder)) = transform_join_filter(join_filter)
                else {
                    return Ok(Transformed::no(plan));
                };

                // If the build side was previously coerced to a single partition by other rules,
                // drop that wrapper for SpatialJoinExec.
                let left = nlj.left().clone();
                let right = nlj.right().clone();

                if !is_spatial_predicate_supported(
                    &spatial_predicate,
                    &left.schema(),
                    &right.schema(),
                )? {
                    return Ok(Transformed::no(plan));
                }

                let exec = SpatialJoinExec::try_new(
                    left,
                    right,
                    spatial_predicate,
                    remainder,
                    nlj.join_type(),
                    nlj.projection().cloned(),
                    &ext.spatial_join,
                )?;

                Ok(Transformed::yes(Arc::new(exec) as Arc<dyn ExecutionPlan>))
            })?
            .data;

        Ok(Some(transformed))
    }
}
