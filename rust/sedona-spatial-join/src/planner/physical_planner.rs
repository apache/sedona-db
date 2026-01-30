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

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use datafusion::execution::context::QueryPlanner;
use datafusion::execution::session_state::{SessionState, SessionStateBuilder};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner};
use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_common::{plan_err, Result};
use datafusion_expr::logical_plan::{Join as LogicalJoin, UserDefinedLogicalNode};
use datafusion_expr::{JoinConstraint, LogicalPlan};
use datafusion_physical_plan::joins::NestedLoopJoinExec;

use crate::exec::SpatialJoinExec;
use crate::planner::logical_plan_node::SpatialJoinPlanNode;
use crate::planner::spatial_expr_utils::{is_spatial_predicate_supported, transform_join_filter};
use sedona_common::option::SedonaOptions;

/// Registers a query planner that can produce [`SpatialJoinExec`] from a logical extension node.
pub fn register_spatial_join_planner(builder: SessionStateBuilder) -> SessionStateBuilder {
    builder.with_query_planner(Arc::new(SedonaSpatialQueryPlanner))
}

/// Query planner that enables Sedona's spatial join planning.
///
/// Installs an [`ExtensionPlanner`] that recognizes `SpatialJoinPlanNode` and produces
/// `SpatialJoinExec` when supported and enabled.
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
        let physical_planner = DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(
            SpatialJoinExtensionPlanner {},
        )]);
        physical_planner
            .create_physical_plan(logical_plan, session_state)
            .await
    }
}

/// Physical planner hook for `SpatialJoinPlanNode`.
struct SpatialJoinExtensionPlanner;

#[async_trait]
impl ExtensionPlanner for SpatialJoinExtensionPlanner {
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
