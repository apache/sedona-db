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

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use arrow_schema::Schema;

use datafusion::execution::context::QueryPlanner;
use datafusion::execution::session_state::{SessionState, SessionStateBuilder};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner};
use datafusion_common::{plan_err, DFSchema, Result};
use datafusion_expr::logical_plan::UserDefinedLogicalNode;
use datafusion_expr::LogicalPlan;
use datafusion_physical_expr::create_physical_expr;
use datafusion_physical_plan::joins::utils::JoinFilter;
use datafusion_physical_plan::joins::NestedLoopJoinExec;
use sedona_common::sedona_internal_err;

use crate::exec::SpatialJoinExec;
use crate::planner::logical_plan_node::SpatialJoinPlanNode;
use crate::planner::spatial_expr_utils::{is_spatial_predicate_supported, transform_join_filter};
use crate::spatial_predicate::SpatialPredicate;
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
        _planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        session_state: &SessionState,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let Some(spatial_node) = node.as_any().downcast_ref::<SpatialJoinPlanNode>() else {
            return Ok(None);
        };

        let Some(ext) = session_state
            .config_options()
            .extensions
            .get::<SedonaOptions>()
        else {
            return sedona_internal_err!("SedonaOptions not found in session state extensions");
        };

        if !ext.spatial_join.enable {
            return sedona_internal_err!("Spatial join is disabled in SedonaOptions");
        }

        if logical_inputs.len() != 2 || physical_inputs.len() != 2 {
            return plan_err!("SpatialJoinPlanNode expects 2 inputs");
        }

        let join_type = &spatial_node.join_type;

        let (physical_left, physical_right) =
            (physical_inputs[0].clone(), physical_inputs[1].clone());

        let join_filter = logical_join_filter_to_physical(
            spatial_node,
            session_state,
            &physical_left,
            &physical_right,
        )?;

        let Some((spatial_predicate, remainder)) = transform_join_filter(&join_filter) else {
            let nlj = NestedLoopJoinExec::try_new(
                physical_left,
                physical_right,
                Some(join_filter),
                join_type,
                None,
            )?;
            return Ok(Some(Arc::new(nlj)));
        };

        if !is_spatial_predicate_supported(
            &spatial_predicate,
            &physical_left.schema(),
            &physical_right.schema(),
        )? {
            let nlj = NestedLoopJoinExec::try_new(
                physical_left,
                physical_right,
                Some(join_filter),
                join_type,
                None,
            )?;
            return Ok(Some(Arc::new(nlj)));
        }

        let should_swap = !matches!(spatial_predicate, SpatialPredicate::KNearestNeighbors(_))
            && join_type.supports_swap()
            && should_swap_join_order(physical_left.as_ref(), physical_right.as_ref())?;

        let exec = SpatialJoinExec::try_new(
            physical_left,
            physical_right,
            spatial_predicate,
            remainder,
            join_type,
            None,
            &ext.spatial_join,
        )?;

        if should_swap {
            exec.swap_inputs().map(Some)
        } else {
            Ok(Some(Arc::new(exec) as Arc<dyn ExecutionPlan>))
        }
    }
}

fn should_swap_join_order(left: &dyn ExecutionPlan, right: &dyn ExecutionPlan) -> Result<bool> {
    let left_stats = left.partition_statistics(None)?;
    let right_stats = right.partition_statistics(None)?;

    match (
        left_stats.total_byte_size.get_value(),
        right_stats.total_byte_size.get_value(),
    ) {
        (Some(l), Some(r)) => Ok(l > r),
        _ => match (
            left_stats.num_rows.get_value(),
            right_stats.num_rows.get_value(),
        ) {
            (Some(l), Some(r)) => Ok(l > r),
            _ => Ok(false),
        },
    }
}

/// This function is mostly taken from the match arm for handling LogicalPlan::Join in
/// https://github.com/apache/datafusion/blob/51.0.0/datafusion/core/src/physical_planner.rs#L1144-L1245
fn logical_join_filter_to_physical(
    plan_node: &SpatialJoinPlanNode,
    session_state: &SessionState,
    physical_left: &Arc<dyn ExecutionPlan>,
    physical_right: &Arc<dyn ExecutionPlan>,
) -> Result<JoinFilter> {
    let SpatialJoinPlanNode {
        left,
        right,
        filter,
        ..
    } = plan_node;

    let left_df_schema = left.schema();
    let right_df_schema = right.schema();

    // Extract columns from filter expression and saved in a HashSet
    let cols = filter.column_refs();

    // Collect left & right field indices, the field indices are sorted in ascending order
    let mut left_field_indices = cols
        .iter()
        .filter_map(|c| left_df_schema.index_of_column(c).ok())
        .collect::<Vec<_>>();
    left_field_indices.sort_unstable();

    let mut right_field_indices = cols
        .iter()
        .filter_map(|c| right_df_schema.index_of_column(c).ok())
        .collect::<Vec<_>>();
    right_field_indices.sort_unstable();

    // Collect DFFields and Fields required for intermediate schemas
    let (filter_df_fields, filter_fields): (Vec<_>, Vec<_>) = left_field_indices
        .clone()
        .into_iter()
        .map(|i| {
            (
                left_df_schema.qualified_field(i),
                physical_left.schema().field(i).clone(),
            )
        })
        .chain(right_field_indices.clone().into_iter().map(|i| {
            (
                right_df_schema.qualified_field(i),
                physical_right.schema().field(i).clone(),
            )
        }))
        .unzip();
    let filter_df_fields = filter_df_fields
        .into_iter()
        .map(|(qualifier, field)| (qualifier.cloned(), Arc::new(field.clone())))
        .collect::<Vec<_>>();

    let metadata: HashMap<_, _> = left_df_schema
        .metadata()
        .clone()
        .into_iter()
        .chain(right_df_schema.metadata().clone())
        .collect();

    // Construct intermediate schemas used for filtering data and
    // convert logical expression to physical according to filter schema
    let filter_df_schema = DFSchema::new_with_metadata(filter_df_fields, metadata.clone())?;
    let filter_schema = Schema::new_with_metadata(filter_fields, metadata);

    let filter_expr =
        create_physical_expr(filter, &filter_df_schema, session_state.execution_props())?;
    let column_indices = JoinFilter::build_column_indices(left_field_indices, right_field_indices);

    let join_filter = JoinFilter::new(filter_expr, column_indices, Arc::new(filter_schema));
    Ok(join_filter)
}
