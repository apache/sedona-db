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

//! Extension planner for spatial join plan nodes.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use arrow_schema::Schema;

use datafusion::config::ConfigOptions;
use datafusion::execution::session_state::SessionState;
use datafusion::physical_planner::{ExtensionPlanner, PhysicalPlanner};
use datafusion_common::{plan_err, DFSchema, Result};
use datafusion_expr::logical_plan::UserDefinedLogicalNode;
use datafusion_expr::{JoinType, LogicalPlan};
use datafusion_physical_expr::create_physical_expr;
use datafusion_physical_plan::joins::utils::JoinFilter;
use datafusion_physical_plan::joins::NestedLoopJoinExec;
use datafusion_physical_plan::ExecutionPlan;
use sedona_common::option::SedonaOptions;
use sedona_common::{sedona_internal_err, SpatialJoinOptions};

use crate::logical_plan_node::SpatialJoinPlanNode;
use crate::spatial_expr_utils::transform_join_filter;
use crate::spatial_predicate::SpatialPredicate;

/// Arguments passed to a [`SpatialJoinPhysicalPlanner`] when planning a spatial join.
pub struct PlanSpatialJoinArgs<'a> {
    pub physical_left: &'a Arc<dyn ExecutionPlan>,
    pub physical_right: &'a Arc<dyn ExecutionPlan>,
    pub spatial_predicate: &'a SpatialPredicate,
    pub remainder: Option<&'a JoinFilter>,
    pub join_type: &'a JoinType,
    pub join_options: &'a SpatialJoinOptions,
    pub options: &'a Arc<ConfigOptions>,
}

/// Factory trait for creating spatial join physical plans.
///
/// Implementations decide whether they can handle a given spatial predicate and,
/// if so, produce an appropriate [`ExecutionPlan`].
pub trait SpatialJoinPhysicalPlanner: std::fmt::Debug + Send + Sync {
    /// Given the provided arguments, produce an [ExecutionPlan] implementing
    /// the join operation or `None` if this implementation cannot execute the join
    fn plan_spatial_join(
        &self,
        args: &PlanSpatialJoinArgs<'_>,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>>;
}

/// Physical planner hook for [`SpatialJoinPlanNode`].
///
/// Delegates to a list of [`SpatialJoinFactory`] implementations to
/// produce a physical plan for spatial join nodes. Falls back to
/// [`NestedLoopJoinExec`] when no factory can handle the predicate.
#[derive(Clone, Debug)]
pub struct SpatialJoinExtensionPlanner {
    factories: Vec<Arc<dyn SpatialJoinPhysicalPlanner>>,
}

impl SpatialJoinExtensionPlanner {
    /// Create a new planner with the given factories.
    pub fn new(factories: Vec<Arc<dyn SpatialJoinPhysicalPlanner>>) -> Self {
        Self { factories }
    }

    /// Append a new join factory
    ///
    /// Implementations are checked in reverse order such that more recently added
    /// implementations can override the default join.
    pub fn append_spatial_join_physical_planner(
        &mut self,
        factory: Arc<dyn SpatialJoinPhysicalPlanner>,
    ) {
        self.factories.push(factory);
    }
}

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

        let args = PlanSpatialJoinArgs {
            physical_left: &physical_left,
            physical_right: &physical_right,
            spatial_predicate: &spatial_predicate,
            remainder: remainder.as_ref(),
            join_type,
            join_options: &ext.spatial_join,
            options: session_state.config_options(),
        };

        // Iterate over in reverse to handle more recently added factories first
        for factory in self.factories.iter().rev() {
            if let Some(exec) = factory.plan_spatial_join(&args)? {
                return Ok(Some(exec));
            }
        }

        let nlj = NestedLoopJoinExec::try_new(
            physical_left,
            physical_right,
            Some(join_filter),
            join_type,
            None,
        )?;

        Ok(Some(Arc::new(nlj)))
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
        .map(|(qualifier, field)| (qualifier.cloned(), field.clone()))
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
