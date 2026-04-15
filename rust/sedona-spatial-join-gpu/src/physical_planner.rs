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

use crate::join_provider::GpuSpatialJoinProvider;
use crate::options::GpuOptions;
use arrow_schema::Schema;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_common::{DataFusionError, Result};
use datafusion_physical_expr::PhysicalExpr;
use sedona_common::SpatialJoinOptions;
use sedona_query_planner::spatial_join_physical_planner::{
    PlanSpatialJoinArgs, SpatialJoinPhysicalPlanner,
};
use sedona_query_planner::spatial_predicate::{
    RelationPredicate, SpatialPredicate, SpatialRelationType,
};
use sedona_schema::datatypes::SedonaType;
use sedona_schema::matchers::ArgMatcher;
use sedona_spatial_join::SpatialJoinExec;

/// [SpatialJoinFactory] implementation for the default spatial join
///
/// This struct is the entrypoint to ensuring the SedonaQueryPlanner is able
/// to instantiate the [ExecutionPlan] implemented in this crate.
#[derive(Debug)]
pub struct GpuSpatialJoinPhysicalPlanner;

impl GpuSpatialJoinPhysicalPlanner {
    /// Create a new default join factory
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for GpuSpatialJoinPhysicalPlanner {
    fn default() -> Self {
        Self::new()
    }
}

impl SpatialJoinPhysicalPlanner for GpuSpatialJoinPhysicalPlanner {
    fn plan_spatial_join(
        &self,
        args: &PlanSpatialJoinArgs<'_>,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let supported = is_spatial_predicate_supported(
            args.spatial_predicate,
            &args.physical_left.schema(),
            &args.physical_right.schema(),
        )?;
        let gpu_options = args
            .options
            .extensions
            .get::<GpuOptions>()
            .cloned()
            .unwrap_or_default();

        if !gpu_options.enable {
            return Ok(None);
        }

        if !supported {
            if gpu_options.fallback_to_cpu {
                log::warn!("Falling back to CPU spatial join as the spatial predicate is not supported on GPU");
                return Ok(None);
            } else {
                return Err(DataFusionError::Plan("GPU spatial join is enabled, but the spatial predicate is not supported on GPU".into()));
            }
        }

        let should_swap = !matches!(
            args.spatial_predicate,
            SpatialPredicate::KNearestNeighbors(_)
        ) && args.join_type.supports_swap()
            && should_swap_join_order(
                args.join_options,
                args.physical_left.as_ref(),
                args.physical_right.as_ref(),
            )?;

        let exec = SpatialJoinExec::try_new(
            args.physical_left.clone(),
            args.physical_right.clone(),
            args.spatial_predicate.clone(),
            args.remainder.cloned(),
            args.join_type,
            None,
            args.join_options,
        )?;
        let exec =
            exec.with_spatial_join_provider(Arc::new(GpuSpatialJoinProvider::new(gpu_options)));

        if should_swap {
            exec.swap_inputs().map(Some)
        } else {
            Ok(Some(Arc::new(exec) as Arc<dyn ExecutionPlan>))
        }
    }
}

/// Spatial join reordering heuristic:
/// 1. Put the input with fewer rows on the build side, because fewer entries
///    produce a smaller and more efficient spatial index (R-tree).
/// 2. If row-count statistics are unavailable (for example, for CSV sources),
///    fall back to total input size as an estimate.
/// 3. Do not swap the join order if join reordering is disabled or no relevant
///    statistics are available.
fn should_swap_join_order(
    spatial_join_options: &SpatialJoinOptions,
    left: &dyn ExecutionPlan,
    right: &dyn ExecutionPlan,
) -> Result<bool> {
    if !spatial_join_options.spatial_join_reordering {
        log::info!(
            "spatial join swap heuristic disabled via sedona.spatial_join.spatial_join_reordering"
        );
        return Ok(false);
    }

    let left_stats = left.partition_statistics(None)?;
    let right_stats = right.partition_statistics(None)?;

    let left_num_rows = left_stats.num_rows;
    let right_num_rows = right_stats.num_rows;
    let left_total_byte_size = left_stats.total_byte_size;
    let right_total_byte_size = right_stats.total_byte_size;

    let should_swap = match (left_num_rows.get_value(), right_num_rows.get_value()) {
        (Some(l), Some(r)) => l > r,
        _ => match (
            left_total_byte_size.get_value(),
            right_total_byte_size.get_value(),
        ) {
            (Some(l), Some(r)) => l > r,
            _ => false,
        },
    };

    log::info!(
        "spatial join swap heuristic: left_num_rows={left_num_rows:?}, right_num_rows={right_num_rows:?}, left_total_byte_size={left_total_byte_size:?}, right_total_byte_size={right_total_byte_size:?}, should_swap={should_swap}"
    );

    Ok(should_swap)
}

pub fn is_spatial_predicate_supported(
    spatial_predicate: &SpatialPredicate,
    left_schema: &Schema,
    right_schema: &Schema,
) -> Result<bool> {
    fn is_geometry_type_supported(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> Result<bool> {
        let return_field = expr.return_field(schema)?;
        let sedona_type = SedonaType::from_storage_field(&return_field)?;
        Ok(ArgMatcher::is_geometry().match_type(&sedona_type))
    }

    let both_geometry =
        |left: &Arc<dyn PhysicalExpr>, right: &Arc<dyn PhysicalExpr>| -> Result<bool> {
            Ok(is_geometry_type_supported(left, left_schema)?
                && is_geometry_type_supported(right, right_schema)?)
        };

    match spatial_predicate {
        SpatialPredicate::Relation(RelationPredicate {
            left,
            right,
            relation_type,
        }) => {
            if !matches!(
                relation_type,
                SpatialRelationType::Intersects
                    | SpatialRelationType::Contains
                    | SpatialRelationType::Within
                    | SpatialRelationType::Covers
                    | SpatialRelationType::CoveredBy
                    | SpatialRelationType::Touches
                    | SpatialRelationType::Equals
            ) {
                return Ok(false);
            }

            both_geometry(left, right)
        }
        SpatialPredicate::Distance(_) | SpatialPredicate::KNearestNeighbors(_) => Ok(false),
    }
}
