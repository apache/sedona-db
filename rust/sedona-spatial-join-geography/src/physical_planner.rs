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

use arrow_schema::Schema;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_common::Result;
use datafusion_physical_expr::PhysicalExpr;
use sedona_query_planner::{
    spatial_join_physical_planner::{PlanSpatialJoinArgs, SpatialJoinPhysicalPlanner},
    spatial_predicate::{RelationPredicate, SpatialPredicate, SpatialRelationType},
};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};
use sedona_spatial_join::{
    physical_planner::{repartition_probe_side, should_swap_join_order},
    SpatialJoinExec,
};

use crate::join_provider::GeographyJoinProvider;

/// [SpatialJoinPhysicalPlanner] implementation for geography-based spatial joins.
///
/// This struct provides the entrypoint for the SedonaQueryPlanner to instantiate
/// geography-aware spatial join execution plans.
#[derive(Debug)]
pub struct GeographySpatialJoinPhysicalPlanner;

impl GeographySpatialJoinPhysicalPlanner {
    /// Create a new geography join planner
    pub fn new() -> Self {
        Self
    }
}

impl Default for GeographySpatialJoinPhysicalPlanner {
    fn default() -> Self {
        Self::new()
    }
}

impl SpatialJoinPhysicalPlanner for GeographySpatialJoinPhysicalPlanner {
    fn plan_spatial_join(
        &self,
        args: &PlanSpatialJoinArgs<'_>,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        if !is_spatial_predicate_supported(
            args.spatial_predicate,
            &args.physical_left.schema(),
            &args.physical_right.schema(),
        )? {
            return Ok(None);
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

        // Repartition the probe side when enabled. This breaks spatial locality in sorted/skewed
        // datasets, leading to more balanced workloads during out-of-core spatial join.
        // We determine which pre-swap input will be the probe AFTER any potential swap, and
        // repartition it here. swap_inputs() will then carry the RepartitionExec to the correct
        // child position.
        let (physical_left, physical_right) = if args.join_options.repartition_probe_side {
            repartition_probe_side(
                args.physical_left.clone(),
                args.physical_right.clone(),
                args.spatial_predicate,
                should_swap,
            )?
        } else {
            (args.physical_left.clone(), args.physical_right.clone())
        };

        let exec = SpatialJoinExec::try_new(
            physical_left,
            physical_right,
            args.spatial_predicate.clone(),
            args.remainder.cloned(),
            args.join_type,
            None,
            args.join_options,
        )?
        .with_spatial_join_provider(Arc::new(GeographyJoinProvider));

        if should_swap {
            exec.swap_inputs().map(Some)
        } else {
            Ok(Some(Arc::new(exec) as Arc<dyn ExecutionPlan>))
        }
    }
}

pub fn is_spatial_predicate_supported(
    spatial_predicate: &SpatialPredicate,
    left_schema: &Schema,
    right_schema: &Schema,
) -> Result<bool> {
    fn is_geometry_type_supported(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> Result<bool> {
        let return_field = expr.return_field(schema)?;
        let sedona_type = SedonaType::from_storage_field(&return_field)?;
        Ok(ArgMatcher::is_geography().match_type(&sedona_type))
    }

    let both_geography =
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
            if !both_geography(left, right)? {
                return Ok(false);
            }

            if matches!(
                relation_type,
                SpatialRelationType::Intersects
                    | SpatialRelationType::Contains
                    | SpatialRelationType::Within
                    | SpatialRelationType::Equals
            ) {
                return Ok(true);
            }
        }
        SpatialPredicate::Distance(d) => {
            return both_geography(&d.left, &d.right);
        }
        _ => {}
    }

    Ok(false)
}
