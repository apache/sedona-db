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
use sedona_spatial_join::SpatialJoinExec;

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
        Self {}
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
        if is_spatial_predicate_supported(
            args.spatial_predicate,
            args.physical_left.schema().as_ref(),
            args.physical_right.schema().as_ref(),
        )? {
            return Ok(None);
        }

        let exec = SpatialJoinExec::try_new(
            args.physical_left.clone(),
            args.physical_right.clone(),
            args.spatial_predicate.clone(),
            args.remainder.cloned(),
            args.join_type,
            None,
            args.join_options,
        )?;

        Ok(Some(Arc::new(exec.with_spatial_join_provider(Arc::new(
            GeographyJoinProvider::new(),
        )))))
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
