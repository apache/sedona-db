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

use datafusion::physical_plan::ExecutionPlan;
use datafusion_common::Result;
use sedona_query_planner::spatial_join_physical_planner::{
    PlanSpatialJoinArgs, SpatialJoinPhysicalPlanner,
};

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
        _args: &PlanSpatialJoinArgs<'_>,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        // TODO: Implement geography-based spatial join planning
        unimplemented!("Geography spatial join planning is not yet implemented")
    }
}
