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

//! Query planner that delegates to DataFusion's [`DefaultPhysicalPlanner`]
//! with a configurable set of [`ExtensionPlanner`]s.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::execution::context::QueryPlanner;
use datafusion::execution::session_state::SessionState;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner};
use datafusion_common::Result;
use datafusion_expr::LogicalPlan;

use crate::extension_planner::{SpatialJoinExtensionPlanner, SpatialJoinFactory};

/// Query planner that wraps DataFusion's [`DefaultPhysicalPlanner`] with a set
/// of extension planners that handle custom logical nodes (e.g. spatial joins).
pub struct SedonaQueryPlanner {
    spatial_join_planner: SpatialJoinExtensionPlanner,
}

impl SedonaQueryPlanner {
    /// Create a new [`SedonaQueryPlanner`] with the given extension planners.
    pub fn new() -> Self {
        Self {
            spatial_join_planner: SpatialJoinExtensionPlanner::new(vec![]),
        }
    }

    pub fn with_spatial_join_factory(mut self, factory: Arc<dyn SpatialJoinFactory>) -> Self {
        self.spatial_join_planner
            .append_spatial_join_factory(factory);
        self
    }

    fn extension_planners(&self) -> Vec<Arc<dyn ExtensionPlanner + Send + Sync>> {
        vec![Arc::new(self.spatial_join_planner.clone())]
    }
}

impl Default for SedonaQueryPlanner {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for SedonaQueryPlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SedonaQueryPlanner").finish()
    }
}

#[async_trait]
impl QueryPlanner for SedonaQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session_state: &SessionState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let physical_planner =
            DefaultPhysicalPlanner::with_extension_planners(self.extension_planners());
        physical_planner
            .create_physical_plan(logical_plan, session_state)
            .await
    }
}
