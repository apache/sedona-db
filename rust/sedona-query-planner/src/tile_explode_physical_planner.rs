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

//! Extension planner for the [`TileExplodePlanNode`].

use std::sync::Arc;

use async_trait::async_trait;

use datafusion::execution::session_state::SessionState;
use datafusion::physical_planner::{ExtensionPlanner, PhysicalPlanner};
use datafusion_common::{plan_err, Result};
use datafusion_expr::logical_plan::UserDefinedLogicalNode;
use datafusion_expr::LogicalPlan;
use datafusion_physical_plan::ExecutionPlan;

use crate::tile_explode_node::TileExplodePlanNode;

/// Factory trait for building a physical plan for a [`TileExplodePlanNode`].
///
/// A factory resolves the node's argument expressions into concrete tile
/// parameters and produces a streaming tile-explode execution plan, or `None`
/// if it cannot handle the node.
pub trait TileExplodePhysicalPlanner: std::fmt::Debug + Send + Sync {
    /// Produce an [`ExecutionPlan`] that streams the node's tiles over its single
    /// physical input, or `None` if this implementation cannot plan the node.
    fn plan_tile_explode(
        &self,
        node: &TileExplodePlanNode,
        physical_input: Arc<dyn ExecutionPlan>,
        session_state: &SessionState,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>>;
}

/// Physical planner hook for [`TileExplodePlanNode`].
///
/// Delegates to a list of [`TileExplodePhysicalPlanner`] factories, consulting
/// more recently added factories first (mirroring
/// [`crate::spatial_join_physical_planner::SpatialJoinExtensionPlanner`]).
#[derive(Clone, Debug, Default)]
pub struct TileExplodeExtensionPlanner {
    factories: Vec<Arc<dyn TileExplodePhysicalPlanner>>,
}

impl TileExplodeExtensionPlanner {
    /// Create a new planner with the given factories.
    pub fn new(factories: Vec<Arc<dyn TileExplodePhysicalPlanner>>) -> Self {
        Self { factories }
    }

    /// Append a tile-explode factory. Implementations are consulted in reverse
    /// registration order so a more recently added factory can override an
    /// earlier one.
    pub fn append_tile_explode_physical_planner(
        &mut self,
        factory: Arc<dyn TileExplodePhysicalPlanner>,
    ) {
        self.factories.push(factory);
    }
}

#[async_trait]
impl ExtensionPlanner for TileExplodeExtensionPlanner {
    async fn plan_extension(
        &self,
        _planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        _logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        session_state: &SessionState,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let Some(tile_node) = node.as_any().downcast_ref::<TileExplodePlanNode>() else {
            return Ok(None);
        };

        if physical_inputs.len() != 1 {
            return plan_err!("TileExplodePlanNode expects 1 input");
        }
        let physical_input = physical_inputs[0].clone();

        // Iterate in reverse to handle more recently added factories first.
        for factory in self.factories.iter().rev() {
            if let Some(exec) =
                factory.plan_tile_explode(tile_node, physical_input.clone(), session_state)?
            {
                return Ok(Some(exec));
            }
        }

        plan_err!("No tile-explode physical planner is registered to plan RS_TileExplode")
    }
}
