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

//! DataFusion planner integration for Sedona spatial joins.
//!
//! This module wires Sedona's logical optimizer rules and physical planning extensions that
//! can produce `SpatialJoinExec`.

use std::sync::Arc;

use datafusion::execution::SessionStateBuilder;
use datafusion::physical_planner::ExtensionPlanner;
use datafusion_common::Result;

mod physical_planner;

/// Register Sedona spatial join planning hooks.
///
/// Registers logical optimizer rules and returns an extension planner that can
/// plan `SpatialJoinExec`. The caller is responsible for installing the returned
/// extension planner into a [`QueryPlanner`].
pub fn register_planner(
    state_builder: SessionStateBuilder,
) -> Result<(SessionStateBuilder, Arc<dyn ExtensionPlanner + Send + Sync>)> {
    // Enable the logical rewrite that turns Filter(CrossJoin) into Join(filter=...)
    let state_builder = sedona_spatial_join_common::optimizer::register_spatial_join_logical_optimizer(state_builder)?;

    // Return the extension planner for SpatialJoinExec
    Ok((state_builder, physical_planner::spatial_join_extension_planner()))
}
