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

use datafusion::execution::SessionStateBuilder;
use datafusion_common::Result;

mod logical_plan_node;
mod optimizer;
mod physical_planner;
mod spatial_expr_utils;

/// Register Sedona spatial join planning hooks.
///
/// Enables logical rewrites (to surface join filters) and a query planner extension that can
/// plan `SpatialJoinExec`. This is the primary entry point to leveraging the spatial join
/// implementation provided by this crate and ensures joins created by SQL or using
/// a DataFrame API that meet certain conditions (e.g. contain a spatial predicate as
/// a join condition) are executed using the `SpatialJoinExec`.
pub fn register_planner(state_builder: SessionStateBuilder) -> Result<SessionStateBuilder> {
    // Enable the logical rewrite that turns Filter(CrossJoin) into Join(filter=...)
    let state_builder = optimizer::register_spatial_join_logical_optimizer(state_builder)?;

    // Enable planning SpatialJoinExec via an extension node during logical->physical planning.
    Ok(physical_planner::register_spatial_join_planner(
        state_builder,
    ))
}
