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

pub mod evaluated_batch;
pub mod exec;
pub mod index;
pub mod join_provider;
pub mod operand_evaluator;
pub mod partitioning;
pub mod physical_planner;
mod prepare;
mod probe;
pub mod refine;
pub use sedona_query_planner::spatial_predicate;
mod stream;
pub mod utils;

pub use exec::SpatialJoinExec;

// Re-export function for register the spatial join planner
pub use physical_planner::DefaultSpatialJoinPhysicalPlanner;

// Re-export ProbeShuffleExec so that integration tests (and other crates) can verify
// its presence in optimized physical plans.
pub use sedona_query_planner::probe_shuffle_exec::ProbeShuffleExec;

// Re-export types needed for external usage (e.g., in Comet)
pub use spatial_predicate::SpatialPredicate;

// Re-export refiner trait and types for extension crates
pub use index::IndexQueryResult;
pub use refine::IndexQueryResultRefiner;

// Re-export option types from sedona-common for convenience
pub use sedona_common::option::*;
