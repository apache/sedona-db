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

mod build_index;
mod evaluated_batch;
pub mod exec;
mod index;
pub mod operand_evaluator;
pub mod optimizer;
pub mod refine;
pub mod spatial_predicate;
mod stream;
pub mod utils;

pub use exec::SpatialJoinExec;
pub use optimizer::register_spatial_join_optimizer;

// Re-export option types from sedona-common for convenience
pub use sedona_common::option::*;
