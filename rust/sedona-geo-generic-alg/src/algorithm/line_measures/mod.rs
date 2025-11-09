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
//! Generic line measurement algorithms (distance, length, perimeter, metric spaces)
//!
//! Ported (and contains copied code) from `geo::algorithm::line_measures` and related modules:
//! <https://github.com/georust/geo/tree/5d667f844716a3d0a17aa60bc0a58528cb5808c3/geo/src/algorithm/line_measures>.
//! Original code is dual-licensed under Apache-2.0 or MIT; used here under Apache-2.0.
mod distance;
pub use distance::{Distance, DistanceExt};

mod length;
pub use length::LengthMeasurableExt;

pub mod metric_spaces;
pub use metric_spaces::Euclidean;
