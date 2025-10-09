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
//! Generic Distance traits (line measures)
//!
//! Ported (and contains copied code) from `geo::algorithm::line_measures::distance`:
//! <https://github.com/georust/geo/blob/5d667f844716a3d0a17aa60bc0a58528cb5808c3/geo/src/algorithm/line_measures/distance.rs>.
//! Original code is dual-licensed under Apache-2.0 or MIT; used here under Apache-2.0.
/// Calculate the minimum distance between two geometries.
pub trait Distance<F, Origin, Destination> {
    /// Note that not all implementations support all geometry combinations, but at least `Point` to `Point`
    /// is supported.
    /// See [specific implementations](#implementers) for details.
    ///
    /// # Units
    ///
    /// - `origin`, `destination`: geometry where the units of x/y depend on the trait implementation.
    /// - returns: depends on the trait implementation.
    fn distance(&self, origin: Origin, destination: Destination) -> F;
}

// Re-export the DistanceExt trait from the refactored euclidean metric space
pub use super::metric_spaces::euclidean::DistanceExt;
