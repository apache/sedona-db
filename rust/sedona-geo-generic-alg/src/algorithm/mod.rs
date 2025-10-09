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
//! Collection of generic geometry algorithms ported from the `geo` crate.
//!
//! All submodules in this directory are ported (and contain copied code) from the
//! `geo` crate at commit `5d667f844716a3d0a17aa60bc0a58528cb5808c3`:
//! <https://github.com/georust/geo/tree/5d667f844716a3d0a17aa60bc0a58528cb5808c3/geo/src/algorithm>.
//! The original code is dual-licensed under Apache-2.0 or MIT; used here under Apache-2.0.
/// Kernels to compute various predicates
pub mod kernels;
pub use kernels::{Kernel, Orientation};

/// Calculate the area of the surface of a `Geometry`.
pub mod area;
pub use area::Area;

/// Calculate the bounding rectangle of a `Geometry`.
pub mod bounding_rect;
pub use bounding_rect::BoundingRect;

/// Calculate the centroid of a `Geometry`.
pub mod centroid;
pub use centroid::Centroid;

/// Determine whether a `Coord` lies inside, outside, or on the boundary of a geometry.
pub mod coordinate_position;
pub use coordinate_position::CoordinatePosition;

/// Dimensionality of a geometry and its boundary, based on OGC-SFA.
pub mod dimensions;
pub use dimensions::HasDimensions;

/// Calculate the length of a planar line between two `Geometries`.
pub mod euclidean_length;
#[allow(deprecated)]
pub use euclidean_length::EuclideanLength;

/// Determine whether `Geometry` `A` intersects `Geometry` `B`.
pub mod intersects;
pub use intersects::Intersects;

pub mod line_measures;
pub use line_measures::metric_spaces::Euclidean;

pub use line_measures::{Distance, DistanceExt, LengthMeasurableExt};

/// Apply a function to all `Coord`s of a `Geometry`.
pub mod map_coords;
pub use map_coords::{MapCoords, MapCoordsInPlace};
