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

/// Convert the type of a geometry’s coordinate value.
pub mod convert;
pub use convert::{Convert, TryConvert};

/// Convert coordinate angle units between radians and degrees.
pub mod convert_angle_unit;
pub use convert_angle_unit::{ToDegrees, ToRadians};

/// Determine whether a `Coord` lies inside, outside, or on the boundary of a geometry.
pub mod coordinate_position;
pub use coordinate_position::CoordinatePosition;

/// Iterate over geometry coordinates.
pub mod coords_iter;
pub use coords_iter::CoordsIter;
pub use coords_iter::CoordsSeqIter;

/// Dimensionality of a geometry and its boundary, based on OGC-SFA.
pub mod dimensions;
pub use dimensions::HasDimensions;

/// Calculate the length of a planar line between two `Geometries`.
pub mod euclidean_length;
#[allow(deprecated)]
pub use euclidean_length::EuclideanLength;

/// Calculate the extreme coordinates and indices of a geometry.
pub mod extremes;
pub use extremes::Extremes;

/// Determine whether `Geometry` `A` intersects `Geometry` `B`.
pub mod intersects;
pub use intersects::Intersects;

pub mod line_measures;
pub use line_measures::metric_spaces::Euclidean;

pub use line_measures::{Distance, DistanceExt, LengthMeasurableExt};

/// Apply a function to all `Coord`s of a `Geometry`.
pub mod map_coords;
pub use map_coords::{MapCoords, MapCoordsInPlace};

/// Rotate a `Geometry` by an angle given in degrees.
pub mod rotate;
pub use rotate::{Rotate, RotateMut};

/// Scale a `Geometry` up or down by a factor
pub mod scale;
pub use scale::{Scale, ScaleMut};

/// Skew a `Geometry` by shearing it at angles along the x and y dimensions
pub mod skew;
pub use skew::{Skew, SkewMut};

/// Composable affine operations such as rotate, scale, skew, and translate
pub mod affine_ops;
pub use affine_ops::{AffineOps, AffineOpsMut, AffineTransform};

/// Simplify `Geometries` using the Ramer-Douglas-Peucker algorithm.
pub mod simplify;
pub use simplify::{Simplify, SimplifyIdx};

/// Translate a `Geometry` along the given offsets.
pub mod translate;
pub use translate::{Translate, TranslateMut};
