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

//! Raster footprint helpers shared by the raster spatial-predicate kernels and
//! the optimized raster spatial join.
//!
//! A raster's footprint is the convex hull of its four corners in world
//! coordinates. Because the affine geotransform may include skew/rotation, each
//! corner is computed individually rather than assumed axis-aligned.

use datafusion_common::{DataFusionError, Result};
use sedona_geometry::wkb_factory::write_wkb_polygon;
use sedona_raster::affine_transformation::to_world_coordinate;
use sedona_raster::traits::RasterRef;

/// The four corners of a raster's footprint in world coordinates.
///
/// Returned in ring order: upper-left `(0, 0)`, upper-right `(width, 0)`,
/// lower-right `(width, height)`, lower-left `(0, height)`.
pub fn raster_footprint_corners(raster: &dyn RasterRef) -> [(f64, f64); 4] {
    let width = raster.metadata().width();
    let height = raster.metadata().height();

    [
        to_world_coordinate(raster, 0, 0),
        to_world_coordinate(raster, width, 0),
        to_world_coordinate(raster, width, height),
        to_world_coordinate(raster, 0, height),
    ]
}

/// Write WKB for the convex-hull polygon through four footprint `corners`.
///
/// `corners` are in ring order (upper-left, upper-right, lower-right,
/// lower-left, as produced by [`raster_footprint_corners`]); the ring is closed
/// back to the first corner. Shared by the native footprint (corners in the
/// raster's own CRS) and the reprojected footprint (corners transformed into
/// another CRS), so both paths emit byte-identical polygon WKB. This can be used
/// to build Binary arrays, as the arrow-rs `BinaryBuilder` implements
/// [`std::io::Write`].
pub fn write_footprint_wkb(corners: [(f64, f64); 4], out: &mut impl std::io::Write) -> Result<()> {
    let [ul, ur, lr, ll] = corners;

    write_wkb_polygon(out, [ul, ur, lr, ll, ul].into_iter())
        .map_err(|e| DataFusionError::External(e.into()))?;

    Ok(())
}

/// Write WKB for the convex-hull polygon of the raster footprint.
///
/// The ring is the four [`raster_footprint_corners`] closed back to the
/// upper-left corner.
pub fn write_convexhull_wkb(raster: &dyn RasterRef, out: &mut impl std::io::Write) -> Result<()> {
    write_footprint_wkb(raster_footprint_corners(raster), out)
}
