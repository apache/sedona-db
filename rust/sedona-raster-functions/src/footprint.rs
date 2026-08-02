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
use sedona_geometry::interpolate::densify_segment;
use sedona_geometry::wkb_factory::write_wkb_polygon;
use sedona_raster::affine_transformation::to_world_coordinate;
use sedona_raster::traits::RasterRef;

/// Number of interior points interpolated along each footprint edge when a
/// footprint is densified before reprojection.
///
/// Reprojecting a footprint bends its straight edges into curves, so a footprint
/// described only by its four corners chords across each curve and under-/over-
/// covers the true extent. Densifying each edge with a handful of interior points
/// (~10 per edge is a common, cheap choice) and reprojecting every point makes
/// the reprojected footprint follow the curve. A footprint left in its own CRS
/// has exact straight edges and needs no densification.
pub const FOOTPRINT_POINTS_PER_EDGE: usize = 10;

/// The four corners of a raster's footprint in world coordinates.
///
/// Returned in ring order: upper-left `(0, 0)`, upper-right `(width, 0)`,
/// lower-right `(width, height)`, lower-left `(0, height)`.
pub fn raster_footprint_corners(raster: &dyn RasterRef) -> [(f64, f64); 4] {
    let width = raster.width().unwrap();
    let height = raster.height().unwrap();

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

/// Fill `ring` with a closed, densified footprint ring through four `corners`.
///
/// `corners` are in ring order (upper-left, upper-right, lower-right, lower-left,
/// as produced by [`raster_footprint_corners`]). Each edge contributes its start
/// corner followed by [`FOOTPRINT_POINTS_PER_EDGE`] interior points, and the ring
/// is closed back to the upper-left corner. `ring` is cleared first and is a
/// caller-managed scratch buffer so densifying many rasters allocates once.
///
/// Densify in the CRS where the footprint has exact straight edges (its own),
/// then reproject every point of the returned ring: the reprojected ring then
/// follows the curved image of each edge instead of chording across it.
pub fn densify_footprint_ring(corners: [(f64, f64); 4], ring: &mut Vec<(f64, f64)>) {
    ring.clear();
    let [ul, ur, lr, ll] = corners;
    for (start, end) in [(ul, ur), (ur, lr), (lr, ll), (ll, ul)] {
        ring.push(start);
        densify_segment(start, end, FOOTPRINT_POINTS_PER_EDGE, ring);
    }
    ring.push(ul);
}

/// Write WKB for the polygon through a pre-closed footprint `ring` (last point
/// equal to the first), as produced by [`densify_footprint_ring`].
///
/// Companion to [`write_footprint_wkb`] for the variable-length densified ring;
/// both emit a single-ring polygon and can build Binary arrays, as the arrow-rs
/// `BinaryBuilder` implements [`std::io::Write`].
pub fn write_footprint_ring_wkb(ring: &[(f64, f64)], out: &mut impl std::io::Write) -> Result<()> {
    write_wkb_polygon(out, ring.iter().copied())
        .map_err(|e| DataFusionError::External(e.into()))?;

    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;

    /// Index of each corner in a densified ring: the start of edge `k` sits after
    /// `k` corners and `k * FOOTPRINT_POINTS_PER_EDGE` interior points.
    fn corner_index(k: usize) -> usize {
        k * (FOOTPRINT_POINTS_PER_EDGE + 1)
    }

    #[test]
    fn densify_footprint_ring_layout_and_coords() {
        // Axis-aligned 10x10 square in ring order (UL, UR, LR, LL).
        let corners = [(0.0, 10.0), (10.0, 10.0), (10.0, 0.0), (0.0, 0.0)];
        let mut ring = Vec::new();
        densify_footprint_ring(corners, &mut ring);

        // 4 corners + 4 * FOOTPRINT_POINTS_PER_EDGE interior + 1 closing point.
        assert_eq!(ring.len(), 4 * (FOOTPRINT_POINTS_PER_EDGE + 1) + 1);

        // Corners land exactly on the interpolated ring, in order, and the ring
        // closes back to the upper-left corner.
        assert_eq!(ring[corner_index(0)], corners[0]);
        assert_eq!(ring[corner_index(1)], corners[1]);
        assert_eq!(ring[corner_index(2)], corners[2]);
        assert_eq!(ring[corner_index(3)], corners[3]);
        assert_eq!(*ring.last().unwrap(), corners[0]);

        // The first edge (UL -> UR) runs along y = 10 with x strictly increasing
        // through the interior points, spaced at t = i / (points_per_edge + 1).
        for i in 1..=FOOTPRINT_POINTS_PER_EDGE {
            let (x, y) = ring[corner_index(0) + i];
            let t = i as f64 / (FOOTPRINT_POINTS_PER_EDGE + 1) as f64;
            assert!((x - 10.0 * t).abs() < 1e-12, "x={x} t={t}");
            assert_eq!(y, 10.0);
        }
    }

    #[test]
    fn densify_footprint_ring_reuses_scratch() {
        let mut ring = vec![(42.0, 42.0)];
        let corners = [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)];
        densify_footprint_ring(corners, &mut ring);
        // Pre-existing scratch content is cleared, not appended to.
        assert_eq!(ring.len(), 4 * (FOOTPRINT_POINTS_PER_EDGE + 1) + 1);
        assert_eq!(ring[0], corners[0]);
    }
}
