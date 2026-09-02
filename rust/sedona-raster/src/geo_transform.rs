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

//! Ported (and contains copied code) from georust/gdal:
//! <https://github.com/georust/gdal/blob/v0.19.0/src/geo_transform.rs>.
//! Original code is licensed under MIT.
//!
//! GeoTransform type and extension trait.
//!
//! The [`apply`](GeoTransformEx::apply) and [`invert`](GeoTransformEx::invert)
//! methods are pure-Rust reimplementations of GDAL's `GDALApplyGeoTransform`
//! and `GDALInvGeoTransform` (from `alg/gdaltransformer.cpp`). No FFI call or
//! thread-local state is needed.
//!
//! This is the single home for the six-coefficient affine geo-transform used
//! across the raster stack (GDAL, Zarr, and the `RS_` functions). Callers work
//! directly on the `[f64; 6]` a raster stores — [`RasterRef::transform`] returns
//! a `&[f64]`, so [`GeoTransformEx`] is implemented on `[f64]` and its named
//! accessors ([`scale_x`](GeoTransformEx::scale_x), ...) replace magic indices.
//!
//! [`RasterRef::transform`]: crate::traits::RasterRef::transform

use crate::error::RasterError;

/// An affine geo-transform: six coefficients mapping pixel/line to projection coordinates.
///
/// - `[0]`: x-coordinate of the upper-left corner of the upper-left pixel.
/// - `[1]`: W-E pixel resolution (pixel width).
/// - `[2]`: row rotation (typically zero).
/// - `[3]`: y-coordinate of the upper-left corner of the upper-left pixel.
/// - `[4]`: column rotation (typically zero).
/// - `[5]`: N-S pixel resolution (pixel height, negative for North-up).
pub type GeoTransform = [f64; 6];

/// Extension methods on a six-coefficient GDAL geo-transform.
///
/// Implemented on `[f64]` so it applies both to an owned [`GeoTransform`]
/// (`[f64; 6]`) and to the `&[f64]` returned by
/// [`RasterRef::transform`](crate::traits::RasterRef::transform). The accessors
/// panic on a slice shorter than six elements, matching the "rasters carry a
/// 6-element geo-transform" invariant.
pub trait GeoTransformEx {
    /// Apply the geo-transform to a pixel/line coordinate, returning (geo_x, geo_y).
    fn apply(&self, x: f64, y: f64) -> (f64, f64);

    /// Invert this geo-transform, returning the inverse coefficients for
    /// computing (geo_x, geo_y) -> (x, y) transformations.
    fn invert(&self) -> Result<GeoTransform, RasterError>;

    /// Rotation angle (radians) implied by the coefficients.
    ///
    /// Sedona Spark's `RS_Rotation` formula: `acos(scaleX / hypot(scaleX,
    /// skewY))`, negated when `skewY > 0` — the (clockwise-positive) angle of
    /// the grid's column-axis direction `(scaleX, skewY)`, which decomposes a
    /// rigid rotation correctly for any pixel shape. The previous
    /// `atan2(-skewX, scaleX)` mixed the row axis's skew into the answer and
    /// misreported rigid rotations of non-square pixels (and shear).
    fn rotation(&self) -> f64;

    /// x-coordinate of the upper-left corner of the upper-left pixel (`[0]`).
    fn origin_x(&self) -> f64;
    /// W-E pixel resolution / pixel width (`[1]`).
    fn scale_x(&self) -> f64;
    /// Row rotation term, typically zero (`[2]`).
    fn skew_x(&self) -> f64;
    /// y-coordinate of the upper-left corner of the upper-left pixel (`[3]`).
    fn origin_y(&self) -> f64;
    /// Column rotation term, typically zero (`[4]`).
    fn skew_y(&self) -> f64;
    /// N-S pixel resolution / pixel height, negative for North-up (`[5]`).
    fn scale_y(&self) -> f64;
}

impl GeoTransformEx for [f64] {
    /// Pure-Rust equivalent of GDAL's `GDALApplyGeoTransform`.
    #[inline]
    fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let geo_x = self[0] + x * self[1] + y * self[2];
        let geo_y = self[3] + x * self[4] + y * self[5];
        (geo_x, geo_y)
    }

    /// Pure-Rust equivalent of GDAL's `GDALInvGeoTransform`.
    fn invert(&self) -> Result<GeoTransform, RasterError> {
        let gt = self;

        // Fast path: no rotation/skew — avoid determinant and precision issues.
        if gt[2] == 0.0 && gt[4] == 0.0 && gt[1] != 0.0 && gt[5] != 0.0 {
            return Ok([
                -gt[0] / gt[1],
                1.0 / gt[1],
                0.0,
                -gt[3] / gt[5],
                0.0,
                1.0 / gt[5],
            ]);
        }

        // General case: 2x2 matrix inverse via adjugate / determinant.
        let det = gt[1] * gt[5] - gt[2] * gt[4];
        let magnitude = gt[1]
            .abs()
            .max(gt[2].abs())
            .max(gt[4].abs().max(gt[5].abs()));

        if det.abs() <= 1e-10 * magnitude * magnitude {
            return Err(RasterError::Invalid(
                "Geo transform is uninvertible".to_string(),
            ));
        }

        let inv_det = 1.0 / det;

        Ok([
            (gt[2] * gt[3] - gt[0] * gt[5]) * inv_det,
            gt[5] * inv_det,
            -gt[2] * inv_det,
            (-gt[1] * gt[3] + gt[0] * gt[4]) * inv_det,
            -gt[4] * inv_det,
            gt[1] * inv_det,
        ])
    }

    #[inline]
    fn rotation(&self) -> f64 {
        // acos rather than atan2 of the same ratio so an axis-aligned grid
        // reports +0.0 and a mirrored one +pi, matching Sedona Spark exactly.
        let theta = (self.scale_x() / self.scale_x().hypot(self.skew_y())).acos();
        if self.skew_y() > 0.0 {
            -theta
        } else {
            theta
        }
    }

    #[inline]
    fn origin_x(&self) -> f64 {
        self[0]
    }
    #[inline]
    fn scale_x(&self) -> f64 {
        self[1]
    }
    #[inline]
    fn skew_x(&self) -> f64 {
        self[2]
    }
    #[inline]
    fn origin_y(&self) -> f64 {
        self[3]
    }
    #[inline]
    fn skew_y(&self) -> f64 {
        self[4]
    }
    #[inline]
    fn scale_y(&self) -> f64 {
        self[5]
    }
}

/// Derive a north-up [`GeoTransform`] from a spatial bounding box and the grid's
/// spatial shape.
///
/// `bbox` is `[xmin, ymin, xmax, ymax]`; `height` and `width` are the number of
/// cells along the y and x axes. `registration` controls how the bbox maps to
/// the grid and defaults to `"pixel"` when `None`:
/// - `"pixel"` (cell-area): the bbox is the grid's outer edge, spanning all N
///   cells, so the top-left corner is the bbox edge. Matches
///   `rasterio.transform.from_bounds`.
/// - `"node"` (cell-center): the bbox endpoints are the centers of the border
///   cells, so N centers span N-1 intervals and the footprint extends half a
///   cell beyond the bbox.
///
/// The result is axis-aligned (zero skews) with a negative `scale_y` (rows
/// increase downward). Errors on a degenerate/inverted bbox, a zero dimension, a
/// node-registered grid smaller than 2 cells, or an unknown registration.
pub fn geotransform_from_bbox_and_spatial_shape(
    bbox: [f64; 4],
    height: u64,
    width: u64,
    registration: Option<&str>,
) -> Result<GeoTransform, RasterError> {
    let [xmin, ymin, xmax, ymax] = bbox;
    // Reject a degenerate or inverted bbox: a zero span gives a non-invertible
    // (zero-scale) transform and a negative span isn't north-up.
    if !(xmax > xmin && ymax > ymin) {
        return Err(RasterError::Invalid(format!(
            "bbox must have xmin < xmax and ymin < ymax; got [{xmin}, {ymin}, {xmax}, {ymax}]"
        )));
    }
    // A real raster axis is at least 1 cell.
    if height == 0 || width == 0 {
        return Err(RasterError::Invalid(
            "raster spatial dimensions must be non-zero to derive a transform from a bbox".into(),
        ));
    }
    let (height, width) = (height as f64, width as f64);

    // cell-area registration is the conventional default.
    let registration = registration.unwrap_or("pixel");
    // scale_y is negative throughout: rows increase downward.
    let (scale_x, scale_y, offset_x, offset_y) = match registration {
        // Pixel-registered: the bbox is the grid's outer edge, spanning all N
        // cells, so the top-left corner is the bbox edge.
        "pixel" => {
            let scale_x = (xmax - xmin) / width;
            let scale_y = (ymin - ymax) / height;
            (scale_x, scale_y, xmin, ymax)
        }
        // Node-registered: the bbox endpoints are the *centers* of the border
        // cells, so N centers span N-1 intervals and the footprint extends half
        // a cell beyond — the corner sits half a cell outside the bbox.
        "node" => {
            if width < 2.0 || height < 2.0 {
                return Err(RasterError::Invalid(
                    "node-registered grid must be at least 2 cells in each spatial dimension"
                        .into(),
                ));
            }
            let scale_x = (xmax - xmin) / (width - 1.0);
            let scale_y = (ymin - ymax) / (height - 1.0);
            (scale_x, scale_y, xmin - scale_x / 2.0, ymax - scale_y / 2.0)
        }
        other => {
            return Err(RasterError::Invalid(format!(
                "registration must be \"pixel\" or \"node\"; got {other:?}"
            )))
        }
    };

    Ok([offset_x, scale_x, 0.0, offset_y, 0.0, scale_y])
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_apply_no_rotation() {
        // Origin at (100, 200), 10m pixels, north-up
        let gt: GeoTransform = [100.0, 10.0, 0.0, 200.0, 0.0, -10.0];
        let (x, y) = gt.apply(5.0, 3.0);
        assert!((x - 150.0).abs() < 1e-12);
        assert!((y - 170.0).abs() < 1e-12);
    }

    #[test]
    fn test_apply_with_rotation() {
        let gt: GeoTransform = [100.0, 10.0, 2.0, 200.0, 3.0, -10.0];
        let (x, y) = gt.apply(5.0, 3.0);
        // 100 + 5*10 + 3*2 = 156
        assert!((x - 156.0).abs() < 1e-12);
        // 200 + 5*3 + 3*(-10) = 185
        assert!((y - 185.0).abs() < 1e-12);
    }

    #[test]
    fn test_apply_fractional_offset() {
        // Pixel-centroid (+0.5) sampling under skew: matches the closed form
        // 100 + 0.5*1 + 0.5*0.25 = 100.625; 200 + 0.5*0.5 + 0.5*(-2) = 199.25.
        let gt: GeoTransform = [100.0, 1.0, 0.25, 200.0, 0.5, -2.0];
        let (wx, wy) = gt.apply(0.5, 0.5);
        assert_relative_eq!(wx, 100.625, epsilon = 1e-10);
        assert_relative_eq!(wy, 199.25, epsilon = 1e-10);
    }

    #[test]
    fn test_invert_no_rotation() {
        let gt: GeoTransform = [100.0, 10.0, 0.0, 200.0, 0.0, -10.0];
        let inv = gt.invert().unwrap();
        // Round-trip: apply then apply inverse should recover pixel/line.
        let (geo_x, geo_y) = gt.apply(7.0, 4.0);
        let (px, ln) = inv.apply(geo_x, geo_y);
        assert!((px - 7.0).abs() < 1e-10);
        assert!((ln - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_invert_with_rotation() {
        let gt: GeoTransform = [100.0, 10.0, 2.0, 200.0, 3.0, -10.0];
        let inv = gt.invert().unwrap();
        let (geo_x, geo_y) = gt.apply(7.0, 4.0);
        let (px, ln) = inv.apply(geo_x, geo_y);
        assert!((px - 7.0).abs() < 1e-10);
        assert!((ln - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_invert_round_trip_skewed() {
        // A skewed transform round-trips at several pixel offsets, including a
        // fractional one.
        let gt: GeoTransform = [100.0, 1.0, 0.25, 200.0, 0.5, -2.0];
        let inv = gt.invert().unwrap();
        for (x, y) in [(0.0, 0.0), (5.0, 10.0), (9.0, 19.0), (0.5, 0.5)] {
            let (wx, wy) = gt.apply(x, y);
            let (rx, ry) = inv.apply(wx, wy);
            assert_relative_eq!(rx, x, epsilon = 1e-10);
            assert_relative_eq!(ry, y, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_invert_singular() {
        // Determinant is zero: both rows are proportional.
        let gt: GeoTransform = [0.0, 1.0, 2.0, 0.0, 2.0, 4.0];
        assert!(gt.invert().is_err());
    }

    #[test]
    fn test_accessors() {
        // Named accessors read the GDAL-order coefficients.
        let gt: GeoTransform = [100.0, 1.0, 0.25, 200.0, 0.5, -2.0];
        assert_eq!(gt.origin_x(), 100.0);
        assert_eq!(gt.scale_x(), 1.0);
        assert_eq!(gt.skew_x(), 0.25);
        assert_eq!(gt.origin_y(), 200.0);
        assert_eq!(gt.skew_y(), 0.5);
        assert_eq!(gt.scale_y(), -2.0);
    }

    #[test]
    fn test_rotation() {
        // Axis-aligned -> exactly +0.0 (acos yields the positive zero Sedona
        // Spark reports; the previous atan2 spelling returned -0.0).
        let gt: GeoTransform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        assert_eq!(gt.rotation(), 0.0);
        assert!(gt.rotation().is_sign_positive());

        // A rigid rotation by theta of a north-up grid with pixel (px, py):
        // (scale_x, skew_x, skew_y, scale_y) = (px*cos, py*sin, px*sin,
        // -py*cos); reads back as -theta, for any pixel shape.
        let (sin, cos) = (PI / 6.0).sin_cos();
        let gt: GeoTransform = [0.0, cos, sin, 0.0, sin, -cos];
        assert_relative_eq!(gt.rotation(), -PI / 6.0, epsilon = 1e-12);
        let gt: GeoTransform = [0.0, cos, -sin, 0.0, -sin, -cos];
        assert_relative_eq!(gt.rotation(), PI / 6.0, epsilon = 1e-12);
        let gt: GeoTransform = [0.0, 2.0 * cos, 3.0 * sin, 0.0, 2.0 * sin, -3.0 * cos];
        assert_relative_eq!(gt.rotation(), -PI / 6.0, epsilon = 1e-12);

        // Under shear the angle follows the column-axis direction
        // (scale_x, skew_y) — Sedona Spark's convention.
        let gt: GeoTransform = [0.0, 0.2, 0.06, 0.0, 0.08, -0.3];
        assert_relative_eq!(
            gt.rotation(),
            -(0.2f64 / 0.2f64.hypot(0.08)).acos(),
            epsilon = 1e-15
        );

        // A mirrored grid (scale_x < 0) reports +pi.
        let gt: GeoTransform = [0.0, -1.0, 0.0, 0.0, 0.0, -1.0];
        assert_relative_eq!(gt.rotation(), PI, epsilon = 1e-12);
    }

    #[test]
    fn bbox_pixel_registration_derives_transform() {
        // A bbox over a 1000x1000 grid with "pixel" registration. Height/width
        // are the array's own dims.
        let gt = geotransform_from_bbox_and_spatial_shape(
            [600000.0, 5690000.0, 610000.0, 5700000.0],
            1000,
            1000,
            Some("pixel"),
        )
        .unwrap();
        assert_eq!(gt, [600000.0, 10.0, 0.0, 5700000.0, 0.0, -10.0]);
    }

    #[test]
    fn bbox_pixel_registration_matches_rasterio_from_bounds() {
        // Cross-check against rasterio: for the same bounds and raster size,
        // `rasterio.transform.from_bounds(west, south, east, north, width, height)`
        // returns Affine(a, b, c, d, e, f) with a = (east-west)/width = 0.25,
        // e = (south-north)/height = -0.5, c = west = -100, f = north = 40, and
        // b = d = 0. In GDAL order that is [c, a, b, f, d, e].
        let gt =
            geotransform_from_bbox_and_spatial_shape([-100.0, -10.0, 0.0, 40.0], 100, 400, None)
                .unwrap();
        assert_eq!(gt, [-100.0, 0.25, 0.0, 40.0, 0.0, -0.5]);
    }

    #[test]
    fn bbox_registration_defaults_to_pixel() {
        // `None` registration behaves as cell-area ("pixel").
        let gt = geotransform_from_bbox_and_spatial_shape(
            [600000.0, 5690000.0, 610000.0, 5700000.0],
            1000,
            1000,
            None,
        )
        .unwrap();
        assert_eq!(gt, [600000.0, 10.0, 0.0, 5700000.0, 0.0, -10.0]);
    }

    #[test]
    fn bbox_node_registration_uses_n_minus_1_intervals() {
        // "node": the bbox endpoints are cell centers, so 11 centers span 10
        // intervals -> scale = 100 / 10 = 10, and the corner sits half a cell
        // (5) outside the bbox: origin (-5, 105).
        let gt = geotransform_from_bbox_and_spatial_shape(
            [0.0, 0.0, 100.0, 100.0],
            11,
            11,
            Some("node"),
        )
        .unwrap();
        assert_eq!(gt, [-5.0, 10.0, 0.0, 105.0, 0.0, -10.0]);
    }

    #[test]
    fn bbox_node_registration_requires_at_least_two_cells() {
        // A single-cell node grid can't define a spacing from its bbox alone.
        let err =
            geotransform_from_bbox_and_spatial_shape([0.0, 0.0, 100.0, 100.0], 1, 1, Some("node"))
                .unwrap_err()
                .to_string();
        assert!(err.contains("at least 2"), "{err}");
    }

    #[test]
    fn bbox_unknown_registration_errors() {
        let err = geotransform_from_bbox_and_spatial_shape(
            [0.0, 0.0, 100.0, 100.0],
            10,
            10,
            Some("corner"),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("pixel") && err.contains("node"), "{err}");
    }

    #[test]
    fn degenerate_or_inverted_bbox_errors() {
        // Zero x-span (non-invertible) and inverted y (not north-up) are both
        // rejected.
        for bbox in [
            [10.0, 0.0, 10.0, 5.0], // xmin == xmax
            [0.0, 5.0, 5.0, 0.0],   // ymax < ymin
        ] {
            let err = geotransform_from_bbox_and_spatial_shape(bbox, 10, 10, None)
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("xmin < xmax") || err.contains("ymin < ymax"),
                "{err}"
            );
        }
    }

    #[test]
    fn zero_dimension_bbox_errors() {
        let err = geotransform_from_bbox_and_spatial_shape([0.0, 0.0, 10.0, 10.0], 0, 10, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("non-zero"), "{err}");
    }
}
