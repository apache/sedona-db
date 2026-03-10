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

use crate::errors;
use crate::errors::GdalError;

/// An affine geo-transform: six coefficients mapping pixel/line to projection coordinates.
///
/// - `[0]`: x-coordinate of the upper-left corner of the upper-left pixel.
/// - `[1]`: W-E pixel resolution (pixel width).
/// - `[2]`: row rotation (typically zero).
/// - `[3]`: y-coordinate of the upper-left corner of the upper-left pixel.
/// - `[4]`: column rotation (typically zero).
/// - `[5]`: N-S pixel resolution (pixel height, negative for North-up).
pub type GeoTransform = [f64; 6];

/// Extension methods on [`GeoTransform`].
pub trait GeoTransformEx {
    /// Apply the geo-transform to a pixel/line coordinate, returning (geo_x, geo_y).
    fn apply(&self, x: f64, y: f64) -> (f64, f64);

    /// Invert this geo-transform, returning the inverse coefficients for
    /// computing (geo_x, geo_y) -> (x, y) transformations.
    fn invert(&self) -> errors::Result<GeoTransform>;
}

impl GeoTransformEx for GeoTransform {
    /// Pure-Rust equivalent of GDAL's `GDALApplyGeoTransform`.
    fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let geo_x = self[0] + x * self[1] + y * self[2];
        let geo_y = self[3] + x * self[4] + y * self[5];
        (geo_x, geo_y)
    }

    /// Pure-Rust equivalent of GDAL's `GDALInvGeoTransform`.
    fn invert(&self) -> errors::Result<GeoTransform> {
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
            return Err(GdalError::BadArgument(
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_invert_singular() {
        // Determinant is zero: both rows are proportional.
        let gt: GeoTransform = [0.0, 1.0, 2.0, 0.0, 2.0, 4.0];
        assert!(gt.invert().is_err());
    }
}
