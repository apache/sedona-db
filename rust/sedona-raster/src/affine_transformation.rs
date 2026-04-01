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

use crate::traits::RasterRef;
use arrow_schema::ArrowError;

/// Pre-computed affine transformation coefficients.
///
/// Constructing this struct pays the cost of reading the transform once.
/// Subsequent `transform` / `inv_transform` calls are pure arithmetic.
///
/// The 6-element GDAL GeoTransform convention is:
/// `[origin_x, scale_x, skew_x, origin_y, skew_y, scale_y]`
#[derive(Debug, Clone)]
pub struct AffineMatrix {
    pub offset_x: f64,
    pub offset_y: f64,
    pub scale_x: f64,
    pub scale_y: f64,
    pub skew_x: f64,
    pub skew_y: f64,
}

impl AffineMatrix {
    /// Build an `AffineMatrix` from a 6-element GDAL GeoTransform slice.
    ///
    /// Index mapping: `[0]=origin_x, [1]=scale_x, [2]=skew_x, [3]=origin_y, [4]=skew_y, [5]=scale_y`
    #[inline]
    pub fn from_transform(t: &[f64]) -> Self {
        Self {
            offset_x: t[0],
            scale_x: t[1],
            skew_x: t[2],
            offset_y: t[3],
            skew_y: t[4],
            scale_y: t[5],
        }
    }

    /// Forward affine transform: pixel (x, y) → world (wx, wy).
    ///
    /// Accepts `f64` coordinates so callers can pass fractional offsets
    /// (e.g. +0.5 for pixel centroids) without duplicating the math.
    #[inline]
    pub fn transform(&self, x: f64, y: f64) -> (f64, f64) {
        let wx = self.offset_x + x * self.scale_x + y * self.skew_x;
        let wy = self.offset_y + x * self.skew_y + y * self.scale_y;
        (wx, wy)
    }

    /// Inverse affine transform: world (wx, wy) → pixel (x, y).
    ///
    /// Returns an error if the determinant is zero (singular matrix).
    #[inline]
    pub fn inv_transform(&self, world_x: f64, world_y: f64) -> Result<(f64, f64), ArrowError> {
        let det = self.scale_x * self.scale_y - self.skew_x * self.skew_y;

        if det.abs() < f64::EPSILON {
            return Err(ArrowError::InvalidArgumentError(
                "Cannot compute coordinate: determinant is zero.".to_string(),
            ));
        }

        let inv_scale_x = self.scale_y / det;
        let inv_scale_y = self.scale_x / det;
        let inv_skew_x = -self.skew_x / det;
        let inv_skew_y = -self.skew_y / det;

        let dx = world_x - self.offset_x;
        let dy = world_y - self.offset_y;

        let rx = inv_scale_x * dx + inv_skew_x * dy;
        let ry = inv_skew_y * dx + inv_scale_y * dy;

        Ok((rx, ry))
    }

    /// Rotation angle (radians) implied by the affine coefficients.
    #[inline]
    pub fn rotation(&self) -> f64 {
        (-self.skew_x).atan2(self.scale_x)
    }
}

/// Computes the rotation angle (in radians) of the raster based on its geotransform.
#[inline]
pub fn rotation(raster: &dyn RasterRef) -> f64 {
    let t = raster.transform();
    (-t[2]).atan2(t[1]) // skew_x=t[2], scale_x=t[1]
}

/// Performs an affine transformation on the provided x and y coordinates based on the geotransform.
///
/// # Arguments
/// * `raster` - Reference to the raster containing transform
/// * `x` - X coordinate in pixel space (column)
/// * `y` - Y coordinate in pixel space (row)
#[inline]
pub fn to_world_coordinate(raster: &dyn RasterRef, x: i64, y: i64) -> (f64, f64) {
    AffineMatrix::from_transform(raster.transform()).transform(x as f64, y as f64)
}

/// Performs the inverse affine transformation to convert world coordinates back to raster pixel coordinates.
///
/// # Arguments
/// * `raster` - Reference to the raster containing transform
/// * `world_x` - X coordinate in world space
/// * `world_y` - Y coordinate in world space
#[inline]
pub fn to_raster_coordinate(
    raster: &dyn RasterRef,
    world_x: f64,
    world_y: f64,
) -> Result<(i64, i64), ArrowError> {
    let (rx, ry) =
        AffineMatrix::from_transform(raster.transform()).inv_transform(world_x, world_y)?;
    Ok((rx as i64, ry as i64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::FRAC_1_SQRT_2;
    use std::f64::consts::PI;

    /// Minimal RasterRef implementation for testing affine transforms.
    struct TestRaster {
        transform: [f64; 6],
    }

    impl TestRaster {
        fn new(
            origin_x: f64,
            origin_y: f64,
            scale_x: f64,
            scale_y: f64,
            skew_x: f64,
            skew_y: f64,
        ) -> Self {
            Self {
                transform: [origin_x, scale_x, skew_x, origin_y, skew_y, scale_y],
            }
        }
    }

    impl RasterRef for TestRaster {
        fn num_bands(&self) -> usize {
            0
        }
        fn band(&self, _index: usize) -> Option<&dyn crate::traits::BandRef> {
            None
        }
        fn band_name(&self, _index: usize) -> Option<&str> {
            None
        }
        fn crs(&self) -> Option<&str> {
            None
        }
        fn transform(&self) -> &[f64] {
            &self.transform
        }
        fn x_dim(&self) -> &str {
            "x"
        }
        fn y_dim(&self) -> &str {
            "y"
        }
    }

    #[test]
    fn test_rotation() {
        // 0 degree rotation
        let raster = TestRaster::new(0.0, 0.0, 1.0, -1.0, 0.0, 0.0);
        assert_eq!(rotation(&raster), 0.0);

        // pi/2
        let raster = TestRaster::new(0.0, 0.0, 0.0, 0.0, -1.0, 1.0);
        assert_relative_eq!(rotation(&raster), PI / 2.0, epsilon = 1e-6);

        // pi/4
        let raster = TestRaster::new(
            0.0,
            0.0,
            FRAC_1_SQRT_2,
            FRAC_1_SQRT_2,
            -FRAC_1_SQRT_2,
            FRAC_1_SQRT_2,
        );
        assert_relative_eq!(rotation(&raster), PI / 4.0, epsilon = 1e-6);

        // pi/3
        let raster = TestRaster::new(0.0, 0.0, 0.5, 0.5, -0.866025, 0.866025);
        assert_relative_eq!(rotation(&raster), PI / 3.0, epsilon = 1e-6);

        // pi
        let raster = TestRaster::new(0.0, 0.0, -1.0, -1.0, 0.0, 0.0);
        assert_relative_eq!(rotation(&raster), -PI, epsilon = 1e-6);
    }

    #[test]
    fn test_to_world_coordinate() {
        let raster = TestRaster::new(100.0, 200.0, 1.0, -2.0, 0.25, 0.5);

        assert_eq!(to_world_coordinate(&raster, 0, 0), (100.0, 200.0));
        assert_eq!(to_world_coordinate(&raster, 5, 10), (107.5, 182.5));
        assert_eq!(to_world_coordinate(&raster, 9, 19), (113.75, 166.5));
        assert_eq!(to_world_coordinate(&raster, 1, 0), (101.0, 200.5));
        assert_eq!(to_world_coordinate(&raster, 0, 1), (100.25, 198.0));
    }

    #[test]
    fn test_to_raster_coordinate() {
        let raster = TestRaster::new(100.0, 200.0, 1.0, -2.0, 0.25, 0.5);

        assert_eq!(to_raster_coordinate(&raster, 100.0, 200.0).unwrap(), (0, 0));
        assert_eq!(
            to_raster_coordinate(&raster, 107.5, 182.5).unwrap(),
            (5, 10)
        );
        assert_eq!(
            to_raster_coordinate(&raster, 113.75, 166.5).unwrap(),
            (9, 19)
        );
        assert_eq!(to_raster_coordinate(&raster, 101.0, 200.5).unwrap(), (1, 0));
        assert_eq!(to_raster_coordinate(&raster, 100.25, 198.0).unwrap(), (0, 1));

        // Zero determinant
        let bad_raster = TestRaster::new(100.0, 200.0, 1.0, 0.0, 0.0, 0.0);
        let result = to_raster_coordinate(&bad_raster, 100.0, 200.0);
        assert!(result.is_err());
        assert!(result
            .err()
            .unwrap()
            .to_string()
            .contains("determinant is zero."));
    }

    fn test_affine() -> AffineMatrix {
        AffineMatrix {
            offset_x: 100.0,
            offset_y: 200.0,
            scale_x: 1.0,
            scale_y: -2.0,
            skew_x: 0.25,
            skew_y: 0.5,
        }
    }

    #[test]
    fn test_affine_transform() {
        let a = test_affine();
        let (wx, wy) = a.transform(0.5, 0.5);
        assert_relative_eq!(wx, 100.625, epsilon = 1e-10);
        assert_relative_eq!(wy, 199.25, epsilon = 1e-10);
    }

    #[test]
    fn test_affine_round_trip() {
        let a = test_affine();
        let coords = [(0.0, 0.0), (5.0, 10.0), (9.0, 19.0), (0.5, 0.5)];
        for (x, y) in coords {
            let (wx, wy) = a.transform(x, y);
            let (rx, ry) = a.inv_transform(wx, wy).unwrap();
            assert_relative_eq!(rx, x, epsilon = 1e-10);
            assert_relative_eq!(ry, y, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_affine_inv_transform_singular() {
        let a = AffineMatrix {
            offset_x: 0.0,
            offset_y: 0.0,
            scale_x: 1.0,
            scale_y: 0.0,
            skew_x: 0.0,
            skew_y: 0.0,
        };
        let result = a.inv_transform(0.0, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_affine_rotation() {
        let a = AffineMatrix {
            offset_x: 0.0,
            offset_y: 0.0,
            scale_x: FRAC_1_SQRT_2,
            scale_y: FRAC_1_SQRT_2,
            skew_x: -FRAC_1_SQRT_2,
            skew_y: FRAC_1_SQRT_2,
        };
        assert_relative_eq!(a.rotation(), PI / 4.0, epsilon = 1e-6);
    }

    #[test]
    fn test_affine_from_transform() {
        let t = [100.0, 1.0, 0.25, 200.0, 0.5, -2.0];
        let a = AffineMatrix::from_transform(&t);
        assert_eq!(a.offset_x, 100.0);
        assert_eq!(a.scale_x, 1.0);
        assert_eq!(a.skew_x, 0.25);
        assert_eq!(a.offset_y, 200.0);
        assert_eq!(a.skew_y, 0.5);
        assert_eq!(a.scale_y, -2.0);
    }
}
