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

use crate::traits::{MetadataRef, RasterRef};
use arrow_schema::ArrowError;

/// Pre-computed affine transformation coefficients extracted from raster metadata.
///
/// Constructing this struct pays the cost of reading metadata once (which may involve
/// vtable dispatch for Arrow-backed rasters). Subsequent `transform` / `inv_transform`
/// calls are pure arithmetic with no virtual calls.
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
    /// Build an `AffineMatrix` from any `MetadataRef` implementer.
    #[inline]
    pub fn from_metadata(m: &dyn MetadataRef) -> Self {
        Self {
            offset_x: m.upper_left_x(),
            offset_y: m.upper_left_y(),
            scale_x: m.scale_x(),
            scale_y: m.scale_y(),
            skew_x: m.skew_x(),
            skew_y: m.skew_y(),
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

/// Computes the rotation angle (in radians) of the raster based on its geotransform metadata.
#[inline]
pub fn rotation(raster: &dyn RasterRef) -> f64 {
    let metadata = raster.metadata();
    (-metadata.skew_x()).atan2(metadata.scale_x())
}

/// Performs an affine transformation on the provided x and y coordinates based on the geotransform
/// data in the raster.
///
/// # Arguments
/// * `raster` - Reference to the raster containing metadata
/// * `x` - X coordinate in pixel space (column)
/// * `y` - Y coordinate in pixel space (row)
#[inline]
pub fn to_world_coordinate(raster: &dyn RasterRef, x: i64, y: i64) -> (f64, f64) {
    AffineMatrix::from_metadata(raster.metadata()).transform(x as f64, y as f64)
}

/// Performs the inverse affine transformation to convert world coordinates back to raster pixel coordinates.
///
/// # Arguments
/// * `raster` - Reference to the raster containing metadata
/// * `world_x` - X coordinate in world space
/// * `world_y` - Y coordinate in world space
#[inline]
pub fn to_raster_coordinate(
    raster: &dyn RasterRef,
    world_x: f64,
    world_y: f64,
) -> Result<(i64, i64), ArrowError> {
    let (rx, ry) =
        AffineMatrix::from_metadata(raster.metadata()).inv_transform(world_x, world_y)?;
    Ok((rx as i64, ry as i64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{MetadataRef, RasterMetadata};
    use approx::assert_relative_eq;
    use std::f64::consts::FRAC_1_SQRT_2;
    use std::f64::consts::PI;

    struct TestRaster {
        metadata: RasterMetadata,
    }

    impl RasterRef for TestRaster {
        fn metadata(&self) -> &dyn MetadataRef {
            &self.metadata
        }
        fn crs(&self) -> Option<&str> {
            None
        }
        fn bands(&self) -> &dyn crate::traits::BandsRef {
            unimplemented!()
        }
    }

    #[test]
    fn test_rotation() {
        // 0 degree rotation -> gt[1.0, 0.0, 0.0, -1.0]
        let raster = rotation_raster(1.0, -1.0, 0.0, 0.0);
        let rot = rotation(&raster);
        assert_eq!(rot, 0.0);

        // pi/2 -> gt[0.0, -1.0, 1.0, 0.0]
        let raster = rotation_raster(0.0, 0.0, -1.0, 1.0);
        let rot = rotation(&raster);
        assert_relative_eq!(rot, PI / 2.0, epsilon = 1e-6); // 90 degrees in radians

        // pi/4 -> gt[0.70710678, -0.70710678, 0.70710678, 0.70710678]
        let raster = rotation_raster(FRAC_1_SQRT_2, FRAC_1_SQRT_2, -FRAC_1_SQRT_2, FRAC_1_SQRT_2);
        let rot = rotation(&raster);
        assert_relative_eq!(rot, PI / 4.0, epsilon = 1e-6); // 45 degrees in radians

        // pi/3 -> gt[0.5, -0.866025, 0.866025, 0.5]
        let raster = rotation_raster(0.5, 0.5, -0.866025, 0.866025);
        let rot = rotation(&raster);
        assert_relative_eq!(rot, PI / 3.0, epsilon = 1e-6); // 60 degrees in radians

        // pi -> gt[-1.0, 0.0, 0.0, -1.0]
        let raster = rotation_raster(-1.0, -1.0, 0.0, 0.0);
        let rot = rotation(&raster);
        assert_relative_eq!(rot, -PI, epsilon = 1e-6); // 180 degrees in radians
    }

    #[test]
    fn test_to_world_coordinate() {
        // Test case with rotation/skew
        let raster = TestRaster {
            metadata: RasterMetadata {
                width: 10,
                height: 20,
                upperleft_x: 100.0,
                upperleft_y: 200.0,
                scale_x: 1.0,
                scale_y: -2.0,
                skew_x: 0.25,
                skew_y: 0.5,
            },
        };

        let (wx, wy) = to_world_coordinate(&raster, 0, 0);
        assert_eq!((wx, wy), (100.0, 200.0));

        let (wx, wy) = to_world_coordinate(&raster, 5, 10);
        assert_eq!((wx, wy), (107.5, 182.5));

        let (wx, wy) = to_world_coordinate(&raster, 9, 19);
        assert_eq!((wx, wy), (113.75, 166.5));

        let (wx, wy) = to_world_coordinate(&raster, 1, 0);
        assert_eq!((wx, wy), (101.0, 200.5));

        let (wx, wy) = to_world_coordinate(&raster, 0, 1);
        assert_eq!((wx, wy), (100.25, 198.0));
    }

    #[test]
    fn test_to_raster_coordinate() {
        // Test case with rotation/skew
        let raster = TestRaster {
            metadata: RasterMetadata {
                width: 10,
                height: 20,
                upperleft_x: 100.0,
                upperleft_y: 200.0,
                scale_x: 1.0,
                scale_y: -2.0,
                skew_x: 0.25,
                skew_y: 0.5,
            },
        };

        // Reverse of the to_world_coordinate tests
        let (wx, wy) = to_raster_coordinate(&raster, 100.0, 200.0).unwrap();
        assert_eq!((wx, wy), (0, 0));

        let (wx, wy) = to_raster_coordinate(&raster, 107.5, 182.5).unwrap();
        assert_eq!((wx, wy), (5, 10));

        let (wx, wy) = to_raster_coordinate(&raster, 113.75, 166.5).unwrap();
        assert_eq!((wx, wy), (9, 19));

        let (wx, wy) = to_raster_coordinate(&raster, 101.0, 200.5).unwrap();
        assert_eq!((wx, wy), (1, 0));

        let (wx, wy) = to_raster_coordinate(&raster, 100.25, 198.0).unwrap();
        assert_eq!((wx, wy), (0, 1));

        // Check error handling for zero determinant
        let bad_raster = TestRaster {
            metadata: RasterMetadata {
                width: 10,
                height: 20,
                upperleft_x: 100.0,
                upperleft_y: 200.0,
                scale_x: 1.0,
                scale_y: 0.0,
                skew_x: 0.0,
                skew_y: 0.0,
            },
        };
        let result = to_raster_coordinate(&bad_raster, 100.0, 200.0);
        assert!(result.is_err());
        assert!(result
            .err()
            .unwrap()
            .to_string()
            .contains("determinant is zero."));
    }

    fn rotation_raster(scale_x: f64, scale_y: f64, skew_x: f64, skew_y: f64) -> TestRaster {
        TestRaster {
            metadata: RasterMetadata {
                width: 10,
                height: 20,
                upperleft_x: 0.0,
                upperleft_y: 0.0,
                scale_x,
                scale_y,
                skew_x,
                skew_y,
            },
        }
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
        assert!(result
            .err()
            .unwrap()
            .to_string()
            .contains("determinant is zero."));
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
    fn test_affine_from_metadata() {
        let m = RasterMetadata {
            width: 10,
            height: 20,
            upperleft_x: 100.0,
            upperleft_y: 200.0,
            scale_x: 1.0,
            scale_y: -2.0,
            skew_x: 0.25,
            skew_y: 0.5,
        };
        let a = AffineMatrix::from_metadata(&m);
        assert_eq!(a.offset_x, 100.0);
        assert_eq!(a.offset_y, 200.0);
        assert_eq!(a.scale_x, 1.0);
        assert_eq!(a.scale_y, -2.0);
        assert_eq!(a.skew_x, 0.25);
        assert_eq!(a.skew_y, 0.5);
    }
}
