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

//! Raster ↔ world coordinate helpers built on [`GeoTransformEx`].
//!
//! These convenience wrappers read a raster's [`GeoTransform`] once (via
//! [`RasterRef::transform`]) and apply/invert it. The affine math itself lives
//! in [`crate::geo_transform`]; there is exactly one `apply`/`invert`
//! implementation.
//!
//! [`GeoTransform`]: crate::geo_transform::GeoTransform

use crate::geo_transform::GeoTransformEx;
use crate::traits::RasterRef;
use arrow_schema::ArrowError;

/// Computes the rotation angle (in radians) of the raster based on its geotransform metadata.
#[inline]
pub fn rotation(raster: &dyn RasterRef) -> f64 {
    raster.transform().rotation()
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
    raster.transform().apply(x as f64, y as f64)
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
    let inverse = raster.transform().invert().map_err(|_| {
        ArrowError::InvalidArgumentError(
            "Cannot compute coordinate: determinant is zero.".to_string(),
        )
    })?;
    let (rx, ry) = inverse.apply(world_x, world_y);
    Ok((rx as i64, ry as i64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{BandRef, Bands};
    use approx::assert_relative_eq;
    use std::f64::consts::FRAC_1_SQRT_2;
    use std::f64::consts::PI;

    /// Minimal `RasterRef` stub carrying only a geotransform and a spatial
    /// shape — enough for the affine helpers, which read `transform()` and
    /// (via `width()`/`height()`) `spatial_shape()`.
    struct TestRaster {
        transform: [f64; 6],
        spatial_shape: [i64; 2],
    }

    impl RasterRef for TestRaster {
        fn num_bands(&self) -> usize {
            0
        }
        fn bands(&self) -> Bands<'_> {
            Bands::new(self)
        }
        fn band(&self, index: usize) -> Result<Box<dyn BandRef + '_>, ArrowError> {
            Err(ArrowError::InvalidArgumentError(format!(
                "Band index {index} is out of range: this raster has 0 bands"
            )))
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
        fn spatial_dims(&self) -> Vec<&str> {
            vec!["x", "y"]
        }
        fn spatial_shape(&self) -> &[i64] {
            &self.spatial_shape
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
            transform: [100.0, 1.0, 0.25, 200.0, 0.5, -2.0],
            spatial_shape: [10, 20],
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
            transform: [100.0, 1.0, 0.25, 200.0, 0.5, -2.0],
            spatial_shape: [10, 20],
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
            transform: [100.0, 1.0, 0.0, 200.0, 0.0, 0.0],
            spatial_shape: [10, 20],
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
            transform: [0.0, scale_x, skew_x, 0.0, skew_y, scale_y],
            spatial_shape: [10, 20],
        }
    }
}
