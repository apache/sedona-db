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

/// Performs an affine transformation on the provided x and y coordinates based on the geotransform
/// data in the raster.
///
/// # Arguments
/// * `raster` - Reference to the raster containing metadata
/// * `x` - X coordinate in pixel space (column)
/// * `y` - Y coordinate in pixel space (row)
#[inline]
pub fn to_world_coordinate(raster: &dyn RasterRef, x: i64, y: i64) -> (f64, f64) {
    let metadata = raster.metadata();
    let x_f64 = x as f64;
    let y_f64 = y as f64;

    let world_x = metadata.upper_left_x() + x_f64 * metadata.scale_x() + y_f64 * metadata.skew_x();
    let world_y = metadata.upper_left_y() + x_f64 * metadata.skew_y() + y_f64 * metadata.scale_y();

    (world_x, world_y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{MetadataRef, RasterMetadata};

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
    fn test_to_world_coordinate_basic() {
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
}
