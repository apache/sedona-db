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

use std::fmt;

use crate::affine_transformation::to_world_coordinate;
use crate::traits::RasterRef;
use sedona_schema::raster::StorageType;

/// Wrapper for formatting a raster reference as a human-readable string.
///
/// # Format
///
/// Non-skewed rasters:
/// ```text
/// [WxH/nbands] @ [xmin ymin xmax ymax] / CRS
/// ```
///
/// Skewed rasters (includes skew parameters):
/// ```text
/// [WxH/nbands] @ [xmin ymin xmax ymax] skew=(skew_x, skew_y) / CRS
/// ```
///
/// With outdb bands:
/// ```text
/// [WxH/nbands] @ [xmin ymin xmax ymax] / CRS <outdb>
/// ```
///
/// Without CRS:
/// ```text
/// [WxH/nbands] @ [xmin ymin xmax ymax]
/// ```
///
/// # Examples
///
/// ```text
/// [64x32/3] @ [43.08 79.07 171.08 143.07] / OGC:CRS84
/// [3x4/1] @ [3 2.4 3.84 4.24] skew=(0.06, 0.08) / EPSG:2193
/// [10x10/1] @ [0 0 10 10] / OGC:CRS84 <outdb>
/// ```
pub struct RasterDisplay<'a>(pub &'a dyn RasterRef);

impl fmt::Display for RasterDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let raster = self.0;
        let metadata = raster.metadata();
        let bands = raster.bands();

        let width = metadata.width();
        let height = metadata.height();
        let nbands = bands.len();

        // Compute axis-aligned bounding box from 4 corners in world coordinates.
        // This handles both skewed and non-skewed rasters correctly.
        let w = width as i64;
        let h = height as i64;
        let (ulx, uly) = to_world_coordinate(raster, 0, 0);
        let (urx, ury) = to_world_coordinate(raster, w, 0);
        let (lrx, lry) = to_world_coordinate(raster, w, h);
        let (llx, lly) = to_world_coordinate(raster, 0, h);

        let xmin = ulx.min(urx).min(lrx).min(llx);
        let xmax = ulx.max(urx).max(lrx).max(llx);
        let ymin = uly.min(ury).min(lry).min(lly);
        let ymax = uly.max(ury).max(lry).max(lly);

        let skew_x = metadata.skew_x();
        let skew_y = metadata.skew_y();
        let has_skew = skew_x != 0.0 || skew_y != 0.0;

        let has_outdb = bands
            .iter()
            .any(|band| matches!(band.metadata().storage_type(), Ok(StorageType::OutDbRef)));

        // Write: [WxH/nbands] @ [xmin ymin xmax ymax]
        write!(
            f,
            "[{width}x{height}/{nbands}] @ [{xmin} {ymin} {xmax} {ymax}]"
        )?;

        // Conditionally append skew info when the raster is rotated/skewed
        if has_skew {
            write!(f, " skew=({skew_x}, {skew_y})")?;
        }

        // Append CRS if present. For PROJJSON (starts with '{'), show compact placeholder.
        if let Some(crs) = raster.crs() {
            if crs.starts_with('{') {
                write!(f, " / {{...}}")?;
            } else {
                write!(f, " / {crs}")?;
            }
        }

        if has_outdb {
            write!(f, " <outdb>")?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::RasterStructArray;
    use sedona_testing::rasters::generate_test_rasters;

    #[test]
    fn display_non_skewed_raster() {
        // i=0: w=1, h=2, scale=(0.1, -0.2), skew=(0, 0), CRS=OGC:CRS84
        // Bounds: xmin=1, ymin=1.6, xmax=1.1, ymax=2
        let rasters = generate_test_rasters(1, None).unwrap();
        let raster_array = RasterStructArray::new(&rasters);
        let raster = raster_array.get(0).unwrap();

        let display = format!("{}", RasterDisplay(&raster));
        assert_eq!(display, "[1x2/1] @ [1 1.6 1.1 2] / OGC:CRS84");
    }

    #[test]
    fn display_skewed_raster() {
        // i=2: w=3, h=4, scale=(0.2, -0.4), skew=(0.06, 0.08), CRS=OGC:CRS84
        // Corners: (3,4), (3.6,4.24), (3.84,2.64), (3.24,2.4)
        // AABB: xmin=3, ymin=2.4, xmax=3.84, ymax=4.24
        let rasters = generate_test_rasters(3, None).unwrap();
        let raster_array = RasterStructArray::new(&rasters);
        let raster = raster_array.get(2).unwrap();

        let display = format!("{}", RasterDisplay(&raster));
        assert_eq!(
            display,
            "[3x4/1] @ [3 2.4 3.84 4.24] skew=(0.06, 0.08) / OGC:CRS84"
        );
    }

    #[test]
    fn display_write_to_fmt_write() {
        // Verify RasterDisplay works with any fmt::Write target (e.g., String)
        let rasters = generate_test_rasters(1, None).unwrap();
        let raster_array = RasterStructArray::new(&rasters);
        let raster = raster_array.get(0).unwrap();

        let mut buf = String::new();
        use std::fmt::Write;
        write!(buf, "{}", RasterDisplay(&raster)).unwrap();
        assert_eq!(buf, "[1x2/1] @ [1 1.6 1.1 2] / OGC:CRS84");
    }
}
