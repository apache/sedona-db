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
pub struct RasterDisplay<'a>(pub &'a dyn RasterRef);

impl fmt::Display for RasterDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let raster = self.0;

        let width = raster.width().unwrap_or(0);
        let height = raster.height().unwrap_or(0);
        let nbands = raster.num_bands();

        // Compute axis-aligned bounding box from 4 corners in world coordinates.
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

        let t = raster.transform();
        let skew_x = t[2];
        let skew_y = t[4];
        let has_skew = skew_x != 0.0 || skew_y != 0.0;

        let has_outdb = (0..nbands).any(|i| {
            raster
                .band(i)
                .is_some_and(|b| b.outdb_uri().is_some())
        });

        write!(
            f,
            "[{width}x{height}/{nbands}] @ [{xmin} {ymin} {xmax} {ymax}]"
        )?;

        if has_skew {
            write!(f, " skew=({skew_x}, {skew_y})")?;
        }

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
