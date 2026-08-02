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

//! Geometry masking machinery shared by the raster functions that select the
//! pixels a geometry covers (RS_Clip, RS_ZonalStats).
//!
//! A mask is built in two steps: [`envelope_window`] clamps the geometry's
//! envelope to a rectangular pixel window on the raster grid, and
//! [`rasterize_geometry_mask`] burns the geometry into a window-sized 0/1 `u8`
//! mask. Callers then interpret the mask however they need — RS_Clip writes
//! nodata outside it, RS_ZonalStats reads the selected pixel values — so this
//! module owns only the window addressing and rasterization, not the
//! per-pixel consumption.

use datafusion_common::{exec_datafusion_err, Result};
use sedona_gdal::gdal::Gdal;
use sedona_gdal::geo_transform::{GeoTransform, GeoTransformEx};
use sedona_gdal::mem::MemDatasetBuilder;
use sedona_gdal::raster::types::GdalDataType;
use sedona_gdal::vector::geometry::Geometry;

/// A rectangular pixel window (offset + size) into a raster grid.
#[derive(Debug, Clone, Copy)]
pub struct PixelWindow {
    pub col_off: usize,
    pub row_off: usize,
    pub width: usize,
    pub height: usize,
}

/// The geometry's envelope intersected with the raster extent, snapped outward
/// to whole pixels. `None` when the clamped window has no area (the envelope
/// falls entirely outside the raster, or only touches its boundary).
///
/// This is the window PostGIS ST_Clip, `gdalwarp -crop_to_cutline`, and Sedona
/// Spark's raster functions use. All four envelope corners are mapped through
/// the inverse geotransform (so a skewed/rotated raster still gets a correct
/// superset window) and the resulting pixel-space bbox is floored/ceiled to
/// whole pixels. A degenerate envelope (point/line) landing exactly on a grid
/// line is widened to one pixel so the rasterizer — not the snapping — decides
/// whether it burns.
pub fn envelope_window(
    geometry: &Geometry,
    transform: &GeoTransform,
    width: usize,
    height: usize,
) -> Result<Option<PixelWindow>> {
    let env = geometry.envelope();
    let inverse = transform
        .invert()
        .map_err(|e| exec_datafusion_err!("raster mask: geotransform is not invertible: {e}"))?;

    let corners = [
        (env.MinX, env.MinY),
        (env.MinX, env.MaxY),
        (env.MaxX, env.MinY),
        (env.MaxX, env.MaxY),
    ];
    let mut min_col = f64::INFINITY;
    let mut max_col = f64::NEG_INFINITY;
    let mut min_row = f64::INFINITY;
    let mut max_row = f64::NEG_INFINITY;
    for (x, y) in corners {
        let (col, row) = inverse.apply(x, y);
        min_col = min_col.min(col);
        max_col = max_col.max(col);
        min_row = min_row.min(row);
        max_row = max_row.max(row);
    }

    let col0 = min_col.floor();
    let row0 = min_row.floor();
    let col1 = max_col.ceil().max(col0 + 1.0);
    let row1 = max_row.ceil().max(row0 + 1.0);

    // Intersect with the raster extent. `>=` also rejects the NaN envelope of
    // an empty geometry.
    let col0 = col0.max(0.0);
    let row0 = row0.max(0.0);
    let col1 = col1.min(width as f64);
    let row1 = row1.min(height as f64);
    if !(col0 < col1 && row0 < row1) {
        return Ok(None);
    }

    Ok(Some(PixelWindow {
        col_off: col0 as usize,
        row_off: row0 as usize,
        width: (col1 - col0) as usize,
        height: (row1 - row0) as usize,
    }))
}

/// Rasterize `geometry` into a window-sized `u8` mask: 1 where the geometry
/// covers a pixel, 0 elsewhere.
///
/// The mask is written into the caller-owned `out` buffer (cleared first), whose
/// allocation is reused across calls so a per-row rasterization does not allocate
/// a fresh `Vec` each time. It ends up `window.width * window.height` bytes long.
///
/// The mask is a MEM UInt8 dataset covering only `window`, with the raster
/// geotransform shifted to the window's upper-left corner so pixel indices in
/// the mask line up with the same-offset pixels of the source raster. GDAL's
/// MEM driver zero-fills the band on creation, so only the burn (value 1,
/// inside the geometry) has to be written. `geometry` is consumed, since the
/// burn is its only remaining use.
pub fn rasterize_geometry_mask(
    gdal: &Gdal,
    geometry: Geometry,
    transform: &GeoTransform,
    window: &PixelWindow,
    all_touched: bool,
    out: &mut Vec<u8>,
) -> Result<()> {
    let mask_dataset =
        MemDatasetBuilder::create(gdal, window.width, window.height, 1, GdalDataType::UInt8)
            .map_err(|e| exec_datafusion_err!("raster mask: failed to create mask dataset: {e}"))?;
    let (window_ulx, window_uly) = transform.apply(window.col_off as f64, window.row_off as f64);
    let mask_transform = [
        window_ulx,
        transform[1],
        transform[2],
        window_uly,
        transform[4],
        transform[5],
    ];
    mask_dataset
        .set_geo_transform(&mask_transform)
        .map_err(|e| exec_datafusion_err!("raster mask: failed to set mask geotransform: {e}"))?;

    gdal.rasterize_affine(&mask_dataset, &[1], &[geometry], &[1.0], all_touched)
        .map_err(|e| exec_datafusion_err!("raster mask: failed to rasterize geometry: {e}"))?;

    let mask_band = mask_dataset
        .rasterband(1)
        .map_err(|e| exec_datafusion_err!("raster mask: failed to read mask band: {e}"))?;
    let mask_buffer = mask_band
        .read_as::<u8>(
            (0, 0),
            (window.width, window.height),
            (window.width, window.height),
            None,
        )
        .map_err(|e| exec_datafusion_err!("raster mask: failed to read mask: {e}"))?;
    out.clear();
    out.extend_from_slice(mask_buffer.data());
    Ok(())
}
