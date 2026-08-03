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

//! Reusable, tile-at-a-time tiling core shared by raster tiling operators.
//!
//! This module holds the geometry and pixel work of cutting a raster into a
//! grid of tiles, exposed so a caller can drive it one tile at a time:
//! [`tile_grid_dims`] validates the request and reports the grid size, then for
//! each grid position a [`TileWindow`] resolves the source rectangle and output
//! extent and [`append_tile`] materializes exactly one tile raster. The
//! [`crate::rs_tile`] UDF is one such driver (it packages the tiles into a
//! `List<Struct<x, y, tile>>`); the primitives themselves carry no UDF surface.

use datafusion_common::error::Result;
use datafusion_common::{exec_datafusion_err, exec_err};

use sedona_common::sedona_internal_err;
use sedona_raster::array::RasterRefImpl;
use sedona_raster::builder::RasterBuilder;
use sedona_raster::geo_transform::{GeoTransform, GeoTransformEx};
use sedona_raster::traits::{is_spatial_dim_pair, nodata_f64_to_bytes, RasterRef};

use crate::utils::{append_stacked_band, BandHeader};

/// The optional tiling parameters for one tiling call, resolved from the
/// positional arguments for a single row.
pub struct TileParams<'a> {
    /// 1-based band indices to include in each tile, in the given order.
    /// `None` includes every band.
    pub bands: Option<&'a [i64]>,
    /// Pad the last partial row/column of tiles to the full tile size with a
    /// nodata fill. When false the smaller edge tile is emitted.
    pub pad_with_nodata: bool,
    /// The value written to padded pixels. Only meaningful with
    /// `pad_with_nodata = true`; it is an error to set it otherwise. Defaults to
    /// the band's own nodata value, or the band data type's minimum if it has
    /// none. It is an error if the value does not fit the band's data type.
    pub nodata: Option<f64>,
}

/// Validate a tiling request and compute the tile grid dimensions for a raster
/// of `width` x `height` pixels cut into `tile_width` x `tile_height` tiles.
///
/// Tiles are emitted row-major: `num_tile_x = ceil(width / tile_width)` and
/// `num_tile_y = ceil(height / tile_height)`. The total tile count (and hence
/// each grid dimension, since both are >= 1) must fit `i32` because the grid
/// positions are Int32 (Spark parity) and the tiles accumulate into an
/// Int32-offset list.
pub fn tile_grid_dims(
    width: i64,
    height: i64,
    tile_width: i64,
    tile_height: i64,
) -> Result<(usize, usize)> {
    if tile_width < 1 || tile_height < 1 {
        return exec_err!(
            "RS_Tile: tile_width and tile_height must be >= 1, got {tile_width}x{tile_height}"
        );
    }
    if width < 0 || height < 0 {
        return sedona_internal_err!("RS_Tile: negative raster extent {width}x{height}");
    }
    let width = width as usize;
    let height = height as usize;
    let tile_w = tile_width as usize;
    let tile_h = tile_height as usize;

    let num_tile_x = width.div_ceil(tile_w);
    let num_tile_y = height.div_ceil(tile_h);
    // x/y grid positions are Int32 (Spark parity) and the tiles accumulate into
    // an Int32-offset list, so the total tile count (and hence each grid
    // dimension, since both are >= 1) must fit i32. Reject an overflowing grid up
    // front rather than wrap a position or panic building the list offsets.
    if num_tile_x
        .checked_mul(num_tile_y)
        .filter(|&n| n <= i32::MAX as usize)
        .is_none()
    {
        return exec_err!(
            "RS_Tile: tile grid {num_tile_x}x{num_tile_y} exceeds the Int32 tile-count limit"
        );
    }

    Ok((num_tile_x, num_tile_y))
}

/// The pixel window one tile copies from the source, plus the tile's output
/// extent.
pub struct TileWindow {
    /// Source pixel offset of the tile's upper-left corner.
    x0: usize,
    y0: usize,
    /// Source pixels actually available for this tile (<= tile size at the edge).
    rect_w: usize,
    rect_h: usize,
    /// The tile's output extent: the full tile size when padding, otherwise the
    /// (possibly smaller) source rectangle at the edge.
    out_w: usize,
    out_h: usize,
}

impl TileWindow {
    pub fn new(
        tile_x: usize,
        tile_y: usize,
        tile_w: usize,
        tile_h: usize,
        width: usize,
        height: usize,
        pad: bool,
    ) -> Self {
        let x0 = tile_x * tile_w;
        let y0 = tile_y * tile_h;
        let rect_w = tile_w.min(width - x0);
        let rect_h = tile_h.min(height - y0);
        let (out_w, out_h) = if pad {
            (tile_w, tile_h)
        } else {
            (rect_w, rect_h)
        };
        Self {
            x0,
            y0,
            rect_w,
            rect_h,
            out_w,
            out_h,
        }
    }

    /// Whether this tile was actually padded: a short edge tile whose output
    /// extent exceeds the source pixels available. Interior tiles are fully
    /// covered by the source and return false.
    pub fn needs_padding(&self) -> bool {
        self.out_w > self.rect_w || self.out_h > self.rect_h
    }
}

/// Resolve which 1-based band indices to include, validating each against the
/// raster's band count. `None` (band-less overload) and an empty list both
/// select every band in order, matching Sedona Spark.
pub fn resolve_band_indices(bands: Option<&[i64]>, num_bands: usize) -> Result<Vec<usize>> {
    match bands {
        None | Some([]) => Ok((1..=num_bands).collect()),
        Some(bands) => bands
            .iter()
            .map(|&band| {
                if band < 1 || band as usize > num_bands {
                    return exec_err!("RS_Tile: band {band} is out of range (1-{num_bands})");
                }
                Ok(band as usize)
            })
            .collect(),
    }
}

/// Build one tile raster and append it to `rast_builder`.
pub fn append_tile(
    raster: &RasterRefImpl<'_>,
    band_indices: &[usize],
    window: &TileWindow,
    params: &TileParams<'_>,
    rast_builder: &mut RasterBuilder,
) -> Result<()> {
    // The tile's upper-left corner is the source origin translated by the tile's
    // pixel offset (scale/skew unchanged), via the shared geotransform apply.
    // Matches the crop-origin shift in RS_Clip and PostGIS ST_Clip.
    let src: GeoTransform = raster
        .transform()
        .try_into()
        .map_err(|_| exec_datafusion_err!("RS_Tile: expected a 6-element geotransform"))?;
    let (new_ulx, new_uly) = src.apply(window.x0 as f64, window.y0 as f64);
    let tile_transform = [new_ulx, src[1], src[2], new_uly, src[4], src[5]];

    // Spatial extent after tiling. `spatial_dims`/`spatial_shape` are kept in the
    // raster's own axis order (X-first), so map each spatial dim to its tile size
    // by name rather than assuming an order (mirrors RS_Clip).
    let spatial_dims = raster.spatial_dims();
    let x_dim = raster.x_dim();
    let tile_spatial_shape: Vec<i64> = spatial_dims
        .iter()
        .map(|&d| {
            if d == x_dim {
                window.out_w as i64
            } else {
                window.out_h as i64
            }
        })
        .collect();

    rast_builder
        .start_raster_nd(
            &tile_transform,
            &spatial_dims,
            &tile_spatial_shape,
            raster.crs(),
        )
        .map_err(|e| exec_datafusion_err!("RS_Tile: failed to start raster: {e}"))?;

    for &band_idx in band_indices {
        append_tile_band(raster, band_idx, window, params, rast_builder)?;
    }

    rast_builder
        .finish_raster()
        .map_err(|e| exec_datafusion_err!("RS_Tile: failed to finish raster: {e}"))?;
    Ok(())
}

/// Copy one band's tile window and append it to `rast_builder`.
fn append_tile_band(
    raster: &RasterRefImpl<'_>,
    band_idx: usize,
    window: &TileWindow,
    params: &TileParams<'_>,
    rast_builder: &mut RasterBuilder,
) -> Result<()> {
    // `band_idx` is 1-based; the `band`/`band_name` accessors are 0-based.
    let band = raster
        .band(band_idx - 1)
        .map_err(|e| exec_datafusion_err!("RS_Tile: failed to get band {band_idx}: {e}"))?;
    let band_name = raster.band_name(band_idx - 1).map(|s| s.to_string());

    let data_type = band.data_type();
    let byte_size = data_type.byte_size();

    // The trailing two axes are the spatial (y, x) plane; anything before them is
    // a stack of planes the 2-D tiling is broadcast over (mirrors RS_Clip).
    let shape = band.shape().to_vec();
    let dim_names: Vec<String> = band.dim_names().iter().map(|s| s.to_string()).collect();
    let ndim = shape.len();
    if ndim < 2 {
        return exec_err!(
            "RS_Tile: band {band_idx} has {ndim} dimension(s); a 2-D (y, x) plane is required"
        );
    }
    if !is_spatial_dim_pair(&dim_names[ndim - 2], &dim_names[ndim - 1]) {
        return exec_err!(
            "RS_Tile: band {band_idx} trailing dims {:?} are not a (y, x) spatial pair",
            &dim_names[ndim - 2..]
        );
    }
    let (plane_h, plane_w) = (shape[ndim - 2] as usize, shape[ndim - 1] as usize);
    let width = raster.width()? as usize;
    let height = raster.height()? as usize;
    if plane_w != width || plane_h != height {
        return exec_err!(
            "RS_Tile: band {band_idx} spatial extent {plane_w}x{plane_h} does not match the raster {width}x{height}"
        );
    }
    let n_planes: usize = shape[..ndim - 2].iter().map(|&d| d as usize).product();

    // Borrow the source band bytes (read-only; the copy below writes only into
    // the tile's own buffer, so no copy of the source is needed here).
    let nd_buffer = band
        .nd_buffer()
        .map_err(|e| exec_datafusion_err!("RS_Tile: failed to read band {band_idx}: {e}"))?;
    let source = nd_buffer
        .as_contiguous()
        .map_err(|e| exec_datafusion_err!("RS_Tile: band {band_idx} is not contiguous: {e}"))?;
    let in_plane_bytes = width * height * byte_size;
    if source.len() != n_planes * in_plane_bytes {
        return exec_err!(
            "RS_Tile: band {band_idx} byte length {} does not match {n_planes} planes of {width}x{height}",
            source.len()
        );
    }

    // Only tiles that were actually padded carry the nodata fill: interior tiles
    // are fully covered by the source, so they keep the source band's own nodata
    // (matching Sedona Spark, which stamps the pad nodata only on short edge
    // tiles). The fill is the explicit noDataVal, else the band's own nodata,
    // else the data type minimum (matching RS_Clip), guarded so a value that
    // doesn't fit the band dtype errors rather than silently saturating.
    let (pad_fill, tile_nodata): (Option<Vec<u8>>, Option<Vec<u8>>) = if window.needs_padding() {
        let fill = match params.nodata {
            Some(value) => nodata_f64_to_bytes(value, &data_type).map_err(|e| {
                exec_datafusion_err!("RS_Tile: invalid nodata for band {band_idx}: {e}")
            })?,
            None => match band.nodata() {
                Some(bytes) => bytes.to_vec(),
                None => data_type.min_value_le_bytes(),
            },
        };
        if fill.len() != byte_size {
            return sedona_internal_err!(
                "RS_Tile: band {band_idx} nodata fill is {} bytes, expected {byte_size} for {data_type:?}",
                fill.len()
            );
        }
        (Some(fill.clone()), Some(fill))
    } else {
        (None, band.nodata().map(|bytes| bytes.to_vec()))
    };

    let mut out_shape = shape[..ndim - 2].to_vec();
    out_shape.push(window.out_h as i64);
    out_shape.push(window.out_w as i64);
    let dim_names_ref: Vec<&str> = dim_names.iter().map(String::as_str).collect();

    // Copy each source plane's tile window and stack the planes into the N-D tile
    // band. `in_plane_bytes` is the full source plane; the tile window may be
    // smaller (a crop) or larger (padded past the source edge).
    append_stacked_band(
        rast_builder,
        &BandHeader {
            name: band_name.as_deref(),
            dim_names: &dim_names_ref,
            shape: &out_shape,
            data_type,
            nodata: tile_nodata.as_deref(),
        },
        |plane, out| {
            let plane_bytes = &source[plane * in_plane_bytes..(plane + 1) * in_plane_bytes];
            copy_tile_window(
                plane_bytes,
                width,
                window,
                byte_size,
                pad_fill.as_deref(),
                out,
            )
        },
    )
}

/// Copy one source plane's tile window into `out`, padding out-of-bounds pixels
/// with the nodata fill when the tile extends past the source edge. Bytes are
/// appended so one `out` buffer can serve every plane of a band.
///
/// When not padding, `out_w == rect_w` and `out_h == rect_h`, so the padding
/// branches never run and `pad_fill` is unused.
fn copy_tile_window(
    src_plane: &[u8],
    full_width: usize,
    window: &TileWindow,
    byte_size: usize,
    pad_fill: Option<&[u8]>,
    out: &mut Vec<u8>,
) -> Result<()> {
    let out_row_bytes = window.out_w * byte_size;
    let base = out.len();
    out.resize(base + window.out_h * out_row_bytes, 0);
    let dst = &mut out[base..];

    let need_col_pad = window.out_w > window.rect_w;
    let need_row_pad = window.out_h > window.rect_h;
    // A single nodata row is reused for both the padded columns and the padded
    // rows. Only built when padding is actually needed for this tile.
    let nodata_row = if need_col_pad || need_row_pad {
        let fill = pad_fill.ok_or_else(|| {
            exec_datafusion_err!("RS_Tile: padding required but no nodata fill resolved")
        })?;
        Some(fill.repeat(window.out_w))
    } else {
        None
    };

    let copy_bytes = window.rect_w * byte_size;
    for row in 0..window.out_h {
        let dst_row = &mut dst[row * out_row_bytes..(row + 1) * out_row_bytes];
        if row < window.rect_h {
            let src_start = ((window.y0 + row) * full_width + window.x0) * byte_size;
            dst_row[..copy_bytes].copy_from_slice(&src_plane[src_start..src_start + copy_bytes]);
            if need_col_pad {
                let Some(nodata_row) = nodata_row.as_ref() else {
                    return sedona_internal_err!("RS_Tile: padded column without a nodata row");
                };
                dst_row[copy_bytes..].copy_from_slice(&nodata_row[copy_bytes..]);
            }
        } else {
            let Some(nodata_row) = nodata_row.as_ref() else {
                return sedona_internal_err!("RS_Tile: padded row without a nodata row");
            };
            dst_row.copy_from_slice(nodata_row);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow_array::builder::Int32Builder;
    use arrow_array::ArrayRef;
    use sedona_raster::array::RasterStructArray;
    use sedona_testing::raster_spec::{assert_rasters_equal, RasterSpec};

    /// The optional tiling parameters for the core-tiling helper tests.
    fn params(bands: Option<&[i64]>, pad_with_nodata: bool, nodata: Option<f64>) -> TileParams<'_> {
        TileParams {
            bands,
            pad_with_nodata,
            nodata,
        }
    }

    /// A 5x3 EPSG-less raster, origin (0, 3), north-up 1x1 pixels, one UInt8
    /// band with values 1..=15 (row-major). The odd extent makes both the last
    /// column (width 5 vs tile 2) and the last row (height 3 vs tile 2) partial,
    /// exercising the edge-tile paths. Expected tile pixels below come from the
    /// numpy reference in the PR description.
    fn source_5x3() -> RasterSpec {
        RasterSpec::d2(5, 3)
            .crs(None)
            .transform([0.0, 1.0, 0.0, 3.0, 0.0, -1.0])
            .band_values(&[1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    }

    /// A north-up UInt8 tile: origin (`ulx`, `uly`), 1x1 pixels, no nodata.
    fn tile(width: i64, height: i64, ulx: f64, uly: f64, values: &[u8]) -> RasterSpec {
        RasterSpec::d2(width, height)
            .crs(None)
            .transform([ulx, 1.0, 0.0, uly, 0.0, -1.0])
            .band_values(values)
    }

    /// Drive the reusable tiling core over a single source raster the way an
    /// operator does: resolve the grid, then emit one tile per grid position in
    /// row-major order. Returns the grid positions and the tiles as a raster
    /// array (so tiles can be asserted with the declarative
    /// `assert_rasters_equal`).
    fn try_explode(
        spec: &RasterSpec,
        tile_width: i64,
        tile_height: i64,
        tile_params: &TileParams<'_>,
    ) -> Result<(Vec<(i32, i32)>, ArrayRef)> {
        let array = spec.build();
        let rasters = RasterStructArray::try_new(&array).unwrap();
        let raster = rasters.get(0).unwrap();

        let mut x_builder = Int32Builder::new();
        let mut y_builder = Int32Builder::new();
        let mut rast_builder = RasterBuilder::new(4);

        let width = raster.width()?;
        let height = raster.height()?;
        let (num_tile_x, num_tile_y) = tile_grid_dims(width, height, tile_width, tile_height)?;
        // width/height are non-negative and tile sizes are >= 1 (validated above).
        let width = width as usize;
        let height = height as usize;
        let tile_w = tile_width as usize;
        let tile_h = tile_height as usize;

        let band_indices = resolve_band_indices(tile_params.bands, raster.num_bands())?;

        for tile_y in 0..num_tile_y {
            for tile_x in 0..num_tile_x {
                let window = TileWindow::new(
                    tile_x,
                    tile_y,
                    tile_w,
                    tile_h,
                    width,
                    height,
                    tile_params.pad_with_nodata,
                );
                append_tile(
                    &raster,
                    &band_indices,
                    &window,
                    tile_params,
                    &mut rast_builder,
                )?;
                x_builder.append_value(tile_x as i32);
                y_builder.append_value(tile_y as i32);
            }
        }

        let xs = x_builder.finish();
        let ys = y_builder.finish();
        let positions = (0..num_tile_x * num_tile_y)
            .map(|i| (xs.value(i), ys.value(i)))
            .collect();
        let tiles: ArrayRef = Arc::new(rast_builder.finish().unwrap());
        Ok((positions, tiles))
    }

    /// Run the core tiling over a single source raster, returning the tile grid
    /// positions and the tiles as a raster array.
    fn explode(
        spec: &RasterSpec,
        tile_width: i64,
        tile_height: i64,
        tile_params: TileParams<'_>,
    ) -> (Vec<(i32, i32)>, ArrayRef) {
        try_explode(spec, tile_width, tile_height, &tile_params).unwrap()
    }

    /// Run the core tiling and return the error string (helper for the error
    /// tests, which all expect the tiling to fail).
    fn explode_one_error(
        spec: &RasterSpec,
        tile_width: i64,
        tile_height: i64,
        tile_params: TileParams<'_>,
    ) -> String {
        try_explode(spec, tile_width, tile_height, &tile_params)
            .unwrap_err()
            .to_string()
    }

    #[test]
    fn tiles_2x2_without_padding() {
        // 5x3 tiled 2x2 with pad_with_nodata = false: 3x2 grid; the right/bottom
        // edge tiles keep their smaller source dimensions (1x2, 2x1, 1x1).
        let (positions, tiles) = explode(&source_5x3(), 2, 2, params(None, false, None));
        assert_eq!(
            positions,
            vec![(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
        );
        assert_rasters_equal(
            &tiles,
            &[
                Some(tile(2, 2, 0.0, 3.0, &[1, 2, 6, 7])),
                Some(tile(2, 2, 2.0, 3.0, &[3, 4, 8, 9])),
                Some(tile(1, 2, 4.0, 3.0, &[5, 10])),
                Some(tile(2, 1, 0.0, 1.0, &[11, 12])),
                Some(tile(2, 1, 2.0, 1.0, &[13, 14])),
                Some(tile(1, 1, 4.0, 1.0, &[15])),
            ],
        );
    }

    #[test]
    fn tiles_2x2_with_padding() {
        // Same grid, but the short edge tiles (right column / bottom row) are
        // padded to the full 2x2 with nodata 0 and record nodata 0. The two
        // fully-interior tiles need no padding, so they keep the source band's
        // nodata (here none) -- matching Sedona Spark, which stamps the pad
        // nodata only on the tiles it actually padded.
        let (positions, tiles) = explode(&source_5x3(), 2, 2, params(None, true, Some(0.0)));
        assert_eq!(
            positions,
            vec![(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
        );
        let interior = |ulx: f64, uly: f64, values: &[u8]| Some(tile(2, 2, ulx, uly, values));
        let padded =
            |ulx: f64, uly: f64, values: &[u8]| Some(tile(2, 2, ulx, uly, values).nodata(0u8));
        assert_rasters_equal(
            &tiles,
            &[
                interior(0.0, 3.0, &[1, 2, 6, 7]),
                interior(2.0, 3.0, &[3, 4, 8, 9]),
                padded(4.0, 3.0, &[5, 0, 10, 0]),
                padded(0.0, 1.0, &[11, 12, 0, 0]),
                padded(2.0, 1.0, &[13, 14, 0, 0]),
                padded(4.0, 1.0, &[15, 0, 0, 0]),
            ],
        );
    }

    #[test]
    fn tile_size_equal_to_or_larger_than_raster_yields_one_tile() {
        // A tile as big as (5x3) or bigger (8x8) than the raster produces a
        // single tile that is the whole raster verbatim when not padding.
        for (tw, th) in [(5, 3), (8, 8)] {
            let (positions, tiles) = explode(&source_5x3(), tw, th, params(None, false, None));
            assert_eq!(positions, vec![(0, 0)], "tile {tw}x{th}");
            assert_rasters_equal(
                &tiles,
                &[Some(tile(
                    5,
                    3,
                    0.0,
                    3.0,
                    &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                ))],
            );
        }
    }

    #[test]
    fn tile_larger_than_raster_with_padding() {
        // One 8x8 tile, the raster in its top-left corner, the rest nodata 0.
        let (positions, tiles) = explode(&source_5x3(), 8, 8, params(None, true, Some(0.0)));
        assert_eq!(positions, vec![(0, 0)]);
        let mut expected = vec![0u8; 64];
        for (row, chunk) in [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
            .iter()
            .enumerate()
        {
            expected[row * 8..row * 8 + 5].copy_from_slice(chunk);
        }
        assert_rasters_equal(&tiles, &[Some(tile(8, 8, 0.0, 3.0, &expected).nodata(0u8))]);
    }

    #[test]
    fn tiles_1x1() {
        // 1x1 tiles: one tile per source pixel (15), each carrying that pixel
        // and an origin at the pixel's own upper-left corner.
        let (positions, tiles) = explode(&source_5x3(), 1, 1, params(None, false, None));
        assert_eq!(positions.len(), 15);
        let expected_positions: Vec<(i32, i32)> = (0..15).map(|i| (i % 5, i / 5)).collect();
        assert_eq!(positions, expected_positions);

        let expected: Vec<Option<RasterSpec>> = (0..15i64)
            .map(|i| {
                let (col, row) = (i % 5, i / 5);
                Some(tile(1, 1, col as f64, 3.0 - row as f64, &[(i + 1) as u8]))
            })
            .collect();
        assert_rasters_equal(&tiles, &expected);
    }

    #[test]
    fn multiband_selects_and_orders_bands() {
        // A 2x2 raster with two distinct bands; one tile equal to the raster.
        let source = RasterSpec::d2(2, 2)
            .crs(None)
            .transform([0.0, 1.0, 0.0, 2.0, 0.0, -1.0])
            .band_values(&[1u8, 2, 3, 4])
            .band_values(&[10u8, 20, 30, 40]);

        // Default: all bands, in order.
        let (_, tiles) = explode(&source, 2, 2, params(None, false, None));
        assert_rasters_equal(
            &tiles,
            &[Some(
                RasterSpec::d2(2, 2)
                    .crs(None)
                    .transform([0.0, 1.0, 0.0, 2.0, 0.0, -1.0])
                    .band_values(&[1u8, 2, 3, 4])
                    .band_values(&[10u8, 20, 30, 40]),
            )],
        );

        // Explicit selection keeps only band 2.
        let (_, tiles) = explode(&source, 2, 2, params(Some(&[2]), false, None));
        assert_rasters_equal(
            &tiles,
            &[Some(
                RasterSpec::d2(2, 2)
                    .crs(None)
                    .transform([0.0, 1.0, 0.0, 2.0, 0.0, -1.0])
                    .band_values(&[10u8, 20, 30, 40]),
            )],
        );
    }

    #[test]
    fn source_band_nodata_preserved_when_not_padding() {
        // Not padding: the tile keeps the source band's own nodata verbatim
        // (no fill is introduced).
        let source = RasterSpec::d2(2, 1)
            .crs(None)
            .transform([0.0, 1.0, 0.0, 1.0, 0.0, -1.0])
            .band_values(&[7u8, 8])
            .nodata(9u8);
        let (_, tiles) = explode(&source, 1, 1, params(None, false, None));
        assert_rasters_equal(
            &tiles,
            &[
                Some(tile(1, 1, 0.0, 1.0, &[7]).nodata(9u8)),
                Some(tile(1, 1, 1.0, 1.0, &[8]).nodata(9u8)),
            ],
        );
    }

    #[test]
    fn padding_without_nodata_uses_type_minimum() {
        // pad_with_nodata but no explicit nodata and no band nodata: the fill is
        // the band data type minimum (0 for UInt8), recorded as the tile nodata.
        let source = RasterSpec::d2(3, 1)
            .crs(None)
            .transform([0.0, 1.0, 0.0, 1.0, 0.0, -1.0])
            .band_values(&[1u8, 2, 3]);
        let (positions, tiles) = explode(&source, 2, 1, params(None, true, None));
        assert_eq!(positions, vec![(0, 0), (1, 0)]);
        assert_rasters_equal(
            &tiles,
            &[
                // Interior tile: no padding, so it keeps the source nodata (none).
                Some(tile(2, 1, 0.0, 1.0, &[1, 2])),
                // Edge tile padded to width 2 with the UInt8 minimum, recorded as
                // the tile nodata.
                Some(tile(2, 1, 2.0, 1.0, &[3, 0]).nodata(0u8)),
            ],
        );
    }

    #[test]
    fn tiling_is_dtype_agnostic() {
        // The window copy is byte-oriented, so a multi-byte dtype must tile the
        // same way. A 3x1 UInt16 raster tiled 2x1 (one full + one edge tile).
        let source = RasterSpec::d2(3, 1)
            .crs(None)
            .transform([0.0, 1.0, 0.0, 1.0, 0.0, -1.0])
            .band_values(&[100u16, 200, 300]);
        let (positions, tiles) = explode(&source, 2, 1, params(None, false, None));
        assert_eq!(positions, vec![(0, 0), (1, 0)]);
        assert_rasters_equal(
            &tiles,
            &[
                Some(
                    RasterSpec::d2(2, 1)
                        .crs(None)
                        .transform([0.0, 1.0, 0.0, 1.0, 0.0, -1.0])
                        .band_values(&[100u16, 200]),
                ),
                Some(
                    RasterSpec::d2(1, 1)
                        .crs(None)
                        .transform([2.0, 1.0, 0.0, 1.0, 0.0, -1.0])
                        .band_values(&[300u16]),
                ),
            ],
        );
    }

    #[test]
    fn padded_interior_tile_keeps_source_nodata() {
        // With pad_with_nodata and an explicit noDataVal that differs from the
        // source band's own nodata, only the padded edge tile gets the pad
        // nodata; the interior tile keeps the SOURCE nodata (99, not the pad fill
        // 0), matching Sedona Spark's "stamp only the padded tiles" behavior.
        let source = RasterSpec::d2(3, 1)
            .crs(None)
            .transform([0.0, 1.0, 0.0, 1.0, 0.0, -1.0])
            .band_values(&[1u8, 2, 3])
            .nodata(99u8);
        let (positions, tiles) = explode(&source, 2, 1, params(None, true, Some(0.0)));
        assert_eq!(positions, vec![(0, 0), (1, 0)]);
        assert_rasters_equal(
            &tiles,
            &[
                // Interior tile: keeps the source band's nodata (99), not the fill.
                Some(tile(2, 1, 0.0, 1.0, &[1, 2]).nodata(99u8)),
                // Edge tile padded to width 2 with the supplied fill 0.
                Some(tile(2, 1, 2.0, 1.0, &[3, 0]).nodata(0u8)),
            ],
        );
    }

    #[test]
    fn tile_size_below_one_errors() {
        let err = explode_one_error(&source_5x3(), 0, 2, params(None, false, None));
        assert!(err.contains("must be >= 1"), "unexpected error: {err}");
    }

    #[test]
    fn band_out_of_range_errors() {
        // source_5x3 has a single band; band 2 is out of range.
        let err = explode_one_error(&source_5x3(), 2, 2, params(Some(&[2]), false, None));
        assert!(err.contains("out of range"), "unexpected error: {err}");
    }

    #[test]
    fn empty_bands_selects_all_bands() {
        // An empty bandIndices array selects every band, matching Sedona Spark
        // (which treats empty and NULL alike as "all bands"). On the single-band
        // source_5x3 this tiles the whole raster like the band-less overload.
        let (positions, tiles) = explode(&source_5x3(), 5, 3, params(Some(&[]), false, None));
        assert_eq!(positions, vec![(0, 0)]);
        assert_rasters_equal(
            &tiles,
            &[Some(tile(
                5,
                3,
                0.0,
                3.0,
                &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            ))],
        );
    }
}
