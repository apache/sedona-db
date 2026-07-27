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

//! RS_Tile UDF - split a raster into a grid of tiles.
//!
//! Mirrors Sedona Spark's `RS_Tile`: a raster is cut into a grid of
//! `tile_width` x `tile_height` tiles. Spark returns an `Array<Raster>`;
//! SedonaDB returns a scalar `List<Struct<x, y, tile>>` — one list per input
//! raster, each item additionally carrying the tile's grid position `(x, y)`
//! (tile column, tile row) — that callers `UNNEST` to get one row per tile.
//! This mirrors how [`crate::rs_polygonize`] returns `List<Struct<geom, value>>`.
//!
//! The last row/column of tiles may not divide evenly: with
//! `pad_with_nodata = true` the edge tiles are padded to the full tile size with
//! a nodata fill; otherwise the smaller edge tile is emitted as-is.

use std::sync::Arc;

use arrow_array::builder::{Int32Builder, NullBufferBuilder, OffsetBufferBuilder};
use arrow_array::{Array, ArrayRef, ListArray, StructArray};
use arrow_buffer::Buffer;
use arrow_schema::{DataType, Field, Fields};
use datafusion_common::cast::{as_boolean_array, as_float64_array, as_int64_array, as_list_array};
use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_common::{exec_datafusion_err, exec_err, ScalarValue};
use datafusion_expr::{ColumnarValue, Volatility};

use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_gdal::geo_transform::{GeoTransform, GeoTransformEx};
use sedona_raster::array::RasterRefImpl;
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::{is_spatial_dim_pair, nodata_f64_to_bytes, Bands, RasterRef};
use sedona_raster_functions::rs_ensure_loaded::NEEDS_PIXELS_METADATA_KEY;
use sedona_raster_functions::RasterExecutor;
use sedona_schema::datatypes::{SedonaType, RASTER};
use sedona_schema::matchers::{ArgMatcher, TypeMatcher};

/// RS_Tile() scalar UDF implementation.
///
/// Mirrors Sedona Spark's `RS_Tile` positional overloads (Spark expands the
/// trailing optionals `padWithNoData` / `noDataVal` into sub-overloads):
/// - `RS_Tile(raster, width, height[, padWithNoData[, noDataVal]])`
/// - `RS_Tile(raster, bandIndices, width, height[, padWithNoData[, noDataVal]])`
///
/// Each of the two shapes has three sub-overloads (no optional, with
/// `padWithNoData`, with both), for six kernels total. The shapes are told
/// apart by the type of the argument after the raster: an integer (`width`) for
/// the all-bands shape, or an integer list (`bandIndices`) for the band-subset
/// shape.
pub fn rs_tile_udf() -> SedonaScalarUDF {
    let kernel = |band_arg: BandArg, arg_count: usize| {
        Arc::new(RsTile {
            band_arg,
            arg_count,
        })
    };
    SedonaScalarUDF::new(
        "rs_tile",
        vec![
            // RS_Tile(raster, width, height[, padWithNoData[, noDataVal]])
            kernel(BandArg::All, 3),
            kernel(BandArg::All, 4),
            kernel(BandArg::All, 5),
            // RS_Tile(raster, bandIndices, width, height[, padWithNoData[, noDataVal]])
            kernel(BandArg::Array, 4),
            kernel(BandArg::Array, 5),
            kernel(BandArg::Array, 6),
        ],
        Volatility::Immutable,
    )
    // Reads band pixels, so the planner materializes OutDb rasters via
    // RS_EnsureLoaded before this kernel runs. The output is a list of in-db
    // tiles (not a top-level raster), so RETURNS_BYTES does not apply.
    .with_metadata(NEEDS_PIXELS_METADATA_KEY, "true")
}

/// Which band-selector argument a signature carries after the raster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BandArg {
    /// No band argument — every band is tiled: `(raster, width, height, ...)`.
    All,
    /// An array of 1-based band indices: `(raster, bandIndices, width, height, ...)`.
    Array,
}

/// The argument position of each parameter for a given signature shape.
struct ArgLayout {
    /// The band-selector argument (scalar index or list), when present.
    band: Option<usize>,
    width: usize,
    height: usize,
    /// `padWithNoData`, when present in this overload.
    pad: Option<usize>,
    /// `noDataVal`, when present in this overload.
    nodata: Option<usize>,
}

/// The optional arguments for one tiling call, resolved from the positional
/// arguments for a single row.
struct TileParams<'a> {
    /// 1-based band indices to include in each tile, in the given order.
    /// `None` includes every band.
    bands: Option<&'a [i64]>,
    /// Pad the last partial row/column of tiles to the full tile size with a
    /// nodata fill. When false the smaller edge tile is emitted.
    pad_with_nodata: bool,
    /// The value written to padded pixels. Only meaningful with
    /// `pad_with_nodata = true`; it is an error to set it otherwise. Defaults to
    /// the band's own nodata value, or the band data type's minimum if it has
    /// none. It is an error if the value does not fit the band's data type.
    nodata: Option<f64>,
}

/// Kernel implementation for RS_Tile.
#[derive(Debug)]
struct RsTile {
    /// The band-selector shape this kernel matches.
    band_arg: BandArg,
    /// Number of arguments in the matched signature.
    arg_count: usize,
}

impl SedonaScalarKernel for RsTile {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        // Build the matcher ladder from the shape: raster, the band selector (an
        // integer list for `bandIndices`, absent for the all-bands shape), then
        // width/height, then the trailing optionals this overload includes.
        let mut matchers: Vec<Arc<dyn TypeMatcher + Send + Sync>> = vec![ArgMatcher::is_raster()];
        match self.band_arg {
            BandArg::All => {}
            BandArg::Array => matchers.push(ArgMatcher::is_integer_list()),
        }
        matchers.push(ArgMatcher::is_integer()); // width
        matchers.push(ArgMatcher::is_integer()); // height
        let layout = self.layout();
        if layout.pad.is_some() {
            matchers.push(ArgMatcher::is_boolean());
        }
        if layout.nodata.is_some() {
            matchers.push(ArgMatcher::is_numeric());
        }
        if matchers.len() != self.arg_count {
            return sedona_internal_err!(
                "RS_Tile: built {} matchers for arg_count {}",
                matchers.len(),
                self.arg_count
            );
        }

        let matcher = ArgMatcher::new(matchers, SedonaType::Arrow(tile_list_type()?));
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        self.invoke_batch_from_args(arg_types, args, &SedonaType::Arrow(DataType::Null), 0, None)
    }

    fn invoke_batch_from_args(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
        _return_type: &SedonaType,
        _num_rows: usize,
        _config_options: Option<&ConfigOptions>,
    ) -> Result<ColumnarValue> {
        let num_iterations = RasterExecutor::num_iterations_over(args);
        let layout = self.layout();

        // width / height expanded to arrays and iterated in lockstep with the
        // raster (a NULL in either yields a NULL list for that row).
        let tile_w_array = args[layout.width]
            .clone()
            .cast_to(&DataType::Int64, None)?
            .into_array(num_iterations)?;
        let tile_w_array = as_int64_array(&tile_w_array)?.clone();
        let tile_h_array = args[layout.height]
            .clone()
            .cast_to(&DataType::Int64, None)?
            .into_array(num_iterations)?;
        let tile_h_array = as_int64_array(&tile_h_array)?.clone();

        // pad_with_nodata: absent defaults to false; when present, a NULL yields
        // a NULL row (an unresolved flag, like a NULL raster).
        let pad_array = match layout.pad {
            Some(pos) => args[pos]
                .clone()
                .cast_to(&DataType::Boolean, None)?
                .into_array(num_iterations)?,
            None => ScalarValue::Boolean(Some(false)).to_array_of_size(num_iterations)?,
        };
        let pad_array = as_boolean_array(&pad_array)?.clone();

        // no_data_val: absent means "not supplied" (None); NULL is the same
        // meaningful sentinel, so it stays an Option rather than nulling the row.
        let nodata_array = match layout.nodata {
            Some(pos) => args[pos]
                .clone()
                .cast_to(&DataType::Float64, None)?
                .into_array(num_iterations)?,
            None => ScalarValue::Float64(None).to_array_of_size(num_iterations)?,
        };
        let nodata_array = as_float64_array(&nodata_array)?.clone();

        // Band selection array, materialized once for the shape that carries one.
        // Array: a list column whose (Int64) values are the 1-based indices per
        // row.
        let (band_list_array, band_list_values) = match (self.band_arg, layout.band) {
            (BandArg::Array, Some(pos)) => {
                let target = DataType::List(Arc::new(Field::new("item", DataType::Int64, true)));
                let array = args[pos]
                    .clone()
                    .cast_to(&target, None)?
                    .into_array(num_iterations)?;
                let list = as_list_array(&array)?.clone();
                let values = as_int64_array(list.values())?.clone();
                (Some(list), Some(values))
            }
            _ => (None, None),
        };

        let mut x_builder = Int32Builder::new();
        let mut y_builder = Int32Builder::new();
        let mut rast_builder = RasterBuilder::new(num_iterations);
        let mut list_offsets = OffsetBufferBuilder::<i32>::new(num_iterations);
        let mut valid_list_items = NullBufferBuilder::new(num_iterations);
        // Reused per-row scratch for the resolved 1-based band indices, so the
        // Array shape does not allocate a fresh Vec every row.
        let mut band_scratch: Vec<i64> = Vec::new();

        let exec_arg_types = [arg_types[0].clone()];
        let exec_args = [args[0].clone()];
        let executor =
            RasterExecutor::new_with_num_iterations(&exec_arg_types, &exec_args, num_iterations);

        executor.execute_raster_void(|i, raster_opt| {
            let tile_width = (!tile_w_array.is_null(i)).then(|| tile_w_array.value(i));
            let tile_height = (!tile_h_array.is_null(i)).then(|| tile_h_array.value(i));
            let pad = (!pad_array.is_null(i)).then(|| pad_array.value(i));

            // Resolve the row's selected 1-based band indices into `band_scratch`.
            // `None` selects every band; a NULL band argument (a NULL list)
            // yields a NULL row.
            let bands: Option<&[i64]> = match self.band_arg {
                BandArg::All => None,
                BandArg::Array => {
                    let (Some(list), Some(values)) =
                        (band_list_array.as_ref(), band_list_values.as_ref())
                    else {
                        return sedona_internal_err!(
                            "RS_Tile: band list arrays missing for the Array overload"
                        );
                    };
                    if list.is_null(i) {
                        valid_list_items.append_null();
                        list_offsets.push_length(0);
                        return Ok(());
                    }
                    let offsets = list.value_offsets();
                    let (start, end) = (offsets[i] as usize, offsets[i + 1] as usize);
                    band_scratch.clear();
                    for j in start..end {
                        if values.is_null(j) {
                            return exec_err!("RS_Tile: band index must not be null");
                        }
                        band_scratch.push(values.value(j));
                    }
                    Some(band_scratch.as_slice())
                }
            };

            let (Some(raster), Some(tile_width), Some(tile_height), Some(pad_with_nodata)) =
                (raster_opt, tile_width, tile_height, pad)
            else {
                valid_list_items.append_null();
                list_offsets.push_length(0);
                return Ok(());
            };

            let params = TileParams {
                bands,
                pad_with_nodata,
                nodata: (!nodata_array.is_null(i)).then(|| nodata_array.value(i)),
            };

            let num_tiles = explode_raster_tiles(
                raster,
                tile_width,
                tile_height,
                &params,
                &mut rast_builder,
                &mut x_builder,
                &mut y_builder,
            )?;

            valid_list_items.append_non_null();
            list_offsets.push_length(num_tiles);
            Ok(())
        })?;

        let list_array = assemble_tile_list(
            x_builder,
            y_builder,
            rast_builder,
            list_offsets,
            valid_list_items,
        )?;

        RasterExecutor::finish_over(args, Arc::new(list_array))
    }
}

impl RsTile {
    /// The argument positions for this kernel's signature shape.
    fn layout(&self) -> ArgLayout {
        match self.band_arg {
            BandArg::All => ArgLayout {
                band: None,
                width: 1,
                height: 2,
                pad: (self.arg_count >= 4).then_some(3),
                nodata: (self.arg_count >= 5).then_some(4),
            },
            BandArg::Array => ArgLayout {
                band: Some(1),
                width: 2,
                height: 3,
                pad: (self.arg_count >= 5).then_some(4),
                nodata: (self.arg_count >= 6).then_some(5),
            },
        }
    }
}

/// The `List<Struct<x, y, tile>>` element struct fields, mirroring Spark's
/// `(x int, y int, tile raster)` generator output.
fn tile_struct_fields() -> Result<Fields> {
    let tile_field = RASTER
        .to_storage_field("tile", false)
        .map_err(|e| exec_datafusion_err!("RS_Tile: {e}"))?;
    Ok(Fields::from(vec![
        Field::new("x", DataType::Int32, false),
        Field::new("y", DataType::Int32, false),
        tile_field,
    ]))
}

/// The full `List<Struct<x, y, tile>>` return type.
fn tile_list_type() -> Result<DataType> {
    Ok(DataType::List(Arc::new(Field::new(
        "item",
        DataType::Struct(tile_struct_fields()?),
        true,
    ))))
}

/// Split one raster into a grid of tiles, appending each tile's grid position to
/// `x_builder`/`y_builder` and its raster to `rast_builder`. Returns the number
/// of tiles produced (`num_tile_x * num_tile_y`).
///
/// Tiles are emitted row-major, matching Spark: `num_tile_x =
/// ceil(width / tile_width)`, `num_tile_y = ceil(height / tile_height)`, and the
/// tile at grid position `(x, y)` covers source pixels
/// `[x * tile_width, ..)` x `[y * tile_height, ..)`.
fn explode_raster_tiles(
    raster: &RasterRefImpl<'_>,
    tile_width: i64,
    tile_height: i64,
    params: &TileParams<'_>,
    rast_builder: &mut RasterBuilder,
    x_builder: &mut Int32Builder,
    y_builder: &mut Int32Builder,
) -> Result<usize> {
    if tile_width < 1 || tile_height < 1 {
        return exec_err!(
            "RS_Tile: tile_width and tile_height must be >= 1, got {tile_width}x{tile_height}"
        );
    }
    if params.nodata.is_some() && !params.pad_with_nodata {
        return exec_err!("RS_Tile: nodata is only meaningful with pad_with_nodata = true");
    }

    let metadata = raster.metadata();
    let width = metadata.width();
    let height = metadata.height();
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
    let Some(num_tiles) = num_tile_x
        .checked_mul(num_tile_y)
        .filter(|&n| n <= i32::MAX as usize)
    else {
        return exec_err!(
            "RS_Tile: tile grid {num_tile_x}x{num_tile_y} exceeds the Int32 tile-count limit"
        );
    };

    let band_indices = resolve_band_indices(params.bands, raster.num_bands())?;
    let bands = raster.bands();

    for tile_y in 0..num_tile_y {
        for tile_x in 0..num_tile_x {
            let window = TileWindow::new(
                tile_x,
                tile_y,
                tile_w,
                tile_h,
                width,
                height,
                params.pad_with_nodata,
            );
            append_tile(raster, &bands, &band_indices, &window, params, rast_builder)?;
            x_builder.append_value(tile_x as i32);
            y_builder.append_value(tile_y as i32);
        }
    }

    Ok(num_tiles)
}

/// The pixel window one tile copies from the source, plus the tile's output
/// extent.
struct TileWindow {
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
    fn new(
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
    fn needs_padding(&self) -> bool {
        self.out_w > self.rect_w || self.out_h > self.rect_h
    }
}

/// Resolve which 1-based band indices to include, validating each against the
/// raster's band count. `None` (band-less overload) and an empty list both
/// select every band in order, matching Sedona Spark.
fn resolve_band_indices(bands: Option<&[i64]>, num_bands: usize) -> Result<Vec<usize>> {
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
fn append_tile(
    raster: &RasterRefImpl<'_>,
    bands: &Bands<'_>,
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
        append_tile_band(raster, bands, band_idx, window, params, rast_builder)?;
    }

    rast_builder
        .finish_raster()
        .map_err(|e| exec_datafusion_err!("RS_Tile: failed to finish raster: {e}"))?;
    Ok(())
}

/// Copy one band's tile window and append it to `rast_builder`.
fn append_tile_band(
    raster: &RasterRefImpl<'_>,
    bands: &Bands<'_>,
    band_idx: usize,
    window: &TileWindow,
    params: &TileParams<'_>,
    rast_builder: &mut RasterBuilder,
) -> Result<()> {
    let band = bands
        .band(band_idx)
        .map_err(|e| exec_datafusion_err!("RS_Tile: failed to get band {band_idx}: {e}"))?;
    // `band_idx` is 1-based; the `band_name` accessor is 0-based.
    let band_name = raster.band_name(band_idx - 1).map(|s| s.to_string());

    let band_metadata = band.metadata();
    let data_type = band_metadata.data_type()?;
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
    let width = raster.metadata().width() as usize;
    let height = raster.metadata().height() as usize;
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
            None => match band_metadata.nodata_value() {
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
        (
            None,
            band_metadata.nodata_value().map(|bytes| bytes.to_vec()),
        )
    };

    // One output allocation serves the whole band: every plane appends into it,
    // and the finished Vec moves into the Arrow array as a zero-copy view block
    // rather than being copied through the builder. The band data is appended as
    // one binary view (u32-limited), so size the buffer with checked arithmetic
    // and reject an oversize (e.g. huge padded) tile before allocating rather
    // than after filling the whole thing.
    let total_bytes = window
        .out_w
        .checked_mul(window.out_h)
        .and_then(|plane| plane.checked_mul(byte_size))
        .and_then(|plane_bytes| plane_bytes.checked_mul(n_planes))
        .ok_or_else(|| {
            exec_datafusion_err!(
                "RS_Tile: tile band extent {}x{} of {n_planes} planes overflows",
                window.out_w,
                window.out_h
            )
        })?;
    let band_data_len = u32::try_from(total_bytes).map_err(|_| {
        exec_datafusion_err!(
            "RS_Tile: tile band data of {total_bytes} bytes exceeds the binary-view limit"
        )
    })?;
    let mut tile_data = Vec::with_capacity(total_bytes);
    for plane in 0..n_planes {
        let plane_bytes = &source[plane * in_plane_bytes..(plane + 1) * in_plane_bytes];
        copy_tile_window(
            plane_bytes,
            width,
            window,
            byte_size,
            pad_fill.as_deref(),
            &mut tile_data,
        )?;
    }

    let mut out_shape = shape[..ndim - 2].to_vec();
    out_shape.push(window.out_h as i64);
    out_shape.push(window.out_w as i64);

    let dim_names_ref: Vec<&str> = dim_names.iter().map(String::as_str).collect();
    rast_builder
        .start_band_nd(
            band_name.as_deref(),
            &dim_names_ref,
            &out_shape,
            data_type,
            tile_nodata.as_deref(),
            None,
            None,
        )
        .map_err(|e| exec_datafusion_err!("RS_Tile: failed to start band: {e}"))?;

    // Move the band bytes into an Arrow buffer and append them as a view (a
    // refcount bump) instead of copying them through the builder.
    let buffer = Buffer::from(tile_data);
    rast_builder
        .append_band_data_buffer(&buffer, 0, band_data_len)
        .map_err(|e| exec_datafusion_err!("RS_Tile: failed to append band data: {e}"))?;
    rast_builder
        .finish_band()
        .map_err(|e| exec_datafusion_err!("RS_Tile: failed to finish band: {e}"))?;
    Ok(())
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

/// Assemble the per-row `x`/`y`/`tile` builders and list offsets into the
/// `List<Struct<x, y, tile>>` output.
fn assemble_tile_list(
    mut x_builder: Int32Builder,
    mut y_builder: Int32Builder,
    rast_builder: RasterBuilder,
    list_offsets: OffsetBufferBuilder<i32>,
    mut valid_list_items: NullBufferBuilder,
) -> Result<ListArray> {
    let x_array: ArrayRef = Arc::new(x_builder.finish());
    let y_array: ArrayRef = Arc::new(y_builder.finish());
    let tile_array: ArrayRef = Arc::new(
        rast_builder
            .finish()
            .map_err(|e| exec_datafusion_err!("RS_Tile: failed to build tiles: {e}"))?,
    );

    let fields = tile_struct_fields()?;
    let element_struct =
        StructArray::try_new(fields.clone(), vec![x_array, y_array, tile_array], None)
            .map_err(|e| exec_datafusion_err!("RS_Tile: failed to build tile struct: {e}"))?;

    let list_field = Arc::new(Field::new("item", DataType::Struct(fields), true));
    ListArray::try_new(
        list_field,
        list_offsets.finish(),
        Arc::new(element_struct),
        valid_list_items.finish(),
    )
    .map_err(|e| exec_datafusion_err!("RS_Tile: failed to build tile list: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    use datafusion_common::cast::{as_int32_array, as_list_array, as_struct_array};
    use datafusion_expr::ScalarUDF;
    use sedona_raster::array::RasterStructArray;
    use sedona_schema::datatypes::RASTER;
    use sedona_testing::raster_spec::{assert_rasters_equal, raster_array, RasterSpec};
    use sedona_testing::testers::ScalarUdfTester;

    /// The optional tiling parameters for the core-tiling helper tests.
    fn params(bands: Option<&[i64]>, pad_with_nodata: bool, nodata: Option<f64>) -> TileParams<'_> {
        TileParams {
            bands,
            pad_with_nodata,
            nodata,
        }
    }

    /// The registered RS_Tile UDF, used to exercise overload dispatch
    /// end to end (the matcher ladder plus the positional argument reading).
    fn udf() -> ScalarUDF {
        crate::register::default_function_set()
            .scalar_udf("rs_tile")
            .expect("rs_tile is registered")
            .clone()
            .into()
    }

    /// A scalar `ScalarValue::List` of 1-based band indices, as Spark's
    /// `array(...)` produces for the `bandIndices` overload.
    fn band_index_list(indices: &[i32]) -> ColumnarValue {
        let values: Vec<ScalarValue> = indices
            .iter()
            .map(|&i| ScalarValue::Int32(Some(i)))
            .collect();
        ColumnarValue::Scalar(ScalarValue::List(ScalarValue::new_list_nullable(
            &values,
            &DataType::Int32,
        )))
    }

    /// The `tile` column of a single-row (scalar) List result.
    fn scalar_tiles(result: &ColumnarValue) -> ArrayRef {
        let ColumnarValue::Scalar(ScalarValue::List(list)) = result else {
            panic!("expected a scalar List result, got {result:?}");
        };
        let element = as_struct_array(list.values()).unwrap();
        element.column(2).clone()
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

    /// Run the core tiling over a single source raster, returning the tile grid
    /// positions and the tiles as a raster array (so tiles can be asserted with
    /// the declarative `assert_rasters_equal`).
    fn explode(
        spec: &RasterSpec,
        tile_width: i64,
        tile_height: i64,
        tile_params: TileParams<'_>,
    ) -> (Vec<(i32, i32)>, ArrayRef) {
        let array = spec.build();
        let rasters = RasterStructArray::try_new(&array).unwrap();
        let raster = rasters.get(0).unwrap();

        let mut x_builder = Int32Builder::new();
        let mut y_builder = Int32Builder::new();
        let mut rast_builder = RasterBuilder::new(4);
        let num_tiles = explode_raster_tiles(
            &raster,
            tile_width,
            tile_height,
            &tile_params,
            &mut rast_builder,
            &mut x_builder,
            &mut y_builder,
        )
        .unwrap();

        let xs = x_builder.finish();
        let ys = y_builder.finish();
        let positions = (0..num_tiles).map(|i| (xs.value(i), ys.value(i))).collect();
        let tiles: ArrayRef = Arc::new(rast_builder.finish().unwrap());
        (positions, tiles)
    }

    #[test]
    fn return_type_is_list_of_tile_structs() {
        let kernel = RsTile {
            band_arg: BandArg::All,
            arg_count: 3,
        };
        let return_type = kernel
            .return_type(&[
                RASTER,
                SedonaType::Arrow(DataType::Int32),
                SedonaType::Arrow(DataType::Int32),
            ])
            .unwrap();
        let Some(SedonaType::Arrow(DataType::List(field))) = return_type else {
            panic!("expected List return type, got {return_type:?}");
        };
        // The list element carries the (x, y, tile) struct with the raster
        // extension field.
        let DataType::Struct(fields) = field.data_type() else {
            panic!("expected Struct list item");
        };
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0].name(), "x");
        assert_eq!(fields[1].name(), "y");
        assert_eq!(fields[2].name(), "tile");
        assert_eq!(
            SedonaType::from_storage_field(&fields[2]).unwrap(),
            RASTER,
            "tile field must be a raster"
        );
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
    fn udf_over_array_packages_list_and_nulls() {
        // End-to-end through the kernel: a two-row raster column (one raster,
        // one NULL) with scalar tile sizes. The output is a List row per input:
        // 6 tiles for row 0, a NULL list for row 1.
        let kernel = RsTile {
            band_arg: BandArg::All,
            arg_count: 3,
        };
        let raster_col = raster_array(vec![Some(source_5x3()), None]);
        let result = kernel
            .invoke_batch(
                &[
                    RASTER,
                    SedonaType::Arrow(DataType::Int32),
                    SedonaType::Arrow(DataType::Int32),
                ],
                &[
                    ColumnarValue::Array(Arc::new(raster_col)),
                    ColumnarValue::Scalar(ScalarValue::Int32(Some(2))),
                    ColumnarValue::Scalar(ScalarValue::Int32(Some(2))),
                ],
            )
            .unwrap();

        let ColumnarValue::Array(array) = result else {
            panic!("expected an array result");
        };
        let list = as_list_array(&array).unwrap();
        assert_eq!(list.len(), 2);
        assert!(!list.is_null(0));
        assert_eq!(list.value_length(0), 6, "row 0 should have 6 tiles");
        assert!(list.is_null(1), "row 1 (null raster) should be a null list");
        assert_eq!(list.value_length(1), 0);

        // The flattened element struct holds the 6 tiles of row 0.
        let element = as_struct_array(list.values()).unwrap();
        let xs = as_int32_array(element.column(0)).unwrap();
        let ys = as_int32_array(element.column(1)).unwrap();
        assert_eq!(
            (0..6)
                .map(|i| (xs.value(i), ys.value(i)))
                .collect::<Vec<_>>(),
            vec![(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
        );
        let tiles: ArrayRef = element.column(2).clone();
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

    /// A 2x2, three-band UInt8 raster (band 1 = 1..=4, band 2 = 10..=40,
    /// band 3 = 100..=103), so band selection and ordering are observable.
    fn three_band_2x2() -> RasterSpec {
        RasterSpec::d2(2, 2)
            .crs(None)
            .transform([0.0, 1.0, 0.0, 2.0, 0.0, -1.0])
            .band_values(&[1u8, 2, 3, 4])
            .band_values(&[10u8, 20, 30, 40])
            .band_values(&[100u8, 101, 102, 103])
    }

    #[test]
    fn overload_ladder_matches_by_argument_type() {
        // The two shapes are told apart by the argument after the raster: an
        // integer (width) for the all-bands shape, or an integer list
        // (bandIndices) for the band-subset shape. There is no single-`bandIndex`
        // (scalar integer) shape, so `(raster, int, int, int)` matches nothing.
        let int = SedonaType::Arrow(DataType::Int32);
        let boolean = SedonaType::Arrow(DataType::Boolean);
        let double = SedonaType::Arrow(DataType::Float64);
        let int_list = SedonaType::Arrow(DataType::List(Arc::new(Field::new(
            "item",
            DataType::Int32,
            true,
        ))));
        let matches = |band_arg: BandArg, arg_count: usize, args: &[SedonaType]| {
            RsTile {
                band_arg,
                arg_count,
            }
            .return_type(args)
            .unwrap()
            .is_some()
        };

        // (raster, int, int) is the all-bands shape.
        assert!(matches(
            BandArg::All,
            3,
            &[RASTER, int.clone(), int.clone()]
        ));

        // (raster, int, int, bool) is the all-bands + padWithNoData overload.
        let with_pad = [RASTER, int.clone(), int.clone(), boolean.clone()];
        assert!(matches(BandArg::All, 4, &with_pad));

        // A list in the band position selects the bandIndices overload only.
        let band_indices = [RASTER, int_list.clone(), int.clone(), int.clone()];
        assert!(matches(BandArg::Array, 4, &band_indices));
        assert!(!matches(BandArg::All, 4, &band_indices));

        // The single-`bandIndex` shape was dropped: an integer band selector
        // followed by an integer height, i.e. (raster, int, int, int), is not a
        // valid signature for either shape.
        let scalar_band = [RASTER, int.clone(), int.clone(), int.clone()];
        assert!(!matches(BandArg::All, 4, &scalar_band));
        assert!(!matches(BandArg::Array, 4, &scalar_band));

        // The fully-expanded bandIndices overload:
        // (raster, list, int, int, bool, double).
        let band_indices_full = [
            RASTER,
            int_list.clone(),
            int.clone(),
            int.clone(),
            boolean.clone(),
            double.clone(),
        ];
        assert!(matches(BandArg::Array, 6, &band_indices_full));
    }

    #[test]
    fn bandindices_array_overload_selects_and_orders_bands() {
        // RS_Tile(raster, bandIndices, width, height): the array overload
        // keeps exactly the listed bands, in the listed order (3 then 1).
        let tester = ScalarUdfTester::new(
            udf(),
            vec![
                RASTER,
                SedonaType::Arrow(DataType::List(Arc::new(Field::new(
                    "item",
                    DataType::Int32,
                    true,
                )))),
                SedonaType::Arrow(DataType::Int32),
                SedonaType::Arrow(DataType::Int32),
            ],
        );
        let result = tester
            .invoke(vec![
                ColumnarValue::Scalar(three_band_2x2().scalar()),
                band_index_list(&[3, 1]),
                ColumnarValue::Scalar(ScalarValue::Int32(Some(2))), // width
                ColumnarValue::Scalar(ScalarValue::Int32(Some(2))), // height
            ])
            .unwrap();
        assert_rasters_equal(
            &scalar_tiles(&result),
            &[Some(
                RasterSpec::d2(2, 2)
                    .crs(None)
                    .transform([0.0, 1.0, 0.0, 2.0, 0.0, -1.0])
                    .band_values(&[100u8, 101, 102, 103])
                    .band_values(&[1u8, 2, 3, 4]),
            )],
        );
    }

    #[test]
    fn nodataval_supplies_nodata_for_padding_when_band_has_none() {
        // RS_Tile(raster, width, height, padWithNoData, noDataVal): the
        // band has no nodata, so noDataVal (7) supplies the pad fill and becomes
        // the tile's recorded nodata. A 3x1 raster tiled 2x1 pads the edge tile.
        let source = RasterSpec::d2(3, 1)
            .crs(None)
            .transform([0.0, 1.0, 0.0, 1.0, 0.0, -1.0])
            .band_values(&[1u8, 2, 3]);
        let tester = ScalarUdfTester::new(
            udf(),
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Int32),
                SedonaType::Arrow(DataType::Int32),
                SedonaType::Arrow(DataType::Boolean),
                SedonaType::Arrow(DataType::Float64),
            ],
        );
        let result = tester
            .invoke(vec![
                ColumnarValue::Scalar(source.scalar()),
                ColumnarValue::Scalar(ScalarValue::Int32(Some(2))), // width
                ColumnarValue::Scalar(ScalarValue::Int32(Some(1))), // height
                ColumnarValue::Scalar(ScalarValue::Boolean(Some(true))), // padWithNoData
                ColumnarValue::Scalar(ScalarValue::Float64(Some(7.0))), // noDataVal
            ])
            .unwrap();
        assert_rasters_equal(
            &scalar_tiles(&result),
            &[
                // Interior tile: no padding, so noDataVal is not applied and it
                // keeps the source nodata (none).
                Some(tile(2, 1, 0.0, 1.0, &[1, 2])),
                // Edge tile padded to width 2 with the supplied nodata 7.
                Some(tile(2, 1, 2.0, 1.0, &[3, 7]).nodata(7u8)),
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

    #[test]
    fn null_band_indices_yields_null_row() {
        // A NULL bandIndices argument (Array overload) casts to a null list and
        // propagates to a NULL list row rather than erroring on the cast or the
        // offset lookup.
        let kernel = RsTile {
            band_arg: BandArg::Array,
            arg_count: 4,
        };
        let result = kernel
            .invoke_batch(
                &[
                    RASTER,
                    SedonaType::Arrow(DataType::Null),
                    SedonaType::Arrow(DataType::Int32),
                    SedonaType::Arrow(DataType::Int32),
                ],
                &[
                    ColumnarValue::Scalar(source_5x3().scalar()),
                    ColumnarValue::Scalar(ScalarValue::Null),
                    ColumnarValue::Scalar(ScalarValue::Int32(Some(2))),
                    ColumnarValue::Scalar(ScalarValue::Int32(Some(2))),
                ],
            )
            .unwrap();
        let ColumnarValue::Scalar(ScalarValue::List(list)) = result else {
            panic!("expected a scalar list result, got {result:?}");
        };
        assert!(list.is_null(0), "a NULL bandIndices yields a NULL list row");
    }

    #[test]
    fn nodata_without_padding_errors() {
        // noDataVal supplied with padWithNoData = false is a caller mistake.
        let err = explode_one_error(&source_5x3(), 2, 2, params(None, false, Some(5.0)));
        assert!(
            err.contains("only meaningful with pad_with_nodata"),
            "unexpected error: {err}"
        );
    }

    /// Run the core tiling and return the error string (helper for the error
    /// tests, which all expect `explode_raster_tiles` to fail).
    fn explode_one_error(
        spec: &RasterSpec,
        tile_width: i64,
        tile_height: i64,
        tile_params: TileParams<'_>,
    ) -> String {
        let array = spec.build();
        let rasters = RasterStructArray::try_new(&array).unwrap();
        let raster = rasters.get(0).unwrap();
        explode_raster_tiles(
            &raster,
            tile_width,
            tile_height,
            &tile_params,
            &mut RasterBuilder::new(1),
            &mut Int32Builder::new(),
            &mut Int32Builder::new(),
        )
        .unwrap_err()
        .to_string()
    }
}
