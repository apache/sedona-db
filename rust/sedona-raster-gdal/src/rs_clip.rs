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

//! RS_Clip UDF - Clip a raster to a geometry boundary
//!
//! Similar to PostGIS ST_Clip, this function clips a raster to the extent of a geometry.
//! Pixels outside the geometry are set to a nodata value: the explicit `no_data_value`
//! argument if given, otherwise the band's own nodata value, otherwise the minimum value
//! of the band's data type (so masked pixels stay distinguishable from real data).

use std::sync::Arc;

use arrow_array::{Array, ArrayRef};
use datafusion_common::cast::{as_boolean_array, as_float64_array, as_int32_array};
use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_common::exec_err;
use datafusion_common::{exec_datafusion_err, ScalarValue};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_common::sedona_internal_err;
use sedona_gdal::gdal::Gdal;
use sedona_gdal::mem::MemDatasetBuilder;
use sedona_gdal::raster::types::GdalDataType;
use sedona_proj::transform::with_global_proj_engine;

use arrow_schema::DataType;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::array::RasterRefImpl;
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::RasterRef;
use sedona_raster_functions::crs_utils::{crs_transform_wkb, resolve_crs};
use sedona_raster_functions::rs_ensure_loaded::{
    NEEDS_PIXELS_METADATA_KEY, RETURNS_BYTES_METADATA_KEY,
};
use sedona_raster_functions::RasterExecutor;
use sedona_schema::datatypes::{SedonaType, RASTER};
use sedona_schema::matchers::ArgMatcher;
use sedona_schema::raster::BandDataType;

use crate::gdal_common::with_gdal;
use crate::gdal_common::{nodata_bytes_to_f64, nodata_f64_to_bytes};
use crate::gdal_dataset_provider::configure_thread_local_options;

/// RS_Clip() scalar UDF implementation
///
/// Clips a raster to a geometry boundary.
///
/// Signatures:
/// - `RS_Clip(raster, band, geom)` — 3 args
/// - `RS_Clip(raster, band, geom, allTouched)` — 4 args
/// - `RS_Clip(raster, band, geom, allTouched, noDataValue)` — 5 args
/// - `RS_Clip(raster, band, geom, allTouched, noDataValue, crop)` — 6 args
/// - `RS_Clip(raster, band, geom, allTouched, noDataValue, crop, lenient)` — 7 args
pub fn rs_clip_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_clip",
        vec![
            Arc::new(RsClip { arg_count: 3 }), // (raster, band, geom)
            Arc::new(RsClip { arg_count: 4 }), // (raster, band, geom, allTouched)
            Arc::new(RsClip { arg_count: 5 }), // (raster, band, geom, allTouched, noDataValue)
            Arc::new(RsClip { arg_count: 6 }), // (raster, band, geom, allTouched, noDataValue, crop)
            Arc::new(RsClip { arg_count: 7 }), // (raster, band, geom, allTouched, noDataValue, crop, lenient)
        ],
        Volatility::Immutable,
    )
    // Reads band pixels (so the planner materializes OutDb rasters via
    // RS_EnsureLoaded first) and emits a fresh InDb raster (so its output is
    // already loaded and isn't wrapped again).
    .with_metadata(NEEDS_PIXELS_METADATA_KEY, "true")
    .with_metadata(RETURNS_BYTES_METADATA_KEY, "true")
}

/// Kernel implementation for RS_Clip
#[derive(Debug)]
struct RsClip {
    /// Number of arguments in the matched signature (3..=7)
    arg_count: usize,
}

impl SedonaScalarKernel for RsClip {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matchers = match self.arg_count {
            3 => vec![
                // RS_Clip(raster, band, geom)
                ArgMatcher::is_raster(),
                ArgMatcher::is_integer(),
                ArgMatcher::is_geometry_or_geography(),
            ],
            4 => vec![
                // RS_Clip(raster, band, geom, allTouched)
                ArgMatcher::is_raster(),
                ArgMatcher::is_integer(),
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_boolean(),
            ],
            5 => vec![
                // RS_Clip(raster, band, geom, allTouched, noDataValue)
                ArgMatcher::is_raster(),
                ArgMatcher::is_integer(),
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_boolean(),
                ArgMatcher::is_numeric(),
            ],
            6 => vec![
                // RS_Clip(raster, band, geom, allTouched, noDataValue, crop)
                ArgMatcher::is_raster(),
                ArgMatcher::is_integer(),
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_boolean(),
                ArgMatcher::is_numeric(),
                ArgMatcher::is_boolean(),
            ],
            7 => vec![
                // RS_Clip(raster, band, geom, allTouched, noDataValue, crop, lenient)
                ArgMatcher::is_raster(),
                ArgMatcher::is_integer(),
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_boolean(),
                ArgMatcher::is_numeric(),
                ArgMatcher::is_boolean(),
                ArgMatcher::is_boolean(),
            ],
            _ => {
                return sedona_internal_err!("RS_Clip: unexpected arg_count {}", self.arg_count);
            }
        };

        let matcher = ArgMatcher::new(matchers, RASTER);
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
        config_options: Option<&ConfigOptions>,
    ) -> Result<ColumnarValue> {
        let num_iterations = calc_num_iterations(args);

        // Band is always at index 1, geom is always at index 2.
        let geom_arg_idx: usize = 2;

        // Expand band to array
        let band_array = args[1]
            .clone()
            .cast_to(&arrow_schema::DataType::Int32, None)?
            .into_array(num_iterations)?;
        let band_array = as_int32_array(&band_array)?.clone();

        // allTouched at index 3 (when arg_count >= 4)
        let all_touched_array = if self.arg_count >= 4 {
            args[3]
                .clone()
                .cast_to(&arrow_schema::DataType::Boolean, None)?
                .into_array(num_iterations)?
        } else {
            ScalarValue::Boolean(Some(false)).to_array_of_size(num_iterations)?
        };
        let all_touched_array = as_boolean_array(&all_touched_array)?.clone();

        // noDataValue at index 4 (when arg_count >= 5)
        let nodata_array = if self.arg_count >= 5 {
            args[4]
                .clone()
                .cast_to(&arrow_schema::DataType::Float64, None)?
                .into_array(num_iterations)?
        } else {
            ScalarValue::Float64(None).to_array_of_size(num_iterations)?
        };
        let nodata_array = as_float64_array(&nodata_array)?.clone();

        // crop at index 5 (when arg_count >= 6), default true
        let crop_array = if self.arg_count >= 6 {
            args[5]
                .clone()
                .cast_to(&arrow_schema::DataType::Boolean, None)?
                .into_array(num_iterations)?
        } else {
            ScalarValue::Boolean(Some(true)).to_array_of_size(num_iterations)?
        };
        let crop_array = as_boolean_array(&crop_array)?.clone();

        // lenient at index 6 (when arg_count >= 7), default true
        let lenient_array = if self.arg_count >= 7 {
            args[6]
                .clone()
                .cast_to(&arrow_schema::DataType::Boolean, None)?
                .into_array(num_iterations)?
        } else {
            ScalarValue::Boolean(Some(true)).to_array_of_size(num_iterations)?
        };
        let lenient_array = as_boolean_array(&lenient_array)?.clone();

        let mut band_iter = band_array.iter();
        let mut all_touched_iter = all_touched_array.iter();
        let mut nodata_iter = nodata_array.iter();
        let mut crop_iter = crop_array.iter();
        let mut lenient_iter = lenient_array.iter();

        // Build output rasters
        let mut builder = RasterBuilder::new(num_iterations);

        let exec_arg_types = vec![arg_types[0].clone(), arg_types[geom_arg_idx].clone()];
        let exec_args = vec![args[0].clone(), args[geom_arg_idx].clone()];
        let executor =
            RasterExecutor::new_with_num_iterations(&exec_arg_types, &exec_args, num_iterations);

        with_gdal(|gdal| {
            configure_thread_local_options(gdal, config_options)?;
            with_global_proj_engine(|engine| {
                executor.execute_raster_wkb_crs_void(|raster_opt, wkb_opt, geom_crs| {
                let band = band_iter.next().unwrap_or(Some(0)).unwrap_or(0);
                let all_touched = all_touched_iter
                    .next()
                    .unwrap_or(Some(false))
                    .unwrap_or(false);
                let nodata_value = nodata_iter.next().unwrap_or(None);
                let crop = crop_iter.next().unwrap_or(Some(true)).unwrap_or(true);
                let lenient = lenient_iter.next().unwrap_or(Some(true)).unwrap_or(true);

                let (raster, geom_wkb) = match (raster_opt, wkb_opt) {
                    (Some(r), Some(w)) => (r, w),
                    _ => {
                        builder.append_null()?;
                        return Ok(());
                    }
                };

                let raster_crs = resolve_crs(raster.crs())?;
                let geom_wkb = match (geom_crs, raster_crs.as_deref()) {
                    (Some(geom_crs), Some(raster_crs)) => {
                        crs_transform_wkb(geom_wkb, geom_crs, raster_crs, engine)?
                    }
                    (None, None) => geom_wkb.to_vec(),
                    (Some(_), None) => {
                        return exec_err!(
                            "Cannot operate on geometry and raster: raster has no CRS but geometry does"
                        )
                    }
                    (None, Some(_)) => {
                        return exec_err!(
                            "Cannot operate on geometry and raster: geometry has no CRS but raster does"
                        )
                    }
                };

                // Band 0 means "all bands" (handled in clip_raster, which also
                // range-checks the upper bound). A negative band is an error,
                // not a silent clamp to band 1.
                if band < 0 {
                    return exec_err!(
                        "RS_Clip: band must be >= 0 (0 = all bands), got {band}"
                    );
                }
                let band_index = band as usize;
                match clip_raster(
                    gdal,
                    raster,
                    &geom_wkb,
                    band_index,
                    nodata_value,
                    all_touched,
                    crop,
                ) {
                    Ok(Some(clipped_data)) => {
                        build_clipped_raster(&mut builder, raster, &clipped_data)?
                    }
                    Ok(None) => {
                        // `lenient` governs only the no-intersection case: the
                        // raster and geometry don't overlap, so there's nothing
                        // to clip. Lenient yields NULL; strict errors.
                        if lenient {
                            builder.append_null()?;
                        } else {
                            return exec_err!("RS_Clip: raster and geometry do not intersect");
                        }
                    }
                    // A genuine failure (malformed WKB, GDAL error, …) always
                    // propagates — it is not the no-intersection case `lenient`
                    // is meant to soften.
                    Err(e) => return Err(e),
                }

                Ok(())
            })
            })?;

            // Decide array-vs-scalar over *all* args, not just the raster/geom
            // the executor was given: a per-row band/option column over a scalar
            // raster+geom must still yield an N-row array. (`executor.finish`
            // only inspects its two exec args and would collapse to row 0.)
            let out: ArrayRef = Arc::new(builder.finish()?);
            if args.iter().any(|a| matches!(a, ColumnarValue::Array(_))) {
                Ok(ColumnarValue::Array(out))
            } else {
                Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(&out, 0)?))
            }
        })
    }
}

/// One clipped band: masked/cropped bytes plus the N-D layout needed to rebuild
/// it. The clip is a 2-D `(y, x)` operation broadcast across every non-spatial
/// plane, so `dim_names` are unchanged from the source and only the trailing
/// `(y, x)` extent of `shape` shrinks when cropping.
struct ClippedBand {
    /// Masked/cropped bytes, plane-major in the band's dim order.
    data: Vec<u8>,
    /// Visible dim names, unchanged from the source band (e.g. `["time","y","x"]`).
    dim_names: Vec<String>,
    /// Output shape: leading non-spatial dims unchanged, trailing `(y, x)` clipped.
    shape: Vec<i64>,
    data_type: BandDataType,
    /// nodata sentinel bytes written for masked-out pixels.
    nodata: Vec<u8>,
}

/// Data for a clipped raster
struct ClippedRasterData {
    /// One entry per processed band.
    bands: Vec<ClippedBand>,
    /// Crop window in pixel coordinates (col_off, row_off, width, height).
    /// `None` means the full original raster extent was kept (crop=false).
    crop_window: Option<CropWindow>,
}

/// A rectangular crop window in pixel coordinates.
#[derive(Debug, Clone, Copy)]
struct CropWindow {
    col_off: usize,
    row_off: usize,
    width: usize,
    height: usize,
}

/// Clip a raster to a geometry.
///
/// Returns `Ok(None)` when the geometry does not intersect the raster extent
/// (caller decides how to handle based on `lenient`).
fn clip_raster(
    gdal: &Gdal,
    raster: &RasterRefImpl<'_>,
    geom_wkb: &[u8],
    band_num: usize,
    custom_nodata: Option<f64>,
    all_touched: bool,
    crop: bool,
) -> Result<Option<ClippedRasterData>> {
    let metadata = raster.metadata();
    let bands = raster.bands();
    let width = metadata.width() as usize;
    let height = metadata.height() as usize;

    // Parse geometry from WKB
    let geometry = gdal
        .geometry_from_wkb(geom_wkb)
        .map_err(|e| exec_datafusion_err!("Failed to parse geometry from WKB: {}", e))?;

    // Create a mask raster (same dimensions as input)
    let mask_dataset = MemDatasetBuilder::create(gdal, width, height, 1, GdalDataType::UInt8)
        .map_err(|e| exec_datafusion_err!("Failed to create mask dataset: {}", e))?;

    // Set the same geotransform as the input raster
    let geotransform = [
        metadata.upper_left_x(),
        metadata.scale_x(),
        metadata.skew_x(),
        metadata.upper_left_y(),
        metadata.skew_y(),
        metadata.scale_y(),
    ];
    mask_dataset
        .set_geo_transform(&geotransform)
        .map_err(|e| exec_datafusion_err!("Failed to set geotransform: {}", e))?;

    // GDAL's MEM driver zero-fills owned band buffers at creation, so the mask
    // already reads 0 (outside) everywhere; rasterize_affine burns 1 inside the
    // geometry. No explicit zero-init write needed.
    gdal.rasterize_affine(
        &mask_dataset,
        &[1], // band 1
        &[geometry],
        &[1.0], // burn value = 1 (inside)
        all_touched,
    )
    .map_err(|e| exec_datafusion_err!("Failed to rasterize geometry: {}", e))?;

    // Read the mask
    let mask_band = mask_dataset
        .rasterband(1)
        .map_err(|e| exec_datafusion_err!("Failed to get mask band: {}", e))?;
    let mask_buffer = mask_band
        .read_as::<u8>((0, 0), (width, height), (width, height), None)
        .map_err(|e| exec_datafusion_err!("Failed to read mask: {}", e))?;
    let mask = mask_buffer.data();

    // Check if there are any non-zero pixels in the mask (i.e. geometry intersects raster)
    let has_intersection = mask.iter().any(|&v| v != 0);
    if !has_intersection {
        return Ok(None);
    }

    // Compute crop window if crop=true
    let crop_window = if crop {
        compute_crop_window(mask, width, height)
    } else {
        None
    };

    // Determine which bands to process
    let band_indices: Vec<usize> = if band_num == 0 {
        (1..=bands.len()).collect()
    } else {
        if band_num > bands.len() {
            return exec_err!("Band {} is out of range (1-{})", band_num, bands.len());
        }
        vec![band_num]
    };

    // Process each band. The clip is a 2-D (y, x) operation; for an N-D band
    // (extra leading dims such as time) the same mask and crop window are
    // broadcast across every non-spatial plane.
    let mut clipped_bands = Vec::with_capacity(band_indices.len());

    for &band_idx in &band_indices {
        let band = bands
            .band(band_idx)
            .map_err(|e| exec_datafusion_err!("Failed to get band {}: {}", band_idx, e))?;

        let band_metadata = band.metadata();
        let data_type = band_metadata.data_type()?;

        // The trailing two axes are the spatial (y, x) plane; anything before
        // them is a stack of planes the 2-D clip is broadcast over.
        let shape = band.shape().to_vec();
        let dim_names: Vec<String> = band.dim_names().iter().map(|s| s.to_string()).collect();
        let ndim = shape.len();
        if ndim < 2 {
            return exec_err!(
                "RS_Clip: band {} has {} dimension(s); a 2-D (y, x) plane is required",
                band_idx,
                ndim
            );
        }
        let (plane_h, plane_w) = (shape[ndim - 2] as usize, shape[ndim - 1] as usize);
        if plane_w != width || plane_h != height {
            return exec_err!(
                "RS_Clip: band {} spatial extent {}x{} does not match the raster {}x{}",
                band_idx,
                plane_w,
                plane_h,
                width,
                height
            );
        }
        let n_planes: usize = shape[..ndim - 2].iter().map(|&d| d as usize).product();

        let original_data = band
            .nd_buffer()
            .map_err(|e| exec_datafusion_err!("RS_Clip: failed to read band {}: {}", band_idx, e))?
            .as_contiguous()
            .map_err(|e| {
                exec_datafusion_err!("RS_Clip: band {} is not contiguous: {}", band_idx, e)
            })?
            .to_vec();

        // nodata precedence: the explicit argument, then the band's own nodata,
        // then the band data type's minimum value — never a silent 0.0, which
        // would be indistinguishable from real zero-valued pixels.
        let nodata = custom_nodata
            .or_else(|| nodata_bytes_to_f64(band_metadata.nodata_value(), &data_type))
            .unwrap_or_else(|| band_data_type_min(&data_type));

        let byte_size = data_type.byte_size();
        let in_plane_bytes = width * height * byte_size;
        if original_data.len() != n_planes * in_plane_bytes {
            return exec_err!(
                "RS_Clip: band {} byte length {} does not match {} planes of {}x{}",
                band_idx,
                original_data.len(),
                n_planes,
                width,
                height
            );
        }

        // Apply the (shared) mask/crop to each plane, then concatenate.
        let mut clipped_data = Vec::new();
        for plane in 0..n_planes {
            let plane_bytes = &original_data[plane * in_plane_bytes..(plane + 1) * in_plane_bytes];
            let clipped_plane = if let Some(cw) = crop_window {
                apply_mask_and_crop(plane_bytes, mask, width, &data_type, nodata, &cw)?
            } else {
                apply_mask_to_band(plane_bytes, mask, width, height, &data_type, nodata)?
            };
            clipped_data.extend_from_slice(&clipped_plane);
        }

        // Output shape: leading dims unchanged; trailing (y, x) becomes the crop
        // window when cropping, else the original plane extent.
        let (out_h, out_w) = match crop_window {
            Some(cw) => (cw.height as i64, cw.width as i64),
            None => (height as i64, width as i64),
        };
        let mut out_shape = shape[..ndim - 2].to_vec();
        out_shape.push(out_h);
        out_shape.push(out_w);

        clipped_bands.push(ClippedBand {
            data: clipped_data,
            dim_names,
            shape: out_shape,
            data_type,
            nodata: nodata_f64_to_bytes(nodata, &data_type),
        });
    }

    Ok(Some(ClippedRasterData {
        bands: clipped_bands,
        crop_window,
    }))
}

/// Compute the minimal bounding pixel window that contains all non-zero mask pixels.
fn compute_crop_window(mask: &[u8], width: usize, height: usize) -> Option<CropWindow> {
    let mut min_col = width;
    let mut max_col = 0usize;
    let mut min_row = height;
    let mut max_row = 0usize;

    for row in 0..height {
        for col in 0..width {
            if mask[row * width + col] != 0 {
                min_col = min_col.min(col);
                max_col = max_col.max(col);
                min_row = min_row.min(row);
                max_row = max_row.max(row);
            }
        }
    }

    if min_col > max_col || min_row > max_row {
        return None; // no non-zero pixels (shouldn't happen if caller checked)
    }

    Some(CropWindow {
        col_off: min_col,
        row_off: min_row,
        width: max_col - min_col + 1,
        height: max_row - min_row + 1,
    })
}

/// Apply mask to band data (no cropping — preserves original dimensions).
fn apply_mask_to_band(
    original_data: &[u8],
    mask: &[u8],
    width: usize,
    height: usize,
    data_type: &BandDataType,
    nodata: f64,
) -> Result<Vec<u8>> {
    let byte_size = data_type.byte_size();
    let nodata_bytes = nodata_f64_to_bytes(nodata, data_type);
    let mut result = original_data.to_vec();

    for (pixel_idx, &mask_val) in mask.iter().enumerate().take(width * height) {
        if mask_val == 0 {
            // Pixel is outside geometry - set to nodata
            let byte_offset = pixel_idx * byte_size;
            result[byte_offset..byte_offset + byte_size].copy_from_slice(&nodata_bytes);
        }
    }

    Ok(result)
}

/// Apply mask AND crop to the given crop window in one pass.
fn apply_mask_and_crop(
    original_data: &[u8],
    mask: &[u8],
    full_width: usize,
    data_type: &BandDataType,
    nodata: f64,
    cw: &CropWindow,
) -> Result<Vec<u8>> {
    let byte_size = data_type.byte_size();
    let crop_pixel_count = cw.width * cw.height;
    let nodata_bytes = nodata_f64_to_bytes(nodata, data_type);
    let mut result = vec![0u8; crop_pixel_count * byte_size];

    for row in 0..cw.height {
        let src_row = cw.row_off + row;
        for col in 0..cw.width {
            let src_col = cw.col_off + col;
            let src_pixel_idx = src_row * full_width + src_col;
            let dst_pixel_idx = row * cw.width + col;
            let src_byte_offset = src_pixel_idx * byte_size;
            let dst_byte_offset = dst_pixel_idx * byte_size;

            if mask[src_pixel_idx] != 0 {
                // Inside geometry — copy original pixel
                result[dst_byte_offset..dst_byte_offset + byte_size]
                    .copy_from_slice(&original_data[src_byte_offset..src_byte_offset + byte_size]);
            } else {
                // Outside geometry — write nodata
                result[dst_byte_offset..dst_byte_offset + byte_size].copy_from_slice(&nodata_bytes);
            }
        }
    }

    Ok(result)
}

/// Build the clipped raster via the N-D builder. A 2-D raster is just the
/// `["y", "x"]` case; an N-D raster keeps its non-spatial dims and only its
/// `(y, x)` extent changes when cropping.
fn build_clipped_raster(
    builder: &mut RasterBuilder,
    original_raster: &RasterRefImpl<'_>,
    clipped_data: &ClippedRasterData,
) -> Result<()> {
    // Geotransform is 2-D and shared across all planes. A crop shifts the
    // upper-left by the pixel offset; scale/skew are unchanged.
    // Layout: [upper_left_x, scale_x, skew_x, upper_left_y, skew_y, scale_y].
    let src = original_raster.transform();
    let transform: [f64; 6] = if let Some(cw) = clipped_data.crop_window {
        let new_ulx = src[0] + cw.col_off as f64 * src[1] + cw.row_off as f64 * src[2];
        let new_uly = src[3] + cw.row_off as f64 * src[5] + cw.col_off as f64 * src[4];
        [new_ulx, src[1], src[2], new_uly, src[4], src[5]]
    } else {
        [src[0], src[1], src[2], src[3], src[4], src[5]]
    };

    // Spatial extent after the clip. `spatial_dims`/`spatial_shape` are kept in
    // the raster's own axis order (X-first, as the readers emit), so map each
    // spatial dim to its clipped size by name rather than assuming an order.
    let spatial_dims = original_raster.spatial_dims();
    let spatial_shape: Vec<i64> = match clipped_data.crop_window {
        None => original_raster.spatial_shape().to_vec(),
        Some(cw) => {
            let x_dim = original_raster.x_dim();
            spatial_dims
                .iter()
                .map(|&d| {
                    if d == x_dim {
                        cw.width as i64
                    } else {
                        cw.height as i64
                    }
                })
                .collect()
        }
    };

    builder
        .start_raster_nd(
            &transform,
            &spatial_dims,
            &spatial_shape,
            original_raster.crs(),
        )
        .map_err(|e| exec_datafusion_err!("Failed to start raster: {}", e))?;

    for band in &clipped_data.bands {
        let dim_names: Vec<&str> = band.dim_names.iter().map(String::as_str).collect();
        builder
            .start_band_nd(
                None,
                &dim_names,
                &band.shape,
                band.data_type,
                Some(&band.nodata),
                None,
                None,
            )
            .map_err(|e| exec_datafusion_err!("Failed to start band: {}", e))?;
        builder.band_data_writer().append_value(&band.data);
        builder
            .finish_band()
            .map_err(|e| exec_datafusion_err!("Failed to finish band: {}", e))?;
    }

    builder
        .finish_raster()
        .map_err(|e| exec_datafusion_err!("Failed to finish raster: {}", e))?;

    Ok(())
}

/// The minimum representable value of a band data type, as `f64` — the default
/// nodata sentinel when neither an explicit value nor the band's own nodata is
/// available, so masked-out pixels stay distinguishable from real data.
fn band_data_type_min(data_type: &BandDataType) -> f64 {
    match data_type {
        BandDataType::UInt8
        | BandDataType::UInt16
        | BandDataType::UInt32
        | BandDataType::UInt64 => 0.0,
        BandDataType::Int8 => i8::MIN as f64,
        BandDataType::Int16 => i16::MIN as f64,
        BandDataType::Int32 => i32::MIN as f64,
        BandDataType::Int64 => i64::MIN as f64,
        BandDataType::Float32 => f32::MIN as f64,
        BandDataType::Float64 => f64::MIN,
    }
}

fn calc_num_iterations(args: &[ColumnarValue]) -> usize {
    for arg in args {
        if let ColumnarValue::Array(array) = arg {
            return array.len();
        }
    }
    1
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::StructArray;
    use sedona_expr::scalar_udf::SedonaScalarKernel;
    use sedona_raster::array::RasterStructArray;
    use sedona_schema::crs::deserialize_crs;
    use sedona_schema::datatypes::Edges;
    use sedona_testing::raster_spec::RasterSpec;

    fn wkb_from_wkt(gdal: &sedona_gdal::gdal::Gdal, wkt: &str) -> Result<Vec<u8>> {
        let geometry = gdal.geometry_from_wkt(wkt).unwrap();
        geometry.wkb().map_err(|e| exec_datafusion_err!("{e}"))
    }

    /// A 4×2 EPSG:4326 raster, origin (0, 2), 1×1 north-up pixels — world extent
    /// x ∈ [0, 4], y ∈ [0, 2]. One UInt8 band with values 1..=8 (row-major).
    fn test_raster_array() -> StructArray {
        RasterSpec::d2(4, 2)
            .band_values(&[1u8, 2, 3, 4, 5, 6, 7, 8])
            .crs(Some("EPSG:4326"))
            .transform([0.0, 1.0, 0.0, 2.0, 0.0, -1.0])
            .build()
    }

    #[test]
    fn test_rs_clip_basic() {
        // crop=false: the output keeps the original extent and band byte length;
        // pixels outside the clip polygon are set to nodata.
        let array = test_raster_array();
        with_gdal(|gdal| {
            let rasters = RasterStructArray::try_new(&array).unwrap();
            let raster = rasters.get(0).unwrap();

            // Left half of the raster: x ∈ [0, 2], y ∈ [0, 2].
            let geom_wkb = wkb_from_wkt(gdal, "POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))")?;
            let clipped = clip_raster(gdal, &raster, &geom_wkb, 0, None, false, false)?
                .expect("Should have intersection");

            let original_len = raster
                .bands()
                .band(1)
                .unwrap()
                .nd_buffer()
                .unwrap()
                .as_contiguous()
                .unwrap()
                .len();
            assert!(!clipped.bands.is_empty(), "Should have at least one band");
            assert_eq!(
                clipped.bands[0].data.len(),
                original_len,
                "Clipped band should have same size as original when crop=false"
            );
            assert!(
                clipped.crop_window.is_none(),
                "crop_window should be None when crop=false"
            );
            Ok::<_, datafusion_common::DataFusionError>(())
        })
        .unwrap();
    }

    #[test]
    fn test_rs_clip_crop() {
        // crop=true: the output shrinks to the geometry's bbox window.
        let array = test_raster_array();
        with_gdal(|gdal| {
            let rasters = RasterStructArray::try_new(&array).unwrap();
            let raster = rasters.get(0).unwrap();
            let metadata = raster.metadata();

            // Top-left quadrant: x ∈ [0, 2], y ∈ [1, 2] — covers pixel centers
            // (0.5, 1.5) and (1.5, 1.5), i.e. a 2×1 window.
            let geom_wkb = wkb_from_wkt(gdal, "POLYGON((0 1, 2 1, 2 2, 0 2, 0 1))")?;
            let clipped = clip_raster(gdal, &raster, &geom_wkb, 0, None, false, true)?
                .expect("Should have intersection");
            let cw = clipped
                .crop_window
                .expect("crop_window should be set when crop=true");

            let byte_size = clipped.bands[0].data_type.byte_size();
            assert_eq!(
                clipped.bands[0].data.len(),
                cw.width * cw.height * byte_size,
                "Cropped band data size should match crop window"
            );
            assert!(
                (cw.width as i64) < metadata.width(),
                "Cropped width should be smaller"
            );
            assert!(
                (cw.height as i64) < metadata.height(),
                "Cropped height should be smaller"
            );
            Ok::<_, datafusion_common::DataFusionError>(())
        })
        .unwrap();
    }

    #[test]
    fn test_rs_clip_no_intersection() {
        let array = test_raster_array();
        with_gdal(|gdal| {
            let rasters = RasterStructArray::try_new(&array).unwrap();
            let raster = rasters.get(0).unwrap();
            // Far outside the raster's [0,4]×[0,2] extent.
            let geom_wkb = wkb_from_wkt(
                gdal,
                "POLYGON((100 100, 101 100, 101 101, 100 101, 100 100))",
            )?;
            let result = clip_raster(gdal, &raster, &geom_wkb, 0, None, false, true)?;
            assert!(result.is_none(), "Should return None for no intersection");
            Ok::<_, datafusion_common::DataFusionError>(())
        })
        .unwrap();
    }

    #[test]
    fn test_rs_clip_crs_mismatch() {
        // A geometry given in EPSG:3857 must be reprojected to the raster's
        // EPSG:4326 before clipping, yielding the same result as the equivalent
        // EPSG:4326 geometry.
        let array = test_raster_array();

        // Build the EPSG:3857 geometry with the same PROJ engine the UDF uses,
        // so the test is robust to axis-order / normalization across builds.
        let crs_4326 = deserialize_crs("EPSG:4326").unwrap().unwrap();
        let crs_3857 = deserialize_crs("EPSG:3857").unwrap().unwrap();

        let geom_wkb_4326 =
            with_gdal(|gdal| wkb_from_wkt(gdal, "POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))")).unwrap();
        let geom_wkb_3857 = with_global_proj_engine(|engine| {
            crs_transform_wkb(&geom_wkb_4326, crs_4326.as_ref(), crs_3857.as_ref(), engine)
        })
        .unwrap();

        // 3-arg variant: RS_Clip(raster, band, geom).
        let kernel = RsClip { arg_count: 3 };
        let raster_scalar = ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(array)));
        let band_type = SedonaType::Arrow(DataType::Int32);
        let band_val = ColumnarValue::Scalar(ScalarValue::Int32(Some(1)));

        let clip_band1 = |geom_type: SedonaType, geom_wkb: Vec<u8>| -> Vec<u8> {
            let result = kernel
                .invoke_batch(
                    &[RASTER, band_type.clone(), geom_type],
                    &[
                        raster_scalar.clone(),
                        band_val.clone(),
                        ColumnarValue::Scalar(ScalarValue::Binary(Some(geom_wkb))),
                    ],
                )
                .unwrap();
            match result {
                ColumnarValue::Scalar(ScalarValue::Struct(struct_array)) => {
                    let array = RasterStructArray::try_new(struct_array.as_ref()).unwrap();
                    array
                        .get(0)
                        .unwrap()
                        .bands()
                        .band(1)
                        .unwrap()
                        .nd_buffer()
                        .unwrap()
                        .as_contiguous()
                        .unwrap()
                        .to_vec()
                }
                _ => panic!("Expected raster scalar result"),
            }
        };

        let band_data_4326 = clip_band1(
            SedonaType::Wkb(Edges::Planar, Some(crs_4326)),
            geom_wkb_4326,
        );
        let band_data_3857 = clip_band1(
            SedonaType::Wkb(Edges::Planar, Some(crs_3857)),
            geom_wkb_3857,
        );

        assert_eq!(band_data_4326, band_data_3857);
    }

    #[test]
    fn test_rs_clip_nd_broadcasts_across_planes() {
        // A [time=2, y=2, x=4] raster: clipping with a 2-D polygon crops the
        // (y, x) plane and broadcasts the same mask across both time planes,
        // preserving the time dimension. Values 1..=16 (C order): time 0 is
        // rows [1,2,3,4] / [5,6,7,8], time 1 is [9..12] / [13..16].
        let array = RasterSpec::nd(&["time", "y", "x"], &[2, 2, 4])
            .band_values(&[1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
            .crs(Some("EPSG:4326"))
            .transform([0.0, 1.0, 0.0, 2.0, 0.0, -1.0])
            .build();

        let crs_4326 = deserialize_crs("EPSG:4326").unwrap().unwrap();
        let geom_wkb =
            with_gdal(|gdal| wkb_from_wkt(gdal, "POLYGON((0 1, 2 1, 2 2, 0 2, 0 1))")).unwrap();

        // RS_Clip(raster, band, geom, allTouched, noData, crop=true).
        let kernel = RsClip { arg_count: 6 };
        let result = kernel
            .invoke_batch(
                &[
                    RASTER,
                    SedonaType::Arrow(DataType::Int32),
                    SedonaType::Wkb(Edges::Planar, Some(crs_4326)),
                    SedonaType::Arrow(DataType::Boolean),
                    SedonaType::Arrow(DataType::Float64),
                    SedonaType::Arrow(DataType::Boolean),
                ],
                &[
                    ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(array))),
                    ColumnarValue::Scalar(ScalarValue::Int32(Some(1))),
                    ColumnarValue::Scalar(ScalarValue::Binary(Some(geom_wkb))),
                    ColumnarValue::Scalar(ScalarValue::Boolean(Some(false))),
                    ColumnarValue::Scalar(ScalarValue::Float64(Some(0.0))),
                    ColumnarValue::Scalar(ScalarValue::Boolean(Some(true))),
                ],
            )
            .unwrap();

        let out = match result {
            ColumnarValue::Scalar(ScalarValue::Struct(s)) => s,
            _ => panic!("Expected raster scalar result"),
        };
        let rasters = RasterStructArray::try_new(out.as_ref()).unwrap();
        let raster = rasters.get(0).unwrap();
        let band = raster.bands().band(1).unwrap();

        // The time dim is preserved; (y, x) is cropped to the 1×2 mask window.
        assert_eq!(band.dim_names(), vec!["time", "y", "x"]);
        assert_eq!(band.shape(), &[2, 1, 2]);

        // The crop window is cols 0-1 of row 0, applied to both planes:
        // time 0 -> [1, 2], time 1 -> [9, 10].
        let bytes = band.nd_buffer().unwrap().as_contiguous().unwrap();
        assert_eq!(bytes, &[1u8, 2, 9, 10]);
    }

    #[test]
    fn test_band_data_type_min() {
        // Unsigned types floor at 0; signed/float at their most-negative value.
        assert_eq!(band_data_type_min(&BandDataType::UInt8), 0.0);
        assert_eq!(band_data_type_min(&BandDataType::UInt64), 0.0);
        assert_eq!(band_data_type_min(&BandDataType::Int8), -128.0);
        assert_eq!(band_data_type_min(&BandDataType::Int16), i16::MIN as f64);
        assert_eq!(band_data_type_min(&BandDataType::Int32), i32::MIN as f64);
        assert_eq!(band_data_type_min(&BandDataType::Float32), f32::MIN as f64);
        assert_eq!(band_data_type_min(&BandDataType::Float64), f64::MIN);
    }

    /// A 2×1, two-band EPSG:4326 raster (band 1 = [1,2], band 2 = [10,20]) — the
    /// two bands differ so a per-band clip is observably distinct.
    fn two_band_raster() -> StructArray {
        RasterSpec::d2(2, 1)
            .band_values(&[1u8, 2])
            .band_values(&[10u8, 20])
            .crs(Some("EPSG:4326"))
            .transform([0.0, 1.0, 0.0, 1.0, 0.0, -1.0])
            .build()
    }

    #[test]
    fn scalar_raster_geom_with_band_column_yields_all_rows() {
        // Regression: a constant raster+geom with a per-row band column must
        // produce an N-row array, not collapse to row 0. (The executor only sees
        // [raster, geom], so output packaging must consider all args.)
        let crs_4326 = deserialize_crs("EPSG:4326").unwrap().unwrap();
        let geom_wkb =
            with_gdal(|gdal| wkb_from_wkt(gdal, "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))")).unwrap();

        let kernel = RsClip { arg_count: 3 };
        let result = kernel
            .invoke_batch(
                &[
                    RASTER,
                    SedonaType::Arrow(DataType::Int32),
                    SedonaType::Wkb(Edges::Planar, Some(crs_4326)),
                ],
                &[
                    ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(two_band_raster()))),
                    ColumnarValue::Array(Arc::new(arrow_array::Int32Array::from(vec![1, 2]))),
                    ColumnarValue::Scalar(ScalarValue::Binary(Some(geom_wkb))),
                ],
            )
            .unwrap();

        // Must be a 2-row array (not a broadcast scalar), with row 0 clipping
        // band 1 and row 1 clipping band 2 — distinct outputs.
        let arr = match result {
            ColumnarValue::Array(a) => a,
            ColumnarValue::Scalar(_) => panic!("expected an array; the batch collapsed to row 0"),
        };
        let rasters =
            RasterStructArray::try_new(arr.as_any().downcast_ref::<StructArray>().unwrap())
                .unwrap();
        assert_eq!(rasters.len(), 2);
        let band_data = |row: usize| {
            rasters
                .get(row)
                .unwrap()
                .bands()
                .band(1)
                .unwrap()
                .nd_buffer()
                .unwrap()
                .as_contiguous()
                .unwrap()
                .to_vec()
        };
        // Row 0 clipped band 1 (values 1,2); row 1 clipped band 2 (values 10,20).
        assert_ne!(band_data(0), band_data(1));
    }

    #[test]
    fn band_zero_clips_all_bands() {
        // Band 0 means "all bands" — it must reach clip_raster as 0, not be
        // clamped to 1.
        let crs_4326 = deserialize_crs("EPSG:4326").unwrap().unwrap();
        let geom_wkb =
            with_gdal(|gdal| wkb_from_wkt(gdal, "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))")).unwrap();
        let kernel = RsClip { arg_count: 3 };
        let result = kernel
            .invoke_batch(
                &[
                    RASTER,
                    SedonaType::Arrow(DataType::Int32),
                    SedonaType::Wkb(Edges::Planar, Some(crs_4326)),
                ],
                &[
                    ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(two_band_raster()))),
                    ColumnarValue::Scalar(ScalarValue::Int32(Some(0))),
                    ColumnarValue::Scalar(ScalarValue::Binary(Some(geom_wkb))),
                ],
            )
            .unwrap();
        let out = match result {
            ColumnarValue::Scalar(ScalarValue::Struct(s)) => s,
            _ => panic!("expected raster scalar"),
        };
        let raster = RasterStructArray::try_new(out.as_ref())
            .unwrap()
            .get(0)
            .unwrap();
        assert_eq!(raster.bands().len(), 2, "band 0 should clip all bands");
    }

    #[test]
    fn negative_band_errors() {
        let crs_4326 = deserialize_crs("EPSG:4326").unwrap().unwrap();
        let geom_wkb =
            with_gdal(|gdal| wkb_from_wkt(gdal, "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))")).unwrap();
        let kernel = RsClip { arg_count: 3 };
        let err = kernel
            .invoke_batch(
                &[
                    RASTER,
                    SedonaType::Arrow(DataType::Int32),
                    SedonaType::Wkb(Edges::Planar, Some(crs_4326)),
                ],
                &[
                    ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(two_band_raster()))),
                    ColumnarValue::Scalar(ScalarValue::Int32(Some(-1))),
                    ColumnarValue::Scalar(ScalarValue::Binary(Some(geom_wkb))),
                ],
            )
            .unwrap_err()
            .to_string();
        assert!(err.contains("band must be >= 0"), "unexpected error: {err}");
    }

    #[test]
    fn malformed_geometry_errors_even_when_lenient() {
        // `lenient` (default true) softens only the no-intersection case; a
        // genuine failure (garbage WKB) must still error, not become NULL.
        let kernel = RsClip { arg_count: 3 };
        let err = kernel
            .invoke_batch(
                &[
                    RASTER,
                    SedonaType::Arrow(DataType::Int32),
                    SedonaType::Wkb(Edges::Planar, None),
                ],
                &[
                    // No CRS on raster or geom, so we reach rasterization with the
                    // garbage WKB rather than erroring on a CRS mismatch first.
                    ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(
                        RasterSpec::d2(2, 1)
                            .band_values(&[1u8, 2])
                            .crs(None)
                            .transform([0.0, 1.0, 0.0, 1.0, 0.0, -1.0])
                            .build(),
                    ))),
                    ColumnarValue::Scalar(ScalarValue::Int32(Some(1))),
                    ColumnarValue::Scalar(ScalarValue::Binary(Some(vec![0xff, 0xff, 0xff, 0xff]))),
                ],
            )
            .unwrap_err();
        // The point is it errored rather than returning a NULL raster.
        let _ = err;
    }
}
