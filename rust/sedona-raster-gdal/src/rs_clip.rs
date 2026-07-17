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

use arrow_array::ArrayRef;
use arrow_buffer::Buffer;
use datafusion_common::cast::{as_boolean_array, as_float64_array, as_int32_array};
use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_common::exec_err;
use datafusion_common::{exec_datafusion_err, ScalarValue};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_common::sedona_internal_err;
use sedona_gdal::gdal::Gdal;
use sedona_gdal::geo_transform::{GeoTransform, GeoTransformEx};
use sedona_gdal::mem::MemDatasetBuilder;
use sedona_gdal::raster::types::GdalDataType;
use sedona_gdal::vector::geometry::Geometry;

use arrow_schema::DataType;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::array::RasterRefImpl;
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::{is_spatial_dim_pair, RasterRef};
use sedona_raster_functions::crs_utils::{crs_transform_wkb, resolve_crs, with_crs_engine};
use sedona_raster_functions::rs_ensure_loaded::{
    NEEDS_PIXELS_METADATA_KEY, RETURNS_BYTES_METADATA_KEY,
};
use sedona_raster_functions::RasterExecutor;
use sedona_schema::datatypes::{SedonaType, RASTER};
use sedona_schema::matchers::ArgMatcher;
use sedona_schema::raster::BandDataType;

use crate::gdal_common::with_gdal;
use crate::gdal_dataset_provider::configure_thread_local_options;
use sedona_raster::traits::nodata_f64_to_bytes;

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
        let num_iterations = RasterExecutor::num_iterations_over(args);

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
            with_crs_engine(config_options, |engine| {
                executor.execute_raster_wkb_crs_void(|raster_opt, wkb_opt, geom_crs| {
                // Advance every option iterator in lockstep. A NULL in the band
                // or any boolean flag propagates to a NULL output row (SQL
                // semantics) — in particular a NULL band must NOT fall through
                // to the "band 0 = all bands" mode. `no_data_value` is the
                // exception: its NULL is the meaningful "not supplied" sentinel
                // (the 3/4-arg signatures pass NULL here), so it stays an Option.
                let band = band_iter.next().flatten();
                let all_touched = all_touched_iter.next().flatten();
                let nodata_value = nodata_iter.next().flatten();
                let crop = crop_iter.next().flatten();
                let lenient = lenient_iter.next().flatten();
                let (Some(band), Some(all_touched), Some(crop), Some(lenient)) =
                    (band, all_touched, crop, lenient)
                else {
                    builder.append_null()?;
                    return Ok(());
                };

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
                        build_clipped_raster(&mut builder, raster, clipped_data)?
                    }
                    Ok(None) => {
                        // The clip mask is empty — no pixel was selected. `lenient`
                        // yields NULL either way; strict errors, with a message
                        // conditioned on `all_touched`: when it is already true an
                        // empty mask means the geometry is genuinely disjoint,
                        // whereas with the default `all_touched = false` the
                        // geometry may still overlap but fall between pixel centers.
                        if lenient {
                            builder.append_null()?;
                        } else if all_touched {
                            return exec_err!("RS_Clip: raster and geometry do not intersect");
                        } else {
                            return exec_err!(
                                "RS_Clip: geometry selects no pixels; it may fall between \
                                 pixel centers — pass all_touched => true to keep any pixel it touches"
                            );
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
            // raster+geom must still yield an N-row array.
            let out: ArrayRef = Arc::new(builder.finish()?);
            RasterExecutor::finish_over(args, out)
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
    /// Source band name, preserved so clipping a named/N-D band keeps it
    /// addressable by name on the output.
    name: Option<String>,
}

/// Data for a clipped raster
struct ClippedRasterData {
    /// One entry per processed band.
    bands: Vec<ClippedBand>,
    /// Crop window in pixel coordinates (col_off, row_off, width, height):
    /// the geometry's envelope intersected with the raster extent, snapped
    /// outward to the pixel grid. `None` means the full original raster
    /// extent was kept (crop=false).
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

    let geotransform = [
        metadata.upper_left_x(),
        metadata.scale_x(),
        metadata.skew_x(),
        metadata.upper_left_y(),
        metadata.skew_y(),
        metadata.scale_y(),
    ];

    // The clip window is the geometry's envelope intersected with the raster
    // extent, snapped outward to the pixel grid — the window PostGIS ST_Clip,
    // `gdalwarp -crop_to_cutline`, and Sedona Spark's RS_Clip use. Knowing it
    // up front rejects disjoint geometries before any GDAL work and bounds the
    // mask to the window instead of the full raster.
    let Some(window) = envelope_window(&geometry, &geotransform, width, height)? else {
        return Ok(None);
    };

    // Create a mask raster covering only the clip window, with the geotransform
    // shifted to the window's upper-left corner.
    let mask_dataset =
        MemDatasetBuilder::create(gdal, window.width, window.height, 1, GdalDataType::UInt8)
            .map_err(|e| exec_datafusion_err!("Failed to create mask dataset: {}", e))?;
    let (window_ulx, window_uly) = geotransform.apply(window.col_off as f64, window.row_off as f64);
    let mask_geotransform = [
        window_ulx,
        geotransform[1],
        geotransform[2],
        window_uly,
        geotransform[4],
        geotransform[5],
    ];
    mask_dataset
        .set_geo_transform(&mask_geotransform)
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

    // Read the (window-sized) mask
    let mask_band = mask_dataset
        .rasterband(1)
        .map_err(|e| exec_datafusion_err!("Failed to get mask band: {}", e))?;
    let mask_buffer = mask_band
        .read_as::<u8>(
            (0, 0),
            (window.width, window.height),
            (window.width, window.height),
            None,
        )
        .map_err(|e| exec_datafusion_err!("Failed to read mask: {}", e))?;
    let mask = mask_buffer.data();

    // The envelope may overlap the raster while the geometry itself selects no
    // pixel (e.g. it falls between pixel centers); that is still the
    // no-intersection case.
    let has_intersection = mask.iter().any(|&v| v != 0);
    if !has_intersection {
        return Ok(None);
    }

    let crop_window = if crop { Some(window) } else { None };

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
        // `band_idx` is 1-based; the `band_name` accessor is 0-based.
        let band_name = raster.band_name(band_idx - 1).map(|s| s.to_string());

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
        // The trailing two dims must actually be the spatial (y, x) pair. A band
        // whose trailing dims are e.g. ["x","y"] is legal at the format level and
        // would otherwise be masked/cropped transposed on a square raster; mirror
        // the `is_spatial_dim_pair` guard the GDAL bridge (gdal_common) performs.
        if !is_spatial_dim_pair(&dim_names[ndim - 2], &dim_names[ndim - 1]) {
            return exec_err!(
                "RS_Clip: band {} trailing dims {:?} are not a (y, x) spatial pair",
                band_idx,
                &dim_names[ndim - 2..]
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

        // `as_contiguous` borrows the band bytes; we only ever read them here
        // (the mask/crop helpers write into the band's output buffer), so no
        // copy is needed.
        let nd_buffer = band.nd_buffer().map_err(|e| {
            exec_datafusion_err!("RS_Clip: failed to read band {}: {}", band_idx, e)
        })?;
        let original_data = nd_buffer.as_contiguous().map_err(|e| {
            exec_datafusion_err!("RS_Clip: band {} is not contiguous: {}", band_idx, e)
        })?;

        // nodata precedence: the explicit argument, then the band's own nodata
        // bytes (used verbatim — no lossy f64 round-trip for Int64/UInt64),
        // then the band data type's minimum value. An explicit value that is
        // not exactly representable in the band's data type (fractional for an
        // integer band, out of range, or a 64-bit integer beyond 2^53) errors
        // rather than silently saturating — e.g. -9999 on UInt8 would collide
        // with real zero-adjacent data.
        let nodata_bytes: Vec<u8> = match custom_nodata {
            Some(cn) => nodata_f64_to_bytes(cn, &data_type).map_err(|e| {
                exec_datafusion_err!("RS_Clip: invalid no_data_value for band {band_idx}: {e}")
            })?,
            None => match band_metadata.nodata_value() {
                Some(bytes) => bytes.to_vec(),
                None => data_type.min_value_le_bytes(),
            },
        };

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

        // Apply the (shared) mask/crop to each plane. One output allocation
        // serves the whole band: every plane appends into it, and the
        // finished Vec later moves into the Arrow array as a zero-copy view
        // block rather than being copied through the builder.
        let out_plane_bytes = match crop_window {
            Some(cw) => cw.width * cw.height * byte_size,
            None => in_plane_bytes,
        };
        let mut clipped_data = Vec::with_capacity(n_planes * out_plane_bytes);
        for plane in 0..n_planes {
            let plane_bytes = &original_data[plane * in_plane_bytes..(plane + 1) * in_plane_bytes];
            if let Some(cw) = crop_window {
                apply_mask_and_crop(
                    plane_bytes,
                    mask,
                    width,
                    &data_type,
                    &nodata_bytes,
                    &cw,
                    &mut clipped_data,
                )?;
            } else {
                apply_mask_to_band(
                    plane_bytes,
                    mask,
                    width,
                    &data_type,
                    &nodata_bytes,
                    &window,
                    &mut clipped_data,
                )?;
            }
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
            nodata: nodata_bytes,
            name: band_name,
        });
    }

    Ok(Some(ClippedRasterData {
        bands: clipped_bands,
        crop_window,
    }))
}

/// Compute the clip window: the geometry's envelope intersected with the
/// raster extent, snapped outward to the pixel grid. Returns `None` when the
/// envelope is disjoint from the raster extent (no clipping possible).
///
/// The envelope corners are mapped through the inverse geotransform (all four,
/// so a skewed/rotated raster still gets a correct superset window) and the
/// resulting pixel-space bbox is floored/ceiled to whole pixels. A degenerate
/// envelope (point/line) landing exactly on a grid line is widened to one
/// pixel so the rasterizer — not the snapping — decides whether it burns.
fn envelope_window(
    geometry: &Geometry,
    geotransform: &GeoTransform,
    width: usize,
    height: usize,
) -> Result<Option<CropWindow>> {
    let env = geometry.envelope();
    let inverse = geotransform
        .invert()
        .map_err(|e| exec_datafusion_err!("RS_Clip: geotransform is not invertible: {}", e))?;

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

    Ok(Some(CropWindow {
        col_off: col0 as usize,
        row_off: row0 as usize,
        width: (col1 - col0) as usize,
        height: (row1 - row0) as usize,
    }))
}

/// Apply mask to band data (no cropping — preserves original dimensions).
/// The mask covers only `window`; every pixel outside it is outside the
/// geometry's envelope and therefore nodata. The plane's bytes are appended
/// to `out` — the caller-owned per-band buffer — so one allocation serves
/// every plane of the band instead of one per plane.
fn apply_mask_to_band(
    original_data: &[u8],
    mask: &[u8],
    width: usize,
    data_type: &BandDataType,
    nodata_bytes: &[u8],
    window: &CropWindow,
    out: &mut Vec<u8>,
) -> Result<()> {
    let byte_size = data_type.byte_size();
    let row_bytes = width * byte_size;
    let height = original_data.len() / row_bytes;
    let nodata_row: Vec<u8> = nodata_bytes.repeat(width);
    let base = out.len();
    out.resize(base + width * height * byte_size, 0);
    let result = &mut out[base..];

    for row in 0..height {
        let dst_row = &mut result[row * row_bytes..(row + 1) * row_bytes];
        if row < window.row_off || row >= window.row_off + window.height {
            dst_row.copy_from_slice(&nodata_row);
            continue;
        }
        // Within a window row: nodata left and right of the window, source
        // bytes inside it (masked-out pixels overwritten below).
        let win_start = window.col_off * byte_size;
        let win_end = (window.col_off + window.width) * byte_size;
        dst_row[..win_start].copy_from_slice(&nodata_row[..win_start]);
        dst_row[win_end..].copy_from_slice(&nodata_row[win_end..]);
        let src_start = row * row_bytes + win_start;
        dst_row[win_start..win_end]
            .copy_from_slice(&original_data[src_start..src_start + (win_end - win_start)]);

        let mask_row_start = (row - window.row_off) * window.width;
        for col in 0..window.width {
            if mask[mask_row_start + col] == 0 {
                let dst = win_start + col * byte_size;
                dst_row[dst..dst + byte_size].copy_from_slice(nodata_bytes);
            }
        }
    }

    Ok(())
}

/// Apply mask AND crop to the given crop window in one pass. The mask is
/// window-sized (row-major over `cw.width`×`cw.height`); the source data is
/// the full raster plane. The plane's bytes are appended to `out` — the
/// caller-owned per-band buffer — so one allocation serves every plane of
/// the band instead of one per plane.
fn apply_mask_and_crop(
    original_data: &[u8],
    mask: &[u8],
    full_width: usize,
    data_type: &BandDataType,
    nodata_bytes: &[u8],
    cw: &CropWindow,
    out: &mut Vec<u8>,
) -> Result<()> {
    let byte_size = data_type.byte_size();
    let row_bytes = cw.width * byte_size;
    let base = out.len();
    out.resize(base + cw.height * row_bytes, 0);
    let result = &mut out[base..];

    // The crop-window columns of a source row are contiguous, so copy each row
    // in one bulk memcpy (which vectorizes) rather than per pixel, then overwrite
    // only the masked-out pixels with nodata. Masked pixels are written twice —
    // once by the bulk copy, once here — but the bulk memcpy is far cheaper than
    // the dynamic-width per-pixel copy it replaces.
    for row in 0..cw.height {
        let src_row = cw.row_off + row;
        let src_start = (src_row * full_width + cw.col_off) * byte_size;
        let dst_start = row * row_bytes;
        result[dst_start..dst_start + row_bytes]
            .copy_from_slice(&original_data[src_start..src_start + row_bytes]);

        let mask_row_start = row * cw.width;
        for col in 0..cw.width {
            if mask[mask_row_start + col] == 0 {
                let dst = dst_start + col * byte_size;
                result[dst..dst + byte_size].copy_from_slice(nodata_bytes);
            }
        }
    }

    Ok(())
}

/// Build the clipped raster via the N-D builder. A 2-D raster is just the
/// `["y", "x"]` case; an N-D raster keeps its non-spatial dims and only its
/// `(y, x)` extent changes when cropping.
fn build_clipped_raster(
    builder: &mut RasterBuilder,
    original_raster: &RasterRefImpl<'_>,
    clipped_data: ClippedRasterData,
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

    for band in clipped_data.bands {
        let dim_names: Vec<&str> = band.dim_names.iter().map(String::as_str).collect();
        builder
            .start_band_nd(
                band.name.as_deref(),
                &dim_names,
                &band.shape,
                band.data_type,
                Some(&band.nodata),
                None,
                None,
            )
            .map_err(|e| exec_datafusion_err!("Failed to start band: {}", e))?;
        // Move the band bytes into an Arrow buffer and append them as a view
        // (a refcount bump), instead of copying them through the builder.
        let len = u32::try_from(band.data.len()).map_err(|_| {
            exec_datafusion_err!(
                "RS_Clip: band data of {} bytes exceeds the binary-view limit",
                band.data.len()
            )
        })?;
        let buffer = Buffer::from(band.data);
        builder
            .append_band_data_buffer(&buffer, 0, len)
            .map_err(|e| exec_datafusion_err!("Failed to append band data: {}", e))?;
        builder
            .finish_band()
            .map_err(|e| exec_datafusion_err!("Failed to finish band: {}", e))?;
    }

    builder
        .finish_raster()
        .map_err(|e| exec_datafusion_err!("Failed to finish raster: {}", e))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{cast::AsArray, StructArray};
    use sedona_expr::scalar_udf::SedonaScalarKernel;
    use sedona_proj::transform::{with_global_proj_engine, LazyProjEngine};
    use sedona_raster::array::RasterStructArray;
    use sedona_schema::crs::deserialize_crs;
    use sedona_schema::datatypes::Edges;
    use sedona_testing::create::make_wkb;
    use sedona_testing::raster_spec::{
        assert_raster_scalar_equals, assert_rasters_equal, raster_array, RasterSpec,
    };
    use sedona_testing::testers::ScalarUdfTester;

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
            let geom_wkb = make_wkb("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))");
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
            // The polygon covers cols 0-1 of both rows; cols 2-3 lie outside
            // the geometry envelope entirely and must also read nodata (0, the
            // UInt8 minimum) — this pins the outside-window fill.
            assert_eq!(clipped.bands[0].data, vec![1u8, 2, 0, 0, 5, 6, 0, 0]);
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
            let geom_wkb = make_wkb("POLYGON((0 1, 2 1, 2 2, 0 2, 0 1))");
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
            // Both window pixels are inside the polygon, so the cropped band
            // holds the source values for that 2×1 window (row-major) — this
            // pins the row-copy offsets in `apply_mask_and_crop`.
            assert_eq!(
                clipped.bands[0].data,
                vec![1u8, 2],
                "cropped band should keep the source pixel values in the window"
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
    fn crop_uses_envelope_window_not_tight_mask_bbox() {
        // PostGIS ST_Clip / gdalwarp -crop_to_cutline semantics: the crop
        // window is the geometry's envelope ∩ raster extent snapped to the
        // grid, keeping unselected envelope pixels as nodata padding — not the
        // tight bbox of the selected mask pixels.
        let array = test_raster_array();
        with_gdal(|gdal| {
            let rasters = RasterStructArray::try_new(&array).unwrap();
            let raster = rasters.get(0).unwrap();

            // Right triangle with envelope x ∈ [0, 3], y ∈ [0, 2] — a 3×2
            // pixel window. Only three pixel centers fall inside it: (0.5, 1.5)
            // -> 1, (0.5, 0.5) -> 5, (1.5, 0.5) -> 6. The tight mask bbox
            // would be 2×2; the envelope window must be 3×2 with nodata (0)
            // padding on the unselected pixels.
            let geom_wkb = make_wkb("POLYGON((0 0, 3 0, 0 2, 0 0))");
            let clipped = clip_raster(gdal, &raster, &geom_wkb, 0, None, false, true)?
                .expect("Should have intersection");

            let cw = clipped.crop_window.expect("crop_window should be set");
            assert_eq!(
                (cw.col_off, cw.row_off, cw.width, cw.height),
                (0, 0, 3, 2),
                "crop window should be the snapped envelope, not the mask bbox"
            );
            assert_eq!(clipped.bands[0].data, vec![1u8, 0, 0, 5, 6, 0]);
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
            let geom_wkb = make_wkb("POLYGON((100 100, 101 100, 101 101, 100 101, 100 100))");
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

        let geom_wkb_4326 = make_wkb("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))");
        let geom_wkb_3857 = with_global_proj_engine(|engine| {
            crs_transform_wkb(&geom_wkb_4326, crs_4326.as_ref(), crs_3857.as_ref(), engine)
        })
        .unwrap();

        // 3-arg variant: RS_Clip(raster, band, geom).
        let kernel = RsClip { arg_count: 3 };
        let raster_scalar = ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(array)));
        let band_type = SedonaType::Arrow(DataType::Int32);
        let band_val = ColumnarValue::Scalar(ScalarValue::Int32(Some(1)));

        // Both the native-CRS and the reprojected geometry must produce the
        // same clip: the polygon covers columns 0-1 of both rows, cropped to
        // that 2x2 window with the UInt8 minimum recorded as nodata.
        let expected = RasterSpec::d2(2, 2)
            .crs(Some("EPSG:4326"))
            .transform([0.0, 1.0, 0.0, 2.0, 0.0, -1.0])
            .band_values(&[1u8, 2, 5, 6])
            .nodata(0u8);
        let clip_band1 = |geom_type: SedonaType, geom_wkb: Vec<u8>| {
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
            let ColumnarValue::Scalar(scalar) = result else {
                panic!("Expected raster scalar result");
            };
            assert_raster_scalar_equals(&scalar, &expected);
        };

        clip_band1(
            SedonaType::Wkb(Edges::Planar, Some(crs_4326)),
            geom_wkb_4326,
        );
        clip_band1(
            SedonaType::Wkb(Edges::Planar, Some(crs_3857)),
            geom_wkb_3857,
        );
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
        let geom_wkb = make_wkb("POLYGON((0 1, 2 1, 2 2, 0 2, 0 1))");

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

        let ColumnarValue::Scalar(scalar) = result else {
            panic!("Expected raster scalar result");
        };
        // The time dim is preserved; (y, x) is cropped to the 1×2 mask
        // window, applied to both planes: time 0 -> [1, 2], time 1 -> [9, 10].
        assert_raster_scalar_equals(
            &scalar,
            &RasterSpec::nd(&["time", "y", "x"], &[2, 1, 2])
                .band_values(&[1u8, 2, 9, 10])
                .crs(Some("EPSG:4326"))
                .transform([0.0, 1.0, 0.0, 2.0, 0.0, -1.0])
                .nodata(0u8),
        );
    }

    #[test]
    fn test_default_nodata_sentinel_is_type_minimum() {
        // Unsigned types floor at 0; signed/float at their most-negative value.
        assert_eq!(BandDataType::UInt8.min_value_le_bytes(), vec![0u8]);
        assert_eq!(
            BandDataType::UInt64.min_value_le_bytes(),
            0u64.to_le_bytes()
        );
        assert_eq!(
            BandDataType::Int8.min_value_le_bytes(),
            i8::MIN.to_le_bytes()
        );
        assert_eq!(
            BandDataType::Int64.min_value_le_bytes(),
            i64::MIN.to_le_bytes()
        );
        assert_eq!(
            BandDataType::Float64.min_value_le_bytes(),
            f64::MIN.to_le_bytes()
        );
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
        let geom_wkb = make_wkb("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))");

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
        // band 1 and row 1 clipping band 2 — distinct outputs, each cropped
        // to the single covered pixel.
        let arr = match result {
            ColumnarValue::Array(a) => a,
            ColumnarValue::Scalar(_) => panic!("expected an array; the batch collapsed to row 0"),
        };
        let row = |values: &[u8]| {
            Some(
                RasterSpec::d2(1, 1)
                    .crs(Some("EPSG:4326"))
                    .transform([0.0, 1.0, 0.0, 1.0, 0.0, -1.0])
                    .band_values(values)
                    .nodata(0u8),
            )
        };
        assert_rasters_equal(&arr, &[row(&[1u8]), row(&[10u8])]);
    }

    #[test]
    fn band_zero_clips_all_bands() {
        // Band 0 means "all bands" — it must reach clip_raster as 0, not be
        // clamped to 1.
        let crs_4326 = deserialize_crs("EPSG:4326").unwrap().unwrap();
        let geom_wkb = make_wkb("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))");
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
        let ColumnarValue::Scalar(scalar) = result else {
            panic!("expected raster scalar");
        };
        // Band 0 clips every band: both come through, each cropped to the
        // covered pixel.
        assert_raster_scalar_equals(
            &scalar,
            &RasterSpec::d2(1, 1)
                .crs(Some("EPSG:4326"))
                .transform([0.0, 1.0, 0.0, 1.0, 0.0, -1.0])
                .band_values(&[1u8])
                .nodata(0u8)
                .band_values(&[10u8])
                .nodata(0u8),
        );
    }

    #[test]
    fn negative_band_errors() {
        let crs_4326 = deserialize_crs("EPSG:4326").unwrap().unwrap();
        let geom_wkb = make_wkb("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))");
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

    #[test]
    fn null_band_yields_null() {
        // A NULL band must propagate to NULL, not silently clip every band.
        let crs_4326 = deserialize_crs("EPSG:4326").unwrap().unwrap();
        let geom_wkb = make_wkb("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))");
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
                    ColumnarValue::Scalar(ScalarValue::Int32(None)),
                    ColumnarValue::Scalar(ScalarValue::Binary(Some(geom_wkb))),
                ],
            )
            .unwrap();
        match result {
            ColumnarValue::Scalar(sv) => {
                assert!(
                    sv.is_null(),
                    "NULL band should yield NULL, not clip all bands"
                )
            }
            other => panic!("expected scalar, got {other:?}"),
        }
    }

    #[test]
    fn custom_nodata_out_of_range_errors() {
        // A UInt8 band can't represent -9999; reject it rather than saturating
        // it to 0 (which would collide with real zero-valued data).
        let array = test_raster_array(); // one UInt8 band
        with_gdal(|gdal| {
            let rasters = RasterStructArray::try_new(&array).unwrap();
            let raster = rasters.get(0).unwrap();
            let geom_wkb = make_wkb("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))");

            let err = match clip_raster(gdal, &raster, &geom_wkb, 1, Some(-9999.0), false, false) {
                Err(e) => e.to_string(),
                Ok(_) => panic!("expected out-of-range nodata to error"),
            };
            assert!(
                err.contains("not a valid UInt8 value"),
                "unexpected error: {err}"
            );

            // A fractional nodata can't be represented in an integer band either.
            let err = match clip_raster(gdal, &raster, &geom_wkb, 1, Some(2.5), false, false) {
                Err(e) => e.to_string(),
                Ok(_) => panic!("expected fractional nodata to error"),
            };
            assert!(
                err.contains("not a valid UInt8 value"),
                "unexpected error: {err}"
            );

            // An in-range custom nodata is accepted.
            let ok = clip_raster(gdal, &raster, &geom_wkb, 1, Some(200.0), false, false)?;
            assert!(ok.is_some(), "in-range custom nodata should be accepted");
            Ok::<_, datafusion_common::DataFusionError>(())
        })
        .unwrap();
    }

    #[test]
    fn preserves_band_name() {
        // Clipping a named band keeps the name on the output band.
        let array = RasterSpec::d2(4, 2)
            .band_values(&[1u8, 2, 3, 4, 5, 6, 7, 8])
            .name("elevation")
            .crs(Some("EPSG:4326"))
            .transform([0.0, 1.0, 0.0, 2.0, 0.0, -1.0])
            .build();
        with_gdal(|gdal| {
            let rasters = RasterStructArray::try_new(&array).unwrap();
            let raster = rasters.get(0).unwrap();
            let geom_wkb = make_wkb("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))");
            let clipped = clip_raster(gdal, &raster, &geom_wkb, 1, None, false, false)?
                .expect("Should have intersection");
            assert_eq!(
                clipped.bands[0].name.as_deref(),
                Some("elevation"),
                "clipped band should keep the source band name"
            );
            Ok::<_, datafusion_common::DataFusionError>(())
        })
        .unwrap();
    }

    #[test]
    fn strict_no_pixel_message_depends_on_all_touched() {
        // With lenient=false and an empty clip mask, the strict error is
        // conditioned on all_touched: a plain "do not intersect" when every
        // touched pixel was already considered, and a sub-pixel hint otherwise.
        let geom_wkb = make_wkb("POLYGON((100 100, 101 100, 101 101, 100 101, 100 100))");
        // No CRS on raster or geom, so we reach the mask check rather than a
        // CRS-mismatch error first.
        let raster = RasterSpec::d2(4, 2)
            .band_values(&[1u8, 2, 3, 4, 5, 6, 7, 8])
            .crs(None)
            .transform([0.0, 1.0, 0.0, 2.0, 0.0, -1.0])
            .build();
        let arg_types = [
            RASTER,
            SedonaType::Arrow(DataType::Int32),
            SedonaType::Wkb(Edges::Planar, None),
            SedonaType::Arrow(DataType::Boolean),
            SedonaType::Arrow(DataType::Float64),
            SedonaType::Arrow(DataType::Boolean),
            SedonaType::Arrow(DataType::Boolean),
        ];
        let kernel = RsClip { arg_count: 7 };
        let invoke = |all_touched: bool| {
            kernel
                .invoke_batch(
                    &arg_types,
                    &[
                        ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(raster.clone()))),
                        ColumnarValue::Scalar(ScalarValue::Int32(Some(1))),
                        ColumnarValue::Scalar(ScalarValue::Binary(Some(geom_wkb.clone()))),
                        ColumnarValue::Scalar(ScalarValue::Boolean(Some(all_touched))),
                        ColumnarValue::Scalar(ScalarValue::Float64(None)),
                        ColumnarValue::Scalar(ScalarValue::Boolean(Some(false))), // crop
                        ColumnarValue::Scalar(ScalarValue::Boolean(Some(false))), // lenient
                    ],
                )
                .unwrap_err()
                .to_string()
        };
        assert!(
            invoke(true).contains("do not intersect"),
            "all_touched=true should give the disjoint message"
        );
        let msg = invoke(false);
        assert!(
            msg.contains("selects no pixels") && msg.contains("all_touched"),
            "all_touched=false should hint at the sub-pixel case: {msg}"
        );
    }

    #[test]
    fn test_rs_clip_band_data_is_block_backed() {
        // The clipped band bytes move into the output as a zero-copy view
        // block; pin that so a refactor can't silently reintroduce the copy.
        // (Views at or under the inline threshold store their bytes inline,
        // so the band must be bigger than that.)
        let values: Vec<u8> = (0..32).collect();
        let tester = ScalarUdfTester::new(
            rs_clip_udf().into(),
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Int32),
                SedonaType::Wkb(Edges::Planar, None),
            ],
        );
        let result = tester
            .invoke_array_scalar_scalar(
                Arc::new(raster_array([Some(
                    RasterSpec::d2(8, 4).crs(None).band_values(&values),
                )])),
                1,
                ScalarValue::Binary(Some(make_wkb("POLYGON((0 0, 8 0, 8 -4, 0 -4, 0 0))"))),
            )
            .unwrap();

        let rasters = RasterStructArray::try_new(result.as_struct()).unwrap();
        let band_data = rasters.band_data_array();
        assert_eq!(
            band_data.value(rasters.band_data_row(0, 0)),
            values.as_slice()
        );
        assert_eq!(
            band_data.data_buffers().len(),
            1,
            "band bytes should be appended as a view block, not copied"
        );
    }

    #[test]
    fn test_rs_clip_reprojects_with_tester_crs_engine() {
        // Through the tester, the kernel reads its CRS engine from the
        // SedonaOptions in the tester's config options: a reprojecting clip
        // errors under the default engine and succeeds once a real engine is
        // supplied via `with_crs_engine`. (Config-less direct invocation,
        // covered by test_rs_clip_crs_mismatch, falls back to the global
        // engine instead.)
        let spec = RasterSpec::d2(4, 2)
            .band_values(&[1u8, 2, 3, 4, 5, 6, 7, 8])
            .crs(Some("EPSG:4326"))
            .transform([0.0, 1.0, 0.0, 2.0, 0.0, -1.0]);

        let crs_4326 = deserialize_crs("EPSG:4326").unwrap().unwrap();
        let crs_3857 = deserialize_crs("EPSG:3857").unwrap().unwrap();
        let geom_wkb_4326 = make_wkb("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))");
        let geom_wkb_3857 = with_global_proj_engine(|engine| {
            crs_transform_wkb(&geom_wkb_4326, crs_4326.as_ref(), crs_3857.as_ref(), engine)
        })
        .unwrap();

        let udf: datafusion_expr::ScalarUDF = rs_clip_udf().into();
        let arg_types = vec![
            RASTER,
            SedonaType::Arrow(DataType::Int32),
            SedonaType::Wkb(Edges::Planar, Some(crs_3857)),
        ];

        let tester = ScalarUdfTester::new(udf.clone(), arg_types.clone());
        let err = tester
            .invoke_scalar_scalar_scalar(
                spec.scalar(),
                1,
                ScalarValue::Binary(Some(geom_wkb_3857.clone())),
            )
            .unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("crs"),
            "the default engine should error, got: {err}"
        );

        let tester = ScalarUdfTester::new(udf, arg_types).with_crs_engine(Arc::new(LazyProjEngine));
        let result = tester
            .invoke_scalar_scalar_scalar(spec.scalar(), 1, ScalarValue::Binary(Some(geom_wkb_3857)))
            .unwrap();

        // The polygon covers columns 0-1 of both rows; with the default
        // crop the output is that 2x2 window, with the raster's CRS and
        // origin carried through and the UInt8 minimum recorded as nodata.
        assert_raster_scalar_equals(
            &result,
            &RasterSpec::d2(2, 2)
                .crs(Some("EPSG:4326"))
                .transform([0.0, 1.0, 0.0, 2.0, 0.0, -1.0])
                .band_values(&[1u8, 2, 5, 6])
                .nodata(0u8),
        );
    }
}
