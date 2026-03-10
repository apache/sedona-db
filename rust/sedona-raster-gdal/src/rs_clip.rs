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
//! Pixels outside the geometry are set to nodata (or the minimum possible value for the
//! band pixel data type if nodata is not specified).

use std::convert::TryFrom;
use std::sync::Arc;

use arrow_array::Array;
use datafusion_common::cast::{as_boolean_array, as_float64_array, as_int32_array};
use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_common::exec_err;
use datafusion_common::{exec_datafusion_err, ScalarValue};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_common::sedona_internal_err;
use sedona_gdal::gdal::Gdal;
use sedona_gdal::mem::MemDatasetBuilder;
use sedona_gdal::raster::Buffer;
use sedona_gdal::raster::GdalDataType;
use sedona_proj::transform::with_global_proj_engine;

use arrow_schema::DataType;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::array::RasterRefImpl;
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::{BandMetadata, RasterMetadata, RasterRef};
use sedona_raster_functions::crs_utils::{crs_transform_wkb, resolve_crs};
use sedona_raster_functions::RasterExecutor;
use sedona_schema::datatypes::{SedonaType, RASTER};
use sedona_schema::matchers::ArgMatcher;
use sedona_schema::raster::{BandDataType, StorageType};

use crate::gdal_common::with_gdal;
use crate::gdal_common::{nodata_bytes_to_f64, nodata_f64_to_bytes};
use crate::gdal_dataset_provider::configure_thread_local_options;
use crate::raster_band_reader::RasterBandReader;

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

                let band_index = usize::try_from(band.max(1)).unwrap_or(1);
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
                        // No intersection between raster and geometry
                        if lenient {
                            builder.append_null()?;
                        } else {
                            return exec_err!("RS_Clip: raster and geometry do not intersect");
                        }
                    }
                    Err(e) => {
                        if lenient {
                            eprintln!("RS_Clip error: {}", e);
                            builder.append_null()?;
                        } else {
                            return Err(e);
                        }
                    }
                }

                Ok(())
            })
            })?;

            executor.finish(Arc::new(builder.finish()?))
        })
    }
}

/// Data for a clipped raster
struct ClippedRasterData {
    /// Clipped band data (one Vec<u8> per band)
    band_data: Vec<Vec<u8>>,
    /// Band metadata (data types, nodata values)
    band_metadata: Vec<BandMetadata>,
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
    let mut band_reader = RasterBandReader::new(gdal, raster);
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

    // Initialize mask to 0 (outside)
    let mask_band = mask_dataset
        .rasterband(1)
        .map_err(|e| exec_datafusion_err!("Failed to get mask band: {}", e))?;
    let zeros = vec![0u8; width * height];
    let mut buffer = Buffer::new((width, height), zeros);
    mask_band
        .write((0, 0), (width, height), &mut buffer)
        .map_err(|e| exec_datafusion_err!("Failed to initialize mask: {}", e))?;

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

    // Process each band
    let mut clipped_band_data = Vec::new();
    let mut clipped_band_metadata = Vec::new();

    for &band_idx in &band_indices {
        let band = bands
            .band(band_idx)
            .map_err(|e| exec_datafusion_err!("Failed to get band {}: {}", band_idx, e))?;

        let band_metadata = band.metadata();
        let data_type = band_metadata.data_type()?;
        let original_data = band_reader.read_band_bytes(band_idx)?;

        // Determine nodata value
        let nodata = custom_nodata
            .or_else(|| nodata_bytes_to_f64(band_metadata.nodata_value(), &data_type))
            .unwrap_or(0.0);

        // Apply mask to band data, optionally cropping
        let clipped_data = if let Some(cw) = crop_window {
            apply_mask_and_crop(&original_data, mask, width, &data_type, nodata, &cw)?
        } else {
            apply_mask_to_band(&original_data, mask, width, height, &data_type, nodata)?
        };

        // Build band metadata
        let new_band_metadata = BandMetadata {
            nodata_value: Some(nodata_f64_to_bytes(nodata, &data_type)),
            storage_type: StorageType::InDb,
            datatype: data_type,
            outdb_url: None,
            outdb_band_id: None,
        };

        clipped_band_data.push(clipped_data);
        clipped_band_metadata.push(new_band_metadata);
    }

    Ok(Some(ClippedRasterData {
        band_data: clipped_band_data,
        band_metadata: clipped_band_metadata,
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
    let byte_size = data_type_byte_size(data_type);
    let mut result = original_data.to_vec();

    for (pixel_idx, &mask_val) in mask.iter().enumerate().take(width * height) {
        if mask_val == 0 {
            // Pixel is outside geometry - set to nodata
            let byte_offset = pixel_idx * byte_size;
            write_nodata_value(&mut result, byte_offset, data_type, nodata)?;
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
    let byte_size = data_type_byte_size(data_type);
    let crop_pixel_count = cw.width * cw.height;
    let nodata_bytes = nodata_value_bytes(data_type, nodata);
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

/// Convert a nodata f64 value to its byte representation for the given data type.
fn nodata_value_bytes(data_type: &BandDataType, nodata: f64) -> Vec<u8> {
    match data_type {
        BandDataType::UInt8 => vec![nodata as u8],
        BandDataType::Int8 => (nodata as i8).to_le_bytes().to_vec(),
        BandDataType::UInt16 => (nodata as u16).to_le_bytes().to_vec(),
        BandDataType::Int16 => (nodata as i16).to_le_bytes().to_vec(),
        BandDataType::UInt32 => (nodata as u32).to_le_bytes().to_vec(),
        BandDataType::Int32 => (nodata as i32).to_le_bytes().to_vec(),
        BandDataType::UInt64 => (nodata as u64).to_le_bytes().to_vec(),
        BandDataType::Int64 => (nodata as i64).to_le_bytes().to_vec(),
        BandDataType::Float32 => (nodata as f32).to_le_bytes().to_vec(),
        BandDataType::Float64 => nodata.to_le_bytes().to_vec(),
    }
}

/// Write nodata value to band data at specified offset
fn write_nodata_value(
    data: &mut [u8],
    offset: usize,
    data_type: &BandDataType,
    nodata: f64,
) -> Result<()> {
    match data_type {
        BandDataType::UInt8 => {
            data[offset] = nodata as u8;
        }
        BandDataType::Int8 => {
            data[offset] = (nodata as i8).to_le_bytes()[0];
        }
        BandDataType::UInt16 => {
            let bytes = (nodata as u16).to_le_bytes();
            data[offset..offset + 2].copy_from_slice(&bytes);
        }
        BandDataType::Int16 => {
            let bytes = (nodata as i16).to_le_bytes();
            data[offset..offset + 2].copy_from_slice(&bytes);
        }
        BandDataType::UInt32 => {
            let bytes = (nodata as u32).to_le_bytes();
            data[offset..offset + 4].copy_from_slice(&bytes);
        }
        BandDataType::Int32 => {
            let bytes = (nodata as i32).to_le_bytes();
            data[offset..offset + 4].copy_from_slice(&bytes);
        }
        BandDataType::UInt64 => {
            let bytes = (nodata as u64).to_le_bytes();
            data[offset..offset + 8].copy_from_slice(&bytes);
        }
        BandDataType::Int64 => {
            let bytes = (nodata as i64).to_le_bytes();
            data[offset..offset + 8].copy_from_slice(&bytes);
        }
        BandDataType::Float32 => {
            let bytes = (nodata as f32).to_le_bytes();
            data[offset..offset + 4].copy_from_slice(&bytes);
        }
        BandDataType::Float64 => {
            let bytes = nodata.to_le_bytes();
            data[offset..offset + 8].copy_from_slice(&bytes);
        }
    }
    Ok(())
}

/// Build clipped raster using RasterBuilder
fn build_clipped_raster(
    builder: &mut RasterBuilder,
    original_raster: &RasterRefImpl<'_>,
    clipped_data: &ClippedRasterData,
) -> Result<()> {
    let original_metadata = original_raster.metadata();

    let metadata = if let Some(cw) = clipped_data.crop_window {
        // Cropped: adjust dimensions and upper-left coordinate.
        // new_upper_left = original_upper_left + pixel_offset * scale + pixel_offset * skew
        let new_upper_left_x = original_metadata.upper_left_x()
            + cw.col_off as f64 * original_metadata.scale_x()
            + cw.row_off as f64 * original_metadata.skew_x();
        let new_upper_left_y = original_metadata.upper_left_y()
            + cw.row_off as f64 * original_metadata.scale_y()
            + cw.col_off as f64 * original_metadata.skew_y();

        RasterMetadata {
            width: cw.width as u64,
            height: cw.height as u64,
            upperleft_x: new_upper_left_x,
            upperleft_y: new_upper_left_y,
            scale_x: original_metadata.scale_x(),
            scale_y: original_metadata.scale_y(),
            skew_x: original_metadata.skew_x(),
            skew_y: original_metadata.skew_y(),
        }
    } else {
        // No crop: use original raster dimensions and geotransform
        RasterMetadata {
            width: original_metadata.width(),
            height: original_metadata.height(),
            upperleft_x: original_metadata.upper_left_x(),
            upperleft_y: original_metadata.upper_left_y(),
            scale_x: original_metadata.scale_x(),
            scale_y: original_metadata.scale_y(),
            skew_x: original_metadata.skew_x(),
            skew_y: original_metadata.skew_y(),
        }
    };

    builder
        .start_raster(&metadata, original_raster.crs())
        .map_err(|e| exec_datafusion_err!("Failed to start raster: {}", e))?;

    // Add clipped bands
    for (band_data, band_metadata) in clipped_data
        .band_data
        .iter()
        .zip(clipped_data.band_metadata.iter())
    {
        builder
            .start_band(band_metadata.clone())
            .map_err(|e| exec_datafusion_err!("Failed to start band: {}", e))?;
        builder.band_data_writer().append_value(band_data);
        builder
            .finish_band()
            .map_err(|e| exec_datafusion_err!("Failed to finish band: {}", e))?;
    }

    builder
        .finish_raster()
        .map_err(|e| exec_datafusion_err!("Failed to finish raster: {}", e))?;

    Ok(())
}

/// Get byte size of data type
fn data_type_byte_size(data_type: &BandDataType) -> usize {
    match data_type {
        BandDataType::UInt8 => 1,
        BandDataType::Int8 => 1,
        BandDataType::UInt16 | BandDataType::Int16 => 2,
        BandDataType::UInt32 | BandDataType::Int32 | BandDataType::Float32 => 4,
        BandDataType::UInt64 | BandDataType::Int64 => 8,
        BandDataType::Float64 => 8,
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
    use sedona_raster::array::RasterStructArray;
    use sedona_schema::crs::deserialize_crs;
    use sedona_schema::datatypes::Edges;

    fn wkb_from_wkt(gdal: &sedona_gdal::gdal::Gdal, wkt: &str) -> Result<Vec<u8>> {
        let geometry = gdal.geometry_from_wkt(wkt).unwrap();
        geometry.wkb().map_err(|e| exec_datafusion_err!("{e}"))
    }

    #[test]
    fn test_rs_clip_basic() {
        // Load test raster
        let test_file = sedona_testing::data::test_raster("test4.tiff").unwrap();
        with_gdal(|gdal| {
            let raster_array = crate::utils::load_as_indb_raster(gdal, &test_file)?;
            let raster_struct = RasterStructArray::new(&raster_array);
            let raster = raster_struct.get(0).unwrap();

            let metadata = raster.metadata();
            let min_x = metadata.upper_left_x();
            let max_y = metadata.upper_left_y();
            let max_x = min_x + (metadata.width() as f64 * metadata.scale_x()) / 2.0;
            let min_y = max_y + (metadata.height() as f64 * metadata.scale_y()) / 2.0;

            let wkt = format!(
                "POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))",
                min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y, min_x, min_y
            );

            let geom_wkb = wkb_from_wkt(gdal, &wkt)?;
            let clipped = clip_raster(gdal, &raster, &geom_wkb, 0, None, false, false)?
                .expect("Should have intersection");
            let mut reader = RasterBandReader::new(gdal, &raster);
            let original_len = reader.read_band_bytes(1)?.len();
            assert!(
                !clipped.band_data.is_empty(),
                "Should have at least one band"
            );
            assert_eq!(
                clipped.band_data[0].len(),
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
        let test_file = sedona_testing::data::test_raster("test4.tiff").unwrap();
        with_gdal(|gdal| {
            let raster_array = crate::utils::load_as_indb_raster(gdal, &test_file)?;
            let raster_struct = RasterStructArray::new(&raster_array);
            let raster = raster_struct.get(0).unwrap();

            let metadata = raster.metadata();
            let min_x = metadata.upper_left_x();
            let max_y = metadata.upper_left_y();
            let max_x = min_x + (metadata.width() as f64 * metadata.scale_x()) / 4.0;
            let min_y = max_y + (metadata.height() as f64 * metadata.scale_y()) / 4.0;

            let wkt = format!(
                "POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))",
                min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y, min_x, min_y
            );

            let geom_wkb = wkb_from_wkt(gdal, &wkt)?;
            let clipped = clip_raster(gdal, &raster, &geom_wkb, 0, None, false, true)?
                .expect("Should have intersection");
            assert!(
                clipped.crop_window.is_some(),
                "crop_window should be set when crop=true"
            );

            let cw = clipped.crop_window.unwrap();
            let byte_size = data_type_byte_size(&clipped.band_metadata[0].datatype);
            assert_eq!(
                clipped.band_data[0].len(),
                cw.width * cw.height * byte_size,
                "Cropped band data size should match crop window"
            );
            assert!(
                (cw.width as u64) < metadata.width(),
                "Cropped width should be smaller"
            );
            assert!(
                (cw.height as u64) < metadata.height(),
                "Cropped height should be smaller"
            );
            Ok::<_, datafusion_common::DataFusionError>(())
        })
        .unwrap();
    }

    #[test]
    fn test_rs_clip_no_intersection() {
        let test_file = sedona_testing::data::test_raster("test4.tiff").unwrap();
        with_gdal(|gdal| {
            let raster_array = crate::utils::load_as_indb_raster(gdal, &test_file)?;
            let raster_struct = RasterStructArray::new(&raster_array);
            let raster = raster_struct.get(0).unwrap();
            let wkt = "POLYGON((1000 1000, 1001 1000, 1001 1001, 1000 1001, 1000 1000))";
            let geom_wkb = wkb_from_wkt(gdal, wkt)?;
            let result = clip_raster(gdal, &raster, &geom_wkb, 0, None, false, true)?;
            assert!(result.is_none(), "Should return None for no intersection");
            Ok::<_, datafusion_common::DataFusionError>(())
        })
        .unwrap();
    }

    #[test]
    fn test_rs_clip_crs_mismatch() {
        use sedona_expr::scalar_udf::SedonaScalarKernel;

        let test_file = sedona_testing::data::test_raster("test4.tiff").unwrap();
        let (raster_array, geom_wkb) = with_gdal(|gdal| {
            let raster_array = crate::utils::load_as_indb_raster(gdal, &test_file)?;
            let raster_struct = RasterStructArray::new(&raster_array);
            let raster = raster_struct.get(0).unwrap();

            let metadata = raster.metadata();
            let min_x = metadata.upper_left_x();
            let max_y = metadata.upper_left_y();
            let max_x = min_x + (metadata.width() as f64 * metadata.scale_x()) / 2.0;
            let min_y = max_y + (metadata.height() as f64 * metadata.scale_y()) / 2.0;

            let wkt = format!(
                "POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))",
                min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y, min_x, min_y
            );
            let geom_wkb = wkb_from_wkt(gdal, &wkt)?;
            Ok::<_, datafusion_common::DataFusionError>((raster_array, geom_wkb))
        })
        .unwrap();

        // Generate the EPSG:3857 geometry using the same PROJ engine that the
        // UDF uses for CRS transforms. This makes the test robust to axis-order
        // and normalization differences between build configurations.
        let crs_4326 = deserialize_crs("EPSG:4326").unwrap().unwrap();
        let crs_3857 = deserialize_crs("EPSG:3857").unwrap().unwrap();

        let geom_wkb_merc = with_global_proj_engine(|engine| {
            crs_transform_wkb(&geom_wkb, crs_4326.as_ref(), crs_3857.as_ref(), engine)
        })
        .unwrap();

        // Test with 3-arg variant: RS_Clip(raster, band, geom)
        let kernel = RsClip { arg_count: 3 };

        let raster_scalar = ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(raster_array)));
        let geom_type_4326 = SedonaType::Wkb(Edges::Planar, Some(crs_4326));
        let geom_type_3857 = SedonaType::Wkb(Edges::Planar, Some(crs_3857));
        let band_type = SedonaType::Arrow(DataType::Int32);
        let band_val = ColumnarValue::Scalar(ScalarValue::Int32(Some(1)));

        let result_4326 = kernel
            .invoke_batch(
                &[RASTER, band_type.clone(), geom_type_4326],
                &[
                    raster_scalar.clone(),
                    band_val.clone(),
                    ColumnarValue::Scalar(ScalarValue::Binary(Some(geom_wkb))),
                ],
            )
            .unwrap();

        let result_3857 = kernel
            .invoke_batch(
                &[RASTER, band_type, geom_type_3857],
                &[
                    raster_scalar,
                    band_val,
                    ColumnarValue::Scalar(ScalarValue::Binary(Some(geom_wkb_merc))),
                ],
            )
            .unwrap();

        let band_data_4326 = match result_4326 {
            ColumnarValue::Scalar(ScalarValue::Struct(struct_array)) => {
                let array = RasterStructArray::new(struct_array.as_ref());
                let raster = array.get(0).unwrap();
                let data = raster.bands().band(1).unwrap().data().to_vec();
                data
            }
            _ => panic!("Expected raster scalar result"),
        };

        let band_data_3857 = match result_3857 {
            ColumnarValue::Scalar(ScalarValue::Struct(struct_array)) => {
                let array = RasterStructArray::new(struct_array.as_ref());
                let raster = array.get(0).unwrap();
                let data = raster.bands().band(1).unwrap().data().to_vec();
                data
            }
            _ => panic!("Expected raster scalar result"),
        };

        assert_eq!(band_data_4326, band_data_3857);
    }

    #[test]
    fn test_write_nodata_value() {
        let mut data = vec![0u8; 8];

        // Test UInt8
        write_nodata_value(&mut data, 0, &BandDataType::UInt8, 255.0).unwrap();
        assert_eq!(data[0], 255);

        // Test Float32
        write_nodata_value(&mut data, 0, &BandDataType::Float32, -9999.0).unwrap();
        let value = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        assert!((value - (-9999.0)).abs() < 0.001);
    }
}
