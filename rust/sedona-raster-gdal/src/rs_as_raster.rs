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

//! RS_AsRaster UDF - Rasterize a vector geometry onto a raster grid.
//!
//! RS_AsRaster converts a vector geometry into a raster dataset by assigning a
//! specified value to all pixels covered by the geometry.

use std::sync::Arc;

use arrow_array::Array;
use datafusion_common::cast::{
    as_binary_array, as_boolean_array, as_float64_array, as_string_array,
};
use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_common::{exec_datafusion_err, exec_err, ScalarValue};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_gdal::dataset::Dataset;
use sedona_gdal::gdal::Gdal;
use sedona_gdal::mem::MemDatasetBuilder;
use sedona_gdal::raster::Buffer;

use arrow_schema::DataType;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::array::RasterRefImpl;
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::{BandMetadata, RasterMetadata, RasterRef};
use sedona_raster_functions::RasterExecutor;
use sedona_schema::datatypes::{SedonaType, RASTER};
use sedona_schema::matchers::ArgMatcher;
use sedona_schema::raster::{BandDataType, StorageType};

use crate::gdal_common::{band_data_type_to_gdal, nodata_f64_to_bytes, with_gdal};
use crate::gdal_dataset_provider::configure_thread_local_options;

/// RS_AsRaster() scalar UDF implementation
pub fn rs_as_raster_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_asraster",
        vec![
            Arc::new(RsAsRaster { arg_count: 3 }),
            Arc::new(RsAsRaster { arg_count: 4 }),
            Arc::new(RsAsRaster { arg_count: 5 }),
            Arc::new(RsAsRaster { arg_count: 6 }),
            Arc::new(RsAsRaster { arg_count: 7 }),
        ],
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct RsAsRaster {
    /// Number of arguments in the matched signature (3..=7)
    arg_count: usize,
}

impl SedonaScalarKernel for RsAsRaster {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let mut matchers = vec![
            ArgMatcher::is_geometry_or_geography(),
            ArgMatcher::is_raster(),
            ArgMatcher::is_string(),
        ];

        if self.arg_count >= 4 {
            matchers.push(ArgMatcher::is_boolean());
        }
        if self.arg_count >= 5 {
            matchers.push(ArgMatcher::is_numeric());
        }
        if self.arg_count >= 6 {
            matchers.push(ArgMatcher::is_numeric());
        }
        if self.arg_count >= 7 {
            matchers.push(ArgMatcher::is_boolean());
        }

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

        // Convert all non-raster/non-geometry args to arrays upfront via into_array
        // arg[2]: pixelType (always present, string)
        let pixel_type_array = args[2]
            .clone()
            .cast_to(&DataType::Utf8, None)?
            .into_array(num_iterations)?;
        let pixel_type_array = as_string_array(&pixel_type_array)?;

        // arg[3]: all_touched (if arg_count >= 4, boolean; default false)
        let all_touched_array = if self.arg_count >= 4 {
            args[3]
                .clone()
                .cast_to(&DataType::Boolean, None)?
                .into_array(num_iterations)?
        } else {
            ScalarValue::Boolean(Some(false)).to_array_of_size(num_iterations)?
        };
        let all_touched_array = as_boolean_array(&all_touched_array)?;

        // arg[4]: burn_value (if arg_count >= 5, numeric -> f64; default 1.0)
        let burn_value_array = if self.arg_count >= 5 {
            args[4]
                .clone()
                .cast_to(&DataType::Float64, None)?
                .into_array(num_iterations)?
        } else {
            ScalarValue::Float64(Some(1.0)).to_array_of_size(num_iterations)?
        };
        let burn_value_array = as_float64_array(&burn_value_array)?;

        // arg[5]: nodata_value (if arg_count >= 6, numeric -> f64; default None)
        let nodata_value_array = if self.arg_count >= 6 {
            args[5]
                .clone()
                .cast_to(&DataType::Float64, None)?
                .into_array(num_iterations)?
        } else {
            ScalarValue::Float64(None).to_array_of_size(num_iterations)?
        };
        let nodata_value_array = as_float64_array(&nodata_value_array)?;

        // arg[6]: use_geometry_extent (if arg_count >= 7, boolean; default true)
        let use_geom_extent_array = if self.arg_count >= 7 {
            args[6]
                .clone()
                .cast_to(&DataType::Boolean, None)?
                .into_array(num_iterations)?
        } else {
            ScalarValue::Boolean(Some(true)).to_array_of_size(num_iterations)?
        };
        let use_geom_extent_array = as_boolean_array(&use_geom_extent_array)?;

        // Convert geometry (arg[0]) to binary array
        let geom_array = args[0].clone().into_array(num_iterations)?;
        let geom_array = as_binary_array(&geom_array)?;
        let mut geom_iter = geom_array.iter();

        let mut pixel_type_iter = pixel_type_array.iter();
        let mut all_touched_iter = all_touched_array.iter();
        let mut burn_value_iter = burn_value_array.iter();
        let mut nodata_value_iter = nodata_value_array.iter();
        let mut use_geom_extent_iter = use_geom_extent_array.iter();

        let mut builder = RasterBuilder::new(num_iterations);

        // Raster is at arg[1] — create executor with raster-only subset
        let exec_arg_types = vec![arg_types[1].clone()];
        let exec_args = vec![args[1].clone()];
        let executor =
            RasterExecutor::new_with_num_iterations(&exec_arg_types, &exec_args, num_iterations);

        with_gdal(|gdal| {
            configure_thread_local_options(gdal, config_options)?;
            executor.execute_raster_void(|_i, raster_opt| {
                let geom_opt = geom_iter.next().unwrap();
                let pixel_type_opt = pixel_type_iter.next().unwrap();
                let all_touched_opt = all_touched_iter.next().unwrap();
                let burn_value_opt = burn_value_iter.next().unwrap();
                let nodata_value_opt = nodata_value_iter.next().unwrap();
                let use_geom_extent_opt = use_geom_extent_iter.next().unwrap();

                let raster = match raster_opt {
                    Some(r) => r,
                    None => {
                        builder.append_null()?;
                        return Ok(());
                    }
                };
                let geom_wkb = match geom_opt {
                    Some(g) => g,
                    None => {
                        builder.append_null()?;
                        return Ok(());
                    }
                };
                let pixel_type_str = match pixel_type_opt {
                    Some(s) => s,
                    None => {
                        builder.append_null()?;
                        return Ok(());
                    }
                };

                let band_type = parse_pixel_type(pixel_type_str)?;
                let all_touched = all_touched_opt.unwrap_or(false);
                let burn_value = burn_value_opt.unwrap_or(1.0);
                let nodata_value = nodata_value_opt;
                let use_geometry_extent = use_geom_extent_opt.unwrap_or(true);

                match as_raster(
                    gdal,
                    geom_wkb,
                    raster,
                    band_type,
                    all_touched,
                    burn_value,
                    nodata_value,
                    use_geometry_extent,
                ) {
                    Ok((out_metadata, out_band_metadata, out_band_bytes)) => {
                        builder
                            .start_raster(&out_metadata, raster.crs())
                            .map_err(|e| {
                                exec_datafusion_err!("Failed to start output raster: {}", e)
                            })?;

                        builder.start_band(out_band_metadata).map_err(|e| {
                            exec_datafusion_err!("Failed to start output raster band: {}", e)
                        })?;

                        builder.band_data_writer().append_value(&out_band_bytes);
                        builder.finish_band().map_err(|e| {
                            exec_datafusion_err!("Failed to finish output raster band: {}", e)
                        })?;

                        builder.finish_raster().map_err(|e| {
                            exec_datafusion_err!("Failed to finish output raster: {}", e)
                        })?;
                    }
                    Err(e) => {
                        eprintln!("RS_AsRaster error: {}", e);
                        builder.append_null()?;
                    }
                }

                Ok(())
            })?;

            // Use finish_result to check ALL original args for scalar/array decision,
            // since the executor only has the raster subset.
            finish_result(args, Arc::new(builder.finish()?))
        })
    }
}

fn parse_pixel_type(s: &str) -> Result<BandDataType> {
    match s.trim().to_ascii_uppercase().as_str() {
        "D" => Ok(BandDataType::Float64),
        "F" => Ok(BandDataType::Float32),
        "I" => Ok(BandDataType::Int32),
        "S" => Ok(BandDataType::Int16),
        "US" => Ok(BandDataType::UInt16),
        "B" => Ok(BandDataType::UInt8),
        "I8" | "INT8" => Ok(BandDataType::Int8),
        "U64" | "UINT64" => Ok(BandDataType::UInt64),
        "I64" | "INT64" => Ok(BandDataType::Int64),
        other => exec_err!(
            "Unsupported pixelType: {} (expected one of D, F, I, S, US, B, I8, U64, I64)",
            other
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn as_raster(
    gdal: &Gdal,
    geom_wkb: &[u8],
    reference_raster: &RasterRefImpl<'_>,
    band_type: BandDataType,
    all_touched: bool,
    burn_value: f64,
    nodata_value: Option<f64>,
    use_geometry_extent: bool,
) -> Result<(RasterMetadata, BandMetadata, Vec<u8>)> {
    let ref_md = reference_raster.metadata();

    if ref_md.skew_x() != 0.0 || ref_md.skew_y() != 0.0 {
        return exec_err!(
            "RS_AsRaster currently requires skew_x=0 and skew_y=0 in the reference raster"
        );
    }

    // Parse geometry
    let geometry = gdal
        .geometry_from_wkb(geom_wkb)
        .map_err(|e| exec_datafusion_err!("Failed to parse geometry from WKB: {}", e))?;

    // Compute output grid
    let (out_width, out_height, out_ulx, out_uly) = if use_geometry_extent {
        let env = geometry.envelope();
        let ulx = ref_md.upper_left_x();
        let uly = ref_md.upper_left_y();
        let scale_x = ref_md.scale_x();
        let scale_y = ref_md.scale_y();

        if scale_x == 0.0 || scale_y == 0.0 {
            return exec_err!("Reference raster has zero scale");
        }

        let start_col = ((env.MinX - ulx) / scale_x).floor() as isize;
        let end_col_excl = ((env.MaxX - ulx) / scale_x).ceil() as isize;

        // Note: scale_y is typically negative.
        let start_row = ((env.MaxY - uly) / scale_y).floor() as isize;
        let end_row_excl = ((env.MinY - uly) / scale_y).ceil() as isize;

        let width = (end_col_excl - start_col).max(0) as usize;
        let height = (end_row_excl - start_row).max(0) as usize;

        if width == 0 || height == 0 {
            return exec_err!("Geometry extent produced an empty raster");
        }

        let out_ulx = ulx + (start_col as f64) * scale_x;
        let out_uly = uly + (start_row as f64) * scale_y;
        (width, height, out_ulx, out_uly)
    } else {
        (
            ref_md.width() as usize,
            ref_md.height() as usize,
            ref_md.upper_left_x(),
            ref_md.upper_left_y(),
        )
    };

    // Create output GDAL dataset
    let gdal_type = band_data_type_to_gdal(&band_type);
    let out_dataset = MemDatasetBuilder::create(gdal, out_width, out_height, 1, gdal_type)
        .map_err(|e| exec_datafusion_err!("Failed to create dataset: {}", e))?;

    let geotransform = [
        out_ulx,
        ref_md.scale_x(),
        ref_md.skew_x(),
        out_uly,
        ref_md.skew_y(),
        ref_md.scale_y(),
    ];
    out_dataset
        .set_geo_transform(&geotransform)
        .map_err(|e| exec_datafusion_err!("Failed to set geotransform: {}", e))?;

    // Set spatial reference based on reference raster dataset (if present)
    let provider = crate::gdal_dataset_provider::thread_local_provider(gdal)
        .map_err(|e| exec_datafusion_err!("Failed to init GDAL provider: {}", e))?;
    let ref_raster_ds = provider
        .raster_ref_to_gdal(reference_raster)
        .map_err(|e| exec_datafusion_err!("Failed to create GDAL dataset: {}", e))?;
    if let Ok(srs) = ref_raster_ds.as_dataset().spatial_ref() {
        out_dataset
            .set_spatial_ref(&srs)
            .map_err(|e| exec_datafusion_err!("Failed to set spatial reference: {}", e))?;
    }

    // Initialize output band to nodata (if provided) or 0
    let init_value = nodata_value.unwrap_or(0.0);
    initialize_band(&out_dataset, &band_type, out_width, out_height, init_value)?;

    // Set nodata metadata on band
    if let Some(nodata) = nodata_value {
        let band = out_dataset
            .rasterband(1)
            .map_err(|e| exec_datafusion_err!("Failed to get output band: {}", e))?;
        match band_type {
            BandDataType::UInt64 => {
                band.set_no_data_value_u64(Some(nodata as u64))
                    .map_err(|e| exec_datafusion_err!("Failed to set nodata value: {}", e))?;
            }
            BandDataType::Int64 => {
                band.set_no_data_value_i64(Some(nodata as i64))
                    .map_err(|e| exec_datafusion_err!("Failed to set nodata value: {}", e))?;
            }
            _ => band
                .set_no_data_value(Some(nodata))
                .map_err(|e| exec_datafusion_err!("Failed to set nodata value: {}", e))?,
        }
    }

    gdal.rasterize_affine(&out_dataset, &[1], &[geometry], &[burn_value], all_touched)
        .map_err(|e| exec_datafusion_err!("Failed to rasterize geometry: {}", e))?;

    // Read band data as bytes
    let band_bytes = read_band_as_bytes(&out_dataset, 1, out_width, out_height, &band_type)?;

    let out_metadata = RasterMetadata {
        width: out_width as u64,
        height: out_height as u64,
        upperleft_x: out_ulx,
        upperleft_y: out_uly,
        scale_x: ref_md.scale_x(),
        scale_y: ref_md.scale_y(),
        skew_x: ref_md.skew_x(),
        skew_y: ref_md.skew_y(),
    };

    let out_band_metadata = BandMetadata {
        nodata_value: nodata_value.map(|v| nodata_f64_to_bytes(v, &band_type)),
        storage_type: StorageType::InDb,
        datatype: band_type,
        outdb_url: None,
        outdb_band_id: None,
    };

    Ok((out_metadata, out_band_metadata, band_bytes))
}

fn initialize_band(
    dataset: &Dataset,
    band_type: &BandDataType,
    width: usize,
    height: usize,
    init_value: f64,
) -> Result<()> {
    match band_type {
        BandDataType::UInt8 => initialize_band_t::<u8>(dataset, width, height, init_value as u8),
        BandDataType::Int8 => initialize_band_t::<i8>(dataset, width, height, init_value as i8),
        BandDataType::UInt16 => initialize_band_t::<u16>(dataset, width, height, init_value as u16),
        BandDataType::Int16 => initialize_band_t::<i16>(dataset, width, height, init_value as i16),
        BandDataType::UInt32 => initialize_band_t::<u32>(dataset, width, height, init_value as u32),
        BandDataType::Int32 => initialize_band_t::<i32>(dataset, width, height, init_value as i32),
        BandDataType::UInt64 => initialize_band_t::<u64>(dataset, width, height, init_value as u64),
        BandDataType::Int64 => initialize_band_t::<i64>(dataset, width, height, init_value as i64),
        BandDataType::Float32 => {
            initialize_band_t::<f32>(dataset, width, height, init_value as f32)
        }
        BandDataType::Float64 => initialize_band_t::<f64>(dataset, width, height, init_value),
    }
}

fn initialize_band_t<T: sedona_gdal::raster::GdalType + Copy>(
    dataset: &Dataset,
    width: usize,
    height: usize,
    init_value: T,
) -> Result<()> {
    let band = dataset
        .rasterband(1)
        .map_err(|e| exec_datafusion_err!("Failed to get output band: {}", e))?;

    let values = vec![init_value; width * height];
    let mut buffer = Buffer::new((width, height), values);
    band.write((0, 0), (width, height), &mut buffer)
        .map_err(|e| exec_datafusion_err!("Failed to initialize band: {}", e))?;

    Ok(())
}

fn read_band_as_bytes(
    dataset: &Dataset,
    band_idx: usize,
    width: usize,
    height: usize,
    band_type: &BandDataType,
) -> Result<Vec<u8>> {
    let band = dataset
        .rasterband(band_idx)
        .map_err(|e| exec_datafusion_err!("Failed to get band {}: {}", band_idx, e))?;

    let data = match band_type {
        BandDataType::UInt8 => {
            let buffer = band
                .read_as::<u8>((0, 0), (width, height), (width, height), None)
                .map_err(|e| {
                    exec_datafusion_err!("Failed to read band {} data: {}", band_idx, e)
                })?;
            buffer.data().to_vec()
        }
        BandDataType::Int8 => {
            let buffer = band
                .read_as::<i8>((0, 0), (width, height), (width, height), None)
                .map_err(|e| {
                    exec_datafusion_err!("Failed to read band {} data: {}", band_idx, e)
                })?;
            buffer.data().iter().map(|v| *v as u8).collect()
        }
        BandDataType::UInt16 => {
            let buffer = band
                .read_as::<u16>((0, 0), (width, height), (width, height), None)
                .map_err(|e| {
                    exec_datafusion_err!("Failed to read band {} data: {}", band_idx, e)
                })?;
            buffer.data().iter().flat_map(|v| v.to_le_bytes()).collect()
        }
        BandDataType::Int16 => {
            let buffer = band
                .read_as::<i16>((0, 0), (width, height), (width, height), None)
                .map_err(|e| {
                    exec_datafusion_err!("Failed to read band {} data: {}", band_idx, e)
                })?;
            buffer.data().iter().flat_map(|v| v.to_le_bytes()).collect()
        }
        BandDataType::UInt32 => {
            let buffer = band
                .read_as::<u32>((0, 0), (width, height), (width, height), None)
                .map_err(|e| {
                    exec_datafusion_err!("Failed to read band {} data: {}", band_idx, e)
                })?;
            buffer.data().iter().flat_map(|v| v.to_le_bytes()).collect()
        }
        BandDataType::Int32 => {
            let buffer = band
                .read_as::<i32>((0, 0), (width, height), (width, height), None)
                .map_err(|e| {
                    exec_datafusion_err!("Failed to read band {} data: {}", band_idx, e)
                })?;
            buffer.data().iter().flat_map(|v| v.to_le_bytes()).collect()
        }
        BandDataType::UInt64 => {
            let buffer = band
                .read_as::<u64>((0, 0), (width, height), (width, height), None)
                .map_err(|e| {
                    exec_datafusion_err!("Failed to read band {} data: {}", band_idx, e)
                })?;
            buffer.data().iter().flat_map(|v| v.to_le_bytes()).collect()
        }
        BandDataType::Int64 => {
            let buffer = band
                .read_as::<i64>((0, 0), (width, height), (width, height), None)
                .map_err(|e| {
                    exec_datafusion_err!("Failed to read band {} data: {}", band_idx, e)
                })?;
            buffer.data().iter().flat_map(|v| v.to_le_bytes()).collect()
        }
        BandDataType::Float32 => {
            let buffer = band
                .read_as::<f32>((0, 0), (width, height), (width, height), None)
                .map_err(|e| {
                    exec_datafusion_err!("Failed to read band {} data: {}", band_idx, e)
                })?;
            buffer.data().iter().flat_map(|v| v.to_le_bytes()).collect()
        }
        BandDataType::Float64 => {
            let buffer = band
                .read_as::<f64>((0, 0), (width, height), (width, height), None)
                .map_err(|e| {
                    exec_datafusion_err!("Failed to read band {} data: {}", band_idx, e)
                })?;
            buffer.data().iter().flat_map(|v| v.to_le_bytes()).collect()
        }
    };

    Ok(data)
}

// -----------------------------------------------------------------------------
// ColumnarValue helpers
// -----------------------------------------------------------------------------

fn calc_num_iterations(args: &[ColumnarValue]) -> usize {
    for arg in args {
        if let ColumnarValue::Array(array) = arg {
            return array.len();
        }
    }
    1
}

fn finish_result(args: &[ColumnarValue], out: Arc<dyn Array>) -> Result<ColumnarValue> {
    for arg in args {
        if let ColumnarValue::Array(_) = arg {
            return Ok(ColumnarValue::Array(out));
        }
    }
    Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(&out, 0)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sedona_raster::array::RasterStructArray;

    fn wkb_from_wkt(gdal: &sedona_gdal::gdal::Gdal, wkt: &str) -> Result<Vec<u8>> {
        let geom = gdal.geometry_from_wkt(wkt).unwrap();
        geom.wkb().map_err(|e| exec_datafusion_err!("{e}"))
    }

    fn bytes_to_f64_vec(bytes: &[u8]) -> Vec<f64> {
        bytes
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn test_parse_pixel_type() {
        assert_eq!(parse_pixel_type("D").unwrap(), BandDataType::Float64);
        assert_eq!(parse_pixel_type("f").unwrap(), BandDataType::Float32);
        assert_eq!(parse_pixel_type("I").unwrap(), BandDataType::Int32);
        assert_eq!(parse_pixel_type("S").unwrap(), BandDataType::Int16);
        assert_eq!(parse_pixel_type("US").unwrap(), BandDataType::UInt16);
        assert_eq!(parse_pixel_type("B").unwrap(), BandDataType::UInt8);
        assert_eq!(parse_pixel_type("I8").unwrap(), BandDataType::Int8);
        assert_eq!(parse_pixel_type("U64").unwrap(), BandDataType::UInt64);
        assert_eq!(parse_pixel_type("I64").unwrap(), BandDataType::Int64);
    }

    #[test]
    fn test_rs_as_raster_use_reference_extent() {
        let test_file = sedona_testing::data::test_raster("test4.tiff").unwrap();
        with_gdal(|gdal| {
            let raster_array = crate::utils::load_as_indb_raster(gdal, &test_file)?;
            let raster_struct = RasterStructArray::new(&raster_array);
            let raster = raster_struct.get(0).unwrap();
            let md = raster.metadata();
            let ulx = md.upper_left_x();
            let uly = md.upper_left_y();
            let scale_x = md.scale_x();
            let scale_y = md.scale_y();

            let minx = ulx;
            let maxx = ulx + scale_x;
            let maxy = uly;
            let miny = uly + scale_y;

            let wkt = format!(
                "POLYGON(({minx} {miny}, {minx} {maxy}, {maxx} {maxy}, {maxx} {miny}, {minx} {miny}))"
            );
            let geom_wkb = wkb_from_wkt(gdal, &wkt)?;

            let (out_md, _band_md, out_bytes) = as_raster(
                gdal,
                &geom_wkb,
                &raster,
                BandDataType::Float64,
                false,
                255.0,
                Some(0.0),
                false,
            )?;

            assert_eq!(out_md.width, md.width() as u64);
            assert_eq!(out_md.height, md.height() as u64);
            assert_eq!(out_md.upperleft_x, md.upper_left_x());
            assert_eq!(out_md.upperleft_y, md.upper_left_y());

            let values = bytes_to_f64_vec(&out_bytes);
            assert_eq!(values[0], 255.0);
            Ok::<_, datafusion_common::DataFusionError>(())
        })
        .unwrap();
    }

    #[test]
    fn test_rs_as_raster_use_geometry_extent() {
        let test_file = sedona_testing::data::test_raster("test4.tiff").unwrap();
        with_gdal(|gdal| {
            let raster_array = crate::utils::load_as_indb_raster(gdal, &test_file)?;
            let raster_struct = RasterStructArray::new(&raster_array);
            let raster = raster_struct.get(0).unwrap();
            let md = raster.metadata();

            let ulx = md.upper_left_x();
            let uly = md.upper_left_y();
            let scale_x = md.scale_x();
            let scale_y = md.scale_y();

            let minx = ulx;
            let maxx = ulx + scale_x;
            let maxy = uly;
            let miny = uly + scale_y;

            let wkt = format!(
                "POLYGON(({minx} {miny}, {minx} {maxy}, {maxx} {maxy}, {maxx} {miny}, {minx} {miny}))"
            );
            let geom_wkb = wkb_from_wkt(gdal, &wkt)?;

            let (out_md, _band_md, out_bytes) = as_raster(
                gdal,
                &geom_wkb,
                &raster,
                BandDataType::Float64,
                false,
                255.0,
                Some(0.0),
                true,
            )?;

            assert_eq!(out_md.width, 1);
            assert_eq!(out_md.height, 1);
            assert_eq!(out_md.upperleft_x, md.upper_left_x());
            assert_eq!(out_md.upperleft_y, md.upper_left_y());

            let values = bytes_to_f64_vec(&out_bytes);
            assert_eq!(values.len(), 1);
            assert_eq!(values[0], 255.0);
            Ok::<_, datafusion_common::DataFusionError>(())
        })
        .unwrap();
    }
}
