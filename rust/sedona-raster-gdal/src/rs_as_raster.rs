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

//! RS_AsRaster UDF - rasterize a vector geometry onto a raster grid.

use std::{convert::TryFrom, sync::Arc};

use arrow_array::ArrayRef;
use arrow_schema::DataType;
use datafusion_common::cast::{as_boolean_array, as_float64_array, as_string_array};
use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_common::{exec_datafusion_err, exec_err, ScalarValue};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_common::SedonaOptions;
use sedona_expr::{
    item_crs::parse_item_crs_arg_type,
    scalar_udf::{SedonaScalarKernel, SedonaScalarUDF},
};
use sedona_gdal::dataset::Dataset;
use sedona_gdal::gdal::Gdal;
use sedona_gdal::mem::MemDatasetBuilder;
use sedona_gdal::raster::{rasterband::RasterBand, types::Buffer};
use sedona_raster::array::RasterRefImpl;
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::{BandMetadata, RasterMetadata, RasterRef};
use sedona_raster_functions::{
    crs_utils::{align_wkb_to_crs, resolve_crs},
    RasterExecutor,
};
use sedona_schema::datatypes::{SedonaType, RASTER};
use sedona_schema::matchers::ArgMatcher;
use sedona_schema::raster::{BandDataType, StorageType};

use crate::gdal_common::{band_data_type_to_gdal, with_gdal};
use crate::gdal_dataset_provider::{configure_thread_local_options, thread_local_provider};

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
    arg_count: usize,
}

impl SedonaScalarKernel for RsAsRaster {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let Some(first_arg) = args.first() else {
            return Ok(None);
        };
        let (geometry_type, _) = parse_item_crs_arg_type(first_arg)?;
        let mut arg_types = args.to_vec();
        arg_types[0] = geometry_type;

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

        ArgMatcher::new(matchers, RASTER).match_args(&arg_types)
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
        let num_iterations = args
            .iter()
            .find_map(|arg| match arg {
                ColumnarValue::Array(array) => Some(array.len()),
                ColumnarValue::Scalar(_) => None,
            })
            .unwrap_or(1);

        let exec_arg_types = vec![arg_types[1].clone(), arg_types[0].clone()];
        let exec_args = vec![args[1].clone(), args[0].clone()];
        let executor =
            RasterExecutor::new_with_num_iterations(&exec_arg_types, &exec_args, num_iterations);

        let pixel_type_array = args[2]
            .clone()
            .cast_to(&DataType::Utf8, None)?
            .into_array(num_iterations)?;
        let pixel_type_array = as_string_array(&pixel_type_array)?;

        let all_touched_array = if self.arg_count >= 4 {
            args[3]
                .clone()
                .cast_to(&DataType::Boolean, None)?
                .into_array(num_iterations)?
        } else {
            ScalarValue::Boolean(Some(false)).to_array_of_size(num_iterations)?
        };
        let all_touched_array = as_boolean_array(&all_touched_array)?;

        let burn_value_array = if self.arg_count >= 5 {
            args[4]
                .clone()
                .cast_to(&DataType::Float64, None)?
                .into_array(num_iterations)?
        } else {
            ScalarValue::Float64(Some(1.0)).to_array_of_size(num_iterations)?
        };
        let burn_value_array = as_float64_array(&burn_value_array)?;

        let nodata_value_array = if self.arg_count >= 6 {
            args[5]
                .clone()
                .cast_to(&DataType::Float64, None)?
                .into_array(num_iterations)?
        } else {
            ScalarValue::Float64(None).to_array_of_size(num_iterations)?
        };
        let nodata_value_array = as_float64_array(&nodata_value_array)?;

        let use_geom_extent_array = if self.arg_count >= 7 {
            args[6]
                .clone()
                .cast_to(&DataType::Boolean, None)?
                .into_array(num_iterations)?
        } else {
            ScalarValue::Boolean(Some(true)).to_array_of_size(num_iterations)?
        };
        let use_geom_extent_array = as_boolean_array(&use_geom_extent_array)?;

        let mut builder = RasterBuilder::new(num_iterations);

        let mut pixel_type_iter = pixel_type_array.iter();
        let mut all_touched_iter = all_touched_array.iter();
        let mut burn_value_iter = burn_value_array.iter();
        let mut nodata_value_iter = nodata_value_array.iter();
        let mut use_geom_extent_iter = use_geom_extent_array.iter();

        let default_options = SedonaOptions::default();
        let sedona_options = config_options
            .and_then(|c| c.extensions.get::<SedonaOptions>())
            .unwrap_or(&default_options);
        let engine = sedona_options.runtime.crs_engine();

        with_gdal(|gdal| {
            configure_thread_local_options(gdal, config_options)?;

            executor.execute_raster_wkb_crs_void(|raster_opt, geom_opt, geom_crs| {
                let pixel_type_opt = pixel_type_iter.next().unwrap();
                let all_touched_opt = all_touched_iter.next().unwrap().unwrap_or(false);
                let burn_value_opt = burn_value_iter.next().unwrap().unwrap_or(1.0);
                let nodata_value_opt = nodata_value_iter.next().unwrap();
                let use_geometry_extent_opt = use_geom_extent_iter.next().unwrap().unwrap_or(true);

                let raster = match raster_opt {
                    Some(raster) => raster,
                    None => {
                        builder.append_null()?;
                        return Ok(());
                    }
                };
                let geom_wkb = match geom_opt {
                    Some(geom_wkb) => geom_wkb,
                    None => {
                        builder.append_null()?;
                        return Ok(());
                    }
                };
                let pixel_type = match pixel_type_opt {
                    Some(pixel_type) => pixel_type,
                    None => {
                        builder.append_null()?;
                        return Ok(());
                    }
                };

                let band_type = parse_pixel_type(pixel_type)?;
                let raster_crs = resolve_crs(raster.crs())?;
                let geom_wkb = align_wkb_to_crs(
                    geom_wkb,
                    geom_crs,
                    raster_crs.as_deref(),
                    "geometry",
                    "reference raster",
                    engine.as_ref(),
                )?;

                let (out_metadata, out_band_metadata, out_band_bytes) = as_raster(
                    gdal,
                    &geom_wkb,
                    raster,
                    band_type,
                    all_touched_opt,
                    burn_value_opt,
                    nodata_value_opt,
                    use_geometry_extent_opt,
                )
                .map_err(|e| exec_datafusion_err!("RS_AsRaster failed: {}", e))?;

                builder
                    .start_raster(&out_metadata, raster.crs())
                    .map_err(|e| exec_datafusion_err!("Failed to start output raster: {}", e))?;
                builder.start_band(out_band_metadata).map_err(|e| {
                    exec_datafusion_err!("Failed to start output raster band: {}", e)
                })?;
                builder.band_data_writer().append_value(out_band_bytes);
                builder.finish_band().map_err(|e| {
                    exec_datafusion_err!("Failed to finish output raster band: {}", e)
                })?;
                builder
                    .finish_raster()
                    .map_err(|e| exec_datafusion_err!("Failed to finish output raster: {}", e))?;

                Ok(())
            })
        })?;

        let result: ArrayRef = Arc::new(builder.finish()?);
        executor.finish(result)
    }
}

fn parse_pixel_type(value: &str) -> Result<BandDataType> {
    match value.trim().to_ascii_lowercase().as_str() {
        "d" | "float64" => Ok(BandDataType::Float64),
        "f" | "float32" => Ok(BandDataType::Float32),
        "i" | "int32" => Ok(BandDataType::Int32),
        "ui" | "uint32" => Ok(BandDataType::UInt32),
        "s" | "int16" => Ok(BandDataType::Int16),
        "us" | "uint16" => Ok(BandDataType::UInt16),
        "b" | "uint8" => Ok(BandDataType::UInt8),
        "i8" | "int8" => Ok(BandDataType::Int8),
        "u64" | "uint64" => Ok(BandDataType::UInt64),
        "i64" | "int64" => Ok(BandDataType::Int64),
        other => exec_err!(
            "Unsupported pixelType: {} (expected one of D/F/I/UI/S/US/B/I8/U64/I64 or int8/uint8/int16/uint16/int32/uint32/int64/uint64/float32/float64)",
            other
        ),
    }
}

#[derive(Clone, Copy, Debug)]
enum TypedBandValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    UInt64(u64),
    Int64(i64),
    Float32(f32),
    Float64(f64),
}

impl TypedBandValue {
    fn metadata_bytes(self) -> Vec<u8> {
        match self {
            Self::UInt8(value) => vec![value],
            Self::Int8(value) => value.to_le_bytes().to_vec(),
            Self::UInt16(value) => value.to_le_bytes().to_vec(),
            Self::Int16(value) => value.to_le_bytes().to_vec(),
            Self::UInt32(value) => value.to_le_bytes().to_vec(),
            Self::Int32(value) => value.to_le_bytes().to_vec(),
            Self::UInt64(value) => value.to_le_bytes().to_vec(),
            Self::Int64(value) => value.to_le_bytes().to_vec(),
            Self::Float32(value) => value.to_le_bytes().to_vec(),
            Self::Float64(value) => value.to_le_bytes().to_vec(),
        }
    }
}

fn checked_floor_to_i64(value: f64, label: &str) -> Result<i64> {
    checked_float_to_i64(value.floor(), label)
}

fn checked_ceil_to_i64(value: f64, label: &str) -> Result<i64> {
    checked_float_to_i64(value.ceil(), label)
}

fn checked_float_to_i64(value: f64, label: &str) -> Result<i64> {
    if !value.is_finite() {
        return exec_err!("{} must be finite", label);
    }

    if value < i64::MIN as f64 || value > i64::MAX as f64 {
        return exec_err!("{} is out of range for i64: {}", label, value);
    }

    Ok(value as i64)
}

fn cast_f64_to_band_value(
    value: f64,
    band_type: BandDataType,
    role: &str,
) -> Result<TypedBandValue> {
    if !value.is_finite() && !matches!(band_type, BandDataType::Float32 | BandDataType::Float64) {
        return exec_err!("{} must be finite for {:?}: {}", role, band_type, value);
    }

    match band_type {
        BandDataType::UInt8 => Ok(TypedBandValue::UInt8(checked_integral_cast::<u8>(
            value, role, band_type,
        )?)),
        BandDataType::Int8 => Ok(TypedBandValue::Int8(checked_integral_cast::<i8>(
            value, role, band_type,
        )?)),
        BandDataType::UInt16 => Ok(TypedBandValue::UInt16(checked_integral_cast::<u16>(
            value, role, band_type,
        )?)),
        BandDataType::Int16 => Ok(TypedBandValue::Int16(checked_integral_cast::<i16>(
            value, role, band_type,
        )?)),
        BandDataType::UInt32 => Ok(TypedBandValue::UInt32(checked_integral_cast::<u32>(
            value, role, band_type,
        )?)),
        BandDataType::Int32 => Ok(TypedBandValue::Int32(checked_integral_cast::<i32>(
            value, role, band_type,
        )?)),
        BandDataType::UInt64 => Ok(TypedBandValue::UInt64(checked_integral_cast::<u64>(
            value, role, band_type,
        )?)),
        BandDataType::Int64 => Ok(TypedBandValue::Int64(checked_integral_cast::<i64>(
            value, role, band_type,
        )?)),
        BandDataType::Float32 => {
            let casted = value as f32;
            if value.is_finite() && !casted.is_finite() {
                return exec_err!("{} is out of range for {:?}: {}", role, band_type, value);
            }
            Ok(TypedBandValue::Float32(casted))
        }
        BandDataType::Float64 => Ok(TypedBandValue::Float64(value)),
    }
}

fn checked_integral_cast<T>(value: f64, role: &str, band_type: BandDataType) -> Result<T>
where
    T: TryFrom<i128>,
    <T as TryFrom<i128>>::Error: std::fmt::Debug,
{
    if !value.is_finite() {
        return exec_err!("{} must be finite for {:?}: {}", role, band_type, value);
    }
    if value.fract() != 0.0 {
        return exec_err!("{} must be an integer for {:?}: {}", role, band_type, value);
    }

    let integer = value as i128;
    T::try_from(integer).map_err(|_| {
        datafusion_common::DataFusionError::Execution(format!(
            "{} is out of range for {:?}: {}",
            role, band_type, value
        ))
    })
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

    let geometry = gdal
        .geometry_from_wkb(geom_wkb)
        .map_err(|e| exec_datafusion_err!("Failed to parse geometry from WKB: {}", e))?;

    let (out_width, out_height, out_ulx, out_uly) = if use_geometry_extent {
        let env = geometry.envelope();
        let ulx = ref_md.upper_left_x();
        let uly = ref_md.upper_left_y();
        let scale_x = ref_md.scale_x();
        let scale_y = ref_md.scale_y();

        if scale_x == 0.0 || scale_y == 0.0 {
            return exec_err!("Reference raster has zero scale");
        }

        let start_col = checked_floor_to_i64((env.MinX - ulx) / scale_x, "start_col")?;
        let end_col_excl = checked_ceil_to_i64((env.MaxX - ulx) / scale_x, "end_col_excl")?;
        let start_row = checked_floor_to_i64((env.MaxY - uly) / scale_y, "start_row")?;
        let end_row_excl = checked_ceil_to_i64((env.MinY - uly) / scale_y, "end_row_excl")?;

        let width = usize::try_from((end_col_excl - start_col).max(0)).map_err(|_| {
            datafusion_common::DataFusionError::Execution(
                "Geometry extent width is out of range for usize".to_string(),
            )
        })?;
        let height = usize::try_from((end_row_excl - start_row).max(0)).map_err(|_| {
            datafusion_common::DataFusionError::Execution(
                "Geometry extent height is out of range for usize".to_string(),
            )
        })?;

        if width == 0 || height == 0 {
            return exec_err!("Geometry extent produced an empty raster");
        }

        (
            width,
            height,
            ulx + (start_col as f64) * scale_x,
            uly + (start_row as f64) * scale_y,
        )
    } else {
        (
            ref_md.width() as usize,
            ref_md.height() as usize,
            ref_md.upper_left_x(),
            ref_md.upper_left_y(),
        )
    };

    let out_dataset = MemDatasetBuilder::create(
        gdal,
        out_width,
        out_height,
        1,
        band_data_type_to_gdal(&band_type),
    )
    .map_err(|e| exec_datafusion_err!("Failed to create dataset: {}", e))?;

    out_dataset
        .set_geo_transform(&[
            out_ulx,
            ref_md.scale_x(),
            ref_md.skew_x(),
            out_uly,
            ref_md.skew_y(),
            ref_md.scale_y(),
        ])
        .map_err(|e| exec_datafusion_err!("Failed to set geotransform: {}", e))?;

    let provider = thread_local_provider(gdal)
        .map_err(|e| exec_datafusion_err!("Failed to init GDAL provider: {}", e))?;
    let ref_raster_ds = provider
        .raster_ref_to_gdal(reference_raster)
        .map_err(|e| exec_datafusion_err!("Failed to create GDAL dataset: {}", e))?;
    if let Ok(srs) = ref_raster_ds.as_dataset().spatial_ref() {
        out_dataset
            .set_spatial_ref(&srs)
            .map_err(|e| exec_datafusion_err!("Failed to set spatial reference: {}", e))?;
    }

    let init_value =
        cast_f64_to_band_value(nodata_value.unwrap_or(0.0), band_type, "initial fill value")?;
    initialize_band(&out_dataset, out_width, out_height, init_value)?;

    if let Some(nodata) = nodata_value {
        let nodata = cast_f64_to_band_value(nodata, band_type, "nodata value")?;
        let band = out_dataset
            .rasterband(1)
            .map_err(|e| exec_datafusion_err!("Failed to get output band: {}", e))?;
        set_band_nodata(&band, nodata)?;
    }

    gdal.rasterize_affine(&out_dataset, &[1], &[geometry], &[burn_value], all_touched)
        .map_err(|e| exec_datafusion_err!("Failed to rasterize geometry: {}", e))?;

    let band = out_dataset
        .rasterband(1)
        .map_err(|e| exec_datafusion_err!("Failed to get output band: {}", e))?;
    let band_bytes = band
        .read_as_bytes(
            (0, 0),
            (out_width, out_height),
            (out_width, out_height),
            None,
        )
        .map_err(|e| exec_datafusion_err!("Failed to read band data: {}", e))?;

    Ok((
        RasterMetadata {
            width: out_width as i64,
            height: out_height as i64,
            upperleft_x: out_ulx,
            upperleft_y: out_uly,
            scale_x: ref_md.scale_x(),
            scale_y: ref_md.scale_y(),
            skew_x: ref_md.skew_x(),
            skew_y: ref_md.skew_y(),
        },
        BandMetadata {
            nodata_value: nodata_value
                .map(|value| cast_f64_to_band_value(value, band_type, "nodata value"))
                .transpose()?
                .map(TypedBandValue::metadata_bytes),
            storage_type: StorageType::InDb,
            datatype: band_type,
            outdb_url: None,
            outdb_band_id: None,
        },
        band_bytes,
    ))
}

fn initialize_band(
    dataset: &Dataset,
    width: usize,
    height: usize,
    init_value: TypedBandValue,
) -> Result<()> {
    match init_value {
        TypedBandValue::UInt8(value) => initialize_band_t::<u8>(dataset, width, height, value),
        TypedBandValue::Int8(value) => initialize_band_t::<i8>(dataset, width, height, value),
        TypedBandValue::UInt16(value) => initialize_band_t::<u16>(dataset, width, height, value),
        TypedBandValue::Int16(value) => initialize_band_t::<i16>(dataset, width, height, value),
        TypedBandValue::UInt32(value) => initialize_band_t::<u32>(dataset, width, height, value),
        TypedBandValue::Int32(value) => initialize_band_t::<i32>(dataset, width, height, value),
        TypedBandValue::UInt64(value) => initialize_band_t::<u64>(dataset, width, height, value),
        TypedBandValue::Int64(value) => initialize_band_t::<i64>(dataset, width, height, value),
        TypedBandValue::Float32(value) => initialize_band_t::<f32>(dataset, width, height, value),
        TypedBandValue::Float64(value) => initialize_band_t::<f64>(dataset, width, height, value),
    }
}

fn set_band_nodata(band: &RasterBand<'_>, nodata: TypedBandValue) -> Result<()> {
    match nodata {
        TypedBandValue::UInt64(value) => band
            .set_no_data_value_u64(Some(value))
            .map_err(|e| exec_datafusion_err!("Failed to set nodata value: {}", e)),
        TypedBandValue::Int64(value) => band
            .set_no_data_value_i64(Some(value))
            .map_err(|e| exec_datafusion_err!("Failed to set nodata value: {}", e)),
        TypedBandValue::UInt8(value) => band
            .set_no_data_value(Some(value as f64))
            .map_err(|e| exec_datafusion_err!("Failed to set nodata value: {}", e)),
        TypedBandValue::Int8(value) => band
            .set_no_data_value(Some(value as f64))
            .map_err(|e| exec_datafusion_err!("Failed to set nodata value: {}", e)),
        TypedBandValue::UInt16(value) => band
            .set_no_data_value(Some(value as f64))
            .map_err(|e| exec_datafusion_err!("Failed to set nodata value: {}", e)),
        TypedBandValue::Int16(value) => band
            .set_no_data_value(Some(value as f64))
            .map_err(|e| exec_datafusion_err!("Failed to set nodata value: {}", e)),
        TypedBandValue::UInt32(value) => band
            .set_no_data_value(Some(value as f64))
            .map_err(|e| exec_datafusion_err!("Failed to set nodata value: {}", e)),
        TypedBandValue::Int32(value) => band
            .set_no_data_value(Some(value as f64))
            .map_err(|e| exec_datafusion_err!("Failed to set nodata value: {}", e)),
        TypedBandValue::Float32(value) => band
            .set_no_data_value(Some(value as f64))
            .map_err(|e| exec_datafusion_err!("Failed to set nodata value: {}", e)),
        TypedBandValue::Float64(value) => band
            .set_no_data_value(Some(value))
            .map_err(|e| exec_datafusion_err!("Failed to set nodata value: {}", e)),
    }
}

fn initialize_band_t<T: sedona_gdal::raster::types::GdalType + Copy>(
    dataset: &Dataset,
    width: usize,
    height: usize,
    init_value: T,
) -> Result<()> {
    let band = dataset
        .rasterband(1)
        .map_err(|e| exec_datafusion_err!("Failed to get output band: {}", e))?;
    let mut buffer = Buffer::new((width, height), vec![init_value; width * height]);
    band.write((0, 0), (width, height), &mut buffer)
        .map_err(|e| exec_datafusion_err!("Failed to initialize band: {}", e))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{BooleanArray, Float64Array, StringArray};
    use datafusion_expr::{ScalarUDF, ScalarUDFImpl};
    use sedona_raster::array::RasterStructArray;
    use sedona_schema::crs::deserialize_crs;
    use sedona_schema::datatypes::Edges;
    use sedona_schema::datatypes::RASTER;
    use sedona_schema::datatypes::WKB_GEOMETRY;
    use sedona_schema::raster::BandDataType;
    use sedona_testing::raster_spec::{assert_rasters_equal, RasterSpec};
    use sedona_testing::{
        create::{create_array, create_array_item_crs, make_wkb},
        testers::ScalarUdfTester,
    };

    use super::*;
    use crate::gdal_common::bytes_to_f64;

    fn reference_raster_spec() -> RasterSpec {
        RasterSpec::d2(4, 3)
            .transform([10.0, 2.0, 0.0, 20.0, 0.0, -2.0])
            .crs(Some("EPSG:4326"))
            .band_values(&[0u8; 12])
            .nodata(0u8)
    }

    fn load_reference_raster() -> RasterMetadata {
        let raster_array = reference_raster_spec().build();
        let raster_struct = RasterStructArray::try_new(&raster_array).unwrap();
        raster_struct.get(0).unwrap().metadata()
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
        assert_eq!(parse_pixel_type("uint32").unwrap(), BandDataType::UInt32);
        assert_eq!(parse_pixel_type("U64").unwrap(), BandDataType::UInt64);
        assert_eq!(parse_pixel_type("I64").unwrap(), BandDataType::Int64);
        assert_eq!(parse_pixel_type("float64").unwrap(), BandDataType::Float64);
        assert_eq!(parse_pixel_type("float32").unwrap(), BandDataType::Float32);
        assert_eq!(parse_pixel_type("int16").unwrap(), BandDataType::Int16);
        assert_eq!(parse_pixel_type("uint16").unwrap(), BandDataType::UInt16);
        assert_eq!(parse_pixel_type("int32").unwrap(), BandDataType::Int32);
        assert_eq!(parse_pixel_type("uint8").unwrap(), BandDataType::UInt8);
    }

    #[test]
    fn test_checked_integral_nodata_cast_errors() {
        let err = cast_f64_to_band_value(1.5, BandDataType::UInt8, "nodata value").unwrap_err();
        assert!(err.to_string().contains("must be an integer"));

        let err = cast_f64_to_band_value(-1.0, BandDataType::UInt8, "nodata value").unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn test_checked_extent_conversion_errors_on_infinite_values() {
        let err = checked_floor_to_i64(f64::INFINITY, "start_col").unwrap_err();
        assert!(err.to_string().contains("must be finite"));
    }

    #[test]
    fn test_float32_nodata_overflow_errors() {
        let err =
            cast_f64_to_band_value(f64::MAX, BandDataType::Float32, "nodata value").unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn test_rs_as_raster_use_reference_extent() {
        let raster_array = reference_raster_spec().build();
        with_gdal(|gdal| {
            let raster_struct = RasterStructArray::try_new(&raster_array).unwrap();
            let raster = raster_struct.get(0).unwrap();
            let md = raster.metadata();

            let wkt = format!(
                "POLYGON(({x0} {y1}, {x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}))",
                x0 = md.upper_left_x(),
                x1 = md.upper_left_x() + md.scale_x(),
                y0 = md.upper_left_y(),
                y1 = md.upper_left_y() + md.scale_y(),
            );
            let geom_wkb = make_wkb(&wkt);

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

            assert_eq!(out_md.width, md.width());
            assert_eq!(out_md.height, md.height());
            assert_eq!(out_md.upperleft_x, md.upper_left_x());
            assert_eq!(out_md.upperleft_y, md.upper_left_y());
            assert_eq!(
                bytes_to_f64(&out_bytes[..8], &BandDataType::Float64).unwrap(),
                255.0
            );
            Ok::<_, datafusion_common::DataFusionError>(())
        })
        .unwrap();
    }

    #[test]
    fn test_rs_as_raster_use_geometry_extent() {
        let raster_array = reference_raster_spec().build();
        with_gdal(|gdal| {
            let raster_struct = RasterStructArray::try_new(&raster_array).unwrap();
            let raster = raster_struct.get(0).unwrap();
            let md = raster.metadata();

            let wkt = format!(
                "POLYGON(({x0} {y1}, {x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}))",
                x0 = md.upper_left_x(),
                x1 = md.upper_left_x() + md.scale_x(),
                y0 = md.upper_left_y(),
                y1 = md.upper_left_y() + md.scale_y(),
            );
            let geom_wkb = make_wkb(&wkt);

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
            assert_eq!(
                bytes_to_f64(&out_bytes, &BandDataType::Float64).unwrap(),
                255.0
            );
            Ok::<_, datafusion_common::DataFusionError>(())
        })
        .unwrap();
    }

    #[test]
    fn test_rs_as_raster_udf_name() {
        assert_eq!(rs_as_raster_udf().name(), "rs_asraster");
    }

    #[test]
    fn test_rs_as_raster_udf_batch() {
        let metadata = load_reference_raster();
        let geometry_type = SedonaType::Wkb(Edges::Planar, deserialize_crs("EPSG:4326").unwrap());
        let geom = format!(
            "POLYGON(({x0} {y1}, {x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}))",
            x0 = metadata.upper_left_x(),
            x1 = metadata.upper_left_x() + metadata.scale_x(),
            y0 = metadata.upper_left_y(),
            y1 = metadata.upper_left_y() + metadata.scale_y(),
        );

        let udf: ScalarUDF = rs_as_raster_udf().into();
        let tester = ScalarUdfTester::new(
            udf,
            vec![
                geometry_type.clone(),
                RASTER,
                SedonaType::Arrow(DataType::Utf8),
                SedonaType::Arrow(DataType::Boolean),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Boolean),
            ],
        );

        let result = tester
            .invoke_arrays(vec![
                create_array(&[Some(geom.as_str())], &geometry_type),
                Arc::new(reference_raster_spec().build()),
                Arc::new(StringArray::from(vec!["D"])),
                Arc::new(BooleanArray::from(vec![false])),
                Arc::new(Float64Array::from(vec![255.0])),
                Arc::new(Float64Array::from(vec![0.0])),
                Arc::new(BooleanArray::from(vec![true])),
            ])
            .unwrap();

        let expected = RasterSpec::d2(1, 1)
            .transform([10.0, 2.0, 0.0, 20.0, 0.0, -2.0])
            .crs(Some("EPSG:4326"))
            .band_values(&[255.0f64])
            .nodata(0.0f64);
        assert_rasters_equal(&result, &[Some(expected)]);
    }

    #[test]
    fn test_rs_as_raster_udf_rejects_one_sided_crs() {
        let geometry_with_crs =
            SedonaType::Wkb(Edges::Planar, deserialize_crs("EPSG:4326").unwrap());
        let udf: ScalarUDF = rs_as_raster_udf().into();

        let tester = ScalarUdfTester::new(
            udf.clone(),
            vec![WKB_GEOMETRY, RASTER, SedonaType::Arrow(DataType::Utf8)],
        );
        let err = tester
            .invoke_arrays(vec![
                create_array(
                    &[Some("POLYGON((10 20, 10 18, 12 18, 12 20, 10 20))")],
                    &WKB_GEOMETRY,
                ),
                Arc::new(reference_raster_spec().build()),
                Arc::new(StringArray::from(vec!["D"])),
            ])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("reference raster has a CRS but the geometry does not"),
            "unexpected error: {err}"
        );

        let tester = ScalarUdfTester::new(
            udf,
            vec![
                geometry_with_crs.clone(),
                RASTER,
                SedonaType::Arrow(DataType::Utf8),
            ],
        );
        let err = tester
            .invoke_arrays(vec![
                create_array(
                    &[Some("POLYGON((0 0, 0 1, 1 1, 1 0, 0 0))")],
                    &geometry_with_crs,
                ),
                Arc::new(RasterSpec::d2(1, 1).crs(None).band_values(&[0u8]).build()),
                Arc::new(StringArray::from(vec!["D"])),
            ])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("geometry has a CRS but the reference raster does not"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_rs_as_raster_udf_reprojects_type_level_crs_geometry() {
        let geometry_type = SedonaType::Wkb(Edges::Planar, deserialize_crs("OGC:CRS84").unwrap());
        let udf: ScalarUDF = rs_as_raster_udf().into();
        let mut tester = ScalarUdfTester::new(
            udf,
            vec![
                geometry_type.clone(),
                RASTER,
                SedonaType::Arrow(DataType::Utf8),
                SedonaType::Arrow(DataType::Boolean),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Boolean),
            ],
        );
        tester.sedona_options_mut().runtime = tester
            .sedona_options()
            .runtime
            .with_crs_engine(Arc::new(sedona_proj::transform::LazyProjEngine));
        let reference = RasterSpec::d2(1, 1)
            .transform([111_000.0, 1_000.0, 0.0, 112_000.0, 0.0, -1_000.0])
            .crs(Some("EPSG:3857"))
            .band_values(&[0u8]);

        let result = tester
            .invoke_arrays(vec![
                create_array(
                    &[Some(
                        "POLYGON((0.99 0.99, 0.99 1.01, 1.01 1.01, 1.01 0.99, 0.99 0.99))",
                    )],
                    &geometry_type,
                ),
                Arc::new(reference.build()),
                Arc::new(StringArray::from(vec!["D"])),
                Arc::new(BooleanArray::from(vec![false])),
                Arc::new(Float64Array::from(vec![1.0])),
                Arc::new(Float64Array::from(vec![None])),
                Arc::new(BooleanArray::from(vec![false])),
            ])
            .unwrap();

        let expected = RasterSpec::d2(1, 1)
            .transform([111_000.0, 1_000.0, 0.0, 112_000.0, 0.0, -1_000.0])
            .crs(Some("EPSG:3857"))
            .band_values(&[1.0f64]);
        assert_rasters_equal(&result, &[Some(expected)]);
    }

    #[test]
    fn test_rs_as_raster_udf_reprojects_item_crs_geometry() {
        let udf: ScalarUDF = rs_as_raster_udf().into();
        let mut tester = ScalarUdfTester::new(
            udf,
            vec![
                SedonaType::new_item_crs(&WKB_GEOMETRY).unwrap(),
                RASTER,
                SedonaType::Arrow(DataType::Utf8),
                SedonaType::Arrow(DataType::Boolean),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Boolean),
            ],
        );
        tester.sedona_options_mut().runtime = tester
            .sedona_options()
            .runtime
            .with_crs_engine(Arc::new(sedona_proj::transform::LazyProjEngine));
        let reference = RasterSpec::d2(1, 1)
            .transform([111_000.0, 1_000.0, 0.0, 112_000.0, 0.0, -1_000.0])
            .crs(Some("EPSG:3857"))
            .band_values(&[0u8]);

        let result = tester
            .invoke_arrays(vec![
                create_array_item_crs(
                    &[Some(
                        "POLYGON((0.99 0.99, 0.99 1.01, 1.01 1.01, 1.01 0.99, 0.99 0.99))",
                    )],
                    [Some("OGC:CRS84")],
                    &WKB_GEOMETRY,
                ),
                Arc::new(reference.build()),
                Arc::new(StringArray::from(vec!["D"])),
                Arc::new(BooleanArray::from(vec![false])),
                Arc::new(Float64Array::from(vec![1.0])),
                Arc::new(Float64Array::from(vec![None])),
                Arc::new(BooleanArray::from(vec![false])),
            ])
            .unwrap();

        let expected = RasterSpec::d2(1, 1)
            .transform([111_000.0, 1_000.0, 0.0, 112_000.0, 0.0, -1_000.0])
            .crs(Some("EPSG:3857"))
            .band_values(&[1.0f64]);
        assert_rasters_equal(&result, &[Some(expected)]);
    }

    #[test]
    fn test_rs_as_raster_udf_accepts_item_crs_geometry() {
        let metadata = load_reference_raster();
        let geom = format!(
            "POLYGON(({x0} {y1}, {x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}))",
            x0 = metadata.upper_left_x(),
            x1 = metadata.upper_left_x() + metadata.scale_x(),
            y0 = metadata.upper_left_y(),
            y1 = metadata.upper_left_y() + metadata.scale_y(),
        );

        let udf: ScalarUDF = rs_as_raster_udf().into();
        let tester = ScalarUdfTester::new(
            udf,
            vec![
                SedonaType::new_item_crs(&WKB_GEOMETRY).unwrap(),
                RASTER,
                SedonaType::Arrow(DataType::Utf8),
            ],
        );

        let result = tester
            .invoke_arrays(vec![
                create_array_item_crs(&[Some(geom.as_str())], [Some("EPSG:4326")], &WKB_GEOMETRY),
                Arc::new(reference_raster_spec().build()),
                Arc::new(StringArray::from(vec!["D"])),
            ])
            .unwrap();

        let expected = RasterSpec::d2(1, 1)
            .transform([10.0, 2.0, 0.0, 20.0, 0.0, -2.0])
            .crs(Some("EPSG:4326"))
            .band_values(&[1.0f64]);
        assert_rasters_equal(&result, &[Some(expected)]);
    }
}
