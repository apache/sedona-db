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

//! Common utilities for sedona-raster-gdal benchmarks

#![allow(dead_code)]

use std::hint::black_box;
use std::sync::Arc;

use arrow_array::{ArrayRef, BinaryArray, StringArray, StructArray};
use arrow_schema::Field;
use datafusion_common::{config::ConfigOptions, ScalarValue};
use datafusion_expr::{ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF};
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::{BandMetadata, RasterMetadata};
use sedona_schema::crs::lnglat;
use sedona_schema::datatypes::SedonaType;
use sedona_schema::raster::{BandDataType, StorageType};

/// Load test rasters from GeoTIFF files using GDAL
///
/// This creates in-db raster arrays by parsing GeoTIFF bytes via GDAL
pub fn load_rasters_from_geotiff(name: &str, count: usize) -> StructArray {
    let test_file = sedona_testing::data::test_raster(name).expect("Failed to find test raster");
    let content = std::fs::read(&test_file).expect("Failed to read test raster file");

    // Create multiple copies of the raster content
    let binary_data: Vec<Option<&[u8]>> = (0..count).map(|_| Some(content.as_slice())).collect();
    let binary_array = BinaryArray::from(binary_data);

    // Use the UDF directly
    let udf: ScalarUDF = sedona_raster_gdal::rs_from_gdal_raster_udf().into();

    let args = ScalarFunctionArgs {
        args: vec![ColumnarValue::Array(Arc::new(binary_array) as ArrayRef)],
        arg_fields: vec![Arc::new(Field::new(
            "binary",
            arrow_schema::DataType::Binary,
            true,
        ))],
        number_rows: count,
        return_field: Arc::new(Field::new(
            "raster",
            sedona_schema::datatypes::RASTER.storage_type().clone(),
            true,
        )),
        config_options: Arc::new(ConfigOptions::default()),
    };

    let result = udf
        .invoke_with_args(args)
        .expect("Failed to invoke RS_FromGDALRaster");

    match result {
        ColumnarValue::Array(array) => array
            .as_any()
            .downcast_ref::<StructArray>()
            .expect("Expected StructArray")
            .clone(),
        ColumnarValue::Scalar(scalar) => {
            if let ScalarValue::Struct(arc_struct) = scalar {
                arc_struct.as_ref().clone()
            } else {
                panic!("Expected Struct scalar");
            }
        }
    }
}

/// Load a single raster and wrap as scalar for benchmarks
pub fn load_raster_as_scalar(name: &str) -> ColumnarValue {
    let raster = load_rasters_from_geotiff(name, 1);
    let scalar = ScalarValue::try_from_array(&raster, 0).expect("Failed to create scalar");
    ColumnarValue::Scalar(scalar)
}

/// Load rasters as array for benchmarks
pub fn load_rasters_as_array(name: &str, count: usize) -> ColumnarValue {
    let raster = load_rasters_from_geotiff(name, count);
    ColumnarValue::Array(Arc::new(raster) as ArrayRef)
}

/// Invoke a UDF with given arguments and consume the result
///
/// For proper type matching, pass arg_types alongside args to specify SedonaTypes
/// (especially important for RASTER types).
pub fn invoke_udf(
    udf: &ScalarUDF,
    args: &[ColumnarValue],
    arg_types: &[SedonaType],
) -> datafusion_common::Result<()> {
    // Get number of rows from first array argument, default to 1
    let number_rows = args
        .iter()
        .find_map(|arg| {
            if let ColumnarValue::Array(array) = arg {
                Some(array.len())
            } else {
                None
            }
        })
        .unwrap_or(1);

    // Create fields for each argument using the provided SedonaTypes
    let arg_fields: Vec<Arc<Field>> = arg_types
        .iter()
        .enumerate()
        .map(|(i, sedona_type)| {
            Arc::new(
                sedona_type
                    .to_storage_field(&format!("arg{}", i), true)
                    .unwrap(),
            )
        })
        .collect();

    let scalar_arguments = args
        .iter()
        .map(|arg| match arg {
            ColumnarValue::Scalar(scalar) => Some(scalar),
            ColumnarValue::Array(_) => None,
        })
        .collect::<Vec<_>>();

    let return_field = udf.return_field_from_args(ReturnFieldArgs {
        arg_fields: &arg_fields,
        scalar_arguments: &scalar_arguments,
    })?;

    let func_args = ScalarFunctionArgs {
        args: args.to_vec(),
        arg_fields,
        number_rows,
        return_field,
        config_options: Arc::new(ConfigOptions::default()),
    };

    let result = udf.invoke_with_args(func_args)?;
    black_box(result);
    Ok(())
}

/// Create a string scalar value
pub fn string_scalar(s: &str) -> ColumnarValue {
    ColumnarValue::Scalar(ScalarValue::Utf8(Some(s.to_string())))
}

/// Create an i32 scalar value
pub fn int32_scalar(v: i32) -> ColumnarValue {
    ColumnarValue::Scalar(ScalarValue::Int32(Some(v)))
}

/// Create an f64 scalar value
pub fn float64_scalar(v: f64) -> ColumnarValue {
    ColumnarValue::Scalar(ScalarValue::Float64(Some(v)))
}

/// Create a string array with repeated values
#[allow(dead_code)]
pub fn string_array(s: &str, count: usize) -> ColumnarValue {
    let strings: Vec<&str> = (0..count).map(|_| s).collect();
    let array = StringArray::from(strings);
    ColumnarValue::Array(Arc::new(array) as ArrayRef)
}

/// Default batch size for benchmarks
#[allow(dead_code)]
pub const BENCH_BATCH_SIZE: usize = 100;

/// Small batch size for slower operations
#[allow(dead_code)]
pub const SMALL_BATCH_SIZE: usize = 10;

/// Tiny batch size for very slow operations (like polygonize, clip)
#[allow(dead_code)]
pub const TINY_BATCH_SIZE: usize = 5;

/// Generate synthetic in-db rasters for benchmarking
///
/// This creates rasters without GDAL dependency, useful for pure-Rust benchmarks
#[allow(dead_code)]
pub fn generate_synthetic_rasters(count: usize, width: usize, height: usize) -> StructArray {
    let mut builder = RasterBuilder::new(count);
    let crs = lnglat().unwrap().to_crs_string();

    for _ in 0..count {
        let raster_metadata = RasterMetadata {
            width: width as u64,
            height: height as u64,
            upperleft_x: 0.0,
            upperleft_y: 0.0,
            scale_x: 1.0,
            scale_y: -1.0,
            skew_x: 0.0,
            skew_y: 0.0,
        };

        builder.start_raster(&raster_metadata, Some(&crs)).unwrap();

        // Add 3 bands (like RGB)
        for _ in 0..3 {
            builder
                .start_band(BandMetadata {
                    datatype: BandDataType::UInt8,
                    nodata_value: Some(vec![0u8]),
                    storage_type: StorageType::InDb,
                    outdb_url: None,
                    outdb_band_id: None,
                })
                .unwrap();

            // Fill with random-ish data
            let pixel_count = width * height;
            let band_data: Vec<u8> = (0..pixel_count).map(|i| (i % 256) as u8).collect();
            builder.band_data_writer().append_value(&band_data);
            builder.finish_band().unwrap();
        }

        builder.finish_raster().unwrap();
    }

    builder.finish().unwrap()
}
