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

//! RS_MapAlgebra UDF - Apply a map algebra expression on raster(s)
//!
//! This function evaluates a mathematical expression for each pixel in the input raster(s)
//! and produces an output raster. The expression can reference input raster bands using
//! `rast[band_index]` syntax (or `rast0[band_index]` and `rast1[band_index]` for two-raster
//! operations).
//!
//! # Expression Syntax
//!
//! The expression evaluator supports standard mathematical operations:
//! - Arithmetic: `+`, `-`, `*`, `/`, `%` (modulo), `^` (power)
//! - Comparison: `==`, `!=`, `<`, `<=`, `>`, `>=`
//! - Logic: `&&`, `||`, `!`
//! - Functions: `min`, `max`, `abs`, `sqrt`, `sin`, `cos`, `tan`, `ln`, `log`, `exp`, `floor`, `ceil`, `round`
//! - Conditionals: `if(condition, true_value, false_value)`
//!
//! # Variables
//!
//! For single-raster operations:
//! - `rast` or `rast0`, `rast1`, ..., `rastN`: Band values (where N is band index, 0-based)
//!
//! For two-raster operations:
//! - `rast0_0`, `rast0_1`, ...: First raster's band values
//! - `rast1_0`, `rast1_1`, ...: Second raster's band values
//!
//! Additional variables:
//! - `x`: Current pixel column (0-based)
//! - `y`: Current pixel row (0-based)
//! - `width`: Raster width
//! - `height`: Raster height

use std::sync::Arc;

use datafusion_common::cast::{as_float64_array, as_int32_array, as_string_array};
use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_common::{exec_datafusion_err, exec_err, ScalarValue};
use datafusion_expr::{ColumnarValue, Volatility};
use evalexpr::{build_operator_tree, ContextWithMutableVariables, HashMapContext, Value};
use sedona_gdal::gdal::Gdal;

use arrow_schema::DataType;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::array::RasterRefImpl;
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::{BandMetadata, RasterMetadata, RasterRef};
use sedona_raster_functions::RasterExecutor;
use sedona_schema::datatypes::{SedonaType, RASTER};
use sedona_schema::matchers::ArgMatcher;
use sedona_schema::raster::{BandDataType, StorageType};

use crate::gdal_common::nodata_f64_to_bytes;
use crate::gdal_common::with_gdal;
use crate::gdal_dataset_provider::configure_thread_local_options;
use crate::raster_band_reader::RasterBandReader;

/// RS_MapAlgebra() scalar UDF implementation
///
/// Apply a map algebra expression on raster(s)
pub fn rs_map_algebra_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_mapalgebra",
        vec![
            // Single raster variants
            Arc::new(RsMapAlgebra {
                two_raster: false,
                with_nodata: false,
                with_num_bands: false,
            }),
            Arc::new(RsMapAlgebra {
                two_raster: false,
                with_nodata: true,
                with_num_bands: false,
            }),
            Arc::new(RsMapAlgebra {
                two_raster: false,
                with_nodata: true,
                with_num_bands: true,
            }),
            // Two raster variants
            Arc::new(RsMapAlgebra {
                two_raster: true,
                with_nodata: false,
                with_num_bands: false,
            }),
            Arc::new(RsMapAlgebra {
                two_raster: true,
                with_nodata: true,
                with_num_bands: false,
            }),
            Arc::new(RsMapAlgebra {
                two_raster: true,
                with_nodata: true,
                with_num_bands: true,
            }),
        ],
        Volatility::Immutable,
    )
}

/// Kernel implementation for RS_MapAlgebra
#[derive(Debug)]
struct RsMapAlgebra {
    two_raster: bool,
    with_nodata: bool,
    with_num_bands: bool,
}

impl SedonaScalarKernel for RsMapAlgebra {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matchers = if self.two_raster {
            if self.with_num_bands {
                vec![
                    ArgMatcher::is_raster(),
                    ArgMatcher::is_raster(),
                    ArgMatcher::is_string(),
                    ArgMatcher::is_string(),
                    ArgMatcher::is_numeric(),
                    ArgMatcher::is_integer(),
                ]
            } else if self.with_nodata {
                vec![
                    ArgMatcher::is_raster(),
                    ArgMatcher::is_raster(),
                    ArgMatcher::is_string(),
                    ArgMatcher::is_string(),
                    ArgMatcher::is_numeric(),
                ]
            } else {
                vec![
                    ArgMatcher::is_raster(),
                    ArgMatcher::is_raster(),
                    ArgMatcher::is_string(),
                    ArgMatcher::is_string(),
                ]
            }
        } else if self.with_num_bands {
            vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_string(),
                ArgMatcher::is_string(),
                ArgMatcher::is_numeric(),
                ArgMatcher::is_integer(),
            ]
        } else if self.with_nodata {
            vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_string(),
                ArgMatcher::is_string(),
                ArgMatcher::is_numeric(),
            ]
        } else {
            vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_string(),
                ArgMatcher::is_string(),
            ]
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
        let executor = RasterExecutor::new(arg_types, args);
        let num_iterations = executor.num_iterations();

        // Parse arguments based on signature
        let (pixel_type_idx, script_idx, nodata_idx, num_bands_idx) = if self.two_raster {
            if self.with_num_bands {
                (2, 3, Some(4), Some(5))
            } else if self.with_nodata {
                (2, 3, Some(4), None)
            } else {
                (2, 3, None, None)
            }
        } else if self.with_num_bands {
            (1, 2, Some(3), Some(4))
        } else if self.with_nodata {
            (1, 2, Some(3), None)
        } else {
            (1, 2, None, None)
        };

        // Convert all non-raster args to arrays upfront via into_array
        let pixel_type_array = args[pixel_type_idx]
            .clone()
            .cast_to(&DataType::Utf8, None)?
            .into_array(num_iterations)?;
        let pixel_type_array = as_string_array(&pixel_type_array)?;

        let script_array = args[script_idx]
            .clone()
            .cast_to(&DataType::Utf8, None)?
            .into_array(num_iterations)?;
        let script_array = as_string_array(&script_array)?;

        let nodata_array = match nodata_idx {
            Some(idx) => args[idx]
                .clone()
                .cast_to(&DataType::Float64, None)?
                .into_array(num_iterations)?,
            None => ScalarValue::Float64(None).to_array_of_size(num_iterations)?,
        };
        let nodata_array = as_float64_array(&nodata_array)?;

        let num_bands_array = match num_bands_idx {
            Some(idx) => args[idx]
                .clone()
                .cast_to(&DataType::Int32, None)?
                .into_array(num_iterations)?,
            None => ScalarValue::Int32(Some(1)).to_array_of_size(num_iterations)?,
        };
        let num_bands_array = as_int32_array(&num_bands_array)?;

        let mut pixel_type_iter = pixel_type_array.iter();
        let mut script_iter = script_array.iter();
        let mut nodata_iter = nodata_array.iter();
        let mut num_bands_iter = num_bands_array.iter();

        let mut builder = RasterBuilder::new(num_iterations);

        with_gdal(|gdal| {
            configure_thread_local_options(gdal, config_options)?;
            if self.two_raster {
                executor.execute_raster_raster_void(|_i, raster0_opt, raster1_opt| {
                    let pixel_type_opt = pixel_type_iter.next().unwrap();
                    let script_opt = script_iter.next().unwrap();
                    let nodata_opt = nodata_iter.next().unwrap();
                    let num_bands_opt = num_bands_iter.next().unwrap();

                    let raster0 = match raster0_opt {
                        Some(r) => r,
                        None => {
                            builder.append_null()?;
                            return Ok(());
                        }
                    };
                    let raster1 = match raster1_opt {
                        Some(r) => r,
                        None => {
                            builder.append_null()?;
                            return Ok(());
                        }
                    };

                    process_map_algebra_row(
                        &mut builder,
                        gdal,
                        raster0,
                        Some(raster1),
                        pixel_type_opt,
                        script_opt,
                        nodata_opt,
                        num_bands_opt,
                    )
                })?;
            } else {
                executor.execute_raster_void(|_i, raster0_opt| {
                    let pixel_type_opt = pixel_type_iter.next().unwrap();
                    let script_opt = script_iter.next().unwrap();
                    let nodata_opt = nodata_iter.next().unwrap();
                    let num_bands_opt = num_bands_iter.next().unwrap();

                    let raster0 = match raster0_opt {
                        Some(r) => r,
                        None => {
                            builder.append_null()?;
                            return Ok(());
                        }
                    };

                    process_map_algebra_row(
                        &mut builder,
                        gdal,
                        raster0,
                        None,
                        pixel_type_opt,
                        script_opt,
                        nodata_opt,
                        num_bands_opt,
                    )
                })?;
            }

            executor.finish(Arc::new(builder.finish()?))
        })
    }
}

/// Process a single row of map algebra (shared between single- and two-raster paths)
#[allow(clippy::too_many_arguments)]
fn process_map_algebra_row(
    builder: &mut RasterBuilder,
    gdal: &Gdal,
    raster0: &RasterRefImpl<'_>,
    raster1: Option<&RasterRefImpl<'_>>,
    pixel_type_opt: Option<&str>,
    script_opt: Option<&str>,
    nodata_opt: Option<f64>,
    num_bands_opt: Option<i32>,
) -> Result<()> {
    let pixel_type_str = match pixel_type_opt {
        Some(s) => s,
        None => {
            builder.append_null()?;
            return Ok(());
        }
    };
    let script = match script_opt {
        Some(s) => s,
        None => {
            builder.append_null()?;
            return Ok(());
        }
    };

    let output_type = parse_pixel_type(pixel_type_str)?;
    let compiled_expr = build_operator_tree(script)
        .map_err(|e| exec_datafusion_err!("Failed to parse expression '{}': {}", script, e))?;
    let nodata = nodata_opt;
    let num_bands = num_bands_opt.map(|v| v.max(1) as usize).unwrap_or(1);

    match apply_map_algebra(
        gdal,
        raster0,
        raster1,
        &compiled_expr,
        &output_type,
        nodata,
        num_bands,
    ) {
        Ok(result_data) => {
            build_result_raster(builder, raster0, &result_data)?;
        }
        Err(e) => {
            eprintln!("RS_MapAlgebra error: {}", e);
            builder.append_null()?;
        }
    }
    Ok(())
}

/// Output data from map algebra operation
struct MapAlgebraResult {
    band_data: Vec<Vec<u8>>,
    band_metadata: Vec<BandMetadata>,
}

/// Parse pixel type string to BandDataType
fn parse_pixel_type(pixel_type: &str) -> Result<BandDataType> {
    match pixel_type.to_uppercase().as_str() {
        "B" | "BYTE" | "UINT8" => Ok(BandDataType::UInt8),
        "I8" | "INT8" => Ok(BandDataType::Int8),
        "S" | "SHORT" | "INT16" => Ok(BandDataType::Int16),
        "US" | "USHORT" | "UINT16" => Ok(BandDataType::UInt16),
        "I" | "INT" | "INT32" => Ok(BandDataType::Int32),
        "UI" | "UINT" | "UINT32" => Ok(BandDataType::UInt32),
        "U64" | "UINT64" => Ok(BandDataType::UInt64),
        "I64" | "INT64" => Ok(BandDataType::Int64),
        "F" | "FLOAT" | "FLOAT32" => Ok(BandDataType::Float32),
        "D" | "DOUBLE" | "FLOAT64" => Ok(BandDataType::Float64),
        _ => exec_err!(
            "Unknown pixel type '{}'. Use: B(yte), I8, S(hort), I(nt), U64, I64, F(loat), D(ouble)",
            pixel_type
        ),
    }
}

/// Apply map algebra expression to raster(s)
fn apply_map_algebra(
    gdal: &Gdal,
    raster0: &RasterRefImpl<'_>,
    raster1: Option<&RasterRefImpl<'_>>,
    expr: &evalexpr::Node,
    output_type: &BandDataType,
    nodata: Option<f64>,
    num_bands: usize,
) -> Result<MapAlgebraResult> {
    let metadata = raster0.metadata();
    let width = metadata.width() as usize;
    let height = metadata.height() as usize;
    let pixel_count = width * height;

    // Read all band data from first raster
    let bands0 = raster0.bands();
    let mut reader0 = RasterBandReader::new(gdal, raster0);
    let band_data0: Vec<Vec<f64>> = (1..=bands0.len())
        .map(|i| reader0.read_band_f64(i))
        .collect::<Result<Vec<_>>>()?;

    // Read all band data from second raster (if present)
    let band_data1: Option<Vec<Vec<f64>>> = if let Some(r1) = raster1 {
        // Validate dimensions match
        let m1 = r1.metadata();
        if m1.width() != metadata.width() || m1.height() != metadata.height() {
            return exec_err!("Raster dimensions must match for two-raster map algebra");
        }
        let bands1 = r1.bands();
        let mut reader1 = RasterBandReader::new(gdal, r1);
        Some(
            (1..=bands1.len())
                .map(|i| reader1.read_band_f64(i))
                .collect::<Result<Vec<_>>>()?,
        )
    } else {
        None
    };

    // Allocate output band data
    let byte_size = data_type_byte_size(output_type);
    let mut output_bands: Vec<Vec<u8>> = (0..num_bands)
        .map(|_| vec![0u8; pixel_count * byte_size])
        .collect();

    // Determine nodata value
    let nodata_val = nodata.unwrap_or(0.0);

    // Create evaluation context
    let mut context = HashMapContext::new();

    // Set constant variables
    context
        .set_value("width".to_string(), Value::Float(width as f64))
        .map_err(|e| exec_datafusion_err!("Failed to set width: {}", e))?;
    context
        .set_value("height".to_string(), Value::Float(height as f64))
        .map_err(|e| exec_datafusion_err!("Failed to set height: {}", e))?;

    // Evaluate expression for each pixel
    for pixel_idx in 0..pixel_count {
        let x = pixel_idx % width;
        let y = pixel_idx / width;

        // Set position variables
        context
            .set_value("x".to_string(), Value::Float(x as f64))
            .map_err(|e| exec_datafusion_err!("Failed to set x: {}", e))?;
        context
            .set_value("y".to_string(), Value::Float(y as f64))
            .map_err(|e| exec_datafusion_err!("Failed to set y: {}", e))?;

        // Set band values for first raster
        // Support both rast0, rast1, ... and rast0_0, rast0_1, ... syntax
        for (band_idx, band_values) in band_data0.iter().enumerate() {
            let value = band_values[pixel_idx];
            // rast0, rast1, rast2, ... (single raster syntax)
            context
                .set_value(format!("rast{}", band_idx), Value::Float(value))
                .map_err(|e| exec_datafusion_err!("Failed to set rast{}: {}", band_idx, e))?;
            // rast0_0, rast0_1, ... (two-raster syntax, first raster)
            context
                .set_value(format!("rast0_{}", band_idx), Value::Float(value))
                .map_err(|e| exec_datafusion_err!("Failed to set rast0_{}: {}", band_idx, e))?;
        }

        // Set band values for second raster (if present)
        if let Some(ref bands1) = band_data1 {
            for (band_idx, band_values) in bands1.iter().enumerate() {
                let value = band_values[pixel_idx];
                context
                    .set_value(format!("rast1_{}", band_idx), Value::Float(value))
                    .map_err(|e| exec_datafusion_err!("Failed to set rast1_{}: {}", band_idx, e))?;
            }
        }

        // Evaluate expression
        let result = expr.eval_with_context(&context).map_err(|e| {
            exec_datafusion_err!(
                "Expression evaluation failed at pixel ({}, {}): {}",
                x,
                y,
                e
            )
        })?;

        // Handle the result based on number of output bands
        if num_bands == 1 {
            // Single output band - use the result directly
            let value = value_to_f64(&result)?;
            write_pixel_value(&mut output_bands[0], pixel_idx, output_type, value);
        } else {
            // Multiple output bands - expect a tuple result or set all bands to same value
            match result {
                Value::Tuple(values) => {
                    for (band_idx, val) in values.iter().enumerate().take(num_bands) {
                        let value = value_to_f64(val)?;
                        write_pixel_value(
                            &mut output_bands[band_idx],
                            pixel_idx,
                            output_type,
                            value,
                        );
                    }
                    // If tuple has fewer values than bands, fill remaining with nodata
                    for band in output_bands.iter_mut().take(num_bands).skip(values.len()) {
                        write_pixel_value(band, pixel_idx, output_type, nodata_val);
                    }
                }
                _ => {
                    // Single value - apply to first band, nodata for rest
                    let value = value_to_f64(&result)?;
                    write_pixel_value(&mut output_bands[0], pixel_idx, output_type, value);
                    for band in output_bands.iter_mut().take(num_bands).skip(1) {
                        write_pixel_value(band, pixel_idx, output_type, nodata_val);
                    }
                }
            }
        }
    }

    // Build band metadata
    let band_metadata: Vec<BandMetadata> = (0..num_bands)
        .map(|_| BandMetadata {
            nodata_value: nodata.map(|v| nodata_f64_to_bytes(v, output_type)),
            storage_type: StorageType::InDb,
            datatype: *output_type,
            outdb_url: None,
            outdb_band_id: None,
        })
        .collect();

    Ok(MapAlgebraResult {
        band_data: output_bands,
        band_metadata,
    })
}

/// Convert evalexpr Value to f64
fn value_to_f64(value: &Value) -> Result<f64> {
    match value {
        Value::Float(f) => Ok(*f),
        Value::Int(i) => Ok(*i as f64),
        Value::Boolean(b) => Ok(if *b { 1.0 } else { 0.0 }),
        _ => exec_err!("Cannot convert {:?} to numeric value", value),
    }
}

/// Write a pixel value to band data
fn write_pixel_value(data: &mut [u8], offset: usize, data_type: &BandDataType, value: f64) {
    let byte_size = data_type_byte_size(data_type);
    let byte_offset = offset * byte_size;

    match data_type {
        BandDataType::UInt8 => {
            data[byte_offset] = value.clamp(0.0, 255.0) as u8;
        }
        BandDataType::Int8 => {
            let v = value.clamp(i8::MIN as f64, i8::MAX as f64) as i8;
            data[byte_offset] = v as u8;
        }
        BandDataType::UInt16 => {
            let v = value.clamp(0.0, u16::MAX as f64) as u16;
            data[byte_offset..byte_offset + 2].copy_from_slice(&v.to_le_bytes());
        }
        BandDataType::Int16 => {
            let v = value.clamp(i16::MIN as f64, i16::MAX as f64) as i16;
            data[byte_offset..byte_offset + 2].copy_from_slice(&v.to_le_bytes());
        }
        BandDataType::UInt32 => {
            let v = value.clamp(0.0, u32::MAX as f64) as u32;
            data[byte_offset..byte_offset + 4].copy_from_slice(&v.to_le_bytes());
        }
        BandDataType::Int32 => {
            let v = value.clamp(i32::MIN as f64, i32::MAX as f64) as i32;
            data[byte_offset..byte_offset + 4].copy_from_slice(&v.to_le_bytes());
        }
        BandDataType::UInt64 => {
            let v = value.clamp(0.0, u64::MAX as f64) as u64;
            data[byte_offset..byte_offset + 8].copy_from_slice(&v.to_le_bytes());
        }
        BandDataType::Int64 => {
            let v = value.clamp(i64::MIN as f64, i64::MAX as f64) as i64;
            data[byte_offset..byte_offset + 8].copy_from_slice(&v.to_le_bytes());
        }
        BandDataType::Float32 => {
            let v = value as f32;
            data[byte_offset..byte_offset + 4].copy_from_slice(&v.to_le_bytes());
        }
        BandDataType::Float64 => {
            data[byte_offset..byte_offset + 8].copy_from_slice(&value.to_le_bytes());
        }
    }
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

/// Build result raster using RasterBuilder
fn build_result_raster(
    builder: &mut RasterBuilder,
    original_raster: &RasterRefImpl<'_>,
    result: &MapAlgebraResult,
) -> Result<()> {
    let original_metadata = original_raster.metadata();

    let metadata = RasterMetadata {
        width: original_metadata.width(),
        height: original_metadata.height(),
        upperleft_x: original_metadata.upper_left_x(),
        upperleft_y: original_metadata.upper_left_y(),
        scale_x: original_metadata.scale_x(),
        scale_y: original_metadata.scale_y(),
        skew_x: original_metadata.skew_x(),
        skew_y: original_metadata.skew_y(),
    };

    builder
        .start_raster(&metadata, original_raster.crs())
        .map_err(|e| exec_datafusion_err!("Failed to start raster: {}", e))?;

    for (band_data, band_metadata) in result.band_data.iter().zip(result.band_metadata.iter()) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use sedona_raster::array::RasterStructArray;

    #[test]
    fn test_parse_pixel_type() {
        assert_eq!(parse_pixel_type("B").unwrap(), BandDataType::UInt8);
        assert_eq!(parse_pixel_type("byte").unwrap(), BandDataType::UInt8);
        assert_eq!(parse_pixel_type("I8").unwrap(), BandDataType::Int8);
        assert_eq!(parse_pixel_type("S").unwrap(), BandDataType::Int16);
        assert_eq!(parse_pixel_type("I").unwrap(), BandDataType::Int32);
        assert_eq!(parse_pixel_type("U64").unwrap(), BandDataType::UInt64);
        assert_eq!(parse_pixel_type("I64").unwrap(), BandDataType::Int64);
        assert_eq!(parse_pixel_type("F").unwrap(), BandDataType::Float32);
        assert_eq!(parse_pixel_type("D").unwrap(), BandDataType::Float64);
        assert_eq!(parse_pixel_type("FLOAT64").unwrap(), BandDataType::Float64);
        assert!(parse_pixel_type("X").is_err());
    }

    #[test]
    fn test_value_to_f64() {
        let pi = std::f64::consts::PI;
        assert!((value_to_f64(&Value::Float(pi)).unwrap() - pi).abs() < f64::EPSILON);
        assert_eq!(value_to_f64(&Value::Int(42)).unwrap(), 42.0);
        assert_eq!(value_to_f64(&Value::Boolean(true)).unwrap(), 1.0);
        assert_eq!(value_to_f64(&Value::Boolean(false)).unwrap(), 0.0);
    }

    #[test]
    fn test_write_pixel_value() {
        let mut data = vec![0u8; 8];

        // Test UInt8
        write_pixel_value(&mut data, 0, &BandDataType::UInt8, 128.0);
        assert_eq!(data[0], 128);

        // Test Float64
        let mut data64 = vec![0u8; 8];
        let pi = std::f64::consts::PI;
        write_pixel_value(&mut data64, 0, &BandDataType::Float64, pi);
        let read_back = f64::from_le_bytes([
            data64[0], data64[1], data64[2], data64[3], data64[4], data64[5], data64[6], data64[7],
        ]);
        assert!((read_back - pi).abs() < 1e-10);
    }

    #[test]
    fn test_expression_evaluation() {
        let expr = build_operator_tree("rast0 * 2 + 1").unwrap();
        let mut context = HashMapContext::new();
        context
            .set_value("rast0".to_string(), Value::Float(10.0))
            .unwrap();
        let result = expr.eval_with_context(&context).unwrap();
        assert_eq!(value_to_f64(&result).unwrap(), 21.0);
    }

    #[test]
    fn test_ndvi_expression() {
        // NDVI = (NIR - Red) / (NIR + Red)
        let expr = build_operator_tree("(rast3 - rast0) / (rast3 + rast0)").unwrap();
        let mut context = HashMapContext::new();
        // Simulate: Red=100, NIR=200
        context
            .set_value("rast0".to_string(), Value::Float(100.0))
            .unwrap();
        context
            .set_value("rast3".to_string(), Value::Float(200.0))
            .unwrap();
        let result = expr.eval_with_context(&context).unwrap();
        let ndvi = value_to_f64(&result).unwrap();
        // NDVI = (200-100)/(200+100) = 100/300 = 0.333...
        assert!((ndvi - 0.333333).abs() < 0.001);
    }

    #[test]
    fn test_map_algebra_with_test_raster() {
        let test_file = sedona_testing::data::test_raster("test4.tiff").unwrap();
        let result = with_gdal(|gdal| {
            let raster_array = crate::utils::load_as_indb_raster(gdal, &test_file)?;
            let raster_struct = RasterStructArray::new(&raster_array);
            let raster = raster_struct.get(0).unwrap();
            let expr = build_operator_tree("rast0 * 2").unwrap();
            let metadata = raster.metadata();
            let expected_size =
                (metadata.width() * metadata.height()) as usize * std::mem::size_of::<f64>();
            let output =
                apply_map_algebra(gdal, &raster, None, &expr, &BandDataType::Float64, None, 1)?;
            Ok::<_, datafusion_common::DataFusionError>((output, expected_size))
        });
        assert!(
            result.is_ok(),
            "Map algebra should succeed: {:?}",
            result.err()
        );

        let (output, expected_size) = result.unwrap();
        assert_eq!(output.band_data.len(), 1);
        assert_eq!(output.band_data[0].len(), expected_size);
    }

    #[test]
    fn test_map_algebra_multi_band_output() {
        let test_file = sedona_testing::data::test_raster("test4.tiff").unwrap();
        let result = with_gdal(|gdal| {
            let raster_array = crate::utils::load_as_indb_raster(gdal, &test_file)?;
            let raster_struct = RasterStructArray::new(&raster_array);
            let raster = raster_struct.get(0).unwrap();
            let expr = build_operator_tree("rast0 + rast0").unwrap();
            apply_map_algebra(
                gdal,
                &raster,
                None,
                &expr,
                &BandDataType::Float32,
                Some(0.0),
                2,
            )
        });
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.band_data.len(), 2);
        assert_eq!(output.band_metadata.len(), 2);
    }
}
