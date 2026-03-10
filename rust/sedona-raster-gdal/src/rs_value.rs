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

//! RS_Value UDF - Get raster pixel value at a point or grid coordinate
//!
//! Returns the value at the given point in the raster. If no band number is specified,
//! it defaults to 1. If the CRS of the input point differs from the raster CRS,
//! the point will be transformed to match the raster CRS.

use std::convert::TryInto;
use std::sync::Arc;

use arrow_array::builder::Float64Builder;
use arrow_schema::DataType;
use datafusion_common::cast::as_int32_array;
use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_common::{exec_datafusion_err, exec_err, ScalarValue};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_gdal::gdal::Gdal;
use sedona_proj::transform::with_global_proj_engine;

use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::affine_transformation::to_raster_coordinate;
use sedona_raster::array::RasterRefImpl;
use sedona_raster::traits::RasterRef;
use sedona_raster_functions::crs_utils::{crs_transform_wkb, resolve_crs};
use sedona_raster_functions::RasterExecutor;
use sedona_schema::datatypes::SedonaType;
use sedona_schema::matchers::ArgMatcher;
use sedona_schema::raster::BandDataType;

use crate::gdal_common::with_gdal;
use crate::gdal_dataset_provider::configure_thread_local_options;
use crate::raster_band_reader::RasterBandReader;

/// RS_Value() scalar UDF implementation
///
/// Returns the value at the given point in the raster
pub fn rs_value_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_value",
        vec![
            Arc::new(RsValuePoint { with_band: false }),
            Arc::new(RsValuePoint { with_band: true }),
            Arc::new(RsValueGrid),
        ],
        Volatility::Immutable,
    )
}

/// Kernel for RS_Value with point geometry argument
#[derive(Debug)]
struct RsValuePoint {
    with_band: bool,
}

impl SedonaScalarKernel for RsValuePoint {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matchers = if self.with_band {
            vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_integer(),
            ]
        } else {
            vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_geometry_or_geography(),
            ]
        };

        let matcher = ArgMatcher::new(matchers, SedonaType::Arrow(DataType::Float64));
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
        // Use a full executor to compute num_iterations from all args
        let full_executor = RasterExecutor::new(arg_types, args);
        let num_iterations = full_executor.num_iterations();
        let mut builder = Float64Builder::with_capacity(num_iterations);

        let band_array = if self.with_band {
            args[2]
                .clone()
                .cast_to(&DataType::Int32, None)?
                .into_array(num_iterations)?
        } else {
            ScalarValue::Int32(Some(1)).to_array_of_size(num_iterations)?
        };
        let band_array = as_int32_array(&band_array)?.clone();
        let mut band_iter = band_array.iter();

        let exec_arg_types = vec![arg_types[0].clone(), arg_types[1].clone()];
        let exec_args = vec![args[0].clone(), args[1].clone()];
        let executor =
            RasterExecutor::new_with_num_iterations(&exec_arg_types, &exec_args, num_iterations);

        with_gdal(|gdal| {
            configure_thread_local_options(gdal, config_options)?;
            with_global_proj_engine(|engine| {
                executor.execute_raster_wkb_crs_void(|raster_opt, wkb_opt, maybe_point_crs| {
                    let band_num = band_iter
                        .next()
                        .flatten()
                        .unwrap_or(1)
                        .max(1)
                        .try_into()
                        .unwrap_or(1);
                    let (raster, point_wkb) = match (raster_opt, wkb_opt) {
                        (Some(raster), Some(point_wkb)) => (raster, point_wkb),
                        _ => {
                            builder.append_null();
                            return Ok(());
                        }
                    };

                    let raster_crs = resolve_crs(raster.crs())?;

                    let point_wkb = match (maybe_point_crs, raster_crs.as_deref()) {
                        (Some(point_crs), Some(raster_crs)) => {
                            crs_transform_wkb(point_wkb, point_crs, raster_crs, engine)?
                        }
                        (None, None) => point_wkb.to_vec(),
                        (Some(_), None) => {
                            return exec_err!(
                            "Cannot operate on point and raster: raster has no CRS but point does"
                        )
                        }
                        (None, Some(_)) => {
                            return exec_err!(
                            "Cannot operate on point and raster: point has no CRS but raster does"
                        )
                        }
                    };

                    match get_value_at_point(gdal, raster, &point_wkb, band_num) {
                        Ok(Some(value)) => builder.append_value(value),
                        Ok(None) => builder.append_null(),
                        Err(_) => builder.append_null(),
                    }

                    Ok(())
                })
            })
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

/// Kernel for RS_Value with grid coordinates
#[derive(Debug)]
struct RsValueGrid;

impl SedonaScalarKernel for RsValueGrid {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_integer(),
                ArgMatcher::is_integer(),
                ArgMatcher::is_integer(),
            ],
            SedonaType::Arrow(DataType::Float64),
        );
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
        let mut builder = Float64Builder::with_capacity(num_iterations);

        // Convert col_x, row_y, band to arrays
        let col_x_array = args[1]
            .clone()
            .cast_to(&DataType::Int32, None)?
            .into_array(num_iterations)?;
        let col_x_array = as_int32_array(&col_x_array)?;
        let row_y_array = args[2]
            .clone()
            .cast_to(&DataType::Int32, None)?
            .into_array(num_iterations)?;
        let row_y_array = as_int32_array(&row_y_array)?;
        let band_array = args[3]
            .clone()
            .cast_to(&DataType::Int32, None)?
            .into_array(num_iterations)?;
        let band_array = as_int32_array(&band_array)?;

        let mut col_x_iter = col_x_array.iter();
        let mut row_y_iter = row_y_array.iter();
        let mut band_iter = band_array.iter();

        with_gdal(|gdal| {
            configure_thread_local_options(gdal, config_options)?;
            executor.execute_raster_void(|_i, raster_opt| {
                let col_x_opt = col_x_iter.next().unwrap();
                let row_y_opt = row_y_iter.next().unwrap();
                let band_opt = band_iter.next().unwrap();

                let raster = match (raster_opt, col_x_opt, row_y_opt, band_opt) {
                    (Some(raster), Some(_), Some(_), Some(_)) => raster,
                    _ => {
                        builder.append_null();
                        return Ok(());
                    }
                };

                let x = col_x_opt.unwrap() as i64;
                let y = row_y_opt.unwrap() as i64;
                let band_num: usize = band_opt.unwrap().max(1).try_into().unwrap_or(1);

                match get_value_at_grid(gdal, raster, x, y, band_num) {
                    Ok(Some(value)) => builder.append_value(value),
                    Ok(None) => builder.append_null(),
                    Err(_) => builder.append_null(),
                }

                Ok(())
            })
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

/// Get pixel value at a point geometry
fn get_value_at_point(
    gdal: &Gdal,
    raster: &RasterRefImpl<'_>,
    point_wkb: &[u8],
    band_num: usize,
) -> Result<Option<f64>> {
    // Parse point from WKB
    let (x, y) = parse_point_from_wkb(point_wkb)?;

    // Convert world coordinates to raster coordinates
    let (col, row) = to_raster_coordinate(raster, x, y)
        .map_err(|e| exec_datafusion_err!("Failed to convert coordinates: {}", e))?;

    get_value_at_grid(gdal, raster, col, row, band_num)
}

/// Get pixel value at grid coordinates
fn get_value_at_grid(
    gdal: &Gdal,
    raster: &RasterRefImpl<'_>,
    col: i64,
    row: i64,
    band_num: usize,
) -> Result<Option<f64>> {
    let metadata = raster.metadata();
    let width = metadata.width() as i64;
    let height = metadata.height() as i64;

    // Check bounds
    if col < 0 || col >= width || row < 0 || row >= height {
        return Ok(None);
    }

    let bands = raster.bands();
    if band_num == 0 || band_num > bands.len() {
        return exec_err!("Band {} is out of range (1-{})", band_num, bands.len());
    }

    let band = bands
        .band(band_num)
        .map_err(|e| exec_datafusion_err!("Failed to get band {}: {}", band_num, e))?;

    let band_metadata = band.metadata();
    let mut band_reader = RasterBandReader::new(gdal, raster);
    let value = band_reader.read_pixel_f64(band_num, col as usize, row as usize)?;

    // Check for nodata
    if let Some(nodata_bytes) = band_metadata.nodata_value() {
        let nodata = read_nodata_value(nodata_bytes, band_metadata.data_type()?)?;
        if (value - nodata).abs() < f64::EPSILON {
            return Ok(None);
        }
    }

    Ok(Some(value))
}

/// Parse point coordinates from WKB
fn parse_point_from_wkb(wkb: &[u8]) -> Result<(f64, f64)> {
    // WKB Point structure:
    // - 1 byte: byte order (01 = little endian, 00 = big endian)
    // - 4 bytes: geometry type (1 = Point)
    // - 8 bytes: X coordinate (f64)
    // - 8 bytes: Y coordinate (f64)

    if wkb.len() < 21 {
        return exec_err!("Invalid WKB: too short for Point geometry");
    }

    let byte_order = wkb[0];
    let geom_type = if byte_order == 0x01 {
        // Little endian
        u32::from_le_bytes([wkb[1], wkb[2], wkb[3], wkb[4]])
    } else {
        // Big endian
        u32::from_be_bytes([wkb[1], wkb[2], wkb[3], wkb[4]])
    };

    // Check geometry type (1 = Point, may have Z/M flags in higher bits)
    let base_type = geom_type & 0xFF;
    if base_type != 1 {
        return exec_err!("Expected Point geometry (type 1), got type {}", base_type);
    }

    let (x, y) = if byte_order == 0x01 {
        // Little endian
        let x = f64::from_le_bytes([
            wkb[5], wkb[6], wkb[7], wkb[8], wkb[9], wkb[10], wkb[11], wkb[12],
        ]);
        let y = f64::from_le_bytes([
            wkb[13], wkb[14], wkb[15], wkb[16], wkb[17], wkb[18], wkb[19], wkb[20],
        ]);
        (x, y)
    } else {
        // Big endian
        let x = f64::from_be_bytes([
            wkb[5], wkb[6], wkb[7], wkb[8], wkb[9], wkb[10], wkb[11], wkb[12],
        ]);
        let y = f64::from_be_bytes([
            wkb[13], wkb[14], wkb[15], wkb[16], wkb[17], wkb[18], wkb[19], wkb[20],
        ]);
        (x, y)
    };

    Ok((x, y))
}

/// Read nodata value from bytes
fn read_nodata_value(bytes: &[u8], data_type: BandDataType) -> Result<f64> {
    match data_type {
        BandDataType::UInt8 => {
            if !bytes.is_empty() {
                Ok(bytes[0] as f64)
            } else {
                exec_err!("Invalid nodata bytes")
            }
        }
        BandDataType::Int8 => {
            if !bytes.is_empty() {
                Ok(bytes[0] as i8 as f64)
            } else {
                exec_err!("Invalid nodata bytes")
            }
        }
        BandDataType::UInt16 => {
            if bytes.len() >= 2 {
                Ok(u16::from_le_bytes([bytes[0], bytes[1]]) as f64)
            } else {
                exec_err!("Invalid nodata bytes")
            }
        }
        BandDataType::Int16 => {
            if bytes.len() >= 2 {
                Ok(i16::from_le_bytes([bytes[0], bytes[1]]) as f64)
            } else {
                exec_err!("Invalid nodata bytes")
            }
        }
        BandDataType::UInt32 => {
            if bytes.len() >= 4 {
                Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64)
            } else {
                exec_err!("Invalid nodata bytes")
            }
        }
        BandDataType::Int32 => {
            if bytes.len() >= 4 {
                Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64)
            } else {
                exec_err!("Invalid nodata bytes")
            }
        }
        BandDataType::UInt64 => {
            if bytes.len() >= 8 {
                Ok(u64::from_le_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                ]) as f64)
            } else {
                exec_err!("Invalid nodata bytes")
            }
        }
        BandDataType::Int64 => {
            if bytes.len() >= 8 {
                Ok(i64::from_le_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                ]) as f64)
            } else {
                exec_err!("Invalid nodata bytes")
            }
        }
        BandDataType::Float32 => {
            if bytes.len() >= 4 {
                Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64)
            } else {
                exec_err!("Invalid nodata bytes")
            }
        }
        BandDataType::Float64 => {
            if bytes.len() >= 8 {
                Ok(f64::from_le_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                ]))
            } else {
                exec_err!("Invalid nodata bytes")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gdal_common::with_gdal;
    use sedona_raster::affine_transformation::to_world_coordinate;
    use sedona_raster::array::RasterStructArray;
    use sedona_raster_functions::crs_utils::crs_transform_coord;
    use sedona_schema::crs::deserialize_crs;
    use sedona_schema::datatypes::{Edges, RASTER};
    use sedona_schema::raster::BandDataType;
    use sedona_testing::create::make_wkb;

    #[test]
    fn test_parse_point_from_wkb() {
        // Little-endian WKB for POINT(1.0, 2.0)
        let wkb = [
            0x01, // Little endian
            0x01, 0x00, 0x00, 0x00, // Point type
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F, // X = 1.0
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, // Y = 2.0
        ];

        let (x, y) = parse_point_from_wkb(&wkb).unwrap();
        assert!((x - 1.0).abs() < f64::EPSILON);
        assert!((y - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_read_pixel_value_uint8() {
        let data = vec![42u8, 100, 200];
        let raster_array = sedona_testing::rasters::raster_from_single_band(
            3,
            1,
            BandDataType::UInt8,
            &data,
            None,
        );
        let raster_struct = RasterStructArray::new(&raster_array);
        let raster = raster_struct.get(0).unwrap();
        let value = with_gdal(|gdal| {
            let mut reader = RasterBandReader::new(gdal, &raster);
            reader.read_pixel_f64(1, 1, 0)
        })
        .unwrap();
        assert!((value - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_read_pixel_value_float32() {
        let mut data = vec![0u8; 12]; // 3 float32 values
        let values: [f32; 3] = [1.5, 2.5, 3.5];
        for (i, &v) in values.iter().enumerate() {
            data[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
        }
        let raster_array = sedona_testing::rasters::raster_from_single_band(
            3,
            1,
            BandDataType::Float32,
            &data,
            None,
        );
        let raster_struct = RasterStructArray::new(&raster_array);
        let raster = raster_struct.get(0).unwrap();
        let value = with_gdal(|gdal| {
            let mut reader = RasterBandReader::new(gdal, &raster);
            reader.read_pixel_f64(1, 1, 0)
        })
        .unwrap();
        assert!((value - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rs_value_grid_with_test_raster() {
        // Load test raster and read value at grid coordinates
        let test_file = sedona_testing::data::test_raster("test4.tiff").unwrap();
        let (value, center_value, out_of_bounds) = with_gdal(|gdal| {
            let raster_array = crate::utils::load_as_indb_raster(gdal, &test_file)?;
            let raster_struct = RasterStructArray::new(&raster_array);
            let raster = raster_struct.get(0).unwrap();

            let value = get_value_at_grid(gdal, &raster, 0, 0, 1)?;
            let center_value = get_value_at_grid(gdal, &raster, 5, 5, 1)?;
            let out_of_bounds = get_value_at_grid(gdal, &raster, 100, 100, 1)?;
            Ok::<_, datafusion_common::DataFusionError>((value, center_value, out_of_bounds))
        })
        .unwrap();
        assert!(value.is_some());
        assert!(center_value.is_some());
        assert!(out_of_bounds.is_none());
    }

    #[test]
    fn test_rs_value_invoke_grid() {
        // Test invoking RS_Value with grid coordinates
        use arrow_schema::DataType;
        use sedona_expr::scalar_udf::SedonaScalarKernel;
        use sedona_schema::datatypes::RASTER;

        let test_file = sedona_testing::data::test_raster("test4.tiff").unwrap();
        let raster_array =
            with_gdal(|gdal| crate::utils::load_as_indb_raster(gdal, &test_file)).unwrap();

        let kernel = RsValueGrid;

        // Test return type
        let arg_types = vec![
            RASTER,
            SedonaType::Arrow(DataType::Int32),
            SedonaType::Arrow(DataType::Int32),
            SedonaType::Arrow(DataType::Int32),
        ];
        let return_type = kernel.return_type(&arg_types).unwrap();
        assert!(return_type.is_some());

        // Test invoke_batch
        let args = vec![
            ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(raster_array))),
            ColumnarValue::Scalar(ScalarValue::Int32(Some(0))), // col_x
            ColumnarValue::Scalar(ScalarValue::Int32(Some(0))), // row_y
            ColumnarValue::Scalar(ScalarValue::Int32(Some(1))), // band
        ];

        let result = kernel.invoke_batch(&arg_types, &args).unwrap();

        match result {
            ColumnarValue::Scalar(ScalarValue::Float64(Some(value))) => {
                // Value should be a valid pixel value
                assert!(value.is_finite());
            }
            _ => panic!("Expected Float64 scalar result"),
        }
    }

    #[test]
    fn test_rs_value_point_crs_transform() {
        let test_file = sedona_testing::data::test_raster("test4.tiff").unwrap();
        let raster_array =
            with_gdal(|gdal| crate::utils::load_as_indb_raster(gdal, &test_file)).unwrap();

        let raster_struct = RasterStructArray::new(&raster_array);
        let raster = raster_struct.get(0).unwrap();
        let width = raster.metadata().width() as i64;
        let height = raster.metadata().height() as i64;
        let col = width / 2;
        let row = height / 2;
        let (lon, lat) = to_world_coordinate(&raster, col, row);

        let point_wkt = format!("POINT ({} {})", lon, lat);
        let point_wkb = make_wkb(&point_wkt);
        let (x_merc, y_merc) = with_global_proj_engine(|engine| {
            crs_transform_coord(engine, (lon, lat), "OGC:CRS84", "EPSG:3857")
        })
        .unwrap();
        let point_merc_wkt = format!("POINT ({} {})", x_merc, y_merc);
        let point_merc_wkb = make_wkb(&point_merc_wkt);

        let raster_scalar = ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(raster_array)));

        let geom_type_4326 = SedonaType::Wkb(Edges::Planar, deserialize_crs("EPSG:4326").unwrap());
        let geom_type_3857 = SedonaType::Wkb(Edges::Planar, deserialize_crs("EPSG:3857").unwrap());

        let kernel = RsValuePoint { with_band: false };

        let result_4326 = kernel
            .invoke_batch(
                &[RASTER, geom_type_4326],
                &[
                    raster_scalar.clone(),
                    ColumnarValue::Scalar(ScalarValue::Binary(Some(point_wkb))),
                ],
            )
            .unwrap();

        let value_4326 = match result_4326 {
            ColumnarValue::Scalar(ScalarValue::Float64(Some(value))) => value,
            _ => panic!("Expected Float64 scalar result"),
        };

        let result_3857 = kernel
            .invoke_batch(
                &[RASTER, geom_type_3857],
                &[
                    raster_scalar,
                    ColumnarValue::Scalar(ScalarValue::Binary(Some(point_merc_wkb))),
                ],
            )
            .unwrap();

        let value_3857 = match result_3857 {
            ColumnarValue::Scalar(ScalarValue::Float64(Some(value))) => value,
            _ => panic!("Expected Float64 scalar result"),
        };

        assert_eq!(value_4326, value_3857);
    }
}
