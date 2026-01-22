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
use std::{sync::Arc, vec};

use crate::executor::RasterExecutor;
use arrow_array::builder::{BinaryBuilder, Float64Builder};
use arrow_schema::DataType;
use datafusion_common::cast::as_int64_array;
use datafusion_common::error::Result;
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::affine_transformation::to_world_coordinate;
use sedona_schema::datatypes::Edges;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

/// RS_RasterToWorldCoordY() scalar UDF implementation
///
/// Converts pixel coordinates to world Y coordinate
pub fn rs_rastertoworldcoordy_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_rastertoworldcoordy",
        vec![Arc::new(RsCoordinateMapper { coord: Coord::Y })],
        Volatility::Immutable,
        Some(rs_rastertoworldcoordy_doc()),
    )
}

/// RS_RasterToWorldCoordX() scalar UDF documentation
///
/// Converts pixel coordinates to world X coordinate
pub fn rs_rastertoworldcoordx_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_rastertoworldcoordx",
        vec![Arc::new(RsCoordinateMapper { coord: Coord::X })],
        Volatility::Immutable,
        Some(rs_rastertoworldcoordx_doc()),
    )
}

/// RS_RasterToWorldCoord() scalar UDF documentation
///
/// Converts pixel coordinates to world coordinates
pub fn rs_rastertoworldcoord_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_rastertoworldcoord",
        vec![Arc::new(RsCoordinatePoint {})],
        Volatility::Immutable,
        Some(rs_rastertoworldcoord_doc()),
    )
}

fn rs_rastertoworldcoordy_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the upper left Y coordinate of the given row and column of the given raster geometric units of the geo-referenced raster. If any out of bounds values are given, the Y coordinate of the assumed point considering existing raster pixel size and skew values will be returned.".to_string(),
        "RS_RasterToWorldCoordY(raster: Raster)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_argument("x", "Integer: Column x into the raster")
    .with_argument("y", "Integer: Row y into the raster")
    .with_sql_example("SELECT RS_RasterToWorldCoordY(RS_Example(), 0, 0)".to_string())
    .build()
}

fn rs_rastertoworldcoordx_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the upper left X coordinate of the given row and column of the given raster geometric units of the geo-referenced raster. If any out of bounds values are given, the X coordinate of the assumed point considering existing raster pixel size and skew values will be returned.".to_string(),
        "RS_RasterToWorldCoordX(raster: Raster)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_argument("x", "Integer: Column x into the raster")
    .with_argument("y", "Integer: Row y into the raster")
    .with_sql_example("SELECT RS_RasterToWorldCoordX(RS_Example(), 0, 0)".to_string())
    .build()
}

fn rs_rastertoworldcoord_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the upper left X and Y coordinates of the given row and column of the given raster geometric units of the geo-referenced raster as a Point geometry. If any out of bounds values are given, the X and Y coordinates of the assumed point considering existing raster pixel size and skew values will be returned.".to_string(),
        "RS_RasterToWorldCoord(raster: Raster, x: Integer, y: Integer)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_argument("x", "Integer: Column x into the raster")
    .with_argument("y", "Integer: Row y into the raster")
    .with_sql_example("SELECT RS_RasterToWorldCoord(RS_Example(), 0, 0)".to_string())
    .build()
}

#[derive(Debug, Clone)]
enum Coord {
    X,
    Y,
}

#[derive(Debug)]
struct RsCoordinateMapper {
    coord: Coord,
}

impl SedonaScalarKernel for RsCoordinateMapper {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_raster(),
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
        let executor = RasterExecutor::new(arg_types, args);
        let mut builder = Float64Builder::with_capacity(executor.num_iterations());

        // Expand x and y coordinate parameters to arrays and cast to Int64
        let x_array = args[1].clone().cast_to(&DataType::Int64, None)?;
        let x_array = x_array.into_array(executor.num_iterations())?;
        let x_array = as_int64_array(&x_array)?;
        let y_array = args[2].clone().cast_to(&DataType::Int64, None)?;
        let y_array = y_array.into_array(executor.num_iterations())?;
        let y_array = as_int64_array(&y_array)?;
        let mut x_iter = x_array.iter();
        let mut y_iter = y_array.iter();

        executor.execute_raster_void(|_i, raster_opt| {
            let x_opt = x_iter.next().unwrap();
            let y_opt = y_iter.next().unwrap();

            match (raster_opt, x_opt, y_opt) {
                (Some(raster), Some(x), Some(y)) => {
                    let (world_x, world_y) = to_world_coordinate(&raster, x, y);
                    match self.coord {
                        Coord::X => builder.append_value(world_x),
                        Coord::Y => builder.append_value(world_y),
                    };
                }
                (_, _, _) => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

#[derive(Debug)]
struct RsCoordinatePoint;
impl SedonaScalarKernel for RsCoordinatePoint {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_integer(),
                ArgMatcher::is_integer(),
            ],
            SedonaType::Wkb(Edges::Planar, None),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let mut item: [u8; 21] = [0x00; 21];
        item[0] = 0x01;
        item[1] = 0x01;
        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            item.len() * executor.num_iterations(),
        );

        // Expand x and y coordinate parameters to arrays and cast to Int64
        let x_array = args[1].clone().cast_to(&DataType::Int64, None)?;
        let x_array = x_array.into_array(executor.num_iterations())?;
        let x_array = as_int64_array(&x_array)?;
        let y_array = args[2].clone().cast_to(&DataType::Int64, None)?;
        let y_array = y_array.into_array(executor.num_iterations())?;
        let y_array = as_int64_array(&y_array)?;
        let mut x_iter = x_array.iter();
        let mut y_iter = y_array.iter();

        executor.execute_raster_void(|_i, raster_opt| {
            let x_opt = x_iter.next().unwrap();
            let y_opt = y_iter.next().unwrap();

            match (raster_opt, x_opt, y_opt) {
                (Some(raster), Some(x), Some(y)) => {
                    let (world_x, world_y) = to_world_coordinate(&raster, x, y);
                    item[5..13].copy_from_slice(&world_x.to_le_bytes());
                    item[13..21].copy_from_slice(&world_y.to_le_bytes());
                    builder.append_value(item);
                }
                (_, _, _) => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Array;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{RASTER, WKB_GEOMETRY};
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::create::create_array;
    use sedona_testing::rasters::generate_test_rasters;
    use sedona_testing::testers::ScalarUdfTester;

    #[test]
    fn udf_docs() {
        let udf: ScalarUDF = rs_rastertoworldcoordy_udf().into();
        assert_eq!(udf.name(), "rs_rastertoworldcoordy");
        assert!(udf.documentation().is_some());

        let udf: ScalarUDF = rs_rastertoworldcoordx_udf().into();
        assert_eq!(udf.name(), "rs_rastertoworldcoordx");
        assert!(udf.documentation().is_some());

        let udf: ScalarUDF = rs_rastertoworldcoord_udf().into();
        assert_eq!(udf.name(), "rs_rastertoworldcoord");
        assert!(udf.documentation().is_some());
    }

    #[rstest]
    fn udf_invoke_xy(#[values(Coord::Y, Coord::X)] coord: Coord) {
        let udf = match coord {
            Coord::X => rs_rastertoworldcoordx_udf(),
            Coord::Y => rs_rastertoworldcoordy_udf(),
        };
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Int32),
                SedonaType::Arrow(DataType::Int32),
            ],
        );

        let rasters = generate_test_rasters(3, Some(1)).unwrap();
        // At 0,0 expect the upper left corner of the test values
        let expected_values = match coord {
            Coord::X => vec![Some(1.0), None, Some(3.0)],
            Coord::Y => vec![Some(2.0), None, Some(4.0)],
        };
        let expected: Arc<dyn arrow_array::Array> =
            Arc::new(arrow_array::Float64Array::from(expected_values));

        let result = tester
            .invoke_array_scalar_scalar(Arc::new(rasters), 0_i32, 0_i32)
            .unwrap();
        assert_array_equal(&result, &expected);
    }

    #[rstest]
    fn udf_invoke_pt() {
        let udf = rs_rastertoworldcoord_udf();
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Int32),
                SedonaType::Arrow(DataType::Int32),
            ],
        );

        let rasters = generate_test_rasters(3, Some(1)).unwrap();
        // At 0,0 expect the upper left corner of the test values
        let expected = &create_array(
            &[Some("POINT (1 2)"), None, Some("POINT (3 4)")],
            &WKB_GEOMETRY,
        );

        let result = tester
            .invoke_array_scalar_scalar(Arc::new(rasters), 0_i32, 0_i32)
            .unwrap();
        assert_array_equal(&result, expected);
    }

    #[rstest]
    fn udf_invoke_xy_with_array_coords(#[values(Coord::Y, Coord::X)] coord: Coord) {
        let udf = match coord {
            Coord::X => rs_rastertoworldcoordx_udf(),
            Coord::Y => rs_rastertoworldcoordy_udf(),
        };
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Int32),
                SedonaType::Arrow(DataType::Int32),
            ],
        );

        let rasters = generate_test_rasters(3, Some(1)).unwrap();
        // Test with different pixel coordinates for each raster
        // Raster 0: upper_left=(1,2), scales=(0,0), so any pixel gives (1,2)
        // Raster 1: null
        // Raster 2: upper_left=(3,4), scale_x=0.2, scale_y=-0.4, skew_x=0.06, skew_y=0.08
        //           At pixel (1,2): x = 3 + 0.2*1 + 0.06*2 = 3.32, y = 4 + 0.08*1 + (-0.4)*2 = 3.28
        let x_coords = Arc::new(arrow_array::Int32Array::from(vec![0, 0, 1]));
        let y_coords = Arc::new(arrow_array::Int32Array::from(vec![0, 0, 2]));

        let result = tester
            .invoke_arrays(vec![Arc::new(rasters), x_coords, y_coords])
            .unwrap();

        let float_array = result
            .as_any()
            .downcast_ref::<arrow_array::Float64Array>()
            .expect("Expected Float64Array");

        // Check each value with approximate comparison due to floating point
        match coord {
            Coord::X => {
                assert!((float_array.value(0) - 1.0).abs() < 1e-10);
                assert!(float_array.is_null(1));
                assert!((float_array.value(2) - 3.32).abs() < 1e-10);
            }
            Coord::Y => {
                assert!((float_array.value(0) - 2.0).abs() < 1e-10);
                assert!(float_array.is_null(1));
                assert!((float_array.value(2) - 3.28).abs() < 1e-10);
            }
        }
    }
}
