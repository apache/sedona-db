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
use arrow_array::builder::{BinaryBuilder, Int64Builder};
use arrow_schema::DataType;
use datafusion_common::ScalarValue;
use datafusion_common::{error::Result, exec_err};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::affine_transformation::to_raster_coordinate;
use sedona_schema::datatypes::Edges;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

/// RS_WorldToRasterCoordY() scalar UDF documentation
///
/// Converts world coordinates to raster Y coordinate
pub fn rs_worldtorastercoordy_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_worldtorastercoordy",
        vec![Arc::new(RsCoordinateMapper { coord: Coord::Y })],
        Volatility::Immutable,
        Some(rs_worldtorastercoordy_doc()),
    )
}

/// RS_WorldToRasterCoordX() scalar UDF documentation
///
/// Converts world coordinates to raster X coordinate
pub fn rs_worldtorastercoordx_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_worldtorastercoordx",
        vec![Arc::new(RsCoordinateMapper { coord: Coord::X })],
        Volatility::Immutable,
        Some(rs_worldtorastercoordx_doc()),
    )
}

/// RS_WorldToRasterCoord() scalar UDF documentation
///
/// Converts world coordinates to raster coordinates as a Point
pub fn rs_worldtorastercoord_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_worldtorastercoord",
        vec![Arc::new(RsCoordinatePoint {})],
        Volatility::Immutable,
        Some(rs_worldtorastercoord_doc()),
    )
}

fn rs_worldtorastercoordx_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the X coordinate of the grid coordinate of the given world coordinates as an integer.".to_string(),
        "RS_WorldToRasterCoord(raster: Raster, x: Float, y: Float)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_argument("x", "Float: World x coordinate")
    .with_argument("y", "Float: World y coordinate")
    .with_sql_example("SELECT RS_WorldToRasterCoordX(RS_Example(), 34.865965, -111.812498)".to_string())
    .build()
}

fn rs_worldtorastercoordy_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the Y coordinate of the grid coordinate of the given world coordinates as an integer.".to_string(),
        "RS_WorldToRasterCoord(raster: Raster, x: Float, y: Float)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_argument("x", "Float: World x coordinate")
    .with_argument("y", "Float: World y coordinate")
    .with_sql_example("SELECT RS_WorldToRasterCoordY(RS_Example(), 34.865965, -111.812498)".to_string())
    .build()
}

fn rs_worldtorastercoord_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the grid coordinate of the given world coordinates as a Point.".to_string(),
        "RS_WorldToRasterCoord(raster: Raster, x: Float, y: Float)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_argument("x", "Float: World x coordinate")
    .with_argument("y", "Float: World y coordinate")
    .with_sql_example(
        "SELECT RS_WorldToRasterCoord(RS_Example(), 34.865965, -111.812498)".to_string(),
    )
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
                ArgMatcher::is_numeric(),
                ArgMatcher::is_numeric(),
            ],
            SedonaType::Arrow(DataType::Int64),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let mut builder = Int64Builder::with_capacity(executor.num_iterations());

        let coord_opt = extract_scalar_coord(&args[1], &args[2])?;
        executor.execute_raster_void(|_i, raster_opt| {
            match (raster_opt, coord_opt) {
                (Some(raster), (Some(x), Some(y))) => {
                    let (raster_x, raster_y) = to_raster_coordinate(&raster, x, y)?;
                    match self.coord {
                        Coord::X => builder.append_value(raster_x),
                        Coord::Y => builder.append_value(raster_y),
                    };
                }
                (_, _) => builder.append_null(),
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
                ArgMatcher::is_numeric(),
                ArgMatcher::is_numeric(),
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

        let coord_opt = extract_scalar_coord(&args[1], &args[2])?;
        executor.execute_raster_void(|_i, raster_opt| {
            match (raster_opt, coord_opt) {
                (Some(raster), (Some(world_x), Some(world_y))) => {
                    let (raster_x, raster_y) = to_raster_coordinate(&raster, world_x, world_y)?;
                    item[5..13].copy_from_slice(&(raster_x as f64).to_le_bytes());
                    item[13..21].copy_from_slice(&(raster_y as f64).to_le_bytes());
                    builder.append_value(item);
                }
                (_, _) => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn extract_float_scalar(arg: &ColumnarValue) -> Result<Option<f64>> {
    match arg {
        ColumnarValue::Scalar(scalar) => {
            let f64_val = scalar.cast_to(&DataType::Float64)?;
            match f64_val {
                ScalarValue::Float64(Some(v)) => Ok(Some(v)),
                _ => Ok(None),
            }
        }
        _ => exec_err!("Expected scalar float argument for coordinate"),
    }
}

fn extract_scalar_coord(
    x_arg: &ColumnarValue,
    y_arg: &ColumnarValue,
) -> Result<(Option<f64>, Option<f64>)> {
    let x_opt = extract_float_scalar(x_arg)?;
    let y_opt = extract_float_scalar(y_arg)?;
    Ok((x_opt, y_opt))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::DataType;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{RASTER, WKB_GEOMETRY};
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::create::create_array;
    use sedona_testing::rasters::generate_test_rasters;
    use sedona_testing::testers::ScalarUdfTester;

    #[test]
    fn udf_docs() {
        let udf: ScalarUDF = rs_worldtorastercoordy_udf().into();
        assert_eq!(udf.name(), "rs_worldtorastercoordy");
        assert!(udf.documentation().is_some());

        let udf: ScalarUDF = rs_worldtorastercoordx_udf().into();
        assert_eq!(udf.name(), "rs_worldtorastercoordx");
        assert!(udf.documentation().is_some());

        let udf: ScalarUDF = rs_worldtorastercoord_udf().into();
        assert_eq!(udf.name(), "rs_worldtorastercoord");
        assert!(udf.documentation().is_some());
    }

    #[rstest]
    fn udf_invoke_xy(#[values(Coord::Y, Coord::X)] coord: Coord) {
        let udf = match coord {
            Coord::X => rs_worldtorastercoordx_udf(),
            Coord::Y => rs_worldtorastercoordy_udf(),
        };
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
            ],
        );

        let rasters = generate_test_rasters(2, Some(0)).unwrap();
        // Nulling out raster 0 since it has a non-invertible geotransform
        // since it has zeros in skews and scales.
        // (2.0,3.0) is upper left corner for raster 1, which has raster coords (0,0)
        let expected_values = match coord {
            Coord::X => vec![None, Some(0_i64)],
            Coord::Y => vec![None, Some(0_i64)],
        };
        let expected: Arc<dyn arrow_array::Array> =
            Arc::new(arrow_array::Int64Array::from(expected_values));

        let result = tester
            .invoke_array_scalar_scalar(Arc::new(rasters), 2.0_f64, 3.0_f64)
            .unwrap();
        assert_array_equal(&result, &expected);

        // Test that we correctly handle non-invertible geotransforms
        // using non-invertible raster 0
        let noninvertible_rasters = generate_test_rasters(2, None).unwrap();
        let result_err =
            tester.invoke_array_scalar_scalar(Arc::new(noninvertible_rasters), 2.0_f64, 3.0_f64);
        assert!(result_err.is_err());
        assert!(result_err
            .err()
            .unwrap()
            .to_string()
            .contains("determinant is zero"));
    }

    #[rstest]
    fn udf_invoke_pt() {
        let udf = rs_worldtorastercoord_udf();
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
            ],
        );

        let rasters = generate_test_rasters(2, Some(0)).unwrap();
        let expected = &create_array(&[None, Some("POINT (0 0)")], &WKB_GEOMETRY);

        let result = tester
            .invoke_array_scalar_scalar(Arc::new(rasters), 2.0_f64, 3.0_f64)
            .unwrap();
        assert_array_equal(&result, expected);

        // Test that we correctly handle non-invertible geotransforms
        let noninvertible_rasters = generate_test_rasters(2, None).unwrap();
        let result_err =
            tester.invoke_array_scalar_scalar(Arc::new(noninvertible_rasters), 2.0_f64, 3.0_f64);
        assert!(result_err.is_err());
        assert!(result_err
            .err()
            .unwrap()
            .to_string()
            .contains("determinant is zero"));
    }
}
