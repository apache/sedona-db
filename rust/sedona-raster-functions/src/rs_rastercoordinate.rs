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
use arrow_array::Float64Array;
use arrow_array::builder::{BinaryBuilder, Int64Builder};
use arrow_schema::DataType;
use datafusion_common::cast::as_float64_array;
use datafusion_common::error::Result;
use datafusion_common::exec_datafusion_err;
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::types::Edges;
use sedona_geometry::wkb_header::read_point_xy;
use sedona_raster::affine_transformation::to_raster_coordinate;
use sedona_schema::{
    datatypes::SedonaType,
    matchers::{ArgMatcher, TypeMatcher},
};

/// RS_WorldToRasterCoordY() scalar UDF documentation
///
/// Converts world coordinates to raster Y coordinate
pub fn rs_worldtorastercoordy_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_worldtorastercoordy",
        vec![
            Arc::new(RsCoordinateMapper {
                coord: Coord::Y,
                from_point: false,
            }),
            Arc::new(RsCoordinateMapper {
                coord: Coord::Y,
                from_point: true,
            }),
        ],
        Volatility::Immutable,
    )
}

/// RS_WorldToRasterCoordX() scalar UDF documentation
///
/// Converts world coordinates to raster X coordinate
pub fn rs_worldtorastercoordx_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_worldtorastercoordx",
        vec![
            Arc::new(RsCoordinateMapper {
                coord: Coord::X,
                from_point: false,
            }),
            Arc::new(RsCoordinateMapper {
                coord: Coord::X,
                from_point: true,
            }),
        ],
        Volatility::Immutable,
    )
}

/// RS_WorldToRasterCoord() scalar UDF documentation
///
/// Converts world coordinates to raster coordinates as a Point
pub fn rs_worldtorastercoord_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_worldtorastercoord",
        vec![
            Arc::new(RsCoordinatePoint { from_point: false }),
            Arc::new(RsCoordinatePoint { from_point: true }),
        ],
        Volatility::Immutable,
    )
}

#[derive(Debug, Clone)]
enum Coord {
    X,
    Y,
}

impl Coord {
    fn pick(&self, raster_x: i64, raster_y: i64) -> i64 {
        match self {
            Coord::X => raster_x,
            Coord::Y => raster_y,
        }
    }
}

#[derive(Debug)]
struct RsCoordinateMapper {
    coord: Coord,
    /// `false`: `(raster, worldX, worldY)`. `true`: the Sedona Spark parity
    /// overload `(raster, point)`, which reads the world coordinate from a
    /// point geometry.
    from_point: bool,
}

impl SedonaScalarKernel for RsCoordinateMapper {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            coord_input_matchers(self.from_point),
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

        if self.from_point {
            executor.execute_raster_wkb_crs_void(|raster_opt, wkb_opt, _crs| {
                match (raster_opt, point_xy(wkb_opt)?) {
                    (Some(raster), Some((x, y))) => {
                        let (raster_x, raster_y) = to_raster_coordinate(raster, x, y)?;
                        builder.append_value(self.coord.pick(raster_x, raster_y));
                    }
                    _ => builder.append_null(),
                }
                Ok(())
            })?;
        } else {
            let (world_x, world_y) = world_xy_arrays(args, executor.num_iterations())?;
            let mut world_x_iter = world_x.iter();
            let mut world_y_iter = world_y.iter();
            executor.execute_raster_void(|_i, raster_opt| {
                match (
                    raster_opt,
                    world_x_iter.next().unwrap(),
                    world_y_iter.next().unwrap(),
                ) {
                    (Some(raster), Some(x), Some(y)) => {
                        let (raster_x, raster_y) = to_raster_coordinate(raster, x, y)?;
                        builder.append_value(self.coord.pick(raster_x, raster_y));
                    }
                    (_, _, _) => builder.append_null(),
                }
                Ok(())
            })?;
        }

        executor.finish(Arc::new(builder.finish()))
    }
}

#[derive(Debug)]
struct RsCoordinatePoint {
    /// `false`: `(raster, worldX, worldY)`. `true`: the Sedona Spark parity
    /// overload `(raster, point)`.
    from_point: bool,
}
impl SedonaScalarKernel for RsCoordinatePoint {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            coord_input_matchers(self.from_point),
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

        if self.from_point {
            executor.execute_raster_wkb_crs_void(|raster_opt, wkb_opt, _crs| {
                match (raster_opt, point_xy(wkb_opt)?) {
                    (Some(raster), Some((x, y))) => {
                        let (raster_x, raster_y) = to_raster_coordinate(raster, x, y)?;
                        item[5..13].copy_from_slice(&(raster_x as f64).to_le_bytes());
                        item[13..21].copy_from_slice(&(raster_y as f64).to_le_bytes());
                        builder.append_value(item);
                    }
                    _ => builder.append_null(),
                }
                Ok(())
            })?;
        } else {
            let (world_x, world_y) = world_xy_arrays(args, executor.num_iterations())?;
            let mut world_x_iter = world_x.iter();
            let mut world_y_iter = world_y.iter();
            executor.execute_raster_void(|_i, raster_opt| {
                match (
                    raster_opt,
                    world_x_iter.next().unwrap(),
                    world_y_iter.next().unwrap(),
                ) {
                    (Some(raster), Some(world_x), Some(world_y)) => {
                        let (raster_x, raster_y) = to_raster_coordinate(raster, world_x, world_y)?;
                        item[5..13].copy_from_slice(&(raster_x as f64).to_le_bytes());
                        item[13..21].copy_from_slice(&(raster_y as f64).to_le_bytes());
                        builder.append_value(item);
                    }
                    (_, _, _) => builder.append_null(),
                }
                Ok(())
            })?;
        }

        executor.finish(Arc::new(builder.finish()))
    }
}

/// Input matchers for a world→raster coordinate function: the numeric
/// `(raster, worldX, worldY)` form, or the `(raster, point)` overload.
fn coord_input_matchers(from_point: bool) -> Vec<Arc<dyn TypeMatcher + Send + Sync>> {
    if from_point {
        vec![
            ArgMatcher::is_raster(),
            ArgMatcher::is_geometry_or_geography(),
        ]
    } else {
        vec![
            ArgMatcher::is_raster(),
            ArgMatcher::is_numeric(),
            ArgMatcher::is_numeric(),
        ]
    }
}

/// The world `(x, y)` carried by a point geometry, or `None` for a null or
/// empty point. Errors on a non-point geometry. The point's coordinates are
/// used as-is, in the raster's own space — the numeric form takes no CRS
/// either, so the point overload does not reproject.
fn point_xy(wkb_opt: Option<&[u8]>) -> Result<Option<(f64, f64)>> {
    match wkb_opt {
        Some(wkb) => {
            read_point_xy(wkb).map_err(|e| exec_datafusion_err!("RS_WorldToRasterCoord: {e}"))
        }
        None => Ok(None),
    }
}

/// The world X and Y numeric arguments (args 1 and 2) as Float64 row arrays.
fn world_xy_arrays(args: &[ColumnarValue], n: usize) -> Result<(Float64Array, Float64Array)> {
    let x = args[1]
        .clone()
        .cast_to(&DataType::Float64, None)?
        .into_array(n)?;
    let y = args[2]
        .clone()
        .cast_to(&DataType::Float64, None)?
        .into_array(n)?;
    Ok((as_float64_array(&x)?.clone(), as_float64_array(&y)?.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::DataType;
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{RASTER, WKB_GEOMETRY};
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::create::create_array;
    use sedona_testing::rasters::{build_noninvertible_raster, generate_test_rasters};
    use sedona_testing::testers::ScalarUdfTester;

    #[test]
    fn udf_docs() {
        let udf: ScalarUDF = rs_worldtorastercoordy_udf().into();
        assert_eq!(udf.name(), "rs_worldtorastercoordy");

        let udf: ScalarUDF = rs_worldtorastercoordx_udf().into();
        assert_eq!(udf.name(), "rs_worldtorastercoordx");

        let udf: ScalarUDF = rs_worldtorastercoord_udf().into();
        assert_eq!(udf.name(), "rs_worldtorastercoord");
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
        let noninvertible_rasters = build_noninvertible_raster();
        let result_err =
            tester.invoke_array_scalar_scalar(Arc::new(noninvertible_rasters), 2.0_f64, 3.0_f64);
        assert!(result_err.is_err());
        assert!(
            result_err
                .err()
                .unwrap()
                .to_string()
                .contains("determinant is zero")
        );
    }

    #[rstest]
    fn udf_invoke_xy_from_point(#[values(Coord::Y, Coord::X)] coord: Coord) {
        // The (raster, point) overload reads the world coordinate from a point
        // geometry and answers exactly like the numeric form: world (2, 3) is
        // raster 1's origin -> raster coordinate (0, 0). Raster 0 is nulled.
        let udf = match coord {
            Coord::X => rs_worldtorastercoordx_udf(),
            Coord::Y => rs_worldtorastercoordy_udf(),
        };
        let tester = ScalarUdfTester::new(udf.into(), vec![RASTER, WKB_GEOMETRY]);
        let rasters = generate_test_rasters(2, Some(0)).unwrap();
        let points = create_array(&[Some("POINT (2 3)"), Some("POINT (2 3)")], &WKB_GEOMETRY);
        let expected: Arc<dyn arrow_array::Array> =
            Arc::new(arrow_array::Int64Array::from(vec![None, Some(0_i64)]));
        let result = tester
            .invoke_array_array(Arc::new(rasters), points)
            .unwrap();
        assert_array_equal(&result, &expected);
    }

    #[rstest]
    fn udf_invoke_pt_from_point() {
        // The combined (raster, point) overload returns the raster-coordinate
        // point, matching the numeric form: world (2, 3) -> POINT (0 0).
        let tester = ScalarUdfTester::new(
            rs_worldtorastercoord_udf().into(),
            vec![RASTER, WKB_GEOMETRY],
        );
        let rasters = generate_test_rasters(2, Some(0)).unwrap();
        let points = create_array(&[Some("POINT (2 3)"), Some("POINT (2 3)")], &WKB_GEOMETRY);
        let expected = &create_array(&[None, Some("POINT (0 0)")], &WKB_GEOMETRY);
        let result = tester
            .invoke_array_array(Arc::new(rasters), points)
            .unwrap();
        assert_array_equal(&result, expected);
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
        let noninvertible_rasters = build_noninvertible_raster();
        let result_err =
            tester.invoke_array_scalar_scalar(Arc::new(noninvertible_rasters), 2.0_f64, 3.0_f64);
        assert!(result_err.is_err());
        assert!(
            result_err
                .err()
                .unwrap()
                .to_string()
                .contains("determinant is zero")
        );
    }

    #[rstest]
    fn udf_invoke_xy_with_array_coords(#[values(Coord::Y, Coord::X)] coord: Coord) {
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

        // Use only raster 1 (invertible) with different world coordinates
        // Raster 1: upper_left=(2,3), scale_x=0.1, scale_y=-0.2, skew_x=0.03, skew_y=0.04
        // For world coords (2,3) -> raster (0,0) (the upper left corner)
        // For world coords (2.1, 2.8) -> need to solve the inverse
        let rasters = generate_test_rasters(2, Some(0)).unwrap();
        let world_x = Arc::new(arrow_array::Float64Array::from(vec![2.0, 2.0]));
        let world_y = Arc::new(arrow_array::Float64Array::from(vec![3.0, 3.0]));

        let expected_values = match coord {
            Coord::X => vec![None, Some(0_i64)],
            Coord::Y => vec![None, Some(0_i64)],
        };
        let expected: Arc<dyn arrow_array::Array> =
            Arc::new(arrow_array::Int64Array::from(expected_values));

        let result = tester
            .invoke_arrays(vec![Arc::new(rasters), world_x, world_y])
            .unwrap();
        assert_array_equal(&result, &expected);
    }

    #[rstest]
    fn udf_invoke_xy_with_scalar_raster_array_coords(#[values(Coord::Y, Coord::X)] coord: Coord) {
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
        let scalar_raster = ScalarValue::try_from_array(&rasters, 1).unwrap();

        let world_x = Arc::new(arrow_array::Float64Array::from(vec![2.0, 2.5, 3.25]));
        let world_y = Arc::new(arrow_array::Float64Array::from(vec![3.0, 2.5, 1.75]));

        let expected_coords = match coord {
            Coord::X => vec![Some(0_i64), Some(4_i64), Some(10_i64)],
            Coord::Y => vec![Some(0_i64), Some(3_i64), Some(8_i64)],
        };

        let result = tester
            .invoke(vec![
                ColumnarValue::Scalar(scalar_raster),
                ColumnarValue::Array(world_x.clone()),
                ColumnarValue::Array(world_y.clone()),
            ])
            .unwrap();

        let array = match result {
            ColumnarValue::Array(array) => array,
            ColumnarValue::Scalar(_) => panic!("Expected array result"),
        };

        let expected: Arc<dyn arrow_array::Array> =
            Arc::new(arrow_array::Int64Array::from(expected_coords));
        assert_array_equal(&array, &expected);
    }

    #[test]
    fn udf_invoke_pt_with_scalar_raster_array_coords() {
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
        let scalar_raster = ScalarValue::try_from_array(&rasters, 1).unwrap();

        let world_x = Arc::new(arrow_array::Float64Array::from(vec![2.0, 2.5, 3.25]));
        let world_y = Arc::new(arrow_array::Float64Array::from(vec![3.0, 2.5, 1.75]));

        let result = tester
            .invoke(vec![
                ColumnarValue::Scalar(scalar_raster),
                ColumnarValue::Array(world_x.clone()),
                ColumnarValue::Array(world_y.clone()),
            ])
            .unwrap();

        let array = match result {
            ColumnarValue::Array(array) => array,
            ColumnarValue::Scalar(_) => panic!("Expected array result"),
        };

        let expected = create_array(
            &[
                Some("POINT (0 0)"),
                Some("POINT (4 3)"),
                Some("POINT (10 8)"),
            ],
            &WKB_GEOMETRY,
        );
        assert_array_equal(&array, &expected);
    }
}
