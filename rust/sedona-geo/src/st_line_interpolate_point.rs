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
use std::{io::Write, sync::Arc};

use arrow_array::builder::BinaryBuilder;
use arrow_schema::DataType;
use datafusion_common::{
    DataFusionError, cast::as_float64_array, error::Result, exec_datafusion_err,
};
use datafusion_expr::ColumnarValue;
use geo_traits::{CoordTrait, GeometryTrait, LineStringTrait};
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{ScalarKernelRef, SedonaScalarKernel},
};
use sedona_functions::executor::WkbExecutor;
use sedona_geometry::{
    error::SedonaGeometryError,
    wkb_factory::{write_wkb_coord_trait, write_wkb_point_header},
};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};
use wkb::reader::Wkb;

/// ST_LineInterpolatePoint() implementation
pub fn st_line_interpolate_point_impl() -> Vec<ScalarKernelRef> {
    ItemCrsKernel::wrap_impl(STLineInterpolatePoint {})
}

#[derive(Debug)]
struct STLineInterpolatePoint {}

impl SedonaScalarKernel for STLineInterpolatePoint {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry(), ArgMatcher::is_numeric()],
            WKB_GEOMETRY,
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let mut builder =
            BinaryBuilder::with_capacity(executor.num_iterations(), 25 * executor.num_iterations());

        let arg1_array = args[1]
            .cast_to(&DataType::Float64, None)?
            .to_array(executor.num_iterations())?;
        let arg1_float = as_float64_array(&arg1_array)?;

        let mut arg1_iter = arg1_float.into_iter();
        executor.execute_wkb_void(|maybe_wkb| {
            let maybe_ratio = arg1_iter.next().unwrap();
            match (maybe_wkb, maybe_ratio) {
                (Some(wkb), Some(ratio)) => {
                    invoke_scalar(wkb, ratio, &mut builder)?;
                    builder.append_value([]);
                }
                _ => builder.append_null(),
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

/// Writes the point at `ratio` of the line's 2D length as a WKB point
///
/// The point has the dimensions of the input line: its Z and M values are
/// interpolated between the two vertices on either side of it. A ratio below
/// 0 or above 1 gives the first or last vertex of the line.
fn invoke_scalar(geom: &Wkb, ratio: f64, writer: &mut impl Write) -> Result<()> {
    let geo_traits::GeometryType::LineString(line) = geom.as_type() else {
        return Err(exec_datafusion_err!(
            "Input to STLineInterpolatePoint must be linestring"
        ));
    };

    let (Some(first), Some(last)) = (line.coords().next(), line.coords().next_back()) else {
        return Err(exec_datafusion_err!("Failed to interpolate point"));
    };

    let segments = || line.coords().zip(line.coords().skip(1));
    let length = segments().fold(0.0, |total, (start, end)| total + distance_2d(&start, &end));
    let distance = ratio * length;

    let result = if distance <= 0.0 {
        write_wkb_point_from_coord(writer, &first)
    } else {
        let mut distance_remaining = distance;
        let mut segment = None;
        for (start, end) in segments() {
            let segment_length = distance_2d(&start, &end);
            if segment_length < distance_remaining {
                distance_remaining -= segment_length;
            } else {
                segment = Some((start, end, segment_length));
                break;
            }
        }

        match segment {
            Some((start, end, segment_length)) => {
                write_wkb_point_between(writer, &start, &end, distance_remaining, segment_length)
            }
            // The distance is at least the length of the line
            None => write_wkb_point_from_coord(writer, &last),
        }
    };

    result.map_err(|e| DataFusionError::External(Box::new(e)))
}

fn distance_2d(start: &impl CoordTrait<T = f64>, end: &impl CoordTrait<T = f64>) -> f64 {
    (end.x() - start.x()).hypot(end.y() - start.y())
}

fn write_wkb_point_from_coord(
    writer: &mut impl Write,
    coord: &impl CoordTrait<T = f64>,
) -> std::result::Result<(), SedonaGeometryError> {
    write_wkb_point_header(writer, coord.dim())?;
    write_wkb_coord_trait(writer, coord)
}

/// Writes the point `distance` along the segment from `start` to `end`,
/// interpolating every dimension of the coordinates
fn write_wkb_point_between(
    writer: &mut impl Write,
    start: &impl CoordTrait<T = f64>,
    end: &impl CoordTrait<T = f64>,
    distance: f64,
    segment_length: f64,
) -> std::result::Result<(), SedonaGeometryError> {
    let dim = start.dim();
    write_wkb_point_header(writer, dim)?;
    for n in 0..dim.size() {
        let (start_n, end_n) = (start.nth_or_panic(n), end.nth_or_panic(n));
        let value = start_n + (end_n - start_n) * distance / segment_length;
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use arrow_array::{ArrayRef, Float64Array};
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS, WKB_VIEW_GEOMETRY};
    use sedona_testing::{
        compare::{assert_array_equal, assert_scalar_equal_wkb_geometry},
        create::create_array,
        testers::ScalarUdfTester,
    };

    use super::*;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf =
            SedonaScalarUDF::from_impl("st_lineinterpolatepoint", st_line_interpolate_point_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![sedona_type, SedonaType::Arrow(DataType::Float64)],
        );

        assert_eq!(tester.return_type().unwrap(), WKB_GEOMETRY);

        assert_scalar_equal_wkb_geometry(
            &tester
                .invoke_scalar_scalar("LINESTRING (0 0, 1 0)", 0.5)
                .unwrap(),
            Some("POINT (0.5 0)"),
        );
    }

    #[rstest]
    fn udf_invoke_item_crs(#[values(WKB_GEOMETRY_ITEM_CRS.clone())] sedona_type: SedonaType) {
        let udf =
            SedonaScalarUDF::from_impl("st_lineinterpolatepoint", st_line_interpolate_point_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Float64)],
        );
        tester.assert_return_type(sedona_type);

        let result = tester
            .invoke_scalar_scalar("LINESTRING (0 0, 1 0)", 0.5)
            .unwrap();
        tester.assert_scalar_result_equals(result, "POINT (0.5 0)");
    }

    #[rstest]
    fn udf_zm(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf =
            SedonaScalarUDF::from_impl("st_lineinterpolatepoint", st_line_interpolate_point_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Float64)],
        );

        let cases = [
            ("LINESTRING Z (0 0 10, 0 2 12)", 0.5, "POINT Z (0 1 11)"),
            ("LINESTRING M (0 0 10, 0 2 12)", 0.5, "POINT M (0 1 11)"),
            (
                "LINESTRING ZM (0 0 10 20, 0 1 11 21, 0 2 12 22)",
                0.25,
                "POINT ZM (0 0.5 10.5 20.5)",
            ),
            (
                "LINESTRING ZM (0 0 10 20, 0 1 11 21, 0 2 12 22)",
                0.75,
                "POINT ZM (0 1.5 11.5 21.5)",
            ),
            // The ends of the line, and fractions past them, return its first or last vertex
            ("LINESTRING Z (0 0 10, 0 2 12)", 0.0, "POINT Z (0 0 10)"),
            ("LINESTRING Z (0 0 10, 0 2 12)", 1.0, "POINT Z (0 2 12)"),
            ("LINESTRING Z (0 0 10, 0 2 12)", -0.5, "POINT Z (0 0 10)"),
            ("LINESTRING Z (0 0 10, 0 2 12)", 1.5, "POINT Z (0 2 12)"),
        ];
        for (line, fraction, expected) in cases {
            assert_scalar_equal_wkb_geometry(
                &tester.invoke_scalar_scalar(line, fraction).unwrap(),
                Some(expected),
            );
        }

        let lines = create_array(
            &[
                Some("LINESTRING Z (0 0 10, 0 2 12)"),
                Some("LINESTRING M (0 0 10, 0 2 12)"),
                None,
                Some("LINESTRING ZM (0 0 10 20, 0 1 11 21, 0 2 12 22)"),
            ],
            &sedona_type,
        );
        let fractions: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(0.5),
            Some(0.25),
            Some(0.5),
            None,
        ]));
        let expected = create_array(
            &[
                Some("POINT Z (0 1 11)"),
                Some("POINT M (0 0.5 10.5)"),
                None,
                None,
            ],
            &WKB_GEOMETRY,
        );
        assert_array_equal(
            &tester.invoke_arrays(vec![lines, fractions]).unwrap(),
            &expected,
        );
    }

    #[test]
    fn udf_errors() {
        let udf =
            SedonaScalarUDF::from_impl("st_lineinterpolatepoint", st_line_interpolate_point_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![WKB_GEOMETRY, SedonaType::Arrow(DataType::Float64)],
        );

        let err = tester
            .invoke_scalar_scalar("LINESTRING EMPTY", 0.5)
            .unwrap_err();
        assert!(err.to_string().contains("Failed to interpolate point"));

        let err = tester.invoke_scalar_scalar("POINT (0 0)", 0.5).unwrap_err();
        assert!(err.to_string().contains("must be linestring"));
    }
}
