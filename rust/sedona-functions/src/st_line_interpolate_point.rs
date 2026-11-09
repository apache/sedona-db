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
use datafusion_expr::{ColumnarValue, Volatility};
use geo_traits::{CoordTrait, Dimensions, GeometryTrait, GeometryType, LineStringTrait};
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{SedonaScalarKernel, SedonaScalarUDF},
};
use sedona_geometry::wkb_factory::{write_wkb_empty_point, write_wkb_point_header};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

use crate::executor::WkbExecutor;

/// ST_LineInterpolatePoint() scalar UDF implementation
///
/// Native implementation of ST_LineInterpolatePoint for geometry
pub fn st_line_interpolate_point_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_lineinterpolatepoint",
        ItemCrsKernel::wrap_impl(vec![Arc::new(STLineInterpolatePoint)]),
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct STLineInterpolatePoint;

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

        // Reused for every line to avoid allocating per row. A scalar geometry is
        // the same line in every row, so it is only filled in once.
        let geometry_is_scalar = matches!(args[0], ColumnarValue::Scalar(_));
        let mut prepared = PreparedLine::default();

        executor.execute_wkb_void(|maybe_wkb| {
            let maybe_fraction = arg1_iter.next().unwrap();
            match (maybe_wkb, maybe_fraction) {
                (Some(wkb), Some(fraction)) => {
                    let GeometryType::LineString(line) = wkb.as_type() else {
                        return Err(exec_datafusion_err!(
                            "Input to STLineInterpolatePoint must be linestring"
                        ));
                    };
                    if !geometry_is_scalar || prepared.coords.is_empty() {
                        prepared.fill(line);
                    }
                    prepared.write_point(fraction, line.dim(), &mut builder)?;
                    builder.append_value([]);
                }
                _ => builder.append_null(),
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

/// The vertices of a line, and how far along the line each of them is in 2D
#[derive(Default)]
struct PreparedLine {
    coords: Vec<[f64; 4]>,
    cumulative_lengths: Vec<f64>,
}

impl PreparedLine {
    /// Replaces the contents with the vertices of `line`
    fn fill(&mut self, line: &impl LineStringTrait<T = f64>) {
        let num_dims = line.dim().size();

        self.coords.clear();
        self.coords.extend(line.coords().map(|coord| {
            let mut values = [0.0; 4];
            for (n, value) in values.iter_mut().enumerate().take(num_dims) {
                *value = coord.nth_or_panic(n);
            }
            values
        }));

        self.cumulative_lengths.clear();
        self.cumulative_lengths.push(0.0);
        let mut length = 0.0;
        for pair in self.coords.windows(2) {
            length += (pair[1][0] - pair[0][0]).hypot(pair[1][1] - pair[0][1]);
            self.cumulative_lengths.push(length);
        }
    }

    /// Writes the point at `fraction` of the line's 2D length as a WKB point
    ///
    /// The point has the dimensions of the line: its Z and M values are
    /// interpolated between the two vertices on either side of it. A fraction at
    /// or below 0 gives the first vertex of the line, and a fraction at or above 1
    /// gives the last one. A line with a single vertex gives that vertex for any
    /// fraction. Otherwise, a NaN fraction gives an empty point.
    fn write_point(&self, fraction: f64, dim: Dimensions, writer: &mut impl Write) -> Result<()> {
        let (Some(&first), Some(&last)) = (self.coords.first(), self.coords.last()) else {
            return Err(exec_datafusion_err!("Failed to interpolate point"));
        };

        let point = if self.coords.len() == 1 {
            first
        } else if fraction.is_nan() {
            return write_wkb_empty_point(writer, dim)
                .map_err(|e| DataFusionError::External(Box::new(e)));
        } else if fraction >= 1.0 {
            last
        } else {
            self.point_before_end(fraction, first, dim.size())
        };

        write_wkb_point_header(writer, dim).map_err(|e| DataFusionError::External(Box::new(e)))?;
        for value in &point[..dim.size()] {
            writer.write_all(&value.to_le_bytes())?;
        }

        Ok(())
    }

    /// The point at a `fraction` below 1 of a line with at least two vertices
    fn point_before_end(&self, fraction: f64, first: [f64; 4], num_dims: usize) -> [f64; 4] {
        let length = self.cumulative_lengths[self.cumulative_lengths.len() - 1];
        let distance = fraction * length;
        if distance <= 0.0 {
            return first;
        }

        // The segment that starts at or before the distance and ends after it
        let segment = self
            .cumulative_lengths
            .partition_point(|&d| d <= distance)
            .saturating_sub(1)
            .min(self.coords.len() - 2);
        let (start, end) = (self.coords[segment], self.coords[segment + 1]);
        let segment_length = (end[0] - start[0]).hypot(end[1] - start[1]);

        // Dividing before multiplying keeps the result finite for large coordinates
        let segment_fraction = if segment_length > 0.0 {
            ((distance - self.cumulative_lengths[segment]) / segment_length).min(1.0)
        } else {
            0.0
        };

        let mut values = [0.0; 4];
        for n in 0..num_dims {
            values[n] = start[n] + (end[n] - start[n]) * segment_fraction;
        }
        values
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{ArrayRef, Float64Array};
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS, WKB_VIEW_GEOMETRY};
    use sedona_testing::{
        compare::{assert_array_equal, assert_scalar_equal_wkb_geometry},
        create::create_array,
        testers::ScalarUdfTester,
    };

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_line_interpolate_point_udf().into();
        assert_eq!(udf.name(), "st_lineinterpolatepoint");
    }

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(
            st_line_interpolate_point_udf().into(),
            vec![sedona_type, SedonaType::Arrow(DataType::Float64)],
        );

        assert_eq!(tester.return_type().unwrap(), WKB_GEOMETRY);

        let cases = [
            ("LINESTRING (0 0, 1 0)", 0.5, "POINT (0.5 0)"),
            ("LINESTRING (0 0, 1 0, 1 2)", 0.25, "POINT (0.75 0)"),
            ("LINESTRING (0 0, 1 0, 1 2)", 0.5, "POINT (1 0.5)"),
            // A fraction that lands on a vertex gives that vertex
            ("LINESTRING (0 0, 1 0, 1 3)", 0.25, "POINT (1 0)"),
            // A zero-length segment is skipped
            ("LINESTRING (0 0, 1 0, 1 0, 1 2)", 0.5, "POINT (1 0.5)"),
            // The ends of the line, and fractions past them, give its first or last vertex
            ("LINESTRING (0 0, 1 0)", 0.0, "POINT (0 0)"),
            ("LINESTRING (0 0, 1 0)", -0.5, "POINT (0 0)"),
            ("LINESTRING (0 0, 1 0)", 1.5, "POINT (1 0)"),
            // A fraction of 1 gives the last vertex exactly, not a point computed along the
            // last segment that rounds away from it
            ("LINESTRING (5.7 8, 0.6 1.2)", 1.0, "POINT (0.6 1.2)"),
            // A line without length gives its first vertex
            ("LINESTRING (1 1, 1 1)", 0.5, "POINT (1 1)"),
            ("LINESTRING (1 1)", 0.5, "POINT (1 1)"),
            // A line with a single vertex gives that vertex for any fraction
            ("LINESTRING (1 1)", f64::NAN, "POINT (1 1)"),
            ("LINESTRING (1 1)", f64::NEG_INFINITY, "POINT (1 1)"),
            // A NaN fraction has no position on the line
            ("LINESTRING (0 0, 1 0)", f64::NAN, "POINT EMPTY"),
        ];
        for (line, fraction, expected) in cases {
            assert_scalar_equal_wkb_geometry(
                &tester.invoke_scalar_scalar(line, fraction).unwrap(),
                Some(expected),
            );
        }
    }

    #[rstest]
    fn udf_invoke_item_crs(#[values(WKB_GEOMETRY_ITEM_CRS.clone())] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(
            st_line_interpolate_point_udf().into(),
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
        let tester = ScalarUdfTester::new(
            st_line_interpolate_point_udf().into(),
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
            // The ends of the line, and fractions past them, give its first or last vertex
            ("LINESTRING Z (0 0 10, 0 2 12)", 0.0, "POINT Z (0 0 10)"),
            ("LINESTRING Z (0 0 10, 0 2 12)", 1.0, "POINT Z (0 2 12)"),
            ("LINESTRING Z (0 0 10, 0 2 12)", -0.5, "POINT Z (0 0 10)"),
            ("LINESTRING Z (0 0 10, 0 2 12)", 1.5, "POINT Z (0 2 12)"),
            ("LINESTRING Z (0 0 10, 0 2 12)", f64::NAN, "POINT Z EMPTY"),
            // Large Z values stay finite
            (
                "LINESTRING Z (0 0 0, 1e150 0 1e200)",
                0.5,
                "POINT Z (5e149 0 5e199)",
            ),
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
                Some("LINESTRING (0 0, 4 0)"),
            ],
            &sedona_type,
        );
        let fractions: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(0.5),
            Some(0.25),
            Some(0.5),
            None,
            Some(0.75),
        ]));
        let expected = create_array(
            &[
                Some("POINT Z (0 1 11)"),
                Some("POINT M (0 0.5 10.5)"),
                None,
                None,
                Some("POINT (3 0)"),
            ],
            &WKB_GEOMETRY,
        );
        assert_array_equal(
            &tester.invoke_arrays(vec![lines, fractions]).unwrap(),
            &expected,
        );
    }

    #[rstest]
    fn udf_scalar_line(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(
            st_line_interpolate_point_udf().into(),
            vec![sedona_type, SedonaType::Arrow(DataType::Float64)],
        );

        let fractions: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(0.75),
            None,
            Some(0.0),
            Some(f64::NAN),
            Some(0.25),
            Some(1.0),
        ]));
        let expected = create_array(
            &[
                Some("POINT Z (1 1.25 12.875)"),
                None,
                Some("POINT Z (0 0 10)"),
                Some("POINT Z EMPTY"),
                Some("POINT Z (0.75 0 10.75)"),
                Some("POINT Z (1 2 14)"),
            ],
            &WKB_GEOMETRY,
        );
        assert_array_equal(
            &tester
                .invoke_scalar_array("LINESTRING Z (0 0 10, 1 0 11, 1 2 14)", fractions)
                .unwrap(),
            &expected,
        );
    }

    #[test]
    fn udf_errors() {
        let tester = ScalarUdfTester::new(
            st_line_interpolate_point_udf().into(),
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
