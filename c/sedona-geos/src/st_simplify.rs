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

use std::sync::Arc;

use arrow_array::builder::BinaryBuilder;
use arrow_schema::DataType;
use datafusion_common::{cast::as_float64_array, DataFusionError, Result};
use datafusion_expr::ColumnarValue;
use geos::{Geom, Geometry, GeometryTypes};
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_geometry::wkb_factory::WKB_MIN_PROBABLE_BYTES;
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

use crate::executor::GeosExecutor;

/// ST_Simplify() implementation using the geos crate
pub fn st_simplify_impl() -> ScalarKernelRef {
    Arc::new(STSimplify {})
}

#[derive(Debug)]
struct STSimplify {}

impl SedonaScalarKernel for STSimplify {
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
        let executor = GeosExecutor::new(arg_types, args);

        let tolerance_value = args[1]
            .cast_to(&DataType::Float64, None)?
            .to_array(executor.num_iterations())?;
        let tolerance_array = as_float64_array(&tolerance_value)?;
        let mut tolerance_iter = tolerance_array.iter();

        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
        );

        executor.execute_wkb_void(|wkb| {
            match (wkb, tolerance_iter.next().unwrap()) {
                (Some(wkb), Some(tolerance)) => {
                    invoke_scalar(&wkb, tolerance, &mut builder)?;
                    builder.append_value([]);
                }
                _ => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(
    geos_geom: &geos::Geometry,
    tolerance: f64,
    writer: &mut impl std::io::Write,
) -> Result<()> {
    let initial_type = geos_geom.geometry_type();
    let geometry = geos_geom
        .simplify(tolerance)
        .map_err(|e| DataFusionError::Execution(format!("Failed to simplify geometry: {e}")))?;

    // GEOS inherently "promotes" a multi-geometry type (e.g., MultiPolygon, MultiLineString)
    // to its simpler, single-component counterpart (Polygon, LineString) if the
    // simplification process reduces the number of resulting components to exactly one.
    //
    // However, to ensure that the geometry type remains invariant after ST_Simplify
    // (e.g., an initial MultiPolygon must always return a MultiPolygon), we revert
    // this promotion by re-wrapping the single component back into its original
    // multi-geometry container.
    let geometry = match (initial_type, geometry.geometry_type()) {
        // If the original was a MultiPolygon but GEOS returned a Polygon (promotion),
        // wrap the Polygon back into a MultiPolygon.
        (GeometryTypes::MultiPolygon, GeometryTypes::Polygon) => {
            Geometry::create_multipolygon(vec![geometry]).map_err(|e| {
                DataFusionError::Execution(format!("Failed to revert geometry promotion: {e}"))
            })?
        }
        // If the original was a MultiLineString but GEOS returned a LineString (promotion),
        // wrap the LineString back into a MultiLineString.
        (GeometryTypes::MultiLineString, GeometryTypes::LineString) => {
            Geometry::create_multiline_string(vec![geometry]).map_err(|e| {
                DataFusionError::Execution(format!("Failed to revert geometry promotion: {e}"))
            })?
        }
        // Keep as is (type is correct)
        _ => geometry,
    };

    let wkb = geometry
        .to_wkb()
        .map_err(|e| DataFusionError::Execution(format!("Failed to convert to wkb: {e}")))?;

    writer.write_all(wkb.as_ref())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Float64Array;
    use datafusion_common::ScalarValue;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::{
        compare::assert_array_equal, create::create_array, testers::ScalarUdfTester,
    };

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_simplify", st_simplify_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Float64)],
        );
        tester.assert_return_type(WKB_GEOMETRY);

        let result = tester
            .invoke_scalar_scalar("LINESTRING(0 0, 1 1, 2 0, 3 1, 4 0)", 1.5)
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING(0 0, 4 0)");

        let wkt_line = "LINESTRING(0 0, 1 1, 2 0, 3 1, 4 0)";
        let result = tester.invoke_scalar_scalar(wkt_line, 0.0).unwrap();
        tester.assert_scalar_result_equals(result, wkt_line);

        let wkt_polygon = "POLYGON((0 0, 0 10, 1 11, 10 10, 10 0, 0 0))";
        let result = tester.invoke_scalar_scalar(wkt_polygon, 1.5).unwrap();
        tester.assert_scalar_result_equals(result, "POLYGON((0 0, 0 10, 10 10, 10 0, 0 0))");

        let wkt_point = "POINT(10 20)";
        let result = tester.invoke_scalar_scalar(wkt_point, 10.0).unwrap();
        tester.assert_scalar_result_equals(result, wkt_point);

        let result = tester.invoke_scalar_scalar(ScalarValue::Null, 1.0).unwrap();
        assert!(result.is_null());

        let result = tester
            .invoke_scalar_scalar("LINESTRING(0 0, 1 1, 2 2)", ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());

        let empty_line = "LINESTRING EMPTY";
        let result = tester.invoke_scalar_scalar(empty_line, 1.0).unwrap();
        tester.assert_scalar_result_equals(result, empty_line);

        let wkt_line_short = "LINESTRING(0 0, 0 1)";
        let result = tester.invoke_scalar_scalar(wkt_line_short, 1.0).unwrap();
        tester.assert_scalar_result_equals(result, wkt_line_short);

        let input_wkt_t_scalar = vec![
            Some("LINESTRING(0 0, 1 1, 2 0, 3 1, 4 0)"), // Simplify: (0 0, 4 0)
            Some("POLYGON((0 0, 0 10, 1 11, 10 10, 10 0, 0 0))"), // Simplify: (0 0, 0 10, 10 10, 10 0, 0 0)
            Some("POINT(5 5)"),
            Some("LINESTRING EMPTY"),
            None,
        ];

        let expected_t_scalar = create_array(
            &[
                Some("LINESTRING (0 0, 4 0)"),
                Some("POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))"),
                Some("POINT (5 5)"),
                Some("LINESTRING EMPTY"),
                None,
            ],
            &WKB_GEOMETRY,
        );

        assert_array_equal(
            &tester
                .invoke_wkb_array_scalar(input_wkt_t_scalar, 1.5)
                .unwrap(),
            &expected_t_scalar,
        );

        let input_wkt_array = vec![
            Some("LINESTRING (0 0, 0 10, 0 51, 50 20, 30 20, 7 32)"),
            Some("LINESTRING (0 0, 0 10, 0 51, 50 20, 30 20, 7 32)"),
            Some("LINESTRING (0 0, 0 10, 0 51, 50 20, 30 20, 7 32)"),
            None,
        ];

        let tolerance_array: Arc<Float64Array> = Arc::new(Float64Array::from(vec![
            Some(2.0),
            Some(10.0),
            Some(50.0),
            Some(5.0),
        ]));

        let expected_array = create_array(
            &[
                Some("LINESTRING (0 0, 0 51, 50 20, 30 20, 7 32)"), // Tolerance 2.0
                Some("LINESTRING (0 0, 0 51, 50 20, 7 32)"),        // Tolerance 10.0
                Some("LINESTRING (0 0, 7 32)"),                     // Tolerance 50.0
                None,
            ],
            &WKB_GEOMETRY,
        );

        assert_array_equal(
            &tester
                .invoke_arrays(vec![
                    create_array(&input_wkt_array, &sedona_type),
                    tolerance_array,
                ])
                .unwrap(),
            &expected_array,
        );
    }

    #[rstest]
    fn simplify_collapsed(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_simplify", st_simplify_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Float64)],
        );
        tester.assert_return_type(WKB_GEOMETRY);

        // A LineString that is simplified, but does not collapse.
        let result = tester
            .invoke_scalar_scalar("LINESTRING(0 0, 1 0, 2 0.1, 3 0)", 5.0)
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING(0 0, 3 0)");

        // A narrow polygon that collapses to a line, which is invalid.
        // Should result in an empty polygon.
        let result = tester
            .invoke_scalar_scalar("POLYGON((0 0, 10 0, 10 0.1, 0 0.1, 0 0))", 1.0)
            .unwrap();
        tester.assert_scalar_result_equals(result, "POLYGON EMPTY");

        // A small polygon that collapses to a point, which is invalid.
        // Should result in an empty polygon.
        let result = tester
            .invoke_scalar_scalar("POLYGON((0 0, 0.1 0, 0.1 0.1, 0 0.1, 0 0))", 1.0)
            .unwrap();
        tester.assert_scalar_result_equals(result, "POLYGON EMPTY");

        // A triangle that collapses.
        let result = tester
            .invoke_scalar_scalar("POLYGON((0 0, 10 0, 5 1, 0 0))", 2.0)
            .unwrap();
        tester.assert_scalar_result_equals(result, "POLYGON EMPTY");

        // A LineString that looks like a point but is preserved.
        // Simplify only removes vertices, it doesn't change geometry type.
        let result = tester
            .invoke_scalar_scalar("LINESTRING(0 0, 0.1 0.1, 0.2 0.2)", 1.0)
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING(0 0, 0.2 0.2)");

        // MultiPoint is unaffected by simplify.
        let result = tester
            .invoke_scalar_scalar("MULTIPOINT((0 0), (0.1 0.1), (5 5))", 1.0)
            .unwrap();
        tester.assert_scalar_result_equals(result, "MULTIPOINT((0 0), (0.1 0.1), (5 5))");

        // MultiLineString where one component is simplified.
        let result = tester
            .invoke_scalar_scalar(
                "MULTILINESTRING((0 0, 5 0.1, 10 0), (20 20, 21 21, 22 22))",
                1.0,
            )
            .unwrap();
        tester.assert_scalar_result_equals(result, "MULTILINESTRING((0 0, 10 0), (20 20, 22 22))");

        // MultiPolygon where one component collapses and is removed.
        let result = tester
            .invoke_scalar_scalar(
                "MULTIPOLYGON(((0 0, 0.1 0, 0.1 0.1, 0 0.1, 0 0)), ((10 10, 20 10, 20 20, 10 20, 10 10)))",
                1.0,
            )
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            "MULTIPOLYGON(((10 10, 20 10, 20 20, 10 20, 10 10)))",
        );

        // MultiPolygon where all components collapse.
        let result = tester
            .invoke_scalar_scalar(
                "MULTIPOLYGON(((0 0, 0.1 0, 0.1 0.1, 0 0.1, 0 0)), ((1 1, 1.1 1, 1.1 1.1, 1 1.1, 1 1)))",
                1.0,
            )
            .unwrap();
        tester.assert_scalar_result_equals(result, "MULTIPOLYGON EMPTY");

        // GeometryCollection where a component collapses, and the collection is "promoted" to a single geometry.
        let result = tester
            .invoke_scalar_scalar(
                "GEOMETRYCOLLECTION(POLYGON((0 0, 0.1 0, 0.1 0.1, 0 0.1, 0 0)), LINESTRING(10 10, 15 10.1, 20 10))",
                1.0,
            )
            .unwrap();
        tester.assert_scalar_result_equals(result, "GEOMETRYCOLLECTION(LINESTRING(10 10, 20 10))");

        // GeometryCollection where all components collapse.
        let result = tester
            .invoke_scalar_scalar(
                "GEOMETRYCOLLECTION(POLYGON((0 0, 0.1 0, 0.1 0.1, 0 0.1, 0 0)), POLYGON((1 1, 1.1 1, 1.1 1.1, 1 1.1, 1 1)))",
                1.0,
            )
            .unwrap();
        tester.assert_scalar_result_equals(result, "GEOMETRYCOLLECTION EMPTY");
    }

    #[rstest]
    fn simplify_edge_cases(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_simplify", st_simplify_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Float64)],
        );

        let result = tester
            .invoke_scalar_scalar("LINESTRING(0 0, 1 1, 2 0, 3 1, 4 0)", 0.0)
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING(0 0, 1 1, 2 0, 3 1, 4 0)");

        // Very large tolerance
        let result = tester
            .invoke_scalar_scalar("LINESTRING(0 0, 1 1, 2 2, 3 3)", 1000.0)
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING(0 0, 3 3)");

        // Two-point LineString should never simplify to single point
        let result = tester
            .invoke_scalar_scalar("LINESTRING(0 0, 10 10)", 100.0)
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING(0 0, 10 10)");
    }

    #[rstest]
    fn simplify_polygon_with_holes(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let udf = SedonaScalarUDF::from_kernel("st_simplify", st_simplify_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Float64)],
        );

        let result = tester
            .invoke_scalar_scalar(
                "POLYGON((0 0, 0 100, 1 101, 100 100, 100 0, 0 0), (20 20, 20 80, 21 81, 80 80, 80 20, 20 20))",
                10.0,
            )
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            "POLYGON((0 0, 0 100, 100 100, 100 0, 0 0), (20 20, 20 80, 80 80, 80 20, 20 20))",
        );

        let result = tester
            .invoke_scalar_scalar(
                "POLYGON((0 0, 0 100, 100 100, 100 0, 0 0), (40 40, 40.1 40, 40.1 40.1, 40 40.1, 40 40))",
                1.0,
            )
            .unwrap();
        // When hole becomes invalid, it should be removed, leaving just outer ring
        tester.assert_scalar_result_equals(result, "POLYGON((0 0, 0 100, 100 100, 100 0, 0 0))");

        // Polygon where outer ring collapses - entire polygon becomes empty
        let result = tester
            .invoke_scalar_scalar(
                "POLYGON((0 0, 0.1 0, 0.1 0.1, 0 0.1, 0 0), (0.02 0.02, 0.08 0.02, 0.08 0.08, 0.02 0.08, 0.02 0.02))",
                1.0,
            )
            .unwrap();
        tester.assert_scalar_result_equals(result, "POLYGON EMPTY");
    }

    #[rstest]
    fn simplify_multi_geometries_mixed(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let udf = SedonaScalarUDF::from_kernel("st_simplify", st_simplify_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Float64)],
        );

        // MultiLineString with one empty component
        let result = tester
            .invoke_scalar_scalar("MULTILINESTRING((0 0, 10 0), EMPTY)", 1.0)
            .unwrap();
        tester.assert_scalar_result_equals(result, "MULTILINESTRING((0 0, 10 0))");

        // MultiLineString where multiple components simplify differently
        let result = tester
            .invoke_scalar_scalar(
                "MULTILINESTRING((0 0, 1 0.1, 2 0.2, 3 0), (10 10, 11 10, 12 10), (20 20, 21 25, 22 20))",
                1.0,
            )
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            "MULTILINESTRING((0 0, 3 0), (10 10, 12 10), (20 20, 21 25, 22 20))",
        );

        // MultiPolygon with mixed valid/collapsed polygons (should keep MultiPolygon type)
        let result = tester
            .invoke_scalar_scalar(
                "MULTIPOLYGON(((0 0, 100 0, 100 100, 0 100, 0 0)), ((200 200, 200.1 200, 200.1 200.1, 200 200.1, 200 200)))",
                1.0,
            )
            .unwrap();
        // This should remain a MULTIPOLYGON, not unwrap to POLYGON
        tester.assert_scalar_result_equals(
            result,
            "MULTIPOLYGON(((0 0, 100 0, 100 100, 0 100, 0 0)))",
        );

        // MultiPolygon with three polygons, middle one collapses
        let result = tester
            .invoke_scalar_scalar(
                "MULTIPOLYGON(((0 0, 10 0, 10 10, 0 10, 0 0)), ((20 20, 20.1 20, 20.1 20.1, 20 20.1, 20 20)), ((30 30, 40 30, 40 40, 30 40, 30 30)))",
                1.0,
            )
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            "MULTIPOLYGON(((0 0, 10 0, 10 10, 0 10, 0 0)), ((30 30, 40 30, 40 40, 30 40, 30 30)))",
        );
    }

    #[rstest]
    fn simplify_geometry_collection_complex(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let udf = SedonaScalarUDF::from_kernel("st_simplify", st_simplify_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Float64)],
        );

        let result = tester
            .invoke_scalar_scalar(
                "GEOMETRYCOLLECTION(POINT EMPTY, LINESTRING(10 10, 20 10))",
                1.0,
            )
            .unwrap();
        tester.assert_scalar_result_equals(result, "GEOMETRYCOLLECTION(LINESTRING(10 10, 20 10))");

        // GeometryCollection with mixed types - all valid
        let result = tester
            .invoke_scalar_scalar(
                "GEOMETRYCOLLECTION(POINT(0 0), LINESTRING(10 10, 11 10.1, 12 10), POLYGON((20 20, 30 20, 30 30, 20 30, 20 20)))",
                1.0,
            )
            .unwrap();
        tester.assert_scalar_result_equals(
                result,
                "GEOMETRYCOLLECTION(POINT(0 0), LINESTRING(10 10, 12 10), POLYGON((20 20, 30 20, 30 30, 20 30, 20 20)))"
            );

        // GeometryCollection with some collapsed geometries
        let result = tester
            .invoke_scalar_scalar(
                "GEOMETRYCOLLECTION(POINT(0 0), POLYGON((1 1, 1.1 1, 1.1 1.1, 1 1.1, 1 1)), LINESTRING(10 10, 20 10))",
                1.0,
            )
            .unwrap();
        // Collapsed polygon should be removed
        tester.assert_scalar_result_equals(
            result,
            "GEOMETRYCOLLECTION(POINT(0 0), LINESTRING(10 10, 20 10))",
        );

        // Nested GeometryCollection (if GEOS supports it)
        let result = tester
            .invoke_scalar_scalar(
                "GEOMETRYCOLLECTION(GEOMETRYCOLLECTION(POINT(0 0)), LINESTRING(10 10, 11 10.1, 12 10))",
                1.0,
            )
            .unwrap();
        // Should preserve nesting or flatten according to GEOS behavior
        tester.assert_scalar_result_equals(
            result,
            "GEOMETRYCOLLECTION(GEOMETRYCOLLECTION(POINT(0 0)), LINESTRING(10 10, 12 10))",
        );

        // GeometryCollection with MultiGeometry inside
        let result = tester
            .invoke_scalar_scalar(
                "GEOMETRYCOLLECTION(MULTIPOINT((0 0), (1 1)), MULTILINESTRING((10 10, 11 10.1, 12 10), (20 20, 21 21)))",
                1.0,
            )
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            "GEOMETRYCOLLECTION(MULTIPOINT((0 0), (1 1)), MULTILINESTRING((10 10, 12 10), (20 20, 21 21)))"
        );
    }

    #[rstest]
    fn simplify_collinear_points(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let udf = SedonaScalarUDF::from_kernel("st_simplify", st_simplify_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Float64)],
        );

        // Perfectly collinear points - intermediate ones should be removed
        let result = tester
            .invoke_scalar_scalar("LINESTRING(0 0, 1 0, 2 0, 3 0, 4 0, 5 0)", 0.0)
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING(0 0, 5 0)");

        // Nearly collinear points within tolerance
        let result = tester
            .invoke_scalar_scalar("LINESTRING(0 0, 1 0.01, 2 0.02, 3 0.01, 4 0)", 0.1)
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING(0 0, 4 0)");
    }

    #[rstest]
    fn simplify_self_intersecting(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let udf = SedonaScalarUDF::from_kernel("st_simplify", st_simplify_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Float64)],
        );

        // ST_Simplify doesn't guarantee output validity, so these should process without error
        // Self-intersecting LineString (bowtie)
        let result = tester
            .invoke_scalar_scalar("LINESTRING(0 0, 10 10, 10 0, 0 10)", 1.0)
            .unwrap();
        assert!(!result.is_null());

        // Invalid polygon (self-intersecting) - figure-8 shape
        let result = tester
            .invoke_scalar_scalar("POLYGON((0 0, 10 10, 10 0, 0 10, 0 0))", 0.5)
            .unwrap();
        assert!(!result.is_null());
    }

    #[rstest]
    fn simplify_very_small_geometries(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let udf = SedonaScalarUDF::from_kernel("st_simplify", st_simplify_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Float64)],
        );

        // Microscopic polygon relative to tolerance
        let result = tester
            .invoke_scalar_scalar(
                "POLYGON((0 0, 0.00001 0, 0.00001 0.00001, 0 0.00001, 0 0))",
                1.0,
            )
            .unwrap();
        tester.assert_scalar_result_equals(result, "POLYGON EMPTY");

        // Very small LineString
        let result = tester
            .invoke_scalar_scalar("LINESTRING(0 0, 0.00001 0.00001, 0.00002 0.00002)", 1.0)
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING(0 0, 0.00002 0.00002)");
    }

    #[rstest]
    fn simplify_closed_rings(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_simplify", st_simplify_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Float64)],
        );

        // Closed LineString (not a polygon) - should remain closed after simplification
        let result = tester
            .invoke_scalar_scalar("LINESTRING(0 0, 10 0, 10 10, 5 15, 0 10, 0 0)", 5.0)
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING(0 0, 10 0, 5 15, 0 0)");
    }
}
