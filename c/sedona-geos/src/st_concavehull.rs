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
use datafusion_common::cast::{as_boolean_array, as_float64_array};
use datafusion_common::error::Result;
use datafusion_common::{DataFusionError, ScalarValue};
use datafusion_expr::ColumnarValue;
use geos::Geom;
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_geometry::wkb_factory::WKB_MIN_PROBABLE_BYTES;
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

use crate::executor::GeosExecutor;

/// ST_ConcaveHull() implementation using the geos crate
pub fn st_concave_hull_allow_holes_impl() -> ScalarKernelRef {
    Arc::new(STConcaveHullAllowHoles {})
}

#[derive(Debug)]
struct STConcaveHullAllowHoles {}

impl SedonaScalarKernel for STConcaveHullAllowHoles {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_geometry(),
                ArgMatcher::is_numeric(),
                ArgMatcher::is_boolean(),
            ],
            WKB_GEOMETRY,
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        invoke_batch_impl(arg_types, args)
    }
}

pub fn st_concave_hull_impl() -> ScalarKernelRef {
    Arc::new(STConcaveHull {})
}

#[derive(Debug)]
struct STConcaveHull {}

impl SedonaScalarKernel for STConcaveHull {
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
        invoke_batch_impl(arg_types, args)
    }
}

fn invoke_batch_impl(arg_types: &[SedonaType], args: &[ColumnarValue]) -> Result<ColumnarValue> {
    let executor = GeosExecutor::new(arg_types, args);
    let mut builder = BinaryBuilder::with_capacity(
        executor.num_iterations(),
        WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
    );

    let pct_convex_val = args[1]
        .cast_to(&DataType::Float64, None)?
        .to_array(executor.num_iterations())?;
    let pct_convex_array = as_float64_array(&pct_convex_val)?;
    let mut pct_convex_iter = pct_convex_array.iter();

    let allow_holes_val = args
        .get(2)
        .unwrap_or(&ColumnarValue::Scalar(ScalarValue::Boolean(Some(false))))
        .cast_to(&DataType::Boolean, None)?
        .to_array(executor.num_iterations())?;
    let allow_holes_array = as_boolean_array(&allow_holes_val)?;
    let mut allow_holes_iter = allow_holes_array.iter();

    executor.execute_wkb_void(|maybe_wkb| {
        match (
            maybe_wkb,
            pct_convex_iter.next().unwrap(),
            allow_holes_iter.next().unwrap(),
        ) {
            (Some(wkb), Some(pct_convex), Some(allow_holes)) => {
                invoke_scalar(&wkb, pct_convex, allow_holes, &mut builder)?;
                builder.append_value([]);
            }
            _ => builder.append_null(),
        }
        Ok(())
    })?;

    executor.finish(Arc::new(builder.finish()))
}

fn invoke_scalar<W: std::io::Write>(
    geos_geom: &geos::Geometry,
    pct_convex: f64,
    allow_holes: bool,
    writer: &mut W,
) -> Result<()> {
    let geometry = geos_geom
        .concave_hull(pct_convex, allow_holes)
        .map_err(|e| {
            DataFusionError::Execution(format!("Failed to calculate concave hull: {e}"))
        })?;

    let geo_wkb = geometry
        .to_wkb()
        .map_err(|e| DataFusionError::Execution(format!("Failed to convert to WKB: {e}")))?;

    writer.write_all(&geo_wkb)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use arrow_array::Float64Array;
    use arrow_schema::DataType;
    use datafusion_common::ScalarValue;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::{
        compare::{assert_array_equal, assert_scalar_equal_wkb_geometry},
        create::create_array,
        testers::ScalarUdfTester,
    };

    use super::*;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_concavehull", st_concave_hull_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Float64)],
        );
        assert_eq!(tester.return_type().unwrap(), WKB_GEOMETRY);

        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());

        let result = tester
            .invoke_scalar_scalar("POLYGON ((70 80, 50 60, 100 150, 160 170, 70 80))", 0.2)
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            "POLYGON ((70 80, 50 60, 100 150, 160 170, 70 80))",
        );

        let input_wkt_array = vec![
            None,
            Some("POINT EMPTY"),
            Some("POINT (2.5 3.1)"),
            Some("LINESTRING EMPTY"),
            Some("LINESTRING (50 50, 150 150, 150 50)"),
            Some("LINESTRING (100 150, 50 60, 70 80, 160 170)"),
        ];

        let tolerance_array: Arc<Float64Array> = Arc::new(Float64Array::from(vec![
            Some(0.1),
            Some(0.1),
            Some(0.1),
            Some(0.1),
            Some(0.1),
            Some(0.3),
        ]));

        let expected_array = create_array(
            &[
                None,
                Some("POLYGON EMPTY"),
                Some("POINT (2.5 3.1)"),
                Some("POLYGON EMPTY"),
                Some("POLYGON ((50 50, 150 150, 150 50, 50 50))"),
                Some("POLYGON ((70 80, 50 60, 100 150, 160 170, 70 80))"),
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

        let input_wkt_array = vec![
            Some("MULTIPOINT EMPTY"),
            Some("MULTIPOINT ((100 150), (160 170))"),
            Some("MULTIPOINT ((0 0), (10 0), (0 10), (10 10), (5 5))"),
            Some("MULTILINESTRING ((50 150, 50 200), (50 50, 50 100))"),
            Some("MULTIPOLYGON EMPTY"),
            Some("MULTIPOLYGON (((2 2, 2 5, 5 5, 5 2, 2 2)), ((6 3, 8 3, 8 1, 6 1, 6 3)))"),
        ];

        let tolerance_array: Arc<Float64Array> = Arc::new(Float64Array::from(vec![
            Some(0.1),
            Some(0.2),
            Some(0.1),
            Some(0.2),
            Some(0.3),
            Some(0.1),
        ]));

        let expected_array = create_array(
            &[
                Some("POLYGON EMPTY"),
                Some("LINESTRING(100 150, 160 170)"),
                Some("POLYGON ((5 5, 0 10, 10 10, 10 0, 0 0, 5 5))"),
                Some("LINESTRING (50 50, 50 200)"),
                Some("POLYGON EMPTY"),
                Some("POLYGON ((5 2, 2 2, 2 5, 5 5, 6 3, 8 3, 8 1, 6 1, 5 2))"),
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

        let input_wkt_array = vec![
            Some("MULTIPOLYGON(((26 125, 26 200, 126 200, 126 125, 26 125 ),( 51 150, 101 150, 76 175, 51 150 )),(( 151 100, 151 200, 176 175, 151 100 )))"),
            Some("GEOMETRYCOLLECTION EMPTY"),
            Some("GEOMETRYCOLLECTION (MULTIPOINT((1 1), (3 3)), POINT(5 6), LINESTRING(4 5, 5 6))"),
            Some("GEOMETRYCOLLECTION(LINESTRING(1 1,2 2),GEOMETRYCOLLECTION(POLYGON((3 3,4 4,5 5,3 3)),GEOMETRYCOLLECTION(LINESTRING(6 6,7 7),POLYGON((8 8,9 9,10 10,8 8)))))"),
        ];

        let tolerance_array: Arc<Float64Array> = Arc::new(Float64Array::from(vec![
            Some(0.1),
            Some(0.3),
            Some(0.1),
            Some(0.1),
        ]));

        let expected_array = create_array(
            &[
                Some("POLYGON((51 150, 26 125, 26 200, 76 175, 126 200, 151 200, 176 175, 151 100, 126 125, 101 150, 51 150))"),
                Some("POLYGON EMPTY"),
                Some("POLYGON ((3 3, 1 1, 4 5, 5 6, 3 3))"),
                Some("LINESTRING(1 1,10 10)"),
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
    fn udf_allow_holes(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf =
            SedonaScalarUDF::from_kernel("st_concavehull", st_concave_hull_allow_holes_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![
                sedona_type.clone(),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Boolean),
            ],
        );
        assert_eq!(tester.return_type().unwrap(), WKB_GEOMETRY);

        let result = tester
            .invoke_scalar_scalar_scalar("POINT EMPTY", 0.1, true)
            .unwrap();
        tester.assert_scalar_result_equals(result, "POLYGON EMPTY");

        let result = tester
            .invoke_scalar_scalar_scalar("POINT (2.5 3.1)", 0.1, true)
            .unwrap();
        tester.assert_scalar_result_equals(result, "POINT (2.5 3.1)");

        let result = tester
            .invoke_scalar_scalar_scalar("POINT (2.5 3.1)", 0.1, true)
            .unwrap();
        tester.assert_scalar_result_equals(result, "POINT (2.5 3.1)");

        let result = tester
            .invoke_scalar_scalar_scalar("LINESTRING EMPTY", 0.2, true)
            .unwrap();
        tester.assert_scalar_result_equals(result, "POLYGON EMPTY");

        let result = tester
            .invoke_scalar_scalar_scalar("LINESTRING (100 150, 50 60, 70 80, 160 170)", 0.2, true)
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            "POLYGON ((50 60, 100 150, 160 170, 70 80, 50 60))",
        );

        let result = tester
            .invoke_scalar_scalar_scalar("LINESTRING (100 150, 50 60, 70 80, 160 170)", 0.2, false)
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            "POLYGON ((70 80, 50 60, 100 150, 160 170, 70 80))",
        );

        let result = tester
            .invoke_scalar_scalar_scalar(
                "POLYGON ((70 80, 50 60, 100 150, 160 170, 70 80))",
                0.2,
                false,
            )
            .unwrap();
        assert_scalar_equal_wkb_geometry(
            &result,
            Some("POLYGON((70 80,50 60,100 150,160 170,70 80))"),
        );

        let result = tester
            .invoke_scalar_scalar_scalar(
                "POLYGON ((70 80, 50 60, 100 150, 160 170, 70 80))",
                0.2,
                true,
            )
            .unwrap();
        assert_scalar_equal_wkb_geometry(
            &result,
            Some("POLYGON((50 60,100 150,160 170,70 80,50 60))"),
        );

        let result = tester
            .invoke_scalar_scalar_scalar("MULTIPOINT EMPTY", 0.2, false)
            .unwrap();
        tester.assert_scalar_result_equals(result, "POLYGON EMPTY");

        let result = tester
            .invoke_scalar_scalar_scalar(
                "MULTIPOINT ((10 40), (40 30), (20 20), (30 10))",
                0.1,
                true,
            )
            .unwrap();
        tester.assert_scalar_result_equals(result, "POLYGON ((40 30, 30 10, 20 20, 10 40, 40 30))");

        let result = tester
            .invoke_scalar_scalar_scalar(
                "MULTIPOINT ((10 40), (40 30), (20 20), (30 10))",
                0.1,
                false,
            )
            .unwrap();
        tester.assert_scalar_result_equals(result, "POLYGON ((20 20, 10 40, 40 30, 30 10, 20 20))");

        let result = tester
            .invoke_scalar_scalar_scalar("MULTILINESTRING EMPTY", 0.1, false)
            .unwrap();
        tester.assert_scalar_result_equals(result, "POLYGON EMPTY");

        let result = tester
            .invoke_scalar_scalar_scalar(
                "MULTILINESTRING ((10 10, 20 20, 10 40), (40 40, 30 30, 40 20, 30 10))",
                0.1,
                true,
            )
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            "POLYGON ((30 30, 40 40, 40 20, 30 10, 10 10, 20 20, 10 40, 30 30))",
        );

        let result = tester
            .invoke_scalar_scalar_scalar(
                "MULTILINESTRING ((10 10, 20 20, 10 40), (40 40, 30 30, 40 20, 30 10))",
                0.1,
                false,
            )
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            "POLYGON ((20 20, 10 40, 30 30, 40 40, 40 20, 30 10, 10 10, 20 20))",
        );

        let result = tester
            .invoke_scalar_scalar_scalar("MULTIPOLYGON EMPTY", 0.1, false)
            .unwrap();
        tester.assert_scalar_result_equals(result, "POLYGON EMPTY");

        let result = tester
            .invoke_scalar_scalar_scalar(
                "MULTIPOLYGON (((2 2, 2 5, 5 5, 5 2, 2 2)), ((6 3, 8 3, 8 1, 6 1, 6 3)))",
                0.1,
                true,
            )
            .unwrap();
        tester
            .assert_scalar_result_equals(result, "POLYGON((2 2,2 5,5 5,6 3,8 3,8 1,6 1,5 2,2 2))");

        let result = tester
            .invoke_scalar_scalar_scalar(
                "MULTIPOLYGON (((2 2, 2 5, 5 5, 5 2, 2 2)), ((6 3, 8 3, 8 1, 6 1, 6 3)))",
                0.1,
                false,
            )
            .unwrap();
        tester
            .assert_scalar_result_equals(result, "POLYGON((5 2,2 2,2 5,5 5,6 3,8 3,8 1,6 1,5 2))");

        let result = tester
            .invoke_scalar_scalar_scalar("GEOMETRYCOLLECTION EMPTY", 0.1, true)
            .unwrap();
        tester.assert_scalar_result_equals(result, "POLYGON EMPTY");

        let result = tester
            .invoke_scalar_scalar_scalar(
                "GEOMETRYCOLLECTION (MULTIPOINT((1 1), (3 3)), POINT(5 6), LINESTRING(4 5, 5 6))",
                0.1,
                true,
            )
            .unwrap();
        tester.assert_scalar_result_equals(result, "POLYGON ((1 1, 4 5, 5 6, 3 3, 1 1))");

        let result = tester
            .invoke_scalar_scalar_scalar(
                "GEOMETRYCOLLECTION (MULTIPOINT((1 1), (3 3)), POINT(5 6), LINESTRING(4 5, 5 6))",
                0.1,
                false,
            )
            .unwrap();
        tester.assert_scalar_result_equals(result, "POLYGON ((3 3, 1 1, 4 5, 5 6, 3 3))");
    }
}
