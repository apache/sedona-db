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

use arrow_array::builder::Float64Builder;
use arrow_schema::DataType;
use datafusion_common::error::Result;
use datafusion_expr::ColumnarValue;
use geo_types::Point;
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{ScalarKernelRef, SedonaScalarKernel},
};
use sedona_functions::executor::{PointOrWkb, PointXYExecutor};
use sedona_geo_generic_alg::{HasDimensions, line_measures::DistanceExt};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

/// ST_Distance() implementation using [DistanceExt]
pub fn st_distance_impl() -> Vec<ScalarKernelRef> {
    ItemCrsKernel::wrap_impl(STDistance {})
}

#[derive(Debug)]
struct STDistance {}

impl SedonaScalarKernel for STDistance {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry(), ArgMatcher::is_geometry()],
            SedonaType::Arrow(DataType::Float64),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        // The PointXY executor decodes each operand as a `PointOrWkb`: a finite
        // Point yields its `(x, y)` straight from the WKB offset (no parse),
        // anything else is parsed. Scalars are decoded once and reused per row.
        let executor = PointXYExecutor::new(arg_types, args);
        let mut builder = Float64Builder::with_capacity(executor.num_iterations());

        executor.execute_wkb_wkb_void(|maybe0, maybe1| {
            match (maybe0, maybe1) {
                (Some(a), Some(b)) => match point_or_wkb_distance(a, b) {
                    Some(distance) => builder.append_value(distance),
                    None => builder.append_null(),
                },
                _ => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

/// Euclidean distance between two operands. Point-to-Point is computed directly
/// from the coordinates; any other combination defers to the generic
/// [DistanceExt] (a Point operand becomes a [`geo_types::Point`] so the mixed
/// Point/geometry case never re-parses the Point). Shared with `ST_DWithin`.
pub(crate) fn point_or_wkb_distance(a: &PointOrWkb, b: &PointOrWkb) -> Option<f64> {
    match (a, b) {
        (PointOrWkb::Point(ax, ay), PointOrWkb::Point(bx, by)) => {
            let dx = ax - bx;
            let dy = ay - by;
            Some((dx * dx + dy * dy).sqrt())
        }
        (PointOrWkb::Point(x, y), PointOrWkb::Other(wkb)) => {
            (!wkb.is_empty()).then(|| Point::new(*x, *y).distance_ext(wkb))
        }
        (PointOrWkb::Other(wkb), PointOrWkb::Point(x, y)) => {
            (!wkb.is_empty()).then(|| wkb.distance_ext(&Point::new(*x, *y)))
        }
        (PointOrWkb::Other(wkb0), PointOrWkb::Other(wkb1)) => {
            (!wkb0.is_empty() && !wkb1.is_empty()).then(|| wkb0.distance_ext(wkb1))
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{ArrayRef, Float64Array};
    use datafusion_common::scalar::ScalarValue;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS, WKB_VIEW_GEOMETRY};
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::create::{create_array, create_scalar};
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn udf(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY, WKB_GEOMETRY_ITEM_CRS.clone())]
        left_sedona_type: SedonaType,
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY, WKB_GEOMETRY_ITEM_CRS.clone())]
        right_sedona_type: SedonaType,
    ) {
        let udf = SedonaScalarUDF::from_impl("st_distance", st_distance_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![left_sedona_type.clone(), right_sedona_type.clone()],
        );

        assert_eq!(
            tester.return_type().unwrap(),
            SedonaType::Arrow(DataType::Float64)
        );

        // Test distance between two points (3-4-5 triangle)
        let point_0_0 = create_scalar(Some("POINT (0 0)"), &left_sedona_type);
        let point_3_4 = create_scalar(Some("POINT (3 4)"), &right_sedona_type);

        let result = tester
            .invoke_scalar_scalar(point_0_0.clone(), point_3_4.clone())
            .unwrap();
        if let ScalarValue::Float64(Some(distance)) = result {
            assert!((distance - 5.0).abs() < 1e-10);
        } else {
            panic!("Expected Float64 result");
        }

        // Test with null values
        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, point_3_4.clone())
            .unwrap();
        assert!(result.is_null());
        let result = tester
            .invoke_scalar_scalar(point_0_0.clone(), ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());

        // Distance involving an empty geometry is undefined.
        let result = tester
            .invoke_scalar_scalar(
                create_scalar(Some("POINT EMPTY"), &left_sedona_type),
                point_3_4.clone(),
            )
            .unwrap();
        assert!(result.is_null());
        let result = tester
            .invoke_scalar_scalar(
                point_0_0,
                create_scalar(Some("POLYGON EMPTY"), &right_sedona_type),
            )
            .unwrap();
        assert!(result.is_null());
    }

    /// Exercises the array/scalar fast-path routing: a scalar operand is parsed
    /// once and reused per row, the point-to-point fast path matches the general
    /// distance, and mixed Point/non-Point inputs fall back correctly. Covered
    /// across storage encodings, including item-CRS (which arrives unwrapped as
    /// plain WKB, so the parse-once scalar handling still applies).
    #[rstest]
    fn array_scalar_routing(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY, WKB_GEOMETRY_ITEM_CRS.clone())] left: SedonaType,
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY, WKB_GEOMETRY_ITEM_CRS.clone())] right: SedonaType,
    ) {
        let udf = SedonaScalarUDF::from_impl("st_distance", st_distance_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![left.clone(), right.clone()]);

        // Array of Points vs a scalar Point: the point-to-point fast path.
        let result = tester
            .invoke_wkb_array_scalar(
                vec![Some("POINT (3 4)"), None, Some("POINT (0 0)")],
                create_scalar(Some("POINT (0 0)"), &right),
            )
            .unwrap();
        let expected: ArrayRef = Arc::new(Float64Array::from(vec![Some(5.0), None, Some(0.0)]));
        assert_array_equal(&result, &expected);

        // Array of Polygons vs a scalar Point: the scalar is parsed once and
        // reused across rows; each row falls back to the general distance. The
        // scalar Point sits on a polygon vertex, so every distance is 0.
        let result = tester
            .invoke_wkb_array_scalar(
                vec![
                    Some("POLYGON ((10 10, 10 20, 20 20, 20 10, 10 10))"),
                    Some("POLYGON ((10 10, 10 20, 20 20, 20 10, 10 10))"),
                ],
                create_scalar(Some("POINT (10 10)"), &right),
            )
            .unwrap();
        let expected: ArrayRef = Arc::new(Float64Array::from(vec![Some(0.0), Some(0.0)]));
        assert_array_equal(&result, &expected);

        // Scalar Point vs array of Points (mirror orientation).
        let result = tester
            .invoke_scalar_array(
                create_scalar(Some("POINT (0 0)"), &left),
                create_array(&[Some("POINT (3 4)"), Some("POINT (0 0)")], &right),
            )
            .unwrap();
        let expected: ArrayRef = Arc::new(Float64Array::from(vec![Some(5.0), Some(0.0)]));
        assert_array_equal(&result, &expected);

        // Array vs array with mixed Point / Polygon rows.
        let arg0 = create_array(
            &[
                Some("POINT (0 0)"),
                Some("POLYGON ((10 10, 10 20, 20 20, 20 10, 10 10))"),
            ],
            &left,
        );
        let arg1 = create_array(&[Some("POINT (3 4)"), Some("POINT (10 10)")], &right);
        let expected: ArrayRef = Arc::new(Float64Array::from(vec![Some(5.0), Some(0.0)]));
        assert_array_equal(&tester.invoke_array_array(arg0, arg1).unwrap(), &expected);
    }

    #[rstest]
    fn crossing_linestrings_have_zero_distance(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY, WKB_GEOMETRY_ITEM_CRS.clone())] left: SedonaType,
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY, WKB_GEOMETRY_ITEM_CRS.clone())] right: SedonaType,
    ) {
        let udf = SedonaScalarUDF::from_impl("st_distance", st_distance_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![left.clone(), right.clone()]);
        let result = tester
            .invoke_scalar_scalar(
                create_scalar(Some("LINESTRING (0 0, 2 2)"), &left),
                create_scalar(Some("LINESTRING (0 2, 2 0)"), &right),
            )
            .unwrap();

        assert_eq!(result, ScalarValue::Float64(Some(0.0)));
    }

    #[test]
    fn nonempty_collection_with_empty_point() {
        let point = create_scalar(Some("POINT (0 0)"), &WKB_GEOMETRY);
        let mixed = create_scalar(
            Some("GEOMETRYCOLLECTION (POINT EMPTY, POINT (10 0))"),
            &WKB_GEOMETRY,
        );
        let distance = ScalarUdfTester::new(
            SedonaScalarUDF::from_impl("st_distance", st_distance_impl()).into(),
            vec![WKB_GEOMETRY, WKB_GEOMETRY],
        )
        .invoke_scalar_scalar(point.clone(), mixed.clone())
        .unwrap();
        let dwithin = ScalarUdfTester::new(
            SedonaScalarUDF::from_impl("st_dwithin", crate::st_dwithin::st_dwithin_impl()).into(),
            vec![
                WKB_GEOMETRY,
                WKB_GEOMETRY,
                SedonaType::Arrow(DataType::Float64),
            ],
        )
        .invoke_scalar_scalar_scalar(point, mixed, ScalarValue::Float64(Some(1.0)))
        .unwrap();
        assert_eq!(
            (distance, dwithin),
            (
                ScalarValue::Float64(Some(10.0)),
                ScalarValue::Boolean(Some(false))
            )
        );
    }

    #[rstest]
    #[case("POINT (0 0)", "GEOMETRYCOLLECTION (POINT EMPTY, POINT (10 0))", 10.0)]
    #[case("GEOMETRYCOLLECTION (POINT EMPTY, POINT (10 0))", "POINT (0 0)", 10.0)]
    #[case(
        "GEOMETRYCOLLECTION (POINT EMPTY, POINT (0 3))",
        "GEOMETRYCOLLECTION (LINESTRING EMPTY, POINT (4 0))",
        5.0
    )]
    #[case(
        "LINESTRING (0 0, 0 1)",
        "GEOMETRYCOLLECTION (POLYGON EMPTY, POINT (3 1))",
        3.0
    )]
    #[case(
        "POINT (0 0)",
        "GEOMETRYCOLLECTION (GEOMETRYCOLLECTION (POINT EMPTY), POINT (10 0))",
        10.0
    )]
    fn collection_with_empty_members(#[case] lhs: &str, #[case] rhs: &str, #[case] expected: f64) {
        let tester = ScalarUdfTester::new(
            SedonaScalarUDF::from_impl("st_distance", st_distance_impl()).into(),
            vec![WKB_GEOMETRY, WKB_GEOMETRY],
        );
        let result = tester
            .invoke_scalar_scalar(
                create_scalar(Some(lhs), &WKB_GEOMETRY),
                create_scalar(Some(rhs), &WKB_GEOMETRY),
            )
            .unwrap();
        assert_eq!(result, ScalarValue::Float64(Some(expected)));
    }

    #[test]
    fn collection_member_at_the_largest_distance() {
        // f64::MAX is a real distance, not a marker for a collection without members
        let point = create_scalar(Some("POINT (0 0)"), &WKB_GEOMETRY);
        let far = create_scalar(
            Some("GEOMETRYCOLLECTION (POINT EMPTY, POINT (1.7976931348623157e308 0))"),
            &WKB_GEOMETRY,
        );
        let distance = ScalarUdfTester::new(
            SedonaScalarUDF::from_impl("st_distance", st_distance_impl()).into(),
            vec![WKB_GEOMETRY, WKB_GEOMETRY],
        )
        .invoke_scalar_scalar(point.clone(), far.clone())
        .unwrap();
        let dwithin = ScalarUdfTester::new(
            SedonaScalarUDF::from_impl("st_dwithin", crate::st_dwithin::st_dwithin_impl()).into(),
            vec![
                WKB_GEOMETRY,
                WKB_GEOMETRY,
                SedonaType::Arrow(DataType::Float64),
            ],
        )
        .invoke_scalar_scalar_scalar(point, far, ScalarValue::Float64(Some(1.0)))
        .unwrap();
        assert_eq!(
            (distance, dwithin),
            (
                ScalarValue::Float64(Some(f64::MAX)),
                ScalarValue::Boolean(Some(false))
            )
        );
    }
}
