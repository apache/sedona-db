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

use arrow_array::ArrayRef;
use arrow_schema::DataType;
use datafusion_common::ScalarValue;
use rstest::rstest;
use sedona_expr::scalar_udf::SedonaScalarUDF;
use sedona_s2geography::s2geography::s2_scalar_kernels;
use sedona_schema::datatypes::{SedonaType, WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY};
use sedona_testing::{
    create::{create_array, create_scalar},
    testers::ScalarUdfTester,
};

fn s2_udf(name: &str) -> SedonaScalarUDF {
    for (kernel_name, kernel) in s2_scalar_kernels().unwrap() {
        if name == kernel_name {
            return SedonaScalarUDF::from_impl(name, kernel);
        }
    }

    panic!("Can't find s2_scalar_udf named '{name}'")
}

#[rstest]
fn unary_scalar_kernel(#[values(WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY)] sedona_type: SedonaType) {
    let udf = s2_udf("st_length");
    let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type]);
    assert_eq!(
        tester.return_type().unwrap(),
        SedonaType::Arrow(DataType::Float64)
    );

    // Array -> Array
    let result = tester
        .invoke_wkb_array(vec![
            Some("POINT (0 1)"),
            Some("LINESTRING (0 0, 0 1)"),
            Some("POLYGON ((0 0, 1 0, 0 1, 0 0))"),
            None,
        ])
        .unwrap();

    let expected: ArrayRef = arrow_array::create_array!(
        Float64,
        [Some(0.0), Some(111195.10117748393), Some(0.0), None]
    );

    assert_eq!(&result, &expected);

    // Scalar -> Scalar
    let result = tester
        .invoke_wkb_scalar(Some("LINESTRING (0 0, 0 1)"))
        .unwrap();
    assert_eq!(result, ScalarValue::Float64(Some(111195.10117748393)));

    // Null scalar -> Null
    let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
    assert_eq!(result, ScalarValue::Float64(None));
}

#[rstest]
fn binary_scalar_kernel(#[values(WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY)] sedona_type: SedonaType) {
    let udf = s2_udf("st_intersects");
    let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type.clone(), sedona_type.clone()]);
    assert_eq!(
        tester.return_type().unwrap(),
        SedonaType::Arrow(DataType::Boolean)
    );

    let point_array = create_array(
        &[Some("POINT (0.25 0.25)"), Some("POINT (10 10)"), None],
        &sedona_type,
    );
    let polygon_scalar = create_scalar(Some("POLYGON ((0 0, 1 0, 0 1, 0 0))"), &sedona_type);
    let point_scalar = create_scalar(Some("POINT (0.25 0.25)"), &sedona_type);

    let expected: ArrayRef = arrow_array::create_array!(Boolean, [Some(true), Some(false), None]);

    // Array, Scalar -> Array
    let result = tester
        .invoke_array_scalar(point_array.clone(), polygon_scalar.clone())
        .unwrap();
    assert_eq!(&result, &expected);

    // Scalar, Array -> Array
    let result = tester
        .invoke_scalar_array(polygon_scalar.clone(), point_array.clone())
        .unwrap();
    assert_eq!(&result, &expected);

    // Scalar, Scalar -> Scalar
    let result = tester
        .invoke_scalar_scalar(polygon_scalar, point_scalar)
        .unwrap();
    assert_eq!(result, ScalarValue::Boolean(Some(true)));

    // Null scalars -> Null
    let result = tester
        .invoke_scalar_scalar(ScalarValue::Null, ScalarValue::Null)
        .unwrap();
    assert_eq!(result, ScalarValue::Boolean(None));
}

#[test]
fn area() {
    let udf = s2_udf("st_area");
    let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOGRAPHY]);

    tester.assert_return_type(DataType::Float64);
    let result = tester
        .invoke_scalar("POLYGON ((0 0, 0 1, 1 0, 0 0))")
        .unwrap();
    tester.assert_scalar_result_equals(result, 6182489130.907195);
}

#[test]
fn centroid() {
    let udf = s2_udf("st_centroid");
    let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOGRAPHY]);
    tester.assert_return_type(WKB_GEOGRAPHY);
    let result = tester.invoke_scalar("LINESTRING (0 0, 0 1)").unwrap();
    tester.assert_scalar_result_equals(result, "POINT (0 0.5)");
}

#[test]
fn closest_point() {
    let udf = s2_udf("st_closestpoint");
    let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOGRAPHY, WKB_GEOGRAPHY]);
    tester.assert_return_type(WKB_GEOGRAPHY);
    let result = tester
        .invoke_scalar_scalar("LINESTRING (0 0, 0 1)", "POINT (-1 -1)")
        .unwrap();
    tester.assert_scalar_result_equals(result, "POINT (0 0)");
}

#[test]
fn contains() {
    let udf = s2_udf("st_contains");
    let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOGRAPHY, WKB_GEOGRAPHY]);
    tester.assert_return_type(DataType::Boolean);
    let result = tester
        .invoke_scalar_scalar("POLYGON ((0 0, 0 1, 1 0, 0 0))", "POINT (0.25 0.25)")
        .unwrap();
    tester.assert_scalar_result_equals(result, true);
}

#[test]
fn difference() {
    let udf = s2_udf("st_difference");
    let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOGRAPHY, WKB_GEOGRAPHY]);
    tester.assert_return_type(WKB_GEOGRAPHY);
    let result = tester
        .invoke_scalar_scalar("POINT (0 0)", "POINT (0 0)")
        .unwrap();
    tester.assert_scalar_result_equals(result, "GEOMETRYCOLLECTION EMPTY");
}

#[test]
fn distance() {
    let udf = s2_udf("st_distance");
    let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOGRAPHY, WKB_GEOGRAPHY]);
    tester.assert_return_type(DataType::Float64);
    let result = tester
        .invoke_scalar_scalar("POINT (0 0)", "POINT (0 0)")
        .unwrap();
    tester.assert_scalar_result_equals(result, 0.0);
}

#[test]
fn equals() {
    let udf = s2_udf("st_equals");
    let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOGRAPHY, WKB_GEOGRAPHY]);
    tester.assert_return_type(DataType::Boolean);
    let result1 = tester
        .invoke_scalar_scalar("POINT (0 0)", "POINT (0 0)")
        .unwrap();
    tester.assert_scalar_result_equals(result1, true);
    let result2 = tester
        .invoke_scalar_scalar("POINT (0 0)", "POINT (0 1)")
        .unwrap();
    tester.assert_scalar_result_equals(result2, false);
}

#[test]
fn intersection() {
    let udf = s2_udf("st_intersection");
    let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOGRAPHY, WKB_GEOGRAPHY]);
    tester.assert_return_type(WKB_GEOGRAPHY);
    let result = tester
        .invoke_scalar_scalar("POINT (0 0)", "POINT (0 0)")
        .unwrap();
    tester.assert_scalar_result_equals(result, "POINT (0 0)");
}

#[test]
fn intersects() {
    let udf = s2_udf("st_intersects");
    let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOGRAPHY, WKB_GEOGRAPHY]);
    tester.assert_return_type(DataType::Boolean);
    let result1 = tester
        .invoke_scalar_scalar("POLYGON ((0 0, 0 1, 1 0, 0 0))", "POINT (0.25 0.25)")
        .unwrap();
    tester.assert_scalar_result_equals(result1, true);
    let result2 = tester
        .invoke_scalar_scalar("POLYGON ((0 0, 0 1, 1 0, 0 0))", "POINT (-1 -1)")
        .unwrap();
    tester.assert_scalar_result_equals(result2, false);
}

#[test]
fn length() {
    let udf = s2_udf("st_length");
    let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOGRAPHY]);
    tester.assert_return_type(DataType::Float64);
    let result = tester.invoke_scalar("LINESTRING (0 0, 0 1)").unwrap();
    tester.assert_scalar_result_equals(result, 111195.10117748393);
}

#[test]
fn line_interpolate_point() {
    let udf = s2_udf("st_lineinterpolatepoint");
    let tester = ScalarUdfTester::new(
        udf.into(),
        vec![WKB_GEOGRAPHY, SedonaType::Arrow(DataType::Float64)],
    );
    tester.assert_return_type(WKB_GEOGRAPHY);
    let result = tester
        .invoke_scalar_scalar("LINESTRING (0 0, 0 1)", 0.5)
        .unwrap();
    tester.assert_scalar_result_equals(result, "POINT (0 0.5)");
}

#[test]
fn line_locate_point() {
    let udf = s2_udf("st_linelocatepoint");
    let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOGRAPHY, WKB_GEOGRAPHY]);
    tester.assert_return_type(DataType::Float64);
    let result = tester
        .invoke_scalar_scalar("LINESTRING (0 0, 0 1)", "POINT (0 0.5)")
        .unwrap();
    tester.assert_scalar_result_equals(result, 0.5);
}

#[test]
fn max_distance() {
    let udf = s2_udf("st_maxdistance");
    let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOGRAPHY, WKB_GEOGRAPHY]);
    tester.assert_return_type(DataType::Float64);
    let result = tester
        .invoke_scalar_scalar("POINT (0 0)", "LINESTRING (0 0, 0 1)")
        .unwrap();
    tester.assert_scalar_result_equals(result, 111195.10117748393);
}

#[test]
fn perimeter() {
    let udf = s2_udf("st_perimeter");
    let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOGRAPHY]);
    let result = tester
        .invoke_scalar("POLYGON ((0 0, 0 1, 1 0, 0 0))")
        .unwrap();
    tester.assert_scalar_result_equals(result, 379639.8304474758);
}

#[test]
fn shortest_line() {
    let udf = s2_udf("st_shortestline");
    let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOGRAPHY, WKB_GEOGRAPHY]);
    tester.assert_return_type(WKB_GEOGRAPHY);
    let result = tester
        .invoke_scalar_scalar("LINESTRING (0 0, 0 1)", "POINT (0 -1)")
        .unwrap();
    tester.assert_scalar_result_equals(result, "LINESTRING (0 0, 0 -1)");
}

#[test]
fn sym_difference() {
    let udf = s2_udf("st_symdifference");
    let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOGRAPHY, WKB_GEOGRAPHY]);
    tester.assert_return_type(WKB_GEOGRAPHY);
    let result = tester
        .invoke_scalar_scalar("POINT (0 0)", "POINT (0 0)")
        .unwrap();
    tester.assert_scalar_result_equals(result, "GEOMETRYCOLLECTION EMPTY");
}

#[test]
fn union() {
    let udf = s2_udf("st_union");
    let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOGRAPHY, WKB_GEOGRAPHY]);
    tester.assert_return_type(WKB_GEOGRAPHY);
    let result = tester
        .invoke_scalar_scalar("POINT (0 0)", "POINT (0 0)")
        .unwrap();
    tester.assert_scalar_result_equals(result, "POINT (0 0)");
}
