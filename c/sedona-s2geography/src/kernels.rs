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

use arrow_schema::DataType;
use datafusion_common::{Result, ScalarValue};
use datafusion_expr::ColumnarValue;
use sedona_common::{sedona_internal_datafusion_err, sedona_internal_err};
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_extension::{extension::SedonaCScalarKernel, scalar_kernel::ImportedScalarKernel};
use sedona_functions::executor::WkbBytesExecutor;
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOGRAPHY},
    matchers::ArgMatcher,
};

use crate::{s2geog_check, s2geography_c_bindgen::*};

pub fn s2_scalar_kernels() -> Result<Vec<(String, ScalarKernelRef)>> {
    let mut ffi_scalar_kernels = Vec::<SedonaCScalarKernel>::new();
    ffi_scalar_kernels.resize_with(unsafe { S2GeogNumKernels() }, Default::default);

    unsafe {
        s2geog_check!(S2GeogInitKernels(
            ffi_scalar_kernels.as_mut_ptr() as _,
            size_of::<SedonaCScalarKernel>() * ffi_scalar_kernels.len(),
            S2GEOGRAPHY_KERNEL_FORMAT_SEDONA_UDF,
        ))
        .map_err(|e| sedona_internal_datafusion_err!("{e}"))?;
    }

    let mut kernels = ffi_scalar_kernels
        .into_iter()
        .map(|c_kernel| {
            let imported_kernel = ImportedScalarKernel::try_from(c_kernel)?;
            Ok((
                imported_kernel.function_name().unwrap().to_string(),
                Arc::new(imported_kernel) as ScalarKernelRef,
            ))
        })
        .collect::<Result<Vec<_>>>()?;

    // A few functions need explicit NULL type matching. This is mostly a DataFusion
    // thing (NULL can happen when binding parameters) so we handle it here and not
    // in s2geography.

    // Binary (geography, geography) -> Boolean
    let binary_bool_fns = [
        "st_contains",
        "st_disjoint",
        "st_equals",
        "st_intersects",
        "st_within",
    ];
    for fn_name in binary_bool_fns {
        kernels.push((
            fn_name.to_string(),
            Arc::new(NullKernelHelper::new(ArgMatcher::new(
                vec![ArgMatcher::is_geography(), ArgMatcher::is_null()],
                SedonaType::Arrow(DataType::Boolean),
            ))),
        ));
        kernels.push((
            fn_name.to_string(),
            Arc::new(NullKernelHelper::new(ArgMatcher::new(
                vec![ArgMatcher::is_null(), ArgMatcher::is_geography()],
                SedonaType::Arrow(DataType::Boolean),
            ))),
        ));
    }

    // Binary (geography, geography) -> Float64
    let binary_float_fns = ["st_distance", "st_linelocatepoint", "st_maxdistance"];
    for fn_name in binary_float_fns {
        kernels.push((
            fn_name.to_string(),
            Arc::new(NullKernelHelper::new(ArgMatcher::new(
                vec![ArgMatcher::is_geography(), ArgMatcher::is_null()],
                SedonaType::Arrow(DataType::Float64),
            ))),
        ));
        kernels.push((
            fn_name.to_string(),
            Arc::new(NullKernelHelper::new(ArgMatcher::new(
                vec![ArgMatcher::is_null(), ArgMatcher::is_geography()],
                SedonaType::Arrow(DataType::Float64),
            ))),
        ));
    }

    // Binary (geography, geography) -> geography
    let binary_geog_fns = [
        "st_closestpoint",
        "st_difference",
        "st_intersection",
        "st_longestline",
        "st_shortestline",
        "st_symdifference",
        "st_union",
    ];
    for fn_name in binary_geog_fns {
        kernels.push((
            fn_name.to_string(),
            Arc::new(NullKernelHelper::new(ArgMatcher::new(
                vec![ArgMatcher::is_geography(), ArgMatcher::is_null()],
                WKB_GEOGRAPHY,
            ))),
        ));
        kernels.push((
            fn_name.to_string(),
            Arc::new(NullKernelHelper::new(ArgMatcher::new(
                vec![ArgMatcher::is_null(), ArgMatcher::is_geography()],
                WKB_GEOGRAPHY,
            ))),
        ));
    }

    // Ternary (geography, geography, float64) -> Boolean
    let ternary_bool_fns = ["st_dwithin"];
    for fn_name in ternary_bool_fns {
        kernels.push((
            fn_name.to_string(),
            Arc::new(NullKernelHelper::new(ArgMatcher::new(
                vec![
                    ArgMatcher::is_geography(),
                    ArgMatcher::is_null(),
                    ArgMatcher::is_numeric(),
                ],
                SedonaType::Arrow(DataType::Boolean),
            ))),
        ));
        kernels.push((
            fn_name.to_string(),
            Arc::new(NullKernelHelper::new(ArgMatcher::new(
                vec![
                    ArgMatcher::is_null(),
                    ArgMatcher::is_geography(),
                    ArgMatcher::is_numeric(),
                ],
                SedonaType::Arrow(DataType::Boolean),
            ))),
        ));
    }

    // st_buffer(geography, NULL) -> geography
    kernels.push((
        "st_buffer".to_string(),
        Arc::new(NullKernelHelper::new(ArgMatcher::new(
            vec![ArgMatcher::is_geography(), ArgMatcher::is_null()],
            WKB_GEOGRAPHY,
        ))),
    ));

    // st_reduceprecision(geography, NULL) -> geography
    kernels.push((
        "st_reduceprecision".to_string(),
        Arc::new(NullKernelHelper::new(ArgMatcher::new(
            vec![ArgMatcher::is_geography(), ArgMatcher::is_null()],
            WKB_GEOGRAPHY,
        ))),
    ));

    // st_buffer(geography, NULL, NULL) -> geography
    kernels.push((
        "st_buffer".to_string(),
        Arc::new(NullKernelHelper::new(ArgMatcher::new(
            vec![
                ArgMatcher::is_geography(),
                ArgMatcher::is_null(),
                ArgMatcher::is_null(),
            ],
            WKB_GEOGRAPHY,
        ))),
    ));

    // s2_coveringcellids(NULL) -> List<Int64>
    kernels.push((
        "s2_coveringcellids".to_string(),
        Arc::new(NullKernelHelper::new(ArgMatcher::new(
            vec![ArgMatcher::is_null()],
            SedonaType::Arrow(DataType::List(Arc::new(arrow_schema::Field::new(
                "item",
                DataType::Int64,
                true,
            )))),
        ))),
    ));

    Ok(kernels)
}

#[derive(Debug)]
struct NullKernelHelper {
    matcher: ArgMatcher,
}

impl NullKernelHelper {
    fn new(matcher: ArgMatcher) -> Self {
        Self { matcher }
    }
}

impl SedonaScalarKernel for NullKernelHelper {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        self.matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        _arg_types: &[SedonaType],
        _args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        sedona_internal_err!("invoke_batch() should not be called")
    }

    fn invoke_batch_from_args(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
        return_type: &SedonaType,
        num_rows: usize,
        _config_options: Option<&datafusion_common::config::ConfigOptions>,
    ) -> Result<ColumnarValue> {
        let executor = WkbBytesExecutor::new(arg_types, args);
        let scalar_out = ScalarValue::try_new_null(return_type.storage_type())?;
        executor.finish(scalar_out.to_array_of_size(num_rows)?)
    }
}

#[cfg(test)]
mod test {
    use arrow_array::ArrayRef;
    use arrow_schema::DataType;
    use datafusion_common::ScalarValue;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{SedonaType, WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY};
    use sedona_testing::{
        create::{create_array, create_scalar},
        testers::ScalarUdfTester,
    };

    use super::*;

    #[test]
    fn test_s2_scalar_kernels() {
        let kernels = s2_scalar_kernels().unwrap();
        assert!(!kernels.is_empty());
    }

    fn s2_udf(name: &str) -> SedonaScalarUDF {
        let mut udf: Option<SedonaScalarUDF> = None;
        for (kernel_name, kernel) in s2_scalar_kernels().unwrap() {
            if name == kernel_name {
                match &mut udf {
                    Some(u) => u.add_kernels(kernel),
                    None => udf = Some(SedonaScalarUDF::from_impl(name, kernel)),
                }
            }
        }

        udf.unwrap_or_else(|| panic!("Can't find s2_scalar_udf named '{name}'"))
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
        let tester =
            ScalarUdfTester::new(udf.into(), vec![sedona_type.clone(), sedona_type.clone()]);
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

        let expected: ArrayRef =
            arrow_array::create_array!(Boolean, [Some(true), Some(false), None]);

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
        tester.assert_scalar_result_equals(result, "POINT EMPTY");
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
        tester.assert_scalar_result_equals(result, "POINT EMPTY");
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

    #[test]
    fn reduce_precision() {
        let udf = s2_udf("st_reduceprecision");
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![WKB_GEOGRAPHY, SedonaType::Arrow(DataType::Float64)],
        );
        tester.assert_return_type(WKB_GEOGRAPHY);
        let result = tester.invoke_scalar_scalar("POINT (0.1 0.2)", 1).unwrap();
        tester.assert_scalar_result_equals(result, "POINT (0 0)");
    }

    #[test]
    fn simplify() {
        let udf = s2_udf("st_simplify");
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![WKB_GEOGRAPHY, SedonaType::Arrow(DataType::Float64)],
        );
        tester.assert_return_type(WKB_GEOGRAPHY);
        let result = tester
            .invoke_scalar_scalar("LINESTRING (0 0, 0 5, 0 10)", 0.1)
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING (0 0, 0 10)");
    }

    #[test]
    fn buffer() {
        let udf = s2_udf("st_buffer");
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![WKB_GEOGRAPHY, SedonaType::Arrow(DataType::Float64)],
        );
        tester.assert_return_type(WKB_GEOGRAPHY);
        let result = tester
            .invoke_scalar_scalar("POLYGON ((0 0, 0.01 0, 0 0.01, 0 0))", -100000)
            .unwrap();
        tester.assert_scalar_result_equals(result, "POLYGON EMPTY");
    }

    #[test]
    fn null_type_matcher() {
        // Test that the NullKernelHelper matches arguments as intended and produces
        // either arrays of nulls or a null scalar of the correct type.
        let udf = s2_udf("st_intersects");

        let point_array = create_array(
            &[Some("POINT (0.25 0.25)"), Some("POINT (10 10)"), None],
            &WKB_GEOGRAPHY,
        );
        let expected: ArrayRef = arrow_array::create_array!(Boolean, [None, None, None]);

        // Test (geography, Null) signature
        let tester_geog_null = ScalarUdfTester::new(
            udf.clone().into(),
            vec![WKB_GEOGRAPHY, SedonaType::Arrow(DataType::Null)],
        );

        let result = tester_geog_null
            .invoke_array_scalar(point_array.clone(), ScalarValue::Null)
            .unwrap();
        assert_eq!(&result, &expected);

        let result = tester_geog_null
            .invoke_scalar_scalar("POINT (0 1)", ScalarValue::Null)
            .unwrap();
        assert_eq!(result, ScalarValue::Boolean(None));

        // Test (Null, geography) signature
        let tester_null_geog = ScalarUdfTester::new(
            udf.into(),
            vec![SedonaType::Arrow(DataType::Null), WKB_GEOGRAPHY],
        );

        let result = tester_null_geog
            .invoke_scalar_array(ScalarValue::Null, point_array)
            .unwrap();
        assert_eq!(&result, &expected);

        let result = tester_null_geog
            .invoke_scalar_scalar(ScalarValue::Null, "POINT (0 1)")
            .unwrap();
        assert_eq!(result, ScalarValue::Boolean(None));
    }
}
