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

use crate::geos::{
    BinaryPredicate, Contains, CoveredBy, Covers, Disjoint, Equals, GeosPredicate, Intersects,
    Touches, Within,
};
use arrow_array::builder::BooleanBuilder;
use arrow_schema::DataType;
use datafusion_common::{error::Result, DataFusionError};
use datafusion_expr::ColumnarValue;
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::executor::GeosExecutor;

pub fn st_contains_impl() -> ScalarKernelRef {
    Arc::new(GeosPredicate::<Contains>::default())
}

pub fn st_covered_by_impl() -> ScalarKernelRef {
    Arc::new(GeosPredicate::<CoveredBy>::default())
}

pub fn st_covers_impl() -> ScalarKernelRef {
    Arc::new(GeosPredicate::<Covers>::default())
}

pub fn st_disjoint_impl() -> ScalarKernelRef {
    Arc::new(GeosPredicate::<Disjoint>::default())
}

pub fn st_equals_impl() -> ScalarKernelRef {
    Arc::new(GeosPredicate::<Equals>::default())
}

pub fn st_intersects_impl() -> ScalarKernelRef {
    Arc::new(GeosPredicate::<Intersects>::default())
}

pub fn st_touches_impl() -> ScalarKernelRef {
    Arc::new(GeosPredicate::<Touches>::default())
}

pub fn st_within_impl() -> ScalarKernelRef {
    Arc::new(GeosPredicate::<Within>::default())
}

impl<Op: BinaryPredicate> SedonaScalarKernel for GeosPredicate<Op> {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher: ArgMatcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry(), ArgMatcher::is_geometry()],
            SedonaType::Arrow(DataType::Boolean),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = GeosExecutor::new(arg_types, args);
        let mut builder = BooleanBuilder::with_capacity(executor.num_iterations());
        executor.execute_wkb_wkb_void(|lhs, rhs| {
            match (lhs, rhs) {
                (Some(lhs), Some(rhs)) => {
                    builder.append_value(Op::evaluate(lhs, rhs).map_err(|e| {
                        DataFusionError::Execution(format!("Failed to evaluate predicate: {e}"))
                    })?);
                }
                _ => builder.append_null(),
            };
            Ok(())
        })?;
        executor.finish(Arc::new(builder.finish()))
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{create_array as arrow_array, ArrayRef};
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::create::create_array;
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn contains_udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        use datafusion_common::ScalarValue;

        let udf = SedonaScalarUDF::from_kernel("st_contains", st_contains_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type.clone(), sedona_type]);
        tester.assert_return_type(DataType::Boolean);

        let result = tester
            .invoke_scalar_scalar("POINT (0 0)", "LINESTRING (2 0, 0 2)")
            .unwrap();
        tester.assert_scalar_result_equals(result, false);

        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());

        let arg1 = create_array(
            &[
                Some("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
                Some("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
                None,
            ],
            &WKB_GEOMETRY,
        );
        let arg2 = create_array(
            &[
                Some("POINT (0.5 0.5)"),
                Some("POINT (5 5)"),
                Some("POINT (0 0)"),
            ],
            &WKB_GEOMETRY,
        );

        let expected: ArrayRef = arrow_array!(Boolean, [Some(true), Some(false), None]);
        assert_array_equal(&tester.invoke_array_array(arg1, arg2).unwrap(), &expected);
    }

    #[rstest]
    fn covered_by_udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        use datafusion_common::ScalarValue;

        let udf = SedonaScalarUDF::from_kernel("st_coveredby", st_covered_by_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type.clone(), sedona_type]);
        tester.assert_return_type(DataType::Boolean);

        let result = tester
            .invoke_scalar_scalar("POINT (0 0)", "LINESTRING (2 0, 0 2)")
            .unwrap();
        tester.assert_scalar_result_equals(result, false);

        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());

        let arg1 = create_array(
            &[
                Some("POINT (0.5 0.5)"),
                Some("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
                None,
            ],
            &WKB_GEOMETRY,
        );
        let arg2 = create_array(
            &[
                Some("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
                Some("POINT (5 5)"),
                Some("POINT (0 0)"),
            ],
            &WKB_GEOMETRY,
        );

        let expected: ArrayRef = arrow_array!(Boolean, [Some(true), Some(false), None]);
        assert_array_equal(&tester.invoke_array_array(arg1, arg2).unwrap(), &expected);
    }

    #[rstest]
    fn covers_udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        use datafusion_common::ScalarValue;

        let udf = SedonaScalarUDF::from_kernel("st_covers", st_covers_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type.clone(), sedona_type]);
        tester.assert_return_type(DataType::Boolean);

        let result = tester
            .invoke_scalar_scalar("POINT (0 0)", "LINESTRING (2 0, 0 2)")
            .unwrap();
        tester.assert_scalar_result_equals(result, false);

        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());

        let arg1 = create_array(
            &[
                Some("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
                Some("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
                None,
            ],
            &WKB_GEOMETRY,
        );
        let arg2 = create_array(
            &[
                Some("POINT (0.5 0.5)"),
                Some("POINT (5 5)"),
                Some("POINT (0 0)"),
            ],
            &WKB_GEOMETRY,
        );

        let expected: ArrayRef = arrow_array!(Boolean, [Some(true), Some(false), None]);
        assert_array_equal(&tester.invoke_array_array(arg1, arg2).unwrap(), &expected);
    }

    #[rstest]
    fn disjoint_udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        use datafusion_common::ScalarValue;

        let udf = SedonaScalarUDF::from_kernel("st_disjoint", st_disjoint_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type.clone(), sedona_type]);
        tester.assert_return_type(DataType::Boolean);

        let result = tester
            .invoke_scalar_scalar("POINT (0 0)", "LINESTRING (2 0, 0 2)")
            .unwrap();
        tester.assert_scalar_result_equals(result, true);

        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());

        let arg1 = create_array(
            &[
                Some("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
                Some("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
                None,
            ],
            &WKB_GEOMETRY,
        );
        let arg2 = create_array(
            &[
                Some("POINT (5 5)"),
                Some("POINT (0.5 0.5)"),
                Some("POINT (0 0)"),
            ],
            &WKB_GEOMETRY,
        );

        let expected: ArrayRef = arrow_array!(Boolean, [Some(true), Some(false), None]);
        assert_array_equal(&tester.invoke_array_array(arg1, arg2).unwrap(), &expected);
    }

    #[rstest]
    fn equals_udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        use datafusion_common::ScalarValue;

        let udf = SedonaScalarUDF::from_kernel("st_equals", st_equals_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type.clone(), sedona_type]);
        tester.assert_return_type(DataType::Boolean);

        let result = tester
            .invoke_scalar_scalar("POINT (0 0)", "LINESTRING (2 0, 0 2)")
            .unwrap();
        tester.assert_scalar_result_equals(result, false);

        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());

        let arg1 = create_array(
            &[
                Some("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
                Some("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
                None,
            ],
            &WKB_GEOMETRY,
        );
        let arg2 = create_array(
            &[
                Some("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
                Some("POINT (0.5 0.5)"),
                Some("POINT (0 0)"),
            ],
            &WKB_GEOMETRY,
        );

        let expected: ArrayRef = arrow_array!(Boolean, [Some(true), Some(false), None]);
        assert_array_equal(&tester.invoke_array_array(arg1, arg2).unwrap(), &expected);
    }

    #[rstest]
    fn intersects_udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        use datafusion_common::ScalarValue;

        let udf = SedonaScalarUDF::from_kernel("st_intersects", st_intersects_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type.clone(), sedona_type]);
        tester.assert_return_type(DataType::Boolean);

        let result = tester
            .invoke_scalar_scalar("POINT (0 0)", "LINESTRING (2 0, 0 2)")
            .unwrap();
        tester.assert_scalar_result_equals(result, false);

        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());

        let arg1 = create_array(
            &[
                Some("POINT (0 0)"),
                Some("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
                None,
            ],
            &WKB_GEOMETRY,
        );
        let arg2 = create_array(
            &[
                Some("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
                Some("POINT (5 5)"),
                Some("POINT (0 0)"),
            ],
            &WKB_GEOMETRY,
        );

        let expected: ArrayRef = arrow_array!(Boolean, [Some(true), Some(false), None]);
        assert_array_equal(&tester.invoke_array_array(arg1, arg2).unwrap(), &expected);
    }

    #[rstest]
    fn within_udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        use datafusion_common::ScalarValue;

        let udf = SedonaScalarUDF::from_kernel("st_within", st_within_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type.clone(), sedona_type]);
        tester.assert_return_type(DataType::Boolean);

        let result = tester
            .invoke_scalar_scalar("POINT (0 0)", "LINESTRING (2 0, 0 2)")
            .unwrap();
        tester.assert_scalar_result_equals(result, false);

        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());

        let arg1 = create_array(
            &[
                Some("POINT (0.5 0.5)"),
                Some("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
                None,
            ],
            &WKB_GEOMETRY,
        );
        let arg2 = create_array(
            &[
                Some("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
                Some("POINT (5 5)"),
                Some("POINT (0 0)"),
            ],
            &WKB_GEOMETRY,
        );

        let expected: ArrayRef = arrow_array!(Boolean, [Some(true), Some(false), None]);
        assert_array_equal(&tester.invoke_array_array(arg1, arg2).unwrap(), &expected);
    }
}
