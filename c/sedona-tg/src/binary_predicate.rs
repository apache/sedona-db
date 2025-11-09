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

use arrow_array::builder::BooleanBuilder;
use arrow_schema::DataType;
use datafusion_common::error::Result;
use datafusion_expr::ColumnarValue;
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::{executor::TgGeomExecutor, tg};

/// ST_Equals() implementation using tg
pub fn st_equals_impl() -> ScalarKernelRef {
    Arc::new(TgPredicate::<tg::Equals>::default())
}

/// ST_Intersects() implementation using tg
pub fn st_intersects_impl() -> ScalarKernelRef {
    Arc::new(TgPredicate::<tg::Intersects>::default())
}

/// ST_Disjoint() implementation using tg
pub fn st_disjoint_impl() -> ScalarKernelRef {
    Arc::new(TgPredicate::<tg::Disjoint>::default())
}

/// ST_Contains() implementation using tg
pub fn st_contains_impl() -> ScalarKernelRef {
    Arc::new(TgPredicate::<tg::Contains>::default())
}

/// ST_Within() implementation using tg
pub fn st_within_impl() -> ScalarKernelRef {
    Arc::new(TgPredicate::<tg::Within>::default())
}

/// ST_Covers() implementation using tg
pub fn st_covers_impl() -> ScalarKernelRef {
    Arc::new(TgPredicate::<tg::Covers>::default())
}

/// ST_CoveredBy() implementation using tg
pub fn st_covered_by_impl() -> ScalarKernelRef {
    Arc::new(TgPredicate::<tg::CoveredBy>::default())
}

/// ST_Touches() implementation using tg
pub fn st_touches_impl() -> ScalarKernelRef {
    Arc::new(TgPredicate::<tg::Touches>::default())
}

#[derive(Debug, Default)]
struct TgPredicate<Op> {
    _op: Op,
}

impl<Op: tg::BinaryPredicate> SedonaScalarKernel for TgPredicate<Op> {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
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
        let executor = TgGeomExecutor::new(arg_types, args);
        let mut builder = BooleanBuilder::with_capacity(executor.num_iterations());
        executor.execute_wkb_wkb_void(|lhs, rhs| {
            match (lhs, rhs) {
                (Some(lhs), Some(rhs)) => {
                    builder.append_value(Op::evaluate(lhs, rhs));
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
    use arrow_array::{create_array, ArrayRef};
    use datafusion_common::scalar::ScalarValue;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::WKB_GEOMETRY;
    use sedona_testing::{
        create::{create_array, create_scalar},
        testers::ScalarUdfTester,
    };

    use super::*;

    #[test]
    fn scalar_scalar() {
        let udf = SedonaScalarUDF::from_kernel("st_intersects", st_intersects_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOMETRY, WKB_GEOMETRY]);
        tester.assert_return_type(DataType::Boolean);

        let polygon_scalar = create_scalar(Some("POLYGON ((0 0, 1 0, 0 1, 0 0))"), &WKB_GEOMETRY);

        // Check something that intersects with both argument orders
        let result = tester
            .invoke_scalar_scalar("POINT (0.25 0.25)", polygon_scalar.clone())
            .unwrap();
        tester.assert_scalar_result_equals(result, true);

        let result = tester
            .invoke_scalar_scalar(polygon_scalar.clone(), "POINT (0.25 0.25)")
            .unwrap();
        tester.assert_scalar_result_equals(result, true);

        // Check something that doesn't intersect with both argument orders
        let result = tester
            .invoke_scalar_scalar("POINT (10 10)", polygon_scalar.clone())
            .unwrap();
        tester.assert_scalar_result_equals(result, false);

        let result = tester
            .invoke_scalar_scalar(polygon_scalar.clone(), "POINT (10 10)")
            .unwrap();
        tester.assert_scalar_result_equals(result, false);

        // Check a null in both argument orders
        let result = tester
            .invoke_scalar_scalar(polygon_scalar.clone(), ScalarValue::Null)
            .unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Null);

        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, polygon_scalar.clone())
            .unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Null);

        // ...and check a null as both arguments
        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, ScalarValue::Null)
            .unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Null);
    }

    #[test]
    fn scalar_array() {
        let udf = SedonaScalarUDF::from_kernel("st_intersects", st_intersects_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOMETRY, WKB_GEOMETRY]);
        tester.assert_return_type(DataType::Boolean);

        let point_array = create_array(
            &[Some("POINT (0.25 0.25)"), Some("POINT (10 10)"), None],
            &WKB_GEOMETRY,
        );
        let polygon_scalar = create_scalar(Some("POLYGON ((0 0, 1 0, 0 1, 0 0))"), &WKB_GEOMETRY);

        // Array, Scalar -> Array
        let expected: ArrayRef = create_array!(Boolean, [Some(true), Some(false), None]);
        assert_eq!(
            &tester
                .invoke_array_scalar(point_array.clone(), polygon_scalar.clone())
                .unwrap(),
            &expected
        );
        assert_eq!(
            &tester
                .invoke_scalar_array(polygon_scalar.clone(), point_array.clone())
                .unwrap(),
            &expected
        );
    }

    #[test]
    fn array_array() {
        let udf = SedonaScalarUDF::from_kernel("st_intersects", st_intersects_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOMETRY, WKB_GEOMETRY]);
        tester.assert_return_type(DataType::Boolean);

        let point_array = create_array(
            &[
                Some("POINT (0.25 0.25)"),
                Some("POINT (10 10)"),
                None,
                Some("POINT (0.25 0.25)"),
            ],
            &WKB_GEOMETRY,
        );
        let polygon_array = create_array(
            &[
                Some("POLYGON ((0 0, 1 0, 0 1, 0 0))"),
                Some("POLYGON ((0 0, 1 0, 0 1, 0 0))"),
                Some("POLYGON ((0 0, 1 0, 0 1, 0 0))"),
                None,
            ],
            &WKB_GEOMETRY,
        );

        // Array, Array -> Array
        let expected: ArrayRef = create_array!(Boolean, [Some(true), Some(false), None, None]);
        assert_eq!(
            &tester
                .invoke_array_array(point_array, polygon_array)
                .unwrap(),
            &expected
        );
    }
}
