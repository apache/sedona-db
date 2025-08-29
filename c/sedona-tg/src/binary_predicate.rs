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
use sedona_expr::scalar_udf::{ArgMatcher, ScalarKernelRef, SedonaScalarKernel};
use sedona_schema::datatypes::SedonaType;

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
            DataType::Boolean.try_into().unwrap(),
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
    use arrow_array::create_array;
    use datafusion_common::scalar::ScalarValue;
    use sedona_functions::register::stubs::st_intersects_udf;
    use sedona_schema::datatypes::WKB_GEOMETRY;
    use sedona_testing::{
        compare::assert_value_equal, create::create_array_value, create::create_scalar_value,
    };

    use super::*;

    #[test]
    fn scalar_scalar() {
        let mut udf = st_intersects_udf();
        udf.add_kernel(st_intersects_impl());

        let point_scalar = create_scalar_value(Some("POINT (0.25 0.25)"), &WKB_GEOMETRY);
        let point2_scalar = create_scalar_value(Some("POINT (10 10)"), &WKB_GEOMETRY);
        let polygon_scalar =
            create_scalar_value(Some("POLYGON ((0 0, 1 0, 0 1, 0 0))"), &WKB_GEOMETRY);
        let null_scalar = create_scalar_value(None, &WKB_GEOMETRY);

        // Check something that intersects with both argument orders
        assert_value_equal(
            &udf.invoke_batch(&[point_scalar.clone(), polygon_scalar.clone()], 1)
                .unwrap(),
            &ScalarValue::Boolean(Some(true)).into(),
        );

        assert_value_equal(
            &udf.invoke_batch(&[polygon_scalar.clone(), point_scalar.clone()], 1)
                .unwrap(),
            &ScalarValue::Boolean(Some(true)).into(),
        );

        // Check something that doesn't intersect with both argument orders
        assert_value_equal(
            &udf.invoke_batch(&[point2_scalar.clone(), polygon_scalar.clone()], 1)
                .unwrap(),
            &ScalarValue::Boolean(Some(false)).into(),
        );

        assert_value_equal(
            &udf.invoke_batch(&[polygon_scalar.clone(), point2_scalar.clone()], 1)
                .unwrap(),
            &ScalarValue::Boolean(Some(false)).into(),
        );

        // Check a null in both argument orders
        assert_value_equal(
            &udf.invoke_batch(&[null_scalar.clone(), polygon_scalar.clone()], 1)
                .unwrap(),
            &ScalarValue::Boolean(None).into(),
        );

        assert_value_equal(
            &udf.invoke_batch(&[polygon_scalar.clone(), null_scalar.clone()], 1)
                .unwrap(),
            &ScalarValue::Boolean(None).into(),
        );

        // ...and check a null as both arguments
        assert_value_equal(
            &udf.invoke_batch(&[null_scalar.clone(), null_scalar.clone()], 1)
                .unwrap(),
            &ScalarValue::Boolean(None).into(),
        );
    }

    #[test]
    fn scalar_array() {
        let mut udf = st_intersects_udf();
        udf.add_kernel(st_intersects_impl());

        let point_array = create_array_value(
            &[Some("POINT (0.25 0.25)"), Some("POINT (10 10)"), None],
            &WKB_GEOMETRY,
        );
        let polygon_scalar =
            create_scalar_value(Some("POLYGON ((0 0, 1 0, 0 1, 0 0))"), &WKB_GEOMETRY);

        // Array, Scalar -> Array
        assert_value_equal(
            &udf.invoke_batch(&[point_array.clone(), polygon_scalar.clone()], 1)
                .unwrap(),
            &ColumnarValue::Array(create_array!(Boolean, [Some(true), Some(false), None])),
        );

        // Scalar, Array -> Array
        assert_value_equal(
            &udf.invoke_batch(&[polygon_scalar.clone(), point_array.clone()], 1)
                .unwrap(),
            &ColumnarValue::Array(create_array!(Boolean, [Some(true), Some(false), None])),
        );
    }

    #[test]
    fn array_array() {
        let mut udf = st_intersects_udf();
        udf.add_kernel(st_intersects_impl());

        let point_array = create_array_value(
            &[
                Some("POINT (0.25 0.25)"),
                Some("POINT (10 10)"),
                None,
                Some("POINT (0.25 0.25)"),
            ],
            &WKB_GEOMETRY,
        );
        let polygon_array = create_array_value(
            &[
                Some("POLYGON ((0 0, 1 0, 0 1, 0 0))"),
                Some("POLYGON ((0 0, 1 0, 0 1, 0 0))"),
                Some("POLYGON ((0 0, 1 0, 0 1, 0 0))"),
                None,
            ],
            &WKB_GEOMETRY,
        );

        // Array, Array -> Array
        assert_value_equal(
            &udf.invoke_batch(&[point_array, polygon_array], 1).unwrap(),
            &ColumnarValue::Array(create_array!(
                Boolean,
                [Some(true), Some(false), None, None]
            )),
        );
    }
}
