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
use geo_generic_alg::Intersects;
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_functions::executor::WkbExecutor;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};
use wkb::reader::Wkb;

/// ST_Intersects() implementation using [Intersects]
pub fn st_intersects_impl() -> ScalarKernelRef {
    Arc::new(STIntersects {})
}

#[derive(Debug)]
struct STIntersects {}

impl SedonaScalarKernel for STIntersects {
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
        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = BooleanBuilder::with_capacity(executor.num_iterations());
        executor.execute_wkb_wkb_void(|maybe_wkb0, maybe_wkb1| {
            match (maybe_wkb0, maybe_wkb1) {
                (Some(wkb0), Some(wkb1)) => {
                    builder.append_value(invoke_scalar(wkb0, wkb1)?);
                }
                _ => builder.append_null(),
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(item_a: &Wkb, geom_b: &Wkb) -> Result<bool> {
    Ok(item_a.intersects(&geom_b))
}

#[cfg(test)]
mod tests {
    use arrow_array::{create_array, ArrayRef};
    use datafusion_common::scalar::ScalarValue;
    use sedona_functions::register::stubs::st_intersects_udf;
    use sedona_schema::datatypes::WKB_GEOMETRY;
    use sedona_testing::{
        create::{create_array, create_scalar},
        testers::ScalarUdfTester,
    };

    use super::*;

    #[test]
    fn scalar_scalar() {
        let mut udf = st_intersects_udf();
        udf.add_kernel(st_intersects_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOMETRY, WKB_GEOMETRY]);

        let point = create_scalar(Some("POINT (0.25 0.25)"), &WKB_GEOMETRY);
        let point2 = create_scalar(Some("POINT (10 10)"), &WKB_GEOMETRY);
        let polygon = create_scalar(Some("POLYGON ((0 0, 1 0, 0 1, 0 0))"), &WKB_GEOMETRY);

        // Check something that is true
        assert_eq!(
            tester
                .invoke_scalar_scalar(point.clone(), polygon.clone())
                .unwrap(),
            ScalarValue::Boolean(Some(true))
        );
        assert_eq!(
            tester
                .invoke_scalar_scalar(polygon.clone(), point.clone())
                .unwrap(),
            ScalarValue::Boolean(Some(true))
        );

        // Check something that is false
        assert_eq!(
            tester
                .invoke_scalar_scalar(point2.clone(), polygon.clone())
                .unwrap(),
            ScalarValue::Boolean(Some(false))
        );
        assert_eq!(
            tester
                .invoke_scalar_scalar(polygon.clone(), point2.clone())
                .unwrap(),
            ScalarValue::Boolean(Some(false))
        );

        // Check one null
        assert_eq!(
            tester
                .invoke_scalar_scalar(polygon.clone(), ScalarValue::Null)
                .unwrap(),
            ScalarValue::Boolean(None)
        );
        assert_eq!(
            tester
                .invoke_scalar_scalar(ScalarValue::Null, polygon.clone())
                .unwrap(),
            ScalarValue::Boolean(None)
        );

        // Check both nulls
        assert_eq!(
            tester
                .invoke_scalar_scalar(ScalarValue::Null, ScalarValue::Null)
                .unwrap(),
            ScalarValue::Boolean(None)
        );
    }

    #[test]
    fn scalar_array() {
        let mut udf = st_intersects_udf();
        udf.add_kernel(st_intersects_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOMETRY, WKB_GEOMETRY]);

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

        // Scalar, Array -> Array
        assert_eq!(
            &tester
                .invoke_scalar_array(polygon_scalar.clone(), point_array.clone())
                .unwrap(),
            &expected
        );
    }

    #[test]
    fn array_array() {
        let mut udf = st_intersects_udf();
        udf.add_kernel(st_intersects_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOMETRY, WKB_GEOMETRY]);

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
