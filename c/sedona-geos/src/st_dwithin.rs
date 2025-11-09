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
use datafusion_common::{cast::as_float64_array, error::Result, DataFusionError};
use datafusion_expr::ColumnarValue;
use geos::Geom;
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::executor::GeosExecutor;

/// Implementation of ST_DWithin using the geos crate
pub fn st_dwithin_impl() -> ScalarKernelRef {
    Arc::new(STDWithin {})
}

#[derive(Debug)]
struct STDWithin {}

impl SedonaScalarKernel for STDWithin {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher: ArgMatcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_geometry(),
                ArgMatcher::is_geometry(),
                ArgMatcher::is_numeric(),
            ],
            SedonaType::Arrow(DataType::Boolean),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let arg2 = args[2].cast_to(&DataType::Float64, None)?;
        let executor = GeosExecutor::new(arg_types, args);
        let arg2_array = arg2.to_array(executor.num_iterations())?;
        let arg2_f64_array = as_float64_array(&arg2_array)?;
        let mut arg2_iter = arg2_f64_array.iter();
        let mut builder = BooleanBuilder::with_capacity(executor.num_iterations());
        executor.execute_wkb_wkb_void(|lhs, rhs| {
            match (lhs, rhs, arg2_iter.next().unwrap()) {
                (Some(lhs), Some(rhs), Some(distance)) => {
                    builder.append_value(invoke_scalar(lhs, rhs, distance)?);
                }
                _ => builder.append_null(),
            };
            Ok(())
        })?;
        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(lhs: &geos::Geometry, rhs: &geos::Geometry, distance: f64) -> Result<bool> {
    let dist_between = lhs
        .distance(rhs)
        .map_err(|e| DataFusionError::Execution(format!("Failed to calculate dwithin: {e}")))?;
    Ok(dist_between <= distance)
}

#[cfg(test)]
mod tests {
    use arrow_array::{create_array as arrow_array, ArrayRef};
    use datafusion_common::ScalarValue;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::create::create_array;
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_dwithin", st_dwithin_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![
                sedona_type.clone(),
                sedona_type,
                SedonaType::Arrow(DataType::Float64),
            ],
        );
        tester.assert_return_type(DataType::Boolean);

        let result = tester
            .invoke_scalar_scalar_scalar("POINT (0 0)", "POINT (0 0)", 0.0)
            .unwrap();
        tester.assert_scalar_result_equals(result, true);

        let result = tester
            .invoke_scalar_scalar_scalar(ScalarValue::Null, ScalarValue::Null, ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());

        let arg1 = create_array(
            &[
                Some("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
                Some("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
                None,
                Some("POINT EMPTY"),
            ],
            &WKB_GEOMETRY,
        );
        let arg2 = create_array(
            &[
                Some("POINT (0.5 0.5)"),
                Some("POINT (5 5)"),
                Some("POINT (0 0)"),
                Some("POINT EMPTY"),
            ],
            &WKB_GEOMETRY,
        );
        let distance = 1;

        let expected: ArrayRef = arrow_array!(Boolean, [Some(true), Some(false), None, Some(true)]);
        assert_array_equal(
            &tester
                .invoke_array_array_scalar(Arc::clone(&arg1), Arc::clone(&arg2), distance)
                .unwrap(),
            &expected,
        );

        let distance = arrow_array!(Int32, [Some(1), Some(1), Some(1), Some(1)]);
        let expected: ArrayRef = arrow_array!(Boolean, [Some(true), Some(false), None, Some(true)]);
        assert_array_equal(
            &tester.invoke_arrays(vec![arg1, arg2, distance]).unwrap(),
            &expected,
        );
    }
}
