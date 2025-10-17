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
use datafusion_common::{cast::as_float64_array, error::Result};
use datafusion_expr::ColumnarValue;
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_functions::executor::WkbExecutor;
use sedona_geo_generic_alg::line_measures::DistanceExt;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};
use wkb::reader::Wkb;

/// ST_DWithin() implementation using [DistanceExt]
pub fn st_dwithin_impl() -> ScalarKernelRef {
    Arc::new(STDWithin {})
}

#[derive(Debug)]
struct STDWithin {}

impl SedonaScalarKernel for STDWithin {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
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
        let executor = WkbExecutor::new(arg_types, args);
        let arg2_array = arg2.to_array(executor.num_iterations())?;
        let arg2_f64_array = as_float64_array(&arg2_array)?;
        let mut arg2_iter = arg2_f64_array.iter();
        let mut builder = BooleanBuilder::with_capacity(executor.num_iterations());
        executor.execute_wkb_wkb_void(|maybe_wkb0, maybe_wkb1| {
            match (maybe_wkb0, maybe_wkb1, arg2_iter.next().unwrap()) {
                (Some(wkb0), Some(wkb1), Some(distance)) => {
                    builder.append_value(invoke_scalar(wkb0, wkb1, distance)?);
                }
                _ => builder.append_null(),
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(wkb_a: &Wkb, wkb_b: &Wkb, distance_bound: f64) -> Result<bool> {
    let actual_distance = wkb_a.distance_ext(wkb_b);
    Ok(actual_distance <= distance_bound)
}

#[cfg(test)]
mod tests {
    use arrow_array::{create_array as arrow_array, ArrayRef};
    use datafusion_common::scalar::ScalarValue;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::create::create_scalar;
    use sedona_testing::testers::ScalarUdfTester;
    use sedona_testing::{compare::assert_array_equal, create::create_array};

    use super::*;

    #[rstest]
    fn udf(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] left_sedona_type: SedonaType,
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] right_sedona_type: SedonaType,
    ) {
        let udf = SedonaScalarUDF::from_kernel("st_dwithin", st_dwithin_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![
                left_sedona_type.clone(),
                right_sedona_type.clone(),
                SedonaType::Arrow(DataType::Float64),
            ],
        );

        assert_eq!(
            tester.return_type().unwrap(),
            SedonaType::Arrow(DataType::Boolean)
        );

        // Test points within distance (3-4-5 triangle, distance = 5.0)
        let point_0_0 = create_scalar(Some("POINT (0 0)"), &left_sedona_type);
        let point_3_4 = create_scalar(Some("POINT (3 4)"), &right_sedona_type);
        let distance_5 = ScalarValue::Float64(Some(5.0));
        let distance_4 = ScalarValue::Float64(Some(4.0));

        let result = tester
            .invoke_scalar_scalar_scalar(point_0_0.clone(), point_3_4.clone(), distance_5.clone())
            .unwrap();
        assert_eq!(result, ScalarValue::Boolean(Some(true)));

        // Test points outside distance
        let result = tester
            .invoke_scalar_scalar_scalar(point_0_0.clone(), point_3_4.clone(), distance_4.clone())
            .unwrap();
        assert_eq!(result, ScalarValue::Boolean(Some(false)));

        // Test with null values
        let result = tester
            .invoke_scalar_scalar_scalar(ScalarValue::Null, point_3_4.clone(), distance_5.clone())
            .unwrap();
        assert!(result.is_null());
        let result = tester
            .invoke_scalar_scalar_scalar(point_0_0.clone(), ScalarValue::Null, distance_5.clone())
            .unwrap();
        assert!(result.is_null());

        // Test with null distance
        let result = tester
            .invoke_scalar_scalar_scalar(
                point_0_0.clone(),
                point_3_4.clone(),
                ScalarValue::Float64(None),
            )
            .unwrap();
        assert!(result.is_null());

        // Test with array args
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
        let distance = arrow_array!(Int32, [Some(1), Some(1), Some(1), Some(1)]);
        let expected: ArrayRef = arrow_array!(Boolean, [Some(true), Some(false), None, Some(true)]);
        assert_array_equal(
            &tester.invoke_arrays(vec![arg1, arg2, distance]).unwrap(),
            &expected,
        );
    }
}
