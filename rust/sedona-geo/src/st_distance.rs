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
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_functions::executor::WkbExecutor;
use sedona_geo_generic_alg::line_measures::DistanceExt;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};
use wkb::reader::Wkb;

/// ST_Distance() implementation using [DistanceExt]
pub fn st_distance_impl() -> ScalarKernelRef {
    Arc::new(STDistance {})
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
        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = Float64Builder::with_capacity(executor.num_iterations());
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

fn invoke_scalar(wkb_a: &Wkb, wkb_b: &Wkb) -> Result<f64> {
    Ok(wkb_a.distance_ext(wkb_b))
}

#[cfg(test)]
mod tests {
    use datafusion_common::scalar::ScalarValue;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::create::create_scalar;
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn udf(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] left_sedona_type: SedonaType,
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] right_sedona_type: SedonaType,
    ) {
        let udf = SedonaScalarUDF::from_kernel("st_distance", st_distance_impl());
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
    }
}
