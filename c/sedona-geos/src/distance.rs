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
use datafusion_common::{error::Result, DataFusionError};
use datafusion_expr::ColumnarValue;
use geos::Geom;
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::executor::GeosExecutor;

/// ST_Distance() implementation using the geos crate
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
        let executor = GeosExecutor::new(arg_types, args);
        let mut builder = Float64Builder::with_capacity(executor.num_iterations());
        executor.execute_wkb_wkb_void(|lhs, rhs| {
            match (lhs, rhs) {
                (Some(lhs), Some(rhs)) => {
                    builder.append_value(invoke_scalar(lhs, rhs)?);
                }
                _ => builder.append_null(),
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(geos_geom: &geos::Geometry, other_geos_geom: &geos::Geometry) -> Result<f64> {
    let distance = geos_geom
        .distance(other_geos_geom)
        .map_err(|e| DataFusionError::Execution(format!("Failed to calculate distance: {e}")))?;

    Ok(distance)
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
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        use datafusion_common::ScalarValue;

        let udf = SedonaScalarUDF::from_kernel("st_distance", st_distance_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type.clone(), sedona_type]);
        tester.assert_return_type(DataType::Float64);

        let result = tester
            .invoke_scalar_scalar("POINT (0 0)", "POINT (3 4)")
            .unwrap();
        tester.assert_scalar_result_equals(result, 5.0);

        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());

        let lhs = create_array(
            &[Some("POINT (72 42)"), Some("POINT EMPTY"), None],
            &WKB_GEOMETRY,
        );
        let rhs = create_array(
            &[
                Some("LINESTRING(-72 -42, 82 92)"),
                Some("POINT (5 5)"),
                Some("POINT (0 0)"),
            ],
            &WKB_GEOMETRY,
        );

        let expected: ArrayRef = arrow_array!(Float64, [Some(31.155515639003543), Some(0.0), None]);
        assert_array_equal(&tester.invoke_array_array(lhs, rhs).unwrap(), &expected);
    }
}
