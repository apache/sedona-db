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

use crate::executor::GeosExecutor;
use arrow_array::builder::Int32Builder;
use arrow_schema::DataType;
use datafusion_common::{error::Result, DataFusionError};
use datafusion_expr::ColumnarValue;
use geos::{Geom, Geometry, GeometryTypes};
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

pub fn st_num_points_impl() -> ScalarKernelRef {
    Arc::new(STNumPoints {})
}

#[derive(Debug)]
struct STNumPoints {}

impl SedonaScalarKernel for STNumPoints {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry()],
            SedonaType::Arrow(DataType::Int32),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = GeosExecutor::new(arg_types, args);
        let mut builder = Int32Builder::with_capacity(executor.num_iterations());
        executor.execute_wkb_void(|maybe_geom| {
            match maybe_geom {
                None => builder.append_null(),
                Some(geom) => {
                    let res = invoke_scalar(&geom)?;
                    match res {
                        Some(n) => builder.append_value(n),
                        None => builder.append_null(),
                    }
                }
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(geom: &Geometry) -> Result<Option<i32>> {
    match geom
        .geometry_type()
        .map_err(|e| DataFusionError::Execution(format!("Failed to get geometry type: {e}")))?
    {
        GeometryTypes::LineString => {
            let count = geom.get_num_points().map_err(|e| {
                DataFusionError::Execution(format!("Failed to get num points: {e}"))
            })?;
            Ok(Some(count as i32))
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int32Array};
    use arrow_schema::DataType;
    use datafusion_common::ScalarValue;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{SedonaType, WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_numpoints", st_num_points_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type]);
        tester.assert_return_type(DataType::Int32);
        let result = tester.invoke_scalar("LINESTRING (1 2, 3 4)").unwrap();
        tester.assert_scalar_result_equals(result, 2_i32);

        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        assert!(result.is_null());

        let input_wkt = vec![
            None,
            Some("POINT (1 2)"),
            Some("LINESTRING (0 0, 1 1, 2 2)"),
            Some("POLYGON EMPTY"),
            Some("LINESTRING(0 0, 1 1, 1 2, 2 2)"),
            Some("MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))"),
            Some("POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))"),
            Some("GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (0 0, 1 1))"),
        ];

        let expected: ArrayRef = Arc::new(Int32Array::from(vec![
            None,
            None,
            Some(3),
            None,
            Some(4),
            None,
            None,
            None,
        ]));

        let result = tester.invoke_wkb_array(input_wkt).unwrap();
        assert_array_equal(&result, &expected);
    }
}
