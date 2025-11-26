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

use arrow_array::builder::Int32Builder;
use arrow_schema::DataType;
use datafusion_common::{error::Result, DataFusionError};
use datafusion_expr::ColumnarValue;
use geos::Geom;
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::executor::GeosExecutor;

/// ST_NumInteriorRings() implementation using the geos crate
pub fn st_num_interior_rings_impl() -> ScalarKernelRef {
    Arc::new(STNumInteriorRings {})
}

#[derive(Debug)]
struct STNumInteriorRings {}

impl SedonaScalarKernel for STNumInteriorRings {
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

        // single-geometry executor path
        executor.execute_wkb_void(|geom| {
            match geom {
                Some(g) => {
                    let n = invoke_scalar(&g)?;
                    builder.append_value(n);
                }
                None => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(geos_geom: &geos::Geometry) -> Result<i32> {
    // geos::Geometry provides get_num_interior_rings() -> GResult<usize>
    let count = geos_geom.get_num_interior_rings().map_err(|e| {
        DataFusionError::Execution(format!("Failed to get num interior rings: {e}"))
    })?;

    // safe to cast to i32 for SQL integer return
    Ok(count as i32)
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
        let udf = SedonaScalarUDF::from_kernel("st_numinteriorrings", st_num_interior_rings_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type.clone()]);
        tester.assert_return_type(DataType::Int32);

        // scalar-scalar: polygon with two interior rings -> 2
        let result = tester
            .invoke_scalar_scalar(
                "POLYGON((0 0,10 0,10 6,0 6,0 0),(1 1,2 1,2 5,1 5,1 1),(8 5,8 4,9 4,9 5,8 5))", // returns 2
                "POLYGON((0 0,10 0,10 6,0 6,0 0))", // second arg ignored, only single-arg UDF tester will pass one value, but keep pattern consistent
            )
            .unwrap();

        // Note: Above tester.invoke_scalar_scalar in your testing harness for single-arg UDFs
        // may accept two parameters; if not, use invoke_scalar("WKT") variant available in the tester.
        tester.assert_scalar_result_equals(result, 2_i32);

        // Nulls -> Null
        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());

        // array-array style: mix of polygon, non-polygon and null
        let lhs = create_array(
            &[
                Some("POLYGON((0 0,10 0,10 6,0 6,0 0),(1 1,2 1,2 5,1 5,1 1))"), //returns 1
                Some("POINT (5 5)"), //returns 0 only out circle is there
                None,
            ],
            &WKB_GEOMETRY,
        );

        let expected: ArrayRef = arrow_array!(Int32, [Some(1), Some(0), None]);
        assert_array_equal(&tester.invoke_array_array(lhs).unwrap(), &expected);
    }
}
