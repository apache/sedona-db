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

use arrow_array::builder::StringBuilder;
use arrow_schema::DataType;
use datafusion_common::error::Result;
use datafusion_common::DataFusionError;
use datafusion_expr::ColumnarValue;
use geos::Geom;
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{ScalarKernelRef, SedonaScalarKernel},
};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::executor::GeosExecutor;

/// ST_Relate implementation using GEOS
pub fn st_relate_impl() -> Vec<ScalarKernelRef> {
    ItemCrsKernel::wrap_impl(STRelate {})
}

#[derive(Debug)]
struct STRelate {}

impl SedonaScalarKernel for STRelate {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry(), ArgMatcher::is_geometry()],
            SedonaType::Arrow(DataType::Utf8),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = GeosExecutor::new(arg_types, args);

        // ST_Relate returns a 9-char DE-9IM string per row; 9 bytes * n rows
        let mut builder =
            StringBuilder::with_capacity(executor.num_iterations(), 9 * executor.num_iterations());

        executor.execute_wkb_wkb_void(|wkb1, wkb2| {
            match (wkb1, wkb2) {
                (Some(g1), Some(g2)) => {
                    let relate = g1
                        .relate(g2)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;

                    builder.append_value(relate);
                }
                _ => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
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
        let udf = SedonaScalarUDF::from_impl("st_relate", st_relate_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type.clone(), sedona_type]);
        tester.assert_return_type(DataType::Utf8);

        // Two disjoint points — DE-9IM should be "FF0FFF0F2"
        let result = tester
            .invoke_scalar_scalar("POINT (0 0)", "POINT (1 1)")
            .unwrap();
        tester.assert_scalar_result_equals(result, "FF0FFF0F2");

        // NULL inputs should return NULL
        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());

        // Array inputs
        let lhs = create_array(
            &[
                Some("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
                Some("POINT (0.5 0.5)"),
                None,
            ],
            &WKB_GEOMETRY,
        );
        let rhs = create_array(
            &[
                Some("POINT (0.5 0.5)"),
                Some("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
                Some("POINT (0 0)"),
            ],
            &WKB_GEOMETRY,
        );

        // actual values from GEOS
        let expected: ArrayRef = arrow_array!(Utf8, [Some("0F2FF1FF2"), Some("0FFFFF212"), None]);
        assert_array_equal(&tester.invoke_array_array(lhs, rhs).unwrap(), &expected);
    }
}
