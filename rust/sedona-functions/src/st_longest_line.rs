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

use std::{io::Write, sync::Arc};

use arrow_array::builder::BinaryBuilder;
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::{ColumnarValue, Volatility};
use geo_traits::Dimensions;
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{SedonaScalarKernel, SedonaScalarUDF},
};
use sedona_geometry::wkb_factory::{WKB_MIN_PROBABLE_BYTES, write_wkb_linestring_header};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

use crate::{executor::WkbExecutor, st_max_distance::farthest_coords};

/// ST_LongestLine() scalar UDF.
pub fn st_longest_line_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_longestline",
        ItemCrsKernel::wrap_impl(STLongestLine {}),
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct STLongestLine {}

impl SedonaScalarKernel for STLongestLine {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry(), ArgMatcher::is_geometry()],
            WKB_GEOMETRY,
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
        );
        let mut lhs_coords = Vec::new();
        let mut rhs_coords = Vec::new();

        executor.execute_wkb_wkb_void(|lhs, rhs| {
            match (lhs, rhs) {
                (Some(lhs), Some(rhs)) => {
                    match farthest_coords(lhs, rhs, &mut lhs_coords, &mut rhs_coords) {
                        Some((a, b)) => {
                            write_wkb_linestring_header(&mut builder, Dimensions::Xy, 2)
                                .map_err(|e| DataFusionError::External(Box::new(e)))?;
                            for ordinate in [a.0, a.1, b.0, b.1] {
                                builder.write_all(&ordinate.to_le_bytes())?;
                            }
                            builder.append_value([]);
                        }
                        None => builder.append_null(),
                    }
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
    use datafusion_common::ScalarValue;
    use rstest::rstest;
    use sedona_schema::datatypes::{
        WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS, WKB_LARGE_GEOMETRY, WKB_VIEW_GEOMETRY,
    };
    use sedona_testing::{
        compare::assert_array_equal, create::create_array, testers::ScalarUdfTester,
    };

    use super::*;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_LARGE_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(
            st_longest_line_udf().into(),
            vec![sedona_type.clone(), sedona_type],
        );
        tester.assert_return_type(WKB_GEOMETRY);

        let result = tester
            .invoke_scalar_scalar("LINESTRING (0 0, 0 1)", "POINT (1 0)")
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING (0 1, 1 0)");

        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());

        let lhs = create_array(
            &[Some("POINT (0 0)"), Some("POINT EMPTY"), None],
            &WKB_GEOMETRY,
        );
        let rhs = create_array(
            &[
                Some("LINESTRING (0 0, 0 2)"),
                Some("POINT (0 0)"),
                Some("POINT (0 0)"),
            ],
            &WKB_GEOMETRY,
        );
        let expected = create_array(&[Some("LINESTRING (0 0, 0 2)"), None, None], &WKB_GEOMETRY);
        assert_array_equal(&tester.invoke_array_array(lhs, rhs).unwrap(), &expected);
    }

    #[test]
    fn udf_invoke_item_crs() {
        let sedona_type = WKB_GEOMETRY_ITEM_CRS.clone();
        let tester = ScalarUdfTester::new(
            st_longest_line_udf().into(),
            vec![sedona_type.clone(), sedona_type.clone()],
        );
        tester.assert_return_type(sedona_type);

        let result = tester
            .invoke_scalar_scalar("POINT (0 0)", "POINT (3 4)")
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING (0 0, 3 4)");
    }
}
