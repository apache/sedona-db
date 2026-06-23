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
use geo::{CoordsIter, Geometry};
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{ScalarKernelRef, SedonaScalarKernel},
};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::to_geo::GeoTypesExecutor;

/// ST_MaxDistance() — max pairwise 2D vertex distance using geo-types CoordsIter
pub fn st_max_distance_impl() -> Vec<ScalarKernelRef> {
    ItemCrsKernel::wrap_impl(STMaxDistance {})
}

#[derive(Debug)]
struct STMaxDistance {}

impl SedonaScalarKernel for STMaxDistance {
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
        let executor = GeoTypesExecutor::new(arg_types, args);
        let mut builder = Float64Builder::with_capacity(executor.num_iterations());
        executor.execute_wkb_wkb_void(|lhs, rhs| {
            match (lhs, rhs) {
                (Some(lhs), Some(rhs)) => match invoke_scalar(lhs, rhs) {
                    Some(dist) => builder.append_value(dist),
                    None => builder.append_null(),
                },
                _ => builder.append_null(),
            }
            Ok(())
        })?;
        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(lhs: &Geometry, rhs: &Geometry) -> Option<f64> {
    let lhs_coords: Vec<_> = lhs.coords_iter().collect();
    let rhs_coords: Vec<_> = rhs.coords_iter().collect();

    if lhs_coords.is_empty() || rhs_coords.is_empty() {
        return None;
    }

    let mut max_dist_sq = f64::NEG_INFINITY;
    for a in &lhs_coords {
        for b in &rhs_coords {
            let dx = a.x - b.x;
            let dy = a.y - b.y;
            let d2 = dx * dx + dy * dy;
            if d2 > max_dist_sq {
                max_dist_sq = d2;
            }
        }
    }

    Some(max_dist_sq.max(0.0).sqrt())
}

#[cfg(test)]
mod tests {
    use arrow_array::{create_array as arrow_array, ArrayRef};
    use datafusion_common::ScalarValue;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS, WKB_VIEW_GEOMETRY};
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::create::create_array;
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_impl("st_maxdistance", st_max_distance_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type.clone(), sedona_type]);
        tester.assert_return_type(DataType::Float64);

        let result = tester
            .invoke_scalar_scalar("POINT (0 0)", "POINT (3 4)")
            .unwrap();
        tester.assert_scalar_result_equals(result, 5.0);

        let result = tester
            .invoke_scalar_scalar("POINT (0 0)", "LINESTRING (0 0, 0 2)")
            .unwrap();
        tester.assert_scalar_result_equals(result, 2.0);

        let result = tester
            .invoke_scalar_scalar("LINESTRING (0 0, 10 0)", "LINESTRING (0 10, 10 10)")
            .unwrap();
        tester.assert_scalar_result_equals(result, 200f64.sqrt());

        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());

        let result = tester
            .invoke_scalar_scalar("LINESTRING EMPTY", "POINT (0 0)")
            .unwrap();
        assert!(result.is_null(), "expected NULL for empty lhs");

        let result = tester
            .invoke_scalar_scalar("POINT (0 0)", "LINESTRING EMPTY")
            .unwrap();
        assert!(result.is_null(), "expected NULL for empty rhs");

        let result = tester
            .invoke_scalar_scalar("LINESTRING EMPTY", "LINESTRING EMPTY")
            .unwrap();
        assert!(result.is_null(), "expected NULL for both empty");

        let lhs = create_array(
            &[Some("POINT (0 0)"), Some("POINT (0 0)"), None],
            &WKB_GEOMETRY,
        );
        let rhs = create_array(
            &[Some("POINT (3 4)"), None, Some("POINT (1 1)")],
            &WKB_GEOMETRY,
        );
        let expected: ArrayRef = arrow_array!(Float64, [Some(5.0), None, None]);
        assert_array_equal(&tester.invoke_array_array(lhs, rhs).unwrap(), &expected);
    }

    #[rstest]
    fn udf_invoke_item_crs(#[values(WKB_GEOMETRY_ITEM_CRS.clone())] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_impl("st_maxdistance", st_max_distance_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type.clone(), sedona_type]);
        tester.assert_return_type(DataType::Float64);

        let result = tester
            .invoke_scalar_scalar("POINT (0 0)", "POINT (3 4)")
            .unwrap();
        tester.assert_scalar_result_equals(result, 5.0);
    }
}
