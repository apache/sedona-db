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
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::ColumnarValue;
use geos::{Geom, GeometryTypes};
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{ScalarKernelRef, SedonaScalarKernel},
};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::executor::GeosExecutor;

/// ST_LineLocatePoint() implementation using the geos crate
pub fn st_line_locate_point_impl() -> Vec<ScalarKernelRef> {
    ItemCrsKernel::wrap_impl(STLineLocatePoint {})
}

#[derive(Debug)]
struct STLineLocatePoint {}

impl SedonaScalarKernel for STLineLocatePoint {
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
        executor.execute_wkb_wkb_void(|line, point| {
            match (line, point) {
                (Some(line), Some(point)) => match invoke_scalar(line, point)? {
                    Some(fraction) => builder.append_value(fraction),
                    None => builder.append_null(),
                },
                _ => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(line: &geos::Geometry, point: &geos::Geometry) -> Result<Option<f64>> {
    // Empty input -> NULL (consistent with SedonaDB's general approach to empties)
    let line_empty = line
        .is_empty()
        .map_err(|e| DataFusionError::Execution(format!("Failed to check empty: {e}")))?;
    let point_empty = point
        .is_empty()
        .map_err(|e| DataFusionError::Execution(format!("Failed to check empty: {e}")))?;
    if line_empty || point_empty {
        return Ok(None);
    }

    match line
        .geometry_type()
        .map_err(|e| DataFusionError::Execution(format!("Failed to get geometry type: {e}")))?
    {
        GeometryTypes::LineString => {
            let fraction = line.project_normalized(point).map_err(|e| {
                DataFusionError::Execution(format!("Failed to locate point on line: {e}"))
            })?;
            Ok(Some(fraction))
        }
        _ => Err(DataFusionError::Execution(
            "First argument to ST_LineLocatePoint must be a single linestring".to_string(),
        )),
    }
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
        let udf = SedonaScalarUDF::from_impl("st_linelocatepoint", st_line_locate_point_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type.clone(), sedona_type]);
        tester.assert_return_type(DataType::Float64);

        let result = tester
            .invoke_scalar_scalar("LINESTRING (0 0, 0 1)", "POINT (0 0.5)")
            .unwrap();
        tester.assert_scalar_result_equals(result, 0.5);

        let result = tester
            .invoke_scalar_scalar("LINESTRING (0 0, 0 10)", "POINT (0 0)")
            .unwrap();
        tester.assert_scalar_result_equals(result, 0.0);

        let result = tester
            .invoke_scalar_scalar("LINESTRING (0 0, 0 10)", "POINT (0 10)")
            .unwrap();
        tester.assert_scalar_result_equals(result, 1.0);

        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());

        // Non-LineString first arg errors, matching the geography implementation.
        let result =
            tester.invoke_scalar_scalar("POLYGON ((0 0, 1 0, 1 1, 0 0))", "POINT (0.5 0.5)");
        assert!(result.is_err());

        // Empty inputs return NULL
        let result = tester
            .invoke_scalar_scalar("LINESTRING EMPTY", "POINT (0 0)")
            .unwrap();
        assert!(result.is_null(), "expected NULL for empty line");

        let result = tester
            .invoke_scalar_scalar("LINESTRING (0 0, 1 1)", "POINT EMPTY")
            .unwrap();
        assert!(result.is_null(), "expected NULL for empty point");

        let lines = create_array(
            &[
                Some("LINESTRING (0 0, 10 0)"),
                Some("LINESTRING (0 0, 10 0)"),
                None,
            ],
            &WKB_GEOMETRY,
        );
        let points = create_array(
            &[Some("POINT (5 0)"), None, Some("POINT (0 0)")],
            &WKB_GEOMETRY,
        );
        let expected: ArrayRef = arrow_array!(Float64, [Some(0.5), None, None]);
        assert_array_equal(
            &tester.invoke_array_array(lines, points).unwrap(),
            &expected,
        );
    }

    #[rstest]
    fn udf_invoke_item_crs(#[values(WKB_GEOMETRY_ITEM_CRS.clone())] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_impl("st_linelocatepoint", st_line_locate_point_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type.clone(), sedona_type]);
        tester.assert_return_type(DataType::Float64);

        let result = tester
            .invoke_scalar_scalar("LINESTRING (0 0, 0 1)", "POINT (0 0.5)")
            .unwrap();
        tester.assert_scalar_result_equals(result, 0.5);
    }
}
