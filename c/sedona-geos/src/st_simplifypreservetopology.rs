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

use arrow_array::builder::BinaryBuilder;
use arrow_schema::DataType;
use datafusion_common::cast::as_float64_array;
use datafusion_common::error::Result;
use datafusion_common::DataFusionError;
use datafusion_expr::ColumnarValue;
use geos::Geom;
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_geometry::wkb_factory::WKB_MIN_PROBABLE_BYTES;
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

use crate::executor::GeosExecutor;

/// ST_SimplifyPreserveTopology() implementation using the geos crate
pub fn st_simplify_preserve_topology_impl() -> ScalarKernelRef {
    Arc::new(STSimplifyPreserveTopology {})
}

#[derive(Debug)]
struct STSimplifyPreserveTopology {}

impl SedonaScalarKernel for STSimplifyPreserveTopology {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry(), ArgMatcher::is_numeric()],
            WKB_GEOMETRY,
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = GeosExecutor::new(arg_types, args);

        let tolerance_value = args[1]
            .cast_to(&DataType::Float64, None)?
            .to_array(executor.num_iterations())?;
        let tolerance_array = as_float64_array(&tolerance_value)?;

        let mut tolerance_iter = tolerance_array.iter();

        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
        );

        executor.execute_wkb_void(|wkb| {
            match (wkb, tolerance_iter.next().unwrap()) {
                (Some(wkb), Some(tolerance)) => {
                    invoke_scalar(&wkb, tolerance, &mut builder)?;
                    builder.append_value([]);
                }
                _ => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(
    geos_geom: &geos::Geometry,
    tolerance: f64,
    writer: &mut impl std::io::Write,
) -> Result<()> {
    let geometry = geos_geom
        .topology_preserve_simplify(tolerance)
        .map_err(|e| DataFusionError::Execution(format!("Failed to simplify geometry: {e}")))?;

    let wkb = geometry
        .to_wkb()
        .map_err(|e| DataFusionError::Execution(format!("Failed to convert to wkb: {e}")))?;

    writer.write_all(wkb.as_ref())?;
    Ok(())
}

#[cfg(test)]
mod tests {
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
        let udf = SedonaScalarUDF::from_kernel(
            "st_simplifypreservetopology",
            st_simplify_preserve_topology_impl(),
        );
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Float64)],
        );
        tester.assert_return_type(WKB_GEOMETRY);

        // Should remove the point (0 10)
        let result = tester
            .invoke_scalar_scalar("LINESTRING(0 0, 0 10, 0 51, 50 20, 30 20, 7 32)", 2.0)
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING(0 0,0 51,50 20,30 20,7 32)");

        // Short linestring should preserve endpoints
        let result = tester
            .invoke_scalar_scalar("LINESTRING(0 0,0 10)", 2.0)
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING(0 0,0 10)");

        // Null geometry
        let result = tester.invoke_scalar_scalar(ScalarValue::Null, 1.0).unwrap();
        assert!(result.is_null());

        // Null tolerance
        let result = tester
            .invoke_scalar_scalar("LINESTRING(0 0,0 10)", ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());

        // Array processing with tolerance 2.0
        let input_wkt_t2 = vec![
            Some("LINESTRING(0 0, 0 10, 0 51, 50 20, 30 20, 7 32)"),
            Some("POLYGON((0 0, 0 10, 0 11, 0 20, 10 20, 10 0, 0 0))"),
            Some("LINESTRING EMPTY"),
            Some("POINT EMPTY"),
            Some("POLYGON EMPTY"),
            None,
        ];

        let expected_t2 = create_array(
            &[
                Some("LINESTRING(0 0,0 51,50 20,30 20,7 32)"),
                Some("POLYGON((0 0,0 20,10 20,10 0,0 0))"),
                Some("LINESTRING EMPTY"),
                Some("POINT EMPTY"),
                Some("POLYGON EMPTY"),
                None,
            ],
            &WKB_GEOMETRY,
        );
        assert_array_equal(
            &tester.invoke_wkb_array_scalar(input_wkt_t2, 2.0).unwrap(),
            &expected_t2,
        );

        // Array processing with tolerance 20.0
        let input_wkt_t20 = vec![
            Some("LINESTRING(0 0,0 10)"),
            Some("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0),(5 5, 5 6, 6 6, 8 5, 5 5))"),
            Some("MULTIPOLYGON(((100 100, 100 130, 130 130, 130 100, 100 100)),((0 0, 10 0, 10 10, 0 10, 0 0),(5 5, 5 6, 6 6, 8 5, 5 5)))"),
        ];

        let expected_t20 = create_array(
            &[
                Some("LINESTRING(0 0,0 10)"),
                Some("POLYGON((0 0,10 0,10 10,0 10,0 0),(5 5,5 6,8 5,5 5))"),
                Some("MULTIPOLYGON(((100 100,100 130,130 130,130 100,100 100)),((0 0,10 0,10 10,0 10,0 0),(5 5,5 6,8 5,5 5)))"),
            ],
            &WKB_GEOMETRY,
        );
        assert_array_equal(
            &tester.invoke_wkb_array_scalar(input_wkt_t20, 20.0).unwrap(),
            &expected_t20,
        );

        // Test array tolerance input - same geometry with different tolerance values
        let input_wkt_array = vec![
            Some("LINESTRING (0 0, 0 10, 0 51, 50 20, 30 20, 7 32)"),
            Some("LINESTRING (0 0, 0 10, 0 51, 50 20, 30 20, 7 32)"),
            Some("LINESTRING (0 0, 0 10, 0 51, 50 20, 30 20, 7 32)"),
            None,
        ];

        use arrow_array::Float64Array;
        let tolerance_array: Arc<Float64Array> = Arc::new(Float64Array::from(vec![
            Some(2.0),
            Some(10.0),
            Some(50.0),
            Some(5.0),
        ]));

        let expected_array = create_array(
            &[
                Some("LINESTRING (0 0, 0 51, 50 20, 30 20, 7 32)"), // Tolerance 2.0
                Some("LINESTRING (0 0, 0 51, 50 20, 7 32)"),        // Tolerance 10.0
                Some("LINESTRING (0 0, 7 32)"),                     // Tolerance 50.0
                None,
            ],
            &WKB_GEOMETRY,
        );

        assert_array_equal(
            &tester
                .invoke_arrays(vec![
                    create_array(&input_wkt_array, &sedona_type),
                    tolerance_array,
                ])
                .unwrap(),
            &expected_array,
        );
    }
}
