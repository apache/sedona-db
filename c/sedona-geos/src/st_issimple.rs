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
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::ColumnarValue;
use geos::Geom;
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::executor::GeosExecutor;

/// ST_IsSimple() implementation using the geos crate
pub fn st_is_simple_impl() -> ScalarKernelRef {
    Arc::new(STIsSimple {})
}

#[derive(Debug)]
struct STIsSimple {}

impl SedonaScalarKernel for STIsSimple {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry()],
            SedonaType::Arrow(DataType::Boolean),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = GeosExecutor::new(arg_types, args);
        let mut builder = BooleanBuilder::with_capacity(executor.num_iterations());
        executor.execute_wkb_void(|maybe_wkb| {
            match maybe_wkb {
                Some(wkb) => {
                    builder.append_value(invoke_scalar(&wkb)?);
                }
                _ => builder.append_null(),
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(geos_geom: &geos::Geometry) -> Result<bool> {
    geos_geom.is_simple().map_err(|e| {
        DataFusionError::Execution(format!("Failed to check if geometry is simple: {e}"))
    })
}

#[cfg(test)]
mod tests {

    use arrow_array::{ArrayRef, BooleanArray};
    use datafusion_common::ScalarValue;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_issimple", st_is_simple_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type]);
        tester.assert_return_type(DataType::Boolean);

        // Simple Polygon
        let result = tester
            .invoke_scalar("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")
            .unwrap();
        tester.assert_scalar_result_equals(result, true);

        // Complex Polygon (self-intersecting)
        let result = tester
            .invoke_scalar("POLYGON ((0 0, 1 1, 0 1, 1 0, 0 0))")
            .unwrap();
        tester.assert_scalar_result_equals(result, false);

        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        assert!(result.is_null());

        let input_wkt = vec![
            None,                                                                            // Null
            Some("POINT (1 1)"),                // Points are always simple (T)
            Some("MULTIPOINT (1 1, 2 2, 3 3)"), // Points are always simple (T)
            Some("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"), // Simple Polygon (T)
            Some("POLYGON ((0 0, 1 1, 0 1, 1 0, 0 0))"), // Complex Polygon (F)
            Some("POLYGON((1 2, 3 4, 5 6, 1 2))"), // POSTGIS Reference (F)
            Some("LINESTRING (0 0, 1 1)"),      // Simple LineString (T)
            Some("LINESTRING (0 0, 1 1, 0 1, 1 0)"), // Complex LineString (F)
            Some("LINESTRING(1 1,2 2,2 3.5,1 3,1 2,2 1)"), // POSTGIS Reference (F)
            Some("MULTILINESTRING ((0 0, 1 1), (1 1, 2 2))"), // Simple MultiLineString (T)
            Some("MULTILINESTRING ((0 0, 2 2), (0 2, 2 0))"), // Complex MultiLineString (F)
            Some("POINT (10 10)"),              // Point (T)
            Some("GEOMETRYCOLLECTION EMPTY"),   // Empty (T)
            Some("Polygon((0 0, 2 0, 1 1, 2 2, 0 2, 1 1, 0 0))"), // Complex Polygon (F)
            Some("MULTIPOLYGON (((0 0, 0 1, 1 1, 1 0, 0 0)), ((2 2, 2 3, 3 3, 3 2, 2 2)))"), // Holes are fine (T)
            Some("POLYGON ((0 0, 3 0, 3 3, 0 3, 0 0), (1 1, 0 2, 2 2, 1 1))"), // Holes are fine (T)
        ];

        let expected: ArrayRef = Arc::new(BooleanArray::from(vec![
            None,
            Some(true),
            Some(true),
            Some(true),
            Some(false),
            Some(false),
            Some(true),
            Some(false),
            Some(false),
            Some(true),
            Some(false),
            Some(true),
            Some(true),
            Some(false),
            Some(true),
            Some(true),
        ]));
        assert_eq!(&tester.invoke_wkb_array(input_wkt).unwrap(), &expected);
    }
}
