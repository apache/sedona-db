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
use datafusion_common::{error::Result, DataFusionError};
use datafusion_expr::ColumnarValue;
use geos::{Geom, GeometryTypes};
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::executor::GeosExecutor;

/// ST_IsRing() implementation using the geos crate
pub fn st_is_ring_impl() -> ScalarKernelRef {
    Arc::new(STIsRing {})
}

#[derive(Debug)]
struct STIsRing {}

impl SedonaScalarKernel for STIsRing {
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
    // Check if geometry is empty - (PostGIS compatibility)
    let is_empty = geos_geom.is_empty().map_err(|e| {
        DataFusionError::Execution(format!("Failed to check if geometry is a ring: {e}"))
    })?;

    if is_empty {
        return Ok(false);
    }

    // Check if geometry is a LineString - (PostGIS compatibility)
    if geos_geom.geometry_type() != GeometryTypes::LineString {
        return Err(DataFusionError::Execution(
            "ST_IsRing() should only be called on a linear feature".to_string(),
        ));
    }

    geos_geom.is_ring().map_err(|e| {
        DataFusionError::Execution(format!("Failed to check if geometry is a ring: {e}"))
    })
}

#[cfg(test)]
mod tests {
    use arrow_array::{create_array as arrow_array, ArrayRef};
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_isring", st_is_ring_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type]);
        tester.assert_return_type(DataType::Boolean);

        // Valid ring (closed + simple) - square
        let result = tester
            .invoke_scalar("LINESTRING(0 0, 0 1, 1 1, 1 0, 0 0)")
            .unwrap();
        tester.assert_scalar_result_equals(result, true);

        // Valid ring (closed + simple) - triangle
        let result = tester
            .invoke_scalar("LINESTRING(0 0, 1 0, 1 1, 0 0)")
            .unwrap();
        tester.assert_scalar_result_equals(result, true);

        // Non-LineString types should throw errors (PostGIS compatibility)

        // Point (not a linestring) - should error
        let result = tester.invoke_scalar("POINT(21 52)");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("should only be called on a linear feature"));

        // Polygon (not a linestring) - should error
        let result = tester.invoke_scalar("POLYGON((0 0, 0 5, 5 5, 5 0, 0 0))");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("should only be called on a linear feature"));

        // MultiLineString (collection) - should error
        let result = tester.invoke_scalar("MULTILINESTRING((0 0, 0 1, 1 1, 1 0, 0 0))");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("should only be called on a linear feature"));

        // GeometryCollection - should error
        let result =
            tester.invoke_scalar("GEOMETRYCOLLECTION(LINESTRING(0 0, 0 1, 1 1, 1 0, 0 0))");
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("should only be called on a linear feature"));

        let input_wkt = vec![
            Some("LINESTRING(0 0, 0 1, 1 1, 1 0, 0 0)"), // Valid ring => true
            Some("LINESTRING(0 0, 0 1, 1 0, 1 1, 0 0)"), // Self-intersecting => false
            Some("LINESTRING(0 0, 2 2)"),                // Not closed => false
            Some("LINESTRING EMPTY"),                    // Empty => false
            Some("POINT EMPTY"),                         // Empty => false
            None,                                        // NULL => null
        ];

        let expected: ArrayRef = arrow_array!(
            Boolean,
            [
                Some(true),
                Some(false),
                Some(false),
                Some(false),
                Some(false),
                None
            ]
        );

        assert_array_equal(&tester.invoke_wkb_array(input_wkt).unwrap(), &expected);
    }
}
