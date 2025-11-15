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
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::ColumnarValue;
use geos::Geom;
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_geometry::wkb_factory::WKB_MIN_PROBABLE_BYTES;
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

use crate::executor::GeosExecutor;

/// ST_MakeValid() implementation using the geos crate
pub fn st_make_valid_impl() -> ScalarKernelRef {
    Arc::new(STMakeValid {})
}

#[derive(Debug)]
struct STMakeValid {}

impl SedonaScalarKernel for STMakeValid {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOMETRY);

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = GeosExecutor::new(arg_types, args);
        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
        );

        executor.execute_wkb_void(|maybe_wkb| {
            match maybe_wkb {
                Some(wkb) => {
                    invoke_scalar(&wkb, &mut builder)?;
                    builder.append_value([]);
                }
                _ => builder.append_null(),
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(geos_geom: &geos::Geometry, writer: &mut impl std::io::Write) -> Result<()> {
    let geometry = geos_geom
        .make_valid()
        .map_err(|e| DataFusionError::Execution(format!("Failed to make geometry valid: {e}")))?;

    let wkb = geometry
        .to_wkb()
        .map_err(|e| DataFusionError::Execution(format!("Failed to convert to wkb: {e}")))?;

    writer.write_all(wkb.as_ref())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::WKB_VIEW_GEOMETRY;
    use sedona_testing::{create::create_array, testers::ScalarUdfTester};

    use super::*;

    #[rstest]
    fn st_make_valid_udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_makevalid", st_make_valid_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type.clone()]);
        tester.assert_return_type(WKB_GEOMETRY);

        let input_wkt = vec![
            Some("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"), // Already valid polygon should remain unchanged
            Some("POLYGON ((0 0, 2 2, 2 0, 0 2, 0 0))"), // Self-intersecting polygon (bowtie) should be fixed
            Some("POLYGON ((0 0, 0 3, 3 3, 3 0, 0 0), (1 1, 1 2, 2 2, 2 1, 1 1))"), // Polygon with incorrect ring orientation should be fixed
            Some("POLYGON ((0 0, 0 1, 0 1, 1 1, 1 0, 0 0, 0 0))"), //Polygon with repeated points should be cleaned
            Some("LINESTRING (0 0, 1 1, 2 2)"), // LineString that is already valid
            Some("LINESTRING (0 0, 0 0, 1 1, 1 1, 2 2)"), // LineString with repeated points should be simplified
            Some("MULTIPOLYGON (((0 0, 1 1, 1 0, 0 1, 0 0)), ((2 2, 3 3, 3 2, 2 3, 2 2)))"), // MultiPolygon with invalid components
            Some("POINT (1 1)"), // Point geometry (always valid)
            Some("GEOMETRYCOLLECTION (POINT (1 1), POLYGON ((0 0, 2 2, 2 0, 0 2, 0 0)))"), // GeometryCollection with mixed valid/invalid geometries
            Some("POLYGON EMPTY"), // Empty geometry
            Some("POLYGON ((0 0, 3 0, 3 3, 2 1, 1 3, 0 3, 0 0))"), // Polygon with spike (almost self-intersecting)
        ];

        let expected = create_array(&[
            Some("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
            Some("MULTIPOLYGON(((0 2,1 1,0 0,0 2)),((2 0,1 1,2 2,2 0)))"),
            Some("POLYGON((0 0,0 3,3 3,3 0,0 0),(1 1,1 2,2 2,2 1,1 1))"),
            Some("POLYGON((0 0,0 1,0 1,1 1,1 0,0 0,0 0))"),
            Some("LINESTRING (0 0, 1 1, 2 2)"),
            Some("LINESTRING(0 0,0 0,1 1,1 1,2 2)"),
            Some("MULTIPOLYGON(((0.5 0.5,0 0,0 1,0.5 0.5)),((0.5 0.5,1 1,1 0,0.5 0.5)),((2.5 2.5,2 2,2 3,2.5 2.5)),((2.5 2.5,3 3,3 2,2.5 2.5)))"),
            Some("POINT (1 1)"),
            Some("GEOMETRYCOLLECTION(POINT(1 1),MULTIPOLYGON(((0 2,1 1,0 0,0 2)),((2 0,1 1,2 2,2 0))))"),
            Some("POLYGON EMPTY"),
            Some("POLYGON((0 0,3 0,3 3,2 1,1 3,0 3,0 0))"),
        ], &WKB_GEOMETRY);

        assert_eq!(&tester.invoke_wkb_array(input_wkt).unwrap(), &expected);
    }

    #[rstest]
    fn st_make_valid_edge_cases(#[values(WKB_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_makevalid", st_make_valid_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type.clone()]);

        // Test with very close points (floating point precision issues)
        let result = tester
            .invoke_scalar("POLYGON ((0 0, 0 1, 1 1, 1 0, 0.0000000001 0.0000000001, 0 0))")
            .unwrap();
        tester.assert_scalar_result_equals(result, "POLYGON((0 0,0 1,1 1,1 0,1e-10 1e-10,0 0))");

        // Test with degenerate polygon (collinear points)
        let result = tester
            .invoke_scalar("POLYGON ((0 0, 1 1, 2 2, 3 3, 0 0))")
            .unwrap();
        tester
            .assert_scalar_result_equals(result, "MULTILINESTRING((0 0,1 1),(1 1,2 2),(2 2,3 3))");

        // Test with polygon that has a tiny hole
        let result = tester
            .invoke_scalar("POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0), (5 5, 5 5.0001, 5.0001 5.0001, 5.0001 5, 5 5))")
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            "POLYGON((0 0,0 10,10 10,10 0,0 0),(5 5,5 5.0001,5.0001 5.0001,5.0001 5,5 5))",
        );
    }
}
