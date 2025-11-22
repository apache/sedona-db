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

/// ST_Polygonize() scalar implementation using GEOS
pub fn st_polygonize_impl() -> ScalarKernelRef {
    Arc::new(STPolygonize {})
}

#[derive(Debug)]
struct STPolygonize {}

impl SedonaScalarKernel for STPolygonize {
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
    let result = geos::Geometry::polygonize(&[geos_geom])
        .map_err(|e| DataFusionError::Execution(format!("Failed to polygonize: {e}")))?;

    let wkb = result
        .to_wkb()
        .map_err(|e| DataFusionError::Execution(format!("Failed to convert result to WKB: {e}")))?;
    writer
        .write_all(wkb.as_ref())
        .map_err(|e| DataFusionError::Execution(format!("Failed to write result WKB: {e}")))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{SedonaType, WKB_VIEW_GEOMETRY};
    use sedona_testing::{
        compare::assert_array_equal, create::create_array, testers::ScalarUdfTester,
    };

    use super::*;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_polygonize", st_polygonize_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type.clone()]);
        tester.assert_return_type(WKB_GEOMETRY);

        let input_geometries = create_array(
            &[
                Some("LINESTRING(0 0, 0 1, 1 1, 1 0, 0 0)"),
                Some("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))"),
                Some("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 2 8, 8 8, 8 2, 2 2))"),
                Some("MULTILINESTRING((0 0, 0 1, 1 1, 1 0, 0 0), (10 10, 10 11, 11 11, 11 10, 10 10))"),
                Some("MULTILINESTRING((0 0, 10 0), (10 0, 10 10), (10 10, 0 0))"),
                Some("MULTIPOLYGON(((0 0, 1 0, 0 1, 0 0)), ((10 10, 11 10, 10 11, 10 10)))"),
                Some("GEOMETRYCOLLECTION(POINT(5 5), LINESTRING(0 0, 0 1, 1 1, 1 0, 0 0))"),
                Some("GEOMETRYCOLLECTION(GEOMETRYCOLLECTION(LINESTRING(0 0, 10 0)), LINESTRING(10 0, 10 10), LINESTRING(10 10, 0 0))"),
                Some("LINESTRING(0 0, 10 10)"),
                Some("POINT(0 0)"),
                Some("MULTIPOINT((0 0), (1 1))"),
                Some("LINESTRING EMPTY"),
                None,
            ],
            &WKB_GEOMETRY,
        );

        let expected_geometries = create_array(
            &[
                Some("GEOMETRYCOLLECTION(POLYGON((0 0, 0 1, 1 1, 1 0, 0 0)))"),
                Some("GEOMETRYCOLLECTION(POLYGON((0 0, 0 10, 10 10, 10 0, 0 0)))"),
                Some("GEOMETRYCOLLECTION(POLYGON((0 0,0 10,10 10,10 0,0 0),(2 2,8 2,8 8,2 8,2 2)),POLYGON((2 2,2 8,8 8,8 2,2 2)))"),
                Some("GEOMETRYCOLLECTION(POLYGON((0 0,0 1,1 1,1 0,0 0)),POLYGON((10 10,10 11,11 11,11 10,10 10)))"),
                Some("GEOMETRYCOLLECTION(POLYGON((10 0, 0 0, 10 10, 10 0)))"),
                Some("GEOMETRYCOLLECTION(POLYGON((0 0,0 1,1 0,0 0)),POLYGON((10 10,10 11,11 10,10 10)))"),
                Some("GEOMETRYCOLLECTION(POLYGON((0 0, 0 1, 1 1, 1 0, 0 0)))"),
                Some("GEOMETRYCOLLECTION(POLYGON((10 0, 0 0, 10 10, 10 0)))"),
                Some("GEOMETRYCOLLECTION EMPTY"),
                Some("GEOMETRYCOLLECTION EMPTY"),
                Some("GEOMETRYCOLLECTION EMPTY"),
                Some("GEOMETRYCOLLECTION EMPTY"),
                None,
            ],
            &WKB_GEOMETRY,
        );

        assert_array_equal(
            &tester.invoke_arrays(vec![input_geometries]).unwrap(),
            &expected_geometries,
        );
    }
}
