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

/// ST_MinimumClearanceLine() implementation using the geos crate
pub fn st_minimum_clearance_line_impl() -> ScalarKernelRef {
    Arc::new(STMinimumClearanceLine {})
}

#[derive(Debug)]
struct STMinimumClearanceLine {}

impl SedonaScalarKernel for STMinimumClearanceLine {
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
    let geometry = geos_geom.minimum_clearance_line().map_err(|e| {
        DataFusionError::Execution(format!(
            "Failed to calculate geometry's minimum clearance line: {e}"
        ))
    })?;

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
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel(
            "st_minimumclearanceline",
            st_minimum_clearance_line_impl(),
        );
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type.clone()]);
        tester.assert_return_type(WKB_GEOMETRY);

        let input_wkt = vec![
            None,
            Some("POLYGON ((0 0, 1 0, 1 1, 0.5 3.2e-4, 0 0))"),
            Some("MULTIPOLYGON(((26 125, 26 200, 126 200, 126 125, 26 125 ),( 51 150, 101 150, 76 175, 51 150 )),(( 151 100, 151 200, 176 175, 151 100 )))"),
            Some("LINESTRING (5 107, 54 84, 101 100)"),
            Some("POLYGON((0 0,0 3,3 3,3 0,0 0),(1 1,1 2,2 2,2 1,1 1))"),
            Some("POLYGON((0 0,0 1,0 1,1 1,1 0,0 0,0 0))"),
            Some("LINESTRING (0 0, 1 1, 2 2)"),
            Some("MULTIPOLYGON(((0.5 0.5,0 0,0 1,0.5 0.5)),((0.5 0.5,1 1,1 0,0.5 0.5)),((2.5 2.5,2 2,2 3,2.5 2.5)),((2.5 2.5,3 3,3 2,2.5 2.5)))"),
            Some("POINT (1 1)"),
            Some("GEOMETRYCOLLECTION(POINT(1 1),MULTIPOLYGON(((0 2,1 1,0 0,0 2)),((2 0,1 1,2 2,2 0))))"),
            Some("POLYGON EMPTY"),
            Some("POLYGON((0 0,3 0,3 3,2 1,1 3,0 3,0 0))"),
        ];
        let expected = create_array(
            &[
                None,
                Some("LINESTRING(0.5 0.00032,0.5 0)"),
                Some("LINESTRING(76 175,76 150)"),
                Some("LINESTRING(54 84,101 100)"),
                Some("LINESTRING(1 1,1 2)"),
                Some("LINESTRING(0 0,0 1)"),
                Some("LINESTRING(0 0,1 1)"),
                Some("LINESTRING(2.5 2.5,3 2.5)"),
                Some("LINESTRING EMPTY"),
                Some("LINESTRING(1 1,2 1)"),
                Some("LINESTRING EMPTY"),
                Some("LINESTRING(1 3,0 3)"),
            ],
            &WKB_GEOMETRY,
        );

        assert_eq!(&tester.invoke_wkb_array(input_wkt).unwrap(), &expected);
    }
}
