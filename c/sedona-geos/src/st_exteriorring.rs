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

use crate::executor::GeosExecutor;
use arrow_array::builder::BinaryBuilder;
use datafusion_common::{error::Result, DataFusionError};
use datafusion_expr::ColumnarValue;
use geos::{Geom, GeometryTypes::Polygon};
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};
use std::sync::Arc;

pub fn st_exterior_ring_impl() -> ScalarKernelRef {
    Arc::new(STExteriorRing {})
}

#[derive(Debug)]
struct STExteriorRing {}

impl SedonaScalarKernel for STExteriorRing {
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
            100 * executor.num_iterations(),
        );

        executor.execute_wkb_void(|maybe_wkb| {
            match maybe_wkb {
                Some(wkb) => {
                    invoke_scalar(&wkb, &mut builder)?;
                }
                _ => builder.append_null(),
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(geos_geom: &geos::Geometry, builder: &mut BinaryBuilder) -> Result<()> {
    match geos_geom.geometry_type() {
        Polygon => {
            let ring = geos_geom.get_exterior_ring().map_err(|e| {
                DataFusionError::Execution(format!("Failed to get exterior ring: {e}"))
            })?;

            let wkb = ring.to_wkb().map_err(|e| {
                DataFusionError::Execution(format!("Failed to convert to wkb: {e}"))
            })?;

            builder.append_value(wkb);
        }
        _ => builder.append_null(),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_common::ScalarValue;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::create::create_array;
    use sedona_testing::testers::ScalarUdfTester;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_exterior_ring", st_exterior_ring_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type]);

        tester.assert_return_type(WKB_GEOMETRY);

        let result = tester
            .invoke_scalar("POLYGON((0 0, 1 0, 1 1, 0 0))")
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING (0 0, 1 0, 1 1, 0 0)");

        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        assert!(result.is_null());

        let input_wkt = vec![
            Some("POINT(1 2)"),
            None,
            Some("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))"),
            Some("LINESTRING (0 0, 1 0, 0 1)"),
        ];

        let expected = create_array(
            &[
                None,
                None,
                Some("LINESTRING (0 0, 10 0, 10 10, 0 10, 0 0)"),
                None,
            ],
            &WKB_GEOMETRY,
        );

        assert_array_equal(&tester.invoke_wkb_array(input_wkt).unwrap(), &expected);
    }
}
