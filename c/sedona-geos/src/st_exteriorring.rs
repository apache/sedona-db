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

use arrow_array::builder::GenericBinaryBuilder;
use arrow_schema::DataType;
use datafusion_common::{error::Result, DataFusionError};
use datafusion_expr::ColumnarValue;
use geos::{GResult, Geom, GeometryTypes::Polygon};
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::executor::GeosExecutor;

pub fn st_exterior_ring_impl() -> ScalarKernelRef {
    Arc::new(STExteriorRing {})
}

#[derive(Debug)]
struct STExteriorRing {}

impl SedonaScalarKernel for STExteriorRing {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry()],
            SedonaType::Arrow(DataType::Binary),
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = GeosExecutor::new(arg_types, args);
        let mut builder = GenericBinaryBuilder::<i32>::with_capacity(
            executor.num_iterations(),
            executor.num_iterations() * 100,
        );

        executor.execute_wkb_void(|maybe_wkb| {
            match maybe_wkb {
                Some(wkb) => {
                    let result_wkb = invoke_scalar(&wkb).map_err(|e| {
                        DataFusionError::Execution(format!("Failed to get exterior ring: {e}"))
                    })?;

                    match result_wkb {
                        Some(val) => builder.append_value(val),
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

fn invoke_scalar(geos_geom: &geos::Geometry) -> GResult<Option<Vec<u8>>> {
    match geos_geom.geometry_type() {
        Polygon => {
            let ring = geos_geom.get_exterior_ring()?;
            Ok(Some(ring.to_wkb()?.into()))
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{builder::GenericBinaryBuilder, ArrayRef};
    use datafusion_common::ScalarValue;
    use geos::Geom;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    fn wkt_to_wkb(wkt: &str) -> Vec<u8> {
        geos::Geometry::new_from_wkt(wkt)
            .expect("Invalid WKT in test expectation")
            .to_wkb()
            .expect("Failed to convert to WKB")
            .into()
    }

    fn build_expected_geometry_array(wkts: Vec<Option<&str>>) -> ArrayRef {
        let mut builder = GenericBinaryBuilder::<i32>::new();
        for wkt in wkts {
            if let Some(w) = wkt {
                builder.append_value(wkt_to_wkb(w));
            } else {
                builder.append_null();
            }
        }
        Arc::new(builder.finish())
    }

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_exterior_ring", st_exterior_ring_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type]);

        tester.assert_return_type(DataType::Binary);

        let result = tester
            .invoke_scalar("POLYGON((0 0, 1 0, 1 1, 0 0))")
            .unwrap();
        match result {
            ScalarValue::Binary(Some(val)) => {
                assert_eq!(val, wkt_to_wkb("LINESTRING(0 0, 1 0, 1 1, 0 0)"));
            }
            _ => panic!("Expected Binary ScalarValue with value, got {:?}", result),
        }

        let result = tester.invoke_scalar("LINESTRING(0 0, 1 0, 1 1)").unwrap();
        assert!(result.is_null());

        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        assert!(result.is_null());

        let input_wkt = vec![
            Some("POINT(1 2)"),
            None,
            Some("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))"),
            Some("LINESTRING (0 0, 1 0, 0 1)"),
        ];

        let expected_wkt = vec![
            None,
            None,
            Some("LINESTRING(0 0, 10 0, 10 10, 0 10, 0 0)"),
            None,
        ];

        let expected_array = build_expected_geometry_array(expected_wkt);
        let result_array = tester.invoke_wkb_array(input_wkt).unwrap();

        assert_eq!(&result_array, &expected_array);
    }
}
