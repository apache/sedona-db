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
use datafusion_common::error::Result;
use datafusion_expr::ColumnarValue;
use geo_generic_alg::Centroid;
use sedona_expr::scalar_udf::{ArgMatcher, ScalarKernelRef, SedonaScalarKernel};
use sedona_functions::executor::WkbExecutor;
use sedona_functions::st_isempty::is_wkb_empty;
use sedona_schema::datatypes::{SedonaType, WKB_GEOMETRY};
use wkb::reader::Wkb;

use sedona_geometry::wkb_factory;
use sedona_testing::fixtures::POINT_EMPTY_WKB;

/// ST_Centroid() implementation using centroid extraction
pub fn st_centroid_impl() -> ScalarKernelRef {
    Arc::new(STCentroid {})
}

#[derive(Debug)]
struct STCentroid {}

impl SedonaScalarKernel for STCentroid {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOMETRY);

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = BinaryBuilder::with_capacity(executor.num_iterations(), 1024);
        executor.execute_wkb_void(|maybe_wkb| {
            match maybe_wkb {
                Some(wkb) => {
                    let centroid_wkb = invoke_scalar(&wkb)?;
                    builder.append_value(&centroid_wkb);
                }
                _ => builder.append_null(),
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(wkb: &Wkb) -> Result<Vec<u8>> {
    // Check for empty geometries first - they should return POINT EMPTY
    if is_wkb_empty(wkb)? {
        return Ok(POINT_EMPTY_WKB.to_vec());
    }

    // Use Centroid trait directly on WKB, similar to how st_area uses unsigned_area()
    if let Some(centroid_point) = wkb.centroid() {
        // Extract coordinates from the centroid point
        let x = centroid_point.x();
        let y = centroid_point.y();

        wkb_factory::wkb_point((x, y))
            .map_err(|e| datafusion_common::error::DataFusionError::External(Box::new(e)))
    } else {
        // This should not happen for non-empty geometries, return POINT EMPTY as fallback
        Ok(POINT_EMPTY_WKB.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{Array, BinaryArray};
    use datafusion_common::scalar::ScalarValue;
    use rstest::rstest;
    use sedona_functions::register::stubs::st_centroid_udf;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::testers::ScalarUdfTester;
    use wkb::reader::Wkb;

    use super::*;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let mut udf = st_centroid_udf();
        udf.add_kernel(st_centroid_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type]);

        assert_eq!(tester.return_type().unwrap(), WKB_GEOMETRY);

        // Test with a polygon
        let result = tester
            .invoke_wkb_scalar(Some("POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))"))
            .unwrap();

        if let ScalarValue::Binary(Some(wkb_data)) = result {
            let wkb = Wkb::try_new(&wkb_data).unwrap();
            if let Some(centroid_point) = wkb.centroid() {
                assert_eq!(centroid_point.x(), 1.0);
                assert_eq!(centroid_point.y(), 1.0);
            } else {
                panic!("Expected centroid point");
            }
        } else {
            panic!("Expected Binary result");
        }

        // Test with array
        let input_wkt = vec![
            Some("POINT(1 2)"),
            None,
            Some("POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))"),
        ];
        let result_array = tester.invoke_wkb_array(input_wkt).unwrap();
        let binary_array = result_array.as_any().downcast_ref::<BinaryArray>().unwrap();

        // First element: POINT(1 2) - centroid should be (1, 2)
        assert!(!binary_array.value(0).is_empty());
        // Second element: NULL
        assert!(binary_array.is_null(1));
        // Third element: POLYGON centroid should be (1, 1)
        assert!(!binary_array.value(2).is_empty());
    }
}
