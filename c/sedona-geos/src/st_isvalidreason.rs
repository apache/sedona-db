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

use arrow_array::builder::StringBuilder;
use arrow_schema::DataType;
use datafusion_common::Result;
use geos::Geom;
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{ScalarKernelRef, SedonaScalarKernel},
};
use sedona_functions::executor::WkbExecutor;
use sedona_geometry::wkb_factory::WKB_MIN_PROBABLE_BYTES;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};
use wkb::reader::Wkb;

use crate::wkb_to_geos::GEOSWkbFactory;

/// ST_IsValidReason() implementation using the geos crate
pub fn st_is_valid_reason_impl() -> Vec<ScalarKernelRef> {
    ItemCrsKernel::wrap_impl(STIsValidReason {})
}

#[derive(Debug)]
struct STIsValidReason {}

impl SedonaScalarKernel for STIsValidReason {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry()],
            SedonaType::Arrow(DataType::Utf8),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[datafusion_expr::ColumnarValue],
    ) -> Result<datafusion_expr::ColumnarValue> {
        // Build the GEOS geometry per row (rather than with the GeosExecutor) so that a
        // geometry GEOS cannot build gets the GEOS error as its reason instead of failing
        // the batch. Malformed WKB still fails in the executor.
        let executor = WkbExecutor::new(arg_types, args);
        let factory = GEOSWkbFactory::new();
        let mut builder = StringBuilder::with_capacity(
            executor.num_iterations(),
            WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
        );
        executor.execute_wkb_void(|maybe_wkb| {
            match maybe_wkb {
                Some(wkb) => {
                    builder.append_value(invoke_scalar(&factory, wkb));
                }
                _ => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

/// A geometry GEOS cannot build or validate is not valid (see `ST_IsValid`). Its reason is
/// the GEOS error, e.g. "IllegalArgumentException: Segment vertex does not intersect ring".
fn invoke_scalar(factory: &GEOSWkbFactory, wkb: &Wkb) -> String {
    factory
        .create(wkb)
        .and_then(|geos_geom| geos_geom.is_valid_reason())
        .unwrap_or_else(|err| match err {
            // Some GEOS messages end with a newline
            geos::Error::GeosError((_, Some(message))) => message.trim_end().to_string(),
            err => err.to_string(),
        })
}

#[cfg(test)]
mod tests {
    use arrow_array::StringArray;
    use arrow_schema::DataType;
    use datafusion_common::ScalarValue;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS, WKB_VIEW_GEOMETRY};
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn udf(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY, WKB_GEOMETRY_ITEM_CRS.clone())]
        sedona_type: SedonaType,
    ) {
        use arrow_array::Array;

        let udf = SedonaScalarUDF::from_impl("st_isvalidreason", st_is_valid_reason_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type]);
        tester.assert_return_type(DataType::Utf8);

        // Test with a valid geometry
        let result = tester
            .invoke_scalar("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")
            .unwrap();
        tester.assert_scalar_result_equals(result, "Valid Geometry");

        // Test with an invalid geometry (self-intersection)
        let result = tester
            .invoke_scalar("POLYGON ((0 0, 1 1, 0 1, 1 0, 0 0))")
            .unwrap();
        if let ScalarValue::Utf8(Some(reason)) = result {
            assert!(reason.starts_with("Self-intersection"));
        } else {
            panic!("Expected a reason string for invalid geometry");
        }

        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        assert!(result.is_null());

        let input_wkt = vec![
            None,
            Some("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
            Some("POLYGON ((0 0, 1 1, 0 1, 1 0, 0 0))"),
            Some("LINESTRING (0 0, 1 1)"),
            Some("Polygon((0 0, 2 0, 1 1, 2 2, 0 2, 1 1, 0 0))"),
        ];

        let result_array = tester.invoke_wkb_array(input_wkt).unwrap();
        let result_array = result_array.as_any().downcast_ref::<StringArray>().unwrap();

        assert!(result_array.is_null(0));
        assert_eq!(result_array.value(1), "Valid Geometry");
        assert!(result_array.value(2).starts_with("Self-intersection"));
        assert_eq!(result_array.value(3), "Valid Geometry");
        assert!(result_array.value(4).starts_with("Ring Self-intersection"),);
    }

    #[rstest]
    fn udf_geos_error_is_reason(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY, WKB_GEOMETRY_ITEM_CRS.clone())]
        sedona_type: SedonaType,
    ) {
        use arrow_array::Array;

        let udf = SedonaScalarUDF::from_impl("st_isvalidreason", st_is_valid_reason_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type]);

        // The GEOS validator throws on this polygon: near f64::MAX its orientation test overflows
        let validator_error = "POLYGON ((-1.7e308 -10, 1.7e308 -10, 1.7e308 0, 1.7e308 10, -1.7e308 10, -1.7e308 -10), (-1e308 0, -5e307 5, -5e307 -5, -1e308 0))";
        let validator_error_reason =
            "IllegalArgumentException: Segment vertex does not intersect ring";

        for (wkt, reason) in [
            (validator_error, validator_error_reason),
            // GEOS refuses to build these: an unclosed ring, a two-point ring, a one-point line
            (
                "POLYGON ((0 0, 1 0, 1 1, 0 1))",
                "IllegalArgumentException: Points of LinearRing do not form a closed linestring",
            ),
            (
                "POLYGON ((0 0, 0 0))",
                "IllegalArgumentException: Invalid number of points in LinearRing found 2 - must be 0 or >= 3",
            ),
            (
                "LINESTRING (0 0)",
                "IllegalArgumentException: point array must contain 0 or >1 elements",
            ),
        ] {
            let result = tester.invoke_scalar(wkt).unwrap();
            tester.assert_scalar_result_equals(result, reason);
        }

        let input_wkt = vec![
            Some("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
            Some(validator_error),
            None,
            Some("POLYGON ((0 0, 1 1, 0 1, 1 0, 0 0))"),
        ];
        let result_array = tester.invoke_wkb_array(input_wkt).unwrap();
        let result_array = result_array.as_any().downcast_ref::<StringArray>().unwrap();

        assert_eq!(result_array.value(0), "Valid Geometry");
        assert_eq!(result_array.value(1), validator_error_reason);
        assert!(result_array.is_null(2));
        assert!(result_array.value(3).starts_with("Self-intersection"));
    }
}
