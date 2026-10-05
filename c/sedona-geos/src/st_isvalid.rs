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
use datafusion_common::Result;
use datafusion_expr::ColumnarValue;
use geos::Geom;
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{ScalarKernelRef, SedonaScalarKernel},
};
use sedona_functions::executor::WkbExecutor;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};
use wkb::reader::Wkb;

use crate::wkb_to_geos::GEOSWkbFactory;

/// ST_IsValid() implementation using the geos crate
pub fn st_is_valid_impl() -> Vec<ScalarKernelRef> {
    ItemCrsKernel::wrap_impl(STIsValid {})
}

#[derive(Debug)]
struct STIsValid {}

impl SedonaScalarKernel for STIsValid {
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
        // Build the GEOS geometry per row (rather than with the GeosExecutor) so that a
        // geometry GEOS cannot build is reported as invalid instead of failing the batch.
        // Malformed WKB still fails in the executor.
        let executor = WkbExecutor::new(arg_types, args);
        let factory = GEOSWkbFactory::new();
        let mut builder = BooleanBuilder::with_capacity(executor.num_iterations());
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

/// GEOS reports some invalid geometries as errors rather than as `false`: it refuses to
/// build rings that are not closed or have too few points, and its validator can throw
/// (e.g., when coordinates near `f64::MAX` overflow its orientation test). A geometry that
/// cannot be built or validated is not valid.
fn invoke_scalar(factory: &GEOSWkbFactory, wkb: &Wkb) -> bool {
    factory
        .create(wkb)
        .and_then(|geos_geom| geos_geom.is_valid())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, BooleanArray};
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
        let udf = SedonaScalarUDF::from_impl("st_isvalid", st_is_valid_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type]);
        tester.assert_return_type(DataType::Boolean);

        // Valid polygon
        let result = tester
            .invoke_scalar("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")
            .unwrap();
        tester.assert_scalar_result_equals(result, true);

        // Invalid polygon (self-intersecting)
        let result = tester
            .invoke_scalar("POLYGON ((0 0, 1 1, 0 1, 1 0, 0 0))")
            .unwrap();
        tester.assert_scalar_result_equals(result, false);

        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        assert!(result.is_null());

        let input_wkt = vec![
            None,
            Some("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
            Some("POLYGON ((0 0, 1 1, 0 1, 1 0, 0 0))"),
            Some("LINESTRING (0 0, 1 1)"),
            Some("Polygon((0 0, 2 0, 1 1, 2 2, 0 2, 1 1, 0 0))"),
        ];

        let expected: ArrayRef = Arc::new(BooleanArray::from(vec![
            None,
            Some(true),
            Some(false),
            Some(true),
            Some(false),
        ]));
        assert_eq!(&tester.invoke_wkb_array(input_wkt).unwrap(), &expected);
    }

    #[rstest]
    fn udf_geos_error_is_invalid(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY, WKB_GEOMETRY_ITEM_CRS.clone())]
        sedona_type: SedonaType,
    ) {
        let udf = SedonaScalarUDF::from_impl("st_isvalid", st_is_valid_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type]);

        // The GEOS validator throws on this polygon: near f64::MAX its orientation test overflows
        let validator_error = "POLYGON ((-1.7e308 -10, 1.7e308 -10, 1.7e308 0, 1.7e308 10, -1.7e308 10, -1.7e308 -10), (-1e308 0, -5e307 5, -5e307 -5, -1e308 0))";
        // GEOS refuses to build these: an unclosed ring, a two-point ring, a one-point line
        let unclosed_ring = "POLYGON ((0 0, 1 0, 1 1, 0 1))";
        let two_point_ring = "POLYGON ((0 0, 0 0))";
        let one_point_line = "LINESTRING (0 0)";

        for wkt in [
            validator_error,
            unclosed_ring,
            two_point_ring,
            one_point_line,
        ] {
            let result = tester.invoke_scalar(wkt).unwrap();
            tester.assert_scalar_result_equals(result, false);
        }

        let input_wkt = vec![
            Some("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
            Some(validator_error),
            None,
            Some("POLYGON ((0 0, 1 1, 0 1, 1 0, 0 0))"),
            Some(unclosed_ring),
            Some(two_point_ring),
            Some(one_point_line),
            Some("POLYGON EMPTY"),
        ];
        let expected: ArrayRef = Arc::new(BooleanArray::from(vec![
            Some(true),
            Some(false),
            None,
            Some(false),
            Some(false),
            Some(false),
            Some(false),
            Some(true),
        ]));
        assert_eq!(&tester.invoke_wkb_array(input_wkt).unwrap(), &expected);
    }
}
