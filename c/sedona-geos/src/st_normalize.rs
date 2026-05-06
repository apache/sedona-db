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
use datafusion_common::{error::Result, DataFusionError};
use datafusion_expr::ColumnarValue;
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{ScalarKernelRef, SedonaScalarKernel},
};
use sedona_geometry::wkb_factory::WKB_MIN_PROBABLE_BYTES;
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

use crate::executor::GeosExecutor;
use crate::geos_to_wkb::write_geos_geometry;

/// ST_Normalize() implementation using the geos crate
pub fn st_normalize_impl() -> Vec<ScalarKernelRef> {
    ItemCrsKernel::wrap_impl(STNormalize {})
}

#[derive(Debug)]
struct STNormalize {}

impl SedonaScalarKernel for STNormalize {
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
    let mut geometry = Clone::clone(geos_geom);
    geometry
        .normalize()
        .map_err(|e| DataFusionError::Execution(format!("Failed to normalize geometry: {e}")))?;

    write_geos_geometry(&geometry, writer)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use datafusion_common::ScalarValue;
    use geos::{Geom, Geometry};
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS, WKB_VIEW_GEOMETRY};
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_impl("st_normalize", st_normalize_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type.clone()]);

        tester.assert_return_type(WKB_GEOMETRY);

        let input_wkt = "POLYGON((1 1, 1 0, 0 0, 0 1, 1 1))";
        let expected_wkt = "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))";

        let result = tester.invoke_scalar(input_wkt).unwrap();
        tester.assert_scalar_result_equals(result, expected_wkt);

        let result = tester
            .invoke_scalar("MULTILINESTRING ((2 2, 1 1), (4 4, 3 3))")
            .unwrap();
        tester.assert_scalar_result_equals(result, "MULTILINESTRING ((3 3, 4 4), (1 1, 2 2))");

        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        assert!(result.is_null());

        let batch_input = vec![
            Some("POINT(2 1)"),
            None,
            Some("POLYGON((1 1, 1 0, 0 0, 0 1, 1 1))"),
            Some("MULTILINESTRING ((2 2, 1 1), (4 4, 3 3))"),
        ];

        let batch_result = tester.invoke_wkb_array(batch_input).unwrap();
        assert_eq!(batch_result.len(), 4);

        assert!(batch_result.is_null(1));

        let expected = sedona_testing::create::create_array(
            &[
                Some("POINT (2 1)"),
                None,
                Some("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
                Some("MULTILINESTRING ((3 3, 4 4), (1 1, 2 2))"),
            ],
            &WKB_GEOMETRY,
        );
        sedona_testing::compare::assert_array_equal(&batch_result, &expected);
    }

    #[rstest]
    fn udf_invoke_item_crs(#[values(WKB_GEOMETRY_ITEM_CRS.clone())] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_impl("st_normalize", st_normalize_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type.clone()]);

        tester.assert_return_type(sedona_type);

        let result = tester
            .invoke_scalar("POLYGON((1 1, 1 0, 0 0, 0 1, 1 1))")
            .unwrap();
        tester.assert_scalar_result_equals(result, "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))");
    }

    #[rstest]
    fn udf_already_normalized(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_impl("st_normalize", st_normalize_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type]);
        let already_normal = "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))";
        let result = tester.invoke_scalar(already_normal).unwrap();
        tester.assert_scalar_result_equals(result, already_normal);
    }

    #[test]
    fn invoke_scalar_normalizes_via_geos_without_mutating_input() {
        let geom = Geometry::new_from_wkt("POLYGON((1 1, 1 0, 0 0, 0 1, 1 1))").unwrap();
        let original_wkt = geom.to_wkt().unwrap();

        let mut normalized_wkb = Vec::new();
        invoke_scalar(&geom, &mut normalized_wkb).unwrap();

        let normalized = Geometry::new_from_wkb(&normalized_wkb).unwrap();
        assert_eq!(
            normalized.to_wkt().unwrap(),
            "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"
        );

        assert_eq!(geom.to_wkt().unwrap(), original_wkt);
    }
}
