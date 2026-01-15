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
use datafusion_common::{error::Result, DataFusionError, ScalarValue};
use datafusion_expr::ColumnarValue;
use geos::Geom;
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{ScalarKernelRef, SedonaScalarKernel},
};
use sedona_geometry::wkb_factory::WKB_MIN_PROBABLE_BYTES;
use sedona_schema::{datatypes::WKB_GEOMETRY, matchers::ArgMatcher};

use crate::executor::GeosExecutor;
use crate::geos_to_wkb::write_geos_geometry;

pub fn st_line_merge_impl() -> Vec<ScalarKernelRef> {
    ItemCrsKernel::wrap_impl(STLineMerge {})
}

#[derive(Debug)]
struct STLineMerge {}

impl SedonaScalarKernel for STLineMerge {
    fn return_type(
        &self,
        args: &[sedona_schema::datatypes::SedonaType],
    ) -> datafusion_common::Result<Option<sedona_schema::datatypes::SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_geometry(),
                ArgMatcher::optional(ArgMatcher::is_boolean()),
            ],
            WKB_GEOMETRY,
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[sedona_schema::datatypes::SedonaType],
        args: &[datafusion_expr::ColumnarValue],
    ) -> datafusion_common::Result<datafusion_expr::ColumnarValue> {
        let executor = GeosExecutor::new(arg_types, args);
        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
        );

        let directed = match args.get(1) {
            Some(ColumnarValue::Scalar(ScalarValue::Boolean(Some(opt_bool)))) => *opt_bool,
            _ => false,
        };

        executor.execute_wkb_void(|maybe_wkb| {
            match maybe_wkb {
                Some(wkb) => {
                    invoke_scalar(&wkb, &mut builder, directed)?;
                    builder.append_value([]);
                }
                None => builder.append_null(),
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(
    geos_geom: &geos::Geometry,
    writer: &mut impl std::io::Write,
    directed: bool,
) -> Result<()> {
    // PostGIS seems to return the original geometry if it is empty
    let is_empty = geos_geom.is_empty().map_err(|e| {
        DataFusionError::Execution(format!("Failed to check if the geometry is empty: {e}"))
    })?;
    if is_empty {
        write_geos_geometry(geos_geom, writer)?;
        return Ok(());
    }

    let result = if directed {
        geos_geom.line_merge_directed()
    } else {
        geos_geom.line_merge()
    };

    let geom =
        result.map_err(|e| DataFusionError::Execution(format!("Failed to merge lines: {e}")))?;

    write_geos_geometry(&geom, writer)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use arrow_array::ArrayRef;
    use datafusion_common::ScalarValue;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{
        SedonaType, WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS, WKB_VIEW_GEOMETRY,
    };
    use sedona_testing::create::create_array;
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        use arrow_schema::DataType;

        let udf = SedonaScalarUDF::from_impl("st_linemerge", st_line_merge_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![sedona_type, SedonaType::Arrow(DataType::Boolean)],
        );
        tester.assert_return_type(WKB_GEOMETRY);

        let input = vec![
            Some("MULTILINESTRING ((0 0, 1 0), (1 0, 1 1))"),
            Some("MULTILINESTRING ((0 0, 1 0), (1 1, 1 0))"), // opposite direction
            Some("MULTILINESTRING ((0 0, 1 0), (8 8, 9 9))"), // doesn't touch
        ];

        let expected: ArrayRef = create_array(
            &[
                Some("LINESTRING (0 0, 1 0, 1 1)"),
                Some("LINESTRING (0 0, 1 0, 1 1)"),
                Some("MULTILINESTRING ((0 0, 1 0), (8 8, 9 9))"),
            ],
            &WKB_GEOMETRY,
        );

        assert_eq!(
            &tester
                .invoke_wkb_array_scalar(input.clone(), false)
                .unwrap(),
            &expected
        );

        // If directed is true, lines with opposite directions won't be merged

        let expected_directed: ArrayRef = create_array(
            &[
                Some("LINESTRING (0 0, 1 0, 1 1)"),
                Some("MULTILINESTRING ((0 0, 1 0), (1 1, 1 0))"),
                Some("MULTILINESTRING ((0 0, 1 0), (8 8, 9 9))"),
            ],
            &WKB_GEOMETRY,
        );

        assert_eq!(
            &tester.invoke_wkb_array_scalar(input, true).unwrap(),
            &expected_directed
        );

        // handle NULL

        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, false)
            .unwrap();
        assert!(result.is_null());
    }

    #[rstest]
    fn udf_invoke_item_crs(#[values(WKB_GEOMETRY_ITEM_CRS.clone())] sedona_type: SedonaType) {
        use arrow_schema::DataType;

        let udf = SedonaScalarUDF::from_impl("st_linemerge", st_line_merge_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Boolean)],
        );
        tester.assert_return_type(sedona_type);

        let result = tester
            .invoke_scalar_scalar("MULTILINESTRING ((0 0, 1 0), (1 0, 1 1))", false)
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING (0 0, 1 0, 1 1)");
    }
}
