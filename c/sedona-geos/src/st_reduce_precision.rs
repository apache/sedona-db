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
use arrow_schema::DataType;
use datafusion_common::{cast::as_float64_array, DataFusionError, Result};
use datafusion_expr::ColumnarValue;
use geos::{Geom, Precision};
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

/// ST_ReducePrecision() implementation using the geos crate
pub fn st_reduce_precision_impl() -> Vec<ScalarKernelRef> {
    ItemCrsKernel::wrap_impl(STReducePrecision {})
}

#[derive(Debug)]
struct STReducePrecision {}

impl SedonaScalarKernel for STReducePrecision {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry(), ArgMatcher::is_numeric()],
            WKB_GEOMETRY,
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = GeosExecutor::new(arg_types, args);

        let grid_size_value = args[1]
            .cast_to(&DataType::Float64, None)?
            .to_array(executor.num_iterations())?;
        let grid_size_array = as_float64_array(&grid_size_value)?;
        let mut grid_size_iter = grid_size_array.iter();

        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
        );

        executor.execute_wkb_void(|wkb| {
            match (wkb, grid_size_iter.next().unwrap()) {
                (Some(wkb), Some(grid_size)) => {
                    invoke_scalar(wkb, grid_size, &mut builder)?;
                    builder.append_value([]);
                }
                _ => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(
    geos_geom: &geos::Geometry,
    grid_size: f64,
    writer: &mut impl std::io::Write,
) -> Result<()> {
    // Reducing the precision of an empty geometry is a no-op. Skip GEOS
    // set_precision here because it promotes empty geometries to the Z
    // dimension (e.g. LINESTRING EMPTY -> LINESTRING Z EMPTY), which would
    // change the output dimensionality of the input.
    let is_empty = geos_geom
        .is_empty()
        .map_err(|e| DataFusionError::Execution(format!("Failed to check is_empty: {e}")))?;
    if is_empty {
        return write_geos_geometry(geos_geom, writer);
    }

    let geometry = geos_geom
        .set_precision(grid_size, Precision::ValidOutput)
        .map_err(|e| DataFusionError::Execution(format!("Failed to reduce precision: {e}")))?;
    write_geos_geometry(&geometry, writer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_common::ScalarValue;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS, WKB_VIEW_GEOMETRY};
    use sedona_testing::{
        compare::assert_array_equal, create::create_array, testers::ScalarUdfTester,
    };

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_impl("st_reduceprecision", st_reduce_precision_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Float64)],
        );
        tester.assert_return_type(WKB_GEOMETRY);

        let result = tester
            .invoke_scalar_scalar("POINT (1.123456789 2.987654321)", 0.001)
            .unwrap();
        tester.assert_scalar_result_equals(result, "POINT (1.123 2.988)");

        let result = tester.invoke_scalar_scalar("POINT (0.1 0.2)", 1.0).unwrap();
        tester.assert_scalar_result_equals(result, "POINT (0 0)");

        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, 0.001)
            .unwrap();
        assert!(result.is_null());

        let result = tester
            .invoke_scalar_scalar("POINT (1 2)", ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());

        // Empty geometry input is a no-op: empty in -> empty out, with the
        // input dimensionality preserved (no Z promotion from GEOS).
        let result = tester
            .invoke_scalar_scalar("LINESTRING EMPTY", 1.0)
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING EMPTY");

        let result = tester.invoke_scalar_scalar("POLYGON EMPTY", 1.0).unwrap();
        tester.assert_scalar_result_equals(result, "POLYGON EMPTY");

        let input_wkt = vec![
            Some("LINESTRING (1.123 2.456, 3.789 4.012)"),
            Some("LINESTRING (1.3 2.7, 3.2 4.8)"),
            None,
        ];
        let expected = create_array(
            &[
                Some("LINESTRING (1 2, 4 4)"),
                Some("LINESTRING (1 3, 3 5)"),
                None,
            ],
            &WKB_GEOMETRY,
        );
        assert_array_equal(
            &tester.invoke_wkb_array_scalar(input_wkt, 1.0).unwrap(),
            &expected,
        );
    }

    #[rstest]
    fn udf_invoke_item_crs(#[values(WKB_GEOMETRY_ITEM_CRS.clone())] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_impl("st_reduceprecision", st_reduce_precision_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Float64)],
        );
        tester.assert_return_type(sedona_type);

        let result = tester.invoke_scalar_scalar("POINT (0.1 0.2)", 1.0).unwrap();
        tester.assert_scalar_result_equals(result, "POINT (0 0)");
    }
}
