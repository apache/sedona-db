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
use datafusion_common::cast::{as_boolean_array, as_float64_array};
use datafusion_common::error::Result;
use datafusion_common::{DataFusionError, ScalarValue};
use datafusion_expr::ColumnarValue;
use geos::Geom;
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_geometry::wkb_factory::WKB_MIN_PROBABLE_BYTES;
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

use crate::executor::GeosExecutor;

/// ST_ConcaveHull() implementation using the geos crate
pub fn st_concave_hull_impl() -> ScalarKernelRef {
    Arc::new(STConcaveHull {})
}

#[derive(Debug)]
struct STConcaveHull {}

impl SedonaScalarKernel for STConcaveHull {
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
        invoke_batch_impl(arg_types, args)
    }
}

fn invoke_batch_impl(arg_types: &[SedonaType], args: &[ColumnarValue]) -> Result<ColumnarValue> {
    let executor = GeosExecutor::new(arg_types, args);
    let mut builder = BinaryBuilder::with_capacity(
        executor.num_iterations(),
        WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
    );

    let pct_convex_val = args[1]
        .cast_to(&DataType::Float64, None)?
        .to_array(executor.num_iterations())?;
    let pct_convex_array = as_float64_array(&pct_convex_val)?;
    let mut pct_convex_iter = pct_convex_array.iter();

    let allow_holes_val = args
        .get(2)
        .unwrap_or(&ColumnarValue::Scalar(ScalarValue::Boolean(Some(false))))
        .cast_to(&DataType::Boolean, None)?
        .to_array(executor.num_iterations())?;
    let allow_holes_array = as_boolean_array(&allow_holes_val)?;
    let mut allow_holes_iter = allow_holes_array.iter();

    executor.execute_wkb_void(|maybe_wkb| {
        match (
            maybe_wkb,
            pct_convex_iter.next().unwrap(),
            allow_holes_iter.next().unwrap(),
        ) {
            (Some(wkb), Some(pct_convex), Some(allow_holes)) => {
                invoke_scalar(&wkb, pct_convex, allow_holes, &mut builder)?;
                builder.append_value([]);
            }
            _ => builder.append_null(),
        }
        Ok(())
    })?;

    executor.finish(Arc::new(builder.finish()))
}

fn invoke_scalar(
    geos_geom: &geos::Geometry,
    pct_convex: f64,
    allow_holes: bool,
    writer: &mut impl std::io::Write,
) -> Result<()> {
    let input_wkb = geos_geom
        .to_wkb()
        .map_err(|e| DataFusionError::Execution(format!("Failed to convert to WKB: {e}")))?;
    let wkb_bytes: &[u8] = input_wkb.as_ref();

    let result_wkb = concave_hull_via_geos_sys(wkb_bytes, pct_convex, allow_holes)?;

    writer.write_all(&result_wkb)?;
    Ok(())
}

fn concave_hull_via_geos_sys(input_wkb: &[u8], ratio: f64, allow_holes: bool) -> Result<Vec<u8>> {
    unsafe {
        let ctx = geos_sys::GEOS_init_r();
        if ctx.is_null() {
            return Err(DataFusionError::Execution(
                "Failed to initialize GEOS context".to_string(),
            ));
        }

        let wkb_reader = geos_sys::GEOSWKBReader_create_r(ctx);
        if wkb_reader.is_null() {
            geos_sys::GEOS_finish_r(ctx);
            return Err(DataFusionError::Execution(
                "Failed to create WKB reader".to_string(),
            ));
        }

        let geom_ptr =
            geos_sys::GEOSWKBReader_read_r(ctx, wkb_reader, input_wkb.as_ptr(), input_wkb.len());
        geos_sys::GEOSWKBReader_destroy_r(ctx, wkb_reader);

        if geom_ptr.is_null() {
            geos_sys::GEOS_finish_r(ctx);
            return Err(DataFusionError::Execution("Failed to read WKB".to_string()));
        }

        let hull_ptr =
            geos_sys::GEOSConcaveHull_r(ctx, geom_ptr, ratio, if allow_holes { 1 } else { 0 });

        geos_sys::GEOSGeom_destroy_r(ctx, geom_ptr);

        if hull_ptr.is_null() {
            geos_sys::GEOS_finish_r(ctx);
            return Err(DataFusionError::Execution(
                "GEOSConcaveHull_r returned null".to_string(),
            ));
        }

        let wkb_writer = geos_sys::GEOSWKBWriter_create_r(ctx);
        if wkb_writer.is_null() {
            geos_sys::GEOSGeom_destroy_r(ctx, hull_ptr);
            geos_sys::GEOS_finish_r(ctx);
            return Err(DataFusionError::Execution(
                "Failed to create WKB writer".to_string(),
            ));
        }

        let mut wkb_size: usize = 0;
        let wkb_ptr = geos_sys::GEOSWKBWriter_write_r(ctx, wkb_writer, hull_ptr, &mut wkb_size);

        geos_sys::GEOSWKBWriter_destroy_r(ctx, wkb_writer);
        geos_sys::GEOSGeom_destroy_r(ctx, hull_ptr);

        if wkb_ptr.is_null() {
            geos_sys::GEOS_finish_r(ctx);
            return Err(DataFusionError::Execution(
                "Failed to write WKB".to_string(),
            ));
        }

        let wkb_slice = std::slice::from_raw_parts(wkb_ptr, wkb_size);
        let result = wkb_slice.to_vec();

        geos_sys::GEOSFree_r(ctx, wkb_ptr as *mut _);
        geos_sys::GEOS_finish_r(ctx);

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use arrow_schema::DataType;
    use datafusion_common::ScalarValue;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_concavehull", st_concave_hull_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Float64)],
        );
        assert_eq!(tester.return_type().unwrap(), WKB_GEOMETRY);

        let result = tester.invoke_scalar_scalar("POINT (2.5 3.1)", 0.1).unwrap();
        tester.assert_scalar_result_equals(result, "POINT (2.5 3.1)");

        let result = tester
            .invoke_scalar_scalar("LINESTRING (50 50, 150 150, 150 50)", 0.1)
            .unwrap();
        tester.assert_scalar_result_equals(result, "POLYGON ((50 50, 150 150, 150 50, 50 50))");

        let result = tester
            .invoke_scalar_scalar("LINESTRING (100 150, 50 60, 70 80, 160 170)", 0.3)
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            "POLYGON ((70 80, 50 60, 100 150, 160 170, 70 80))",
        );

        let result = tester
            .invoke_scalar_scalar(
                "GEOMETRYCOLLECTION (MULTIPOINT((1 1), (3 3)), POINT(5 6), LINESTRING(4 5, 5 6))",
                0.1,
            )
            .unwrap();
        tester.assert_scalar_result_equals(result, "POLYGON ((3 3, 1 1, 4 5, 5 6, 3 3))");

        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());
    }
}
