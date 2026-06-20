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
use geos::{Geom, Geometry};
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

/// ST_DelaunayTriangles(geom) — no tolerance, returns polygons
pub fn st_delaunay_triangles_impl() -> Vec<ScalarKernelRef> {
    ItemCrsKernel::wrap_impl(STDelaunayTriangles { tolerance: false })
}

/// ST_DelaunayTriangles(geom, tolerance) — with tolerance
pub fn st_delaunay_triangles_tolerance_impl() -> Vec<ScalarKernelRef> {
    ItemCrsKernel::wrap_impl(STDelaunayTriangles { tolerance: true })
}

#[derive(Debug)]
struct STDelaunayTriangles {
    tolerance: bool,
}

impl SedonaScalarKernel for STDelaunayTriangles {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let arg_matchers = if self.tolerance {
            vec![ArgMatcher::is_geometry(), ArgMatcher::is_numeric()]
        } else {
            vec![ArgMatcher::is_geometry()]
        };
        let matcher = ArgMatcher::new(arg_matchers, WKB_GEOMETRY);
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

        if self.tolerance {
            let tolerance_value = args[1]
                .cast_to(&DataType::Float64, None)?
                .to_array(executor.num_iterations())?;
            let tolerance_array = as_float64_array(&tolerance_value)?;
            let mut tolerance_iter = tolerance_array.iter();

            executor.execute_wkb_void(|maybe_geom| {
                match (maybe_geom, tolerance_iter.next().unwrap()) {
                    (Some(geom), Some(tol)) => {
                        invoke_scalar(&geom, tol, &mut builder)?;
                        builder.append_value([]);
                    }
                    _ => builder.append_null(),
                }
                Ok(())
            })?;
        } else {
            executor.execute_wkb_void(|maybe_geom| {
                match maybe_geom {
                    Some(geom) => {
                        invoke_scalar(&geom, 0.0, &mut builder)?;
                        builder.append_value([]);
                    }
                    _ => builder.append_null(),
                }
                Ok(())
            })?;
        }

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(geom: &Geometry, tolerance: f64, writer: &mut impl std::io::Write) -> Result<()> {
    let result = geom
        .delaunay_triangulation(tolerance, false)
        .map_err(|e| DataFusionError::Execution(format!("ST_DelaunayTriangles failed: {e}")))?;
    write_geos_geometry(&result, writer)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use datafusion_common::ScalarValue;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS};
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn udf_no_tolerance(#[values(WKB_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_impl("st_delaunaytriangles", st_delaunay_triangles_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type.clone()]);

        tester.assert_return_type(WKB_GEOMETRY);

        let result = tester
            .invoke_scalar("MULTIPOINT ((0 0), (1 1), (0 1))")
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            "GEOMETRYCOLLECTION (POLYGON ((0 1, 0 0, 1 1, 0 1)))",
        );

        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        assert!(result.is_null());
    }

    #[rstest]
    fn udf_invoke_item_crs(#[values(WKB_GEOMETRY_ITEM_CRS.clone())] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_impl("st_delaunaytriangles", st_delaunay_triangles_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type.clone()]);
        tester.assert_return_type(sedona_type);
    }
}
