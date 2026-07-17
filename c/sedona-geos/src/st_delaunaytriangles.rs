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
use datafusion_common::{
    cast::{as_float64_array, as_int64_array},
    exec_datafusion_err, Result,
};
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

fn invoke_scalar(
    geom: &Geometry,
    tolerance: f64,
    only_edges: bool,
    writer: &mut impl std::io::Write,
) -> Result<()> {
    let result = geom
        .delaunay_triangulation(tolerance, only_edges)
        .map_err(|e| exec_datafusion_err!("ST_DelaunayTriangles failed: {e}"))?;
    write_geos_geometry(&result, writer)?;
    Ok(())
}

// ── 1-arg: ST_DelaunayTriangles(geom) ────────────────────────────────────────

pub fn st_delaunay_triangles_impl() -> Vec<ScalarKernelRef> {
    ItemCrsKernel::wrap_impl(STDelaunayTriangles)
}

#[derive(Debug)]
struct STDelaunayTriangles;

impl SedonaScalarKernel for STDelaunayTriangles {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOMETRY).match_args(args)
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
        executor.execute_wkb_void(|maybe_geom| {
            match maybe_geom {
                Some(geom) => {
                    invoke_scalar(geom, 0.0, false, &mut builder)?;
                    builder.append_value([]);
                }
                None => builder.append_null(),
            }
            Ok(())
        })?;
        executor.finish(Arc::new(builder.finish()))
    }
}

// ── 2-arg: ST_DelaunayTriangles(geom, tolerance) ─────────────────────────────

pub fn st_delaunay_triangles_tolerance_impl() -> Vec<ScalarKernelRef> {
    ItemCrsKernel::wrap_impl(STDelaunayTrianglesWithTolerance)
}

#[derive(Debug)]
struct STDelaunayTrianglesWithTolerance;

impl SedonaScalarKernel for STDelaunayTrianglesWithTolerance {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        ArgMatcher::new(
            vec![ArgMatcher::is_geometry(), ArgMatcher::is_numeric()],
            WKB_GEOMETRY,
        )
        .match_args(args)
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
        let tol_value = args[1]
            .cast_to(&DataType::Float64, None)?
            .to_array(executor.num_iterations())?;
        let tol_array = as_float64_array(&tol_value)?;
        let mut tol_iter = tol_array.iter();
        executor.execute_wkb_void(|maybe_geom| {
            match (maybe_geom, tol_iter.next().unwrap()) {
                (Some(geom), Some(tol)) => {
                    invoke_scalar(geom, tol, false, &mut builder)?;
                    builder.append_value([]);
                }
                _ => builder.append_null(),
            }
            Ok(())
        })?;
        executor.finish(Arc::new(builder.finish()))
    }
}

// -- 3-arg: ST_DelaunayTriangles(geom, tolerance, flags)
// flags=0 -> polygon output (default), flags=1 -> multilinestring edges only

pub fn st_delaunay_triangles_flags_impl() -> Vec<ScalarKernelRef> {
    ItemCrsKernel::wrap_impl(STDelaunayTrianglesWithFlags)
}

#[derive(Debug)]
struct STDelaunayTrianglesWithFlags;

impl SedonaScalarKernel for STDelaunayTrianglesWithFlags {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        ArgMatcher::new(
            vec![
                ArgMatcher::is_geometry(),
                ArgMatcher::is_numeric(),
                ArgMatcher::is_integer(),
            ],
            WKB_GEOMETRY,
        )
        .match_args(args)
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
        let tol_value = args[1]
            .cast_to(&DataType::Float64, None)?
            .to_array(executor.num_iterations())?;
        let tol_array = as_float64_array(&tol_value)?;
        let mut tol_iter = tol_array.iter();

        let flags_value = args[2]
            .cast_to(&DataType::Int64, None)?
            .to_array(executor.num_iterations())?;
        let flags_array = as_int64_array(&flags_value)?;
        let mut flags_iter = flags_array.iter();

        executor.execute_wkb_void(|maybe_geom| {
            match (
                maybe_geom,
                tol_iter.next().unwrap(),
                flags_iter.next().unwrap(),
            ) {
                (Some(geom), Some(tol), Some(flag)) => {
                    let only_edges = match flag {
                        0 => false,
                        1 => true,
                        _ => {
                            return Err(exec_datafusion_err!(
                                "ST_DelaunayTriangles flags must be 0 or 1, got {flag}"
                            ))
                        }
                    };
                    invoke_scalar(geom, tol, only_edges, &mut builder)?;
                    builder.append_value([]);
                }
                _ => builder.append_null(),
            }
            Ok(())
        })?;
        executor.finish(Arc::new(builder.finish()))
    }
}

#[cfg(test)]
mod tests {
    use datafusion_common::ScalarValue;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS};
    use sedona_testing::{create::create_scalar, testers::ScalarUdfTester};

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

    #[rstest]
    fn udf_flags(#[values(WKB_GEOMETRY)] sedona_type: SedonaType) {
        let udf =
            SedonaScalarUDF::from_impl("st_delaunaytriangles", st_delaunay_triangles_flags_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![
                sedona_type.clone(),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Int64),
            ],
        );

        tester.assert_return_type(WKB_GEOMETRY);

        let geom = create_scalar(Some("MULTIPOINT ((0 0), (1 1), (0 1))"), &sedona_type);

        let result = tester
            .invoke_scalar_scalar_scalar(
                geom.clone(),
                ScalarValue::Float64(Some(0.0)),
                ScalarValue::Int64(Some(1)),
            )
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            "MULTILINESTRING ((0 1, 1 1), (0 0, 0 1), (0 0, 1 1))",
        );

        let result = tester
            .invoke_scalar_scalar_scalar(
                geom,
                ScalarValue::Float64(Some(0.0)),
                ScalarValue::Int64(Some(0)),
            )
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            "GEOMETRYCOLLECTION (POLYGON ((0 1, 0 0, 1 1, 0 1)))",
        );
    }
}
