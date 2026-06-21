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

use arrow_array::builder::Float64Builder;
use arrow_schema::DataType;
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::ColumnarValue;
use geos::{GResult, Geom, GeometryTypes};
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::executor::GeosExecutor;

/// ST_MaxDistance() implementation using the geos crate
pub fn st_max_distance_impl() -> ScalarKernelRef {
    Arc::new(STMaxDistance {})
}

#[derive(Debug)]
struct STMaxDistance {}

impl SedonaScalarKernel for STMaxDistance {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry(), ArgMatcher::is_geometry()],
            SedonaType::Arrow(DataType::Float64),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = GeosExecutor::new(arg_types, args);
        let mut builder = Float64Builder::with_capacity(executor.num_iterations());
        executor.execute_wkb_wkb_void(|lhs, rhs| {
            match (lhs, rhs) {
                (Some(lhs), Some(rhs)) => {
                    builder.append_value(invoke_scalar(lhs, rhs).map_err(|e| {
                        DataFusionError::Execution(format!("Failed to calculate max distance: {e}"))
                    })?);
                }
                _ => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn collect_vertices<G: Geom>(geom: &G, coords: &mut Vec<(f64, f64)>) -> GResult<()> {
    match geom.geometry_type()? {
        GeometryTypes::Point | GeometryTypes::LineString | GeometryTypes::LinearRing => {
            let seq = geom.get_coord_seq()?;
            let size = seq.size()?;
            for i in 0..size {
                coords.push((seq.get_x(i)?, seq.get_y(i)?));
            }
        }
        GeometryTypes::Polygon => {
            let ext = geom.get_exterior_ring()?;
            collect_vertices(&ext, coords)?;
            let n = geom.get_num_interior_rings()?;
            for i in 0..n {
                let ring = geom.get_interior_ring_n(i)?;
                collect_vertices(&ring, coords)?;
            }
        }
        _ => {
            let n = geom.get_num_geometries()?;
            for i in 0..n {
                let sub = geom.get_geometry_n(i)?;
                collect_vertices(&sub, coords)?;
            }
        }
    }
    Ok(())
}

fn invoke_scalar(lhs: &geos::Geometry, rhs: &geos::Geometry) -> GResult<f64> {
    let mut lhs_coords = Vec::new();
    let mut rhs_coords = Vec::new();
    collect_vertices(lhs, &mut lhs_coords)?;
    collect_vertices(rhs, &mut rhs_coords)?;

    let mut max_dist_sq = f64::NEG_INFINITY;
    for (ax, ay) in &lhs_coords {
        for (bx, by) in &rhs_coords {
            let dx = ax - bx;
            let dy = ay - by;
            let d2 = dx * dx + dy * dy;
            if d2 > max_dist_sq {
                max_dist_sq = d2;
            }
        }
    }

    Ok(max_dist_sq.max(0.0).sqrt())
}

#[cfg(test)]
mod tests {
    use arrow_array::{create_array as arrow_array, ArrayRef};
    use datafusion_common::ScalarValue;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::create::create_array;
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_impl("st_maxdistance", st_max_distance_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type.clone(), sedona_type]);
        tester.assert_return_type(DataType::Float64);

        let result = tester
            .invoke_scalar_scalar("POINT (0 0)", "POINT (3 4)")
            .unwrap();
        tester.assert_scalar_result_equals(result, 5.0);

        // Line vs point — max is distance to farthest vertex
        let result = tester
            .invoke_scalar_scalar("POINT (0 0)", "LINESTRING (0 0, 0 2)")
            .unwrap();
        tester.assert_scalar_result_equals(result, 2.0);

        // Two lines — max is between farthest-apart vertices
        let result = tester
            .invoke_scalar_scalar("LINESTRING (0 0, 10 0)", "LINESTRING (0 10, 10 10)")
            .unwrap();
        // max of (0,0)-(0,10)=10, (0,0)-(10,10)=sqrt(200), (10,0)-(0,10)=sqrt(200), (10,0)-(10,10)=10
        tester.assert_scalar_result_equals(result, 200f64.sqrt());

        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());

        let lhs = create_array(
            &[Some("POINT (0 0)"), Some("POINT (0 0)"), None],
            &WKB_GEOMETRY,
        );
        let rhs = create_array(
            &[Some("POINT (3 4)"), None, Some("POINT (1 1)")],
            &WKB_GEOMETRY,
        );
        let expected: ArrayRef = arrow_array!(Float64, [Some(5.0), None, None]);
        assert_array_equal(&tester.invoke_array_array(lhs, rhs).unwrap(), &expected);
    }
}
