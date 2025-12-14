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

use crate::executor::GeosExecutor;
use arrow_array::builder::Int32Builder;
use arrow_schema::DataType;
use datafusion_common::{error::Result, DataFusionError};
use datafusion_expr::ColumnarValue;
use geos::{Geom, GeometryTypes};
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

pub fn st_nrings_impl() -> ScalarKernelRef {
    Arc::new(STNRings {})
}

#[derive(Debug)]
struct STNRings {}

impl SedonaScalarKernel for STNRings {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry()],
            SedonaType::Arrow(DataType::Int32),
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = GeosExecutor::new(arg_types, args);
        let mut builder = Int32Builder::with_capacity(executor.num_iterations());
        executor.execute_wkb_void(|maybe_geom| {
            match maybe_geom {
                None => builder.append_null(),
                Some(geom) => {
                    let val = invoke_scalar(&geom)?;
                    builder.append_value(val);
                }
            }
            Ok(())
        })?;
        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar<G: Geom>(geom: &G) -> Result<i32> {
    match geom
        .geometry_type()
        .map_err(|e| DataFusionError::Execution(format!("Failed to get geometry type: {e}")))?
    {
        GeometryTypes::Polygon => {
            if geom
                .is_empty()
                .map_err(|e| DataFusionError::Execution(format!("{e}")))?
            {
                return Ok(0);
            }
            let num_interior = geom
                .get_num_interior_rings()
                .map_err(|e| DataFusionError::Execution(format!("{e}")))?;
            Ok((num_interior + 1) as i32)
        }
        GeometryTypes::MultiPolygon | GeometryTypes::GeometryCollection => {
            if geom
                .is_empty()
                .map_err(|e| DataFusionError::Execution(format!("{e}")))?
            {
                return Ok(0);
            }
            let num_geoms = geom
                .get_num_geometries()
                .map_err(|e| DataFusionError::Execution(format!("{e}")))?;
            let mut total_rings = 0;
            for i in 0..num_geoms {
                let sub_geom = geom
                    .get_geometry_n(i)
                    .map_err(|e| DataFusionError::Execution(format!("{e}")))?;
                total_rings += invoke_scalar(&sub_geom)?;
            }
            Ok(total_rings)
        }
        _ => Ok(0),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int32Array};
    use arrow_schema::DataType;
    use datafusion_common::ScalarValue;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{SedonaType, WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_nrings", st_nrings_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type]);
        tester.assert_return_type(DataType::Int32);

        let result = tester
            .invoke_scalar(
                "POLYGON((0 0,10 0,10 6,0 6,0 0),(1 1,2 1,2 5,1 5,1 1),(8 5,8 4,9 4,9 5,8 5))",
            )
            .unwrap();
        tester.assert_scalar_result_equals(result, 3_i32);

        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        assert!(result.is_null());

        let input_wkt = vec![
            None,
            Some("POINT (1 2)"),
            Some("LINESTRING (0 0, 1 1, 2 2)"),
            Some("POLYGON EMPTY"),
            Some("POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))"),
            Some("POLYGON ((0 0,6 0,6 6,0 6,0 0),(2 2,4 2,4 4,2 4,2 2))"),
            Some(
                "POLYGON ((0 0,10 0,10 6,0 6,0 0),(1 1,2 1,2 5,1 5,1 1),(8 5,8 4,9 4,9 5,8 5))",
            ),
            Some(
                "MULTIPOLYGON (((0 0,5 0,5 5,0 5,0 0),(1 1,2 1,2 2,1 2,1 1)),((10 10,14 10,14 14,10 14,10 10)))",
            ),
            Some(
                "GEOMETRYCOLLECTION (POINT (1 2),POLYGON ((0 0,3 0,3 3,0 3,0 0)))",
            ),
            Some("POLYGON Z ((0 0 1, 1 0 1, 1 1 1, 0 1 1, 0 0 1))"),
            Some("GEOMETRYCOLLECTION(POINT(2 3), LINESTRING(0 0, 1 1, 2 2), POLYGON((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 2 1, 2 2, 1 2, 1 1)), MULTIPOLYGON(((5 5, 6 5, 6 6, 5 6, 5 5)), ((10 10, 12 10, 12 12, 10 12, 10 10), (10.5 10.5, 11 10.5, 11 11, 10.5 11, 10.5 10.5))), GEOMETRYCOLLECTION(POLYGON((20 20, 22 20, 22 22, 20 22, 20 20)), POINT(30 30)))"),
        ];

        let expected: ArrayRef = Arc::new(Int32Array::from(vec![
            None,
            Some(0),
            Some(0),
            Some(0),
            Some(1),
            Some(2),
            Some(3),
            Some(3),
            Some(1),
            Some(1),
            Some(6),
        ]));

        let result = tester.invoke_wkb_array(input_wkt).unwrap();
        assert_array_equal(&result, &expected);
    }
}
