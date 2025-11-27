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
<<<<<<< HEAD
use geos::{Geom, Geometry, GeometryTypes};
=======
use geos::Geom;
use geos::GeometryTypes;
>>>>>>> d797f12 (modified num of rings file)
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

pub fn st_num_interior_rings_impl() -> ScalarKernelRef {
    Arc::new(STNumInteriorRings {})
}

#[derive(Debug)]
struct STNumInteriorRings {}

impl SedonaScalarKernel for STNumInteriorRings {
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
                    let res = invoke_scalar(&geom)?;
                    match res {
                        Some(n) => builder.append_value(n),
                        None => builder.append_null(),
                    }
                }
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

<<<<<<< HEAD
fn invoke_scalar(geom: &Geometry) -> Result<Option<i32>> {
    match geom.geometry_type() {
        GeometryTypes::Polygon => {
            let is_empty = geom.is_empty().map_err(|e| {
                DataFusionError::Execution(format!("Failed to check if geometry is empty: {e}"))
            })?;

            if is_empty {
                Ok(Some(0))
            } else {
                let count = geom.get_num_interior_rings().map_err(|e| {
                    DataFusionError::Execution(format!("Failed to get num interior rings: {e}"))
                })?;
                Ok(Some(count as i32))
            }
        }
        _ => Ok(None),
=======
fn invoke_scalar(geos_geom: &geos::Geometry) -> Result<i32> {
    // Only polygons have interior rings; for everything else, return 0.
    if matches!(geos_geom.geometry_type(), GeometryTypes::Polygon) {
        let count = geos_geom.get_num_interior_rings().map_err(|e| {
            DataFusionError::Execution(format!("Failed to get num interior rings: {e}"))
        })?;

        Ok(count as i32)
    } else {
        // POINT, LINESTRING, MULTIPOINT, etc. -> 0 holes
        Ok(0)
>>>>>>> d797f12 (modified num of rings file)
    }
}
#[cfg(test)]
mod tests {
<<<<<<< HEAD
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int32Array};
    use arrow_schema::DataType;
    use datafusion_common::ScalarValue;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{SedonaType, WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::compare::assert_array_equal;
=======
    use arrow_array::{create_array, ArrayRef};
    use datafusion_common::ScalarValue;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
>>>>>>> d797f12 (modified num of rings file)
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_numinteriorrings", st_num_interior_rings_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type]);
        tester.assert_return_type(DataType::Int32);

<<<<<<< HEAD
        // Polygon with 2 interior rings -> 2
        let result = tester
            .invoke_scalar(
                "POLYGON((0 0,10 0,10 6,0 6,0 0),(1 1,2 1,2 5,1 5,1 1),(8 5,8 4,9 4,9 5,8 5))",
=======
        // Polygon with two interior rings -> 2
        let result = tester
            .invoke_scalar(
                "POLYGON(
                    (0 0,10 0,10 6,0 6,0 0),
                    (1 1,2 1,2 5,1 5,1 1),
                    (8 5,8 4,9 4,9 5,8 5)
                )",
>>>>>>> d797f12 (modified num of rings file)
            )
            .unwrap();
        tester.assert_scalar_result_equals(result, 2_i32);

<<<<<<< HEAD
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
        ];
        let expected: ArrayRef = Arc::new(Int32Array::from(vec![
            None,
            None,
            None,
            Some(0),
            Some(0),
            Some(1),
            Some(2),
            None,
            None,
        ]));

        let result = tester.invoke_wkb_array(input_wkt).unwrap();
        assert_array_equal(&result, &expected);
=======
        // NULL -> NULL
        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        assert!(result.is_null());

        let input_wkt = vec![
            None,
        ];

        let expected: ArrayRef = create_array!(Int32, [Some(1), Some(0), None]);
        assert_eq!(&tester.invoke_wkb_array(input_wkt).unwrap(), &expected);
>>>>>>> d797f12 (modified num of rings file)
    }
}
