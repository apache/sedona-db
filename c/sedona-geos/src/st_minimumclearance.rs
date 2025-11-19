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
use geos::Geom;
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::executor::GeosExecutor;

/// ST_MinimumClearance() implementation using the geos crate
pub fn st_minimum_clearance_impl() -> ScalarKernelRef {
    Arc::new(STMinimumClearance {})
}

#[derive(Debug)]
struct STMinimumClearance {}

impl SedonaScalarKernel for STMinimumClearance {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry()],
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
        executor.execute_wkb_void(|maybe_wkb| {
            match maybe_wkb {
                Some(wkb) => {
                    builder.append_value(invoke_scalar(&wkb)?);
                }
                _ => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(geos_geom: &geos::Geometry) -> Result<f64> {
    geos_geom.minimum_clearance().map_err(|e| {
        DataFusionError::Execution(format!("Failed to calculate minimum clearance: {e}"))
    })
}

#[cfg(test)]
mod tests {
    use arrow_array::{create_array, ArrayRef};
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_minimumclearance", st_minimum_clearance_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type]);
        tester.assert_return_type(DataType::Float64);

        let input_wkt = vec![
            Some("POINT (1 1)"),
            Some("LINESTRING (5 107, 54 84, 101 100)"),
            Some("LINESTRING (0 0, 1 1, 2 2)"),
            Some("LINESTRING(0 0,0 0,1 1,1 1,2 2)"),
            Some("POLYGON((0 0,0 3,3 3,3 0,0 0),(1 1,1 2,2 2,2 1,1 1))"),
            Some("POLYGON((0 0,0 1,0 1,1 1,1 0,0 0,0 0))"),
            Some("POLYGON EMPTY"),
            Some("POLYGON((0 0,3 0,3 3,2 1,1 3,0 3,0 0))"),
            Some("MULTIPOINT (10 40, 40 30)"),
            Some("MULTIPOINT ((10 10), (20 20), (30 30))"),
            Some("MULTILINESTRING ((10 10, 20 20), (30 30, 40 40))"),
            Some("MULTILINESTRING ((5 5, 10 5, 10 15), (20 20, 25 10))"),
            Some("MULTIPOLYGON(((26 125, 26 200, 126 200, 126 125, 26 125 ),( 51 150, 101 150, 76 175, 51 150 )),(( 151 100, 151 200, 176 175, 151 100 )))"),
            Some("MULTIPOLYGON(((0.5 0.5,0 0,0 1,0.5 0.5)),((0.5 0.5,1 1,1 0,0.5 0.5)),((2.5 2.5,2 2,2 3,2.5 2.5)),((2.5 2.5,3 3,3 2,2.5 2.5)))"),
            Some("GEOMETRYCOLLECTION(POINT(1 1),MULTIPOLYGON(((0 2,1 1,0 0,0 2)),((2 0,1 1,2 2,2 0))))"),
        ];

        let expected: ArrayRef = create_array!(
            Float64,
            [
                Some(f64::INFINITY),
                Some(49.64876634922564),
                Some(std::f64::consts::SQRT_2),
                Some(std::f64::consts::SQRT_2),
                Some(1.0),
                Some(1.0),
                Some(f64::INFINITY),
                Some(1.0),
                Some(31.622776601683793),
                Some(14.142135623730951),
                Some(14.142135623730951),
                Some(5.0),
                Some(25.0),
                Some(0.5),
                Some(1.0)
            ]
        );

        assert_eq!(&tester.invoke_wkb_array(input_wkt).unwrap(), &expected);
    }
}
