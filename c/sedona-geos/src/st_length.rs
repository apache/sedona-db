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
use datafusion_common::{error::Result, DataFusionError};
use datafusion_expr::ColumnarValue;
use geos::{
    GResult, Geom,
    GeometryTypes::{GeometryCollection, LineString, MultiLineString},
};
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::executor::GeosExecutor;

/// ST_Length() implementation using the geos crate
pub fn st_length_impl() -> ScalarKernelRef {
    Arc::new(STLength {})
}

#[derive(Debug)]
struct STLength {}

impl SedonaScalarKernel for STLength {
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
                    builder.append_value(invoke_scalar(&wkb).map_err(|e| {
                        DataFusionError::Execution(format!("Failed to calculate length: {e}"))
                    })?);
                }
                _ => builder.append_null(),
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(geos_geom: &geos::Geometry) -> GResult<f64> {
    // The .length() property may return non-zero values for non line geometries,
    // so we check for the geometry type here
    match geos_geom.geometry_type() {
        LineString => Ok(geos_geom.length()?),
        MultiLineString => Ok(geos_geom.length()?),
        GeometryCollection => {
            let mut sum = 0.0;
            for i in 0..geos_geom.get_num_geometries()? {
                let geom = geos_geom.get_geometry_n(i)?;
                match geom.geometry_type() {
                    LineString => sum += geom.length()?,
                    MultiLineString => sum += geom.length()?,
                    _ => {}
                }
            }
            Ok(sum)
        }
        _ => Ok(0.0),
    }
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
        use datafusion_common::ScalarValue;

        let udf = SedonaScalarUDF::from_kernel("st_length", st_length_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type]);
        tester.assert_return_type(DataType::Float64);

        let result = tester.invoke_scalar("LINESTRING (0 0, 1 0, 0 1)").unwrap();
        tester.assert_scalar_result_equals(result, 2.414213562373095);

        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        assert!(result.is_null());

        let input_wkt = vec![
            Some("POINT(1 2)"),
            None,
            Some("POLYGON ((0 0, 1 0, 0 1, 0 0))"),
            Some("LINESTRING (0 0, 1 0, 0 1)"),
            Some("GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 0, 0 1), LINESTRING (0 0, 1 0, 0 1), POLYGON ((0 0, 1 0, 0 1, 0 0)))"),
        ];
        let expected: ArrayRef = create_array!(
            Float64,
            [
                Some(0.0),
                None,
                Some(0.0),
                Some(2.414213562373095),
                Some(4.82842712474619)
            ]
        );
        assert_eq!(&tester.invoke_wkb_array(input_wkt).unwrap(), &expected);
    }
}
