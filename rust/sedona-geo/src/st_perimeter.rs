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
use datafusion_common::error::Result;
use datafusion_expr::ColumnarValue;
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_functions::executor::WkbExecutor;
use sedona_geo_generic_alg::algorithm::{line_measures::Euclidean, LengthMeasurableExt};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};
use wkb::reader::Wkb;

/// ST_Perimeter() implementation using [LengthMeasurableExt::perimeter_ext] with Euclidean metric
pub fn st_perimeter_impl() -> ScalarKernelRef {
    Arc::new(STPerimeter {})
}

#[derive(Debug)]
struct STPerimeter {}

impl SedonaScalarKernel for STPerimeter {
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
        let executor = WkbExecutor::new(arg_types, args);
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

fn invoke_scalar(wkb: &Wkb) -> Result<f64> {
    Ok(wkb.perimeter_ext(&Euclidean))
}

#[cfg(test)]
mod tests {
    use arrow_array::{create_array, ArrayRef};
    use datafusion_common::scalar::ScalarValue;
    use rstest::rstest;
    use sedona_functions::register::stubs::st_perimeter_udf;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let mut udf = st_perimeter_udf();
        udf.add_kernel(st_perimeter_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type]);

        assert_eq!(
            tester.return_type().unwrap(),
            SedonaType::Arrow(DataType::Float64)
        );

        // Test with a square polygon
        assert_eq!(
            tester
                .invoke_wkb_scalar(Some("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"))
                .unwrap(),
            ScalarValue::Float64(Some(4.0))
        );

        let input_wkt = vec![
            Some("POINT(1 2)"), // Point should have 0 perimeter
            None,
            Some("LINESTRING (0 0, 3 4)"), // LineString perimeter equals length (0.0)
            Some("POLYGON ((0 0, 4 0, 4 3, 0 3, 0 0))"), // Rectangle perimeter: 2*(4+3) = 14
            Some("POLYGON ((0 0, 1 0, 0.5 1, 0 0))"), // Triangle with sides approx 1, 1.118, 1.118
            Some("MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((2 2, 3 2, 3 3, 2 3, 2 2)))"), // Two unit squares
        ];
        let expected: ArrayRef = create_array!(
            Float64,
            [
                Some(0.0),
                None,
                Some(0.0),
                Some(14.0),
                Some(3.236_067_977_499_79), // 1 + sqrt(1.25) + sqrt(1.25)
                Some(8.0)                   // 4 + 4
            ]
        );
        assert_eq!(&tester.invoke_wkb_array(input_wkt).unwrap(), &expected);
    }

    #[test]
    fn test_polygon_with_hole() {
        let mut udf = st_perimeter_udf();
        udf.add_kernel(st_perimeter_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOMETRY]);

        // Polygon with a hole: outer ring 40, inner ring 24
        assert_eq!(
            tester
                .invoke_wkb_scalar(Some(
                    "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 8 2, 8 8, 2 8, 2 2))"
                ))
                .unwrap(),
            ScalarValue::Float64(Some(64.0))
        );
    }
}
