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
use datafusion_common::{exec_datafusion_err, Result};
use datafusion_expr::ColumnarValue;
use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_functions::executor::WkbBytesExecutor;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::geography::{Geography, GeographyFactory};
use crate::rect_bounder::RectBounder;

/// Returns a vector of (function_name, kernel) tuples for XY min/max functions
pub fn st_xy_minmax_kernels() -> Vec<(String, ScalarKernelRef)> {
    vec![
        (
            "st_xmin".to_string(),
            Arc::new(STXyMinMax {
                dim: "x",
                is_max: false,
            }) as ScalarKernelRef,
        ),
        (
            "st_xmax".to_string(),
            Arc::new(STXyMinMax {
                dim: "x",
                is_max: true,
            }) as ScalarKernelRef,
        ),
        (
            "st_ymin".to_string(),
            Arc::new(STXyMinMax {
                dim: "y",
                is_max: false,
            }) as ScalarKernelRef,
        ),
        (
            "st_ymax".to_string(),
            Arc::new(STXyMinMax {
                dim: "y",
                is_max: true,
            }) as ScalarKernelRef,
        ),
    ]
}

#[derive(Debug)]
struct STXyMinMax {
    dim: &'static str,
    is_max: bool,
}

impl SedonaScalarKernel for STXyMinMax {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geography()],
            SedonaType::Arrow(DataType::Float64),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbBytesExecutor::new(arg_types, args);
        let mut builder = Float64Builder::with_capacity(executor.num_iterations());
        let mut factory = GeographyFactory::new();
        let mut geog = Geography::new();
        let mut bounder = RectBounder::new();

        executor.execute_wkb_void(|maybe_wkb| {
            match maybe_wkb {
                Some(wkb) => {
                    factory
                        .init_from_wkb(wkb, &mut geog)
                        .map_err(|e| exec_datafusion_err!("Error parsing geography: {e}"))?;
                    builder.append_option(invoke_scalar(
                        &geog,
                        &mut bounder,
                        self.dim,
                        self.is_max,
                    )?);
                }
                None => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(
    geog: &Geography,
    bounder: &mut RectBounder,
    dim: &'static str,
    is_max: bool,
) -> Result<Option<f64>> {
    // Clear the bounder for reuse
    bounder.clear();

    // Add the geography to the bounder
    bounder
        .bound(geog)
        .map_err(|e| exec_datafusion_err!("Error bounding geography: {e}"))?;

    // Get the bounding rectangle
    let Some((xmin, ymin, xmax, ymax)) = bounder
        .finish()
        .map_err(|e| exec_datafusion_err!("Error finishing bounds: {e}"))?
    else {
        // Empty geography
        return Ok(None);
    };

    match (dim, is_max) {
        ("x", false) => Ok(Some(xmin)),
        ("x", true) => Ok(Some(xmax)),
        ("y", false) => Ok(Some(ymin)),
        ("y", true) => Ok(Some(ymax)),
        _ => sedona_internal_err!("unexpected dim: {dim}"),
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{create_array as arrow_array, ArrayRef};
    use datafusion_common::ScalarValue;
    use rstest::rstest;
    use sedona_schema::datatypes::{WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY};
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;
    use sedona_expr::scalar_udf::SedonaScalarUDF;

    fn create_udf(name: &str) -> SedonaScalarUDF {
        for (kernel_name, kernel) in st_xy_minmax_kernels() {
            if kernel_name == name {
                return SedonaScalarUDF::from_impl(name, kernel);
            }
        }
        panic!("Kernel not found: {name}");
    }

    fn assert_approx_equal(actual: ScalarValue, expected: f64) {
        match actual {
            ScalarValue::Float64(Some(val)) => {
                assert!(
                    (val - expected).abs() < 1e-10,
                    "Expected {expected}, got {val}"
                );
            }
            _ => panic!("Expected Float64, got {actual:?}"),
        }
    }

    #[rstest]
    fn udf(#[values(WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY)] sedona_type: SedonaType) {
        let xmin_tester =
            ScalarUdfTester::new(create_udf("st_xmin").into(), vec![sedona_type.clone()]);
        let ymin_tester =
            ScalarUdfTester::new(create_udf("st_ymin").into(), vec![sedona_type.clone()]);
        let xmax_tester =
            ScalarUdfTester::new(create_udf("st_xmax").into(), vec![sedona_type.clone()]);
        let ymax_tester =
            ScalarUdfTester::new(create_udf("st_ymax").into(), vec![sedona_type.clone()]);

        xmin_tester.assert_return_type(DataType::Float64);
        ymin_tester.assert_return_type(DataType::Float64);
        xmax_tester.assert_return_type(DataType::Float64);
        ymax_tester.assert_return_type(DataType::Float64);

        // Test with a polygon
        let input_wkt = "POLYGON ((-1 0, 0 -2, 3 1, 0 4, -1 0))";
        assert_approx_equal(xmin_tester.invoke_scalar(input_wkt).unwrap(), -1.0);
        assert_approx_equal(ymin_tester.invoke_scalar(input_wkt).unwrap(), -2.0);
        assert_approx_equal(xmax_tester.invoke_scalar(input_wkt).unwrap(), 3.0);
        assert_approx_equal(ymax_tester.invoke_scalar(input_wkt).unwrap(), 4.0);

        // Test with a linestring
        let input_wkt = "LINESTRING (1 2, 5 6)";
        assert_approx_equal(xmin_tester.invoke_scalar(input_wkt).unwrap(), 1.0);
        assert_approx_equal(xmax_tester.invoke_scalar(input_wkt).unwrap(), 5.0);
        assert_approx_equal(ymin_tester.invoke_scalar(input_wkt).unwrap(), 2.0);
        assert_approx_equal(ymax_tester.invoke_scalar(input_wkt).unwrap(), 6.0);

        // Test array input with empty geometries
        let input_wkt = vec![None, Some("POINT EMPTY"), Some("GEOMETRYCOLLECTION EMPTY")];

        let expected: ArrayRef = arrow_array!(Float64, [None, None, None]);
        assert_array_equal(
            &xmin_tester.invoke_wkb_array(input_wkt.clone()).unwrap(),
            &expected,
        );
        assert_array_equal(
            &ymin_tester.invoke_wkb_array(input_wkt.clone()).unwrap(),
            &expected,
        );
        assert_array_equal(
            &xmax_tester.invoke_wkb_array(input_wkt.clone()).unwrap(),
            &expected,
        );
        assert_array_equal(
            &ymax_tester.invoke_wkb_array(input_wkt.clone()).unwrap(),
            &expected,
        );
    }

    #[test]
    fn test_point() {
        let tester = ScalarUdfTester::new(create_udf("st_xmin").into(), vec![WKB_GEOGRAPHY]);
        let result = tester.invoke_scalar("POINT (10 20)").unwrap();
        assert_approx_equal(result, 10.0);

        let tester = ScalarUdfTester::new(create_udf("st_ymin").into(), vec![WKB_GEOGRAPHY]);
        let result = tester.invoke_scalar("POINT (10 20)").unwrap();
        assert_approx_equal(result, 20.0);
    }

    #[test]
    fn test_null_scalar() {
        let tester = ScalarUdfTester::new(create_udf("st_xmin").into(), vec![WKB_GEOGRAPHY]);
        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        assert_eq!(result, ScalarValue::Float64(None));
    }
}
