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
use datafusion_common::{error::Result, DataFusionError};
use datafusion_expr::ColumnarValue;
use geos::Geom;
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_geometry::wkb_factory::WKB_MIN_PROBABLE_BYTES;
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

use crate::executor::GeosExecutor;

pub fn st_intersection_impl() -> ScalarKernelRef {
    Arc::new(BinaryOverlay {
        name: "intersection",
        func: geos::Geometry::intersection,
    })
}

pub fn st_union_impl() -> ScalarKernelRef {
    Arc::new(BinaryOverlay {
        name: "union",
        func: geos::Geometry::union,
    })
}

pub fn st_difference_impl() -> ScalarKernelRef {
    Arc::new(BinaryOverlay {
        name: "difference",
        func: geos::Geometry::difference,
    })
}

pub fn st_sym_difference_impl() -> ScalarKernelRef {
    Arc::new(BinaryOverlay {
        name: "sym_difference",
        func: geos::Geometry::sym_difference,
    })
}

struct BinaryOverlay<F> {
    name: &'static str,
    func: F,
}

// Debug trait is necessary to implement SedonaScalarKernel
impl<F> std::fmt::Debug for BinaryOverlay<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BinaryOverlay")
            .field("name", &self.name)
            .finish()
    }
}

impl<F> SedonaScalarKernel for BinaryOverlay<F>
where
    F: Fn(&geos::Geometry, &geos::Geometry) -> geos::GResult<geos::Geometry>,
{
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry(), ArgMatcher::is_geometry()],
            WKB_GEOMETRY,
        );
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

        let func = &self.func;

        executor.execute_wkb_wkb_void(|lhs, rhs| {
            match (lhs, rhs) {
                (Some(lhs), Some(rhs)) => {
                    let geom = func(lhs, rhs).map_err(|e| {
                        DataFusionError::Execution(format!(
                            "Failed to calculate {}: {e}",
                            self.name
                        ))
                    })?;

                    let wkb = geom.to_wkb().map_err(|e| {
                        DataFusionError::Execution(format!("Failed to convert to WKB: {e}"))
                    })?;

                    builder.append_value(&wkb);
                }
                _ => builder.append_null(),
            };
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
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::create::create_array;
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn intersection_udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_intersection", st_intersection_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type.clone(), sedona_type]);
        tester.assert_return_type(WKB_GEOMETRY);

        let result = tester
            .invoke_scalar_scalar("POINT (0 0)", "LINESTRING (0 0, 0 2)")
            .unwrap();
        tester.assert_scalar_result_equals(result, "POINT (0 0)");

        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());

        let arg1 = create_array(
            &[
                Some("POLYGON ((1 1, 8 1, 8 8, 1 8, 1 1))"),
                Some("POLYGON ((0 0, 1 0, 0 1, 0 0))"),
                Some("POINT (0 0)"),
            ],
            &WKB_GEOMETRY,
        );
        let arg2 = create_array(
            &[
                Some("POLYGON ((2 2, 9 2, 9 9, 2 9, 2 2))"),
                Some("POINT (2 2)"),
                None,
            ],
            &WKB_GEOMETRY,
        );
        let expected = create_array(
            &[
                Some("POLYGON ((8 8, 8 2, 2 2, 2 8, 8 8))"),
                Some("POINT EMPTY"),
                None,
            ],
            &WKB_GEOMETRY,
        );
        assert_array_equal(&tester.invoke_array_array(arg1, arg2).unwrap(), &expected);
    }

    #[rstest]
    fn union_udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_union", st_union_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type.clone(), sedona_type]);
        tester.assert_return_type(WKB_GEOMETRY);

        let result = tester
            .invoke_scalar_scalar("POINT (0 0)", "LINESTRING (0 0, 0 2)")
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING (0 0, 0 2)");

        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());

        let arg1 = create_array(
            &[
                Some("POLYGON ((1 1, 8 1, 8 8, 1 8, 1 1))"),
                Some("POINT (1 2)"),
                Some("POINT (0 0)"),
            ],
            &WKB_GEOMETRY,
        );
        let arg2 = create_array(
            &[Some("POINT(-2 3)"), Some("POINT(-2 3)"), None],
            &WKB_GEOMETRY,
        );
        let expected = create_array(
            &[
                Some("GEOMETRYCOLLECTION (POLYGON((1 1, 8 1, 8 8, 1 8, 1 1)), POINT(-2 3))"),
                Some("MULTIPOINT (1 2, -2 3)"),
                None,
            ],
            &WKB_GEOMETRY,
        );
        assert_array_equal(&tester.invoke_array_array(arg1, arg2).unwrap(), &expected);
    }

    #[rstest]
    fn difference_udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_difference", st_difference_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type.clone(), sedona_type]);
        tester.assert_return_type(WKB_GEOMETRY);

        let result = tester
            .invoke_scalar_scalar("LINESTRING (50 100, 50 200)", "LINESTRING (50 50, 50 150)")
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING (50 150, 50 200)");

        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());

        let arg1 = create_array(
            &[
                Some("POLYGON ((1 1, 8 1, 8 8, 1 8, 1 1))"),
                Some("MULTIPOINT (1 2, 2 2)"),
                Some("POINT (0 0)"),
            ],
            &WKB_GEOMETRY,
        );
        let arg2 = create_array(
            &[Some("POINT(-2 3)"), Some("POINT(1 2)"), None],
            &WKB_GEOMETRY,
        );
        let expected = create_array(
            &[
                Some("POLYGON ((1 8, 8 8, 8 1, 1 1, 1 8))"),
                Some("POINT (2 2)"),
                None,
            ],
            &WKB_GEOMETRY,
        );
        assert_array_equal(&tester.invoke_array_array(arg1, arg2).unwrap(), &expected);
    }

    #[rstest]
    fn sym_difference_udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_sym_difference", st_sym_difference_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type.clone(), sedona_type]);
        tester.assert_return_type(WKB_GEOMETRY);

        let result = tester
            .invoke_scalar_scalar("LINESTRING (50 100, 50 200)", "LINESTRING (50 50, 50 150)")
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            "MULTILINESTRING ((50 150, 50 200), (50 50, 50 100))",
        );

        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());

        let arg1 = create_array(
            &[
                Some("POLYGON ((1 1, 8 1, 8 8, 1 8, 1 1))"),
                Some("LINESTRING (1 2, 2 2)"),
                Some("POINT (0 0)"),
            ],
            &WKB_GEOMETRY,
        );
        let arg2 = create_array(
            &[Some("POINT (1 1)"), Some("POINT (3 3)"), None],
            &WKB_GEOMETRY,
        );
        let expected = create_array(
            &[
                Some("POLYGON ((1 8, 8 8, 8 1, 1 1, 1 8))"),
                Some("GEOMETRYCOLLECTION (LINESTRING (1 2, 2 2), POINT (3 3))"),
                None,
            ],
            &WKB_GEOMETRY,
        );
        assert_array_equal(&tester.invoke_array_array(arg1, arg2).unwrap(), &expected);
    }
}
