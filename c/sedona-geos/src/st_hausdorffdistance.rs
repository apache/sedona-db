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
use datafusion_common::{
    cast::as_float64_array, error::Result, exec_datafusion_err, DataFusionError,
};
use datafusion_expr::ColumnarValue;
use geos::Geom;
use sedona_common::sedona_internal_datafusion_err;
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{ScalarKernelRef, SedonaScalarKernel},
};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::executor::GeosExecutor;

/// ST_HausdorffDistance(geometry, geometry) implementation using the geos crate
pub fn st_hausdorff_distance_impl() -> Vec<ScalarKernelRef> {
    ItemCrsKernel::wrap_impl(STHausdorffDistance {})
}

/// ST_HausdorffDistance(geometry, geometry, densifyFrac) implementation using the geos crate
pub fn st_hausdorff_distance_densify_impl() -> Vec<ScalarKernelRef> {
    ItemCrsKernel::wrap_impl(STHausdorffDistanceDensify {})
}

#[derive(Debug)]
struct STHausdorffDistance {}

impl SedonaScalarKernel for STHausdorffDistance {
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
                    if let Some(distance) = invoke_scalar(lhs, rhs)? {
                        builder.append_value(distance);
                    } else {
                        builder.append_null();
                    }
                }
                _ => builder.append_null(),
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(lhs: &geos::Geometry, rhs: &geos::Geometry) -> Result<Option<f64>> {
    // Return NULL for empty geometries
    if lhs
        .is_empty()
        .map_err(|e| sedona_internal_datafusion_err!("is_empty() call failed: {e}"))?
        || rhs
            .is_empty()
            .map_err(|e| sedona_internal_datafusion_err!("is_empty() call failed: {e}"))?
    {
        return Ok(None);
    }
    let distance = lhs
        .hausdorff_distance(rhs)
        .map_err(|e| exec_datafusion_err!("Failed to calculate hausdorff distance: {e}"))?;
    Ok(Some(distance))
}

#[derive(Debug)]
struct STHausdorffDistanceDensify {}

impl SedonaScalarKernel for STHausdorffDistanceDensify {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_geometry(),
                ArgMatcher::is_geometry(),
                ArgMatcher::is_numeric(),
            ],
            SedonaType::Arrow(DataType::Float64),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let arg2 = args[2].cast_to(&DataType::Float64, None)?;
        let executor = GeosExecutor::new(arg_types, args);
        let arg2_array = arg2.to_array(executor.num_iterations())?;
        let arg2_f64_array = as_float64_array(&arg2_array)?;
        let mut arg2_iter = arg2_f64_array.iter();
        let mut builder = Float64Builder::with_capacity(executor.num_iterations());
        executor.execute_wkb_wkb_void(|lhs, rhs| {
            match (lhs, rhs, arg2_iter.next().unwrap()) {
                (Some(lhs), Some(rhs), Some(densify_frac)) => {
                    if let Some(distance) = invoke_scalar_densify(lhs, rhs, densify_frac)? {
                        builder.append_value(distance);
                    } else {
                        builder.append_null();
                    }
                }
                _ => builder.append_null(),
            };
            Ok(())
        })?;
        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar_densify(
    lhs: &geos::Geometry,
    rhs: &geos::Geometry,
    densify_frac: f64,
) -> Result<Option<f64>> {
    // Return NULL for empty geometries (PostGIS compatibility)
    if lhs
        .is_empty()
        .map_err(|e| sedona_internal_datafusion_err!("is_empty() call failed: {e}"))?
        || rhs
            .is_empty()
            .map_err(|e| sedona_internal_datafusion_err!("is_empty() call failed: {e}"))?
    {
        return Ok(None);
    }

    let distance = lhs
        .hausdorff_distance_densify(rhs, densify_frac)
        .map_err(|e| {
            DataFusionError::Execution(format!("Failed to calculate hausdorff distance: {e}"))
        })?;
    Ok(Some(distance))
}

#[cfg(test)]
mod tests {
    use arrow_array::{create_array as arrow_array, Array, ArrayRef};
    use datafusion_common::cast::as_float64_array;
    use datafusion_common::ScalarValue;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS, WKB_VIEW_GEOMETRY};
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::create::create_array;
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_impl("st_hausdorffdistance", st_hausdorff_distance_impl());
        let tester =
            ScalarUdfTester::new(udf.into(), vec![sedona_type.clone(), sedona_type.clone()]);
        tester.assert_return_type(DataType::Float64);

        // Point to point - Hausdorff distance equals Euclidean distance
        let result = tester
            .invoke_scalar_scalar("POINT (0 0)", "POINT (3 4)")
            .unwrap();
        tester.assert_scalar_result_equals(result, 5.0);

        // NULL handling
        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());

        let result = tester
            .invoke_scalar_scalar("POINT (0 0)", ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());

        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, "POINT (0 0)")
            .unwrap();
        assert!(result.is_null());

        // EMPTY geometries return NULL (PostGIS compatibility)
        let result = tester
            .invoke_scalar_scalar("POINT EMPTY", "POINT (0 0)")
            .unwrap();
        assert!(result.is_null());

        let result = tester
            .invoke_scalar_scalar("POINT EMPTY", "POINT EMPTY")
            .unwrap();
        assert!(result.is_null());

        // LineString to LineString
        let result = tester
            .invoke_scalar_scalar("LINESTRING (0 0, 2 0)", "LINESTRING (0 1, 1 1, 2 1)")
            .unwrap();
        tester.assert_scalar_result_equals(result, 1.0);

        // Polygon to Polygon
        let result = tester
            .invoke_scalar_scalar(
                "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
                "POLYGON ((2 0, 3 0, 3 1, 2 1, 2 0))",
            )
            .unwrap();
        tester.assert_scalar_result_equals(result, 2.0);

        // Array batch test with multiple geometry types
        let arg1 = create_array(
            &[
                Some("POINT (0 0)"),
                Some("LINESTRING (0 0, 1 0)"),
                Some("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"),
                Some("MULTIPOINT ((0 0), (1 0))"),
                Some("MULTILINESTRING ((0 0, 1 0), (0 1, 1 1))"),
                Some("MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)))"),
                Some("GEOMETRYCOLLECTION (POINT (0 0))"),
                None,
            ],
            &sedona_type,
        );
        let arg2 = create_array(
            &[
                Some("POINT (3 4)"),
                Some("LINESTRING (0 1, 1 1)"),
                Some("POLYGON ((2 0, 3 0, 3 1, 2 1, 2 0))"),
                Some("MULTIPOINT ((3 4), (4 4))"),
                Some("MULTILINESTRING ((0 2, 1 2))"),
                Some("MULTIPOLYGON (((2 0, 3 0, 3 1, 2 1, 2 0)))"),
                Some("GEOMETRYCOLLECTION (POINT (3 4))"),
                Some("POINT (0 0)"),
            ],
            &sedona_type,
        );
        let expected: ArrayRef = arrow_array!(
            Float64,
            [
                Some(5.0), // Point to Point
                Some(1.0), // LineString to LineString
                Some(2.0), // Polygon to Polygon
                Some(5.0), // MultiPoint to MultiPoint
                Some(2.0), // MultiLineString to MultiLineString
                Some(2.0), // MultiPolygon to MultiPolygon
                Some(5.0), // GeometryCollection to GeometryCollection
                None       // NULL
            ]
        );
        assert_array_equal(&tester.invoke_arrays(vec![arg1, arg2]).unwrap(), &expected);
    }

    #[rstest]
    fn udf_densify(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY, WKB_GEOMETRY_ITEM_CRS.clone())]
        sedona_type: SedonaType,
    ) {
        let udf = SedonaScalarUDF::from_impl(
            "st_hausdorffdistance",
            st_hausdorff_distance_densify_impl(),
        );
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![
                sedona_type.clone(),
                sedona_type.clone(),
                SedonaType::Arrow(DataType::Float64),
            ],
        );
        tester.assert_return_type(DataType::Float64);

        // With densify fraction - this example shows different results for different fractions
        // The densify fraction of 0.5 adds interpolated points, improving the accuracy
        let result = tester
            .invoke_scalar_scalar_scalar(
                "LINESTRING (130 0, 0 0, 0 150)",
                "LINESTRING (10 10, 10 150, 130 10)",
                0.5,
            )
            .unwrap();
        let result_f64: f64 = result.try_into().unwrap();
        assert!(
            (result_f64 - 70.0).abs() < 0.01,
            "Expected ~70.0, got {result_f64}"
        );

        // NULL handling
        let result = tester
            .invoke_scalar_scalar_scalar(ScalarValue::Null, ScalarValue::Null, ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());

        // Array batch test - varying densify fractions show different results
        let arg1 = create_array(
            &[
                Some("LINESTRING (130 0, 0 0, 0 150)"),
                Some("LINESTRING (130 0, 0 0, 0 150)"),
                Some("LINESTRING (130 0, 0 0, 0 150)"),
                None,
                Some("POINT EMPTY"),
            ],
            &sedona_type,
        );
        let arg2 = create_array(
            &[
                Some("LINESTRING (10 10, 10 150, 130 10)"),
                Some("LINESTRING (10 10, 10 150, 130 10)"),
                Some("LINESTRING (10 10, 10 150, 130 10)"),
                Some("LINESTRING (0 0, 1 1)"),
                Some("POINT EMPTY"),
            ],
            &sedona_type,
        );
        // Different densify fractions: smaller fractions create more segments, improving accuracy
        let densify_frac = arrow_array!(
            Float64,
            [Some(0.5), Some(0.25), Some(0.1), Some(0.5), Some(0.5)]
        );

        let result = tester
            .invoke_arrays(vec![arg1, arg2, densify_frac])
            .unwrap();
        assert_eq!(result.len(), 5);
        // First three results should all be close to ~70.0 (the continuous Hausdorff distance)
        let result_array = as_float64_array(&result).unwrap();
        assert!(
            (result_array.value(0) - 70.0).abs() < 0.01,
            "Expected ~70.0, got {}",
            result_array.value(0)
        );
        assert!(
            (result_array.value(1) - 70.0).abs() < 0.01,
            "Expected ~70.0, got {}",
            result_array.value(1)
        );
        assert!(
            (result_array.value(2) - 70.0).abs() < 0.01,
            "Expected ~70.0, got {}",
            result_array.value(2)
        );
        assert!(result_array.is_null(3)); // NULL input
        assert!(result_array.is_null(4)); // EMPTY geometry
    }

    #[test]
    fn udf_densify_invalid_frac() {
        let udf = SedonaScalarUDF::from_impl(
            "st_hausdorffdistance",
            st_hausdorff_distance_densify_impl(),
        );
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![
                WKB_GEOMETRY,
                WKB_GEOMETRY,
                SedonaType::Arrow(DataType::Float64),
            ],
        );

        // densifyFrac less than 0 should error
        let result = tester.invoke_scalar_scalar_scalar(
            "LINESTRING (0 0, 100 0)",
            "LINESTRING (0 1, 100 1)",
            -0.1,
        );
        assert!(result.is_err());

        // densifyFrac greater than 1 should error
        let result = tester.invoke_scalar_scalar_scalar(
            "LINESTRING (0 0, 100 0)",
            "LINESTRING (0 1, 100 1)",
            1.5,
        );
        assert!(result.is_err());
    }
}
