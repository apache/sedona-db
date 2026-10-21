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
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::ColumnarValue;
use geos::{CoordSeq, Geom, Geometry};
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

/// ST_ClosestPoint() implementation using GEOS.
pub fn st_closest_point_impl() -> Vec<ScalarKernelRef> {
    ItemCrsKernel::wrap_impl(STClosestPoint {})
}

/// ST_ShortestLine() implementation using GEOS.
pub fn st_shortest_line_impl() -> Vec<ScalarKernelRef> {
    ItemCrsKernel::wrap_impl(STShortestLine {})
}

#[derive(Debug)]
struct STClosestPoint {}

impl SedonaScalarKernel for STClosestPoint {
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

        executor.execute_wkb_wkb_void(|geom1, geom2| {
            match (geom1, geom2) {
                (Some(geom1), Some(geom2)) => match invoke_closest_point(geom1, geom2)? {
                    Some(point) => {
                        write_geos_geometry(&point, &mut builder)?;
                        builder.append_value([]);
                    }
                    None => builder.append_null(),
                },
                _ => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn nearest_points(geom1: &geos::Geometry, geom2: &geos::Geometry) -> Result<Option<CoordSeq>> {
    let geom1_empty = geom1
        .is_empty()
        .map_err(|e| DataFusionError::Execution(format!("Failed to check empty geometry: {e}")))?;
    let geom2_empty = geom2
        .is_empty()
        .map_err(|e| DataFusionError::Execution(format!("Failed to check empty geometry: {e}")))?;
    if geom1_empty || geom2_empty {
        return Ok(None);
    }

    let nearest = geom1.nearest_points(geom2).map_err(|e| {
        DataFusionError::Execution(format!("Failed to calculate nearest points: {e}"))
    })?;
    Ok(Some(nearest))
}

fn invoke_closest_point(
    geom1: &geos::Geometry,
    geom2: &geos::Geometry,
) -> Result<Option<Geometry>> {
    let Some(nearest) = nearest_points(geom1, geom2)? else {
        return Ok(None);
    };
    let x = nearest.get_x(0).map_err(|e| {
        DataFusionError::Execution(format!("Failed to read closest point x coordinate: {e}"))
    })?;
    let y = nearest.get_y(0).map_err(|e| {
        DataFusionError::Execution(format!("Failed to read closest point y coordinate: {e}"))
    })?;
    let point_coords = CoordSeq::new_from_vec(&[[x, y]]).map_err(|e| {
        DataFusionError::Execution(format!("Failed to create closest point coordinates: {e}"))
    })?;
    let point = Geometry::create_point(point_coords).map_err(|e| {
        DataFusionError::Execution(format!("Failed to create closest point geometry: {e}"))
    })?;

    Ok(Some(point))
}

#[derive(Debug)]
struct STShortestLine {}

impl SedonaScalarKernel for STShortestLine {
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

        executor.execute_wkb_wkb_void(|geom1, geom2| {
            match (geom1, geom2) {
                (Some(geom1), Some(geom2)) => match nearest_points(geom1, geom2)? {
                    Some(nearest) => {
                        let line = Geometry::create_line_string(nearest).map_err(|e| {
                            DataFusionError::Execution(format!(
                                "Failed to create shortest line geometry: {e}"
                            ))
                        })?;
                        write_geos_geometry(&line, &mut builder)?;
                        builder.append_value([]);
                    }
                    None => builder.append_null(),
                },
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
    use sedona_schema::datatypes::{
        SedonaType, WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS, WKB_VIEW_GEOMETRY,
    };
    use sedona_testing::{
        compare::assert_array_equal, create::create_array, testers::ScalarUdfTester,
    };

    use super::*;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_impl("st_closestpoint", st_closest_point_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type.clone(), sedona_type]);
        tester.assert_return_type(WKB_GEOMETRY);

        let result = tester
            .invoke_scalar_scalar("LINESTRING (0 0, 10 10)", "POINT (5 0)")
            .unwrap();
        tester.assert_scalar_result_equals(result, "POINT (2.5 2.5)");

        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());

        let geom1 = create_array(
            &[
                Some("POINT (5 0)"),
                Some("POINT EMPTY"),
                Some("POINT (0 0)"),
                None,
            ],
            &WKB_GEOMETRY,
        );
        let geom2 = create_array(
            &[
                Some("LINESTRING (0 0, 10 10)"),
                Some("POINT (0 0)"),
                Some("POINT EMPTY"),
                Some("POINT (0 0)"),
            ],
            &WKB_GEOMETRY,
        );
        let expected = create_array(&[Some("POINT (5 0)"), None, None, None], &WKB_GEOMETRY);
        assert_array_equal(&tester.invoke_array_array(geom1, geom2).unwrap(), &expected);
    }

    #[test]
    fn udf_invoke_item_crs() {
        let sedona_type = WKB_GEOMETRY_ITEM_CRS.clone();
        let udf = SedonaScalarUDF::from_impl("st_closestpoint", st_closest_point_impl());
        let tester =
            ScalarUdfTester::new(udf.into(), vec![sedona_type.clone(), sedona_type.clone()]);
        tester.assert_return_type(sedona_type);

        let result = tester
            .invoke_scalar_scalar("LINESTRING (0 0, 10 10)", "POINT (5 0)")
            .unwrap();
        tester.assert_scalar_result_equals(result, "POINT (2.5 2.5)");
    }

    #[rstest]
    fn shortest_line_udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_impl("st_shortestline", st_shortest_line_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type.clone(), sedona_type]);
        tester.assert_return_type(WKB_GEOMETRY);

        let result = tester
            .invoke_scalar_scalar("LINESTRING (0 0, 10 10)", "POINT (5 0)")
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING (2.5 2.5, 5 0)");

        let geom1 = create_array(
            &[
                Some("POINT (5 0)"),
                Some("POINT EMPTY"),
                Some("POINT (0 0)"),
                None,
            ],
            &WKB_GEOMETRY,
        );
        let geom2 = create_array(
            &[
                Some("LINESTRING (0 0, 10 10)"),
                Some("POINT (0 0)"),
                Some("POINT EMPTY"),
                Some("POINT (0 0)"),
            ],
            &WKB_GEOMETRY,
        );
        let expected = create_array(
            &[Some("LINESTRING (5 0, 2.5 2.5)"), None, None, None],
            &WKB_GEOMETRY,
        );
        assert_array_equal(&tester.invoke_array_array(geom1, geom2).unwrap(), &expected);
    }

    #[test]
    fn shortest_line_udf_invoke_item_crs() {
        let sedona_type = WKB_GEOMETRY_ITEM_CRS.clone();
        let udf = SedonaScalarUDF::from_impl("st_shortestline", st_shortest_line_impl());
        let tester =
            ScalarUdfTester::new(udf.into(), vec![sedona_type.clone(), sedona_type.clone()]);
        tester.assert_return_type(sedona_type);

        let result = tester
            .invoke_scalar_scalar("LINESTRING (0 0, 10 10)", "POINT (5 0)")
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING (2.5 2.5, 5 0)");
    }
}
