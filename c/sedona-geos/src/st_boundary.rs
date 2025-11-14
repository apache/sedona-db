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
use geos::{Geom, Geometry, GeometryTypes};
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_geometry::wkb_factory::WKB_MIN_PROBABLE_BYTES;
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

use crate::executor::GeosExecutor;

/// ST_Boundary() implementation using the geos crate
pub fn st_boundary_impl() -> ScalarKernelRef {
    Arc::new(STBoundary {})
}

#[derive(Debug)]
struct STBoundary {}

impl SedonaScalarKernel for STBoundary {
    fn return_type(&self, args: &[SedonaType]) -> datafusion_common::Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOMETRY);

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> datafusion_common::Result<ColumnarValue> {
        let executor = GeosExecutor::new(arg_types, args);
        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
        );

        executor.execute_wkb_void(|maybe_wkb| {
            match maybe_wkb {
                Some(wkb) => {
                    invoke_scalar(&wkb, &mut builder)?;
                    builder.append_value([]);
                }
                _ => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(geos_geom: &geos::Geometry, writer: &mut BinaryBuilder) -> Result<()> {
    let result_geom = geos_boundary(geos_geom)?;

    let wkb = result_geom
        .to_wkb()
        .map_err(|e| DataFusionError::Execution(format!("Failed to convert to wkb: {e}")))?;

    writer.append_value(wkb.as_ref());
    Ok(())
}

/// For simple geometries, it calls `geometry.boundary()`.
/// For a `GeometryCollection`, it recursively computes the boundary for each
/// component and then re-aggregates the resulting geometry types (Point, LineString, etc.)
/// into their respective Multi-geometry types (MultiPoint, MultiLineString, etc.)
/// before forming the final `GeometryCollection`. This aggregation step is crucial
/// for adhering to OGC specifications for `GeometryCollection` boundaries.
fn geos_boundary(geometry: &impl Geom) -> Result<Geometry> {
    if geometry.geometry_type() == GeometryTypes::GeometryCollection {
        let num_geometries = geometry.get_num_geometries().map_err(|e| {
            DataFusionError::Execution(format!("Failed to get number of geometries: {e}"))
        })?;

        let mut empty_collections: Vec<Geometry> = Vec::new();
        let mut points: Vec<Geometry> = Vec::new();
        let mut lines: Vec<Geometry> = Vec::new();
        let mut polygons: Vec<Geometry> = Vec::new();

        for i in 0..num_geometries {
            let child_geom = geometry.get_geometry_n(i).map_err(|e| {
                DataFusionError::Execution(format!("Failed to get {}th child geometry: {e}", i + 1))
            })?;

            // Recursively calculate the boundary of the child geometry
            let child_boundary = geos_boundary(&child_geom)?;

            // Collect components based on whether they are an empty GeometryCollection
            if is_empty_geometry_collection(&child_boundary)? {
                empty_collections.push(child_boundary);
            } else {
                // Collect and group non-empty boundary components (Points, LineStrings, etc.)
                collect_boundary_components(
                    &child_boundary,
                    &mut points,
                    &mut lines,
                    &mut polygons,
                )?;
            }
        }

        let mut result_components: Vec<Geometry> = Vec::new();

        // Aggregate the result
        result_components.extend(empty_collections);

        if points.len() == 1 {
            result_components.push(points.into_iter().next().unwrap());
        } else if !points.is_empty() {
            let multi_point = Geometry::create_multipoint(points).map_err(|e| {
                DataFusionError::Execution(format!("Failed to create multipoint: {e}"))
            })?;
            result_components.push(multi_point);
        }

        if lines.len() == 1 {
            result_components.push(lines.into_iter().next().unwrap());
        } else if !lines.is_empty() {
            let multi_line = Geometry::create_multiline_string(lines).map_err(|e| {
                DataFusionError::Execution(format!("Failed to create multilinestring: {e}"))
            })?;
            result_components.push(multi_line);
        }

        if polygons.len() == 1 {
            result_components.push(polygons.into_iter().next().unwrap());
        } else if !polygons.is_empty() {
            let multi_polygon = Geometry::create_multipolygon(polygons).map_err(|e| {
                DataFusionError::Execution(format!("Failed to create multipolygon: {e}"))
            })?;
            result_components.push(multi_polygon);
        }

        if result_components.len() == 1 {
            Ok(Geom::clone(result_components.first().unwrap()))
        } else {
            Geometry::create_geometry_collection(result_components).map_err(|e| {
                DataFusionError::Execution(format!("Failed to create geometry collection: {e}"))
            })
        }
    } else {
        // For simple geometries, use the standard geos boundary function
        geometry
            .boundary()
            .map_err(|e| DataFusionError::Execution(format!("Failed to calculate boundary: {e}")))
    }
}

/// Checks if a geometry is an empty `GeometryCollection`.
fn is_empty_geometry_collection(geom: &Geometry) -> Result<bool> {
    if geom.geometry_type() == GeometryTypes::GeometryCollection {
        let num = geom.get_num_geometries().map_err(|e| {
            DataFusionError::Execution(format!("Failed to get number of geometries: {e}"))
        })?;
        Ok(num == 0)
    } else {
        Ok(false)
    }
}

/// Recursively collects and groups individual boundary components (Point, LineString, etc.)
/// from a given boundary geometry (which could be a `GeometryCollection` itself)
/// into mutable vectors based on their type.
fn collect_boundary_components(
    boundary: &Geometry,
    points: &mut Vec<Geometry>,
    lines: &mut Vec<Geometry>,
    polygons: &mut Vec<Geometry>,
) -> Result<()> {
    match boundary.geometry_type() {
        // Recurse into sub-geometries if it's a collection
        GeometryTypes::GeometryCollection => {
            let num_geoms = boundary.get_num_geometries().map_err(|e| {
                DataFusionError::Execution(format!("Failed to get number of geometries: {e}"))
            })?;

            for i in 0..num_geoms {
                let component = boundary.get_geometry_n(i).map_err(|e| {
                    DataFusionError::Execution(format!("Failed to get {}th geometry: {e}", i + 1))
                })?;
                let owned_component = Geom::clone(&component);
                collect_boundary_components(&owned_component, points, lines, polygons)?;
            }
        }
        // Collect simple, single-part components
        GeometryTypes::Point => {
            points.push(Geom::clone(boundary));
        }
        GeometryTypes::LineString => {
            lines.push(Geom::clone(boundary));
        }
        GeometryTypes::Polygon => {
            polygons.push(Geom::clone(boundary));
        }
        // Decompose Multi-geometries and collect their parts
        GeometryTypes::MultiPoint => {
            let num_points = boundary.get_num_geometries().map_err(|e| {
                DataFusionError::Execution(format!("Failed to get number of points: {e}"))
            })?;
            for i in 0..num_points {
                let point = boundary.get_geometry_n(i).map_err(|e| {
                    DataFusionError::Execution(format!("Failed to get {}th point: {e}", i + 1))
                })?;
                points.push(Geom::clone(&point));
            }
        }
        GeometryTypes::MultiLineString => {
            let num_lines = boundary.get_num_geometries().map_err(|e| {
                DataFusionError::Execution(format!("Failed to get number of linestrings: {e}"))
            })?;
            for i in 0..num_lines {
                let line = boundary.get_geometry_n(i).map_err(|e| {
                    DataFusionError::Execution(format!("Failed to get {}th linestring: {e}", i + 1))
                })?;
                lines.push(Geom::clone(&line));
            }
        }
        GeometryTypes::MultiPolygon => {
            let num_polygons = boundary.get_num_geometries().map_err(|e| {
                DataFusionError::Execution(format!("Failed to get number of polygons: {e}"))
            })?;
            for i in 0..num_polygons {
                let polygon = boundary.get_geometry_n(i).map_err(|e| {
                    DataFusionError::Execution(format!("Failed to get {}th polygon: {e}", i + 1))
                })?;
                polygons.push(Geom::clone(&polygon));
            }
        }
        // Ignore other types (e.g., empty geometries)
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_boundary", st_boundary_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type]);
        tester.assert_return_type(WKB_GEOMETRY);

        let result = tester
            .invoke_scalar(
                "GEOMETRYCOLLECTION(LINESTRING(1 1,2 2),GEOMETRYCOLLECTION(POLYGON((3 3,4 4,5 5,3 3)),GEOMETRYCOLLECTION(LINESTRING(6 6,7 7),POLYGON((8 8,9 9,10 10,8 8)))))",
            )
            .unwrap();
        tester.assert_scalar_result_equals(result, "GEOMETRYCOLLECTION(MULTIPOINT((1 1),(2 2),(6 6),(7 7)),MULTILINESTRING((3 3,4 4,5 5,3 3),(8 8,9 9,10 10,8 8)))");

        let result = tester
            .invoke_scalar("LINESTRING(100 150,50 60, 70 80, 160 170)")
            .unwrap();
        tester.assert_scalar_result_equals(result, "MULTIPOINT((100 150),(160 170))");

        let result = tester
            .invoke_scalar(
                "POLYGON (( 10 130, 50 190, 110 190, 140 150, 150 80, 100 10, 20 40, 10 130 ), ( 70 40, 100 50, 120 80, 80 110, 50 90, 70 40 ))"
            )
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            "MULTILINESTRING((10 130,50 190,110 190,140 150,150 80,100 10,20 40,10 130), (70 40,100 50,120 80,80 110,50 90,70 40))"
        );

        let result = tester
            .invoke_scalar("MULTILINESTRING ((10 10, 20 20), (30 30, 40 40, 30 30))")
            .unwrap();
        tester.assert_scalar_result_equals(result, "MULTIPOINT (10 10, 20 20)");

        let result = tester.invoke_scalar("GEOMETRYCOLLECTION(MULTIPOINT(-2 3, -2 2), LINESTRING(5 5, 10 10), POLYGON((-7 4.2, -7.1 5, -7.1 4.3, -7 4.2)))").unwrap();
        tester.assert_scalar_result_equals(
            result,
            "GEOMETRYCOLLECTION(GEOMETRYCOLLECTION EMPTY, MULTIPOINT(5 5, 10 10), LINESTRING(-7 4.2, -7.1 5, -7.1 4.3, -7 4.2))"
        );

        let result = tester.invoke_scalar("POINT (10 20)").unwrap();
        tester.assert_scalar_result_equals(result, "GEOMETRYCOLLECTION EMPTY");

        let result = tester
            .invoke_scalar("MULTIPOINT (5 5, 10 10, 15 15)")
            .unwrap();
        tester.assert_scalar_result_equals(result, "GEOMETRYCOLLECTION EMPTY");

        let result = tester
            .invoke_scalar("LINESTRING (0 0, 1 1, 0 1, 0 0)")
            .unwrap();
        tester.assert_scalar_result_equals(result, "MULTIPOINT EMPTY");

        let result = tester
            .invoke_scalar("POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))")
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING(0 0,0 10,10 10,10 0,0 0)");

        let result = tester
            .invoke_scalar(
                "MULTIPOLYGON (((0 0, 0 1, 1 1, 1 0, 0 0)), ((10 10, 10 11, 11 11, 11 10, 10 10)))",
            )
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            "MULTILINESTRING((0 0,0 1,1 1,1 0,0 0),(10 10,10 11,11 11,11 10,10 10))",
        );

        let result = tester
            .invoke_scalar("GEOMETRYCOLLECTION(POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0)), GEOMETRYCOLLECTION(LINESTRING(10 10, 10 20)))")
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            "GEOMETRYCOLLECTION(MULTIPOINT((10 10),(10 20)), LINESTRING(0 0,0 1,1 1,1 0,0 0))",
        );

        let result = tester.invoke_scalar("GEOMETRYCOLLECTION EMPTY").unwrap();
        tester.assert_scalar_result_equals(result, "GEOMETRYCOLLECTION EMPTY");
    }
}
