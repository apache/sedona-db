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
use datafusion_expr::{ColumnarValue, Volatility};
use geo_traits::{
    CoordTrait, GeometryCollectionTrait, GeometryTrait, GeometryType, LineStringTrait,
    MultiLineStringTrait, MultiPointTrait, MultiPolygonTrait, PointTrait, PolygonTrait,
};
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{SedonaScalarKernel, SedonaScalarUDF},
};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};
use wkb::reader::Wkb;

use crate::executor::WkbExecutor;

/// ST_MaxDistance() scalar UDF
///
/// Native implementation to calculate max pairwise 2D vertex distance using geo-traits iteration
pub fn st_max_distance_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_maxdistance",
        ItemCrsKernel::wrap_impl(STMaxDistance {}),
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct STMaxDistance {}

impl SedonaScalarKernel for STMaxDistance {
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
        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = Float64Builder::with_capacity(executor.num_iterations());
        let mut lhs_coords = Vec::new();
        let mut rhs_coords = Vec::new();

        executor.execute_wkb_wkb_void(|lhs, rhs| {
            match (lhs, rhs) {
                (Some(lhs), Some(rhs)) => {
                    match invoke_scalar(lhs, rhs, &mut lhs_coords, &mut rhs_coords) {
                        Some(dist) => builder.append_value(dist),
                        None => builder.append_null(),
                    }
                }
                _ => builder.append_null(),
            }
            Ok(())
        })?;
        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(
    lhs: &Wkb,
    rhs: &Wkb,
    lhs_coords: &mut Vec<(f64, f64)>,
    rhs_coords: &mut Vec<(f64, f64)>,
) -> Option<f64> {
    lhs_coords.clear();
    rhs_coords.clear();

    collect_coords(lhs, lhs_coords);
    collect_coords(rhs, rhs_coords);

    if lhs_coords.is_empty() || rhs_coords.is_empty() {
        return None;
    }

    let mut max_dist_sq = f64::NEG_INFINITY;
    for a in lhs_coords {
        for b in &*rhs_coords {
            let dx = a.0 - b.0;
            let dy = a.1 - b.1;
            let d2 = dx * dx + dy * dy;
            if d2 > max_dist_sq {
                max_dist_sq = d2;
            }
        }
    }

    Some(max_dist_sq.max(0.0).sqrt())
}

fn collect_coords(geom: &impl GeometryTrait<T = f64>, coords: &mut Vec<(f64, f64)>) {
    match geom.as_type() {
        GeometryType::Point(point) => {
            if let Some(coord) = point.coord() {
                coords.push(coord.x_y());
            }
        }
        GeometryType::LineString(line_string) => collect_line_string(line_string, coords),
        GeometryType::Polygon(polygon) => {
            if let Some(exterior) = polygon.exterior() {
                collect_line_string(&exterior, coords);
            }
            for interior in polygon.interiors() {
                collect_line_string(&interior, coords);
            }
        }
        GeometryType::MultiPoint(multi_point) => {
            for point in multi_point.points() {
                if let Some(coord) = point.coord() {
                    coords.push(coord.x_y());
                }
            }
        }
        GeometryType::MultiLineString(multi_line_string) => {
            for line_string in multi_line_string.line_strings() {
                collect_line_string(&line_string, coords);
            }
        }
        GeometryType::MultiPolygon(multi_polygon) => {
            for polygon in multi_polygon.polygons() {
                if let Some(exterior) = polygon.exterior() {
                    collect_line_string(&exterior, coords);
                }
                for interior in polygon.interiors() {
                    collect_line_string(&interior, coords);
                }
            }
        }
        GeometryType::GeometryCollection(geometry_collection) => {
            for child in geometry_collection.geometries() {
                collect_coords(&child, coords);
            }
        }
        _ => {}
    }
}

fn collect_line_string<LS: LineStringTrait<T = f64>>(
    line_string: &LS,
    coords: &mut Vec<(f64, f64)>,
) {
    coords.extend(line_string.coords().map(|coord| coord.x_y()));
}

#[cfg(test)]
mod tests {
    use arrow_array::{create_array as arrow_array, ArrayRef};
    use datafusion_common::ScalarValue;
    use rstest::rstest;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS, WKB_VIEW_GEOMETRY};
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::create::create_array;
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(
            st_max_distance_udf().into(),
            vec![sedona_type.clone(), sedona_type],
        );
        tester.assert_return_type(DataType::Float64);

        let result = tester
            .invoke_scalar_scalar("POINT (0 0)", "POINT (3 4)")
            .unwrap();
        tester.assert_scalar_result_equals(result, 5.0);

        let result = tester
            .invoke_scalar_scalar("POINT (0 0)", "LINESTRING (0 0, 0 2)")
            .unwrap();
        tester.assert_scalar_result_equals(result, 2.0);

        let result = tester
            .invoke_scalar_scalar("LINESTRING (0 0, 10 0)", "LINESTRING (0 10, 10 10)")
            .unwrap();
        tester.assert_scalar_result_equals(result, 200f64.sqrt());

        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());

        let result = tester
            .invoke_scalar_scalar("LINESTRING EMPTY", "POINT (0 0)")
            .unwrap();
        assert!(result.is_null(), "expected NULL for empty lhs");

        let result = tester
            .invoke_scalar_scalar("POINT EMPTY", "POINT (0 0)")
            .unwrap();
        assert!(result.is_null(), "expected NULL for empty point lhs");

        let result = tester
            .invoke_scalar_scalar("POINT (0 0)", "LINESTRING EMPTY")
            .unwrap();
        assert!(result.is_null(), "expected NULL for empty rhs");

        let result = tester
            .invoke_scalar_scalar("POINT (0 0)", "POINT EMPTY")
            .unwrap();
        assert!(result.is_null(), "expected NULL for empty point rhs");

        let result = tester
            .invoke_scalar_scalar("LINESTRING EMPTY", "LINESTRING EMPTY")
            .unwrap();
        assert!(result.is_null(), "expected NULL for both empty");

        let lhs = create_array(
            &[Some("POINT (0 0)"), Some("POINT (0 0)"), None],
            &WKB_GEOMETRY,
        );
        let rhs = create_array(
            &[Some("POINT (3 4)"), None, Some("POINT (1 1)")],
            &WKB_GEOMETRY,
        );
        let expected: ArrayRef = arrow_array!(Float64, [Some(5.0), None, None]);
        assert_array_equal(&tester.invoke_array_array(lhs, rhs).unwrap(), &expected);
    }

    #[rstest]
    fn udf_invoke_item_crs(#[values(WKB_GEOMETRY_ITEM_CRS.clone())] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(
            st_max_distance_udf().into(),
            vec![sedona_type.clone(), sedona_type],
        );
        tester.assert_return_type(DataType::Float64);

        let result = tester
            .invoke_scalar_scalar("POINT (0 0)", "POINT (3 4)")
            .unwrap();
        tester.assert_scalar_result_equals(result, 5.0);
    }
}
