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
use geo_traits::{
    GeometryCollectionTrait, GeometryTrait, GeometryType, LineStringTrait, MultiLineStringTrait,
    MultiPointTrait, MultiPolygonTrait, PointTrait, PolygonTrait,
};

/// Counts the number of points in a geometry
pub fn count_points<G: GeometryTrait>(geom: &G) -> i64 {
    match geom.as_type() {
        GeometryType::Point(pt) => PointTrait::coord(pt).is_some() as i64,
        GeometryType::MultiPoint(mp) => mp.num_points() as i64,
        GeometryType::LineString(ls) => ls.num_coords() as i64,
        GeometryType::Line(_) => 2,
        GeometryType::Polygon(poly) => {
            let mut count = 0;
            if let Some(exterior) = poly.exterior() {
                count += exterior.num_coords();
            }
            for interior in poly.interiors() {
                count += interior.num_coords();
            }
            count as i64
        }
        GeometryType::MultiLineString(mls) => {
            let mut count = 0;
            for i in 0..mls.num_line_strings() {
                if let Some(ls) = mls.line_string(i) {
                    count += ls.num_coords();
                }
            }
            count as i64
        }
        GeometryType::MultiPolygon(mp) => {
            let mut count = 0;
            for i in 0..mp.num_polygons() {
                if let Some(poly) = mp.polygon(i) {
                    if let Some(exterior) = poly.exterior() {
                        count += exterior.num_coords();
                    }
                    for interior in poly.interiors() {
                        count += interior.num_coords();
                    }
                }
            }
            count as i64
        }
        GeometryType::GeometryCollection(gc) => {
            let mut count = 0;
            for g in gc.geometries() {
                count += count_points(&g);
            }
            count
        }
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wkb_factory;
    use wkb::reader::Wkb;

    // Helper function to create WKB for tests
    fn create_test_wkb(geom_type: TestGeometry) -> Wkb<'static> {
        let wkb_bytes = match geom_type {
            TestGeometry::Point(pt) => wkb_factory::wkb_point(pt).unwrap(),
            TestGeometry::LineString(pts) => wkb_factory::wkb_linestring(pts.into_iter()).unwrap(),
            TestGeometry::Polygon(pts) => wkb_factory::wkb_polygon(pts.into_iter()).unwrap(),
            TestGeometry::MultiLineString(lines) => {
                wkb_factory::wkb_multilinestring(lines.into_iter()).unwrap()
            }
            TestGeometry::MultiPolygon(polys) => {
                wkb_factory::wkb_multipolygon(polys.into_iter()).unwrap()
            }
        };

        // Convert to static slice for testing
        let static_bytes = Box::leak(wkb_bytes.into_boxed_slice());
        Wkb::try_new(static_bytes).expect("Failed to create WKB")
    }

    // Define test geometry types
    enum TestGeometry {
        Point((f64, f64)),
        LineString(Vec<(f64, f64)>),
        Polygon(Vec<(f64, f64)>),
        MultiLineString(Vec<Vec<(f64, f64)>>),
        MultiPolygon(Vec<Vec<(f64, f64)>>),
    }

    #[test]
    fn test_count_points_point() {
        let point_wkb = create_test_wkb(TestGeometry::Point((1.0, 1.0)));
        assert_eq!(count_points(&point_wkb), 1);
    }

    #[test]
    fn test_count_points_linestring() {
        // Empty linestring
        let empty_ls_wkb = create_test_wkb(TestGeometry::LineString(vec![]));
        assert_eq!(count_points(&empty_ls_wkb), 0);

        // Single-point linestring
        let single_ls_wkb = create_test_wkb(TestGeometry::LineString(vec![(1.0, 1.0)]));
        assert_eq!(count_points(&single_ls_wkb), 1);

        // Multiple-point linestring
        let multi_ls_wkb = create_test_wkb(TestGeometry::LineString(vec![
            (1.0, 1.0),
            (2.0, 2.0),
            (3.0, 3.0),
        ]));
        assert_eq!(count_points(&multi_ls_wkb), 3);
    }

    #[test]
    fn test_count_points_polygon() {
        // Empty polygon
        let empty_poly_wkb = create_test_wkb(TestGeometry::Polygon(vec![]));
        assert_eq!(count_points(&empty_poly_wkb), 0);

        // Simple triangle
        let triangle_wkb = create_test_wkb(TestGeometry::Polygon(vec![
            (0.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (0.0, 0.0),
        ]));
        assert_eq!(count_points(&triangle_wkb), 4); // Includes closing point

        // Square
        let square_wkb = create_test_wkb(TestGeometry::Polygon(vec![
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (1.0, 0.0),
            (0.0, 0.0),
        ]));
        assert_eq!(count_points(&square_wkb), 5); // Includes closing point
    }

    #[test]
    fn test_count_points_multilinestring() {
        // Empty multilinestring
        let empty_mls_wkb = create_test_wkb(TestGeometry::MultiLineString(vec![]));
        assert_eq!(count_points(&empty_mls_wkb), 0);

        // Single linestring in multilinestring
        let single_mls_wkb = create_test_wkb(TestGeometry::MultiLineString(vec![vec![
            (1.0, 1.0),
            (2.0, 2.0),
        ]]));
        assert_eq!(count_points(&single_mls_wkb), 2);

        // Multiple linestrings in multilinestring
        let multi_mls_wkb = create_test_wkb(TestGeometry::MultiLineString(vec![
            vec![(1.0, 1.0), (2.0, 2.0)],
            vec![(3.0, 3.0), (4.0, 4.0), (5.0, 5.0)],
        ]));
        assert_eq!(count_points(&multi_mls_wkb), 5); // 2 + 3
    }

    #[test]
    fn test_count_points_multipolygon() {
        // Empty multipolygon
        let empty_mp_wkb = create_test_wkb(TestGeometry::MultiPolygon(vec![]));
        assert_eq!(count_points(&empty_mp_wkb), 0);

        // Single polygon in multipolygon
        let single_mp_wkb = create_test_wkb(TestGeometry::MultiPolygon(vec![vec![
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (1.0, 0.0),
            (0.0, 0.0),
        ]]));
        assert_eq!(count_points(&single_mp_wkb), 5);

        // Multiple polygons in multipolygon
        let multi_mp_wkb = create_test_wkb(TestGeometry::MultiPolygon(vec![
            vec![(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)],
            vec![(2.0, 2.0), (2.0, 3.0), (3.0, 3.0), (3.0, 2.0), (2.0, 2.0)],
        ]));
        assert_eq!(count_points(&multi_mp_wkb), 10); // 5 + 5
    }
}
