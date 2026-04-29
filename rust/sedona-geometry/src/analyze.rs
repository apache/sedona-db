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
use crate::{
    error::SedonaGeometryError,
    point_count::count_points,
    types::{GeometryTypeAndDimensions, GeometryTypeId},
};
use wkb::reader::Wkb;

/// Captures the size and type-derived counts for a single geometry.
/// Used as the per-geometry input that eventually feeds aggregated `GeoStatistics`.
#[derive(Debug, Clone)]
pub struct GeometrySummary {
    pub size_bytes: usize,
    pub point_count: i64,
    pub geometry_type: GeometryTypeAndDimensions,
    pub puntal_count: i64,
    pub lineal_count: i64,
    pub polygonal_count: i64,
    pub collection_count: i64,
}

/// Analyzes a WKB geometry and returns its size, point count, and dimensions
pub fn analyze_wkb(geom: &Wkb) -> Result<GeometrySummary, SedonaGeometryError> {
    // Get size in bytes directly from WKB buffer
    let size_bytes = geom.buf().len();

    // Get point count directly using the geometry traits
    let point_count = count_points(geom);

    // Derive geometry type and dimensions from the geometry
    let geometry_type = GeometryTypeAndDimensions::try_from_geom(geom)?;

    // Determine geometry type counts directly
    let puntal_count = matches!(
        geometry_type.geometry_type(),
        GeometryTypeId::Point | GeometryTypeId::MultiPoint
    ) as i64;

    let lineal_count = matches!(
        geometry_type.geometry_type(),
        GeometryTypeId::LineString | GeometryTypeId::MultiLineString
    ) as i64;

    let polygonal_count = matches!(
        geometry_type.geometry_type(),
        GeometryTypeId::Polygon | GeometryTypeId::MultiPolygon
    ) as i64;

    let collection_count = matches!(
        geometry_type.geometry_type(),
        GeometryTypeId::GeometryCollection
    ) as i64;

    Ok(GeometrySummary {
        size_bytes,
        point_count,
        geometry_type,
        puntal_count,
        lineal_count,
        polygonal_count,
        collection_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::types::GeometryTypeId;
    use crate::wkb_factory;
    use geo_traits::Dimensions;

    // Helper function to create WKB for tests
    fn create_test_wkb(geom_type: TestGeometry) -> Wkb<'static> {
        let wkb_bytes = match geom_type {
            TestGeometry::Point(pt) => wkb_factory::wkb_point(pt).unwrap(),
            TestGeometry::LineString(pts) => wkb_factory::wkb_linestring(pts.into_iter()).unwrap(),
            TestGeometry::Polygon(pts) => wkb_factory::wkb_polygon(pts.into_iter()).unwrap(),
            TestGeometry::MultiLineString(lines) => {
                wkb_factory::wkb_multilinestring(lines.into_iter()).unwrap()
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
    }

    #[test]
    fn test_analyze_wkb() {
        // Test point
        let point_wkb = create_test_wkb(TestGeometry::Point((1.0, 2.0)));
        let point_analysis = analyze_wkb(&point_wkb).unwrap();

        assert_eq!(point_analysis.point_count, 1);
        assert_eq!(
            point_analysis.geometry_type,
            GeometryTypeAndDimensions::new(GeometryTypeId::Point, Dimensions::Xy)
        );
        assert!(point_analysis.size_bytes > 0);

        // Test linestring
        let linestring_wkb = create_test_wkb(TestGeometry::LineString(vec![
            (1.0, 2.0),
            (3.0, 4.0),
            (5.0, 6.0),
        ]));
        let linestring_analysis = analyze_wkb(&linestring_wkb).unwrap();

        assert_eq!(linestring_analysis.point_count, 3);
        assert_eq!(
            linestring_analysis.geometry_type,
            GeometryTypeAndDimensions::new(GeometryTypeId::LineString, Dimensions::Xy)
        );
        assert!(linestring_analysis.size_bytes > 0);

        // Test polygon
        let polygon_wkb = create_test_wkb(TestGeometry::Polygon(vec![
            (0.0, 0.0),
            (0.0, 3.0),
            (3.0, 3.0),
            (3.0, 0.0),
            (0.0, 0.0),
        ]));
        let polygon_analysis = analyze_wkb(&polygon_wkb).unwrap();

        assert_eq!(polygon_analysis.point_count, 5);
        assert_eq!(
            polygon_analysis.geometry_type,
            GeometryTypeAndDimensions::new(GeometryTypeId::Polygon, Dimensions::Xy)
        );
        assert!(polygon_analysis.size_bytes > 0);

        // Test multilinestring (as a substitute for collection for now)
        let multilinestring_wkb = create_test_wkb(TestGeometry::MultiLineString(vec![
            vec![(1.0, 1.0)],             // Point equivalent
            vec![(0.0, 0.0), (2.0, 2.0)], // LineString
        ]));
        let multilinestring_analysis = analyze_wkb(&multilinestring_wkb).unwrap();

        assert_eq!(multilinestring_analysis.point_count, 3); // 1 for point + 2 for linestring
        assert_eq!(
            multilinestring_analysis.geometry_type,
            GeometryTypeAndDimensions::new(GeometryTypeId::MultiLineString, Dimensions::Xy)
        );
        assert!(multilinestring_analysis.size_bytes > 0);

        // Test empty multilinestring
        let empty_multilinestring_wkb = create_test_wkb(TestGeometry::MultiLineString(vec![]));
        let empty_multilinestring_analysis = analyze_wkb(&empty_multilinestring_wkb).unwrap();

        assert_eq!(empty_multilinestring_analysis.point_count, 0);
        assert_eq!(
            empty_multilinestring_analysis.geometry_type,
            GeometryTypeAndDimensions::new(GeometryTypeId::MultiLineString, Dimensions::Xy)
        );
    }

    #[test]
    fn test_empty_geometries() {
        // Test empty linestring
        let empty_linestring_wkb = create_test_wkb(TestGeometry::LineString(vec![]));
        let empty_linestring_analysis = analyze_wkb(&empty_linestring_wkb).unwrap();

        assert_eq!(empty_linestring_analysis.point_count, 0);
        assert_eq!(empty_linestring_analysis.lineal_count, 1);

        // Test empty polygon
        let empty_polygon_wkb = create_test_wkb(TestGeometry::Polygon(vec![]));
        let empty_polygon_analysis = analyze_wkb(&empty_polygon_wkb).unwrap();

        assert_eq!(empty_polygon_analysis.point_count, 0);
        assert_eq!(empty_polygon_analysis.polygonal_count, 1);
    }
}
