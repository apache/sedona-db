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
//! Centroid extraction functionality for WKB geometries

use datafusion_common::{error::DataFusionError, Result};
use geo_traits::CoordTrait;
use geo_traits::GeometryTrait;
use geo_traits::PointTrait;
use sedona_geo_generic_alg::Centroid;
use sedona_geo_generic_alg::HasDimensions;

use crate::to_geo::item_to_geometry;

/// Extract the centroid from a WKB geometry in 2D space.
///
/// For Point geometries, returns the point coordinates directly.
/// For other geometry types, computes the centroid using the Centroid trait.
/// If centroid computation fails, falls back to the first coordinate of the geometry.
///
/// # Arguments
/// * `wkb` - The WKB geometry to extract centroid from
///
/// # Returns
/// * `Ok((x, y))` - The centroid coordinates
/// * `Err` - If the WKB cannot be parsed or no valid centroid can be extracted
///
pub fn extract_centroid_2d(geo: impl GeometryTrait<T = f64>) -> Result<(f64, f64)> {
    match geo.as_type() {
        geo_traits::GeometryType::Point(point) => {
            if let Some(coord) = PointTrait::coord(point) {
                Ok((coord.x(), coord.y()))
            } else {
                Ok((f64::NAN, f64::NAN))
            }
        }
        // For other geometries, compute centroid
        _ => {
            let geom = item_to_geometry(geo)?;
            if let Some(centroid) = geom.centroid() {
                Ok((centroid.x(), centroid.y()))
            } else if geom.is_empty() {
                // Return POINT EMPTY as (NaN, NaN)
                Ok((f64::NAN, f64::NAN))
            } else {
                Err(DataFusionError::Internal(
                    "Centroid computation failed.".to_string(),
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sedona_testing::create::make_wkb;
    use wkb::reader::Wkb;

    fn create_wkb_from_wkt(wkt: &str) -> Vec<u8> {
        make_wkb(wkt)
    }

    #[test]
    fn test_extract_centroid_point() {
        // Create a point at (1.0, 2.0)
        let wkb_data = create_wkb_from_wkt("POINT (1.0 2.0)");
        let wkb = Wkb::try_new(&wkb_data).unwrap();

        let (x, y) = extract_centroid_2d(&wkb).unwrap();
        assert_eq!(x, 1.0);
        assert_eq!(y, 2.0);
    }

    #[test]
    fn test_extract_centroid_linestring() {
        // Create a linestring from (0,0) to (4,0)
        let wkb_data = create_wkb_from_wkt("LINESTRING (0.0 0.0, 4.0 0.0)");
        let wkb = Wkb::try_new(&wkb_data).unwrap();

        let (x, y) = extract_centroid_2d(&wkb).unwrap();
        assert_eq!(x, 2.0);
        assert_eq!(y, 0.0);
    }

    #[test]
    fn test_extract_centroid_polygon() {
        // Create a square polygon (0,0), (2,0), (2,2), (0,2), (0,0)
        let wkb_data =
            create_wkb_from_wkt("POLYGON ((0.0 0.0, 2.0 0.0, 2.0 2.0, 0.0 2.0, 0.0 0.0))");
        let wkb = Wkb::try_new(&wkb_data).unwrap();

        let (x, y) = extract_centroid_2d(&wkb).unwrap();
        assert_eq!(x, 1.0);
        assert_eq!(y, 1.0);
    }

    #[test]
    fn test_extract_centroid_empty_linestring() {
        // Should return POINT EMPTY as (NaN, NaN)
        let wkb_data = create_wkb_from_wkt("LINESTRING EMPTY");
        let wkb = Wkb::try_new(&wkb_data).unwrap();

        let result = extract_centroid_2d(&wkb).unwrap();
        assert!(result.0.is_nan() && result.1.is_nan());
    }

    #[test]
    fn test_extract_centroid_multipoint() {
        // Create a multipoint with three points: (0,0), (2,0), (1,3)
        let wkb_data = create_wkb_from_wkt("MULTIPOINT ((0.0 0.0), (2.0 0.0), (1.0 3.0))");
        let wkb = Wkb::try_new(&wkb_data).unwrap();

        let (x, y) = extract_centroid_2d(&wkb).unwrap();
        assert_eq!(x, 1.0);
        assert_eq!(y, 1.0);
    }

    #[test]
    fn test_extract_centroid_triangle() {
        // Create a triangle polygon
        let wkb_data = create_wkb_from_wkt("POLYGON ((0.0 0.0, 3.0 0.0, 1.5 3.0, 0.0 0.0))");
        let wkb = Wkb::try_new(&wkb_data).unwrap();

        let (x, y) = extract_centroid_2d(&wkb).unwrap();
        assert_eq!(x, 1.5);
        assert_eq!(y, 1.0);
    }

    #[test]
    fn test_extract_centroid_multipolygon() {
        // Create a multipolygon with two squares: one at (0,0)-(1,1) and another at (2,0)-(3,1)
        // First square centroid: (0.5, 0.5), Second square centroid: (2.5, 0.5)
        // Combined centroid should be approximately at (1.5, 0.5)
        let wkb_data = create_wkb_from_wkt(
            "MULTIPOLYGON (((0.0 0.0, 1.0 0.0, 1.0 1.0, 0.0 1.0, 0.0 0.0)), \
                           ((2.0 0.0, 3.0 0.0, 3.0 1.0, 2.0 1.0, 2.0 0.0)))",
        );
        let wkb = Wkb::try_new(&wkb_data).unwrap();

        let (x, y) = extract_centroid_2d(&wkb).unwrap();
        assert_eq!(x, 1.5);
        assert_eq!(y, 0.5);
    }

    #[test]
    fn test_extract_centroid_geometry_collection() {
        // Create a geometry collection with a point and a linestring
        let wkb_data = create_wkb_from_wkt(
            "GEOMETRYCOLLECTION (POINT (0.0 0.0), LINESTRING (2.0 0.0, 4.0 0.0))",
        );
        let wkb = Wkb::try_new(&wkb_data).unwrap();

        let (x, y) = extract_centroid_2d(&wkb).unwrap();
        assert!(x.is_finite());
        assert!(y.is_finite());
    }

    #[test]
    fn test_extract_centroid_multipoint_empty() {
        // Test with empty multipoint
        let wkb_data = create_wkb_from_wkt("MULTIPOINT EMPTY");
        let wkb = Wkb::try_new(&wkb_data).unwrap();

        let result = extract_centroid_2d(&wkb).unwrap();
        assert!(result.0.is_nan() && result.1.is_nan());
    }

    #[test]
    fn test_extract_centroid_multilinestring() {
        // Create a multilinestring with two line segments
        let wkb_data =
            create_wkb_from_wkt("MULTILINESTRING ((0.0 0.0, 2.0 0.0), (0.0 1.0, 2.0 1.0))");
        let wkb = Wkb::try_new(&wkb_data).unwrap();

        let (x, y) = extract_centroid_2d(&wkb).unwrap();
        assert_eq!(x, 1.0);
        assert_eq!(y, 0.5);
    }

    #[test]
    fn test_extract_centroid_invalid_wkb() {
        // Create invalid WKB data
        let invalid_wkb = vec![0xFF, 0xFF, 0xFF, 0xFF];
        // This should fail at the Wkb::try_new level, so let's test the actual extract_centroid error
        if let Ok(wkb) = Wkb::try_new(&invalid_wkb) {
            let result = extract_centroid_2d(&wkb);
            assert!(result.is_err());
        }
    }
}
