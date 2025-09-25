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

use crate::error::SedonaGeometryError;
use geo_traits::{
    GeometryCollectionTrait, GeometryTrait, LineStringTrait, MultiLineStringTrait, MultiPointTrait,
    MultiPolygonTrait, PointTrait, PolygonTrait,
};

pub fn is_geometry_empty<G: GeometryTrait<T = f64>>(
    geometry: &G,
) -> Result<bool, SedonaGeometryError> {
    match geometry.as_type() {
        geo_traits::GeometryType::Point(point) => Ok(point.coord().is_none()),
        geo_traits::GeometryType::LineString(linestring) => Ok(linestring.num_coords() == 0),
        geo_traits::GeometryType::Polygon(polygon) => {
            Ok(polygon.num_interiors() == 0 && polygon.exterior().is_none())
        }
        geo_traits::GeometryType::MultiPoint(multipoint) => Ok(multipoint.num_points() == 0),
        geo_traits::GeometryType::MultiLineString(multilinestring) => {
            Ok(multilinestring.num_line_strings() == 0)
        }
        geo_traits::GeometryType::MultiPolygon(multipolygon) => {
            Ok(multipolygon.num_polygons() == 0)
        }
        geo_traits::GeometryType::GeometryCollection(geometrycollection) => {
            Ok(geometrycollection.num_geometries() == 0)
        }
        _ => Err(SedonaGeometryError::Invalid(
            "Invalid geometry type".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;
    use wkb::reader::read_wkb;
    use wkb::writer::{write_geometry, WriteOptions};
    use wkb::Endianness;
    use wkt::Wkt;

    fn create_wkb_bytes_from_wkt(wkt_str: &str) -> Vec<u8> {
        let wkt: Wkt = Wkt::from_str(wkt_str).unwrap();
        let mut wkb_bytes = vec![];
        write_geometry(
            &mut wkb_bytes,
            &wkt,
            &WriteOptions {
                endianness: Endianness::LittleEndian,
            },
        )
        .unwrap();
        wkb_bytes
    }

    #[test]
    fn test_is_geometry_empty_points() {
        // Empty point
        let empty_point_bytes = create_wkb_bytes_from_wkt("POINT EMPTY");
        let empty_point = read_wkb(&empty_point_bytes).unwrap();
        assert!(is_geometry_empty(&empty_point).unwrap());

        // Non-empty point
        let point_bytes = create_wkb_bytes_from_wkt("POINT (1 2)");
        let point = read_wkb(&point_bytes).unwrap();
        assert!(!is_geometry_empty(&point).unwrap());
    }

    #[test]
    fn test_is_geometry_empty_linestrings() {
        // Empty linestring
        let empty_linestring_bytes = create_wkb_bytes_from_wkt("LINESTRING EMPTY");
        let empty_linestring = read_wkb(&empty_linestring_bytes).unwrap();
        assert!(is_geometry_empty(&empty_linestring).unwrap());

        // Non-empty linestring
        let linestring_bytes = create_wkb_bytes_from_wkt("LINESTRING (1 2, 2 2)");
        let linestring = read_wkb(&linestring_bytes).unwrap();
        assert!(!is_geometry_empty(&linestring).unwrap());
    }

    #[test]
    fn test_is_geometry_empty_polygons() {
        // Empty polygon
        let empty_polygon_bytes = create_wkb_bytes_from_wkt("POLYGON EMPTY");
        let empty_polygon = read_wkb(&empty_polygon_bytes).unwrap();
        assert!(is_geometry_empty(&empty_polygon).unwrap());

        // Non-empty polygon
        let polygon_bytes = create_wkb_bytes_from_wkt("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))");
        let polygon = read_wkb(&polygon_bytes).unwrap();
        assert!(!is_geometry_empty(&polygon).unwrap());
    }

    #[test]
    fn test_is_geometry_empty_multigeometries() {
        // Empty multipoint
        let empty_multipoint_bytes = create_wkb_bytes_from_wkt("MULTIPOINT EMPTY");
        let empty_multipoint = read_wkb(&empty_multipoint_bytes).unwrap();
        assert!(is_geometry_empty(&empty_multipoint).unwrap());

        // Non-empty multipoint
        let multipoint_bytes = create_wkb_bytes_from_wkt("MULTIPOINT ((0 0))");
        let multipoint = read_wkb(&multipoint_bytes).unwrap();
        assert!(!is_geometry_empty(&multipoint).unwrap());

        // Empty multilinestring
        let empty_multilinestring_bytes = create_wkb_bytes_from_wkt("MULTILINESTRING EMPTY");
        let empty_multilinestring = read_wkb(&empty_multilinestring_bytes).unwrap();
        assert!(is_geometry_empty(&empty_multilinestring).unwrap());

        // Non-empty multilinestring
        let multilinestring_bytes =
            create_wkb_bytes_from_wkt("MULTILINESTRING ((0 0, 1 0), (1 1, 0 1))");
        let multilinestring = read_wkb(&multilinestring_bytes).unwrap();
        assert!(!is_geometry_empty(&multilinestring).unwrap());

        // Empty multipolygon
        let empty_multipolygon_bytes = create_wkb_bytes_from_wkt("MULTIPOLYGON EMPTY");
        let empty_multipolygon = read_wkb(&empty_multipolygon_bytes).unwrap();
        assert!(is_geometry_empty(&empty_multipolygon).unwrap());

        // Non-empty multipolygon
        let multipolygon_bytes =
            create_wkb_bytes_from_wkt("MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)))");
        let multipolygon = read_wkb(&multipolygon_bytes).unwrap();
        assert!(!is_geometry_empty(&multipolygon).unwrap());
    }

    #[test]
    fn test_is_geometry_empty_collections() {
        // Empty collection
        let empty_collection_bytes = create_wkb_bytes_from_wkt("GEOMETRYCOLLECTION EMPTY");
        let empty_collection = read_wkb(&empty_collection_bytes).unwrap();
        assert!(is_geometry_empty(&empty_collection).unwrap());

        // Non-empty collection
        let collection_bytes =
            create_wkb_bytes_from_wkt("GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (1 2, 2 2))");
        let collection = read_wkb(&collection_bytes).unwrap();
        assert!(!is_geometry_empty(&collection).unwrap());
    }
}
