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
use std::{fs::File, path::PathBuf, str::FromStr};

use geo_types::{LineString, MultiPolygon, Point, Polygon};
use wkt::{TryFromWkt, WktFloat};

/// A well-known binary blob of MULTIPOINT (EMPTY)
///
/// The wkt crate's parser rejects this; however, it's a corner case that may show
/// up in WKB generated externally.
pub const MULTIPOINT_WITH_EMPTY_CHILD_WKB: [u8; 30] = [
    0x01, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f,
];

/// A well-known binary blob of MULTIPOINT ((1 2 3)) where outer dimension is specified for xy
/// while inner point's dimension is actually xyz
pub const MULTIPOINT_WITH_INFERRED_Z_DIMENSION_WKB: [u8; 38] = [
    0x01, // byte-order
    0x04, 0x00, 0x00, 0x00, // multipoint with xy-dimension specified
    0x01, 0x00, 0x00, 0x00, // 1 point
    // nested point geom
    0x01, // byte-order
    0xe9, 0x03, 0x00, 0x00, // point with xyz-dimension specified
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, // x-coordinate of point
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, // y-coordinate of point
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x40, // z-coordinate of point
];

/// EWKB for POINT (1 2) with SRID 4326
/// Little endian, geometry type 1 (POINT) with SRID flag (0x20000000)
pub const POINT_WITH_SRID_4326_EWKB: [u8; 25] = [
    0x01, // byte-order
    0x01, 0x00, 0x00, 0x20, // geometry type 1 (POINT) with SRID flag (0x20000000)
    0xe6, 0x10, 0x00, 0x00, // SRID 4326
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, // x-coordinate 1.0
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, // y-coordinate 2.0
];

/// EWKB for POINT Z (1 2 3) with SRID 3857
/// Little endian, geometry type 1001 (POINT Z) with SRID flag
pub const POINT_Z_WITH_SRID_3857_EWKB: [u8; 33] = [
    0x01, // byte-order
    0x01, 0x00, 0x00, 0xa0, // geometry type
    // 0xe9, 0x03, 0x00, 0x20, // geometry type 1001 (POINT Z) with SRID flag
    0x11, 0x0f, 0x00, 0x00, // SRID 3857
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, // x-coordinate 1.0
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, // y-coordinate 2.0
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x40, // z-coordinate 3.0
];

pub const POINT_M_WITH_SRID_4326_EWKB: [u8; 33] = [
    0x01, // byte-order
    0x01, 0x00, 0x00, 0x60, // geometry type
    0xe6, 0x10, 0x00, 0x00, // SRID
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, // x-coordinate 1.0
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, // y-coordinate 2.0
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x40, // m-coordinate 3.0
];

/// EWKB for POINT ZM (1 2 3 4) with SRID 4326
pub const POINT_ZM_WITH_SRID_4326_EWKB: [u8; 41] = [
    0x01, // byte-order
    0x01, 0x00, 0x00, 0xe0, // geometry type
    // 0xb9, 0x0b, 0x00, 0x20, // geometry type 3001 (POINT ZM) with SRID flag
    0xe6, 0x10, 0x00, 0x00, // SRID 4326
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, // x = 1.0
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, // y = 2.0
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x40, // z = 3.0
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x40, // m = 4.0
];

/// EWKB for LINESTRING (1 2, 3 4) with SRID 4326
/// Little endian, geometry type 2 (LINESTRING) with SRID flag
pub const LINESTRING_WITH_SRID_4326_EWKB: [u8; 45] = [
    0x01, // byte-order
    0x02, 0x00, 0x00, 0x20, // geometry type
    0xe6, 0x10, 0x00, 0x00, // SRID 4326
    0x02, 0x00, 0x00, 0x00, // number of points (2)
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, // x1 = 1.0
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, // y1 = 2.0
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x40, // x2 = 3.0
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x40, // y2 = 4.0
];

/// EWKB for POLYGON ((0 0, 0 1, 1 0, 0 0)) with SRID 4326
/// Little endian, geometry type 3 (POLYGON) with SRID flag
pub const POLYGON_WITH_SRID_4326_EWKB: [u8; 81] = [
    0x01, // byte-order
    0x03, 0x00, 0x00, 0x20, // geometry type 3 (POLYGON) with SRID flag
    0xe6, 0x10, 0x00, 0x00, // SRID 4326
    0x01, 0x00, 0x00, 0x00, // number of rings (1)
    0x04, 0x00, 0x00, 0x00, // number of points in exterior ring (4)
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // x1 = 0.0
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // y1 = 0.0
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // x2 = 0.0
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, // y2 = 1.0
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, // x3 = 1.0
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // y3 = 0.0
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // x4 = 0.0
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // y4 = 0.0
];

/// EWKB for MULTIPOINT ((1 2), (3 4)) with SRID 4326
/// Little endian, geometry type 4 (MULTIPOINT) with SRID flag
pub const MULTIPOINT_WITH_SRID_4326_EWKB: [u8; 55] = [
    0x01, // byte-order
    0x04, 0x00, 0x00, 0x20, // geometry type 4 (MULTIPOINT) with SRID flag
    0xe6, 0x10, 0x00, 0x00, // SRID 4326
    0x02, 0x00, 0x00, 0x00, // number of points (2)
    // First point
    0x01, // byte-order
    0x01, 0x00, 0x00, 0x00, // geometry type 1 (POINT) - no SRID flag
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, // x1 = 1.0
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, // y1 = 2.0
    // Second point
    0x01, // byte-order
    0x01, 0x00, 0x00, 0x00, // geometry type 1 (POINT) - no SRID flag
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x40, // x2 = 3.0
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x40, // y2 = 4.0
];

/// EWKB for GEOMETRYCOLLECTION (POINT (1 2)) with SRID 4326
/// Little endian, geometry type 7 (GEOMETRYCOLLECTION) with SRID flag
pub const GEOMETRYCOLLECTION_POINT_WITH_SRID_4326_EWKB: [u8; 34] = [
    0x01, // byte-order
    0x07, 0x00, 0x00, 0x20, // geometry type 7 (GEOMETRYCOLLECTION) with SRID flag
    0xe6, 0x10, 0x00, 0x00, // SRID 4326
    0x01, 0x00, 0x00, 0x00, // number of geometries (1)
    // Nested POINT
    0x01, // byte-order
    0x01, 0x00, 0x00, 0x00, // geometry type 1 (POINT) - no SRID flag
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, // x = 1.0
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, // y = 2.0
];

/// EWKB for GEOMETRYCOLLECTION (POINT Z (1 2 3)) with SRID 4326
/// Little endian, geometry type 7 (GEOMETRYCOLLECTION) with SRID flag; nested POINT Z (Z flag set)
pub const GEOMETRYCOLLECTION_POINT_Z_WITH_SRID_4326_EWKB: [u8; 42] = [
    0x01, // byte-order
    0x07, 0x00, 0x00, 0x20, // geometry type 7 (GEOMETRYCOLLECTION) with SRID flag
    0xe6, 0x10, 0x00, 0x00, // SRID 4326
    0x01, 0x00, 0x00, 0x00, // number of geometries (1)
    // Nested POINT Z
    0x01, // byte-order
    0x01, 0x00, 0x00, 0x80, // geometry type 1 (POINT) with Z flag
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, // x = 1.0
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, // y = 2.0
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x40, // z = 3.0
];

/// EWKB for GEOMETRYCOLLECTION (POINT M (1 2 4)) with SRID 4326
/// Little endian, geometry type 7 (GEOMETRYCOLLECTION) with SRID flag; nested POINT M (M flag set)
pub const GEOMETRYCOLLECTION_POINT_M_WITH_SRID_4326_EWKB: [u8; 42] = [
    0x01, // byte-order
    0x07, 0x00, 0x00, 0x20, // geometry type 7 (GEOMETRYCOLLECTION) with SRID flag
    0xe6, 0x10, 0x00, 0x00, // SRID 4326
    0x01, 0x00, 0x00, 0x00, // number of geometries (1)
    // Nested POINT M
    0x01, // byte-order
    0x01, 0x00, 0x00, 0x40, // geometry type 1 (POINT) with M flag
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, // x = 1.0
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, // y = 2.0
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x40, // m = 4.0
];

/// EWKB for GEOMETRYCOLLECTION (POINT ZM (1 2 3 4)) with SRID 4326
/// Little endian, geometry type 7 (GEOMETRYCOLLECTION) with SRID flag; nested POINT ZM (Z and M flags set)
pub const GEOMETRYCOLLECTION_POINT_ZM_WITH_SRID_4326_EWKB: [u8; 50] = [
    0x01, // byte-order
    0x07, 0x00, 0x00, 0x20, // geometry type 7 (GEOMETRYCOLLECTION) with SRID flag
    0xe6, 0x10, 0x00, 0x00, // SRID 4326
    0x01, 0x00, 0x00, 0x00, // number of geometries (1)
    // Nested POINT ZM
    0x01, // byte-order
    0x01, 0x00, 0x00, 0xc0, // geometry type 1 (POINT) with Z and M flags
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, // x = 1.0
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, // y = 2.0
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x08, 0x40, // z = 3.0
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x40, // m = 4.0
];

/// EWKB for POINT EMPTY with SRID 4326
/// Little endian, geometry type 1 (POINT) with SRID flag
pub const POINT_EMPTY_WITH_SRID_4326_EWKB: [u8; 25] = [
    0x01, // byte-order
    0x01, 0x00, 0x00, 0x20, // geometry type 1 (POINT) with SRID flag
    0xe6, 0x10, 0x00, 0x00, // SRID 4326
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f, // x = NaN
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f, // y = NaN
];

/// EWKB for GEOMETRYCOLLECTION EMPTY with SRID 4326
/// Little endian, geometry type 7 (GEOMETRYCOLLECTION) with SRID flag
pub const GEOMETRYCOLLECTION_EMPTY_WITH_SRID_4326_EWKB: [u8; 13] = [
    0x01, // byte-order
    0x07, 0x00, 0x00, 0x20, // geometry type 7 (GEOMETRYCOLLECTION) with SRID flag
    0xe6, 0x10, 0x00, 0x00, // SRID 4326
    0x00, 0x00, 0x00, 0x00, // number of geometries (0)
];

pub fn louisiana<T>() -> LineString<T>
where
    T: WktFloat + Default + FromStr,
{
    line_string("louisiana.wkt")
}

pub fn baton_rouge<T>() -> Point<T>
where
    T: WktFloat + Default + FromStr,
{
    let x = T::from(-91.147385).unwrap();
    let y = T::from(30.471165).unwrap();
    Point::new(x, y)
}

pub fn east_baton_rouge<T>() -> Polygon<T>
where
    T: WktFloat + Default + FromStr,
{
    polygon("east_baton_rouge.wkt")
}

pub fn norway_main<T>() -> LineString<T>
where
    T: WktFloat + Default + FromStr,
{
    line_string("norway_main.wkt")
}

pub fn norway_concave_hull<T>() -> LineString<T>
where
    T: WktFloat + Default + FromStr,
{
    line_string("norway_concave_hull.wkt")
}

pub fn norway_convex_hull<T>() -> LineString<T>
where
    T: WktFloat + Default + FromStr,
{
    line_string("norway_convex_hull.wkt")
}

pub fn norway_nonconvex_hull<T>() -> LineString<T>
where
    T: WktFloat + Default + FromStr,
{
    line_string("norway_nonconvex_hull.wkt")
}

pub fn vw_orig<T>() -> LineString<T>
where
    T: WktFloat + Default + FromStr,
{
    line_string("vw_orig.wkt")
}

pub fn vw_simplified<T>() -> LineString<T>
where
    T: WktFloat + Default + FromStr,
{
    line_string("vw_simplified.wkt")
}

pub fn poly1<T>() -> LineString<T>
where
    T: WktFloat + Default + FromStr,
{
    line_string("poly1.wkt")
}

pub fn poly1_hull<T>() -> LineString<T>
where
    T: WktFloat + Default + FromStr,
{
    line_string("poly1_hull.wkt")
}

pub fn poly2<T>() -> LineString<T>
where
    T: WktFloat + Default + FromStr,
{
    line_string("poly2.wkt")
}

pub fn poly2_hull<T>() -> LineString<T>
where
    T: WktFloat + Default + FromStr,
{
    line_string("poly2_hull.wkt")
}

pub fn poly_in_ring<T>() -> LineString<T>
where
    T: WktFloat + Default + FromStr,
{
    line_string("poly_in_ring.wkt")
}

pub fn ring<T>() -> LineString<T>
where
    T: WktFloat + Default + FromStr,
{
    line_string("ring.wkt")
}

pub fn shell<T>() -> LineString<T>
where
    T: WktFloat + Default + FromStr,
{
    line_string("shell.wkt")
}

// From https://geodata.nationaalgeoregister.nl/kadastralekaart/wfs/v4_0?request=GetFeature&service=WFS&srsName=EPSG:4326&typeName=kadastralekaartv4:perceel&version=2.0.0&outputFormat=json&bbox=165593,480993,166125,481552
pub fn nl_zones<T>() -> MultiPolygon<T>
where
    T: WktFloat + Default + FromStr,
{
    multi_polygon("nl_zones.wkt")
}

// From https://afnemers.ruimtelijkeplannen.nl/afnemers/services?request=GetFeature&service=WFS&srsName=EPSG:4326&typeName=Enkelbestemming&version=2.0.0&bbox=165618,480983,166149,481542";
pub fn nl_plots_wgs84<T>() -> MultiPolygon<T>
where
    T: WktFloat + Default + FromStr,
{
    multi_polygon("nl_plots.wkt")
}

pub fn nl_plots_epsg_28992<T>() -> MultiPolygon<T>
where
    T: WktFloat + Default + FromStr,
{
    // https://epsg.io/28992
    multi_polygon("nl_plots_epsg_28992.wkt")
}

fn line_string<T>(name: &str) -> LineString<T>
where
    T: WktFloat + Default + FromStr,
{
    LineString::try_from_wkt_reader(file(name)).unwrap()
}

pub fn polygon<T>(name: &str) -> Polygon<T>
where
    T: WktFloat + Default + FromStr,
{
    Polygon::try_from_wkt_reader(file(name)).unwrap()
}

pub fn multi_polygon<T>(name: &str) -> MultiPolygon<T>
where
    T: WktFloat + Default + FromStr,
{
    MultiPolygon::try_from_wkt_reader(file(name)).unwrap()
}

pub fn file(name: &str) -> File {
    let base = crate::data::sedona_testing_dir()
        .expect("sedona-testing directory should resolve when accessing fixtures");

    let mut path = PathBuf::from(base);
    path.push("data");
    path.push("wkts");
    path.push("geo-test-fixtures");
    path.push(name);

    File::open(&path).unwrap_or_else(|_| panic!("Can't open file: {path:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norway_main_linestring_has_vertices() {
        let ls = norway_main::<f64>();
        assert!(
            !ls.0.is_empty(),
            "LineString loaded from norway_main.wkt should have vertices"
        );

        let first = ls.0.first().expect("expected at least one coordinate");
        assert!(first.x.is_finite(), "first coordinate x should be finite");
        assert!(first.y.is_finite(), "first coordinate y should be finite");
    }

    #[test]
    fn nl_zones_multipolygon_not_empty() {
        let mp = nl_zones::<f64>();
        assert!(
            !mp.0.is_empty(),
            "MultiPolygon from nl_zones.wkt should contain polygons"
        );

        let polygon = mp.0.first().expect("expected at least one polygon");
        assert!(
            !polygon.exterior().0.is_empty(),
            "polygon exterior ring should contain coordinates"
        );
    }
}
