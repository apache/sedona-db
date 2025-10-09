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

//! Tests for the WKB extension traits implemented in `wkb_ext`.

use geo_traits::GeometryTrait;
use rstest::rstest;
use sedona_geo_traits_ext::*;
use std::str::FromStr;
use wkb::{reader::Wkb, Endianness};
use wkt::Wkt;

/// Helper to create WKB from WKT string using the wkb writer
fn wkb_from_wkt(wkt_str: &str) -> Vec<u8> {
    wkb_from_wkt_with_endianness(wkt_str, wkb::Endianness::LittleEndian)
}

/// Helper to create WKB from WKT string using the wkb writer
fn wkb_from_wkt_with_endianness(wkt_str: &str, endianness: wkb::Endianness) -> Vec<u8> {
    let geometry = Wkt::<f64>::from_str(wkt_str).unwrap();
    let mut buf = Vec::new();
    let options = wkb::writer::WriteOptions { endianness };
    wkb::writer::write_geometry(&mut buf, &geometry, &options).unwrap();
    buf
}

#[rstest]
fn test_geo_coord(
    #[values(Endianness::LittleEndian, Endianness::BigEndian)] endianness: Endianness,
) {
    let buf = wkb_from_wkt_with_endianness("POINT (1.0 2.0)", endianness);
    let wkb = Wkb::try_new(&buf).unwrap();
    let geo_traits::GeometryType::Point(pt) = wkb.as_type() else {
        panic!("expected point")
    };
    let coord = pt.geo_coord().unwrap();
    assert_eq!(coord.x, 1.0);
    assert_eq!(coord.y, 2.0);

    let buf = wkb_from_wkt_with_endianness("POINT EMPTY", endianness);
    let wkb = Wkb::try_new(&buf).unwrap();
    let geo_traits::GeometryType::Point(pt) = wkb.as_type() else {
        panic!("expected point")
    };
    let coord = pt.geo_coord();
    assert!(coord.is_none());
}

#[rstest]
fn test_linestring_iterators(
    #[values(Endianness::LittleEndian, Endianness::BigEndian)] endianness: Endianness,
) {
    let buf = wkb_from_wkt_with_endianness("LINESTRING(0 0, 1 1, 2 1.5)", endianness);
    let wkb = Wkb::try_new(&buf).unwrap();
    let GeometryTypeExt::LineString(ls) = wkb.as_type_ext() else {
        panic!("expected linestring")
    };

    let coords = &[(0.0, 0.0), (1.0, 1.0), (2.0, 1.5)];
    let v: Vec<_> = ls.coord_iter().collect();
    assert_eq!(v.len(), coords.len());
    for (got, (ex_x, ex_y)) in v.iter().zip(coords.iter()) {
        assert!((got.x - ex_x).abs() < 1e-9);
        assert!((got.y - ex_y).abs() < 1e-9);
    }
    let segs: Vec<_> = ls.lines().collect();
    assert_eq!(segs.len(), coords.len() - 1);
    assert_eq!(segs[0].start.x, 0.0);
    assert_eq!(segs[0].end.x, 1.0);

    // Empty linestring
    let buf = wkb_from_wkt_with_endianness("LINESTRING EMPTY", endianness);
    let wkb = Wkb::try_new(&buf).unwrap();
    let GeometryTypeExt::LineString(ls) = wkb.as_type_ext() else {
        panic!("expected linestring")
    };
    assert_eq!(ls.coord_iter().count(), 0);
    assert_eq!(ls.lines().count(), 0);
}

#[test]
fn test_geometry_collection_ext() {
    let buf = wkb_from_wkt("GEOMETRYCOLLECTION(POINT(0 0), POINT(1 1))");
    let wkb = Wkb::try_new(&buf).unwrap();

    // GeometryTraitExt is implemented for Wkb in wkb_ext. Use those helpers.
    assert!(wkb.is_collection());
    assert_eq!(wkb.num_geometries_ext(), 2);

    let child0 = wkb.geometry_ext(0).unwrap();
    let GeometryTypeExt::Point(_) = child0.as_type_ext() else {
        panic!("child0 expected point");
    };

    // Iterate via geometries_ext
    let types: Vec<_> = wkb
        .geometries_ext()
        .map(|g| match g.as_type_ext() {
            GeometryTypeExt::Point(_) => "P",
            _ => "?",
        })
        .collect();
    assert_eq!(types, vec!["P", "P"]);
}

#[test]
fn test_linestring_rev_lines() {
    // Empty linestring
    let buf = wkb_from_wkt("LINESTRING EMPTY");
    let wkb = Wkb::try_new(&buf).unwrap();
    let geo_traits::GeometryType::LineString(ls) = wkb.as_type() else {
        panic!("expected linestring")
    };
    assert_eq!(ls.rev_lines().count(), 0);

    // Two-point linestring: 1 segment
    let buf = wkb_from_wkt("LINESTRING(0 0, 1 1)");
    let wkb = Wkb::try_new(&buf).unwrap();
    let geo_traits::GeometryType::LineString(ls) = wkb.as_type() else {
        panic!("expected linestring")
    };
    let forward: Vec<_> = ls.lines().collect();
    let reverse: Vec<_> = ls.rev_lines().collect();
    assert_eq!(forward.len(), 1);
    assert_eq!(reverse.len(), 1);
    assert_eq!(forward[0].start.x, reverse[0].start.x);
    assert_eq!(forward[0].end.x, reverse[0].end.x);

    // Multi-point linestring: rev_lines should produce segments in reverse order
    let buf = wkb_from_wkt("LINESTRING(0 0, 2 0, 2 2, 0 2)");
    let wkb = Wkb::try_new(&buf).unwrap();
    let geo_traits::GeometryType::LineString(ls) = wkb.as_type() else {
        panic!("expected linestring")
    };
    let forward: Vec<_> = ls.lines().collect();
    let reverse: Vec<_> = ls.rev_lines().collect();
    assert_eq!(forward.len(), 3);
    assert_eq!(reverse.len(), 3);
    for i in 0..forward.len() {
        let f_rev = &forward[forward.len() - 1 - i];
        let r = &reverse[i];
        assert_eq!(f_rev.start.x, r.start.x);
        assert_eq!(f_rev.end.x, r.end.x);
    }
}

#[test]
fn test_linestring_is_closed() {
    // Empty line string is considered closed
    let buf = wkb_from_wkt("LINESTRING EMPTY");
    let wkb = Wkb::try_new(&buf).unwrap();
    let geo_traits::GeometryType::LineString(ls) = wkb.as_type() else {
        panic!("expected linestring")
    };
    assert!(ls.is_closed());

    // Non-closed line string
    let buf = wkb_from_wkt("LINESTRING(0 0, 1 0, 2 0)");
    let wkb = Wkb::try_new(&buf).unwrap();
    let geo_traits::GeometryType::LineString(ls) = wkb.as_type() else {
        panic!("expected linestring")
    };
    assert!(!ls.is_closed());

    // Closed linestring (square ring) with repeated first/last
    let buf = wkb_from_wkt("LINESTRING(0 0, 1 0, 1 1, 0 1, 0 0)");
    let wkb = Wkb::try_new(&buf).unwrap();
    let geo_traits::GeometryType::LineString(ls) = wkb.as_type() else {
        panic!("expected linestring")
    };
    assert!(ls.is_closed());
}

#[test]
fn test_linestring_triangles() {
    // Empty - no triangles
    let buf = wkb_from_wkt("LINESTRING EMPTY");
    let wkb = Wkb::try_new(&buf).unwrap();
    let geo_traits::GeometryType::LineString(ls) = wkb.as_type() else {
        panic!("expected linestring")
    };
    assert_eq!(ls.triangles().count(), 0);

    // Two points - no triangles (need at least 3)
    let buf = wkb_from_wkt("LINESTRING(0 0, 1 1)");
    let wkb = Wkb::try_new(&buf).unwrap();
    let geo_traits::GeometryType::LineString(ls) = wkb.as_type() else {
        panic!("expected linestring")
    };
    assert_eq!(ls.triangles().count(), 0);

    // Three points - one triangle
    let buf = wkb_from_wkt("LINESTRING(0 0, 1 0, 1 1)");
    let wkb = Wkb::try_new(&buf).unwrap();
    let geo_traits::GeometryType::LineString(ls) = wkb.as_type() else {
        panic!("expected linestring")
    };
    assert_eq!(ls.triangles().count(), 1);

    // Four points - two triangles
    let buf = wkb_from_wkt("LINESTRING(0 0, 2 0, 2 2, 0 2)");
    let wkb = Wkb::try_new(&buf).unwrap();
    let geo_traits::GeometryType::LineString(ls) = wkb.as_type() else {
        panic!("expected linestring")
    };
    assert_eq!(ls.triangles().count(), 2);

    // Single point - degenerate case
    let buf = wkb_from_wkt("LINESTRING(5 5)");
    let wkb = Wkb::try_new(&buf).unwrap();
    let geo_traits::GeometryType::LineString(ls) = wkb.as_type() else {
        panic!("expected linestring")
    };
    assert_eq!(ls.triangles().count(), 0);
    assert_eq!(ls.lines().len(), 0);
}
