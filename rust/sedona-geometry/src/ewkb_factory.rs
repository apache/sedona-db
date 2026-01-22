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

use std::io::Write;

use geo_traits::{
    GeometryCollectionTrait, GeometryTrait, LineStringTrait, MultiLineStringTrait, MultiPointTrait,
    MultiPolygonTrait, PointTrait, PolygonTrait,
};
use wkb::reader::Wkb;

use crate::{
    error::SedonaGeometryError,
    wkb_factory::{write_wkb_coord_trait, write_wkb_empty_point},
};

const EWKB_Z_BIT: u32 = 0x80000000;
const EWKB_M_BIT: u32 = 0x40000000;
const EWKB_SRID_BIT: u32 = 0x20000000;

pub fn write_ewkb_geometry(
    geom: &Wkb,
    srid: Option<u32>,
    buf: &mut impl Write,
) -> Result<(), SedonaGeometryError> {
    match geom.as_type() {
        geo_traits::GeometryType::Point(p) => write_ewkb_point(p, srid, buf),
        geo_traits::GeometryType::LineString(ls) => write_ewkb_line_string(ls, srid, buf),
        geo_traits::GeometryType::Polygon(poly) => write_ewkb_polygon(poly, srid, buf),
        geo_traits::GeometryType::MultiPoint(mp) => write_ewkb_multi_point(mp, srid, buf),
        geo_traits::GeometryType::MultiLineString(mls) => {
            write_ewkb_multi_line_string(mls, srid, buf)
        }
        geo_traits::GeometryType::MultiPolygon(mpoly) => write_ewkb_multi_polygon(mpoly, srid, buf),
        geo_traits::GeometryType::GeometryCollection(gc) => {
            write_ewkb_geometry_collection(gc, srid, buf)
        }
        _ => Err(SedonaGeometryError::Invalid(
            "Unsupported EWKB geometry type".to_string(),
        )),
    }
}

fn write_ewkb_point(
    geom: &wkb::reader::Point,
    srid: Option<u32>,
    buf: &mut impl Write,
) -> Result<(), SedonaGeometryError> {
    write_geometry_type_and_srid(1, geom.dimension(), srid, buf)?;
    match geom.byte_order() {
        wkb::Endianness::BigEndian => match geom.coord() {
            Some(c) => write_wkb_coord_trait(buf, &c)?,
            None => write_wkb_empty_point(buf, geom.dim())?,
        },
        wkb::Endianness::LittleEndian => buf.write_all(geom.coord_slice())?,
    }

    Ok(())
}

fn write_ewkb_line_string(
    geom: &wkb::reader::LineString,
    srid: Option<u32>,
    buf: &mut impl Write,
) -> Result<(), SedonaGeometryError> {
    write_geometry_type_and_srid(2, geom.dimension(), srid, buf)?;
    let num_coords = geom.num_coords() as u32;
    buf.write_all(&num_coords.to_le_bytes())?;
    match geom.byte_order() {
        wkb::Endianness::BigEndian => {
            for c in geom.coords() {
                write_wkb_coord_trait(buf, &c)?;
            }
        }
        wkb::Endianness::LittleEndian => buf.write_all(geom.coords_slice())?,
    }

    Ok(())
}

fn write_linearring(
    geom: &wkb::reader::LinearRing,
    buf: &mut impl Write,
) -> Result<(), SedonaGeometryError> {
    let num_coords = geom.num_coords() as u32;
    buf.write_all(&num_coords.to_le_bytes())?;
    match geom.byte_order() {
        wkb::Endianness::BigEndian => {
            for c in geom.coords() {
                write_wkb_coord_trait(buf, &c)?;
            }
        }
        wkb::Endianness::LittleEndian => buf.write_all(geom.coords_slice())?,
    }

    Ok(())
}

fn write_ewkb_polygon(
    geom: &wkb::reader::Polygon,
    srid: Option<u32>,
    buf: &mut impl Write,
) -> Result<(), SedonaGeometryError> {
    write_geometry_type_and_srid(3, geom.dimension(), srid, buf)?;
    let num_rings = geom.num_interiors() as u32 + geom.exterior().is_some() as u32;
    buf.write_all(&num_rings.to_le_bytes())?;

    if let Some(exterior) = geom.exterior() {
        write_linearring(exterior, buf)?;
    }

    for interior in geom.interiors() {
        write_linearring(interior, buf)?;
    }

    Ok(())
}

fn write_ewkb_multi_point(
    geom: &wkb::reader::MultiPoint,
    srid: Option<u32>,
    buf: &mut impl Write,
) -> Result<(), SedonaGeometryError> {
    write_geometry_type_and_srid(4, geom.dimension(), srid, buf)?;
    let num_children = geom.num_points() as u32;
    buf.write_all(&num_children.to_le_bytes())?;

    for child in geom.points() {
        write_ewkb_point(&child, None, buf)?;
    }

    Ok(())
}

fn write_ewkb_multi_line_string(
    geom: &wkb::reader::MultiLineString,
    srid: Option<u32>,
    buf: &mut impl Write,
) -> Result<(), SedonaGeometryError> {
    write_geometry_type_and_srid(5, geom.dimension(), srid, buf)?;
    let num_children = geom.num_line_strings() as u32;
    buf.write_all(&num_children.to_le_bytes())?;

    for child in geom.line_strings() {
        write_ewkb_line_string(child, None, buf)?;
    }

    Ok(())
}

fn write_ewkb_multi_polygon(
    geom: &wkb::reader::MultiPolygon,
    srid: Option<u32>,
    buf: &mut impl Write,
) -> Result<(), SedonaGeometryError> {
    write_geometry_type_and_srid(6, geom.dimension(), srid, buf)?;
    let num_children = geom.num_polygons() as u32;
    buf.write_all(&num_children.to_le_bytes())?;

    for child in geom.polygons() {
        write_ewkb_polygon(child, None, buf)?;
    }

    Ok(())
}

fn write_ewkb_geometry_collection(
    geom: &wkb::reader::GeometryCollection,
    srid: Option<u32>,
    buf: &mut impl Write,
) -> Result<(), SedonaGeometryError> {
    write_geometry_type_and_srid(7, geom.dimension(), srid, buf)?;
    let num_children = geom.num_geometries() as u32;
    buf.write_all(&num_children.to_le_bytes())?;

    for child in geom.geometries() {
        write_ewkb_geometry(child, None, buf)?;
    }

    Ok(())
}

fn write_geometry_type_and_srid(
    mut base_type: u32,
    dimensions: wkb::reader::Dimension,
    srid: Option<u32>,
    buf: &mut impl Write,
) -> Result<(), SedonaGeometryError> {
    buf.write_all(&[0x01])?;

    match dimensions {
        wkb::reader::Dimension::Xy => {}
        wkb::reader::Dimension::Xyz => base_type |= EWKB_Z_BIT,
        wkb::reader::Dimension::Xym => base_type |= EWKB_M_BIT,
        wkb::reader::Dimension::Xyzm => {
            base_type |= EWKB_Z_BIT;
            base_type |= EWKB_M_BIT;
        }
    }

    if let Some(srid) = srid {
        base_type |= EWKB_SRID_BIT;
        buf.write_all(&base_type.to_le_bytes())?;
        buf.write_all(&srid.to_le_bytes())?;
    } else {
        buf.write_all(&base_type.to_le_bytes())?;
    }

    Ok(())
}

#[cfg(test)]
mod test {

    use rstest::rstest;
    use wkb::{writer::WriteOptions, Endianness};

    use super::*;

    #[rstest]
    fn test_roundtrip(
        #[values(Endianness::LittleEndian, Endianness::BigEndian)] endianness: Endianness,
    ) {
        for wkt_str in ROUNDTRIP_CASES {
            let wkt: wkt::Wkt<f64> = wkt_str.parse().unwrap();

            let mut iso_wkb = Vec::new();
            wkb::writer::write_geometry(&mut iso_wkb, &wkt, &WriteOptions { endianness }).unwrap();
            let wkb_geom = wkb::reader::read_wkb(&iso_wkb).unwrap();

            let mut ewkb_no_srid = Vec::new();
            write_ewkb_geometry(&wkb_geom, None, &mut ewkb_no_srid).unwrap();

            let mut ewkb_with_srid = Vec::new();
            write_ewkb_geometry(&wkb_geom, Some(4326), &mut ewkb_with_srid).unwrap();

            // Check that the ewkbs have the correct number of bytes
            assert_eq!(ewkb_no_srid.len(), iso_wkb.len());
            assert_eq!(ewkb_with_srid.len(), ewkb_no_srid.len() + size_of::<u32>());

            // Check the rendered WKT of the no srid EWKB
            let wkb_geom_roundtrip_no_srid = wkb::reader::read_wkb(&ewkb_no_srid).unwrap();
            let mut wkt_roundtrip_no_srid = String::new();
            wkt::to_wkt::write_geometry(&mut wkt_roundtrip_no_srid, &wkb_geom_roundtrip_no_srid)
                .unwrap();
            assert_eq!(wkt_roundtrip_no_srid, wkt_str);

            // Check the rendered WKT of the srid EWKB
            let wkb_geom_roundtrip_with_srid = wkb::reader::read_wkb(&ewkb_with_srid).unwrap();
            let mut wkt_roundtrip_with_srid = String::new();
            wkt::to_wkt::write_geometry(&mut wkt_roundtrip_with_srid, &wkb_geom_roundtrip_with_srid)
                .unwrap();
            assert_eq!(wkt_roundtrip_with_srid, wkt_str);
        }
    }

    const ROUNDTRIP_CASES: [&str; 35] = [
        // XY dimensions
        "POINT (1 2)",
        "LINESTRING (1 2, 3 4, 5 6)",
        "POLYGON ((0 1, 2 0, 2 3, 0 3, 0 1))",
        "MULTIPOINT ((1 2), (3 4))",
        "MULTILINESTRING ((1 2, 3 4), (5 6, 7 8))",
        "MULTIPOLYGON (((0 1, 2 0, 2 3, 0 3, 0 1)))",
        "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (3 4, 5 6))",
        // XYZ dimensions
        "POINT Z (1 2 3)",
        "LINESTRING Z (1 2 3, 4 5 6)",
        "POLYGON Z ((0 1 2, 3 0 2, 3 4 2, 0 4 2, 0 1 2))",
        "MULTIPOINT Z ((1 2 3), (4 5 6))",
        "MULTILINESTRING Z ((1 2 3, 4 5 6), (7 8 9, 10 11 12))",
        "MULTIPOLYGON Z (((0 1 2, 3 0 2, 3 4 2, 0 4 2, 0 1 2)))",
        "GEOMETRYCOLLECTION Z (POINT Z (1 2 3))",
        // XYM dimensions
        "POINT M (1 2 3)",
        "LINESTRING M (1 2 3, 4 5 6)",
        "POLYGON M ((0 1 2, 3 0 2, 3 4 2, 0 4 2, 0 1 2))",
        "MULTIPOINT M ((1 2 3), (4 5 6))",
        "MULTILINESTRING M ((1 2 3, 4 5 6), (7 8 9, 10 11 12))",
        "MULTIPOLYGON M (((0 1 2, 3 0 2, 3 4 2, 0 4 2, 0 1 2)))",
        "GEOMETRYCOLLECTION M (POINT M (1 2 3))",
        // XYZM dimensions
        "POINT ZM (1 2 3 4)",
        "LINESTRING ZM (1 2 3 4, 5 6 7 8)",
        "POLYGON ZM ((0 1 2 3, 4 0 2 3, 4 5 2 3, 0 5 2 3, 0 1 2 3))",
        "MULTIPOINT ZM ((1 2 3 4), (5 6 7 8))",
        "MULTILINESTRING ZM ((1 2 3 4, 5 6 7 8), (9 10 11 12, 13 14 15 16))",
        "MULTIPOLYGON ZM (((0 1 2 3, 4 0 2 3, 4 5 2 3, 0 5 2 3, 0 1 2 3)))",
        "GEOMETRYCOLLECTION ZM (POINT ZM (1 2 3 4))",
        // Empty geometries
        "POINT EMPTY",
        "LINESTRING EMPTY",
        "POLYGON EMPTY",
        "MULTIPOINT EMPTY",
        "MULTILINESTRING EMPTY",
        "MULTIPOLYGON EMPTY",
        "GEOMETRYCOLLECTION EMPTY",
    ];
}
