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
use std::fmt::Write;

use geo_traits::{
    CoordTrait, Dimensions, GeometryCollectionTrait, GeometryTrait, GeometryType, LineStringTrait,
    MultiLineStringTrait, MultiPointTrait, MultiPolygonTrait, PointTrait, PolygonTrait,
};
use wkt::error::Error;

/// The name of a geometry type in each of the dimensions XY, XYZ, XYM and XYZM
type TypeNames = [&'static str; 4];

const POLYGON: TypeNames = ["POLYGON", "POLYGON Z", "POLYGON M", "POLYGON ZM"];
const MULTIPOINT: TypeNames = [
    "MULTIPOINT",
    "MULTIPOINT Z",
    "MULTIPOINT M",
    "MULTIPOINT ZM",
];
const MULTILINESTRING: TypeNames = [
    "MULTILINESTRING",
    "MULTILINESTRING Z",
    "MULTILINESTRING M",
    "MULTILINESTRING ZM",
];
const MULTIPOLYGON: TypeNames = [
    "MULTIPOLYGON",
    "MULTIPOLYGON Z",
    "MULTIPOLYGON M",
    "MULTIPOLYGON ZM",
];
const GEOMETRYCOLLECTION: TypeNames = [
    "GEOMETRYCOLLECTION",
    "GEOMETRYCOLLECTION Z",
    "GEOMETRYCOLLECTION M",
    "GEOMETRYCOLLECTION ZM",
];

/// Write a geometry as WKT
///
/// This writes what [wkt::to_wkt::write_geometry] writes, except that an empty member of a
/// multi-geometry, and an empty interior ring of a polygon, are written `EMPTY`, as in the WKT
/// grammar of Simple Features Access. The wkt crate (0.14) panics on an empty point or polygon in
/// a multi-geometry and writes an empty linestring in one as `()`, which is not WKT
/// (apache/sedona-db#1531, georust/wkt#157). Points and linestrings, which have no members, are
/// written by the wkt crate.
pub(crate) fn write_geometry(
    f: &mut impl Write,
    geom: &impl GeometryTrait<T = f64>,
) -> Result<(), Error> {
    match geom.as_type() {
        GeometryType::Polygon(polygon) => {
            let size = write_type(f, &POLYGON, polygon.dim())?;
            if is_empty_polygon(polygon) {
                Ok(f.write_str(" EMPTY")?)
            } else {
                write_rings(f, polygon, size)
            }
        }
        GeometryType::MultiPoint(multipoint) => {
            let size = write_type(f, &MULTIPOINT, multipoint.dim())?;
            write_members(f, multipoint.points(), |f, point| match point.coord() {
                Some(coord) => {
                    f.write_char('(')?;
                    write_coord(f, &coord, size)?;
                    Ok(f.write_char(')')?)
                }
                None => Ok(f.write_str("EMPTY")?),
            })
        }
        GeometryType::MultiLineString(multilinestring) => {
            let size = write_type(f, &MULTILINESTRING, multilinestring.dim())?;
            write_members(f, multilinestring.line_strings(), |f, linestring| {
                write_linestring_text(f, &linestring, size)
            })
        }
        GeometryType::MultiPolygon(multipolygon) => {
            let size = write_type(f, &MULTIPOLYGON, multipolygon.dim())?;
            write_members(f, multipolygon.polygons(), |f, polygon| {
                if is_empty_polygon(&polygon) {
                    Ok(f.write_str("EMPTY")?)
                } else {
                    write_rings(f, &polygon, size)
                }
            })
        }
        GeometryType::GeometryCollection(collection) => {
            write_type(f, &GEOMETRYCOLLECTION, collection.dim())?;
            write_members(f, collection.geometries(), |f, geom| {
                write_geometry(f, &geom)
            })
        }
        _ => wkt::to_wkt::write_geometry(f, geom),
    }
}

/// Write the name of a geometry type in the given dimensions, returning the number of values in
/// each coordinate
fn write_type(f: &mut impl Write, names: &TypeNames, dim: Dimensions) -> Result<usize, Error> {
    let name = match dim {
        Dimensions::Xy | Dimensions::Unknown(2) => names[0],
        Dimensions::Xyz | Dimensions::Unknown(3) => names[1],
        Dimensions::Xym => names[2],
        Dimensions::Xyzm | Dimensions::Unknown(4) => names[3],
        Dimensions::Unknown(_) => return Err(Error::UnknownDimension),
    };
    f.write_str(name)?;
    Ok(dim.size())
}

/// Write the members of a multi-geometry or a collection in parentheses, or ` EMPTY` where it has
/// none
fn write_members<W: Write, M>(
    f: &mut W,
    mut members: impl Iterator<Item = M>,
    mut write_member: impl FnMut(&mut W, M) -> Result<(), Error>,
) -> Result<(), Error> {
    let Some(first) = members.next() else {
        return Ok(f.write_str(" EMPTY")?);
    };
    f.write_char('(')?;
    write_member(f, first)?;
    for member in members {
        f.write_char(',')?;
        write_member(f, member)?;
    }
    Ok(f.write_char(')')?)
}

/// Whether a polygon is empty, as the wkt crate has it: without an exterior ring or with an empty
/// one
fn is_empty_polygon(polygon: &impl PolygonTrait<T = f64>) -> bool {
    polygon
        .exterior()
        .is_none_or(|exterior| exterior.num_coords() == 0)
}

/// Write the rings of a non-empty polygon in parentheses
fn write_rings(
    f: &mut impl Write,
    polygon: &impl PolygonTrait<T = f64>,
    size: usize,
) -> Result<(), Error> {
    f.write_char('(')?;
    if let Some(exterior) = polygon.exterior() {
        write_linestring_text(f, &exterior, size)?;
    }
    for interior in polygon.interiors() {
        f.write_char(',')?;
        write_linestring_text(f, &interior, size)?;
    }
    Ok(f.write_char(')')?)
}

/// Write a linestring, or a ring, without its type: its coordinates in parentheses, or `EMPTY`
fn write_linestring_text(
    f: &mut impl Write,
    linestring: &impl LineStringTrait<T = f64>,
    size: usize,
) -> Result<(), Error> {
    let mut coords = linestring.coords();
    let Some(first) = coords.next() else {
        return Ok(f.write_str("EMPTY")?);
    };
    f.write_char('(')?;
    write_coord(f, &first, size)?;
    for coord in coords {
        f.write_char(',')?;
        write_coord(f, &coord, size)?;
    }
    Ok(f.write_char(')')?)
}

/// Write the values of a coordinate, separated by spaces
fn write_coord(
    f: &mut impl Write,
    coord: &impl CoordTrait<T = f64>,
    size: usize,
) -> Result<(), Error> {
    write!(f, "{} {}", coord.x(), coord.y())?;
    for n in 2..size {
        write!(f, " {}", coord.nth_or_panic(n))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use rstest::rstest;
    use wkt::Wkt;

    use super::*;

    #[rstest]
    fn writes_what_the_wkt_crate_writes(
        #[values(
            "POINT EMPTY",
            "POINT (1 2)",
            "POINT Z (0.1 -2.5e-10 3)",
            "LINESTRING EMPTY",
            "LINESTRING M (1 2 3, 4 5 6)",
            "POLYGON EMPTY",
            "POLYGON ((0 0, 4 0, 4 4, 0 0), (1 1, 2 1, 2 2, 1 1))",
            "POLYGON ZM ((0 0 1 2, 4 0 1 2, 4 4 1 2, 0 0 1 2))",
            "MULTIPOINT EMPTY",
            "MULTIPOINT ((1 2), (3 4))",
            "MULTIPOINT Z ((1 2 3))",
            "MULTILINESTRING EMPTY",
            "MULTILINESTRING M ((1 2 3, 4 5 6), (7 8 9, 10 11 12))",
            "MULTIPOLYGON EMPTY",
            "MULTIPOLYGON (((0 0, 4 0, 4 4, 0 0)), ((10 10, 11 10, 11 11, 10 10), (10.5 10.2, 10.6 10.2, 10.6 10.3, 10.5 10.2)))",
            "MULTIPOLYGON ZM (((0 0 1 2, 4 0 1 2, 4 4 1 2, 0 0 1 2)))",
            "GEOMETRYCOLLECTION EMPTY",
            "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (1 2, 3 4), GEOMETRYCOLLECTION (MULTIPOINT ((1 2))))",
            "GEOMETRYCOLLECTION Z (POINT Z (1 2 3), MULTIPOLYGON Z (((0 0 1, 4 0 1, 4 4 1, 0 0 1))))"
        )]
        wkt: &str,
    ) {
        // Without empty members, the WKT is the wkt crate's
        let mut wkb = vec![];
        wkb::writer::write_geometry(
            &mut wkb,
            &Wkt::<f64>::from_str(wkt).unwrap(),
            &Default::default(),
        )
        .unwrap();
        let geom = wkb::reader::read_wkb(&wkb).unwrap();

        let mut expected = String::new();
        wkt::to_wkt::write_geometry(&mut expected, &geom).unwrap();
        let mut actual = String::new();
        write_geometry(&mut actual, &geom).unwrap();
        assert_eq!(actual, expected);
    }
}
