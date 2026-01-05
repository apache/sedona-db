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

use byteorder::{LittleEndian, WriteBytesExt};
use datafusion_common::{error::Result, exec_err, DataFusionError};
use geos::{Geom, Geometry, GeometryTypes};

// TODO:
const NATIVE_ENDIANNESS: u8 = 1;

/// Write a GEOS geometry to WKB format.
///
/// This is a fast, custom implementation that directly extracts coordinates
/// from GEOS geometries and writes them in WKB format into a buffer.
pub fn write_geos_geometry(geom: &Geometry, writer: &mut impl Write) -> Result<()> {
    write_geometry(geom, writer)
}

// actually this is for geo-traits, not geos
// use geo_traits::GeometryTrait;
// use wkb::{writer::{WriteOptions, write_geometry}, Endianness};

// const WRITE_OPTIONS: WriteOptions = WriteOptions {
//     endianness: Endianness::LittleEndian,
// };

// pub fn write_geos_geometry(geom: &impl GeometryTrait<T = f64>, writer: &mut impl Write) -> Result<()> {
//     write_geometry(writer, geom, &WRITE_OPTIONS).map_err(|e| DataFusionError::Execution(format!("Failed to write geometry: {e}")))?;
//     Ok(())
// }

fn write_geometry(geom: &impl Geom, writer: &mut impl Write) -> Result<()> {
    let geom_type = geom
        .geometry_type()
        .map_err(|e| DataFusionError::Execution(format!("Failed to get geometry type: {e}")))?;

    match geom_type {
        GeometryTypes::Point => write_point(geom, writer),
        GeometryTypes::LineString => write_line_string(geom, writer),
        GeometryTypes::Polygon => write_polygon(geom, writer),
        GeometryTypes::MultiPoint => write_multi_point(geom, writer),
        GeometryTypes::MultiLineString => write_multi_line_string(geom, writer),
        GeometryTypes::MultiPolygon => write_multi_polygon(geom, writer),
        GeometryTypes::GeometryCollection => write_geometry_collection(geom, writer),
        _ => Err(DataFusionError::Execution(format!(
            "Unsupported geometry type: {geom_type:?}"
        ))),
    }
}

fn write_point(geom: &impl Geom, writer: &mut impl Write) -> Result<()> {
    // Write byte order
    writer.write_u8(NATIVE_ENDIANNESS)?;

    let has_z = geom
        .has_z()
        .map_err(|e| DataFusionError::Execution(format!("Failed to check has_z: {e}")))?;
    let has_m = geom
        .has_m()
        .map_err(|e| DataFusionError::Execution(format!("Failed to check has_m: {e}")))?;

    let wkb_type = match (has_z, has_m) {
        (false, false) => 1,   // Point
        (true, false) => 1001, // Point Z
        (false, true) => 2001, // Point M
        (true, true) => 3001,  // Point ZM
    };

    writer.write_u32::<LittleEndian>(wkb_type)?;

    let is_empty = geom
        .is_empty()
        .map_err(|e| DataFusionError::Execution(format!("Failed to check if empty: {e}")))?;

    if is_empty {
        // Write NaN coordinates for empty point
        writer.write_f64::<LittleEndian>(f64::NAN)?; // x
        writer.write_f64::<LittleEndian>(f64::NAN)?; // y
        if has_z {
            writer.write_f64::<LittleEndian>(f64::NAN)?; // z
        }
        if has_m {
            writer.write_f64::<LittleEndian>(f64::NAN)?; // m
        }
    } else {
        let coord_seq = geom
            .get_coord_seq()
            .map_err(|e| DataFusionError::Execution(format!("Failed to get coord seq: {e}")))?;

        let num_coords = coord_seq.size().map_err(|e| {
            DataFusionError::Execution(format!("Failed to get coord seq size: {e}"))
        })?;

        if num_coords == 0 {
            return exec_err!("Point has no coordinates");
        }

        write_coord_seq(&coord_seq, writer, has_z, has_m)?;
    }

    Ok(())
}

fn write_line_string(geom: &impl Geom, writer: &mut impl Write) -> Result<()> {
    writer.write_u8(NATIVE_ENDIANNESS)?;

    let has_z = geom
        .has_z()
        .map_err(|e| DataFusionError::Execution(format!("Failed to check has_z: {e}")))?;

    let has_m = geom
        .has_m()
        .map_err(|e| DataFusionError::Execution(format!("Failed to check has_m: {e}")))?;

    let wkb_type = match (has_z, has_m) {
        (false, false) => 2,   // LineString
        (true, false) => 1002, // LineString Z
        (false, true) => 2002, // LineString M
        (true, true) => 3002,  // LineString ZM
    };

    writer.write_u32::<LittleEndian>(wkb_type)?;

    let is_empty = geom
        .is_empty()
        .map_err(|e| DataFusionError::Execution(format!("Failed to check if empty: {e}")))?;

    if is_empty {
        writer.write_u32::<LittleEndian>(0)?; // 0 points
    } else {
        let coord_seq = geom
            .get_coord_seq()
            .map_err(|e| DataFusionError::Execution(format!("Failed to get coord seq: {e}")))?;

        let num_points = coord_seq.size().map_err(|e| {
            DataFusionError::Execution(format!("Failed to get coord seq size: {e}"))
        })?;

        writer.write_u32::<LittleEndian>(num_points as u32)?;

        write_coord_seq(&coord_seq, writer, has_z, has_m)?;
    }

    Ok(())
}

fn write_polygon(geom: &impl Geom, writer: &mut impl Write) -> Result<()> {
    writer.write_u8(NATIVE_ENDIANNESS)?;

    let has_z = geom
        .has_z()
        .map_err(|e| DataFusionError::Execution(format!("Failed to check has_z: {e}")))?;
    let has_m = geom
        .has_m()
        .map_err(|e| DataFusionError::Execution(format!("Failed to check has_m: {e}")))?;

    let wkb_type = match (has_z, has_m) {
        (false, false) => 3,   // Polygon
        (true, false) => 1003, // Polygon Z
        (false, true) => 2003, // Polygon M
        (true, true) => 3003,  // Polygon ZM
    };

    writer.write_u32::<LittleEndian>(wkb_type)?;

    let is_empty = geom
        .is_empty()
        .map_err(|e| DataFusionError::Execution(format!("Failed to check if empty: {e}")))?;

    if is_empty {
        writer.write_u32::<LittleEndian>(0)?; // 0 rings
    } else {
        let exterior = geom
            .get_exterior_ring()
            .map_err(|e| DataFusionError::Execution(format!("Failed to get exterior ring: {e}")))?;

        let exterior_coord_seq = exterior.get_coord_seq().map_err(|e| {
            DataFusionError::Execution(format!("Failed to get exterior coord seq: {e}"))
        })?;

        let num_interior_rings = geom.get_num_interior_rings().map_err(|e| {
            DataFusionError::Execution(format!("Failed to get num interior rings: {e}"))
        })?;

        // Number of rings (interior rings + 1 exterior ring)
        writer.write_u32::<LittleEndian>((num_interior_rings + 1) as u32)?;

        let exterior_size = exterior_coord_seq
            .size()
            .map_err(|e| DataFusionError::Execution(format!("Failed to get exterior size: {e}")))?;

        // Number of points in exterior ring
        writer.write_u32::<LittleEndian>(exterior_size as u32)?;
        write_coord_seq(&exterior_coord_seq, writer, has_z, has_m)?;

        // Write interior rings
        for i in 0..num_interior_rings {
            let interior = geom.get_interior_ring_n(i).map_err(|e| {
                DataFusionError::Execution(format!("Failed to get interior ring {i}: {e}"))
            })?;

            let interior_coord_seq = interior.get_coord_seq().map_err(|e| {
                DataFusionError::Execution(format!("Failed to get interior coord seq: {e}"))
            })?;

            let interior_size = interior_coord_seq.size().map_err(|e| {
                DataFusionError::Execution(format!("Failed to get interior size: {e}"))
            })?;

            writer.write_u32::<LittleEndian>(interior_size as u32)?;
            write_coord_seq(&interior_coord_seq, writer, has_z, has_m)?;
        }
    }

    Ok(())
}

fn write_multi_point(geom: &impl Geom, writer: &mut impl Write) -> Result<()> {
    writer.write_u8(NATIVE_ENDIANNESS)?;

    let has_z = geom
        .has_z()
        .map_err(|e| DataFusionError::Execution(format!("Failed to check has_z: {e}")))?;
    let has_m = geom
        .has_m()
        .map_err(|e| DataFusionError::Execution(format!("Failed to check has_m: {e}")))?;

    let wkb_type = match (has_z, has_m) {
        (false, false) => 4,   // MultiPoint
        (true, false) => 1004, // MultiPoint Z
        (false, true) => 2004, // MultiPoint M
        (true, true) => 3004,  // MultiPoint ZM
    };

    writer.write_u32::<LittleEndian>(wkb_type)?;

    let is_empty = geom
        .is_empty()
        .map_err(|e| DataFusionError::Execution(format!("Failed to check if empty: {e}")))?;

    if is_empty {
        writer.write_u32::<LittleEndian>(0)?; // 0 points
    } else {
        let num_points = geom.get_num_geometries().map_err(|e| {
            DataFusionError::Execution(format!("Failed to get num geometries: {e}"))
        })?;

        writer.write_u32::<LittleEndian>(num_points as u32)?;

        for i in 0..num_points {
            let point = geom
                .get_geometry_n(i)
                .map_err(|e| DataFusionError::Execution(format!("Failed to get point {i}: {e}")))?;

            write_point(&point, writer)?;
        }
    }

    Ok(())
}

fn write_multi_line_string(geom: &impl Geom, writer: &mut impl Write) -> Result<()> {
    writer.write_u8(NATIVE_ENDIANNESS)?;

    let has_z = geom
        .has_z()
        .map_err(|e| DataFusionError::Execution(format!("Failed to check has_z: {e}")))?;
    let has_m = geom
        .has_m()
        .map_err(|e| DataFusionError::Execution(format!("Failed to check has_m: {e}")))?;

    let wkb_type = match (has_z, has_m) {
        (false, false) => 5,   // MultiLineString
        (true, false) => 1005, // MultiLineString Z
        (false, true) => 2005, // MultiLineString M
        (true, true) => 3005,  // MultiLineString ZM
    };

    writer.write_u32::<LittleEndian>(wkb_type)?;

    let is_empty = geom
        .is_empty()
        .map_err(|e| DataFusionError::Execution(format!("Failed to check if empty: {e}")))?;

    if is_empty {
        writer.write_u32::<LittleEndian>(0)?; // 0 line strings
    } else {
        let num_line_strings = geom.get_num_geometries().map_err(|e| {
            DataFusionError::Execution(format!("Failed to get num geometries: {e}"))
        })?;

        writer.write_u32::<LittleEndian>(num_line_strings as u32)?;

        for i in 0..num_line_strings {
            let ls = geom.get_geometry_n(i).map_err(|e| {
                DataFusionError::Execution(format!("Failed to get line string {i}: {e}"))
            })?;

            write_line_string(&ls, writer)?;
        }
    }

    Ok(())
}

fn write_multi_polygon(geom: &impl Geom, writer: &mut impl Write) -> Result<()> {
    writer.write_u8(NATIVE_ENDIANNESS)?;

    let has_z = geom
        .has_z()
        .map_err(|e| DataFusionError::Execution(format!("Failed to check has_z: {e}")))?;
    let has_m = geom
        .has_m()
        .map_err(|e| DataFusionError::Execution(format!("Failed to check has_m: {e}")))?;

    let wkb_type = match (has_z, has_m) {
        (false, false) => 6,   // MultiPolygon
        (true, false) => 1006, // MultiPolygon Z
        (false, true) => 2006, // MultiPolygon M
        (true, true) => 3006,  // MultiPolygon ZM
    };

    writer.write_u32::<LittleEndian>(wkb_type)?;

    let is_empty = geom
        .is_empty()
        .map_err(|e| DataFusionError::Execution(format!("Failed to check if empty: {e}")))?;

    if is_empty {
        writer.write_u32::<LittleEndian>(0)?; // 0 polygons
    } else {
        let num_polygons = geom.get_num_geometries().map_err(|e| {
            DataFusionError::Execution(format!("Failed to get num geometries: {e}"))
        })?;

        writer.write_u32::<LittleEndian>(num_polygons as u32)?;

        for i in 0..num_polygons {
            let poly = geom.get_geometry_n(i).map_err(|e| {
                DataFusionError::Execution(format!("Failed to get polygon {i}: {e}"))
            })?;

            write_polygon(&poly, writer)?;
        }
    }
    Ok(())
}

fn write_geometry_collection(geom: &impl Geom, writer: &mut impl Write) -> Result<()> {
    writer.write_u8(NATIVE_ENDIANNESS)?;

    let has_z = geom
        .has_z()
        .map_err(|e| DataFusionError::Execution(format!("Failed to check has_z: {e}")))?;
    let has_m = geom
        .has_m()
        .map_err(|e| DataFusionError::Execution(format!("Failed to check has_m: {e}")))?;

    let wkb_type = match (has_z, has_m) {
        (false, false) => 7,   // GeometryCollection
        (true, false) => 1007, // GeometryCollection Z
        (false, true) => 2007, // GeometryCollection M
        (true, true) => 3007,  // GeometryCollection ZM
    };

    writer.write_u32::<LittleEndian>(wkb_type)?;

    let num_geometries = geom
        .get_num_geometries()
        .map_err(|e| DataFusionError::Execution(format!("Failed to get num geometries: {e}")))?;

    writer.write_u32::<LittleEndian>(num_geometries as u32)?;

    for i in 0..num_geometries {
        let sub_geom = geom
            .get_geometry_n(i)
            .map_err(|e| DataFusionError::Execution(format!("Failed to get geometry {i}: {e}")))?;

        write_geometry(&sub_geom, writer)?;
    }

    Ok(())
}

fn write_coord_seq(
    coord_seq: &geos::CoordSeq,
    writer: &mut impl Write,
    has_z: bool,
    has_m: bool,
) -> Result<()> {
    let dims = match (has_z, has_m) {
        (true, true) => 4,
        (true, false) | (false, true) => 3,
        (false, false) => 2,
    };

    let coords = coord_seq
        .as_buffer(Some(dims))
        .map_err(|e| DataFusionError::Execution(format!("Failed to get coord seq buffer: {e}")))?;

    // Cast Vec<f64> to &[u8] so we can write the bytes directly to the writer buffer
    let byte_slice: &[u8] = bytemuck::cast_slice(&coords);
    writer.write_all(byte_slice)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper function to test WKB round-trip: create geometry from WKT, write to WKB, read back, verify
    fn test_wkb_round_trip(wkt: &str) {
        let geos_geom = geos::Geometry::new_from_wkt(wkt).unwrap();
        let expected_wkt = geos_geom.to_wkt().unwrap();

        // Write to WKB from Geos object using our method
        let mut wkb_buf = Vec::new();
        write_geos_geometry(&geos_geom, &mut wkb_buf).unwrap();
        let geos_from_wkb = geos::Geometry::new_from_wkb(&wkb_buf).unwrap();

        // Compare them as WKT
        let geos_from_wkb_wkt = geos_from_wkb.to_wkt().unwrap();
        assert_eq!(geos_from_wkb_wkt, expected_wkt);
    }

    // Point tests
    #[test]
    fn test_write_point_xy() {
        test_wkb_round_trip("POINT (0 1)");
        test_wkb_round_trip("POINT (1.5 2.5)");
        test_wkb_round_trip("POINT (-10.5 -20.5)");
    }

    #[test]
    fn test_write_point_xyz() {
        test_wkb_round_trip("POINT Z (0 1 10)");
        test_wkb_round_trip("POINT Z (1.5 2.5 3.5)");
        test_wkb_round_trip("POINT Z (-10.5 -20.5 -30.5)");
    }

    #[test]
    fn test_write_point_xyzm() {
        test_wkb_round_trip("POINT ZM (0 1 10 100)");
        test_wkb_round_trip("POINT ZM (1.5 2.5 3.5 4.5)");
        test_wkb_round_trip("POINT ZM (-10.5 -20.5 -30.5 -40.5)");
    }

    #[test]
    fn test_write_point_empty() {
        test_wkb_round_trip("POINT EMPTY");
        test_wkb_round_trip("POINT Z EMPTY");
        test_wkb_round_trip("POINT ZM EMPTY");
    }

    // LineString tests
    #[test]
    fn test_write_linestring_xy() {
        test_wkb_round_trip("LINESTRING (0 0, 1 1)");
        test_wkb_round_trip("LINESTRING (0 0, 1 1, 2 2)");
        test_wkb_round_trip("LINESTRING (0 0, 1 1, 2 2, 3 3)");
    }

    #[test]
    fn test_write_linestring_xyz() {
        test_wkb_round_trip("LINESTRING Z (0 0 0, 1 1 1)");
        test_wkb_round_trip("LINESTRING Z (0 0 0, 1 1 1, 2 2 2)");
        test_wkb_round_trip("LINESTRING Z (0 0 10, 1 1 11, 2 2 12)");
    }

    #[test]
    fn test_write_linestring_xyzm() {
        test_wkb_round_trip("LINESTRING ZM (0 0 1 2, 1 1 3 4)");
        test_wkb_round_trip("LINESTRING ZM (0 0 1 2, 1 1 3 4, 2 2 5 6)");
        test_wkb_round_trip("LINESTRING ZM (0 0 10 20, 1 1 11 21, 2 2 12 22)");
    }

    #[test]
    fn test_write_linestring_empty() {
        test_wkb_round_trip("LINESTRING EMPTY");
        test_wkb_round_trip("LINESTRING Z EMPTY");
        test_wkb_round_trip("LINESTRING ZM EMPTY");
    }

    // Polygon tests
    #[test]
    fn test_write_polygon_xy() {
        test_wkb_round_trip("POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))");
        test_wkb_round_trip("POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 1 2, 2 2, 2 1, 1 1))");
    }

    #[test]
    fn test_write_polygon_xyz() {
        test_wkb_round_trip("POLYGON Z ((0 0 10, 4 0 10, 4 4 10, 0 4 10, 0 0 10))");
        test_wkb_round_trip("POLYGON Z ((0 0 0, 1 0 0, 0 1 0, 0 0 0))");
        test_wkb_round_trip(
            "POLYGON Z ((0 0 10, 4 0 10, 4 4 10, 0 4 10, 0 0 10), (1 1 5, 1 2 5, 2 2 5, 2 1 5, 1 1 5))",
        );
    }

    #[test]
    fn test_write_polygon_xyzm() {
        test_wkb_round_trip("POLYGON ZM ((0 0 10 1, 4 0 10 2, 4 4 10 3, 0 4 10 4, 0 0 10 5))");
        test_wkb_round_trip(
            "POLYGON ZM ((0 0 10 1, 4 0 10 2, 4 4 10 3, 0 4 10 4, 0 0 10 5), (1 1 5 10, 1 2 5 11, 2 2 5 12, 2 1 5 13, 1 1 5 10))",
        );
    }

    #[test]
    fn test_write_polygon_empty() {
        test_wkb_round_trip("POLYGON EMPTY");
        test_wkb_round_trip("POLYGON Z EMPTY");
        test_wkb_round_trip("POLYGON ZM EMPTY");
    }

    // MultiPoint tests
    #[test]
    fn test_write_multipoint_xy() {
        test_wkb_round_trip("MULTIPOINT ((0 0), (1 1))");
        test_wkb_round_trip("MULTIPOINT ((0 0), (1 1), (2 2))");
    }

    #[test]
    fn test_write_multipoint_xyz() {
        test_wkb_round_trip("MULTIPOINT Z ((0 0 0), (1 1 1))");
        test_wkb_round_trip("MULTIPOINT Z ((0 0 0), (1 1 1), (2 2 2))");
    }

    #[test]
    fn test_write_multipoint_xyzm() {
        test_wkb_round_trip("MULTIPOINT ZM ((0 0 1 2), (1 1 3 4))");
        test_wkb_round_trip("MULTIPOINT ZM ((0 0 1 2), (1 1 3 4), (2 2 5 6))");
    }

    #[test]
    fn test_write_multipoint_empty() {
        test_wkb_round_trip("MULTIPOINT EMPTY");
        test_wkb_round_trip("MULTIPOINT Z EMPTY");
        test_wkb_round_trip("MULTIPOINT ZM EMPTY");
    }

    // MultiLineString tests
    #[test]
    fn test_write_multilinestring_xy() {
        test_wkb_round_trip("MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))");
        test_wkb_round_trip("MULTILINESTRING ((0 0, 1 1), (2 2, 3 3), (4 4, 5 5))");
    }

    #[test]
    fn test_write_multilinestring_xyz() {
        test_wkb_round_trip("MULTILINESTRING Z ((0 0 0, 1 1 1), (2 2 2, 3 3 3))");
        test_wkb_round_trip("MULTILINESTRING Z ((0 0 0, 1 1 1), (2 2 2, 3 3 3), (4 4 4, 5 5 5))");
    }

    #[test]
    fn test_write_multilinestring_xyzm() {
        test_wkb_round_trip("MULTILINESTRING ZM ((0 0 1 2, 1 1 3 4), (2 2 5 6, 3 3 7 8))");
    }

    #[test]
    fn test_write_multilinestring_empty() {
        test_wkb_round_trip("MULTILINESTRING EMPTY");
        test_wkb_round_trip("MULTILINESTRING Z EMPTY");
        test_wkb_round_trip("MULTILINESTRING ZM EMPTY");
    }

    // MultiPolygon tests
    #[test]
    fn test_write_multipolygon_xy() {
        test_wkb_round_trip(
            "MULTIPOLYGON (((0 0, 4 0, 4 4, 0 4, 0 0)), ((5 5, 6 5, 6 6, 5 6, 5 5)))",
        );
    }

    #[test]
    fn test_write_multipolygon_xyz() {
        test_wkb_round_trip(
            "MULTIPOLYGON Z (((0 0 10, 4 0 10, 4 4 10, 0 4 10, 0 0 10)), ((5 5 20, 6 5 20, 6 6 20, 5 6 20, 5 5 20)))",
        );
    }

    #[test]
    fn test_write_multipolygon_xyzm() {
        test_wkb_round_trip(
            "MULTIPOLYGON ZM (((0 0 10 1, 4 0 10 2, 4 4 10 3, 0 4 10 4, 0 0 10 5)), ((5 5 20 10, 6 5 20 11, 6 6 20 12, 5 6 20 13, 5 5 20 10)))",
        );
    }

    #[test]
    fn test_write_multipolygon_empty() {
        test_wkb_round_trip("MULTIPOLYGON EMPTY");
        test_wkb_round_trip("MULTIPOLYGON Z EMPTY");
        test_wkb_round_trip("MULTIPOLYGON ZM EMPTY");
    }

    // GeometryCollection tests
    #[test]
    fn test_write_geometrycollection_xy() {
        test_wkb_round_trip("GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (0 0, 1 1))");
        test_wkb_round_trip(
            "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (0 0, 1 1), POLYGON ((0 0, 1 0, 0 1, 0 0)))",
        );
    }

    #[test]
    fn test_write_geometrycollection_xyz() {
        test_wkb_round_trip("GEOMETRYCOLLECTION Z (POINT Z (1 2 3), LINESTRING Z (0 0 0, 1 1 1))");
        test_wkb_round_trip(
            "GEOMETRYCOLLECTION Z (POINT Z (1 2 3), LINESTRING Z (0 0 0, 1 1 1), POLYGON Z ((0 0 10, 4 0 10, 4 4 10, 0 4 10, 0 0 10)))",
        );
    }

    #[test]
    fn test_write_geometrycollection_xyzm() {
        test_wkb_round_trip(
            "GEOMETRYCOLLECTION ZM (POINT ZM (1 2 3 4), LINESTRING ZM (0 0 1 2, 1 1 3 4))",
        );
    }

    #[test]
    fn test_write_geometrycollection_mixed_dimensions() {
        // Test that dimension is inferred from nested geometries when not specified on collection
        test_wkb_round_trip("GEOMETRYCOLLECTION (POINT Z (1 2 3), LINESTRING Z (0 0 0, 1 1 1))");
        test_wkb_round_trip(
            "GEOMETRYCOLLECTION (POINT ZM (1 2 3 4), LINESTRING ZM (0 0 1 2, 1 1 3 4))",
        );
    }

    #[test]
    fn test_write_geometrycollection_empty() {
        test_wkb_round_trip("GEOMETRYCOLLECTION EMPTY");
        test_wkb_round_trip("GEOMETRYCOLLECTION Z EMPTY");
        test_wkb_round_trip("GEOMETRYCOLLECTION ZM EMPTY");
    }

    #[test]
    fn test_write_geometrycollection_nested() {
        test_wkb_round_trip(
            "GEOMETRYCOLLECTION (GEOMETRYCOLLECTION (POINT (1 2), POINT (3 4)), POINT (5 6))",
        );
        test_wkb_round_trip(
            "GEOMETRYCOLLECTION Z (GEOMETRYCOLLECTION Z (POINT Z (1 2 3), POINT Z (4 5 6)), POINT Z (7 8 9))",
        );
    }
}
