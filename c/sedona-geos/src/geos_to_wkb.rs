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
use datafusion_common::{error::Result, DataFusionError};
use geo_traits::Dimensions;
use geos::{CoordType, Geom, Geometry, GeometryTypes};
use sedona_common::sedona_internal_err;
use sedona_geometry::wkb_factory::{
    write_wkb_geometrycollection_header, write_wkb_linestring_header,
    write_wkb_multilinestring_header, write_wkb_multipoint_header, write_wkb_multipolygon_header,
    write_wkb_point_header, write_wkb_polygon_header,
};

/// Write a GEOS geometry to WKB format.
///
/// This is a fast, custom implementation that directly extracts coordinates
/// from GEOS geometries and writes them in WKB format into a buffer.
pub fn write_geos_geometry(geom: &Geometry, writer: &mut impl Write) -> Result<()> {
    write_geometry(geom, writer)
}

fn write_geometry(geom: &impl Geom, writer: &mut impl Write) -> Result<()> {
    let geom_type = geom
        .geometry_type()
        .map_err(|e| DataFusionError::Execution(format!("Failed to get geometry type: {e}")))?;

    let has_z = geom
        .has_z()
        .map_err(|e| DataFusionError::Execution(format!("Failed to check has_z: {e}")))?;
    let has_m = geom
        .has_m()
        .map_err(|e| DataFusionError::Execution(format!("Failed to check has_m: {e}")))?;

    let dim = match (has_z, has_m) {
        (false, false) => Dimensions::Xy,
        (true, false) => Dimensions::Xyz,
        (false, true) => Dimensions::Xym,
        (true, true) => Dimensions::Xyzm,
    };
    match geom_type {
        GeometryTypes::Point => write_point(geom, dim, writer),
        GeometryTypes::LineString => write_line_string(geom, dim, writer),
        GeometryTypes::Polygon => write_polygon(geom, dim, writer),
        GeometryTypes::MultiPoint => write_multi_point(geom, dim, writer),
        GeometryTypes::MultiLineString => write_multi_line_string(geom, dim, writer),
        GeometryTypes::MultiPolygon => write_multi_polygon(geom, dim, writer),
        GeometryTypes::GeometryCollection => write_geometry_collection(geom, dim, writer),
        _ => Err(DataFusionError::Execution(format!(
            "Unsupported geometry type: {geom_type:?}"
        ))),
    }
}

fn write_point(geom: &impl Geom, dim: Dimensions, writer: &mut impl Write) -> Result<()> {
    write_wkb_point_header(writer, dim)
        .map_err(|e| DataFusionError::Execution(format!("Failed to write point header: {e}")))?;

    let is_empty = geom
        .is_empty()
        .map_err(|e| DataFusionError::Execution(format!("Failed to check if empty: {e}")))?;

    if is_empty {
        // Write NaN coordinates for empty point
        writer.write_f64::<LittleEndian>(f64::NAN)?; // x
        writer.write_f64::<LittleEndian>(f64::NAN)?; // y
        if matches!(dim, Dimensions::Xyz | Dimensions::Xyzm) {
            writer.write_f64::<LittleEndian>(f64::NAN)?; // z
        }
        if matches!(dim, Dimensions::Xym | Dimensions::Xyzm) {
            writer.write_f64::<LittleEndian>(f64::NAN)?; // m
        }
    } else {
        let coord_seq = geom
            .get_coord_seq()
            .map_err(|e| DataFusionError::Execution(format!("Failed to get coord seq: {e}")))?;

        write_coord_seq(&coord_seq, dim, writer)?;
    }

    Ok(())
}

fn write_line_string(geom: &impl Geom, dim: Dimensions, writer: &mut impl Write) -> Result<()> {
    let num_points = geom
        .get_num_points()
        .map_err(|e| DataFusionError::Execution(format!("Failed to get num points: {e}")))?;

    write_wkb_linestring_header(writer, dim, num_points).map_err(|e| {
        DataFusionError::Execution(format!("Failed to write linestring header: {e}"))
    })?;

    if num_points > 0 {
        let coord_seq = geom
            .get_coord_seq()
            .map_err(|e| DataFusionError::Execution(format!("Failed to get coord seq: {e}")))?;

        write_coord_seq(&coord_seq, dim, writer)?;
    }

    Ok(())
}

fn write_polygon(geom: &impl Geom, dim: Dimensions, writer: &mut impl Write) -> Result<()> {
    let is_empty = geom
        .is_empty()
        .map_err(|e| DataFusionError::Execution(format!("Failed to check if empty: {e}")))?;

    let num_interior_rings = geom.get_num_interior_rings().map_err(|e| {
        DataFusionError::Execution(format!("Failed to get num interior rings: {e}"))
    })?;

    let num_rings = match (is_empty, num_interior_rings) {
        (true, _) => 0,
        (false, 0) => 1,
        (false, _) => num_interior_rings + 1,
    };

    write_wkb_polygon_header(writer, dim, num_rings)
        .map_err(|e| DataFusionError::Execution(format!("Failed to write polygon header: {e}")))?;

    if num_rings > 0 {
        let exterior = geom
            .get_exterior_ring()
            .map_err(|e| DataFusionError::Execution(format!("Failed to get exterior ring: {e}")))?;

        let exterior_coord_seq = exterior.get_coord_seq().map_err(|e| {
            DataFusionError::Execution(format!("Failed to get exterior coord seq: {e}"))
        })?;

        let exterior_size = exterior_coord_seq
            .size()
            .map_err(|e| DataFusionError::Execution(format!("Failed to get exterior size: {e}")))?;

        // Number of points in exterior ring
        writer.write_u32::<LittleEndian>(exterior_size as u32)?;
        write_coord_seq(&exterior_coord_seq, dim, writer)?;

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
            write_coord_seq(&interior_coord_seq, dim, writer)?;
        }
    }

    Ok(())
}

fn write_multi_point(geom: &impl Geom, dim: Dimensions, writer: &mut impl Write) -> Result<()> {
    let num_points = geom
        .get_num_geometries()
        .map_err(|e| DataFusionError::Execution(format!("Failed to get num geometries: {e}")))?;

    write_wkb_multipoint_header(writer, dim, num_points).map_err(|e| {
        DataFusionError::Execution(format!("Failed to write multipoint header: {e}"))
    })?;

    for i in 0..num_points {
        let point = geom
            .get_geometry_n(i)
            .map_err(|e| DataFusionError::Execution(format!("Failed to get point {i}: {e}")))?;

        write_point(&point, dim, writer)
            .map_err(|e| DataFusionError::Execution(format!("Failed to write point: {e}")))?;
    }

    Ok(())
}

fn write_multi_line_string(
    geom: &impl Geom,
    dim: Dimensions,
    writer: &mut impl Write,
) -> Result<()> {
    let num_line_strings = geom
        .get_num_geometries()
        .map_err(|e| DataFusionError::Execution(format!("Failed to get num geometries: {e}")))?;

    write_wkb_multilinestring_header(writer, dim, num_line_strings).map_err(|e| {
        DataFusionError::Execution(format!("Failed to write multilinestring header: {e}"))
    })?;

    for i in 0..num_line_strings {
        let line_string = geom.get_geometry_n(i).map_err(|e| {
            DataFusionError::Execution(format!("Failed to get line string {i}: {e}"))
        })?;
        write_line_string(&line_string, dim, writer)
            .map_err(|e| DataFusionError::Execution(format!("Failed to write line string: {e}")))?;
    }

    Ok(())
}

fn write_multi_polygon(geom: &impl Geom, dim: Dimensions, writer: &mut impl Write) -> Result<()> {
    let num_polygons = geom
        .get_num_geometries()
        .map_err(|e| DataFusionError::Execution(format!("Failed to get num geometries: {e}")))?;

    write_wkb_multipolygon_header(writer, dim, num_polygons).map_err(|e| {
        DataFusionError::Execution(format!("Failed to write multipolygon header: {e}"))
    })?;

    for i in 0..num_polygons {
        let poly = geom
            .get_geometry_n(i)
            .map_err(|e| DataFusionError::Execution(format!("Failed to get polygon {i}: {e}")))?;

        write_polygon(&poly, dim, writer)?;
    }
    Ok(())
}

fn write_geometry_collection(
    geom: &impl Geom,
    dim: Dimensions,
    writer: &mut impl Write,
) -> Result<()> {
    let num_geometries = geom
        .get_num_geometries()
        .map_err(|e| DataFusionError::Execution(format!("Failed to get num geometries: {e}")))?;

    write_wkb_geometrycollection_header(writer, dim, num_geometries).map_err(|e| {
        DataFusionError::Execution(format!("Failed to write geometry collection header: {e}"))
    })?;

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
    dim: Dimensions,
    writer: &mut impl Write,
) -> Result<()> {
    let coord_type = match dim {
        Dimensions::Xy => CoordType::XY,
        Dimensions::Xyz => CoordType::XYZ,
        Dimensions::Xym => CoordType::XYM,
        Dimensions::Xyzm => CoordType::XYZM,
        _ => return sedona_internal_err!("Unexpecte dimensions {dim:?}"),
    };

    let coords = coord_seq
        .as_buffer(Some(coord_type))
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
    fn test_write_linestring_xym() {
        test_wkb_round_trip("LINESTRING M (0 0 0, 1 1 1)");
        test_wkb_round_trip("LINESTRING M (0 0 0, 1 1 1, 2 2 2)");
        test_wkb_round_trip("LINESTRING M (0 0 10, 1 1 11, 2 2 12)");
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
