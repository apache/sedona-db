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

    // GeometryCollection's dimension might be specified on the child geometries only
    let dimension = get_geomcol_dimension(geom)?;
    writer.write_u32::<LittleEndian>(dimension)?;

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

// TBH. um not sure this is needed for geos
fn get_geomcol_dimension(geomcol: &impl Geom) -> Result<u32> {
    let has_z = geomcol
        .has_z()
        .map_err(|e| DataFusionError::Execution(format!("Failed to check has_z: {e}")))?;

    let has_m = geomcol
        .has_m()
        .map_err(|e| DataFusionError::Execution(format!("Failed to check has_m: {e}")))?;

    let wkb_type = match (has_z, has_m) {
        (true, false) => Some(1007), // GeometryCollection Z
        (false, true) => Some(2007), // GeometryCollection M
        (true, true) => Some(3007),  // GeometryCollection ZM
        (false, false) => None,      // GeometryCollection
    };

    // If found non-xy dimension, use it
    if let Some(wkb_type) = wkb_type {
        return Ok(wkb_type);
    }

    // Try inferring from first nested geom
    let num_geometries = geomcol
        .get_num_geometries()
        .map_err(|e| DataFusionError::Execution(format!("Failed to get num geometries: {e}")))?;

    if num_geometries == 0 {
        // Guaranteed to be XY only
        return Ok(7);
    }

    // Check the first geom's type
    let sub_geom = geomcol.get_geometry_n(0).map_err(|e| {
        DataFusionError::Execution(format!("Failed to get first nested geometry: {e}"))
    })?;
    let sub_geom_type = sub_geom
        .geometry_type()
        .map_err(|e| DataFusionError::Execution(format!("Failed to get geometry type: {e}")))?;

    // If it's a nested geom collection, recurse into that
    if sub_geom_type == GeometryTypes::GeometryCollection {
        return get_geomcol_dimension(&sub_geom);
    }

    let has_z = sub_geom
        .has_z()
        .map_err(|e| DataFusionError::Execution(format!("Failed to check has_z: {e}")))?;
    let has_m = sub_geom
        .has_m()
        .map_err(|e| DataFusionError::Execution(format!("Failed to check has_m: {e}")))?;
    let wkb_type = match (has_z, has_m) {
        (false, false) => 7,   // GeometryCollection
        (true, false) => 1007, // GeometryCollection Z
        (false, true) => 2007, // GeometryCollection M
        (true, true) => 3007,  // GeometryCollection ZM
    };
    Ok(wkb_type)
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

    // TODO: I guess we could compute the exact size buffer we expect here and check it. Should we do it or just assume geos correct?
    // TODO: do we need to consider endianness??

    // Cast Vec<f64> to &[u8] so we can write the bytes directly to the writer buffer
    let byte_slice: &[u8] = bytemuck::cast_slice(&coords);
    writer.write_all(byte_slice)?;

    Ok(())

    // OLD Approach with get_x, get_y, get_z, get_m functions

    // let actual_size = coord_seq
    //     .size()
    //     .map_err(|e| DataFusionError::Execution(format!("Failed to get coord seq size: {e}")))?;

    // let coords_to_write = if num_coords == 0 { actual_size } else { num_coords };

    // TODO: check size in advance to avoid conditions inside of the loop to leverage SIMD

    // #[inline(always)]
    // fn write_xy_coord(i: usize, coord_seq: &geos::CoordSeq, writer: &mut dyn Write) -> Result<()> {
    //     let x = coord_seq.get_x(i).unwrap();
    //     let y = coord_seq.get_y(i).unwrap();
    //     writer.write_f64::<LittleEndian>(x)?;
    //     writer.write_f64::<LittleEndian>(y)?;
    //     Ok(())
    // }

    // #[inline(always)]
    // fn write_xyz_coord(
    //     i: usize,
    //     coord_seq: &geos::CoordSeq,
    //     writer: &mut dyn Write,
    // ) -> Result<()> {
    //     write_xy_coord(i, coord_seq, writer)?;
    //     let z = coord_seq.get_z(i).unwrap();
    //     writer.write_f64::<LittleEndian>(z)?;
    //     Ok(())
    // }

    // fn write_xyzm_coord()  // not possible bc geos-rust doesn't support .get_m() yet

    // // TODO: use it
    // let write_coord: fn(usize, &geos::CoordSeq, &mut dyn Write) -> Result<()> = if has_z {
    //     write_xyz_coord
    // } else {
    //     write_xy_coord
    // };

    // for i in 0..coords_to_write {
    //
    // }
}
