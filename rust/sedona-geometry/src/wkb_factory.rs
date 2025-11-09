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
use geo_traits::{CoordTrait, Dimensions};
use std::io::Write;

pub const WKB_MIN_PROBABLE_BYTES: usize = 21;
pub const WKB_POINT_TYPE: u32 = 1;
pub const WKB_LINESTRING_TYPE: u32 = 2;
pub const WKB_POLYGON_TYPE: u32 = 3;
pub const WKB_MULTIPOINT_TYPE: u32 = 4;
pub const WKB_MULTILINESTRING_TYPE: u32 = 5;
pub const WKB_MULTIPOLYGON_TYPE: u32 = 6;
pub const WKB_GEOMETRYCOLLECTION_TYPE: u32 = 7;

/// Create WKB representing a POINT
///
/// A convenience wrapper for [write_wkb_point] that creates a Vec,
/// which is useful for creating DataFusion scalar values.
/// This function always writes little endian coordinates.
pub fn wkb_point(pt: (f64, f64)) -> Result<Vec<u8>, SedonaGeometryError> {
    let mut out_wkb = Vec::with_capacity(WKB_MIN_PROBABLE_BYTES);
    write_wkb_point(&mut out_wkb, pt)?;
    Ok(out_wkb)
}

/// Write WKB representing a POINT into a buffer
///
/// This can be used to build Binary arrays, as the arrow-rs BinaryBuilder
/// implements Write.
/// This function always writes little endian coordinates.
pub fn write_wkb_point(buf: &mut impl Write, pt: (f64, f64)) -> Result<(), SedonaGeometryError> {
    write_wkb_point_header(buf, Dimensions::Xy)?;
    write_wkb_coord(buf, pt)?;
    Ok(())
}

/// Write WKB representing an empty POINT
///
/// A convenient way to write WKB for an empty point.
/// This function always writes little endian coordinates.
pub fn write_wkb_empty_point(
    buf: &mut impl Write,
    dims: Dimensions,
) -> Result<(), SedonaGeometryError> {
    write_wkb_point_header(buf, dims)?;

    let count = dims.size();
    for _ in 0..count {
        buf.write_all(&f64::NAN.to_le_bytes())?;
    }
    Ok(())
}

/// Write WKB header for POINT
///
/// Writes the WKB header for a POINT geometry.
/// Should be followed by writing the coordinates
/// with [write_wkb_coords].
/// This function always writes little endian coordinates.
pub fn write_wkb_point_header(
    buf: &mut impl Write,
    dims: Dimensions,
) -> Result<(), SedonaGeometryError> {
    buf.write_all(&[0x01])?;
    let code: u32 = WKB_POINT_TYPE + wkb_dim_code(dims)?;
    buf.write_all(&code.to_le_bytes())?;
    Ok(())
}

/// Write WKB header for LINESTRING
///
/// The number of points in the linestring is provided as `count`.
/// The actual writing of the coordinates is expected to follow
/// with a call to [write_wkb_coords].
/// This function always writes little endian coordinates.
pub fn write_wkb_linestring_header(
    buf: &mut impl Write,
    dims: Dimensions,
    count: usize,
) -> Result<(), SedonaGeometryError> {
    buf.write_all(&[0x01])?;
    let code: u32 = WKB_LINESTRING_TYPE + wkb_dim_code(dims)?;
    buf.write_all(&code.to_le_bytes())?;
    buf.write_all(&count_to_u32(count)?.to_le_bytes())?;
    Ok(())
}

/// Create WKB representing a LINESTRING
///
/// A convenience wrapper for [write_wkb_linestring] that creates a Vec,
/// which is useful for creating DataFusion scalar values.
/// This function always writes little endian coordinates.
pub fn wkb_linestring<I: ExactSizeIterator<Item = (f64, f64)>>(
    pts: I,
) -> Result<Vec<u8>, SedonaGeometryError> {
    let mut out_wkb = Vec::with_capacity(5 + 4 + pts.len() * 16);
    write_wkb_linestring(&mut out_wkb, pts)?;
    Ok(out_wkb)
}

/// Write WKB representing a LINESTRING into a buffer
///
/// This can be used to build Binary arrays, as the arrow-rs BinaryBuilder
/// implements Write.
/// This function always writes little endian coordinates.
pub fn write_wkb_linestring<I: ExactSizeIterator<Item = (f64, f64)>>(
    buf: &mut impl Write,
    pts: I,
) -> Result<(), SedonaGeometryError> {
    write_wkb_linestring_header(buf, Dimensions::Xy, pts.len())?;
    write_wkb_coords(buf, pts)?;
    Ok(())
}

/// Create WKB representing a MULTILINESTRING
///
/// A convenience wrapper for [write_wkb_multilinestring] that creates a Vec,
/// which is useful for creating DataFusion scalar values.
/// This function always writes little endian coordinates.
pub fn wkb_multilinestring<I>(linestrings: I) -> Result<Vec<u8>, SedonaGeometryError>
where
    I: ExactSizeIterator<Item = Vec<(f64, f64)>> + Clone,
{
    // Base size + linestrings count + estimated per-linestring overhead
    let total_points: usize = linestrings.clone().map(|ls| ls.len()).sum();
    let capacity = 5 + 4 + (linestrings.len() * 9) + (total_points * 16);
    let mut out_wkb = Vec::with_capacity(capacity);
    write_wkb_multilinestring(&mut out_wkb, linestrings)?;
    Ok(out_wkb)
}

/// Write WKB representing a MULTILINESTRING into a buffer
///
/// This can be used to build Binary arrays, as the arrow-rs BinaryBuilder
/// implements Write.
/// This function always writes little endian coordinates.
pub fn write_wkb_multilinestring<I>(
    buf: &mut impl Write,
    linestrings: I,
) -> Result<(), SedonaGeometryError>
where
    I: ExactSizeIterator<Item = Vec<(f64, f64)>>,
{
    write_wkb_multilinestring_header(buf, Dimensions::Xy, linestrings.len())?;

    // For each linestring, write a complete linestring WKB
    for linestring in linestrings {
        write_wkb_linestring_header(buf, Dimensions::Xy, linestring.len())?;
        write_wkb_coords(buf, linestring.into_iter())?;
    }

    Ok(())
}

/// Create WKB representing a POLYGON
///
/// A convenience wrapper for [write_wkb_polygon] that creates a Vec,
/// which is useful for creating DataFusion scalar values.
/// This function always writes little endian coordinates.
pub fn wkb_polygon<I: ExactSizeIterator<Item = (f64, f64)>>(
    pts: I,
) -> Result<Vec<u8>, SedonaGeometryError> {
    let mut out_wkb = Vec::with_capacity(5 + 4 + 4 + pts.len() * 16);
    write_wkb_polygon(&mut out_wkb, pts)?;
    Ok(out_wkb)
}

/// Write WKB representing a POLYGON into a buffer
///
/// This can be used to build Binary arrays, as the arrow-rs BinaryBuilder
/// implements Write.
/// This function always writes little endian coordinates.
pub fn write_wkb_polygon<I: ExactSizeIterator<Item = (f64, f64)>>(
    buf: &mut impl Write,
    pts: I,
) -> Result<(), SedonaGeometryError> {
    let count: usize = if pts.len() == 0 { 0 } else { 1 };
    write_wkb_polygon_header(buf, Dimensions::Xy, count)?;
    if pts.len() == 0 {
        return Ok(());
    }
    write_wkb_polygon_ring_header(buf, pts.len())?;
    write_wkb_coords(buf, pts)?;

    Ok(())
}

/// Write WKB header for POLYGON
///
/// The number of points in the polygon is provided as `count`.
/// Each ring in the polygon is expected to have its own header,
/// which is provided by calling [write_wkb_polygon_ring_header].
/// This is followed by writing the ring's coordinates with [write_wkb_coords].
/// This function always writes little endian coordinates.
pub fn write_wkb_polygon_header(
    buf: &mut impl Write,
    dims: Dimensions,
    count: usize,
) -> Result<(), SedonaGeometryError> {
    buf.write_all(&[0x01])?;
    let code = WKB_POLYGON_TYPE + wkb_dim_code(dims)?;
    buf.write_all(&code.to_le_bytes())?;
    buf.write_all(&count_to_u32(count)?.to_le_bytes())?;
    Ok(())
}

/// Write WKB header for a polygon ring
///
/// This writes the number of points in the ring, which is
/// expected to be followed by the actual coordinates
/// This function always writes little endian coordinates.
pub fn write_wkb_polygon_ring_header(
    buf: &mut impl Write,
    count: usize,
) -> Result<(), SedonaGeometryError> {
    // Write the number of points in the ring
    let num_points: u32 = count.try_into()?;
    buf.write_all(&num_points.to_le_bytes())?;
    Ok(())
}

/// Create WKB representing a MULTIPOLYGON
///
/// A convenience wrapper for [write_wkb_multipolygon] that creates a Vec,
/// which is useful for creating DataFusion scalar values.
/// This function always writes little endian coordinates.
pub fn wkb_multipolygon<I>(polygons: I) -> Result<Vec<u8>, SedonaGeometryError>
where
    I: ExactSizeIterator<Item = Vec<(f64, f64)>>,
{
    // Base size + polygons count + memory for coordinate pairs and ring headers
    let capacity = 5 + 4 + polygons.len() * (4 + 16 * polygons.len());
    let mut out_wkb = Vec::with_capacity(capacity);
    write_wkb_multipolygon(&mut out_wkb, polygons)?;
    Ok(out_wkb)
}

/// Write WKB representing a MULTIPOLYGON into a buffer
///
/// This can be used to build Binary arrays, as the arrow-rs BinaryBuilder
/// implements Write.
/// This function always writes little endian coordinates.
pub fn write_wkb_multipolygon<I>(
    buf: &mut impl Write,
    polygons: I,
) -> Result<(), SedonaGeometryError>
where
    I: ExactSizeIterator<Item = Vec<(f64, f64)>>,
{
    write_wkb_multipolygon_header(buf, Dimensions::Xy, polygons.len())?;

    // For each polygon, write a complete polygon WKB
    for polygon in polygons {
        // Each polygon needs its own byte order and type
        buf.write_all(&[0x01, 0x03, 0x00, 0x00, 0x00])?;

        // If polygon is empty
        if polygon.is_empty() {
            buf.write_all(&[0x00, 0x00, 0x00, 0x00])?; // 0 rings
            continue;
        }

        // Write 1 ring (exterior only for now)
        buf.write_all(&[0x01, 0x00, 0x00, 0x00])?;

        // Write number of points in the polygon
        let num_points: u32 = polygon.len().try_into()?;
        buf.write_all(&num_points.to_le_bytes())?;

        // Write each point coordinate
        for pt in polygon {
            buf.write_all(&pt.0.to_le_bytes())?;
            buf.write_all(&pt.1.to_le_bytes())?;
        }
    }

    Ok(())
}

/// Create WKB representing a MULTIPOINT
///
/// A convenience wrapper for [write_wkb_multipoint] that creates a Vec,
/// which is useful for creating DataFusion scalar values.
/// This function always writes little endian coordinates.
pub fn wkb_multipoint<I>(points: I) -> Result<Vec<u8>, SedonaGeometryError>
where
    I: ExactSizeIterator<Item = (f64, f64)>,
{
    // Base size + points count
    let capacity = 5 + 4 + points.len() * 16;
    let mut out_wkb = Vec::with_capacity(capacity);
    write_wkb_multipoint(&mut out_wkb, points)?;
    Ok(out_wkb)
}

/// Write WKB representing a MULTIPOINT into a buffer
///
/// A convenience wrapper for [write_wkb_multipoint] that creates a Vec,
/// which is useful for creating DataFusion scalar values.
/// This function always writes little endian coordinates.
pub fn write_wkb_multipoint<I>(buf: &mut impl Write, points: I) -> Result<(), SedonaGeometryError>
where
    I: ExactSizeIterator<Item = (f64, f64)>,
{
    write_wkb_multipoint_header(buf, Dimensions::Xy, points.len())?;
    for pt in points {
        write_wkb_point_header(buf, Dimensions::Xy)?;
        write_wkb_coord(buf, pt)?;
    }

    Ok(())
}

/// Write WKB header for GEOMETRYCOLLECTION
/// This function always writes little endian coordinates.
pub fn write_wkb_geometrycollection_header(
    buf: &mut impl Write,
    dims: Dimensions,
    count: usize,
) -> Result<(), SedonaGeometryError> {
    buf.write_all(&[0x01])?;
    let code = WKB_GEOMETRYCOLLECTION_TYPE + wkb_dim_code(dims)?;
    buf.write_all(&code.to_le_bytes())?;
    buf.write_all(&count_to_u32(count)?.to_le_bytes())?;
    Ok(())
}

/// Write WKB header for MULTIPOLYGON
/// This function always writes little endian coordinates.
pub fn write_wkb_multipolygon_header(
    buf: &mut impl Write,
    dims: Dimensions,
    count: usize,
) -> Result<(), SedonaGeometryError> {
    buf.write_all(&[0x01])?;
    let code = WKB_MULTIPOLYGON_TYPE + wkb_dim_code(dims)?;
    buf.write_all(&code.to_le_bytes())?;
    buf.write_all(&count_to_u32(count)?.to_le_bytes())?;
    Ok(())
}

/// Write WKB header for MULTIPOINT
/// This function always writes little endian coordinates.
pub fn write_wkb_multipoint_header(
    buf: &mut impl Write,
    dims: Dimensions,
    count: usize,
) -> Result<(), SedonaGeometryError> {
    buf.write_all(&[0x01])?;
    let code = WKB_MULTIPOINT_TYPE + wkb_dim_code(dims)?;
    buf.write_all(&code.to_le_bytes())?;
    buf.write_all(&count_to_u32(count)?.to_le_bytes())?;
    Ok(())
}

/// Write WKB header for MULTILINESTRING
/// This function always writes little endian coordinates.
pub fn write_wkb_multilinestring_header(
    buf: &mut impl Write,
    dims: Dimensions,
    count: usize,
) -> Result<(), SedonaGeometryError> {
    buf.write_all(&[0x01])?;
    let code: u32 = WKB_MULTILINESTRING_TYPE + wkb_dim_code(dims)?;
    buf.write_all(&code.to_le_bytes())?;
    buf.write_all(&count_to_u32(count)?.to_le_bytes())?;
    Ok(())
}

/// Write a single coordinate to WKB
/// This function always writes little endian coordinates.
pub fn write_wkb_coord<C>(buf: &mut impl Write, coord: C) -> Result<(), SedonaGeometryError>
where
    C: WritableCoord,
{
    coord.write_to(buf)?;
    Ok(())
}

/// Write a single coordinate of CoordTrait to WKB
/// This function always writes little endian coordinates.
pub fn write_wkb_coord_trait<C>(buf: &mut impl Write, coord: &C) -> Result<(), SedonaGeometryError>
where
    C: CoordTrait<T = f64>,
{
    match coord.dim().size() {
        2 => {
            let coord_tuple = coord.x_y();
            write_wkb_coord(buf, coord_tuple)
        }
        3 => {
            let coord_tuple: (<C as CoordTrait>::T, _, _) =
                (coord.x(), coord.y(), coord.nth_or_panic(2));
            write_wkb_coord(buf, coord_tuple)
        }
        4 => {
            let coord_tuple = (
                coord.x(),
                coord.y(),
                coord.nth_or_panic(2),
                coord.nth_or_panic(3),
            );
            write_wkb_coord(buf, coord_tuple)
        }
        _ => Err(SedonaGeometryError::Invalid(
            "Unsupported number of dimensions".to_string(),
        )),
    }
}

/// Write multiple coordinates to WKB
///
/// This function takes an iterator of coordinates and writes them to the provided buffer.
/// This function always writes little endian coordinates.
pub fn write_wkb_coords<C, I>(buf: &mut impl Write, coords: I) -> Result<(), SedonaGeometryError>
where
    C: WritableCoord,
    I: Iterator<Item = C>,
{
    for coord in coords {
        coord.write_to(buf)?;
    }
    Ok(())
}

/// Trait to allow writing of coordinates for multiple dimensions
pub trait WritableCoord {
    fn write_to(&self, buf: &mut impl Write) -> Result<(), SedonaGeometryError>;
}

/// Implementation of WritableCoord for 2D
impl WritableCoord for (f64, f64) {
    fn write_to(&self, buf: &mut impl Write) -> Result<(), SedonaGeometryError> {
        buf.write_all(&self.0.to_le_bytes())?;
        buf.write_all(&self.1.to_le_bytes())?;
        Ok(())
    }
}

/// Implementation of WritableCoord for 3D
/// This function always writes little endian coordinates.
impl WritableCoord for (f64, f64, f64) {
    fn write_to(&self, buf: &mut impl Write) -> Result<(), SedonaGeometryError> {
        buf.write_all(&self.0.to_le_bytes())?;
        buf.write_all(&self.1.to_le_bytes())?;
        buf.write_all(&self.2.to_le_bytes())?;
        Ok(())
    }
}

/// Implementation of WritableCoord for 4D
/// This function always writes little endian coordinates.
impl WritableCoord for (f64, f64, f64, f64) {
    fn write_to(&self, buf: &mut impl Write) -> Result<(), SedonaGeometryError> {
        buf.write_all(&self.0.to_le_bytes())?;
        buf.write_all(&self.1.to_le_bytes())?;
        buf.write_all(&self.2.to_le_bytes())?;
        buf.write_all(&self.3.to_le_bytes())?;
        Ok(())
    }
}

fn wkb_dim_code(dimension: Dimensions) -> Result<u32, SedonaGeometryError> {
    match dimension {
        Dimensions::Xy => Ok(0),
        Dimensions::Xyz => Ok(1000),
        Dimensions::Xym => Ok(2000),
        Dimensions::Xyzm => Ok(3000),
        _ => Err(SedonaGeometryError::Invalid(
            "Unsupported dimensions writing wkb: ".to_string(),
        )),
    }
}

/// Convert a count of elements to u32
fn count_to_u32(count: usize) -> Result<u32, SedonaGeometryError> {
    count.try_into().map_err(|_| {
        SedonaGeometryError::Invalid(
            "Collection contains too many elements for WKB format".to_string(),
        )
    })
}

#[cfg(test)]
mod test {
    use std::str::FromStr;
    use wkb::reader::read_wkb;
    use wkb::writer::{write_geometry, WriteOptions};
    use wkb::Endianness;
    use wkt::Wkt;

    use super::*;

    #[test]
    fn test_wkb_point() {
        let wkt: Wkt = Wkt::from_str("POINT (0 1)").unwrap();
        let mut wkb = vec![];
        write_geometry(
            &mut wkb,
            &wkt,
            &WriteOptions {
                endianness: Endianness::LittleEndian,
            },
        )
        .unwrap();
        assert_eq!(wkb_point((0.0, 1.0)).unwrap(), wkb);
    }

    #[test]
    fn test_wkb_empty_point() {
        let mut wkb = Vec::new();
        write_wkb_empty_point(&mut wkb, Dimensions::Xy).unwrap();
        check_bytes(&wkb, "POINT EMPTY");

        wkb.clear();
        write_wkb_empty_point(&mut wkb, Dimensions::Xyz).unwrap();
        check_bytes(&wkb, "POINT Z EMPTY");

        wkb.clear();
        write_wkb_empty_point(&mut wkb, Dimensions::Xym).unwrap();
        check_bytes(&wkb, "POINT M EMPTY");

        wkb.clear();
        write_wkb_empty_point(&mut wkb, Dimensions::Xyzm).unwrap();
        check_bytes(&wkb, "POINT ZM EMPTY");
    }

    #[test]
    fn test_wkb_point_header() {
        let mut wkb = Vec::new();
        write_wkb_point_header(&mut wkb, Dimensions::Xy).unwrap();
        write_wkb_coords(&mut wkb, vec![(f64::NAN, f64::NAN)].into_iter()).unwrap();
        check_bytes(&wkb, "POINT EMPTY");

        wkb.clear();
        write_wkb_point_header(&mut wkb, Dimensions::Xy).unwrap();
        write_wkb_coords(&mut wkb, vec![(0.0, 1.0)].into_iter()).unwrap();
        check_bytes(&wkb, "POINT(0 1)");

        wkb.clear();
        write_wkb_point_header(&mut wkb, Dimensions::Xyz).unwrap();
        write_wkb_coords(&mut wkb, vec![(1.0, 1.0, 2.0)].into_iter()).unwrap();
        check_bytes(&wkb, "POINT Z(1 1 2)");

        wkb.clear();
        write_wkb_point_header(&mut wkb, Dimensions::Xym).unwrap();
        write_wkb_coords(&mut wkb, vec![(6.0, 7.0, 8.0)].into_iter()).unwrap();
        check_bytes(&wkb, "POINT M(6 7 8)");

        wkb.clear();
        write_wkb_point_header(&mut wkb, Dimensions::Xyzm).unwrap();
        write_wkb_coords(&mut wkb, vec![(12.0, 13.0, 14.0, 15.0)].into_iter()).unwrap();
        check_bytes(&wkb, "POINT ZM(12 13 14 15)");
    }

    #[test]
    fn test_write_wkb_coord_trait() {
        let cases = [
            (None, None, "POINT(0 1)"),
            (Some(2.0), None, "POINT Z(0 1 2)"),
            (None, Some(3.0), "POINT M(0 1 3)"),
            (Some(2.0), Some(3.0), "POINT ZM(0 1 2 3)"),
        ];
        let mut wkb = vec![];

        for (z, m, expected) in cases {
            let coord = wkt::types::Coord {
                x: 0.0,
                y: 1.0,
                z,
                m,
            };

            wkb.clear();
            write_wkb_point_header(&mut wkb, coord.dim()).unwrap();
            write_wkb_coord_trait(&mut wkb, &coord).unwrap();
            check_bytes(&wkb, expected);
        }
    }

    #[test]
    fn test_wkb_linestring() {
        let wkt: Wkt = Wkt::from_str("LINESTRING EMPTY").unwrap();
        let mut wkb = vec![];
        write_geometry(
            &mut wkb,
            &wkt,
            &WriteOptions {
                endianness: Endianness::LittleEndian,
            },
        )
        .unwrap();
        assert_eq!(wkb_linestring([].into_iter()).unwrap(), wkb);

        let wkt: Wkt = Wkt::from_str("LINESTRING (0 1, 2 3)").unwrap();
        let mut wkb = vec![];
        write_geometry(
            &mut wkb,
            &wkt,
            &WriteOptions {
                endianness: Endianness::LittleEndian,
            },
        )
        .unwrap();
        assert_eq!(
            wkb_linestring([(0.0, 1.0), (2.0, 3.0)].into_iter()).unwrap(),
            wkb
        );
    }

    #[test]
    fn test_wkb_linestring_header() {
        let mut wkb = Vec::new();
        write_wkb_linestring_header(&mut wkb, Dimensions::Xy, 0).unwrap();
        check_bytes(&wkb, "LINESTRING EMPTY");

        wkb.clear();
        write_wkb_linestring_header(&mut wkb, Dimensions::Xy, 1).unwrap();
        write_wkb_coords(&mut wkb, vec![(0.0, 1.0)].into_iter()).unwrap();
        check_bytes(&wkb, "LINESTRING(0 1)");

        wkb.clear();
        write_wkb_linestring_header(&mut wkb, Dimensions::Xyz, 2).unwrap();
        write_wkb_coords(&mut wkb, vec![(1.0, 1.0, 2.0), (3.0, 4.0, 5.0)].into_iter()).unwrap();
        check_bytes(&wkb, "LINESTRING Z(1 1 2,3 4 5)");

        wkb.clear();
        write_wkb_linestring_header(&mut wkb, Dimensions::Xym, 2).unwrap();
        write_wkb_coords(
            &mut wkb,
            vec![(6.0, 7.0, 8.0), (9.0, 10.0, 11.0)].into_iter(),
        )
        .unwrap();
        check_bytes(&wkb, "LINESTRING M(6 7 8,9 10 11)");

        wkb.clear();
        write_wkb_linestring_header(&mut wkb, Dimensions::Xyzm, 2).unwrap();
        write_wkb_coords(
            &mut wkb,
            vec![(12.0, 13.0, 14.0, 15.0), (16.0, 17.0, 18.0, 19.0)].into_iter(),
        )
        .unwrap();
        check_bytes(&wkb, "LINESTRING ZM(12 13 14 15,16 17 18 19)");
    }

    #[test]
    fn test_wkb_multilinestring() {
        let wkt: Wkt = Wkt::from_str("MULTILINESTRING EMPTY").unwrap();
        let mut wkb = vec![];
        write_geometry(
            &mut wkb,
            &wkt,
            &WriteOptions {
                endianness: Endianness::LittleEndian,
            },
        )
        .unwrap();
        assert_eq!(wkb_multilinestring([].into_iter()).unwrap(), wkb);

        let wkt: Wkt = Wkt::from_str("MULTILINESTRING ((0 0, 1 1, 2 2), (3 3, 4 4))").unwrap();
        let mut wkb = vec![];
        write_geometry(
            &mut wkb,
            &wkt,
            &WriteOptions {
                endianness: Endianness::LittleEndian,
            },
        )
        .unwrap();

        let linestrings = vec![
            vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
            vec![(3.0, 3.0), (4.0, 4.0)],
        ];

        assert_eq!(wkb_multilinestring(linestrings.into_iter()).unwrap(), wkb);
    }

    #[test]
    fn test_wkb_polygon() {
        let wkt: Wkt = Wkt::from_str("POLYGON EMPTY").unwrap();
        let mut wkb = vec![];
        write_geometry(
            &mut wkb,
            &wkt,
            &WriteOptions {
                endianness: Endianness::LittleEndian,
            },
        )
        .unwrap();
        assert_eq!(wkb_polygon([].into_iter()).unwrap(), wkb);

        let wkt: Wkt = Wkt::from_str("POLYGON ((0 0, 1 0, 0 1, 0 0))").unwrap();
        let mut wkb = vec![];
        write_geometry(
            &mut wkb,
            &wkt,
            &WriteOptions {
                endianness: Endianness::LittleEndian,
            },
        )
        .unwrap();
        assert_eq!(
            wkb_polygon([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)].into_iter()).unwrap(),
            wkb
        );
    }

    #[test]
    fn test_wkb_polygon_header() {
        let mut wkb = Vec::new();
        write_wkb_polygon_header(&mut wkb, Dimensions::Xy, 0).unwrap();
        check_bytes(&wkb, "POLYGON EMPTY");

        wkb.clear();
        write_wkb_polygon_header(&mut wkb, Dimensions::Xy, 1).unwrap();
        write_wkb_polygon_ring_header(&mut wkb, 3).unwrap();
        write_wkb_coords(
            &mut wkb,
            vec![(0.0, 1.0), (1.0, 1.0), (0.0, 1.0)].into_iter(),
        )
        .unwrap();
        check_bytes(&wkb, "POLYGON((0 1,1 1,0 1))");

        wkb.clear();
        write_wkb_polygon_header(&mut wkb, Dimensions::Xyz, 1).unwrap();
        write_wkb_polygon_ring_header(&mut wkb, 3).unwrap();
        write_wkb_coords(
            &mut wkb,
            vec![(1.0, 1.0, 2.0), (3.0, 4.0, 5.0), (1.0, 1.0, 2.0)].into_iter(),
        )
        .unwrap();
        check_bytes(&wkb, "POLYGON Z((1 1 2,3 4 5,1 1 2))");

        wkb.clear();
        write_wkb_polygon_header(&mut wkb, Dimensions::Xym, 1).unwrap();
        write_wkb_polygon_ring_header(&mut wkb, 3).unwrap();
        write_wkb_coords(
            &mut wkb,
            vec![(6.0, 7.0, 8.0), (9.0, 10.0, 11.0), (6.0, 7.0, 8.0)].into_iter(),
        )
        .unwrap();
        check_bytes(&wkb, "POLYGON M((6 7 8,9 10 11,6 7 8))");

        wkb.clear();
        write_wkb_polygon_header(&mut wkb, Dimensions::Xyzm, 1).unwrap();
        write_wkb_polygon_ring_header(&mut wkb, 3).unwrap();
        write_wkb_coords(
            &mut wkb,
            vec![
                (12.0, 13.0, 14.0, 15.0),
                (16.0, 17.0, 18.0, 19.0),
                (12.0, 13.0, 14.0, 15.0),
            ]
            .into_iter(),
        )
        .unwrap();
        check_bytes(&wkb, "POLYGON ZM((12 13 14 15,16 17 18 19,12 13 14 15))");

        // Polygon with inner rings
        wkb.clear();
        write_wkb_polygon_header(&mut wkb, Dimensions::Xy, 3).unwrap();
        write_wkb_polygon_ring_header(&mut wkb, 3).unwrap();
        write_wkb_coords(
            &mut wkb,
            vec![(0.0, 1.0), (1.0, 1.0), (0.0, 1.0)].into_iter(),
        )
        .unwrap();
        write_wkb_polygon_ring_header(&mut wkb, 4).unwrap();
        write_wkb_coords(
            &mut wkb,
            vec![(0.0, 2.0), (2.0, 2.0), (3.0, 3.0), (0.0, 2.0)].into_iter(),
        )
        .unwrap();
        write_wkb_polygon_ring_header(&mut wkb, 4).unwrap();
        write_wkb_coords(
            &mut wkb,
            vec![(0.0, 4.0), (4.0, 4.0), (5.0, 5.0), (0.0, 2.0)].into_iter(),
        )
        .unwrap();
        check_bytes(
            &wkb,
            "POLYGON((0 1,1 1,0 1),(0 2,2 2,3 3,0 2),(0 4,4 4,5 5,0 2))",
        );
    }

    #[test]
    fn test_wkb_multipolygon() {
        let wkt: Wkt = Wkt::from_str("MULTIPOLYGON EMPTY").unwrap();
        let mut wkb = vec![];
        write_geometry(
            &mut wkb,
            &wkt,
            &WriteOptions {
                endianness: Endianness::LittleEndian,
            },
        )
        .unwrap();
        assert_eq!(wkb_multipolygon([].into_iter()).unwrap(), wkb);

        let wkt: Wkt =
            Wkt::from_str("MULTIPOLYGON (((0 0, 1 0, 0 1, 0 0)), ((2 2, 3 2, 2 3, 2 2)))").unwrap();
        let mut wkb = vec![];
        write_geometry(
            &mut wkb,
            &wkt,
            &WriteOptions {
                endianness: Endianness::LittleEndian,
            },
        )
        .unwrap();

        let polygons = vec![
            vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)],
            vec![(2.0, 2.0), (3.0, 2.0), (2.0, 3.0), (2.0, 2.0)],
        ];

        assert_eq!(wkb_multipolygon(polygons.into_iter()).unwrap(), wkb);
    }

    #[test]
    fn test_wkb_multipoint() {
        let wkt: Wkt = Wkt::from_str("MULTIPOINT EMPTY").unwrap();
        let mut wkb = vec![];
        write_geometry(
            &mut wkb,
            &wkt,
            &WriteOptions {
                endianness: Endianness::LittleEndian,
            },
        )
        .unwrap();
        assert_eq!(wkb_multipoint([].into_iter()).unwrap(), wkb);

        let wkt: Wkt = Wkt::from_str("MULTIPOINT ((0 0), (1 1))").unwrap();
        let mut wkb = vec![];
        write_geometry(
            &mut wkb,
            &wkt,
            &WriteOptions {
                endianness: Endianness::LittleEndian,
            },
        )
        .unwrap();

        let points = vec![(0.0, 0.0), (1.0, 1.0)];
        assert_eq!(wkb_multipoint(points.into_iter()).unwrap(), wkb);
    }

    #[test]
    fn test_wkb_multilinestring_header() {
        // Test empty header
        let mut wkb = Vec::new();
        write_wkb_multilinestring_header(&mut wkb, Dimensions::Xy, 0).unwrap();

        // Expected bytes for empty multilinestring:
        // - 0x01 for little endian byte order
        // - 0x05, 0x00, 0x00, 0x00 for geometry type 5 (MultiLineString)
        // - 0x00, 0x00, 0x00, 0x00 for count of 0 linestrings
        let expected = vec![0x01, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(wkb, expected);

        // Test non-empty header
        let mut wkb = Vec::new();
        write_wkb_multilinestring_header(&mut wkb, Dimensions::Xy, 3).unwrap();

        // Expected bytes for multilinestring with 3 linestrings:
        // - 0x01 for little endian byte order
        // - 0x05, 0x00, 0x00, 0x00 for geometry type 5 (MultiLineString)
        // - 0x03, 0x00, 0x00, 0x00 for count of 3 linestrings
        let expected = vec![0x01, 0x05, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00];
        assert_eq!(wkb, expected);

        let mut wkb = Vec::new();
        write_wkb_multilinestring_header(&mut wkb, Dimensions::Xy, 0).unwrap();
        check_bytes(&wkb, "MULTILINESTRING EMPTY");

        wkb.clear();
        write_wkb_multilinestring_header(&mut wkb, Dimensions::Xy, 2).unwrap();
        write_wkb_linestring_header(&mut wkb, Dimensions::Xy, 1).unwrap();
        write_wkb_coords(&mut wkb, vec![(0.0, 1.0)].into_iter()).unwrap();
        write_wkb_linestring_header(&mut wkb, Dimensions::Xy, 3).unwrap();
        write_wkb_coords(
            &mut wkb,
            vec![(2.0, 3.0), (4.0, 5.0), (6.0, 7.0)].into_iter(),
        )
        .unwrap();
        check_bytes(&wkb, "MULTILINESTRING((0 1),(2 3,4 5,6 7))");

        wkb.clear();
        write_wkb_multilinestring_header(&mut wkb, Dimensions::Xyz, 2).unwrap();
        write_wkb_linestring_header(&mut wkb, Dimensions::Xyz, 2).unwrap();
        write_wkb_coords(&mut wkb, vec![(1.0, 1.0, 2.0), (3.0, 4.0, 5.0)].into_iter()).unwrap();
        write_wkb_linestring_header(&mut wkb, Dimensions::Xyz, 2).unwrap();
        write_wkb_coords(
            &mut wkb,
            vec![(6.0, 7.0, 8.0), (9.0, 10.0, 11.0)].into_iter(),
        )
        .unwrap();
        check_bytes(&wkb, "MULTILINESTRING Z((1 1 2,3 4 5),(6 7 8,9 10 11))");

        wkb.clear();
        write_wkb_multilinestring_header(&mut wkb, Dimensions::Xym, 2).unwrap();
        write_wkb_linestring_header(&mut wkb, Dimensions::Xym, 2).unwrap();
        write_wkb_coords(
            &mut wkb,
            vec![(12.0, 13.0, 14.0), (15.0, 16.0, 17.0)].into_iter(),
        )
        .unwrap();
        write_wkb_linestring_header(&mut wkb, Dimensions::Xym, 2).unwrap();
        write_wkb_coords(
            &mut wkb,
            vec![(18.0, 19.0, 20.0), (21.0, 22.0, 23.0)].into_iter(),
        )
        .unwrap();
        check_bytes(
            &wkb,
            "MULTILINESTRING M((12 13 14,15 16 17),(18 19 20,21 22 23))",
        );

        wkb.clear();
        write_wkb_multilinestring_header(&mut wkb, Dimensions::Xyzm, 2).unwrap();
        write_wkb_linestring_header(&mut wkb, Dimensions::Xyzm, 2).unwrap();
        write_wkb_coords(
            &mut wkb,
            vec![(24.0, 25.0, 26.0, 27.0), (28.0, 29.0, 30.0, 31.0)].into_iter(),
        )
        .unwrap();
        write_wkb_linestring_header(&mut wkb, Dimensions::Xyzm, 2).unwrap();
        write_wkb_coords(
            &mut wkb,
            vec![(32.0, 33.0, 34.0, 35.0), (36.0, 37.0, 38.0, 39.0)].into_iter(),
        )
        .unwrap();
        check_bytes(
            &wkb,
            "MULTILINESTRING ZM((24 25 26 27,28 29 30 31),(32 33 34 35,36 37 38 39))",
        );
    }

    #[test]
    fn test_wkb_multipoint_header() {
        // Test empty header
        let mut wkb = Vec::new();
        write_wkb_multipoint_header(&mut wkb, Dimensions::Xy, 0).unwrap();

        // Expected bytes for empty multipoint:
        // - 0x01 for little endian byte order
        // - 0x04, 0x00, 0x00, 0x00 for geometry type 4 (MultiPoint)
        // - 0x00, 0x00, 0x00, 0x00 for count of 0 points
        let expected = vec![0x01, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(wkb, expected);

        // Test non-empty header
        let mut wkb = Vec::new();
        write_wkb_multipoint_header(&mut wkb, Dimensions::Xy, 5).unwrap();

        // Expected bytes for multipoint with 5 points:
        // - 0x01 for little endian byte order
        // - 0x04, 0x00, 0x00, 0x00 for geometry type 4 (MultiPoint)
        // - 0x05, 0x00, 0x00, 0x00 for count of 5 points
        let expected = vec![0x01, 0x04, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00];
        assert_eq!(wkb, expected);

        let mut wkb = Vec::new();
        write_wkb_multipoint_header(&mut wkb, Dimensions::Xy, 0).unwrap();
        check_bytes(&wkb, "MULTIPOINT EMPTY");

        wkb.clear();
        write_wkb_multipoint_header(&mut wkb, Dimensions::Xy, 3).unwrap();
        for pt in vec![(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)].into_iter() {
            write_wkb_point_header(&mut wkb, Dimensions::Xy).unwrap();
            write_wkb_coord(&mut wkb, pt).unwrap();
        }
        check_bytes(&wkb, "MULTIPOINT((0 1),(2 3),(4 5))");

        wkb.clear();
        write_wkb_multipoint_header(&mut wkb, Dimensions::Xyz, 3).unwrap();
        for pt in vec![(1.0, 1.0, 2.0), (3.0, 4.0, 5.0), (6.0, 7.0, 8.0)].into_iter() {
            write_wkb_point_header(&mut wkb, Dimensions::Xyz).unwrap();
            write_wkb_coord(&mut wkb, pt).unwrap();
        }
        check_bytes(&wkb, "MULTIPOINT Z((1 1 2),(3 4 5),(6 7 8))");

        wkb.clear();
        write_wkb_multipoint_header(&mut wkb, Dimensions::Xym, 3).unwrap();
        for pt in vec![(9.0, 10.0, 11.0), (12.0, 13.0, 14.0), (15.0, 16.0, 17.0)].into_iter() {
            write_wkb_point_header(&mut wkb, Dimensions::Xym).unwrap();
            write_wkb_coord(&mut wkb, pt).unwrap();
        }
        check_bytes(&wkb, "MULTIPOINT M((9 10 11),(12 13 14),(15 16 17))");

        wkb.clear();
        write_wkb_multipoint_header(&mut wkb, Dimensions::Xyzm, 3).unwrap();
        for pt in [
            (18.0, 19.0, 20.0, 21.0),
            (22.0, 23.0, 24.0, 25.0),
            (26.0, 27.0, 28.0, 29.0),
        ] {
            write_wkb_point_header(&mut wkb, Dimensions::Xyzm).unwrap();
            write_wkb_coord(&mut wkb, pt).unwrap();
        }
        check_bytes(
            &wkb,
            "MULTIPOINT ZM((18 19 20 21),(22 23 24 25),(26 27 28 29))",
        );
    }

    #[test]
    fn test_wkb_multipolygon_header() {
        // Test empty header
        let mut wkb = Vec::new();
        write_wkb_multipolygon_header(&mut wkb, Dimensions::Xy, 0).unwrap();

        // Expected bytes for empty multipolygon:
        // - 0x01 for little endian byte order
        // - 0x06, 0x00, 0x00, 0x00 for geometry type 6 (MultiPolygon)
        // - 0x00, 0x00, 0x00, 0x00 for count of 0 polygons
        let expected = vec![0x01, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(wkb, expected);

        // Test non-empty header
        let mut wkb = Vec::new();
        write_wkb_multipolygon_header(&mut wkb, Dimensions::Xy, 4).unwrap();

        // Expected bytes for multipolygon with 4 polygons:
        // - 0x01 for little endian byte order
        // - 0x06, 0x00, 0x00, 0x00 for geometry type 6 (MultiPolygon)
        // - 0x04, 0x00, 0x00, 0x00 for count of 4 polygons
        let expected = vec![0x01, 0x06, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00];
        assert_eq!(wkb, expected);

        let mut wkb = Vec::new();
        write_wkb_multipolygon_header(&mut wkb, Dimensions::Xy, 0).unwrap();
        check_bytes(&wkb, "MULTIPOLYGON EMPTY");

        wkb.clear();
        write_wkb_multipolygon_header(&mut wkb, Dimensions::Xy, 2).unwrap();
        let polygons = vec![
            vec![(0.0, 1.0), (1.0, 1.0), (0.0, 1.0)],
            vec![(2.0, 3.0), (4.0, 5.0), (6.0, 7.0)],
        ];
        for polygon in polygons {
            write_wkb_polygon_header(&mut wkb, Dimensions::Xy, 1).unwrap();
            write_wkb_polygon_ring_header(&mut wkb, polygon.len()).unwrap();
            write_wkb_coords(&mut wkb, polygon.into_iter()).unwrap();
        }
        check_bytes(&wkb, "MULTIPOLYGON(((0 1,1 1,0 1)),((2 3,4 5,6 7)))");

        wkb.clear();
        write_wkb_multipolygon_header(&mut wkb, Dimensions::Xyz, 2).unwrap();
        let polygons = vec![
            vec![(1.0, 1.0, 2.0), (3.0, 4.0, 5.0), (1.0, 1.0, 2.0)],
            vec![(6.0, 7.0, 8.0), (9.0, 10.0, 11.0), (12.0, 13.0, 14.0)],
        ];
        for polygon in polygons {
            write_wkb_polygon_header(&mut wkb, Dimensions::Xyz, 1).unwrap();
            write_wkb_polygon_ring_header(&mut wkb, polygon.len()).unwrap();
            write_wkb_coords(&mut wkb, polygon.into_iter()).unwrap();
        }
        check_bytes(
            &wkb,
            "MULTIPOLYGON Z(((1 1 2,3 4 5,1 1 2)),((6 7 8,9 10 11,12 13 14)))",
        );

        wkb.clear();
        write_wkb_multipolygon_header(&mut wkb, Dimensions::Xym, 2).unwrap();
        let polygons = vec![
            vec![(1.0, 1.0, 2.0), (3.0, 4.0, 5.0), (1.0, 1.0, 2.0)],
            vec![(6.0, 7.0, 8.0), (9.0, 10.0, 11.0), (12.0, 13.0, 14.0)],
        ];
        for polygon in polygons {
            write_wkb_polygon_header(&mut wkb, Dimensions::Xym, 1).unwrap();
            write_wkb_polygon_ring_header(&mut wkb, polygon.len()).unwrap();
            write_wkb_coords(&mut wkb, polygon.into_iter()).unwrap();
        }
        check_bytes(
            &wkb,
            "MULTIPOLYGON M(((1 1 2,3 4 5,1 1 2)),((6 7 8,9 10 11,12 13 14)))",
        );

        wkb.clear();
        write_wkb_multipolygon_header(&mut wkb, Dimensions::Xyzm, 2).unwrap();
        let polygons = vec![
            vec![
                (1.0, 1.0, 2.0, 3.0),
                (4.0, 5.0, 6.0, 7.0),
                (1.0, 1.0, 2.0, 3.0),
            ],
            vec![
                (8.0, 9.0, 10.0, 11.0),
                (12.0, 13.0, 14.0, 15.0),
                (16.0, 17.0, 18.0, 19.0),
            ],
        ];
        for polygon in polygons {
            write_wkb_polygon_header(&mut wkb, Dimensions::Xyzm, 1).unwrap();
            write_wkb_polygon_ring_header(&mut wkb, polygon.len()).unwrap();
            write_wkb_coords(&mut wkb, polygon.into_iter()).unwrap();
        }
        check_bytes(
            &wkb,
            "MULTIPOLYGON ZM(((1 1 2 3,4 5 6 7,1 1 2 3)),((8 9 10 11,12 13 14 15,16 17 18 19)))",
        );
    }

    #[test]
    fn test_wkb_geometrycollection_header() {
        let mut wkb = Vec::new();
        write_wkb_geometrycollection_header(&mut wkb, Dimensions::Xy, 0).unwrap();
        let expected = vec![0x01, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(wkb, expected);

        let mut wkb = Vec::new();
        write_wkb_geometrycollection_header(&mut wkb, Dimensions::Xy, 2).unwrap();

        // Expected bytes:
        // - 0x01 for little endian byte order
        // - 0x07, 0x00, 0x00, 0x00 for geometry type 7 (GeometryCollection)
        // - 0x02, 0x00, 0x00, 0x00 for count of 2 geometries
        let expected = vec![0x01, 0x07, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00];
        assert_eq!(wkb, expected);

        let mut wkb = Vec::new();
        write_wkb_geometrycollection_header(&mut wkb, Dimensions::Xy, 0).unwrap();
        check_bytes(&wkb, "GEOMETRYCOLLECTION EMPTY");

        wkb.clear();
        write_wkb_geometrycollection_header(&mut wkb, Dimensions::Xy, 2).unwrap();
        write_wkb_point_header(&mut wkb, Dimensions::Xy).unwrap();
        write_wkb_coord(&mut wkb, (0.0, 1.0)).unwrap();
        write_wkb_linestring_header(&mut wkb, Dimensions::Xy, 2).unwrap();
        write_wkb_coords(&mut wkb, vec![(2.0, 3.0), (4.0, 5.0)].into_iter()).unwrap();
        check_bytes(&wkb, "GEOMETRYCOLLECTION(POINT(0 1),LINESTRING(2 3,4 5))");

        wkb.clear();
        write_wkb_geometrycollection_header(&mut wkb, Dimensions::Xyz, 2).unwrap();
        write_wkb_point_header(&mut wkb, Dimensions::Xyz).unwrap();
        write_wkb_coord(&mut wkb, (1.0, 1.0, 2.0)).unwrap();
        write_wkb_linestring_header(&mut wkb, Dimensions::Xyz, 2).unwrap();
        write_wkb_coords(&mut wkb, vec![(3.0, 4.0, 5.0), (6.0, 7.0, 8.0)].into_iter()).unwrap();
        check_bytes(
            &wkb,
            "GEOMETRYCOLLECTION Z(POINT Z(1 1 2),LINESTRING Z(3 4 5,6 7 8))",
        );

        wkb.clear();
        write_wkb_geometrycollection_header(&mut wkb, Dimensions::Xym, 2).unwrap();
        write_wkb_point_header(&mut wkb, Dimensions::Xym).unwrap();
        write_wkb_coord(&mut wkb, (9.0, 10.0, 11.0)).unwrap();
        write_wkb_linestring_header(&mut wkb, Dimensions::Xym, 2).unwrap();
        write_wkb_coords(
            &mut wkb,
            vec![(12.0, 13.0, 14.0), (15.0, 16.0, 17.0)].into_iter(),
        )
        .unwrap();
        check_bytes(
            &wkb,
            "GEOMETRYCOLLECTION M(POINT M(9 10 11),LINESTRING M(12 13 14,15 16 17))",
        );

        wkb.clear();
        write_wkb_geometrycollection_header(&mut wkb, Dimensions::Xyzm, 2).unwrap();
        write_wkb_point_header(&mut wkb, Dimensions::Xyzm).unwrap();
        write_wkb_coord(&mut wkb, (18.0, 19.0, 20.0, 21.0)).unwrap();
        write_wkb_linestring_header(&mut wkb, Dimensions::Xyzm, 2).unwrap();
        write_wkb_coords(
            &mut wkb,
            vec![(22.0, 23.0, 24.0, 25.0), (26.0, 27.0, 28.0, 29.0)].into_iter(),
        )
        .unwrap();
        check_bytes(
            &wkb,
            "GEOMETRYCOLLECTION ZM(POINT ZM(18 19 20 21),LINESTRING ZM(22 23 24 25,26 27 28 29))",
        );
    }

    fn check_bytes(buf: &[u8], expected: &str) {
        let mut wkt = String::new();
        let wkb_reader = read_wkb(buf).unwrap();
        wkt::to_wkt::write_geometry(&mut wkt, &wkb_reader).unwrap();
        assert_eq!(wkt, expected);
    }
}
