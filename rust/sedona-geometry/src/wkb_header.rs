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

use geo_traits::Dimensions;

use crate::error::SedonaGeometryError;
use crate::types::GeometryTypeId;

const Z_FLAG_BIT: u32 = 0x80000000;
const M_FLAG_BIT: u32 = 0x40000000;
const SRID_FLAG_BIT: u32 = 0x20000000;

/// Fast-path WKB header parser
/// Performs operations lazily and caches them after the first computation
#[derive(Debug)]
pub struct WkbHeader {
    geometry_type: u32,
    // Not applicable for a point
    // number of points for a linestring
    // number of rings for a polygon
    // number of geometries for a MULTIPOINT, MULTILINESTRING, MULTIPOLYGON, or GEOMETRYCOLLECTION
    size: u32,
    // SRID if given buffer was EWKB. Otherwise, 0.
    srid: u32,
    // First x,y coordinates for a point. Otherwise (f64::NAN, f64::NAN) if empty
    first_xy: (f64, f64),
    // Dimensions of the first nested geometry of a collection or None if empty
    // For POINT, LINESTRING, POLYGON, returns the dimensions of the geometry
    first_geom_dimensions: Option<Dimensions>,
}

impl WkbHeader {
    /// Creates a new [WkbHeader] from a buffer
    pub fn try_new(buf: &[u8]) -> Result<Self, SedonaGeometryError> {
        let mut wkb_buffer = WkbBuffer::new(buf);

        wkb_buffer.read_endian()?;

        let geometry_type = wkb_buffer.read_u32()?;

        let geometry_type_id = GeometryTypeId::try_from_wkb_id(geometry_type & 0x7)?;

        let mut srid = 0;
        // if EWKB
        if geometry_type & SRID_FLAG_BIT != 0 {
            srid = wkb_buffer.read_u32()?;
        }

        let size = if geometry_type_id == GeometryTypeId::Point {
            // Dummy value for a point
            1
        } else {
            wkb_buffer.read_u32()?
        };

        // Default values for empty geometries
        let first_x;
        let first_y;
        let first_geom_dimensions: Option<Dimensions>;

        wkb_buffer.set_offset(0);

        let first_geom_idx = wkb_buffer.first_geom_idx()?;
        if let Some(i) = first_geom_idx {
            // Reset to first_geom_idx and parse the dimensions
            wkb_buffer.set_offset(i);
            // Parse dimension
            wkb_buffer.read_endian()?;
            let code = wkb_buffer.read_u32()?;
            first_geom_dimensions = Some(calc_dimensions(code)?);

            // For first_xy_coord, we need to pass the buffer starting from the geometry header
            wkb_buffer.set_offset(i);
            (first_x, first_y) = wkb_buffer.first_xy_coord()?;
        } else {
            first_geom_dimensions = None;
            first_x = f64::NAN;
            first_y = f64::NAN;
        }

        Ok(Self {
            geometry_type,
            srid,
            size,
            first_xy: (first_x, first_y),
            first_geom_dimensions,
        })
    }

    /// Returns the [GeometryTypeId] of the WKB by only parsing the header instead of the entire WKB
    pub fn geometry_type_id(&self) -> Result<GeometryTypeId, SedonaGeometryError> {
        // Only low 3 bits is for the base type, high bits include additional info
        let code = self.geometry_type & 0x7;

        let geometry_type_id = GeometryTypeId::try_from_wkb_id(code)?;

        Ok(geometry_type_id)
    }

    /// Returns the size of the geometry
    ///
    /// - 1 for Points
    /// - Number of points for a linestring
    /// - Number of rings for a polygon
    /// - Number of geometries for a MULTIPOINT, MULTILINESTRING, MULTIPOLYGON, or GEOMETRYCOLLECTION
    pub fn size(&self) -> u32 {
        self.size
    }

    /// Returns the SRID if given buffer was EWKB. Otherwise, 0.
    pub fn srid(&self) -> u32 {
        self.srid
    }

    /// Returns the first x, y coordinates for a point. Otherwise (f64::NAN, f64::NAN) if empty
    pub fn first_xy(&self) -> (f64, f64) {
        self.first_xy
    }

    /// Returns the top-level dimension of the WKB
    pub fn dimensions(&self) -> Result<Dimensions, SedonaGeometryError> {
        calc_dimensions(self.geometry_type)
    }

    /// Returns the dimensions of the first coordinate of the geometry
    pub fn first_geom_dimensions(&self) -> Option<Dimensions> {
        self.first_geom_dimensions
    }

    /// Returns true if this geometry is EMPTY or false otherwise
    pub fn is_empty(&self) -> Result<bool, SedonaGeometryError> {
        let geometry_type_id = self.geometry_type_id()?;
        if geometry_type_id == GeometryTypeId::Point {
            let (x, y) = self.first_xy();
            return Ok(x.is_nan() && y.is_nan());
        }
        Ok(self.size == 0)
    }
}

/// The coordinate layout of a WKB Point, discovered by reading only its header.
///
/// This is the fast-path primitive for sampling/indexing by Point. A Point WKB
/// has a fixed layout (`byte order`, `type code`, optional SRID, then the
/// coordinate), so [`try_from_wkb`](Self::try_from_wkb) can cheaply decide
/// whether a buffer is a Point — reading only the first few header bytes, never
/// a coordinate — and a later [`read_xy`](Self::read_xy) reads the coordinate
/// from the offset it already found.
///
/// Splitting the header inspection from the coordinate read lets a binary
/// function (e.g. distance) test *both* operands first and read coordinates only
/// when both are Points, without re-parsing the header or wasting a coordinate
/// read on rows that fall back to the general parser.
#[derive(Debug, Clone, Copy)]
pub struct WkbPointLayout {
    little_endian: bool,
    coord_offset: usize,
}

impl WkbPointLayout {
    /// Inspects a WKB header, returning the Point layout or `Ok(None)` if `buf`
    /// is not a Point. Reads only the byte order and type code (recognizing the
    /// EWKB SRID prefix) — never a coordinate. Errors only on a missing/invalid
    /// byte order or a truncated type code. Accepts both ISO (with Z/M dimension
    /// flags) and EWKB (with Z/M/SRID flags) encodings.
    #[inline]
    pub fn try_from_wkb(buf: &[u8]) -> Result<Option<Self>, SedonaGeometryError> {
        let little_endian = match buf.first() {
            Some(1) => true,
            Some(0) => false,
            _ => {
                return Err(SedonaGeometryError::Invalid(
                    "WKB: missing or invalid byte order".to_string(),
                ))
            }
        };

        let type_bytes: [u8; 4] = buf
            .get(1..5)
            .ok_or_else(|| SedonaGeometryError::Invalid("WKB: truncated header".to_string()))?
            .try_into()
            .unwrap();
        let type_code = if little_endian {
            u32::from_le_bytes(type_bytes)
        } else {
            u32::from_be_bytes(type_bytes)
        };

        // The low three bits of the base type identify a Point (1) regardless of
        // the ISO dimension digits or EWKB high-bit flags.
        if type_code & 0x7 != 1 {
            return Ok(None);
        }

        // EWKB carries a 4-byte SRID between the type code and the coordinate.
        let has_srid = type_code & SRID_FLAG_BIT != 0;
        let coord_offset = if has_srid { 9 } else { 5 };

        Ok(Some(Self {
            little_endian,
            coord_offset,
        }))
    }

    /// Reads the `(x, y)` from `buf`, which must be the same buffer passed to
    /// [`try_from_wkb`](Self::try_from_wkb). Returns `Ok(None)` for `POINT EMPTY`
    /// (encoded as `NaN`/`NaN`); errors if the coordinate is truncated.
    #[inline]
    pub fn read_xy(&self, buf: &[u8]) -> Result<Option<(f64, f64)>, SedonaGeometryError> {
        let x = self.read_f64(buf, self.coord_offset)?;
        let y = self.read_f64(buf, self.coord_offset + 8)?;
        if x.is_nan() && y.is_nan() {
            Ok(None)
        } else {
            Ok(Some((x, y)))
        }
    }

    #[inline]
    fn read_f64(&self, buf: &[u8], offset: usize) -> Result<f64, SedonaGeometryError> {
        let bytes: [u8; 8] = buf
            .get(offset..offset + 8)
            .ok_or_else(|| SedonaGeometryError::Invalid("WKB: truncated coordinate".to_string()))?
            .try_into()
            .unwrap();
        Ok(if self.little_endian {
            f64::from_le_bytes(bytes)
        } else {
            f64::from_be_bytes(bytes)
        })
    }
}

/// Read the `(x, y)` of a WKB Point without parsing the geometry generally.
///
/// Convenience wrapper over [`WkbPointLayout`] for the single-Point case: it
/// deliberately handles only Points — any other geometry is an error.
///
/// Returns `Ok(None)` for `POINT EMPTY` (encoded as `NaN`/`NaN`), `Ok(Some((x, y)))`
/// for a finite point, and an error if the bytes are not a Point or are truncated.
/// Accepts both ISO (with Z/M dimension flags) and EWKB (with Z/M/SRID flags)
/// encodings; only the first two ordinates are read.
pub fn read_point_xy(buf: &[u8]) -> Result<Option<(f64, f64)>, SedonaGeometryError> {
    match WkbPointLayout::try_from_wkb(buf)? {
        Some(layout) => layout.read_xy(buf),
        None => Err(SedonaGeometryError::Invalid(
            "WKB: expected a Point".to_string(),
        )),
    }
}

// A helper struct for calculating the WKBHeader
struct WkbBuffer<'a> {
    buf: &'a [u8],
    offset: usize,
    remaining: usize,
    last_endian: u8,
}

impl<'a> WkbBuffer<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            offset: 0,
            remaining: buf.len(),
            last_endian: 0,
        }
    }

    // For MULITPOINT, MULTILINESTRING, MULTIPOLYGON, or GEOMETRYCOLLECTION, returns the index to the first nested
    // non-collection geometry (POINT, LINESTRING, or POLYGON), or None if empty
    // For POINT, LINESTRING, POLYGON, returns 0 as it already is a non-collection geometry
    fn first_geom_idx(&mut self) -> Result<Option<usize>, SedonaGeometryError> {
        // Record the start of this geometry header so we can return an absolute index
        let start_offset = self.offset;

        self.read_endian()?;
        let geometry_type = self.read_u32()?;
        let geometry_type_id = GeometryTypeId::try_from_wkb_id(geometry_type & 0x7)?;

        match geometry_type_id {
            GeometryTypeId::Point | GeometryTypeId::LineString | GeometryTypeId::Polygon => {
                // Return absolute offset to the start of this geometry header
                Ok(Some(start_offset))
            }
            GeometryTypeId::MultiPoint
            | GeometryTypeId::MultiLineString
            | GeometryTypeId::MultiPolygon
            | GeometryTypeId::GeometryCollection => {
                if geometry_type & SRID_FLAG_BIT != 0 {
                    // Skip the SRID
                    self.read_u32()?;
                }

                let num_geometries = self.read_u32()?;

                if num_geometries == 0 {
                    return Ok(None);
                }

                // Recursive call to get first non-collection geometry
                self.first_geom_idx()
            }
            _ => Err(SedonaGeometryError::Invalid(format!(
                "Unexpected geometry type: {geometry_type_id:?}"
            ))),
        }
    }

    // Given a point, linestring, or polygon, return the first xy coordinate
    // If the geometry, is empty, (NaN, NaN) is returned
    fn first_xy_coord(&mut self) -> Result<(f64, f64), SedonaGeometryError> {
        self.read_endian()?;
        let geometry_type = self.read_u32()?;

        let geometry_type_id = GeometryTypeId::try_from_wkb_id(geometry_type & 0x7)?;

        // Skip the SRID if it's present
        if geometry_type & SRID_FLAG_BIT != 0 {
            self.read_u32()?;
        }

        match geometry_type_id {
            GeometryTypeId::LineString => {
                let size = self.read_u32()?;
                if size == 0 {
                    return Ok((f64::NAN, f64::NAN));
                }
            }
            GeometryTypeId::Polygon => {
                let size = self.read_u32()?;
                if size == 0 {
                    return Ok((f64::NAN, f64::NAN));
                }
                let ring0_num_points = self.read_u32()?;
                if ring0_num_points == 0 {
                    return Ok((f64::NAN, f64::NAN));
                }
            }
            _ => {}
        }

        let x = self.read_coord()?;
        let y = self.read_coord()?;
        Ok((x, y))
    }

    fn read_endian(&mut self) -> Result<(), SedonaGeometryError> {
        if self.remaining < 1 {
            return Err(SedonaGeometryError::Invalid(format!(
                "Invalid WKB: buffer too small. At offset: {}. Need 1 byte.",
                self.offset
            )));
        }
        self.last_endian = self.buf[self.offset];
        self.remaining -= 1;
        self.offset += 1;
        Ok(())
    }

    fn read_u32(&mut self) -> Result<u32, SedonaGeometryError> {
        if self.remaining < 4 {
            return Err(SedonaGeometryError::Invalid(format!(
                "Invalid WKB: buffer too small. At offset: {}. Need 4 bytes.",
                self.offset
            )));
        }

        let off = self.offset;
        let num = match self.last_endian {
            0 => u32::from_be_bytes([
                self.buf[off],
                self.buf[off + 1],
                self.buf[off + 2],
                self.buf[off + 3],
            ]),
            1 => u32::from_le_bytes([
                self.buf[off],
                self.buf[off + 1],
                self.buf[off + 2],
                self.buf[off + 3],
            ]),
            other => {
                return Err(SedonaGeometryError::Invalid(format!(
                    "Unexpected byte order: {other:?}"
                )))
            }
        };
        self.remaining -= 4;
        self.offset += 4;
        Ok(num)
    }

    // Given a buffer starting at the coordinate itself, parse the x and y coordinates
    fn read_coord(&mut self) -> Result<f64, SedonaGeometryError> {
        if self.remaining < 8 {
            return Err(SedonaGeometryError::Invalid(format!(
                "Invalid WKB: buffer too small. At offset: {}. Need 8 bytes.",
                self.offset
            )));
        }

        let buf = &self.buf;
        let off = self.offset;
        let coord: f64 = match self.last_endian {
            0 => f64::from_be_bytes([
                buf[off],
                buf[off + 1],
                buf[off + 2],
                buf[off + 3],
                buf[off + 4],
                buf[off + 5],
                buf[off + 6],
                buf[off + 7],
            ]),
            1 => f64::from_le_bytes([
                buf[off],
                buf[off + 1],
                buf[off + 2],
                buf[off + 3],
                buf[off + 4],
                buf[off + 5],
                buf[off + 6],
                buf[off + 7],
            ]),
            other => {
                return Err(SedonaGeometryError::Invalid(format!(
                    "Unexpected byte order: {other:?}"
                )))
            }
        };
        self.remaining -= 8;
        self.offset += 8;

        Ok(coord)
    }

    fn set_offset(&mut self, offset: usize) {
        self.offset = offset;
        self.remaining = self.buf.len() - offset;
    }
}

fn calc_dimensions(code: u32) -> Result<Dimensions, SedonaGeometryError> {
    // Check for EWKB Z and M flags
    let hasz = (code & Z_FLAG_BIT) != 0;
    let hasm = (code & M_FLAG_BIT) != 0;

    match (hasz, hasm) {
        (false, false) => {}
        // If either flag is set, this must be EWKB (and not ISO WKB)
        (true, false) => return Ok(Dimensions::Xyz),
        (false, true) => return Ok(Dimensions::Xym),
        (true, true) => return Ok(Dimensions::Xyzm),
    }

    // if SRID flag is set, then it must be EWKB with no z or m
    if code & SRID_FLAG_BIT != 0 {
        return Ok(Dimensions::Xy);
    }

    // Interpret as ISO WKB
    match code / 1000 {
        0 => Ok(Dimensions::Xy),
        1 => Ok(Dimensions::Xyz),
        2 => Ok(Dimensions::Xym),
        3 => Ok(Dimensions::Xyzm),
        _ => Err(SedonaGeometryError::Invalid(format!(
            "Unexpected code: {code:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sedona_testing::fixtures::*;
    use std::str::FromStr;
    use wkb::writer::{write_geometry, WriteOptions};
    use wkt::Wkt;

    fn make_wkb(wkt_value: &'static str) -> Vec<u8> {
        let geom = Wkt::<f64>::from_str(wkt_value).unwrap();
        let mut buf: Vec<u8> = vec![];
        write_geometry(&mut buf, &geom, &WriteOptions::default()).unwrap();
        buf
    }

    #[test]
    fn geometry_type_id() {
        let wkb = make_wkb("POINT (1 2)");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.geometry_type_id().unwrap(), GeometryTypeId::Point);

        let wkb = make_wkb("LINESTRING (1 2, 3 4)");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(
            header.geometry_type_id().unwrap(),
            GeometryTypeId::LineString
        );

        let wkb = make_wkb("POLYGON ((0 0, 0 1, 1 0, 0 0))");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.geometry_type_id().unwrap(), GeometryTypeId::Polygon);

        let wkb = make_wkb("MULTIPOINT ((1 2), (3 4))");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(
            header.geometry_type_id().unwrap(),
            GeometryTypeId::MultiPoint
        );

        let wkb = make_wkb("MULTILINESTRING ((1 2, 3 4))");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(
            header.geometry_type_id().unwrap(),
            GeometryTypeId::MultiLineString
        );

        let wkb = make_wkb("MULTIPOLYGON (((0 0, 0 1, 1 0, 0 0)))");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(
            header.geometry_type_id().unwrap(),
            GeometryTypeId::MultiPolygon
        );

        let wkb = make_wkb("GEOMETRYCOLLECTION (POINT (1 2))");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(
            header.geometry_type_id().unwrap(),
            GeometryTypeId::GeometryCollection
        );

        // Some cases with z and m dimensions
        let wkb = make_wkb("POINT Z (1 2 3)");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.geometry_type_id().unwrap(), GeometryTypeId::Point);

        let wkb = make_wkb("LINESTRING Z (1 2 3, 4 5 6)");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(
            header.geometry_type_id().unwrap(),
            GeometryTypeId::LineString
        );

        let wkb = make_wkb("POLYGON M ((0 0 0, 0 1 0, 1 0 0, 0 0 0))");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.geometry_type_id().unwrap(), GeometryTypeId::Polygon);
    }

    #[test]
    fn size() {
        let wkb = make_wkb("LINESTRING (1 2, 3 4)");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.size(), 2);

        let wkb = make_wkb("POLYGON ((0 0, 0 1, 1 0, 0 0), (1 1, 1 2, 2 1, 1 1))");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.size(), 2);

        let wkb = make_wkb("MULTIPOINT ((1 2), (3 4))");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.size(), 2);

        let wkb = make_wkb("MULTILINESTRING ((1 2, 3 4, 5 6), (7 8, 9 10, 11 12))");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.size(), 2);

        let wkb = make_wkb("MULTIPOLYGON (((0 0, 0 1, 1 0, 0 0)), ((1 1, 1 2, 2 1, 1 1)))");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.size(), 2);

        let wkb = make_wkb("GEOMETRYCOLLECTION (POINT (1 2))");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.size(), 1);

        let wkb = make_wkb("GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (1 2, 3 4), POLYGON ((0 0, 0 1, 1 0, 0 0)))");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.size(), 3);
    }

    #[test]
    fn empty_size() {
        let wkb = make_wkb("LINESTRING EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.size(), 0);

        let wkb = make_wkb("POLYGON EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.size(), 0);

        let wkb = make_wkb("MULTIPOINT EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.size(), 0);

        let wkb = make_wkb("MULTILINESTRING EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.size(), 0);

        let wkb = make_wkb("MULTIPOLYGON EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.size(), 0);

        let wkb = make_wkb("GEOMETRYCOLLECTION EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.size(), 0);

        let wkb = make_wkb("GEOMETRYCOLLECTION Z EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.size(), 0);
    }

    #[test]
    fn ewkb() {
        // Test POINT with SRID 4326
        let header = WkbHeader::try_new(&POINT_WITH_SRID_4326_EWKB).unwrap();
        assert_eq!(header.srid(), 4326);
        assert_eq!(header.geometry_type_id().unwrap(), GeometryTypeId::Point);
        assert_eq!(header.first_xy(), (1.0, 2.0));
        assert_eq!(header.dimensions().unwrap(), Dimensions::Xy);

        // Test POINT Z with SRID 3857
        let header = WkbHeader::try_new(&POINT_Z_WITH_SRID_3857_EWKB).unwrap();
        assert_eq!(header.srid(), 3857);
        assert_eq!(header.geometry_type_id().unwrap(), GeometryTypeId::Point);
        assert_eq!(header.first_xy(), (1.0, 2.0));
        assert_eq!(header.dimensions().unwrap(), Dimensions::Xyz);

        // Test POINT M with SRID 4326
        let header = WkbHeader::try_new(&POINT_M_WITH_SRID_4326_EWKB).unwrap();
        assert_eq!(header.srid(), 4326);
        assert_eq!(header.geometry_type_id().unwrap(), GeometryTypeId::Point);
        assert_eq!(header.first_xy(), (1.0, 2.0));
        assert_eq!(header.dimensions().unwrap(), Dimensions::Xym);

        // Test POINT ZM with SRID 4326
        let header = WkbHeader::try_new(&POINT_ZM_WITH_SRID_4326_EWKB).unwrap();
        assert_eq!(header.srid(), 4326);
        assert_eq!(header.geometry_type_id().unwrap(), GeometryTypeId::Point);
        assert_eq!(header.first_xy(), (1.0, 2.0));
        assert_eq!(header.dimensions().unwrap(), Dimensions::Xyzm);

        // Test GEOMETRYCOLLECTION with SRID 4326
        let header = WkbHeader::try_new(&GEOMETRYCOLLECTION_POINT_WITH_SRID_4326_EWKB).unwrap();
        assert_eq!(header.srid(), 4326);
        assert_eq!(
            header.geometry_type_id().unwrap(),
            GeometryTypeId::GeometryCollection
        );
        assert_eq!(header.size(), 1);
        assert_eq!(header.first_xy(), (1.0, 2.0));
        assert_eq!(header.dimensions().unwrap(), Dimensions::Xy);
        assert_eq!(header.first_geom_dimensions().unwrap(), Dimensions::Xy);

        // Test GEOMETRYCOLLECTION Z with SRID 4326
        let header = WkbHeader::try_new(&GEOMETRYCOLLECTION_POINT_Z_WITH_SRID_4326_EWKB).unwrap();
        assert_eq!(header.srid(), 4326);
        assert_eq!(
            header.geometry_type_id().unwrap(),
            GeometryTypeId::GeometryCollection
        );
        assert_eq!(header.size(), 1);
        assert_eq!(header.first_xy(), (1.0, 2.0));
        // Outer dimension specified as Xy, but inner dimension is Xyz
        assert_eq!(header.dimensions().unwrap(), Dimensions::Xy);
        assert_eq!(header.first_geom_dimensions().unwrap(), Dimensions::Xyz);

        // Test GEOMETRYCOLLECTION M with SRID 4326
        let header = WkbHeader::try_new(&GEOMETRYCOLLECTION_POINT_M_WITH_SRID_4326_EWKB).unwrap();
        assert_eq!(header.srid(), 4326);
        assert_eq!(
            header.geometry_type_id().unwrap(),
            GeometryTypeId::GeometryCollection
        );
        assert_eq!(header.size(), 1);
        assert_eq!(header.first_xy(), (1.0, 2.0));
        // Outer dimension specified as Xy, but inner dimension is Xym
        assert_eq!(header.dimensions().unwrap(), Dimensions::Xy);
        assert_eq!(header.first_geom_dimensions().unwrap(), Dimensions::Xym);

        // Test GEOMETRYCOLLECTION ZM with SRID 4326
        let header = WkbHeader::try_new(&GEOMETRYCOLLECTION_POINT_ZM_WITH_SRID_4326_EWKB).unwrap();
        assert_eq!(header.srid(), 4326);
        assert_eq!(
            header.geometry_type_id().unwrap(),
            GeometryTypeId::GeometryCollection
        );
        assert_eq!(header.size(), 1);
        assert_eq!(header.first_xy(), (1.0, 2.0));
        // Outer dimension specified as Xy, but inner dimension is Xyzm
        assert_eq!(header.dimensions().unwrap(), Dimensions::Xy);
        assert_eq!(header.first_geom_dimensions().unwrap(), Dimensions::Xyzm);
    }

    #[test]
    fn srid_linestring() {
        let header = WkbHeader::try_new(&LINESTRING_WITH_SRID_4326_EWKB).unwrap();
        assert_eq!(header.srid(), 4326);
        assert_eq!(
            header.geometry_type_id().unwrap(),
            GeometryTypeId::LineString
        );
        assert_eq!(header.size(), 2);
        assert_eq!(header.first_xy(), (1.0, 2.0));
        assert_eq!(header.dimensions().unwrap(), Dimensions::Xy);
    }

    #[test]
    fn srid_polygon() {
        let header = WkbHeader::try_new(&POLYGON_WITH_SRID_4326_EWKB).unwrap();
        assert_eq!(header.srid(), 4326);
        assert_eq!(header.geometry_type_id().unwrap(), GeometryTypeId::Polygon);
        assert_eq!(header.size(), 1);
        assert_eq!(header.first_xy(), (0.0, 0.0));
        assert_eq!(header.dimensions().unwrap(), Dimensions::Xy);
    }

    #[test]
    fn multipoint_with_srid() {
        let header = WkbHeader::try_new(&MULTIPOINT_WITH_SRID_4326_EWKB).unwrap();
        assert_eq!(header.srid(), 4326);
        assert_eq!(
            header.geometry_type_id().unwrap(),
            GeometryTypeId::MultiPoint
        );
        assert_eq!(header.size(), 2);
        assert_eq!(header.first_xy(), (1.0, 2.0));
        assert_eq!(header.dimensions().unwrap(), Dimensions::Xy);
    }

    #[test]
    fn srid_empty_geometries_with_srid() {
        // Test POINT EMPTY with SRID
        let header = WkbHeader::try_new(&POINT_EMPTY_WITH_SRID_4326_EWKB).unwrap();
        assert_eq!(header.srid(), 4326);
        assert_eq!(header.geometry_type_id().unwrap(), GeometryTypeId::Point);
        assert_eq!(header.dimensions().unwrap(), Dimensions::Xy);

        // Test GEOMETRYCOLLECTION EMPTY with SRID
        let header = WkbHeader::try_new(&GEOMETRYCOLLECTION_EMPTY_WITH_SRID_4326_EWKB).unwrap();
        assert_eq!(header.srid(), 4326);
        assert_eq!(
            header.geometry_type_id().unwrap(),
            GeometryTypeId::GeometryCollection
        );
        assert_eq!(header.size(), 0);
        assert_eq!(header.dimensions().unwrap(), Dimensions::Xy);
        assert_eq!(header.first_geom_dimensions(), None);
    }

    #[test]
    fn srid_no_srid_flag() {
        // Test that regular WKB (without SRID flag) returns 0 for SRID
        let wkb = make_wkb("POINT (1 2)");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.srid(), 0);
    }

    #[test]
    fn first_xy() {
        let wkb = make_wkb("POINT (-5 -2)");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.first_xy(), (-5.0, -2.0));

        let wkb = make_wkb("LINESTRING (1 2, 3 4)");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.first_xy(), (1.0, 2.0));

        let wkb = make_wkb("POLYGON ((0 0, 0 1, 1 0, 0 0))");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.first_xy(), (0.0, 0.0));

        // Another polygon test since that logic is more complicated
        let wkb = make_wkb("POLYGON ((1.5 0.5, 1.5 1.5, 1.5 0.5), (0 0, 0 1, 1 0, 0 0))");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.first_xy(), (1.5, 0.5));

        let wkb = make_wkb("MULTIPOINT ((1 2), (3 4))");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.first_xy(), (1.0, 2.0));

        let wkb = make_wkb("MULTILINESTRING ((3 4, 1 2), (5 6, 7 8))");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.first_xy(), (3.0, 4.0));

        let wkb = make_wkb("MULTIPOLYGON (((-1 -1, 0 1, 1 -1, -1 -1)), ((0 0, 0 1, 1 0, 0 0)))");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.first_xy(), (-1.0, -1.0));

        let wkb = make_wkb("GEOMETRYCOLLECTION (POINT (1 2))");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.first_xy(), (1.0, 2.0));

        let wkb = make_wkb("GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (1 2, 3 4), POLYGON ((0 0, 0 1, 1 0, 0 0)))");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.first_xy(), (1.0, 2.0));
    }

    #[test]
    fn empty_first_xy() {
        let wkb = make_wkb("POINT EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        let (x, y) = header.first_xy();
        assert!(x.is_nan());
        assert!(y.is_nan());

        let wkb = make_wkb("LINESTRING EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        let (x, y) = header.first_xy();
        assert!(x.is_nan());
        assert!(y.is_nan());

        let wkb = make_wkb("GEOMETRYCOLLECTION Z EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        let (x, y) = header.first_xy();
        assert!(x.is_nan());
        assert!(y.is_nan());
    }

    #[test]
    fn empty_geometry_type_id() {
        let wkb = make_wkb("POINT EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.geometry_type_id().unwrap(), GeometryTypeId::Point);

        let wkb = make_wkb("LINESTRING EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(
            header.geometry_type_id().unwrap(),
            GeometryTypeId::LineString
        );

        let wkb = make_wkb("POLYGON EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.geometry_type_id().unwrap(), GeometryTypeId::Polygon);

        let wkb = make_wkb("MULTIPOINT EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(
            header.geometry_type_id().unwrap(),
            GeometryTypeId::MultiPoint
        );

        let wkb = make_wkb("MULTILINESTRING EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(
            header.geometry_type_id().unwrap(),
            GeometryTypeId::MultiLineString
        );

        let wkb = make_wkb("MULTIPOLYGON EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(
            header.geometry_type_id().unwrap(),
            GeometryTypeId::MultiPolygon
        );

        let wkb = make_wkb("GEOMETRYCOLLECTION EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(
            header.geometry_type_id().unwrap(),
            GeometryTypeId::GeometryCollection
        );

        // z, m cases
        let wkb = make_wkb("POINT Z EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.geometry_type_id().unwrap(), GeometryTypeId::Point);

        let wkb = make_wkb("POINT M EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.geometry_type_id().unwrap(), GeometryTypeId::Point);

        let wkb = make_wkb("LINESTRING ZM EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(
            header.geometry_type_id().unwrap(),
            GeometryTypeId::LineString
        );
    }

    #[test]
    fn dimensions() {
        let wkb = make_wkb("POINT (1 2)");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.dimensions().unwrap(), Dimensions::Xy);

        let wkb = make_wkb("POINT Z (1 2 3)");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.dimensions().unwrap(), Dimensions::Xyz);

        let wkb = make_wkb("POINT M (1 2 3)");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.dimensions().unwrap(), Dimensions::Xym);

        let wkb = make_wkb("POINT ZM (1 2 3 4)");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.dimensions().unwrap(), Dimensions::Xyzm);
    }

    #[test]
    fn empty_geometry_dimensions() {
        // POINTs
        let wkb = make_wkb("POINT EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.dimensions().unwrap(), Dimensions::Xy);

        let wkb = make_wkb("POINT Z EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.dimensions().unwrap(), Dimensions::Xyz);

        let wkb = make_wkb("POINT M EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.dimensions().unwrap(), Dimensions::Xym);

        let wkb = make_wkb("POINT ZM EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.dimensions().unwrap(), Dimensions::Xyzm);

        // GEOMETRYCOLLECTIONs
        let wkb = make_wkb("GEOMETRYCOLLECTION EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.dimensions().unwrap(), Dimensions::Xy);

        let wkb = make_wkb("GEOMETRYCOLLECTION Z EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.dimensions().unwrap(), Dimensions::Xyz);

        let wkb = make_wkb("GEOMETRYCOLLECTION M EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.dimensions().unwrap(), Dimensions::Xym);

        let wkb = make_wkb("GEOMETRYCOLLECTION ZM EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.dimensions().unwrap(), Dimensions::Xyzm);
    }

    #[test]
    fn first_geom_dimensions() {
        // Top-level dimension is xy, while nested geometry is xyz
        let wkb = make_wkb("GEOMETRYCOLLECTION (POINT Z (1 2 3))");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.first_geom_dimensions().unwrap(), Dimensions::Xyz);
        assert_eq!(header.dimensions().unwrap(), Dimensions::Xy);

        let wkb = make_wkb("GEOMETRYCOLLECTION (POINT ZM (1 2 3 4))");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.first_geom_dimensions().unwrap(), Dimensions::Xyzm);

        let wkb = make_wkb("GEOMETRYCOLLECTION (POINT M (1 2 3))");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.first_geom_dimensions().unwrap(), Dimensions::Xym);

        let wkb = make_wkb("GEOMETRYCOLLECTION (POINT ZM (1 2 3 4))");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.first_geom_dimensions().unwrap(), Dimensions::Xyzm);
    }

    #[test]
    fn empty_geometry_first_geom_dimensions() {
        let wkb = make_wkb("POINT EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.first_geom_dimensions(), Some(Dimensions::Xy));

        let wkb = make_wkb("LINESTRING EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.first_geom_dimensions(), Some(Dimensions::Xy));

        let wkb = make_wkb("POLYGON Z EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.first_geom_dimensions(), Some(Dimensions::Xyz));

        // Empty collections should return None
        let wkb = make_wkb("MULTIPOINT EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.first_geom_dimensions(), None);

        let wkb = make_wkb("MULTILINESTRING Z EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.first_geom_dimensions(), None);

        let wkb = make_wkb("MULTIPOLYGON M EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.first_geom_dimensions(), None);

        let wkb = make_wkb("GEOMETRYCOLLECTION ZM EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(header.first_geom_dimensions(), None);
    }

    #[test]
    fn incomplete_buffers() {
        // Test various incomplete buffer scenarios to ensure proper error handling

        // Empty buffer
        let result = WkbHeader::try_new(&[]);
        assert!(result.is_err());

        // Test truncation of a simple POINT
        let wkb = make_wkb("POINT (1 2)");
        for i in 1..wkb.len() - 1 {
            assert!(
                WkbHeader::try_new(&wkb[0..i]).is_err(),
                "0..{i} unexpectedly succeeded"
            );
        }

        // Test truncation of a POINT ZM
        // Iterate through all i that is less than the number needed for the first_xy coord
        // 1 byte_order + 4 geometry type + 8 x + 8 y
        let last_i = 1 + 4 + 8 + 8;
        let wkb = make_wkb("POINT ZM (1 2 3 4)");
        for i in 1..last_i {
            assert!(
                WkbHeader::try_new(&wkb[0..i]).is_err(),
                "0..{i} unexpectedly succeeded"
            );
        }

        // Test truncation of a GEOMETRYCOLLECTION with nested geometries
        // Iterate through all i that is less than the number needed for the first_xy coord
        // 1 byte_order + 4 geometry type + 4 size + 8 x + 8 y
        let last_i = 1 + 4 + 4 + 8 + 8;
        let wkb = make_wkb("GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (1 2, 3 4))");
        for i in 1..last_i {
            assert!(
                WkbHeader::try_new(&wkb[0..i]).is_err(),
                "0..{i} unexpectedly succeeded"
            );
        }
    }

    #[test]
    fn incomplete_ewkb_buffers() {
        // Test incomplete EWKB buffers

        // 1 byte_order + 4 geometry type + 4 srid + 8 x + 8 y
        let wkb = POINT_WITH_SRID_4326_EWKB;
        let last_i = 1 + 4 + 4 + 8 + 8;
        for i in 1..last_i {
            assert!(
                WkbHeader::try_new(&wkb[0..i]).is_err(),
                "0..{i} unexpectedly succeeded"
            );
        }

        // 1 byte_order + 4 geometry type + 4 srid + 4 size + 1 byte_order + 4 geometry type + 8 x + 8 y
        let last_i = 1 + 4 + 4 + 4 + 1 + 4 + 8 + 8;
        let wkb = MULTIPOINT_WITH_SRID_4326_EWKB;
        for i in 1..last_i {
            assert!(
                WkbHeader::try_new(&wkb[0..i]).is_err(),
                "0..{i} unexpectedly succeeded"
            );
        }

        // 1 byte_order + 4 geometry type + 4 srid + 4 size + 1 byte_order + 4 geometry type + 8 x + 8 y
        let last_i = 1 + 4 + 4 + 4 + 1 + 4 + 8 + 8;
        let wkb = GEOMETRYCOLLECTION_POINT_ZM_WITH_SRID_4326_EWKB;
        for i in 1..last_i {
            assert!(
                WkbHeader::try_new(&wkb[0..i]).is_err(),
                "0..{i} unexpectedly succeeded"
            );
        }
    }

    #[test]
    fn invalid_byte_order() {
        // Test invalid byte order values
        let result = WkbHeader::try_new(&[0x02, 0x01, 0x00, 0x00, 0x00]);
        assert!(result.is_err());

        let result = WkbHeader::try_new(&[0xff, 0x01, 0x00, 0x00, 0x00]);
        assert!(result.is_err());
    }

    #[test]
    fn nested_geometry_collections() {
        let wkb = make_wkb("GEOMETRYCOLLECTION (GEOMETRYCOLLECTION (POINT (1 2)))");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(
            header.geometry_type_id().unwrap(),
            GeometryTypeId::GeometryCollection
        );
        assert_eq!(header.size(), 1);
        assert_eq!(header.first_xy(), (1.0, 2.0));
        assert_eq!(header.dimensions().unwrap(), Dimensions::Xy);
        assert_eq!(header.first_geom_dimensions().unwrap(), Dimensions::Xy);

        let wkb = make_wkb("GEOMETRYCOLLECTION (GEOMETRYCOLLECTION (POINT ZM (1 2 3 4)))");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert_eq!(
            header.geometry_type_id().unwrap(),
            GeometryTypeId::GeometryCollection
        );
        assert_eq!(header.size(), 1);
        assert_eq!(header.first_xy(), (1.0, 2.0));
        assert_eq!(header.dimensions().unwrap(), Dimensions::Xy);
        assert_eq!(header.first_geom_dimensions().unwrap(), Dimensions::Xyzm);
    }

    #[test]
    fn is_empty() {
        let wkb = make_wkb("POINT EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert!(header.is_empty().unwrap());

        let wkb = make_wkb("POINT Z EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert!(header.is_empty().unwrap());

        let wkb = make_wkb("LINESTRING EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert!(header.is_empty().unwrap());

        let wkb = make_wkb("POLYGON EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert!(header.is_empty().unwrap());

        let wkb = make_wkb("MULTIPOINT EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert!(header.is_empty().unwrap());

        let wkb = make_wkb("MULTILINESTRING EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert!(header.is_empty().unwrap());

        let wkb = make_wkb("MULTIPOLYGON EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert!(header.is_empty().unwrap());

        let wkb = make_wkb("GEOMETRYCOLLECTION EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert!(header.is_empty().unwrap());

        let wkb = make_wkb("GEOMETRYCOLLECTION Z EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert!(header.is_empty().unwrap());

        let wkb = make_wkb("GEOMETRYCOLLECTION M EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert!(header.is_empty().unwrap());

        let wkb = make_wkb("GEOMETRYCOLLECTION ZM EMPTY");
        let header = WkbHeader::try_new(&wkb).unwrap();
        assert!(header.is_empty().unwrap());
    }

    #[test]
    fn wkb_point_layout_accepts_point_encodings() {
        // ISO points of every dimensionality: the layout is found and the
        // coordinate read matches the leading (x, y).
        for wkt_value in [
            "POINT (1 2)",
            "POINT Z (1 2 3)",
            "POINT M (1 2 3)",
            "POINT ZM (1 2 3 4)",
        ] {
            let wkb = make_wkb(wkt_value);
            let layout = WkbPointLayout::try_from_wkb(&wkb)
                .unwrap()
                .unwrap_or_else(|| panic!("{wkt_value} not recognized as a Point"));
            assert_eq!(
                layout.read_xy(&wkb).unwrap(),
                Some((1.0, 2.0)),
                "{wkt_value}"
            );
        }

        // EWKB points: the SRID between the type code and the coordinate must be
        // skipped for every dimensionality.
        for wkb in [
            &POINT_WITH_SRID_4326_EWKB[..],
            &POINT_Z_WITH_SRID_3857_EWKB[..],
            &POINT_M_WITH_SRID_4326_EWKB[..],
            &POINT_ZM_WITH_SRID_4326_EWKB[..],
        ] {
            let layout = WkbPointLayout::try_from_wkb(wkb).unwrap().unwrap();
            assert_eq!(layout.read_xy(wkb).unwrap(), Some((1.0, 2.0)));
        }

        // Big-endian EWKB: SRID flag and coordinate both decoded big-endian.
        let be_ewkb: [u8; 25] = [
            0x00, // big-endian
            0x20, 0x00, 0x00, 0x01, // type code 1 (Point) with SRID flag
            0x00, 0x00, 0x10, 0xE6, // SRID 4326
            0x3F, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // x = 1.0
            0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // y = 2.0
        ];
        let layout = WkbPointLayout::try_from_wkb(&be_ewkb).unwrap().unwrap();
        assert_eq!(layout.read_xy(&be_ewkb).unwrap(), Some((1.0, 2.0)));
    }

    #[test]
    fn wkb_point_layout_non_point_is_none() {
        for wkt_value in [
            "LINESTRING (1 2, 3 4)",
            "POLYGON ((0 0, 0 1, 1 0, 0 0))",
            "MULTIPOINT ((1 2))",
            "GEOMETRYCOLLECTION (POINT (1 2))",
        ] {
            let wkb = make_wkb(wkt_value);
            assert!(
                WkbPointLayout::try_from_wkb(&wkb).unwrap().is_none(),
                "{wkt_value} unexpectedly recognized as a Point"
            );
        }

        // Non-point EWKB: the SRID flag must not confuse type detection.
        assert!(
            WkbPointLayout::try_from_wkb(&LINESTRING_WITH_SRID_4326_EWKB)
                .unwrap()
                .is_none()
        );
        assert!(
            WkbPointLayout::try_from_wkb(&MULTIPOINT_WITH_SRID_4326_EWKB)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn wkb_point_layout_incomplete_iso_buffer() {
        let wkb = make_wkb("POINT (1 2)");

        // No byte order at all, and an invalid byte-order flag.
        assert!(WkbPointLayout::try_from_wkb(&[]).is_err());
        assert!(WkbPointLayout::try_from_wkb(&[0x02, 0, 0, 0, 1]).is_err());

        // Shorter than byte order + type code: the header itself is truncated.
        for i in 1..5 {
            assert!(
                WkbPointLayout::try_from_wkb(&wkb[0..i]).is_err(),
                "header 0..{i} unexpectedly succeeded"
            );
        }

        // Header is readable but the coordinate is truncated: read_xy must
        // error rather than read out of bounds.
        for i in 5..wkb.len() {
            let layout = WkbPointLayout::try_from_wkb(&wkb[0..i]).unwrap().unwrap();
            assert!(
                layout.read_xy(&wkb[0..i]).is_err(),
                "coordinate 0..{i} unexpectedly succeeded"
            );
        }
    }

    #[test]
    fn wkb_point_layout_incomplete_ewkb_buffer() {
        // 1 byte order + 4 type code + 4 SRID + 8 x + 8 y
        let wkb = POINT_WITH_SRID_4326_EWKB;
        assert_eq!(wkb.len(), 25);

        for i in 1..5 {
            assert!(
                WkbPointLayout::try_from_wkb(&wkb[0..i]).is_err(),
                "header 0..{i} unexpectedly succeeded"
            );
        }

        // Everything shorter than the full SRID + coordinate must error: the
        // coordinate offset accounts for the SRID, so a buffer that would fit an
        // ISO point is still too short here.
        for i in 5..wkb.len() {
            let layout = WkbPointLayout::try_from_wkb(&wkb[0..i]).unwrap().unwrap();
            assert!(
                layout.read_xy(&wkb[0..i]).is_err(),
                "coordinate 0..{i} unexpectedly succeeded"
            );
        }
        let layout = WkbPointLayout::try_from_wkb(&wkb).unwrap().unwrap();
        assert_eq!(layout.read_xy(&wkb).unwrap(), Some((1.0, 2.0)));
    }

    #[test]
    fn read_point_xy_iso_little_endian() {
        // 2-D / 3-D / 4-D ISO points all expose the same leading (x, y), and the
        // result must agree with the general parser's `first_xy`.
        for wkt_value in [
            "POINT (1 2)",
            "POINT Z (1 2 3)",
            "POINT M (1 2 3)",
            "POINT ZM (1 2 3 4)",
        ] {
            let wkb = make_wkb(wkt_value);
            assert_eq!(
                read_point_xy(&wkb).unwrap(),
                Some((1.0, 2.0)),
                "{wkt_value}"
            );
            assert_eq!(
                read_point_xy(&wkb).unwrap(),
                Some(WkbHeader::try_new(&wkb).unwrap().first_xy()),
                "disagrees with first_xy for {wkt_value}"
            );
        }
    }

    #[test]
    fn read_point_xy_big_endian() {
        // Byte order 0x00, type 1 (BE), x = 1.0, y = 2.0 — all big-endian.
        let wkb: [u8; 21] = [
            0x00, // big-endian
            0x00, 0x00, 0x00, 0x01, // type code 1 (Point)
            0x3F, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // x = 1.0
            0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // y = 2.0
        ];
        assert_eq!(read_point_xy(&wkb).unwrap(), Some((1.0, 2.0)));
    }

    #[test]
    fn read_point_xy_ewkb_with_srid() {
        // The SRID sits between the type code and the coordinate; the reader must
        // skip it for every dimensionality.
        for wkb in [
            &POINT_WITH_SRID_4326_EWKB[..],
            &POINT_Z_WITH_SRID_3857_EWKB[..],
            &POINT_M_WITH_SRID_4326_EWKB[..],
            &POINT_ZM_WITH_SRID_4326_EWKB[..],
        ] {
            assert_eq!(read_point_xy(wkb).unwrap(), Some((1.0, 2.0)));
            assert_eq!(
                read_point_xy(wkb).unwrap(),
                Some(WkbHeader::try_new(wkb).unwrap().first_xy())
            );
        }
    }

    #[test]
    fn read_point_xy_empty_is_none() {
        // POINT EMPTY is encoded as NaN/NaN and reads back as "no coordinate".
        assert_eq!(read_point_xy(&make_wkb("POINT EMPTY")).unwrap(), None);
        assert_eq!(read_point_xy(&make_wkb("POINT Z EMPTY")).unwrap(), None);
        assert_eq!(
            read_point_xy(&POINT_EMPTY_WITH_SRID_4326_EWKB).unwrap(),
            None
        );
    }

    #[test]
    fn read_point_xy_rejects_non_point() {
        for wkt_value in [
            "LINESTRING (1 2, 3 4)",
            "POLYGON ((0 0, 0 1, 1 0, 0 0))",
            "MULTIPOINT ((1 2))",
            "GEOMETRYCOLLECTION (POINT (1 2))",
        ] {
            let err = read_point_xy(&make_wkb(wkt_value)).unwrap_err().to_string();
            assert!(err.contains("expected a Point"), "{wkt_value}: {err}");
        }
    }

    #[test]
    fn read_point_xy_rejects_truncated() {
        let wkb = make_wkb("POINT (1 2)");
        // Empty input has no byte order; everything shorter than a full 2-D point
        // header + coordinate must error rather than read past the end.
        assert!(read_point_xy(&[]).is_err());
        for i in 1..wkb.len() {
            assert!(
                read_point_xy(&wkb[0..i]).is_err(),
                "0..{i} unexpectedly succeeded"
            );
        }
        // Invalid byte-order flag.
        assert!(read_point_xy(&[0x02, 0, 0, 0, 1]).is_err());
    }
}
