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

use crate::types::GeometryTypeId;
use datafusion_common::error::{DataFusionError, Result};
use geo_traits::Dimensions;
use sedona_common::sedona_internal_err;

/// Fast-path WKB header parser
/// Performs operations lazily and caches them after the first computation
pub struct WkbHeader<'a> {
    buf: &'a [u8],
    geometry_type_id: Option<GeometryTypeId>,
    dimensions: Option<Dimensions>,
}

impl<'a> WkbHeader<'a> {
    /// Creates a new [WkbHeader] from a buffer
    pub fn new(buf: &'a [u8]) -> Result<Self> {
        Ok(Self {
            buf,
            geometry_type_id: None,
            dimensions: None,
        })
    }

    /// Returns the geometry type id of the WKB by only parsing the header instead of the entire WKB
    /// 1 -> Point
    /// 2 -> LineString
    /// 3 -> Polygon
    /// 4 -> MultiPoint
    /// 5 -> MultiLineString
    /// 6 -> MultiPolygon
    /// 7 -> GeometryCollection
    ///
    /// Spec: https://libgeos.org/specifications/wkb/
    pub fn geometry_type_id(mut self) -> Result<GeometryTypeId> {
        if self.geometry_type_id.is_none() {
            self.geometry_type_id = Some(parse_geometry_type_id(self.buf)?);
        }
        self.geometry_type_id.ok_or_else(|| {
            DataFusionError::External("Unexpected internal state in WkbHeader".into())
        })
    }

    /// Returns the dimension of the WKB by only parsing what's minimally necessary instead of the entire WKB
    pub fn dimension(mut self) -> Result<Dimensions> {
        // Calculate the dimension if we haven't already
        if self.dimensions.is_none() {
            self.dimensions = Some(parse_dimension(self.buf)?);
        }
        self.dimensions.ok_or_else(|| {
            DataFusionError::External("Unexpected internal state in WkbHeader".into())
        })
    }
}

fn parse_dimension(buf: &[u8]) -> Result<Dimensions> {
    if buf.len() < 5 {
        return sedona_internal_err!("Invalid WKB: buffer too small ({} bytes)", buf.len());
    }

    let byte_order = buf[0];

    let code = match byte_order {
        0 => u32::from_be_bytes([buf[1], buf[2], buf[3], buf[4]]),
        1 => u32::from_le_bytes([buf[1], buf[2], buf[3], buf[4]]),
        other => return sedona_internal_err!("Unexpected byte order: {other}"),
    };

    // 0000 -> xy or unspecified
    // 1000 -> xyz
    // 2000 -> xym
    // 3000 -> xyzm
    match code / 1000 {
        // If xy, it's possible we need to infer the dimension
        0 => {}
        1 => return Ok(Dimensions::Xyz),
        2 => return Ok(Dimensions::Xym),
        3 => return Ok(Dimensions::Xyzm),
        _ => return sedona_internal_err!("Unexpected code: {code}"),
    };

    // Try to infer dimension
    // If geometry is a collection (MULTIPOINT, ... GEOMETRYCOLLECTION, code 4-7), we need to check the dimension of the first geometry
    if code & 0x7 >= 4 {
        // The next 4 bytes are the number of geometries in the collection
        let num_geometries = match byte_order {
            0 => u32::from_be_bytes([buf[5], buf[6], buf[7], buf[8]]),
            1 => u32::from_le_bytes([buf[5], buf[6], buf[7], buf[8]]),
            other => return sedona_internal_err!("Unexpected byte order: {other}"),
        };
        // Check the dimension of the first geometry since they all have to be the same dimension
        // Note: Attempting to create the following geometries error and are thus not possible to create:
        // - Nested geometry dimension doesn't match the **specified** geom collection z-dimension
        //   - GEOMETRYCOLLECTION M (POINT Z (1 1 1))
        // - Nested geometry doesn't have the specified dimension
        //   - GEOMETRYCOLLECTION Z (POINT (1 1))
        // - Nested geometries have different dimensions
        //   - GEOMETRYCOLLECTION (POINT Z (1 1 1), POINT (1 1))
        if num_geometries >= 1 {
            return parse_dimension(&buf[9..]);
        }
        // If empty geometry (num_geometries == 0), fallback to below logic to check the geom collection's dimension
        // GEOMETRY COLLECTION Z EMPTY hasz -> true
    }

    Ok(Dimensions::Xy)
}

fn parse_geometry_type_id(buf: &[u8]) -> Result<GeometryTypeId> {
    if buf.len() < 5 {
        return sedona_internal_err!("Invalid WKB: buffer too small");
    };

    let byte_order = buf[0];

    // Parse geometry type
    let code = match byte_order {
        0 => u32::from_be_bytes([buf[1], buf[2], buf[3], buf[4]]),
        1 => u32::from_le_bytes([buf[1], buf[2], buf[3], buf[4]]),
        other => return sedona_internal_err!("Unexpected byte order: {other}"),
    };

    // Only low 3 bits is for the base type, high bits include additional info
    let code = code & 0x7;

    let geometry_type_id = GeometryTypeId::try_from_wkb_id(code)
        .map_err(|e| DataFusionError::External(Box::new(e)))?;

    Ok(geometry_type_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;
    use wkt::Wkt;

    fn make_wkb(wkt_value: &'static str) -> Vec<u8> {
        let geom = Wkt::<f64>::from_str(wkt_value).unwrap();
        let mut buf: Vec<u8> = vec![];
        wkb::writer::write_geometry(&mut buf, &geom, Default::default()).unwrap();
        buf
    }

    #[test]
    fn geometry_type_id() {
        let wkb = make_wkb("POINT (1 2)");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(header.geometry_type_id().unwrap(), GeometryTypeId::Point);

        let wkb = make_wkb("LINESTRING (1 2, 3 4)");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(
            header.geometry_type_id().unwrap(),
            GeometryTypeId::LineString
        );

        let wkb = make_wkb("POLYGON ((0 0, 0 1, 1 0, 0 0))");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(header.geometry_type_id().unwrap(), GeometryTypeId::Polygon);

        let wkb = make_wkb("MULTIPOINT ((1 2), (3 4))");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(
            header.geometry_type_id().unwrap(),
            GeometryTypeId::MultiPoint
        );

        let wkb = make_wkb("MULTILINESTRING ((1 2, 3 4))");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(
            header.geometry_type_id().unwrap(),
            GeometryTypeId::MultiLineString
        );

        let wkb = make_wkb("MULTIPOLYGON (((0 0, 0 1, 1 0, 0 0)))");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(
            header.geometry_type_id().unwrap(),
            GeometryTypeId::MultiPolygon
        );

        let wkb = make_wkb("GEOMETRYCOLLECTION (POINT (1 2))");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(
            header.geometry_type_id().unwrap(),
            GeometryTypeId::GeometryCollection
        );

        // Some cases with z and m dimensions
        let wkb = make_wkb("POINT Z (1 2 3)");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(header.geometry_type_id().unwrap(), GeometryTypeId::Point);

        let wkb = make_wkb("LINESTRING Z (1 2 3, 4 5 6)");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(
            header.geometry_type_id().unwrap(),
            GeometryTypeId::LineString
        );

        let wkb = make_wkb("POLYGON M ((0 0 0, 0 1 0, 1 0 0, 0 0 0))");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(header.geometry_type_id().unwrap(), GeometryTypeId::Polygon);
    }

    #[test]
    fn empty_geometry_type_id() {
        let wkb = make_wkb("POINT EMPTY");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(header.geometry_type_id().unwrap(), GeometryTypeId::Point);

        let wkb = make_wkb("LINESTRING EMPTY");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(
            header.geometry_type_id().unwrap(),
            GeometryTypeId::LineString
        );

        let wkb = make_wkb("POLYGON EMPTY");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(header.geometry_type_id().unwrap(), GeometryTypeId::Polygon);

        let wkb = make_wkb("MULTIPOINT EMPTY");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(
            header.geometry_type_id().unwrap(),
            GeometryTypeId::MultiPoint
        );

        let wkb = make_wkb("MULTILINESTRING EMPTY");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(
            header.geometry_type_id().unwrap(),
            GeometryTypeId::MultiLineString
        );

        let wkb = make_wkb("MULTIPOLYGON EMPTY");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(
            header.geometry_type_id().unwrap(),
            GeometryTypeId::MultiPolygon
        );

        let wkb = make_wkb("GEOMETRYCOLLECTION EMPTY");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(
            header.geometry_type_id().unwrap(),
            GeometryTypeId::GeometryCollection
        );

        // z, m cases
        let wkb = make_wkb("POINT Z EMPTY");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(header.geometry_type_id().unwrap(), GeometryTypeId::Point);

        let wkb = make_wkb("POINT M EMPTY");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(header.geometry_type_id().unwrap(), GeometryTypeId::Point);

        let wkb = make_wkb("LINESTRING ZM EMPTY");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(
            header.geometry_type_id().unwrap(),
            GeometryTypeId::LineString
        );
    }

    #[test]
    fn dimension() {
        let wkb = make_wkb("POINT (1 2)");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(header.dimension().unwrap(), Dimensions::Xy);

        let wkb = make_wkb("POINT Z (1 2 3)");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(header.dimension().unwrap(), Dimensions::Xyz);

        let wkb = make_wkb("POINT M (1 2 3)");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(header.dimension().unwrap(), Dimensions::Xym);

        let wkb = make_wkb("POINT ZM (1 2 3 4)");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(header.dimension().unwrap(), Dimensions::Xyzm);
    }

    #[test]
    fn inferred_collections_dimension() {
        let wkb = make_wkb("GEOMETRYCOLLECTION (POINT Z (1 2 3))");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(header.dimension().unwrap(), Dimensions::Xyz);

        let wkb = make_wkb("GEOMETRYCOLLECTION (POINT ZM (1 2 3 4))");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(header.dimension().unwrap(), Dimensions::Xyzm);

        let wkb = make_wkb("GEOMETRYCOLLECTION (POINT M (1 2 3))");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(header.dimension().unwrap(), Dimensions::Xym);

        let wkb = make_wkb("GEOMETRYCOLLECTION (POINT ZM (1 2 3 4))");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(header.dimension().unwrap(), Dimensions::Xyzm);
    }

    #[test]
    fn empty_geometry_dimension() {
        // POINTs
        let wkb = make_wkb("POINT EMPTY");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(header.dimension().unwrap(), Dimensions::Xy);

        let wkb = make_wkb("POINT Z EMPTY");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(header.dimension().unwrap(), Dimensions::Xyz);

        let wkb = make_wkb("POINT M EMPTY");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(header.dimension().unwrap(), Dimensions::Xym);

        let wkb = make_wkb("POINT ZM EMPTY");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(header.dimension().unwrap(), Dimensions::Xyzm);

        // GEOMETRYCOLLECTIONs
        let wkb = make_wkb("GEOMETRYCOLLECTION EMPTY");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(header.dimension().unwrap(), Dimensions::Xy);

        let wkb = make_wkb("GEOMETRYCOLLECTION Z EMPTY");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(header.dimension().unwrap(), Dimensions::Xyz);

        let wkb = make_wkb("GEOMETRYCOLLECTION M EMPTY");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(header.dimension().unwrap(), Dimensions::Xym);

        let wkb = make_wkb("GEOMETRYCOLLECTION ZM EMPTY");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(header.dimension().unwrap(), Dimensions::Xyzm);

        let wkb = make_wkb("GEOMETRYCOLLECTION (POINT Z EMPTY)");
        let header = WkbHeader::new(&wkb).unwrap();
        assert_eq!(header.dimension().unwrap(), Dimensions::Xyz);
    }
}
