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
use std::{io::Write, sync::Arc, vec};

use arrow_array::builder::BinaryBuilder;
use arrow_schema::DataType;
use datafusion_common::{
    cast::{as_string_view_array, as_struct_array},
    error::Result,
    exec_datafusion_err, exec_err, ScalarValue,
};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use geo_traits::{
    GeometryCollectionTrait, GeometryTrait, LineStringTrait, MultiLineStringTrait, MultiPointTrait,
    MultiPolygonTrait, PointTrait, PolygonTrait,
};
use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::wkb_factory::{
    write_wkb_coord_trait, write_wkb_empty_point, WKB_MIN_PROBABLE_BYTES,
};
use sedona_schema::{crs::deserialize_crs, datatypes::SedonaType, matchers::ArgMatcher};
use wkb::reader::{Dimension, Wkb};

use crate::executor::WkbExecutor;

/// ST_AsEWKB() scalar UDF implementation
///
/// An implementation of WKB writing using GeoRust's wkt crate.
pub fn st_asewkb_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_asewkb",
        vec![Arc::new(STAsEWKBItemCrs {}), Arc::new(STAsEWKB {})],
        Volatility::Immutable,
        Some(st_asewkb_doc()),
    )
}

fn st_asewkb_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        r#"Return the Extended Well-Known Binary (EWKB) representation of a geometry or geography.

Compared to ST_AsBinary(), this function embeds an integer SRID derived from the type or dervied
from the item-level CRS for item CRS types. This is particularly useful for integration with
PostGIS"#,
        "ST_AsEWKB (A: Geometry)",
    )
    .with_argument("geom", "geometry: Input geometry or geography")
    .with_sql_example("SELECT ST_AsEWKB(ST_Point(1.0, 2.0, 4326))")
    .build()
}

#[derive(Debug)]
struct STAsEWKB {}

impl SedonaScalarKernel for STAsEWKB {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry_or_geography()],
            SedonaType::Arrow(DataType::Binary),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
        );

        let maybe_srid = match &arg_types[0] {
            SedonaType::Wkb(_, crs) | SedonaType::WkbView(_, crs) => match crs {
                Some(crs) => match crs.srid()? {
                    Some(srid) if srid != 0 => Some(srid),
                    _ => None,
                },
                None => None,
            },
            _ => return sedona_internal_err!("Unexpected input to invoke_batch in ST_AsEWKB"),
        };

        executor.execute_wkb_void(|maybe_wkb| {
            match maybe_wkb {
                Some(wkb) => {
                    write_geometry(&wkb, maybe_srid, &mut builder)?;
                    builder.append_value([]);
                }
                None => builder.append_null(),
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

#[derive(Debug)]
struct STAsEWKBItemCrs {}

impl SedonaScalarKernel for STAsEWKBItemCrs {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_item_crs()],
            SedonaType::Arrow(DataType::Binary),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
        );

        let crs_array_ref = match &args[0] {
            ColumnarValue::Array(array) => {
                let struct_array = as_struct_array(array)?;
                struct_array.column(1).clone()
            }
            ColumnarValue::Scalar(ScalarValue::Struct(struct_array)) => {
                struct_array.column(1).clone()
            }
            _ => return sedona_internal_err!("Unexpected item_crs type"),
        };

        let crs_array = as_string_view_array(&crs_array_ref)?;
        let mut srid_iter = crs_array
            .into_iter()
            .map(|maybe_crs_str| match maybe_crs_str {
                None => Ok(None),
                Some(crs_str) => match deserialize_crs(crs_str)? {
                    None => Ok(None),
                    Some(crs) => match crs.srid()? {
                        Some(srid) => Ok(Some(srid)),
                        None => exec_err!("CRS {crs} cannot be represented by a single SRID"),
                    },
                },
            });

        executor.execute_wkb_void(|maybe_wkb| {
            match maybe_wkb {
                Some(wkb) => {
                    write_geometry(&wkb, srid_iter.next().unwrap()?, &mut builder)?;
                    builder.append_value([]);
                }
                None => builder.append_null(),
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

const EWKB_Z_BIT: u32 = 0x80000000;
const EWKB_M_BIT: u32 = 0x40000000;
const EWKB_SRID_BIT: u32 = 0x20000000;

fn write_geometry(geom: &Wkb, srid: Option<u32>, buf: &mut impl Write) -> Result<()> {
    match geom.as_type() {
        geo_traits::GeometryType::Point(p) => write_point(p, srid, buf),
        geo_traits::GeometryType::LineString(ls) => write_linestring(ls, srid, buf),
        geo_traits::GeometryType::Polygon(poly) => write_polygon(poly, srid, buf),
        geo_traits::GeometryType::MultiPoint(mp) => write_multipoint(mp, srid, buf),
        geo_traits::GeometryType::MultiLineString(mls) => write_multilinestring(mls, srid, buf),
        geo_traits::GeometryType::MultiPolygon(mpoly) => write_multipolygon(mpoly, srid, buf),
        geo_traits::GeometryType::GeometryCollection(gc) => write_geometrycollection(gc, srid, buf),
        _ => exec_err!("Unsupported geometry type in ST_AsEWKB()"),
    }
}

fn write_point(geom: &wkb::reader::Point, srid: Option<u32>, buf: &mut impl Write) -> Result<()> {
    write_geometry_type_and_srid(1, geom.dimension(), srid, buf)?;
    match geom.byte_order() {
        wkb::Endianness::BigEndian => match geom.coord() {
            Some(c) => {
                write_wkb_coord_trait(buf, &c).map_err(|e| exec_datafusion_err!("write err {e}"))?
            }
            None => write_wkb_empty_point(buf, geom.dim())
                .map_err(|e| exec_datafusion_err!("write err {e}"))?,
        },
        wkb::Endianness::LittleEndian => buf.write_all(geom.coord_slice())?,
    }

    Ok(())
}

fn write_linestring(
    geom: &wkb::reader::LineString,
    srid: Option<u32>,
    buf: &mut impl Write,
) -> Result<()> {
    write_geometry_type_and_srid(2, geom.dimension(), srid, buf)?;
    let num_coords = geom.num_coords() as u32;
    buf.write_all(&num_coords.to_le_bytes())?;
    match geom.byte_order() {
        wkb::Endianness::BigEndian => {
            for c in geom.coords() {
                write_wkb_coord_trait(buf, &c)
                    .map_err(|e| exec_datafusion_err!("write err {e}"))?;
            }
        }
        wkb::Endianness::LittleEndian => buf.write_all(geom.coords_slice())?,
    }

    Ok(())
}

fn write_linearring(geom: &wkb::reader::LinearRing, buf: &mut impl Write) -> Result<()> {
    let num_coords = geom.num_coords() as u32;
    buf.write_all(&num_coords.to_le_bytes())?;
    match geom.byte_order() {
        wkb::Endianness::BigEndian => {
            for c in geom.coords() {
                write_wkb_coord_trait(buf, &c)
                    .map_err(|e| exec_datafusion_err!("write err {e}"))?;
            }
        }
        wkb::Endianness::LittleEndian => buf.write_all(geom.coords_slice())?,
    }

    Ok(())
}

fn write_polygon(
    geom: &wkb::reader::Polygon,
    srid: Option<u32>,
    buf: &mut impl Write,
) -> Result<()> {
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

fn write_multipoint(
    geom: &wkb::reader::MultiPoint,
    srid: Option<u32>,
    buf: &mut impl Write,
) -> Result<()> {
    write_geometry_type_and_srid(4, geom.dimension(), srid, buf)?;
    let num_children = geom.num_points();
    buf.write_all(&num_children.to_le_bytes())?;

    for child in geom.points() {
        write_point(&child, None, buf)?;
    }

    Ok(())
}

fn write_multilinestring(
    geom: &wkb::reader::MultiLineString,
    srid: Option<u32>,
    buf: &mut impl Write,
) -> Result<()> {
    write_geometry_type_and_srid(4, geom.dimension(), srid, buf)?;
    let num_children = geom.num_line_strings();
    buf.write_all(&num_children.to_le_bytes())?;

    for child in geom.line_strings() {
        write_linestring(child, None, buf)?;
    }

    Ok(())
}

fn write_multipolygon(
    geom: &wkb::reader::MultiPolygon,
    srid: Option<u32>,
    buf: &mut impl Write,
) -> Result<()> {
    write_geometry_type_and_srid(4, geom.dimension(), srid, buf)?;
    let num_children = geom.num_polygons();
    buf.write_all(&num_children.to_le_bytes())?;

    for child in geom.polygons() {
        write_polygon(child, None, buf)?;
    }

    Ok(())
}

fn write_geometrycollection(
    geom: &wkb::reader::GeometryCollection,
    srid: Option<u32>,
    buf: &mut impl Write,
) -> Result<()> {
    write_geometry_type_and_srid(4, geom.dimension(), srid, buf)?;
    let num_children = geom.num_geometries();
    buf.write_all(&num_children.to_le_bytes())?;

    for child in geom.geometries() {
        write_geometry(child, None, buf)?;
    }

    Ok(())
}

fn write_geometry_type_and_srid(
    mut base_type: u32,
    dimensions: Dimension,
    srid: Option<u32>,
    buf: &mut impl Write,
) -> Result<()> {
    buf.write_all(&[0x01])?;

    match dimensions {
        Dimension::Xy => {}
        Dimension::Xyz => base_type |= EWKB_Z_BIT,
        Dimension::Xym => base_type |= EWKB_M_BIT,
        Dimension::Xyzm => {
            base_type |= EWKB_Z_BIT;
            base_type |= EWKB_Z_BIT;
        }
    }

    if let Some(srid) = srid {
        base_type |= EWKB_SRID_BIT;
        buf.write_all(&base_type.to_le_bytes())?;
        buf.write_all(&srid.to_le_bytes())?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use arrow_array::{ArrayRef, BinaryArray, BinaryViewArray};
    use datafusion_common::scalar::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{
        WKB_GEOGRAPHY, WKB_GEOGRAPHY_ITEM_CRS, WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS,
        WKB_VIEW_GEOGRAPHY, WKB_VIEW_GEOMETRY,
    };
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    const POINT12: [u8; 21] = [
        0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x40,
    ];

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_asewkb_udf().into();
        assert_eq!(udf.name(), "st_asewkb");
        assert!(udf.documentation().is_some())
    }

    #[rstest]
    fn udf_geometry_input(
        #[values(
            WKB_GEOMETRY,
            WKB_GEOGRAPHY,
            WKB_GEOMETRY_ITEM_CRS.clone(),
            WKB_GEOGRAPHY_ITEM_CRS.clone(),
        )]
        sedona_type: SedonaType,
    ) {
        let udf = st_asewkb_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type]);

        assert_eq!(
            tester.invoke_wkb_scalar(Some("POINT (1 2)")).unwrap(),
            ScalarValue::Binary(Some(POINT12.to_vec()))
        );

        assert_eq!(
            tester.invoke_wkb_scalar(None).unwrap(),
            ScalarValue::Binary(None)
        );

        let expected_array: BinaryArray = [Some(POINT12), None, Some(POINT12)].iter().collect();
        assert_eq!(
            &tester
                .invoke_wkb_array(vec![Some("POINT (1 2)"), None, Some("POINT (1 2)")])
                .unwrap(),
            &(Arc::new(expected_array) as ArrayRef)
        );
    }

    #[rstest]
    fn udf_geometry_view_input(
        #[values(WKB_VIEW_GEOMETRY, WKB_VIEW_GEOGRAPHY)] sedona_type: SedonaType,
    ) {
        let udf = st_asewkb_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type]);

        assert_eq!(
            tester.invoke_wkb_scalar(Some("POINT (1 2)")).unwrap(),
            ScalarValue::BinaryView(Some(POINT12.to_vec()))
        );

        assert_eq!(
            tester.invoke_wkb_scalar(None).unwrap(),
            ScalarValue::BinaryView(None)
        );

        let expected_array: BinaryViewArray = [Some(POINT12), None, Some(POINT12)].iter().collect();
        assert_eq!(
            &tester
                .invoke_wkb_array(vec![Some("POINT (1 2)"), None, Some("POINT (1 2)")])
                .unwrap(),
            &(Arc::new(expected_array) as ArrayRef)
        );
    }

    #[test]
    fn aliases() {
        let udf: ScalarUDF = st_asewkb_udf().into();
        assert!(udf.aliases().contains(&"st_aswkb".to_string()));
    }
}
