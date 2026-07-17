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
use std::sync::Arc;

use arrow_array::builder::BinaryBuilder;
use datafusion_common::{exec_datafusion_err, Result};
use datafusion_expr::ColumnarValue;
use datafusion_expr::Volatility;
use geo_traits::Dimensions;
use geo_traits::{
    GeometryCollectionTrait, GeometryTrait, LineStringTrait, MultiLineStringTrait,
    MultiPolygonTrait, PolygonTrait,
};
use sedona_expr::item_crs::ItemCrsKernel;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::{
    error::SedonaGeometryError,
    wkb_factory::{
        write_wkb_geometrycollection_header, write_wkb_linestring_header,
        write_wkb_multilinestring_header, write_wkb_multipolygon_header, write_wkb_polygon_header,
        write_wkb_polygon_ring_header, WKB_MIN_PROBABLE_BYTES,
    },
};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOGRAPHY, WKB_GEOMETRY},
    matchers::ArgMatcher,
};
use wkb::reader::Wkb;
use wkb::Endianness;

use crate::executor::WkbExecutor;

/// ST_Reverse() scalar UDF
///
/// Native implementation to reverse the vertices in a geometry
pub fn st_reverse_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_reverse",
        ItemCrsKernel::wrap_impl(vec![
            Arc::new(STReverse {
                matcher: ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOMETRY),
            }),
            Arc::new(STReverse {
                matcher: ArgMatcher::new(vec![ArgMatcher::is_geography()], WKB_GEOGRAPHY),
            }),
        ]),
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct STReverse {
    matcher: ArgMatcher,
}

impl SedonaScalarKernel for STReverse {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        self.matcher.match_args(args)
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

        executor.execute_wkb_void(|maybe_wkb| {
            match maybe_wkb {
                Some(wkb) => {
                    invoke_scalar(wkb, &mut builder)
                        .map_err(|e| exec_datafusion_err!("ST_Reverse error: {e}"))?;
                    builder.append_value([]);
                }
                _ => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(geom: &Wkb, writer: &mut impl Write) -> Result<(), SedonaGeometryError> {
    let dims = geom.dim();
    match geom.as_type() {
        geo_traits::GeometryType::Point(_) | geo_traits::GeometryType::MultiPoint(_) => {
            // Note: in the case of big endian input, this may result in mixed endian output.
            // Mixed endian output should be handled by all readers but is probably not well tested.
            writer.write_all(geom.buf())?;
        }

        geo_traits::GeometryType::LineString(ls) => {
            write_reversed_linestring(writer, ls, dims)?;
        }

        geo_traits::GeometryType::Polygon(pgn) => {
            write_reversed_polygon(writer, pgn, dims)?;
        }

        geo_traits::GeometryType::MultiLineString(mls) => {
            write_wkb_multilinestring_header(writer, dims, mls.num_line_strings())?;
            for ls in mls.line_strings() {
                write_reversed_linestring(writer, ls, dims)?;
            }
        }

        geo_traits::GeometryType::MultiPolygon(mpgn) => {
            write_wkb_multipolygon_header(writer, dims, mpgn.num_polygons())?;
            for pgn in mpgn.polygons() {
                write_reversed_polygon(writer, pgn, dims)?;
            }
        }

        geo_traits::GeometryType::GeometryCollection(gcn) => {
            write_wkb_geometrycollection_header(writer, dims, gcn.num_geometries())?;
            for geom in gcn.geometries() {
                invoke_scalar(geom, writer)?;
            }
        }

        _ => {
            return Err(SedonaGeometryError::Invalid(
                "Unsupported geometry type for reversal operation".to_string(),
            ));
        }
    }
    Ok(())
}

fn write_reversed_linestring(
    writer: &mut impl Write,
    ls: &wkb::reader::LineString,
    dims: Dimensions,
) -> Result<(), SedonaGeometryError> {
    write_wkb_linestring_header(writer, dims, ls.num_coords())?;
    write_reversed_coords(writer, ls.coords_slice(), dims.size(), ls.byte_order())?;
    Ok(())
}

fn write_reversed_polygon(
    writer: &mut impl Write,
    pgn: &wkb::reader::Polygon,
    dims: Dimensions,
) -> Result<(), SedonaGeometryError> {
    let num_rings = pgn.num_interiors() + pgn.exterior().is_some() as usize;
    write_wkb_polygon_header(writer, dims, num_rings)?;

    if let Some(exterior) = pgn.exterior() {
        write_reversed_ring(writer, exterior, dims.size())?;
    }

    for interior in pgn.interiors() {
        write_reversed_ring(writer, interior, dims.size())?;
    }

    Ok(())
}

fn write_reversed_ring(
    writer: &mut impl Write,
    ring: &wkb::reader::LinearRing,
    dim_size: usize,
) -> Result<(), SedonaGeometryError> {
    write_wkb_polygon_ring_header(writer, ring.num_coords())?;
    write_reversed_coords(writer, ring.coords_slice(), dim_size, ring.byte_order())
}

fn write_reversed_coords(
    writer: &mut impl Write,
    coords: &[u8],
    dim_size: usize,
    endianness: Endianness,
) -> Result<(), SedonaGeometryError> {
    let coord_bytes = dim_size * size_of::<f64>();
    let needs_byteswap = matches!(endianness, Endianness::BigEndian);

    if needs_byteswap {
        let mut ord_reversed = [0u8; size_of::<f64>()];
        for coord in coords.rchunks_exact(coord_bytes) {
            for ord in coord.chunks_exact(size_of::<f64>()) {
                ord_reversed.copy_from_slice(ord);
                ord_reversed.reverse();
                writer.write_all(&ord_reversed)?;
            }
        }
    } else {
        for coord in coords.rchunks_exact(coord_bytes) {
            writer.write_all(coord)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use datafusion_common::ScalarValue;
    use rstest::rstest;
    use sedona_schema::datatypes::{WKB_GEOGRAPHY_ITEM_CRS, WKB_GEOMETRY_ITEM_CRS};
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::create::create_array;
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_GEOGRAPHY)] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(st_reverse_udf().into(), vec![sedona_type.clone()]);
        tester.assert_return_type(sedona_type.clone());

        let result = tester.invoke_scalar("POINT EMPTY").unwrap();
        tester.assert_scalar_result_equals(result, "POINT EMPTY");

        let result = tester.invoke_scalar("POINT (30 10)").unwrap();
        tester.assert_scalar_result_equals(result, "POINT (30 10)");

        let result = tester
            .invoke_scalar("LINESTRING (30 10, 10 30, 40 40)")
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING (40 40, 10 30, 30 10)");

        let result = tester
            .invoke_scalar("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")
            .unwrap();
        tester.assert_scalar_result_equals(result, "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))");

        let result = tester
            .invoke_scalar("MULTIPOINT ((10 40), (40 30), (20 20), (30 10))")
            .unwrap();
        tester
            .assert_scalar_result_equals(result, "MULTIPOINT ((10 40), (40 30), (20 20), (30 10))");

        let result = tester
            .invoke_scalar("MULTILINESTRING ((10 10, 20 20), (15 15, 30 15))")
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            "MULTILINESTRING ((20 20, 10 10), (30 15, 15 15))",
        );

        let result = tester
            .invoke_scalar("MULTIPOLYGON (((10 10, 10 20, 20 20, 20 15, 10 10)), ((60 60, 70 70, 80 60, 60 60)))")
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            "MULTIPOLYGON (((10 10, 20 15, 20 20, 10 20, 10 10)), ((60 60, 80 60, 70 70, 60 60)))",
        );

        let result = tester
            .invoke_scalar(
                "GEOMETRYCOLLECTION (MULTIPOINT (3 4, 1 2, 7 8, 5 6), LINESTRING (1 10, 1 2))",
            )
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            "GEOMETRYCOLLECTION (MULTIPOINT ((3 4), (1 2), (7 8), (5 6)), LINESTRING (1 2, 1 10))",
        );

        let result = tester
            .invoke_scalar("GEOMETRYCOLLECTION (POINT (10 40), LINESTRING (30 10, 10 30, 40 40), POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10)))")
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            "GEOMETRYCOLLECTION (POINT (10 40), LINESTRING (40 40, 10 30, 30 10), POLYGON ((30 10, 10 20, 20 40, 40 40, 30 10)))",
        );

        let result = tester
            .invoke_scalar(
                "GEOMETRYCOLLECTION (
                POINT (10 10),
                LINESTRING (10 20, 20 20, 20 30),
                GEOMETRYCOLLECTION (
                    POLYGON ((40 40, 50 50, 60 40, 40 40)),
                    MULTIPOINT (70 70, 80 80)
                ),
                GEOMETRYCOLLECTION (
                    LINESTRING (90 90, 100 100),
                    POINT (95 95)
                )
            )",
            )
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            "GEOMETRYCOLLECTION(
            POINT(10 10),
            LINESTRING(20 30,20 20,10 20),
            GEOMETRYCOLLECTION(
                POLYGON((40 40,60 40,50 50,40 40)),
                MULTIPOINT((70 70),(80 80))
            ),
            GEOMETRYCOLLECTION(
                LINESTRING(100 100,90 90),
                POINT(95 95)
            )
            )",
        );

        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        assert!(result.is_null());

        let input_wkt = vec![
            // Null case
            None,
            // POINT types
            Some("POINT EMPTY"),
            Some("POINT (1 2)"),
            Some("POINT Z EMPTY"),
            Some("POINT Z (1 2 3)"),
            Some("POINT M EMPTY"),
            Some("POINT M (1 2 3)"),
            Some("POINT ZM EMPTY"),
            Some("POINT ZM (1 2 3 4)"),
            // LINESTRING types
            Some("LINESTRING EMPTY"),
            Some("LINESTRING (1 2, 1 10)"),
            Some("LINESTRING (0 0, 1 1, 2 2)"),
            Some("LINESTRING (10 20, 30 40)"),
            Some("LINESTRING Z EMPTY"),
            Some("LINESTRING Z (1 2 3, 4 5 6)"),
            Some("LINESTRING M EMPTY"),
            Some("LINESTRING M (1 2 3, 4 5 6)"),
            Some("LINESTRING ZM EMPTY"),
            Some("LINESTRING ZM (1 2 3 4, 5 6 7 8)"),
            // POLYGON types
            Some("POLYGON EMPTY"),
            Some("POLYGON ((2 2, 2 3, 3 3, 3 2, 2 2))"),
            Some("POLYGON Z EMPTY"),
            Some("POLYGON Z ((0 0 0, 0 1 0, 1 1 0, 1 0 0, 0 0 0))"),
            Some("POLYGON M EMPTY"),
            Some("POLYGON M ((0 0 0, 0 1 0, 1 1 0, 1 0 0, 0 0 0))"),
            Some("POLYGON ZM EMPTY"),
            Some("POLYGON ZM ((0 0 0 0, 0 1 0 0, 1 1 0 0, 1 0 0 0, 0 0 0 0))"),
            // MULTIPOINT types
            Some("MULTIPOINT EMPTY"),
            Some("MULTIPOINT((3 4),(1 2),(7 8),(5 6))"),
            Some("MULTIPOINT Z EMPTY"),
            Some("MULTIPOINT Z ((1 2 3), (4 5 6))"),
            Some("MULTIPOINT M EMPTY"),
            Some("MULTIPOINT M ((1 2 3), (4 5 6))"),
            Some("MULTIPOINT ZM EMPTY"),
            Some("MULTIPOINT ZM ((1 2 3 4), (5 6 7 8))"),
            // MULTILINESTRING types
            Some("MULTILINESTRING EMPTY"),
            Some("MULTILINESTRING ((1 2, 3 4), (5 6, 7 8))"),
            Some("MULTILINESTRING Z EMPTY"),
            Some("MULTILINESTRING Z ((1 2 3, 4 5 6), (7 8 9, 10 11 12))"),
            Some("MULTILINESTRING M EMPTY"),
            Some("MULTILINESTRING M ((1 2 3, 4 5 6), (7 8 9, 10 11 12))"),
            Some("MULTILINESTRING ZM EMPTY"),
            Some("MULTILINESTRING ZM ((1 2 3 4, 5 6 7 8), (9 10 11 12, 13 14 15 16))"),
            // MULTIPOLYGON types
            Some("MULTIPOLYGON EMPTY"),
            Some("MULTIPOLYGON (((0 0, 0 1, 1 1, 1 0, 0 0)), ((2 2, 2 3, 3 3, 3 2, 2 2)))"),
            Some("MULTIPOLYGON Z EMPTY"),
            Some("MULTIPOLYGON Z (((0 0 0, 0 1 0, 1 1 0, 1 0 0, 0 0 0)))"),
            Some("MULTIPOLYGON M EMPTY"),
            Some("MULTIPOLYGON M (((0 0 0, 0 1 0, 1 1 0, 1 0 0, 0 0 0)))"),
            Some("MULTIPOLYGON ZM EMPTY"),
            Some("MULTIPOLYGON ZM (((0 0 0 0, 0 1 0 0, 1 1 0 0, 1 0 0 0, 0 0 0 0)))"),
            // GEOMETRYCOLLECTION types
            Some("GEOMETRYCOLLECTION EMPTY"),
            Some(
                "GEOMETRYCOLLECTION (MULTIPOINT((3 4),(1 2),(7 8),(5 6)), LINESTRING (1 10, 1 2))",
            ),
            Some("GEOMETRYCOLLECTION (POINT Z (1 2 3), LINESTRING Z (1 2 3, 4 5 6))"),
            Some("GEOMETRYCOLLECTION (POINT M (1 2 3), LINESTRING M (1 2 3, 4 5 6))"),
            Some("GEOMETRYCOLLECTION (POINT ZM (1 2 3 4), LINESTRING ZM (1 2 3 4, 5 6 7 8))"),
        ];

        let expected = create_array(
            &[
                // Null case
                None,
                // POINT types (unchanged - points have no direction)
                Some("POINT EMPTY"),
                Some("POINT (1 2)"),
                Some("POINT Z EMPTY"),
                Some("POINT Z (1 2 3)"),
                Some("POINT M EMPTY"),
                Some("POINT M (1 2 3)"),
                Some("POINT ZM EMPTY"),
                Some("POINT ZM (1 2 3 4)"),
                // LINESTRING types (vertex order reversed)
                Some("LINESTRING EMPTY"),
                Some("LINESTRING (1 10, 1 2)"),
                Some("LINESTRING (2 2, 1 1, 0 0)"),
                Some("LINESTRING (30 40, 10 20)"),
                Some("LINESTRING Z EMPTY"),
                Some("LINESTRING Z (4 5 6, 1 2 3)"),
                Some("LINESTRING M EMPTY"),
                Some("LINESTRING M (4 5 6, 1 2 3)"),
                Some("LINESTRING ZM EMPTY"),
                Some("LINESTRING ZM (5 6 7 8, 1 2 3 4)"),
                // POLYGON types (ring vertex order reversed)
                Some("POLYGON EMPTY"),
                Some("POLYGON ((2 2, 3 2, 3 3, 2 3, 2 2))"),
                Some("POLYGON Z EMPTY"),
                Some("POLYGON Z ((0 0 0, 1 0 0, 1 1 0, 0 1 0, 0 0 0))"),
                Some("POLYGON M EMPTY"),
                Some("POLYGON M ((0 0 0, 1 0 0, 1 1 0, 0 1 0, 0 0 0))"),
                Some("POLYGON ZM EMPTY"),
                Some("POLYGON ZM ((0 0 0 0, 1 0 0 0, 1 1 0 0, 0 1 0 0, 0 0 0 0))"),
                // MULTIPOINT types (no change)
                Some("MULTIPOINT EMPTY"),
                Some("MULTIPOINT((3 4),(1 2),(7 8),(5 6))"),
                Some("MULTIPOINT Z EMPTY"),
                Some("MULTIPOINT Z ((1 2 3), (4 5 6))"),
                Some("MULTIPOINT M EMPTY"),
                Some("MULTIPOINT M ((1 2 3), (4 5 6))"),
                Some("MULTIPOINT ZM EMPTY"),
                Some("MULTIPOINT ZM ((1 2 3 4), (5 6 7 8))"),
                // MULTILINESTRING types (each linestring reversed individually)
                Some("MULTILINESTRING EMPTY"),
                Some("MULTILINESTRING ((3 4, 1 2), (7 8, 5 6))"),
                Some("MULTILINESTRING Z EMPTY"),
                Some("MULTILINESTRING Z ((4 5 6, 1 2 3), (10 11 12, 7 8 9))"),
                Some("MULTILINESTRING M EMPTY"),
                Some("MULTILINESTRING M ((4 5 6, 1 2 3), (10 11 12, 7 8 9))"),
                Some("MULTILINESTRING ZM EMPTY"),
                Some("MULTILINESTRING ZM ((5 6 7 8, 1 2 3 4), (13 14 15 16, 9 10 11 12))"),
                // MULTIPOLYGON types (each polygon reversed individually)
                Some("MULTIPOLYGON EMPTY"),
                Some("MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)), ((2 2, 3 2, 3 3, 2 3, 2 2)))"),
                Some("MULTIPOLYGON Z EMPTY"),
                Some("MULTIPOLYGON Z (((0 0 0, 1 0 0, 1 1 0, 0 1 0, 0 0 0)))"),
                Some("MULTIPOLYGON M EMPTY"),
                Some("MULTIPOLYGON M (((0 0 0, 1 0 0, 1 1 0, 0 1 0, 0 0 0)))"),
                Some("MULTIPOLYGON ZM EMPTY"),
                Some("MULTIPOLYGON ZM (((0 0 0 0, 1 0 0 0, 1 1 0 0, 0 1 0 0, 0 0 0 0)))"),
                // GEOMETRYCOLLECTION types (each member geometry reversed)
                Some("GEOMETRYCOLLECTION EMPTY"),
                Some("GEOMETRYCOLLECTION (MULTIPOINT((3 4),(1 2),(7 8),(5 6)), LINESTRING (1 2, 1 10))"),
                Some("GEOMETRYCOLLECTION (POINT Z (1 2 3), LINESTRING Z (4 5 6, 1 2 3))"),
                Some("GEOMETRYCOLLECTION (POINT M (1 2 3), LINESTRING M (4 5 6, 1 2 3))"),
                Some("GEOMETRYCOLLECTION (POINT ZM (1 2 3 4), LINESTRING ZM (5 6 7 8, 1 2 3 4))"),
            ],
            &sedona_type,
        );

        assert_array_equal(&tester.invoke_wkb_array(input_wkt).unwrap(), &expected);
    }

    #[test]
    fn udf_big_endian_linestring() {
        // Big-endian WKB for LINESTRING (1 2, 3 4)
        #[rustfmt::skip]
        let big_endian_wkb: &[u8] = &[
            0x00,                               // big-endian
            0x00, 0x00, 0x00, 0x02,              // LINESTRING
            0x00, 0x00, 0x00, 0x02,              // 2 points
            0x3F, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // x=1.0
            0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // y=2.0
            0x40, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // x=3.0
            0x40, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // y=4.0
        ];

        let tester = ScalarUdfTester::new(st_reverse_udf().into(), vec![WKB_GEOMETRY.clone()]);
        let input = Arc::new(arrow_array::BinaryArray::from_vec(vec![big_endian_wkb]));
        let result = tester.invoke_array(input).unwrap();

        // Result should be LINESTRING (3 4, 1 2) in little-endian WKB
        let expected = create_array(&[Some("LINESTRING (3 4, 1 2)")], &WKB_GEOMETRY);
        assert_array_equal(&result, &expected);
    }

    #[rstest]
    fn udf_invoke_item_crs(
        #[values(WKB_GEOMETRY_ITEM_CRS.clone(), WKB_GEOGRAPHY_ITEM_CRS.clone())]
        sedona_type: SedonaType,
    ) {
        let tester = ScalarUdfTester::new(st_reverse_udf().into(), vec![sedona_type.clone()]);
        tester.assert_return_type(sedona_type);

        let result = tester
            .invoke_scalar("LINESTRING (30 10, 10 30, 40 40)")
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING (40 40, 10 30, 30 10)");
    }
}
