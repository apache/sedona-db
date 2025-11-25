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
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::ColumnarValue;
use datafusion_expr::{scalar_doc_sections::DOC_SECTION_OTHER, Documentation, Volatility};
use geo_traits::{
    CoordTrait, Dimensions, GeometryCollectionTrait, GeometryTrait, LineStringTrait,
    MultiLineStringTrait, MultiPointTrait, MultiPolygonTrait, PointTrait, PolygonTrait,
};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::wkb_factory::{
    write_wkb_coord, write_wkb_empty_point, write_wkb_geometrycollection_header,
    write_wkb_linestring_header, write_wkb_multilinestring_header, write_wkb_multipoint_header,
    write_wkb_multipolygon_header, write_wkb_point_header, write_wkb_polygon_header,
    write_wkb_polygon_ring_header, WKB_MIN_PROBABLE_BYTES,
};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

use crate::executor::WkbExecutor;

/// ST_Reverse() scalar UDF
///
/// Native implementation to reverse the vertices in a geometry
pub fn st_reverse_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_reverse",
        vec![Arc::new(STReverse)],
        Volatility::Immutable,
        Some(st_reverse_doc()),
    )
}

fn st_reverse_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Can be used on any geometry and reverses the order of the vertices.",
        "ST_Reverse (geom: Geometry)",
    )
    .with_argument("geom", "geometry: Input geometry")
    .with_sql_example("SELECT ST_AsText(ST_Reverse('POLYGON ((2 2, 2 3, 3 3, 3 2, 2 2))'))")
    .build()
}

#[derive(Debug)]
struct STReverse;

impl SedonaScalarKernel for STReverse {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOMETRY);

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

        executor.execute_wkb_void(|maybe_wkb| {
            match maybe_wkb {
                Some(wkb) => {
                    invoke_scalar(&wkb, &mut builder)?;
                    builder.append_value([]);
                }
                _ => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(geom: &impl GeometryTrait<T = f64>, writer: &mut impl Write) -> Result<()> {
    let dims = geom.dim();
    match geom.as_type() {
        geo_traits::GeometryType::Point(pt) => {
            if pt.coord().is_some() {
                write_wkb_point_header(writer, dims)
                    .map_err(|e| DataFusionError::Execution(e.to_string()))?;
                write_coord(writer, pt.coord().unwrap())?;
            } else {
                write_wkb_empty_point(writer, dims)
                    .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            }
        }

        geo_traits::GeometryType::MultiPoint(multi_point) => {
            write_wkb_multipoint_header(writer, dims, multi_point.points().count())
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            for pt in multi_point.points() {
                invoke_scalar(&pt, writer)?;
            }
        }

        geo_traits::GeometryType::LineString(ls) => {
            let coords: Vec<_> = ls.coords().collect();
            write_wkb_linestring_header(writer, dims, coords.len())
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            for coord in coords.into_iter().rev() {
                write_coord(writer, coord)?;
            }
        }

        geo_traits::GeometryType::Polygon(pgn) => {
            let num_rings = pgn.interiors().count() + pgn.exterior().is_some() as usize;
            write_wkb_polygon_header(writer, dims, num_rings)
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;

            if let Some(exterior) = pgn.exterior() {
                write_reversed_ring(writer, exterior)?;
            }

            for interior in pgn.interiors() {
                write_reversed_ring(writer, interior)?;
            }
        }

        geo_traits::GeometryType::MultiLineString(mls) => {
            write_wkb_multilinestring_header(writer, dims, mls.line_strings().count())
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            for ls in mls.line_strings() {
                invoke_scalar(&ls, writer)?;
            }
        }

        geo_traits::GeometryType::MultiPolygon(mpgn) => {
            write_wkb_multipolygon_header(writer, dims, mpgn.polygons().count())
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            for pgn in mpgn.polygons() {
                invoke_scalar(&pgn, writer)?;
            }
        }

        geo_traits::GeometryType::GeometryCollection(gcn) => {
            write_wkb_geometrycollection_header(writer, dims, gcn.geometries().count())
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            for geom in gcn.geometries() {
                invoke_scalar(&geom, writer)?;
            }
        }

        _ => {
            return Err(DataFusionError::Execution(
                "Unsupported geometry type for reversal operation".to_string(),
            ));
        }
    }
    Ok(())
}

fn write_reversed_ring(writer: &mut impl Write, ring: impl LineStringTrait<T = f64>) -> Result<()> {
    let coords: Vec<_> = ring.coords().collect();
    write_wkb_polygon_ring_header(writer, coords.len())
        .map_err(|e| DataFusionError::Execution(e.to_string()))?;
    for coord in coords.into_iter().rev() {
        write_coord(writer, coord)?;
    }
    Ok(())
}

fn write_coord(writer: &mut impl Write, coord: impl CoordTrait<T = f64>) -> Result<()> {
    match coord.dim() {
        Dimensions::Xy => {
            write_wkb_coord(writer, (coord.x(), coord.y()))
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
        }
        Dimensions::Xyz => {
            write_wkb_coord(writer, (coord.x(), coord.y(), coord.nth_or_panic(2)))
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
        }
        Dimensions::Xym => {
            write_wkb_coord(writer, (coord.x(), coord.y(), coord.nth_or_panic(2)))
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
        }
        Dimensions::Xyzm => {
            write_wkb_coord(
                writer,
                (
                    coord.x(),
                    coord.y(),
                    coord.nth_or_panic(2),
                    coord.nth_or_panic(3),
                ),
            )
            .map_err(|e| DataFusionError::Execution(e.to_string()))?;
        }
        _ => {
            return Err(DataFusionError::Execution(
                "Unsupported dimensions for coordinate".to_string(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use datafusion_common::ScalarValue;
    use rstest::rstest;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::create::create_array;
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(st_reverse_udf().into(), vec![sedona_type]);
        tester.assert_return_type(WKB_GEOMETRY);

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
            Some("POLYGON ((2 2, 2 3, 3 3, 3 2, 2 2))"),
            Some("POINT EMPTY"),
            Some("POINT (1 2)"),
            Some("LINESTRING (1 2, 1 10)"),
            Some("GEOMETRYCOLLECTION (MULTIPOINT (3 4, 1 2, 7 8, 5 6), LINESTRING (1 10, 1 2))"),
            None,
        ];

        let expected = create_array(
            &[
                Some("POLYGON ((2 2, 3 2, 3 3, 2 3, 2 2))"),
                Some("POINT EMPTY"),
                Some("POINT (1 2)"),
                Some("LINESTRING (1 10, 1 2)"),
                Some("GEOMETRYCOLLECTION (MULTIPOINT ((3 4), (1 2), (7 8), (5 6)), LINESTRING (1 2, 1 10))"),
                None,
            ],
            &WKB_GEOMETRY,
        );

        assert_array_equal(&tester.invoke_wkb_array(input_wkt).unwrap(), &expected);
    }
}
