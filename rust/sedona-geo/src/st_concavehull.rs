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

use std::sync::Arc;

use arrow_array::builder::BinaryBuilder;
use datafusion_common::error::Result;
use datafusion_common::{cast::as_float64_array, DataFusionError};
use datafusion_expr::ColumnarValue;
use geo::{ConcaveHull, CoordsIter, Geometry, GeometryCollection, Point, Polygon};
use geo_traits::to_geo::{ToGeoGeometry, ToGeoPoint};
use geo_traits::{GeometryCollectionTrait, GeometryTrait, MultiPointTrait, PointTrait};
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_functions::executor::WkbExecutor;
use sedona_geometry::is_empty;
use sedona_geometry::wkb_factory::WKB_MIN_PROBABLE_BYTES;
use sedona_schema::datatypes::SedonaType;
use sedona_schema::{datatypes::WKB_GEOMETRY, matchers::ArgMatcher};
use wkb::reader::Wkb;
use wkb::writer::{write_geometry, WriteOptions};

use crate::to_geo::item_to_geometry;

/// ST_ConcaveHull implementation using [ConcaveHull]
///
/// Geo returns a Polygon for every concave hull computation
/// whereas the Geos implementation returns a MultiPolygon for
/// certain geometries concave hull computation.
pub fn st_concavehull_impl() -> ScalarKernelRef {
    Arc::new(STConcaveHull {})
}

#[derive(Debug)]
struct STConcaveHull {}

impl SedonaScalarKernel for STConcaveHull {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry(), ArgMatcher::is_numeric()],
            WKB_GEOMETRY,
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        invoke_batch_impl(arg_types, args)
    }
}

fn invoke_batch_impl(arg_types: &[SedonaType], args: &[ColumnarValue]) -> Result<ColumnarValue> {
    let executor = WkbExecutor::new(arg_types, args);
    let mut builder = BinaryBuilder::with_capacity(
        executor.num_iterations(),
        WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
    );

    // Extract Args
    let pct_convex_val = args[1]
        .cast_to(&arrow_schema::DataType::Float64, None)?
        .to_array(executor.num_iterations())?;
    let pct_convex_array = as_float64_array(&pct_convex_val)?;
    let mut pct_convex_iter = pct_convex_array.iter();

    executor.execute_wkb_void(|maybe_wkb| {
        match (maybe_wkb, pct_convex_iter.next().unwrap()) {
            (Some(wkb), Some(pct_convex)) => {
                invoke_scalar(&wkb, pct_convex, &mut builder)?;
                builder.append_value([]);
            }
            _ => builder.append_null(),
        }

        Ok(())
    })?;

    executor.finish(Arc::new(builder.finish()))
}

fn invoke_scalar(geom: &Wkb, pct_convex: f64, writer: &mut impl std::io::Write) -> Result<()> {
    if is_empty::is_geometry_empty(geom).map_err(|e| DataFusionError::Execution(e.to_string()))? {
        write_geometry(writer, &Polygon::<f64>::empty(), &write_opts())
            .map_err(|e| DataFusionError::Execution(e.to_string()))?;
        return Ok(());
    }
    compute_and_write_hull(&normalize_geometry(geom)?, pct_convex, writer)
}

fn write_opts() -> WriteOptions {
    WriteOptions {
        endianness: wkb::Endianness::LittleEndian,
    }
}

fn normalize_geometry(geom: &Wkb) -> Result<Geometry> {
    match geom.as_type() {
        geo_traits::GeometryType::GeometryCollection(gc) => {
            let filtered: Vec<Geometry> = gc
                .geometries()
                .filter(|g| !is_empty::is_geometry_empty(g).unwrap_or(true))
                .map(|g| g.to_geometry())
                .collect();
            Ok(Geometry::GeometryCollection(GeometryCollection::new_from(
                filtered,
            )))
        }

        geo_traits::GeometryType::MultiPoint(mp) => {
            let filtered: Vec<geo_types::Point> = mp
                .points()
                .filter_map(|pt| pt.coord().map(|_| pt.to_point()))
                .collect();
            Ok(Geometry::MultiPoint(geo_types::MultiPoint::new(filtered)))
        }

        geo_traits::GeometryType::LineString(ls) => item_to_geometry(ls),
        geo_traits::GeometryType::MultiLineString(mls) => item_to_geometry(mls),
        geo_traits::GeometryType::Polygon(pgn) => item_to_geometry(pgn),
        geo_traits::GeometryType::MultiPolygon(mpgn) => item_to_geometry(mpgn),
        geo_traits::GeometryType::Point(pt) => item_to_geometry(pt),

        _ => Err(DataFusionError::Execution(
            "Unsupported geometry type".to_string(),
        )),
    }
}

fn compute_and_write_hull(
    geom: &Geometry,
    pct_convex: f64,
    writer: &mut impl std::io::Write,
) -> Result<()> {
    match geom.as_type() {
        geo_traits::GeometryType::Point(pt) => wkb::writer::write_point(writer, pt, &write_opts())
            .map_err(|e| DataFusionError::Execution(e.to_string()))?,

        geo_traits::GeometryType::MultiPoint(mpt) => {
            let hull = geo_types::MultiPoint::new(
                mpt.points()
                    .filter(|pt| pt.coord().is_some())
                    .copied()
                    .collect::<Vec<Point>>(),
            )
            .concave_hull(pct_convex);
            write_concave_hull(writer, hull)?;
        }

        geo_traits::GeometryType::LineString(ls) => {
            let hull = ls.concave_hull(pct_convex);
            write_concave_hull(writer, hull)?;
        }

        geo_traits::GeometryType::Polygon(pgn) => {
            let hull = pgn.concave_hull(pct_convex);
            write_concave_hull(writer, hull)?;
        }

        geo_traits::GeometryType::MultiLineString(mls) => {
            let hull = mls.concave_hull(pct_convex);
            write_concave_hull(writer, hull)?;
        }

        geo_traits::GeometryType::MultiPolygon(mpgn) => {
            let hull = mpgn.concave_hull(pct_convex);
            write_concave_hull(writer, hull)?;
        }

        geo_traits::GeometryType::GeometryCollection(gcn) => {
            let coords: Vec<geo_types::Coord> = gcn
                .geometries()
                .flat_map(|geom| geom.coords_iter())
                .collect();

            let multi_point = geo_types::MultiPoint::new(
                coords.into_iter().map(geo_types::Point::from).collect(),
            );

            let hull = multi_point.concave_hull(pct_convex);
            write_concave_hull(writer, hull)?;
        }

        _ => {
            return Err(DataFusionError::Execution(
                "Unsupported geometry type for concave hull".to_string(),
            ))
        }
    }

    Ok(())
}

fn write_concave_hull<W: std::io::Write>(
    writer: &mut W,
    hull: impl GeometryTrait<T = f64>,
) -> Result<()> {
    wkb::writer::write_geometry(writer, &hull, &write_opts())
        .map_err(|e| DataFusionError::Execution(e.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Float64Array;
    use arrow_schema::DataType;
    use datafusion_common::ScalarValue;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::{
        compare::{assert_array_equal, assert_scalar_equal_wkb_geometry_topologically},
        create::create_array,
        testers::ScalarUdfTester,
    };

    /// Helper to initialize the UDF tester to avoid boilerplate in every test.
    fn create_tester(sedona_type: SedonaType) -> ScalarUdfTester {
        let udf = SedonaScalarUDF::from_kernel("st_concavehull", st_concavehull_impl());
        ScalarUdfTester::new(
            udf.into(),
            vec![sedona_type, SedonaType::Arrow(DataType::Float64)],
        )
    }

    #[rstest]
    fn test_return_type(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = create_tester(sedona_type);
        assert_eq!(tester.return_type().unwrap(), WKB_GEOMETRY);
    }

    #[rstest]
    fn test_null_inputs(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = create_tester(sedona_type);
        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());
    }

    #[rstest]
    fn test_empty_geometries(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_concavehull", st_concavehull_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Float64)],
        );

        let input_wkt_array = vec![
            Some("POINT EMPTY"),
            Some("LINESTRING EMPTY"),
            Some("POLYGON EMPTY"),
            Some("MULTIPOINT EMPTY"),
            Some("MULTILINESTRING EMPTY"),
            Some("MULTIPOLYGON EMPTY"),
            Some("GEOMETRYCOLLECTION EMPTY"),
        ];

        let tolerance_array: Arc<Float64Array> = Arc::new(Float64Array::from(vec![
            Some(0.1),
            Some(0.1),
            Some(0.1),
            Some(0.1),
            Some(0.1),
            Some(0.1),
            Some(0.1),
        ]));

        let expected_array = create_array(
            &[
                Some("POLYGON EMPTY"),
                Some("POLYGON EMPTY"),
                Some("POLYGON EMPTY"),
                Some("POLYGON EMPTY"),
                Some("POLYGON EMPTY"),
                Some("POLYGON EMPTY"),
                Some("POLYGON EMPTY"),
            ],
            &WKB_GEOMETRY,
        );

        assert_array_equal(
            &tester
                .invoke_arrays(vec![
                    create_array(&input_wkt_array, &sedona_type),
                    tolerance_array,
                ])
                .unwrap(),
            &expected_array,
        );
    }

    /// For testing here, the Python method was adopted as the `assert_array_equal`
    /// doesn't assert `topologically`
    #[rstest]
    fn test_different_geometry_types_fixed_parameter(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let tester = create_tester(sedona_type);

        // =========================================================================
        // Differing Geometry Types, Identical Percentage (0.1)
        // =========================================================================
        let test_cases = vec![
            ("POINT (2.5 3.1)", "POINT (2.5 3.1)"),
            (
                "LINESTRING (50 50, 150 150, 150 50)",
                "POLYGON ((50 50, 150 150, 150 50, 50 50))",
            ),
            (
                "MULTIPOINT ((0 0), (10 0), (0 10), (10 10), (5 5))",
                "POLYGON ((10 0, 10 10, 0 10, 0 0, 5 5, 10 0))",
            ),
            (
                "MULTILINESTRING ((10 10, 20 20, 10 40), (40 40, 30 30, 40 20, 30 10))",
                "POLYGON ((20 20, 10 40, 30 30, 40 40, 40 20, 30 10, 10 10, 20 20))",
            ),
            (
                "MULTIPOLYGON (((2 2, 2 5, 5 5, 5 2, 2 2)), ((6 3, 8 3, 8 1, 6 1, 6 3)))",
                "POLYGON ((5 2, 2 2, 2 5, 5 5, 6 3, 8 3, 8 1, 6 1, 5 2))",
            ),
            (
                "MULTIPOLYGON (\
                    ((26 125, 26 200, 126 200, 126 125, 26 125 ),\
                    ( 51 150, 101 150, 76 175, 51 150 )),\
                    (( 151 100, 151 200, 176 175, 151 100 ))\
                )",
                "POLYGON ((151 100, 176 175, 151 200, 126 200, 26 200, 26 125, 126 125, 151 100))",
            ),
            (
                "GEOMETRYCOLLECTION (MULTIPOINT((1 1), (3 3)), POINT(5 6), LINESTRING(4 5, 5 6))",
                "POLYGON ((3 3, 1 1, 4 5, 5 6, 3 3))",
            ),
            (
                "GEOMETRYCOLLECTION (MULTIPOINT((1 1), (3 3)), POINT EMPTY, LINESTRING(4 5, 5 6))",
                "POLYGON ((3 3, 1 1, 4 5, 5 6, 3 3))",
            ),
            (
                "GEOMETRYCOLLECTION ( LINESTRING(1 1,2 2),\
                    GEOMETRYCOLLECTION( POLYGON((3 3,4 4,5 5,3 3)),\
                        GEOMETRYCOLLECTION (LINESTRING(6 6,7 7), POLYGON((8 8,9 9,10 10,8 8)))\
                    )\
                )",
                "POLYGON ((10 10, 1 1, 3 3, 3 3, 4 4, 5 5, 8 8, 9 9, 10 10))",
            ),
        ];

        for (input_wkt, expected_wkt) in test_cases {
            let result = tester.invoke_scalar_scalar(input_wkt, 0.1).unwrap();
            assert_scalar_equal_wkb_geometry_topologically(&result, Some(expected_wkt));
        }
    }

    /// For testing here, the Python method was adopted as the `assert_array_equal`
    /// doesn't assert `topologically`
    #[rstest]
    fn test_different_parameters_fixed_geometry(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let tester = create_tester(sedona_type);

        // =========================================================================
        // Identical Geometry Types, Differing Percentages
        // =========================================================================
        let test_cases = vec![
            // LINESTRING with different pctconvex values
            (
                "LINESTRING (100 150, 50 60, 70 80, 160 170)",
                0.1,
                "POLYGON ((70 80, 50 60, 100 150, 160 170, 70 80))",
            ),
            (
                "LINESTRING (100 150, 50 60, 70 80, 160 170)",
                0.3,
                "POLYGON ((70 80, 50 60, 100 150, 160 170, 70 80))",
            ),
            // Test POLYGON with different pctconvex values
            (
                "POLYGON ((70 80, 50 60, 100 150, 160 170, 70 80))",
                0.1,
                "POLYGON ((70 80, 50 60, 100 150, 160 170, 70 80))",
            ),
            (
                "POLYGON ((70 80, 50 60, 100 150, 160 170, 70 80))",
                0.2,
                "POLYGON ((70 80, 50 60, 100 150, 160 170, 70 80))",
            ),
            // Test MULTIPOINT with different pctconvex values
            (
                "MULTIPOINT ((100 150), (160 170))",
                0.1,
                "POLYGON ((100 150, 160 170, 100 150))",
            ),
            (
                "MULTIPOINT ((100 150), (160 170))",
                0.2,
                "POLYGON ((100 150, 160 170, 100 150))",
            ),
            // Test MULTILINESTRING with different pctconvex values
            (
                "MULTILINESTRING ((50 150, 50 200), (50 50, 50 100))",
                0.1,
                "POLYGON ((50 200, 50 50, 50 100, 50 150, 50 200))",
            ),
            (
                "MULTILINESTRING ((50 150, 50 200), (50 50, 50 100))",
                0.2,
                "POLYGON ((50 200, 50 50, 50 100, 50 150, 50 200))",
            ),
            // Test MULTIPOLYGON with different pctconvex values
            (
                "MULTIPOLYGON (((26 125, 26 200, 126 200, 126 125, 26 125 ),\
                    ( 51 150, 101 150, 76 175, 51 150 )), (( 151 100, 151 200, 176 175, 151 100 )))",
                0.1,
                "POLYGON ((151 100, 176 175, 151 200, 126 200, 26 200, 26 125, 126 125, 151 100))"
            ),
            (
                "MULTIPOLYGON (((26 125, 26 200, 126 200, 126 125, 26 125 ),\
                    ( 51 150, 101 150, 76 175, 51 150 )), (( 151 100, 151 200, 176 175, 151 100 )))",
                0.4,
                "POLYGON((151 100,176 175,151 200,26 200,26 125,151 100))"
            ),
            // Test GEOMETRYCOLLECTION with different pctconvex values
            (
                "GEOMETRYCOLLECTION(LINESTRING(1 1,2 2), \
                    GEOMETRYCOLLECTION(POLYGON((3 3,4 4,5 5,3 3)), \
                    GEOMETRYCOLLECTION(LINESTRING(6 6,7 7), POLYGON((8 8,9 9,10 10,8 8)))))",
                0.1,
                "POLYGON ((10 10, 1 1, 3 3, 3 3, 4 4, 5 5, 8 8, 9 9, 10 10))"
            ),
            (
                "GEOMETRYCOLLECTION(LINESTRING(1 1,2 2), \
                    GEOMETRYCOLLECTION(POLYGON((3 3,4 4,5 5,3 3)), \
                    GEOMETRYCOLLECTION(LINESTRING(6 6,7 7), POLYGON((8 8,9 9,10 10,8 8)))))",
                0.6,
                "POLYGON ((10 10, 1 1, 3 3, 3 3, 4 4, 5 5, 8 8, 9 9, 10 10))"
            ),
        ];

        for (wkt, pctconvex, expected) in test_cases {
            let result = tester.invoke_scalar_scalar(wkt, pctconvex).unwrap();
            assert_scalar_equal_wkb_geometry_topologically(&result, Some(expected));
        }
    }
}
