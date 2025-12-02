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
use datafusion_common::cast::as_int64_array;
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::{scalar_doc_sections::DOC_SECTION_OTHER, Documentation};
use geo_traits::{GeometryTrait, LineStringTrait, PolygonTrait};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::wkb_factory::{
    write_wkb_coord_trait, write_wkb_linestring_header, WKB_MIN_PROBABLE_BYTES,
};
use sedona_schema::datatypes::SedonaType;
use sedona_schema::{datatypes::WKB_GEOMETRY, matchers::ArgMatcher};
use wkb::reader::Wkb;

use crate::executor::WkbExecutor;

/// ST_InteriorRingN() scalar UDF
///
/// Native implementation to get the nth interior ring (hole) of a Polygon
pub fn st_interiorringn_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_interiorringn",
        vec![Arc::new(STInteriorRingN)],
        datafusion_expr::Volatility::Immutable,
        Some(st_interiorringn_doc()),
    )
}

fn st_interiorringn_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the Nth interior ring (hole) of a POLYGON geometry as a LINESTRING. \
        The index starts at 1. Returns NULL if the geometry is not a polygon or the index is out of range.",
        "ST_GeometryN (geom: Geometry, n: integer)")
    .with_argument("geom", "geometry: Input Polygon")
    .with_argument("n", "n: Index")
    .with_sql_example("SELECT ST_InteriorRingN('POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 1 2, 2 2, 2 1, 1 1))', 1)")
    .build()
}

#[derive(Debug)]
struct STInteriorRingN;

impl SedonaScalarKernel for STInteriorRingN {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry(), ArgMatcher::is_integer()],
            WKB_GEOMETRY,
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[datafusion_expr::ColumnarValue],
    ) -> Result<datafusion_expr::ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
        );

        let integer_value = args[1]
            .cast_to(&arrow_schema::DataType::Int64, None)?
            .to_array(executor.num_iterations())?;
        let index_array = as_int64_array(&integer_value)?;
        let mut index_iter = index_array.iter();

        executor.execute_wkb_void(|maybe_wkb| {
            match (maybe_wkb, index_iter.next().unwrap()) {
                (Some(wkb), Some(index)) => {
                    if invoke_scalar(&wkb, (index - 1) as usize, &mut builder)? {
                        builder.append_value([]);
                    } else {
                        // Unsupported Geometry Type, Invalid index encountered
                        builder.append_null();
                    }
                }
                _ => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(geom: &Wkb, index: usize, writer: &mut impl std::io::Write) -> Result<bool> {
    let geometry = match geom.as_type() {
        geo_traits::GeometryType::Polygon(pgn) => pgn.interior(index),
        _ => None,
    };

    if let Some(wkb) = geometry {
        write_wkb_linestring_header(writer, wkb.dim(), wkb.num_coords())
            .map_err(|e| DataFusionError::Execution(e.to_string()))?;
        wkb.coords().try_for_each(|coord| {
            write_wkb_coord_trait(writer, &coord)
                .map_err(|e| DataFusionError::Execution(e.to_string()))
        })?;
        Ok(true)
    } else {
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use sedona_schema::datatypes::WKB_VIEW_GEOMETRY;
    use sedona_testing::{
        compare::assert_array_equal, create::create_array, testers::ScalarUdfTester,
    };

    use super::*;

    fn setup_tester(sedona_type: SedonaType) -> ScalarUdfTester {
        let tester = ScalarUdfTester::new(
            st_interiorringn_udf().into(),
            vec![
                sedona_type,
                SedonaType::Arrow(arrow_schema::DataType::Int64),
            ],
        );
        tester.assert_return_type(WKB_GEOMETRY);
        tester
    }

    // 1. Tests for Non-Polygon Geometries (Should return NULL)
    #[rstest]
    fn test_st_interiorringn_non_polygons(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let tester = setup_tester(sedona_type);

        let input_wkt = create_array(
            &[
                None,                                               // NULL input
                Some("POINT (0 0)"),                                // POINT
                Some("POINT EMPTY"),                                // POINT EMPTY
                Some("LINESTRING (0 0, 0 1, 1 2)"),                 // LINESTRING
                Some("LINESTRING EMPTY"),                           // LINESTRING EMPTY
                Some("MULTIPOINT ((0 0), (1 1))"),                  // MULTIPOINT
                Some("MULTIPOLYGON (((1 1, 1 3, 3 3, 3 1, 1 1)))"), // MULTIPOLYGON
                Some("GEOMETRYCOLLECTION (POINT(1 1))"),            // GEOMETRYCOLLECTION
            ],
            &WKB_GEOMETRY,
        );
        let integers = arrow_array::create_array!(
            Int64,
            [
                Some(1),
                Some(1),
                Some(1),
                Some(1),
                Some(1),
                Some(1),
                Some(1),
                Some(1)
            ]
        );
        let expected = create_array(
            &[None, None, None, None, None, None, None, None],
            &WKB_GEOMETRY,
        );

        assert_array_equal(
            &tester.invoke_arrays(vec![input_wkt, integers]).unwrap(),
            &expected,
        );
    }

    // 2. Tests for Polygon Edge Cases (No holes, Invalid index)
    #[rstest]
    fn test_st_interiorringn_polygon_edge_cases(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let tester = setup_tester(sedona_type);

        let input_wkt = create_array(
            &[
                Some("POLYGON EMPTY"),                       // POLYGON EMPTY
                Some("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"), // Polygon with NO interior rings (n=1)
                Some("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"), // Invalid index n=0
                Some("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"), // Index n too high (n=2)
            ],
            &WKB_GEOMETRY,
        );
        let integers = arrow_array::create_array!(Int64, [Some(1), Some(1), Some(0), Some(2)]);
        let expected = create_array(
            &[
                None, // POLYGON EMPTY
                None, // Polygon with NO interior rings
                None, // Invalid index n=0 (Assuming NULL/None on invalid index)
                None, // Index n too high
            ],
            &WKB_GEOMETRY,
        );

        assert_array_equal(
            &tester.invoke_arrays(vec![input_wkt, integers]).unwrap(),
            &expected,
        );
    }

    // 3. Tests for Valid Polygons (Correct Extraction)
    #[rstest]
    fn test_st_interiorringn_valid_polygons(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let tester = setup_tester(sedona_type);

        let input_wkt = create_array(
            &[
                Some("POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 1 2, 2 2, 2 1, 1 1))"),                                  // Single hole, n=1
                Some("POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 1 2, 2 2, 2 1, 1 1))"),                                  // Single hole, n=1
                Some("POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 1 2, 2 2, 2 1, 1 1))"),                                  // Single hole, n=2 (too high)
                Some("POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0), (1 1, 1 2, 2 2, 2 1, 1 1), (4 4, 4 5, 5 5, 5 4, 4 4))"),       // Two holes, n=1
                Some("POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0), (1 1, 1 2, 2 2, 2 1, 1 1), (4 4, 4 5, 5 5, 5 4, 4 4))"),       // Two holes, n=2
                Some("POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0), (1 1, 1 2, 2 2, 2 1, 1 1), (4 4, 4 5, 5 5, 5 4, 4 4))"),       // Two holes, n=3 (too high)
            ],
            &WKB_GEOMETRY,
        );
        let integers = arrow_array::create_array!(
            Int64,
            [Some(1), Some(-1), Some(2), Some(1), Some(2), Some(3)]
        );
        let expected = create_array(
            &[
                Some("LINESTRING (1 1, 1 2, 2 2, 2 1, 1 1)"),
                None,
                None,
                Some("LINESTRING (1 1, 1 2, 2 2, 2 1, 1 1)"),
                Some("LINESTRING (4 4, 4 5, 5 5, 5 4, 4 4)"),
                None,
            ],
            &WKB_GEOMETRY,
        );

        assert_array_equal(
            &tester.invoke_arrays(vec![input_wkt, integers]).unwrap(),
            &expected,
        );
    }

    // 4. Tests for Invalid/Malformed Polygons (Checking for error/extraction)
    #[rstest]
    fn test_st_interiorringn_invalid_polygons(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let tester = setup_tester(sedona_type);

        let input_wkt = create_array(
            &[
                Some("POLYGON ((0 0, 1 0, 1 1))"),                                                            // Unclosed/Malformed WKT
                Some("POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (5 5, 5 6, 6 6, 6 5, 5 5))"),                       // External hole
                Some("POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 1 3, 3 3, 3 1, 1 1), (2 2, 2 2.5, 2.5 2.5, 2.5 2, 2 2))"), // Intersecting holes
            ],
            &WKB_GEOMETRY,
        );
        let integers = arrow_array::create_array!(Int64, [Some(1), Some(1), Some(2)]);
        let expected = create_array(
            &[
                None, // parsing/validation returns None/NULL for invalid geometry (Unclosed)
                Some("LINESTRING (5 5, 5 6, 6 6, 6 5, 5 5)"), // Extraction works even if topologically invalid (external)
                Some("LINESTRING (2 2, 2 2.5, 2.5 2.5, 2.5 2, 2 2)"), // Extraction works even if topologically invalid (intersecting)
            ],
            &WKB_GEOMETRY,
        );

        assert_array_equal(
            &tester.invoke_arrays(vec![input_wkt, integers]).unwrap(),
            &expected,
        );
    }

    #[rstest]
    fn test_st_interiorringn_z_dimensions(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let tester = setup_tester(sedona_type);

        let input_wkt = create_array(
            &[
                // Valid Polygon Z extraction
                Some("POLYGON Z ((0 0 10, 4 0 10, 4 4 10, 0 4 10, 0 0 10), (1 1 5, 1 2 5, 2 2 5, 2 1 5, 1 1 5))"),
                // Non-Polygon Z (Should be NULL)
                Some("POINT Z (1 1 5)"),
                // Polygon Z with no hole (Should be NULL)
                Some("POLYGON Z ((0 0 10, 4 0 10, 4 4 10, 0 4 10, 0 0 10))"),
            ],
            &WKB_GEOMETRY
        );
        let integers = arrow_array::create_array!(Int64, [Some(1), Some(1), Some(1)]);
        let expected = create_array(
            &[
                Some("LINESTRING Z (1 1 5, 1 2 5, 2 2 5, 2 1 5, 1 1 5)"),
                None,
                None,
            ],
            &WKB_GEOMETRY,
        );

        assert_array_equal(
            &tester.invoke_arrays(vec![input_wkt, integers]).unwrap(),
            &expected,
        );
    }

    #[rstest]
    fn test_st_interiorringn_m_dimensions(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let tester = setup_tester(sedona_type);

        let input_wkt = create_array(
            &[
                // Valid Polygon M extraction
                Some("POLYGON M ((0 0 1, 4 0 2, 4 4 3, 0 4 4, 0 0 5), (1 1 6, 1 2 7, 2 2 8, 2 1 9, 1 1 10))"),
                // Non-Polygon M (Should be NULL)
                Some("LINESTRING M (0 0 1, 1 1 2)"),
                // Polygon M with no hole (Should be NULL)
                Some("POLYGON M ((0 0 1, 4 0 2, 4 4 3, 0 4 4, 0 0 5))"),
            ],
            &WKB_GEOMETRY
        );
        let integers = arrow_array::create_array!(Int64, [Some(1), Some(1), Some(1)]);
        let expected = create_array(
            &[
                Some("LINESTRING M (1 1 6, 1 2 7, 2 2 8, 2 1 9, 1 1 10)"),
                None,
                None,
            ],
            &WKB_GEOMETRY,
        );

        assert_array_equal(
            &tester.invoke_arrays(vec![input_wkt, integers]).unwrap(),
            &expected,
        );
    }

    #[rstest]
    fn test_st_interiorringn_zm_dimensions(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let tester = setup_tester(sedona_type);

        let input_wkt = create_array(
            &[
                // Valid Polygon ZM extraction (n=1)
                Some("POLYGON ZM ((0 0 10 1, 4 0 10 2, 4 4 10 3, 0 4 10 4, 0 0 10 5), (1 1 5 6, 1 2 5 7, 2 2 5 8, 2 1 5 9, 1 1 5 10))"),
                // Index too high (n=2)
                Some("POLYGON ZM ((0 0 10 1, 4 0 10 2, 4 4 10 3, 0 4 10 4, 0 0 10 5), (1 1 5 6, 1 2 5 7, 2 2 5 8, 2 1 5 9, 1 1 5 10))"),
                // POLYGON ZM EMPTY (Should be NULL)
                Some("POLYGON ZM EMPTY"),
            ],
            &WKB_GEOMETRY
        );
        let integers = arrow_array::create_array!(Int64, [Some(1), Some(2), Some(1)]);
        let expected = create_array(
            &[
                Some("LINESTRING ZM (1 1 5 6, 1 2 5 7, 2 2 5 8, 2 1 5 9, 1 1 5 10)"),
                None,
                None,
            ],
            &WKB_GEOMETRY,
        );

        assert_array_equal(
            &tester.invoke_arrays(vec![input_wkt, integers]).unwrap(),
            &expected,
        );
    }
}
