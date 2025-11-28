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
use geo_traits::to_geo::ToGeoLineString;
use geo_traits::{GeometryTrait, PolygonTrait};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::wkb_factory::{write_wkb_linestring, WKB_MIN_PROBABLE_BYTES};
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

    if let Some(buf) = geometry {
        write_wkb_linestring(
            writer,
            buf.to_line_string()
                .coords()
                .map(|c| c.x_y())
                .collect::<Vec<(f64, f64)>>()
                .into_iter(),
        )
        .map_err(|e| DataFusionError::Execution(format!("{e}")))?;
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

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(
            st_interiorringn_udf().into(),
            vec![
                sedona_type.clone(),
                SedonaType::Arrow(arrow_schema::DataType::Int64),
            ],
        );
        tester.assert_return_type(WKB_GEOMETRY);

        let input_wkt = create_array(
            &[
                // I. Null/Empty/Non-Polygon Inputs
                None,                                               // NULL
                Some("POINT (0 0)"),                                // POINT
                Some("POINT EMPTY"),                                // POINT EMPTY
                Some("LINESTRING (0 0, 0 1, 1 2)"),                 // LINESTRING
                Some("LINESTRING EMPTY"),                           // LINESTRING EMPTY
                Some("MULTIPOINT ((0 0), (1 1))"),                  // MULTIPOINT
                Some("MULTIPOLYGON (((1 1, 1 3, 3 3, 3 1, 1 1)))"), // MULTIPOLYGON
                Some("GEOMETRYCOLLECTION (POINT(1 1))"),            // GEOMETRYCOLLECTION
                // II. Polygon Edge Cases
                Some("POLYGON EMPTY"),                       // POLYGON EMPTY
                Some("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"), //  Polygon with NO interior rings
                Some("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"), // Invalid index n=0
                Some("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"), // Index n too high
                // III. Valid Polygon with Interior Ring(s)
                Some("POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 1 2, 2 2, 2 1, 1 1))"), // Single hole, n=1
                Some("POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 1 2, 2 2, 2 1, 1 1))"), // Single hole, n=2
                Some("POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0), (1 1, 1 2, 2 2, 2 1, 1 1), (4 4, 4 5, 5 5, 5 4, 4 4))"), // Two holes, n=1
                Some("POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0), (1 1, 1 2, 2 2, 2 1, 1 1), (4 4, 4 5, 5 5, 5 4, 4 4))"), // Two holes, n=2
                Some("POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0), (1 1, 1 2, 2 2, 2 1, 1 1), (4 4, 4 5, 5 5, 5 4, 4 4))"), // Two holes, n=3

                // // IV. Invalid/Malformed Polygon Input
                Some("POLYGON ((0 0, 1 0, 1 1))"), // Unclosed/Malformed WKT (invalid polygon geometry)
                Some("POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (5 5, 5 6, 6 6, 6 5, 5 5))"), // External hole
                Some("POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 1 3, 3 3, 3 1, 1 1), (2 2, 2 2.5, 2.5 2.5, 2.5 2, 2 2))"), //  Intersecting holes
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
                Some(1), // I.
                Some(1),
                Some(1),
                Some(0),
                Some(2), // II.
                Some(1),
                Some(2),
                Some(1),
                Some(2),
                Some(3),
                // III.
                Some(1),
                Some(1),
                Some(2) // IV.
            ]
        );

        let expected = create_array(
            &[
                // I. Null/Empty/Non-Polygon Inputs
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                // II. Polygon Edge Cases
                None, // POLYGON EMPTY
                None, // Polygon with NO interior rings
                None, // Invalid index n=0 (Assuming NULL/None on invalid index)
                None, // Index n too high
                // III. Valid Polygon with Interior Ring(s)
                Some("LINESTRING (1 1, 1 2, 2 2, 2 1, 1 1)"),
                None,
                Some("LINESTRING (1 1, 1 2, 2 2, 2 1, 1 1)"),
                Some("LINESTRING (4 4, 4 5, 5 5, 5 4, 4 4)"),
                None,
                // IV. Invalid/Malformed Polygon Input
                None, // WKT parsing/validation returns None/NULL for invalid geometry
                Some("LINESTRING(5 5,5 6,6 6,6 5,5 5)"),
                Some("LINESTRING(2 2,2 2.5,2.5 2.5,2.5 2,2 2)"),
            ],
            &WKB_GEOMETRY,
        );

        assert_array_equal(
            &tester.invoke_arrays(vec![input_wkt, integers]).unwrap(),
            &expected,
        );
    }
}
