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

use crate::executor::WkbExecutor;
use arrow_array::builder::BinaryBuilder;
use datafusion_common::{cast::as_int64_array, Result};
use datafusion_expr::scalar_doc_sections::DOC_SECTION_OTHER;
use datafusion_expr::ColumnarValue;
use datafusion_expr::Documentation;
use datafusion_expr::Volatility;
use geo_traits::{
    GeometryCollectionTrait, GeometryTrait, MultiLineStringTrait, MultiPointTrait,
    MultiPolygonTrait,
};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::wkb_factory::WKB_MIN_PROBABLE_BYTES;
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};
use wkb::reader::Wkb;

/// ST_GeometryN() scalar UDF
///
/// Native implementation to get the nth Geometry in a Collection
pub fn st_geometryn_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_geometryn",
        vec![Arc::new(STGeometryN)],
        Volatility::Immutable,
        Some(st_geometryn_doc()),
    )
}

fn st_geometryn_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Return the 1-based Nth element geometry of an input geometry which is a GEOMETRYCOLLECTION, MULTIPOINT, MULTILINESTRING, MULTICURVE, MULTI)POLYGON, or POLYHEDRALSURFACE. Otherwise, returns NULL.",
        "ST_GeometryN (geom: Geometry, n: integer)",
    )
    .with_argument("geom", "geometry: Input geometry")
    .with_argument("n", "n: Index")
    .with_sql_example("SELECT ST_GeometryN('GEOMETRYCOLLECTION(POINT(10 10), LINESTRING(20 20, 30 30), POLYGON((1 1, 2 2, 1 2, 1 1)))', 1)")
    .build()
}

#[derive(Debug)]
struct STGeometryN;

impl SedonaScalarKernel for STGeometryN {
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
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
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
        geo_traits::GeometryType::GeometryCollection(collection) => {
            collection.geometry(index).map(|item| item.buf())
        }
        geo_traits::GeometryType::MultiLineString(mul_ls) => {
            mul_ls.line_string(index).map(|ls| ls.buf())
        }
        geo_traits::GeometryType::MultiPolygon(mul_pgn) => {
            mul_pgn.polygon(index).map(|pgn| pgn.buf())
        }
        geo_traits::GeometryType::MultiPoint(mul_pt) => mul_pt.point(index).map(|pt| pt.buf()),
        // PostGIS returns `Self` for Simple Geometries
        _ if index == 0 => Some(geom.buf()),
        _ => None,
    };

    if let Some(buf) = geometry {
        writer.write_all(buf)?;
        Ok(true)
    } else {
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use sedona_schema::datatypes::WKB_VIEW_GEOMETRY;
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        use sedona_testing::{compare::assert_array_equal, create::create_array};

        let tester = ScalarUdfTester::new(
            st_geometryn_udf().into(),
            vec![
                sedona_type.clone(),
                SedonaType::Arrow(arrow_schema::DataType::Int64),
            ],
        );
        tester.assert_return_type(WKB_GEOMETRY);

        let input_wkt = create_array(
            &[
                // 1. POINT
                Some("POINT(1 1)"), //  n=1 (Valid)
                Some("POINT(1 1)"), //  n=2 (OOB)
                Some("POINT(1 1)"), //  n=99 (Large OOB)
                // 2. LINESTRING
                Some("LINESTRING(2 2, 3 3, 4 4)"), //  n=1 (Valid)
                None,                              //  Null input (n=2)
                Some("LINESTRING(2 2, 3 3, 4 4)"), //  n=0 (OOB)
                // 3. POLYGON
                Some("POLYGON((0 0, 1 0, 1 1, 0 0))"), //  n=1 (Valid)
                Some("POLYGON((0 0, 1 0, 1 1, 0 0))"), //  n=3 (OOB)
                // 4. MULTIPOINT
                Some("MULTIPOINT((1 1), (2 2), (3 3))"), //  n=2 (Valid) - Original
                Some("MULTIPOINT((1 1), (2 2), (3 3))"), //  n=3 (Valid) - Original
                None,                                    //  Null Input (n=0) - Original
                Some("MULTIPOINT((1 1), (2 2), (3 3))"), //  n=1 (Valid)
                Some("MULTIPOINT((1 1), (2 2), (3 3))"), //  n=0 (OOB)
                // 5. MULTILINESTRING
                Some("MULTILINESTRING((1 1, 2 2), (3 3, 4 4))"), //  n=1 (Valid) - Original
                Some("MULTILINESTRING((1 1, 2 2), (3 3, 4 4))"), //  n=3 (OOB) - Original
                Some("MULTILINESTRING((1 1, 2 2), (3 3, 4 4))"), //  n=2 (Valid)
                // 6. MULTIPOLYGON
                Some("MULTIPOLYGON(((0 0, 1 1, 0 1, 0 0)), ((5 5, 6 6, 5 6, 5 5)))"), //   n=2 (Valid) - Original
                Some("MULTIPOLYGON(((0 0, 1 1, 0 1, 0 0)))"),                          //  n=2 (OOB) - Original
                Some("MULTIPOLYGON(((0 0, 1 1, 0 1, 0 0)), ((5 5, 6 6, 5 6, 5 5)))"), //  n=1 (Valid)
                Some("MULTIPOLYGON EMPTY"),                                            //  Empty Multi (n=1)
                // 7. GEOMETRYCOLLECTION (7 cases)
                Some("GEOMETRYCOLLECTION(POINT(10 10), LINESTRING(20 20, 30 30), POLYGON((1 1, 2 2, 1 2, 1 1)))"), //  n=1 (Point) - Original
                Some("GEOMETRYCOLLECTION(POINT(10 10), LINESTRING(20 20, 30 30), POLYGON((1 1, 2 2, 1 2, 1 1)))"), //  n=2 (LineString) - Original
                Some("GEOMETRYCOLLECTION(POINT(10 10))"), //  n=2 (OOB) - Original
                Some("GEOMETRYCOLLECTION(POINT(1 1), GEOMETRYCOLLECTION(LINESTRING(2 2, 3 3)))"), //  n=1 (Nested: Point)
                Some("GEOMETRYCOLLECTION(POINT(1 1), GEOMETRYCOLLECTION(LINESTRING(2 2, 3 3)))"), //  n=2 (Nested: GC)
                Some("GEOMETRYCOLLECTION(POINT(1 1))"), //  n=0 (OOB)
            ],
            &WKB_GEOMETRY,
        );

        let integers = arrow_array::create_array!(
            Int64,
            [
                // 1. POINT
                Some(1),  // n=1
                Some(2),  // n=2 (OOB)
                Some(99), //  n=99 (OOB)
                // 2. LINESTRING
                Some(1), //  n=1
                Some(2), //  Null input
                Some(0), //  n=0 (OOB)
                // 3. POLYGON
                Some(1), //  n=1
                Some(3), //  n=3 (OOB)
                // 4. MULTIPOINT
                Some(2), //  n=2
                Some(3), //  n=3
                Some(0), //  n=0 (Null input)
                Some(1), //  n=1
                Some(0), //  n=0 (OOB)
                // 5. MULTILINESTRING
                Some(1), //  n=1
                Some(3), //  n=3 (OOB)
                Some(2), //  n=2
                // 6. MULTIPOLYGON
                Some(2), //  n=2
                Some(2), //  n=2 (OOB)
                Some(1), //  n=1
                Some(1), //  n=1 (Empty)
                // 7. GEOMETRYCOLLECTION
                Some(1), //  n=1 (Point)
                Some(2), //  n=2 (LineString)
                Some(2), //  n=2 (OOB)
                Some(1), //  n=1 (Nested: Point)
                Some(2), //  n=2 (Nested: GC)
                Some(0)  //  n=0 (OOB)
            ]
        );

        let expected = create_array(
            &[
                // 1. POINT
                Some("POINT(1 1)"),
                None,
                None,
                // 2. LINESTRING
                Some("LINESTRING(2 2, 3 3, 4 4)"),
                None,
                None,
                // 3. POLYGON
                Some("POLYGON((0 0, 1 0, 1 1, 0 0))"),
                None,
                // 4. MULTIPOINT
                Some("POINT(2 2)"),
                Some("POINT(3 3)"),
                None,
                Some("POINT(1 1)"),
                None,
                // 5. MULTILINESTRING
                Some("LINESTRING(1 1, 2 2)"),
                None,
                Some("LINESTRING(3 3, 4 4)"),
                // 6. MULTIPOLYGON
                Some("POLYGON((5 5, 6 6, 5 6, 5 5))"),
                None,
                Some("POLYGON((0 0, 1 1, 0 1, 0 0))"),
                None,
                // 7. GEOMETRYCOLLECTION
                Some("POINT(10 10)"),
                Some("LINESTRING(20 20, 30 30)"),
                None,
                Some("POINT(1 1)"),
                Some("GEOMETRYCOLLECTION(LINESTRING(2 2, 3 3))"), // The WKB of the GC component itself
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
