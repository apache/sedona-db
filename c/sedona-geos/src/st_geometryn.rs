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
use datafusion_common::{cast::as_int64_array, DataFusionError, Result};
use datafusion_expr::ColumnarValue;
use geo_traits::{
    GeometryCollectionTrait, GeometryTrait, MultiLineStringTrait, MultiPointTrait,
    MultiPolygonTrait,
};
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_functions::executor::WkbExecutor;
use sedona_geometry::wkb_factory::WKB_MIN_PROBABLE_BYTES;
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};
use wkb::reader::Wkb;

/// ST_GeometryN() implementation using geo-traits
pub fn st_geometryn_impl() -> ScalarKernelRef {
    Arc::new(STGeometryN {})
}

#[derive(Debug)]
struct STGeometryN {}

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
                    if invoke_scalar(&wkb, (index - 1) as usize, &mut builder).is_err() {
                        // Unsupported Geometry Type, Invalid index encountered
                        builder.append_null();
                    } else {
                        builder.append_value([]);
                    }
                }
                _ => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(geom: &Wkb, index: usize, writer: &mut impl std::io::Write) -> Result<()> {
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

    let buf = geometry
        .ok_or_else(|| DataFusionError::Execution(format!("Invalid index specified: {index}")))?;

    writer.write_all(buf)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::WKB_VIEW_GEOMETRY;
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        use sedona_testing::{compare::assert_array_equal, create::create_array};

        let udf = SedonaScalarUDF::from_kernel("st_geometryn", st_geometryn_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![
                sedona_type.clone(),
                SedonaType::Arrow(arrow_schema::DataType::Int64),
            ],
        );
        tester.assert_return_type(WKB_GEOMETRY);

        let input_wkt = create_array(
         &[
            Some("GEOMETRYCOLLECTION(POINT(1 1),MULTIPOLYGON(((0 2,1 1,0 0,0 2)),((2 0,1 1,2 2,2 0))))"),
            Some("MULTIPOLYGON(((26 125, 26 200, 126 200, 126 125, 26 125 ),( 51 150, 101 150, 76 175, 51 150 )),(( 151 100, 151 200, 176 175, 151 100 )))"),
            Some("MULTILINESTRING((1 2, 3 4), (4 5, 6 7), (8 9, 10 11))"),
            Some("MULTIPOINT((1 1), (2 2), (5 5), (6 6))"),
            Some("MULTIPOINT((1 1), (2 2))"),
            Some("MULTIPOLYGON(((1 1, 2 2, 1 2, 1 1)))"),
            Some("POINT(10 10)"),
            Some("POINT(20 20)"),
            Some("GEOMETRYCOLLECTION(POINT(1 1), POINT(2 2))"),
            Some("GEOMETRYCOLLECTION EMPTY"),
            None,
            Some("MULTIPOINT((0 0))"),
         ],
         &WKB_GEOMETRY,
         );

        let integers = arrow_array::create_array!(
            Int64,
            [
                Some(1),
                Some(2),
                Some(2),
                Some(3),
                Some(3),
                Some(0),
                Some(1),
                Some(2),
                Some(3),
                Some(1),
                Some(1),
                None
            ]
        );

        let expected = create_array(
            &[
                Some("POINT(1 1)"),
                Some("POLYGON((151 100,151 200,176 175,151 100))"),
                Some("LINESTRING(4 5,6 7)"),
                Some("POINT(5 5)"),
                None,
                None,
                Some("POINT(10 10)"),
                None,
                None,
                None,
                None,
                None,
            ],
            &WKB_GEOMETRY,
        );

        let nog = &tester
            .invoke_arrays(vec![input_wkt.clone(), integers.clone()])
            .unwrap();
        println!("{:?}", nog);

        assert_array_equal(
            &tester.invoke_arrays(vec![input_wkt, integers]).unwrap(),
            &expected,
        );
    }
}
