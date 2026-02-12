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

use crate::executor::WkbExecutor;
use arrow_array::builder::BinaryBuilder;
use arrow_schema::DataType;
use datafusion_common::cast::as_binary_array;
use datafusion_common::ScalarValue;
use datafusion_expr::scalar_doc_sections::DOC_SECTION_OTHER;
use datafusion_expr::{ColumnarValue, Documentation, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::wkb_factory::WKB_MIN_PROBABLE_BYTES;
use sedona_schema::crs::deserialize_crs;
use sedona_schema::datatypes::{Edges, SedonaType, WKB_GEOMETRY};
use sedona_schema::matchers::ArgMatcher;
use sedona_serde::deserialize::deserialize;
use std::sync::Arc;

fn to_crs_str(scalar_arg: &ScalarValue) -> Option<String> {
    if let Ok(ScalarValue::Utf8(Some(crs))) = scalar_arg.cast_to(&DataType::Utf8) {
        return Some(crs);
    }

    None
}

#[derive(Debug)]
struct STGeomFromSedonaSpark {
    out_type: SedonaType,
}

pub fn st_geomfromsedona_udf() -> SedonaScalarUDF {
    let kernel = Arc::new(STGeomFromSedonaSpark {
        out_type: WKB_GEOMETRY,
    });

    SedonaScalarUDF::new(
        "st_geomfromsedonaspark",
        vec![kernel],
        Volatility::Immutable,
        Some(doc()),
    )
}

impl SedonaScalarKernel for STGeomFromSedonaSpark {
    fn return_type(&self, args: &[SedonaType]) -> datafusion_common::Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_binary(), ArgMatcher::is_string()],
            self.out_type.clone(),
        );

        matcher.match_args(args)
    }

    fn return_type_from_args_and_scalars(
        &self,
        args: &[SedonaType],
        _scalar_args: &[Option<&ScalarValue>],
    ) -> datafusion_common::Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_binary(), ArgMatcher::is_string()],
            self.out_type.clone(),
        );

        if !matcher.matches(args) {
            return Ok(None);
        }

        let crs_scalar = _scalar_args.get(1).unwrap();

        let crs_str_opt = if let Some(scalar_crs) = crs_scalar {
            to_crs_str(scalar_crs)
        } else {
            None
        };

        match crs_str_opt {
            Some(to_crs) => Ok(Some(SedonaType::Wkb(
                Edges::Planar,
                deserialize_crs(&to_crs)?,
            ))),
            _ => Ok(Some(SedonaType::Wkb(Edges::Planar, None))),
        }
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> datafusion_common::Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let arg_array = args[0]
            .cast_to(&DataType::Binary, None)?
            .to_array(executor.num_iterations())?;

        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
        );

        for sedona_bytes in as_binary_array(&arg_array)?.into_iter().flatten() {
            deserialize(&mut builder, sedona_bytes)?;
            builder.append_value(vec![]);
        }

        let new_array = builder.finish();
        executor.finish(Arc::new(new_array))
    }
}

fn doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Internal only, it's function used in the vectorized UDFs to translate Sedona Spark binary format to WKB format.",
        "ST_GeomFromSedonaSpark (geom: binary, crs: string)",
    )
        .with_argument("geom", "sedona spark geometry binary")
        .with_argument("crs", "crs: coordinate reference system")
        .with_sql_example("SELECT ST_GeomFromSedonaSpark(X'1200000001000000000000000000F03F000000000000F03F', 'EPSG:4326')")
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use sedona_testing::testers::ScalarUdfTester;

    fn get_tester() -> ScalarUdfTester {
        ScalarUdfTester::new(
            st_geomfromsedona_udf().into(),
            vec![
                SedonaType::Arrow(DataType::Binary),
                SedonaType::Arrow(DataType::Utf8),
            ],
        )
    }

    fn fixture_to_bytes(wkb: &str) -> Vec<u8> {
        wkb.split("\n")
            .filter(|line| !line.starts_with("//") && !line.is_empty())
            .flat_map(|s| s.split_whitespace())
            .map(|num| num.parse::<u8>().expect("invalid byte"))
            .collect::<Vec<u8>>()
    }

    const POINT_WKT: &str = "POINT (1 1)";
    const LINESTRING_WKT: &str = "LINESTRING (0 0, 1 1, 2 2)";
    const MULTILINESTRING_WKT: &str = "MULTILINESTRING ((1 1, 2 2), (4 5, 6 7))";
    const MULTIPOINT_WKT: &str = "MULTIPOINT ((1 1), (2 2), (4 5))";
    const POLYGON_WKT: &str = "POLYGON (
        (1 1, 10 1, 10 10, 1 10, 1 1),
        (2 2, 4 2, 4 4, 2 4, 2 2),
        (6 6, 8 6, 8 8, 6 8, 6 6)
    )";
    const MULTIPOLYGON_WKT: &str = "MULTIPOLYGON (
        (
            (1 1, 10 1, 10 10, 1 10, 1 1),
            (2 2, 4 2, 4 4, 2 4, 2 2),
            (6 6, 8 6, 8 8, 6 8, 6 6)
        ),
         (
            (12 1, 20 1, 20 9, 12 9, 12 1),
            (13 2, 15 2, 15 4, 13 4, 13 2),
            (17 5, 19 5, 19 7, 17 7, 17 5)
         )
     )";
    const GEOMETRYCOLLECTION_WKT: &str = "GEOMETRYCOLLECTION (
        POINT (4 6),
        LINESTRING (4 6,7 10),
        POLYGON((4 6,7 10,4 10,4 6))
    )";

    const COMPLEX_GEOMETRYCOLLECTION_WKT: &str = "GEOMETRYCOLLECTION(
        POINT(4 6),
        LINESTRING(4 6,7 10),
        POLYGON((4 6,7 10,4 10,4 6)),
        MULTIPOINT((1 2),(3 4))
    )";
    const NESTED_GEOMETRYCOLLECTION_WKT: &str = "GEOMETRYCOLLECTION (
        POINT (1 1),
        GEOMETRYCOLLECTION (
            LINESTRING (0 0, 1 1),
            POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))
        )
     )";

    const FLOATING_POLYGON_WKT: &str = "POLYGON (
        (
            12.345678901234 45.678901234567,
            23.456789012345 67.890123456789,
            34.567890123456 56.789012345678,
            45.678901234567 34.567890123456,
            29.876543210987 22.345678901234,
            12.345678901234 45.678901234567
        ),
        (
            25.123456789012 45.987654321098,
            30.987654321098 50.123456789012,
            35.456789012345 45.456789012345,
            30.234567890123 40.987654321098,
            25.123456789012 45.987654321098
        )
    )";

    #[rstest]
    fn test_geometries_deserialization(
        #[values(
        (POINT_WKT, include_str!("fixtures/point.sedona")),
        (LINESTRING_WKT, include_str!("fixtures/linestring.sedona")),
        (MULTILINESTRING_WKT, include_str!("fixtures/multilinestring.sedona")),
        (MULTIPOINT_WKT, include_str!("fixtures/multipoint.sedona")),
        (POLYGON_WKT, include_str!("fixtures/polygon.sedona")),
        (MULTIPOLYGON_WKT, include_str!("fixtures/multipolygon.sedona")),
        (GEOMETRYCOLLECTION_WKT, include_str!("fixtures/geometrycollection.sedona")),
        (COMPLEX_GEOMETRYCOLLECTION_WKT, include_str!("fixtures/geometrycollectioncomplex.sedona")),
        (NESTED_GEOMETRYCOLLECTION_WKT, include_str!("fixtures/nested_geometry_collection.sedona")),
        ("POINT EMPTY", include_str!("fixtures/empty_point.sedona")),
        ("LINESTRING EMPTY", include_str!("fixtures/empty_linestring.sedona")),
        ("POLYGON EMPTY", include_str!("fixtures/empty_polygon.sedona")),
        ("MULTIPOINT EMPTY", include_str!("fixtures/multipoint_empty.sedona")),
        ("MULTIPOLYGON EMPTY", include_str!("fixtures/empty_multipolygon.sedona")),
        ("MULTILINESTRING EMPTY", include_str!("fixtures/empty_multilinestring.sedona")),
        ("GEOMETRYCOLLECTION EMPTY", include_str!("fixtures/empty_geometry_collection.sedona")),
        (FLOATING_POLYGON_WKT, include_str!("fixtures/point_float_coords.sedona"))
        )]
        value: (&str, &str),
    ) {
        let (expected_wkt, input_bytes) = value;

        let binary_geometry = fixture_to_bytes(input_bytes);
        let tester = get_tester();

        let result = tester
            .invoke_scalar_scalar(binary_geometry, ScalarValue::Utf8(Some("4326".to_string())))
            .unwrap();

        tester.assert_scalar_result_equals(result, expected_wkt);
    }
}
