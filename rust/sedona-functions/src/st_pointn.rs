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
use arrow_array::builder::BinaryBuilder;
use arrow_schema::DataType;
use datafusion_common::{error::Result, ScalarValue};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use geo_traits::{CoordTrait, GeometryTrait, LineStringTrait};
use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::{
    error::SedonaGeometryError,
    wkb_factory::{write_wkb_coord_trait, write_wkb_point_header, WKB_MIN_PROBABLE_BYTES},
};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};
use std::{io::Write, sync::Arc};

use crate::executor::WkbExecutor;

/// ST_PointN() scalar UDF
///
/// Native implementation to get the nth point of a LINESTRING geometry.
pub fn st_pointn_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_pointn",
        vec![Arc::new(STPointN)],
        Volatility::Immutable,
        Some(st_pointn_doc()),
    )
}

fn st_pointn_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the nth point of a geometry. Returns NULL if the geometry is empty or not a LINESTRING. Negative values are counted backwards from the end.",
        "ST_PointN (geom: Geometry, n: integer)",
    )
    .with_argument("geom", "geometry: Input geometry")
    .with_argument("n", "n: Index")
    .with_sql_example("SELECT ST_PointN(ST_GeomFromWKT('LINESTRING(0 1, 2 3, 4 5)'), 2)")
    .build()
}

#[derive(Debug)]
struct STPointN;

impl SedonaScalarKernel for STPointN {
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

        let maybe_n: Option<i64> = match &args[1].cast_to(&DataType::Int64, None)? {
            ColumnarValue::Scalar(ScalarValue::Int64(maybe_n)) => *maybe_n,
            _ => None, // pass invalid n value so that all
        };

        executor.execute_wkb_void(|maybe_wkb| {
            let n = match maybe_n {
                // n is 1-origin, so 0 is invalid value
                Some(n) if n != 0 => n,
                _ => {
                    builder.append_null();
                    return Ok(());
                }
            };

            if let Some(wkb) = maybe_wkb {
                if let geo_traits::GeometryType::LineString(line_string) = wkb.as_type() {
                    let num_coords = line_string.num_coords() as i64;

                    // if n is out of the range, return NULL
                    if n.abs() > num_coords {
                        builder.append_null();
                        return Ok(());
                    }

                    // Negative values are counted backwards from the end
                    let n = if n > 0 { n - 1 } else { num_coords + n } as usize;

                    if let Some(coord) = line_string.coord(n) {
                        if write_wkb_point_from_coord(&mut builder, coord).is_err() {
                            return sedona_internal_err!("Failed to write WKB point");
                        };
                        builder.append_value([]);
                        return Ok(());
                    }
                }
            }

            builder.append_null();
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn write_wkb_point_from_coord(
    buf: &mut impl Write,
    coord: impl CoordTrait<T = f64>,
) -> Result<(), SedonaGeometryError> {
    write_wkb_point_header(buf, coord.dim())?;
    write_wkb_coord_trait(buf, &coord)
}

#[cfg(test)]
mod tests {
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::WKB_VIEW_GEOMETRY;
    use sedona_testing::{
        compare::assert_array_equal, create::create_array, testers::ScalarUdfTester,
    };

    use super::*;

    #[test]
    fn udf_metadata() {
        let st_pointn_udf: ScalarUDF = st_pointn_udf().into();
        assert_eq!(st_pointn_udf.name(), "st_pointn");
        assert!(st_pointn_udf.documentation().is_some());
    }

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester_pointn = ScalarUdfTester::new(
            st_pointn_udf().into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Int64)],
        );

        // valid cases
        let input_linestrings = create_array(
            &[
                Some("LINESTRING (11 12, 21 22, 31 32, 41 42)"),
                Some("LINESTRING Z (11 12 13, 21 22 23, 31 32 33, 41 42 43)"),
                Some("LINESTRING M (11 12 13, 21 22 23, 31 32 33, 41 42 43)"),
                Some("LINESTRING ZM (11 12 13 14, 21 22 23 24, 31 32 33 34, 41 42 43 44)"),
            ],
            &sedona_type,
        );

        // first points
        let expected1 = create_array(
            &[
                Some("POINT (11 12)"),
                Some("POINT Z (11 12 13)"),
                Some("POINT M (11 12 13)"),
                Some("POINT ZM (11 12 13 14)"),
            ],
            &WKB_GEOMETRY,
        );

        let result1 = tester_pointn
            .invoke_array_scalar(input_linestrings.clone(), ScalarValue::Int64(Some(1)))
            .unwrap();
        assert_array_equal(&result1, &expected1);

        // second points
        let expected2 = create_array(
            &[
                Some("POINT (21 22)"),
                Some("POINT Z (21 22 23)"),
                Some("POINT M (21 22 23)"),
                Some("POINT ZM (21 22 23 24)"),
            ],
            &WKB_GEOMETRY,
        );

        let result2 = tester_pointn
            .invoke_array_scalar(input_linestrings.clone(), ScalarValue::Int64(Some(2)))
            .unwrap();
        assert_array_equal(&result2, &expected2);

        // second points from tail
        let expected2_tail = create_array(
            &[
                Some("POINT (31 32)"),
                Some("POINT Z (31 32 33)"),
                Some("POINT M (31 32 33)"),
                Some("POINT ZM (31 32 33 34)"),
            ],
            &WKB_GEOMETRY,
        );

        let result2_tail = tester_pointn
            .invoke_array_scalar(input_linestrings.clone(), ScalarValue::Int64(Some(-2)))
            .unwrap();
        assert_array_equal(&result2_tail, &expected2_tail);

        // out of range or 0
        let expected_null = create_array(&[None, None, None, None], &WKB_GEOMETRY);

        let result_zero = tester_pointn
            .invoke_array_scalar(input_linestrings.clone(), ScalarValue::Int64(Some(0)))
            .unwrap();
        assert_array_equal(&result_zero, &expected_null);

        let result_too_big = tester_pointn
            .invoke_array_scalar(input_linestrings.clone(), ScalarValue::Int64(Some(5)))
            .unwrap();
        assert_array_equal(&result_too_big, &expected_null);

        let result_too_big_neg = tester_pointn
            .invoke_array_scalar(input_linestrings.clone(), ScalarValue::Int64(Some(-5)))
            .unwrap();
        assert_array_equal(&result_too_big_neg, &expected_null);

        let result_null = tester_pointn
            .invoke_array_scalar(input_linestrings.clone(), ScalarValue::Int64(None))
            .unwrap();
        assert_array_equal(&result_null, &expected_null);

        // invalid cases
        let input_others = create_array(
            &[
                Some("POINT (1 2)"),
                Some("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"),
                Some("MULTIPOINT (0 0, 10 0, 10 10, 0 10, 0 0)"),
                Some("MULTILINESTRING ((1 2, 3 4), (5 6, 7 8))"),
                Some("MULTIPOLYGON (((0 0, 10 0, 10 10, 0 10, 0 0)))"),
                Some("GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (3 4, 5 6))"),
                Some("POINT EMPTY"),
                Some("LINESTRING EMPTY"),
                Some("POLYGON EMPTY"),
                Some("MULTIPOINT EMPTY"),
                Some("MULTILINESTRING EMPTY"),
                Some("MULTIPOLYGON EMPTY"),
                Some("GEOMETRYCOLLECTION EMPTY"),
                None,
            ],
            &sedona_type,
        );

        // all NULL
        let expected_others = create_array(
            &[
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            ],
            &WKB_GEOMETRY,
        );
        let result_others = tester_pointn
            .invoke_array_scalar(input_others.clone(), ScalarValue::Int64(Some(2)))
            .unwrap();
        assert_array_equal(&result_others, &expected_others);
    }
}
