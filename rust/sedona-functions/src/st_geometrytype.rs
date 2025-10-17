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

use crate::executor::WkbBytesExecutor;
use arrow_array::builder::StringBuilder;
use arrow_schema::DataType;
use datafusion_common::error::Result;
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

pub fn st_geometry_type_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_geometrytype",
        vec![Arc::new(STGeometryType {})],
        Volatility::Immutable,
        Some(st_geometry_type_doc()),
    )
}

fn st_geometry_type_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Return the type of a geometry",
        "ST_GeometryType (A: Geometry)",
    )
    .with_argument("geom", "geometry: Input geometry")
    .with_sql_example("SELECT ST_GeometryType(ST_GeomFromWKT('POLYGON ((0 0, 1 0, 0 1, 0 0))'))")
    .build()
}

#[derive(Debug)]
struct STGeometryType {}

impl SedonaScalarKernel for STGeometryType {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry()],
            SedonaType::Arrow(DataType::Utf8),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbBytesExecutor::new(arg_types, args);
        let min_output_size = "ST_POINT".len() * executor.num_iterations();
        let mut builder = StringBuilder::with_capacity(executor.num_iterations(), min_output_size);

        // Iterate over raw WKB bytes for faster type inference
        executor.execute_wkb_void(|maybe_bytes| {
            match maybe_bytes {
                Some(bytes) => {
                    let name = infer_geometry_type_name(bytes)?;
                    builder.append_value(name);
                }
                None => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

/// Fast-path inference of geometry type name from raw WKB bytes
/// An error will be thrown for invalid WKB bytes input
///
/// Spec: https://libgeos.org/specifications/wkb/
#[inline]
fn infer_geometry_type_name(buf: &[u8]) -> Result<&'static str> {
    if buf.len() < 5 {
        return sedona_internal_err!("Invalid WKB: buffer too small ({} bytes)", buf.len());
    }

    let byte_order = buf[0];
    let code = match byte_order {
        0 => u32::from_be_bytes([buf[1], buf[2], buf[3], buf[4]]),
        1 => u32::from_le_bytes([buf[1], buf[2], buf[3], buf[4]]),
        other => return sedona_internal_err!("Unexpected byte order: {other}"),
    };

    // Only low 3 bits is for the base type, high bits include additional info
    match code & 0x7 {
        1 => Ok("ST_Point"),
        2 => Ok("ST_LineString"),
        3 => Ok("ST_Polygon"),
        4 => Ok("ST_MultiPoint"),
        5 => Ok("ST_MultiLineString"),
        6 => Ok("ST_MultiPolygon"),
        7 => Ok("ST_GeometryCollection"),
        _ => sedona_internal_err!("WKB type code out of range. Got: {}", code),
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{create_array, ArrayRef};
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_geometry_type_udf().into();
        assert_eq!(udf.name(), "st_geometrytype");
        assert!(udf.documentation().is_some())
    }

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf: ScalarUDF = st_geometry_type_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![sedona_type]);
        tester.assert_return_type(DataType::Utf8);

        let result = tester
            .invoke_scalar("POLYGON ((0 0, 1 0, 0 1, 0 0))")
            .unwrap();
        tester.assert_scalar_result_equals(result, "ST_Polygon");

        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        assert!(result.is_null());

        let input_wkt = vec![
            None,
            Some("POINT (1 2)"),
            Some("POLYGON ((0 0, 1 0, 0 1, 0 0))"),
            Some("LINESTRING (0 0, 1 0, 0 1)"),
            Some("MULTIPOINT ((0 1), (2 3))"),
            Some("MULTILINESTRING ((0 1, 2 3))"),
            Some("MULTIPOLYGON (((0 0, 0 1, 1 0, 0 0)))"),
            Some("GEOMETRYCOLLECTION (POINT (0 1))"),
        ];
        let expected: ArrayRef = create_array!(
            Utf8,
            [
                None,
                Some("ST_Point"),
                Some("ST_Polygon"),
                Some("ST_LineString"),
                Some("ST_MultiPoint"),
                Some("ST_MultiLineString"),
                Some("ST_MultiPolygon"),
                Some("ST_GeometryCollection")
            ]
        );
        assert_eq!(&tester.invoke_wkb_array(input_wkt).unwrap(), &expected);
    }
}
