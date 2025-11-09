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
use datafusion_expr::{scalar_doc_sections::DOC_SECTION_OTHER, Documentation, Volatility};
use sedona_expr::scalar_udf::SedonaScalarUDF;
use sedona_schema::{datatypes::WKB_GEOMETRY, matchers::ArgMatcher};

/// ST_Buffer() scalar UDF stub
pub fn st_buffer_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new_stub(
        "st_buffer",
        ArgMatcher::new(
            vec![
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_numeric(),
            ],
            WKB_GEOMETRY,
        ),
        Volatility::Immutable,
        Some(st_buffer_doc()),
    )
}

fn st_buffer_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns a geometry that represents all points whose distance from this Geometry is less than or equal to distance",
        "st_buffer (A: Geometry, B: Double)"
    )
    .with_argument("geom", "geometry: Input geometry")
    .with_argument("distance", "distance: Radius of the buffer")
    .with_sql_example("SELECT ST_Buffer(ST_GeomFromText('POLYGON ((10 10, 11 10, 10 11, 10 10))'), 1.0)".to_string())
    .build()
}

#[cfg(test)]
mod tests {
    use datafusion_expr::ScalarUDF;

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_buffer_udf().into();
        assert_eq!(udf.name(), "st_buffer");
        assert!(udf.documentation().is_some())
    }
}
