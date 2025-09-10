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

/// ST_Centroid() scalar UDF stub
///
/// Stub function for centroid calculation.
pub fn st_centroid_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new_stub(
        "st_centroid",
        ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOMETRY),
        Volatility::Immutable,
        Some(st_centroid_doc()),
    )
}

fn st_centroid_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the centroid of geom",
        "ST_Centroid (A: Geometry)",
    )
    .with_argument("geom", "geometry: Input geometry")
    .with_sql_example("SELECT ST_Centroid(ST_GeomFromText('POLYGON ((1 1, 11 1, 1 11, 0 0))'))")
    .build()
}

#[cfg(test)]
mod tests {
    use datafusion_expr::ScalarUDF;

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_centroid_udf().into();
        assert_eq!(udf.name(), "st_centroid");
        assert!(udf.documentation().is_some())
    }
}
