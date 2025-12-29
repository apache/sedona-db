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
use arrow_schema::DataType;
use datafusion_expr::{scalar_doc_sections::DOC_SECTION_OTHER, Documentation, Volatility};
use sedona_expr::scalar_udf::SedonaScalarUDF;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

/// ST_AsGeoJSON() scalar UDF implementation
///
/// Stub function for GeoJSON conversion.
pub fn st_asgeojson_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new_stub(
        "st_asgeojson",
        ArgMatcher::new(
            vec![ArgMatcher::is_geometry_or_geography()],
            SedonaType::Arrow(DataType::Utf8),
        ),
        Volatility::Immutable,
        Some(st_asgeojson_doc()),
    )
}

fn st_asgeojson_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Return the GeoJSON representation of a geometry",
        "ST_AsGeoJSON (A: Geometry [, type: String])",
    )
    .with_argument("geom", "geometry: Input geometry")
    .with_argument(
        "type",
        "string (optional): Output type - 'Simple' (default), 'Feature', or 'FeatureCollection'",
    )
    .with_sql_example("SELECT ST_AsGeoJSON(ST_Point(1.0, 2.0))")
    .with_sql_example("SELECT ST_AsGeoJSON(ST_Point(1.0, 2.0), 'Feature')")
    .with_sql_example("SELECT ST_AsGeoJSON(ST_Point(1.0, 2.0), 'FeatureCollection')")
    .with_related_udf("ST_GeomFromGeoJSON")
    .build()
}

#[cfg(test)]
mod tests {
    use datafusion_expr::ScalarUDF;

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_asgeojson_udf().into();
        assert_eq!(udf.name(), "st_asgeojson");
        assert!(udf.documentation().is_some())
    }
}
