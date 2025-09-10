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

/// ST_Perimeter() scalar UDF implementation
///
/// Stub function for perimeter calculation.
pub fn st_perimeter_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new_stub(
        "st_perimeter",
        ArgMatcher::new(
            vec![
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_optional(ArgMatcher::is_boolean()),
                ArgMatcher::is_optional(ArgMatcher::is_boolean()),
            ],
            SedonaType::Arrow(DataType::Float64),
        ),
        Volatility::Immutable,
        Some(st_perimeter_doc()),
    )
}

fn st_perimeter_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Introduction: This function calculates the 2D perimeter of a given geometry. It supports Polygon, MultiPolygon, and GeometryCollection geometries (as long as the GeometryCollection contains polygonal geometries). For other types, it returns 0. To measure lines, use ST_Length.\n\
        To get the perimeter in meters, set use_spheroid to true. This calculates the geodesic perimeter using the WGS84 spheroid. When using use_spheroid, the lenient parameter defaults to true, assuming the geometry uses EPSG:4326. To throw an exception instead, set lenient to false.",
        "ST_Perimeter(geom: Geometry)"
         )
        .with_syntax_example("ST_Perimeter(geom: Geometry, use_spheroid: Boolean)")
        .with_syntax_example("ST_Perimeter(geom: Geometry, use_spheroid: Boolean, lenient: Boolean = True)")
        .with_sql_example("SELECT ST_Perimeter(ST_GeomFromText('POLYGON((0 0, 0 5, 5 5, 5 0, 0 0))'))")
        .with_sql_example("SELECT ST_Perimeter(ST_GeomFromText('POLYGON((0 0, 0 5, 5 5, 5 0, 0 0))', 4326), true, false)")
        .with_argument("geom", "geometry: Input geometry")
        .with_argument("use_spheroid", "boolean: If true, calculates the geodesic perimeter using the WGS84 spheroid. Defaults to false.")
        .with_argument("lenient", "boolean: If true, assumes the geometry uses EPSG:4326 when use_spheroid is true. Defaults to true.")
        .build()
}

#[cfg(test)]
mod tests {
    use datafusion_expr::ScalarUDF;

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_perimeter_udf().into();
        assert_eq!(udf.name(), "st_perimeter");
        assert!(udf.documentation().is_some());
    }
}
