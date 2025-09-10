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

/// ST_Length() scalar UDF implementation
///
/// Stub function for length calculation.
pub fn st_length_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new_stub(
        "st_length",
        ArgMatcher::new(
            vec![ArgMatcher::is_geometry_or_geography()],
            SedonaType::Arrow(DataType::Float64),
        ),
        Volatility::Immutable,
        Some(st_length_doc()),
    )
}

fn st_length_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the length of geom\
         This function only supports LineString, MultiLineString, and GeometryCollections \
         containing linear geometries. Use ST_Perimeter for polygons.\
        ",
        "ST_Length (A: Geometry)",
    )
    .with_argument("geom", "geometry: Input geometry")
    .with_sql_example(
        "SELECT ST_Length(ST_GeomFromWKT('LINESTRING(38 16,38 50,65 50,66 16,38 16)'))",
    )
    .with_related_udf("ST_Perimeter")
    .build()
}

#[cfg(test)]
mod tests {
    use datafusion_expr::ScalarUDF;

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_length_udf().into();
        assert_eq!(udf.name(), "st_length");
        assert!(udf.documentation().is_some())
    }
}
