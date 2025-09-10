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

/// ST_DWithin() scalar UDF stub
pub fn st_dwithin_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new_stub(
        "st_dwithin",
        ArgMatcher::new(
            vec![
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_numeric(),
            ],
            SedonaType::Arrow(DataType::Boolean),
        ),
        Volatility::Immutable,
        Some(dwithin_doc()),
    )
}

fn dwithin_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Return true if geomA is within distance of geomB",
        "ST_DWithin (A: Geometry, B: Geometry, distance: Double)"
    )
    .with_argument("geomA", "geometry: Input geometry or geography")
    .with_argument("geomB", "geometry: Input geometry or geography")
    .with_argument("distance", "double: Distance in units of the geometry's coordinate system")
    .with_sql_example("SELECT ST_DWithin(ST_Point(0.25 0.25), ST_GeomFromText('POLYGON ((0 0, 1 0, 0 1, 0 0))'), 0.5)")
    .build()
}

#[cfg(test)]
mod tests {
    use datafusion_expr::ScalarUDF;

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_dwithin_udf().into();
        assert_eq!(udf.name(), "st_dwithin");
        assert!(udf.documentation().is_some())
    }
}
