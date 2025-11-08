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
use std::vec;

use datafusion_expr::{scalar_doc_sections::DOC_SECTION_OTHER, Documentation, Volatility};
use sedona_expr::aggregate_udf::SedonaAggregateUDF;
use sedona_schema::{datatypes::WKB_GEOMETRY, matchers::ArgMatcher};

/// ST_Polygonize() aggregate UDF implementation
///
/// Creates polygons from a set of linework that forms closed rings.
pub fn st_polygonize_udf() -> SedonaAggregateUDF {
    SedonaAggregateUDF::new_stub(
        "st_polygonize",
        ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOMETRY),
        Volatility::Immutable,
        Some(st_polygonize_doc()),
    )
}

fn st_polygonize_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Creates a GeometryCollection containing polygons formed from the linework of a set of geometries. \
         Returns an empty GeometryCollection if no polygons can be formed.",
        "ST_Polygonize (geom: Geometry)",
    )
    .with_argument("geom", "geometry: Input geometry (typically linestrings that form closed rings)")
    .with_sql_example(
        "SELECT ST_AsText(ST_Polygonize(geom)) FROM (VALUES \
         (ST_GeomFromText('LINESTRING (0 0, 10 0)')), \
         (ST_GeomFromText('LINESTRING (10 0, 10 10)')), \
         (ST_GeomFromText('LINESTRING (10 10, 0 0)'))  \
         ) AS t(geom)"
    )
    .build()
}

#[cfg(test)]
mod test {
    use datafusion_expr::AggregateUDF;

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: AggregateUDF = st_polygonize_udf().into();
        assert_eq!(udf.name(), "st_polygonize");
        assert!(udf.documentation().is_some());
    }
}
