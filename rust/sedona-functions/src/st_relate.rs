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
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

/// ST_Relate() scalar UDF implementation
pub fn st_relate_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::from_impl(
        "st_relate",
        ArgMatcher::new(
            vec![ArgMatcher::is_geometry(), ArgMatcher::is_geometry()],
            SedonaType::Arrow(arrow_schema::DataType::Utf8),
        ),
        Volatility::Immutable,
        Some(st_relate_doc()),
    )
}

fn st_relate_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the DE-9IM intersection matrix for two geometries",
        "ST_Relate (geomA: Geometry, geomB: Geometry)",
    )
    .with_argument("geomA", "First input geometry")
    .with_argument("geomB", "Second input geometry")
    .with_sql_example(
        "SELECT ST_Relate(
            ST_GeomFromWKT('POINT(0 0)'),
            ST_GeomFromWKT('POINT(1 1)')
        )",
    )
    .build()
}
