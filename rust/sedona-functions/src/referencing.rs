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
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

/// ST_LineLocatePoint() scalar UDF implementation
pub fn st_line_locate_point_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new_stub(
        "st_line_locate_point",
        ArgMatcher::new(
            vec![ArgMatcher::is_geometry(), ArgMatcher::is_geometry()],
            SedonaType::Arrow(DataType::Float64),
        ),
        Volatility::Immutable,
        Some(st_line_locate_point_doc()),
    )
}

fn st_line_locate_point_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the distance along a linear geometry required to reach the closest point to target",
        "ST_LineLocatePoint (geom: Geometry, target: Geometry)",
    )
    .with_argument("geom", "geometry: Input geometry")
    .with_argument("target", "geometry: Point to locate")
    .with_sql_example(
        "SELECT ST_LineLocatePoint(ST_GeomFromWKT('LINESTRING(38 16, 38 50, 65 50, 66 16, 38 16)'), ST_Point(38, 50))",
    )
    .build()
}

/// ST_LineInterpolatePoint() scalar UDF implementation
pub fn st_line_interpolate_point_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new_stub(
        "st_line_interpolate_point",
        ArgMatcher::new(
            vec![ArgMatcher::is_geometry(), ArgMatcher::is_numeric()],
            WKB_GEOMETRY,
        ),
        Volatility::Immutable,
        Some(st_line_interpolate_point_doc()),
    )
}

fn st_line_interpolate_point_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the point at a given relative distance (0 to 1) along a linear geometry",
        "ST_LineInterpolatePoint (geom: Geometry, distance: double)",
    )
    .with_argument("geom", "geometry: Input geometry")
    .with_argument("distance", "double: Relative distance along geom")
    .with_sql_example(
        "SELECT ST_LineInterpolatePoint(ST_GeomFromWKT('LINESTRING(38 16, 38 50, 65 50, 66 16, 38 16)'), 0.25)",
    )
    .build()
}

#[cfg(test)]
mod tests {
    use datafusion_expr::ScalarUDF;

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_line_interpolate_point_udf().into();
        assert_eq!(udf.name(), "st_line_interpolate_point");
        assert!(udf.documentation().is_some());

        let udf: ScalarUDF = st_line_locate_point_udf().into();
        assert_eq!(udf.name(), "st_line_locate_point");
        assert!(udf.documentation().is_some());
    }
}
