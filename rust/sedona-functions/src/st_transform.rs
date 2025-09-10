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
use sedona_expr::scalar_udf::SedonaScalarUDF;
use sedona_schema::{
    datatypes::{Edges, SedonaType},
    matchers::ArgMatcher,
};

/// St_Transform() UDF implementation
///
/// An implementation of intersection calculation.
pub fn st_transform_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new_stub(
        "st_transform",
        ArgMatcher::new(
            vec![
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_string(),
                ArgMatcher::is_optional(ArgMatcher::is_string()),
                ArgMatcher::is_optional(ArgMatcher::is_boolean()),
            ],
            SedonaType::Wkb(Edges::Planar, None),
        ),
        Volatility::Immutable,
        Some(st_transform_doc()),
    )
}

fn st_transform_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Introduction:
Transform the Spatial Reference System / Coordinate Reference System of A, from SourceCRS to TargetCRS. If the SourceCRS is not specified, CRS will be fetched from the geometry column.

Lon/Lat Order in the input geometry

If the input geometry is in lat/lon order, it might throw an error such as too close to pole, latitude or longitude exceeded limits, or give unexpected results. You need to make sure that the input geometry is in lon/lat order. If the input geometry is in lat/lon order, you can use ST_FlipCoordinates to swap X and Y.

Lon/Lat Order in the source and target CRS

Sedona will make sure the source and target CRS are in lon/lat order. If the source CRS or target CRS is in lat/lon order, these CRS will be swapped to lon/lat order.

CRS code

The CRS code is the code of the CRS in the official EPSG database (https://epsg.org/) in the format of EPSG:XXXX. A community tool https://spatialreference.org can help you quick identify a CRS code. For example, the code of WGS84 is EPSG:4326.

CRS format

You can use any string accepted by PROJ to specify a CRS (e.g., PROJJSON, WKT1/2, authority/code in the form authority:code).
","ST_Transform (A: Geometry, SourceCRS: String, TargetCRS: String)")
        .with_argument("geom", "geometry: Input geometry or geography")
        .with_argument("source_crs", "string: Source CRS code or WKT")
        .with_argument("target_crs", "string: Target CRS code or WKT")
        .with_argument("lenient", "boolean: If true, assumes the geometry uses EPSG:4326 when source_crs is not specified. Defaults to true.")
        .with_sql_example(
            "
            SELECT ST_Transform(ST_GeomFromWkt('POLYGON((170 50,170 72,-130 72,-130 50,170 50))'),'EPSG:4326', 'EPSG:32649')"
        )
        .with_sql_example("SELECT ST_Transform(ST_GeomFromWkt('POLYGON((170 50,170 72,-130 72,-130 50,170 50))'),'EPSG:4326', 'EPSG:32649', false)")
        .with_syntax_example("ST_Transform (A: Geometry, SourceCRS: String, TargetCRS: String)")
        .with_syntax_example("ST_Transform (A: Geometry, TargetCRS: String)")
        .build()
}

#[cfg(test)]
mod test {
    use super::*;
    use datafusion_expr::ScalarUDFImpl;

    #[test]
    fn udf_metadata() {
        let udf: SedonaScalarUDF = st_transform_udf();
        assert_eq!(udf.name(), "st_transform");
        assert!(udf.documentation().is_some());
    }
}
