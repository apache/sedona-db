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
use std::sync::Arc;

use arrow_array::builder::StringBuilder;
use arrow_schema::DataType;
use datafusion_common::error::{DataFusionError, Result};
use datafusion_expr::ColumnarValue;
use geo_traits::{GeometryTrait, PointTrait, PolygonTrait};
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{ScalarKernelRef, SedonaScalarKernel},
};
use sedona_functions::executor::WkbExecutor;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};
use wkb::reader::Wkb;

use crate::to_geo::item_to_geometry;

/// ST_AsGeoJSON() kernel implementation using WkbExecutor
pub fn st_asgeojson_impl() -> Vec<ScalarKernelRef> {
    ItemCrsKernel::wrap_impl(STAsGeoJSON {})
}

#[derive(Debug)]
struct STAsGeoJSON {}

impl SedonaScalarKernel for STAsGeoJSON {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry()],
            SedonaType::Arrow(DataType::Utf8),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);

        // Estimate the minimum probable memory requirement of the output.
        // GeoJSON is typically longer than WKT due to JSON formatting.
        let min_probable_geojson_size = executor.num_iterations() * 33;

        // Initialize an output builder of the appropriate type
        let mut builder =
            StringBuilder::with_capacity(executor.num_iterations(), min_probable_geojson_size);

        executor.execute_wkb_void(|maybe_wkb| {
            match maybe_wkb {
                Some(wkb) => {
                    let json_str = geom_to_geojson(&wkb)?;
                    builder.append_value(&json_str);
                }
                None => builder.append_null(),
            };

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

/// Convert a WKB geometry to GeoJSON string, handling special cases for empty geometries
fn geom_to_geojson(geom: &Wkb) -> Result<String> {
    // Special case handling for geometries that geo_types::Geometry cannot represent
    match geom.as_type() {
        geo_traits::GeometryType::Point(pt) => {
            if pt.coord().is_none() {
                // Empty point - geo_types cannot represent this
                return Ok(r#"{"type":"Point","coordinates":[]}"#.to_string());
            }
        }
        geo_traits::GeometryType::Polygon(poly) => {
            if poly.exterior().is_none() {
                // Empty polygon - to match PostGIS behavior
                return Ok(r#"{"type":"Polygon","coordinates":[]}"#.to_string());
            }
        }
        _ => {}
    }

    // For all other geometries (including other empty geometries), convert to geo_types::Geometry
    let geo_geom = item_to_geometry(geom)?;

    let geojson_value = geojson::Value::from(&geo_geom);
    let geojson_geom = geojson::Geometry::new(geojson_value);

    serde_json::to_string(&geojson_geom).map_err(|err| DataFusionError::External(Box::new(err)))
}

#[cfg(test)]
mod tests {
    use datafusion_common::scalar::ScalarValue;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS};
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn test_simple_geojson(
        #[values(WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS.clone())] sedona_type: SedonaType,
    ) {
        let kernel = st_asgeojson_impl();
        let udf = SedonaScalarUDF::from_impl("st_asgeojson", kernel);
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type]);

        // Test with a simple point
        let result = tester.invoke_wkb_scalar(Some("POINT (1 2)")).unwrap();
        tester.assert_scalar_result_equals(result, r#"{"type":"Point","coordinates":[1.0,2.0]}"#);

        // Test with null
        let result = tester.invoke_wkb_scalar(None).unwrap();
        assert_eq!(result, ScalarValue::Utf8(None));
    }

    #[test]
    fn test_linestring() {
        let kernel = st_asgeojson_impl();
        let udf = SedonaScalarUDF::from_impl("st_asgeojson", kernel);
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOMETRY]);

        let result = tester
            .invoke_wkb_scalar(Some("LINESTRING (0 0, 1 1, 2 2)"))
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            r#"{"type":"LineString","coordinates":[[0.0,0.0],[1.0,1.0],[2.0,2.0]]}"#,
        );
    }

    #[test]
    fn test_polygon() {
        let kernel = st_asgeojson_impl();
        let udf = SedonaScalarUDF::from_impl("st_asgeojson", kernel);
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOMETRY]);

        let result = tester
            .invoke_wkb_scalar(Some("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"))
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            r#"{"type":"Polygon","coordinates":[[[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0],[0.0,0.0]]]}"#,
        );
    }

    #[test]
    fn test_geometry_collection() {
        let kernel = st_asgeojson_impl();
        let udf = SedonaScalarUDF::from_impl("st_asgeojson", kernel);
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOMETRY]);

        let result = tester
            .invoke_wkb_scalar(Some("GEOMETRYCOLLECTION(POINT(1 2), LINESTRING(0 0, 1 1))"))
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            r#"{"type":"GeometryCollection","geometries":[{"type":"Point","coordinates":[1.0,2.0]},{"type":"LineString","coordinates":[[0.0,0.0],[1.0,1.0]]}]}"#,
        );
    }

    #[test]
    fn test_empty_point() {
        let kernel = st_asgeojson_impl();
        let udf = SedonaScalarUDF::from_impl("st_asgeojson", kernel);
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOMETRY]);

        let result = tester.invoke_wkb_scalar(Some("POINT EMPTY")).unwrap();
        tester.assert_scalar_result_equals(result, r#"{"type":"Point","coordinates":[]}"#);
    }

    #[test]
    fn test_empty_polygon() {
        let kernel = st_asgeojson_impl();
        let udf = SedonaScalarUDF::from_impl("st_asgeojson", kernel);
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOMETRY]);

        let result = tester.invoke_wkb_scalar(Some("POLYGON EMPTY")).unwrap();
        tester.assert_scalar_result_equals(result, r#"{"type":"Polygon","coordinates":[]}"#);
    }
}
