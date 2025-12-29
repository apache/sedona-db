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

use crate::to_geo::GeoTypesExecutor;
use arrow_array::builder::StringBuilder;
use arrow_schema::DataType;
use datafusion_common::error::{DataFusionError, Result};
use datafusion_expr::ColumnarValue;
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

/// ST_AsGeoJSON() kernel implementation using GeoTypesExecutor
pub fn st_asgeojson_impl() -> ScalarKernelRef {
    Arc::new(STAsGeoJSON {})
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
        let executor = GeoTypesExecutor::new(arg_types, args);

        // Estimate the minimum probable memory requirement of the output.
        // GeoJSON is typically longer than WKT due to JSON formatting.
        let min_probable_geojson_size = executor.num_iterations() * 50;

        // Initialize an output builder of the appropriate type
        let mut builder =
            StringBuilder::with_capacity(executor.num_iterations(), min_probable_geojson_size);

        executor.execute_wkb_void(|maybe_geom| {
            match maybe_geom {
                Some(geom) => {
                    // Convert geo_types::Geometry to geojson::Geometry
                    let geojson_geom: geojson::Geometry = (&geom).into();

                    // Serialize to JSON string
                    let json_str = serde_json::to_string(&geojson_geom)
                        .map_err(|err| DataFusionError::External(Box::new(err)))?;
                    builder.append_value(&json_str);
                }
                None => builder.append_null(),
            };

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

#[cfg(test)]
mod tests {
    use datafusion_common::scalar::ScalarValue;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::WKB_GEOMETRY;
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[test]
    fn test_simple_geojson() {
        let kernel = st_asgeojson_impl();
        let udf = SedonaScalarUDF::from_kernel("st_asgeojson", kernel);
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOMETRY]);

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
        let udf = SedonaScalarUDF::from_kernel("st_asgeojson", kernel);
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
        let udf = SedonaScalarUDF::from_kernel("st_asgeojson", kernel);
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
        let udf = SedonaScalarUDF::from_kernel("st_asgeojson", kernel);
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOMETRY]);

        let result = tester
            .invoke_wkb_scalar(Some("GEOMETRYCOLLECTION(POINT(1 2), LINESTRING(0 0, 1 1))"))
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            r#"{"type":"GeometryCollection","geometries":[{"type":"Point","coordinates":[1.0,2.0]},{"type":"LineString","coordinates":[[0.0,0.0],[1.0,1.0]]}]}"#,
        );
    }
}
