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
use datafusion_common::exec_err;
use datafusion_expr::ColumnarValue;
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

/// Output format type for GeoJSON
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GeoJsonType {
    Simple,
    Feature,
    FeatureCollection,
}

impl GeoJsonType {
    pub fn from_geojson_type_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "simple" => Ok(GeoJsonType::Simple),
            "feature" => Ok(GeoJsonType::Feature),
            "featurecollection" => Ok(GeoJsonType::FeatureCollection),
            _ => exec_err!(
                "Invalid GeoJSON type '{}'. Valid options are: 'Simple', 'Feature', 'FeatureCollection'",
                s
            ),
        }
    }
}

/// ST_AsGeoJSON() kernel implementation using GeoTypesExecutor
pub fn st_asgeojson_impl(geojson_type: GeoJsonType) -> ScalarKernelRef {
    Arc::new(STAsGeoJSON { geojson_type })
}

/// ST_AsGeoJSON() kernel implementation with dynamic type parameter
pub fn st_asgeojson_with_type_impl() -> ScalarKernelRef {
    Arc::new(STAsGeoJSONWithType {})
}

#[derive(Debug)]
struct STAsGeoJSON {
    geojson_type: GeoJsonType,
}

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
        // Feature and FeatureCollection add extra wrapping
        let base_size = match self.geojson_type {
            GeoJsonType::Simple => 50,
            GeoJsonType::Feature => 100,
            GeoJsonType::FeatureCollection => 150,
        };
        let min_probable_geojson_size = executor.num_iterations() * base_size;

        // Initialize an output builder of the appropriate type
        let mut builder =
            StringBuilder::with_capacity(executor.num_iterations(), min_probable_geojson_size);

        executor.execute_wkb_void(|maybe_geom| {
            match maybe_geom {
                Some(geom) => {
                    // Convert geo_types::Geometry to geojson::Geometry
                    let geojson_geom: geojson::Geometry = (&geom).into();

                    // Wrap the geometry based on the type parameter and serialize
                    match self.geojson_type {
                        GeoJsonType::Simple => {
                            let json_str = serde_json::to_string(&geojson_geom)
                                .map_err(|err| DataFusionError::External(Box::new(err)))?;
                            builder.append_value(&json_str);
                        }
                        GeoJsonType::Feature => {
                            let feature = geojson::Feature {
                                bbox: None,
                                geometry: Some(geojson_geom),
                                id: None,
                                properties: None,
                                foreign_members: None,
                            };
                            let json_str = serde_json::to_string(&feature)
                                .map_err(|err| DataFusionError::External(Box::new(err)))?;
                            builder.append_value(&json_str);
                        }
                        GeoJsonType::FeatureCollection => {
                            let feature = geojson::Feature {
                                bbox: None,
                                geometry: Some(geojson_geom),
                                id: None,
                                properties: None,
                                foreign_members: None,
                            };
                            let feature_collection = geojson::FeatureCollection {
                                bbox: None,
                                features: vec![feature],
                                foreign_members: None,
                            };
                            let json_str = serde_json::to_string(&feature_collection)
                                .map_err(|err| DataFusionError::External(Box::new(err)))?;
                            builder.append_value(&json_str);
                        }
                    }
                }
                None => builder.append_null(),
            };

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

#[derive(Debug)]
struct STAsGeoJSONWithType {}

impl SedonaScalarKernel for STAsGeoJSONWithType {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry(), ArgMatcher::is_string()],
            SedonaType::Arrow(DataType::Utf8),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        // Extract the type parameter
        let geojson_type = match &args[1] {
            ColumnarValue::Scalar(datafusion_common::ScalarValue::Utf8(Some(type_str))) => {
                GeoJsonType::from_geojson_type_str(type_str.as_str())?
            }
            ColumnarValue::Scalar(datafusion_common::ScalarValue::Utf8(None)) => {
                GeoJsonType::Simple // Default to Simple if NULL
            }
            _ => {
                return exec_err!("Second argument to ST_AsGeoJSON must be a string literal");
            }
        };

        // Delegate to the appropriate kernel based on the type parameter
        let kernel = st_asgeojson_impl(geojson_type);
        kernel.invoke_batch(&arg_types[..1], &args[..1])
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
        let kernel = st_asgeojson_impl(GeoJsonType::Simple);
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
    fn test_feature_geojson() {
        let kernel = st_asgeojson_impl(GeoJsonType::Feature);
        let udf = SedonaScalarUDF::from_kernel("st_asgeojson", kernel);
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOMETRY]);

        let result = tester.invoke_wkb_scalar(Some("POINT (1 2)")).unwrap();
        tester.assert_scalar_result_equals(
            result,
            r#"{"type":"Feature","geometry":{"type":"Point","coordinates":[1.0,2.0]},"properties":null}"#,
        );
    }

    #[test]
    fn test_feature_collection_geojson() {
        let kernel = st_asgeojson_impl(GeoJsonType::FeatureCollection);
        let udf = SedonaScalarUDF::from_kernel("st_asgeojson", kernel);
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOMETRY]);

        let result = tester.invoke_wkb_scalar(Some("POINT (1 2)")).unwrap();
        tester.assert_scalar_result_equals(
            result,
            r#"{"type":"FeatureCollection","features":[{"type":"Feature","geometry":{"type":"Point","coordinates":[1.0,2.0]},"properties":null}]}"#,
        );
    }

    #[test]
    fn test_geometry_collection() {
        let kernel = st_asgeojson_impl(GeoJsonType::Simple);
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

    #[test]
    fn test_with_type_parameter() {
        let kernel = st_asgeojson_with_type_impl();
        let udf = SedonaScalarUDF::from_kernel("st_asgeojson", kernel);
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![WKB_GEOMETRY, SedonaType::Arrow(DataType::Utf8)],
        );

        // Test with 'Simple' type
        let result = tester
            .invoke_scalar_scalar("POINT (1 2)", "Simple")
            .unwrap();
        tester.assert_scalar_result_equals(result, r#"{"type":"Point","coordinates":[1.0,2.0]}"#);

        // Test with 'Feature' type
        let result = tester
            .invoke_scalar_scalar("POINT (1 2)", "Feature")
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            r#"{"type":"Feature","geometry":{"type":"Point","coordinates":[1.0,2.0]},"properties":null}"#,
        );

        // Test with 'FeatureCollection' type
        let result = tester
            .invoke_scalar_scalar("POINT (1 2)", "FeatureCollection")
            .unwrap();
        tester.assert_scalar_result_equals(
            result,
            r#"{"type":"FeatureCollection","features":[{"type":"Feature","geometry":{"type":"Point","coordinates":[1.0,2.0]},"properties":null}]}"#,
        );
    }

    #[test]
    fn test_invalid_type_string() {
        let kernel = st_asgeojson_with_type_impl();
        let udf = SedonaScalarUDF::from_kernel("st_asgeojson", kernel);
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![WKB_GEOMETRY, SedonaType::Arrow(DataType::Utf8)],
        );

        // Test with invalid type string
        let result = tester.invoke_scalar_scalar("POINT (1 2)", "InvalidType");
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Invalid GeoJSON type"));
    }
}
