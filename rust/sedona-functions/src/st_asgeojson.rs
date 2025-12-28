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

use crate::executor::WkbExecutor;
use arrow_array::builder::StringBuilder;
use arrow_schema::DataType;
use datafusion_common::error::{DataFusionError, Result};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use geo_traits::to_geo::{
    ToGeoLineString, ToGeoMultiLineString, ToGeoMultiPoint, ToGeoMultiPolygon, ToGeoPoint,
    ToGeoPolygon,
};
use geo_traits::{GeometryCollectionTrait, GeometryTrait, GeometryType};
use geo_types::Geometry;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

/// Output format type for GeoJSON
#[derive(Debug, Clone, Copy, PartialEq)]
enum GeoJsonType {
    Simple,
    Feature,
    FeatureCollection,
}

impl GeoJsonType {
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "simple" => Ok(GeoJsonType::Simple),
            "feature" => Ok(GeoJsonType::Feature),
            "featurecollection" => Ok(GeoJsonType::FeatureCollection),
            _ => Err(DataFusionError::Execution(format!(
                "Invalid GeoJSON type '{}'. Valid options are: 'Simple', 'Feature', 'FeatureCollection'",
                s
            ))),
        }
    }
}

/// ST_AsGeoJSON() scalar UDF implementation
///
/// An implementation of GeoJSON writing using the geojson crate.
pub fn st_asgeojson_udf() -> SedonaScalarUDF {
    let udf = SedonaScalarUDF::new(
        "st_asgeojson",
        vec![
            Arc::new(STAsGeoJSON {}),
            Arc::new(STAsGeoJSONWithType {}),
        ],
        Volatility::Immutable,
        Some(st_asgeojson_doc()),
    );
    udf
}

fn st_asgeojson_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Return the GeoJSON representation of a geometry or geography",
        "ST_AsGeoJSON (A: Geometry [, type: String])",
    )
    .with_argument("geom", "geometry: Input geometry or geography")
    .with_argument("type", "string (optional): Output type - 'Simple' (default), 'Feature', or 'FeatureCollection'")
    .with_sql_example("SELECT ST_AsGeoJSON(ST_Point(1.0, 2.0))")
    .with_sql_example("SELECT ST_AsGeoJSON(ST_Point(1.0, 2.0), 'Feature')")
    .with_sql_example("SELECT ST_AsGeoJSON(ST_Point(1.0, 2.0), 'FeatureCollection')")
    .with_related_udf("ST_GeomFromGeoJSON")
    .build()
}

#[derive(Debug)]
struct STAsGeoJSON {}

impl SedonaScalarKernel for STAsGeoJSON {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry_or_geography()],
            SedonaType::Arrow(DataType::Utf8),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        convert_to_geojson(arg_types, args, GeoJsonType::Simple)
    }
}

#[derive(Debug)]
struct STAsGeoJSONWithType {}

impl SedonaScalarKernel for STAsGeoJSONWithType {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_string(),
            ],
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
                GeoJsonType::from_str(type_str.as_str())?
            }
            ColumnarValue::Scalar(datafusion_common::ScalarValue::Utf8(None)) => {
                GeoJsonType::Simple // Default to Simple if NULL
            }
            _ => {
                return Err(DataFusionError::Execution(
                    "Second argument to ST_AsGeoJSON must be a string literal".to_string(),
                ));
            }
        };

        convert_to_geojson(&arg_types[..1], &args[..1], geojson_type)
    }
}

fn convert_to_geojson(
    arg_types: &[SedonaType],
    args: &[ColumnarValue],
    geojson_type: GeoJsonType,
) -> Result<ColumnarValue> {
    let executor = WkbExecutor::new(arg_types, args);

    // Estimate the minimum probable memory requirement of the output.
    // GeoJSON is typically longer than WKT due to JSON formatting.
    // Feature and FeatureCollection add extra wrapping
    let base_size = match geojson_type {
        GeoJsonType::Simple => 50,
        GeoJsonType::Feature => 100,
        GeoJsonType::FeatureCollection => 150,
    };
    let min_probable_geojson_size = executor.num_iterations() * base_size;

    // Initialize an output builder of the appropriate type
    let mut builder =
        StringBuilder::with_capacity(executor.num_iterations(), min_probable_geojson_size);

    executor.execute_wkb_void(|maybe_item| {
        match maybe_item {
            Some(item) => {
                // Convert WKB geometry to geo_types::Geometry using geo-traits
                let geo_geometry = wkb_to_geometry(item)?;

                match geo_geometry {
                    Some(geom) => {
                        // Convert geo_types::Geometry to geojson::Geometry
                        let geojson_geom: geojson::Geometry = (&geom).try_into()
                            .map_err(|e| DataFusionError::Execution(format!("Failed to convert to GeoJSON: {:?}", e)))?;

                        // Wrap the geometry based on the type parameter and serialize
                        let geojson_output = match geojson_type {
                            GeoJsonType::Simple => {
                                geojson_geom
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
                                return Ok(());
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
                                return Ok(());
                            }
                        };

                        // For Simple type, serialize the geometry
                        let json_str = serde_json::to_string(&geojson_output)
                            .map_err(|err| DataFusionError::External(Box::new(err)))?;
                        builder.append_value(&json_str);
                    }
                    None => {
                        return Err(DataFusionError::NotImplemented(
                            "Unsupported geometry type for GeoJSON conversion".to_string(),
                        ));
                    }
                }
            }
            None => builder.append_null(),
        };

        Ok(())
    })?;

    executor.finish(Arc::new(builder.finish()))
}

/// Convert a WKB geometry to geo_types::Geometry, including GeometryCollection support
fn wkb_to_geometry(item: impl GeometryTrait<T = f64>) -> Result<Option<Geometry>> {
    let geo_geometry = match item.as_type() {
        GeometryType::Point(geom) => geom.try_to_point().map(Geometry::Point),
        GeometryType::LineString(geom) => Some(Geometry::LineString(geom.to_line_string())),
        GeometryType::Polygon(geom) => Some(Geometry::Polygon(geom.to_polygon())),
        GeometryType::MultiPoint(geom) => geom.try_to_multi_point().map(Geometry::MultiPoint),
        GeometryType::MultiLineString(geom) => {
            Some(Geometry::MultiLineString(geom.to_multi_line_string()))
        }
        GeometryType::MultiPolygon(geom) => {
            Some(Geometry::MultiPolygon(geom.to_multi_polygon()))
        }
        GeometryType::GeometryCollection(geom) => convert_geometry_collection(geom)?,
        _ => None,
    };
    Ok(geo_geometry)
}

/// Convert a GeometryCollection to geo_types::Geometry
/// Handles up to 1 level of nesting to avoid compiler issues with recursion
fn convert_geometry_collection<GC: GeometryCollectionTrait<T = f64>>(
    geom: &GC,
) -> Result<Option<Geometry>> {
    let geometries: Result<Vec<_>> = geom
        .geometries()
        .map(|child| {
            let child_geom = match child.as_type() {
                GeometryType::Point(g) => g.try_to_point().map(Geometry::Point),
                GeometryType::LineString(g) => Some(Geometry::LineString(g.to_line_string())),
                GeometryType::Polygon(g) => Some(Geometry::Polygon(g.to_polygon())),
                GeometryType::MultiPoint(g) => g.try_to_multi_point().map(Geometry::MultiPoint),
                GeometryType::MultiLineString(g) => {
                    Some(Geometry::MultiLineString(g.to_multi_line_string()))
                }
                GeometryType::MultiPolygon(g) => {
                    Some(Geometry::MultiPolygon(g.to_multi_polygon()))
                }
                GeometryType::GeometryCollection(g) => {
                    // Support one level of nested GeometryCollection
                    return convert_geometry_collection(g)?
                        .ok_or_else(|| {
                            DataFusionError::NotImplemented(
                                "Nested GeometryCollection with unsupported types".to_string(),
                            )
                        });
                }
                _ => None,
            };

            child_geom.ok_or_else(|| {
                DataFusionError::NotImplemented(
                    "GeometryCollection contains unsupported geometry types".to_string(),
                )
            })
        })
        .collect();

    let geometries = geometries?;

    Ok(Some(Geometry::GeometryCollection(
        geo_types::GeometryCollection(geometries),
    )))
}

#[cfg(test)]
mod tests {
    use datafusion_common::scalar::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{
        WKB_GEOGRAPHY, WKB_GEOMETRY, WKB_VIEW_GEOGRAPHY,
    };
    use sedona_testing::{compare::assert_scalar_equal, testers::ScalarUdfTester};

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_asgeojson_udf().into();
        assert_eq!(udf.name(), "st_asgeojson");
        assert!(udf.documentation().is_some())
    }

    #[rstest]
    fn udf(
        #[values(WKB_GEOMETRY, WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY, WKB_VIEW_GEOGRAPHY)]
        sedona_type: SedonaType,
    ) {
        let udf = st_asgeojson_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type]);

        // Test with a simple point
        let result = tester.invoke_wkb_scalar(Some("POINT (1 2)")).unwrap();
        if let ScalarValue::Utf8(Some(json_str)) = result {
            // Verify it's valid JSON and contains expected structure
            let parsed: serde_json::Value = serde_json::from_str(json_str.as_str()).unwrap();
            assert_eq!(parsed["type"], "Point");
            assert!(parsed["coordinates"].is_array());
        } else {
            panic!("Expected Utf8 scalar value");
        }

        // Test with null
        assert_scalar_equal(
            &tester.invoke_wkb_scalar(None).unwrap(),
            &ScalarValue::Utf8(None),
        );

        // Test with array
        let result_array = tester
            .invoke_wkb_array(vec![Some("POINT(1 2)"), None, Some("POINT(3 5)")])
            .unwrap();

        // Verify the array has the expected number of elements
        assert_eq!((*result_array).len(), 3);
    }

    #[rstest]
    fn geometry_collection(
        #[values(WKB_GEOMETRY, WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY, WKB_VIEW_GEOGRAPHY)]
        sedona_type: SedonaType,
    ) {
        let udf = st_asgeojson_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type]);

        // Test with a simple GeometryCollection
        let result = tester
            .invoke_wkb_scalar(Some("GEOMETRYCOLLECTION(POINT(1 2), LINESTRING(0 0, 1 1))"))
            .unwrap();
        if let ScalarValue::Utf8(Some(json_str)) = result {
            // Verify it's valid JSON and contains expected structure
            let parsed: serde_json::Value = serde_json::from_str(json_str.as_str()).unwrap();
            assert_eq!(parsed["type"], "GeometryCollection");
            assert!(parsed["geometries"].is_array());
            let geometries = parsed["geometries"].as_array().unwrap();
            assert_eq!(geometries.len(), 2);
            assert_eq!(geometries[0]["type"], "Point");
            assert_eq!(geometries[1]["type"], "LineString");
        } else {
            panic!("Expected Utf8 scalar value");
        }

        // Test with empty GeometryCollection
        let result = tester
            .invoke_wkb_scalar(Some("GEOMETRYCOLLECTION EMPTY"))
            .unwrap();
        if let ScalarValue::Utf8(Some(json_str)) = result {
            let parsed: serde_json::Value = serde_json::from_str(json_str.as_str()).unwrap();
            assert_eq!(parsed["type"], "GeometryCollection");
            let geometries = parsed["geometries"].as_array().unwrap();
            assert_eq!(geometries.len(), 0);
        } else {
            panic!("Expected Utf8 scalar value");
        }
    }

    #[rstest]
    fn geojson_type_parameter(
        #[values(WKB_GEOMETRY, WKB_GEOGRAPHY)]
        sedona_type: SedonaType,
    ) {
        // Test Simple type (default)
        let tester = ScalarUdfTester::new(st_asgeojson_udf().into(), vec![sedona_type.clone()]);
        let result = tester.invoke_wkb_scalar(Some("POINT (1 2)")).unwrap();
        if let ScalarValue::Utf8(Some(json_str)) = result {
            let parsed: serde_json::Value = serde_json::from_str(json_str.as_str()).unwrap();
            assert_eq!(parsed["type"], "Point");
            assert!(parsed.get("geometry").is_none()); // No wrapping
        } else {
            panic!("Expected Utf8 scalar value");
        }

        // Test Feature type
        let tester_with_type = ScalarUdfTester::new(
            st_asgeojson_udf().into(),
            vec![sedona_type, SedonaType::Arrow(DataType::Utf8)],
        );
        let result = tester_with_type
            .invoke_scalar_scalar("POINT (1 2)", "Feature")
            .unwrap();
        if let ScalarValue::Utf8(Some(json_str)) = result {
            let parsed: serde_json::Value = serde_json::from_str(json_str.as_str()).unwrap();
            assert_eq!(parsed["type"], "Feature");
            assert!(parsed["geometry"].is_object());
            assert_eq!(parsed["geometry"]["type"], "Point");
            assert!(parsed["geometry"]["coordinates"].is_array());
        } else {
            panic!("Expected Utf8 scalar value");
        }

        // Test FeatureCollection type
        let result = tester_with_type
            .invoke_scalar_scalar("POINT (1 2)", "FeatureCollection")
            .unwrap();
        if let ScalarValue::Utf8(Some(json_str)) = result {
            let parsed: serde_json::Value = serde_json::from_str(json_str.as_str()).unwrap();
            assert_eq!(parsed["type"], "FeatureCollection");
            assert!(parsed["features"].is_array());
            let features = parsed["features"].as_array().unwrap();
            assert_eq!(features.len(), 1);
            assert_eq!(features[0]["type"], "Feature");
            assert_eq!(features[0]["geometry"]["type"], "Point");
        } else {
            panic!("Expected Utf8 scalar value");
        }
    }

    #[test]
    fn invalid_geojson_type() {
        let udf = st_asgeojson_udf();
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![WKB_GEOMETRY, SedonaType::Arrow(DataType::Utf8)],
        );

        // Test with invalid type
        let result = tester.invoke_scalar_scalar("POINT (1 2)", "InvalidType");

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Invalid GeoJSON type"));
    }
}
