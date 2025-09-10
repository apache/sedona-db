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
use crate::executor::WkbExecutor;
use arrow_array::builder::StringBuilder;
use arrow_array::builder::UInt32Builder;
use arrow_array::Array;
use arrow_schema::DataType;
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_schema::datatypes::SedonaType;
use sedona_schema::matchers::ArgMatcher;
use std::{sync::Arc, vec};

/// ST_Srid() scalar UDF implementation
///
/// Scalar function to return the SRID of a geometry or geography
pub fn st_srid_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_srid",
        vec![Arc::new(StSrid {})],
        Volatility::Immutable,
        Some(st_srid_doc()),
    )
}

/// ST_Crs() scalar UDF implementation
///
/// Scalar function to return the CRS of a geometry or geography
pub fn st_crs_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_crs",
        vec![Arc::new(StCrs {})],
        Volatility::Immutable,
        Some(st_crs_doc()),
    )
}

fn st_srid_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Return the spatial reference system identifier (SRID) of the geometry.",
        "ST_SRID (geom: Geometry)",
    )
    .with_argument("geom", "geometry: Input geometry or geography")
    .with_sql_example("SELECT ST_SRID(polygon)".to_string())
    .build()
}

fn st_crs_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Return the coordinate reference system (CRS) of the geometry.",
        "ST_CRS (geom: Geometry)",
    )
    .with_argument("geom", "geometry: Input geometry or geography")
    .with_sql_example("SELECT ST_CRS(polygon)".to_string())
    .build()
}

#[derive(Debug)]
struct StSrid {}

impl SedonaScalarKernel for StSrid {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry_or_geography()],
            SedonaType::Arrow(DataType::UInt32),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = UInt32Builder::with_capacity(executor.num_iterations());
        let srid_opt = match &arg_types[0] {
            SedonaType::Wkb(_, Some(crs)) | SedonaType::WkbView(_, Some(crs)) => {
                match crs.srid()? {
                    Some(srid) => Some(srid),
                    None => return Err(DataFusionError::Execution("CRS has no SRID".to_string())),
                }
            }
            _ => Some(0),
        };

        match &args[0] {
            ColumnarValue::Array(array) => {
                (0..array.len()).for_each(|i| {
                    builder.append_option(if array.is_null(i) { None } else { srid_opt });
                });
            }
            ColumnarValue::Scalar(scalar) => {
                builder.append_option(if scalar.is_null() { None } else { srid_opt });
            }
        }

        executor.finish(Arc::new(builder.finish()))
    }
}

#[derive(Debug)]
struct StCrs {}

impl SedonaScalarKernel for StCrs {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry_or_geography()],
            SedonaType::Arrow(DataType::Utf8View),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let preallocate_bytes = "EPSG:4326".len() * executor.num_iterations();
        let mut builder =
            StringBuilder::with_capacity(executor.num_iterations(), preallocate_bytes);
        let crs_opt: Option<String> = match &arg_types[0] {
            SedonaType::Wkb(_, Some(crs)) | SedonaType::WkbView(_, Some(crs)) => {
                Some(crs.to_json())
            }
            _ => None,
        };

        match &args[0] {
            ColumnarValue::Array(array) => {
                (0..array.len()).for_each(|i| {
                    builder.append_option(if array.is_null(i) {
                        None
                    } else {
                        crs_opt.clone()
                    });
                });
            }
            ColumnarValue::Scalar(scalar) => {
                builder.append_option(if scalar.is_null() { None } else { crs_opt });
            }
        }

        executor.finish(Arc::new(builder.finish()))
    }
}

#[cfg(test)]
mod test {
    use arrow_array::create_array;
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use sedona_schema::crs::deserialize_crs;
    use sedona_schema::datatypes::Edges;
    use sedona_testing::create::create_array;
    use sedona_testing::testers::ScalarUdfTester;
    use std::str::FromStr;

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_srid_udf().into();
        assert_eq!(udf.name(), "st_srid");
        assert!(udf.documentation().is_some());

        let udf: ScalarUDF = st_crs_udf().into();
        assert_eq!(udf.name(), "st_crs");
        assert!(udf.documentation().is_some())
    }

    #[test]
    fn udf_srid() {
        let udf: ScalarUDF = st_srid_udf().into();

        // Test that when no CRS is set, SRID is 0
        let sedona_type = SedonaType::Wkb(Edges::Planar, None);
        let tester = ScalarUdfTester::new(udf.clone(), vec![sedona_type]);
        tester.assert_return_type(DataType::UInt32);
        let result = tester
            .invoke_scalar("POLYGON ((0 0, 1 0, 0 1, 0 0))")
            .unwrap();
        tester.assert_scalar_result_equals(result, 0_u32);

        // Test that NULL input returns NULL output
        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Null);

        // Test with a CRS with an EPSG code
        let crs_value = serde_json::Value::String("EPSG:4837".to_string());
        let crs = deserialize_crs(&crs_value).unwrap();
        let sedona_type = SedonaType::Wkb(Edges::Planar, crs.clone());
        let tester = ScalarUdfTester::new(udf.clone(), vec![sedona_type.clone()]);
        let result = tester
            .invoke_scalar("POLYGON ((0 0, 1 0, 0 1, 0 0))")
            .unwrap();
        tester.assert_scalar_result_equals(result, 4837_u32);

        // Test with a CRS but null geom
        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Null);

        // Call with an array
        let wkb_array = create_array(
            &[Some("POINT (1 2)"), None, Some("MULTIPOINT (3 4)")],
            &sedona_type,
        );
        let expected = create_array!(UInt32, [Some(4837_u32), None, Some(4837_u32)]);
        assert_eq!(
            &tester.invoke_array(wkb_array).unwrap().as_ref(),
            &expected.as_ref()
        );

        // Call with a CRS with no SRID (should error)
        let crs_value = serde_json::Value::from_str("{}");
        let crs = deserialize_crs(&crs_value.unwrap()).unwrap();
        let sedona_type = SedonaType::Wkb(Edges::Planar, crs.clone());
        let tester = ScalarUdfTester::new(udf.clone(), vec![sedona_type]);
        let result = tester.invoke_scalar("POINT (0 1)");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("CRS has no SRID"));
    }

    #[test]
    fn udf_crs() {
        let udf: ScalarUDF = st_crs_udf().into();

        // Test that when no CRS is set, CRS is null
        let sedona_type = SedonaType::Wkb(Edges::Planar, None);
        let tester = ScalarUdfTester::new(udf.clone(), vec![sedona_type]);
        tester.assert_return_type(DataType::Utf8View);
        let result = tester
            .invoke_scalar("POLYGON ((0 0, 1 0, 0 1, 0 0))")
            .unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Utf8(None));

        // Test that NULL input returns NULL output
        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Null);

        // Test with a CRS with an EPSG code
        let crs_value = serde_json::Value::String("EPSG:4837".to_string());
        let crs = deserialize_crs(&crs_value).unwrap();
        let sedona_type = SedonaType::Wkb(Edges::Planar, crs.clone());
        let tester = ScalarUdfTester::new(udf.clone(), vec![sedona_type.clone()]);
        let expected_crs = "\"EPSG:4837\"".to_string();
        let result = tester
            .invoke_scalar("POLYGON ((0 0, 1 0, 0 1, 0 0))")
            .unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Utf8(Some(expected_crs.clone())));

        // Call with an array
        let wkb_array = create_array(
            &[Some("POINT (1 2)"), None, Some("MULTIPOINT (3 4)")],
            &sedona_type,
        );
        let expected = create_array!(
            Utf8,
            [Some(expected_crs.clone()), None, Some(expected_crs.clone())]
        );
        assert_eq!(
            &tester.invoke_array(wkb_array).unwrap().as_ref(),
            &expected.as_ref()
        );

        // Test with a CRS but null geom
        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Null);
    }
}
