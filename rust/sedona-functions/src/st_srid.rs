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
use arrow_array::{
    builder::{StringViewBuilder, UInt32Builder},
    Array,
};
use arrow_schema::DataType;
use datafusion_common::{
    cast::{as_string_view_array, as_struct_array},
    DataFusionError, Result, ScalarValue,
};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_schema::crs::CachedCrsToSRIDMapping;
use sedona_schema::datatypes::SedonaType;
use sedona_schema::matchers::ArgMatcher;
use std::{iter::zip, sync::Arc};

/// ST_Srid() scalar UDF implementation
///
/// Scalar function to return the SRID of a geometry or geography
pub fn st_srid_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_srid",
        vec![Arc::new(StSridItemCrs {}), Arc::new(StSrid {})],
        Volatility::Immutable,
    )
}

/// ST_Crs() scalar UDF implementation
///
/// Scalar function to return the CRS of a geometry or geography
pub fn st_crs_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_crs",
        vec![Arc::new(StCrsItemCrs {}), Arc::new(StCrs {})],
        Volatility::Immutable,
    )
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
struct StSridItemCrs {}

impl SedonaScalarKernel for StSridItemCrs {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_item_crs()],
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

        let item_crs_struct_array = match &args[0] {
            ColumnarValue::Array(array) => as_struct_array(array)?,
            ColumnarValue::Scalar(ScalarValue::Struct(struct_array)) => struct_array.as_ref(),
            ColumnarValue::Scalar(ScalarValue::Null) => {
                return Ok(ColumnarValue::Scalar(ScalarValue::UInt32(None)));
            }
            _ => return sedona_internal_err!("Unexpected input to ST_SRID"),
        };

        let item_array = item_crs_struct_array.column(0);
        let crs_string_array = as_string_view_array(item_crs_struct_array.column(1))?;
        let mut crs_to_srid_mapping = CachedCrsToSRIDMapping::with_capacity(item_array.len());

        if let Some(item_nulls) = item_array.nulls() {
            for (is_valid, maybe_crs) in zip(item_nulls, crs_string_array) {
                if !is_valid {
                    builder.append_null();
                    continue;
                }

                let srid = crs_to_srid_mapping.get_srid(maybe_crs)?;
                builder.append_value(srid);
            }
        } else {
            for maybe_crs in crs_string_array {
                let srid = crs_to_srid_mapping.get_srid(maybe_crs)?;
                builder.append_value(srid);
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
        let mut builder = StringViewBuilder::with_capacity(executor.num_iterations());
        let crs_opt: Option<String> = match &arg_types[0] {
            SedonaType::Wkb(_, Some(crs)) | SedonaType::WkbView(_, Some(crs)) => {
                Some(crs.to_authority_code()?.unwrap_or_else(|| crs.to_json()))
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

#[derive(Debug)]
struct StCrsItemCrs {}

impl SedonaScalarKernel for StCrsItemCrs {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_item_crs()],
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

        let item_crs_struct_array = match &args[0] {
            ColumnarValue::Array(array) => as_struct_array(array)?,
            ColumnarValue::Scalar(ScalarValue::Struct(struct_array)) => struct_array.as_ref(),
            ColumnarValue::Scalar(ScalarValue::Null) => {
                return Ok(ColumnarValue::Scalar(ScalarValue::Utf8View(None)));
            }
            _ => return sedona_internal_err!("Unexpected input to ST_Crs"),
        };

        let item_array = item_crs_struct_array.column(0);
        let crs_array = item_crs_struct_array.column(1);

        if crs_array.null_count() == 0 && item_crs_struct_array.null_count() == 0 {
            return executor.finish(crs_array.clone());
        }

        // Otherwise we need to build the output. We could potentially do some unioning
        // of null buffers in the case where we have zero NULL crses but some null items.
        let item_nulls = item_array.nulls();
        let crs_string_array = as_string_view_array(crs_array)?;
        let mut builder = StringViewBuilder::with_capacity(executor.num_iterations());

        if let Some(item_nulls) = item_nulls {
            for (is_valid, maybe_crs) in zip(item_nulls, crs_string_array) {
                if is_valid {
                    builder.append_value(maybe_crs.unwrap_or("0"))
                } else {
                    builder.append_null();
                }
            }
        } else {
            for maybe_crs in crs_string_array {
                builder.append_value(maybe_crs.unwrap_or("0"))
            }
        }

        executor.finish(Arc::new(builder.finish()))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use arrow_array::{create_array, ArrayRef};
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use sedona_schema::crs::deserialize_crs;
    use sedona_schema::datatypes::{Edges, WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS};
    use sedona_testing::create::{create_array, create_array_item_crs, create_scalar_item_crs};
    use sedona_testing::testers::ScalarUdfTester;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_srid_udf().into();
        assert_eq!(udf.name(), "st_srid");

        let udf: ScalarUDF = st_crs_udf().into();
        assert_eq!(udf.name(), "st_crs");
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
        let crs = deserialize_crs("EPSG:4837").unwrap();
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
        let crs = deserialize_crs("{}").unwrap();
        let sedona_type = SedonaType::Wkb(Edges::Planar, crs.clone());
        let tester = ScalarUdfTester::new(udf.clone(), vec![sedona_type]);
        let result = tester.invoke_scalar("POINT (0 1)");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("CRS has no SRID"));
    }

    #[test]
    fn udf_item_srid() {
        let tester =
            ScalarUdfTester::new(st_srid_udf().into(), vec![WKB_GEOMETRY_ITEM_CRS.clone()]);
        tester.assert_return_type(DataType::UInt32);

        let result = tester
            .invoke_scalar(create_scalar_item_crs(
                Some("POINT (0 1)"),
                None,
                &WKB_GEOMETRY,
            ))
            .unwrap();
        tester.assert_scalar_result_equals(result, 0);

        let result = tester
            .invoke_scalar(create_scalar_item_crs(
                Some("POINT (0 1)"),
                Some("EPSG:3857"),
                &WKB_GEOMETRY,
            ))
            .unwrap();
        tester.assert_scalar_result_equals(result, 3857);

        let item_crs_array = create_array_item_crs(
            &[
                Some("POINT (0 1)"),
                Some("POINT (2 3)"),
                Some("POINT (4 5)"),
                Some("POINT (6 7)"),
                None,
            ],
            [
                Some("OGC:CRS84"),
                Some("EPSG:3857"),
                Some("EPSG:3857"),
                None,
                None,
            ],
            &WKB_GEOMETRY,
        );
        let expected_srid =
            create_array!(UInt32, [Some(4326), Some(3857), Some(3857), Some(0), None]) as ArrayRef;

        let result = tester.invoke_array(item_crs_array).unwrap();
        assert_eq!(&result, &expected_srid);
    }

    #[test]
    fn udf_srid_wkt() {
        // A WKT CRS carrying an EPSG authority tag resolves to that SRID; an
        // authority-less WKT (e.g. a bespoke LCC) has no SRID.
        const WKT_3857: &str = r#"PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],AUTHORITY["EPSG","4326"]],PROJECTION["Mercator_1SP"],AUTHORITY["EPSG","3857"]]"#;
        const WKT_LCC_NO_AUTHORITY: &str = r#"PROJCS["Custom LCC",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["standard_parallel_1",33],PARAMETER["standard_parallel_2",45],PARAMETER["latitude_of_origin",39],PARAMETER["central_meridian",-96],UNIT["metre",1]]"#;

        let udf: ScalarUDF = st_srid_udf().into();

        // Type-level WKT with an EPSG authority -> that SRID.
        let crs = deserialize_crs(WKT_3857).unwrap();
        let tester = ScalarUdfTester::new(udf.clone(), vec![SedonaType::Wkb(Edges::Planar, crs)]);
        let result = tester.invoke_scalar("POINT (0 1)").unwrap();
        tester.assert_scalar_result_equals(result, 3857_u32);

        // Type-level authority-less WKT -> no SRID -> error.
        let crs = deserialize_crs(WKT_LCC_NO_AUTHORITY).unwrap();
        let tester = ScalarUdfTester::new(udf.clone(), vec![SedonaType::Wkb(Edges::Planar, crs)]);
        let err = tester.invoke_scalar("POINT (0 1)").unwrap_err();
        assert!(err.to_string().contains("CRS has no SRID"), "{err}");

        // Item-level WKT with an EPSG authority -> that SRID.
        let tester = ScalarUdfTester::new(udf.clone(), vec![WKB_GEOMETRY_ITEM_CRS.clone()]);
        let arr = create_array_item_crs(&[Some("POINT (0 1)")], [Some(WKT_3857)], &WKB_GEOMETRY);
        let result = tester.invoke_array(arr).unwrap();
        assert_eq!(
            &result,
            &(create_array!(UInt32, [Some(3857_u32)]) as ArrayRef)
        );

        // Item-level authority-less WKT -> error from the CRS->SRID mapping.
        let arr = create_array_item_crs(
            &[Some("POINT (0 1)")],
            [Some(WKT_LCC_NO_AUTHORITY)],
            &WKB_GEOMETRY,
        );
        let err = tester.invoke_array(arr).unwrap_err();
        assert!(
            err.to_string()
                .contains("Can't extract SRID from item-level CRS"),
            "{err}"
        );
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
        let crs = deserialize_crs("EPSG:4837").unwrap();
        let sedona_type = SedonaType::Wkb(Edges::Planar, crs.clone());
        let tester = ScalarUdfTester::new(udf.clone(), vec![sedona_type.clone()]);
        let result = tester
            .invoke_scalar("POLYGON ((0 0, 1 0, 0 1, 0 0))")
            .unwrap();
        tester.assert_scalar_result_equals(result, "EPSG:4837");

        // Call with an array
        let wkb_array = create_array(
            &[Some("POINT (1 2)"), None, Some("MULTIPOINT (3 4)")],
            &sedona_type,
        );
        let expected = create_array!(Utf8View, [Some("EPSG:4837"), None, Some("EPSG:4837")]);
        assert_eq!(
            &tester.invoke_array(wkb_array).unwrap().as_ref(),
            &expected.as_ref()
        );

        // Test with a CRS but null geom
        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Null);
    }

    #[test]
    fn udf_crs_wkt() {
        const WKT_3857: &str = r#"PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],AUTHORITY["EPSG","4326"]],PROJECTION["Mercator_1SP"],AUTHORITY["EPSG","3857"]]"#;
        const WKT_LCC_NO_AUTHORITY: &str = r#"PROJCS["Custom LCC",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["standard_parallel_1",33],PARAMETER["standard_parallel_2",45],PARAMETER["latitude_of_origin",39],PARAMETER["central_meridian",-96],UNIT["metre",1]]"#;

        let udf: ScalarUDF = st_crs_udf().into();

        // WKT with an EPSG authority -> ST_CRS returns the authority code.
        let crs = deserialize_crs(WKT_3857).unwrap();
        let tester = ScalarUdfTester::new(udf.clone(), vec![SedonaType::Wkb(Edges::Planar, crs)]);
        let result = tester.invoke_scalar("POINT (0 1)").unwrap();
        tester.assert_scalar_result_equals(result, "EPSG:3857");

        // Authority-less WKT -> falls back to the JSON serialization (`to_json`).
        let crs = deserialize_crs(WKT_LCC_NO_AUTHORITY).unwrap();
        let expected = crs.as_ref().unwrap().to_json();
        let tester = ScalarUdfTester::new(udf.clone(), vec![SedonaType::Wkb(Edges::Planar, crs)]);
        let result = tester.invoke_scalar("POINT (0 1)").unwrap();
        tester.assert_scalar_result_equals(result, expected.as_str());
    }

    #[test]
    fn udf_item_crs() {
        let tester = ScalarUdfTester::new(
            st_crs_udf().into(),
            vec![SedonaType::new_item_crs(&WKB_GEOMETRY).unwrap()],
        );
        tester.assert_return_type(DataType::Utf8View);

        let result = tester
            .invoke_scalar(create_scalar_item_crs(
                Some("POINT (0 1)"),
                None,
                &WKB_GEOMETRY,
            ))
            .unwrap();
        tester.assert_scalar_result_equals(result, "0");

        let result = tester
            .invoke_scalar(create_scalar_item_crs(
                Some("POINT (0 1)"),
                Some("EPSG:3857"),
                &WKB_GEOMETRY,
            ))
            .unwrap();
        tester.assert_scalar_result_equals(result, "EPSG:3857");

        let item_crs_array = create_array_item_crs(
            &[
                Some("POINT (0 1)"),
                Some("POINT (2 3)"),
                Some("POINT (4 5)"),
                Some("POINT (6 7)"),
                None,
            ],
            [
                Some("OGC:CRS84"),
                Some("EPSG:3857"),
                Some("EPSG:3857"),
                None,
                None,
            ],
            &WKB_GEOMETRY,
        );
        let expected_crs = create_array!(
            Utf8View,
            [
                Some("OGC:CRS84"),
                Some("EPSG:3857"),
                Some("EPSG:3857"),
                Some("0"),
                None
            ]
        ) as ArrayRef;

        let result = tester.invoke_array(item_crs_array).unwrap();
        assert_eq!(&result, &expected_crs);
    }
}
