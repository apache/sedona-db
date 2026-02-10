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
use std::{sync::Arc, vec};

use crate::executor::RasterExecutor;
use arrow_array::builder::StringBuilder;
use arrow_array::builder::UInt32Builder;
use arrow_schema::DataType;
use datafusion_common::error::Result;
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::traits::RasterRef;
use sedona_schema::crs::CachedCrsToSRIDMapping;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

/// RS_SRID() scalar UDF implementation
///
/// Extract the SRID (Spatial Reference ID) of the raster
pub fn rs_srid_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_srid",
        vec![Arc::new(RsSrid {})],
        Volatility::Immutable,
        Some(rs_srid_doc()),
    )
}

/// RS_CRS() scalar UDF implementation
///
/// Extract the CRS (Coordinate Reference System) of the raster
pub fn rs_crs_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_crs",
        vec![Arc::new(RsCrs {})],
        Volatility::Immutable,
        Some(rs_crs_doc()),
    )
}

fn rs_srid_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Return the spatial reference system identifier (SRID) of the raster".to_string(),
        "RS_SRID(raster: Raster)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_sql_example("SELECT RS_SRID(RS_Example())".to_string())
    .build()
}

fn rs_crs_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Return the coordinate reference system (CRS) of the raster".to_string(),
        "RS_CRS(raster: Raster)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_sql_example("SELECT RS_CRS(RS_Example())".to_string())
    .build()
}

#[derive(Debug)]
struct RsSrid {}

impl SedonaScalarKernel for RsSrid {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_raster()],
            SedonaType::Arrow(DataType::UInt32),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let mut builder = UInt32Builder::with_capacity(executor.num_iterations());

        let mut crs_to_srid_mapping =
            CachedCrsToSRIDMapping::with_capacity(executor.num_iterations());
        executor.execute_raster_void(|_i, raster_opt| {
            let Some(raster) = raster_opt else {
                builder.append_null();
                return Ok(());
            };

            let maybe_crs = raster.crs_str_ref();
            let srid = crs_to_srid_mapping.get_srid(maybe_crs)?;
            builder.append_value(srid);
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

#[derive(Debug)]
struct RsCrs {}

impl SedonaScalarKernel for RsCrs {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_raster()],
            SedonaType::Arrow(DataType::Utf8),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let preallocate_bytes = "EPSG:4326".len() * executor.num_iterations();
        let mut builder =
            StringBuilder::with_capacity(executor.num_iterations(), preallocate_bytes);

        executor.execute_raster_void(|_i, raster_opt| {
            let Some(raster) = raster_opt else {
                builder.append_null();
                return Ok(());
            };

            // This is similar to ST_CRS: if no CRS is set, return "0"
            let crs_str = raster.crs().unwrap_or("0");
            builder.append_value(crs_str);
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{StringArray, UInt32Array};
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use sedona_raster::builder::RasterBuilder;
    use sedona_raster::traits::{BandMetadata, RasterMetadata};
    use sedona_schema::datatypes::RASTER;
    use sedona_schema::raster::{BandDataType, StorageType};
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::rasters::generate_test_rasters;
    use sedona_testing::testers::ScalarUdfTester;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = rs_srid_udf().into();
        assert_eq!(udf.name(), "rs_srid");
        assert!(udf.documentation().is_some());

        let udf: ScalarUDF = rs_crs_udf().into();
        assert_eq!(udf.name(), "rs_crs");
        assert!(udf.documentation().is_some());
    }

    #[test]
    fn udf_srid() {
        let udf: ScalarUDF = rs_srid_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);

        tester.assert_return_type(DataType::UInt32);

        // Test with rasters that have CRS set (generate_test_rasters sets OGC:CRS84 which maps to 4326)
        let rasters = generate_test_rasters(3, Some(1)).unwrap();
        let expected: Arc<dyn arrow_array::Array> =
            Arc::new(UInt32Array::from(vec![Some(4326), None, Some(4326)]));

        let result = tester.invoke_array(Arc::new(rasters)).unwrap();
        assert_array_equal(&result, &expected);

        // Test with null scalar
        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::UInt32(None));

        // Test with raster missing CRS
        let mut builder = RasterBuilder::new(1);
        append_1x1_raster_with_crs(&mut builder, None);
        let rasters = builder.finish().unwrap();

        let result = tester.invoke_array(Arc::new(rasters)).unwrap();
        let expected: Arc<dyn arrow_array::Array> = Arc::new(UInt32Array::from(vec![Some(0)]));
        assert_array_equal(&result, &expected);
    }

    #[test]
    fn udf_srid_missing_srid_returns_error() {
        let udf: ScalarUDF = rs_srid_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);

        // A PROJJSON CRS without an authority identifier should error.
        let projjson_crs = "{\"type\":\"GeographicCRS\",\"name\":\"No authority id\"}";
        let mut builder = RasterBuilder::new(1);
        append_1x1_raster_with_crs(&mut builder, Some(projjson_crs));
        let rasters = builder.finish().unwrap();

        let err = tester.invoke_array(Arc::new(rasters)).unwrap_err();
        assert!(
            err.to_string()
                .contains("Can't extract SRID from item-level CRS"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn udf_crs() {
        let udf: ScalarUDF = rs_crs_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);

        tester.assert_return_type(DataType::Utf8);

        // Test with rasters that have CRS set (generate_test_rasters sets OGC:CRS84)
        let rasters = generate_test_rasters(3, Some(1)).unwrap();
        let expected_crs = "OGC:CRS84".to_string();
        let expected: Arc<dyn arrow_array::Array> = Arc::new(StringArray::from(vec![
            Some(expected_crs.clone()),
            None,
            Some(expected_crs.clone()),
        ]));

        let result = tester.invoke_array(Arc::new(rasters)).unwrap();
        assert_array_equal(&result, &expected);

        // Test with raster missing CRS
        let mut builder = RasterBuilder::new(1);
        append_1x1_raster_with_crs(&mut builder, None);
        let rasters = builder.finish().unwrap();

        let result = tester.invoke_array(Arc::new(rasters)).unwrap();
        let expected: Arc<dyn arrow_array::Array> = Arc::new(StringArray::from(vec![Some("0")]));
        assert_array_equal(&result, &expected);

        // Test with null scalar
        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Utf8(None));
    }

    fn append_1x1_raster_with_crs(builder: &mut RasterBuilder, crs: Option<&str>) {
        let raster_metadata = RasterMetadata {
            width: 1,
            height: 1,
            upperleft_x: 0.0,
            upperleft_y: 0.0,
            scale_x: 1.0,
            scale_y: -1.0,
            skew_x: 0.0,
            skew_y: 0.0,
        };
        builder.start_raster(&raster_metadata, crs).unwrap();
        builder
            .start_band(BandMetadata {
                datatype: BandDataType::UInt8,
                nodata_value: None,
                storage_type: StorageType::InDb,
                outdb_url: None,
                outdb_band_id: None,
            })
            .unwrap();
        builder.band_data_writer().append_value([0u8]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
    }
}
