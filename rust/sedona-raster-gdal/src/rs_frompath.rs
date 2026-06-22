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

//! RS_FromPath UDF - Load out-db raster from file path.

use std::sync::Arc;

use arrow_array::Array;
use arrow_schema::DataType;
use datafusion_common::cast::as_string_array;
use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_functions::executor::WkbBytesExecutor;
use sedona_raster::builder::RasterBuilder;
use sedona_schema::datatypes::{SedonaType, RASTER};
use sedona_schema::matchers::ArgMatcher;

use crate::gdal_common::with_gdal;
use crate::gdal_dataset_provider::configure_thread_local_options;
use crate::utils::append_as_outdb_raster;

pub fn rs_frompath_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_frompath",
        vec![Arc::new(RsFromPath)],
        Volatility::Volatile,
    )
}

#[derive(Debug)]
pub(crate) struct RsFromPath;

impl SedonaScalarKernel for RsFromPath {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        ArgMatcher::new(vec![ArgMatcher::is_string()], RASTER).match_args(args)
    }

    fn invoke_batch_from_args(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
        _return_type: &SedonaType,
        _num_rows: usize,
        config_options: Option<&ConfigOptions>,
    ) -> Result<ColumnarValue> {
        with_gdal(|gdal| {
            configure_thread_local_options(gdal, config_options)?;
            let executor = WkbBytesExecutor::new(arg_types, args);

            let paths = args[0]
                .cast_to(&DataType::Utf8, None)?
                .into_array_of_size(executor.num_iterations())?;
            let path_array = as_string_array(&paths)?;

            let mut builder = RasterBuilder::new(path_array.len());
            for path_opt in path_array {
                if let Some(path) = path_opt {
                    append_as_outdb_raster(gdal, path, &mut builder)?;
                } else {
                    builder.append_null()?;
                }
            }

            let result: Arc<dyn Array> = Arc::new(builder.finish()?);
            executor.finish(result)
        })
    }

    fn invoke_batch(
        &self,
        _arg_types: &[SedonaType],
        _args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        sedona_internal_err!("Should not be called because invoke_batch_from_args() is implemented")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{StringArray, StructArray};
    use datafusion_common::cast::as_struct_array;
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDFImpl;
    use sedona_raster::array::RasterStructArray;
    use sedona_raster::traits::RasterRef;
    use sedona_testing::data::test_raster;

    #[test]
    fn test_rs_from_path_udf_name() {
        assert_eq!(rs_frompath_udf().name(), "rs_frompath");
    }

    fn assert_raster_dimensions(
        result: &ColumnarValue,
        expected_len: usize,
        width: i64,
        height: i64,
    ) {
        fn assert_struct_array_dimensions(
            struct_arr: &StructArray,
            expected_len: usize,
            width: i64,
            height: i64,
        ) {
            let raster_array = RasterStructArray::try_new(struct_arr).unwrap();
            assert_eq!(raster_array.len(), expected_len);

            for idx in 0..expected_len {
                let raster = raster_array.get(idx).unwrap();
                assert_eq!(raster.metadata().width(), width);
                assert_eq!(raster.metadata().height(), height);
            }
        }

        match result {
            ColumnarValue::Array(arr) => {
                let struct_arr = as_struct_array(arr).unwrap();
                assert_struct_array_dimensions(struct_arr, expected_len, width, height);
            }
            ColumnarValue::Scalar(ScalarValue::Struct(struct_arr)) => {
                assert_struct_array_dimensions(struct_arr, expected_len, width, height);
            }
            other => panic!("Unexpected result: {other:?}"),
        }
    }

    #[test]
    fn test_invoke_rs_from_path() {
        let path = test_raster("test4.tiff").expect("test4.tiff should exist");

        let paths = Arc::new(StringArray::from(vec![path.as_str()]));
        let input = ColumnarValue::Array(paths);

        let kernel = RsFromPath;
        let result = kernel
            .invoke_batch_from_args(&[], &[input], &SedonaType::Arrow(DataType::Null), 0, None)
            .expect("Should invoke successfully");

        assert_raster_dimensions(&result, 1, 10, 10);

        let scalar_input = ColumnarValue::Scalar(ScalarValue::Utf8(Some(path.clone())));
        let scalar_result = kernel
            .invoke_batch_from_args(
                &[],
                &[scalar_input],
                &SedonaType::Arrow(DataType::Null),
                0,
                None,
            )
            .expect("Should invoke successfully for scalar path");

        assert_raster_dimensions(&scalar_result, 1, 10, 10);

        let multi_paths = Arc::new(StringArray::from(vec![path.as_str(), path.as_str()]));
        let multi_result = kernel
            .invoke_batch_from_args(
                &[],
                &[ColumnarValue::Array(multi_paths)],
                &SedonaType::Arrow(DataType::Null),
                0,
                None,
            )
            .expect("Should invoke successfully for multiple paths");

        assert_raster_dimensions(&multi_result, 2, 10, 10);

        let empty_paths = Arc::new(StringArray::from(Vec::<&str>::new()));
        let empty_result = kernel
            .invoke_batch_from_args(
                &[],
                &[ColumnarValue::Array(empty_paths)],
                &SedonaType::Arrow(DataType::Null),
                0,
                None,
            )
            .expect("Should invoke successfully for empty paths");

        match empty_result {
            ColumnarValue::Array(arr) => {
                let struct_arr = as_struct_array(&arr).unwrap();
                assert_eq!(struct_arr.len(), 0);
            }
            other => panic!("Expected empty array result, got {other:?}"),
        }
    }

    #[test]
    fn test_invoke_rs_from_path_propagates_nulls() {
        let path = test_raster("test4.tiff").expect("test4.tiff should exist");

        let input =
            ColumnarValue::Array(Arc::new(StringArray::from(vec![Some(path.as_str()), None])));

        let result = RsFromPath
            .invoke_batch_from_args(&[], &[input], &SedonaType::Arrow(DataType::Null), 0, None)
            .expect("Should invoke successfully for null-containing input");

        match result {
            ColumnarValue::Array(arr) => {
                let struct_arr = as_struct_array(&arr).unwrap();
                assert_eq!(struct_arr.len(), 2);
                assert!(!struct_arr.is_null(0));
                assert!(struct_arr.is_null(1));

                let raster_array = RasterStructArray::try_new(struct_arr).unwrap();
                let raster = raster_array.get(0).unwrap();
                assert_eq!(raster.metadata().width(), 10);
                assert_eq!(raster.metadata().height(), 10);
            }
            other => panic!("Expected array result, got {other:?}"),
        }
    }

    #[test]
    fn test_invoke_rs_from_path_invalid_path_errors() {
        let missing_path = "/definitely/missing/rs_from_path_test.tif";
        let input = ColumnarValue::Scalar(ScalarValue::Utf8(Some(missing_path.to_string())));

        let err = RsFromPath
            .invoke_batch_from_args(&[], &[input], &SedonaType::Arrow(DataType::Null), 0, None)
            .expect_err("Missing path should return an error");

        let err_message = err.to_string();
        assert!(err_message.contains(&format!(
            "Failed to open raster file '{}' (GDAL path '{}')",
            missing_path, missing_path
        )));
    }

    #[test]
    fn test_invoke_rs_from_path_scalar_ignores_num_rows_for_shape() {
        let path = test_raster("test4.tiff").expect("test4.tiff should exist");

        let result = RsFromPath
            .invoke_batch_from_args(
                &[],
                &[ColumnarValue::Scalar(ScalarValue::Utf8(Some(path)))],
                &SedonaType::Arrow(DataType::Null),
                32,
                None,
            )
            .expect("Should invoke successfully for scalar path with larger num_rows");

        assert!(matches!(result, ColumnarValue::Scalar(_)));
        assert_raster_dimensions(&result, 1, 10, 10);
    }
}
