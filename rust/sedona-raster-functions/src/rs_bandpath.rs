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
use arrow_array::{cast::AsArray, types::Int32Type, Array};
use arrow_schema::DataType;
use datafusion_common::error::Result;
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::traits::RasterRef;
use sedona_schema::raster::StorageType;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

/// RS_BandPath() scalar UDF implementation
///
/// Returns the path to the raster file referenced by the out-db band.
/// If the band is an in-db band, this function returns null.
/// Accepts an optional band_index parameter (1-based, default is 1).
pub fn rs_bandpath_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_bandpath",
        vec![
            Arc::new(RsBandPath {}),
            Arc::new(RsBandPathWithBandIndex {}),
        ],
        Volatility::Immutable,
        Some(rs_bandpath_doc()),
    )
}

fn rs_bandpath_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the path to the raster file referenced by the out-db band. If the band is an in-db band, this function returns null.".to_string(),
        "RS_BandPath(raster: Raster, band_index: Integer = 1)".to_string(),
    )
    .with_argument("raster", "Raster: Input raster")
    .with_argument("band_index", "Integer: 1-based band index (default is 1)")
    .with_sql_example("SELECT RS_BandPath(RS_FromPath('s3://bucket/path/to/raster.tif'))".to_string())
    .build()
}

/// One-argument kernel: RS_BandPath(raster) - uses band 1 by default
#[derive(Debug)]
struct RsBandPath {}

const PREALLOC_SIZE_PER_PATH: usize = 256;

impl SedonaScalarKernel for RsBandPath {
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

        let preallocate_bytes = PREALLOC_SIZE_PER_PATH * executor.num_iterations();
        let mut builder =
            StringBuilder::with_capacity(executor.num_iterations(), preallocate_bytes);

        executor
            .execute_raster_void(|_i, raster_opt| get_band_path(raster_opt, 1, &mut builder))?;

        executor.finish(Arc::new(builder.finish()))
    }
}

/// Two-argument kernel: RS_BandPath(raster, band_index)
#[derive(Debug)]
struct RsBandPathWithBandIndex {}

impl SedonaScalarKernel for RsBandPathWithBandIndex {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_raster(), ArgMatcher::is_integer()],
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

        // Expand the band_index parameter to an array
        let band_index_array = args[1].clone().into_array(executor.num_iterations())?;
        let band_index_array = band_index_array.as_primitive::<Int32Type>();

        let preallocate_bytes = PREALLOC_SIZE_PER_PATH * executor.num_iterations();
        let mut builder =
            StringBuilder::with_capacity(executor.num_iterations(), preallocate_bytes);

        executor.execute_raster_void(|i, raster_opt| {
            let band_index = if band_index_array.is_null(i) {
                1 // Default to band 1 if null
            } else {
                band_index_array.value(i)
            };
            get_band_path(raster_opt, band_index, &mut builder)
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

/// Get the band path for a raster at the specified band index
fn get_band_path(
    raster_opt: Option<&sedona_raster::array::RasterRefImpl<'_>>,
    band_index: i32,
    builder: &mut StringBuilder,
) -> Result<()> {
    match raster_opt {
        None => builder.append_null(),
        Some(raster) => {
            let bands = raster.bands();
            let num_bands = bands.len() as i32;
            if band_index < 1 || band_index > num_bands {
                builder.append_null();
            } else {
                let band = bands.band(band_index as usize)?;
                let band_metadata = band.metadata();

                if band_metadata.storage_type()? == StorageType::OutDbRef {
                    match band_metadata.outdb_url() {
                        Some(url) => builder.append_value(url),
                        None => builder.append_null(),
                    }
                } else {
                    builder.append_null()
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, Int32Array, StringArray};
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use sedona_schema::datatypes::RASTER;
    use sedona_testing::rasters::generate_test_rasters;
    use sedona_testing::testers::ScalarUdfTester;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = rs_bandpath_udf().into();
        assert_eq!(udf.name(), "rs_bandpath");
        assert!(udf.documentation().is_some());
    }

    #[test]
    fn udf_bandpath_indb_rasters_default_band() {
        let udf: ScalarUDF = rs_bandpath_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);

        tester.assert_return_type(DataType::Utf8);

        // Test with in-db rasters - should all return null (default band_index = 1)
        let rasters = generate_test_rasters(3, Some(1)).unwrap();
        let result = tester.invoke_array(Arc::new(rasters)).unwrap();

        let string_array = result
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("Expected StringArray");

        // All in-db rasters should return null
        assert!(string_array.is_null(0));
        assert!(string_array.is_null(1));
        assert!(string_array.is_null(2));
    }

    #[test]
    fn udf_bandpath_indb_rasters_with_band_index() {
        let udf: ScalarUDF = rs_bandpath_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Int32)]);

        tester.assert_return_type(DataType::Utf8);

        // Test with in-db rasters and explicit band index
        let rasters = generate_test_rasters(3, Some(3)).unwrap(); // 3 bands
        let band_indices = Int32Array::from(vec![1, 2, 3]);
        let result = tester
            .invoke_arrays(vec![Arc::new(rasters), Arc::new(band_indices)])
            .unwrap();

        let string_array = result
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("Expected StringArray");

        // All in-db bands should return null regardless of band index
        assert!(string_array.is_null(0));
        assert!(string_array.is_null(1));
        assert!(string_array.is_null(2));
    }

    #[test]
    fn udf_bandpath_invalid_band_index() {
        let udf: ScalarUDF = rs_bandpath_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Int32)]);

        // Test with invalid band indices (out of range)
        let rasters = generate_test_rasters(3, Some(2)).unwrap(); // 2 bands
        let band_indices = Int32Array::from(vec![0, 3, -1]); // All invalid indices
        let result = tester
            .invoke_arrays(vec![Arc::new(rasters), Arc::new(band_indices)])
            .unwrap();

        let string_array = result
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("Expected StringArray");

        // Invalid band indices should return null
        assert!(string_array.is_null(0)); // band 0 is invalid (1-based)
        assert!(string_array.is_null(1)); // band 3 is out of range
        assert!(string_array.is_null(2)); // negative band index is invalid
    }

    #[test]
    fn udf_bandpath_null_scalar() {
        let udf: ScalarUDF = rs_bandpath_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);

        // Test with null scalar
        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Utf8(None));
    }
}
