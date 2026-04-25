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

//! RS_FromGDALRaster UDF - Parse binary content using GDAL driver as in-db raster
//!
//! Similar to PostGIS's ST_FromGDALRaster. Parses binary content using GDAL driver
//! and loads it as an in-db raster with all band data stored inline.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, StructArray};
use arrow_schema::DataType;
use datafusion_common::cast::as_binary_array;
use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_common::exec_datafusion_err;
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_gdal::gdal::Gdal;
use sedona_raster::builder::RasterBuilder;
use sedona_schema::datatypes::{SedonaType, RASTER};
use sedona_schema::matchers::ArgMatcher;

use crate::gdal_common::with_gdal;
use crate::gdal_dataset_provider::configure_thread_local_options;

/// Counter for generating unique VSI memory file names
static VSI_FILE_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// RS_FromGDALRaster() scalar UDF implementation
///
/// Parse binary content using GDAL driver and load it as in-db raster
pub fn rs_from_gdal_raster_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_fromgdalraster",
        vec![Arc::new(RsFromGDALRaster)],
        Volatility::Immutable,
    )
}

/// Kernel implementation for RS_FromGDALRaster
#[derive(Debug)]
pub(crate) struct RsFromGDALRaster;

impl RsFromGDALRaster {
    /// Generate a unique VSI memory file path
    fn generate_vsi_path() -> String {
        let counter = VSI_FILE_COUNTER.fetch_add(1, Ordering::SeqCst);
        let thread_id = std::thread::current().id();
        format!(
            "/vsimem/rs_from_gdal_raster_{:?}_{}.bin",
            thread_id, counter
        )
    }

    /// Parse binary content and create an in-db raster
    pub(crate) fn parse_gdal_raster(gdal: &Gdal, content: &[u8]) -> Result<StructArray> {
        // Create a temporary VSI memory file
        let vsi_path = Self::generate_vsi_path();
        let content_copy = content.to_vec();

        // Write content to VSI memory file
        gdal.create_mem_file(&vsi_path, &content_copy)
            .map_err(|e| exec_datafusion_err!("Failed to create VSI memory file: {}", e))?;

        // Delegate to load_as_indb_raster, then always clean up
        let result = crate::utils::load_as_indb_raster(gdal, &vsi_path);
        let _ = gdal.unlink_mem_file(&vsi_path);
        result
    }
}

impl SedonaScalarKernel for RsFromGDALRaster {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(vec![ArgMatcher::is_binary()], RASTER);
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        self.invoke_batch_from_args(arg_types, args, &SedonaType::Arrow(DataType::Null), 0, None)
    }

    fn invoke_batch_from_args(
        &self,
        _arg_types: &[SedonaType],
        args: &[ColumnarValue],
        _return_type: &SedonaType,
        _num_rows: usize,
        config_options: Option<&ConfigOptions>,
    ) -> Result<ColumnarValue> {
        with_gdal(|gdal| {
            configure_thread_local_options(gdal, config_options)?;

            let content_array = match &args[0] {
                ColumnarValue::Scalar(scalar) => scalar.to_array().map_err(|e| {
                    exec_datafusion_err!("Failed to convert scalar to array: {}", e)
                })?,
                ColumnarValue::Array(array) => array.clone(),
            };

            let binary_array = as_binary_array(&content_array)?;

            let len = binary_array.len();

            if len == 0 {
                let builder = RasterBuilder::new(0);
                let result = builder
                    .finish()
                    .map_err(|e| exec_datafusion_err!("Failed to build empty raster: {}", e))?;
                return Ok(ColumnarValue::Array(Arc::new(result)));
            }

            let mut combined_arrays: Vec<ArrayRef> = Vec::with_capacity(len);

            for i in 0..len {
                if binary_array.is_null(i) {
                    let mut builder = RasterBuilder::new(1);
                    builder
                        .append_null()
                        .map_err(|e| exec_datafusion_err!("Failed to append null: {}", e))?;
                    let result = builder
                        .finish()
                        .map_err(|e| exec_datafusion_err!("Failed to build null raster: {}", e))?;
                    combined_arrays.push(Arc::new(result));
                } else {
                    let content = binary_array.value(i);
                    let raster = Self::parse_gdal_raster(gdal, content)?;
                    combined_arrays.push(Arc::new(raster));
                }
            }

            let refs: Vec<&dyn Array> = combined_arrays.iter().map(|a| a.as_ref()).collect();
            let result = arrow::compute::concat(&refs)
                .map_err(|e| exec_datafusion_err!("Failed to concatenate rasters: {}", e))?;

            match &args[0] {
                ColumnarValue::Scalar(_) => {
                    let scalar = datafusion_common::ScalarValue::try_from_array(&result, 0)?;
                    Ok(ColumnarValue::Scalar(scalar))
                }
                ColumnarValue::Array(_) => Ok(ColumnarValue::Array(result)),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gdal_common::with_gdal;
    use datafusion_common::cast::as_struct_array;
    use sedona_raster::traits::RasterRef;

    #[test]
    fn test_generate_vsi_path() {
        let path1 = RsFromGDALRaster::generate_vsi_path();
        let path2 = RsFromGDALRaster::generate_vsi_path();

        assert!(path1.starts_with("/vsimem/rs_from_gdal_raster_"));
        assert!(path2.starts_with("/vsimem/rs_from_gdal_raster_"));
        assert_ne!(path1, path2);
    }

    #[test]
    fn udf_from_gdal_raster() {
        let udf: datafusion_expr::ScalarUDF = rs_from_gdal_raster_udf().into();
        assert_eq!(udf.name(), "rs_fromgdalraster");
    }

    #[test]
    fn test_parse_geotiff_bytes() {
        use sedona_raster::array::RasterStructArray;
        use sedona_testing::data::test_raster;

        // Read test4.tiff file into bytes
        let path = test_raster("test4.tiff").expect("test4.tiff should exist");
        let content = std::fs::read(&path).expect("Should read file");

        // Parse the GeoTiff bytes into a raster
        let result = with_gdal(|gdal| RsFromGDALRaster::parse_gdal_raster(gdal, &content))
            .expect("Should parse GeoTiff bytes");

        // Verify the raster
        let raster_array = RasterStructArray::new(&result);
        assert_eq!(raster_array.len(), 1);

        let raster = raster_array.get(0).expect("Should get raster");
        assert_eq!(raster.metadata().width(), 10);
        assert_eq!(raster.metadata().height(), 10);
        assert_eq!(raster.bands().len(), 1);
        // Check CRS - test4.tiff has EPSG:4326
        assert!(raster.crs().is_some());

        // Verify it's an in-db raster (should have band data, not outdb_url)
        let band = raster.bands().band(1).expect("Should have band 1");
        assert!(
            band.metadata().outdb_url().is_none(),
            "In-db raster should not have outdb_url"
        );
    }

    #[test]
    fn test_invoke_rs_from_gdal_raster() {
        use arrow_array::BinaryArray;
        use sedona_expr::scalar_udf::SedonaScalarKernel;
        use sedona_testing::data::test_raster;

        // Read test file into bytes
        let path = test_raster("test4.tiff").expect("test4.tiff should exist");
        let content = std::fs::read(&path).expect("Should read file");

        // Create binary array with the content
        let binary_arr = Arc::new(BinaryArray::from(vec![content.as_slice()]));
        let input = ColumnarValue::Array(binary_arr);

        // Invoke the UDF
        let kernel = RsFromGDALRaster;
        let result = kernel
            .invoke_batch_from_args(&[], &[input], &SedonaType::Arrow(DataType::Null), 0, None)
            .expect("Should invoke successfully");

        // Verify result
        match result {
            ColumnarValue::Array(arr) => {
                let struct_arr = as_struct_array(&arr).unwrap();
                let raster_array = sedona_raster::array::RasterStructArray::new(struct_arr);
                assert_eq!(raster_array.len(), 1);
                let raster = raster_array.get(0).expect("Should get raster");
                assert_eq!(raster.metadata().width(), 10);
                assert_eq!(raster.metadata().height(), 10);
            }
            _ => panic!("Expected array result"),
        }
    }
}
