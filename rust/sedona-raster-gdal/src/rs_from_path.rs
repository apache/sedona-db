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

use arrow::compute::cast;
use arrow_array::{Array, ArrayRef, StructArray};
use arrow_schema::DataType;
use datafusion_common::cast::as_string_array;
use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_common::exec_datafusion_err;
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_gdal::gdal::Gdal;
use sedona_gdal::gdal_dyn_bindgen::{GDAL_OF_RASTER, GDAL_OF_READONLY};
use sedona_gdal::raster::types::DatasetOptions;
use sedona_gdal::spatial_ref::SpatialRef;
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::{BandMetadata, RasterMetadata};
use sedona_schema::datatypes::{SedonaType, RASTER};
use sedona_schema::matchers::ArgMatcher;
use sedona_schema::raster::StorageType;

use crate::gdal_common::{
    gdal_to_band_data_type, nodata_f64_to_bytes, normalize_outdb_source_path, with_gdal,
};
use crate::gdal_dataset_provider::configure_thread_local_options;

pub fn rs_from_path_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_frompath",
        vec![Arc::new(RsFromPath)],
        Volatility::Volatile,
    )
}

#[derive(Debug)]
pub(crate) struct RsFromPath;

impl RsFromPath {
    fn load_outdb_raster(gdal: &Gdal, path: &str) -> Result<StructArray> {
        let gdal_path = normalize_outdb_source_path(path);
        let dataset = gdal
            .open_ex_with_options(
                &gdal_path,
                DatasetOptions {
                    open_flags: GDAL_OF_RASTER | GDAL_OF_READONLY,
                    ..Default::default()
                },
            )
            .map_err(|e| {
                exec_datafusion_err!(
                    "Failed to open raster file '{}'(GDAL path '{}'): {}",
                    path,
                    gdal_path,
                    e
                )
            })?;

        let (width, height) = dataset.raster_size();
        let geotransform = dataset
            .geo_transform()
            .map_err(|e| exec_datafusion_err!("Failed to get geotransform: {}", e))?;

        let metadata = RasterMetadata {
            width: width as u64,
            height: height as u64,
            upperleft_x: geotransform[0],
            upperleft_y: geotransform[3],
            scale_x: geotransform[1],
            scale_y: geotransform[5],
            skew_x: geotransform[2],
            skew_y: geotransform[4],
        };

        let crs = dataset
            .spatial_ref()
            .ok()
            .and_then(|sr: SpatialRef| sr.to_projjson().ok());

        let mut builder = RasterBuilder::new(1);
        builder
            .start_raster(&metadata, crs.as_deref())
            .map_err(|e| exec_datafusion_err!("Failed to start raster: {}", e))?;

        let band_count = dataset.raster_count();
        for band_idx in 1..=band_count {
            let band = dataset
                .rasterband(band_idx)
                .map_err(|e| exec_datafusion_err!("Failed to get band {}: {}", band_idx, e))?;

            let gdal_type = band.band_type();
            let band_data_type = gdal_to_band_data_type(gdal_type)
                .map_err(|_| exec_datafusion_err!("Unsupported band data type: {:?}", gdal_type))?;

            let nodata_bytes = band
                .no_data_value()
                .map(|no_data| nodata_f64_to_bytes(no_data, &band_data_type));

            let band_metadata = BandMetadata {
                nodata_value: nodata_bytes,
                storage_type: StorageType::OutDbRef,
                datatype: band_data_type,
                outdb_url: Some(path.to_string()),
                outdb_band_id: Some(band_idx as u32),
            };

            builder
                .start_band(band_metadata)
                .map_err(|e| exec_datafusion_err!("Failed to start band: {}", e))?;

            builder.band_data_writer().append_value([]);

            builder
                .finish_band()
                .map_err(|e| exec_datafusion_err!("Failed to finish band: {}", e))?;
        }

        builder
            .finish_raster()
            .map_err(|e| exec_datafusion_err!("Failed to finish raster: {}", e))?;

        builder
            .finish()
            .map_err(|e| exec_datafusion_err!("Failed to build raster: {}", e))
    }
}

impl SedonaScalarKernel for RsFromPath {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        ArgMatcher::new(vec![ArgMatcher::is_string()], RASTER).match_args(args)
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

            let paths = match &args[0] {
                ColumnarValue::Scalar(scalar) => scalar.to_array().map_err(|e| {
                    exec_datafusion_err!("Failed to convert scalar to array: {}", e)
                })?,
                ColumnarValue::Array(array) => array.clone(),
            };

            let paths = cast(&paths, &DataType::Utf8)?;
            let path_array = as_string_array(&paths)?;

            let len = path_array.len();
            if len == 0 {
                let builder = RasterBuilder::new(0);
                let result = builder
                    .finish()
                    .map_err(|e| exec_datafusion_err!("Failed to build empty raster: {}", e))?;
                return Ok(ColumnarValue::Array(Arc::new(result)));
            }

            let mut combined_arrays: Vec<ArrayRef> = Vec::with_capacity(len);
            for i in 0..len {
                if path_array.is_null(i) {
                    let mut builder = RasterBuilder::new(1);
                    builder
                        .append_null()
                        .map_err(|e| exec_datafusion_err!("Failed to append null: {}", e))?;
                    let result = builder
                        .finish()
                        .map_err(|e| exec_datafusion_err!("Failed to build null raster: {}", e))?;
                    combined_arrays.push(Arc::new(result));
                } else {
                    let path = path_array.value(i);
                    let raster = Self::load_outdb_raster(gdal, path)?;
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

    #[test]
    fn udf_from_path() {
        let udf: datafusion_expr::ScalarUDF = rs_from_path_udf().into();
        assert_eq!(udf.name(), "rs_frompath");
    }

    #[test]
    fn test_load_outdb_raster_from_file() {
        use sedona_testing::data::test_raster;

        let path = test_raster("test4.tiff").expect("test4.tiff should exist");

        let raster = with_gdal(|gdal| RsFromPath::load_outdb_raster(gdal, &path))
            .expect("Should load raster from path");

        assert_eq!(raster.len(), 1);

        use datafusion_common::cast::{
            as_list_array, as_string_array, as_string_view_array, as_struct_array, as_uint32_array,
            as_uint64_array,
        };
        use sedona_schema::raster::{
            band_indices, band_metadata_indices, metadata_indices, raster_indices,
        };

        let metadata_struct = as_struct_array(raster.column(raster_indices::METADATA)).unwrap();
        let width = as_uint64_array(metadata_struct.column(metadata_indices::WIDTH))
            .unwrap()
            .value(0);
        let height = as_uint64_array(metadata_struct.column(metadata_indices::HEIGHT))
            .unwrap()
            .value(0);

        assert_eq!(width, 10);
        assert_eq!(height, 10);

        let crs = as_string_view_array(raster.column(raster_indices::CRS)).unwrap();
        assert!(!crs.is_null(0));

        let bands_list = as_list_array(raster.column(raster_indices::BANDS)).unwrap();
        let bands_struct = as_struct_array(bands_list.values()).unwrap();
        let band_metadata_struct =
            as_struct_array(bands_struct.column(band_indices::METADATA)).unwrap();

        let outdb_url =
            as_string_array(band_metadata_struct.column(band_metadata_indices::OUTDB_URL)).unwrap();
        assert!(!outdb_url.is_null(0));
        assert!(outdb_url.value(0).contains("test4.tiff"));

        let storage_type =
            as_uint32_array(band_metadata_struct.column(band_metadata_indices::STORAGE_TYPE))
                .unwrap();
        assert_eq!(
            storage_type.value(0),
            sedona_schema::raster::StorageType::OutDbRef as u32
        );
    }

    #[test]
    fn test_invoke_rs_from_path() {
        use arrow_array::StringArray;
        use datafusion_common::cast::{as_struct_array, as_uint64_array};
        use sedona_expr::scalar_udf::SedonaScalarKernel;
        use sedona_schema::raster::{metadata_indices, raster_indices};
        use sedona_testing::data::test_raster;

        let path = test_raster("test4.tiff").expect("test4.tiff should exist");

        let paths = Arc::new(StringArray::from(vec![path.as_str()]));
        let input = ColumnarValue::Array(paths);

        let kernel = RsFromPath;
        let result = kernel
            .invoke_batch_from_args(&[], &[input], &SedonaType::Arrow(DataType::Null), 0, None)
            .expect("Should invoke successfully");

        match result {
            ColumnarValue::Array(arr) => {
                let struct_arr = as_struct_array(&arr).unwrap();
                assert_eq!(struct_arr.len(), 1);

                let metadata_struct =
                    as_struct_array(struct_arr.column(raster_indices::METADATA)).unwrap();
                let width = as_uint64_array(metadata_struct.column(metadata_indices::WIDTH))
                    .unwrap()
                    .value(0);
                let height = as_uint64_array(metadata_struct.column(metadata_indices::HEIGHT))
                    .unwrap()
                    .value(0);

                assert_eq!(width, 10);
                assert_eq!(height, 10);
            }
            _ => panic!("Expected array result"),
        }
    }
}
