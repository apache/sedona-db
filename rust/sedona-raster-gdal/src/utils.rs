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

//! Utility functions for loading raster data via GDAL.

use arrow_array::StructArray;
use datafusion_common::error::Result;
use datafusion_common::exec_datafusion_err;
use sedona_gdal::dataset::Dataset;
use sedona_gdal::gdal::Gdal;
use sedona_gdal::gdal_dyn_bindgen::{GDAL_OF_RASTER, GDAL_OF_READONLY};
use sedona_gdal::raster::types::DatasetOptions;
use sedona_gdal::spatial_ref::SpatialRef;

use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::{BandMetadata, RasterMetadata};
use sedona_schema::raster::{BandDataType, StorageType};

use crate::gdal_common::{gdal_to_band_data_type, nodata_f64_to_bytes};

/// Load a raster from any GDAL-openable path as an in-db raster `StructArray`.
///
/// The `path` can be a regular file path, a `/vsimem/` memory path,
/// a `/vsicurl/` URL, or any other GDAL virtual filesystem path.
pub fn load_as_indb_raster(gdal: &Gdal, path: &str) -> Result<StructArray> {
    // Open dataset from path
    let dataset = gdal
        .open_ex_with_options(
            path,
            DatasetOptions {
                open_flags: GDAL_OF_RASTER | GDAL_OF_READONLY,
                ..Default::default()
            },
        )
        .map_err(|e| exec_datafusion_err!("Failed to open raster from {}: {}", path, e))?;

    // Get raster dimensions
    let (width, height) = dataset.raster_size();

    // Get geotransform
    let geotransform = dataset
        .geo_transform()
        .map_err(|e| exec_datafusion_err!("Failed to get geotransform: {}", e))?;

    // Build RasterMetadata
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

    // Get CRS as PROJJSON if available
    let crs = dataset
        .spatial_ref()
        .ok()
        .and_then(|sr: SpatialRef| sr.to_projjson().ok());

    // Build the raster array
    let mut builder = RasterBuilder::new(1);
    builder
        .start_raster(&metadata, crs.as_deref())
        .map_err(|e| exec_datafusion_err!("Failed to start raster: {}", e))?;

    // Add bands with in-db data
    let band_count = dataset.raster_count();
    for band_idx in 1..=band_count {
        let band = dataset
            .rasterband(band_idx)
            .map_err(|e| exec_datafusion_err!("Failed to get band {}: {}", band_idx, e))?;

        let gdal_type = band.band_type();
        let band_data_type = gdal_to_band_data_type(gdal_type)
            .map_err(|_| exec_datafusion_err!("Unsupported band data type: {:?}", gdal_type))?;

        // Get nodata value
        let nodata_bytes = band
            .no_data_value()
            .map(|no_data| nodata_f64_to_bytes(no_data, &band_data_type));

        let band_metadata = BandMetadata {
            nodata_value: nodata_bytes,
            storage_type: StorageType::InDb,
            datatype: band_data_type,
            outdb_url: None,
            outdb_band_id: None,
        };

        builder
            .start_band(band_metadata)
            .map_err(|e| exec_datafusion_err!("Failed to start band: {}", e))?;

        // Read and store band data
        let band_data = read_band_data(&dataset, band_idx, width, height, band_data_type)?;
        builder.band_data_writer().append_value(&band_data);

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

/// Read band data as bytes from a GDAL dataset.
fn read_band_data(
    dataset: &Dataset,
    band_idx: usize,
    width: usize,
    height: usize,
    band_type: BandDataType,
) -> Result<Vec<u8>> {
    let band = dataset
        .rasterband(band_idx)
        .map_err(|e| exec_datafusion_err!("Failed to get band {}: {}", band_idx, e))?;

    // Read band data based on type
    macro_rules! read_as {
        ($T:ty) => {{
            let buffer = band
                .read_as::<$T>((0, 0), (width, height), (width, height), None)
                .map_err(|e| {
                    exec_datafusion_err!("Failed to read band {} data: {}", band_idx, e)
                })?;
            buffer.data().iter().flat_map(|v| v.to_le_bytes()).collect()
        }};
    }

    let data: Vec<u8> = match band_type {
        BandDataType::UInt8 => read_as!(u8),
        BandDataType::Int8 => read_as!(i8),
        BandDataType::UInt16 => read_as!(u16),
        BandDataType::Int16 => read_as!(i16),
        BandDataType::UInt32 => read_as!(u32),
        BandDataType::Int32 => read_as!(i32),
        BandDataType::UInt64 => read_as!(u64),
        BandDataType::Int64 => read_as!(i64),
        BandDataType::Float32 => read_as!(f32),
        BandDataType::Float64 => read_as!(f64),
    };

    Ok(data)
}
