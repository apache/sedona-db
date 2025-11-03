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

use crate::dataset::{geotransform_components, tile_size, to_banddatatype};
use arrow::array::StructArray;
use arrow::error::ArrowError;
use gdal::raster::RasterBand;
use gdal::Dataset;
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::{BandMetadata, RasterMetadata};
use sedona_raster::utils::{bytes_per_pixel, f64_to_bandtype_bytes};
use sedona_schema::raster::{BandDataType, StorageType};
use std::sync::Arc;

/// Reads a raster file using GDAL and converts it into a StructArray of rasters
/// 
/// # Arguments
/// * `filepath` - Path to the raster file
/// * `tile_width` - Optional tile width to override dataset metadata (uses dataset metadata if None)
/// * `tile_height` - Optional tile height to override dataset metadata (uses dataset metadata if None)
pub fn read_raster_geotiff(
    filepath: &str,
    tile_width: Option<usize>,
    tile_height: Option<usize>,
) -> Result<Arc<StructArray>, ArrowError> {
    // TODO: validate filepath contains a geotiff extension
    // or change names if this just works for other raster types
    // and gdal handles proper failing... test it!
    let dataset = Dataset::open(filepath.to_string())
        .map_err(|err| ArrowError::ParseError(err.to_string()))?;

    // Extract geotransform components
    let (origin_x, origin_y, pixel_width, pixel_height, rotation_x, rotation_y) =
        geotransform_components(&dataset)?;

    // Extract the tile size from dataset metadata
    let (dataset_tile_width, dataset_tile_height) = tile_size(&dataset)?;

    let (raster_width, raster_height) = dataset.raster_size();

    // Use provided tile dimensions or fall back to dataset metadata
    let (tile_width, tile_height) = match (tile_width, tile_height) {
        (Some(w), Some(h)) => (w, h),
        (Some(w), None) => (w, dataset_tile_height),
        (None, Some(h)) => (dataset_tile_width, h),
        (None, None) => (dataset_tile_width, dataset_tile_height),
    };

    let x_tile_count = (raster_width + tile_width - 1) / tile_width;
    let y_tile_count = (raster_height + tile_height - 1) / tile_height;

    let mut raster_builder = RasterBuilder::new(x_tile_count * y_tile_count);
    let band_count = dataset.raster_count();

    for tile_y in 0..y_tile_count {
        for tile_x in 0..x_tile_count {
            let x_offset = tile_x * tile_width;
            let y_offset = tile_y * tile_height;

            // Calculate geographic coordinates for this tile
            // using the geotransform from the original raster
            let tile_origin_x =
                origin_x + (x_offset as f64) * pixel_width + (y_offset as f64) * rotation_x;
            let tile_origin_y =
                origin_y + (x_offset as f64) * rotation_y + (y_offset as f64) * pixel_height;
            
            // Create raster metadata for this tile with actual geotransform values
            let tile_metadata = RasterMetadata {
                width: tile_width as u64,
                height: tile_height as u64,
                upperleft_x: tile_origin_x,
                upperleft_y: tile_origin_y,
                scale_x: pixel_width,
                scale_y: pixel_height,
                skew_x: rotation_x,
                skew_y: rotation_y,
            };

            raster_builder.start_raster(&tile_metadata, None)?;

            for band_number in 1..=band_count {
                let band: RasterBand = dataset.rasterband(band_number).unwrap();
                
                // The band size is always the full raster size, not the tile size
                // We read a specific window from the band
                let actual_tile_width = tile_width.min(raster_width - x_offset);
                let actual_tile_height = tile_height.min(raster_height - y_offset);

                let data_type = to_banddatatype(band.band_type())?;
                let data_type_bytes = bytes_per_pixel(data_type.clone())?;
                let buffer_size_bytes = actual_tile_width * actual_tile_height * data_type_bytes;

                // Start the band with metadata first
                let nodata_value = match band.no_data_value() {
                    Some(val) => Some(f64_to_bandtype_bytes(val, data_type.clone())?),
                    None => None,
                };

                let band_metadata = BandMetadata {
                    nodata_value,
                    storage_type: StorageType::InDb,
                    datatype: data_type,
                    outdb_url: None,
                    outdb_band_id: None,
                };

                raster_builder.start_band(band_metadata)?;

                // Allocate a buffer for GDAL to read into
                let mut band_data = vec![0u8; buffer_size_bytes];

                // TODO: Do we need resampling? If so set buffer_size to different from window_size
                //       and have a ResampleAlgorithm.
                band.read_into_slice(
                    (x_offset as isize, y_offset as isize), // window_origin
                    (actual_tile_width, actual_tile_height), // window_size
                    (actual_tile_width, actual_tile_height), // buffer_size (no resampling)
                    &mut band_data,                         // buffer
                    None,                                   // resampling algorithms
                )
                .map_err(|e| {
                    ArrowError::ParseError(format!("Failed to read band {band_number} {e}"))
                })?;

                // Write the band data
                raster_builder.band_data_writer().append_value(&band_data);

                // Finalize the band (no arguments)
                raster_builder.finish_band()?;
            }

            // Finalize the raster
            raster_builder.finish_raster()?;
        }
    }

    // Finalize the raster struct array
    let raster_struct = raster_builder.finish()?;
    Ok(Arc::new(raster_struct))
}

/// Write a tiled raster StructArray to a GeoTIFF file using GDAL
pub fn write_geotiff(raster_array: &StructArray, filepath: &str) -> Result<(), ArrowError> {
    use gdal::{DriverManager, Metadata};
    use sedona_raster::array::RasterStructArray;
    use sedona_raster::traits::RasterRef;
    
    let raster_struct_array = RasterStructArray::new(raster_array);
    
    if raster_struct_array.len() == 0 {
        return Err(ArrowError::InvalidArgumentError(
            "Cannot write empty raster array".to_string(),
        ));
    }
    
    // Get the first raster to determine dimensions and band count
    let first_raster = raster_struct_array.get(0)?;
    let first_metadata = first_raster.metadata();
    let band_count = first_raster.bands().len();
    
    if band_count == 0 {
        return Err(ArrowError::InvalidArgumentError(
            "Raster has no bands".to_string(),
        ));
    }
    
    let first_band = first_raster.bands().band(1)?;
    let data_type = first_band.metadata().data_type();
    
    // Calculate total raster dimensions by analyzing tile positions
    let mut max_x: f64 = 0.0;
    let mut max_y: f64 = 0.0;
    let mut min_x: f64 = f64::MAX;
    let mut min_y: f64 = f64::MAX;
    
    for i in 0..raster_struct_array.len() {
        let raster = raster_struct_array.get(i)?;
        let metadata = raster.metadata();
        
        let tile_max_x = metadata.upper_left_x() + (metadata.width() as f64) * metadata.scale_x();
        let tile_max_y = metadata.upper_left_y() + (metadata.height() as f64) * metadata.scale_y();
        
        max_x = max_x.max(tile_max_x);
        max_y = max_y.max(tile_max_y);
        min_x = min_x.min(metadata.upper_left_x());
        min_y = min_y.min(metadata.upper_left_y());
    }
    
    // Calculate total raster dimensions
    let scale_x = first_metadata.scale_x();
    let scale_y = first_metadata.scale_y();
    let total_width = ((max_x - min_x) / scale_x).abs().round() as usize;
    let total_height = ((max_y - min_y) / scale_y).abs().round() as usize;
    
    // Get the GTiff driver
    let driver = DriverManager::get_driver_by_name("GTiff").map_err(|e| {
        ArrowError::ParseError(format!("Failed to get GTiff driver: {e}"))
    })?;
    
    // Create dataset based on data type
    let mut dataset = match data_type {
        BandDataType::UInt8 => driver.create_with_band_type::<u8, _>(filepath, total_width, total_height, band_count),
        BandDataType::UInt16 => driver.create_with_band_type::<u16, _>(filepath, total_width, total_height, band_count),
        BandDataType::Int16 => driver.create_with_band_type::<i16, _>(filepath, total_width, total_height, band_count),
        BandDataType::UInt32 => driver.create_with_band_type::<u32, _>(filepath, total_width, total_height, band_count),
        BandDataType::Int32 => driver.create_with_band_type::<i32, _>(filepath, total_width, total_height, band_count),
        BandDataType::Float32 => driver.create_with_band_type::<f32, _>(filepath, total_width, total_height, band_count),
        BandDataType::Float64 => driver.create_with_band_type::<f64, _>(filepath, total_width, total_height, band_count),
    }.map_err(|e| ArrowError::ParseError(format!("Failed to create GeoTIFF: {e}")))?;
    
    // Set geotransform
    dataset
        .set_geo_transform(&[
            min_x,
            scale_x,
            first_metadata.skew_x(),
            min_y,
            first_metadata.skew_y(),
            scale_y,
        ])
        .map_err(|e| ArrowError::ParseError(format!("Failed to set geotransform: {e}")))?;
    
    // Set tile size metadata if tiles are being used
    let tile_width = first_metadata.width();
    let tile_height = first_metadata.height();
    dataset
        .set_metadata_item("TILEWIDTH", &tile_width.to_string(), "")
        .map_err(|e| ArrowError::ParseError(format!("Failed to set TILEWIDTH metadata: {e}")))?;
    dataset
        .set_metadata_item("TILEHEIGHT", &tile_height.to_string(), "")
        .map_err(|e| ArrowError::ParseError(format!("Failed to set TILEHEIGHT metadata: {e}")))?;
    
    // Write each tile to the dataset
    for i in 0..raster_struct_array.len() {
        let raster = raster_struct_array.get(i)?;
        let raster_metadata = raster.metadata();
        
        // Calculate pixel offset for this tile
        let x_offset = ((raster_metadata.upper_left_x() - min_x) / scale_x).round() as isize;
        let y_offset = ((raster_metadata.upper_left_y() - min_y) / scale_y).round() as isize;
        
        let tile_width = raster_metadata.width() as usize;
        let tile_height = raster_metadata.height() as usize;
        
        // Write each band
        for band_index in 0..band_count {
            let band = raster.bands().band(band_index + 1)?;
            let mut gdal_band = dataset.rasterband(band_index + 1).map_err(|e| {
                ArrowError::ParseError(format!("Failed to get GDAL band {}: {e}", band_index + 1))
            })?;
            
            let band_data = band.data();
            let band_datatype = band.metadata().data_type();
            
            // Write the band data to the appropriate location in the dataset
            // We need to convert the byte slice to the appropriate type for GDAL
            write_band_data(
                &mut gdal_band,
                (x_offset, y_offset),
                (tile_width, tile_height),
                band_data,
                band_datatype,
            )?;
            
            // Set nodata value if present
            if let Some(nodata_bytes) = band.metadata().nodata_value() {
                if let Some(nodata_f64) = bytes_to_f64(nodata_bytes, band.metadata().data_type()) {
                    gdal_band.set_no_data_value(Some(nodata_f64)).map_err(|e| {
                        ArrowError::ParseError(format!("Failed to set nodata value: {e}"))
                    })?;
                }
            }
        }
    }
    
    // Flush the dataset to disk
    dataset.flush_cache().map_err(|e| {
        ArrowError::ParseError(format!("Failed to flush cache when writing GeoTIFF: {e}"))
    })?;
    
    Ok(())
}

/// Helper function to write band data to GDAL, converting from bytes to the appropriate type
fn write_band_data(
    gdal_band: &mut RasterBand,
    window_origin: (isize, isize),
    window_size: (usize, usize),
    band_data: &[u8],
    data_type: BandDataType,
) -> Result<(), ArrowError> {
    use gdal::raster::Buffer;
    
    let (width, height) = window_size;
    let pixel_count = width * height;
    
    match data_type {
        BandDataType::UInt8 => {
            let data: Vec<u8> = band_data.to_vec();
            let mut buffer = Buffer::new(window_size, data);
            gdal_band.write(window_origin, window_size, &mut buffer)
        }
        BandDataType::UInt16 => {
            let mut data = Vec::with_capacity(pixel_count);
            for chunk in band_data.chunks_exact(2) {
                data.push(u16::from_ne_bytes([chunk[0], chunk[1]]));
            }
            let mut buffer = Buffer::new(window_size, data);
            gdal_band.write(window_origin, window_size, &mut buffer)
        }
        BandDataType::Int16 => {
            let mut data = Vec::with_capacity(pixel_count);
            for chunk in band_data.chunks_exact(2) {
                data.push(i16::from_ne_bytes([chunk[0], chunk[1]]));
            }
            let mut buffer = Buffer::new(window_size, data);
            gdal_band.write(window_origin, window_size, &mut buffer)
        }
        BandDataType::UInt32 => {
            let mut data = Vec::with_capacity(pixel_count);
            for chunk in band_data.chunks_exact(4) {
                data.push(u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            let mut buffer = Buffer::new(window_size, data);
            gdal_band.write(window_origin, window_size, &mut buffer)
        }
        BandDataType::Int32 => {
            let mut data = Vec::with_capacity(pixel_count);
            for chunk in band_data.chunks_exact(4) {
                data.push(i32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            let mut buffer = Buffer::new(window_size, data);
            gdal_band.write(window_origin, window_size, &mut buffer)
        }
        BandDataType::Float32 => {
            let mut data = Vec::with_capacity(pixel_count);
            for chunk in band_data.chunks_exact(4) {
                data.push(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            let mut buffer = Buffer::new(window_size, data);
            gdal_band.write(window_origin, window_size, &mut buffer)
        }
        BandDataType::Float64 => {
            let mut data = Vec::with_capacity(pixel_count);
            for chunk in band_data.chunks_exact(8) {
                data.push(f64::from_ne_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]));
            }
            let mut buffer = Buffer::new(window_size, data);
            gdal_band.write(window_origin, window_size, &mut buffer)
        }
    }
    .map_err(|e| ArrowError::ParseError(format!("Failed to write band data: {e}")))
}

/// Convert bytes back to f64 for nodata value
fn bytes_to_f64(bytes: &[u8], data_type: BandDataType) -> Option<f64> {
    match data_type {
        BandDataType::UInt8 if bytes.len() >= 1 => Some(bytes[0] as f64),
        BandDataType::UInt16 if bytes.len() >= 2 => {
            Some(u16::from_ne_bytes([bytes[0], bytes[1]]) as f64)
        }
        BandDataType::Int16 if bytes.len() >= 2 => {
            Some(i16::from_ne_bytes([bytes[0], bytes[1]]) as f64)
        }
        BandDataType::UInt32 if bytes.len() >= 4 => {
            Some(u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64)
        }
        BandDataType::Int32 if bytes.len() >= 4 => {
            Some(i32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64)
        }
        BandDataType::Float32 if bytes.len() >= 4 => {
            Some(f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64)
        }
        BandDataType::Float64 if bytes.len() >= 8 => Some(f64::from_ne_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ])),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Array;
    use sedona_raster::array::RasterStructArray;
    use sedona_raster::traits::RasterRef;
    use sedona_schema::raster::BandDataType;
    use sedona_testing::rasters::raster_arrays_equal;
    

    #[test]
    fn test_read_write_raster() {
        // Create an array of random rasters
        let raster_array = generate_tiled_rasters(8, 4, 6, 4).unwrap();

        // Write them as a single GeoTIFF to /tmp/test_output.tif
        let filepath = "/tmp/test_output.tif";
        write_geotiff(&raster_array, filepath).unwrap();

        // Read the rasters back in from the GeoTIFF using the tile metadata we wrote
        let read_raster_array = read_raster_geotiff(filepath, None, None).unwrap();
        assert_eq!(raster_array.len(), read_raster_array.len());

        // Compare the original and read rasters for equality
        let raster_struct = RasterStructArray::new(&raster_array);
        let read_raster_struct = RasterStructArray::new(read_raster_array.as_ref());
        assert!(raster_arrays_equal(
            &raster_struct,
            &read_raster_struct
        ));
        
        // Re-Read with new tiling parameters
        let (tile_width, tile_height) = (4, 2);
        let read_raster_array_tiled = read_raster_geotiff(filepath, Some(tile_width), Some(tile_height)).unwrap();
        let raster_tiled_array = RasterStructArray::new(&read_raster_array_tiled);
        let expected_tile_count = ((6 * 8 + tile_width - 1) / tile_width) * ((4 * 4 + tile_height - 1) / tile_height);
        assert_eq!(expected_tile_count, raster_tiled_array.len());
        let raster = raster_tiled_array.get(0).unwrap();
        let metadata = raster.metadata();
        assert_eq!(metadata.width(), tile_width as u64);
        assert_eq!(metadata.height(), tile_height as u64);

        // Clean up
        //std::fs::remove_file(filepath).unwrap();
    }

    fn generate_tiled_rasters(tile_width: usize, tile_height: usize, x_tiles: usize, y_tiles: usize) -> Result<StructArray, ArrowError> {
        let mut raster_builder = RasterBuilder::new(x_tiles * y_tiles);
        let band_count = 3;

        for tile_y in 0..y_tiles {
            for tile_x in 0..x_tiles {
                let origin_x = (tile_x * tile_width) as f64;
                let origin_y = (tile_y * tile_height) as f64;

                let raster_metadata = RasterMetadata {
                    width: tile_width as u64,
                    height: tile_height as u64,
                    upperleft_x: origin_x,
                    upperleft_y: origin_y,
                    scale_x: 1.0,
                    scale_y: 1.0,
                    skew_x: 0.0,
                    skew_y: 0.0,
                };

                raster_builder.start_raster(&raster_metadata, None)?;

                for _ in 0..band_count {
                    let band_metadata = BandMetadata {
                        nodata_value: None,
                        storage_type: StorageType::InDb,
                        datatype: BandDataType::UInt8,
                        outdb_url: None,
                        outdb_band_id: None,
                    };

                    raster_builder.start_band(band_metadata)?;

                    let buffer_size = tile_width * tile_height; // UInt8
                    let mut band_data = vec![0u8; buffer_size];

                    // Fill with random data
                    for byte in band_data.iter_mut() {
                        *byte = fastrand::u8(..);
                    }

                    raster_builder.band_data_writer().append_value(&band_data);
                    raster_builder.finish_band()?;
                }

                raster_builder.finish_raster()?;
            }
        }

        raster_builder.finish()
    }
}