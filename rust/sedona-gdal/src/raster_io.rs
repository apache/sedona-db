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

// ============================================================================
// Format-specific convenience wrappers
// ============================================================================

/// Reads a GeoTIFF file using GDAL and converts it into a StructArray of rasters.
///
/// This is a convenience wrapper around [`read_raster`] for GeoTIFF files.
///
/// # Arguments
/// * `filepath` - Path to the GeoTIFF file
/// * `tile_size_opt` - Optional tile size to override dataset metadata
pub fn read_geotiff(
    filepath: &str,
    tile_size_opt: Option<(usize, usize)>,
) -> Result<Arc<StructArray>, ArrowError> {
    // Check that the filepath has a GeoTIFF extension
    let filepath_lower = filepath.to_lowercase();
    if !filepath_lower.ends_with(".tif") && !filepath_lower.ends_with(".tiff") {
        return Err(ArrowError::InvalidArgumentError(format!(
            "Expected GeoTIFF file with .tif or .tiff extension, got: {}",
            filepath
        )));
    }
    read_raster(filepath, tile_size_opt)
}

/// Writes a tiled raster StructArray to a GeoTIFF file using GDAL.
///
/// This is a convenience wrapper around [`write_raster`] for GeoTIFF files.
///
/// # Arguments
/// * `raster_array` - The raster struct array to write
/// * `filepath` - Path to the output GeoTIFF file
pub fn write_geotiff(raster_array: &StructArray, filepath: &str) -> Result<(), ArrowError> {
    // Check that the filepath has a GeoTIFF extension
    let filepath_lower = filepath.to_lowercase();
    if !filepath_lower.ends_with(".tif") && !filepath_lower.ends_with(".tiff") {
        return Err(ArrowError::InvalidArgumentError(format!(
            "Expected GeoTIFF file with .tif or .tiff extension, got: {}",
            filepath
        )));
    }
    write_raster(raster_array, filepath, "GTiff")
}

/// Reads a raster file using GDAL and converts it into a StructArray of rasters.
///
/// Currently only supports reading rasters into InDb storage type.
/// OutDb storage types are not yet implemented.
///
/// # Arguments
/// * `filepath` - Path to the raster file
/// * `tile_size_opt` - Optional tile size to override dataset metadata (uses dataset metadata if None)
pub fn read_raster(
    filepath: &str,
    tile_size_opt: Option<(usize, usize)>,
) -> Result<Arc<StructArray>, ArrowError> {
    let dataset = Dataset::open(filepath).map_err(|err| ArrowError::ParseError(err.to_string()))?;

    // Extract geotransform components
    let (origin_x, origin_y, pixel_width, pixel_height, rotation_x, rotation_y) =
        geotransform_components(&dataset)?;

    let (raster_width, raster_height) = dataset.raster_size();

    // Use provided tile dimensions or fall back to dataset metadata
    let (tile_width, tile_height) = match tile_size_opt {
        Some((w, h)) => (w, h),
        _ => tile_size(&dataset)?,
    };

    let x_tile_count = raster_width.div_ceil(tile_width);
    let y_tile_count = raster_height.div_ceil(tile_height);

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

                // Start the band with metadata first
                let nodata_value = match band.no_data_value() {
                    Some(val) => Some(f64_to_bandtype_bytes(val, data_type.clone())?),
                    None => None,
                };

                let band_metadata = BandMetadata {
                    nodata_value,
                    storage_type: StorageType::InDb,
                    datatype: data_type.clone(),
                    outdb_url: None,
                    outdb_band_id: None,
                };

                raster_builder.start_band(band_metadata)?;

                // Reserve capacity and write band data directly from GDAL
                let pixel_count = actual_tile_width * actual_tile_height;
                let bytes_needed = pixel_count * bytes_per_pixel(data_type.clone())?;

                // Pre-allocate a vector of the exact size needed
                let mut band_data = vec![0u8; bytes_needed];

                // Read directly into the pre-allocated buffer
                read_band_data_into(
                    &band,
                    (x_offset as isize, y_offset as isize),
                    (actual_tile_width, actual_tile_height),
                    &data_type,
                    &mut band_data,
                )?;

                // Write the band data
                raster_builder.band_data_writer().append_value(&band_data);

                // Finalize the band
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

/// Helper function to read band data from GDAL directly into a pre-allocated byte buffer
/// Casts the buffer to the appropriate type and reads directly onto `output`
fn read_band_data_into(
    band: &RasterBand,
    window_origin: (isize, isize),
    window_size: (usize, usize),
    data_type: &BandDataType,
    output: &mut [u8],
) -> Result<(), ArrowError> {
    let pixel_count = window_size.0 * window_size.1;

    match data_type {
        BandDataType::UInt8 => {
            band.read_into_slice(window_origin, window_size, window_size, output, None)
                .map_err(|e| ArrowError::ParseError(format!("Failed to read band data: {e}")))?;
            Ok(())
        }
        BandDataType::UInt16 => {
            let typed_output = unsafe {
                std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut u16, pixel_count)
            };
            band.read_into_slice(window_origin, window_size, window_size, typed_output, None)
                .map_err(|e| ArrowError::ParseError(format!("Failed to read band data: {e}")))?;
            Ok(())
        }
        BandDataType::Int16 => {
            let typed_output = unsafe {
                std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut i16, pixel_count)
            };
            band.read_into_slice(window_origin, window_size, window_size, typed_output, None)
                .map_err(|e| ArrowError::ParseError(format!("Failed to read band data: {e}")))?;
            Ok(())
        }
        BandDataType::UInt32 => {
            let typed_output = unsafe {
                std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut u32, pixel_count)
            };
            band.read_into_slice(window_origin, window_size, window_size, typed_output, None)
                .map_err(|e| ArrowError::ParseError(format!("Failed to read band data: {e}")))?;
            Ok(())
        }
        BandDataType::Int32 => {
            let typed_output = unsafe {
                std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut i32, pixel_count)
            };
            band.read_into_slice(window_origin, window_size, window_size, typed_output, None)
                .map_err(|e| ArrowError::ParseError(format!("Failed to read band data: {e}")))?;
            Ok(())
        }
        BandDataType::Float32 => {
            let typed_output = unsafe {
                std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut f32, pixel_count)
            };
            band.read_into_slice(window_origin, window_size, window_size, typed_output, None)
                .map_err(|e| ArrowError::ParseError(format!("Failed to read band data: {e}")))?;
            Ok(())
        }
        BandDataType::Float64 => {
            let typed_output = unsafe {
                std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut f64, pixel_count)
            };
            band.read_into_slice(window_origin, window_size, window_size, typed_output, None)
                .map_err(|e| ArrowError::ParseError(format!("Failed to read band data: {e}")))?;
            Ok(())
        }
    }
}

/// Write a tiled raster StructArray to a raster file using GDAL.
///
/// This is a generic function that works with any GDAL-supported raster format.
/// Currently only supports writing rasters with InDb storage type.
/// OutDb storage types will return a NotYetImplemented error.
///
/// # Arguments
/// * `raster_array` - The raster struct array to write
/// * `filepath` - Path to the output file
/// * `driver_name` - GDAL driver name (e.g., "GTiff", "Zarr")
fn write_raster(
    raster_array: &StructArray,
    filepath: &str,
    driver_name: &str,
) -> Result<(), ArrowError> {
    use gdal::{DriverManager, Metadata};
    use sedona_raster::array::RasterStructArray;
    use sedona_raster::traits::RasterRef;

    let raster_struct_array = RasterStructArray::new(raster_array);

    if raster_struct_array.is_empty() {
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

    if scale_x == 0.0 || scale_y == 0.0 {
        return Err(ArrowError::InvalidArgumentError(
            "Invalid geotransform: scale values cannot be zero".to_string(),
        ));
    }

    let total_width = ((max_x - min_x) / scale_x).abs().round() as usize;
    let total_height = ((max_y - min_y) / scale_y).abs().round() as usize;

    // Get GDAL driver by name
    let driver = DriverManager::get_driver_by_name(driver_name).map_err(|e| {
        ArrowError::ParseError(format!("Failed to get {} driver: {e}", driver_name))
    })?;

    // Create dataset based on data type
    let mut dataset = match data_type {
        BandDataType::UInt8 => {
            driver.create_with_band_type::<u8, _>(filepath, total_width, total_height, band_count)
        }
        BandDataType::UInt16 => {
            driver.create_with_band_type::<u16, _>(filepath, total_width, total_height, band_count)
        }
        BandDataType::Int16 => {
            driver.create_with_band_type::<i16, _>(filepath, total_width, total_height, band_count)
        }
        BandDataType::UInt32 => {
            driver.create_with_band_type::<u32, _>(filepath, total_width, total_height, band_count)
        }
        BandDataType::Int32 => {
            driver.create_with_band_type::<i32, _>(filepath, total_width, total_height, band_count)
        }
        BandDataType::Float32 => {
            driver.create_with_band_type::<f32, _>(filepath, total_width, total_height, band_count)
        }
        BandDataType::Float64 => {
            driver.create_with_band_type::<f64, _>(filepath, total_width, total_height, band_count)
        }
    }
    .map_err(|e| ArrowError::ParseError(format!("Failed to create raster dataset: {e}")))?;

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

            // Check that storage type is InDb
            if band.metadata().storage_type() != StorageType::InDb {
                return Err(ArrowError::NotYetImplemented(
                    format!("Writing rasters with storage type {:?} is not yet implemented. Only InDb storage is currently supported.",
                        band.metadata().storage_type())
                ));
            }

            let mut gdal_band = dataset.rasterband(band_index + 1).map_err(|e| {
                ArrowError::ParseError(format!("Failed to get GDAL band {}: {e}", band_index + 1))
            })?;

            let band_data = band.data();
            let band_datatype = band.metadata().data_type();

            // Write the band data to the appropriate location in the dataset
            // Convert the byte slice to the appropriate type for GDAL
            write_band_data(
                &mut gdal_band,
                (x_offset, y_offset),
                (tile_width, tile_height),
                band_data,
                band_datatype,
            )?;

            // Set nodata value if present
            // Note: Some drivers (e.g., Zarr) don't support nodata values, so we ignore errors
            if let Some(nodata_bytes) = band.metadata().nodata_value() {
                if let Some(nodata_f64) = bytes_to_f64(nodata_bytes, band.metadata().data_type()) {
                    let _ = gdal_band.set_no_data_value(Some(nodata_f64));
                }
            }
        }
    }

    // Flush the dataset
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

    /// Macro to reduce boilerplate for writing typed buffers
    macro_rules! write_typed_buffer {
        ($type:ty, $bytes_per_elem:expr) => {{
            let data = bytes_to_typed_vec::<$type>(band_data, $bytes_per_elem);
            let mut buffer = Buffer::new(window_size, data);
            gdal_band.write(window_origin, window_size, &mut buffer)
        }};
    }

    match data_type {
        BandDataType::UInt8 => {
            // UInt8 is special - no conversion needed
            let data = band_data.to_vec();
            let mut buffer = Buffer::new(window_size, data);
            gdal_band.write(window_origin, window_size, &mut buffer)
        }
        BandDataType::UInt16 => write_typed_buffer!(u16, 2),
        BandDataType::Int16 => write_typed_buffer!(i16, 2),
        BandDataType::UInt32 => write_typed_buffer!(u32, 4),
        BandDataType::Int32 => write_typed_buffer!(i32, 4),
        BandDataType::Float32 => write_typed_buffer!(f32, 4),
        BandDataType::Float64 => write_typed_buffer!(f64, 8),
    }
    .map_err(|e| ArrowError::ParseError(format!("Failed to write band data: {e}")))
}

/// Helper function to convert byte slices to typed vectors
/// Uses chunks_exact to safely handle byte conversion without potential panics
fn bytes_to_typed_vec<T>(bytes: &[u8], element_size: usize) -> Vec<T>
where
    T: FromNeBytes,
{
    bytes
        .chunks_exact(element_size)
        .map(|chunk| T::from_ne_bytes_slice(chunk))
        .collect()
}

/// Trait for converting from native-endian byte slices
trait FromNeBytes: Sized {
    fn from_ne_bytes_slice(bytes: &[u8]) -> Self;
}

impl FromNeBytes for u16 {
    fn from_ne_bytes_slice(bytes: &[u8]) -> Self {
        u16::from_ne_bytes([bytes[0], bytes[1]])
    }
}

impl FromNeBytes for i16 {
    fn from_ne_bytes_slice(bytes: &[u8]) -> Self {
        i16::from_ne_bytes([bytes[0], bytes[1]])
    }
}

impl FromNeBytes for u32 {
    fn from_ne_bytes_slice(bytes: &[u8]) -> Self {
        u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }
}

impl FromNeBytes for i32 {
    fn from_ne_bytes_slice(bytes: &[u8]) -> Self {
        i32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }
}

impl FromNeBytes for f32 {
    fn from_ne_bytes_slice(bytes: &[u8]) -> Self {
        f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }
}

impl FromNeBytes for f64 {
    fn from_ne_bytes_slice(bytes: &[u8]) -> Self {
        f64::from_ne_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ])
    }
}

/// Convert nodata value bytes back to f64
fn bytes_to_f64(bytes: &[u8], data_type: BandDataType) -> Option<f64> {
    let expected_bytes = bytes_per_pixel(data_type.clone()).ok()?;
    if bytes.len() < expected_bytes {
        return None;
    }
    match data_type {
        BandDataType::UInt8 => Some(bytes[0] as f64),
        BandDataType::UInt16 => Some(u16::from_ne_bytes_slice(bytes) as f64),
        BandDataType::Int16 => Some(i16::from_ne_bytes_slice(bytes) as f64),
        BandDataType::UInt32 => Some(u32::from_ne_bytes_slice(bytes) as f64),
        BandDataType::Int32 => Some(i32::from_ne_bytes_slice(bytes) as f64),
        BandDataType::Float32 => Some(f32::from_ne_bytes_slice(bytes) as f64),
        BandDataType::Float64 => Some(f64::from_ne_bytes_slice(bytes)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Array;
    use gdal::vsi;
    use rstest::rstest;
    use sedona_raster::array::RasterStructArray;
    use sedona_raster::traits::RasterRef;
    use sedona_schema::raster::BandDataType;
    use sedona_testing::rasters::{assert_raster_arrays_equal, generate_tiled_rasters};
    use tempfile::tempdir;

    #[rstest]
    fn test_read_write_raster(
        #[values(
            BandDataType::UInt8,
            BandDataType::UInt16,
            BandDataType::Int16,
            BandDataType::UInt32,
            BandDataType::Int32,
            BandDataType::Float32,
            BandDataType::Float64
        )]
        data_type: BandDataType,
    ) {
        let tile_size = (16, 8);
        let tile_count = (4, 4);
        let raster_struct =
            generate_tiled_rasters(tile_size, tile_count, data_type.clone(), Some(43)).unwrap();

        //
        let filepath_str = format!("/vsimem/test_raster_output_{:?}.tif", data_type);
        write_geotiff(&raster_struct, &filepath_str).unwrap();

        // Read the rasters back in from the GeoTIFF using the tile metadata
        let read_raster_struct = read_geotiff(&filepath_str, None).unwrap();
        assert_eq!(raster_struct.len(), read_raster_struct.len());

        // Compare the original and read rasters for equality
        let raster_array = RasterStructArray::new(&raster_struct);
        let read_raster_array = RasterStructArray::new(&read_raster_struct);
        assert_raster_arrays_equal(&raster_array, &read_raster_array);

        // Re-Read with new tiling parameters (swap tile_count and tile_size)
        let (new_tile_width, new_tile_height) = tile_count; // swapped
        let (new_tile_count_x, new_tile_count_y) = tile_size; // swapped
        let read_raster_array_tiled =
            read_raster(&filepath_str, Some((new_tile_width, new_tile_height))).unwrap();
        let raster_retiled_array = RasterStructArray::new(&read_raster_array_tiled);

        // Validate the new tiling
        let expected_tile_count = new_tile_count_x * new_tile_count_y;
        assert_eq!(expected_tile_count, raster_retiled_array.len());
        let raster = raster_retiled_array.get(0).unwrap();
        let metadata = raster.metadata();
        assert_eq!(metadata.width(), new_tile_width as u64);
        assert_eq!(metadata.height(), new_tile_height as u64);

        // Re-Write the re-tiled raster to a new GeoTIFF
        let retiled_filepath_str =
            format!("/vsimem/test_raster_retiled_output_{:?}.tif", data_type);
        write_geotiff(&read_raster_array_tiled, &retiled_filepath_str).unwrap();

        // Re-Read with original tiling parameters
        let read_original_tiling_raster_array =
            read_geotiff(&retiled_filepath_str, Some(tile_size)).unwrap();

        // Validate that we get back the original raster array
        assert_raster_arrays_equal(
            &raster_array,
            &RasterStructArray::new(&read_original_tiling_raster_array),
        );

        // Clean-up
        vsi::unlink_mem_file(filepath_str).unwrap();
        vsi::unlink_mem_file(retiled_filepath_str).unwrap();
    }

    #[test]
    #[ignore]
    // Unit test for writing to disk. Passes, but marking as ignore due to the
    // potential issues of writing to disk in unit tests.  Run with `--ignored`
    fn test_write_geotiff_to_disk() {
        let temp_dir = tempdir().unwrap();
        let filepath = temp_dir.path().join("test_raster_output.tif");
        let filepath_str = filepath.as_os_str().to_str().unwrap();

        let tile_size = (16, 8);
        let tile_count = (4, 4);
        let raster_struct =
            generate_tiled_rasters(tile_size, tile_count, BandDataType::UInt16, Some(43)).unwrap();
        let result = write_geotiff(&raster_struct, filepath_str);
        assert!(result.is_ok());
        let read_raster_struct = read_geotiff(filepath_str, None).unwrap();
        assert_raster_arrays_equal(
            &RasterStructArray::new(&raster_struct),
            &RasterStructArray::new(&read_raster_struct),
        );

        // Clean up
        drop(filepath);
        temp_dir.close().unwrap();
    }

    #[test]
    fn test_filepath_validation() {
        // Create a simple test raster
        let tile_size = (8, 8);
        let tile_count = (2, 2);
        let raster_struct =
            generate_tiled_rasters(tile_size, tile_count, BandDataType::UInt8, Some(43)).unwrap();

        let err = read_geotiff("test.zarr", None).unwrap_err();
        assert!(err.to_string().contains("Expected GeoTIFF"));
        assert!(err.to_string().contains(".tif or .tiff"));

        let err = write_geotiff(&raster_struct, "test.zarr").unwrap_err();
        assert!(err.to_string().contains("Expected GeoTIFF"));
    }

    #[test]
    fn test_round_trip_conversions() {
        // UInt8
        let value: u8 = 255;
        let bytes = value.to_le_bytes();
        assert_eq!(bytes_to_f64(&bytes, BandDataType::UInt8), Some(255.0));

        // UInt16
        let value: u16 = 32768;
        let bytes = value.to_le_bytes();
        assert_eq!(bytes_to_f64(&bytes, BandDataType::UInt16), Some(32768.0));

        // Int16
        let value: i16 = -32768;
        let bytes = value.to_le_bytes();
        assert_eq!(bytes_to_f64(&bytes, BandDataType::Int16), Some(-32768.0));

        // UInt32
        let value: u32 = 2147483648;
        let bytes = value.to_le_bytes();
        assert_eq!(
            bytes_to_f64(&bytes, BandDataType::UInt32),
            Some(2147483648.0)
        );

        // Int32
        let value: i32 = -2147483648;
        let bytes = value.to_le_bytes();
        assert_eq!(
            bytes_to_f64(&bytes, BandDataType::Int32),
            Some(-2147483648.0)
        );

        // Float32
        let value: f32 = 256.0;
        let bytes = value.to_le_bytes();
        assert_eq!(bytes_to_f64(&bytes, BandDataType::Float32), Some(256.0));

        // Float64
        let value: f64 = -2147483648.3;
        let bytes = value.to_le_bytes();
        assert_eq!(
            bytes_to_f64(&bytes, BandDataType::Float64),
            Some(-2147483648.3)
        );
    }
}
