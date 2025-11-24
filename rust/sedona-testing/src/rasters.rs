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
use arrow_array::StructArray;
use datafusion_common::Result;
use fastrand::Rng;
use sedona_raster::array::RasterStructArray;
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::{BandMetadata, RasterMetadata, RasterRef};
use sedona_schema::raster::{BandDataType, StorageType};

/// Generate a StructArray of rasters with sequentially increasing dimensions and pixel values
/// These tiny rasters are to provide fast, easy and predictable test data for unit tests.
pub fn generate_test_rasters(
    count: usize,
    null_raster_index: Option<usize>,
) -> Result<StructArray> {
    let mut builder = RasterBuilder::new(count);
    for i in 0..count {
        // If a null raster index is specified and that matches the current index,
        // append a null raster
        if matches!(null_raster_index, Some(index) if index == i) {
            builder.append_null()?;
            continue;
        }

        let raster_metadata = RasterMetadata {
            width: i as u64 + 1,
            height: i as u64 + 2,
            upperleft_x: i as f64 + 1.0,
            upperleft_y: i as f64 + 2.0,
            scale_x: i as f64 * 0.1,
            scale_y: i as f64 * 0.2,
            skew_x: i as f64 * 0.3,
            skew_y: i as f64 * 0.4,
        };
        builder.start_raster(&raster_metadata, None)?;
        builder.start_band(BandMetadata {
            datatype: BandDataType::UInt16,
            nodata_value: Some(vec![0u8; 2]),
            storage_type: StorageType::InDb,
            outdb_url: None,
            outdb_band_id: None,
        })?;

        let pixel_count = (i + 1) * (i + 2); // width * height
        let mut band_data = Vec::with_capacity(pixel_count * 2); // 2 bytes per u16
        for pixel_value in 0..pixel_count as u16 {
            band_data.extend_from_slice(&pixel_value.to_le_bytes());
        }

        builder.band_data_writer().append_value(&band_data);
        builder.finish_band()?;
        builder.finish_raster()?;
    }

    Ok(builder.finish()?)
}

/// Generates a set of tiled rasters arranged in a grid
/// - Each raster tile has specified dimensions and random pixel values
/// - Each raster has 3 bands which can be interpreted as RGB values
///   and the result can be visualized as a mosaic of tiles.
/// - There are nodata values at the 4 corners of the overall mosaic.
pub fn generate_tiled_rasters(
    tile_size: (usize, usize),
    number_of_tiles: (usize, usize),
    data_type: BandDataType,
    seed: Option<u64>,
) -> Result<StructArray> {
    let mut rng = match seed {
        Some(s) => Rng::with_seed(s),
        None => Rng::new(),
    };
    let (tile_width, tile_height) = tile_size;
    let (x_tiles, y_tiles) = number_of_tiles;
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
                // Set a nodata value appropriate for the data type
                let nodata_value = get_nodata_value_for_type(&data_type);

                let band_metadata = BandMetadata {
                    nodata_value: nodata_value.clone(),
                    storage_type: StorageType::InDb,
                    datatype: data_type.clone(),
                    outdb_url: None,
                    outdb_band_id: None,
                };

                raster_builder.start_band(band_metadata)?;

                let pixel_count = tile_width * tile_height;

                // Determine which corner position (if any) should have nodata in this tile
                let corner_position =
                    get_corner_position(tile_x, tile_y, x_tiles, y_tiles, tile_width, tile_height);
                let band_data = generate_random_band_data(
                    pixel_count,
                    &data_type,
                    nodata_value.as_deref(),
                    corner_position,
                    &mut rng,
                );

                raster_builder.band_data_writer().append_value(&band_data);
                raster_builder.finish_band()?;
            }

            raster_builder.finish_raster()?;
        }
    }

    Ok(raster_builder.finish()?)
}

/// Determine if this tile contains a corner of the overall grid and return its position
/// Returns Some(position) if this tile contains a corner, None otherwise
fn get_corner_position(
    tile_x: usize,
    tile_y: usize,
    x_tiles: usize,
    y_tiles: usize,
    tile_width: usize,
    tile_height: usize,
) -> Option<usize> {
    // Top-left corner (tile 0,0, pixel 0)
    if tile_x == 0 && tile_y == 0 {
        return Some(0);
    }
    // Top-right corner (tile x_tiles-1, 0, pixel tile_width-1)
    if tile_x == x_tiles - 1 && tile_y == 0 {
        return Some(tile_width - 1);
    }
    // Bottom-left corner (tile 0, y_tiles-1, pixel (tile_height-1)*tile_width)
    if tile_x == 0 && tile_y == y_tiles - 1 {
        return Some((tile_height - 1) * tile_width);
    }
    // Bottom-right corner (tile x_tiles-1, y_tiles-1, pixel tile_height*tile_width-1)
    if tile_x == x_tiles - 1 && tile_y == y_tiles - 1 {
        return Some(tile_height * tile_width - 1);
    }
    None
}

fn generate_random_band_data(
    pixel_count: usize,
    data_type: &BandDataType,
    nodata_bytes: Option<&[u8]>,
    corner_position: Option<usize>,
    rng: &mut Rng,
) -> Vec<u8> {
    match data_type {
        BandDataType::UInt8 => {
            let mut data: Vec<u8> = (0..pixel_count).map(|_| rng.u8(..)).collect();
            // Set corner pixel to nodata value if this tile contains a corner
            if let (Some(nodata), Some(pos)) = (nodata_bytes, corner_position) {
                if !nodata.is_empty() && pos < data.len() {
                    data[pos] = nodata[0];
                }
            }
            data
        }
        BandDataType::UInt16 => {
            let mut data = Vec::with_capacity(pixel_count * 2);
            for _ in 0..pixel_count {
                data.extend_from_slice(&rng.u16(..).to_ne_bytes());
            }
            // Set corner pixel to nodata value if this tile contains a corner
            if let (Some(nodata), Some(pos)) = (nodata_bytes, corner_position) {
                if nodata.len() >= 2 && pos * 2 + 2 <= data.len() {
                    data[pos * 2..(pos * 2) + 2].copy_from_slice(&nodata[0..2]);
                }
            }
            data
        }
        BandDataType::Int16 => {
            let mut data = Vec::with_capacity(pixel_count * 2);
            for _ in 0..pixel_count {
                data.extend_from_slice(&rng.i16(..).to_ne_bytes());
            }
            // Set corner pixel to nodata value if this tile contains a corner
            if let (Some(nodata), Some(pos)) = (nodata_bytes, corner_position) {
                if nodata.len() >= 2 && pos * 2 + 2 <= data.len() {
                    data[pos * 2..(pos * 2) + 2].copy_from_slice(&nodata[0..2]);
                }
            }
            data
        }
        BandDataType::UInt32 => {
            let mut data = Vec::with_capacity(pixel_count * 4);
            for _ in 0..pixel_count {
                data.extend_from_slice(&rng.u32(..).to_ne_bytes());
            }
            // Set corner pixel to nodata value if this tile contains a corner
            if let (Some(nodata), Some(pos)) = (nodata_bytes, corner_position) {
                if nodata.len() >= 4 && pos * 4 + 4 <= data.len() {
                    data[pos * 4..(pos * 4) + 4].copy_from_slice(&nodata[0..4]);
                }
            }
            data
        }
        BandDataType::Int32 => {
            let mut data = Vec::with_capacity(pixel_count * 4);
            for _ in 0..pixel_count {
                data.extend_from_slice(&rng.i32(..).to_ne_bytes());
            }
            // Set corner pixel to nodata value if this tile contains a corner
            if let (Some(nodata), Some(pos)) = (nodata_bytes, corner_position) {
                if nodata.len() >= 4 && pos * 4 + 4 <= data.len() {
                    data[pos * 4..(pos * 4) + 4].copy_from_slice(&nodata[0..4]);
                }
            }
            data
        }
        BandDataType::Float32 => {
            let mut data = Vec::with_capacity(pixel_count * 4);
            for _ in 0..pixel_count {
                data.extend_from_slice(&rng.f32().to_ne_bytes());
            }
            // Set corner pixel to nodata value if this tile contains a corner
            if let (Some(nodata), Some(pos)) = (nodata_bytes, corner_position) {
                if nodata.len() >= 4 && pos * 4 + 4 <= data.len() {
                    data[pos * 4..(pos * 4) + 4].copy_from_slice(&nodata[0..4]);
                }
            }
            data
        }
        BandDataType::Float64 => {
            let mut data = Vec::with_capacity(pixel_count * 8);
            for _ in 0..pixel_count {
                data.extend_from_slice(&rng.f64().to_ne_bytes());
            }
            // Set corner pixel to nodata value if this tile contains a corner
            if let (Some(nodata), Some(pos)) = (nodata_bytes, corner_position) {
                if nodata.len() >= 8 && pos * 8 + 8 <= data.len() {
                    data[pos * 8..(pos * 8) + 8].copy_from_slice(&nodata[0..8]);
                }
            }
            data
        }
    }
}

fn get_nodata_value_for_type(data_type: &BandDataType) -> Option<Vec<u8>> {
    match data_type {
        BandDataType::UInt8 => Some(vec![255u8]),
        BandDataType::UInt16 => Some(u16::MAX.to_ne_bytes().to_vec()),
        BandDataType::Int16 => Some(i16::MIN.to_ne_bytes().to_vec()),
        BandDataType::UInt32 => Some(u32::MAX.to_ne_bytes().to_vec()),
        BandDataType::Int32 => Some(i32::MIN.to_ne_bytes().to_vec()),
        BandDataType::Float32 => Some(f32::NAN.to_ne_bytes().to_vec()),
        BandDataType::Float64 => Some(f64::NAN.to_ne_bytes().to_vec()),
    }
}

/// Compare two RasterStructArrays for equality
pub fn assert_raster_arrays_equal(
    raster_array1: &RasterStructArray,
    raster_array2: &RasterStructArray,
) {
    assert_eq!(
        raster_array1.len(),
        raster_array2.len(),
        "Raster array lengths do not match"
    );

    for i in 0..raster_array1.len() {
        let raster1 = raster_array1.get(i).unwrap();
        let raster2 = raster_array2.get(i).unwrap();
        assert_raster_equal(&raster1, &raster2);
    }
}

/// Compare two rasters for equality
pub fn assert_raster_equal(raster1: &impl RasterRef, raster2: &impl RasterRef) {
    // Compare metadata
    let meta1 = raster1.metadata();
    let meta2 = raster2.metadata();
    assert_eq!(meta1.width(), meta2.width(), "Raster widths do not match");
    assert_eq!(
        meta1.height(),
        meta2.height(),
        "Raster heights do not match"
    );
    assert_eq!(
        meta1.upper_left_x(),
        meta2.upper_left_x(),
        "Raster upper left x does not match"
    );
    assert_eq!(
        meta1.upper_left_y(),
        meta2.upper_left_y(),
        "Raster upper left y does not match"
    );
    assert_eq!(
        meta1.scale_x(),
        meta2.scale_x(),
        "Raster scale x does not match"
    );
    assert_eq!(
        meta1.scale_y(),
        meta2.scale_y(),
        "Raster scale y does not match"
    );
    assert_eq!(
        meta1.skew_x(),
        meta2.skew_x(),
        "Raster skew x does not match"
    );
    assert_eq!(
        meta1.skew_y(),
        meta2.skew_y(),
        "Raster skew y does not match"
    );

    // Compare bands
    let bands1 = raster1.bands();
    let bands2 = raster2.bands();
    assert_eq!(bands1.len(), bands2.len(), "Number of bands do not match");

    for band_index in 0..bands1.len() {
        let band1 = bands1.band(band_index + 1).unwrap();
        let band2 = bands2.band(band_index + 1).unwrap();

        let band_meta1 = band1.metadata();
        let band_meta2 = band2.metadata();
        assert_eq!(
            band_meta1.data_type(),
            band_meta2.data_type(),
            "Band data types do not match"
        );
        assert_eq!(
            band_meta1.nodata_value(),
            band_meta2.nodata_value(),
            "Band nodata values do not match"
        );
        assert_eq!(
            band_meta1.storage_type(),
            band_meta2.storage_type(),
            "Band storage types do not match"
        );
        assert_eq!(
            band_meta1.outdb_url(),
            band_meta2.outdb_url(),
            "Band outdb URLs do not match"
        );
        assert_eq!(
            band_meta1.outdb_band_id(),
            band_meta2.outdb_band_id(),
            "Band outdb band IDs do not match"
        );

        assert_eq!(band1.data(), band2.data(), "Band data does not match");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sedona_raster::array::RasterStructArray;
    use sedona_raster::traits::RasterRef;

    #[test]
    fn test_generate_test_rasters() {
        let count = 5;
        let struct_array = generate_test_rasters(count, None).unwrap();
        let raster_array = RasterStructArray::new(&struct_array);
        assert_eq!(raster_array.len(), count);

        for i in 0..count {
            let raster = raster_array.get(i).unwrap();
            let metadata = raster.metadata();
            assert_eq!(metadata.width(), i as u64 + 1);
            assert_eq!(metadata.height(), i as u64 + 2);
            assert_eq!(metadata.upper_left_x(), i as f64 + 1.0);
            assert_eq!(metadata.upper_left_y(), i as f64 + 2.0);
            assert_eq!(metadata.scale_x(), (i as f64) * 0.1);
            assert_eq!(metadata.scale_y(), (i as f64) * 0.2);
            assert_eq!(metadata.skew_x(), (i as f64) * 0.3);
            assert_eq!(metadata.skew_y(), (i as f64) * 0.4);

            let bands = raster.bands();
            let band = bands.band(1).unwrap();
            let band_metadata = band.metadata();
            assert_eq!(band_metadata.data_type(), BandDataType::UInt16);
            assert_eq!(band_metadata.nodata_value(), Some(&[0u8, 0u8][..]));
            assert_eq!(band_metadata.storage_type(), StorageType::InDb);
            assert_eq!(band_metadata.outdb_url(), None);
            assert_eq!(band_metadata.outdb_band_id(), None);

            let band_data = band.data();
            let expected_pixel_count = (i + 1) * (i + 2); // width * height

            // Convert raw bytes back to u16 values for comparison
            let mut actual_pixel_values = Vec::new();
            for chunk in band_data.chunks_exact(2) {
                let value = u16::from_le_bytes([chunk[0], chunk[1]]);
                actual_pixel_values.push(value);
            }
            let expected_pixel_values: Vec<u16> = (0..expected_pixel_count as u16).collect();
            assert_eq!(actual_pixel_values, expected_pixel_values);
        }
    }

    #[test]
    fn test_generate_tiled_rasters() {
        let tile_size = (64, 64);
        let number_of_tiles = (4, 4);
        let data_type = BandDataType::UInt8;
        let struct_array =
            generate_tiled_rasters(tile_size, number_of_tiles, data_type, Some(43)).unwrap();
        let raster_array = RasterStructArray::new(&struct_array);
        assert_eq!(raster_array.len(), 16); // 4x4 tiles
        for i in 0..16 {
            let raster = raster_array.get(i).unwrap();
            let metadata = raster.metadata();
            assert_eq!(metadata.width(), 64);
            assert_eq!(metadata.height(), 64);
            assert_eq!(metadata.upper_left_x(), ((i % 4) * 64) as f64);
            assert_eq!(metadata.upper_left_y(), ((i / 4) * 64) as f64);
            let bands = raster.bands();
            assert_eq!(bands.len(), 3);
            for band_index in 0..3 {
                let band = bands.band(band_index + 1).unwrap();
                let band_metadata = band.metadata();
                assert_eq!(band_metadata.data_type(), BandDataType::UInt8);
                assert_eq!(band_metadata.storage_type(), StorageType::InDb);
                let band_data = band.data();
                assert_eq!(band_data.len(), 64 * 64); // 4096 pixels
            }
        }
    }

    #[test]
    fn test_raster_arrays_equal() {
        let raster_array1 = generate_test_rasters(3, None).unwrap();
        let raster_struct_array1 = RasterStructArray::new(&raster_array1);
        // Test that identical arrays are equal
        assert_raster_arrays_equal(&raster_struct_array1, &raster_struct_array1);
    }

    #[test]
    #[should_panic = "Raster array lengths do not match"]
    fn test_raster_arrays_not_equal() {
        let raster_array1 = generate_test_rasters(3, None).unwrap();
        let raster_struct_array1 = RasterStructArray::new(&raster_array1);

        // Test that arrays with different lengths are not equal
        let raster_array2 = generate_test_rasters(4, None).unwrap();
        let raster_struct_array2 = RasterStructArray::new(&raster_array2);
        assert_raster_arrays_equal(&raster_struct_array1, &raster_struct_array2);
    }

    #[test]
    fn test_raster_equal() {
        let raster_array1 =
            generate_tiled_rasters((256, 256), (1, 1), BandDataType::UInt8, Some(43)).unwrap();
        let raster1 = RasterStructArray::new(&raster_array1).get(0).unwrap();

        // Assert that the rasters are equal to themselves
        assert_raster_equal(&raster1, &raster1);
    }

    #[test]
    #[should_panic = "Band data does not match"]
    fn test_raster_different_band_data() {
        let raster_array1 =
            generate_tiled_rasters((128, 128), (1, 1), BandDataType::UInt8, Some(43)).unwrap();
        let raster_array2 =
            generate_tiled_rasters((128, 128), (1, 1), BandDataType::UInt8, Some(47)).unwrap();

        let raster1 = RasterStructArray::new(&raster_array1).get(0).unwrap();
        let raster2 = RasterStructArray::new(&raster_array2).get(0).unwrap();
        assert_raster_equal(&raster1, &raster2);
    }

    #[test]
    #[should_panic = "Raster upper left x does not match"]
    fn test_raster_different_metadata() {
        let raster_array =
            generate_tiled_rasters((128, 128), (2, 1), BandDataType::UInt8, Some(43)).unwrap();
        let raster1 = RasterStructArray::new(&raster_array).get(0).unwrap();
        let raster2 = RasterStructArray::new(&raster_array).get(1).unwrap();
        assert_raster_equal(&raster1, &raster2);
    }
}
