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
use arrow_schema::ArrowError;
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::{BandMetadata, RasterMetadata};
use sedona_schema::raster::{BandDataType, StorageType};

/// Generate a StructArray of rasters with sequentially increasing dimensions and pixel values
/// These tiny rasters are to provide fast, easy and predictable test data for unit tests.
pub fn generate_test_rasters(
    count: usize,
    null_raster_index: Option<usize>,
) -> Result<StructArray, ArrowError> {
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

        let pixel_count = i * (i + 1);
        let mut band_data = Vec::with_capacity(pixel_count * 2); // 2 bytes per u16
        for pixel_value in 0..pixel_count as u16 {
            band_data.extend_from_slice(&pixel_value.to_le_bytes());
        }

        builder.band_data_writer().append_value(&band_data);
        builder.finish_band()?;
        builder.finish_raster()?;
    }

    builder.finish()
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
            let expected_pixel_count = i * (i + 1); // width * height

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
}
