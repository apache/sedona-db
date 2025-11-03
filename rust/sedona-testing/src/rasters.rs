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
use sedona_raster::{
    array::RasterStructArray,
    builder::RasterBuilder,
    traits::{BandMetadata, RasterMetadata, RasterRef},
};
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

        let pixel_count = (i + 1) * (i + 2); // width * height
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

/// Compare two RasterStructArrays for equality
/// This compares each raster's metadata and band data for equality
pub fn raster_arrays_equal(raster_array1: &RasterStructArray, raster_array2: &RasterStructArray) -> bool {
    if raster_array1.len() != raster_array2.len() {
        return false;
    }

    for i in 0..raster_array1.len() {
        let raster1 = raster_array1.get(i).unwrap();
        let raster2 = raster_array2.get(i).unwrap();
        if !raster_equal(&raster1, &raster2) {
            return false;
        }
    }

    true
}

/// Compare two rasters for equality
/// This compares metadata and band data for equality
pub fn raster_equal(raster1: &impl RasterRef, raster2: &impl RasterRef) -> bool {
    // Compare metadata
    let meta1 = raster1.metadata();
    let meta2 = raster2.metadata();
    if meta1.width() != meta2.width()
        || meta1.height() != meta2.height()
        || meta1.upper_left_x() != meta2.upper_left_x()
        || meta1.upper_left_y() != meta2.upper_left_y()
        || meta1.scale_x() != meta2.scale_x()
        || meta1.scale_y() != meta2.scale_y()
        || meta1.skew_x() != meta2.skew_x()
        || meta1.skew_y() != meta2.skew_y()
    {
        return false;
    }

    // Compare bands
    let bands1 = raster1.bands();
    let bands2 = raster2.bands();
    if bands1.len() != bands2.len() {
        return false;
    }

    for band_index in 0..bands1.len() {
        let band1 = bands1.band(band_index + 1).unwrap();
        let band2 = bands2.band(band_index + 1).unwrap();

        let band_meta1 = band1.metadata();
        let band_meta2 = band2.metadata();
        if band_meta1.data_type() != band_meta2.data_type()
            || band_meta1.nodata_value() != band_meta2.nodata_value()
            || band_meta1.storage_type() != band_meta2.storage_type()
            || band_meta1.outdb_url() != band_meta2.outdb_url()
            || band_meta1.outdb_band_id() != band_meta2.outdb_band_id()
        {
            return false;
        }

        if band1.data() != band2.data() {
            return false;
        }
    }

    true
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
}
