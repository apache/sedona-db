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

use arrow_array::{
    builder::{
        BinaryBuilder, BinaryViewBuilder, BooleanBuilder, Float64Builder, StringBuilder,
        StringViewBuilder, UInt32Builder, UInt64Builder,
    },
    Array, ArrayRef, ListArray, StructArray,
};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{ArrowError, DataType};
use std::sync::Arc;

use sedona_schema::raster::RasterSchema;

use crate::traits::{BandMetadata, MetadataRef};

/// Builder for constructing raster arrays with zero-copy band data writing
///
/// Required steps to build a raster:
/// 1. Create a RasterBuilder with a specified capacity
/// 2. For each raster to add:
///    - Call `start_raster` with the appropriate metadata, CRS
///    - For each band in the raster:
///       - Call `start_band` with the band metadata
///       - Use `band_data_writer` to get a BinaryViewBuilder and write the band data
///       - Call `finish_band` to complete the band
///    - Call `finish_raster` to complete the raster
/// 3. After all rasters are added, call `finish` to get the final StructArray
///
/// Example usage:
/// ```
/// use sedona_raster::traits::{RasterMetadata, BandMetadata};
/// use sedona_schema::raster::{StorageType, BandDataType};
/// use sedona_raster::builder::RasterBuilder;
///
/// let mut builder = RasterBuilder::new(1);
/// let metadata = RasterMetadata {
///     width: 100, height: 100,
///     upperleft_x: 0.0, upperleft_y: 0.0,
///     scale_x: 1.0, scale_y: -1.0,
///     skew_x: 0.0, skew_y: 0.0,
/// };
/// // Start a raster from RasterMetadata struct
/// builder.start_raster(&metadata, Some("EPSG:4326")).unwrap();
///
/// // Add a band:
/// let band_metadata = BandMetadata {
///     nodata_value: Some(vec![0u8]),
///     storage_type: StorageType::InDb,
///     datatype: BandDataType::UInt8,
///     outdb_url: None,
///     outdb_band_id: None,
/// };
/// builder.start_band(band_metadata).unwrap();
/// let band_writer = builder.band_data_writer();
/// band_writer.append_value(&vec![/* band data bytes */]);
/// builder.finish_band().unwrap();
///
/// // Finish the raster
/// builder.finish_raster().unwrap();
///
/// // Finish building and get the StructArray
/// let raster_array = builder.finish().unwrap();
/// ```
pub struct RasterBuilder {
    // Metadata fields
    width: UInt64Builder,
    height: UInt64Builder,
    upper_left_x: Float64Builder,
    upper_left_y: Float64Builder,
    scale_x: Float64Builder,
    scale_y: Float64Builder,
    skew_x: Float64Builder,
    skew_y: Float64Builder,

    // CRS field
    crs: StringViewBuilder,

    // Band metadata fields
    band_nodata: BinaryBuilder,
    band_storage_type: UInt32Builder,
    band_datatype: UInt32Builder,
    band_outdb_url: StringBuilder,
    band_outdb_band_id: UInt32Builder,

    // Band data field
    band_data: BinaryViewBuilder,

    // List structure tracking
    band_offsets: Vec<i32>,  // Track where each raster's bands start/end
    current_band_count: i32, // Track bands in current raster

    raster_validity: BooleanBuilder, // Track which rasters are null
}

impl RasterBuilder {
    /// Create a new raster builder with the specified capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            // Metadata builders
            width: UInt64Builder::with_capacity(capacity),
            height: UInt64Builder::with_capacity(capacity),
            upper_left_x: Float64Builder::with_capacity(capacity),
            upper_left_y: Float64Builder::with_capacity(capacity),
            scale_x: Float64Builder::with_capacity(capacity),
            scale_y: Float64Builder::with_capacity(capacity),
            skew_x: Float64Builder::with_capacity(capacity),
            skew_y: Float64Builder::with_capacity(capacity),

            // CRS builder
            crs: StringViewBuilder::with_capacity(capacity),

            // Band builders - estimate some bands per raster
            // The capacity is at raster level, but each raster has multiple bands and
            // are large. We may want to add an optional parameter to control expected
            // bands per raster or even band size in the future
            band_nodata: BinaryBuilder::with_capacity(capacity, capacity),
            band_storage_type: UInt32Builder::with_capacity(capacity),
            band_datatype: UInt32Builder::with_capacity(capacity),
            band_outdb_url: StringBuilder::with_capacity(capacity, capacity),
            band_outdb_band_id: UInt32Builder::with_capacity(capacity),
            band_data: BinaryViewBuilder::with_capacity(capacity),

            // List tracking
            band_offsets: vec![0],
            current_band_count: 0,

            // Raster-level validity (keeps track of null rasters)
            raster_validity: BooleanBuilder::with_capacity(capacity),
        }
    }

    /// Start a new raster with metadata and optional CRS
    pub fn start_raster(
        &mut self,
        metadata: &dyn MetadataRef,
        crs: Option<&str>,
    ) -> Result<(), ArrowError> {
        self.append_metadata_from_ref(metadata)?;
        self.append_crs(crs)?;

        // Reset band count for this raster
        self.current_band_count = 0;

        Ok(())
    }

    /// Start a new band - this must be called before writing band data
    pub fn start_band(&mut self, band_metadata: BandMetadata) -> Result<(), ArrowError> {
        // Append band metadata
        match band_metadata.nodata_value {
            Some(nodata) => self.band_nodata.append_value(&nodata),
            None => self.band_nodata.append_null(),
        }

        self.band_storage_type
            .append_value(band_metadata.storage_type as u32);
        self.band_datatype
            .append_value(band_metadata.datatype as u32);

        match band_metadata.outdb_url {
            Some(url) => self.band_outdb_url.append_value(&url),
            None => self.band_outdb_url.append_null(),
        }

        match band_metadata.outdb_band_id {
            Some(band_id) => self.band_outdb_band_id.append_value(band_id),
            None => self.band_outdb_band_id.append_null(),
        }

        self.current_band_count += 1;

        Ok(())
    }

    /// Get direct access to the BinaryViewBuilder for writing the current band's data
    /// Must be called after start_band() to write data to the current band
    pub fn band_data_writer(&mut self) -> &mut BinaryViewBuilder {
        &mut self.band_data
    }

    /// Finish writing the current band
    pub fn finish_band(&mut self) -> Result<(), ArrowError> {
        // Band data should already be written via band_data_writer
        // Nothing additional needed here since we're building flat
        Ok(())
    }

    /// Finish all bands for the current raster
    pub fn finish_raster(&mut self) -> Result<(), ArrowError> {
        // Record the end offset for this raster's bands
        let next_offset = self.band_offsets.last().unwrap() + self.current_band_count;
        self.band_offsets.push(next_offset);

        self.raster_validity.append_value(true);

        Ok(())
    }

    /// Append raster metadata from a MetadataRef trait object
    fn append_metadata_from_ref(&mut self, metadata: &dyn MetadataRef) -> Result<(), ArrowError> {
        self.width.append_value(metadata.width());
        self.height.append_value(metadata.height());
        self.upper_left_x.append_value(metadata.upper_left_x());
        self.upper_left_y.append_value(metadata.upper_left_y());
        self.scale_x.append_value(metadata.scale_x());
        self.scale_y.append_value(metadata.scale_y());
        self.skew_x.append_value(metadata.skew_x());
        self.skew_y.append_value(metadata.skew_y());

        Ok(())
    }

    /// Set the CRS for the current raster
    pub fn append_crs(&mut self, crs: Option<&str>) -> Result<(), ArrowError> {
        match crs {
            Some(crs_data) => self.crs.append_value(crs_data),
            None => self.crs.append_null(),
        }
        Ok(())
    }

    /// Append a null raster
    pub fn append_null(&mut self) -> Result<(), ArrowError> {
        // Since metadata fields are non-nullable, provide default values
        self.width.append_value(0u64);
        self.height.append_value(0u64);
        self.upper_left_x.append_value(0.0f64);
        self.upper_left_y.append_value(0.0f64);
        self.scale_x.append_value(0.0f64);
        self.scale_y.append_value(0.0f64);
        self.skew_x.append_value(0.0f64);
        self.skew_y.append_value(0.0f64);

        // Append null CRS
        self.crs.append_null();

        // No bands for null raster
        let current_offset = *self.band_offsets.last().unwrap();
        self.band_offsets.push(current_offset);

        // Mark raster as null
        self.raster_validity.append_null();

        Ok(())
    }

    /// Finish building and return the constructed StructArray
    pub fn finish(mut self) -> Result<StructArray, ArrowError> {
        // Build the metadata struct using the schema
        let metadata_fields = if let DataType::Struct(fields) = RasterSchema::metadata_type() {
            fields
        } else {
            return Err(ArrowError::SchemaError(
                "Expected struct type for metadata".to_string(),
            ));
        };

        let metadata_arrays: Vec<ArrayRef> = vec![
            Arc::new(self.width.finish()),
            Arc::new(self.height.finish()),
            Arc::new(self.upper_left_x.finish()),
            Arc::new(self.upper_left_y.finish()),
            Arc::new(self.scale_x.finish()),
            Arc::new(self.scale_y.finish()),
            Arc::new(self.skew_x.finish()),
            Arc::new(self.skew_y.finish()),
        ];
        let metadata_array = StructArray::new(metadata_fields, metadata_arrays, None);

        // Build the band metadata struct using the schema
        let band_metadata_fields =
            if let DataType::Struct(fields) = RasterSchema::band_metadata_type() {
                fields
            } else {
                return Err(ArrowError::SchemaError(
                    "Expected struct type for band metadata".to_string(),
                ));
            };

        let band_metadata_arrays: Vec<ArrayRef> = vec![
            Arc::new(self.band_nodata.finish()),
            Arc::new(self.band_storage_type.finish()),
            Arc::new(self.band_datatype.finish()),
            Arc::new(self.band_outdb_url.finish()),
            Arc::new(self.band_outdb_band_id.finish()),
        ];
        let band_metadata_array =
            StructArray::new(band_metadata_fields, band_metadata_arrays, None);

        // Build the band struct using the schema
        let band_fields = if let DataType::Struct(fields) = RasterSchema::band_type() {
            fields
        } else {
            return Err(ArrowError::SchemaError(
                "Expected struct type for band".to_string(),
            ));
        };

        let band_arrays: Vec<ArrayRef> = vec![
            Arc::new(band_metadata_array),
            Arc::new(self.band_data.finish()),
        ];
        let band_struct_array = StructArray::new(band_fields, band_arrays, None);

        // Build the bands list array using the schema
        let band_field = if let DataType::List(field) = RasterSchema::bands_type() {
            field
        } else {
            return Err(ArrowError::SchemaError(
                "Expected list type for bands".to_string(),
            ));
        };

        let offsets = OffsetBuffer::new(ScalarBuffer::from(self.band_offsets));
        let bands_list = ListArray::new(band_field, offsets, Arc::new(band_struct_array), None);

        // Build the final raster struct using the schema
        let raster_fields = RasterSchema::fields();
        let raster_arrays: Vec<ArrayRef> = vec![
            Arc::new(metadata_array),
            Arc::new(self.crs.finish()),
            Arc::new(bands_list),
        ];

        let raster_validity_array = self.raster_validity.finish();
        let raster_nulls = raster_validity_array.nulls().cloned();

        Ok(StructArray::new(raster_fields, raster_arrays, raster_nulls))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::RasterStructArray;
    use crate::traits::{RasterMetadata, RasterRef};
    use sedona_schema::raster::{BandDataType, StorageType};

    #[test]
    fn test_iterator_basic_functionality() {
        // Create a simple raster for testing using the correct API
        let mut builder = RasterBuilder::new(10); // capacity

        let metadata = RasterMetadata {
            width: 10,
            height: 10,
            upperleft_x: 0.0,
            upperleft_y: 0.0,
            scale_x: 1.0,
            scale_y: -1.0,
            skew_x: 0.0,
            skew_y: 0.0,
        };

        let epsg4326 = "EPSG:4326";
        builder.start_raster(&metadata, Some(epsg4326)).unwrap();

        let band_metadata = BandMetadata {
            nodata_value: Some(vec![255u8]),
            storage_type: StorageType::InDb,
            datatype: BandDataType::UInt8,
            outdb_url: None,
            outdb_band_id: None,
        };

        // Add a single band with some test data using the correct API
        builder.start_band(band_metadata.clone()).unwrap();
        let test_data = vec![1u8; 100]; // 10x10 raster with value 1
        builder.band_data_writer().append_value(&test_data);
        builder.finish_band().unwrap();
        let result = builder.finish_raster();
        assert!(result.is_ok());

        let raster_array = builder.finish().unwrap();

        // Test the iterator
        let rasters = RasterStructArray::new(&raster_array);

        assert_eq!(rasters.len(), 1);
        assert!(!rasters.is_empty());

        let raster = rasters.get(0).unwrap();
        let metadata = raster.metadata();

        assert_eq!(metadata.width(), 10);
        assert_eq!(metadata.height(), 10);
        assert_eq!(metadata.scale_x(), 1.0);
        assert_eq!(metadata.scale_y(), -1.0);

        let bands = raster.bands();
        assert_eq!(bands.len(), 1);
        assert!(!bands.is_empty());

        // Access band with 1-based band_number
        let band = bands.band(1).unwrap();
        assert_eq!(band.data().len(), 100);
        assert_eq!(band.data()[0], 1u8);

        let band_meta = band.metadata();
        assert_eq!(band_meta.storage_type(), StorageType::InDb);
        assert_eq!(band_meta.data_type(), BandDataType::UInt8);

        let crs = raster.crs().unwrap();
        assert_eq!(crs, epsg4326);

        // Test iterator over bands
        let band_iter: Vec<_> = bands.iter().collect();
        assert_eq!(band_iter.len(), 1);
    }

    #[test]
    fn test_multi_band_iterator() {
        let mut builder = RasterBuilder::new(3);

        let metadata = RasterMetadata {
            width: 5,
            height: 5,
            upperleft_x: 0.0,
            upperleft_y: 0.0,
            scale_x: 1.0,
            scale_y: -1.0,
            skew_x: 0.0,
            skew_y: 0.0,
        };

        builder.start_raster(&metadata, None).unwrap();

        // Add three bands using the correct API
        for band_idx in 0..3 {
            let band_metadata = BandMetadata {
                nodata_value: Some(vec![255u8]),
                storage_type: StorageType::InDb,
                datatype: BandDataType::UInt8,
                outdb_url: None,
                outdb_band_id: None,
            };

            builder.start_band(band_metadata).unwrap();
            let test_data = vec![band_idx as u8; 25]; // 5x5 raster
            builder.band_data_writer().append_value(&test_data);
            builder.finish_band().unwrap();
        }

        let result = builder.finish_raster();
        assert!(result.is_ok());

        let raster_array = builder.finish().unwrap();

        let rasters = RasterStructArray::new(&raster_array);
        let raster = rasters.get(0).unwrap();
        let bands = raster.bands();

        assert_eq!(bands.len(), 3);

        // Test each band has different data
        // Use 1-based band numbers
        for i in 0..3 {
            // Access band with 1-based band_number
            let band = bands.band(i + 1).unwrap();
            let expected_value = i as u8;
            assert!(band.data().iter().all(|&x| x == expected_value));
        }

        // Test iterator
        let band_values: Vec<u8> = bands
            .iter()
            .enumerate()
            .map(|(i, band)| {
                assert_eq!(band.data()[0], i as u8);
                band.data()[0]
            })
            .collect();

        assert_eq!(band_values, vec![0, 1, 2]);
    }

    #[test]
    fn test_copy_metadata_from_iterator() {
        // Create an original raster
        let mut source_builder = RasterBuilder::new(10);

        let original_metadata = RasterMetadata {
            width: 42,
            height: 24,
            upperleft_x: -122.0,
            upperleft_y: 37.8,
            scale_x: 0.1,
            scale_y: -0.1,
            skew_x: 0.0,
            skew_y: 0.0,
        };

        source_builder
            .start_raster(&original_metadata, None)
            .unwrap();

        let band_metadata = BandMetadata {
            nodata_value: Some(vec![255u8]),
            storage_type: StorageType::InDb,
            datatype: BandDataType::UInt8,
            outdb_url: None,
            outdb_band_id: None,
        };

        source_builder.start_band(band_metadata).unwrap();
        let test_data = vec![42u8; 1008]; // 42x24 raster
        source_builder.band_data_writer().append_value(&test_data);
        source_builder.finish_band().unwrap();
        source_builder.finish_raster().unwrap();

        let source_array = source_builder.finish().unwrap();

        // Create a new raster using metadata from the iterator
        let mut target_builder = RasterBuilder::new(10);
        let iterator = RasterStructArray::new(&source_array);
        let source_raster = iterator.get(0).unwrap();

        target_builder
            .start_raster(source_raster.metadata(), source_raster.crs())
            .unwrap();

        // Add new band data while preserving original metadata
        let new_band_metadata = BandMetadata {
            nodata_value: None,
            storage_type: StorageType::InDb,
            datatype: BandDataType::UInt16,
            outdb_url: None,
            outdb_band_id: None,
        };

        target_builder.start_band(new_band_metadata).unwrap();
        let new_data = vec![100u16; 1008]; // Different data, same dimensions
        let new_data_bytes: Vec<u8> = new_data.iter().flat_map(|&x| x.to_le_bytes()).collect();

        target_builder
            .band_data_writer()
            .append_value(&new_data_bytes);
        target_builder.finish_band().unwrap();
        target_builder.finish_raster().unwrap();

        let target_array = target_builder.finish().unwrap();

        // Verify the metadata was copied correctly
        let target_iterator = RasterStructArray::new(&target_array);
        let target_raster = target_iterator.get(0).unwrap();
        let target_metadata = target_raster.metadata();

        // All metadata should match the original
        assert_eq!(target_metadata.width(), 42);
        assert_eq!(target_metadata.height(), 24);
        assert_eq!(target_metadata.upper_left_x(), -122.0);
        assert_eq!(target_metadata.upper_left_y(), 37.8);
        assert_eq!(target_metadata.scale_x(), 0.1);
        assert_eq!(target_metadata.scale_y(), -0.1);

        // But band data and metadata should be different
        let target_band = target_raster.bands().band(1).unwrap();
        let target_band_meta = target_band.metadata();
        assert_eq!(target_band_meta.data_type(), BandDataType::UInt16);
        assert!(target_band_meta.nodata_value().is_none());
        assert_eq!(target_band.data().len(), 2016); // 1008 * 2 bytes per u16

        let result = target_raster.bands().band(0);
        assert!(result.is_err(), "Band number 0 should be invalid");

        let result = target_raster.bands().band(2);
        assert!(result.is_err(), "Band number 2 should be out of range");
    }

    #[test]
    fn test_band_data_types() {
        // Create a test raster with bands of different data types
        let mut builder = RasterBuilder::new(1);

        let metadata = RasterMetadata {
            width: 2,
            height: 2,
            upperleft_x: 0.0,
            upperleft_y: 0.0,
            scale_x: 1.0,
            scale_y: -1.0,
            skew_x: 0.0,
            skew_y: 0.0,
        };

        builder.start_raster(&metadata, None).unwrap();

        // Test all BandDataType variants
        let test_cases = vec![
            (BandDataType::UInt8, vec![1u8, 2u8, 3u8, 4u8]),
            (
                BandDataType::UInt16,
                vec![1u8, 0u8, 2u8, 0u8, 3u8, 0u8, 4u8, 0u8],
            ), // little-endian u16
            (
                BandDataType::Int16,
                vec![255u8, 255u8, 254u8, 255u8, 253u8, 255u8, 252u8, 255u8],
            ), // little-endian i16
            (
                BandDataType::UInt32,
                vec![
                    1u8, 0u8, 0u8, 0u8, 2u8, 0u8, 0u8, 0u8, 3u8, 0u8, 0u8, 0u8, 4u8, 0u8, 0u8, 0u8,
                ],
            ), // little-endian u32
            (
                BandDataType::Int32,
                vec![
                    255u8, 255u8, 255u8, 255u8, 254u8, 255u8, 255u8, 255u8, 253u8, 255u8, 255u8,
                    255u8, 252u8, 255u8, 255u8, 255u8,
                ],
            ), // little-endian i32
            (
                BandDataType::Float32,
                vec![
                    0u8, 0u8, 128u8, 63u8, 0u8, 0u8, 0u8, 64u8, 0u8, 0u8, 64u8, 64u8, 0u8, 0u8,
                    128u8, 64u8,
                ],
            ), // little-endian f32: 1.0, 2.0, 3.0, 4.0
            (
                BandDataType::Float64,
                vec![
                    0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 240u8, 63u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    64u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 8u8, 64u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    16u8, 64u8,
                ],
            ), // little-endian f64: 1.0, 2.0, 3.0, 4.0
        ];

        for (expected_data_type, test_data) in test_cases {
            let band_metadata = BandMetadata {
                nodata_value: None,
                storage_type: StorageType::InDb,
                datatype: expected_data_type.clone(),
                outdb_url: None,
                outdb_band_id: None,
            };

            builder.start_band(band_metadata).unwrap();
            builder.band_data_writer().append_value(&test_data);
            builder.finish_band().unwrap();
        }

        builder.finish_raster().unwrap();
        let raster_array = builder.finish().unwrap();

        // Test the data type conversion for each band
        let iterator = RasterStructArray::new(&raster_array);
        let raster = iterator.get(0).unwrap();
        let bands = raster.bands();

        assert_eq!(bands.len(), 7, "Expected 7 bands for all data types");

        // Verify each band returns the correct data type
        let expected_types = [
            BandDataType::UInt8,
            BandDataType::UInt16,
            BandDataType::Int16,
            BandDataType::UInt32,
            BandDataType::Int32,
            BandDataType::Float32,
            BandDataType::Float64,
        ];

        // i is zero-based index
        for (i, expected_type) in expected_types.iter().enumerate() {
            // Bands are 1-based band_number
            let band = bands.band(i + 1).unwrap();
            let band_metadata = band.metadata();
            let actual_type = band_metadata.data_type();

            assert_eq!(
                actual_type, *expected_type,
                "Band {} expected data type {:?}, got {:?}",
                i, expected_type, actual_type
            );
        }
    }

    #[test]
    fn test_outdb_metadata_fields() {
        // Test creating raster with OutDb reference metadata
        let mut builder = RasterBuilder::new(10);

        let metadata = RasterMetadata {
            width: 1024,
            height: 1024,
            upperleft_x: 0.0,
            upperleft_y: 0.0,
            scale_x: 1.0,
            scale_y: -1.0,
            skew_x: 0.0,
            skew_y: 0.0,
        };

        builder.start_raster(&metadata, None).unwrap();

        // Test InDb band (should have null OutDb fields)
        let indb_band_metadata = BandMetadata {
            nodata_value: Some(vec![255u8]),
            storage_type: StorageType::InDb,
            datatype: BandDataType::UInt8,
            outdb_url: None,
            outdb_band_id: None,
        };

        builder.start_band(indb_band_metadata).unwrap();
        let test_data = vec![1u8; 100];
        builder.band_data_writer().append_value(&test_data);
        builder.finish_band().unwrap();

        // Test OutDbRef band (should have OutDb fields populated)
        let outdb_band_metadata = BandMetadata {
            nodata_value: None,
            storage_type: StorageType::OutDbRef,
            datatype: BandDataType::Float32,
            outdb_url: Some("s3://mybucket/satellite_image.tif".to_string()),
            outdb_band_id: Some(2),
        };

        builder.start_band(outdb_band_metadata).unwrap();
        // For OutDbRef, data field could be empty or contain metadata/thumbnail
        builder.band_data_writer().append_value([]);
        builder.finish_band().unwrap();

        builder.finish_raster().unwrap();
        let raster_array = builder.finish().unwrap();

        // Verify the band metadata
        let iterator = RasterStructArray::new(&raster_array);
        let raster = iterator.get(0).unwrap();
        let bands = raster.bands();

        assert_eq!(bands.len(), 2);

        // Test InDb band
        let indb_band = bands.band(1).unwrap();
        let indb_metadata = indb_band.metadata();
        assert_eq!(indb_metadata.storage_type(), StorageType::InDb);
        assert_eq!(indb_metadata.data_type(), BandDataType::UInt8);
        assert!(indb_metadata.outdb_url().is_none());
        assert!(indb_metadata.outdb_band_id().is_none());
        assert_eq!(indb_band.data().len(), 100);

        // Test OutDbRef band
        let outdb_band = bands.band(2).unwrap();
        let outdb_metadata = outdb_band.metadata();
        assert_eq!(outdb_metadata.storage_type(), StorageType::OutDbRef);
        assert_eq!(outdb_metadata.data_type(), BandDataType::Float32);
        assert_eq!(
            outdb_metadata.outdb_url().unwrap(),
            "s3://mybucket/satellite_image.tif"
        );
        assert_eq!(outdb_metadata.outdb_band_id().unwrap(), 2);
        assert_eq!(outdb_band.data().len(), 0); // Empty data for OutDbRef
    }

    #[test]
    fn test_band_access_errors() {
        // Create a simple raster with one band
        let mut builder = RasterBuilder::new(1);

        let metadata = RasterMetadata {
            width: 10,
            height: 10,
            upperleft_x: 0.0,
            upperleft_y: 0.0,
            scale_x: 1.0,
            scale_y: -1.0,
            skew_x: 0.0,
            skew_y: 0.0,
        };

        builder.start_raster(&metadata, None).unwrap();

        let band_metadata = BandMetadata {
            nodata_value: None,
            storage_type: StorageType::InDb,
            datatype: BandDataType::UInt8,
            outdb_url: None,
            outdb_band_id: None,
        };

        builder.start_band(band_metadata).unwrap();
        builder.band_data_writer().append_value([1u8; 100]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        let raster_array = builder.finish().unwrap();
        let iterator = RasterStructArray::new(&raster_array);
        let raster = iterator.get(0).unwrap();
        let bands = raster.bands();

        // Test invalid band number (0-based)
        let result = bands.band(0);
        assert!(result.is_err());
        let err = result.err().unwrap().to_string();
        assert!(err.contains("band numbers must be 1-based"));

        // Test out of range band number
        let result = bands.band(2);
        assert!(result.is_err());
        let err = result.err().unwrap().to_string();
        assert!(err.contains("is out of range"));

        // Test valid band number should still work
        let result = bands.band(1);
        assert!(result.is_ok());
        let band = result.unwrap();
        assert_eq!(band.data().len(), 100);
    }
}
