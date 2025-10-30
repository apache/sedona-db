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
    Array, ArrayRef, BinaryArray, BinaryViewArray, Float64Array, ListArray, StringArray,
    StringViewArray, StructArray, UInt32Array, UInt64Array,
};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{ArrowError, DataType};
use std::sync::Arc;

use sedona_schema::raster::{
    band_indices, band_metadata_indices, bounding_box_indices, metadata_indices, raster_indices,
    BandDataType, RasterSchema, StorageType,
};

/// Metadata for a raster
#[derive(Debug, Clone)]
pub struct RasterMetadata {
    pub width: u64,
    pub height: u64,
    pub upperleft_x: f64,
    pub upperleft_y: f64,
    pub scale_x: f64,
    pub scale_y: f64,
    pub skew_x: f64,
    pub skew_y: f64,
}

/// Bounding box coordinates
#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

/// Metadata for a single band
#[derive(Debug, Clone)]
pub struct BandMetadata {
    pub nodata_value: Option<Vec<u8>>,
    pub storage_type: StorageType,
    pub datatype: BandDataType,
    /// URL for OutDb reference (only used when storage_type == OutDbRef)
    pub outdb_url: Option<String>,
    /// Band ID within the OutDb resource (only used when storage_type == OutDbRef)
    pub outdb_band_id: Option<u32>,
}

/// Builder for constructing raster arrays with zero-copy band data writing
///
/// Required steps to build a raster:
/// 1. Create a RasterBuilder with a specified capacity
/// 2. For each raster to add:
///    - Call `start_raster` with the appropriate metadata, CRS, and bounding box
///    - For each band in the raster:
///       - Call `start_band` with the band metadata
///       - Use `band_data_writer` to get a BinaryViewBuilder and write the band data
///       - Call `finish_band` to complete the band
///    - Call `finish_raster` to complete the raster
/// 3. After all rasters are added, call `finish` to get the final StructArray
///
/// Example usage:
/// ```
/// use sedona_raster::builder::{RasterBuilder, RasterMetadata, BoundingBox, BandMetadata};
/// use sedona_schema::raster::{StorageType, BandDataType};
///
/// let mut builder = RasterBuilder::new(1);
/// let metadata = RasterMetadata {
///     width: 100, height: 100,
///     upperleft_x: 0.0, upperleft_y: 0.0,
///     scale_x: 1.0, scale_y: -1.0,
///     skew_x: 0.0, skew_y: 0.0,
/// };
/// let bbox = BoundingBox { min_x: 0.0, min_y: 0.0, max_x: 100.0, max_y: 100.0 };
///
/// // Start a raster:
/// // From RasterMetadata struct with separate bounding box
/// builder.start_raster(&metadata, Some("EPSG:4326"), Some(&bbox)).unwrap();
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

    // Bounding box fields
    bbox_min_x: Float64Builder,
    bbox_min_y: Float64Builder,
    bbox_max_x: Float64Builder,
    bbox_max_y: Float64Builder,
    bbox_validity: BooleanBuilder, // Track which bboxes are null

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

            // Bounding box builders
            bbox_min_x: Float64Builder::with_capacity(capacity),
            bbox_min_y: Float64Builder::with_capacity(capacity),
            bbox_max_x: Float64Builder::with_capacity(capacity),
            bbox_max_y: Float64Builder::with_capacity(capacity),
            // Bounding box validity (keeps track of null bounding boxes)
            bbox_validity: BooleanBuilder::with_capacity(capacity),

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

    /// Start a new raster with metadata, optional CRS, and optional bounding box
    pub fn start_raster(
        &mut self,
        metadata: &dyn MetadataRef,
        crs: Option<&str>,
        bbox: Option<&BoundingBox>,
    ) -> Result<(), ArrowError> {
        self.append_metadata_from_ref(metadata)?;
        self.append_crs(crs)?;
        self.append_bounding_box(bbox)?;

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

    /// Append a bounding box to the current raster
    pub fn append_bounding_box(&mut self, bbox: Option<&BoundingBox>) -> Result<(), ArrowError> {
        if let Some(bbox) = bbox {
            self.bbox_min_x.append_value(bbox.min_x);
            self.bbox_min_y.append_value(bbox.min_y);
            self.bbox_max_x.append_value(bbox.max_x);
            self.bbox_max_y.append_value(bbox.max_y);
            self.bbox_validity.append_value(true);
        } else {
            // For null bounding box, append default values since fields are non-nullable
            // but mark the struct as null via validity buffer
            self.bbox_min_x.append_value(0.0);
            self.bbox_min_y.append_value(0.0);
            self.bbox_max_x.append_value(0.0);
            self.bbox_max_y.append_value(0.0);
            self.bbox_validity.append_value(false);
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

        // Append null bounding box
        self.append_bounding_box(None)?;

        // No bands for null raster
        let current_offset = *self.band_offsets.last().unwrap();
        self.band_offsets.push(current_offset);

        // Mark raster as null
        self.raster_validity.append_value(false);

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

        // Build the bounding box struct using the schema
        let bbox_fields = if let DataType::Struct(fields) = RasterSchema::bounding_box_type() {
            fields
        } else {
            return Err(ArrowError::SchemaError(
                "Expected struct type for bounding box".to_string(),
            ));
        };

        let bbox_arrays: Vec<ArrayRef> = vec![
            Arc::new(self.bbox_min_x.finish()),
            Arc::new(self.bbox_min_y.finish()),
            Arc::new(self.bbox_max_x.finish()),
            Arc::new(self.bbox_max_y.finish()),
        ];
        let bbox_validity = self.bbox_validity.finish();
        let bbox_array = StructArray::new(bbox_fields, bbox_arrays, bbox_validity.nulls().cloned());

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
            Arc::new(bbox_array),
            Arc::new(bands_list),
        ];

        let raster_validity_array = self.raster_validity.finish();
        let raster_nulls = raster_validity_array.nulls().cloned();

        Ok(StructArray::new(raster_fields, raster_arrays, raster_nulls))
    }
}

/// Trait for accessing individual band metadata
pub trait BandMetadataRef {
    /// No-data value as raw bytes (None if null)
    fn nodata_value(&self) -> Option<&[u8]>;
    /// Storage type (InDb, OutDbRef, etc)
    fn storage_type(&self) -> StorageType;
    /// Band data type (UInt8, Float32, etc.)
    fn data_type(&self) -> BandDataType;
    /// OutDb URL (only used when storage_type == OutDbRef)
    fn outdb_url(&self) -> Option<&str>;
    /// OutDb band ID (only used when storage_type == OutDbRef)
    fn outdb_band_id(&self) -> Option<u32>;
}

/// Trait for accessing individual band data
pub trait BandRef {
    /// Band metadata accessor
    fn metadata(&self) -> &dyn BandMetadataRef;
    /// Raw band data as bytes (zero-copy access)
    fn data(&self) -> &[u8];
}

/// Trait for accessing all bands in a raster
pub trait BandsRef {
    /// Number of bands in the raster
    fn len(&self) -> usize;
    /// Check if no bands are present
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Get a specific band by number (returns Error if out of bounds)
    /// By convention, band numbers are 1-based
    fn band(&self, number: usize) -> Result<Box<dyn BandRef + '_>, ArrowError>;
    /// Iterator over all bands
    fn iter(&self) -> BandIterator<'_>;
}

/// Trait for accessing complete raster data
pub trait RasterRef {
    /// Raster metadata accessor
    fn metadata(&self) -> &dyn MetadataRef;
    /// CRS accessor
    fn crs(&self) -> Option<&str>;
    /// Bounding box accessor (optional)
    fn bounding_box(&self) -> Option<&dyn BoundingBoxRef>;
    /// Bands accessor
    fn bands(&self) -> &dyn BandsRef;
}

/// Trait for accessing raster metadata (dimensions, geotransform, bounding box, etc.)
pub trait MetadataRef {
    /// Width of the raster in pixels
    fn width(&self) -> u64;
    /// Height of the raster in pixels
    fn height(&self) -> u64;
    /// X coordinate of the upper-left corner
    fn upper_left_x(&self) -> f64;
    /// Y coordinate of the upper-left corner
    fn upper_left_y(&self) -> f64;
    /// X-direction pixel size (scale)
    fn scale_x(&self) -> f64;
    /// Y-direction pixel size (scale)
    fn scale_y(&self) -> f64;
    /// X-direction skew/rotation
    fn skew_x(&self) -> f64;
    /// Y-direction skew/rotation
    fn skew_y(&self) -> f64;
}

/// Trait for accessing raster bounding box coordinates
pub trait BoundingBoxRef {
    /// Minimum X coordinate
    fn min_x(&self) -> f64;
    /// Minimum Y coordinate
    fn min_y(&self) -> f64;
    /// Maximum X coordinate
    fn max_x(&self) -> f64;
    /// Maximum Y coordinate
    fn max_y(&self) -> f64;
}

/// Implement MetadataRef for RasterMetadata to allow direct use with builder
impl MetadataRef for RasterMetadata {
    fn width(&self) -> u64 {
        self.width
    }
    fn height(&self) -> u64 {
        self.height
    }
    fn upper_left_x(&self) -> f64 {
        self.upperleft_x
    }
    fn upper_left_y(&self) -> f64 {
        self.upperleft_y
    }
    fn scale_x(&self) -> f64 {
        self.scale_x
    }
    fn scale_y(&self) -> f64 {
        self.scale_y
    }
    fn skew_x(&self) -> f64 {
        self.skew_x
    }
    fn skew_y(&self) -> f64 {
        self.skew_y
    }
}

/// Implement BoundingBoxRef for BoundingBox to allow direct use with traits
impl BoundingBoxRef for BoundingBox {
    fn min_x(&self) -> f64 {
        self.min_x
    }
    fn min_y(&self) -> f64 {
        self.min_y
    }
    fn max_x(&self) -> f64 {
        self.max_x
    }
    fn max_y(&self) -> f64 {
        self.max_y
    }
}

/// Implementation of MetadataRef for Arrow StructArray
struct MetadataRefImpl<'a> {
    width_array: &'a UInt64Array,
    height_array: &'a UInt64Array,
    upper_left_x_array: &'a Float64Array,
    upper_left_y_array: &'a Float64Array,
    scale_x_array: &'a Float64Array,
    scale_y_array: &'a Float64Array,
    skew_x_array: &'a Float64Array,
    skew_y_array: &'a Float64Array,
    index: usize,
}

impl<'a> MetadataRef for MetadataRefImpl<'a> {
    fn width(&self) -> u64 {
        self.width_array.value(self.index)
    }

    fn height(&self) -> u64 {
        self.height_array.value(self.index)
    }

    fn upper_left_x(&self) -> f64 {
        self.upper_left_x_array.value(self.index)
    }

    fn upper_left_y(&self) -> f64 {
        self.upper_left_y_array.value(self.index)
    }

    fn scale_x(&self) -> f64 {
        self.scale_x_array.value(self.index)
    }

    fn scale_y(&self) -> f64 {
        self.scale_y_array.value(self.index)
    }

    fn skew_x(&self) -> f64 {
        self.skew_x_array.value(self.index)
    }

    fn skew_y(&self) -> f64 {
        self.skew_y_array.value(self.index)
    }
}

/// Implementation of BoundingBoxRef for Arrow StructArray
struct BoundingBoxRefImpl<'a> {
    min_x_array: &'a Float64Array,
    min_y_array: &'a Float64Array,
    max_x_array: &'a Float64Array,
    max_y_array: &'a Float64Array,
    index: usize,
}

impl<'a> BoundingBoxRef for BoundingBoxRefImpl<'a> {
    fn min_x(&self) -> f64 {
        self.min_x_array.value(self.index)
    }

    fn min_y(&self) -> f64 {
        self.min_y_array.value(self.index)
    }

    fn max_x(&self) -> f64 {
        self.max_x_array.value(self.index)
    }

    fn max_y(&self) -> f64 {
        self.max_y_array.value(self.index)
    }
}

/// Implementation of BandMetadataRef for Arrow StructArray
struct BandMetadataRefImpl<'a> {
    nodata_array: &'a BinaryArray,
    storage_type_array: &'a UInt32Array,
    datatype_array: &'a UInt32Array,
    outdb_url_array: &'a StringArray,
    outdb_band_id_array: &'a UInt32Array,
    band_index: usize,
}

impl<'a> BandMetadataRef for BandMetadataRefImpl<'a> {
    fn nodata_value(&self) -> Option<&[u8]> {
        if self.nodata_array.is_null(self.band_index) {
            None
        } else {
            Some(self.nodata_array.value(self.band_index))
        }
    }

    fn storage_type(&self) -> StorageType {
        match self.storage_type_array.value(self.band_index) {
            0 => StorageType::InDb,
            1 => StorageType::OutDbRef,
            _ => panic!(
                "Unknown storage type: {}",
                self.storage_type_array.value(self.band_index)
            ),
        }
    }

    fn data_type(&self) -> BandDataType {
        match self.datatype_array.value(self.band_index) {
            0 => BandDataType::UInt8,
            1 => BandDataType::UInt16,
            2 => BandDataType::Int16,
            3 => BandDataType::UInt32,
            4 => BandDataType::Int32,
            5 => BandDataType::Float32,
            6 => BandDataType::Float64,
            _ => panic!(
                "Unknown band data type: {}",
                self.datatype_array.value(self.band_index)
            ),
        }
    }

    fn outdb_url(&self) -> Option<&str> {
        if self.outdb_url_array.is_null(self.band_index) {
            None
        } else {
            Some(self.outdb_url_array.value(self.band_index))
        }
    }

    fn outdb_band_id(&self) -> Option<u32> {
        if self.outdb_band_id_array.is_null(self.band_index) {
            None
        } else {
            Some(self.outdb_band_id_array.value(self.band_index))
        }
    }
}

/// Implementation of BandRef for accessing individual band data
struct BandRefImpl<'a> {
    band_metadata: BandMetadataRefImpl<'a>,
    band_data: &'a [u8],
}

impl<'a> BandRef for BandRefImpl<'a> {
    fn metadata(&self) -> &dyn BandMetadataRef {
        &self.band_metadata
    }

    fn data(&self) -> &[u8] {
        self.band_data
    }
}

/// Implementation of BandsRef for accessing all bands in a raster
struct BandsRefImpl<'a> {
    bands_list: &'a ListArray,
    raster_index: usize,
}

impl<'a> BandsRef for BandsRefImpl<'a> {
    fn len(&self) -> usize {
        let start = self.bands_list.value_offsets()[self.raster_index] as usize;
        let end = self.bands_list.value_offsets()[self.raster_index + 1] as usize;
        end - start
    }

    /// Get a specific band by number (1-based index)
    fn band(&self, number: usize) -> Result<Box<dyn BandRef + '_>, ArrowError> {
        if number == 0 {
            return Err(ArrowError::InvalidArgumentError(format!(
                "Invalid band number {}: band numbers must be 1-based",
                number
            )));
        }
        // By convention, band numbers are 1-based.
        // Convert to zero-based index.
        let index = number - 1;
        if index >= self.len() {
            return Err(ArrowError::InvalidArgumentError(format!(
                "Band number {} is out of range: this raster has {} bands",
                number,
                self.len()
            )));
        }

        let start = self.bands_list.value_offsets()[self.raster_index] as usize;
        let band_row = start + index;

        let bands_struct = self
            .bands_list
            .values()
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or(ArrowError::SchemaError(
                "Failed to downcast to StructArray".to_string(),
            ))?;

        // Get the metadata substructure from the band struct
        let band_metadata_struct = bands_struct
            .column(band_indices::METADATA)
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or(ArrowError::SchemaError(
                "Failed to downcast metadata to StructArray".to_string(),
            ))?;

        let band_metadata = BandMetadataRefImpl {
            nodata_array: band_metadata_struct
                .column(band_metadata_indices::NODATAVALUE)
                .as_any()
                .downcast_ref::<BinaryArray>()
                .ok_or(ArrowError::SchemaError(
                    "Failed to downcast nodata to BinaryArray".to_string(),
                ))?,
            storage_type_array: band_metadata_struct
                .column(band_metadata_indices::STORAGE_TYPE)
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or(ArrowError::SchemaError(
                    "Failed to downcast storage_type to UInt32Array".to_string(),
                ))?,
            datatype_array: band_metadata_struct
                .column(band_metadata_indices::DATATYPE)
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or(ArrowError::SchemaError(
                    "Failed to downcast datatype to UInt32Array".to_string(),
                ))?,
            outdb_url_array: band_metadata_struct
                .column(band_metadata_indices::OUTDB_URL)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or(ArrowError::SchemaError(
                    "Failed to downcast outdb_url to StringArray".to_string(),
                ))?,
            outdb_band_id_array: band_metadata_struct
                .column(band_metadata_indices::OUTDB_BAND_ID)
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or(ArrowError::SchemaError(
                    "Failed to downcast outdb_band_id to UInt32Array".to_string(),
                ))?,
            band_index: band_row,
        };

        // Get band data from the Binary column within the band struct
        let band_data_array = bands_struct
            .column(band_indices::DATA)
            .as_any()
            .downcast_ref::<BinaryViewArray>()
            .ok_or(ArrowError::SchemaError(
                "Failed to downcast data to BinaryViewArray".to_string(),
            ))?;

        let band_data = band_data_array.value(band_row);

        Ok(Box::new(BandRefImpl {
            band_metadata,
            band_data,
        }))
    }

    fn iter(&self) -> BandIterator<'_> {
        BandIterator {
            bands: self,
            current: 1, // Start at 1 for 1-based band numbering
        }
    }
}

/// Iterator for bands within a raster
pub struct BandIterator<'a> {
    bands: &'a dyn BandsRef,
    current: usize,
}

impl<'a> Iterator for BandIterator<'a> {
    type Item = Box<dyn BandRef + 'a>;

    fn next(&mut self) -> Option<Self::Item> {
        // current is 1-based, compare against len() + 1
        if self.current <= self.bands.len() {
            let band = self.bands.band(self.current).ok(); // Convert Result to Option
            self.current += 1;
            band
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // current is 1-based, so remaining calculation needs adjustment
        let remaining = self.bands.len().saturating_sub(self.current - 1);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for BandIterator<'_> {}

/// Implementation of RasterRef for complete raster access
pub struct RasterRefImpl<'a> {
    metadata: MetadataRefImpl<'a>,
    crs: &'a StringViewArray,
    bbox: Option<BoundingBoxRefImpl<'a>>,
    bands: BandsRefImpl<'a>,
}

impl<'a> RasterRefImpl<'a> {
    /// Create a new RasterRefImpl from a struct array and index using hard-coded indices
    pub fn new(raster_struct: &'a StructArray, raster_index: usize) -> Self {
        let metadata_struct = raster_struct
            .column(raster_indices::METADATA)
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();

        let crs = raster_struct
            .column(raster_indices::CRS)
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap();

        let bbox = raster_struct
            .column(raster_indices::BBOX)
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();

        let bands_list = raster_struct
            .column(raster_indices::BANDS)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();

        let metadata = MetadataRefImpl {
            width_array: metadata_struct
                .column(metadata_indices::WIDTH)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap(),
            height_array: metadata_struct
                .column(metadata_indices::HEIGHT)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap(),
            upper_left_x_array: metadata_struct
                .column(metadata_indices::UPPERLEFT_X)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap(),
            upper_left_y_array: metadata_struct
                .column(metadata_indices::UPPERLEFT_Y)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap(),
            scale_x_array: metadata_struct
                .column(metadata_indices::SCALE_X)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap(),
            scale_y_array: metadata_struct
                .column(metadata_indices::SCALE_Y)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap(),
            skew_x_array: metadata_struct
                .column(metadata_indices::SKEW_X)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap(),
            skew_y_array: metadata_struct
                .column(metadata_indices::SKEW_Y)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap(),
            index: raster_index,
        };

        let bands = BandsRefImpl {
            bands_list,
            raster_index,
        };

        // Create optional bounding box ref if not null
        let bbox_ref = if bbox.is_null(raster_index) {
            None
        } else {
            Some(BoundingBoxRefImpl {
                min_x_array: bbox
                    .column(bounding_box_indices::MIN_X)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap(),
                min_y_array: bbox
                    .column(bounding_box_indices::MIN_Y)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap(),
                max_x_array: bbox
                    .column(bounding_box_indices::MAX_X)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap(),
                max_y_array: bbox
                    .column(bounding_box_indices::MAX_Y)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap(),
                index: raster_index,
            })
        };

        Self {
            metadata,
            crs,
            bbox: bbox_ref,
            bands,
        }
    }

    /// Access the bounding box for this raster
    pub fn bounding_box(&self) -> Option<BoundingBox> {
        self.bbox.as_ref().map(|bbox_ref| BoundingBox {
            min_x: bbox_ref.min_x(),
            min_y: bbox_ref.min_y(),
            max_x: bbox_ref.max_x(),
            max_y: bbox_ref.max_y(),
        })
    }
}

impl<'a> RasterRef for RasterRefImpl<'a> {
    fn metadata(&self) -> &dyn MetadataRef {
        &self.metadata
    }

    fn crs(&self) -> Option<&str> {
        if self.crs.is_null(self.bands.raster_index) {
            None
        } else {
            Some(self.crs.value(self.bands.raster_index))
        }
    }

    fn bounding_box(&self) -> Option<&dyn BoundingBoxRef> {
        self.bbox
            .as_ref()
            .map(|bbox_ref| bbox_ref as &dyn BoundingBoxRef)
    }

    fn bands(&self) -> &dyn BandsRef {
        &self.bands
    }
}

/// Iterate over raster structs in an Arrow StructArray
///
/// This provides efficient, zero-copy access to raster data stored in Arrow format.
/// Each iteration yields a `RasterRefImpl` that provides access to both metadata and band data.
pub struct RasterStructIterator<'a> {
    raster_array: &'a StructArray,
    current_row: usize,
}

impl<'a> RasterStructIterator<'a> {
    /// Create a new iterator over the raster struct array
    pub fn new(raster_array: &'a StructArray) -> Self {
        Self {
            raster_array,
            current_row: 0,
        }
    }

    /// Get the total number of rasters in the array
    pub fn len(&self) -> usize {
        self.raster_array.len()
    }

    /// Check if the array is empty
    pub fn is_empty(&self) -> bool {
        self.raster_array.is_empty()
    }

    /// Get a specific raster by index without consuming the iterator
    pub fn get(&self, index: usize) -> Option<RasterRefImpl<'a>> {
        if index >= self.raster_array.len() {
            return None;
        }

        Some(RasterRefImpl::new(self.raster_array, index))
    }
}

impl<'a> Iterator for RasterStructIterator<'a> {
    type Item = RasterRefImpl<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row >= self.raster_array.len() {
            return None;
        }

        let item = RasterRefImpl::new(self.raster_array, self.current_row);
        self.current_row += 1;
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.raster_array.len() - self.current_row;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for RasterStructIterator<'a> {}

#[cfg(test)]
mod tests {
    use super::*;
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
        let bbox = BoundingBox {
            min_x: 0.0,
            min_y: -10.0,
            max_x: 10.0,
            max_y: 0.0,
        };
        builder
            .start_raster(&metadata, Some(epsg4326), Some(&bbox))
            .unwrap();

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
        let mut iterator = RasterStructIterator::new(&raster_array);

        assert_eq!(iterator.len(), 1);
        assert!(!iterator.is_empty());

        let raster = iterator.next().unwrap();
        let metadata = raster.metadata();

        assert_eq!(metadata.width(), 10);
        assert_eq!(metadata.height(), 10);
        assert_eq!(metadata.scale_x(), 1.0);
        assert_eq!(metadata.scale_y(), -1.0);

        let bbox = raster.bounding_box().unwrap();
        assert_eq!(bbox.min_x, 0.0);
        assert_eq!(bbox.max_x, 10.0);

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

        builder.start_raster(&metadata, None, None).unwrap();

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

        let mut iterator = RasterStructIterator::new(&raster_array);
        let raster = iterator.next().unwrap();
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

        let original_bbox = BoundingBox {
            min_x: -122.0,
            min_y: 35.4,
            max_x: -120.0,
            max_y: 37.8,
        };

        source_builder
            .start_raster(&original_metadata, None, Some(&original_bbox))
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
        let iterator = RasterStructIterator::new(&source_array);
        let source_raster = iterator.get(0).unwrap();

        target_builder
            .start_raster(
                source_raster.metadata(),
                source_raster.crs(),
                source_raster.bounding_box().as_ref(),
            )
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
        let target_iterator = RasterStructIterator::new(&target_array);
        let target_raster = target_iterator.get(0).unwrap();
        let target_metadata = target_raster.metadata();

        // All metadata should match the original
        assert_eq!(target_metadata.width(), 42);
        assert_eq!(target_metadata.height(), 24);
        assert_eq!(target_metadata.upper_left_x(), -122.0);
        assert_eq!(target_metadata.upper_left_y(), 37.8);
        assert_eq!(target_metadata.scale_x(), 0.1);
        assert_eq!(target_metadata.scale_y(), -0.1);

        let target_bbox = target_raster.bounding_box().unwrap();
        assert_eq!(target_bbox.min_x, -122.0);
        assert_eq!(target_bbox.max_x, -120.0);

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

        builder.start_raster(&metadata, None, None).unwrap();

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
        let iterator = RasterStructIterator::new(&raster_array);
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

        builder.start_raster(&metadata, None, None).unwrap();

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
        let iterator = RasterStructIterator::new(&raster_array);
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

        builder.start_raster(&metadata, None, None).unwrap();

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
        let iterator = RasterStructIterator::new(&raster_array);
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
