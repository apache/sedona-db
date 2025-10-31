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
use arrow_schema::{DataType, Field, FieldRef, Fields};

/// Schema for storing raster data in Apache Arrow format.
/// Utilizing nested structs and lists to represent raster metadata and bands.
#[derive(Debug, PartialEq, Clone)]
pub struct RasterSchema;
impl RasterSchema {
    /// Returns the top-level fields for the raster schema structure.
    pub fn fields() -> Fields {
        Fields::from(vec![
            Field::new(column::METADATA, Self::metadata_type(), false),
            Field::new(column::CRS, Self::crs_type(), true), // Optional: may be inferred from data
            Field::new(column::BANDS, Self::bands_type(), true),
        ])
    }

    /// Raster metadata schema
    pub fn metadata_type() -> DataType {
        DataType::Struct(Fields::from(vec![
            // Raster dimensions
            Field::new(column::WIDTH, DataType::UInt64, false),
            Field::new(column::HEIGHT, DataType::UInt64, false),
            // Geospatial transformation parameters
            Field::new(column::UPPERLEFT_X, DataType::Float64, false),
            Field::new(column::UPPERLEFT_Y, DataType::Float64, false),
            Field::new(column::SCALE_X, DataType::Float64, false),
            Field::new(column::SCALE_Y, DataType::Float64, false),
            Field::new(column::SKEW_X, DataType::Float64, false),
            Field::new(column::SKEW_Y, DataType::Float64, false),
        ]))
    }

    /// Bands list schema
    pub fn bands_type() -> DataType {
        DataType::List(FieldRef::new(Field::new(
            column::BAND,
            Self::band_type(),
            false,
        )))
    }

    /// Individual band schema
    pub fn band_type() -> DataType {
        DataType::Struct(Fields::from(vec![
            Field::new(column::METADATA, Self::band_metadata_type(), false),
            Field::new(column::DATA, Self::band_data_type(), false),
        ]))
    }

    /// Band metadata schema
    pub fn band_metadata_type() -> DataType {
        DataType::Struct(Fields::from(vec![
            Field::new(column::NODATAVALUE, DataType::Binary, true), // Optional: null means no nodata value specified
            Field::new(column::STORAGE_TYPE, DataType::UInt32, false),
            Field::new(column::DATATYPE, DataType::UInt32, false),
            // OutDb reference fields - only used when storage_type == OutDbRef
            Field::new(column::OUTDB_URL, DataType::Utf8, true),
            Field::new(column::OUTDB_BAND_ID, DataType::UInt32, true),
        ]))
    }

    /// Band data schema - stores the actual raster pixel data as a binary blob
    pub fn band_data_type() -> DataType {
        DataType::BinaryView
    }

    /// Coordinate Reference System (CRS) schema - stores CRS as JSON string (PROJ or WKT format)
    pub fn crs_type() -> DataType {
        DataType::Utf8View
    }
}

/// Band data type enumeration for raster bands.
///
/// Only supports basic numeric types.
/// In future versions, consider support for complex types used in
/// radar and other wave-based data.
#[repr(u16)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BandDataType {
    UInt8 = 0,
    UInt16 = 1,
    Int16 = 2,
    UInt32 = 3,
    Int32 = 4,
    Float32 = 5,
    Float64 = 6,
}

/// Storage strategy for raster band data within Apache Arrow arrays.
///
/// This enum defines how raster data is physically stored and accessed:
///
/// **InDb**: Raster data is embedded directly in the Arrow array as binary blobs.
///   - Self-contained, no external dependencies, fast access for small-medium rasters
///   - Increases Arrow array size, memory usage grows and copy times increase with raster size
///   - Best for: Tiles, thumbnails, processed results, small rasters (<10MB per band)
///
/// **OutDbRef**: Raster data is stored externally with references in the Arrow array.
///   - Keeps Arrow arrays lightweight, supports massive rasters, enables lazy loading
///   - Requires external storage management, potential for broken references
///   - Best for: Large satellite imagery, time series data, cloud-native workflows
///   - Supported backends: S3, GCS, Azure Blob, local filesystem, HTTP endpoints
#[repr(u16)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StorageType {
    InDb = 0,
    OutDbRef = 1,
}

/// Hard-coded column indices for performant access to nested struct fields.
/// These indices must match the exact order defined in the RasterSchema methods.
///
/// Using compile-time constants avoids string lookups and provides type safety
/// when accessing nested struct fields in Arrow arrays.
pub mod metadata_indices {
    pub const WIDTH: usize = 0;
    pub const HEIGHT: usize = 1;
    pub const UPPERLEFT_X: usize = 2;
    pub const UPPERLEFT_Y: usize = 3;
    pub const SCALE_X: usize = 4;
    pub const SCALE_Y: usize = 5;
    pub const SKEW_X: usize = 6;
    pub const SKEW_Y: usize = 7;
}

pub mod band_metadata_indices {
    pub const NODATAVALUE: usize = 0;
    pub const STORAGE_TYPE: usize = 1;
    pub const DATATYPE: usize = 2;
    pub const OUTDB_URL: usize = 3;
    pub const OUTDB_BAND_ID: usize = 4;
}

pub mod band_indices {
    pub const METADATA: usize = 0;
    pub const DATA: usize = 1;
}

pub mod raster_indices {
    pub const METADATA: usize = 0;
    pub const CRS: usize = 1;
    pub const BANDS: usize = 2;
}

/// Column name constants used throughout the raster schema definition.
/// These string constants ensure consistency across schema creation and field access.
pub mod column {
    pub const METADATA: &str = "metadata";
    pub const BANDS: &str = "bands";
    pub const BAND: &str = "band";
    pub const DATA: &str = "data";

    // Raster metadata fields
    pub const WIDTH: &str = "width";
    pub const HEIGHT: &str = "height";
    pub const UPPERLEFT_X: &str = "upperleft_x";
    pub const UPPERLEFT_Y: &str = "upperleft_y";
    pub const SCALE_X: &str = "scale_x";
    pub const SCALE_Y: &str = "scale_y";
    pub const SKEW_X: &str = "skew_x";
    pub const SKEW_Y: &str = "skew_y";

    // Raster CRS field
    pub const CRS: &str = "crs";
    // Band metadata fields
    pub const NODATAVALUE: &str = "nodata_value";
    pub const STORAGE_TYPE: &str = "storage_type";
    pub const DATATYPE: &str = "data_type";
    pub const OUTDB_URL: &str = "outdb_url";
    pub const OUTDB_BAND_ID: &str = "outdb_band_id";
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Tests that the top-level raster schema has the expected number and names of fields.
    #[test]
    fn test_raster_schema_fields() {
        let fields = RasterSchema::fields();
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0].name(), column::METADATA);
        assert_eq!(fields[1].name(), column::CRS);
        assert_eq!(fields[2].name(), column::BANDS);
    }

    /// Comprehensive test to verify all hard-coded indices match the actual schema.
    /// This ensures that performance optimizations using direct index access remain valid
    /// when the schema structure changes.
    #[test]
    fn test_hardcoded_indices_match_schema() {
        // Test raster-level indices
        let raster_fields = RasterSchema::fields();
        assert_eq!(raster_fields.len(), 3, "Expected exactly 3 raster fields");
        assert_eq!(
            raster_fields[raster_indices::METADATA].name(),
            column::METADATA,
            "Raster metadata index mismatch"
        );
        assert_eq!(
            raster_fields[raster_indices::CRS].name(),
            column::CRS,
            "Raster CRS index mismatch"
        );
        assert_eq!(
            raster_fields[raster_indices::BANDS].name(),
            column::BANDS,
            "Raster BANDS index mismatch"
        );

        // Test metadata indices
        let metadata_type = RasterSchema::metadata_type();
        if let DataType::Struct(metadata_fields) = metadata_type {
            assert_eq!(
                metadata_fields.len(),
                8,
                "Expected exactly 8 metadata fields"
            );
            assert_eq!(
                metadata_fields[metadata_indices::WIDTH].name(),
                column::WIDTH,
                "Metadata width index mismatch"
            );
            assert_eq!(
                metadata_fields[metadata_indices::HEIGHT].name(),
                column::HEIGHT,
                "Metadata height index mismatch"
            );
            assert_eq!(
                metadata_fields[metadata_indices::UPPERLEFT_X].name(),
                column::UPPERLEFT_X,
                "Metadata upperleft_x index mismatch"
            );
            assert_eq!(
                metadata_fields[metadata_indices::UPPERLEFT_Y].name(),
                column::UPPERLEFT_Y,
                "Metadata upperleft_y index mismatch"
            );
            assert_eq!(
                metadata_fields[metadata_indices::SCALE_X].name(),
                column::SCALE_X,
                "Metadata scale_x index mismatch"
            );
            assert_eq!(
                metadata_fields[metadata_indices::SCALE_Y].name(),
                column::SCALE_Y,
                "Metadata scale_y index mismatch"
            );
            assert_eq!(
                metadata_fields[metadata_indices::SKEW_X].name(),
                column::SKEW_X,
                "Metadata skew_x index mismatch"
            );
            assert_eq!(
                metadata_fields[metadata_indices::SKEW_Y].name(),
                column::SKEW_Y,
                "Metadata skew_y index mismatch"
            );
        } else {
            panic!("Expected Struct type for metadata");
        }

        // Test band metadata indices
        let band_metadata_type = RasterSchema::band_metadata_type();
        if let DataType::Struct(band_metadata_fields) = band_metadata_type {
            assert_eq!(
                band_metadata_fields.len(),
                5,
                "Expected exactly 5 band metadata fields"
            );
            assert_eq!(
                band_metadata_fields[band_metadata_indices::NODATAVALUE].name(),
                column::NODATAVALUE,
                "Band metadata nodatavalue index mismatch"
            );
            assert_eq!(
                band_metadata_fields[band_metadata_indices::STORAGE_TYPE].name(),
                column::STORAGE_TYPE,
                "Band metadata storage_type index mismatch"
            );
            assert_eq!(
                band_metadata_fields[band_metadata_indices::DATATYPE].name(),
                column::DATATYPE,
                "Band metadata datatype index mismatch"
            );
            assert_eq!(
                band_metadata_fields[band_metadata_indices::OUTDB_URL].name(),
                column::OUTDB_URL,
                "Band metadata outdb_url index mismatch"
            );
            assert_eq!(
                band_metadata_fields[band_metadata_indices::OUTDB_BAND_ID].name(),
                column::OUTDB_BAND_ID,
                "Band metadata outdb_band_id index mismatch"
            );
        } else {
            panic!("Expected Struct type for band metadata");
        }

        // Test band indices
        let band_type = RasterSchema::band_type();
        if let DataType::Struct(band_fields) = band_type {
            assert_eq!(band_fields.len(), 2, "Expected exactly 2 band fields");
            assert_eq!(
                band_fields[band_indices::METADATA].name(),
                column::METADATA,
                "Band metadata index mismatch"
            );
            assert_eq!(
                band_fields[band_indices::DATA].name(),
                column::DATA,
                "Band data index mismatch"
            );
        } else {
            panic!("Expected Struct type for band");
        }
    }
}
