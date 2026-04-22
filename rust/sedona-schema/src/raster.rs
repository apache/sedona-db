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

/// Schema for storing N-dimensional raster data in Apache Arrow format.
///
/// Each raster has a CRS, an affine transform, explicit spatial dimension names
/// (`x_dim`, `y_dim`), and a list of bands. Each band is an N-D chunk with named
/// dimensions, a shape, and optional strides for zero-copy slicing.
///
/// Legacy 2D rasters are represented as bands with `dim_names=["y","x"]` and
/// `shape=[height, width]`.
#[derive(Debug, PartialEq, Clone)]
pub struct RasterSchema;

impl RasterSchema {
    /// Returns the top-level fields for the raster schema structure.
    pub fn fields() -> Fields {
        Fields::from(vec![
            Field::new(column::CRS, Self::crs_type(), true),
            Field::new(column::TRANSFORM, Self::transform_type(), false),
            Field::new(column::X_DIM, DataType::Utf8View, false),
            Field::new(column::Y_DIM, DataType::Utf8View, false),
            Field::new(column::BANDS, Self::bands_type(), true),
        ])
    }

    /// Affine transform schema — 6-element GDAL GeoTransform:
    /// `[origin_x, scale_x, skew_x, origin_y, skew_y, scale_y]`
    pub fn transform_type() -> DataType {
        DataType::List(FieldRef::new(Field::new("item", DataType::Float64, false)))
    }

    /// Bands list schema
    pub fn bands_type() -> DataType {
        DataType::List(FieldRef::new(Field::new(
            column::BAND,
            Self::band_type(),
            false,
        )))
    }

    /// Individual band schema — flattened N-D band with dimension metadata.
    ///
    /// Out-of-band ("outdb") bands carry two orthogonal identifiers:
    /// - `outdb_uri` is the *location* (what scheme/registry to dispatch to,
    ///   e.g. `s3://bucket/file.tif`, `file:///…`, `mem://…`).
    /// - `outdb_format` is the *format* (how to interpret the bytes, e.g.
    ///   `"geotiff"`, `"zarr"`). Null format means in-memory — the band's
    ///   `data` buffer is authoritative.
    pub fn band_type() -> DataType {
        DataType::Struct(Fields::from(vec![
            Field::new(column::NAME, DataType::Utf8, true),
            Field::new(column::DIM_NAMES, Self::dim_names_type(), false),
            Field::new(column::SHAPE, Self::shape_type(), false),
            Field::new(column::DATATYPE, DataType::UInt32, false),
            Field::new(column::NODATA, DataType::Binary, true),
            Field::new(column::STRIDES, Self::strides_type(), false),
            Field::new(column::OFFSET, DataType::UInt64, false),
            Field::new(column::OUTDB_URI, DataType::Utf8, true),
            Field::new(column::OUTDB_FORMAT, DataType::Utf8View, true),
            Field::new(column::DATA, DataType::BinaryView, false),
        ]))
    }

    /// Dimension names list type
    pub fn dim_names_type() -> DataType {
        DataType::List(FieldRef::new(Field::new("item", DataType::Utf8, false)))
    }

    /// Shape list type (per-dimension sizes)
    pub fn shape_type() -> DataType {
        DataType::List(FieldRef::new(Field::new("item", DataType::UInt64, false)))
    }

    /// Strides list type (per-dimension byte strides)
    pub fn strides_type() -> DataType {
        DataType::List(FieldRef::new(Field::new("item", DataType::Int64, false)))
    }

    /// Coordinate Reference System (CRS) schema — stores CRS as JSON string
    pub fn crs_type() -> DataType {
        DataType::Utf8View
    }
}

/// Band data type enumeration for raster bands.
///
/// Only supports basic numeric types.
/// In future versions, consider support for complex types used in
/// radar and other wave-based data.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Copy)]
pub enum BandDataType {
    UInt8 = 1,
    UInt16 = 2,
    Int16 = 3,
    UInt32 = 4,
    Int32 = 5,
    Float32 = 6,
    Float64 = 7,
    UInt64 = 8,
    Int64 = 9,
    Int8 = 10,
}

impl BandDataType {
    /// Byte size of a single pixel for this data type.
    pub fn byte_size(&self) -> usize {
        match self {
            BandDataType::UInt8 | BandDataType::Int8 => 1,
            BandDataType::UInt16 | BandDataType::Int16 => 2,
            BandDataType::UInt32 | BandDataType::Int32 | BandDataType::Float32 => 4,
            BandDataType::UInt64 | BandDataType::Int64 | BandDataType::Float64 => 8,
        }
    }

    /// Try to convert from a u32 discriminant value.
    pub fn try_from_u32(value: u32) -> Option<Self> {
        match value {
            1 => Some(BandDataType::UInt8),
            2 => Some(BandDataType::UInt16),
            3 => Some(BandDataType::Int16),
            4 => Some(BandDataType::UInt32),
            5 => Some(BandDataType::Int32),
            6 => Some(BandDataType::Float32),
            7 => Some(BandDataType::Float64),
            8 => Some(BandDataType::UInt64),
            9 => Some(BandDataType::Int64),
            10 => Some(BandDataType::Int8),
            _ => None,
        }
    }

    /// Java/Sedona-compatible pixel type name (e.g. `"UNSIGNED_8BITS"`).
    pub fn pixel_type_name(&self) -> &'static str {
        match self {
            BandDataType::UInt8 => "UNSIGNED_8BITS",
            BandDataType::UInt16 => "UNSIGNED_16BITS",
            BandDataType::Int16 => "SIGNED_16BITS",
            BandDataType::Int32 => "SIGNED_32BITS",
            BandDataType::Float32 => "REAL_32BITS",
            BandDataType::Float64 => "REAL_64BITS",
            BandDataType::UInt32 => "UNSIGNED_32BITS",
            BandDataType::UInt64 => "UNSIGNED_64BITS",
            BandDataType::Int64 => "SIGNED_64BITS",
            BandDataType::Int8 => "SIGNED_8BITS",
        }
    }
}

/// Hard-coded column indices for performant access to nested struct fields.
/// These indices must match the exact order defined in the RasterSchema methods.
pub mod raster_indices {
    pub const CRS: usize = 0;
    pub const TRANSFORM: usize = 1;
    pub const X_DIM: usize = 2;
    pub const Y_DIM: usize = 3;
    pub const BANDS: usize = 4;
}

pub mod band_indices {
    pub const NAME: usize = 0;
    pub const DIM_NAMES: usize = 1;
    pub const SHAPE: usize = 2;
    pub const DATA_TYPE: usize = 3;
    pub const NODATA: usize = 4;
    pub const STRIDES: usize = 5;
    pub const OFFSET: usize = 6;
    pub const OUTDB_URI: usize = 7;
    pub const OUTDB_FORMAT: usize = 8;
    pub const DATA: usize = 9;
}

/// Column name constants used throughout the raster schema definition.
pub mod column {
    // Top-level raster fields
    pub const CRS: &str = "crs";
    pub const TRANSFORM: &str = "transform";
    pub const X_DIM: &str = "x_dim";
    pub const Y_DIM: &str = "y_dim";
    pub const BANDS: &str = "bands";
    pub const BAND: &str = "band";

    // Band fields
    pub const NAME: &str = "name";
    pub const DIM_NAMES: &str = "dim_names";
    pub const SHAPE: &str = "shape";
    pub const DATATYPE: &str = "data_type";
    pub const NODATA: &str = "nodata";
    pub const STRIDES: &str = "strides";
    pub const OFFSET: &str = "offset";
    pub const OUTDB_URI: &str = "outdb_uri";
    pub const OUTDB_FORMAT: &str = "outdb_format";
    pub const DATA: &str = "data";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raster_schema_fields() {
        let fields = RasterSchema::fields();
        assert_eq!(fields.len(), 5);
        assert_eq!(fields[0].name(), column::CRS);
        assert_eq!(fields[1].name(), column::TRANSFORM);
        assert_eq!(fields[2].name(), column::X_DIM);
        assert_eq!(fields[3].name(), column::Y_DIM);
        assert_eq!(fields[4].name(), column::BANDS);
    }

    #[test]
    fn test_hardcoded_indices_match_schema() {
        // Test raster-level indices
        let raster_fields = RasterSchema::fields();
        assert_eq!(raster_fields.len(), 5, "Expected exactly 5 raster fields");
        assert_eq!(
            raster_fields[raster_indices::CRS].name(),
            column::CRS,
            "Raster CRS index mismatch"
        );
        assert_eq!(
            raster_fields[raster_indices::TRANSFORM].name(),
            column::TRANSFORM,
            "Raster TRANSFORM index mismatch"
        );
        assert_eq!(
            raster_fields[raster_indices::X_DIM].name(),
            column::X_DIM,
            "Raster X_DIM index mismatch"
        );
        assert_eq!(
            raster_fields[raster_indices::Y_DIM].name(),
            column::Y_DIM,
            "Raster Y_DIM index mismatch"
        );
        assert_eq!(
            raster_fields[raster_indices::BANDS].name(),
            column::BANDS,
            "Raster BANDS index mismatch"
        );

        // Test band indices
        let band_type = RasterSchema::band_type();
        if let DataType::Struct(band_fields) = band_type {
            assert_eq!(band_fields.len(), 10, "Expected exactly 10 band fields");
            assert_eq!(band_fields[band_indices::NAME].name(), column::NAME);
            assert_eq!(
                band_fields[band_indices::DIM_NAMES].name(),
                column::DIM_NAMES
            );
            assert_eq!(band_fields[band_indices::SHAPE].name(), column::SHAPE);
            assert_eq!(
                band_fields[band_indices::DATA_TYPE].name(),
                column::DATATYPE
            );
            assert_eq!(band_fields[band_indices::NODATA].name(), column::NODATA);
            assert_eq!(band_fields[band_indices::STRIDES].name(), column::STRIDES);
            assert_eq!(band_fields[band_indices::OFFSET].name(), column::OFFSET);
            assert_eq!(
                band_fields[band_indices::OUTDB_URI].name(),
                column::OUTDB_URI
            );
            assert_eq!(
                band_fields[band_indices::OUTDB_FORMAT].name(),
                column::OUTDB_FORMAT
            );
            assert_eq!(band_fields[band_indices::DATA].name(), column::DATA);
        } else {
            panic!("Expected Struct type for band");
        }
    }

    #[test]
    fn test_band_data_type_byte_size() {
        assert_eq!(BandDataType::UInt8.byte_size(), 1);
        assert_eq!(BandDataType::Int8.byte_size(), 1);
        assert_eq!(BandDataType::UInt16.byte_size(), 2);
        assert_eq!(BandDataType::Int16.byte_size(), 2);
        assert_eq!(BandDataType::UInt32.byte_size(), 4);
        assert_eq!(BandDataType::Int32.byte_size(), 4);
        assert_eq!(BandDataType::Float32.byte_size(), 4);
        assert_eq!(BandDataType::UInt64.byte_size(), 8);
        assert_eq!(BandDataType::Int64.byte_size(), 8);
        assert_eq!(BandDataType::Float64.byte_size(), 8);
    }

    #[test]
    fn test_band_data_type_try_from_u32() {
        assert_eq!(BandDataType::try_from_u32(1), Some(BandDataType::UInt8));
        assert_eq!(BandDataType::try_from_u32(2), Some(BandDataType::UInt16));
        assert_eq!(BandDataType::try_from_u32(3), Some(BandDataType::Int16));
        assert_eq!(BandDataType::try_from_u32(4), Some(BandDataType::UInt32));
        assert_eq!(BandDataType::try_from_u32(5), Some(BandDataType::Int32));
        assert_eq!(BandDataType::try_from_u32(6), Some(BandDataType::Float32));
        assert_eq!(BandDataType::try_from_u32(7), Some(BandDataType::Float64));
        assert_eq!(BandDataType::try_from_u32(8), Some(BandDataType::UInt64));
        assert_eq!(BandDataType::try_from_u32(9), Some(BandDataType::Int64));
        assert_eq!(BandDataType::try_from_u32(10), Some(BandDataType::Int8));
        assert_eq!(BandDataType::try_from_u32(0), None);
        assert_eq!(BandDataType::try_from_u32(11), None);
        assert_eq!(BandDataType::try_from_u32(u32::MAX), None);
    }

    #[test]
    fn test_band_data_type_roundtrip_u32() {
        // Verify that discriminant → try_from_u32 round-trips for all variants
        let all_types = [
            BandDataType::UInt8,
            BandDataType::UInt16,
            BandDataType::Int16,
            BandDataType::UInt32,
            BandDataType::Int32,
            BandDataType::Float32,
            BandDataType::Float64,
            BandDataType::UInt64,
            BandDataType::Int64,
            BandDataType::Int8,
        ];
        for dt in all_types {
            let value = dt as u32;
            assert_eq!(
                BandDataType::try_from_u32(value),
                Some(dt),
                "Round-trip failed for {dt:?} (discriminant {value})"
            );
        }
    }

    #[test]
    fn test_band_data_type_pixel_type_name() {
        assert_eq!(BandDataType::UInt8.pixel_type_name(), "UNSIGNED_8BITS");
        assert_eq!(BandDataType::Int8.pixel_type_name(), "SIGNED_8BITS");
        assert_eq!(BandDataType::UInt16.pixel_type_name(), "UNSIGNED_16BITS");
        assert_eq!(BandDataType::Int16.pixel_type_name(), "SIGNED_16BITS");
        assert_eq!(BandDataType::UInt32.pixel_type_name(), "UNSIGNED_32BITS");
        assert_eq!(BandDataType::Int32.pixel_type_name(), "SIGNED_32BITS");
        assert_eq!(BandDataType::Float32.pixel_type_name(), "REAL_32BITS");
        assert_eq!(BandDataType::UInt64.pixel_type_name(), "UNSIGNED_64BITS");
        assert_eq!(BandDataType::Int64.pixel_type_name(), "SIGNED_64BITS");
        assert_eq!(BandDataType::Float64.pixel_type_name(), "REAL_64BITS");
    }
}
