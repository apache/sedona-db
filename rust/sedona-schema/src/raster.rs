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
/// Each raster has a CRS, an affine transform, a list of spatial dimension
/// names (`spatial_dims`) and sizes (`spatial_shape`), and a list of bands.
/// Each band is an N-D chunk with named dimensions, a `source_shape`
/// describing the natural extent of its underlying buffer, and a `view`
/// describing the visible region of that buffer.
///
/// `spatial_dims` + `spatial_shape` are the raster-level source of truth for
/// the spatial grid — today length 2 (`["x","y"]`, `[width, height]`),
/// Z-ready for a future 3D phase. All bands must contain every name in
/// `spatial_dims` in their own `dim_names`, with the band's *visible* size
/// for that dim matching `spatial_shape`.
///
/// 2D rasters are represented as bands with `dim_names=["y","x"]` and
/// `source_shape=[height, width]`.
#[derive(Debug, PartialEq, Clone)]
pub struct RasterSchema;

impl RasterSchema {
    /// Returns the top-level fields for the raster schema structure.
    pub fn fields() -> Fields {
        Fields::from(vec![
            Field::new(column::CRS, Self::crs_type(), true), // Optional: may be inferred from data
            Field::new(column::TRANSFORM, Self::transform_type(), false),
            Field::new(column::SPATIAL_DIMS, Self::spatial_dims_type(), false),
            Field::new(column::SPATIAL_SHAPE, Self::spatial_shape_type(), false),
            Field::new(column::BANDS, Self::bands_type(), true),
        ])
    }

    /// Affine transform schema — 6-element GDAL GeoTransform:
    /// `[origin_x, scale_x, skew_x, origin_y, skew_y, scale_y]`
    pub fn transform_type() -> DataType {
        DataType::List(FieldRef::new(Field::new("item", DataType::Float64, false)))
    }

    /// Spatial dimension names schema — list of `Utf8View` strings, one per
    /// spatial axis. Today always `["x","y"]`; becomes `["x","y","z"]` if a
    /// future phase adds Z support.
    pub fn spatial_dims_type() -> DataType {
        DataType::List(FieldRef::new(Field::new("item", DataType::Utf8View, false)))
    }

    /// Spatial shape schema — list of `Int64` sizes in the same order as
    /// `spatial_dims`. Today `[width, height]`.
    pub fn spatial_shape_type() -> DataType {
        DataType::List(FieldRef::new(Field::new("item", DataType::Int64, false)))
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
    /// Pixels live in one of two places:
    /// - **In-db**: `data` holds the band's raw little-endian, C-order buffer
    ///   of `Π source_shape × byte_size` bytes, and `chunks` is null.
    /// - **Out-db**: `data` is empty and `chunks` lists a regular grid of
    ///   chunk references over `source_shape`, `ceil(source_shape /
    ///   chunk_shape)` entries per dimension in row-major order. Every entry
    ///   is one object, or a byte range of one, that the band's `codecs` (a
    ///   Zarr v3 codec pipeline as JSON, shared by all chunks of the band)
    ///   decode into one chunk of `chunk_shape` pixels. A null entry is a
    ///   missing chunk.
    ///
    /// A band may carry both after `RS_EnsureLoaded`: `data` is then
    /// authoritative and `chunks` is provenance that a writer keeps so the
    /// band writes back as references, not pixels.
    ///
    /// The chunk struct uses the field names and types of the Iceberg V4
    /// `file` type and Parquet's `FILE` logical type. `inline` is always null:
    /// in-db pixels stay in `data`, uncompressed, so they remain zero-copy.
    pub fn band_type() -> DataType {
        DataType::Struct(Fields::from(vec![
            Field::new(column::NAME, DataType::Utf8, true),
            Field::new(column::DIM_NAMES, Self::dim_names_type(), false),
            Field::new(column::SOURCE_SHAPE, Self::source_shape_type(), false),
            Field::new(column::DATATYPE, DataType::UInt32, false),
            Field::new(column::NODATA, DataType::Binary, true),
            Field::new(column::VIEW, Self::view_type(), true),
            Field::new(column::DATA, DataType::BinaryView, false),
            Field::new(column::CHUNK_SHAPE, Self::chunk_shape_type(), true),
            Field::new(column::CODECS, Self::codecs_type(), true),
            Field::new(column::CHUNKS, Self::chunks_type(), true),
        ]))
    }

    /// Chunk shape list type — pixels per chunk per dimension, parallel to
    /// the band's `dim_names`. Null for an in-db band.
    pub fn chunk_shape_type() -> DataType {
        DataType::List(FieldRef::new(Field::new("item", DataType::Int64, false)))
    }

    /// Codec pipeline type — a Zarr v3 codec pipeline as a JSON array of
    /// `{"name": …, "configuration": {…}}` objects, shared by every chunk of
    /// the band. Null for an in-db band.
    pub fn codecs_type() -> DataType {
        DataType::Utf8View
    }

    /// Chunks list type — one entry per grid position in row-major order.
    /// Entries are nullable: a null entry is a missing chunk. The list itself
    /// is null for an in-db band.
    pub fn chunks_type() -> DataType {
        DataType::List(FieldRef::new(Field::new(
            column::CHUNK,
            Self::chunk_type(),
            true,
        )))
    }

    /// One chunk reference, shaped like the Iceberg V4 `file` type: `uri`,
    /// the optional byte range `offset`/`size` (`offset` requires `size`),
    /// `content_type` (unused; `codecs` decides decoding), `checksum`
    /// (`ETAG:<etag>` of the source object) and `inline` (always null).
    pub fn chunk_type() -> DataType {
        DataType::Struct(Fields::from(vec![
            Field::new(column::URI, DataType::Utf8View, true),
            Field::new(column::OFFSET, DataType::Int64, true),
            Field::new(column::SIZE, DataType::Int64, true),
            Field::new(column::CONTENT_TYPE, DataType::Utf8View, true),
            Field::new(column::CHECKSUM, DataType::Utf8View, true),
            Field::new(column::INLINE, DataType::BinaryView, true),
        ]))
    }

    /// Dimension names list type
    pub fn dim_names_type() -> DataType {
        DataType::List(FieldRef::new(Field::new("item", DataType::Utf8, false)))
    }

    /// Source shape list type — the natural C-order extent of the band's
    /// `data` buffer (or outdb-resolved source) per dimension. The *visible*
    /// shape exposed to consumers is derived from `view`:
    /// `[entry.steps for entry in view]`.
    pub fn source_shape_type() -> DataType {
        DataType::List(FieldRef::new(Field::new("item", DataType::Int64, false)))
    }

    /// View list type — one entry per dimension in the band's *visible*
    /// order. Each entry is a `(source_axis, start, step, steps)` quadruple
    /// describing how the visible axis maps onto the band's source shape.
    /// The field is nullable: a null view denotes the identity view
    /// `[(i, 0, 1, source_shape[i]) for i in 0..ndim]` and is the canonical
    /// representation for any band whose data has not been sliced. See
    /// `RasterSchema` doc for full semantics.
    pub fn view_type() -> DataType {
        DataType::List(FieldRef::new(Field::new(
            "item",
            DataType::Struct(Fields::from(vec![
                Field::new("source_axis", DataType::Int64, false),
                Field::new("start", DataType::Int64, false),
                Field::new("step", DataType::Int64, false),
                Field::new("steps", DataType::Int64, false),
            ])),
            false,
        )))
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
    // Int8 was added after the original 1-7 set (PR #589) and after the
    // 64-bit additions at 8-9. The discriminants are an Arrow-column
    // contract for the `band.data_type` UInt32 column — reordering would
    // silently misinterpret existing raster data, so new variants append.
    Int8 = 10,
}

impl BandDataType {
    /// The minimum representable value of this data type, as the little-endian
    /// bytes of the type itself. Conventionally used as a nodata sentinel when
    /// no explicit nodata is available — never a silent 0, which would be
    /// indistinguishable from real zero-valued pixels. Byte form (rather than
    /// `f64`) so `Int64`/`UInt64` minima stay exact.
    pub fn min_value_le_bytes(&self) -> Vec<u8> {
        match self {
            BandDataType::UInt8 => u8::MIN.to_le_bytes().to_vec(),
            BandDataType::UInt16 => u16::MIN.to_le_bytes().to_vec(),
            BandDataType::UInt32 => u32::MIN.to_le_bytes().to_vec(),
            BandDataType::UInt64 => u64::MIN.to_le_bytes().to_vec(),
            BandDataType::Int8 => i8::MIN.to_le_bytes().to_vec(),
            BandDataType::Int16 => i16::MIN.to_le_bytes().to_vec(),
            BandDataType::Int32 => i32::MIN.to_le_bytes().to_vec(),
            BandDataType::Int64 => i64::MIN.to_le_bytes().to_vec(),
            BandDataType::Float32 => f32::MIN.to_le_bytes().to_vec(),
            BandDataType::Float64 => f64::MIN.to_le_bytes().to_vec(),
        }
    }

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
            // Extra types present in Rust but not in Java Sedona
            BandDataType::UInt32 => "UNSIGNED_32BITS",
            BandDataType::UInt64 => "UNSIGNED_64BITS",
            BandDataType::Int64 => "SIGNED_64BITS",
            BandDataType::Int8 => "SIGNED_8BITS",
        }
    }
}

/// Hard-coded column indices for performant access to nested struct fields.
/// These indices must match the exact order defined in the RasterSchema methods.
///
/// Using compile-time constants avoids string lookups and provides type safety
/// when accessing nested struct fields in Arrow arrays.
pub mod raster_indices {
    pub const CRS: usize = 0;
    pub const TRANSFORM: usize = 1;
    pub const SPATIAL_DIMS: usize = 2;
    pub const SPATIAL_SHAPE: usize = 3;
    pub const BANDS: usize = 4;
    pub const FIELD_COUNT: usize = 5;
}

pub mod band_indices {
    pub const NAME: usize = 0;
    pub const DIM_NAMES: usize = 1;
    pub const SOURCE_SHAPE: usize = 2;
    pub const DATA_TYPE: usize = 3;
    pub const NODATA: usize = 4;
    pub const VIEW: usize = 5;
    pub const DATA: usize = 6;
    pub const CHUNK_SHAPE: usize = 7;
    pub const CODECS: usize = 8;
    pub const CHUNKS: usize = 9;
    pub const FIELD_COUNT: usize = 10;
}

/// Field indices within the `chunk` struct (the Iceberg V4 `file` shape).
pub mod chunk_indices {
    pub const URI: usize = 0;
    pub const OFFSET: usize = 1;
    pub const SIZE: usize = 2;
    pub const CONTENT_TYPE: usize = 3;
    pub const CHECKSUM: usize = 4;
    pub const INLINE: usize = 5;
    pub const FIELD_COUNT: usize = 6;
}

/// Field indices within the `view` struct (`(source_axis, start, step, steps)`).
pub mod band_view_indices {
    pub const SOURCE_AXIS: usize = 0;
    pub const START: usize = 1;
    pub const STEP: usize = 2;
    pub const STEPS: usize = 3;
    pub const FIELD_COUNT: usize = 4;
}

/// Column name constants used throughout the raster schema definition.
/// These string constants ensure consistency across schema creation and field access.
pub mod column {
    // Top-level raster fields
    pub const CRS: &str = "crs";
    pub const TRANSFORM: &str = "transform";
    pub const SPATIAL_DIMS: &str = "spatial_dims";
    pub const SPATIAL_SHAPE: &str = "spatial_shape";
    pub const BANDS: &str = "bands";
    pub const BAND: &str = "band";

    // Band fields
    pub const NAME: &str = "name";
    pub const DIM_NAMES: &str = "dim_names";
    pub const SOURCE_SHAPE: &str = "source_shape";
    pub const DATATYPE: &str = "data_type";
    pub const NODATA: &str = "nodata";
    pub const VIEW: &str = "view";
    pub const DATA: &str = "data";
    pub const CHUNK_SHAPE: &str = "chunk_shape";
    pub const CODECS: &str = "codecs";
    pub const CHUNKS: &str = "chunks";
    pub const CHUNK: &str = "chunk";

    // Chunk fields (the Iceberg V4 `file` type's names)
    pub const URI: &str = "uri";
    pub const OFFSET: &str = "offset";
    pub const SIZE: &str = "size";
    pub const CONTENT_TYPE: &str = "content_type";
    pub const CHECKSUM: &str = "checksum";
    pub const INLINE: &str = "inline";
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Tests that the top-level raster schema has the expected number and names of fields.
    #[test]
    fn test_raster_schema_fields() {
        let fields = RasterSchema::fields();
        assert_eq!(fields.len(), 5);
        assert_eq!(fields[0].name(), column::CRS);
        assert_eq!(fields[1].name(), column::TRANSFORM);
        assert_eq!(fields[2].name(), column::SPATIAL_DIMS);
        assert_eq!(fields[3].name(), column::SPATIAL_SHAPE);
        assert_eq!(fields[4].name(), column::BANDS);
    }

    /// Comprehensive test to verify all hard-coded indices match the actual schema.
    /// This ensures that performance optimizations using direct index access remain valid
    /// when the schema structure changes.
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
            raster_fields[raster_indices::SPATIAL_DIMS].name(),
            column::SPATIAL_DIMS,
            "Raster SPATIAL_DIMS index mismatch"
        );
        assert_eq!(
            raster_fields[raster_indices::SPATIAL_SHAPE].name(),
            column::SPATIAL_SHAPE,
            "Raster SPATIAL_SHAPE index mismatch"
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
            assert_eq!(
                band_fields[band_indices::SOURCE_SHAPE].name(),
                column::SOURCE_SHAPE
            );
            assert_eq!(
                band_fields[band_indices::DATA_TYPE].name(),
                column::DATATYPE
            );
            assert_eq!(band_fields[band_indices::NODATA].name(), column::NODATA);
            assert_eq!(band_fields[band_indices::VIEW].name(), column::VIEW);
            assert!(
                band_fields[band_indices::VIEW].is_nullable(),
                "view field must be nullable — null encodes the identity view"
            );
            assert_eq!(band_fields[band_indices::DATA].name(), column::DATA);
            assert_eq!(
                band_fields[band_indices::CHUNK_SHAPE].name(),
                column::CHUNK_SHAPE
            );
            assert_eq!(band_fields[band_indices::CODECS].name(), column::CODECS);
            assert_eq!(band_fields[band_indices::CHUNKS].name(), column::CHUNKS);
            for optional in [
                band_indices::CHUNK_SHAPE,
                band_indices::CODECS,
                band_indices::CHUNKS,
            ] {
                assert!(
                    band_fields[optional].is_nullable(),
                    "{} must be nullable: an in-db band has none",
                    band_fields[optional].name()
                );
            }
        } else {
            panic!("Expected Struct type for band");
        }

        // Chunk indices
        let DataType::List(chunk_field) = RasterSchema::chunks_type() else {
            panic!("Expected List type for chunks");
        };
        assert_eq!(chunk_field.name(), column::CHUNK);
        assert!(
            chunk_field.is_nullable(),
            "chunk entries must be nullable: a null entry is a missing chunk"
        );
        let DataType::Struct(chunk_fields) = chunk_field.data_type() else {
            panic!("Expected Struct type inside chunks list");
        };
        assert_eq!(chunk_fields.len(), chunk_indices::FIELD_COUNT);
        assert_eq!(chunk_fields[chunk_indices::URI].name(), column::URI);
        assert_eq!(chunk_fields[chunk_indices::OFFSET].name(), column::OFFSET);
        assert_eq!(chunk_fields[chunk_indices::SIZE].name(), column::SIZE);
        assert_eq!(
            chunk_fields[chunk_indices::CONTENT_TYPE].name(),
            column::CONTENT_TYPE
        );
        assert_eq!(
            chunk_fields[chunk_indices::CHECKSUM].name(),
            column::CHECKSUM
        );
        assert_eq!(chunk_fields[chunk_indices::INLINE].name(), column::INLINE);
        assert_eq!(
            chunk_fields[chunk_indices::URI].data_type(),
            &DataType::Utf8View
        );
        assert_eq!(
            chunk_fields[chunk_indices::OFFSET].data_type(),
            &DataType::Int64
        );
        assert_eq!(
            chunk_fields[chunk_indices::SIZE].data_type(),
            &DataType::Int64
        );
        assert_eq!(
            chunk_fields[chunk_indices::INLINE].data_type(),
            &DataType::BinaryView
        );
        for f in chunk_fields.iter() {
            assert!(f.is_nullable(), "every chunk field is optional");
        }
    }

    #[test]
    fn test_view_type_struct_shape() {
        // The view struct must have exactly 4 Int64 fields in the order
        // expected by band_view_indices.
        let DataType::List(item_field) = RasterSchema::view_type() else {
            panic!("Expected List type for view");
        };
        let DataType::Struct(view_fields) = item_field.data_type() else {
            panic!("Expected Struct type inside view list");
        };
        assert_eq!(view_fields.len(), 4);
        assert_eq!(
            view_fields[band_view_indices::SOURCE_AXIS].name(),
            "source_axis"
        );
        assert_eq!(view_fields[band_view_indices::START].name(), "start");
        assert_eq!(view_fields[band_view_indices::STEP].name(), "step");
        assert_eq!(view_fields[band_view_indices::STEPS].name(), "steps");
        for f in view_fields.iter() {
            assert_eq!(f.data_type(), &DataType::Int64);
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
