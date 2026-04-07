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

use std::borrow::Cow;

use arrow_schema::ArrowError;
use sedona_schema::raster::BandDataType;

/// Zero-copy view into a band's N-D data buffer with layout metadata.
///
/// In Phase 1, strides are always standard C-order contiguous and offset is 0.
/// Phase 2 will introduce non-standard strides for zero-copy slicing.
#[derive(Debug)]
pub struct NdBuffer<'a> {
    pub buffer: &'a [u8],
    pub shape: &'a [u64],
    pub strides: &'a [i64],
    pub offset: u64,
    pub data_type: BandDataType,
}

/// Trait for accessing an N-dimensional raster (top level).
///
/// Replaces the legacy `RasterRef` + `MetadataRef` + `BandsRef` hierarchy with
/// a single flat interface. Bands are 0-indexed.
pub trait RasterRef {
    /// Number of bands/variables
    fn num_bands(&self) -> usize;

    /// Access a band by 0-based index
    fn band(&self, index: usize) -> Option<Box<dyn BandRef + '_>>;

    /// Band name (e.g., Zarr variable name). None for unnamed bands.
    fn band_name(&self, index: usize) -> Option<&str>;

    /// CRS string (PROJJSON, WKT, or authority code). None if not set.
    fn crs(&self) -> Option<&str>;

    /// 6-element affine transform in GDAL GeoTransform order:
    /// `[origin_x, scale_x, skew_x, origin_y, skew_y, scale_y]`
    fn transform(&self) -> &[f64];

    /// Name of the X spatial dimension (e.g., "x", "lon", "easting")
    fn x_dim(&self) -> &str;

    /// Name of the Y spatial dimension (e.g., "y", "lat", "northing")
    fn y_dim(&self) -> &str;

    /// Width in pixels — size of the X spatial dimension in band(0).
    fn width(&self) -> Option<u64> {
        self.band(0).and_then(|b| b.dim_size(self.x_dim()))
    }

    /// Height in pixels — size of the Y spatial dimension in band(0).
    fn height(&self) -> Option<u64> {
        self.band(0).and_then(|b| b.dim_size(self.y_dim()))
    }

    /// Look up a band by name. Returns None if no band has that name.
    fn band_by_name(&self, name: &str) -> Option<Box<dyn BandRef + '_>> {
        (0..self.num_bands())
            .find(|&i| self.band_name(i) == Some(name))
            .and_then(|i| self.band(i))
    }
}

/// Trait for accessing a single band/variable within an N-D raster.
///
/// This is the consumer interface. Implementations handle storage details
/// Two data access paths:
/// - `contiguous_data()` — flat row-major bytes for consumers that don't need
///   stride awareness (most RS_* functions, GDAL boundary, serialization).
/// - `nd_buffer()` — raw buffer + shape + strides + offset for stride-aware
///   consumers (numpy zero-copy views, Arrow FFI) that want to avoid copies.
pub trait BandRef {
    // -- Dimension metadata --

    /// Number of dimensions in this band
    fn ndim(&self) -> usize;

    /// Dimension names in order (e.g., `["time", "y", "x"]`)
    fn dim_names(&self) -> Vec<&str>;

    /// Shape (size of each dimension)
    fn shape(&self) -> &[u64];

    /// Size of a named dimension (None if doesn't exist)
    fn dim_size(&self, name: &str) -> Option<u64> {
        let idx = self.dim_index(name)?;
        Some(self.shape()[idx])
    }

    /// Index of a named dimension (None if doesn't exist)
    fn dim_index(&self, name: &str) -> Option<usize> {
        self.dim_names().iter().position(|n| *n == name)
    }

    // -- Band metadata --

    /// Data type for all elements in this band
    fn data_type(&self) -> BandDataType;

    /// Nodata value as raw bytes (None if not set)
    fn nodata(&self) -> Option<&[u8]>;

    /// OutDb URI (None for in-memory bands)
    fn outdb_uri(&self) -> Option<&str> {
        None
    }

    // -- Data access --

    /// Raw backing buffer + layout. Triggers load for lazy impls.
    /// Returns an NdBuffer with shape, strides, offset, and raw byte buffer.
    fn nd_buffer(&self) -> Result<NdBuffer<'_>, ArrowError>;

    /// Contiguous row-major bytes. Zero-copy (`Cow::Borrowed`) when data
    /// has standard C-order strides; copies into a new buffer only when
    /// strides are non-standard. Most RS_* functions use this.
    fn contiguous_data(&self) -> Result<Cow<'_, [u8]>, ArrowError>;

    /// Nodata value interpreted as f64.
    ///
    /// Returns `Ok(None)` when no nodata value is defined, `Ok(Some(f64))` on
    /// success, or an error when the raw bytes have an unexpected length.
    fn nodata_as_f64(&self) -> Result<Option<f64>, ArrowError> {
        let bytes = match self.nodata() {
            Some(b) => b,
            None => return Ok(None),
        };
        nodata_bytes_to_f64(bytes, &self.data_type()).map(Some)
    }
}

/// Convert raw nodata bytes to f64 given a [`BandDataType`].
///
/// The bytes are expected to be in little-endian order and exactly match the
/// byte size of the data type.
pub fn nodata_bytes_to_f64(bytes: &[u8], dt: &BandDataType) -> Result<f64, ArrowError> {
    macro_rules! read_le {
        ($t:ty, $n:expr) => {{
            let arr: [u8; $n] = bytes.try_into().map_err(|_| {
                ArrowError::InvalidArgumentError(format!(
                    "Invalid nodata byte length for {:?}: expected {}, got {}",
                    dt,
                    $n,
                    bytes.len()
                ))
            })?;
            Ok(<$t>::from_le_bytes(arr) as f64)
        }};
    }

    match dt {
        BandDataType::UInt8 => {
            if bytes.len() != 1 {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "Invalid nodata byte length for UInt8: expected 1, got {}",
                    bytes.len()
                )));
            }
            Ok(bytes[0] as f64)
        }
        BandDataType::Int8 => {
            if bytes.len() != 1 {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "Invalid nodata byte length for Int8: expected 1, got {}",
                    bytes.len()
                )));
            }
            Ok(bytes[0] as i8 as f64)
        }
        BandDataType::UInt16 => read_le!(u16, 2),
        BandDataType::Int16 => read_le!(i16, 2),
        BandDataType::UInt32 => read_le!(u32, 4),
        BandDataType::Int32 => read_le!(i32, 4),
        BandDataType::UInt64 => read_le!(u64, 8),
        BandDataType::Int64 => read_le!(i64, 8),
        BandDataType::Float32 => read_le!(f32, 4),
        BandDataType::Float64 => read_le!(f64, 8),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nodata_bytes_to_f64_uint8() {
        let val = nodata_bytes_to_f64(&[42], &BandDataType::UInt8).unwrap();
        assert_eq!(val, 42.0);
    }

    #[test]
    fn test_nodata_bytes_to_f64_int8() {
        let val = nodata_bytes_to_f64(&[0xFE], &BandDataType::Int8).unwrap();
        assert_eq!(val, -2.0);
    }

    #[test]
    fn test_nodata_bytes_to_f64_float64() {
        let bytes = (-9999.0_f64).to_le_bytes();
        let val = nodata_bytes_to_f64(&bytes, &BandDataType::Float64).unwrap();
        assert_eq!(val, -9999.0);
    }

    #[test]
    fn test_nodata_bytes_to_f64_int32() {
        let bytes = (-1_i32).to_le_bytes();
        let val = nodata_bytes_to_f64(&bytes, &BandDataType::Int32).unwrap();
        assert_eq!(val, -1.0);
    }

    #[test]
    fn test_nodata_bytes_to_f64_wrong_length() {
        let result = nodata_bytes_to_f64(&[1, 2, 3], &BandDataType::Float64);
        assert!(result.is_err());
    }
}
