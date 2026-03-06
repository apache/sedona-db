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

use arrow_schema::ArrowError;

use sedona_schema::raster::{BandDataType, StorageType};

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

/// Trait for accessing complete raster data
pub trait RasterRef {
    /// Raster metadata accessor
    fn metadata(&self) -> &dyn MetadataRef;
    /// CRS accessor
    fn crs(&self) -> Option<&str>;
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
    fn iter(&self) -> Box<dyn BandIterator<'_> + '_>;
}

/// Trait for accessing individual band data
pub trait BandRef {
    /// Band metadata accessor
    fn metadata(&self) -> &dyn BandMetadataRef;
    /// Raw band data as bytes (zero-copy access)
    fn data(&self) -> &[u8];
}

/// Trait for accessing individual band metadata
pub trait BandMetadataRef {
    /// No-data value as raw bytes (None if null)
    fn nodata_value(&self) -> Option<&[u8]>;
    /// Storage type (InDb, OutDbRef, etc)
    fn storage_type(&self) -> Result<StorageType, ArrowError>;
    /// Band data type (UInt8, Float32, etc.)
    fn data_type(&self) -> Result<BandDataType, ArrowError>;
    /// OutDb URL (only used when storage_type == OutDbRef)
    fn outdb_url(&self) -> Option<&str>;
    /// OutDb band ID (only used when storage_type == OutDbRef)
    fn outdb_band_id(&self) -> Option<u32>;

    /// No-data value interpreted as f64.
    ///
    /// Returns `Ok(None)` when no nodata value is defined, `Ok(Some(f64))` on
    /// success, or an error when the raw bytes have an unexpected length for
    /// the band's data type.
    fn nodata_value_as_f64(&self) -> Result<Option<f64>, ArrowError> {
        let bytes = match self.nodata_value() {
            Some(b) => b,
            None => return Ok(None),
        };
        let dt = self.data_type()?;
        nodata_bytes_to_f64(bytes, &dt).map(Some)
    }
}

/// Convert raw nodata bytes to f64 given a [`BandDataType`].
///
/// The bytes are expected to be in little-endian order and exactly match the
/// byte size of the data type.
fn nodata_bytes_to_f64(bytes: &[u8], dt: &BandDataType) -> Result<f64, ArrowError> {
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

/// Trait for iterating over bands within a raster
pub trait BandIterator<'a>: Iterator<Item = Box<dyn BandRef + 'a>> {
    fn len(&self) -> usize;
    /// Check if there are no more bands
    fn is_empty(&self) -> bool {
        self.len() == 0
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
