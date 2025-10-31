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
    fn storage_type(&self) -> StorageType;
    /// Band data type (UInt8, Float32, etc.)
    fn data_type(&self) -> BandDataType;
    /// OutDb URL (only used when storage_type == OutDbRef)
    fn outdb_url(&self) -> Option<&str>;
    /// OutDb band ID (only used when storage_type == OutDbRef)
    fn outdb_band_id(&self) -> Option<u32>;
}

/// Trait for iterating over bands within a raster
pub trait BandIterator<'a>: Iterator<Item = Box<dyn BandRef + 'a>> {
    fn len(&self) -> usize;
    /// Check if there are no more bands
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
