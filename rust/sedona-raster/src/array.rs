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
    Array, BinaryArray, BinaryViewArray, Float64Array, ListArray, StringArray, StringViewArray,
    StructArray, UInt32Array, UInt64Array,
};
use arrow_schema::ArrowError;

use crate::traits::{
    BandIterator, BandMetadataRef, BandRef, BandsRef, BoundingBox, BoundingBoxRef, MetadataRef,
    RasterMetadata, RasterRef,
};
use sedona_schema::raster::{
    band_indices, band_metadata_indices, bounding_box_indices, metadata_indices, raster_indices,
    BandDataType, StorageType,
};

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

    fn iter(&self) -> Box<dyn BandIterator<'_> + '_> {
        Box::new(BandIteratorImpl {
            bands: self,
            current: 1, // Start at 1 for 1-based band numbering
        })
    }
}

/// Concrete implementation of BandIterator trait
pub struct BandIteratorImpl<'a> {
    bands: &'a dyn BandsRef,
    current: usize,
}

impl<'a> Iterator for BandIteratorImpl<'a> {
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

impl<'a> BandIterator<'a> for BandIteratorImpl<'a> {
    fn len(&self) -> usize {
        // current is 1-based, so remaining calculation needs adjustment
        self.bands.len().saturating_sub(self.current - 1)
    }
}

impl ExactSizeIterator for BandIteratorImpl<'_> {}

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

/// Access rasters from the Arrow StructArray
///
/// This provides efficient, zero-copy access to raster data stored in Arrow format.
pub struct RasterStructArray<'a> {
    raster_array: &'a StructArray,
}

impl<'a> RasterStructArray<'a> {
    /// Create a new RasterStructArray from an existing StructArray
    pub fn new(raster_array: &'a StructArray) -> Self {
        Self { raster_array }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::RasterBuilder;
    use crate::traits::{BandMetadata, BoundingBox, RasterMetadata};
    use sedona_schema::raster::{BandDataType, StorageType};

    #[test]
    fn test_array_basic_functionality() {
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

        // Test the array
        let rasters = RasterStructArray::new(&raster_array);

        assert_eq!(rasters.len(), 1);
        assert!(!rasters.is_empty());

        let raster = rasters.get(0).unwrap();
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

        // Test array over bands
        let band_iter: Vec<_> = bands.iter().collect();
        assert_eq!(band_iter.len(), 1);
    }

    #[test]
    fn test_multi_band_array() {
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

        // Test array
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
}
