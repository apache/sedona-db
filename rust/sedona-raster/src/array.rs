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

use arrow_array::{
    Array, BinaryArray, BinaryViewArray, Float64Array, Int64Array, ListArray, StringArray,
    StringViewArray, StructArray, UInt32Array, UInt64Array,
};
use arrow_schema::ArrowError;

use crate::traits::{BandRef, NdBuffer, RasterRef};
use sedona_schema::raster::{band_indices, raster_indices, BandDataType};

// ---------------------------------------------------------------------------
// Band implementation (Arrow-backed)
// ---------------------------------------------------------------------------

/// Arrow-backed implementation of BandRef for a single band within a raster.
struct BandRefImpl<'a> {
    // Band metadata arrays (indexed by absolute band row)
    dim_names_list: &'a ListArray,
    dim_names_values: &'a StringArray,
    shape_list: &'a ListArray,
    shape_values: &'a UInt64Array,
    datatype_array: &'a UInt32Array,
    nodata_array: &'a BinaryArray,
    strides_list: &'a ListArray,
    strides_values: &'a Int64Array,
    offset_array: &'a UInt64Array,
    outdb_uri_array: &'a StringArray,
    outdb_format_array: &'a StringViewArray,
    data_array: &'a BinaryViewArray,
    /// Absolute row index within the flattened bands arrays
    band_row: usize,
}

impl<'a> BandRef for BandRefImpl<'a> {
    fn ndim(&self) -> usize {
        self.shape_list.value_length(self.band_row) as usize
    }

    fn dim_names(&self) -> Vec<&str> {
        let start = self.dim_names_list.value_offsets()[self.band_row] as usize;
        let end = self.dim_names_list.value_offsets()[self.band_row + 1] as usize;
        (start..end)
            .map(|i| self.dim_names_values.value(i))
            .collect()
    }

    fn shape(&self) -> &[u64] {
        let start = self.shape_list.value_offsets()[self.band_row] as usize;
        let end = self.shape_list.value_offsets()[self.band_row + 1] as usize;
        &self.shape_values.values()[start..end]
    }

    fn data_type(&self) -> BandDataType {
        let value = self.datatype_array.value(self.band_row);
        BandDataType::try_from_u32(value)
            .unwrap_or_else(|| panic!("Unknown band data type: {value}"))
    }

    fn nodata(&self) -> Option<&[u8]> {
        if self.nodata_array.is_null(self.band_row) {
            None
        } else {
            Some(self.nodata_array.value(self.band_row))
        }
    }

    fn outdb_uri(&self) -> Option<&str> {
        if self.outdb_uri_array.is_null(self.band_row) {
            None
        } else {
            Some(self.outdb_uri_array.value(self.band_row))
        }
    }

    fn outdb_format(&self) -> Option<&str> {
        if self.outdb_format_array.is_null(self.band_row) {
            None
        } else {
            Some(self.outdb_format_array.value(self.band_row))
        }
    }

    fn nd_buffer(&self) -> Result<NdBuffer<'_>, ArrowError> {
        let strides_start = self.strides_list.value_offsets()[self.band_row] as usize;
        let strides_end = self.strides_list.value_offsets()[self.band_row + 1] as usize;

        Ok(NdBuffer {
            buffer: self.data_array.value(self.band_row),
            shape: self.shape(),
            strides: &self.strides_values.values()[strides_start..strides_end],
            offset: self.offset_array.value(self.band_row),
            data_type: self.data_type(),
        })
    }

    fn contiguous_data(&self) -> Result<Cow<'_, [u8]>, ArrowError> {
        // Phase 1: all data is contiguous, so always return Borrowed
        Ok(Cow::Borrowed(self.data_array.value(self.band_row)))
    }
}

// ---------------------------------------------------------------------------
// Raster implementation (Arrow-backed)
// ---------------------------------------------------------------------------

/// Arrow-backed implementation of RasterRef for a single raster row.
pub struct RasterRefImpl<'a> {
    raster_struct_array: &'a RasterStructArray<'a>,
    raster_index: usize,
}

impl<'a> RasterRefImpl<'a> {
    /// Returns the raw CRS string reference with the array's lifetime.
    pub fn crs_str_ref(&self) -> Option<&'a str> {
        if self
            .raster_struct_array
            .crs_array
            .is_null(self.raster_index)
        {
            None
        } else {
            Some(self.raster_struct_array.crs_array.value(self.raster_index))
        }
    }
}

impl<'a> RasterRef for RasterRefImpl<'a> {
    fn num_bands(&self) -> usize {
        self.raster_struct_array
            .bands_list
            .value_length(self.raster_index) as usize
    }

    fn band(&self, index: usize) -> Option<Box<dyn BandRef + '_>> {
        if index >= self.num_bands() {
            return None;
        }
        let start = self.raster_struct_array.bands_list.value_offsets()[self.raster_index] as usize;
        let band_row = start + index;
        Some(Box::new(BandRefImpl {
            dim_names_list: self.raster_struct_array.band_dim_names_list,
            dim_names_values: self.raster_struct_array.band_dim_names_values,
            shape_list: self.raster_struct_array.band_shape_list,
            shape_values: self.raster_struct_array.band_shape_values,
            datatype_array: self.raster_struct_array.band_datatype_array,
            nodata_array: self.raster_struct_array.band_nodata_array,
            strides_list: self.raster_struct_array.band_strides_list,
            strides_values: self.raster_struct_array.band_strides_values,
            offset_array: self.raster_struct_array.band_offset_array,
            outdb_uri_array: self.raster_struct_array.band_outdb_uri_array,
            outdb_format_array: self.raster_struct_array.band_outdb_format_array,
            data_array: self.raster_struct_array.band_data_array,
            band_row,
        }))
    }

    fn band_name(&self, index: usize) -> Option<&str> {
        if index >= self.num_bands() {
            return None;
        }
        let start = self.raster_struct_array.bands_list.value_offsets()[self.raster_index] as usize;
        let band_row = start + index;
        if self.raster_struct_array.band_name_array.is_null(band_row) {
            None
        } else {
            Some(self.raster_struct_array.band_name_array.value(band_row))
        }
    }

    fn crs(&self) -> Option<&str> {
        self.crs_str_ref()
    }

    fn transform(&self) -> &[f64] {
        let start =
            self.raster_struct_array.transform_list.value_offsets()[self.raster_index] as usize;
        let end =
            self.raster_struct_array.transform_list.value_offsets()[self.raster_index + 1] as usize;
        debug_assert!(
            end - start >= 6,
            "transform list must have at least 6 elements for raster {}, got {}",
            self.raster_index,
            end - start
        );
        &self.raster_struct_array.transform_values.values()[start..start + 6]
    }

    fn spatial_dims(&self) -> Vec<&str> {
        let offsets = self.raster_struct_array.spatial_dims_list.value_offsets();
        let start = offsets[self.raster_index] as usize;
        let end = offsets[self.raster_index + 1] as usize;
        (start..end)
            .map(|i| self.raster_struct_array.spatial_dims_values.value(i))
            .collect()
    }

    fn spatial_shape(&self) -> &[i64] {
        let offsets = self.raster_struct_array.spatial_shape_list.value_offsets();
        let start = offsets[self.raster_index] as usize;
        let end = offsets[self.raster_index + 1] as usize;
        &self.raster_struct_array.spatial_shape_values.values()[start..end]
    }
}

// ---------------------------------------------------------------------------
// RasterStructArray — efficient columnar access to rasters
// ---------------------------------------------------------------------------

/// Access rasters from the Arrow StructArray.
///
/// Provides efficient, zero-copy access to N-D raster data stored in Arrow format.
pub struct RasterStructArray<'a> {
    raster_array: &'a StructArray,
    // Top-level fields
    crs_array: &'a StringViewArray,
    transform_list: &'a ListArray,
    transform_values: &'a Float64Array,
    spatial_dims_list: &'a ListArray,
    spatial_dims_values: &'a StringViewArray,
    spatial_shape_list: &'a ListArray,
    spatial_shape_values: &'a Int64Array,
    bands_list: &'a ListArray,
    // Band-level fields (flattened across all bands in all rasters)
    band_name_array: &'a StringArray,
    band_dim_names_list: &'a ListArray,
    band_dim_names_values: &'a StringArray,
    band_shape_list: &'a ListArray,
    band_shape_values: &'a UInt64Array,
    band_datatype_array: &'a UInt32Array,
    band_nodata_array: &'a BinaryArray,
    band_strides_list: &'a ListArray,
    band_strides_values: &'a Int64Array,
    band_offset_array: &'a UInt64Array,
    band_outdb_uri_array: &'a StringArray,
    band_outdb_format_array: &'a StringViewArray,
    band_data_array: &'a BinaryViewArray,
}

impl<'a> RasterStructArray<'a> {
    /// Create a new RasterStructArray from an existing StructArray.
    #[inline]
    pub fn new(raster_array: &'a StructArray) -> Self {
        // Top-level fields
        let crs_array = raster_array
            .column(raster_indices::CRS)
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap();
        let transform_list = raster_array
            .column(raster_indices::TRANSFORM)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let transform_values = transform_list
            .values()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let spatial_dims_list = raster_array
            .column(raster_indices::SPATIAL_DIMS)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let spatial_dims_values = spatial_dims_list
            .values()
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap();
        let spatial_shape_list = raster_array
            .column(raster_indices::SPATIAL_SHAPE)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let spatial_shape_values = spatial_shape_list
            .values()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        // Bands list and nested struct
        let bands_list = raster_array
            .column(raster_indices::BANDS)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let bands_struct = bands_list
            .values()
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();

        // Band-level fields
        let band_name_array = bands_struct
            .column(band_indices::NAME)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let band_dim_names_list = bands_struct
            .column(band_indices::DIM_NAMES)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let band_dim_names_values = band_dim_names_list
            .values()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let band_shape_list = bands_struct
            .column(band_indices::SHAPE)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let band_shape_values = band_shape_list
            .values()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let band_datatype_array = bands_struct
            .column(band_indices::DATA_TYPE)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let band_nodata_array = bands_struct
            .column(band_indices::NODATA)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap();
        let band_strides_list = bands_struct
            .column(band_indices::STRIDES)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let band_strides_values = band_strides_list
            .values()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let band_offset_array = bands_struct
            .column(band_indices::OFFSET)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let band_outdb_uri_array = bands_struct
            .column(band_indices::OUTDB_URI)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let band_outdb_format_array = bands_struct
            .column(band_indices::OUTDB_FORMAT)
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap();
        let band_data_array = bands_struct
            .column(band_indices::DATA)
            .as_any()
            .downcast_ref::<BinaryViewArray>()
            .unwrap();

        Self {
            raster_array,
            crs_array,
            transform_list,
            transform_values,
            spatial_dims_list,
            spatial_dims_values,
            spatial_shape_list,
            spatial_shape_values,
            bands_list,
            band_name_array,
            band_dim_names_list,
            band_dim_names_values,
            band_shape_list,
            band_shape_values,
            band_datatype_array,
            band_nodata_array,
            band_strides_list,
            band_strides_values,
            band_offset_array,
            band_outdb_uri_array,
            band_outdb_format_array,
            band_data_array,
        }
    }

    /// Get the total number of rasters in the array.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.raster_array.len()
    }

    /// Check if the array is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.raster_array.is_empty()
    }

    /// Get a specific raster by index.
    #[inline(always)]
    pub fn get(&'a self, index: usize) -> Result<RasterRefImpl<'a>, ArrowError> {
        if index >= self.raster_array.len() {
            return Err(ArrowError::InvalidArgumentError(format!(
                "Invalid raster index: {index}"
            )));
        }
        Ok(RasterRefImpl {
            raster_struct_array: self,
            raster_index: index,
        })
    }

    /// Check if a raster at the given index is null.
    #[inline(always)]
    pub fn is_null(&self, index: usize) -> bool {
        self.raster_array.is_null(index)
    }
}
