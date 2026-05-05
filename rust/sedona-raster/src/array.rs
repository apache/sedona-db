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

use crate::traits::{validate_view, BandRef, NdBuffer, RasterRef, ViewEntry};
use sedona_schema::raster::{band_indices, band_view_indices, raster_indices, BandDataType};

// ---------------------------------------------------------------------------
// Band implementation (Arrow-backed)
// ---------------------------------------------------------------------------

/// Arrow-backed implementation of BandRef for a single band within a raster.
///
/// View-derived layout (`visible_shape`, `byte_strides`, `byte_offset`,
/// `is_identity_view`) is computed once at construction and reused by every
/// accessor. Source-shape and dim-name slices are borrowed directly from
/// the underlying Arrow buffers.
struct BandRefImpl<'a> {
    dim_names_list: &'a ListArray,
    dim_names_values: &'a StringArray,
    source_shape_list: &'a ListArray,
    source_shape_values: &'a UInt64Array,
    nodata_array: &'a BinaryArray,
    outdb_uri_array: &'a StringArray,
    outdb_format_array: &'a StringViewArray,
    data_array: &'a BinaryViewArray,
    /// Absolute row index within the flattened bands arrays
    band_row: usize,
    /// Resolved at construction so accessors don't re-decode the discriminant.
    data_type: BandDataType,
    /// Per-visible-axis view, length = ndim
    view_entries: Vec<ViewEntry>,
    /// Visible shape (== `[v.steps for v in view_entries]`), length = ndim
    visible_shape: Vec<u64>,
    /// Byte strides per visible axis. May be 0 (broadcast) or negative.
    byte_strides: Vec<i64>,
    /// Byte offset into `data` of the visible region's `[0,...,0]` element.
    byte_offset: u64,
    /// True iff this view is the identity over a C-order source buffer —
    /// `contiguous_data()` can then borrow `data` directly.
    is_identity_view: bool,
}

impl<'a> BandRef for BandRefImpl<'a> {
    fn ndim(&self) -> usize {
        self.view_entries.len()
    }

    fn dim_names(&self) -> Vec<&str> {
        let start = self.dim_names_list.value_offsets()[self.band_row] as usize;
        let end = self.dim_names_list.value_offsets()[self.band_row + 1] as usize;
        (start..end)
            .map(|i| self.dim_names_values.value(i))
            .collect()
    }

    fn shape(&self) -> &[u64] {
        &self.visible_shape
    }

    fn raw_source_shape(&self) -> &[u64] {
        let start = self.source_shape_list.value_offsets()[self.band_row] as usize;
        let end = self.source_shape_list.value_offsets()[self.band_row + 1] as usize;
        &self.source_shape_values.values()[start..end]
    }

    fn view(&self) -> &[ViewEntry] {
        &self.view_entries
    }

    fn data_type(&self) -> BandDataType {
        self.data_type
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
        Ok(NdBuffer {
            buffer: self.data_array.value(self.band_row),
            shape: &self.visible_shape,
            strides: &self.byte_strides,
            offset: self.byte_offset,
            data_type: self.data_type,
        })
    }

    fn contiguous_data(&self) -> Result<Cow<'_, [u8]>, ArrowError> {
        let buffer = self.data_array.value(self.band_row);
        if self.is_identity_view {
            return Ok(Cow::Borrowed(buffer));
        }
        let dtype_size = self.data_type.byte_size();
        let ndim = self.visible_shape.len();
        let total: u64 = self.visible_shape.iter().product();
        if total == 0 {
            return Ok(Cow::Owned(Vec::new()));
        }
        let total = total as usize;
        let mut out = Vec::with_capacity(total * dtype_size);
        let base = self.byte_offset as i64;

        // Innermost-axis fast path. We always step the innermost (last)
        // visible axis as the inner loop; everything outer drives the
        // starting byte for that row.
        let inner = ndim - 1;
        let inner_steps = self.visible_shape[inner] as usize;
        let inner_stride = self.byte_strides[inner];
        let row_bytes_contiguous =
            inner_stride == dtype_size as i64 && inner_steps > 0 && dtype_size > 0;

        // Precompute a small index vector for outer axes (everything except
        // the innermost). For 1D this is empty and we run a single pass.
        let mut outer_idx = vec![0u64; ndim.saturating_sub(1)];
        loop {
            // Compute the byte offset of the row's first element from the
            // current outer index combination.
            let mut row_off = base;
            for (k, &i) in outer_idx.iter().enumerate() {
                row_off += (i as i64) * self.byte_strides[k];
            }

            if row_bytes_contiguous {
                // Whole row is one flat slice — copy it in one go.
                let len = inner_steps * dtype_size;
                if row_off < 0 || (row_off as usize) + len > buffer.len() {
                    return Err(ArrowError::InvalidArgumentError(format!(
                        "view addresses out-of-bounds byte range {row_off}..{} (buffer len {})",
                        row_off as usize + len,
                        buffer.len()
                    )));
                }
                out.extend_from_slice(&buffer[row_off as usize..row_off as usize + len]);
            } else {
                // Step element-by-element along the innermost axis.
                let mut byte_off = row_off;
                for _ in 0..inner_steps {
                    if byte_off < 0 || (byte_off as usize) + dtype_size > buffer.len() {
                        return Err(ArrowError::InvalidArgumentError(format!(
                            "view addresses out-of-bounds byte offset {byte_off} (buffer len {})",
                            buffer.len()
                        )));
                    }
                    out.extend_from_slice(
                        &buffer[byte_off as usize..byte_off as usize + dtype_size],
                    );
                    byte_off += inner_stride;
                }
            }

            // Increment outer indices in C-order (last outer axis fastest).
            if outer_idx.is_empty() {
                break;
            }
            let mut k = inner; // outer axes are [0..inner)
            let mut done = true;
            while k > 0 {
                k -= 1;
                outer_idx[k] += 1;
                if outer_idx[k] < self.visible_shape[k] {
                    done = false;
                    break;
                }
                outer_idx[k] = 0;
            }
            if done {
                break;
            }
        }
        Ok(Cow::Owned(out))
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

        let arr = self.raster_struct_array;

        // Read source shape slice.
        let ss_start = arr.band_source_shape_list.value_offsets()[band_row] as usize;
        let ss_end = arr.band_source_shape_list.value_offsets()[band_row + 1] as usize;
        let source_shape: &[u64] = &arr.band_source_shape_values.values()[ss_start..ss_end];

        // Phase 1 reject 0-D bands at the read boundary. Schema doesn't
        // forbid them outright but every consumer assumes ndim >= 1.
        if source_shape.is_empty() {
            return None;
        }

        // Resolve data type up front; an unknown discriminant is a
        // schema-corruption bug, not user data, so failing the band is
        // appropriate.
        let data_type_value = arr.band_datatype_array.value(band_row);
        let data_type = BandDataType::try_from_u32(data_type_value)?;

        // Read view entries. A null row encodes the canonical identity view
        // and is synthesised on the fly from `source_shape`.
        let view_entries: Vec<ViewEntry> = if arr.band_view_list.is_null(band_row) {
            source_shape
                .iter()
                .enumerate()
                .map(|(i, &s)| ViewEntry {
                    source_axis: i as i64,
                    start: 0,
                    step: 1,
                    steps: s as i64,
                })
                .collect()
        } else {
            let v_start = arr.band_view_list.value_offsets()[band_row] as usize;
            let v_end = arr.band_view_list.value_offsets()[band_row + 1] as usize;
            (v_start..v_end)
                .map(|i| ViewEntry {
                    source_axis: arr.band_view_source_axis.value(i),
                    start: arr.band_view_start.value(i),
                    step: arr.band_view_step.value(i),
                    steps: arr.band_view_steps.value(i),
                })
                .collect()
        };

        // Full validation: length match, source_axis permutation, bounds,
        // and steps >= 0. Anything malformed is treated as a schema-level
        // corruption — return None so the caller can surface it.
        if validate_view(&view_entries, source_shape).is_err() {
            return None;
        }

        let ndim = view_entries.len();
        // After validate_view: every steps is >= 0, so this cast is safe.
        let visible_shape: Vec<u64> = view_entries.iter().map(|v| v.steps as u64).collect();

        let dtype_size = data_type.byte_size() as i64;

        // C-order byte strides over the source_shape:
        //   source_strides_bytes[k] = dtype_size * Π_{j>k} source_shape[j]
        let mut source_strides_bytes = vec![0i64; source_shape.len()];
        source_strides_bytes[source_shape.len() - 1] = dtype_size;
        for k in (0..source_shape.len() - 1).rev() {
            source_strides_bytes[k] = source_strides_bytes[k + 1] * (source_shape[k + 1] as i64);
        }

        // Compose view → byte strides + offset. validate_view has already
        // guaranteed source_axis is in range and start is non-negative.
        let mut byte_strides = vec![0i64; ndim];
        let mut byte_offset: i64 = 0;
        let mut is_identity_view = true;
        for (k, v) in view_entries.iter().enumerate() {
            let sa_idx = v.source_axis as usize;
            let src_stride = source_strides_bytes[sa_idx];
            byte_strides[k] = v.step * src_stride;
            byte_offset += v.start * src_stride;
            if !(sa_idx == k
                && v.start == 0
                && v.step == 1
                && v.steps as u64 == source_shape[sa_idx])
            {
                is_identity_view = false;
            }
        }
        // byte_offset is non-negative by construction (start >= 0,
        // src_stride > 0), but assert to keep the cast below honest.
        if byte_offset < 0 {
            return None;
        }

        Some(Box::new(BandRefImpl {
            dim_names_list: arr.band_dim_names_list,
            dim_names_values: arr.band_dim_names_values,
            source_shape_list: arr.band_source_shape_list,
            source_shape_values: arr.band_source_shape_values,
            nodata_array: arr.band_nodata_array,
            outdb_uri_array: arr.band_outdb_uri_array,
            outdb_format_array: arr.band_outdb_format_array,
            data_array: arr.band_data_array,
            band_row,
            data_type,
            view_entries,
            visible_shape,
            byte_strides,
            byte_offset: byte_offset as u64,
            is_identity_view,
        }))
    }

    fn band_data_type(&self, index: usize) -> Option<BandDataType> {
        if index >= self.num_bands() {
            return None;
        }
        let start = self.raster_struct_array.bands_list.value_offsets()[self.raster_index] as usize;
        let band_row = start + index;
        let value = self.raster_struct_array.band_datatype_array.value(band_row);
        BandDataType::try_from_u32(value)
    }

    fn band_outdb_uri(&self, index: usize) -> Option<&str> {
        if index >= self.num_bands() {
            return None;
        }
        let start = self.raster_struct_array.bands_list.value_offsets()[self.raster_index] as usize;
        let band_row = start + index;
        let arr = self.raster_struct_array.band_outdb_uri_array;
        if arr.is_null(band_row) {
            None
        } else {
            Some(arr.value(band_row))
        }
    }

    fn band_outdb_format(&self, index: usize) -> Option<&str> {
        if index >= self.num_bands() {
            return None;
        }
        let start = self.raster_struct_array.bands_list.value_offsets()[self.raster_index] as usize;
        let band_row = start + index;
        let arr = self.raster_struct_array.band_outdb_format_array;
        if arr.is_null(band_row) {
            None
        } else {
            Some(arr.value(band_row))
        }
    }

    fn band_nodata(&self, index: usize) -> Option<&[u8]> {
        if index >= self.num_bands() {
            return None;
        }
        let start = self.raster_struct_array.bands_list.value_offsets()[self.raster_index] as usize;
        let band_row = start + index;
        let arr = self.raster_struct_array.band_nodata_array;
        if arr.is_null(band_row) {
            None
        } else {
            Some(arr.value(band_row))
        }
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
        assert!(
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
    band_source_shape_list: &'a ListArray,
    band_source_shape_values: &'a UInt64Array,
    band_datatype_array: &'a UInt32Array,
    band_nodata_array: &'a BinaryArray,
    band_view_list: &'a ListArray,
    band_view_source_axis: &'a Int64Array,
    band_view_start: &'a Int64Array,
    band_view_step: &'a Int64Array,
    band_view_steps: &'a Int64Array,
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
        let band_source_shape_list = bands_struct
            .column(band_indices::SOURCE_SHAPE)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let band_source_shape_values = band_source_shape_list
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
        let band_view_list = bands_struct
            .column(band_indices::VIEW)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let band_view_struct = band_view_list
            .values()
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let band_view_source_axis = band_view_struct
            .column(band_view_indices::SOURCE_AXIS)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let band_view_start = band_view_struct
            .column(band_view_indices::START)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let band_view_step = band_view_struct
            .column(band_view_indices::STEP)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let band_view_steps = band_view_struct
            .column(band_view_indices::STEPS)
            .as_any()
            .downcast_ref::<Int64Array>()
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
            band_source_shape_list,
            band_source_shape_values,
            band_datatype_array,
            band_nodata_array,
            band_view_list,
            band_view_source_axis,
            band_view_start,
            band_view_step,
            band_view_steps,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::RasterBuilder;
    use crate::traits::ViewEntry;
    use arrow_array::{
        types::Int64Type, ArrayRef, ListArray, StructArray, UInt32Array, UInt64Array,
    };
    use arrow_buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
    use arrow_schema::{DataType, Field, Fields};
    use sedona_schema::raster::{band_indices, raster_indices, BandDataType, RasterSchema};
    use std::sync::Arc;

    /// Build a single-raster, single-band raster StructArray with an explicit
    /// view. Used as the input to the surgery helpers below; callers replace
    /// one band-level column to simulate schema corruption.
    fn build_explicit_view_raster() -> StructArray {
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster(&transform, &["x"], &[3], None)
            .unwrap();
        let view = [ViewEntry {
            source_axis: 0,
            start: 1,
            step: 2,
            steps: 3,
        }];
        builder
            .start_band_with_view(
                None,
                &["x"],
                &[8],
                &view,
                BandDataType::UInt8,
                None,
                None,
                None,
            )
            .unwrap();
        builder
            .band_data_writer()
            .append_value(vec![0u8, 1, 2, 3, 4, 5, 6, 7]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        builder.finish().unwrap()
    }

    /// Replace a single column of the bands struct, then rebuild the bands
    /// list and the top-level raster struct. Schema-shape preserving — this
    /// only swaps the array data, never the field type.
    fn replace_band_column(
        array: &StructArray,
        column_index: usize,
        new_column: ArrayRef,
    ) -> StructArray {
        let bands_list = array
            .column(raster_indices::BANDS)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let bands_struct = bands_list
            .values()
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();

        let mut columns: Vec<ArrayRef> = bands_struct.columns().to_vec();
        columns[column_index] = new_column;
        let DataType::Struct(band_fields) = RasterSchema::band_type() else {
            unreachable!("band_type must be Struct")
        };
        let new_bands_struct =
            StructArray::new(band_fields, columns, bands_struct.nulls().cloned());

        let DataType::List(bands_field) = RasterSchema::bands_type() else {
            unreachable!("bands_type must be List")
        };
        let new_bands_list = ListArray::new(
            bands_field,
            bands_list.offsets().clone(),
            Arc::new(new_bands_struct),
            bands_list.nulls().cloned(),
        );

        let mut top_columns: Vec<ArrayRef> = array.columns().to_vec();
        top_columns[raster_indices::BANDS] = Arc::new(new_bands_list);
        let raster_fields = RasterSchema::fields();
        StructArray::new(
            Fields::from(raster_fields.to_vec()),
            top_columns,
            array.nulls().cloned(),
        )
    }

    /// Rebuild the band view list with hand-rolled entries. `entries[i]`
    /// supplies all four `(source_axis, start, step, steps)` Int64 values
    /// for band-row `i`. `nulls` controls per-row validity bits — `None`
    /// means every row is non-null.
    fn make_band_view_list(
        entries: Vec<Vec<(i64, i64, i64, i64)>>,
        nulls: Option<Vec<bool>>,
    ) -> ArrayRef {
        let mut offsets: Vec<i32> = vec![0];
        let mut sa: Vec<i64> = vec![];
        let mut start: Vec<i64> = vec![];
        let mut step: Vec<i64> = vec![];
        let mut steps: Vec<i64> = vec![];
        for row in &entries {
            for &(a, s, k, n) in row {
                sa.push(a);
                start.push(s);
                step.push(k);
                steps.push(n);
            }
            offsets.push(sa.len() as i32);
        }
        let view_struct_fields = Fields::from(vec![
            Field::new("source_axis", DataType::Int64, false),
            Field::new("start", DataType::Int64, false),
            Field::new("step", DataType::Int64, false),
            Field::new("steps", DataType::Int64, false),
        ]);
        let view_struct = StructArray::new(
            view_struct_fields,
            vec![
                Arc::new(arrow_array::PrimitiveArray::<Int64Type>::from(sa)) as ArrayRef,
                Arc::new(arrow_array::PrimitiveArray::<Int64Type>::from(start)) as ArrayRef,
                Arc::new(arrow_array::PrimitiveArray::<Int64Type>::from(step)) as ArrayRef,
                Arc::new(arrow_array::PrimitiveArray::<Int64Type>::from(steps)) as ArrayRef,
            ],
            None,
        );
        let DataType::List(view_field) = RasterSchema::view_type() else {
            unreachable!()
        };
        let null_buf = nulls.map(NullBuffer::from);
        Arc::new(ListArray::new(
            view_field,
            OffsetBuffer::new(ScalarBuffer::from(offsets)),
            Arc::new(view_struct),
            null_buf,
        ))
    }

    // ---- Critical #1: malformed view entries ----

    #[test]
    fn band_returns_none_when_view_has_negative_steps() {
        // Schema accepts negative Int64 in the steps field, but validate_view
        // rejects it. The reader path must surface that as None — never
        // hand back a band whose visible_shape would underflow.
        let array = build_explicit_view_raster();
        let bad_view = make_band_view_list(vec![vec![(0, 0, 1, -1)]], None);
        let mutated = replace_band_column(&array, band_indices::VIEW, bad_view);
        let rasters = RasterStructArray::new(&mutated);
        assert!(rasters.get(0).unwrap().band(0).is_none());
    }

    #[test]
    fn band_returns_none_when_view_source_axis_out_of_range() {
        let array = build_explicit_view_raster();
        let bad_view = make_band_view_list(vec![vec![(5, 0, 1, 3)]], None);
        let mutated = replace_band_column(&array, band_indices::VIEW, bad_view);
        let rasters = RasterStructArray::new(&mutated);
        assert!(rasters.get(0).unwrap().band(0).is_none());
    }

    #[test]
    fn band_returns_none_when_view_length_mismatches_source_shape() {
        // source_shape has 1 dim but view encodes 2 entries.
        let array = build_explicit_view_raster();
        let bad_view = make_band_view_list(vec![vec![(0, 0, 1, 3), (0, 0, 1, 3)]], None);
        let mutated = replace_band_column(&array, band_indices::VIEW, bad_view);
        let rasters = RasterStructArray::new(&mutated);
        assert!(rasters.get(0).unwrap().band(0).is_none());
    }

    #[test]
    fn band_returns_none_when_view_has_duplicate_source_axis() {
        // Need a 2-D source_shape so two entries with source_axis=0 are
        // legal in length but illegal as a permutation.
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder.start_raster(&transform, &[], &[], None).unwrap();
        builder
            .start_band_with_view(
                None,
                &["a", "b"],
                &[2, 3],
                &[
                    ViewEntry {
                        source_axis: 0,
                        start: 0,
                        step: 1,
                        steps: 2,
                    },
                    ViewEntry {
                        source_axis: 1,
                        start: 0,
                        step: 1,
                        steps: 3,
                    },
                ],
                BandDataType::UInt8,
                None,
                None,
                None,
            )
            .unwrap();
        builder.band_data_writer().append_value(vec![0u8; 6]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        let array = builder.finish().unwrap();

        let bad_view = make_band_view_list(vec![vec![(0, 0, 1, 2), (0, 0, 1, 2)]], None);
        let mutated = replace_band_column(&array, band_indices::VIEW, bad_view);
        let rasters = RasterStructArray::new(&mutated);
        assert!(rasters.get(0).unwrap().band(0).is_none());
    }

    // ---- Critical #2: bad data_type discriminant ----

    #[test]
    fn band_and_band_data_type_return_none_for_unknown_discriminant() {
        let array = build_explicit_view_raster();
        let bad_dtype: ArrayRef = Arc::new(UInt32Array::from(vec![0xFFu32]));
        let mutated = replace_band_column(&array, band_indices::DATA_TYPE, bad_dtype);
        let rasters = RasterStructArray::new(&mutated);
        let r = rasters.get(0).unwrap();
        assert!(r.band(0).is_none());
        assert!(r.band_data_type(0).is_none());
    }

    // ---- Critical #3 (reader side): empty source_shape ----

    #[test]
    fn band_returns_none_when_source_shape_is_empty() {
        let array = build_explicit_view_raster();
        // Replace source_shape with a single empty list row.
        let DataType::List(ss_field) = RasterSchema::source_shape_type() else {
            unreachable!()
        };
        let empty_source_shape = ListArray::new(
            ss_field,
            OffsetBuffer::new(ScalarBuffer::from(vec![0i32, 0])),
            Arc::new(UInt64Array::from(Vec::<u64>::new())),
            None,
        );
        let mutated = replace_band_column(
            &array,
            band_indices::SOURCE_SHAPE,
            Arc::new(empty_source_shape),
        );
        let rasters = RasterStructArray::new(&mutated);
        assert!(rasters.get(0).unwrap().band(0).is_none());
    }

    // ---- Important #7: direct fast-path tests ----

    #[test]
    fn raster_ref_fast_paths_return_expected_values() {
        // Single 2-band raster: band 0 has explicit values for nodata,
        // outdb_uri, outdb_format; band 1 has all-nullable fields null.
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster(&transform, &["x", "y"], &[3, 2], None)
            .unwrap();
        builder
            .start_band(
                Some("a"),
                &["y", "x"],
                &[2, 3],
                BandDataType::UInt16,
                Some(&[0xFFu8, 0xFE]),
                Some("s3://bucket/a.tif"),
                Some("GTiff"),
            )
            .unwrap();
        builder.band_data_writer().append_value(vec![0u8; 12]);
        builder.finish_band().unwrap();
        builder
            .start_band(
                Some("b"),
                &["y", "x"],
                &[2, 3],
                BandDataType::Float32,
                None,
                None,
                None,
            )
            .unwrap();
        builder.band_data_writer().append_value(vec![0u8; 24]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::new(&array);
        let r = rasters.get(0).unwrap();

        // Bounds: out-of-range indices yield None on every fast path.
        assert!(r.band_data_type(2).is_none());
        assert!(r.band_outdb_uri(2).is_none());
        assert!(r.band_outdb_format(2).is_none());
        assert!(r.band_nodata(2).is_none());

        // Band 0 — non-null values.
        assert_eq!(r.band_data_type(0), Some(BandDataType::UInt16));
        assert_eq!(r.band_outdb_uri(0), Some("s3://bucket/a.tif"));
        assert_eq!(r.band_outdb_format(0), Some("GTiff"));
        assert_eq!(r.band_nodata(0), Some(&[0xFFu8, 0xFE][..]));

        // Band 1 — null fields.
        assert_eq!(r.band_data_type(1), Some(BandDataType::Float32));
        assert!(r.band_outdb_uri(1).is_none());
        assert!(r.band_outdb_format(1).is_none());
        assert!(r.band_nodata(1).is_none());

        // Cross-check against the BandRef slow path.
        let band0 = r.band(0).unwrap();
        assert_eq!(band0.data_type(), BandDataType::UInt16);
        assert_eq!(band0.outdb_uri(), Some("s3://bucket/a.tif"));
        assert_eq!(band0.outdb_format(), Some("GTiff"));
        assert_eq!(band0.nodata(), Some(&[0xFFu8, 0xFE][..]));
    }

    // ---- Important #9: multi-band, multi-raster mixed identity/explicit ----

    #[test]
    fn multi_raster_mixed_identity_and_explicit_views() {
        // Two rasters. Raster 0 has 3 bands (identity, explicit slice,
        // identity). Raster 1 has 2 bands (explicit broadcast, identity).
        // bands_list.value_offsets() must correctly route each band.
        let mut builder = RasterBuilder::new(2);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];

        // Raster 0
        builder
            .start_raster(&transform, &["x"], &[3], None)
            .unwrap();
        builder
            .start_band(None, &["x"], &[3], BandDataType::UInt8, None, None, None)
            .unwrap();
        builder.band_data_writer().append_value(vec![10u8, 20, 30]);
        builder.finish_band().unwrap();
        builder
            .start_band_with_view(
                None,
                &["x"],
                &[8],
                &[ViewEntry {
                    source_axis: 0,
                    start: 1,
                    step: 2,
                    steps: 3,
                }],
                BandDataType::UInt8,
                None,
                None,
                None,
            )
            .unwrap();
        builder
            .band_data_writer()
            .append_value(vec![0u8, 1, 2, 3, 4, 5, 6, 7]);
        builder.finish_band().unwrap();
        builder
            .start_band(None, &["x"], &[3], BandDataType::UInt8, None, None, None)
            .unwrap();
        builder
            .band_data_writer()
            .append_value(vec![100u8, 101, 102]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        // Raster 1
        builder
            .start_raster(&transform, &["x"], &[4], None)
            .unwrap();
        builder
            .start_band_with_view(
                None,
                &["x"],
                &[1],
                &[ViewEntry {
                    source_axis: 0,
                    start: 0,
                    step: 0,
                    steps: 4,
                }],
                BandDataType::UInt8,
                None,
                None,
                None,
            )
            .unwrap();
        builder.band_data_writer().append_value(vec![42u8]);
        builder.finish_band().unwrap();
        builder
            .start_band(None, &["x"], &[4], BandDataType::UInt8, None, None, None)
            .unwrap();
        builder.band_data_writer().append_value(vec![1u8, 2, 3, 4]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::new(&array);

        // Raster 0 bands: identity (3), slice (3), identity (3).
        let r0 = rasters.get(0).unwrap();
        assert_eq!(r0.num_bands(), 3);
        assert_eq!(r0.band(0).unwrap().shape(), &[3]);
        assert_eq!(
            &*r0.band(0).unwrap().contiguous_data().unwrap(),
            &[10u8, 20, 30]
        );
        assert_eq!(r0.band(1).unwrap().shape(), &[3]);
        assert_eq!(
            &*r0.band(1).unwrap().contiguous_data().unwrap(),
            &[1u8, 3, 5]
        );
        assert_eq!(r0.band(2).unwrap().shape(), &[3]);
        assert_eq!(
            &*r0.band(2).unwrap().contiguous_data().unwrap(),
            &[100u8, 101, 102]
        );

        // Raster 1 bands: broadcast (4 copies of 42), identity (4).
        let r1 = rasters.get(1).unwrap();
        assert_eq!(r1.num_bands(), 2);
        assert_eq!(r1.band(0).unwrap().shape(), &[4]);
        assert_eq!(
            &*r1.band(0).unwrap().contiguous_data().unwrap(),
            &[42u8, 42, 42, 42]
        );
        assert_eq!(r1.band(1).unwrap().shape(), &[4]);
        assert_eq!(
            &*r1.band(1).unwrap().contiguous_data().unwrap(),
            &[1u8, 2, 3, 4]
        );

        // Fast paths must honour the same offsets.
        assert_eq!(r0.band_data_type(1), Some(BandDataType::UInt8));
        assert_eq!(r1.band_data_type(0), Some(BandDataType::UInt8));
        assert_eq!(r1.band_data_type(1), Some(BandDataType::UInt8));
    }

    // ---- Important #10: null raster row, fast path ----

    #[test]
    fn null_raster_row_fast_paths_return_none_after_non_null() {
        // A non-null raster precedes the null one, so the underlying flat
        // band arrays are non-empty. A naive fast path that forgets the
        // bands_list.value_offsets() routing would return *raster 0's*
        // band 0 metadata when asked for raster 1's band 0 — a real bug
        // that a single-null-raster fixture cannot detect.
        let mut builder = RasterBuilder::new(2);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster(&transform, &["x"], &[3], None)
            .unwrap();
        builder
            .start_band(
                Some("a"),
                &["x"],
                &[3],
                BandDataType::UInt16,
                Some(&[0xFFu8, 0xFE]),
                Some("s3://bucket/a.tif"),
                Some("GTiff"),
            )
            .unwrap();
        builder.band_data_writer().append_value(vec![0u8; 6]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        builder.append_null().unwrap();
        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::new(&array);

        // Sanity: raster 0 still resolves correctly.
        let r0 = rasters.get(0).unwrap();
        assert_eq!(r0.band_data_type(0), Some(BandDataType::UInt16));
        assert_eq!(r0.band_outdb_uri(0), Some("s3://bucket/a.tif"));

        // Raster 1 is null with zero bands. Every per-band lookup is
        // out of range and must return None even though the flat
        // underlying arrays still hold raster 0's data.
        assert!(rasters.is_null(1));
        let r1 = rasters.get(1).unwrap();
        assert_eq!(r1.num_bands(), 0);
        assert!(r1.band(0).is_none());
        assert!(r1.band_data_type(0).is_none());
        assert!(r1.band_outdb_uri(0).is_none());
        assert!(r1.band_outdb_format(0).is_none());
        assert!(r1.band_nodata(0).is_none());
    }

    // ---- Fast-path / band(i) divergence on a corrupt view ----

    #[test]
    fn fast_paths_return_columnar_values_when_band_is_corrupt() {
        // band(i) goes through validate_view and returns None for a
        // malformed view; the columnar fast paths read their fields
        // directly without consulting the view at all. Pin down that
        // contract so a future reader doesn't accidentally couple them
        // (or "fix" the divergence in either direction without us
        // noticing). Also catches a regression where a fast path would
        // panic instead of returning the underlying value.
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster(&transform, &["x"], &[3], None)
            .unwrap();
        builder
            .start_band_with_view(
                Some("a"),
                &["x"],
                &[8],
                &[ViewEntry {
                    source_axis: 0,
                    start: 1,
                    step: 2,
                    steps: 3,
                }],
                BandDataType::UInt32,
                Some(&[0u8, 0, 0, 0]),
                Some("s3://bucket/a.tif"),
                Some("GTiff"),
            )
            .unwrap();
        builder.band_data_writer().append_value(vec![0u8; 32]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        let array = builder.finish().unwrap();

        let bad_view = make_band_view_list(vec![vec![(0, 0, 1, -1)]], None);
        let mutated = replace_band_column(&array, band_indices::VIEW, bad_view);
        let rasters = RasterStructArray::new(&mutated);
        let r = rasters.get(0).unwrap();

        // band(i) rejects on validate_view.
        assert!(r.band(0).is_none());

        // Fast paths still surface the underlying columnar values —
        // they don't validate the view, by design. Locking that in.
        assert_eq!(r.band_data_type(0), Some(BandDataType::UInt32));
        assert_eq!(r.band_outdb_uri(0), Some("s3://bucket/a.tif"));
        assert_eq!(r.band_outdb_format(0), Some("GTiff"));
        assert_eq!(r.band_nodata(0), Some(&[0u8, 0, 0, 0][..]));
    }
}
