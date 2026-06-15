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
    Array, BinaryArray, BinaryViewArray, Float64Array, Int64Array, ListArray, StringArray,
    StringViewArray, StructArray, UInt32Array,
};
use arrow_schema::ArrowError;

use crate::traits::{BandRef, Bands, NdBuffer, RasterRef};
use crate::view_entries::{ViewEntries, ViewEntry};
use sedona_schema::raster::{band_indices, band_view_indices, raster_indices, BandDataType};

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
    source_shape_values: &'a Int64Array,
    nodata_array: &'a BinaryArray,
    outdb_uri_array: &'a StringArray,
    outdb_format_array: &'a StringViewArray,
    data_array: &'a BinaryViewArray,
    /// Absolute row index within the flattened bands arrays
    band_row: usize,
    /// Resolved at construction so accessors don't re-decode the discriminant.
    data_type: BandDataType,
    /// Per-visible-axis view, length = ndim
    view_entries: ViewEntries,
    /// Visible shape (== `[v.steps for v in view_entries]`), length = ndim.
    /// `i64` to match `BandRef::shape()`'s return type and the surrounding
    /// view-machinery arithmetic (strides, offsets). `validate_view`
    /// guarantees entries are non-negative.
    visible_shape: Vec<i64>,
    /// Byte strides per visible axis. May be 0 (broadcast) or negative.
    byte_strides: Vec<i64>,
    /// Byte offset into `data` of the visible region's `[0,...,0]` element.
    /// Typed `i64` to match the surrounding stride arithmetic
    /// (`byte_strides` are `i64` to allow negative steps). Always non-negative
    /// by construction — `RasterRefImpl::band` asserts `>= 0` before storing.
    byte_offset: i64,
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

    fn shape(&self) -> &[i64] {
        &self.visible_shape
    }

    fn raw_source_shape(&self) -> &[i64] {
        let start = self.source_shape_list.value_offsets()[self.band_row] as usize;
        let end = self.source_shape_list.value_offsets()[self.band_row + 1] as usize;
        &self.source_shape_values.values()[start..end]
    }

    fn view(&self) -> &[ViewEntry] {
        self.view_entries.as_slice()
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

    fn is_indb(&self) -> bool {
        // A 0-element visible region (any visible dim is 0) holds no readable
        // bytes — trivially fully in-RAM — so it's InDb, not the OutDb
        // empty-`data` sentinel. Otherwise the discriminator is buffer presence.
        self.shape().iter().product::<i64>() == 0
            || !self.data_array.value(self.band_row).is_empty()
    }

    fn nd_buffer(&self) -> Result<NdBuffer<'_>, ArrowError> {
        if !self.is_indb() {
            return Err(ArrowError::NotYetImplemented(
                "OutDb byte access via nd_buffer() is not yet implemented; \
                 backend-specific OutDb resolvers are tracked separately"
                    .to_string(),
            ));
        }
        // shape and strides are owned by NdBuffer (see its doc comment).
        // Cloning here is cheap — both vecs are O(ndim), a handful of values.
        Ok(NdBuffer {
            buffer: self.data_array.value(self.band_row),
            shape: self.visible_shape.clone(),
            strides: self.byte_strides.clone(),
            offset: self.byte_offset,
            data_type: self.data_type,
        })
    }
}

/// Verify that every byte the view can address lies within `buffer_len`
/// and that every stride × index product (and their accumulations) fits
/// in i64.
///
/// **Load-bearing**: this is the *only* bound check between the view's
/// byte-stride description and the data buffer. Stride-aware consumers walk
/// the buffer with plain-arithmetic indexing and rely on this precheck
/// having proven every addressed byte is in range. Two corruption modes it
/// catches:
///
///   1. A writer that lies about `source_shape` (Arrow column shorter
///      than the view promises).
///   2. A composed view whose stride × index product or accumulated
///      offset overflows i64 even though `validate_view` accepted the
///      per-entry bounds.
///
/// Empty visible regions (any axis with `steps == 0`) address no bytes
/// and skip the check.
fn check_view_buffer_bounds(
    buffer_len: usize,
    visible_shape: &[i64],
    byte_strides: &[i64],
    byte_offset: i64,
    dtype_size: usize,
) -> Result<(), ArrowError> {
    if visible_shape.contains(&0) {
        return Ok(());
    }
    let mut min_offset = byte_offset;
    let mut max_offset = byte_offset;
    for (k, &stride) in byte_strides.iter().enumerate() {
        // `validate_view` guarantees `steps >= 0`, so `visible_shape[k] >= 0`
        // and `visible_shape[k] - 1` is in-range for any non-empty axis.
        let last_idx = visible_shape[k] - 1;
        let contribution = last_idx.checked_mul(stride).ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!(
                "max addressable offset on axis {k} overflows i64"
            ))
        })?;
        if contribution > 0 {
            max_offset = max_offset.checked_add(contribution).ok_or_else(|| {
                ArrowError::InvalidArgumentError(
                    "max addressable offset accumulation overflows i64".to_string(),
                )
            })?;
        } else if contribution < 0 {
            min_offset = min_offset.checked_add(contribution).ok_or_else(|| {
                ArrowError::InvalidArgumentError(
                    "min addressable offset accumulation overflows i64".to_string(),
                )
            })?;
        }
    }
    let last_byte = max_offset
        .checked_add(dtype_size as i64 - 1)
        .ok_or_else(|| {
            ArrowError::InvalidArgumentError("max addressable byte overflows i64".to_string())
        })?;
    if min_offset < 0 {
        return Err(ArrowError::InvalidArgumentError(format!(
            "view addresses out-of-bounds negative byte offset {min_offset}"
        )));
    }
    let buffer_len_i64 = i64::try_from(buffer_len).map_err(|_| {
        ArrowError::InvalidArgumentError(format!("buffer length {buffer_len} exceeds i64::MAX"))
    })?;
    if last_byte >= buffer_len_i64 {
        return Err(ArrowError::InvalidArgumentError(format!(
            "view addresses byte {last_byte} but buffer is only {buffer_len} bytes"
        )));
    }
    Ok(())
}

/// Arrow-backed implementation of RasterRef for a single raster row.
///
/// Holds flat references to the underlying Arrow arrays so the impl does
/// not borrow from a `RasterStructArray` wrapper. That keeps
/// `RasterStructArray::get(&self, ...)` callable without a `&'a self`
/// constraint, which would otherwise force callers to hoist the
/// `RasterStructArray` into a `let` binding.
pub struct RasterRefImpl<'a> {
    crs_array: &'a StringViewArray,
    transform_list: &'a ListArray,
    transform_values: &'a Float64Array,
    spatial_dims_list: &'a ListArray,
    spatial_dims_values: &'a StringViewArray,
    spatial_shape_list: &'a ListArray,
    spatial_shape_values: &'a Int64Array,
    bands_list: &'a ListArray,
    band_name_array: &'a StringArray,
    band_dim_names_list: &'a ListArray,
    band_dim_names_values: &'a StringArray,
    band_source_shape_list: &'a ListArray,
    band_source_shape_values: &'a Int64Array,
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
    raster_index: usize,
}

impl<'a> RasterRefImpl<'a> {
    /// Returns the raw CRS string reference with the array's lifetime.
    pub fn crs_str_ref(&self) -> Option<&'a str> {
        if self.crs_array.is_null(self.raster_index) {
            None
        } else {
            Some(self.crs_array.value(self.raster_index))
        }
    }

    /// Read the band's source_shape and convert u64 → i64 with overflow check.
    ///
    /// Rejects 0-D bands (empty source_shape) at the read boundary: the schema
    /// doesn't forbid them outright but every consumer assumes ndim >= 1. Every
    /// downstream consumer in the view machinery wants i64 (matches ViewEntry's
    /// signed fields and the stride arithmetic); converting once here keeps the
    /// rest of band() free of mixed-signedness gymnastics.
    fn read_band_source_shape(&self, band_row: usize) -> Result<Vec<i64>, ArrowError> {
        let ss_start = self.band_source_shape_list.value_offsets()[band_row] as usize;
        let ss_end = self.band_source_shape_list.value_offsets()[band_row + 1] as usize;
        let source_shape: &[i64] = &self.band_source_shape_values.values()[ss_start..ss_end];

        if source_shape.is_empty() {
            return Err(ArrowError::ExternalError(Box::new(
                sedona_common::sedona_internal_datafusion_err!(
                    "band {band_row} has empty source_shape; ndim must be >= 1"
                ),
            )));
        }

        Ok(source_shape.to_vec())
    }

    /// Resolve the band's data-type discriminant or fail. An unknown
    /// discriminant is schema-corruption, not user data.
    fn read_band_data_type_or_err(&self, band_row: usize) -> Result<BandDataType, ArrowError> {
        let data_type_value = self.band_datatype_array.value(band_row);
        BandDataType::try_from_u32(data_type_value).ok_or_else(|| {
            ArrowError::ExternalError(Box::new(sedona_common::sedona_internal_datafusion_err!(
                "band {band_row} has unknown data_type discriminant {data_type_value}"
            )))
        })
    }

    /// Read the band's view-entry list. Identity is encoded exclusively as a
    /// NULL row — an empty (non-null, zero-length) list is malformed and
    /// rejected later by [`ViewEntries::validate`]. The schema (see
    /// `RasterSchema::view_type`) documents this contract.
    fn read_band_view_entries(
        &self,
        band_row: usize,
        source_shape: &[i64],
    ) -> Result<ViewEntries, ArrowError> {
        if self.band_view_list.is_null(band_row) {
            return Ok(ViewEntries::identity_for_shape(source_shape));
        }
        let v_start = self.band_view_list.value_offsets()[band_row] as usize;
        let v_end = self.band_view_list.value_offsets()[band_row + 1] as usize;
        Ok(ViewEntries::new(
            (v_start..v_end)
                .map(|i| ViewEntry {
                    source_axis: self.band_view_source_axis.value(i),
                    start: self.band_view_start.value(i),
                    step: self.band_view_step.value(i),
                    steps: self.band_view_steps.value(i),
                })
                .collect(),
        ))
    }

    /// Resolve a 0-based band `index` to its absolute row in the flattened
    /// bands arrays, bounds-checked against this raster's band count.
    fn resolve_band_row(&self, index: usize) -> Result<usize, ArrowError> {
        let nbands = self.num_bands();
        if index >= nbands {
            return Err(ArrowError::InvalidArgumentError(format!(
                "Band index {index} is out of range: this raster has {nbands} bands"
            )));
        }
        let start = self.bands_list.value_offsets()[self.raster_index] as usize;
        Ok(start + index)
    }

    /// Read the band's view and source shape, validate the view against the
    /// source shape, and compose the byte-stride layout (visible shape, byte
    /// strides, byte offset), checking it against the backing data buffer.
    fn compose_band_layout(
        &self,
        band_row: usize,
        data_type: BandDataType,
    ) -> Result<BandLayout, ArrowError> {
        let source_shape = self.read_band_source_shape(band_row)?;
        let view_entries = self.read_band_view_entries(band_row, &source_shape)?;
        view_entries.validate(&source_shape).map_err(|e| {
            ArrowError::ExternalError(Box::new(sedona_common::sedona_internal_datafusion_err!(
                "band {band_row} has malformed view: {e}"
            )))
        })?;

        let visible_shape = view_entries.visible_shape();
        let (byte_strides, byte_offset) = compose_byte_strides(
            band_row,
            &source_shape,
            &view_entries,
            data_type.byte_size(),
        )?;

        self.check_band_buffer_bounds(
            band_row,
            &visible_shape,
            &byte_strides,
            byte_offset,
            data_type,
        )?;

        Ok(BandLayout {
            view_entries,
            visible_shape,
            byte_strides,
            byte_offset,
        })
    }

    /// For InDb bands, verify the `data` BinaryView is long enough to cover
    /// every byte the composed view can address. [`ViewEntries::validate`]
    /// doesn't know the actual buffer length, so a writer that lies about
    /// `source_shape` vs the bytes written would otherwise slip through and
    /// panic later when a consumer walks the strided buffer. OutDb bands skip
    /// this: their data column is empty by design.
    fn check_band_buffer_bounds(
        &self,
        band_row: usize,
        visible_shape: &[i64],
        byte_strides: &[i64],
        byte_offset: i64,
        data_type: BandDataType,
    ) -> Result<(), ArrowError> {
        let data_bytes = self.band_data_array.value(band_row);
        if data_bytes.is_empty() {
            return Ok(());
        }
        check_view_buffer_bounds(
            data_bytes.len(),
            visible_shape,
            byte_strides,
            byte_offset,
            data_type.byte_size(),
        )
        .map_err(|e| {
            ArrowError::ExternalError(Box::new(sedona_common::sedona_internal_datafusion_err!(
                "band {band_row}: view-buffer bounds check failed: {e}"
            )))
        })
    }
}

/// The composed, validated byte-stride layout for one band's view — everything
/// [`RasterRefImpl::band`] derives before constructing a [`BandRefImpl`].
struct BandLayout {
    view_entries: ViewEntries,
    visible_shape: Vec<i64>,
    byte_strides: Vec<i64>,
    byte_offset: i64,
}

/// Compose a validated view against a source shape into C-order byte strides
/// and a byte offset.
///
/// C-order source strides are dtype-scaled cumulative products of `source_shape`,
/// then each visible axis's stride/offset is composed as `view.step *
/// src_stride` / `view.start * src_stride`. All arithmetic is checked: even
/// after `ViewEntries::validate`, the cumulative byte product can overflow
/// `i64` for cosmically large shapes, and a corrupt source_shape whose product
/// wraps would otherwise silently pass downstream bound checks. The returned
/// `byte_offset` is non-negative by construction (start >= 0, src_stride > 0);
/// the defensive sign check guards future refactors that might break that
/// invariant before we cross the i64 → u64 boundary in `nd_buffer()`.
fn compose_byte_strides(
    band_row: usize,
    source_shape: &[i64],
    view_entries: &ViewEntries,
    dtype_byte_size: usize,
) -> Result<(Vec<i64>, i64), ArrowError> {
    let overflow_err = |msg: &str| {
        ArrowError::ExternalError(Box::new(sedona_common::sedona_internal_datafusion_err!(
            "band {band_row}: {msg}"
        )))
    };

    let dtype_size = dtype_byte_size as i64;

    let mut source_strides_bytes = vec![0i64; source_shape.len()];
    source_strides_bytes[source_shape.len() - 1] = dtype_size;
    for k in (0..source_shape.len() - 1).rev() {
        source_strides_bytes[k] = source_strides_bytes[k + 1]
            .checked_mul(source_shape[k + 1])
            .ok_or_else(|| overflow_err("source-stride product overflows i64"))?;
    }

    let mut byte_strides = vec![0i64; view_entries.len()];
    let mut byte_offset: i64 = 0;
    for (k, v) in view_entries.iter().enumerate() {
        let src_stride = source_strides_bytes[v.source_axis as usize];
        byte_strides[k] = v
            .step
            .checked_mul(src_stride)
            .ok_or_else(|| overflow_err("view step × source-stride overflows i64"))?;
        let start_off = v
            .start
            .checked_mul(src_stride)
            .ok_or_else(|| overflow_err("view start × source-stride overflows i64"))?;
        byte_offset = byte_offset
            .checked_add(start_off)
            .ok_or_else(|| overflow_err("view offset accumulation overflows i64"))?;
    }

    if byte_offset < 0 {
        return Err(overflow_err("composed byte_offset is negative"));
    }

    Ok((byte_strides, byte_offset))
}

impl<'a> RasterRef for RasterRefImpl<'a> {
    fn num_bands(&self) -> usize {
        self.bands_list.value_length(self.raster_index) as usize
    }

    fn bands(&self) -> Bands<'_> {
        Bands::new(self)
    }

    fn band(&self, index: usize) -> Result<Box<dyn BandRef + '_>, ArrowError> {
        let band_row = self.resolve_band_row(index)?;
        let data_type = self.read_band_data_type_or_err(band_row)?;
        let layout = self.compose_band_layout(band_row, data_type)?;

        Ok(Box::new(BandRefImpl {
            dim_names_list: self.band_dim_names_list,
            dim_names_values: self.band_dim_names_values,
            source_shape_list: self.band_source_shape_list,
            source_shape_values: self.band_source_shape_values,
            nodata_array: self.band_nodata_array,
            outdb_uri_array: self.band_outdb_uri_array,
            outdb_format_array: self.band_outdb_format_array,
            data_array: self.band_data_array,
            band_row,
            data_type,
            view_entries: layout.view_entries,
            visible_shape: layout.visible_shape,
            byte_strides: layout.byte_strides,
            byte_offset: layout.byte_offset,
        }))
    }

    fn band_data_type(&self, index: usize) -> Option<BandDataType> {
        if index >= self.num_bands() {
            return None;
        }
        let start = self.bands_list.value_offsets()[self.raster_index] as usize;
        let band_row = start + index;
        let value = self.band_datatype_array.value(band_row);
        BandDataType::try_from_u32(value)
    }

    fn band_outdb_uri(&self, index: usize) -> Option<&str> {
        if index >= self.num_bands() {
            return None;
        }
        let start = self.bands_list.value_offsets()[self.raster_index] as usize;
        let band_row = start + index;
        if self.band_outdb_uri_array.is_null(band_row) {
            None
        } else {
            Some(self.band_outdb_uri_array.value(band_row))
        }
    }

    fn band_outdb_format(&self, index: usize) -> Option<&str> {
        if index >= self.num_bands() {
            return None;
        }
        let start = self.bands_list.value_offsets()[self.raster_index] as usize;
        let band_row = start + index;
        if self.band_outdb_format_array.is_null(band_row) {
            None
        } else {
            Some(self.band_outdb_format_array.value(band_row))
        }
    }

    fn band_nodata(&self, index: usize) -> Option<&[u8]> {
        if index >= self.num_bands() {
            return None;
        }
        let start = self.bands_list.value_offsets()[self.raster_index] as usize;
        let band_row = start + index;
        if self.band_nodata_array.is_null(band_row) {
            None
        } else {
            Some(self.band_nodata_array.value(band_row))
        }
    }

    fn band_name(&self, index: usize) -> Option<&str> {
        if index >= self.num_bands() {
            return None;
        }
        let start = self.bands_list.value_offsets()[self.raster_index] as usize;
        let band_row = start + index;
        if self.band_name_array.is_null(band_row) {
            None
        } else {
            Some(self.band_name_array.value(band_row))
        }
    }

    fn crs(&self) -> Option<&str> {
        self.crs_str_ref()
    }

    fn transform(&self) -> &[f64] {
        let start = self.transform_list.value_offsets()[self.raster_index] as usize;
        let end = self.transform_list.value_offsets()[self.raster_index + 1] as usize;
        assert!(
            end - start >= 6,
            "transform list must have at least 6 elements for raster {}, got {}",
            self.raster_index,
            end - start
        );
        &self.transform_values.values()[start..start + 6]
    }

    fn spatial_dims(&self) -> Vec<&str> {
        let offsets = self.spatial_dims_list.value_offsets();
        let start = offsets[self.raster_index] as usize;
        let end = offsets[self.raster_index + 1] as usize;
        (start..end)
            .map(|i| self.spatial_dims_values.value(i))
            .collect()
    }

    fn spatial_shape(&self) -> &[i64] {
        let offsets = self.spatial_shape_list.value_offsets();
        let start = offsets[self.raster_index] as usize;
        let end = offsets[self.raster_index + 1] as usize;
        &self.spatial_shape_values.values()[start..end]
    }
}

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
    band_source_shape_values: &'a Int64Array,
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
            .downcast_ref::<Int64Array>()
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
    pub fn get(&self, index: usize) -> Result<RasterRefImpl<'a>, ArrowError> {
        if index >= self.raster_array.len() {
            return Err(ArrowError::InvalidArgumentError(format!(
                "Invalid raster index: {index}"
            )));
        }
        Ok(RasterRefImpl {
            crs_array: self.crs_array,
            transform_list: self.transform_list,
            transform_values: self.transform_values,
            spatial_dims_list: self.spatial_dims_list,
            spatial_dims_values: self.spatial_dims_values,
            spatial_shape_list: self.spatial_shape_list,
            spatial_shape_values: self.spatial_shape_values,
            bands_list: self.bands_list,
            band_name_array: self.band_name_array,
            band_dim_names_list: self.band_dim_names_list,
            band_dim_names_values: self.band_dim_names_values,
            band_source_shape_list: self.band_source_shape_list,
            band_source_shape_values: self.band_source_shape_values,
            band_datatype_array: self.band_datatype_array,
            band_nodata_array: self.band_nodata_array,
            band_view_list: self.band_view_list,
            band_view_source_axis: self.band_view_source_axis,
            band_view_start: self.band_view_start,
            band_view_step: self.band_view_step,
            band_view_steps: self.band_view_steps,
            band_outdb_uri_array: self.band_outdb_uri_array,
            band_outdb_format_array: self.band_outdb_format_array,
            band_data_array: self.band_data_array,
            raster_index: index,
        })
    }

    /// Check if a raster at the given index is null.
    #[inline(always)]
    pub fn is_null(&self, index: usize) -> bool {
        self.raster_array.is_null(index)
    }

    /// The flattened band `data` column (BinaryView) shared by every raster
    /// in this array. Pair with [`Self::band_data_row`] to address a single
    /// band's bytes — e.g. for zero-copy passthrough into a [`RasterBuilder`]
    /// via `append_band_data_from`.
    #[inline(always)]
    pub fn band_data_array(&self) -> &'a BinaryViewArray {
        self.band_data_array
    }

    /// Absolute row of band `band_idx` of raster `raster_idx` within the
    /// flattened band arrays (such as [`Self::band_data_array`]).
    #[inline(always)]
    pub fn band_data_row(&self, raster_idx: usize, band_idx: usize) -> usize {
        self.bands_list.value_offsets()[raster_idx] as usize + band_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::{RasterBuilder, StartBandWithViewArgs};
    use crate::traits::{BandMetadata, RasterMetadata};
    use crate::view_entries::ViewEntry;
    use arrow_array::{types::Int64Type, ArrayRef, ListArray, StructArray, UInt32Array};
    use arrow_buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
    use arrow_schema::{DataType, Field, Fields};
    use sedona_schema::raster::{
        band_indices, raster_indices, BandDataType, RasterSchema, StorageType,
    };
    use sedona_testing::rasters::generate_test_rasters;
    use std::sync::Arc;

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

        builder.start_raster(&metadata, Some(epsg4326)).unwrap();

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

        let bands = raster.bands();
        assert_eq!(bands.len(), 1);
        assert!(!bands.is_empty());

        // Access band with 1-based band_number
        let band = bands.band(1).unwrap();
        assert_eq!(
            band.nd_buffer().unwrap().as_contiguous().unwrap().len(),
            100
        );
        assert_eq!(band.nd_buffer().unwrap().as_contiguous().unwrap()[0], 1u8);

        let band_meta = band.metadata();
        assert_eq!(band_meta.storage_type().unwrap(), StorageType::InDb);
        assert_eq!(band_meta.data_type().unwrap(), BandDataType::UInt8);

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

        builder.start_raster(&metadata, None).unwrap();

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
            assert!(band
                .nd_buffer()
                .unwrap()
                .as_contiguous()
                .unwrap()
                .iter()
                .all(|&x| x == expected_value));
        }

        // Test array
        let band_values: Vec<u8> = bands
            .iter()
            .enumerate()
            .map(|(i, band)| {
                let band = band.unwrap();
                assert_eq!(
                    band.nd_buffer().unwrap().as_contiguous().unwrap()[0],
                    i as u8
                );
                band.nd_buffer().unwrap().as_contiguous().unwrap()[0]
            })
            .collect();

        assert_eq!(band_values, vec![0, 1, 2]);
    }

    #[test]
    fn test_raster_is_null() {
        let raster_array = generate_test_rasters(2, Some(1)).unwrap();
        let rasters = RasterStructArray::new(&raster_array);
        assert_eq!(rasters.len(), 2);
        assert!(!rasters.is_null(0));
        assert!(rasters.is_null(1));
    }

    /// Build a single-raster, single-band raster StructArray with an explicit
    /// view. Used as the input to the surgery helpers below; callers replace
    /// one band-level column to simulate schema corruption.
    fn build_explicit_view_raster() -> StructArray {
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster_nd(&transform, &["x"], &[3], None)
            .unwrap();
        let view = [ViewEntry {
            source_axis: 0,
            start: 1,
            step: 2,
            steps: 3,
        }];
        builder
            .start_band_with_view(StartBandWithViewArgs {
                name: None,
                dim_names: &["x"],
                source_shape: &[8],
                view: &view,
                data_type: BandDataType::UInt8,
                nodata: None,
                outdb_uri: None,
                outdb_format: None,
            })
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
    fn band_returns_none_when_view_length_mismatches_source_shape() {
        // source_shape has 1 dim but view encodes 2 entries.
        let array = build_explicit_view_raster();
        let bad_view = make_band_view_list(vec![vec![(0, 0, 1, 3), (0, 0, 1, 3)]], None);
        let mutated = replace_band_column(&array, band_indices::VIEW, bad_view);
        let rasters = RasterStructArray::new(&mutated);
        assert!(rasters.get(0).unwrap().band(0).is_err());
    }

    // ---- Critical #2: bad data_type discriminant ----

    #[test]
    fn band_and_band_data_type_surface_corruption_for_unknown_discriminant() {
        let array = build_explicit_view_raster();
        let bad_dtype: ArrayRef = Arc::new(UInt32Array::from(vec![0xFFu32]));
        let mutated = replace_band_column(&array, band_indices::DATA_TYPE, bad_dtype);
        let rasters = RasterStructArray::new(&mutated);
        let r = rasters.get(0).unwrap();
        // band() surfaces the corruption through the standardized
        // SedonaDB-internal-error message routed via ArrowError::ExternalError.
        // `Box<dyn BandRef>` isn't `Debug`, so unwrap_err doesn't compile —
        // pull the error out via `.err().unwrap()` on the `Option<E>` side.
        let err = r.band(0).err().unwrap();
        assert!(err.to_string().contains("SedonaDB internal error"));
        assert!(err.to_string().contains("data_type discriminant"));
        // band_data_type retains its `Option` fast-path shape — corrupt
        // discriminant collapses to None for consistency with the existing
        // accessor's contract.
        assert!(r.band_data_type(0).is_none());
    }

    // empty source_shape

    #[test]
    fn band_surfaces_internal_error_when_source_shape_is_empty() {
        let array = build_explicit_view_raster();
        // Replace source_shape with a single empty list row.
        let DataType::List(ss_field) = RasterSchema::source_shape_type() else {
            unreachable!()
        };
        let empty_source_shape = ListArray::new(
            ss_field,
            OffsetBuffer::new(ScalarBuffer::from(vec![0i32, 0])),
            Arc::new(Int64Array::from(Vec::<i64>::new())),
            None,
        );
        let mutated = replace_band_column(
            &array,
            band_indices::SOURCE_SHAPE,
            Arc::new(empty_source_shape),
        );
        let rasters = RasterStructArray::new(&mutated);
        let err = rasters.get(0).unwrap().band(0).err().unwrap();
        assert!(err.to_string().contains("SedonaDB internal error"));
        assert!(err.to_string().contains("empty source_shape"));
    }

    #[test]
    fn band_surfaces_internal_error_when_data_column_shorter_than_view() {
        // build_explicit_view_raster writes 8 UInt8 source bytes with view
        // (start=1, step=2, steps=3) which addresses bytes 1, 3, 5.
        // Inflate source_shape to [16] and the view to cover steps=10 along
        // the (now nominally-larger) source axis: the byte range jumps past
        // the actual 8-byte data column and the precheck must fire.
        let array = build_explicit_view_raster();
        // source_shape := [16]
        let new_source_shape = make_band_source_shape_list(vec![vec![16i64]]);
        let mutated_ss = replace_band_column(&array, band_indices::SOURCE_SHAPE, new_source_shape);
        // view := (source_axis=0, start=0, step=1, steps=10) — addresses
        // bytes 0..10 but the underlying data column only has 8 bytes.
        let new_view = make_band_view_list(vec![vec![(0, 0, 1, 10)]], None);
        let mutated = replace_band_column(&mutated_ss, band_indices::VIEW, new_view);
        let rasters = RasterStructArray::new(&mutated);
        let err = rasters.get(0).unwrap().band(0).err().unwrap();
        assert!(err.to_string().contains("SedonaDB internal error"));
        assert!(err.to_string().contains("view-buffer bounds check failed"));
    }

    #[test]
    fn band_rejects_empty_non_null_view_row() {
        // The identity view is encoded exclusively as a NULL row; a
        // non-null zero-length list is malformed and must error rather
        // than silently fall back to identity. (Pre-rev behaviour
        // accepted it — see `RasterSchema::view_type` for the contract.)
        let array = build_explicit_view_raster();
        let empty_non_null_view = make_band_view_list(vec![vec![]], Some(vec![true]));
        let mutated = replace_band_column(&array, band_indices::VIEW, empty_non_null_view);
        let rasters = RasterStructArray::new(&mutated);
        let err = rasters.get(0).unwrap().band(0).err().unwrap();
        assert!(err.to_string().contains("view length"), "got: {err}");
    }

    // ---- Stride composition overflow guards ----

    /// Build a band source_shape list with hand-rolled i64 entries so tests
    /// can inject values that the builder's writer-side checks would refuse.
    fn make_band_source_shape_list(rows: Vec<Vec<i64>>) -> ArrayRef {
        let mut offsets: Vec<i32> = vec![0];
        let mut values: Vec<i64> = vec![];
        for row in &rows {
            values.extend_from_slice(row);
            offsets.push(values.len() as i32);
        }
        let DataType::List(field) = RasterSchema::source_shape_type() else {
            unreachable!()
        };
        Arc::new(ListArray::new(
            field,
            OffsetBuffer::new(ScalarBuffer::from(offsets)),
            Arc::new(Int64Array::from(values)),
            None,
        ))
    }

    #[test]
    fn band_returns_none_when_source_strides_product_overflows() {
        // dtype_size × Π source_shape[j>k] must not silently wrap. With a
        // 3-D source_shape of `[1, 1<<32, 1<<32]` the product (1<<32) ×
        // (1<<32) = 1<<64 overflows i64 in the source-stride build.
        let array = build_explicit_view_raster();
        let new_source_shape =
            make_band_source_shape_list(vec![vec![1i64, 1i64 << 32, 1i64 << 32]]);
        let mutated_ss = replace_band_column(&array, band_indices::SOURCE_SHAPE, new_source_shape);
        // Pad the view to 3 entries; steps=0 on the giant axes keeps
        // validate_view's start/last checks out of the casts-from-u64 path.
        let new_view =
            make_band_view_list(vec![vec![(0, 0, 1, 1), (1, 0, 1, 0), (2, 0, 1, 0)]], None);
        let mutated = replace_band_column(&mutated_ss, band_indices::VIEW, new_view);
        let rasters = RasterStructArray::new(&mutated);
        assert!(rasters.get(0).unwrap().band(0).is_err());
    }

    #[test]
    fn band_returns_none_when_view_step_times_source_stride_overflows() {
        // `validate_view` bounds (steps-1)*step + start on the SOURCE axis
        // but doesn't bound v.step × cumulative_byte_stride. A view with a
        // small visible region but a step large enough to wrap the byte
        // stride must be rejected at construction.
        //
        // Source `[3, 1<<60]`, dtype_size=1 (UInt8) → src_stride[0] = 1<<60.
        // View on axis 0 with step=8 makes byte_strides[0] = 8 × (1<<60) =
        // 1<<63 which overflows i64. The view itself only walks 1 step on
        // that axis so validate_view's (steps-1)*step bound holds.
        let array = build_explicit_view_raster();
        let new_source_shape = make_band_source_shape_list(vec![vec![3i64, 1i64 << 60]]);
        let mutated_ss = replace_band_column(&array, band_indices::SOURCE_SHAPE, new_source_shape);
        let new_view = make_band_view_list(vec![vec![(0, 0, 8, 1), (1, 0, 1, 1)]], None);
        let mutated = replace_band_column(&mutated_ss, band_indices::VIEW, new_view);
        let rasters = RasterStructArray::new(&mutated);
        assert!(rasters.get(0).unwrap().band(0).is_err());
    }

    #[test]
    fn raster_ref_fast_paths_return_expected_values() {
        // Single 2-band raster: band 0 has explicit values for nodata,
        // outdb_uri, outdb_format; band 1 has all-nullable fields null.
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster_nd(&transform, &["x", "y"], &[3, 2], None)
            .unwrap();
        builder
            .start_band_nd(
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
            .start_band_nd(
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

        // bands() view: 1-based band(N), len, is_empty, iter — same shape as
        // pre-N-D callers expect. Exercise via the concrete type and via a
        // `&dyn RasterRef` to confirm both dispatch paths work.
        let bands = r.bands();
        assert_eq!(bands.len(), 2);
        assert!(!bands.is_empty());
        assert_eq!(bands.band(1).unwrap().data_type(), BandDataType::UInt16);
        assert_eq!(bands.band(2).unwrap().data_type(), BandDataType::Float32);
        assert!(bands.band(0).is_err()); // 0 is invalid (1-based)
        assert!(bands.band(3).is_err()); // out of range
        assert_eq!(bands.iter().count(), 2);
        let dyn_r: &dyn RasterRef = &r;
        assert_eq!(dyn_r.bands().len(), 2);

        // metadata() shim: concrete RasterMetadata/BandMetadata values.
        let m = r.metadata();
        assert_eq!(m.width(), 3);
        assert_eq!(m.height(), 2);
        assert_eq!(m.upper_left_x(), 0.0);
        assert_eq!(m.scale_x(), 1.0);
        let b0 = r.band(0).unwrap();
        let bm0 = b0.metadata();
        assert_eq!(bm0.data_type().unwrap(), BandDataType::UInt16);
        assert_eq!(
            bm0.storage_type().unwrap(),
            sedona_schema::raster::StorageType::InDb
        );
        assert_eq!(bm0.nodata_value(), Some(&[0xFFu8, 0xFE][..]));
        // Band 0 is InDb (has bytes), so outdb_* are hidden via the shim
        // even though the row carries an outdb_uri hint.
        assert!(bm0.outdb_url().is_none());
        assert!(bm0.outdb_band_id().is_none());
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
            .start_raster_nd(&transform, &["x"], &[3], None)
            .unwrap();
        builder
            .start_band_nd(None, &["x"], &[3], BandDataType::UInt8, None, None, None)
            .unwrap();
        builder.band_data_writer().append_value(vec![10u8, 20, 30]);
        builder.finish_band().unwrap();
        builder
            .start_band_with_view(StartBandWithViewArgs {
                name: None,
                dim_names: &["x"],
                source_shape: &[8],
                view: &[ViewEntry {
                    source_axis: 0,
                    start: 1,
                    step: 2,
                    steps: 3,
                }],
                data_type: BandDataType::UInt8,
                nodata: None,
                outdb_uri: None,
                outdb_format: None,
            })
            .unwrap();
        builder
            .band_data_writer()
            .append_value(vec![0u8, 1, 2, 3, 4, 5, 6, 7]);
        builder.finish_band().unwrap();
        builder
            .start_band_nd(None, &["x"], &[3], BandDataType::UInt8, None, None, None)
            .unwrap();
        builder
            .band_data_writer()
            .append_value(vec![100u8, 101, 102]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        // Raster 1
        builder
            .start_raster_nd(&transform, &["x"], &[4], None)
            .unwrap();
        builder
            .start_band_with_view(StartBandWithViewArgs {
                name: None,
                dim_names: &["x"],
                source_shape: &[1],
                view: &[ViewEntry {
                    source_axis: 0,
                    start: 0,
                    step: 0,
                    steps: 4,
                }],
                data_type: BandDataType::UInt8,
                nodata: None,
                outdb_uri: None,
                outdb_format: None,
            })
            .unwrap();
        builder.band_data_writer().append_value(vec![42u8]);
        builder.finish_band().unwrap();
        builder
            .start_band_nd(None, &["x"], &[4], BandDataType::UInt8, None, None, None)
            .unwrap();
        builder.band_data_writer().append_value(vec![1u8, 2, 3, 4]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::new(&array);

        // Raster 0 bands: identity (3), slice (3), identity (3). The identity
        // bands are contiguous and borrow zero-copy; the step=2 slice is
        // strided so `as_contiguous` rejects it.
        let r0 = rasters.get(0).unwrap();
        assert_eq!(r0.num_bands(), 3);
        assert_eq!(r0.band(0).unwrap().shape(), &[3]);
        let b0 = r0.band(0).unwrap();
        let nd0 = b0.nd_buffer().unwrap();
        assert_eq!(nd0.as_contiguous().unwrap(), &[10u8, 20, 30]);
        assert_eq!(r0.band(1).unwrap().shape(), &[3]);
        let b1 = r0.band(1).unwrap();
        let nd1 = b1.nd_buffer().unwrap();
        assert!(!nd1.is_contiguous());
        assert!(nd1.as_contiguous().is_err());
        assert_eq!(r0.band(2).unwrap().shape(), &[3]);
        let b2 = r0.band(2).unwrap();
        let nd2 = b2.nd_buffer().unwrap();
        assert_eq!(nd2.as_contiguous().unwrap(), &[100u8, 101, 102]);

        // Raster 1 bands: broadcast (4 copies of 42), identity (4). The
        // broadcast band has a zero stride so it is non-contiguous and
        // rejected; the identity band borrows zero-copy.
        let r1 = rasters.get(1).unwrap();
        assert_eq!(r1.num_bands(), 2);
        assert_eq!(r1.band(0).unwrap().shape(), &[4]);
        let r1b0 = r1.band(0).unwrap();
        let r1nd0 = r1b0.nd_buffer().unwrap();
        assert!(!r1nd0.is_contiguous());
        assert!(r1nd0.as_contiguous().is_err());
        assert_eq!(r1.band(1).unwrap().shape(), &[4]);
        let r1b1 = r1.band(1).unwrap();
        let r1nd1 = r1b1.nd_buffer().unwrap();
        assert_eq!(r1nd1.as_contiguous().unwrap(), &[1u8, 2, 3, 4]);

        // Fast paths must honour the same offsets.
        assert_eq!(r0.band_data_type(1), Some(BandDataType::UInt8));
        assert_eq!(r1.band_data_type(0), Some(BandDataType::UInt8));
        assert_eq!(r1.band_data_type(1), Some(BandDataType::UInt8));
    }

    // null raster row, fast path

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
            .start_raster_nd(&transform, &["x"], &[3], None)
            .unwrap();
        builder
            .start_band_nd(
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
        // out of range — `band()` surfaces an out-of-range error,
        // the fast-path accessors return None.
        assert!(rasters.is_null(1));
        let r1 = rasters.get(1).unwrap();
        assert_eq!(r1.num_bands(), 0);
        assert!(r1.band(0).is_err());
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
            .start_raster_nd(&transform, &["x"], &[3], None)
            .unwrap();
        builder
            .start_band_with_view(StartBandWithViewArgs {
                name: Some("a"),
                dim_names: &["x"],
                source_shape: &[8],
                view: &[ViewEntry {
                    source_axis: 0,
                    start: 1,
                    step: 2,
                    steps: 3,
                }],
                data_type: BandDataType::UInt32,
                nodata: Some(&[0u8, 0, 0, 0]),
                outdb_uri: Some("s3://bucket/a.tif"),
                outdb_format: Some("GTiff"),
            })
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
        assert!(r.band(0).is_err());

        // Fast paths still surface the underlying columnar values —
        // they don't validate the view, by design. Locking that in.
        assert_eq!(r.band_data_type(0), Some(BandDataType::UInt32));
        assert_eq!(r.band_outdb_uri(0), Some("s3://bucket/a.tif"));
        assert_eq!(r.band_outdb_format(0), Some("GTiff"));
        assert_eq!(r.band_nodata(0), Some(&[0u8, 0, 0, 0][..]));
    }

    #[test]
    fn zero_element_indb_band_classifies_as_indb() {
        // A band with a 0-size dim (here `time = 0`) legitimately holds 0 bytes.
        // Its empty `data` column must NOT be mistaken for the OutDb sentinel:
        // a 0-element band has nothing to load, so it's InDb.
        let mut builder = RasterBuilder::new(1);
        builder
            .start_raster_2d(2, 2, 0.0, 2.0, 1.0, -1.0, 0.0, 0.0, None)
            .unwrap();
        builder
            .start_band_nd(
                Some("empty_time"),
                &["time", "y", "x"],
                &[0, 2, 2],
                BandDataType::UInt8,
                None,
                None,
                None,
            )
            .unwrap();
        builder.band_data_writer().append_value([]); // 0 bytes, legitimately
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        let arr = builder.finish().unwrap();

        let rasters = RasterStructArray::new(&arr);
        let r = rasters.get(0).unwrap();
        let band = r.band(0).unwrap();
        assert!(
            band.is_indb(),
            "a 0-element band holds 0 bytes legitimately and must be InDb"
        );
        assert_eq!(band.metadata().storage_type().unwrap(), StorageType::InDb);
    }

    #[test]
    fn test_as_contiguous_borrows_identity_view() {
        let mut builder = RasterBuilder::new(1);
        builder
            .start_raster_2d(4, 4, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, None)
            .unwrap();
        builder.start_band_2d(BandDataType::UInt8, None).unwrap();
        builder.band_data_writer().append_value([1u8; 16]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::new(&array);
        let r = rasters.get(0).unwrap();
        let band = r.band(0).unwrap();

        let ndb = band.nd_buffer().unwrap();
        // Identity-view bands are always contiguous, so as_contiguous borrows
        // the underlying bytes zero-copy rather than erroring.
        assert!(ndb.is_contiguous());
        let data = ndb.as_contiguous().unwrap();
        assert_eq!(data.len(), 16);
    }

    #[test]
    fn test_nd_buffer_strides_various_types() {
        // Each raster exercises a different shape; strict spatial-grid
        // validation forbids mixing bands of disagreeing spatial sizes within
        // one raster.
        let mut builder = RasterBuilder::new(3);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];

        // Raster 0 — UInt8: element size = 1, shape [3, 4] → strides [4, 1]
        builder
            .start_raster_nd(&transform, &["x", "y"], &[4, 3], None)
            .unwrap();
        builder
            .start_band_nd(
                None,
                &["y", "x"],
                &[3, 4],
                BandDataType::UInt8,
                None,
                None,
                None,
            )
            .unwrap();
        builder.band_data_writer().append_value(vec![0u8; 12]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        // Raster 1 — Float64: element size = 8, shape [2, 3, 5] → strides [120, 40, 8]
        builder
            .start_raster_nd(&transform, &["x", "y"], &[5, 3], None)
            .unwrap();
        builder
            .start_band_nd(
                None,
                &["z", "y", "x"],
                &[2, 3, 5],
                BandDataType::Float64,
                None,
                None,
                None,
            )
            .unwrap();
        builder
            .band_data_writer()
            .append_value(vec![0u8; 2 * 3 * 5 * 8]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        // Raster 2 — UInt16: element size = 2, shape [10] → strides [2].
        // Only has an "x" dim, so declare spatial_dims=["x"].
        builder
            .start_raster_nd(&transform, &["x"], &[10], None)
            .unwrap();
        builder
            .start_band_nd(None, &["x"], &[10], BandDataType::UInt16, None, None, None)
            .unwrap();
        builder.band_data_writer().append_value(vec![0u8; 20]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::new(&array);

        let r0 = rasters.get(0).unwrap();
        let b0 = r0.band(0).unwrap();
        assert_eq!(b0.nd_buffer().unwrap().strides, &[4, 1]); // UInt8 [3, 4]

        let r1 = rasters.get(1).unwrap();
        let b1 = r1.band(0).unwrap();
        assert_eq!(b1.nd_buffer().unwrap().strides, &[120, 40, 8]); // Float64 [2, 3, 5]

        let r2 = rasters.get(2).unwrap();
        let b2 = r2.band(0).unwrap();
        assert_eq!(b2.nd_buffer().unwrap().strides, &[2]); // UInt16 [10]
    }

    #[test]
    fn test_as_contiguous_identity_via_start_band_borrows() {
        // Canonical identity: the row's view list is null, and the read path
        // synthesises the identity view. Should still hand the underlying
        // bytes back without copying.
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster_nd(&transform, &["x", "y"], &[3, 2], None)
            .unwrap();
        builder
            .start_band_nd(
                None,
                &["y", "x"],
                &[2, 3],
                BandDataType::UInt8,
                None,
                None,
                None,
            )
            .unwrap();
        let pixels: Vec<u8> = (0..6).collect();
        builder.band_data_writer().append_value(pixels.clone());
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::new(&array);
        let r = rasters.get(0).unwrap();
        let band = r.band(0).unwrap();

        // Visible shape comes from the synthesised identity view.
        assert_eq!(band.shape(), &[2, 3]);
        assert_eq!(band.raw_source_shape(), &[2, 3]);

        let buf = band.nd_buffer().unwrap();
        assert_eq!(buf.strides, &[3, 1]);
        assert_eq!(buf.offset, 0);
        assert!(buf.is_contiguous());
        assert_eq!(buf.as_contiguous().unwrap(), pixels.as_slice());
    }

    #[test]
    fn test_as_contiguous_explicit_identity_view_borrows() {
        // Identity expressed *explicitly* through start_band_with_view must be
        // indistinguishable to consumers from the null-row identity above —
        // same visible shape, same byte strides, same zero-copy borrow.
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster_nd(&transform, &["x", "y"], &[3, 2], None)
            .unwrap();
        let view = crate::view_entries![0:2, 0:3];
        builder
            .start_band_with_view(StartBandWithViewArgs {
                name: None,
                dim_names: &["y", "x"],
                source_shape: &[2, 3],
                view: view.as_slice(),
                data_type: BandDataType::UInt8,
                nodata: None,
                outdb_uri: None,
                outdb_format: None,
            })
            .unwrap();
        let pixels: Vec<u8> = (0..6).collect();
        builder.band_data_writer().append_value(pixels.clone());
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::new(&array);
        let r = rasters.get(0).unwrap();
        let band = r.band(0).unwrap();

        assert_eq!(band.shape(), &[2, 3]);
        let buf = band.nd_buffer().unwrap();
        assert_eq!(buf.strides, &[3, 1]);
        assert_eq!(buf.offset, 0);
        assert!(buf.is_contiguous());
        assert_eq!(buf.as_contiguous().unwrap(), pixels.as_slice());
    }

    #[test]
    fn test_zero_step_broadcast_2d_is_strided_and_rejected() {
        // 2D broadcast: source shape [1, 3], view broadcasts axis 0 four
        // times so the visible region is 4×3. Each visible row must equal the
        // source's only row.
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder.start_raster_nd(&transform, &[], &[], None).unwrap();
        let view = [
            ViewEntry {
                source_axis: 0,
                start: 0,
                step: 0,
                steps: 4,
            },
            ViewEntry {
                source_axis: 1,
                start: 0,
                step: 1,
                steps: 3,
            },
        ];
        builder
            .start_band_with_view(StartBandWithViewArgs {
                name: None,
                dim_names: &["row", "col"],
                source_shape: &[1, 3],
                view: &view,
                data_type: BandDataType::UInt8,
                nodata: None,
                outdb_uri: None,
                outdb_format: None,
            })
            .unwrap();
        builder.band_data_writer().append_value(vec![10u8, 20, 30]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::new(&array);
        let r = rasters.get(0).unwrap();
        let band = r.band(0).unwrap();

        let buf = band.nd_buffer().unwrap();
        assert_eq!(buf.shape, &[4, 3]);
        // Broadcast row stride is 0; column stride is 1 byte per UInt8.
        assert_eq!(buf.strides, &[0, 1]);
        assert_eq!(buf.offset, 0);

        // A zero stride is not C-order packed, so the buffer is non-contiguous
        // and as_contiguous rejects it (repacking lives behind
        // RS_EnsureContiguous, https://github.com/apache/sedona-db/issues/899).
        assert!(!buf.is_contiguous());
        assert!(buf.as_contiguous().is_err());
    }

    #[test]
    fn test_negative_step_strided_reverse_is_rejected() {
        // 1D source [0..8] with start=6, step=-2, steps=3 picks every other
        // element walking backwards: {6, 4, 2}.
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder.start_raster_nd(&transform, &[], &[], None).unwrap();
        let view = [ViewEntry {
            source_axis: 0,
            start: 6,
            step: -2,
            steps: 3,
        }];
        builder
            .start_band_with_view(StartBandWithViewArgs {
                name: None,
                dim_names: &["x"],
                source_shape: &[8],
                view: &view,
                data_type: BandDataType::UInt8,
                nodata: None,
                outdb_uri: None,
                outdb_format: None,
            })
            .unwrap();
        builder
            .band_data_writer()
            .append_value(vec![0u8, 1, 2, 3, 4, 5, 6, 7]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::new(&array);
        let r = rasters.get(0).unwrap();
        let band = r.band(0).unwrap();

        let buf = band.nd_buffer().unwrap();
        assert_eq!(buf.shape, &[3]);
        assert_eq!(buf.strides, &[-2]);
        assert_eq!(buf.offset, 6);

        // A negative stride is not C-order packed → non-contiguous, rejected.
        assert!(!buf.is_contiguous());
        assert!(buf.as_contiguous().is_err());
    }

    #[test]
    fn test_outer_axis_slice_float32_is_contiguous() {
        // Multi-byte dtype outer-axis slice: a 2D view over Float32 that
        // takes the leading rows from offset 0 is contiguous-but-not-identity,
        // so as_contiguous borrows the source prefix zero-copy. Catches a
        // regression where contiguity assumed dtype_size == 1.
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster_nd(&transform, &["x", "y"], &[3, 2], None)
            .unwrap();
        // Slice the outer axis: take rows 0 and 1 of a 3-row source. With
        // start=0, step=1, steps=2 over an axis of size 3, the view is not
        // identity, but its byte strides are still C-order packed from
        // offset 0, so the buffer is contiguous and borrows zero-copy.
        let view = crate::view_entries![0:2, 0:3];
        builder
            .start_band_with_view(StartBandWithViewArgs {
                name: None,
                dim_names: &["y", "x"],
                source_shape: &[3, 3], // 3x3 source
                view: view.as_slice(),
                data_type: BandDataType::Float32,
                nodata: None,
                outdb_uri: None,
                outdb_format: None,
            })
            .unwrap();
        let source: Vec<f32> = (0..9).map(|i| i as f32).collect();
        let source_bytes: Vec<u8> = source.iter().flat_map(|f| f.to_le_bytes()).collect();
        builder
            .band_data_writer()
            .append_value(source_bytes.clone());
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::new(&array);
        let r = rasters.get(0).unwrap();
        let band = r.band(0).unwrap();

        // Visible shape is [2, 3]; the first 6 source floats (rows 0,1) are
        // exactly the visible pixels — i.e. the first 24 source bytes.
        let buf = band.nd_buffer().unwrap();
        assert!(buf.is_contiguous());
        assert_eq!(buf.as_contiguous().unwrap(), &source_bytes[0..24]);
    }

    #[test]
    fn test_outer_axis_slice_3d_is_contiguous() {
        // 3D source [T=3, Y=2, X=3] of UInt8. View slices T to T=1..3
        // (start=1, step=1, steps=2), keeps Y and X identity. The visible
        // region is a contiguous source sub-range (offset 6, C-order packed
        // strides), so as_contiguous borrows it zero-copy.
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster_nd(&transform, &["x", "y"], &[3, 2], None)
            .unwrap();
        let view = crate::view_entries![1:3, 0:2, 0:3];
        builder
            .start_band_with_view(StartBandWithViewArgs {
                name: None,
                dim_names: &["t", "y", "x"],
                source_shape: &[3, 2, 3],
                view: view.as_slice(),
                data_type: BandDataType::UInt8,
                nodata: None,
                outdb_uri: None,
                outdb_format: None,
            })
            .unwrap();
        let source: Vec<u8> = (0..18).collect();
        builder.band_data_writer().append_value(source.clone());
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::new(&array);
        let r = rasters.get(0).unwrap();
        let band = r.band(0).unwrap();

        // Visible region = source[6..18] (T=1 and T=2 planes).
        assert_eq!(band.shape(), &[2, 2, 3]);
        let buf = band.nd_buffer().unwrap();
        assert!(buf.is_contiguous());
        assert_eq!(buf.as_contiguous().unwrap(), &source[6..18]);
    }

    #[test]
    fn test_nd_buffer_permutation_and_slice_combined() {
        // 2D source [Y=4, X=3]. View permutes (visible order [X, Y]) and
        // slices Y from 1, step 2, steps 2. Expected:
        //   visible_shape = [3, 2]
        //   byte_strides  = [step_X * stride_X_src, step_Y * stride_Y_src]
        //                 = [1 * 1, 2 * 3] = [1, 6]
        //   byte_offset   = start_X * stride_X_src + start_Y * stride_Y_src
        //                 = 0 * 1 + 1 * 3 = 3
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder.start_raster_nd(&transform, &[], &[], None).unwrap();
        let view = [
            ViewEntry {
                source_axis: 1,
                start: 0,
                step: 1,
                steps: 3,
            }, // X
            ViewEntry {
                source_axis: 0,
                start: 1,
                step: 2,
                steps: 2,
            }, // Y
        ];
        builder
            .start_band_with_view(StartBandWithViewArgs {
                name: None,
                dim_names: &["x", "y"],
                source_shape: &[4, 3],
                view: &view,
                data_type: BandDataType::UInt8,
                nodata: None,
                outdb_uri: None,
                outdb_format: None,
            })
            .unwrap();
        builder
            .band_data_writer()
            .append_value((0u8..12).collect::<Vec<u8>>());
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::new(&array);
        let r = rasters.get(0).unwrap();
        let band = r.band(0).unwrap();
        let buf = band.nd_buffer().unwrap();
        assert_eq!(buf.shape, &[3, 2]);
        assert_eq!(buf.strides, &[1, 6]);
        assert_eq!(buf.offset, 3);

        // The permuted+strided layout (strides [1, 6]) is not C-order packed,
        // so the buffer is non-contiguous and as_contiguous rejects it.
        assert!(!buf.is_contiguous());
        assert!(buf.as_contiguous().is_err());
    }
}
