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
use datafusion_common::cast::{
    as_binary_array, as_binary_view_array, as_float64_array, as_int64_array, as_list_array,
    as_string_array, as_string_view_array, as_struct_array, as_uint32_array,
};

use crate::builder::RasterBuilder;
use crate::traits::{BandRef, NdBuffer, RasterRef};
use crate::view_entries::{ViewEntries, ViewEntry};
use sedona_schema::raster::{band_indices, band_view_indices, raster_indices, BandDataType};

/// Arrow-backed implementation of BandRef for a single band within a raster.
///
/// View-derived layout (`visible_shape`, `byte_strides`, `byte_offset`) is
/// composed once at construction from the band's stored view (identity when
/// the view row is null, otherwise the persisted `ViewEntry` list) and reused
/// by every accessor. Source-shape and dim-name slices are borrowed directly
/// from the underlying Arrow buffers.
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
    /// Per-visible-axis view, length = ndim.
    view_entries: ViewEntries,
    /// Visible shape (`[v.steps for v in view_entries]`), length = ndim.
    visible_shape: Vec<i64>,
    /// Byte strides per visible axis. May be 0 (broadcast) or negative
    /// (reverse iteration) for a non-identity view.
    byte_strides: Vec<i64>,
    /// Byte offset into `data` of the visible region's `[0,...,0]` element.
    /// Non-negative by construction — `RasterRefImpl::band` composes it from
    /// non-negative `start × source_stride` terms and rejects a negative or
    /// overflowing result before the band is built.
    byte_offset: u64,
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
        // empty-`data` sentinel. Test emptiness with `contains(&0)` rather than
        // `Π shape`: a broadcast axis can push the visible element count past
        // i64::MAX (e.g. a huge `time` axis), and the product would
        // overflow-panic in debug / wrap in release. For non-negative shapes
        // (guaranteed by `validate`) the two are equivalent. Otherwise the
        // discriminator is buffer presence.
        self.shape().contains(&0) || !self.data_array.value(self.band_row).is_empty()
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

    /// Zero-copy override: share the source row's backing `Buffer` into the
    /// builder (refcount bump) instead of copying the visible bytes. OutDb
    /// bands have an empty data column by design.
    fn append_data_into(&self, builder: &mut RasterBuilder) -> Result<(), ArrowError> {
        if self.is_indb() {
            builder.append_band_data_from(self.data_array, self.band_row)
        } else {
            builder.band_data_writer().append_value([]);
            Ok(())
        }
    }
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

    /// Read the band's stored view-entry list. Identity is encoded exclusively
    /// as a NULL row and is synthesised as the canonical identity over
    /// `source_shape`; a non-null row is decoded from the four parallel view
    /// columns. An empty (non-null, zero-length) row is malformed and is
    /// rejected downstream by [`ViewEntries::validate`].
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
        // The list offsets are authored by whoever wrote the Arrow array — for a
        // view round-tripped in from another engine over IPC/FFI (validation is
        // skipped on import), the four parallel child columns may be shorter
        // than the offsets claim, or carry a null in a field the schema declares
        // non-null. Either would panic or silently misread `.value(i)` below, so
        // validate the child arrays against the offsets before indexing.
        for (name, arr) in [
            ("source_axis", self.band_view_source_axis),
            ("start", self.band_view_start),
            ("step", self.band_view_step),
            ("steps", self.band_view_steps),
        ] {
            if v_end > arr.len() {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "band {band_row}: view '{name}' child array has {} elements but the \
                     view list addresses up to {v_end}",
                    arr.len()
                )));
            }
            if arr.null_count() > 0 && (v_start..v_end).any(|i| arr.is_null(i)) {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "band {band_row}: view '{name}' child array has a null in \
                     [{v_start}, {v_end}); view fields must be non-null"
                )));
            }
        }
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
}

/// Compose a validated view against a source shape into C-order byte strides
/// and a byte offset.
///
/// C-order source strides are dtype-scaled cumulative products of
/// `source_shape`; each visible axis's byte stride is `view.step ×
/// src_stride` and its offset contribution is `view.start × src_stride`. All
/// arithmetic is checked: even after [`ViewEntries::validate`], the cumulative
/// byte product can overflow `i64` for cosmically large shapes, and a corrupt
/// `source_shape` whose product wraps would otherwise silently pass the
/// downstream bound check. The returned `byte_offset` is non-negative by
/// construction (`start >= 0`, `src_stride > 0`); the defensive sign check
/// guards a future refactor before we cross the `i64` → `u64` boundary in
/// `band()`.
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

/// Verify that every byte the view can address lies within `buffer_len` and
/// that every stride × index product (and their accumulations) fits in i64.
///
/// **Load-bearing**: this is the *only* bound check between the view's
/// byte-stride description and the data buffer. Stride-aware consumers walk
/// the buffer with plain-arithmetic indexing and rely on this precheck having
/// proven every addressed byte is in range. Two corruption modes it catches:
///
///   1. A writer that lies about `source_shape` (Arrow column shorter than the
///      view promises).
///   2. A composed view whose stride × index product or accumulated offset
///      overflows i64 even though `validate` accepted the per-entry bounds.
///
/// Empty visible regions (any axis with `steps == 0`) address no bytes and
/// skip the check.
fn check_view_buffer_bounds(
    buffer_len: usize,
    visible_shape: &[i64],
    byte_strides: &[i64],
    byte_offset: i64,
    dtype_size: usize,
) -> Result<(), ArrowError> {
    if visible_shape.contains(&0) {
        // An empty visible region addresses no elements, but the composed
        // `byte_offset` (from a large `start` on a non-empty axis) must still
        // stay within the buffer so the `NdBuffer.offset <= buffer.len()`
        // invariant always holds.
        let buffer_len_i64 = i64::try_from(buffer_len).map_err(|_| {
            ArrowError::InvalidArgumentError(format!("buffer length {buffer_len} exceeds i64::MAX"))
        })?;
        if byte_offset > buffer_len_i64 {
            return Err(ArrowError::InvalidArgumentError(format!(
                "view byte offset {byte_offset} exceeds buffer length {buffer_len} \
                 for an empty visible region"
            )));
        }
        return Ok(());
    }
    let mut min_offset = byte_offset;
    let mut max_offset = byte_offset;
    for (k, &stride) in byte_strides.iter().enumerate() {
        // `validate` guarantees `steps >= 0`, so `visible_shape[k] - 1` is
        // in-range for any non-empty axis.
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

impl<'a> RasterRef for RasterRefImpl<'a> {
    fn num_bands(&self) -> usize {
        self.bands_list.value_length(self.raster_index) as usize
    }

    fn band(&self, index: usize) -> Result<Box<dyn BandRef + '_>, ArrowError> {
        let nbands = self.num_bands();
        if index >= nbands {
            return Err(ArrowError::InvalidArgumentError(format!(
                "Band index {index} is out of range: this raster has {nbands} bands"
            )));
        }
        let start = self.bands_list.value_offsets()[self.raster_index] as usize;
        let band_row = start + index;

        // Read source shape slice.
        let ss_start = self.band_source_shape_list.value_offsets()[band_row] as usize;
        let ss_end = self.band_source_shape_list.value_offsets()[band_row + 1] as usize;
        let source_shape: &[i64] = &self.band_source_shape_values.values()[ss_start..ss_end];

        // Reject 0-D bands at the read boundary. Schema doesn't forbid them
        // outright but every consumer assumes ndim >= 1.
        if source_shape.is_empty() {
            return Err(ArrowError::ExternalError(Box::new(
                sedona_common::sedona_internal_datafusion_err!(
                    "band {band_row} has empty source_shape; ndim must be >= 1"
                ),
            )));
        }

        // Resolve data type up front; an unknown discriminant is a
        // schema-corruption bug, not user data, so failing the band loudly
        // here is appropriate.
        let data_type_value = self.band_datatype_array.value(band_row);
        let data_type = BandDataType::try_from_u32(data_type_value).ok_or_else(|| {
            ArrowError::ExternalError(Box::new(sedona_common::sedona_internal_datafusion_err!(
                "band {band_row} has unknown data_type discriminant {data_type_value}"
            )))
        })?;

        // Read the band's view (identity when the row is null, otherwise the
        // persisted ViewEntry list) and validate it against the source shape —
        // a malformed or corrupt view surfaces loudly rather than mislocating
        // bytes.
        let view_entries = self.read_band_view_entries(band_row, source_shape)?;
        view_entries.validate(source_shape).map_err(|e| {
            ArrowError::ExternalError(Box::new(sedona_common::sedona_internal_datafusion_err!(
                "band {band_row} has malformed view: {e}"
            )))
        })?;

        // The visible shape is view-derived and needed for every band. The
        // InDb C-order byte-stride layout, by contrast, is only meaningful for
        // InDb bands: an OutDb band (empty `data`, non-empty visible region)
        // never dereferences its strides — `nd_buffer()` errors for it — so we
        // skip composing them. That also lets an OutDb band whose described
        // `source_shape` has a stride product exceeding i64 read its metadata
        // without tripping the InDb source-stride overflow guard.
        let visible_shape = view_entries.visible_shape();
        let data_bytes = self.band_data_array.value(band_row);

        // Mirror `BandRef::is_indb`: an empty visible region or non-empty data
        // buffer is InDb; empty data with a non-empty visible region is OutDb.
        let is_indb = visible_shape.contains(&0) || !data_bytes.is_empty();

        let (byte_strides, byte_offset) = if is_indb {
            // Compose the view onto the source's natural C-order byte strides to
            // get per-axis byte strides (0 for broadcast, negative for reverse)
            // and the byte offset of element [0,...,0].
            let (byte_strides, byte_offset_i64) =
                compose_byte_strides(band_row, source_shape, &view_entries, data_type.byte_size())?;

            // Verify the data column is long enough to cover every byte the view
            // can address. The view-machinery validation above doesn't know the
            // actual `data` BinaryView length — a writer that lies about
            // source_shape vs the bytes written would otherwise slip through and
            // panic later when a consumer walks the strided buffer. Skipped when
            // there are no bytes (an empty visible region).
            if !data_bytes.is_empty() {
                check_view_buffer_bounds(
                    data_bytes.len(),
                    &visible_shape,
                    &byte_strides,
                    byte_offset_i64,
                    data_type.byte_size(),
                )
                .map_err(|e| {
                    ArrowError::ExternalError(Box::new(
                        sedona_common::sedona_internal_datafusion_err!(
                            "band {band_row}: view-buffer bounds check failed: {e}"
                        ),
                    ))
                })?;
            }

            // `compose_byte_strides` guarantees a non-negative offset; cross into
            // `u64` for storage with a checked conversion that upholds that at
            // the boundary.
            let byte_offset = u64::try_from(byte_offset_i64).map_err(|_| {
                ArrowError::ExternalError(Box::new(sedona_common::sedona_internal_datafusion_err!(
                    "band {band_row}: composed byte_offset {byte_offset_i64} is negative"
                )))
            })?;
            (byte_strides, byte_offset)
        } else {
            // OutDb: the byte strides/offset are never read (`nd_buffer()`
            // errors for OutDb bands), so leave them zeroed rather than compute
            // an InDb layout for bytes that live elsewhere.
            (vec![0i64; view_entries.len()], 0u64)
        };

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
            view_entries,
            visible_shape,
            byte_strides,
            byte_offset,
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
    ///
    /// Returns an error if the array doesn't have the expected raster schema.
    #[inline]
    pub fn try_new(raster_array: &'a StructArray) -> Result<Self, ArrowError> {
        if raster_array.fields().len() != raster_indices::FIELD_COUNT {
            return Err(ArrowError::SchemaError(
                "Unexpected column count for raster array".to_string(),
            ));
        }

        // Top-level fields
        let crs_array = as_string_view_array(raster_array.column(raster_indices::CRS))?;
        let transform_list = as_list_array(raster_array.column(raster_indices::TRANSFORM))?;
        let transform_values = as_float64_array(transform_list.values())?;
        let spatial_dims_list = as_list_array(raster_array.column(raster_indices::SPATIAL_DIMS))?;
        let spatial_dims_values = as_string_view_array(spatial_dims_list.values())?;
        let spatial_shape_list = as_list_array(raster_array.column(raster_indices::SPATIAL_SHAPE))?;
        let spatial_shape_values = as_int64_array(spatial_shape_list.values())?;

        // Bands list and nested struct
        let bands_list = as_list_array(raster_array.column(raster_indices::BANDS))?;
        let bands_struct = as_struct_array(bands_list.values())?;

        if bands_struct.fields().len() != band_indices::FIELD_COUNT {
            return Err(ArrowError::SchemaError(
                "Unexpected column count for band array".to_string(),
            ));
        }

        // Band-level fields
        let band_name_array = as_string_array(bands_struct.column(band_indices::NAME))?;
        let band_dim_names_list = as_list_array(bands_struct.column(band_indices::DIM_NAMES))?;
        let band_dim_names_values = as_string_array(band_dim_names_list.values())?;
        let band_source_shape_list =
            as_list_array(bands_struct.column(band_indices::SOURCE_SHAPE))?;
        let band_source_shape_values = as_int64_array(band_source_shape_list.values())?;
        let band_datatype_array = as_uint32_array(bands_struct.column(band_indices::DATA_TYPE))?;
        let band_nodata_array = as_binary_array(bands_struct.column(band_indices::NODATA))?;
        let band_view_list = as_list_array(bands_struct.column(band_indices::VIEW))?;
        let band_view_struct = as_struct_array(band_view_list.values())?;
        let band_view_source_axis =
            as_int64_array(band_view_struct.column(band_view_indices::SOURCE_AXIS))?;
        let band_view_start = as_int64_array(band_view_struct.column(band_view_indices::START))?;
        let band_view_step = as_int64_array(band_view_struct.column(band_view_indices::STEP))?;
        let band_view_steps = as_int64_array(band_view_struct.column(band_view_indices::STEPS))?;
        let band_outdb_uri_array = as_string_array(bands_struct.column(band_indices::OUTDB_URI))?;
        let band_outdb_format_array =
            as_string_view_array(bands_struct.column(band_indices::OUTDB_FORMAT))?;
        let band_data_array = as_binary_view_array(bands_struct.column(band_indices::DATA))?;

        Ok(Self {
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
        })
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
    use crate::builder::{RasterBuilder, StartBandArgs};
    use crate::traits::BandOverrides;
    use arrow_array::{ArrayRef, ListArray, StructArray, UInt32Array};
    use arrow_buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
    use arrow_schema::{DataType, Field, Fields};
    use sedona_schema::raster::{band_indices, raster_indices, BandDataType, RasterSchema};
    use sedona_testing::rasters::generate_test_rasters;
    use std::sync::Arc;

    #[test]
    fn copy_into_shares_buffer_zero_copy_and_overrides() {
        // 16-byte InDb band (> inline threshold, so block-backed and shareable).
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        let mut ib = RasterBuilder::new(1);
        ib.start_raster_nd(&transform, &["x"], &[16], None).unwrap();
        ib.start_band(StartBandArgs {
            name: Some("orig"),
            ..StartBandArgs::new(&["x"], &[16], BandDataType::UInt8)
        })
        .unwrap();
        ib.band_data_writer()
            .append_value((0u8..16).collect::<Vec<u8>>());
        ib.finish_band().unwrap();
        ib.finish_raster().unwrap();
        let input_array = ib.finish().unwrap();
        let input_rasters = RasterStructArray::try_new(&input_array).unwrap();
        let input_raster = input_rasters.get(0).unwrap();
        let input_band = input_raster.band(0).unwrap();
        let input_ptr = input_band.nd_buffer().unwrap().buffer.as_ptr();

        // copy_into with a name override; everything else inherited.
        let mut ob = RasterBuilder::new(1);
        ob.start_raster_nd(&transform, &["x"], &[16], None).unwrap();
        input_band
            .copy_into(
                &mut ob,
                BandOverrides {
                    name: Some("derived"),
                    ..Default::default()
                },
            )
            .unwrap();
        ob.finish_band().unwrap();
        ob.finish_raster().unwrap();
        let out_array = ob.finish().unwrap();
        let out_rasters = RasterStructArray::try_new(&out_array).unwrap();
        let out_raster = out_rasters.get(0).unwrap();
        let out_band = out_raster.band(0).unwrap();

        // Zero-copy: the derived band references the same backing bytes.
        assert_eq!(
            input_ptr,
            out_band.nd_buffer().unwrap().buffer.as_ptr(),
            "copy_into must share the source buffer, not copy it"
        );
        assert_eq!(
            out_band.nd_buffer().unwrap().as_contiguous().unwrap(),
            (0u8..16).collect::<Vec<u8>>().as_slice()
        );
        // Name overridden; dim names + data type inherited from the source.
        assert_eq!(out_raster.band_name(0), Some("derived"));
        assert_eq!(out_band.dim_names(), vec!["x"]);
        assert_eq!(out_band.data_type(), BandDataType::UInt8);
    }

    #[test]
    fn copy_into_with_identity_override_view_succeeds() {
        // An explicit identity override composes back to the identity, so it is
        // accepted and behaves exactly like the inherited (None) case — this
        // exercises the new `BandOverrides::view` path end to end.
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        let mut ib = RasterBuilder::new(1);
        ib.start_raster_nd(&transform, &["x"], &[4], None).unwrap();
        ib.start_band(StartBandArgs {
            name: Some("orig"),
            ..StartBandArgs::new(&["x"], &[4], BandDataType::UInt8)
        })
        .unwrap();
        ib.band_data_writer().append_value(vec![1u8, 2, 3, 4]);
        ib.finish_band().unwrap();
        ib.finish_raster().unwrap();
        let in_array = ib.finish().unwrap();
        let in_rasters = RasterStructArray::try_new(&in_array).unwrap();
        let in_raster = in_rasters.get(0).unwrap();
        let in_band = in_raster.band(0).unwrap();

        let identity = [ViewEntry {
            source_axis: 0,
            start: 0,
            step: 1,
            steps: 4,
        }];
        let mut ob = RasterBuilder::new(1);
        ob.start_raster_nd(&transform, &["x"], &[4], None).unwrap();
        in_band
            .copy_into(
                &mut ob,
                BandOverrides {
                    view: Some(&identity),
                    ..Default::default()
                },
            )
            .unwrap();
        ob.finish_band().unwrap();
        ob.finish_raster().unwrap();
        let out_array = ob.finish().unwrap();
        let out_rasters = RasterStructArray::try_new(&out_array).unwrap();
        let out_raster = out_rasters.get(0).unwrap();
        let out_band = out_raster.band(0).unwrap();
        assert_eq!(
            out_band.nd_buffer().unwrap().as_contiguous().unwrap(),
            &[1u8, 2, 3, 4]
        );
    }

    #[test]
    fn test_array_basic_functionality() {
        // Create a simple raster for testing using the correct API
        let mut builder = RasterBuilder::new(10); // capacity

        let epsg4326 = "EPSG:4326";

        builder
            .start_raster_2d(10, 10, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, Some(epsg4326))
            .unwrap();

        // Add a single band with some test data using the correct API
        builder
            .start_band_2d(BandDataType::UInt8, Some(&[255u8]))
            .unwrap();
        let test_data = vec![1u8; 100]; // 10x10 raster with value 1
        builder.band_data_writer().append_value(&test_data);
        builder.finish_band().unwrap();
        let result = builder.finish_raster();
        assert!(result.is_ok());

        let raster_array = builder.finish().unwrap();

        // Test the array
        let rasters = RasterStructArray::try_new(&raster_array).unwrap();

        assert_eq!(rasters.len(), 1);
        assert!(!rasters.is_empty());

        let raster = rasters.get(0).unwrap();

        assert_eq!(raster.width().unwrap(), 10);
        assert_eq!(raster.height().unwrap(), 10);
        assert_eq!(raster.transform()[1], 1.0);
        assert_eq!(raster.transform()[5], -1.0);

        assert_eq!(raster.num_bands(), 1);
        assert_ne!(raster.num_bands(), 0);

        // Bands are 0-based.
        let band = raster.band(0).unwrap();
        assert_eq!(
            band.nd_buffer().unwrap().as_contiguous().unwrap().len(),
            100
        );
        assert_eq!(band.nd_buffer().unwrap().as_contiguous().unwrap()[0], 1u8);

        assert!(band.is_indb());
        assert_eq!(band.data_type(), BandDataType::UInt8);

        let crs = raster.crs().unwrap();
        assert_eq!(crs, epsg4326);

        // Test array over bands
        let band_iter: Vec<_> = (0..raster.num_bands()).map(|i| raster.band(i)).collect();
        assert_eq!(band_iter.len(), 1);
    }

    #[test]
    fn test_multi_band_array() {
        let mut builder = RasterBuilder::new(3);

        builder
            .start_raster_2d(5, 5, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, None)
            .unwrap();

        // Add three bands using the correct API
        for band_idx in 0..3 {
            builder
                .start_band_2d(BandDataType::UInt8, Some(&[255u8]))
                .unwrap();
            let test_data = vec![band_idx as u8; 25]; // 5x5 raster
            builder.band_data_writer().append_value(&test_data);
            builder.finish_band().unwrap();
        }

        let result = builder.finish_raster();
        assert!(result.is_ok());

        let raster_array = builder.finish().unwrap();

        let rasters = RasterStructArray::try_new(&raster_array).unwrap();
        let raster = rasters.get(0).unwrap();

        assert_eq!(raster.num_bands(), 3);

        // Test each band has different data (bands are 0-based).
        for i in 0..3 {
            let band = raster.band(i).unwrap();
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
        let band_values: Vec<u8> = (0..raster.num_bands())
            .map(|i| raster.band(i))
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
        let rasters = RasterStructArray::try_new(&raster_array).unwrap();
        assert_eq!(rasters.len(), 2);
        assert!(!rasters.is_null(0));
        assert!(rasters.is_null(1));
    }

    /// Build a single-raster, single-band raster StructArray with the
    /// canonical identity view. Used as the baseline input to the surgery
    /// helpers below; callers replace one band-level column to simulate
    /// schema corruption on non-view fields.
    fn build_identity_raster() -> StructArray {
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster_nd(&transform, &["x"], &[3], None)
            .unwrap();
        builder
            .start_band(StartBandArgs::new(&["x"], &[3], BandDataType::UInt8))
            .unwrap();
        builder.band_data_writer().append_value(vec![0u8, 1, 2]);
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

    // bad data_type discriminant

    #[test]
    fn band_and_band_data_type_surface_corruption_for_unknown_discriminant() {
        let array = build_identity_raster();
        let bad_dtype: ArrayRef = Arc::new(UInt32Array::from(vec![0xFFu32]));
        let mutated = replace_band_column(&array, band_indices::DATA_TYPE, bad_dtype);
        let rasters = RasterStructArray::try_new(&mutated).unwrap();
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
        let array = build_identity_raster();
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
        let rasters = RasterStructArray::try_new(&mutated).unwrap();
        let err = rasters.get(0).unwrap().band(0).err().unwrap();
        assert!(err.to_string().contains("SedonaDB internal error"));
        assert!(err.to_string().contains("empty source_shape"));
    }

    // ---- Read path: non-identity view decoding + corruption rejection ----

    /// Build a single-raster, single-band StructArray carrying an explicit,
    /// non-identity view (`start=1, step=2, steps=3` over an 8-byte source).
    /// The corruption tests replace one band-level column of this baseline.
    fn build_explicit_view_raster() -> StructArray {
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster_nd(&transform, &["x"], &[3], None)
            .unwrap();
        builder
            .start_band(StartBandArgs {
                view: Some(&[ViewEntry {
                    source_axis: 0,
                    start: 1,
                    step: 2,
                    steps: 3,
                }]),
                ..StartBandArgs::new(&["x"], &[8], BandDataType::UInt8)
            })
            .unwrap();
        builder
            .band_data_writer()
            .append_value(vec![0u8, 1, 2, 3, 4, 5, 6, 7]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        builder.finish().unwrap()
    }

    /// Rebuild the band view list from hand-rolled entries. `entries[i]`
    /// supplies all four `(source_axis, start, step, steps)` values for band
    /// row `i`; `nulls` controls per-row validity (`None` → every row non-null).
    fn make_band_view_list(
        entries: Vec<Vec<(i64, i64, i64, i64)>>,
        nulls: Option<Vec<bool>>,
    ) -> ArrayRef {
        let mut offsets: Vec<i32> = vec![0];
        let (mut sa, mut start, mut step, mut steps) = (vec![], vec![], vec![], vec![]);
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
                Arc::new(Int64Array::from(sa)) as ArrayRef,
                Arc::new(Int64Array::from(start)) as ArrayRef,
                Arc::new(Int64Array::from(step)) as ArrayRef,
                Arc::new(Int64Array::from(steps)) as ArrayRef,
            ],
            None,
        );
        let DataType::List(view_field) = RasterSchema::view_type() else {
            unreachable!()
        };
        Arc::new(ListArray::new(
            view_field,
            OffsetBuffer::new(ScalarBuffer::from(offsets)),
            Arc::new(view_struct),
            nulls.map(NullBuffer::from),
        ))
    }

    /// Build a band source_shape list with hand-rolled i64 entries so tests can
    /// inject values the builder's writer-side checks would refuse.
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
    fn band_decodes_explicit_view_end_to_end() {
        // The baseline non-identity fixture must decode to the right visible
        // shape, strides, and offset — the read path's happy case for a
        // persisted view read out of the Arrow columns.
        let array = build_explicit_view_raster();
        let rasters = RasterStructArray::try_new(&array).unwrap();
        let r = rasters.get(0).unwrap();
        let band = r.band(0).unwrap();
        assert_eq!(band.shape(), &[3]);
        assert_eq!(band.raw_source_shape(), &[8]);
        let buf = band.nd_buffer().unwrap();
        assert_eq!(buf.strides, &[2]);
        assert_eq!(buf.offset, 1);
        assert!(!buf.is_contiguous());
    }

    #[test]
    fn band_errors_when_view_length_mismatches_source_shape() {
        // source_shape has 1 dim but the view encodes 2 entries.
        let array = build_explicit_view_raster();
        let bad_view = make_band_view_list(vec![vec![(0, 0, 1, 3), (0, 0, 1, 3)]], None);
        let mutated = replace_band_column(&array, band_indices::VIEW, bad_view);
        let rasters = RasterStructArray::try_new(&mutated).unwrap();
        let err = rasters.get(0).unwrap().band(0).err().unwrap();
        assert!(err.to_string().contains("malformed view"), "got: {err}");
    }

    #[test]
    fn band_rejects_empty_non_null_view_row() {
        // Identity is encoded exclusively as a NULL row; a non-null zero-length
        // list is malformed and must error rather than fall back to identity.
        let array = build_explicit_view_raster();
        let empty_non_null_view = make_band_view_list(vec![vec![]], Some(vec![true]));
        let mutated = replace_band_column(&array, band_indices::VIEW, empty_non_null_view);
        let rasters = RasterStructArray::try_new(&mutated).unwrap();
        let err = rasters.get(0).unwrap().band(0).err().unwrap();
        assert!(err.to_string().contains("view length"), "got: {err}");
    }

    #[test]
    fn band_errors_when_data_column_shorter_than_view() {
        // Inflate source_shape to [16] and the view to steps=10 along it: the
        // addressed byte range (0..10) jumps past the actual 8-byte data
        // column, so the InDb bounds precheck must fire.
        let array = build_explicit_view_raster();
        let new_source_shape = make_band_source_shape_list(vec![vec![16i64]]);
        let mutated_ss = replace_band_column(&array, band_indices::SOURCE_SHAPE, new_source_shape);
        let new_view = make_band_view_list(vec![vec![(0, 0, 1, 10)]], None);
        let mutated = replace_band_column(&mutated_ss, band_indices::VIEW, new_view);
        let rasters = RasterStructArray::try_new(&mutated).unwrap();
        let err = rasters.get(0).unwrap().band(0).err().unwrap();
        assert!(err.to_string().contains("SedonaDB internal error"));
        assert!(err.to_string().contains("view-buffer bounds check failed"));
    }

    #[test]
    fn band_errors_when_source_stride_product_overflows() {
        // dtype_size × Π source_shape[j>k] must not silently wrap. A 3-D source
        // shape of [1, 1<<32, 1<<32] makes (1<<32) × (1<<32) = 1<<64 overflow
        // i64 during the source-stride build.
        let array = build_explicit_view_raster();
        let new_source_shape =
            make_band_source_shape_list(vec![vec![1i64, 1i64 << 32, 1i64 << 32]]);
        let mutated_ss = replace_band_column(&array, band_indices::SOURCE_SHAPE, new_source_shape);
        // steps=0 on the giant axes keeps validate's start/last checks trivial.
        let new_view =
            make_band_view_list(vec![vec![(0, 0, 1, 1), (1, 0, 1, 0), (2, 0, 1, 0)]], None);
        let mutated = replace_band_column(&mutated_ss, band_indices::VIEW, new_view);
        let rasters = RasterStructArray::try_new(&mutated).unwrap();
        assert!(rasters.get(0).unwrap().band(0).is_err());
    }

    #[test]
    fn fast_paths_return_columnar_values_when_band_view_is_corrupt() {
        // band(i) validates the view and errors on a malformed one; the
        // columnar fast paths read their fields directly without consulting the
        // view. Pin the contract so a future reader doesn't accidentally couple
        // them.
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster_nd(&transform, &["x"], &[3], None)
            .unwrap();
        builder
            .start_band(StartBandArgs {
                name: Some("a"),
                view: Some(&[ViewEntry {
                    source_axis: 0,
                    start: 1,
                    step: 2,
                    steps: 3,
                }]),
                nodata: Some(&[0u8, 0, 0, 0]),
                outdb_uri: Some("s3://bucket/a.tif"),
                outdb_format: Some("GTiff"),
                ..StartBandArgs::new(&["x"], &[8], BandDataType::UInt32)
            })
            .unwrap();
        builder.band_data_writer().append_value(vec![0u8; 32]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        let array = builder.finish().unwrap();

        // Corrupt the view with a negative steps value.
        let bad_view = make_band_view_list(vec![vec![(0, 0, 1, -1)]], None);
        let mutated = replace_band_column(&array, band_indices::VIEW, bad_view);
        let rasters = RasterStructArray::try_new(&mutated).unwrap();
        let r = rasters.get(0).unwrap();

        assert!(r.band(0).is_err());
        assert_eq!(r.band_data_type(0), Some(BandDataType::UInt32));
        assert_eq!(r.band_outdb_uri(0), Some("s3://bucket/a.tif"));
        assert_eq!(r.band_outdb_format(0), Some("GTiff"));
        assert_eq!(r.band_nodata(0), Some(&[0u8, 0, 0, 0][..]));
    }

    // direct fast-path tests

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
            .start_band(StartBandArgs {
                name: Some("a"),
                nodata: Some(&[0xFFu8, 0xFE]),
                outdb_uri: Some("s3://bucket/a.tif"),
                outdb_format: Some("GTiff"),
                ..StartBandArgs::new(&["y", "x"], &[2, 3], BandDataType::UInt16)
            })
            .unwrap();
        builder.band_data_writer().append_value(vec![0u8; 12]);
        builder.finish_band().unwrap();
        builder
            .start_band(StartBandArgs {
                name: Some("b"),
                ..StartBandArgs::new(&["y", "x"], &[2, 3], BandDataType::Float32)
            })
            .unwrap();
        builder.band_data_writer().append_value(vec![0u8; 24]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::try_new(&array).unwrap();
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

        // num_bands / band(0-based) / iteration. Exercise via the concrete type
        // and via a `&dyn RasterRef` to confirm both dispatch paths work.
        assert_eq!(r.num_bands(), 2);
        assert_ne!(r.num_bands(), 0);
        assert_eq!(r.band(0).unwrap().data_type(), BandDataType::UInt16);
        assert_eq!(r.band(1).unwrap().data_type(), BandDataType::Float32);
        assert!(r.band(2).is_err()); // out of range
        assert_eq!((0..r.num_bands()).filter_map(|i| r.band(i).ok()).count(), 2);
        let dyn_r: &dyn RasterRef = &r;
        assert_eq!(dyn_r.num_bands(), 2);

        // Raster-level geometry via the direct accessors.
        assert_eq!(r.width().unwrap(), 3);
        assert_eq!(r.height().unwrap(), 2);
        assert_eq!(r.transform()[0], 0.0);
        assert_eq!(r.transform()[1], 1.0);
        // Band 0 has bytes, so it reports InDb even though the row carries an
        // outdb_uri hint.
        assert!(r.band(0).unwrap().is_indb());
    }

    // multi-band, multi-raster identity

    #[test]
    fn multi_raster_identity_views() {
        // Two rasters with multiple identity bands each. Exercises the
        // `bands_list.value_offsets()` routing for every per-band lookup —
        // a naive reader that forgets to add the per-raster offset would
        // hand back data from the wrong band.
        let mut builder = RasterBuilder::new(2);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];

        // Raster 0: three identity bands.
        builder
            .start_raster_nd(&transform, &["x"], &[3], None)
            .unwrap();
        builder
            .start_band(StartBandArgs::new(&["x"], &[3], BandDataType::UInt8))
            .unwrap();
        builder.band_data_writer().append_value(vec![10u8, 20, 30]);
        builder.finish_band().unwrap();
        builder
            .start_band(StartBandArgs::new(&["x"], &[3], BandDataType::UInt8))
            .unwrap();
        builder.band_data_writer().append_value(vec![40u8, 50, 60]);
        builder.finish_band().unwrap();
        builder
            .start_band(StartBandArgs::new(&["x"], &[3], BandDataType::UInt8))
            .unwrap();
        builder
            .band_data_writer()
            .append_value(vec![100u8, 101, 102]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        // Raster 1: two identity bands of a different shape.
        builder
            .start_raster_nd(&transform, &["x"], &[4], None)
            .unwrap();
        builder
            .start_band(StartBandArgs::new(&["x"], &[4], BandDataType::UInt8))
            .unwrap();
        builder
            .band_data_writer()
            .append_value(vec![42u8, 43, 44, 45]);
        builder.finish_band().unwrap();
        builder
            .start_band(StartBandArgs::new(&["x"], &[4], BandDataType::UInt8))
            .unwrap();
        builder.band_data_writer().append_value(vec![1u8, 2, 3, 4]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::try_new(&array).unwrap();

        let r0 = rasters.get(0).unwrap();
        assert_eq!(r0.num_bands(), 3);
        assert_eq!(r0.band(0).unwrap().shape(), &[3]);
        assert_eq!(
            r0.band(0)
                .unwrap()
                .nd_buffer()
                .unwrap()
                .as_contiguous()
                .unwrap(),
            &[10u8, 20, 30]
        );
        assert_eq!(r0.band(1).unwrap().shape(), &[3]);
        assert_eq!(
            r0.band(1)
                .unwrap()
                .nd_buffer()
                .unwrap()
                .as_contiguous()
                .unwrap(),
            &[40u8, 50, 60]
        );
        assert_eq!(r0.band(2).unwrap().shape(), &[3]);
        assert_eq!(
            r0.band(2)
                .unwrap()
                .nd_buffer()
                .unwrap()
                .as_contiguous()
                .unwrap(),
            &[100u8, 101, 102]
        );

        let r1 = rasters.get(1).unwrap();
        assert_eq!(r1.num_bands(), 2);
        assert_eq!(r1.band(0).unwrap().shape(), &[4]);
        assert_eq!(
            r1.band(0)
                .unwrap()
                .nd_buffer()
                .unwrap()
                .as_contiguous()
                .unwrap(),
            &[42u8, 43, 44, 45]
        );
        assert_eq!(r1.band(1).unwrap().shape(), &[4]);
        assert_eq!(
            r1.band(1)
                .unwrap()
                .nd_buffer()
                .unwrap()
                .as_contiguous()
                .unwrap(),
            &[1u8, 2, 3, 4]
        );

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
            .start_band(StartBandArgs {
                name: Some("a"),
                nodata: Some(&[0xFFu8, 0xFE]),
                outdb_uri: Some("s3://bucket/a.tif"),
                outdb_format: Some("GTiff"),
                ..StartBandArgs::new(&["x"], &[3], BandDataType::UInt16)
            })
            .unwrap();
        builder.band_data_writer().append_value(vec![0u8; 6]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        builder.append_null().unwrap();
        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::try_new(&array).unwrap();

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
            .start_band(StartBandArgs {
                name: Some("empty_time"),
                ..StartBandArgs::new(&["time", "y", "x"], &[0, 2, 2], BandDataType::UInt8)
            })
            .unwrap();
        builder.band_data_writer().append_value([]); // 0 bytes, legitimately
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        let arr = builder.finish().unwrap();

        let rasters = RasterStructArray::try_new(&arr).unwrap();
        let r = rasters.get(0).unwrap();
        let band = r.band(0).unwrap();
        assert!(
            band.is_indb(),
            "a 0-element band holds 0 bytes legitimately and must be InDb"
        );
    }

    #[test]
    fn is_indb_does_not_overflow_on_broadcast_axis_exceeding_i64_max() {
        // A broadcast axis with huge `steps` can push the visible element count
        // past i64::MAX. `is_indb()` must classify by buffer presence without
        // computing `Π shape` (which would overflow-panic in debug / wrap in
        // release), and `nd_buffer()` must not panic either.
        //
        // Visible shape = [4, 2^62] → product 2^64 overflows i64. The giant
        // axis is a broadcast (step=0) over a size-1 source axis, so it
        // addresses one byte per position and the 4-byte buffer suffices.
        let big: i64 = 1 << 62;
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster_nd(&transform, &["y", "t"], &[4, big], None)
            .unwrap();
        builder
            .start_band(StartBandArgs {
                name: Some("broadcast_time"),
                view: Some(&[
                    ViewEntry {
                        source_axis: 0,
                        start: 0,
                        step: 1,
                        steps: 4,
                    },
                    ViewEntry {
                        source_axis: 1,
                        start: 0,
                        step: 0,
                        steps: big,
                    },
                ]),
                ..StartBandArgs::new(&["y", "t"], &[4, 1], BandDataType::UInt8)
            })
            .unwrap();
        builder.band_data_writer().append_value(vec![0u8, 1, 2, 3]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        let array = builder.finish().unwrap();

        let rasters = RasterStructArray::try_new(&array).unwrap();
        let raster = rasters.get(0).unwrap();
        let band = raster.band(0).unwrap();
        assert_eq!(band.shape(), &[4, big]);
        assert!(
            band.is_indb(),
            "a band with bytes is InDb and must not overflow the product"
        );
        assert!(
            band.nd_buffer().is_ok(),
            "nd_buffer() must not panic on a broadcast axis"
        );
    }

    #[test]
    fn band_errors_when_view_child_arrays_shorter_than_offsets() {
        // Round-tripping the view columns in from another engine over IPC/FFI
        // (validation skipped on import) can yield a list whose offsets claim
        // more entries than the child struct actually holds. Reading it must
        // surface an error, not panic in `.value(i)`.
        use arrow_data::ArrayData;
        let array = build_explicit_view_raster();

        // A valid length-1 view struct, but list offsets [0, 3] claiming three
        // entries. Built unchecked because the safe constructors reject it.
        let valid = make_band_view_list(vec![vec![(0, 0, 1, 3)]], None);
        let valid_list = valid.as_any().downcast_ref::<ListArray>().unwrap();
        let struct_values = valid_list.values().to_data();
        let DataType::List(view_field) = RasterSchema::view_type() else {
            unreachable!()
        };
        let over_offsets = arrow_buffer::Buffer::from_slice_ref([0i32, 3i32]);
        // SAFETY: deliberately building an invalid array (offsets over-run the
        // child) to exercise the read-path guard; nothing dereferences it
        // beyond the guarded accessor under test.
        let list_data = unsafe {
            ArrayData::builder(DataType::List(view_field))
                .len(1)
                .add_buffer(over_offsets)
                .add_child_data(struct_values)
                .build_unchecked()
        };
        let bad_view: ArrayRef = Arc::new(ListArray::from(list_data));
        let mutated = replace_band_column(&array, band_indices::VIEW, bad_view);
        let rasters = RasterStructArray::try_new(&mutated).unwrap();
        let err = rasters.get(0).unwrap().band(0).err().unwrap();
        assert!(
            err.to_string().contains("child array") && err.to_string().contains("addresses up to"),
            "got: {err}"
        );
    }

    #[test]
    fn outdb_band_with_large_source_shape_reads_metadata() {
        // An OutDb band never dereferences InDb byte strides, so a described
        // source_shape whose C-order stride product overflows i64 must not
        // block reading the band's metadata. Before the OutDb read fast-path,
        // band() always composed strides and tripped the source-stride overflow
        // guard even though the bytes live elsewhere.
        //
        // source_shape [1, 2^32, 2^32]: the C-order stride of axis 0 is
        // 2^32 × 2^32 = 2^64, which overflows i64.
        let big: i64 = 1 << 32;
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster_nd(&transform, &["y", "x"], &[big, big], None)
            .unwrap();
        builder
            .start_band(StartBandArgs {
                name: Some("external"),
                nodata: Some(&[7u8]),
                outdb_uri: Some("s3://bucket/huge.tif#band=1"),
                outdb_format: Some("geotiff"),
                ..StartBandArgs::new(&["z", "y", "x"], &[1, big, big], BandDataType::UInt8)
            })
            .unwrap();
        builder.band_data_writer().append_value([]); // empty → OutDb
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        let array = builder.finish().unwrap();

        let rasters = RasterStructArray::try_new(&array).unwrap();
        let raster = rasters.get(0).unwrap();
        let band = raster.band(0).unwrap();
        assert!(
            !band.is_indb(),
            "empty data + non-empty visible region → OutDb"
        );
        assert_eq!(band.shape(), &[1, big, big]);
        assert_eq!(band.nodata(), Some(&[7u8][..]));
        assert_eq!(band.outdb_uri(), Some("s3://bucket/huge.tif#band=1"));
    }

    #[test]
    fn band_errors_when_empty_region_offset_escapes_buffer() {
        // An empty visible region (steps=0) addresses no elements, but a large
        // `start` still composes a byte_offset. That offset must stay within
        // the data buffer so the NdBuffer.offset invariant holds — a view whose
        // offset runs past the buffer end must error even though it's empty.
        let array = build_explicit_view_raster(); // source_shape [8], 8 data bytes
                                                  // start=100 with steps=0: validate skips the start bound for empty axes,
                                                  // so this composes byte_offset=100 over the 8-byte buffer.
        let escaping_view = make_band_view_list(vec![vec![(0, 100, 1, 0)]], None);
        let mutated = replace_band_column(&array, band_indices::VIEW, escaping_view);
        let rasters = RasterStructArray::try_new(&mutated).unwrap();
        let err = rasters.get(0).unwrap().band(0).err().unwrap();
        assert!(
            err.to_string().contains("view-buffer bounds check failed")
                && err.to_string().contains("exceeds buffer length"),
            "got: {err}"
        );
    }
}
