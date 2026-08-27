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

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{
    builder::{
        ArrayBuilder, BinaryBuilder, BinaryViewBuilder, Int64Builder, StringBuilder,
        StringViewBuilder, UInt32Builder,
    },
    ArrayRef, BinaryViewArray, ListArray, StructArray,
};
use arrow_buffer::{Buffer, NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::DataType;

use sedona_schema::raster::RasterSchema;

use crate::builder::StartBandArgs;
use crate::error::RasterError;

/// Maximum byte length of an inline `BinaryViewArray` view. Views this short
/// store their bytes in the 16-byte view itself; longer views reference a data
/// block by `(buffer_index, offset)`. Fixed by the Arrow columnar format spec.
const MAX_INLINE_VIEW_LEN: u32 = 12;

/// The subset of [`BandArrayBuilder`]'s API that [`crate::traits::BandRef::
/// copy_into`] and [`crate::traits::BandRef::append_data_into`] need, as a
/// trait so those methods work whether the derived band is being written into
/// a bare [`BandArrayBuilder`] (no enclosing raster) or into a
/// [`crate::builder::RasterBuilder`] (which implements this trait by
/// delegating to its own internal `BandArrayBuilder` plus its own per-raster
/// bookkeeping).
///
/// `start_band` returns the band's recorded `(dim_names, visible_shape)` —
/// `RasterBuilder`'s implementation needs it to validate the band against its
/// current raster's spatial grid; a bare `BandArrayBuilder` caller can ignore
/// it.
pub trait BandWriter {
    fn start_band(
        &mut self,
        args: StartBandArgs<'_>,
    ) -> Result<(Vec<String>, Vec<i64>), RasterError>;
    fn band_data_writer(&mut self) -> &mut BinaryViewBuilder;
    fn append_band_data_from(
        &mut self,
        src: &BinaryViewArray,
        row: usize,
    ) -> Result<(), RasterError>;
}

/// Builder for the flat, per-band columns of `RasterSchema::band_type()` —
/// one row per band, with none of the raster-envelope concerns (`crs`,
/// `transform`, `spatial_dims`/`spatial_shape`, or grouping bands into a
/// per-raster list). [`crate::builder::RasterBuilder`] holds one of these
/// internally and wraps it with that envelope; a caller with no envelope to
/// carry (e.g. a bare Band-shaped value with no raster around it) can use
/// this directly.
///
/// Required steps to build a band: call [`Self::start_band`], write its data
/// via [`Self::band_data_writer`] (or [`Self::append_band_data_from`] to
/// share an existing row's backing buffer zero-copy), then
/// [`Self::finish_band`]. After all bands are added, [`Self::finish`] returns
/// the band `StructArray`.
pub struct BandArrayBuilder {
    name: StringBuilder,
    dim_names_values: StringBuilder,
    dim_names_offsets: Vec<i32>,
    shape_values: Int64Builder,
    shape_offsets: Vec<i32>,
    datatype: UInt32Builder,
    nodata: BinaryBuilder,
    // VIEW field — one entry per visible dimension per band. Stored as four
    // parallel Int64 columns + a List offset vector; assembled into a
    // `ListArray<StructArray<Int64,Int64,Int64,Int64>>` in `finish()`.
    view_source_axis_values: Int64Builder,
    view_start_values: Int64Builder,
    view_step_values: Int64Builder,
    view_steps_values: Int64Builder,
    view_offsets: Vec<i32>,
    // Per-band validity for the view list. `false` means the row is null —
    // the canonical representation of an identity view. `true` means the row
    // carries an explicit view in the four parallel value builders.
    view_validity: Vec<bool>,
    outdb_uri: StringBuilder,
    outdb_format: StringViewBuilder,
    data: BinaryViewBuilder,

    // Track band data count at the start of each band for finish_band validation.
    data_count_at_start: usize,

    // Zero-copy band-data dedup: maps an already-appended source `Buffer`'s
    // data pointer to its block index in `data`, so the same backing buffer
    // (e.g. many bands sharing one source column block) is attached once and
    // referenced by multiple views. See `append_band_data_buffer`.
    data_blocks: HashMap<usize, u32>,
}

impl BandArrayBuilder {
    /// Create a new band builder with the specified capacity (in bands).
    pub fn new(capacity: usize) -> Self {
        Self {
            name: StringBuilder::with_capacity(capacity, capacity),
            dim_names_values: StringBuilder::with_capacity(capacity * 2, capacity * 4),
            dim_names_offsets: vec![0],
            shape_values: Int64Builder::with_capacity(capacity * 2),
            shape_offsets: vec![0],
            datatype: UInt32Builder::with_capacity(capacity),
            nodata: BinaryBuilder::with_capacity(capacity, capacity),
            view_source_axis_values: Int64Builder::with_capacity(capacity * 2),
            view_start_values: Int64Builder::with_capacity(capacity * 2),
            view_step_values: Int64Builder::with_capacity(capacity * 2),
            view_steps_values: Int64Builder::with_capacity(capacity * 2),
            view_offsets: vec![0],
            view_validity: Vec::with_capacity(capacity),
            outdb_uri: StringBuilder::with_capacity(capacity, capacity),
            outdb_format: StringViewBuilder::with_capacity(capacity),
            data: BinaryViewBuilder::with_capacity(capacity),

            data_count_at_start: 0,
            data_blocks: HashMap::new(),
        }
    }

    /// Number of band rows appended so far (via [`Self::start_band`]).
    pub fn len(&self) -> usize {
        self.name.len()
    }

    /// True iff no bands have been appended yet.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Start a new band with explicit N-D parameters. See
    /// [`crate::builder::RasterBuilder::start_band`] for the full
    /// documentation of `args`' fields — this is the same logic, minus the
    /// per-raster bookkeeping `RasterBuilder`'s own `start_band` layers on
    /// top.
    ///
    /// Returns the band's recorded `(dim_names, visible_shape)` — see
    /// [`BandWriter::start_band`].
    fn start_band_impl(
        &mut self,
        args: StartBandArgs<'_>,
    ) -> Result<(Vec<String>, Vec<i64>), RasterError> {
        let StartBandArgs {
            name,
            dim_names,
            source_shape,
            view,
            data_type,
            nodata,
            outdb_uri,
            outdb_format,
        } = args;

        // A caller-supplied view is validated against source_shape. An identity
        // view falls through to the null-sentinel storage path below (identical
        // to `None`) so downstream readers agree on the canonical
        // representation; a genuinely non-identity view is persisted explicitly
        // into the band's four parallel view columns here.
        if let Some(view) = view {
            let ndim = dim_names.len();
            if ndim == 0 {
                return Err(RasterError::Invalid(
                    "start_band: 0-dimensional bands are not supported".into(),
                ));
            }
            if source_shape.len() != ndim || view.len() != ndim {
                return Err(RasterError::Invalid(format!(
                    "start_band: dim_names ({}), source_shape ({}), and view ({}) \
                     must all have the same length",
                    ndim,
                    source_shape.len(),
                    view.len()
                )));
            }
            view.validate(source_shape)?;

            if !view.is_identity(source_shape) {
                // -- Non-identity view: persist the explicit ViewEntry list. --
                match name {
                    Some(n) => self.name.append_value(n),
                    None => self.name.append_null(),
                }

                for dn in dim_names {
                    self.dim_names_values.append_value(dn);
                }
                let next = *self.dim_names_offsets.last().unwrap() + ndim as i32;
                self.dim_names_offsets.push(next);

                // The `shape` column stores the *source* shape; the visible
                // shape is derived from the view at read time.
                for &s in source_shape {
                    self.shape_values.append_value(s);
                }
                let next = *self.shape_offsets.last().unwrap() + ndim as i32;
                self.shape_offsets.push(next);

                self.datatype.append_value(data_type as u32);

                match nodata {
                    Some(b) => self.nodata.append_value(b),
                    None => self.nodata.append_null(),
                }

                // VIEW: one entry per visible axis written into the four
                // parallel columns, offset advanced by the entry count,
                // validity bit set — the mirror of the null-sentinel path below
                // (validity false, offset unchanged).
                for v in view {
                    self.view_source_axis_values.append_value(v.source_axis);
                    self.view_start_values.append_value(v.start);
                    self.view_step_values.append_value(v.step);
                    self.view_steps_values.append_value(v.steps);
                }
                let next = *self.view_offsets.last().unwrap() + ndim as i32;
                self.view_offsets.push(next);
                self.view_validity.push(true);

                match outdb_uri {
                    Some(uri) => self.outdb_uri.append_value(uri),
                    None => self.outdb_uri.append_null(),
                }
                match outdb_format {
                    Some(format) => self.outdb_format.append_value(format),
                    None => self.outdb_format.append_null(),
                }

                self.data_count_at_start = self.data.len();

                // The caller (e.g. `RasterBuilder::finish_raster`) compares
                // the band's *visible* shape against its raster's spatial_shape.
                return Ok((
                    dim_names.iter().map(|s| s.to_string()).collect(),
                    view.visible_shape(),
                ));
            }
        }

        if dim_names.is_empty() {
            return Err(RasterError::Invalid(
                "start_band: 0-dimensional bands are not supported".into(),
            ));
        }
        if dim_names.len() != source_shape.len() {
            return Err(RasterError::Invalid(format!(
                "start_band: dim_names ({}) and shape ({}) must have the same length",
                dim_names.len(),
                source_shape.len(),
            )));
        }
        // Name
        match name {
            Some(n) => self.name.append_value(n),
            None => self.name.append_null(),
        }

        // Dim names
        for dn in dim_names {
            self.dim_names_values.append_value(dn);
        }
        let next = *self.dim_names_offsets.last().unwrap() + dim_names.len() as i32;
        self.dim_names_offsets.push(next);

        // Shape
        for &s in source_shape {
            self.shape_values.append_value(s);
        }
        let next = *self.shape_offsets.last().unwrap() + source_shape.len() as i32;
        self.shape_offsets.push(next);

        // Data type
        self.datatype.append_value(data_type as u32);

        // Nodata
        match nodata {
            Some(nodata_bytes) => self.nodata.append_value(nodata_bytes),
            None => self.nodata.append_null(),
        }

        // VIEW: canonical identity is encoded as a null list entry — no
        // values appended, offset unchanged, validity bit cleared.
        let next = *self.view_offsets.last().unwrap();
        self.view_offsets.push(next);
        self.view_validity.push(false);

        // OutDb URI
        match outdb_uri {
            Some(uri) => self.outdb_uri.append_value(uri),
            None => self.outdb_uri.append_null(),
        }

        // OutDb format
        match outdb_format {
            Some(format) => self.outdb_format.append_value(format),
            None => self.outdb_format.append_null(),
        }

        self.data_count_at_start = self.data.len();

        // Record this band's dims/shape for the caller's own validation.
        Ok((
            dim_names.iter().map(|s| s.to_string()).collect(),
            source_shape.to_vec(),
        ))
    }

    /// Append the current band's data as a **zero-copy** view into an
    /// existing Arrow [`Buffer`], rather than copying bytes via
    /// `append_value`. See
    /// [`crate::builder::RasterBuilder::append_band_data_buffer`] for the
    /// full documentation — identical logic.
    pub fn append_band_data_buffer(
        &mut self,
        buffer: &Buffer,
        offset: u32,
        len: u32,
    ) -> Result<(), RasterError> {
        if len <= MAX_INLINE_VIEW_LEN {
            self.data
                .append_value(&buffer.as_slice()[offset as usize..(offset + len) as usize]);
            return Ok(());
        }
        let block = match self.data_blocks.get(&(buffer.as_ptr() as usize)) {
            Some(&idx) => idx,
            None => {
                // `clone` bumps the buffer's refcount; the bytes are not copied.
                let idx = self.data.append_block(buffer.clone());
                self.data_blocks.insert(buffer.as_ptr() as usize, idx);
                idx
            }
        };
        Ok(self.data.try_append_view(block, offset, len)?)
    }

    /// Finish writing the current band.
    ///
    /// Validates that exactly one data value was appended since `start_band()`.
    pub fn finish_band(&mut self) -> Result<(), RasterError> {
        let current_count = self.data.len();
        if current_count != self.data_count_at_start + 1 {
            return Err(RasterError::Invalid(
                format!(
                    "Expected exactly one band data value per band, but got {} appended since start_band()",
                    current_count - self.data_count_at_start
                ),
            ));
        }
        Ok(())
    }

    /// Finish building and return the flat band `StructArray` — one row per
    /// band appended via [`Self::start_band`], in order.
    pub fn finish(mut self) -> Result<StructArray, RasterError> {
        // Build band dim_names nested list
        let dim_names_values = self.dim_names_values.finish();
        let dim_names_offsets = OffsetBuffer::new(ScalarBuffer::from(self.dim_names_offsets));
        let DataType::List(dim_names_field) = RasterSchema::dim_names_type() else {
            return Err(RasterError::Invalid(
                "Expected list type for dim_names".to_string(),
            ));
        };
        let dim_names_list = ListArray::new(
            dim_names_field,
            dim_names_offsets,
            Arc::new(dim_names_values),
            None,
        );

        // Build band source_shape nested list
        let source_shape_values = self.shape_values.finish();
        let source_shape_offsets = OffsetBuffer::new(ScalarBuffer::from(self.shape_offsets));
        let DataType::List(source_shape_field) = RasterSchema::source_shape_type() else {
            return Err(RasterError::Invalid(
                "Expected list type for source_shape".to_string(),
            ));
        };
        let source_shape_list = ListArray::new(
            source_shape_field,
            source_shape_offsets,
            Arc::new(source_shape_values),
            None,
        );

        // Build band view nested list (List<Struct<Int64×4>>).
        let view_source_axis = self.view_source_axis_values.finish();
        let view_start = self.view_start_values.finish();
        let view_step = self.view_step_values.finish();
        let view_steps = self.view_steps_values.finish();
        let view_offsets = OffsetBuffer::new(ScalarBuffer::from(self.view_offsets));
        let DataType::List(view_list_field) = RasterSchema::view_type() else {
            return Err(RasterError::Invalid(
                "Expected list type for view".to_string(),
            ));
        };
        let DataType::Struct(view_struct_fields) = view_list_field.data_type().clone() else {
            return Err(RasterError::Invalid(
                "Expected struct type inside view list".to_string(),
            ));
        };
        let view_struct = StructArray::new(
            view_struct_fields,
            vec![
                Arc::new(view_source_axis) as ArrayRef,
                Arc::new(view_start) as ArrayRef,
                Arc::new(view_step) as ArrayRef,
                Arc::new(view_steps) as ArrayRef,
            ],
            None,
        );
        let view_nulls = if self.view_validity.iter().all(|&b| b) {
            None
        } else {
            Some(NullBuffer::from_iter(self.view_validity.iter().copied()))
        };
        let view_list = ListArray::new(
            view_list_field,
            view_offsets,
            Arc::new(view_struct),
            view_nulls,
        );

        // Build band struct
        let DataType::Struct(band_fields) = RasterSchema::band_type() else {
            return Err(RasterError::Invalid(
                "Expected struct type for band".to_string(),
            ));
        };

        let band_arrays: Vec<ArrayRef> = vec![
            Arc::new(self.name.finish()),
            Arc::new(dim_names_list),
            Arc::new(source_shape_list),
            Arc::new(self.datatype.finish()),
            Arc::new(self.nodata.finish()),
            Arc::new(view_list),
            Arc::new(self.outdb_uri.finish()),
            Arc::new(self.outdb_format.finish()),
            Arc::new(self.data.finish()),
        ];
        Ok(StructArray::new(band_fields, band_arrays, None))
    }
}

impl BandWriter for BandArrayBuilder {
    fn start_band(
        &mut self,
        args: StartBandArgs<'_>,
    ) -> Result<(Vec<String>, Vec<i64>), RasterError> {
        self.start_band_impl(args)
    }

    fn band_data_writer(&mut self) -> &mut BinaryViewBuilder {
        &mut self.data
    }

    fn append_band_data_from(
        &mut self,
        src: &BinaryViewArray,
        row: usize,
    ) -> Result<(), RasterError> {
        // Arrow BYTE_VIEW layout (u128, little-endian fields), fixed by the
        // columnar format spec:
        //   bits   0..32  length
        //   bits  32..64  prefix
        //   bits  64..96  buffer_index
        //   bits  96..128 offset
        // A view of `length <= MAX_INLINE_VIEW_LEN` stores its bytes inline and
        // has no backing buffer to share.
        let view = src.views()[row];
        let len = view as u32;
        if len <= MAX_INLINE_VIEW_LEN {
            self.data.append_value(src.value(row));
            Ok(())
        } else {
            let buffer_index = (view >> 64) as u32;
            let offset = (view >> 96) as u32;
            self.append_band_data_buffer(&src.data_buffers()[buffer_index as usize], offset, len)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::RasterStructArray;
    use crate::builder::RasterBuilder;
    use crate::traits::{BandOverrides, RasterRef};
    use arrow_array::{Array, BinaryViewArray, StringArray, UInt32Array};
    use sedona_schema::raster::{band_indices, BandDataType};

    /// The point of this whole refactor: a bare `BandArrayBuilder`, with no
    /// enclosing `RasterBuilder`/raster envelope at all, builds a real band
    /// row directly.
    #[test]
    fn bare_band_array_builder_produces_expected_struct_array() {
        let mut builder = BandArrayBuilder::new(2);
        assert!(builder.is_empty());

        builder
            .start_band(StartBandArgs {
                name: Some("first"),
                nodata: Some(&[9u8]),
                ..StartBandArgs::new(&["x"], &[3], BandDataType::UInt8)
            })
            .unwrap();
        builder.band_data_writer().append_value(vec![1u8, 2, 3]);
        builder.finish_band().unwrap();

        builder
            .start_band(StartBandArgs::new(&["x"], &[2], BandDataType::UInt8))
            .unwrap();
        builder.band_data_writer().append_value(vec![4u8, 5]);
        builder.finish_band().unwrap();

        assert_eq!(builder.len(), 2);
        let bands = builder.finish().unwrap();
        assert_eq!(bands.len(), 2);

        let names = bands
            .column(band_indices::NAME)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(names.value(0), "first");
        assert!(names.is_null(1));

        let datatype = bands
            .column(band_indices::DATA_TYPE)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        assert_eq!(datatype.value(0), BandDataType::UInt8 as u32);

        let data = bands
            .column(band_indices::DATA)
            .as_any()
            .downcast_ref::<BinaryViewArray>()
            .unwrap();
        assert_eq!(data.value(0), &[1u8, 2, 3]);
        assert_eq!(data.value(1), &[4u8, 5]);
    }

    /// `copy_into` targets a `&mut dyn BandWriter` precisely so a derived
    /// band can land in a bare `BandArrayBuilder` — no raster envelope, no
    /// `RasterBuilder::finish_raster`/`finish` needed — which this proves
    /// end to end: a real `BandRef` (read from a `RasterBuilder`-built
    /// source) copied into a standalone `BandArrayBuilder`.
    #[test]
    fn copy_into_targets_a_bare_band_array_builder() {
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        let mut source_builder = RasterBuilder::new(1);
        source_builder
            .start_raster_nd(&transform, &["x"], &[3], None)
            .unwrap();
        source_builder
            .start_band(StartBandArgs {
                nodata: Some(&[9u8]),
                ..StartBandArgs::new(&["x"], &[3], BandDataType::UInt8)
            })
            .unwrap();
        source_builder
            .band_data_writer()
            .append_value(vec![1u8, 2, 3]);
        source_builder.finish_band().unwrap();
        source_builder.finish_raster().unwrap();
        let source_array = source_builder.finish().unwrap();
        let source_rasters = RasterStructArray::try_new(&source_array).unwrap();
        let source_raster = source_rasters.get(0).unwrap();
        let source_band = source_raster.band(0).unwrap();

        let mut target = BandArrayBuilder::new(1);
        source_band
            .copy_into(&mut target, BandOverrides::default())
            .unwrap();
        target.finish_band().unwrap();
        assert_eq!(target.len(), 1);

        let bands = target.finish().unwrap();
        let nodata = bands.column(band_indices::NODATA);
        assert_eq!(
            nodata
                .as_any()
                .downcast_ref::<arrow_array::BinaryArray>()
                .unwrap()
                .value(0),
            &[9u8]
        );
        let data = bands
            .column(band_indices::DATA)
            .as_any()
            .downcast_ref::<BinaryViewArray>()
            .unwrap();
        assert_eq!(data.value(0), &[1u8, 2, 3]);
    }
}
