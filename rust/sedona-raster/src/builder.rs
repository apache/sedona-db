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
    builder::{
        ArrayBuilder, BinaryBuilder, BinaryViewBuilder, BooleanBuilder, Float64Builder,
        Int64Builder, StringBuilder, StringViewBuilder, UInt32Builder,
    },
    Array, ArrayRef, BinaryViewArray, ListArray, StructArray,
};
use arrow_buffer::{Buffer, NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{ArrowError, DataType};
use std::collections::HashMap;
use std::sync::Arc;

use sedona_schema::raster::{BandDataType, RasterSchema};

use crate::traits::{BandMetadata, BandOverrides, MetadataRef, RasterRef};
use crate::view_entries::{ViewEntries, ViewEntry};

/// Raster-level metadata overrides for [`RasterBuilder::start_raster_from`] and
/// [`RasterBuilder::copy_raster_from`]. A `None` field inherits the source
/// raster's value. The band-level analog is [`BandOverrides`].
#[derive(Debug, Default, Clone)]
pub struct RasterOverrides {
    /// Override the 6-element GDAL geotransform; `None` inherits the source's.
    pub transform: Option<[f64; 6]>,
}

/// Maximum byte length of an inline `BinaryViewArray` view. Views this short
/// store their bytes in the 16-byte view itself; longer views reference a data
/// block by `(buffer_index, offset)`. Fixed by the Arrow columnar format spec.
const MAX_INLINE_VIEW_LEN: u32 = 12;

/// Builder for constructing raster arrays with zero-copy band data writing
///
/// Required steps to build a raster:
/// 1. Create a RasterBuilder with a specified capacity
/// 2. For each raster to add:
///    - Call `start_raster` with the appropriate metadata, CRS
///    - For each band in the raster:
///       - Call `start_band` with the band metadata
///       - Use `band_data_writer` to get a BinaryViewBuilder and write the band data
///       - Call `finish_band` to complete the band
///    - Call `finish_raster` to complete the raster
/// 3. After all rasters are added, call `finish` to get the final StructArray
///
/// Example usage:
/// ```
/// use sedona_raster::traits::{RasterMetadata, BandMetadata};
/// use sedona_schema::raster::{StorageType, BandDataType};
/// use sedona_raster::builder::RasterBuilder;
///
/// let mut builder = RasterBuilder::new(1);
/// let metadata = RasterMetadata {
///     width: 100, height: 100,
///     upperleft_x: 0.0, upperleft_y: 0.0,
///     scale_x: 1.0, scale_y: -1.0,
///     skew_x: 0.0, skew_y: 0.0,
/// };
/// // Start a raster from RasterMetadata struct
/// builder.start_raster(&metadata, Some("EPSG:4326")).unwrap();
///
/// // Add a band:
/// let band_metadata = BandMetadata {
///     nodata_value: Some(vec![0u8]),
///     storage_type: StorageType::InDb,
///     datatype: BandDataType::UInt8,
///     outdb_url: None,
///     outdb_band_id: None,
/// };
/// builder.start_band(band_metadata).unwrap();
/// let band_writer = builder.band_data_writer();
/// band_writer.append_value(&vec![/* band data bytes */]);
/// builder.finish_band().unwrap();
///
/// // Finish the raster
/// builder.finish_raster().unwrap();
///
/// // Get the final StructArray
/// let raster_array = builder.finish().unwrap();
/// ```
pub struct RasterBuilder {
    // Top-level raster fields
    crs: StringViewBuilder,
    transform_values: Float64Builder,
    transform_offsets: Vec<i32>,
    spatial_dims_values: StringViewBuilder,
    spatial_dims_offsets: Vec<i32>,
    spatial_shape_values: Int64Builder,
    spatial_shape_offsets: Vec<i32>,

    // Band fields (flattened across all bands)
    band_name: StringBuilder,
    band_dim_names_values: StringBuilder,
    band_dim_names_offsets: Vec<i32>,
    band_shape_values: Int64Builder,
    band_shape_offsets: Vec<i32>,
    band_datatype: UInt32Builder,
    band_nodata: BinaryBuilder,
    // VIEW field — one entry per visible dimension per band. Stored as four
    // parallel Int64 columns + a List offset vector; assembled into a
    // `ListArray<StructArray<Int64,Int64,Int64,Int64>>` in `finish()`.
    band_view_source_axis_values: Int64Builder,
    band_view_start_values: Int64Builder,
    band_view_step_values: Int64Builder,
    band_view_steps_values: Int64Builder,
    band_view_offsets: Vec<i32>,
    // Per-band validity for the view list. `false` means the row is null —
    // the canonical representation of an identity view. `true` means the row
    // carries an explicit view in the four parallel value builders.
    band_view_validity: Vec<bool>,
    band_outdb_uri: StringBuilder,
    band_outdb_format: StringViewBuilder,
    band_data: BinaryViewBuilder,

    // List structure tracking
    band_offsets: Vec<i32>,  // Track where each raster's bands start/end
    current_band_count: i32, // Track bands in current raster

    // Current raster state (needed for start_band_2d)
    current_width: i64,
    current_height: i64,

    // Per-raster validation state: spatial dims/shape and recorded bands so
    // finish_raster can check every band matches the top-level spatial grid.
    current_spatial_dims: Vec<String>,
    current_spatial_shape: Vec<i64>,
    current_raster_bands: Vec<(Vec<String>, Vec<i64>)>,

    // Track band_data count at the start of each band for finish_band validation
    band_data_count_at_start: usize,

    // Zero-copy band-data dedup: maps an already-appended source `Buffer`'s
    // data pointer to its block index in `band_data`, so the same backing
    // buffer (e.g. many bands sharing one source column block) is attached
    // once and referenced by multiple views. See `append_band_data_buffer`.
    band_data_blocks: HashMap<usize, u32>,

    raster_validity: BooleanBuilder,
}

/// Arguments to [`RasterBuilder::start_band_with_view`]. Bundled into a
/// struct to keep the call site readable — eight slots is enough that
/// positional args invite mis-ordering bugs.
pub(crate) struct StartBandWithViewArgs<'a> {
    pub name: Option<&'a str>,
    pub dim_names: &'a [&'a str],
    pub source_shape: &'a [i64],
    pub view: &'a [ViewEntry],
    pub data_type: BandDataType,
    pub nodata: Option<&'a [u8]>,
    pub outdb_uri: Option<&'a str>,
    pub outdb_format: Option<&'a str>,
}

impl RasterBuilder {
    /// Create a new raster builder with the specified capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            crs: StringViewBuilder::with_capacity(capacity),
            transform_values: Float64Builder::with_capacity(capacity * 6),
            transform_offsets: vec![0],
            spatial_dims_values: StringViewBuilder::with_capacity(capacity * 2),
            spatial_dims_offsets: vec![0],
            spatial_shape_values: Int64Builder::with_capacity(capacity * 2),
            spatial_shape_offsets: vec![0],

            band_name: StringBuilder::with_capacity(capacity, capacity),
            band_dim_names_values: StringBuilder::with_capacity(capacity * 2, capacity * 4),
            band_dim_names_offsets: vec![0],
            band_shape_values: Int64Builder::with_capacity(capacity * 2),
            band_shape_offsets: vec![0],
            band_datatype: UInt32Builder::with_capacity(capacity),
            band_nodata: BinaryBuilder::with_capacity(capacity, capacity),
            band_view_source_axis_values: Int64Builder::with_capacity(capacity * 2),
            band_view_start_values: Int64Builder::with_capacity(capacity * 2),
            band_view_step_values: Int64Builder::with_capacity(capacity * 2),
            band_view_steps_values: Int64Builder::with_capacity(capacity * 2),
            band_view_offsets: vec![0],
            band_view_validity: Vec::with_capacity(capacity),
            band_outdb_uri: StringBuilder::with_capacity(capacity, capacity),
            band_outdb_format: StringViewBuilder::with_capacity(capacity),
            band_data: BinaryViewBuilder::with_capacity(capacity),

            band_offsets: vec![0],
            current_band_count: 0,
            current_width: 0,
            current_height: 0,

            current_spatial_dims: Vec::new(),
            current_spatial_shape: Vec::new(),
            current_raster_bands: Vec::new(),

            band_data_count_at_start: 0,
            band_data_blocks: HashMap::new(),

            raster_validity: BooleanBuilder::with_capacity(capacity),
        }
    }

    /// Start a new raster with explicit N-D parameters.
    ///
    /// `transform` must be a 6-element GDAL GeoTransform:
    /// `[origin_x, scale_x, skew_x, origin_y, skew_y, scale_y]`
    ///
    /// `spatial_dims` names the raster-level spatial dimensions (today always
    /// length 2, e.g. `["x","y"]`). `spatial_shape` gives their sizes in the
    /// same order. Every band added to this raster must contain each name in
    /// `spatial_dims` within its own `dim_names`, with matching size.
    pub fn start_raster_nd(
        &mut self,
        transform: &[f64; 6],
        spatial_dims: &[&str],
        spatial_shape: &[i64],
        crs: Option<&str>,
    ) -> Result<(), ArrowError> {
        if spatial_dims.len() != spatial_shape.len() {
            return Err(ArrowError::InvalidArgumentError(format!(
                "spatial_dims.len() ({}) must equal spatial_shape.len() ({})",
                spatial_dims.len(),
                spatial_shape.len()
            )));
        }

        // Transform
        for &v in transform {
            self.transform_values.append_value(v);
        }
        let next = *self.transform_offsets.last().unwrap() + 6;
        self.transform_offsets.push(next);

        // Spatial dims + shape
        for d in spatial_dims {
            self.spatial_dims_values.append_value(d);
        }
        let next = *self.spatial_dims_offsets.last().unwrap() + spatial_dims.len() as i32;
        self.spatial_dims_offsets.push(next);

        for &s in spatial_shape {
            self.spatial_shape_values.append_value(s);
        }
        let next = *self.spatial_shape_offsets.last().unwrap() + spatial_shape.len() as i32;
        self.spatial_shape_offsets.push(next);

        // CRS
        match crs {
            Some(crs_data) => self.crs.append_value(crs_data),
            None => self.crs.append_null(),
        }

        self.current_band_count = 0;
        self.current_spatial_dims = spatial_dims.iter().map(|s| s.to_string()).collect();
        self.current_spatial_shape = spatial_shape.to_vec();
        self.current_raster_bands.clear();
        // Preserve legacy current_width/current_height for start_band_2d (set
        // by start_raster_2d). Callers using this direct entry point drive
        // their own shapes via start_band_nd.
        self.current_width = 0;
        self.current_height = 0;

        Ok(())
    }

    /// Start a raster from `source`, copying its geotransform, spatial
    /// dims/shape, and CRS — with [`RasterOverrides`] applied — but **not** its
    /// bands. The caller adds bands (e.g. via
    /// [`BandRef::copy_into`](crate::traits::BandRef::copy_into)) and then calls
    /// [`finish_raster`](Self::finish_raster). Use this when bands need
    /// per-band changes; see [`copy_raster_from`](Self::copy_raster_from) to
    /// copy a raster whole.
    pub fn start_raster_from(
        &mut self,
        source: &dyn RasterRef,
        overrides: RasterOverrides,
    ) -> Result<(), ArrowError> {
        let transform: [f64; 6] = match overrides.transform {
            Some(transform) => transform,
            None => source.transform().try_into().map_err(|_| {
                ArrowError::InvalidArgumentError("raster transform is not 6 elements".to_string())
            })?,
        };
        let spatial_dims = source.spatial_dims();
        self.start_raster_nd(
            &transform,
            &spatial_dims,
            source.spatial_shape(),
            source.crs(),
        )
    }

    /// Copy `source` whole: its metadata (with [`RasterOverrides`] applied) and
    /// every band, each derived via
    /// [`BandRef::copy_into`](crate::traits::BandRef::copy_into) so pixel buffers
    /// are shared zero-copy. Finishes the raster — no further calls are needed
    /// for this row.
    pub fn copy_raster_from(
        &mut self,
        source: &dyn RasterRef,
        overrides: RasterOverrides,
    ) -> Result<(), ArrowError> {
        self.start_raster_from(source, overrides)?;
        for band_idx in 0..source.num_bands() {
            source
                .band(band_idx)?
                .copy_into(self, BandOverrides::default())?;
            self.finish_band()?;
        }
        self.finish_raster()
    }

    /// Convenience: start a 2-D raster with positional geotransform parameters.
    /// Sets `spatial_dims=["x","y"]` and `spatial_shape=[width, height]` and
    /// builds the 6-element GDAL transform internally. The N-D entry point is
    /// [`Self::start_raster_nd`]; the metadata-taking entry is
    /// [`Self::start_raster`].
    #[allow(clippy::too_many_arguments)]
    pub fn start_raster_2d(
        &mut self,
        width: i64,
        height: i64,
        origin_x: f64,
        origin_y: f64,
        scale_x: f64,
        scale_y: f64,
        skew_x: f64,
        skew_y: f64,
        crs: Option<&str>,
    ) -> Result<(), ArrowError> {
        let transform = [origin_x, scale_x, skew_x, origin_y, skew_y, scale_y];
        self.start_raster_nd(&transform, &["x", "y"], &[width, height], crs)?;
        self.current_width = width;
        self.current_height = height;
        Ok(())
    }

    /// Start a 2-D raster from a `&dyn MetadataRef`. Matches the pre-N-D
    /// signature so callers from before the refactor keep compiling without
    /// changing argument lists.
    pub fn start_raster(
        &mut self,
        metadata: &dyn MetadataRef,
        crs: Option<&str>,
    ) -> Result<(), ArrowError> {
        self.start_raster_2d(
            metadata.width(),
            metadata.height(),
            metadata.upper_left_x(),
            metadata.upper_left_y(),
            metadata.scale_x(),
            metadata.scale_y(),
            metadata.skew_x(),
            metadata.skew_y(),
            crs,
        )
    }

    /// Start a new band with explicit N-D parameters.
    ///
    /// `outdb_uri` is the *location* of the external resource (scheme is
    /// resolved by an `ObjectStoreRegistry`). `outdb_format` is the *format*
    /// used to interpret the bytes at that location (e.g. `"geotiff"`,
    /// `"zarr"`). A null `outdb_format` means the band is in-memory — the
    /// band's `data` buffer is authoritative.
    #[allow(clippy::too_many_arguments)]
    pub fn start_band_nd(
        &mut self,
        name: Option<&str>,
        dim_names: &[&str],
        shape: &[i64],
        data_type: BandDataType,
        nodata: Option<&[u8]>,
        outdb_uri: Option<&str>,
        outdb_format: Option<&str>,
    ) -> Result<(), ArrowError> {
        if dim_names.is_empty() {
            return Err(ArrowError::InvalidArgumentError(
                "start_band_nd: 0-dimensional bands are not supported".into(),
            ));
        }
        if dim_names.len() != shape.len() {
            return Err(ArrowError::InvalidArgumentError(format!(
                "start_band_nd: dim_names ({}) and shape ({}) must have the same length",
                dim_names.len(),
                shape.len(),
            )));
        }
        // Name
        match name {
            Some(n) => self.band_name.append_value(n),
            None => self.band_name.append_null(),
        }

        // Dim names
        for dn in dim_names {
            self.band_dim_names_values.append_value(dn);
        }
        let next = *self.band_dim_names_offsets.last().unwrap() + dim_names.len() as i32;
        self.band_dim_names_offsets.push(next);

        // Shape
        for &s in shape {
            self.band_shape_values.append_value(s);
        }
        let next = *self.band_shape_offsets.last().unwrap() + shape.len() as i32;
        self.band_shape_offsets.push(next);

        // Data type
        self.band_datatype.append_value(data_type as u32);

        // Nodata
        match nodata {
            Some(nodata_bytes) => self.band_nodata.append_value(nodata_bytes),
            None => self.band_nodata.append_null(),
        }

        // VIEW: canonical identity is encoded as a null list entry — no
        // values appended, offset unchanged, validity bit cleared.
        let next = *self.band_view_offsets.last().unwrap();
        self.band_view_offsets.push(next);
        self.band_view_validity.push(false);

        // OutDb URI
        match outdb_uri {
            Some(uri) => self.band_outdb_uri.append_value(uri),
            None => self.band_outdb_uri.append_null(),
        }

        // OutDb format
        match outdb_format {
            Some(format) => self.band_outdb_format.append_value(format),
            None => self.band_outdb_format.append_null(),
        }

        self.current_band_count += 1;
        self.band_data_count_at_start = self.band_data.len();

        // Record this band's dims/shape for strict validation at finish_raster.
        self.current_raster_bands.push((
            dim_names.iter().map(|s| s.to_string()).collect(),
            shape.to_vec(),
        ));

        Ok(())
    }

    /// Start a band carrying an explicit `view` — a per-axis window of
    /// offsets/steps over `source_shape` — rather than the implicit identity.
    ///
    /// This is the entry point view persistence will use, so callers such as
    /// [`BandRef::copy_into`] route through it unchanged once persistence
    /// lands. Today the band schema stores a view only as the canonical
    /// identity null sentinel, so a non-identity `view` is rejected; an
    /// identity view is stored exactly as [`Self::start_band_nd`] does. View
    /// persistence is tracked in
    /// <https://github.com/apache/sedona-db/issues/897>.
    pub(crate) fn start_band_with_view(
        &mut self,
        args: StartBandWithViewArgs<'_>,
    ) -> Result<(), ArrowError> {
        let StartBandWithViewArgs {
            name,
            dim_names,
            source_shape,
            view,
            data_type,
            nodata,
            outdb_uri,
            outdb_format,
        } = args;
        let ndim = dim_names.len();
        if ndim == 0 {
            return Err(ArrowError::InvalidArgumentError(
                "start_band_with_view: 0-dimensional bands are not supported".into(),
            ));
        }
        if source_shape.len() != ndim || view.len() != ndim {
            return Err(ArrowError::InvalidArgumentError(format!(
                "start_band_with_view: dim_names ({}), source_shape ({}), and view ({}) \
                 must all have the same length",
                ndim,
                source_shape.len(),
                view.len()
            )));
        }
        let view_entries = ViewEntries::new(view.to_vec());
        view_entries.validate(source_shape)?;
        // The schema stores views only as the identity null sentinel today, so a
        // non-identity view can't round-trip — reject it up front, before any
        // column append, rather than persisting mislocated bytes.
        if !view_entries.is_identity(source_shape) {
            return Err(ArrowError::InvalidArgumentError(
                "start_band_with_view: persisting a non-identity band view is not yet \
                 supported (see https://github.com/apache/sedona-db/issues/897); \
                 materialize the band (e.g. via RS_EnsureContiguous) first"
                    .into(),
            ));
        }
        self.start_band_nd(
            name,
            dim_names,
            source_shape,
            data_type,
            nodata,
            outdb_uri,
            outdb_format,
        )
    }

    /// Convenience: start a 2D band with `dim_names=["y","x"]` and `shape=[height, width]`.
    ///
    /// Must be called after `start_raster_2d` / `start_raster_2d` which sets
    /// the current width/height.
    pub fn start_band_2d(
        &mut self,
        data_type: BandDataType,
        nodata: Option<&[u8]>,
    ) -> Result<(), ArrowError> {
        if self.current_width == 0 && self.current_height == 0 {
            return Err(ArrowError::InvalidArgumentError(
                "start_band_2d requires prior start_raster_2d (width and height are 0)".into(),
            ));
        }
        self.start_band_nd(
            None,
            &["y", "x"],
            &[self.current_height, self.current_width],
            data_type,
            nodata,
            None,
            None,
        )
    }

    /// Start a 2-D band from a concrete [`BandMetadata`] struct. Matches
    /// the pre-N-D signature so callers from before the refactor keep
    /// compiling. For OutDb bands the `outdb_url` + `outdb_band_id` are
    /// recombined into the SedonaDB `<url>#band=N` URI convention.
    pub fn start_band(&mut self, metadata: BandMetadata) -> Result<(), ArrowError> {
        if self.current_width == 0 && self.current_height == 0 {
            return Err(ArrowError::InvalidArgumentError(
                "start_band requires prior start_raster / start_raster_2d (width and height are 0)"
                    .into(),
            ));
        }
        let outdb_uri = match (metadata.outdb_url.as_deref(), metadata.outdb_band_id) {
            (Some(url), Some(band_id)) => Some(format!("{url}#band={band_id}")),
            (Some(url), None) => Some(url.to_string()),
            _ => None,
        };
        self.start_band_nd(
            None,
            &["y", "x"],
            &[self.current_height, self.current_width],
            metadata.datatype,
            metadata.nodata_value.as_deref(),
            outdb_uri.as_deref(),
            None,
        )
    }

    /// Get direct access to the BinaryViewBuilder for writing the current band's data.
    pub fn band_data_writer(&mut self) -> &mut BinaryViewBuilder {
        &mut self.band_data
    }

    /// Append the current band's data as a **zero-copy** view into an existing
    /// Arrow [`Buffer`], rather than copying bytes via `append_value`.
    ///
    /// `buffer` is attached to the output `BinaryViewArray` as a shared data
    /// block (a refcount bump, never a copy) and a single view row referencing
    /// `[offset, offset + len)` is appended. Buffers are de-duplicated by data
    /// pointer, so handing the same backing buffer for many bands attaches it
    /// once and points every view at it.
    ///
    /// Bytes at or under the inline threshold ([`MAX_INLINE_VIEW_LEN`]) are
    /// stored inline (a tiny copy) instead, since the `BinaryViewArray` layout
    /// requires such views to be inline and a block-referencing view of that
    /// size is non-canonical (it fails array validation on roundtrip). Band
    /// data is realistically always larger, so this is the degenerate path.
    ///
    /// Counts as the one data value for the current band (see [`finish_band`]).
    ///
    /// [`finish_band`]: Self::finish_band
    pub fn append_band_data_buffer(
        &mut self,
        buffer: &Buffer,
        offset: u32,
        len: u32,
    ) -> Result<(), ArrowError> {
        if len <= MAX_INLINE_VIEW_LEN {
            self.band_data
                .append_value(&buffer.as_slice()[offset as usize..(offset + len) as usize]);
            return Ok(());
        }
        let block = match self.band_data_blocks.get(&(buffer.as_ptr() as usize)) {
            Some(&idx) => idx,
            None => {
                // `clone` bumps the buffer's refcount; the bytes are not copied.
                let idx = self.band_data.append_block(buffer.clone());
                self.band_data_blocks.insert(buffer.as_ptr() as usize, idx);
                idx
            }
        };
        self.band_data.try_append_view(block, offset, len)
    }

    /// Append the current band's data by copying row `row` of `src` through —
    /// zero-copy when the row's bytes are block-backed (shares the backing
    /// `Buffer` via [`append_band_data_buffer`]), with a small copy only for
    /// inline views (≤ [`MAX_INLINE_VIEW_LEN`] bytes, which live in the view
    /// itself and have no backing buffer).
    ///
    /// Counts as the one data value for the current band (see [`finish_band`]).
    ///
    /// [`finish_band`]: Self::finish_band
    pub fn append_band_data_from(
        &mut self,
        src: &BinaryViewArray,
        row: usize,
    ) -> Result<(), ArrowError> {
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
            self.band_data.append_value(src.value(row));
            Ok(())
        } else {
            let buffer_index = (view >> 64) as u32;
            let offset = (view >> 96) as u32;
            self.append_band_data_buffer(&src.data_buffers()[buffer_index as usize], offset, len)
        }
    }

    /// Finish writing the current band.
    ///
    /// Validates that exactly one data value was appended since `start_band_nd()`.
    pub fn finish_band(&mut self) -> Result<(), ArrowError> {
        let current_count = self.band_data.len();
        if current_count != self.band_data_count_at_start + 1 {
            return Err(ArrowError::InvalidArgumentError(
                format!(
                    "Expected exactly one band data value per band, but got {} appended since start_band_nd()",
                    current_count - self.band_data_count_at_start
                ),
            ));
        }
        Ok(())
    }

    /// Finish all bands for the current raster.
    ///
    /// Strictly validates every band added since `start_raster_nd`: each name in
    /// the top-level `spatial_dims` must appear in the band's own `dim_names`
    /// with a size matching the corresponding entry in `spatial_shape`.
    pub fn finish_raster(&mut self) -> Result<(), ArrowError> {
        for (band_idx, (band_dims, band_shape)) in self.current_raster_bands.iter().enumerate() {
            for (spatial_idx, spatial_dim) in self.current_spatial_dims.iter().enumerate() {
                let pos = band_dims
                    .iter()
                    .position(|d| d == spatial_dim)
                    .ok_or_else(|| {
                        ArrowError::InvalidArgumentError(format!(
                            "Band {band_idx} is missing spatial dimension {spatial_dim:?} \
                         (band dim_names = {band_dims:?})"
                        ))
                    })?;
                let expected = self.current_spatial_shape[spatial_idx];
                let actual = band_shape[pos];
                if actual != expected {
                    return Err(ArrowError::InvalidArgumentError(format!(
                        "Band {band_idx} dimension {spatial_dim:?} has size {actual}, \
                         expected {expected} from top-level spatial_shape"
                    )));
                }
            }
        }

        let next_offset = self.band_offsets.last().unwrap() + self.current_band_count;
        self.band_offsets.push(next_offset);
        self.raster_validity.append_value(true);
        self.current_raster_bands.clear();
        self.current_spatial_dims.clear();
        self.current_spatial_shape.clear();
        Ok(())
    }

    /// Append a null raster.
    pub fn append_null(&mut self) -> Result<(), ArrowError> {
        // Transform: append 6 zeros
        for _ in 0..6 {
            self.transform_values.append_value(0.0);
        }
        let next = *self.transform_offsets.last().unwrap() + 6;
        self.transform_offsets.push(next);

        // Spatial dims + shape: empty list for null rasters.
        let next = *self.spatial_dims_offsets.last().unwrap();
        self.spatial_dims_offsets.push(next);
        let next = *self.spatial_shape_offsets.last().unwrap();
        self.spatial_shape_offsets.push(next);

        // CRS: null
        self.crs.append_null();

        // No bands
        let current_offset = *self.band_offsets.last().unwrap();
        self.band_offsets.push(current_offset);

        // Mark null
        self.raster_validity.append_null();

        Ok(())
    }

    /// Finish building and return the constructed StructArray.
    pub fn finish(mut self) -> Result<StructArray, ArrowError> {
        // Build transform list
        let transform_values = self.transform_values.finish();
        let transform_offsets = OffsetBuffer::new(ScalarBuffer::from(self.transform_offsets));
        let DataType::List(transform_field) = RasterSchema::transform_type() else {
            return Err(ArrowError::SchemaError(
                "Expected list type for transform".to_string(),
            ));
        };
        let transform_list = ListArray::new(
            transform_field,
            transform_offsets,
            Arc::new(transform_values),
            None,
        );

        // Build spatial_dims list
        let spatial_dims_values = self.spatial_dims_values.finish();
        let spatial_dims_offsets = OffsetBuffer::new(ScalarBuffer::from(self.spatial_dims_offsets));
        let DataType::List(spatial_dims_field) = RasterSchema::spatial_dims_type() else {
            return Err(ArrowError::SchemaError(
                "Expected list type for spatial_dims".to_string(),
            ));
        };
        let spatial_dims_list = ListArray::new(
            spatial_dims_field,
            spatial_dims_offsets,
            Arc::new(spatial_dims_values),
            None,
        );

        // Build spatial_shape list
        let spatial_shape_values = self.spatial_shape_values.finish();
        let spatial_shape_offsets =
            OffsetBuffer::new(ScalarBuffer::from(self.spatial_shape_offsets));
        let DataType::List(spatial_shape_field) = RasterSchema::spatial_shape_type() else {
            return Err(ArrowError::SchemaError(
                "Expected list type for spatial_shape".to_string(),
            ));
        };
        let spatial_shape_list = ListArray::new(
            spatial_shape_field,
            spatial_shape_offsets,
            Arc::new(spatial_shape_values),
            None,
        );

        // Build band dim_names nested list
        let dim_names_values = self.band_dim_names_values.finish();
        let dim_names_offsets = OffsetBuffer::new(ScalarBuffer::from(self.band_dim_names_offsets));
        let DataType::List(dim_names_field) = RasterSchema::dim_names_type() else {
            return Err(ArrowError::SchemaError(
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
        let source_shape_values = self.band_shape_values.finish();
        let source_shape_offsets = OffsetBuffer::new(ScalarBuffer::from(self.band_shape_offsets));
        let DataType::List(source_shape_field) = RasterSchema::source_shape_type() else {
            return Err(ArrowError::SchemaError(
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
        let view_source_axis = self.band_view_source_axis_values.finish();
        let view_start = self.band_view_start_values.finish();
        let view_step = self.band_view_step_values.finish();
        let view_steps = self.band_view_steps_values.finish();
        let view_offsets = OffsetBuffer::new(ScalarBuffer::from(self.band_view_offsets));
        let DataType::List(view_list_field) = RasterSchema::view_type() else {
            return Err(ArrowError::SchemaError(
                "Expected list type for view".to_string(),
            ));
        };
        let DataType::Struct(view_struct_fields) = view_list_field.data_type().clone() else {
            return Err(ArrowError::SchemaError(
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
        let view_nulls = if self.band_view_validity.iter().all(|&b| b) {
            None
        } else {
            Some(NullBuffer::from_iter(
                self.band_view_validity.iter().copied(),
            ))
        };
        let view_list = ListArray::new(
            view_list_field,
            view_offsets,
            Arc::new(view_struct),
            view_nulls,
        );

        // Build band struct
        let DataType::Struct(band_fields) = RasterSchema::band_type() else {
            return Err(ArrowError::SchemaError(
                "Expected struct type for band".to_string(),
            ));
        };

        let band_arrays: Vec<ArrayRef> = vec![
            Arc::new(self.band_name.finish()),
            Arc::new(dim_names_list),
            Arc::new(source_shape_list),
            Arc::new(self.band_datatype.finish()),
            Arc::new(self.band_nodata.finish()),
            Arc::new(view_list),
            Arc::new(self.band_outdb_uri.finish()),
            Arc::new(self.band_outdb_format.finish()),
            Arc::new(self.band_data.finish()),
        ];
        let band_struct = StructArray::new(band_fields, band_arrays, None);

        // Build bands list
        let DataType::List(bands_field) = RasterSchema::bands_type() else {
            return Err(ArrowError::SchemaError(
                "Expected list type for bands".to_string(),
            ));
        };
        let band_list_offsets = OffsetBuffer::new(ScalarBuffer::from(self.band_offsets));
        let bands_list =
            ListArray::new(bands_field, band_list_offsets, Arc::new(band_struct), None);

        // Build top-level raster struct
        let raster_fields = RasterSchema::fields();
        let raster_arrays: Vec<ArrayRef> = vec![
            Arc::new(self.crs.finish()),
            Arc::new(transform_list),
            Arc::new(spatial_dims_list),
            Arc::new(spatial_shape_list),
            Arc::new(bands_list),
        ];

        let raster_validity_array = self.raster_validity.finish();
        let raster_nulls = raster_validity_array.nulls().cloned();

        Ok(StructArray::new(raster_fields, raster_arrays, raster_nulls))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::RasterStructArray;
    use crate::traits::{RasterMetadata, RasterRef};
    use arrow_array::RecordBatch;
    use arrow_ipc::reader::StreamReader;
    use arrow_ipc::writer::StreamWriter;
    use arrow_schema::Schema;
    use sedona_schema::raster::StorageType;
    use std::io::Cursor;

    #[test]
    fn test_iterator_basic_functionality() {
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

        // Test the iterator
        let rasters = RasterStructArray::try_new(&raster_array).unwrap();

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

        // Test iterator over bands
        let band_iter: Vec<_> = bands.iter().collect();
        assert_eq!(band_iter.len(), 1);
    }

    #[test]
    fn test_multi_band_iterator() {
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

        let rasters = RasterStructArray::try_new(&raster_array).unwrap();
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

        // Test iterator
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
    fn test_copy_metadata_from_iterator() {
        // Create an original raster
        let mut source_builder = RasterBuilder::new(10);

        let original_metadata = RasterMetadata {
            width: 42,
            height: 24,
            upperleft_x: -122.0,
            upperleft_y: 37.8,
            scale_x: 0.1,
            scale_y: -0.1,
            skew_x: 0.0,
            skew_y: 0.0,
        };

        source_builder
            .start_raster(&original_metadata, None)
            .unwrap();

        let band_metadata = BandMetadata {
            nodata_value: Some(vec![255u8]),
            storage_type: StorageType::InDb,
            datatype: BandDataType::UInt8,
            outdb_url: None,
            outdb_band_id: None,
        };

        source_builder.start_band(band_metadata).unwrap();
        let test_data = vec![42u8; 1008]; // 42x24 raster
        source_builder.band_data_writer().append_value(&test_data);
        source_builder.finish_band().unwrap();
        source_builder.finish_raster().unwrap();

        let source_array = source_builder.finish().unwrap();

        // Create a new raster using metadata from the iterator
        let mut target_builder = RasterBuilder::new(10);
        let iterator = RasterStructArray::try_new(&source_array).unwrap();
        let source_raster = iterator.get(0).unwrap();

        target_builder
            .start_raster(&source_raster.metadata(), source_raster.crs())
            .unwrap();

        // Add new band data while preserving original metadata
        let new_band_metadata = BandMetadata {
            nodata_value: None,
            storage_type: StorageType::InDb,
            datatype: BandDataType::UInt16,
            outdb_url: None,
            outdb_band_id: None,
        };

        target_builder.start_band(new_band_metadata).unwrap();
        let new_data = vec![100u16; 1008]; // Different data, same dimensions
        let new_data_bytes: Vec<u8> = new_data.iter().flat_map(|&x| x.to_le_bytes()).collect();

        target_builder
            .band_data_writer()
            .append_value(&new_data_bytes);
        target_builder.finish_band().unwrap();
        target_builder.finish_raster().unwrap();

        let target_array = target_builder.finish().unwrap();

        // Verify the metadata was copied correctly
        let target_iterator = RasterStructArray::try_new(&target_array).unwrap();
        let target_raster = target_iterator.get(0).unwrap();
        let target_metadata = target_raster.metadata();

        // All metadata should match the original
        assert_eq!(target_metadata.width(), 42);
        assert_eq!(target_metadata.height(), 24);
        assert_eq!(target_metadata.upper_left_x(), -122.0);
        assert_eq!(target_metadata.upper_left_y(), 37.8);
        assert_eq!(target_metadata.scale_x(), 0.1);
        assert_eq!(target_metadata.scale_y(), -0.1);

        // But band data and metadata should be different
        let target_band = target_raster.bands().band(1).unwrap();
        let target_band_meta = target_band.metadata();
        assert_eq!(target_band_meta.data_type().unwrap(), BandDataType::UInt16);
        assert!(target_band_meta.nodata_value().is_none());
        assert_eq!(
            target_band
                .nd_buffer()
                .unwrap()
                .as_contiguous()
                .unwrap()
                .len(),
            2016
        ); // 1008 * 2 bytes per u16

        let result = target_raster.bands().band(0);
        assert!(result.is_err(), "Band number 0 should be invalid");

        let result = target_raster.bands().band(2);
        assert!(result.is_err(), "Band number 2 should be out of range");
    }

    #[test]
    fn copy_raster_from_overrides_transform_and_preserves_bands() {
        use sedona_testing::raster_spec::{assert_rasters_equal, RasterSpec};

        // Source: a CRS, a nodata sentinel, and pixel values to preserve.
        let source = RasterSpec::d2(2, 1)
            .band_values(&[1u8, 2])
            .nodata(9u8)
            .crs(Some("OGC:CRS84"))
            .build();
        let source = RasterStructArray::try_new(&source).unwrap();

        let mut builder = RasterBuilder::new(1);
        builder
            .copy_raster_from(
                &source.get(0).unwrap(),
                RasterOverrides {
                    transform: Some([100.0, 2.0, 0.0, 200.0, 0.0, -3.0]),
                },
            )
            .unwrap();
        let out: ArrayRef = Arc::new(builder.finish().unwrap());

        // Only the transform changed; CRS, nodata, and pixels carried over.
        let expected = RasterSpec::d2(2, 1)
            .band_values(&[1u8, 2])
            .nodata(9u8)
            .crs(Some("OGC:CRS84"))
            .transform([100.0, 2.0, 0.0, 200.0, 0.0, -3.0]);
        assert_rasters_equal(&out, &[Some(expected)]);
    }

    #[test]
    fn test_band_data_types() {
        // Create a test raster with bands of different data types
        let mut builder = RasterBuilder::new(1);

        let metadata = RasterMetadata {
            width: 2,
            height: 2,
            upperleft_x: 0.0,
            upperleft_y: 0.0,
            scale_x: 1.0,
            scale_y: -1.0,
            skew_x: 0.0,
            skew_y: 0.0,
        };

        builder.start_raster(&metadata, None).unwrap();

        // Test all BandDataType variants
        let test_cases = vec![
            (BandDataType::UInt8, vec![1u8, 2u8, 3u8, 4u8]),
            (BandDataType::Int8, vec![255u8, 254u8, 253u8, 252u8]), // -1, -2, -3, -4 as i8
            (
                BandDataType::UInt16,
                vec![1u8, 0u8, 2u8, 0u8, 3u8, 0u8, 4u8, 0u8],
            ), // little-endian u16
            (
                BandDataType::Int16,
                vec![255u8, 255u8, 254u8, 255u8, 253u8, 255u8, 252u8, 255u8],
            ), // little-endian i16
            (
                BandDataType::UInt32,
                vec![
                    1u8, 0u8, 0u8, 0u8, 2u8, 0u8, 0u8, 0u8, 3u8, 0u8, 0u8, 0u8, 4u8, 0u8, 0u8, 0u8,
                ],
            ), // little-endian u32
            (
                BandDataType::Int32,
                vec![
                    255u8, 255u8, 255u8, 255u8, 254u8, 255u8, 255u8, 255u8, 253u8, 255u8, 255u8,
                    255u8, 252u8, 255u8, 255u8, 255u8,
                ],
            ), // little-endian i32
            (
                BandDataType::UInt64,
                vec![
                    1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    3u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 4u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                ],
            ), // little-endian u64
            (
                BandDataType::Int64,
                vec![
                    255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 254u8, 255u8, 255u8,
                    255u8, 255u8, 255u8, 255u8, 255u8, 253u8, 255u8, 255u8, 255u8, 255u8, 255u8,
                    255u8, 255u8, 252u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8, 255u8,
                ],
            ), // little-endian i64: -1, -2, -3, -4
            (
                BandDataType::Float32,
                vec![
                    0u8, 0u8, 128u8, 63u8, 0u8, 0u8, 0u8, 64u8, 0u8, 0u8, 64u8, 64u8, 0u8, 0u8,
                    128u8, 64u8,
                ],
            ), // little-endian f32: 1.0, 2.0, 3.0, 4.0
            (
                BandDataType::Float64,
                vec![
                    0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 240u8, 63u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    64u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 8u8, 64u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    16u8, 64u8,
                ],
            ), // little-endian f64: 1.0, 2.0, 3.0, 4.0
        ];

        for (expected_data_type, test_data) in test_cases {
            let band_metadata = BandMetadata {
                nodata_value: None,
                storage_type: StorageType::InDb,
                datatype: expected_data_type,
                outdb_url: None,
                outdb_band_id: None,
            };

            builder.start_band(band_metadata).unwrap();
            builder.band_data_writer().append_value(&test_data);
            builder.finish_band().unwrap();
        }

        builder.finish_raster().unwrap();
        let raster_array = builder.finish().unwrap();

        // Test the data type conversion for each band
        let iterator = RasterStructArray::try_new(&raster_array).unwrap();
        let raster = iterator.get(0).unwrap();
        let bands = raster.bands();

        assert_eq!(bands.len(), 10, "Expected 10 bands for all data types");

        // Verify each band returns the correct data type
        let expected_types = [
            BandDataType::UInt8,
            BandDataType::Int8,
            BandDataType::UInt16,
            BandDataType::Int16,
            BandDataType::UInt32,
            BandDataType::Int32,
            BandDataType::UInt64,
            BandDataType::Int64,
            BandDataType::Float32,
            BandDataType::Float64,
        ];

        // i is zero-based index
        for (i, expected_type) in expected_types.iter().enumerate() {
            // Bands are 1-based band_number
            let band = bands.band(i + 1).unwrap();
            let band_metadata = band.metadata();
            let actual_type = band_metadata.data_type().unwrap();

            assert_eq!(
                actual_type, *expected_type,
                "Band {i} expected data type {expected_type:?}, got {actual_type:?}"
            );
        }
    }

    #[test]
    fn test_outdb_metadata_fields() {
        // Test creating raster with OutDb reference metadata
        let mut builder = RasterBuilder::new(10);

        let metadata = RasterMetadata {
            width: 1024,
            height: 1024,
            upperleft_x: 0.0,
            upperleft_y: 0.0,
            scale_x: 1.0,
            scale_y: -1.0,
            skew_x: 0.0,
            skew_y: 0.0,
        };

        builder.start_raster(&metadata, None).unwrap();

        // Test InDb band (should have null OutDb fields)
        let indb_band_metadata = BandMetadata {
            nodata_value: Some(vec![255u8]),
            storage_type: StorageType::InDb,
            datatype: BandDataType::UInt8,
            outdb_url: None,
            outdb_band_id: None,
        };

        builder.start_band(indb_band_metadata).unwrap();
        let test_data = vec![1u8; 100];
        builder.band_data_writer().append_value(&test_data);
        builder.finish_band().unwrap();

        // Test OutDbRef band (should have OutDb fields populated)
        let outdb_band_metadata = BandMetadata {
            nodata_value: None,
            storage_type: StorageType::OutDbRef,
            datatype: BandDataType::Float32,
            outdb_url: Some("s3://mybucket/satellite_image.tif".to_string()),
            outdb_band_id: Some(2),
        };

        builder.start_band(outdb_band_metadata).unwrap();
        // For OutDbRef, data field could be empty or contain metadata/thumbnail
        builder.band_data_writer().append_value([]);
        builder.finish_band().unwrap();

        builder.finish_raster().unwrap();
        let raster_array = builder.finish().unwrap();

        // Verify the band metadata
        let iterator = RasterStructArray::try_new(&raster_array).unwrap();
        let raster = iterator.get(0).unwrap();
        let bands = raster.bands();

        assert_eq!(bands.len(), 2);

        // Test InDb band
        let indb_band = bands.band(1).unwrap();
        let indb_metadata = indb_band.metadata();
        assert_eq!(indb_metadata.storage_type().unwrap(), StorageType::InDb);
        assert_eq!(indb_metadata.data_type().unwrap(), BandDataType::UInt8);
        assert!(indb_metadata.outdb_url().is_none());
        assert!(indb_metadata.outdb_band_id().is_none());
        assert!(indb_band.is_indb());

        // Test OutDbRef band
        let outdb_band = bands.band(2).unwrap();
        let outdb_metadata = outdb_band.metadata();
        assert_eq!(
            outdb_metadata.storage_type().unwrap(),
            StorageType::OutDbRef
        );
        assert_eq!(outdb_metadata.data_type().unwrap(), BandDataType::Float32);
        assert_eq!(
            outdb_metadata.outdb_url().unwrap(),
            "s3://mybucket/satellite_image.tif"
        );
        assert_eq!(outdb_metadata.outdb_band_id().unwrap(), 2);
        assert!(!outdb_band.is_indb());
    }

    #[test]
    fn test_band_access_errors() {
        // Create a simple raster with one band
        let mut builder = RasterBuilder::new(1);

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

        builder.start_raster(&metadata, None).unwrap();

        let band_metadata = BandMetadata {
            nodata_value: None,
            storage_type: StorageType::InDb,
            datatype: BandDataType::UInt8,
            outdb_url: None,
            outdb_band_id: None,
        };

        builder.start_band(band_metadata).unwrap();
        builder.band_data_writer().append_value([1u8; 100]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        let raster_array = builder.finish().unwrap();
        let iterator = RasterStructArray::try_new(&raster_array).unwrap();
        let raster = iterator.get(0).unwrap();
        let bands = raster.bands();

        // Test invalid band number (0-based)
        let result = bands.band(0);
        assert!(result.is_err());
        let err = result.err().unwrap().to_string();
        assert!(err.contains("band numbers must be 1-based"));

        // Test out of range band number
        let result = bands.band(2);
        assert!(result.is_err());
        let err = result.err().unwrap().to_string();
        assert!(err.contains("is out of range"));

        // Test valid band number should still work
        let result = bands.band(1);
        assert!(result.is_ok());
        let band = result.unwrap();
        assert_eq!(
            band.nd_buffer().unwrap().as_contiguous().unwrap().len(),
            100
        );
    }

    #[test]
    fn test_roundtrip_2d_raster() {
        let mut builder = RasterBuilder::new(1);
        builder
            .start_raster_2d(
                10,
                20,
                100.0,
                200.0,
                1.0,
                -2.0,
                0.25,
                0.5,
                Some("EPSG:4326"),
            )
            .unwrap();
        builder
            .start_band_2d(BandDataType::UInt8, Some(&[255u8]))
            .unwrap();
        builder.band_data_writer().append_value(vec![1u8; 200]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::try_new(&array).unwrap();
        assert_eq!(rasters.len(), 1);

        let r = rasters.get(0).unwrap();
        assert_eq!(r.width().unwrap(), 10);
        assert_eq!(r.height().unwrap(), 20);
        assert_eq!(r.transform(), &[100.0, 1.0, 0.25, 200.0, 0.5, -2.0]);
        assert_eq!(r.x_dim(), "x");
        assert_eq!(r.y_dim(), "y");
        assert_eq!(r.crs(), Some("EPSG:4326"));
        assert_eq!(r.num_bands(), 1);

        let band = r.band(0).unwrap();
        assert_eq!(band.ndim(), 2);
        assert_eq!(band.dim_names(), vec!["y", "x"]);
        assert_eq!(band.shape(), &[20, 10]);
        assert_eq!(band.data_type(), BandDataType::UInt8);
        assert_eq!(band.nodata(), Some(&[255u8][..]));
        assert_eq!(
            band.nd_buffer().unwrap().as_contiguous().unwrap().len(),
            200
        );
    }

    #[test]
    fn test_roundtrip_multi_band() {
        let mut builder = RasterBuilder::new(1);
        builder
            .start_raster_2d(2, 2, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, None)
            .unwrap();

        // Band 0: UInt8
        builder
            .start_band_2d(BandDataType::UInt8, Some(&[255u8]))
            .unwrap();
        builder.band_data_writer().append_value([1u8, 2, 3, 4]);
        builder.finish_band().unwrap();

        // Band 1: Float32
        builder.start_band_2d(BandDataType::Float32, None).unwrap();
        let f32_data: Vec<u8> = [1.5f32, 2.5, 3.5, 4.5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        builder.band_data_writer().append_value(&f32_data);
        builder.finish_band().unwrap();

        builder.finish_raster().unwrap();
        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::try_new(&array).unwrap();
        let r = rasters.get(0).unwrap();

        assert_eq!(r.num_bands(), 2);

        let b0 = r.band(0).unwrap();
        assert_eq!(b0.data_type(), BandDataType::UInt8);
        assert_eq!(b0.nodata(), Some(&[255u8][..]));

        let b1 = r.band(1).unwrap();
        assert_eq!(b1.data_type(), BandDataType::Float32);
        assert_eq!(b1.nodata(), None);
    }

    #[test]
    fn test_null_raster() {
        let mut builder = RasterBuilder::new(2);
        builder
            .start_raster_2d(1, 1, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, None)
            .unwrap();
        builder.start_band_2d(BandDataType::UInt8, None).unwrap();
        builder.band_data_writer().append_value([0u8]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        builder.append_null().unwrap();

        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::try_new(&array).unwrap();
        assert_eq!(rasters.len(), 2);
        assert!(!rasters.is_null(0));
        assert!(rasters.is_null(1));
    }

    #[test]
    fn test_nd_band() {
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster_nd(&transform, &["x", "y"], &[5, 4], None)
            .unwrap();

        // 3D band: [time=3, y=4, x=5]
        builder
            .start_band_nd(
                Some("temperature"),
                &["time", "y", "x"],
                &[3, 4, 5],
                BandDataType::Float32,
                None,
                None,
                None,
            )
            .unwrap();
        let data = vec![0u8; 3 * 4 * 5 * 4]; // 3*4*5 Float32 elements
        builder.band_data_writer().append_value(&data);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::try_new(&array).unwrap();
        let r = rasters.get(0).unwrap();

        assert_eq!(r.band_name(0), Some("temperature"));
        let band = r.band(0).unwrap();
        assert_eq!(band.ndim(), 3);
        assert_eq!(band.dim_names(), vec!["time", "y", "x"]);
        assert_eq!(band.shape(), &[3, 4, 5]);
        assert_eq!(band.dim_size("time"), Some(3));
        assert_eq!(band.dim_size("y"), Some(4));
        assert_eq!(band.dim_size("x"), Some(5));
        assert_eq!(band.dim_size("z"), None);

        // Verify strides are standard C-order: [4*5*4, 5*4, 4] = [80, 20, 4]
        let buf = band.nd_buffer().unwrap();
        assert_eq!(buf.strides, &[80, 20, 4]);
        assert_eq!(buf.offset, 0);
    }

    #[test]
    fn test_nonstandard_spatial_dim_names() {
        // Zarr-style dataset with lat/lon instead of y/x
        let mut builder = RasterBuilder::new(1);
        let transform = [10.0, 0.01, 0.0, 50.0, 0.0, -0.01];
        builder
            .start_raster_nd(
                &transform,
                &["longitude", "latitude"],
                &[360, 180],
                Some("EPSG:4326"),
            )
            .unwrap();
        builder
            .start_band_nd(
                Some("sst"),
                &["latitude", "longitude"],
                &[180, 360],
                BandDataType::Float32,
                None,
                None,
                None,
            )
            .unwrap();
        let data = vec![0u8; 180 * 360 * 4];
        builder.band_data_writer().append_value(&data);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::try_new(&array).unwrap();
        let r = rasters.get(0).unwrap();

        assert_eq!(r.x_dim(), "longitude");
        assert_eq!(r.y_dim(), "latitude");
        // width = size of "longitude" dim, height = size of "latitude" dim
        assert_eq!(r.width().unwrap(), 360);
        assert_eq!(r.height().unwrap(), 180);
    }

    #[test]
    fn test_mixed_dimensionality_bands() {
        // One 3D band and one 2D band in the same raster
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster_nd(&transform, &["x", "y"], &[64, 64], None)
            .unwrap();

        // Band 0: 3D [time=12, y=64, x=64]
        builder
            .start_band_nd(
                Some("temperature"),
                &["time", "y", "x"],
                &[12, 64, 64],
                BandDataType::Float32,
                None,
                None,
                None,
            )
            .unwrap();
        let data_3d = vec![0u8; 12 * 64 * 64 * 4];
        builder.band_data_writer().append_value(&data_3d);
        builder.finish_band().unwrap();

        // Band 1: 2D [y=64, x=64]
        builder
            .start_band_nd(
                Some("elevation"),
                &["y", "x"],
                &[64, 64],
                BandDataType::Float64,
                None,
                None,
                None,
            )
            .unwrap();
        let data_2d = vec![0u8; 64 * 64 * 8];
        builder.band_data_writer().append_value(&data_2d);
        builder.finish_band().unwrap();

        builder.finish_raster().unwrap();
        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::try_new(&array).unwrap();
        let r = rasters.get(0).unwrap();

        assert_eq!(r.num_bands(), 2);
        // width/height derived from band(0) which is 3D
        assert_eq!(r.width().unwrap(), 64);
        assert_eq!(r.height().unwrap(), 64);

        let b0 = r.band(0).unwrap();
        assert_eq!(b0.ndim(), 3);
        assert_eq!(b0.dim_names(), vec!["time", "y", "x"]);
        assert_eq!(b0.shape(), &[12, 64, 64]);
        assert_eq!(b0.dim_size("time"), Some(12));

        let b1 = r.band(1).unwrap();
        assert_eq!(b1.ndim(), 2);
        assert_eq!(b1.dim_names(), vec!["y", "x"]);
        assert_eq!(b1.shape(), &[64, 64]);
        assert_eq!(b1.dim_size("time"), None);
    }

    #[test]
    fn test_dim_index_lookup() {
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster_nd(&transform, &["x", "y"], &[32, 32], None)
            .unwrap();
        builder
            .start_band_nd(
                None,
                &["time", "pressure", "y", "x"],
                &[6, 10, 32, 32],
                BandDataType::Float32,
                None,
                None,
                None,
            )
            .unwrap();
        let data = vec![0u8; 6 * 10 * 32 * 32 * 4];
        builder.band_data_writer().append_value(&data);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::try_new(&array).unwrap();
        let r = rasters.get(0).unwrap();
        let band = r.band(0).unwrap();

        assert_eq!(band.dim_index("time"), Some(0));
        assert_eq!(band.dim_index("pressure"), Some(1));
        assert_eq!(band.dim_index("y"), Some(2));
        assert_eq!(band.dim_index("x"), Some(3));
        assert_eq!(band.dim_index("wavelength"), None);

        assert_eq!(band.dim_size("time"), Some(6));
        assert_eq!(band.dim_size("pressure"), Some(10));
        assert_eq!(band.dim_size("wavelength"), None);
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
        let rasters = RasterStructArray::try_new(&array).unwrap();
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
        let rasters = RasterStructArray::try_new(&array).unwrap();

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
    fn test_width_height_no_bands() {
        // Zero-band raster — used as a "target grid" specification (GDAL warp
        // pattern). Width/height come from the top-level spatial_shape, not
        // band(0).
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster_nd(&transform, &["x", "y"], &[64, 32], None)
            .unwrap();
        builder.finish_raster().unwrap();

        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::try_new(&array).unwrap();
        let r = rasters.get(0).unwrap();

        assert_eq!(r.num_bands(), 0);
        assert_eq!(r.width().unwrap(), 64);
        assert_eq!(r.height().unwrap(), 32);
    }

    #[test]
    fn test_band_name_nullable() {
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster_nd(&transform, &["x", "y"], &[4, 4], None)
            .unwrap();

        // Named band
        builder
            .start_band_nd(
                Some("temperature"),
                &["y", "x"],
                &[4, 4],
                BandDataType::Float32,
                None,
                None,
                None,
            )
            .unwrap();
        builder.band_data_writer().append_value(vec![0u8; 64]);
        builder.finish_band().unwrap();

        // Unnamed band (via start_band_2d which passes None for name)
        builder.current_width = 4;
        builder.current_height = 4;
        builder.start_band_2d(BandDataType::UInt8, None).unwrap();
        builder.band_data_writer().append_value(vec![0u8; 16]);
        builder.finish_band().unwrap();

        builder.finish_raster().unwrap();
        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::try_new(&array).unwrap();
        let r = rasters.get(0).unwrap();

        assert_eq!(r.band_name(0), Some("temperature"));
        assert_eq!(r.band_name(1), None); // unnamed
        assert_eq!(r.band_name(99), None); // out of range
    }

    #[test]
    fn test_spatial_dims_shape_roundtrip() {
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster_nd(&transform, &["longitude", "latitude"], &[360, 180], None)
            .unwrap();
        builder
            .start_band_nd(
                None,
                &["latitude", "longitude"],
                &[180, 360],
                BandDataType::UInt8,
                None,
                None,
                None,
            )
            .unwrap();
        builder
            .band_data_writer()
            .append_value(vec![0u8; 360 * 180]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::try_new(&array).unwrap();
        let r = rasters.get(0).unwrap();

        assert_eq!(r.spatial_dims(), vec!["longitude", "latitude"]);
        assert_eq!(r.spatial_shape(), &[360, 180]);
        assert_eq!(r.x_dim(), "longitude");
        assert_eq!(r.y_dim(), "latitude");
        assert_eq!(r.width().unwrap(), 360);
        assert_eq!(r.height().unwrap(), 180);
    }

    #[test]
    fn test_zero_band_raster_roundtrip() {
        // Zero-band rasters double as "target grid" specifications. They must
        // round-trip through the builder cleanly.
        let mut builder = RasterBuilder::new(1);
        let transform = [10.0, 1.0, 0.0, 20.0, 0.0, -1.0];
        builder
            .start_raster_nd(&transform, &["x", "y"], &[128, 64], Some("EPSG:3857"))
            .unwrap();
        builder.finish_raster().unwrap();

        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::try_new(&array).unwrap();
        let r = rasters.get(0).unwrap();

        assert_eq!(r.num_bands(), 0);
        assert_eq!(r.spatial_dims(), vec!["x", "y"]);
        assert_eq!(r.spatial_shape(), &[128, 64]);
        assert_eq!(r.width().unwrap(), 128);
        assert_eq!(r.height().unwrap(), 64);
        assert_eq!(r.crs(), Some("EPSG:3857"));
    }

    #[test]
    fn test_band_missing_spatial_dim_errors() {
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster_nd(&transform, &["x", "y"], &[4, 4], None)
            .unwrap();
        // Band is missing "y" entirely.
        builder
            .start_band_nd(None, &["x"], &[4], BandDataType::UInt8, None, None, None)
            .unwrap();
        builder.band_data_writer().append_value(vec![0u8; 4]);
        builder.finish_band().unwrap();

        let err = builder.finish_raster().unwrap_err();
        assert!(
            err.to_string().contains("missing spatial dimension"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_start_band_rejects_zero_dim() {
        // 0-D bands carry no spatial extent and no caller has a use for
        // them. start_band_nd must reject an empty dim_names slice eagerly so
        // the malformed band never reaches the buffer layer.
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder.start_raster_nd(&transform, &[], &[], None).unwrap();
        let err = builder
            .start_band_nd(None, &[], &[], BandDataType::UInt8, None, None, None)
            .unwrap_err();
        assert!(
            err.to_string().contains("0-dimensional"),
            "unexpected error: {err}"
        );
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
        let rasters = RasterStructArray::try_new(&array).unwrap();
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
    fn test_view_field_is_null_for_identity_band() {
        // Schema invariant: identity views are stored as null list rows so
        // the canonical "no slice" case costs no Arrow space. Confirm by
        // poking the raw column.
        use arrow_array::Array;

        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster_nd(&transform, &["x", "y"], &[2, 2], None)
            .unwrap();
        builder
            .start_band_nd(
                None,
                &["y", "x"],
                &[2, 2],
                BandDataType::UInt8,
                None,
                None,
                None,
            )
            .unwrap();
        builder.band_data_writer().append_value(vec![0u8; 4]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        let array = builder.finish().unwrap();
        let bands_list = array
            .column(sedona_schema::raster::raster_indices::BANDS)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let bands_struct = bands_list
            .values()
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let view_list = bands_struct
            .column(sedona_schema::raster::band_indices::VIEW)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        assert_eq!(view_list.len(), 1);
        assert!(
            view_list.is_null(0),
            "identity-view band should serialise as a null view row"
        );
    }

    #[test]
    fn test_band_spatial_dim_size_mismatch_errors() {
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster_nd(&transform, &["x", "y"], &[4, 4], None)
            .unwrap();
        // Band has "x" and "y" but x-size disagrees with top-level shape.
        builder
            .start_band_nd(
                None,
                &["y", "x"],
                &[4, 8],
                BandDataType::UInt8,
                None,
                None,
                None,
            )
            .unwrap();
        builder.band_data_writer().append_value(vec![0u8; 32]);
        builder.finish_band().unwrap();

        let err = builder.finish_raster().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("has size 8") && msg.contains("expected 4"),
            "unexpected error: {msg}"
        );
    }

    #[test]
    fn test_view_null_round_trips_through_arrow_ipc() {
        // Schema invariant: a band built via start_band_nd serialises with a
        // null view row, and the null must survive an Arrow IPC round-trip.
        // If a future change accidentally writes a non-null empty list
        // instead, downstream readers (DuckDB, PyArrow, sedona-py) will
        // disagree about whether the view is identity.

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
        builder.band_data_writer().append_value(vec![0u8; 6]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        let array = builder.finish().unwrap();
        let schema = Arc::new(Schema::new(vec![Arc::new(arrow_schema::Field::new(
            "raster",
            array.data_type().clone(),
            true,
        )) as arrow_schema::FieldRef]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(array.clone())]).unwrap();

        let mut buf: Vec<u8> = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut buf, schema.as_ref()).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
        }

        let cursor = Cursor::new(buf);
        let reader = StreamReader::try_new(cursor, None).unwrap();
        let batches: Vec<_> = reader.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(batches.len(), 1);
        let restored_struct = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();

        let bands_list = restored_struct
            .column(sedona_schema::raster::raster_indices::BANDS)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let bands_struct = bands_list
            .values()
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let view_list = bands_struct
            .column(sedona_schema::raster::band_indices::VIEW)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        assert_eq!(view_list.len(), 1);
        assert!(
            view_list.is_null(0),
            "identity-view band must remain a null view row after IPC round-trip"
        );

        let rasters = RasterStructArray::try_new(restored_struct).unwrap();
        let r0 = rasters.get(0).unwrap();
        assert_eq!(r0.band(0).unwrap().shape(), &[2, 3]);
    }

    /// Navigate an output raster `StructArray` to its bands' `data`
    /// `BinaryViewArray` column.
    fn output_band_data(arr: &StructArray) -> &BinaryViewArray {
        use sedona_schema::raster::{band_indices, raster_indices};
        arr.column(raster_indices::BANDS)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap()
            .values()
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap()
            .column(band_indices::DATA)
            .as_any()
            .downcast_ref::<BinaryViewArray>()
            .unwrap()
    }

    #[test]
    fn append_band_data_buffer_borrows_source_zero_copy() {
        let bytes: Vec<u8> = (10u8..24).collect(); // 14 bytes (> inline threshold)
        let src = Buffer::from_vec(bytes.clone());
        let src_ptr = src.as_ptr();

        let mut builder = RasterBuilder::new(1);
        builder
            .start_raster_2d(7, 2, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, None)
            .unwrap();
        builder.start_band_2d(BandDataType::UInt8, None).unwrap();
        builder
            .append_band_data_buffer(&src, 0, bytes.len() as u32)
            .unwrap();
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        let arr = builder.finish().unwrap();

        let rasters = RasterStructArray::try_new(&arr).unwrap();
        let r = rasters.get(0).unwrap();
        let band = r.band(0).unwrap();
        let out = band.nd_buffer().unwrap().as_contiguous().unwrap();
        assert_eq!(out, bytes.as_slice());
        // Zero-copy: the output borrows the source allocation, not a copy.
        assert_eq!(out.as_ptr(), src_ptr);
    }

    #[test]
    fn append_band_data_buffer_dedups_shared_buffer() {
        // Two bands carved from the same backing buffer attach it once. Each
        // slice is > 12 bytes so it's block-backed (the inline path attaches
        // no block).
        let bytes: Vec<u8> = (0u8..26).collect();
        let src = Buffer::from_vec(bytes.clone());

        let mut builder = RasterBuilder::new(1);
        builder
            .start_raster_2d(13, 1, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, None)
            .unwrap();
        builder.start_band_2d(BandDataType::UInt8, None).unwrap();
        builder.append_band_data_buffer(&src, 0, 13).unwrap();
        builder.finish_band().unwrap();
        builder.start_band_2d(BandDataType::UInt8, None).unwrap();
        builder.append_band_data_buffer(&src, 13, 13).unwrap();
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        let arr = builder.finish().unwrap();

        // Dedup: one shared data block, not two.
        assert_eq!(output_band_data(&arr).data_buffers().len(), 1);

        let rasters = RasterStructArray::try_new(&arr).unwrap();
        let r = rasters.get(0).unwrap();
        assert_eq!(
            r.band(0)
                .unwrap()
                .nd_buffer()
                .unwrap()
                .as_contiguous()
                .unwrap(),
            &bytes[0..13]
        );
        assert_eq!(
            r.band(1)
                .unwrap()
                .nd_buffer()
                .unwrap()
                .as_contiguous()
                .unwrap(),
            &bytes[13..26]
        );
    }

    #[test]
    fn append_band_data_buffer_interleaves_with_append_value() {
        // One band via append_value (in-progress buffer), one via a borrowed
        // block — the view block-indexing must stay correct across the mix.
        // Both bands are > 12 bytes so both are block-backed.
        let src = Buffer::from_vec((100u8..113).collect::<Vec<_>>()); // 13 bytes
        let band0: Vec<u8> = (1u8..14).collect(); // 13 bytes
        let mut builder = RasterBuilder::new(1);
        builder
            .start_raster_2d(13, 1, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, None)
            .unwrap();
        builder.start_band_2d(BandDataType::UInt8, None).unwrap();
        builder.band_data_writer().append_value(&band0);
        builder.finish_band().unwrap();
        builder.start_band_2d(BandDataType::UInt8, None).unwrap();
        builder.append_band_data_buffer(&src, 0, 13).unwrap();
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        let arr = builder.finish().unwrap();

        let rasters = RasterStructArray::try_new(&arr).unwrap();
        let r = rasters.get(0).unwrap();
        assert_eq!(
            r.band(0)
                .unwrap()
                .nd_buffer()
                .unwrap()
                .as_contiguous()
                .unwrap(),
            band0.as_slice()
        );
        assert_eq!(
            r.band(1)
                .unwrap()
                .nd_buffer()
                .unwrap()
                .as_contiguous()
                .unwrap(),
            (100u8..113).collect::<Vec<_>>().as_slice()
        );
    }

    #[test]
    fn append_band_data_from_shares_block_backed_row() {
        // Source raster with block-backed band data (> 12 bytes); copying that
        // row into a new raster must borrow the same backing buffer.
        let bytes: Vec<u8> = (20u8..34).collect(); // 14 bytes
        let mut src_builder = RasterBuilder::new(1);
        src_builder
            .start_raster_2d(7, 2, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, None)
            .unwrap();
        src_builder
            .start_band_2d(BandDataType::UInt8, None)
            .unwrap();
        src_builder.band_data_writer().append_value(&bytes);
        src_builder.finish_band().unwrap();
        src_builder.finish_raster().unwrap();
        let src_arr = src_builder.finish().unwrap();
        let src_data = output_band_data(&src_arr);
        let src_ptr = src_data.value(0).as_ptr();

        let mut builder = RasterBuilder::new(1);
        builder
            .start_raster_2d(7, 2, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, None)
            .unwrap();
        builder.start_band_2d(BandDataType::UInt8, None).unwrap();
        builder.append_band_data_from(src_data, 0).unwrap();
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        let arr = builder.finish().unwrap();

        let rasters = RasterStructArray::try_new(&arr).unwrap();
        let r = rasters.get(0).unwrap();
        let band = r.band(0).unwrap();
        let out = band.nd_buffer().unwrap().as_contiguous().unwrap();
        assert_eq!(out, bytes.as_slice());
        assert_eq!(out.as_ptr(), src_ptr); // zero-copy: same allocation
    }

    #[test]
    fn append_band_data_from_copies_inline_row() {
        // Inline rows (<= 12 bytes) have no backing buffer, so they're copied;
        // verify correctness of that path.
        let bytes: Vec<u8> = vec![1, 2, 3, 4, 5, 6]; // 6 bytes (inline)
        let mut src_builder = RasterBuilder::new(1);
        src_builder
            .start_raster_2d(6, 1, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, None)
            .unwrap();
        src_builder
            .start_band_2d(BandDataType::UInt8, None)
            .unwrap();
        src_builder.band_data_writer().append_value(&bytes);
        src_builder.finish_band().unwrap();
        src_builder.finish_raster().unwrap();
        let src_arr = src_builder.finish().unwrap();
        let src_data = output_band_data(&src_arr);

        let mut builder = RasterBuilder::new(1);
        builder
            .start_raster_2d(6, 1, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, None)
            .unwrap();
        builder.start_band_2d(BandDataType::UInt8, None).unwrap();
        builder.append_band_data_from(src_data, 0).unwrap();
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        let arr = builder.finish().unwrap();

        let rasters = RasterStructArray::try_new(&arr).unwrap();
        assert_eq!(
            rasters
                .get(0)
                .unwrap()
                .band(0)
                .unwrap()
                .nd_buffer()
                .unwrap()
                .as_contiguous()
                .unwrap(),
            bytes.as_slice()
        );
    }

    #[test]
    fn append_band_data_buffer_inlines_small_slice() {
        // A <= 12-byte slice must be stored inline (no data block), since a
        // block-referencing view of that size is non-canonical.
        let bytes: Vec<u8> = vec![1, 2, 3, 4, 5, 6]; // 6 bytes (inline)
        let src = Buffer::from_vec(bytes.clone());

        let mut builder = RasterBuilder::new(1);
        builder
            .start_raster_2d(6, 1, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, None)
            .unwrap();
        builder.start_band_2d(BandDataType::UInt8, None).unwrap();
        builder.append_band_data_buffer(&src, 0, 6).unwrap();
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        let arr = builder.finish().unwrap();

        // Inline: no backing block attached.
        assert_eq!(output_band_data(&arr).data_buffers().len(), 0);

        let rasters = RasterStructArray::try_new(&arr).unwrap();
        let r = rasters.get(0).unwrap();
        let band = r.band(0).unwrap();
        assert_eq!(
            band.nd_buffer().unwrap().as_contiguous().unwrap(),
            bytes.as_slice()
        );
    }
}
