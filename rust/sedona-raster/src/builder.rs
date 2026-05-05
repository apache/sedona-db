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
        Int64Builder, StringBuilder, StringViewBuilder, UInt32Builder, UInt64Builder,
    },
    Array, ArrayRef, ListArray, StructArray,
};
use arrow_buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::ArrowError;
use std::sync::Arc;

use crate::traits::ViewEntry;
use sedona_schema::raster::BandDataType;
use sedona_schema::raster::RasterSchema;

use arrow_schema::DataType;

/// Builder for constructing N-D raster arrays.
///
/// # Usage
///
/// ```
/// use sedona_raster::builder::RasterBuilder;
/// use sedona_schema::raster::BandDataType;
///
/// let mut builder = RasterBuilder::new(1);
///
/// // 2D raster convenience: sets transform, spatial_dims=["x","y"], spatial_shape=[w,h]
/// builder.start_raster_2d(100, 100, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, Some("EPSG:4326")).unwrap();
///
/// // 2D band convenience: sets dim_names=["y","x"], shape=[h,w], contiguous strides
/// builder.start_band_2d(BandDataType::UInt8, Some(&[0u8])).unwrap();
/// builder.band_data_writer().append_value(&vec![0u8; 10000]);
/// builder.finish_band().unwrap();
/// builder.finish_raster().unwrap();
///
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
    band_shape_values: UInt64Builder,
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
    current_width: u64,
    current_height: u64,

    // Per-raster validation state: spatial dims/shape and recorded bands so
    // finish_raster can check every band matches the top-level spatial grid.
    current_spatial_dims: Vec<String>,
    current_spatial_shape: Vec<i64>,
    current_raster_bands: Vec<(Vec<String>, Vec<u64>)>,

    // Track band_data count at the start of each band for finish_band validation
    band_data_count_at_start: usize,

    raster_validity: BooleanBuilder,
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
            band_shape_values: UInt64Builder::with_capacity(capacity * 2),
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
    pub fn start_raster(
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
        // their own shapes via start_band.
        self.current_width = 0;
        self.current_height = 0;

        Ok(())
    }

    /// Convenience: start a 2D raster with the legacy 8-parameter interface.
    ///
    /// Sets `spatial_dims=["x","y"]`, `spatial_shape=[width, height]`, and
    /// builds the 6-element GDAL transform from the individual parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn start_raster_2d(
        &mut self,
        width: u64,
        height: u64,
        origin_x: f64,
        origin_y: f64,
        scale_x: f64,
        scale_y: f64,
        skew_x: f64,
        skew_y: f64,
        crs: Option<&str>,
    ) -> Result<(), ArrowError> {
        let transform = [origin_x, scale_x, skew_x, origin_y, skew_y, scale_y];
        self.start_raster(&transform, &["x", "y"], &[width as i64, height as i64], crs)?;
        self.current_width = width;
        self.current_height = height;
        Ok(())
    }

    /// Start a new band with explicit N-D parameters.
    ///
    /// `outdb_uri` is the *location* of the external resource (scheme is
    /// resolved by an `ObjectStoreRegistry`). `outdb_format` is the *format*
    /// used to interpret the bytes at that location (e.g. `"geotiff"`,
    /// `"zarr"`). A null `outdb_format` means the band is in-memory — the
    /// band's `data` buffer is authoritative.
    #[allow(clippy::too_many_arguments)]
    pub fn start_band(
        &mut self,
        name: Option<&str>,
        dim_names: &[&str],
        shape: &[u64],
        data_type: BandDataType,
        nodata: Option<&[u8]>,
        outdb_uri: Option<&str>,
        outdb_format: Option<&str>,
    ) -> Result<(), ArrowError> {
        if dim_names.is_empty() {
            return Err(ArrowError::InvalidArgumentError(
                "start_band: 0-dimensional bands are not supported".into(),
            ));
        }
        if dim_names.len() != shape.len() {
            return Err(ArrowError::InvalidArgumentError(format!(
                "start_band: dim_names ({}) and shape ({}) must have the same length",
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

    /// Start a band with an explicit non-identity view over `source_shape`.
    ///
    /// Each `ViewEntry` describes one *visible* axis in `dim_names` order:
    /// `(source_axis, start, step, steps)`. Validates that:
    /// - `dim_names`, `source_shape`, and `view` have equal length.
    /// - Across `view`, `source_axis` values form a permutation of
    ///   `0..ndim` (no axis duplicated, none missing).
    /// - For each entry with `steps > 0`: `start` and (when `step != 0`)
    ///   `start + (steps - 1) * step` are in `[0, source_shape[source_axis])`.
    /// - `steps >= 0`.
    ///
    /// On success, the band's `view` field is written verbatim and its
    /// `source_shape` is written from `source_shape`. The visible shape
    /// (== `[v.steps for v in view]`) is what `finish_raster` will compare
    /// against `spatial_shape`.
    #[allow(clippy::too_many_arguments)]
    pub fn start_band_with_view(
        &mut self,
        name: Option<&str>,
        dim_names: &[&str],
        source_shape: &[u64],
        view: &[ViewEntry],
        data_type: BandDataType,
        nodata: Option<&[u8]>,
        outdb_uri: Option<&str>,
        outdb_format: Option<&str>,
    ) -> Result<(), ArrowError> {
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

        // Permutation check on source_axis values + per-entry bound checks.
        let mut seen = vec![false; ndim];
        for (k, v) in view.iter().enumerate() {
            if v.source_axis < 0 || (v.source_axis as usize) >= ndim {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "view[{k}].source_axis = {} is out of range [0, {ndim})",
                    v.source_axis
                )));
            }
            let sa = v.source_axis as usize;
            if seen[sa] {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "view source_axis values must be a permutation of 0..{ndim}; \
                     axis {sa} appears more than once"
                )));
            }
            seen[sa] = true;

            if v.steps < 0 {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "view[{k}].steps = {} must be >= 0",
                    v.steps
                )));
            }
            if v.steps > 0 {
                let s = source_shape[sa] as i64;
                if v.start < 0 || v.start >= s {
                    return Err(ArrowError::InvalidArgumentError(format!(
                        "view[{k}].start = {} is out of range [0, {s}) for source axis {sa}",
                        v.start
                    )));
                }
                if v.step != 0 {
                    let last = v.start + (v.steps - 1) * v.step;
                    if last < 0 || last >= s {
                        return Err(ArrowError::InvalidArgumentError(format!(
                            "view[{k}] addresses element {last} which is out of range \
                             [0, {s}) for source axis {sa}"
                        )));
                    }
                }
            }
        }

        // Write fields.
        match name {
            Some(n) => self.band_name.append_value(n),
            None => self.band_name.append_null(),
        }

        for dn in dim_names {
            self.band_dim_names_values.append_value(dn);
        }
        let next = *self.band_dim_names_offsets.last().unwrap() + ndim as i32;
        self.band_dim_names_offsets.push(next);

        for &s in source_shape {
            self.band_shape_values.append_value(s);
        }
        let next = *self.band_shape_offsets.last().unwrap() + ndim as i32;
        self.band_shape_offsets.push(next);

        self.band_datatype.append_value(data_type as u32);

        match nodata {
            Some(b) => self.band_nodata.append_value(b),
            None => self.band_nodata.append_null(),
        }

        for v in view {
            self.band_view_source_axis_values
                .append_value(v.source_axis);
            self.band_view_start_values.append_value(v.start);
            self.band_view_step_values.append_value(v.step);
            self.band_view_steps_values.append_value(v.steps);
        }
        let next = *self.band_view_offsets.last().unwrap() + ndim as i32;
        self.band_view_offsets.push(next);
        self.band_view_validity.push(true);

        match outdb_uri {
            Some(uri) => self.band_outdb_uri.append_value(uri),
            None => self.band_outdb_uri.append_null(),
        }
        match outdb_format {
            Some(format) => self.band_outdb_format.append_value(format),
            None => self.band_outdb_format.append_null(),
        }

        self.current_band_count += 1;
        self.band_data_count_at_start = self.band_data.len();

        // finish_raster compares visible shape against spatial_shape.
        let visible_shape: Vec<u64> = view.iter().map(|v| v.steps as u64).collect();
        self.current_raster_bands.push((
            dim_names.iter().map(|s| s.to_string()).collect(),
            visible_shape,
        ));

        Ok(())
    }

    /// Convenience: start a 2D band with `dim_names=["y","x"]` and `shape=[height, width]`.
    ///
    /// Must be called after `start_raster_2d` which sets the current width/height.
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
        self.start_band(
            None,
            &["y", "x"],
            &[self.current_height, self.current_width],
            data_type,
            nodata,
            None,
            None,
        )
    }

    /// Get direct access to the BinaryViewBuilder for writing the current band's data.
    pub fn band_data_writer(&mut self) -> &mut BinaryViewBuilder {
        &mut self.band_data
    }

    /// Finish writing the current band.
    ///
    /// Validates that exactly one data value was appended since `start_band()`.
    pub fn finish_band(&mut self) -> Result<(), ArrowError> {
        let current_count = self.band_data.len();
        if current_count != self.band_data_count_at_start + 1 {
            return Err(ArrowError::InvalidArgumentError(
                format!(
                    "Expected exactly one band data value per band, but got {} appended since start_band()",
                    current_count - self.band_data_count_at_start
                ),
            ));
        }
        Ok(())
    }

    /// Finish all bands for the current raster.
    ///
    /// Strictly validates every band added since `start_raster`: each name in
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
                let actual = band_shape[pos] as i64;
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
    use crate::traits::RasterRef;

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
        let rasters = RasterStructArray::new(&array);
        assert_eq!(rasters.len(), 1);

        let r = rasters.get(0).unwrap();
        assert_eq!(r.width(), Some(10));
        assert_eq!(r.height(), Some(20));
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
        assert_eq!(band.contiguous_data().unwrap().len(), 200);
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
        let rasters = RasterStructArray::new(&array);
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
        let rasters = RasterStructArray::new(&array);
        assert_eq!(rasters.len(), 2);
        assert!(!rasters.is_null(0));
        assert!(rasters.is_null(1));
    }

    #[test]
    fn test_nd_band() {
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster(&transform, &["x", "y"], &[5, 4], None)
            .unwrap();

        // 3D band: [time=3, y=4, x=5]
        builder
            .start_band(
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
        let rasters = RasterStructArray::new(&array);
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
            .start_raster(
                &transform,
                &["longitude", "latitude"],
                &[360, 180],
                Some("EPSG:4326"),
            )
            .unwrap();
        builder
            .start_band(
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
        let rasters = RasterStructArray::new(&array);
        let r = rasters.get(0).unwrap();

        assert_eq!(r.x_dim(), "longitude");
        assert_eq!(r.y_dim(), "latitude");
        // width = size of "longitude" dim, height = size of "latitude" dim
        assert_eq!(r.width(), Some(360));
        assert_eq!(r.height(), Some(180));
    }

    #[test]
    fn test_mixed_dimensionality_bands() {
        // One 3D band and one 2D band in the same raster
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster(&transform, &["x", "y"], &[64, 64], None)
            .unwrap();

        // Band 0: 3D [time=12, y=64, x=64]
        builder
            .start_band(
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
            .start_band(
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
        let rasters = RasterStructArray::new(&array);
        let r = rasters.get(0).unwrap();

        assert_eq!(r.num_bands(), 2);
        // width/height derived from band(0) which is 3D
        assert_eq!(r.width(), Some(64));
        assert_eq!(r.height(), Some(64));

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
            .start_raster(&transform, &["x", "y"], &[32, 32], None)
            .unwrap();
        builder
            .start_band(
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
        let rasters = RasterStructArray::new(&array);
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
    fn test_contiguous_data_is_borrowed() {
        use std::borrow::Cow;

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

        let data = band.contiguous_data().unwrap();
        // Phase 1: all data is contiguous, so should be Cow::Borrowed
        assert!(matches!(data, Cow::Borrowed(_)));
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
            .start_raster(&transform, &["x", "y"], &[4, 3], None)
            .unwrap();
        builder
            .start_band(
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
            .start_raster(&transform, &["x", "y"], &[5, 3], None)
            .unwrap();
        builder
            .start_band(
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
            .start_raster(&transform, &["x"], &[10], None)
            .unwrap();
        builder
            .start_band(None, &["x"], &[10], BandDataType::UInt16, None, None, None)
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
    fn test_width_height_no_bands() {
        // Zero-band raster — used as a "target grid" specification (GDAL warp
        // pattern). Width/height come from the top-level spatial_shape, not
        // band(0).
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster(&transform, &["x", "y"], &[64, 32], None)
            .unwrap();
        builder.finish_raster().unwrap();

        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::new(&array);
        let r = rasters.get(0).unwrap();

        assert_eq!(r.num_bands(), 0);
        assert_eq!(r.width(), Some(64));
        assert_eq!(r.height(), Some(32));
    }

    #[test]
    fn test_band_name_nullable() {
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster(&transform, &["x", "y"], &[4, 4], None)
            .unwrap();

        // Named band
        builder
            .start_band(
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
        let rasters = RasterStructArray::new(&array);
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
            .start_raster(&transform, &["longitude", "latitude"], &[360, 180], None)
            .unwrap();
        builder
            .start_band(
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
        let rasters = RasterStructArray::new(&array);
        let r = rasters.get(0).unwrap();

        assert_eq!(r.spatial_dims(), vec!["longitude", "latitude"]);
        assert_eq!(r.spatial_shape(), &[360, 180]);
        assert_eq!(r.x_dim(), "longitude");
        assert_eq!(r.y_dim(), "latitude");
        assert_eq!(r.width(), Some(360));
        assert_eq!(r.height(), Some(180));
    }

    #[test]
    fn test_zero_band_raster_roundtrip() {
        // Zero-band rasters double as "target grid" specifications. They must
        // round-trip through the builder cleanly.
        let mut builder = RasterBuilder::new(1);
        let transform = [10.0, 1.0, 0.0, 20.0, 0.0, -1.0];
        builder
            .start_raster(&transform, &["x", "y"], &[128, 64], Some("EPSG:3857"))
            .unwrap();
        builder.finish_raster().unwrap();

        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::new(&array);
        let r = rasters.get(0).unwrap();

        assert_eq!(r.num_bands(), 0);
        assert_eq!(r.spatial_dims(), vec!["x", "y"]);
        assert_eq!(r.spatial_shape(), &[128, 64]);
        assert_eq!(r.width(), Some(128));
        assert_eq!(r.height(), Some(64));
        assert_eq!(r.crs(), Some("EPSG:3857"));
    }

    #[test]
    fn test_band_missing_spatial_dim_errors() {
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster(&transform, &["x", "y"], &[4, 4], None)
            .unwrap();
        // Band is missing "y" entirely.
        builder
            .start_band(None, &["x"], &[4], BandDataType::UInt8, None, None, None)
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
    fn test_start_band_with_view_identity_matches_start_band() {
        // Identity view through start_band_with_view should produce the same
        // visible shape and byte strides as the convenience start_band path.
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster(&transform, &["x", "y"], &[5, 4], None)
            .unwrap();

        let view = [
            ViewEntry {
                source_axis: 0,
                start: 0,
                step: 1,
                steps: 4,
            },
            ViewEntry {
                source_axis: 1,
                start: 0,
                step: 1,
                steps: 5,
            },
        ];
        builder
            .start_band_with_view(
                None,
                &["y", "x"],
                &[4, 5],
                &view,
                BandDataType::UInt8,
                None,
                None,
                None,
            )
            .unwrap();
        builder.band_data_writer().append_value(vec![0u8; 20]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::new(&array);
        let r = rasters.get(0).unwrap();
        let band = r.band(0).unwrap();
        assert_eq!(band.shape(), &[4, 5]);
        assert_eq!(band.raw_source_shape(), &[4, 5]);
        let buf = band.nd_buffer().unwrap();
        assert_eq!(buf.strides, &[5, 1]);
        assert_eq!(buf.offset, 0);
    }

    #[test]
    fn test_view_slice_nd_buffer_and_contiguous_data() {
        // 1D source of size 8 (UInt8), view (start=1, step=2, steps=3) selects
        // elements at byte offsets 1, 3, 5. Source: 0..8.
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

        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::new(&array);
        let r = rasters.get(0).unwrap();
        let band = r.band(0).unwrap();

        assert_eq!(band.shape(), &[3]);
        assert_eq!(band.raw_source_shape(), &[8]);
        let buf = band.nd_buffer().unwrap();
        assert_eq!(buf.shape, &[3]);
        assert_eq!(buf.strides, &[2]);
        assert_eq!(buf.offset, 1);

        // Materialised contiguous bytes should be [1, 3, 5].
        let bytes = band.contiguous_data().unwrap();
        assert_eq!(&*bytes, &[1u8, 3, 5]);
        assert!(matches!(bytes, std::borrow::Cow::Owned(_)));
    }

    #[test]
    fn test_view_broadcast() {
        // Broadcast: source size 1, step=0 → expose the same byte 4 times.
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster(&transform, &["x"], &[4], None)
            .unwrap();

        let view = [ViewEntry {
            source_axis: 0,
            start: 0,
            step: 0,
            steps: 4,
        }];
        builder
            .start_band_with_view(
                None,
                &["x"],
                &[1],
                &view,
                BandDataType::UInt8,
                None,
                None,
                None,
            )
            .unwrap();
        builder.band_data_writer().append_value(vec![42u8]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::new(&array);
        let r = rasters.get(0).unwrap();
        let band = r.band(0).unwrap();

        let buf = band.nd_buffer().unwrap();
        assert_eq!(buf.shape, &[4]);
        assert_eq!(buf.strides, &[0]);
        assert_eq!(buf.offset, 0);

        let bytes = band.contiguous_data().unwrap();
        assert_eq!(&*bytes, &[42u8, 42, 42, 42]);
    }

    #[test]
    fn test_view_permutation_transpose() {
        // 2×3 source (UInt8), values 0..6 in C-order:
        //   row 0: [0, 1, 2]
        //   row 1: [3, 4, 5]
        // Transposed view exposes axes (cols, rows) → 3×2:
        //   row 0: [0, 3]
        //   row 1: [1, 4]
        //   row 2: [2, 5]
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        // The transposed visible shape on the spatial axes would conflict with
        // a 2D spatial grid; declare a single non-spatial dim "i" so the
        // strict spatial check is trivially satisfied.
        builder.start_raster(&transform, &[], &[], None).unwrap();

        let view = [
            // visible axis 0 reads source axis 1 (cols), full extent 3
            ViewEntry {
                source_axis: 1,
                start: 0,
                step: 1,
                steps: 3,
            },
            // visible axis 1 reads source axis 0 (rows), full extent 2
            ViewEntry {
                source_axis: 0,
                start: 0,
                step: 1,
                steps: 2,
            },
        ];
        builder
            .start_band_with_view(
                None,
                &["a", "b"],
                &[2, 3],
                &view,
                BandDataType::UInt8,
                None,
                None,
                None,
            )
            .unwrap();
        builder
            .band_data_writer()
            .append_value(vec![0u8, 1, 2, 3, 4, 5]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::new(&array);
        let r = rasters.get(0).unwrap();
        let band = r.band(0).unwrap();

        assert_eq!(band.shape(), &[3, 2]);
        assert_eq!(band.raw_source_shape(), &[2, 3]);
        let buf = band.nd_buffer().unwrap();
        assert_eq!(buf.strides, &[1, 3]); // visible axis 0 → source col stride; visible axis 1 → source row stride

        let bytes = band.contiguous_data().unwrap();
        assert_eq!(&*bytes, &[0u8, 3, 1, 4, 2, 5]);
    }

    #[test]
    fn test_view_empty_axis() {
        // steps=0 → empty visible axis. contiguous_data must succeed and
        // return an empty buffer.
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder.start_raster(&transform, &[], &[], None).unwrap();

        let view = [ViewEntry {
            source_axis: 0,
            start: 0,
            step: 1,
            steps: 0,
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

        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::new(&array);
        let r = rasters.get(0).unwrap();
        let band = r.band(0).unwrap();
        assert_eq!(band.shape(), &[0]);
        let buf = band.nd_buffer().unwrap();
        assert_eq!(buf.shape, &[0]);
        let bytes = band.contiguous_data().unwrap();
        assert!(bytes.is_empty());
    }

    #[test]
    fn test_start_band_rejects_zero_dim() {
        // 0-D bands carry no spatial extent and no caller has a use for
        // them. start_band must reject an empty dim_names slice eagerly so
        // the malformed band never reaches the buffer layer.
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder.start_raster(&transform, &[], &[], None).unwrap();
        let err = builder
            .start_band(None, &[], &[], BandDataType::UInt8, None, None, None)
            .unwrap_err();
        assert!(
            err.to_string().contains("0-dimensional"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_start_band_with_view_rejects_zero_dim() {
        // start_band_with_view must apply the same 0-D guard as start_band
        // — accepting empty dim_names would otherwise bypass it via the
        // explicit-view path.
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder.start_raster(&transform, &[], &[], None).unwrap();
        let err = builder
            .start_band_with_view(None, &[], &[], &[], BandDataType::UInt8, None, None, None)
            .unwrap_err();
        assert!(
            err.to_string().contains("0-dimensional"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_view_validation_rejects_out_of_range_start() {
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder.start_raster(&transform, &[], &[], None).unwrap();
        let view = [ViewEntry {
            source_axis: 0,
            start: 8,
            step: 1,
            steps: 1,
        }];
        let err = builder
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
            .unwrap_err();
        assert!(
            err.to_string().contains("out of range"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_view_validation_rejects_step_overrun() {
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder.start_raster(&transform, &[], &[], None).unwrap();
        // start=1, step=2, steps=4 → addresses element 1+(4-1)*2 = 7 which is
        // out of range for a source size of 7.
        let view = [ViewEntry {
            source_axis: 0,
            start: 1,
            step: 2,
            steps: 4,
        }];
        let err = builder
            .start_band_with_view(
                None,
                &["x"],
                &[7],
                &view,
                BandDataType::UInt8,
                None,
                None,
                None,
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("out of range"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_view_validation_rejects_duplicate_source_axis() {
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder.start_raster(&transform, &[], &[], None).unwrap();
        let view = [
            ViewEntry {
                source_axis: 0,
                start: 0,
                step: 1,
                steps: 2,
            },
            ViewEntry {
                source_axis: 0,
                start: 0,
                step: 1,
                steps: 2,
            },
        ];
        let err = builder
            .start_band_with_view(
                None,
                &["a", "b"],
                &[2, 3],
                &view,
                BandDataType::UInt8,
                None,
                None,
                None,
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("permutation"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_contiguous_data_identity_via_start_band_is_borrowed() {
        // Canonical identity: the row's view list is null, and the read path
        // synthesises the identity view. Should still hand the underlying
        // bytes back without copying.
        use std::borrow::Cow;

        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster(&transform, &["x", "y"], &[3, 2], None)
            .unwrap();
        builder
            .start_band(
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

        let bytes = band.contiguous_data().unwrap();
        assert!(matches!(bytes, Cow::Borrowed(_)));
        assert_eq!(&*bytes, pixels.as_slice());
    }

    #[test]
    fn test_contiguous_data_explicit_identity_view_is_borrowed() {
        // Identity expressed *explicitly* through start_band_with_view must be
        // indistinguishable to consumers from the null-row identity above —
        // same visible shape, same byte strides, same Cow::Borrowed fast path.
        use std::borrow::Cow;

        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster(&transform, &["x", "y"], &[3, 2], None)
            .unwrap();
        let view = [
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
        ];
        builder
            .start_band_with_view(
                None,
                &["y", "x"],
                &[2, 3],
                &view,
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

        assert_eq!(band.shape(), &[2, 3]);
        let buf = band.nd_buffer().unwrap();
        assert_eq!(buf.strides, &[3, 1]);
        assert_eq!(buf.offset, 0);

        let bytes = band.contiguous_data().unwrap();
        assert!(matches!(bytes, Cow::Borrowed(_)));
        assert_eq!(&*bytes, pixels.as_slice());
    }

    #[test]
    fn test_contiguous_data_zero_step_broadcast_2d() {
        // 2D broadcast: source shape [1, 3], view broadcasts axis 0 four
        // times so the visible region is 4×3. Each visible row must equal the
        // source's only row.
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder.start_raster(&transform, &[], &[], None).unwrap();
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
            .start_band_with_view(
                None,
                &["row", "col"],
                &[1, 3],
                &view,
                BandDataType::UInt8,
                None,
                None,
                None,
            )
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

        let bytes = band.contiguous_data().unwrap();
        assert_eq!(&*bytes, &[10u8, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 30]);
    }

    #[test]
    fn test_contiguous_data_negative_step_full_reverse() {
        // 1D source [0..8] with start=7, step=-1, steps=8 walks the source
        // backwards. Byte stride must be negative; offset lands on the last
        // source element.
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder.start_raster(&transform, &[], &[], None).unwrap();
        let view = [ViewEntry {
            source_axis: 0,
            start: 7,
            step: -1,
            steps: 8,
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

        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::new(&array);
        let r = rasters.get(0).unwrap();
        let band = r.band(0).unwrap();

        let buf = band.nd_buffer().unwrap();
        assert_eq!(buf.shape, &[8]);
        assert_eq!(buf.strides, &[-1]);
        assert_eq!(buf.offset, 7);

        let bytes = band.contiguous_data().unwrap();
        assert_eq!(&*bytes, &[7u8, 6, 5, 4, 3, 2, 1, 0]);
    }

    #[test]
    fn test_contiguous_data_negative_step_strided_reverse() {
        // 1D source [0..8] with start=6, step=-2, steps=3 picks every other
        // element walking backwards: {6, 4, 2}.
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder.start_raster(&transform, &[], &[], None).unwrap();
        let view = [ViewEntry {
            source_axis: 0,
            start: 6,
            step: -2,
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

        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::new(&array);
        let r = rasters.get(0).unwrap();
        let band = r.band(0).unwrap();

        let buf = band.nd_buffer().unwrap();
        assert_eq!(buf.shape, &[3]);
        assert_eq!(buf.strides, &[-2]);
        assert_eq!(buf.offset, 6);

        let bytes = band.contiguous_data().unwrap();
        assert_eq!(&*bytes, &[6u8, 4, 2]);
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
            .start_raster(&transform, &["x", "y"], &[2, 2], None)
            .unwrap();
        builder
            .start_band(
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
            .start_raster(&transform, &["x", "y"], &[4, 4], None)
            .unwrap();
        // Band has "x" and "y" but x-size disagrees with top-level shape.
        builder
            .start_band(
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
    fn test_contiguous_data_float32_fast_path() {
        // Multi-byte dtype on the contiguous innermost-axis fast path:
        // a 2D explicit-identity view over Float32 should still emit
        // bytes by `extend_from_slice` and produce the exact source
        // payload back. Catches a regression where the fast path
        // assumed dtype_size == 1.
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster(&transform, &["x", "y"], &[3, 2], None)
            .unwrap();
        // Slice the outer axis: take rows 0 and 1 of a 3-row source. With
        // start=0, step=1, steps=2 over an axis of size 3, the view is
        // not identity, so contiguous_data() materialises through the
        // fast path. Inner stride = dtype_size = 4 → fast path is taken.
        let view = [
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
        ];
        builder
            .start_band_with_view(
                None,
                &["y", "x"],
                &[3, 3], // 3x3 source
                &view,
                BandDataType::Float32,
                None,
                None,
                None,
            )
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
        let bytes = band.contiguous_data().unwrap();
        assert!(matches!(bytes, std::borrow::Cow::Owned(_)));
        assert_eq!(&*bytes, &source_bytes[0..24]);
    }

    #[test]
    fn test_contiguous_data_outer_axis_slice_3d() {
        // 3D source [T=3, Y=2, X=3] of UInt8. View slices T to T=1..3
        // (start=1, step=1, steps=2), keeps Y and X identity. Innermost
        // axis is contiguous (step=1, dtype=1) so the fast path emits 6
        // bytes per outer iteration via extend_from_slice.
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster(&transform, &["x", "y"], &[3, 2], None)
            .unwrap();
        let view = [
            ViewEntry {
                source_axis: 0,
                start: 1,
                step: 1,
                steps: 2,
            },
            ViewEntry {
                source_axis: 1,
                start: 0,
                step: 1,
                steps: 2,
            },
            ViewEntry {
                source_axis: 2,
                start: 0,
                step: 1,
                steps: 3,
            },
        ];
        builder
            .start_band_with_view(
                None,
                &["t", "y", "x"],
                &[3, 2, 3],
                &view,
                BandDataType::UInt8,
                None,
                None,
                None,
            )
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
        let bytes = band.contiguous_data().unwrap();
        assert_eq!(&*bytes, &source[6..18]);
    }

    #[test]
    fn test_contiguous_data_strided_inner_falls_back() {
        // Inner stride != dtype_size forces the elementwise fallback. View
        // takes every other column on a 1D UInt16 source. Verifies the
        // slow path still emits correct bytes.
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder.start_raster(&transform, &[], &[], None).unwrap();
        let view = [ViewEntry {
            source_axis: 0,
            start: 0,
            step: 2,
            steps: 3,
        }];
        builder
            .start_band_with_view(
                None,
                &["x"],
                &[6],
                &view,
                BandDataType::UInt16,
                None,
                None,
                None,
            )
            .unwrap();
        let source: Vec<u16> = vec![10, 20, 30, 40, 50, 60];
        let source_bytes: Vec<u8> = source.iter().flat_map(|v| v.to_le_bytes()).collect();
        builder.band_data_writer().append_value(source_bytes);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::new(&array);
        let r = rasters.get(0).unwrap();
        let band = r.band(0).unwrap();

        let bytes = band.contiguous_data().unwrap();
        let expected: Vec<u8> = [10u16, 30, 50]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(&*bytes, expected.as_slice());
    }

    #[test]
    fn test_nd_buffer_multidim_non_zero_starts() {
        // 3D source [T=4, Y=3, X=5], slice T from 1, Y from 1, X identity.
        // visible = [3, 2, 5]. byte_offset must equal 1*Y*X + 1*X = 1*15 + 1*5 = 20.
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder
            .start_raster(&transform, &["x", "y"], &[5, 2], None)
            .unwrap();
        let view = [
            ViewEntry {
                source_axis: 0,
                start: 1,
                step: 1,
                steps: 3,
            },
            ViewEntry {
                source_axis: 1,
                start: 1,
                step: 1,
                steps: 2,
            },
            ViewEntry {
                source_axis: 2,
                start: 0,
                step: 1,
                steps: 5,
            },
        ];
        builder
            .start_band_with_view(
                None,
                &["t", "y", "x"],
                &[4, 3, 5],
                &view,
                BandDataType::UInt8,
                None,
                None,
                None,
            )
            .unwrap();
        builder.band_data_writer().append_value(vec![0u8; 60]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::new(&array);
        let r = rasters.get(0).unwrap();
        let band = r.band(0).unwrap();
        let buf = band.nd_buffer().unwrap();
        assert_eq!(buf.shape, &[3, 2, 5]);
        assert_eq!(buf.strides, &[15, 5, 1]);
        assert_eq!(buf.offset, 20);
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
        builder.start_raster(&transform, &[], &[], None).unwrap();
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
            .start_band_with_view(
                None,
                &["x", "y"],
                &[4, 3],
                &view,
                BandDataType::UInt8,
                None,
                None,
                None,
            )
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

        // contiguous_data must walk the permuted+strided view correctly.
        // Visible (X, Y) → source row Y is 0..3=[0,1,2], 3..6=[3,4,5], 6..9=[6,7,8], 9..12=[9,10,11].
        // We expose Y rows 1 and 3 (start=1 step=2 steps=2). At visible index
        // (x, y), source value = source[(start_Y + y*2) * 3 + x] = source[(1 + y*2)*3 + x].
        // Expected, in C-order with X outer and Y inner:
        //   (x=0,y=0)=src[3]=3,  (x=0,y=1)=src[9]=9,
        //   (x=1,y=0)=src[4]=4,  (x=1,y=1)=src[10]=10,
        //   (x=2,y=0)=src[5]=5,  (x=2,y=1)=src[11]=11
        let bytes = band.contiguous_data().unwrap();
        assert_eq!(&*bytes, &[3u8, 9, 4, 10, 5, 11]);
    }

    #[test]
    fn test_nd_buffer_steps_one_view() {
        // Degenerate inner view: pick a single element on each axis.
        // visible_shape == [1, 1]; byte_strides retain their per-axis values
        // because they're step * source_stride, not skipped on steps==1.
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder.start_raster(&transform, &[], &[], None).unwrap();
        let view = [
            ViewEntry {
                source_axis: 0,
                start: 1,
                step: 1,
                steps: 1,
            },
            ViewEntry {
                source_axis: 1,
                start: 2,
                step: 1,
                steps: 1,
            },
        ];
        builder
            .start_band_with_view(
                None,
                &["a", "b"],
                &[3, 4],
                &view,
                BandDataType::UInt8,
                None,
                None,
                None,
            )
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
        assert_eq!(band.shape(), &[1, 1]);
        let buf = band.nd_buffer().unwrap();
        assert_eq!(buf.shape, &[1, 1]);
        // step * source_stride per axis: source_stride_a = 4, source_stride_b = 1.
        // A regression that zeroed out strides when steps==1 would still pass
        // the shape/offset/data assertions, so pin the strides explicitly.
        assert_eq!(buf.strides, &[4, 1]);
        // start_a * source_stride_a + start_b * source_stride_b
        //  = 1 * 4 + 2 * 1 = 6
        assert_eq!(buf.offset, 6);
        let bytes = band.contiguous_data().unwrap();
        assert_eq!(&*bytes, &[6u8]);
    }

    #[test]
    fn test_nd_buffer_multidim_with_zero_axis() {
        // visible_shape contains a zero axis somewhere in the middle.
        // contiguous_data returns an empty buffer; nd_buffer returns the
        // zero-element shape.
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder.start_raster(&transform, &[], &[], None).unwrap();
        let view = [
            ViewEntry {
                source_axis: 0,
                start: 0,
                step: 1,
                steps: 3,
            },
            ViewEntry {
                source_axis: 1,
                start: 0,
                step: 1,
                steps: 0,
            },
            ViewEntry {
                source_axis: 2,
                start: 0,
                step: 1,
                steps: 5,
            },
        ];
        builder
            .start_band_with_view(
                None,
                &["a", "b", "c"],
                &[3, 4, 5],
                &view,
                BandDataType::UInt8,
                None,
                None,
                None,
            )
            .unwrap();
        builder.band_data_writer().append_value(vec![0u8; 60]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        let array = builder.finish().unwrap();
        let rasters = RasterStructArray::new(&array);
        let r = rasters.get(0).unwrap();
        let band = r.band(0).unwrap();
        assert_eq!(band.shape(), &[3, 0, 5]);
        let buf = band.nd_buffer().unwrap();
        assert_eq!(buf.shape, &[3, 0, 5]);
        let bytes = band.contiguous_data().unwrap();
        assert!(bytes.is_empty());
    }

    #[test]
    fn test_view_null_round_trips_through_arrow_ipc() {
        // Schema invariant: a band built via start_band serialises with a
        // null view row, and the null must survive an Arrow IPC round-trip.
        // If a future change accidentally writes a non-null empty list
        // instead, downstream readers (DuckDB, PyArrow, sedona-py) will
        // disagree about whether the view is identity.
        use arrow_array::RecordBatch;
        use arrow_ipc::reader::StreamReader;
        use arrow_ipc::writer::StreamWriter;
        use arrow_schema::Schema;
        use std::io::Cursor;

        let mut builder = RasterBuilder::new(2);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        // Raster 0: identity-view band → null view row.
        builder
            .start_raster(&transform, &["x", "y"], &[3, 2], None)
            .unwrap();
        builder
            .start_band(
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
        // Raster 1: explicit non-identity view → non-null view row.
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

        // Reach into the restored bands list and confirm the view list
        // preserves null/non-null per row.
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
        assert_eq!(view_list.len(), 2);
        assert!(
            view_list.is_null(0),
            "identity-view band must remain a null view row after IPC round-trip"
        );
        assert!(
            !view_list.is_null(1),
            "explicit-view band must remain non-null after IPC round-trip"
        );

        // Sanity: read paths still produce the expected visible shapes.
        let rasters = RasterStructArray::new(restored_struct);
        let r0 = rasters.get(0).unwrap();
        assert_eq!(r0.band(0).unwrap().shape(), &[2, 3]);
        let r1 = rasters.get(1).unwrap();
        assert_eq!(r1.band(0).unwrap().shape(), &[3]);
    }
}
