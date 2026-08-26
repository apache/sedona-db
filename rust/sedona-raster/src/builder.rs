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
    builder::{BinaryViewBuilder, BooleanBuilder, Float64Builder, Int64Builder, StringViewBuilder},
    Array, ArrayRef, BinaryViewArray, ListArray, StructArray,
};
use arrow_buffer::{Buffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{ArrowError, DataType};
use std::sync::Arc;

use sedona_schema::raster::{BandDataType, RasterSchema};

use crate::band_builder::{BandArrayBuilder, BandWriter};
use crate::traits::{BandOverrides, BandRef, RasterRef};
use crate::view_entries::ViewEntry;

/// Raster-level metadata overrides for [`RasterBuilder::start_raster_from`] and
/// [`RasterBuilder::copy_raster_from`]. A `None` field inherits the source
/// raster's value. The band-level analog is [`BandOverrides`].
#[derive(Debug, Default, Clone)]
pub struct RasterOverrides {
    /// Override the 6-element GDAL geotransform; `None` inherits the source's.
    pub transform: Option<[f64; 6]>,
}

/// Builder for constructing raster arrays with zero-copy band data writing
///
/// Required steps to build a raster:
/// 1. Create a RasterBuilder with a specified capacity
/// 2. For each raster to add:
///    - Call `start_raster_2d` with the geotransform parameters and CRS
///    - For each band in the raster:
///       - Call `start_band_2d` with the band data type and nodata
///       - Use `band_data_writer` to get a BinaryViewBuilder and write the band data
///       - Call `finish_band` to complete the band
///    - Call `finish_raster` to complete the raster
/// 3. After all rasters are added, call `finish` to get the final StructArray
///
/// Example usage:
/// ```
/// use sedona_schema::raster::BandDataType;
/// use sedona_raster::builder::RasterBuilder;
///
/// let mut builder = RasterBuilder::new(1);
/// // Start a 100x100 raster with a north-up geotransform.
/// builder
///     .start_raster_2d(100, 100, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, Some("EPSG:4326"))
///     .unwrap();
///
/// // Add a band:
/// builder.start_band_2d(BandDataType::UInt8, Some(&[0u8])).unwrap();
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

    // The flat, per-band columns — see [`BandArrayBuilder`]. Everything
    // below is raster-envelope bookkeeping layered on top of it: grouping
    // its rows into a per-raster list, and validating them against each
    // raster's spatial grid.
    bands: BandArrayBuilder,

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

    raster_validity: BooleanBuilder,
}

/// Arguments to [`RasterBuilder::start_band`]. Bundled into a struct to keep
/// the call site readable — eight slots is enough that positional args invite
/// mis-ordering bugs.
pub struct StartBandArgs<'a> {
    pub name: Option<&'a str>,
    pub dim_names: &'a [&'a str],
    pub source_shape: &'a [i64],
    /// Per-axis window of offsets/steps over `source_shape`. `None` is the
    /// canonical identity view (the whole source buffer in C order).
    pub view: Option<&'a [ViewEntry]>,
    pub data_type: BandDataType,
    pub nodata: Option<&'a [u8]>,
    pub outdb_uri: Option<&'a str>,
    pub outdb_format: Option<&'a str>,
}

impl<'a> StartBandArgs<'a> {
    /// The required band descriptors; the optional fields (`name`, `view`,
    /// `nodata`, `outdb_uri`, `outdb_format`) default to `None`. Override any of
    /// them with struct-update syntax, e.g.
    /// `StartBandArgs { nodata: Some(&nd), ..StartBandArgs::new(dims, shape, dtype) }`.
    pub fn new(dim_names: &'a [&'a str], source_shape: &'a [i64], data_type: BandDataType) -> Self {
        Self {
            name: None,
            dim_names,
            source_shape,
            view: None,
            data_type,
            nodata: None,
            outdb_uri: None,
            outdb_format: None,
        }
    }
}

/// Arguments to [`RasterBuilder::with_view`]. Mirrors
/// [`StartBandArgs`] minus the two fields `with_view` derives from
/// `input` (`source_shape` from `input.raw_source_shape()`, `data_type` from
/// `input.data_type()`) — accepting those from the caller would let them
/// contradict `input`. `view` here is a *delta* composed against
/// `input.view()`, not the absolute view stored on the band.
pub struct WithViewArgs<'a> {
    pub name: Option<&'a str>,
    pub dim_names: &'a [&'a str],
    pub input: &'a dyn BandRef,
    pub view: &'a [ViewEntry],
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

            bands: BandArrayBuilder::new(capacity),

            band_offsets: vec![0],
            current_band_count: 0,
            current_width: 0,
            current_height: 0,

            current_spatial_dims: Vec::new(),
            current_spatial_shape: Vec::new(),
            current_raster_bands: Vec::new(),

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
        // their own shapes via start_band.
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
            // The band name lives on the *raster* (`RasterRef::band_name`), not
            // on `BandRef`, so `copy_into` has nothing to inherit it from and
            // must be handed it explicitly — otherwise every copied band comes
            // out unnamed.
            source.band(band_idx)?.copy_into(
                self,
                BandOverrides {
                    name: source.band_name(band_idx),
                    ..BandOverrides::default()
                },
            )?;
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

    /// Start a new band with explicit N-D parameters.
    ///
    /// `outdb_uri` is the *location* of the external resource (scheme is
    /// resolved by an `ObjectStoreRegistry`). `outdb_format` is the *format*
    /// used to interpret the bytes at that location (e.g. `"geotiff"`,
    /// `"zarr"`). A null `outdb_format` means the band is in-memory — the
    /// band's `data` buffer is authoritative.
    ///
    /// `view` is a per-axis window of offsets/steps over `source_shape`.
    /// `None` is the canonical identity view — the whole source buffer in C
    /// order — encoded as the identity null sentinel. A `Some` view is
    /// validated against `source_shape`; an identity `Some` view is stored the
    /// same as `None` (the null sentinel), while a non-identity view (a slice,
    /// broadcast, permutation, or reverse) is persisted explicitly into the
    /// band's four parallel view columns. In that case the `shape` column holds
    /// the *source* shape and the visible shape is derived from the view on
    /// read; the source bytes are carried over unchanged.
    pub fn start_band(&mut self, args: StartBandArgs<'_>) -> Result<(), ArrowError> {
        self.start_band_and_record(args)?;
        Ok(())
    }

    /// Shared by the inherent [`Self::start_band`] (used directly by ~20
    /// call sites across the raster function crates, which don't need the
    /// recorded `(dim_names, visible_shape)`) and `impl BandWriter for
    /// RasterBuilder` (used by [`BandRef::copy_into`](crate::traits::
    /// BandRef::copy_into), which does need it — see [`BandWriter::
    /// start_band`]). Both delegate here rather than to each other, so
    /// there's no same-type self-call to disambiguate between the inherent
    /// method and the trait impl of the same name.
    fn start_band_and_record(
        &mut self,
        args: StartBandArgs<'_>,
    ) -> Result<(Vec<String>, Vec<i64>), ArrowError> {
        let (dim_names, shape) = self.bands.start_band(args)?;
        self.current_band_count += 1;
        self.current_raster_bands
            .push((dim_names.clone(), shape.clone()));
        Ok((dim_names, shape))
    }

    /// Build a band that is a new view into an existing band.
    ///
    /// The output band stores a view that is the composition of `input`'s
    /// existing view with the supplied `view`. The supplied `view`'s
    /// `source_axis` entries refer to `input`'s *visible* axes, not its
    /// source axes — composition with `input.view()` translates them, so the
    /// caller expresses the slice in the coordinates it sees.
    ///
    /// `dim_names` names the output's *visible* axes (`len() == view.len()`).
    ///
    /// Storage:
    /// - **InDb input** → output is InDb. The input's source bytes are carried
    ///   over via [`BandRef::append_data_into`] (zero-copy when the impl shares
    ///   its backing `Buffer`); the composed view addresses the visible region
    ///   within them.
    /// - **OutDb input** → output is OutDb. The data column stays empty and the
    ///   input's `outdb_uri` / `outdb_format` are inherited (unless overridden);
    ///   the composed view lives alongside the same external pointer and loading
    ///   is deferred to whoever reads the visible bytes.
    ///
    /// Identity-input shortcut: when `input` carries the identity view, the
    /// composed view equals `view` verbatim.
    pub fn with_view(&mut self, args: WithViewArgs) -> Result<(), ArrowError> {
        let WithViewArgs {
            name,
            dim_names,
            input,
            view,
            nodata,
            outdb_uri,
            outdb_format,
        } = args;
        // Delegate to `copy_into`, which composes `view` (a delta over the
        // input's visible axes) onto the input's own view, carries the source
        // bytes over, and inherits every field left unset here from the input
        // via `.or_else(|| input.<field>())` — including `nodata`, which the
        // earlier hand-rolled implementation forwarded verbatim and thereby
        // dropped. `dim_names` and `view` are always supplied by this call, so
        // they pass through as explicit overrides.
        input.copy_into(
            self,
            BandOverrides {
                name,
                dim_names: Some(dim_names),
                nodata,
                outdb_uri,
                outdb_format,
                view: Some(view),
            },
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
        self.start_band(StartBandArgs {
            nodata,
            ..StartBandArgs::new(
                &["y", "x"],
                &[self.current_height, self.current_width],
                data_type,
            )
        })
    }

    /// Get direct access to the BinaryViewBuilder for writing the current band's data.
    pub fn band_data_writer(&mut self) -> &mut BinaryViewBuilder {
        self.bands.band_data_writer()
    }

    /// Append the current band's data as a **zero-copy** view into an existing
    /// Arrow [`Buffer`], rather than copying bytes via `append_value`. See
    /// [`crate::band_builder::BandArrayBuilder::append_band_data_buffer`] for
    /// the full documentation — identical logic, delegated.
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
        self.bands.append_band_data_buffer(buffer, offset, len)
    }

    /// Append the current band's data by copying row `row` of `src` through —
    /// zero-copy when the row's bytes are block-backed (shares the backing
    /// `Buffer` via [`append_band_data_buffer`]), with a small copy only for
    /// inline views (which live in the view itself and have no backing
    /// buffer).
    ///
    /// Counts as the one data value for the current band (see [`finish_band`]).
    ///
    /// [`finish_band`]: Self::finish_band
    pub fn append_band_data_from(
        &mut self,
        src: &BinaryViewArray,
        row: usize,
    ) -> Result<(), ArrowError> {
        self.bands.append_band_data_from(src, row)
    }

    /// Finish writing the current band.
    ///
    /// Validates that exactly one data value was appended since `start_band()`.
    pub fn finish_band(&mut self) -> Result<(), ArrowError> {
        self.bands.finish_band()
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

        // Build the flat band struct — delegated to `BandArrayBuilder`,
        // which owns all the per-band columns (name, dim_names, shape,
        // datatype, nodata, view, outdb_uri/format, data).
        let band_struct = self.bands.finish()?;

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

/// Lets [`BandRef::copy_into`](crate::traits::BandRef::copy_into) and
/// [`BandRef::append_data_into`](crate::traits::BandRef::append_data_into)
/// target a `RasterBuilder` directly, same as before this trait existed:
/// `start_band` here does the *same* per-raster bookkeeping as the inherent
/// [`RasterBuilder::start_band`] (both delegate to
/// [`Self::start_band_and_record`]) — deriving a band via `copy_into` into a
/// `RasterBuilder` is still validated against that raster's spatial grid at
/// `finish_raster`, exactly as it was when `copy_into` was hardcoded to this
/// type.
impl BandWriter for RasterBuilder {
    fn start_band(
        &mut self,
        args: StartBandArgs<'_>,
    ) -> Result<(Vec<String>, Vec<i64>), ArrowError> {
        self.start_band_and_record(args)
    }

    fn band_data_writer(&mut self) -> &mut BinaryViewBuilder {
        self.bands.band_data_writer()
    }

    fn append_band_data_from(
        &mut self,
        src: &BinaryViewArray,
        row: usize,
    ) -> Result<(), ArrowError> {
        self.bands.append_band_data_from(src, row)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::RasterStructArray;
    use crate::traits::RasterRef;
    use arrow_array::RecordBatch;
    use arrow_ipc::reader::StreamReader;
    use arrow_ipc::writer::StreamWriter;
    use arrow_schema::Schema;
    use std::io::Cursor;

    /// Terse [`ViewEntry`] constructor for the view-persistence tests.
    fn ve(source_axis: i64, start: i64, step: i64, steps: i64) -> ViewEntry {
        ViewEntry {
            source_axis,
            start,
            step,
            steps,
        }
    }

    #[test]
    fn test_iterator_basic_functionality() {
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

        // Test the iterator
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

        // Test iterator over bands
        let band_iter: Vec<_> = (0..raster.num_bands()).map(|i| raster.band(i)).collect();
        assert_eq!(band_iter.len(), 1);
    }

    #[test]
    fn test_multi_band_iterator() {
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

        // Test iterator
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
    fn test_copy_metadata_from_iterator() {
        // Create an original raster
        let mut source_builder = RasterBuilder::new(10);

        source_builder
            .start_raster_2d(42, 24, -122.0, 37.8, 0.1, -0.1, 0.0, 0.0, None)
            .unwrap();

        source_builder
            .start_band_2d(BandDataType::UInt8, Some(&[255u8]))
            .unwrap();
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
            .start_raster_from(&source_raster, RasterOverrides::default())
            .unwrap();

        // Add new band data while preserving original metadata. `start_raster_from`
        // copies the N-D spatial grid, so add the band with an explicit shape
        // matching the source's [height, width].
        target_builder
            .start_band(StartBandArgs::new(
                &["y", "x"],
                &[
                    source_raster.height().unwrap(),
                    source_raster.width().unwrap(),
                ],
                BandDataType::UInt16,
            ))
            .unwrap();
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

        // All metadata should match the original
        assert_eq!(target_raster.width().unwrap(), 42);
        assert_eq!(target_raster.height().unwrap(), 24);
        assert_eq!(target_raster.transform()[0], -122.0);
        assert_eq!(target_raster.transform()[3], 37.8);
        assert_eq!(target_raster.transform()[1], 0.1);
        assert_eq!(target_raster.transform()[5], -0.1);

        // But band data and metadata should be different
        let target_band = target_raster.band(0).unwrap();
        assert_eq!(target_band.data_type(), BandDataType::UInt16);
        assert!(target_band.nodata().is_none());
        assert_eq!(
            target_band
                .nd_buffer()
                .unwrap()
                .as_contiguous()
                .unwrap()
                .len(),
            2016
        ); // 1008 * 2 bytes per u16

        let result = target_raster.band(1);
        assert!(result.is_err(), "Band index 1 should be out of range");
    }

    #[test]
    fn copy_raster_from_overrides_transform_and_preserves_bands() {
        use sedona_testing::raster_spec::{assert_rasters_equal, RasterSpec};

        // Source: a CRS, a nodata sentinel, and pixel values to preserve.
        let source = RasterSpec::d2(2, 1)
            .band_values(&[1u8, 2])
            .name("temperature")
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

        // Only the transform changed; CRS, nodata, band name, and pixels
        // carried over.
        let expected = RasterSpec::d2(2, 1)
            .band_values(&[1u8, 2])
            .name("temperature")
            .nodata(9u8)
            .crs(Some("OGC:CRS84"))
            .transform([100.0, 2.0, 0.0, 200.0, 0.0, -3.0]);
        assert_rasters_equal(&out, &[Some(expected)]);
    }

    #[test]
    fn test_band_data_types() {
        // Create a test raster with bands of different data types
        let mut builder = RasterBuilder::new(1);

        builder
            .start_raster_2d(2, 2, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, None)
            .unwrap();

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
            builder.start_band_2d(expected_data_type, None).unwrap();
            builder.band_data_writer().append_value(&test_data);
            builder.finish_band().unwrap();
        }

        builder.finish_raster().unwrap();
        let raster_array = builder.finish().unwrap();

        // Test the data type conversion for each band
        let iterator = RasterStructArray::try_new(&raster_array).unwrap();
        let raster = iterator.get(0).unwrap();

        assert_eq!(
            raster.num_bands(),
            10,
            "Expected 10 bands for all data types"
        );

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
            let band = raster.band(i).unwrap();
            let actual_type = band.data_type();

            assert_eq!(
                actual_type, *expected_type,
                "Band {i} expected data type {expected_type:?}, got {actual_type:?}"
            );
        }
    }

    #[test]
    fn test_outdb_metadata_fields() {
        // Test creating raster with OutDb reference metadata.
        //
        // 10x10 UInt8 = 100 visible bytes, matching the InDb data buffer
        // written below. `RasterRef::band()` now verifies the data column is
        // long enough to cover the visible region, so the dimensions and the
        // byte count must agree.
        let mut builder = RasterBuilder::new(10);

        builder
            .start_raster_2d(10, 10, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, None)
            .unwrap();

        // Test InDb band (should have null OutDb fields)
        builder
            .start_band_2d(BandDataType::UInt8, Some(&[255u8]))
            .unwrap();
        let test_data = vec![1u8; 100];
        builder.band_data_writer().append_value(&test_data);
        builder.finish_band().unwrap();

        // Test OutDbRef band: an out-db location is carried as an `outdb_uri`
        // with the SedonaDB `#band=N` fragment; the band's own `data` is empty.
        builder
            .start_band(StartBandArgs {
                outdb_uri: Some("s3://mybucket/satellite_image.tif#band=2"),
                ..StartBandArgs::new(&["y", "x"], &[10, 10], BandDataType::Float32)
            })
            .unwrap();
        // For OutDbRef, data field could be empty or contain metadata/thumbnail
        builder.band_data_writer().append_value([]);
        builder.finish_band().unwrap();

        builder.finish_raster().unwrap();
        let raster_array = builder.finish().unwrap();

        // Verify the band metadata
        let iterator = RasterStructArray::try_new(&raster_array).unwrap();
        let raster = iterator.get(0).unwrap();

        assert_eq!(raster.num_bands(), 2);

        // Test InDb band
        let indb_band = raster.band(0).unwrap();
        assert!(indb_band.is_indb());
        assert_eq!(indb_band.data_type(), BandDataType::UInt8);
        assert!(indb_band.outdb_uri().is_none());

        // Test OutDbRef band
        let outdb_band = raster.band(1).unwrap();
        assert!(!outdb_band.is_indb());
        assert_eq!(outdb_band.data_type(), BandDataType::Float32);
        assert_eq!(
            outdb_band.outdb_uri().unwrap(),
            "s3://mybucket/satellite_image.tif#band=2"
        );
    }

    #[test]
    fn test_band_access_errors() {
        // Create a simple raster with one band
        let mut builder = RasterBuilder::new(1);

        builder
            .start_raster_2d(10, 10, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, None)
            .unwrap();

        builder.start_band_2d(BandDataType::UInt8, None).unwrap();
        builder.band_data_writer().append_value([1u8; 100]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        let raster_array = builder.finish().unwrap();
        let iterator = RasterStructArray::try_new(&raster_array).unwrap();
        let raster = iterator.get(0).unwrap();

        // Test out of range band number (bands are 0-based, so index 1 is out
        // of range for this single-band raster)
        let result = raster.band(1);
        assert!(result.is_err());
        let err = result.err().unwrap().to_string();
        assert!(err.contains("is out of range"));

        // Test valid band number should still work
        let result = raster.band(0);
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
            .start_band(StartBandArgs {
                name: Some("temperature"),
                ..StartBandArgs::new(&["time", "y", "x"], &[3, 4, 5], BandDataType::Float32)
            })
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
            .start_band(StartBandArgs {
                name: Some("sst"),
                ..StartBandArgs::new(
                    &["latitude", "longitude"],
                    &[180, 360],
                    BandDataType::Float32,
                )
            })
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
            .start_band(StartBandArgs {
                name: Some("temperature"),
                ..StartBandArgs::new(&["time", "y", "x"], &[12, 64, 64], BandDataType::Float32)
            })
            .unwrap();
        let data_3d = vec![0u8; 12 * 64 * 64 * 4];
        builder.band_data_writer().append_value(&data_3d);
        builder.finish_band().unwrap();

        // Band 1: 2D [y=64, x=64]
        builder
            .start_band(StartBandArgs {
                name: Some("elevation"),
                ..StartBandArgs::new(&["y", "x"], &[64, 64], BandDataType::Float64)
            })
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
            .start_band(StartBandArgs::new(
                &["time", "pressure", "y", "x"],
                &[6, 10, 32, 32],
                BandDataType::Float32,
            ))
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
            .start_band(StartBandArgs::new(
                &["y", "x"],
                &[3, 4],
                BandDataType::UInt8,
            ))
            .unwrap();
        builder.band_data_writer().append_value(vec![0u8; 12]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        // Raster 1 — Float64: element size = 8, shape [2, 3, 5] → strides [120, 40, 8]
        builder
            .start_raster_nd(&transform, &["x", "y"], &[5, 3], None)
            .unwrap();
        builder
            .start_band(StartBandArgs::new(
                &["z", "y", "x"],
                &[2, 3, 5],
                BandDataType::Float64,
            ))
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
            .start_band(StartBandArgs::new(&["x"], &[10], BandDataType::UInt16))
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
            .start_band(StartBandArgs {
                name: Some("temperature"),
                ..StartBandArgs::new(&["y", "x"], &[4, 4], BandDataType::Float32)
            })
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
            .start_band(StartBandArgs::new(
                &["latitude", "longitude"],
                &[180, 360],
                BandDataType::UInt8,
            ))
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
            .start_band(StartBandArgs::new(&["x"], &[4], BandDataType::UInt8))
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
        // them. start_band must reject an empty dim_names slice eagerly so
        // the malformed band never reaches the buffer layer.
        let mut builder = RasterBuilder::new(1);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        builder.start_raster_nd(&transform, &[], &[], None).unwrap();
        let err = builder
            .start_band(StartBandArgs::new(&[], &[], BandDataType::UInt8))
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
            .start_band(StartBandArgs::new(
                &["y", "x"],
                &[2, 3],
                BandDataType::UInt8,
            ))
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

    // ---- Non-identity view persistence: construct → finish → read back ----

    /// Build a single-raster, single-band `UInt8` array carrying an explicit
    /// `view` over `source_shape`, with `data` as the band's raw bytes. Uses
    /// empty top-level spatial dims so `finish_raster` imposes no spatial-shape
    /// constraint on the (arbitrary) view being exercised.
    fn build_viewed_u8(
        source_shape: &[i64],
        dim_names: &[&str],
        view: &[ViewEntry],
        data: Vec<u8>,
    ) -> StructArray {
        let mut b = RasterBuilder::new(1);
        b.start_raster_nd(&[0.0, 1.0, 0.0, 0.0, 0.0, -1.0], &[], &[], None)
            .unwrap();
        b.start_band(StartBandArgs {
            view: Some(view),
            ..StartBandArgs::new(dim_names, source_shape, BandDataType::UInt8)
        })
        .unwrap();
        b.band_data_writer().append_value(data);
        b.finish_band().unwrap();
        b.finish_raster().unwrap();
        b.finish().unwrap()
    }

    /// Walk an `NdBuffer`'s visible region in C-order and collect the single
    /// `UInt8` byte at each visited position. The byte address is hand-computed
    /// from the buffer's own shape/strides/offset, so it verifies exactly the
    /// layout the reader composed — independent of `as_contiguous`, which
    /// refuses strided views. Compare its output against a hand-computed
    /// expectation, never against the buffer itself.
    fn gather_u8(buf: &crate::traits::NdBuffer) -> Vec<u8> {
        assert_eq!(buf.data_type, BandDataType::UInt8);
        let n: i64 = buf.shape.iter().product();
        let mut out = Vec::with_capacity(n.max(0) as usize);
        let mut idx = vec![0i64; buf.shape.len()];
        for _ in 0..n {
            let mut pos = buf.offset as i64;
            for (k, &i) in idx.iter().enumerate() {
                pos += i * buf.strides[k];
            }
            out.push(buf.buffer[pos as usize]);
            // Increment the multi-index in C-order (last axis fastest).
            for k in (0..buf.shape.len()).rev() {
                idx[k] += 1;
                if idx[k] < buf.shape[k] {
                    break;
                }
                idx[k] = 0;
            }
        }
        out
    }

    #[test]
    fn view_identity_via_start_band_is_null_and_borrows() {
        // An identity view passed through start_band as an explicit `Some` view
        // is stored as the canonical null sentinel (not an explicit row), so it
        // is indistinguishable from the `None` path: null view row, same
        // visible shape/strides, zero-copy borrow.
        use arrow_array::Array;
        let view = [ve(0, 0, 1, 2), ve(1, 0, 1, 3)];
        let pixels: Vec<u8> = (0..6).collect();
        let array = build_viewed_u8(&[2, 3], &["y", "x"], &view, pixels.clone());

        let bands_struct = array
            .column(sedona_schema::raster::raster_indices::BANDS)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap()
            .values()
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let view_list = bands_struct
            .column(sedona_schema::raster::band_indices::VIEW)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        assert!(
            view_list.is_null(0),
            "identity view must serialise as a null view row even via an explicit start_band view"
        );

        let rasters = RasterStructArray::try_new(&array).unwrap();
        let r = rasters.get(0).unwrap();
        let band = r.band(0).unwrap();
        assert_eq!(band.shape(), &[2, 3]);
        let buf = band.nd_buffer().unwrap();
        assert_eq!(buf.strides, &[3, 1]);
        assert_eq!(buf.offset, 0);
        assert!(buf.is_contiguous());
        assert_eq!(buf.as_contiguous().unwrap(), pixels.as_slice());
        assert_eq!(gather_u8(&buf), pixels);
    }

    #[test]
    fn view_slice_outer_axis_is_contiguous() {
        // Slice the outer axis of a 3x3 source to its first 2 rows. The view is
        // non-identity, but its byte strides are still C-order packed from
        // offset 0, so the region is contiguous and borrows the source prefix
        // zero-copy.
        let data: Vec<u8> = (0..9).collect();
        let view = [ve(0, 0, 1, 2), ve(1, 0, 1, 3)];
        let array = build_viewed_u8(&[3, 3], &["y", "x"], &view, data.clone());
        let rasters = RasterStructArray::try_new(&array).unwrap();
        let r = rasters.get(0).unwrap();
        let band = r.band(0).unwrap();

        assert_eq!(band.shape(), &[2, 3]);
        assert_eq!(band.raw_source_shape(), &[3, 3]);
        let buf = band.nd_buffer().unwrap();
        assert_eq!(buf.strides, &[3, 1]);
        assert_eq!(buf.offset, 0);
        assert!(buf.is_contiguous());
        assert_eq!(buf.as_contiguous().unwrap(), &data[0..6]);
        assert_eq!(gather_u8(&buf), vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn view_slice_strided_reads_expected_values() {
        // Every-other slice of an 8-element source: start=1, step=2, steps=3
        // addresses source indices 1, 3, 5.
        let data: Vec<u8> = (0..8).collect();
        let array = build_viewed_u8(&[8], &["x"], &[ve(0, 1, 2, 3)], data);
        let rasters = RasterStructArray::try_new(&array).unwrap();
        let r = rasters.get(0).unwrap();
        let band = r.band(0).unwrap();

        assert_eq!(band.shape(), &[3]);
        let buf = band.nd_buffer().unwrap();
        assert_eq!(buf.shape, &[3]);
        assert_eq!(buf.strides, &[2]);
        assert_eq!(buf.offset, 1);
        // Strided → not C-order packed, so as_contiguous rejects it.
        assert!(!buf.is_contiguous());
        assert!(buf.as_contiguous().is_err());
        assert_eq!(gather_u8(&buf), vec![1, 3, 5]);
    }

    #[test]
    fn view_broadcast_zero_stride_repeats_source_row() {
        // 2D broadcast: source shape [1, 3], the view broadcasts axis 0 four
        // times (step 0) so every visible row equals the source's single row.
        let view = [ve(0, 0, 0, 4), ve(1, 0, 1, 3)];
        let array = build_viewed_u8(&[1, 3], &["row", "col"], &view, vec![10u8, 20, 30]);
        let rasters = RasterStructArray::try_new(&array).unwrap();
        let r = rasters.get(0).unwrap();
        let band = r.band(0).unwrap();

        assert_eq!(band.shape(), &[4, 3]);
        let buf = band.nd_buffer().unwrap();
        assert_eq!(buf.shape, &[4, 3]);
        assert_eq!(buf.strides, &[0, 1]);
        assert_eq!(buf.offset, 0);
        // Zero stride is not packed → non-contiguous, rejected.
        assert!(!buf.is_contiguous());
        assert!(buf.as_contiguous().is_err());
        assert_eq!(
            gather_u8(&buf),
            vec![10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 30]
        );
    }

    #[test]
    fn view_axis_permutation_and_slice_reads_expected_values() {
        // 2D source [Y=4, X=3], data = 0..12 row-major. The view permutes to
        // visible order [X, Y] and slices Y from start=1, step=2, steps=2.
        //   byte_strides = [step_X * src_stride_X, step_Y * src_stride_Y]
        //                = [1 * 1, 2 * 3] = [1, 6]
        //   byte_offset  = start_X * src_stride_X + start_Y * src_stride_Y
        //                = 0 * 1 + 1 * 3 = 3
        // Visible[i, j] (i over X 0..3, j over Y {1,3}) sits at source byte
        // 3 + i + 6j → C-order gather = [3, 9, 4, 10, 5, 11].
        let data: Vec<u8> = (0..12).collect();
        let view = [ve(1, 0, 1, 3), ve(0, 1, 2, 2)];
        let array = build_viewed_u8(&[4, 3], &["x", "y"], &view, data);
        let rasters = RasterStructArray::try_new(&array).unwrap();
        let r = rasters.get(0).unwrap();
        let band = r.band(0).unwrap();

        assert_eq!(band.shape(), &[3, 2]);
        let buf = band.nd_buffer().unwrap();
        assert_eq!(buf.shape, &[3, 2]);
        assert_eq!(buf.strides, &[1, 6]);
        assert_eq!(buf.offset, 3);
        assert!(!buf.is_contiguous());
        assert!(buf.as_contiguous().is_err());
        assert_eq!(gather_u8(&buf), vec![3, 9, 4, 10, 5, 11]);
    }

    #[test]
    fn view_reverse_negative_step_reads_expected_values() {
        // 1D source [0..8]; start=6, step=-2, steps=3 walks backwards picking
        // every other element: source indices 6, 4, 2.
        let data: Vec<u8> = (0..8).collect();
        let array = build_viewed_u8(&[8], &["x"], &[ve(0, 6, -2, 3)], data);
        let rasters = RasterStructArray::try_new(&array).unwrap();
        let r = rasters.get(0).unwrap();
        let band = r.band(0).unwrap();

        assert_eq!(band.shape(), &[3]);
        let buf = band.nd_buffer().unwrap();
        assert_eq!(buf.shape, &[3]);
        assert_eq!(buf.strides, &[-2]);
        assert_eq!(buf.offset, 6);
        // Negative stride is not packed → non-contiguous, rejected.
        assert!(!buf.is_contiguous());
        assert!(buf.as_contiguous().is_err());
        assert_eq!(gather_u8(&buf), vec![6, 4, 2]);
    }

    #[test]
    fn view_multidim_with_zero_axis_borrows_empty() {
        // A zero-extent middle axis addresses no bytes: the visible region is
        // empty, trivially contiguous, and as_contiguous borrows an empty slice.
        let view = [ve(0, 0, 1, 3), ve(1, 0, 1, 0), ve(2, 0, 1, 5)];
        let array = build_viewed_u8(&[3, 4, 5], &["a", "b", "c"], &view, vec![0u8; 60]);
        let rasters = RasterStructArray::try_new(&array).unwrap();
        let r = rasters.get(0).unwrap();
        let band = r.band(0).unwrap();

        assert_eq!(band.shape(), &[3, 0, 5]);
        let buf = band.nd_buffer().unwrap();
        assert_eq!(buf.shape, &[3, 0, 5]);
        assert!(buf.is_contiguous());
        assert!(buf.as_contiguous().unwrap().is_empty());
    }

    #[test]
    fn start_band_explicit_view_rejects_zero_dim() {
        // An explicit `Some` view must apply the same 0-D guard as the identity
        // path — accepting empty dim_names would otherwise bypass it via the
        // explicit view path.
        let mut builder = RasterBuilder::new(1);
        builder
            .start_raster_nd(&[0.0, 1.0, 0.0, 0.0, 0.0, -1.0], &[], &[], None)
            .unwrap();
        let err = builder
            .start_band(StartBandArgs {
                view: Some(&[]),
                ..StartBandArgs::new(&[], &[], BandDataType::UInt8)
            })
            .unwrap_err();
        assert!(
            err.to_string().contains("0-dimensional"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn start_band_explicit_view_rejects_step_overrun() {
        // start=1, step=2, steps=4 addresses element 1 + (4-1)*2 = 7, out of
        // range for a source axis of size 7 — validation must reject it before
        // any column is written.
        let mut builder = RasterBuilder::new(1);
        builder
            .start_raster_nd(&[0.0, 1.0, 0.0, 0.0, 0.0, -1.0], &[], &[], None)
            .unwrap();
        let err = builder
            .start_band(StartBandArgs {
                view: Some(&[ve(0, 1, 2, 4)]),
                ..StartBandArgs::new(&["x"], &[7], BandDataType::UInt8)
            })
            .unwrap_err();
        assert!(
            err.to_string().contains("out of range"),
            "unexpected error: {err}"
        );
    }

    // ---- with_view: public "create a new view into an existing band" ----

    /// Build a 1-D UInt8 raster with `source_shape=[8]` and bytes `[0..8]`.
    /// Identity-view; used as input to the with_view tests.
    fn build_1d_identity_raster() -> StructArray {
        let mut b = RasterBuilder::new(1);
        b.start_raster_nd(&[0.0, 1.0, 0.0, 0.0, 0.0, -1.0], &["x"], &[8], None)
            .unwrap();
        b.start_band(StartBandArgs::new(&["x"], &[8], BandDataType::UInt8))
            .unwrap();
        b.band_data_writer()
            .append_value((0u8..8).collect::<Vec<u8>>());
        b.finish_band().unwrap();
        b.finish_raster().unwrap();
        b.finish().unwrap()
    }

    #[test]
    fn with_view_over_identity_input_produces_expected_visible_bytes() {
        // Input is identity over [0..8]. with_view layers a slice
        // (start=1, step=2, steps=3) producing visible bytes [1, 3, 5].
        let input_array = build_1d_identity_raster();
        let input_rasters = RasterStructArray::try_new(&input_array).unwrap();
        let input_raster = input_rasters.get(0).unwrap();
        let input_band = input_raster.band(0).unwrap();

        let mut b = RasterBuilder::new(1);
        b.start_raster_nd(&[0.0, 1.0, 0.0, 0.0, 0.0, -1.0], &["x"], &[3], None)
            .unwrap();
        b.with_view(WithViewArgs {
            name: None,
            dim_names: &["x"],
            input: input_band.as_ref(),
            view: &[ve(0, 1, 2, 3)],
            nodata: None,
            outdb_uri: None,
            outdb_format: None,
        })
        .unwrap();
        b.finish_band().unwrap();
        b.finish_raster().unwrap();
        let out_array = b.finish().unwrap();

        let out_rasters = RasterStructArray::try_new(&out_array).unwrap();
        let out_raster = out_rasters.get(0).unwrap();
        let out_band = out_raster.band(0).unwrap();
        assert_eq!(out_band.shape(), &[3]);
        // The output's source_shape is inherited from the input.
        assert_eq!(out_band.raw_source_shape(), &[8]);
        let buf = out_band.nd_buffer().unwrap();
        assert_eq!(buf.strides, &[2]);
        assert_eq!(buf.offset, 1);
        assert!(!buf.is_contiguous());
        assert_eq!(gather_u8(&buf), vec![1, 3, 5]);
    }

    #[test]
    fn with_view_chained_composes_into_single_view() {
        // Round 1: with_view layers (start=1, step=2, steps=4) → visible bytes
        //          [1, 3, 5, 7] over source [0..8].
        // Round 2: with_view on that, layering (start=1, step=1, steps=2) →
        //          visible bytes [3, 5] (input visible indices 1 and 2).
        // compose collapses the chain into one source-space view; the read
        // back must reference the ORIGINAL 8-byte source.
        let input_array = build_1d_identity_raster();
        let input_rasters = RasterStructArray::try_new(&input_array).unwrap();
        let input_raster = input_rasters.get(0).unwrap();
        let input_band = input_raster.band(0).unwrap();

        // Round 1.
        let mut b1 = RasterBuilder::new(1);
        b1.start_raster_nd(&[0.0, 1.0, 0.0, 0.0, 0.0, -1.0], &["x"], &[4], None)
            .unwrap();
        b1.with_view(WithViewArgs {
            name: None,
            dim_names: &["x"],
            input: input_band.as_ref(),
            view: &[ve(0, 1, 2, 4)],
            nodata: None,
            outdb_uri: None,
            outdb_format: None,
        })
        .unwrap();
        b1.finish_band().unwrap();
        b1.finish_raster().unwrap();
        let mid_array = b1.finish().unwrap();

        let mid_rasters = RasterStructArray::try_new(&mid_array).unwrap();
        let mid_raster = mid_rasters.get(0).unwrap();
        let mid_band = mid_raster.band(0).unwrap();
        assert_eq!(mid_band.shape(), &[4]);
        assert_eq!(gather_u8(&mid_band.nd_buffer().unwrap()), vec![1, 3, 5, 7]);

        // Round 2: with_view applied on the view-bearing mid_band.
        let mut b2 = RasterBuilder::new(1);
        b2.start_raster_nd(&[0.0, 1.0, 0.0, 0.0, 0.0, -1.0], &["x"], &[2], None)
            .unwrap();
        b2.with_view(WithViewArgs {
            name: None,
            dim_names: &["x"],
            input: mid_band.as_ref(),
            view: &[ve(0, 1, 1, 2)],
            nodata: None,
            outdb_uri: None,
            outdb_format: None,
        })
        .unwrap();
        b2.finish_band().unwrap();
        b2.finish_raster().unwrap();
        let final_array = b2.finish().unwrap();

        let final_rasters = RasterStructArray::try_new(&final_array).unwrap();
        let final_raster = final_rasters.get(0).unwrap();
        let final_band = final_raster.band(0).unwrap();
        assert_eq!(final_band.shape(), &[2]);
        // The composed view still references the original 8-byte source.
        assert_eq!(final_band.raw_source_shape(), &[8]);
        let final_buf = final_band.nd_buffer().unwrap();
        assert_eq!(final_buf.strides, &[2]);
        assert_eq!(final_buf.offset, 3);
        assert!(!final_buf.is_contiguous());
        assert_eq!(gather_u8(&final_buf), vec![3, 5]);
    }

    #[test]
    fn with_view_on_outdb_input_produces_outdb_output_with_composed_view() {
        // Viewing an OutDb band doesn't need the source bytes — the output band
        // is itself OutDb, pointing at the same external resource via an
        // inherited outdb_uri, with the composed view describing the slice.
        let mut b = RasterBuilder::new(1);
        b.start_raster_nd(&[0.0, 1.0, 0.0, 0.0, 0.0, -1.0], &["x"], &[8], None)
            .unwrap();
        b.start_band(StartBandArgs {
            outdb_uri: Some("s3://bucket/file.tif#band=1"),
            outdb_format: Some("geotiff"),
            ..StartBandArgs::new(&["x"], &[8], BandDataType::UInt8)
        })
        .unwrap();
        b.band_data_writer().append_value([0u8; 0]); // empty → OutDb
        b.finish_band().unwrap();
        b.finish_raster().unwrap();
        let input_array = b.finish().unwrap();

        let input_rasters = RasterStructArray::try_new(&input_array).unwrap();
        let input_raster = input_rasters.get(0).unwrap();
        let input_band = input_raster.band(0).unwrap();
        assert!(!input_band.is_indb(), "fixture must be OutDb");

        let mut b2 = RasterBuilder::new(1);
        b2.start_raster_nd(&[0.0, 1.0, 0.0, 0.0, 0.0, -1.0], &["x"], &[3], None)
            .unwrap();
        b2.with_view(WithViewArgs {
            name: None,
            dim_names: &["x"],
            input: input_band.as_ref(),
            view: &[ve(0, 1, 2, 3)],
            nodata: None,
            outdb_uri: None,
            outdb_format: None,
        })
        .unwrap();
        b2.finish_band().unwrap();
        b2.finish_raster().unwrap();
        let out_array = b2.finish().unwrap();

        let out_rasters = RasterStructArray::try_new(&out_array).unwrap();
        let out_raster = out_rasters.get(0).unwrap();
        let out_band = out_raster.band(0).unwrap();

        assert!(
            !out_band.is_indb(),
            "output of OutDb-input with_view must be OutDb"
        );
        assert_eq!(out_band.outdb_uri(), Some("s3://bucket/file.tif#band=1"));
        assert_eq!(out_band.outdb_format(), Some("geotiff"));
        // Input had identity view, so composed == supplied view verbatim.
        assert_eq!(out_band.view(), &[ve(0, 1, 2, 3)]);
        assert_eq!(out_band.raw_source_shape(), &[8]);
        assert_eq!(out_band.shape(), &[3]);
    }

    #[test]
    fn with_view_inherits_source_nodata() {
        // Viewing a band must not drop its nodata sentinel. The earlier
        // implementation forwarded the caller's `nodata` (None here) verbatim
        // and never inherited the source's, so former-nodata pixels silently
        // became valid. Delegating through `copy_into` inherits it.
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        let mut ib = RasterBuilder::new(1);
        ib.start_raster_nd(&transform, &["x"], &[4], None).unwrap();
        ib.start_band(StartBandArgs {
            name: Some("orig"),
            nodata: Some(&[255u8]),
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
        assert_eq!(
            in_band.nodata(),
            Some(&[255u8][..]),
            "fixture must carry nodata"
        );

        // A non-identity view (every other element) with no explicit nodata
        // override — the source's [255] must carry over to the derived band.
        let mut ob = RasterBuilder::new(1);
        ob.start_raster_nd(&transform, &["x"], &[2], None).unwrap();
        ob.with_view(WithViewArgs {
            name: None,
            dim_names: &["x"],
            input: in_band.as_ref(),
            view: &[ve(0, 0, 2, 2)],
            nodata: None,
            outdb_uri: None,
            outdb_format: None,
        })
        .unwrap();
        ob.finish_band().unwrap();
        ob.finish_raster().unwrap();
        let out_array = ob.finish().unwrap();
        let out_rasters = RasterStructArray::try_new(&out_array).unwrap();
        let out_raster = out_rasters.get(0).unwrap();
        let out_band = out_raster.band(0).unwrap();

        assert_eq!(
            out_band.nodata(),
            Some(&[255u8][..]),
            "with_view must inherit the source band's nodata"
        );
        // The view is still applied: visible bytes are the every-other slice.
        assert_eq!(out_band.shape(), &[2]);
        assert_eq!(gather_u8(&out_band.nd_buffer().unwrap()), vec![1, 3]);
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
            .start_band(StartBandArgs::new(
                &["y", "x"],
                &[2, 2],
                BandDataType::UInt8,
            ))
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
            .start_band(StartBandArgs::new(
                &["y", "x"],
                &[4, 8],
                BandDataType::UInt8,
            ))
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
        // Schema invariant: a band built via start_band serialises with a
        // null view row, and the null must survive an Arrow IPC round-trip.
        // If a future change accidentally writes a non-null empty list
        // instead, downstream readers (DuckDB, PyArrow, sedona-py) will
        // disagree about whether the view is identity. A second raster carries
        // an explicit non-identity view to confirm the non-null row (and the
        // visible shape it decodes to) also survives the round-trip.

        let mut builder = RasterBuilder::new(2);
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        // Raster 0: identity-view band → null view row.
        builder
            .start_raster_nd(&transform, &["x", "y"], &[3, 2], None)
            .unwrap();
        builder
            .start_band(StartBandArgs::new(
                &["y", "x"],
                &[2, 3],
                BandDataType::UInt8,
            ))
            .unwrap();
        builder.band_data_writer().append_value(vec![0u8; 6]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        // Raster 1: explicit non-identity view → non-null view row.
        builder
            .start_raster_nd(&transform, &["x"], &[3], None)
            .unwrap();
        builder
            .start_band(StartBandArgs {
                view: Some(&[ve(0, 1, 2, 3)]),
                ..StartBandArgs::new(&["x"], &[8], BandDataType::UInt8)
            })
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

        let rasters = RasterStructArray::try_new(restored_struct).unwrap();
        let r0 = rasters.get(0).unwrap();
        assert_eq!(r0.band(0).unwrap().shape(), &[2, 3]);
        let r1 = rasters.get(1).unwrap();
        assert_eq!(r1.band(0).unwrap().shape(), &[3]);
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
