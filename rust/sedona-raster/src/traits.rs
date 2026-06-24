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
use sedona_schema::raster::BandDataType;

use crate::builder::{RasterBuilder, StartBandWithViewArgs};
use crate::view_entries::{ViewEntries, ViewEntry};

/// Recognized spatial dimension-name pairs, in band C-order: the slower-
/// varying Y-like (row) axis first, the faster-varying X-like (column) axis
/// second. A band whose trailing two `dim_names` match one of these pairs is
/// treated as a georeferenced 2-D layout without needing an explicit
/// `spatial:dims` declaration. CF-style `lat`/`lon` and
/// `latitude`/`longitude` layouts are accepted alongside the native `y`/`x`.
pub const SPATIAL_DIM_NAME_PAIRS: [(&str, &str); 3] =
    [("y", "x"), ("lat", "lon"), ("latitude", "longitude")];

/// True iff `(y_like, x_like)` is a recognized spatial dimension-name pair
/// (see [`SPATIAL_DIM_NAME_PAIRS`]). Arguments are in band C-order: the
/// Y-like (row) axis first, the X-like (column) axis second.
pub fn is_spatial_dim_pair(y_like: &str, x_like: &str) -> bool {
    SPATIAL_DIM_NAME_PAIRS
        .iter()
        .any(|&(y, x)| y == y_like && x == x_like)
}

/// View into a band's N-D data buffer with layout metadata.
///
/// `shape`, `strides`, and `offset` describe the *visible* region in
/// byte-stride terms — they are computed by composing the band's
/// `source_shape` (the natural extent of `buffer`) with its `view`
/// (the per-axis `(source_axis, start, step, steps)` slice spec). Stride
/// can be zero (broadcast) or negative (reverse iteration), and may not be
/// C-order. Consumers that need a flat row-major buffer should use
/// `BandRef::contiguous_data()` instead.
///
/// Only `buffer` is tied to the producer's lifetime `'a` (it can be tens of
/// MBs of pixel data and must not be copied). `shape` and `strides` are
/// owned `Vec`s — they're tiny (ndim ≤ a handful) so an allocation here is
/// negligible, and owning them lets an `NdBuffer` outlive the producer's
/// internal layout cache (e.g. cross-thread, return-by-value).
#[derive(Debug)]
pub struct NdBuffer<'a> {
    pub buffer: &'a [u8],
    pub shape: Vec<i64>,
    pub strides: Vec<i64>,
    pub offset: u64,
    pub data_type: BandDataType,
}

impl<'a> NdBuffer<'a> {
    /// True iff the visible region is packed in C-order (row-major) with no
    /// gaps — the byte strides equal the canonical innermost-fastest layout
    /// over `shape`. Offset-agnostic. A strided slice, broadcast, or
    /// permutation is not contiguous.
    pub fn is_contiguous(&self) -> bool {
        if self.shape.len() != self.strides.len() {
            return false;
        }
        // An empty visible region is trivially packed.
        if self.shape.contains(&0) {
            return true;
        }
        // Expected C-order byte strides, innermost first:
        // stride[i] == byte_size × Π_{j>i} shape[j].
        let mut expected = self.data_type.byte_size() as i64;
        for (&dim, &stride) in self.shape.iter().zip(self.strides.iter()).rev() {
            if stride != expected {
                return false;
            }
            expected = expected.saturating_mul(dim);
        }
        true
    }

    /// The visible bytes as a packed row-major slice, borrowed zero-copy
    /// from `buffer`. `Ok` iff [`is_contiguous`](Self::is_contiguous);
    /// otherwise an error directing the caller to materialize via
    /// `RS_EnsureContiguous`
    /// (<https://github.com/apache/sedona-db/issues/899>). Never copies or
    /// allocates — a strided layout returns an error, it is not materialized
    /// here.
    pub fn as_contiguous(&self) -> Result<&'a [u8], ArrowError> {
        if !self.is_contiguous() {
            return Err(ArrowError::InvalidArgumentError(format!(
                "band view is not contiguous (shape {:?}, strides {:?}); \
                 materialize it with RS_EnsureContiguous before contiguous \
                 byte access — see https://github.com/apache/sedona-db/issues/899",
                self.shape, self.strides
            )));
        }
        let elements: i64 = self.shape.iter().product();
        let len = elements as usize * self.data_type.byte_size();
        let start = self.offset as usize;
        let buffer: &'a [u8] = self.buffer;
        match start.checked_add(len) {
            Some(end) if end <= buffer.len() => Ok(&buffer[start..end]),
            _ => Err(ArrowError::ExternalError(Box::new(
                sedona_common::sedona_internal_datafusion_err!(
                    "contiguous region [{start}, {start}+{len}) is out of bounds \
                     for buffer of length {}",
                    buffer.len()
                ),
            ))),
        }
    }
}

/// Concrete raster metadata returned by `RasterRef::metadata()`.
///
/// Restored from the pre-N-D schema to keep callers that pattern-match on
/// `metadata.width`, `metadata.upperleft_x`, etc. compiling. Computed
/// eagerly from `RasterRef::transform()` and `RasterRef::spatial_shape()`.
///
/// Panics on construction (`metadata()`) if the raster lacks width or
/// height — corrupt schemas error through the `width()`/`height()` trait
/// methods directly; the metadata accessor is the convenience surface.
#[derive(Debug, Clone)]
pub struct RasterMetadata {
    pub width: i64,
    pub height: i64,
    pub upperleft_x: f64,
    pub upperleft_y: f64,
    pub scale_x: f64,
    pub scale_y: f64,
    pub skew_x: f64,
    pub skew_y: f64,
}

/// Pre-N-D metadata-accessor trait. Restored so callers from before the
/// N-D refactor that write `fn foo(metadata: &dyn MetadataRef)` keep
/// compiling. `RasterMetadata` is the canonical implementer; new code
/// should reach for `RasterRef::width()? / height()?` instead.
pub trait MetadataRef {
    /// Width of the raster in pixels
    fn width(&self) -> i64;
    /// Height of the raster in pixels
    fn height(&self) -> i64;
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

impl MetadataRef for RasterMetadata {
    fn width(&self) -> i64 {
        self.width
    }
    fn height(&self) -> i64 {
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

impl RasterMetadata {
    pub fn width(&self) -> i64 {
        self.width
    }
    pub fn height(&self) -> i64 {
        self.height
    }
    pub fn upper_left_x(&self) -> f64 {
        self.upperleft_x
    }
    pub fn upper_left_y(&self) -> f64 {
        self.upperleft_y
    }
    pub fn scale_x(&self) -> f64 {
        self.scale_x
    }
    pub fn scale_y(&self) -> f64 {
        self.scale_y
    }
    pub fn skew_x(&self) -> f64 {
        self.skew_x
    }
    pub fn skew_y(&self) -> f64 {
        self.skew_y
    }
}

/// Concrete band metadata returned by `BandRef::metadata()`.
///
/// Restored from the pre-N-D schema. The `outdb_url` and `outdb_band_id`
/// fields are eagerly parsed from the N-D `outdb_uri` (which carries a
/// `#band=N` fragment in the SedonaDB convention) so callers from the
/// pre-N-D era keep compiling against the same field names.
#[derive(Debug, Clone)]
pub struct BandMetadata {
    pub nodata_value: Option<Vec<u8>>,
    pub storage_type: sedona_schema::raster::StorageType,
    pub datatype: BandDataType,
    pub outdb_url: Option<String>,
    pub outdb_band_id: Option<u32>,
}

impl BandMetadata {
    pub fn nodata_value(&self) -> Option<&[u8]> {
        self.nodata_value.as_deref()
    }
    /// Returns the storage type. Wrapped in `Result` to match main's
    /// `BandMetadataRef::storage_type()` signature — our shim
    /// implementation never errors, but the signature is preserved so
    /// existing `matches!(band.metadata().storage_type(), Ok(...))`
    /// patterns from before the N-D refactor keep compiling.
    pub fn storage_type(&self) -> Result<sedona_schema::raster::StorageType, ArrowError> {
        Ok(self.storage_type)
    }
    /// Returns the band data type. Wrapped in `Result` to match main's
    /// `BandMetadataRef::data_type()` signature — see `storage_type()`.
    pub fn data_type(&self) -> Result<BandDataType, ArrowError> {
        Ok(self.datatype)
    }
    pub fn outdb_url(&self) -> Option<&str> {
        self.outdb_url.as_deref()
    }
    pub fn outdb_band_id(&self) -> Option<u32> {
        self.outdb_band_id
    }
    /// Nodata value interpreted as f64. Mirrors the pre-N-D
    /// `BandMetadataRef::nodata_value_as_f64()`. Uses the lossless
    /// conversion (errors on i64/u64 magnitudes > 2^53) so the shim
    /// surface picks up the same correctness fix as
    /// `BandRef::nodata_as_f64()`.
    pub fn nodata_value_as_f64(&self) -> Result<Option<f64>, ArrowError> {
        let bytes = match self.nodata_value.as_deref() {
            Some(b) => b,
            None => return Ok(None),
        };
        nodata_bytes_to_f64_lossless(bytes, &self.datatype).map(Some)
    }
}

/// Parse the SedonaDB `#band=N` fragment out of an out-DB URI.
/// Returns `(base_url, band_id)`; band_id defaults to 1 if absent.
/// Duplicated (intentionally — and minimally) from
/// `sedona-raster-gdal::source_uri` because the shim lives in
/// `sedona-raster` and can't reach across the crate boundary.
fn split_outdb_band_fragment(uri: &str) -> (String, u32) {
    if let Some(hash_pos) = uri.rfind('#') {
        let (base, fragment) = uri.split_at(hash_pos);
        let fragment = &fragment[1..]; // skip the '#'
        if let Some(rest) = fragment.strip_prefix("band=") {
            if let Ok(n) = rest.parse::<u32>() {
                return (base.to_string(), n);
            }
        }
    }
    (uri.to_string(), 1)
}

/// Iteration view over a raster's bands. Returned by `RasterRef::bands()`.
///
/// Wraps a borrowed `&dyn RasterRef` and offers the `len()` / `band(1-based)`
/// / `iter()` shape that callers used before the N-D refactor. New code can
/// equivalently use `RasterRef::num_bands()` and `RasterRef::band(0-based)`
/// directly; both call patterns coexist.
pub struct Bands<'a> {
    raster: &'a dyn RasterRef,
}

impl<'a> Bands<'a> {
    /// Wrap a `&dyn RasterRef` for the legacy 1-based band-access surface.
    pub fn new(raster: &'a dyn RasterRef) -> Self {
        Self { raster }
    }
}

impl<'a> Bands<'a> {
    /// Number of bands in the raster.
    pub fn len(&self) -> usize {
        self.raster.num_bands()
    }

    /// True iff the raster has zero bands.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Look up a band by **1-based** number. Returns an error rather than
    /// `None` so callers can use `?`. For 0-based access, use
    /// `RasterRef::band` directly.
    pub fn band(&self, number: usize) -> Result<Box<dyn BandRef + 'a>, ArrowError> {
        if number == 0 {
            return Err(ArrowError::InvalidArgumentError(format!(
                "Invalid band number {number}: band numbers must be 1-based"
            )));
        }
        self.raster.band(number - 1)
    }

    /// Iterate over every band in 0-based order. Yields `Result` so that
    /// a corrupt band surfaces as an error rather than being silently
    /// dropped from the iteration.
    pub fn iter(&self) -> impl Iterator<Item = Result<Box<dyn BandRef + 'a>, ArrowError>> + 'a {
        let raster = self.raster;
        (0..raster.num_bands()).map(move |i| raster.band(i))
    }
}

/// Trait for accessing an N-dimensional raster (top level).
///
/// Replaces the legacy `RasterRef` + `MetadataRef` + `BandsRef` hierarchy with
/// a single flat interface. Bands are 0-indexed.
pub trait RasterRef {
    /// Number of bands/variables
    fn num_bands(&self) -> usize;

    /// Access a band by 0-based index. Returns an `ArrowError` when the
    /// index is out of range or when the underlying schema is malformed
    /// (unknown data-type discriminant, corrupt view, etc.). The latter
    /// cases route through `sedona_common::sedona_internal_datafusion_err!`
    /// so they carry the standardised "SedonaDB internal error" framing.
    fn band(&self, index: usize) -> Result<Box<dyn BandRef + '_>, ArrowError>;

    /// 1-based band-access view used by callers from before the N-D
    /// refactor. Implementers typically write `Bands::new(self)`.
    fn bands(&self) -> Bands<'_>;

    /// Band name (e.g., Zarr variable name). None for unnamed bands.
    fn band_name(&self, index: usize) -> Option<&str>;

    /// Fast path for band data type — reads the scalar `data_type` column
    /// without materialising a full `BandRef`. UDFs that only need this
    /// metadata field should prefer this over `band(i)?.data_type()`.
    /// Returns None if `index` is out of range or the discriminant is invalid.
    ///
    /// The default implementation delegates to `band(i)`. Backends with a
    /// flat columnar layout should override for the no-allocation fast path.
    fn band_data_type(&self, index: usize) -> Option<BandDataType> {
        // Fast-path accessor: corrupt bands and out-of-range indices both
        // collapse to `None`. Callers that need to distinguish the two
        // should use `band(index)` directly.
        self.band(index).ok().map(|b| b.data_type())
    }

    /// Fast path for band outdb URI — reads the `outdb_uri` column without
    /// materialising a `BandRef`. Returns None if the band has no URI or
    /// if `index` is out of range.
    ///
    /// The default implementation must allocate a `Box<dyn BandRef>`; the
    /// raster-array backend overrides it to read the column directly.
    /// Default returns None because the borrow can't outlive the boxed band.
    fn band_outdb_uri(&self, index: usize) -> Option<&str> {
        let _ = index;
        None
    }

    /// Fast path for band outdb format — reads the `outdb_format` column
    /// without materialising a `BandRef`. Default returns None for the
    /// same lifetime reason as `band_outdb_uri`.
    fn band_outdb_format(&self, index: usize) -> Option<&str> {
        let _ = index;
        None
    }

    /// Fast path for band nodata bytes — reads the `nodata` column without
    /// materialising a `BandRef`. Default returns None for the same
    /// lifetime reason as `band_outdb_uri`.
    fn band_nodata(&self, index: usize) -> Option<&[u8]> {
        let _ = index;
        None
    }

    /// CRS string (PROJJSON, WKT, or authority code). None if not set.
    fn crs(&self) -> Option<&str>;

    /// 6-element affine transform in GDAL GeoTransform order:
    /// `[origin_x, scale_x, skew_x, origin_y, skew_y, scale_y]`
    fn transform(&self) -> &[f64];

    /// Eagerly-computed concrete metadata view (width, height, geotransform
    /// scalars). Mirrors the pre-N-D `RasterRef::metadata()` accessor.
    ///
    /// Panics if `spatial_shape` lacks width/height or `transform` is the
    /// wrong length — those are corrupt-schema cases that error cleanly
    /// through the `width()`/`height()` trait methods, but the metadata
    /// accessor predates that contract and is kept infallible for caller
    /// ergonomics.
    fn metadata(&self) -> RasterMetadata {
        let width = self
            .width()
            .expect("raster has no width (spatial_shape missing); use width()? for error handling");
        let height = self
            .height()
            .expect("raster has no height; use height()? for error handling");
        let t = self.transform();
        if t.len() != 6 {
            panic!("transform must be 6 elements, got {}", t.len());
        }
        RasterMetadata {
            width,
            height,
            upperleft_x: t[0],
            scale_x: t[1],
            skew_x: t[2],
            upperleft_y: t[3],
            skew_y: t[4],
            scale_y: t[5],
        }
    }

    /// Spatial dimension names, in order (today `["x","y"]`; a future Z phase
    /// would extend to `["x","y","z"]`). Every band must contain each of these
    /// names in its own `dim_names`, with matching sizes.
    fn spatial_dims(&self) -> Vec<&str>;

    /// Spatial dimension sizes, in the same order as `spatial_dims`. Today
    /// `[width, height]`.
    fn spatial_shape(&self) -> &[i64];

    /// Name of the X spatial dimension (e.g., "x", "lon", "easting").
    fn x_dim(&self) -> &str {
        let dims = self.spatial_dims();
        dims.into_iter().next().unwrap_or("x")
    }

    /// Name of the Y spatial dimension (e.g., "y", "lat", "northing").
    fn y_dim(&self) -> &str {
        let dims = self.spatial_dims();
        dims.into_iter().nth(1).unwrap_or("y")
    }

    /// Width in pixels — size of the X spatial dimension from the top-level
    /// `spatial_shape`. Errors if `spatial_shape` is empty, which is an
    /// invariant violation rather than a legitimate "no value" state.
    fn width(&self) -> Result<i64, ArrowError> {
        let shape = self.spatial_shape();
        shape.first().copied().ok_or_else(|| {
            ArrowError::InvalidArgumentError(
                "raster has no width (spatial_shape is empty)".to_string(),
            )
        })
    }

    /// Height in pixels — size of the Y spatial dimension from the top-level
    /// `spatial_shape`. Errors if `spatial_shape` has fewer than two entries.
    fn height(&self) -> Result<i64, ArrowError> {
        let shape = self.spatial_shape();
        shape.get(1).copied().ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!(
                "raster has no height (spatial_shape has {} entries, need >= 2)",
                shape.len()
            ))
        })
    }

    /// Look up a band by name. Returns an error if no band has that
    /// name or if the matching band is malformed.
    fn band_by_name(&self, name: &str) -> Result<Box<dyn BandRef + '_>, ArrowError> {
        let i = (0..self.num_bands())
            .find(|&i| self.band_name(i) == Some(name))
            .ok_or_else(|| {
                ArrowError::InvalidArgumentError(format!("Band with name '{name}' not found"))
            })?;
        self.band(i)
    }
}

/// Field overrides for [`BandRef::copy_into`]. Each field defaults to `None`,
/// meaning "inherit from the source band". `name` has no source on a `BandRef`
/// (band names live at the raster level), so it defaults to unnamed.
#[derive(Default)]
pub struct BandOverrides<'a> {
    /// Name for the derived band (the source has none to inherit).
    pub name: Option<&'a str>,
    /// Override the dimension names; `None` inherits the source's.
    pub dim_names: Option<&'a [&'a str]>,
    /// Override the nodata value; `None` inherits the source's.
    pub nodata: Option<&'a [u8]>,
    /// Override the OutDb URI; `None` inherits the source's.
    pub outdb_uri: Option<&'a str>,
    /// Override the OutDb format; `None` inherits the source's.
    pub outdb_format: Option<&'a str>,
    /// View to apply to the derived band, expressed in the **source's visible
    /// coordinates**. [`BandRef::copy_into`] composes it onto the source's own
    /// view for you — you don't manage that composition and don't need to know
    /// whether the source already carries a view. `None` inherits the source's
    /// view unchanged. (A non-identity result isn't persistable yet and
    /// `copy_into` rejects it; see <https://github.com/apache/sedona-db/issues/897>.)
    pub view: Option<&'a [ViewEntry]>,
}

/// Trait for accessing a single band/variable within an N-D raster.
///
/// This is the consumer interface. Implementations handle storage details
/// Two data access paths:
/// - `contiguous_data()` — flat row-major bytes for consumers that don't need
///   stride awareness (most RS_* functions, GDAL boundary, serialization).
/// - `nd_buffer()` — raw buffer + shape + strides + offset for stride-aware
///   consumers (numpy zero-copy views, Arrow FFI) that want to avoid copies.
pub trait BandRef {
    // -- Dimension metadata --

    /// Number of dimensions in this band
    fn ndim(&self) -> usize;

    /// Dimension names in order (e.g., `["time", "y", "x"]`)
    fn dim_names(&self) -> Vec<&str>;

    /// Visible shape — size of each dimension in the band's view, in
    /// `dim_names` order. Derived from `view`: `[v.steps for v in view]`.
    /// This is what almost all consumers want; use `raw_source_shape()` only
    /// when you need to address into the raw `data` buffer (e.g. FFI).
    fn shape(&self) -> &[i64];

    /// **Internal/FFI-only.** Natural C-order extent of the band's
    /// underlying `data` buffer, indexed by *source* axis (not visible
    /// axis). Almost every consumer wants `shape()` instead — that is the
    /// region the band exposes, and is what you compare against
    /// `spatial_shape`, iterate over for pixels, and compose further views
    /// against. The two only agree when the band's view is the identity;
    /// any slice, broadcast, or permutation makes them diverge.
    ///
    /// Use this only when you need to index directly into the raw `data`
    /// bytes (e.g. Arrow C Data Interface, numpy zero-copy views) and you
    /// also handle `view()` and the byte-stride layout from `nd_buffer()`.
    fn raw_source_shape(&self) -> &[i64];

    /// Per-visible-dimension view entries describing how the band's
    /// visible axes map onto its `source_shape`. `view().len() == ndim()`.
    /// See `ViewEntry` for per-entry semantics.
    fn view(&self) -> &[ViewEntry];

    /// Size of a named dimension (None if doesn't exist)
    fn dim_size(&self, name: &str) -> Option<i64> {
        let idx = self.dim_index(name)?;
        Some(self.shape()[idx])
    }

    /// Index of a named dimension (None if doesn't exist)
    fn dim_index(&self, name: &str) -> Option<usize> {
        self.dim_names().iter().position(|n| *n == name)
    }

    /// True iff this band has a recognized 2-D spatial *shape*: exactly two
    /// dimensions named as a spatial pair (`["y", "x"]`, `["lat", "lon"]`, or
    /// `["latitude", "longitude"]`; see [`is_spatial_dim_pair`]).
    ///
    /// This is a shape/naming predicate only — it says nothing about the
    /// band's view or byte layout. Callers that also need the bytes laid out
    /// contiguously (e.g. the GDAL boundary, which hands GDAL a zero-copy
    /// pointer) pair this with a contiguity check via `nd_buffer()?` +
    /// [`NdBuffer::as_contiguous`], which borrows on a contiguous view and
    /// errors otherwise (pointing the caller at `RS_EnsureContiguous`).
    fn is_spatial_2d(&self) -> bool {
        let dims = self.dim_names();
        dims.len() == 2 && is_spatial_dim_pair(dims[0], dims[1])
    }

    // -- Band metadata --

    /// Data type for all elements in this band
    fn data_type(&self) -> BandDataType;

    /// Nodata value as raw bytes (None if not set)
    fn nodata(&self) -> Option<&[u8]>;

    /// OutDb URI — location of the external resource (e.g.
    /// `"s3://bucket/file.tif"`, `"file:///…"`, `"mem://…"`). None for
    /// in-memory bands. Scheme resolution is delegated to an
    /// `ObjectStoreRegistry`; it does *not* imply a format.
    fn outdb_uri(&self) -> Option<&str> {
        None
    }

    /// OutDb format — how to interpret the bytes at `outdb_uri`
    /// (e.g. `"geotiff"`, `"zarr"`). None means in-memory — the band's
    /// `contiguous_data()` / `nd_buffer()` is authoritative.
    fn outdb_format(&self) -> Option<&str> {
        None
    }

    /// True if this band's bytes live in the `data` buffer (in-database).
    /// False if the bytes must be fetched from `outdb_uri` (out-of-database).
    ///
    /// The discriminator is whether the `data` buffer is non-empty —
    /// `outdb_uri` and `outdb_format` are orthogonal location/format hints
    /// that may be set on either kind of band.
    /// A band whose visible region is empty (`Π shape() == 0`) holds no
    /// readable bytes and is always InDb — there is nothing to load. This is
    /// what separates a legitimately-empty InDb band from the OutDb empty-`data`
    /// sentinel.
    ///
    /// Required (no default): a band's storage location is a primitive fact each
    /// impl answers directly. `nd_buffer()` depends on `is_indb()`, so any
    /// `nd_buffer`-based default would recurse.
    fn is_indb(&self) -> bool;

    /// Eagerly-computed concrete band metadata. Mirrors the pre-N-D
    /// `BandRef::metadata()` accessor.
    ///
    /// `outdb_url` and `outdb_band_id` are parsed from `outdb_uri()`'s
    /// SedonaDB `#band=N` fragment convention so callers that pattern-match
    /// on those fields keep compiling.
    fn metadata(&self) -> BandMetadata {
        let is_indb = self.is_indb();
        // Match the pre-N-D contract: outdb_url / outdb_band_id are only
        // populated when storage_type is OutDbRef. The current schema lets
        // the URI hint coexist with InDb data; this surface hides that.
        let (outdb_url, outdb_band_id) = if !is_indb {
            match self.outdb_uri() {
                Some(uri) => {
                    let (base, band) = split_outdb_band_fragment(uri);
                    (Some(base), Some(band))
                }
                None => (None, None),
            }
        } else {
            (None, None)
        };
        BandMetadata {
            nodata_value: self.nodata().map(|b| b.to_vec()),
            storage_type: if is_indb {
                sedona_schema::raster::StorageType::InDb
            } else {
                sedona_schema::raster::StorageType::OutDbRef
            },
            datatype: self.data_type(),
            outdb_url,
            outdb_band_id,
        }
    }

    // -- Data access --

    /// Raw backing buffer + visible-region layout. Triggers load for lazy
    /// impls. The returned `NdBuffer` describes the band's view in
    /// byte-stride terms — `shape` is the visible shape, `strides` and
    /// `offset` are computed by composing the view with the source's
    /// natural C-order byte strides. Strides may be zero (broadcast) or
    /// negative (reverse iteration).
    fn nd_buffer(&self) -> Result<NdBuffer<'_>, ArrowError>;

    /// Nodata value interpreted as f64.
    ///
    /// Returns `Ok(None)` when no nodata value is defined, `Ok(Some(f64))` on
    /// success, or an error when the raw bytes have an unexpected length **or**
    /// when the nodata value cannot be represented exactly in `f64`.
    ///
    /// 64-bit integer bands (`Int64`, `UInt64`) error rather than silently
    /// rounding when the magnitude exceeds 2^53 — values outside
    /// `[-9_007_199_254_740_992, 9_007_199_254_740_992]` can't round-trip
    /// through `f64` and a rounded sentinel can collide with a real pixel
    /// value. Use `nodata()` directly to recover the exact bytes when full
    /// integer precision matters (e.g. when nodata is the type's extreme
    /// value like `0xFF…FF`).
    fn nodata_as_f64(&self) -> Result<Option<f64>, ArrowError> {
        let bytes = match self.nodata() {
            Some(b) => b,
            None => return Ok(None),
        };
        nodata_bytes_to_f64_lossless(bytes, &self.data_type()).map(Some)
    }

    /// Write a derived band into `builder`, inheriting every field not set in
    /// `overrides` from `self`, and carrying the source bytes over.
    ///
    /// This is the canonical "derive a band from an existing one" path — it
    /// replaces hand-rebuilding via `start_band_nd` + a manual data append. The
    /// data transfer is zero-copy when the implementation supports it (see
    /// [`Self::append_data_into`]).
    ///
    /// The derived band's view is the source's own view with any
    /// `overrides.view` **composed on top for you**: express an override in the
    /// source's *visible* coordinates and `copy_into` composes it against the
    /// source's view — callers never manage that composition and don't need to
    /// know whether the source already carries one. `overrides.view = None`
    /// inherits the source's view unchanged.
    ///
    /// The composition + persistence is delegated to
    /// [`RasterBuilder::start_band_with_view`]. Today that stores views only as
    /// the canonical identity null sentinel, so a non-identity effective view
    /// is rejected rather than copying mislocated bytes; in practice the source
    /// is identity-viewed and any override must compose back to the identity.
    /// View persistence is tracked in
    /// <https://github.com/apache/sedona-db/issues/897>.
    fn copy_into(
        &self,
        builder: &mut RasterBuilder,
        overrides: BandOverrides<'_>,
    ) -> Result<(), ArrowError> {
        let inherited_dims = self.dim_names();
        let dim_names: Vec<&str> = match overrides.dim_names {
            Some(d) => d.to_vec(),
            None => inherited_dims,
        };
        let source_shape = self.raw_source_shape().to_vec();
        // Compose the caller's override (if any) onto the source's own view, so
        // the override is interpreted in the source's visible space and the
        // caller doesn't have to. `None` keeps the source view unchanged.
        let source_view = ViewEntries::new(self.view().to_vec());
        let effective_view = match overrides.view {
            Some(v) => source_view.compose(&ViewEntries::new(v.to_vec()))?,
            None => source_view,
        };
        builder.start_band_with_view(StartBandWithViewArgs {
            name: overrides.name,
            dim_names: &dim_names,
            source_shape: &source_shape,
            view: effective_view.as_slice(),
            data_type: self.data_type(),
            nodata: overrides.nodata.or_else(|| self.nodata()),
            outdb_uri: overrides.outdb_uri.or_else(|| self.outdb_uri()),
            outdb_format: overrides.outdb_format.or_else(|| self.outdb_format()),
        })?;
        self.append_data_into(builder)
    }

    /// Append `self`'s band data as the current band's single `data` value.
    ///
    /// The default copies the visible source bytes via `append_value`. Arrow-
    /// backed implementations override this to share the source row's backing
    /// `Buffer` zero-copy (a refcount bump via
    /// [`RasterBuilder::append_band_data_from`]), keeping the buffer plumbing
    /// encapsulated rather than exposing a raw `Buffer` accessor. Call after the
    /// band's schema has been written (e.g. by [`Self::copy_into`]).
    fn append_data_into(&self, builder: &mut RasterBuilder) -> Result<(), ArrowError> {
        if self.is_indb() {
            let ndb = self.nd_buffer()?;
            builder.band_data_writer().append_value(ndb.buffer);
        } else {
            builder.band_data_writer().append_value([]);
        }
        Ok(())
    }
}

/// Convert raw nodata bytes to f64 given a [`BandDataType`].
///
/// The bytes are expected to be in little-endian order and exactly match the
/// byte size of the data type. Internal helper for the lossless wrapper;
/// non-i64/u64 callers reach for `nodata_bytes_to_f64_lossless` instead.
fn nodata_bytes_to_f64(bytes: &[u8], dt: &BandDataType) -> Result<f64, ArrowError> {
    macro_rules! read_le {
        ($t:ty, $n:expr) => {{
            let arr: [u8; $n] = bytes.try_into().map_err(|_| {
                ArrowError::InvalidArgumentError(format!(
                    "Invalid nodata byte length for {:?}: expected {}, got {}",
                    dt,
                    $n,
                    bytes.len()
                ))
            })?;
            Ok(<$t>::from_le_bytes(arr) as f64)
        }};
    }

    match dt {
        BandDataType::UInt8 => {
            if bytes.len() != 1 {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "Invalid nodata byte length for UInt8: expected 1, got {}",
                    bytes.len()
                )));
            }
            Ok(bytes[0] as f64)
        }
        BandDataType::Int8 => {
            if bytes.len() != 1 {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "Invalid nodata byte length for Int8: expected 1, got {}",
                    bytes.len()
                )));
            }
            Ok(bytes[0] as i8 as f64)
        }
        BandDataType::UInt16 => read_le!(u16, 2),
        BandDataType::Int16 => read_le!(i16, 2),
        BandDataType::UInt32 => read_le!(u32, 4),
        BandDataType::Int32 => read_le!(i32, 4),
        BandDataType::UInt64 => read_le!(u64, 8),
        BandDataType::Int64 => read_le!(i64, 8),
        BandDataType::Float32 => read_le!(f32, 4),
        BandDataType::Float64 => read_le!(f64, 8),
    }
}

/// Convert raw nodata bytes to f64, erroring on lossy conversion.
///
/// Like [`nodata_bytes_to_f64`] but rejects 64-bit integer values whose
/// magnitude exceeds 2^53, since they can't round-trip through `f64`.
/// Callers that interpret nodata as a sentinel (e.g. UDFs that compare
/// pixel == nodata) should prefer this over the lossy variant — a rounded
/// `0xFFFF_FFFF_FFFF_FFFE` sentinel can silently collide with a real
/// pixel value.
pub fn nodata_bytes_to_f64_lossless(bytes: &[u8], dt: &BandDataType) -> Result<f64, ArrowError> {
    match dt {
        BandDataType::UInt64 => {
            let arr: [u8; 8] = bytes.try_into().map_err(|_| {
                ArrowError::InvalidArgumentError(format!(
                    "Invalid nodata byte length for UInt64: expected 8, got {}",
                    bytes.len()
                ))
            })?;
            let v = u64::from_le_bytes(arr);
            if v > (1u64 << 53) {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "UInt64 nodata value {v} cannot be represented exactly as f64 \
                     (magnitude > 2^53); use the raw nodata bytes instead"
                )));
            }
            Ok(v as f64)
        }
        BandDataType::Int64 => {
            let arr: [u8; 8] = bytes.try_into().map_err(|_| {
                ArrowError::InvalidArgumentError(format!(
                    "Invalid nodata byte length for Int64: expected 8, got {}",
                    bytes.len()
                ))
            })?;
            let v = i64::from_le_bytes(arr);
            if v.unsigned_abs() > (1u64 << 53) {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "Int64 nodata value {v} cannot be represented exactly as f64 \
                     (magnitude > 2^53); use the raw nodata bytes instead"
                )));
            }
            Ok(v as f64)
        }
        _ => nodata_bytes_to_f64(bytes, dt),
    }
}

/// Pack an `f64` nodata value into the little-endian bytes of a band's data
/// type — the inverse of [`nodata_bytes_to_f64_lossless`].
///
/// Errors when the value can't be represented exactly in `dt`: a non-integral
/// value for an integer type, a value outside the type's range, or a 64-bit
/// integer beyond 2^53 (which can't have arrived losslessly through `f64`).
/// `Float32` is the one lossy case allowed — it rounds to the nearest `f32`, as
/// any f64 → f32 narrowing does.
pub fn nodata_f64_to_bytes(value: f64, dt: &BandDataType) -> Result<Vec<u8>, ArrowError> {
    fn check_integer(value: f64, min: f64, max: f64, dt: &BandDataType) -> Result<(), ArrowError> {
        if value.fract() != 0.0 || value < min || value > max {
            return Err(ArrowError::InvalidArgumentError(format!(
                "nodata value {value} is not a valid {dt:?} value"
            )));
        }
        Ok(())
    }

    // The largest magnitude an integer can have and still round-trip through f64.
    const F64_INT_MAX: f64 = (1u64 << 53) as f64;

    Ok(match dt {
        BandDataType::UInt8 => {
            check_integer(value, 0.0, u8::MAX as f64, dt)?;
            (value as u8).to_le_bytes().to_vec()
        }
        BandDataType::Int8 => {
            check_integer(value, i8::MIN as f64, i8::MAX as f64, dt)?;
            (value as i8).to_le_bytes().to_vec()
        }
        BandDataType::UInt16 => {
            check_integer(value, 0.0, u16::MAX as f64, dt)?;
            (value as u16).to_le_bytes().to_vec()
        }
        BandDataType::Int16 => {
            check_integer(value, i16::MIN as f64, i16::MAX as f64, dt)?;
            (value as i16).to_le_bytes().to_vec()
        }
        BandDataType::UInt32 => {
            check_integer(value, 0.0, u32::MAX as f64, dt)?;
            (value as u32).to_le_bytes().to_vec()
        }
        BandDataType::Int32 => {
            check_integer(value, i32::MIN as f64, i32::MAX as f64, dt)?;
            (value as i32).to_le_bytes().to_vec()
        }
        BandDataType::UInt64 => {
            check_integer(value, 0.0, F64_INT_MAX, dt)?;
            (value as u64).to_le_bytes().to_vec()
        }
        BandDataType::Int64 => {
            check_integer(value, -F64_INT_MAX, F64_INT_MAX, dt)?;
            (value as i64).to_le_bytes().to_vec()
        }
        BandDataType::Float32 => (value as f32).to_le_bytes().to_vec(),
        BandDataType::Float64 => value.to_le_bytes().to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nodata_bytes_to_f64_uint8() {
        let val = nodata_bytes_to_f64(&[42], &BandDataType::UInt8).unwrap();
        assert_eq!(val, 42.0);
    }

    #[test]
    fn test_nodata_f64_to_bytes_round_trips() {
        for (dt, value) in [
            (BandDataType::UInt8, 127.0),
            (BandDataType::Int8, -2.0),
            (BandDataType::UInt16, 65535.0),
            (BandDataType::Int16, -1000.0),
            (BandDataType::UInt32, 4_000_000_000.0),
            (BandDataType::Int32, -2_000_000.0),
            (BandDataType::UInt64, 9_007_199_254_740_992.0), // 2^53
            (BandDataType::Int64, -9_007_199_254_740_992.0), // -2^53
            (BandDataType::Float32, -9999.5),
            (BandDataType::Float64, -9999.0),
        ] {
            let bytes = nodata_f64_to_bytes(value, &dt).unwrap();
            assert_eq!(bytes.len(), dt.byte_size(), "{dt:?}");
            assert_eq!(
                nodata_bytes_to_f64_lossless(&bytes, &dt).unwrap(),
                value,
                "{dt:?} round-trip"
            );
        }
    }

    #[test]
    fn test_nodata_f64_to_bytes_rejects_out_of_range() {
        // Non-integral for an integer type, and out of range.
        nodata_f64_to_bytes(1.5, &BandDataType::UInt8).unwrap_err();
        nodata_f64_to_bytes(300.0, &BandDataType::UInt8).unwrap_err();
        nodata_f64_to_bytes(-1.0, &BandDataType::UInt8).unwrap_err();
        // Beyond 2^53 can't have arrived losslessly through f64.
        nodata_f64_to_bytes(1e18, &BandDataType::Int64).unwrap_err();
    }

    #[test]
    fn test_nodata_bytes_to_f64_int8() {
        let val = nodata_bytes_to_f64(&[0xFE], &BandDataType::Int8).unwrap();
        assert_eq!(val, -2.0);
    }

    #[test]
    fn test_nodata_bytes_to_f64_float64() {
        let bytes = (-9999.0_f64).to_le_bytes();
        let val = nodata_bytes_to_f64(&bytes, &BandDataType::Float64).unwrap();
        assert_eq!(val, -9999.0);
    }

    #[test]
    fn test_nodata_bytes_to_f64_int32() {
        let bytes = (-1_i32).to_le_bytes();
        let val = nodata_bytes_to_f64(&bytes, &BandDataType::Int32).unwrap();
        assert_eq!(val, -1.0);
    }

    #[test]
    fn test_nodata_bytes_to_f64_wrong_length() {
        let result = nodata_bytes_to_f64(&[1, 2, 3], &BandDataType::Float64);
        assert!(result.is_err());
    }

    #[test]
    fn test_nodata_bytes_to_f64_lossless_int64_within_mantissa() {
        // Boundary: 2^53 is the largest magnitude that round-trips exactly.
        let safe = 1i64 << 53;
        let val = nodata_bytes_to_f64_lossless(&safe.to_le_bytes(), &BandDataType::Int64).unwrap();
        assert_eq!(val as i64, safe);

        let neg_safe = -(1i64 << 53);
        let val =
            nodata_bytes_to_f64_lossless(&neg_safe.to_le_bytes(), &BandDataType::Int64).unwrap();
        assert_eq!(val as i64, neg_safe);
    }

    #[test]
    fn test_nodata_bytes_to_f64_lossless_int64_errors_above_mantissa() {
        let big = (1i64 << 53) + 1;
        let err =
            nodata_bytes_to_f64_lossless(&big.to_le_bytes(), &BandDataType::Int64).unwrap_err();
        assert!(
            err.to_string().contains("Int64 nodata value"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_nodata_bytes_to_f64_lossless_uint64_sentinel_errors() {
        // The common sentinel 0xFFFF_FFFF_FFFF_FFFF is exactly the case the
        // review flagged: lossy variant silently rounds to a value that can
        // collide with a real pixel; lossless variant errors.
        let sentinel = u64::MAX;
        let err = nodata_bytes_to_f64_lossless(&sentinel.to_le_bytes(), &BandDataType::UInt64)
            .unwrap_err();
        assert!(
            err.to_string().contains("UInt64 nodata value"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_nodata_bytes_to_f64_lossless_delegates_for_smaller_types() {
        // Non-64-bit types pass through to nodata_bytes_to_f64 unchanged.
        let val = nodata_bytes_to_f64_lossless(&[42], &BandDataType::UInt8).unwrap();
        assert_eq!(val, 42.0);
        let val = nodata_bytes_to_f64_lossless(&[0xFE], &BandDataType::Int8).unwrap();
        assert_eq!(val, -2.0);
    }

    #[test]
    fn test_split_outdb_band_fragment_with_band() {
        let (base, n) = split_outdb_band_fragment("s3://bucket/file.tif#band=42");
        assert_eq!(base, "s3://bucket/file.tif");
        assert_eq!(n, 42);
    }

    #[test]
    fn test_split_outdb_band_fragment_without_band_defaults_to_1() {
        let (base, n) = split_outdb_band_fragment("s3://bucket/file.tif");
        assert_eq!(base, "s3://bucket/file.tif");
        assert_eq!(n, 1);
    }

    #[test]
    fn test_split_outdb_band_fragment_malformed_fragment_defaults_to_1() {
        // `#band=abc` is malformed; treat the whole string as the base URL.
        let (base, n) = split_outdb_band_fragment("s3://bucket/file.tif#band=abc");
        assert_eq!(base, "s3://bucket/file.tif#band=abc");
        assert_eq!(n, 1);
    }

    fn ve(source_axis: i64, start: i64, step: i64, steps: i64) -> ViewEntry {
        ViewEntry {
            source_axis,
            start,
            step,
            steps,
        }
    }

    /// Minimal `BandRef` stub: only the inputs `is_2d` actually inspects
    /// (`dim_names`, `view`, `raw_source_shape`) carry meaningful values;
    /// every other method returns a placeholder we never read.
    struct StubBand {
        dim_names: Vec<String>,
        source_shape: Vec<i64>,
        shape: Vec<i64>,
        view: Vec<ViewEntry>,
    }

    impl BandRef for StubBand {
        fn ndim(&self) -> usize {
            self.dim_names.len()
        }
        fn dim_names(&self) -> Vec<&str> {
            self.dim_names.iter().map(String::as_str).collect()
        }
        fn shape(&self) -> &[i64] {
            &self.shape
        }
        fn raw_source_shape(&self) -> &[i64] {
            &self.source_shape
        }
        fn view(&self) -> &[ViewEntry] {
            &self.view
        }
        fn data_type(&self) -> BandDataType {
            BandDataType::UInt8
        }
        fn nodata(&self) -> Option<&[u8]> {
            None
        }
        fn nd_buffer(&self) -> Result<NdBuffer<'_>, ArrowError> {
            unimplemented!("not used in is_spatial_2d tests")
        }
        fn is_indb(&self) -> bool {
            unimplemented!("not used in is_spatial_2d tests")
        }
    }

    fn band(dims: &[&str], source_shape: &[i64], view: &[ViewEntry]) -> StubBand {
        let shape = view.iter().map(|v| v.steps).collect();
        StubBand {
            dim_names: dims.iter().map(|s| (*s).to_string()).collect(),
            source_shape: source_shape.to_vec(),
            shape,
            view: view.to_vec(),
        }
    }

    #[test]
    fn is_spatial_2d_yx_is_true() {
        let b = band(&["y", "x"], &[4, 5], &[ve(0, 0, 1, 4), ve(1, 0, 1, 5)]);
        assert!(b.is_spatial_2d());
    }

    #[test]
    fn copy_into_rejects_non_identity_source_view() {
        // A sliced source view (step 2 on the outer axis) composes to a
        // non-identity effective view; copy_into can't persist it yet, so it
        // must error rather than copy mislocated bytes.
        let src = band(&["y", "x"], &[4, 5], &[ve(0, 0, 2, 2), ve(1, 0, 1, 5)]);
        let mut ob = RasterBuilder::new(1);
        let err = src
            .copy_into(&mut ob, BandOverrides::default())
            .unwrap_err()
            .to_string();
        assert!(err.contains("non-identity"), "unexpected error: {err}");
    }

    #[test]
    fn copy_into_rejects_non_identity_override_view() {
        // Identity source, but the caller's override slices it (step 2); the
        // composed effective view is non-identity and can't be persisted yet.
        let src = band(&["y", "x"], &[4, 5], &[ve(0, 0, 1, 4), ve(1, 0, 1, 5)]);
        let override_view = [ve(0, 0, 2, 2), ve(1, 0, 1, 5)];
        let mut ob = RasterBuilder::new(1);
        let err = src
            .copy_into(
                &mut ob,
                BandOverrides {
                    view: Some(&override_view),
                    ..Default::default()
                },
            )
            .unwrap_err()
            .to_string();
        assert!(err.contains("non-identity"), "unexpected error: {err}");
    }

    #[test]
    fn is_spatial_2d_latlon_is_true() {
        let b = band(&["lat", "lon"], &[4, 5], &[ve(0, 0, 1, 4), ve(1, 0, 1, 5)]);
        assert!(b.is_spatial_2d());
    }

    #[test]
    fn is_spatial_2d_latitude_longitude_is_true() {
        let b = band(
            &["latitude", "longitude"],
            &[4, 5],
            &[ve(0, 0, 1, 4), ve(1, 0, 1, 5)],
        );
        assert!(b.is_spatial_2d());
    }

    #[test]
    fn is_spatial_2d_3d_is_false() {
        let b = band(
            &["time", "y", "x"],
            &[3, 4, 5],
            &[ve(0, 0, 1, 3), ve(1, 0, 1, 4), ve(2, 0, 1, 5)],
        );
        assert!(!b.is_spatial_2d());
    }

    #[test]
    fn is_spatial_2d_1d_is_false() {
        let b = band(&["x"], &[5], &[ve(0, 0, 1, 5)]);
        assert!(!b.is_spatial_2d());
    }

    #[test]
    fn is_spatial_2d_permuted_xy_is_false() {
        // ["x","y"] is not a recognized (y-like, x-like) pair.
        let b = band(&["x", "y"], &[5, 4], &[ve(0, 0, 1, 5), ve(1, 0, 1, 4)]);
        assert!(!b.is_spatial_2d());
    }

    #[test]
    fn is_spatial_2d_unrecognized_dim_names_is_false() {
        let b = band(
            &["northing", "easting"],
            &[4, 5],
            &[ve(0, 0, 1, 4), ve(1, 0, 1, 5)],
        );
        assert!(!b.is_spatial_2d());
    }

    /// Build a bufferless `NdBuffer` for contiguity checks — `is_contiguous`
    /// inspects only shape/strides/data_type, never the bytes.
    fn ndbuf(shape: &[i64], strides: &[i64], offset: u64) -> NdBuffer<'static> {
        NdBuffer {
            buffer: &[],
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            offset,
            data_type: BandDataType::UInt8,
        }
    }

    #[test]
    fn is_contiguous_packed_identity() {
        // C-order packed strides for shape [2, 3], byte_size 1.
        assert!(ndbuf(&[2, 3], &[3, 1], 0).is_contiguous());
    }

    #[test]
    fn is_contiguous_packed_with_offset() {
        // Offset is irrelevant to contiguity — a packed sub-window still
        // counts (this is the relaxation the GDAL gate relies on).
        assert!(ndbuf(&[2, 3], &[3, 1], 12).is_contiguous());
    }

    #[test]
    fn is_contiguous_strided_is_false() {
        // Inner stride 2 != byte_size 1 — gaps between elements.
        assert!(!ndbuf(&[2, 3], &[6, 2], 0).is_contiguous());
    }

    #[test]
    fn is_contiguous_broadcast_is_false() {
        // Zero stride (broadcast) is not packed.
        assert!(!ndbuf(&[2, 3], &[0, 1], 0).is_contiguous());
    }
}
