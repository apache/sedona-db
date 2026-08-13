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

use crate::builder::{RasterBuilder, StartBandArgs};
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

    /// Materialize the visible region into `out` as a packed row-major
    /// (C-order) byte buffer — the strided-to-contiguous repack behind
    /// `RS_EnsureContiguous`.
    ///
    /// `out` is cleared and refilled with exactly `Π shape × byte_size` bytes:
    /// the visible elements walked in C-order (last axis fastest), each read
    /// from `buffer[pos .. pos + byte_size]` where
    /// `pos = offset + Σ idx[i] × strides[i]`. `strides` and `offset` are byte
    /// units, so this handles zero strides (broadcast — the same source element
    /// is copied repeatedly) and negative strides (reverse iteration) uniformly.
    /// For an already-contiguous view the output equals [`as_contiguous`](Self::as_contiguous).
    ///
    /// Passing the same `out` across calls reuses its capacity — the walk itself
    /// allocates nothing after the first call. Every visited byte position is
    /// in-bounds for any `NdBuffer` a reader composed from a validated view.
    pub fn materialize_into(&self, out: &mut Vec<u8>) {
        out.clear();
        let byte_size = self.data_type.byte_size();
        let n: i64 = self.shape.iter().product();
        // A zero-extent axis (or empty shape product) leaves `out` empty.
        if n <= 0 {
            return;
        }
        out.reserve(n as usize * byte_size);
        let ndim = self.shape.len();
        let mut idx = vec![0i64; ndim];
        for _ in 0..n {
            let mut pos = self.offset as i64;
            for (k, &i) in idx.iter().enumerate() {
                pos += i * self.strides[k];
            }
            let start = pos as usize;
            out.extend_from_slice(&self.buffer[start..start + byte_size]);
            // Increment the multi-index in C-order (last axis fastest).
            for k in (0..ndim).rev() {
                idx[k] += 1;
                if idx[k] < self.shape[k] {
                    break;
                }
                idx[k] = 0;
            }
        }
    }

    /// Materialize the visible region into a fresh packed row-major `Vec<u8>`.
    /// Convenience wrapper over [`materialize_into`](Self::materialize_into) for
    /// callers that want an owned buffer to move onward (e.g. into an Arrow
    /// [`arrow_buffer::Buffer`]) rather than reuse scratch.
    pub fn materialize_contiguous(&self) -> Vec<u8> {
        let mut out = Vec::new();
        self.materialize_into(&mut out);
        out
    }
}

/// Parse the SedonaDB `#band=N` fragment out of an out-DB URI into the base
/// URL and 1-based source band index.
///
/// Behaviour:
///
/// - `<uri>#band=N` where `N` parses as a `u32` in `1..=u32::MAX`: strips the
///   fragment and returns `(<uri>, N)`. `rsplit_once` lets a trailing
///   `#band=N` win over any earlier `#anchor` already in the URI.
/// - `<uri>#band=...` with a value that is not a positive `u32` (zero,
///   negative, non-numeric, empty, or overflowing `u32`): returns an error.
///   The user explicitly asked for a band; we refuse to silently substitute a
///   default.
/// - Any URI whose fragment is not `band=...` (including GDAL-native
///   subdataset URIs like `HDF5:"x.h5":/var`) and plain URIs without a
///   fragment: pass through verbatim with default band index 1.
///
/// This is the shared convention consumers use to recover the source path and
/// band from a band's `outdb_uri()`.
pub fn split_outdb_band_fragment(uri: &str) -> Result<(String, u32), ArrowError> {
    if let Some((prefix, fragment)) = uri.rsplit_once('#') {
        if let Some(band_str) = fragment.strip_prefix("band=") {
            return match band_str.parse::<u32>() {
                Ok(band) if band >= 1 => Ok((prefix.to_string(), band)),
                _ => Err(ArrowError::InvalidArgumentError(format!(
                    "Invalid band index in outdb URI fragment '#band={band_str}': expected a positive integer in 1..=u32::MAX"
                ))),
            };
        }
    }
    Ok((uri.to_string(), 1))
}

/// Trait for accessing an N-dimensional raster (top level).
///
/// A single flat interface: raster-level geometry (`width`/`height`/
/// `transform`) and band access live directly on this trait. Bands are
/// 0-indexed.
pub trait RasterRef {
    /// Number of bands/variables
    fn num_bands(&self) -> usize;

    /// Access a band by 0-based index. Returns an `ArrowError` when the
    /// index is out of range or when the underlying schema is malformed
    /// (unknown data-type discriminant, corrupt view, etc.). The latter
    /// cases route through `sedona_common::sedona_internal_datafusion_err!`
    /// so they carry the standardised "SedonaDB internal error" framing.
    fn band(&self, index: usize) -> Result<Box<dyn BandRef + '_>, ArrowError>;

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
    /// view unchanged. A non-identity result is persisted on the derived band
    /// (as a slice, broadcast, permutation, or reverse) and decoded back by the
    /// reader; the underlying bytes are carried over unchanged.
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
    /// replaces hand-rebuilding via `start_band` + a manual data append. The
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
    /// [`RasterBuilder::start_band`]: an identity effective view is
    /// stored as the canonical null sentinel, and a non-identity one (a slice,
    /// broadcast, permutation, or reverse) is persisted explicitly and decoded
    /// back by the reader. The source bytes are carried over unchanged — the
    /// view relocates the visible window over them, it does not repack them.
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
        builder.start_band(StartBandArgs {
            name: overrides.name,
            view: Some(effective_view.as_slice()),
            nodata: overrides.nodata.or_else(|| self.nodata()),
            outdb_uri: overrides.outdb_uri.or_else(|| self.outdb_uri()),
            outdb_format: overrides.outdb_format.or_else(|| self.outdb_format()),
            ..StartBandArgs::new(&dim_names, &source_shape, self.data_type())
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
        let (base, n) = split_outdb_band_fragment("s3://bucket/file.tif#band=42").unwrap();
        assert_eq!(base, "s3://bucket/file.tif");
        assert_eq!(n, 42);
    }

    #[test]
    fn test_split_outdb_band_fragment_without_band_defaults_to_1() {
        let (base, n) = split_outdb_band_fragment("s3://bucket/file.tif").unwrap();
        assert_eq!(base, "s3://bucket/file.tif");
        assert_eq!(n, 1);
    }

    #[test]
    fn test_split_outdb_band_fragment_malformed_fragment_errors() {
        // `#band=abc` explicitly requests a band but is not a positive integer;
        // reject it rather than silently substituting the default band 1.
        let msg = split_outdb_band_fragment("s3://bucket/file.tif#band=abc")
            .unwrap_err()
            .to_string();
        assert!(msg.contains("band=abc"), "msg was: {msg}");
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
    fn copy_into_persists_non_identity_source_view() {
        // A source band that already carries a non-identity view (an every-
        // other slice over an 8-element axis) is copied with default overrides:
        // the effective view is the source's own slice, and copy_into persists
        // it and carries the source bytes over. Read back, the derived band
        // exposes the same slice — visible values [1, 3, 5].
        use crate::array::RasterStructArray;
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];

        // Source: source_shape [8], data [0..8], view (start=1, step=2, steps=3).
        let mut ib = RasterBuilder::new(1);
        ib.start_raster_nd(&transform, &["x"], &[3], None).unwrap();
        ib.start_band(StartBandArgs {
            view: Some(&[ve(0, 1, 2, 3)]),
            ..StartBandArgs::new(&["x"], &[8], BandDataType::UInt8)
        })
        .unwrap();
        ib.band_data_writer()
            .append_value((0u8..8).collect::<Vec<u8>>());
        ib.finish_band().unwrap();
        ib.finish_raster().unwrap();
        let in_array = ib.finish().unwrap();
        let in_rasters = RasterStructArray::try_new(&in_array).unwrap();
        let in_raster = in_rasters.get(0).unwrap();
        let in_band = in_raster.band(0).unwrap();

        let mut ob = RasterBuilder::new(1);
        ob.start_raster_nd(&transform, &["x"], &[3], None).unwrap();
        in_band
            .copy_into(&mut ob, BandOverrides::default())
            .unwrap();
        ob.finish_band().unwrap();
        ob.finish_raster().unwrap();
        let out_array = ob.finish().unwrap();
        let out_rasters = RasterStructArray::try_new(&out_array).unwrap();
        let out_raster = out_rasters.get(0).unwrap();
        let out_band = out_raster.band(0).unwrap();

        assert_eq!(out_band.view(), &[ve(0, 1, 2, 3)]);
        assert_eq!(out_band.shape(), &[3]);
        assert_eq!(out_band.raw_source_shape(), &[8]);
        let buf = out_band.nd_buffer().unwrap();
        assert_eq!(buf.strides, &[2]);
        assert_eq!(buf.offset, 1);
        // Strided → not contiguous; the visible bytes must be [1, 3, 5].
        assert!(!buf.is_contiguous());
        let visited: Vec<u8> = (0..buf.shape[0])
            .map(|i| buf.buffer[(buf.offset as i64 + i * buf.strides[0]) as usize])
            .collect();
        assert_eq!(visited, vec![1, 3, 5]);
    }

    #[test]
    fn copy_into_persists_non_identity_override_view() {
        // Identity source; the caller's override slices every other element
        // (step 2). copy_into composes the override onto the identity source
        // view and persists the resulting slice — visible values [1, 3].
        use crate::array::RasterStructArray;
        let transform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];

        let mut ib = RasterBuilder::new(1);
        ib.start_raster_nd(&transform, &["x"], &[4], None).unwrap();
        ib.start_band(StartBandArgs::new(&["x"], &[4], BandDataType::UInt8))
            .unwrap();
        ib.band_data_writer().append_value(vec![1u8, 2, 3, 4]);
        ib.finish_band().unwrap();
        ib.finish_raster().unwrap();
        let in_array = ib.finish().unwrap();
        let in_rasters = RasterStructArray::try_new(&in_array).unwrap();
        let in_raster = in_rasters.get(0).unwrap();
        let in_band = in_raster.band(0).unwrap();

        let override_view = [ve(0, 0, 2, 2)];
        let mut ob = RasterBuilder::new(1);
        ob.start_raster_nd(&transform, &["x"], &[2], None).unwrap();
        in_band
            .copy_into(
                &mut ob,
                BandOverrides {
                    view: Some(&override_view),
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

        assert_eq!(out_band.view(), &[ve(0, 0, 2, 2)]);
        assert_eq!(out_band.shape(), &[2]);
        assert_eq!(out_band.raw_source_shape(), &[4]);
        let buf = out_band.nd_buffer().unwrap();
        assert_eq!(buf.strides, &[2]);
        assert_eq!(buf.offset, 0);
        assert!(!buf.is_contiguous());
        let visited: Vec<u8> = (0..buf.shape[0])
            .map(|i| buf.buffer[(buf.offset as i64 + i * buf.strides[0]) as usize])
            .collect();
        assert_eq!(visited, vec![1, 3]);
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

    /// Build an `NdBuffer` over `bytes` with explicit layout for the
    /// materialize tests. Unlike [`ndbuf`], the bytes are meaningful — the
    /// walk reads them.
    fn ndbuf_over<'a>(
        bytes: &'a [u8],
        shape: &[i64],
        strides: &[i64],
        offset: u64,
        data_type: BandDataType,
    ) -> NdBuffer<'a> {
        NdBuffer {
            buffer: bytes,
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            offset,
            data_type,
        }
    }

    #[test]
    fn materialize_into_contiguous_equals_as_contiguous() {
        // Packed [2, 3] view over [0..6]: the walk reproduces the source and
        // matches the zero-copy contiguous borrow byte-for-byte.
        let data: Vec<u8> = (0..6).collect();
        let buf = ndbuf_over(&data, &[2, 3], &[3, 1], 0, BandDataType::UInt8);
        let mut out = Vec::new();
        buf.materialize_into(&mut out);
        assert_eq!(out, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(out.as_slice(), buf.as_contiguous().unwrap());
    }

    #[test]
    fn materialize_into_strided_1d() {
        // Every-other slice of [0..8]: offset 1, stride 2, 3 steps → 1, 3, 5.
        let data: Vec<u8> = (0..8).collect();
        let buf = ndbuf_over(&data, &[3], &[2], 1, BandDataType::UInt8);
        assert!(!buf.is_contiguous());
        assert_eq!(buf.materialize_contiguous(), vec![1, 3, 5]);
    }

    #[test]
    fn materialize_into_broadcast_zero_stride() {
        // 2-D broadcast: [4, 3] over a single source row via a zero outer
        // stride — every visible row repeats [10, 20, 30].
        let data = vec![10u8, 20, 30];
        let buf = ndbuf_over(&data, &[4, 3], &[0, 1], 0, BandDataType::UInt8);
        assert!(!buf.is_contiguous());
        assert_eq!(
            buf.materialize_contiguous(),
            vec![10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 30]
        );
    }

    #[test]
    fn materialize_into_permutation_and_slice() {
        // 2-D source [Y=4, X=3], data 0..12 row-major, viewed as [X, Y] with Y
        // sliced start=1 step=2 steps=2: byte strides [1, 6], offset 3 → the
        // C-order gather is [3, 9, 4, 10, 5, 11] (mirrors the builder's
        // view_axis_permutation_and_slice test).
        let data: Vec<u8> = (0..12).collect();
        let buf = ndbuf_over(&data, &[3, 2], &[1, 6], 3, BandDataType::UInt8);
        assert!(!buf.is_contiguous());
        assert_eq!(buf.materialize_contiguous(), vec![3, 9, 4, 10, 5, 11]);
    }

    #[test]
    fn materialize_into_reverse_negative_stride() {
        // Negative stride walks backwards: offset 6, stride -2, 3 steps → 6, 4, 2.
        let data: Vec<u8> = (0..8).collect();
        let buf = ndbuf_over(&data, &[3], &[-2], 6, BandDataType::UInt8);
        assert!(!buf.is_contiguous());
        assert_eq!(buf.materialize_contiguous(), vec![6, 4, 2]);
    }

    #[test]
    fn materialize_into_multibyte_dtype_respects_element_size() {
        // UInt16 (2 bytes/elem): LE bytes for [1, 2, 3, 4]; a stride of 4 bytes
        // picks every other element (indices 0, 2 → values 1, 3), copying whole
        // 2-byte elements, not single bytes.
        let data: Vec<u8> = vec![1, 0, 2, 0, 3, 0, 4, 0];
        let buf = ndbuf_over(&data, &[2], &[4], 0, BandDataType::UInt16);
        assert!(!buf.is_contiguous());
        assert_eq!(buf.materialize_contiguous(), vec![1, 0, 3, 0]);
    }

    #[test]
    fn materialize_into_zero_extent_axis_is_empty() {
        // A zero-extent axis addresses no elements → empty output.
        let data = vec![0u8; 60];
        let buf = ndbuf_over(&data, &[3, 0, 5], &[20, 5, 1], 0, BandDataType::UInt8);
        buf.materialize_into(&mut Vec::new());
        assert!(buf.materialize_contiguous().is_empty());
    }

    #[test]
    fn materialize_into_reuses_scratch_across_calls() {
        // Reusing the same `out` clears the previous contents each call, so a
        // longer band followed by a shorter one yields exactly the shorter
        // band's bytes — no stale tail.
        let long: Vec<u8> = (0..8).collect();
        let short: Vec<u8> = (100..104).collect();
        let mut scratch = Vec::new();

        ndbuf_over(&long, &[8], &[1], 0, BandDataType::UInt8).materialize_into(&mut scratch);
        assert_eq!(scratch, vec![0, 1, 2, 3, 4, 5, 6, 7]);

        ndbuf_over(&short, &[4], &[1], 0, BandDataType::UInt8).materialize_into(&mut scratch);
        assert_eq!(scratch, vec![100, 101, 102, 103]);
    }

    #[test]
    fn split_outdb_band_one_fragment_round_trips() {
        assert_eq!(
            split_outdb_band_fragment("s3://bucket/file.tif#band=1").unwrap(),
            ("s3://bucket/file.tif".to_string(), 1),
        );
    }

    #[test]
    fn split_outdb_band_max_u32_accepted() {
        let max = u32::MAX;
        let uri = format!("s3://bucket/file.tif#band={max}");
        assert_eq!(
            split_outdb_band_fragment(&uri).unwrap(),
            ("s3://bucket/file.tif".to_string(), max),
        );
    }

    #[test]
    fn split_outdb_band_zero_errors() {
        let msg = split_outdb_band_fragment("s3://bucket/file.tif#band=0")
            .unwrap_err()
            .to_string();
        assert!(msg.contains("band=0"), "msg was: {msg}");
        assert!(msg.contains("positive integer"), "msg was: {msg}");
    }

    #[test]
    fn split_outdb_negative_band_errors() {
        let msg = split_outdb_band_fragment("s3://bucket/file.tif#band=-2")
            .unwrap_err()
            .to_string();
        assert!(msg.contains("band=-2"), "msg was: {msg}");
    }

    #[test]
    fn split_outdb_band_overflow_errors() {
        // 4294967296 = u32::MAX + 1
        let msg = split_outdb_band_fragment("s3://bucket/file.tif#band=4294967296")
            .unwrap_err()
            .to_string();
        assert!(msg.contains("band=4294967296"), "msg was: {msg}");
    }

    #[test]
    fn split_outdb_empty_band_value_errors() {
        let msg = split_outdb_band_fragment("s3://bucket/file.tif#band=")
            .unwrap_err()
            .to_string();
        assert!(msg.contains("band="), "msg was: {msg}");
    }

    #[test]
    fn split_outdb_non_band_fragment_passes_through() {
        let uri = "s3://bucket/file.tif#section";
        assert_eq!(
            split_outdb_band_fragment(uri).unwrap(),
            (uri.to_string(), 1)
        );
    }

    #[test]
    fn split_outdb_empty_fragment_passes_through() {
        let uri = "s3://bucket/file.tif#";
        assert_eq!(
            split_outdb_band_fragment(uri).unwrap(),
            (uri.to_string(), 1)
        );
    }

    #[test]
    fn split_outdb_url_query_string_preserved_with_band_fragment() {
        assert_eq!(
            split_outdb_band_fragment("https://example.com/r.tif?token=abc#band=3").unwrap(),
            ("https://example.com/r.tif?token=abc".to_string(), 3),
        );
    }

    #[test]
    fn split_outdb_url_query_string_with_non_band_fragment_passes_through() {
        let uri = "https://example.com/r.tif?token=abc#anchor";
        assert_eq!(
            split_outdb_band_fragment(uri).unwrap(),
            (uri.to_string(), 1)
        );
    }

    #[test]
    fn split_outdb_local_path_with_band_fragment() {
        assert_eq!(
            split_outdb_band_fragment("/tmp/file.tif#band=5").unwrap(),
            ("/tmp/file.tif".to_string(), 5),
        );
    }

    #[test]
    fn split_outdb_local_path_without_fragment() {
        assert_eq!(
            split_outdb_band_fragment("/tmp/file.tif").unwrap(),
            ("/tmp/file.tif".to_string(), 1),
        );
    }

    #[test]
    fn split_outdb_gdal_subdataset_hdf5_passthrough() {
        let uri = r#"HDF5:"/path/x.h5"://temperature"#;
        assert_eq!(
            split_outdb_band_fragment(uri).unwrap(),
            (uri.to_string(), 1)
        );
    }

    #[test]
    fn split_outdb_gdal_subdataset_netcdf_passthrough() {
        let uri = r#"NETCDF:"/path/file.nc":variable"#;
        assert_eq!(
            split_outdb_band_fragment(uri).unwrap(),
            (uri.to_string(), 1)
        );
    }

    #[test]
    fn split_outdb_gdal_subdataset_gtiff_dir_passthrough() {
        let uri = "GTIFF_DIR:1:/path/multi.tif";
        assert_eq!(
            split_outdb_band_fragment(uri).unwrap(),
            (uri.to_string(), 1)
        );
    }

    #[test]
    fn split_outdb_gdal_subdataset_with_band_fragment_extracts_band() {
        let uri = r#"HDF5:"/path/x.h5":/var#band=3"#;
        let (gdal_uri, band) = split_outdb_band_fragment(uri).unwrap();
        assert_eq!(gdal_uri, r#"HDF5:"/path/x.h5":/var"#);
        assert_eq!(band, 3);
    }

    #[test]
    fn split_outdb_trailing_band_wins_over_earlier_anchor() {
        assert_eq!(
            split_outdb_band_fragment("https://example.com/r.tif#anchor#band=7").unwrap(),
            ("https://example.com/r.tif#anchor".to_string(), 7),
        );
    }

    #[test]
    fn split_outdb_empty_uri() {
        assert_eq!(split_outdb_band_fragment("").unwrap(), (String::new(), 1));
    }
}
