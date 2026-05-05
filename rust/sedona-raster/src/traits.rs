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

use arrow_schema::ArrowError;
use sedona_schema::raster::BandDataType;

/// Zero-copy view into a band's N-D data buffer with layout metadata.
///
/// `shape`, `strides`, and `offset` describe the *visible* region in
/// byte-stride terms — they are computed by composing the band's
/// `source_shape` (the natural extent of `buffer`) with its `view`
/// (the per-axis `(source_axis, start, step, steps)` slice spec). Stride
/// can be zero (broadcast) or negative (reverse iteration), and may not be
/// C-order. Consumers that need a flat row-major buffer should use
/// `BandRef::contiguous_data()` instead.
#[derive(Debug)]
pub struct NdBuffer<'a> {
    pub buffer: &'a [u8],
    pub shape: &'a [u64],
    pub strides: &'a [i64],
    pub offset: u64,
    pub data_type: BandDataType,
}

/// One per-dimension entry of a band's logical view. Describes how a
/// visible axis maps onto an axis of the underlying source buffer.
///
/// - `source_axis`: index into the band's `source_shape` that this visible
///   axis reads from. Across a band's full view, `source_axis` values must
///   form a permutation of `0..ndim` — Phase 1 does not support
///   axis-dropping or axis-introducing views.
/// - `start`: starting index along the source axis (in elements, not bytes).
/// - `step`: stride between consecutive visible elements along the source
///   axis. `step == 0` means broadcast (the same source element is
///   exposed `steps` times); negative `step` means reverse iteration.
/// - `steps`: number of visible elements along this axis. `steps == 0` is
///   allowed (empty axis).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ViewEntry {
    pub source_axis: i64,
    pub start: i64,
    pub step: i64,
    pub steps: i64,
}

/// Trait for accessing an N-dimensional raster (top level).
///
/// Replaces the legacy `RasterRef` + `MetadataRef` + `BandsRef` hierarchy with
/// a single flat interface. Bands are 0-indexed.
pub trait RasterRef {
    /// Number of bands/variables
    fn num_bands(&self) -> usize;

    /// Access a band by 0-based index
    fn band(&self, index: usize) -> Option<Box<dyn BandRef + '_>>;

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
        self.band(index).map(|b| b.data_type())
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
    /// `spatial_shape`.
    fn width(&self) -> Option<u64> {
        self.spatial_shape().first().map(|&v| v as u64)
    }

    /// Height in pixels — size of the Y spatial dimension from the top-level
    /// `spatial_shape`.
    fn height(&self) -> Option<u64> {
        self.spatial_shape().get(1).map(|&v| v as u64)
    }

    /// Look up a band by name. Returns None if no band has that name.
    fn band_by_name(&self, name: &str) -> Option<Box<dyn BandRef + '_>> {
        (0..self.num_bands())
            .find(|&i| self.band_name(i) == Some(name))
            .and_then(|i| self.band(i))
    }
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
    fn shape(&self) -> &[u64];

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
    fn raw_source_shape(&self) -> &[u64];

    /// Per-visible-dimension view entries describing how the band's
    /// visible axes map onto its `source_shape`. `view().len() == ndim()`.
    /// See `ViewEntry` for per-entry semantics.
    fn view(&self) -> &[ViewEntry];

    /// Size of a named dimension (None if doesn't exist)
    fn dim_size(&self, name: &str) -> Option<u64> {
        let idx = self.dim_index(name)?;
        Some(self.shape()[idx])
    }

    /// Index of a named dimension (None if doesn't exist)
    fn dim_index(&self, name: &str) -> Option<usize> {
        self.dim_names().iter().position(|n| *n == name)
    }

    /// True iff this band is shaped exactly like a legacy 2-D raster band:
    /// `dim_names == ["y", "x"]` and the view is the identity over the
    /// band's `raw_source_shape` (no slice, no broadcast, no permutation).
    ///
    /// GDAL-backed SQL functions use this to refuse N-D bands cleanly while
    /// they wait for an MDArray-aware port.
    fn is_2d(&self) -> bool {
        let dims = self.dim_names();
        if dims.len() != 2 || dims[0] != "y" || dims[1] != "x" {
            return false;
        }
        let view = self.view();
        let source_shape = self.raw_source_shape();
        if view.len() != 2 || source_shape.len() != 2 {
            return false;
        }
        view.iter().enumerate().all(|(i, v)| {
            v.source_axis as usize == i
                && v.start == 0
                && v.step == 1
                && v.steps >= 0
                && v.steps as u64 == source_shape[i]
        })
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

    // -- Data access --

    /// Raw backing buffer + visible-region layout. Triggers load for lazy
    /// impls. The returned `NdBuffer` describes the band's view in
    /// byte-stride terms — `shape` is the visible shape, `strides` and
    /// `offset` are computed by composing the view with the source's
    /// natural C-order byte strides. Strides may be zero (broadcast) or
    /// negative (reverse iteration).
    fn nd_buffer(&self) -> Result<NdBuffer<'_>, ArrowError>;

    /// Contiguous row-major bytes covering the *visible* region. Zero-copy
    /// (`Cow::Borrowed`) when the view is full identity over a C-order
    /// source buffer; copies into a new buffer when the view slices,
    /// broadcasts, or permutes. Most RS_* functions use this.
    fn contiguous_data(&self) -> Result<Cow<'_, [u8]>, ArrowError>;

    /// Nodata value interpreted as f64.
    ///
    /// Returns `Ok(None)` when no nodata value is defined, `Ok(Some(f64))` on
    /// success, or an error when the raw bytes have an unexpected length.
    ///
    /// # Warning
    ///
    /// For 64-bit integer bands (`Int64`, `UInt64`), the conversion to `f64`
    /// is lossy when the magnitude exceeds 2^53 — values outside
    /// `[-9_007_199_254_740_992, 9_007_199_254_740_992]` will be rounded to
    /// the nearest representable double. Use `nodata()` directly to recover
    /// the exact bytes if you need full integer precision.
    fn nodata_as_f64(&self) -> Result<Option<f64>, ArrowError> {
        let bytes = match self.nodata() {
            Some(b) => b,
            None => return Ok(None),
        };
        nodata_bytes_to_f64(bytes, &self.data_type()).map(Some)
    }
}

/// Validate a `[ViewEntry]` against a band's `source_shape`.
///
/// Returns `Ok(())` if the view is well-formed under the rules:
/// - `view.len() == source_shape.len()`.
/// - `source_axis` values across `view` form a permutation of
///   `0..source_shape.len()` (no axis duplicated, none missing).
/// - `steps >= 0`.
/// - When `steps > 0`: `start ∈ [0, source_shape[source_axis])`, and when
///   `step != 0` the last addressed element
///   `start + (steps - 1) * step` is also in that range.
///
/// This is the same check the builder runs in `start_band_with_view` and
/// the reader runs when materialising a band — exposed publicly so future
/// view-producing functions (slice composition, transpose, etc.) can
/// validate before they touch a builder.
pub fn validate_view(view: &[ViewEntry], source_shape: &[u64]) -> Result<(), ArrowError> {
    let ndim = source_shape.len();
    if view.len() != ndim {
        return Err(ArrowError::InvalidArgumentError(format!(
            "view length ({}) must equal source_shape length ({ndim})",
            view.len()
        )));
    }
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
                // Use checked arithmetic so a malicious or corrupted view
                // can't silently wrap (steps-1)*step or start+… into an
                // in-range value and bypass the bound check. Any overflow
                // is reported as a normal validation error.
                let last = (v.steps - 1)
                    .checked_mul(v.step)
                    .and_then(|d| v.start.checked_add(d))
                    .ok_or_else(|| {
                        ArrowError::InvalidArgumentError(format!(
                            "view[{k}] last-element index overflows i64 for \
                             start={}, step={}, steps={} on source axis {sa}",
                            v.start, v.step, v.steps
                        ))
                    })?;
                if last < 0 || last >= s {
                    return Err(ArrowError::InvalidArgumentError(format!(
                        "view[{k}] addresses element {last} which is out of range \
                         [0, {s}) for source axis {sa}"
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Convert raw nodata bytes to f64 given a [`BandDataType`].
///
/// The bytes are expected to be in little-endian order and exactly match the
/// byte size of the data type.
pub fn nodata_bytes_to_f64(bytes: &[u8], dt: &BandDataType) -> Result<f64, ArrowError> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nodata_bytes_to_f64_uint8() {
        let val = nodata_bytes_to_f64(&[42], &BandDataType::UInt8).unwrap();
        assert_eq!(val, 42.0);
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
    fn test_nodata_as_f64_int64_loses_precision_above_2_pow_53() {
        // Locks in the documented warning: nodata bytes for Int64 values
        // beyond f64's 53-bit mantissa silently round on conversion.
        // The expected f64 is hard-coded — deriving it via `as f64` would
        // mean the test invokes the same primitive cast it claims to test.
        let big = (1i64 << 53) + 1; // 2^53 + 1; not representable in f64
        let bytes = big.to_le_bytes();
        let val = nodata_bytes_to_f64(&bytes, &BandDataType::Int64).unwrap();
        assert_eq!(val, 9007199254740992.0_f64);
        assert_ne!(val as i64, big);
    }

    fn ve(source_axis: i64, start: i64, step: i64, steps: i64) -> ViewEntry {
        ViewEntry {
            source_axis,
            start,
            step,
            steps,
        }
    }

    #[test]
    fn validate_view_accepts_identity() {
        let v = [ve(0, 0, 1, 4), ve(1, 0, 1, 5)];
        validate_view(&v, &[4, 5]).unwrap();
    }

    #[test]
    fn validate_view_rejects_length_mismatch() {
        let v = [ve(0, 0, 1, 4)];
        let err = validate_view(&v, &[4, 5]).unwrap_err();
        assert!(err.to_string().contains("must equal"), "got {err}");
    }

    #[test]
    fn validate_view_rejects_negative_source_axis() {
        let v = [ve(-1, 0, 1, 4)];
        let err = validate_view(&v, &[4]).unwrap_err();
        assert!(err.to_string().contains("source_axis"), "got {err}");
    }

    #[test]
    fn validate_view_rejects_oob_source_axis() {
        let v = [ve(2, 0, 1, 4)];
        let err = validate_view(&v, &[4]).unwrap_err();
        assert!(err.to_string().contains("source_axis"), "got {err}");
    }

    #[test]
    fn validate_view_rejects_duplicate_source_axis() {
        let v = [ve(0, 0, 1, 2), ve(0, 0, 1, 2)];
        let err = validate_view(&v, &[2, 3]).unwrap_err();
        assert!(err.to_string().contains("permutation"), "got {err}");
    }

    #[test]
    fn validate_view_rejects_negative_steps() {
        let v = [ve(0, 0, 1, -1)];
        let err = validate_view(&v, &[4]).unwrap_err();
        assert!(err.to_string().contains("steps"), "got {err}");
    }

    #[test]
    fn validate_view_rejects_negative_start() {
        let v = [ve(0, -1, 1, 1)];
        let err = validate_view(&v, &[4]).unwrap_err();
        assert!(err.to_string().contains("start"), "got {err}");
    }

    #[test]
    fn validate_view_rejects_start_at_source_size() {
        // start == S is one past the end. Forbidden even with steps=1.
        let v = [ve(0, 4, 1, 1)];
        let err = validate_view(&v, &[4]).unwrap_err();
        assert!(err.to_string().contains("start"), "got {err}");
    }

    #[test]
    fn validate_view_rejects_negative_step_underrun() {
        // start=0, step=-1, steps=2 addresses element 0 then -1 → underrun.
        // The most likely real bug in step != 0 arithmetic.
        let v = [ve(0, 0, -1, 2)];
        let err = validate_view(&v, &[4]).unwrap_err();
        assert!(err.to_string().contains("out of range"), "got {err}");
    }

    #[test]
    fn validate_view_accepts_negative_step_full_reverse() {
        // start=3, step=-1, steps=4 addresses 3,2,1,0 — all in range.
        let v = [ve(0, 3, -1, 4)];
        validate_view(&v, &[4]).unwrap();
    }

    #[test]
    fn validate_view_accepts_steps_zero_with_unconstrained_start() {
        // Empty axis short-circuits the bound check on start.
        let v = [ve(0, 999, 1, 0)];
        validate_view(&v, &[4]).unwrap();
    }

    #[test]
    fn validate_view_steps_one_only_checks_start() {
        // steps=1, step=999 — only `start` matters; the would-be next index
        // (start + 1*999) is never addressed and must not be checked.
        let v = [ve(0, 3, 999, 1)];
        validate_view(&v, &[4]).unwrap();
    }

    #[test]
    fn validate_view_step_zero_broadcast_within_bounds() {
        // step=0 broadcasts. start ∈ [0, S) is the only check.
        let v_ok = [ve(0, 3, 0, 100)];
        validate_view(&v_ok, &[4]).unwrap();
        let v_bad = [ve(0, 4, 0, 1)];
        let err = validate_view(&v_bad, &[4]).unwrap_err();
        assert!(err.to_string().contains("start"), "got {err}");
    }

    #[test]
    fn validate_view_permutation_with_slice_ok() {
        // Mix permutation and slicing — both legal as long as source_axis
        // values are a permutation and bounds hold per axis.
        let v = [ve(1, 0, 1, 3), ve(0, 1, 1, 1)];
        validate_view(&v, &[2, 3]).unwrap();
    }

    #[test]
    fn validate_view_rejects_i64_overflow_in_last_element() {
        // start=10, step=i64::MAX, steps=3 wraps `(steps-1)*step` to a
        // small negative i64; without checked arithmetic the naive sum
        // becomes 8 — falsely "in range" for a source of size 100. With
        // checked arithmetic, validate_view must reject it as overflow.
        // This was a real bug: in release the wrapped value passed all
        // bounds; in debug, the multiply would panic.
        let v = [ve(0, 10, i64::MAX, 3)];
        let err = validate_view(&v, &[100]).unwrap_err();
        assert!(
            err.to_string().contains("overflow"),
            "expected overflow error, got: {err}"
        );
    }

    #[test]
    fn validate_view_rejects_i64_overflow_in_start_plus_offset() {
        // (steps-1)*step = i64::MAX - 1 fits in i64. Adding a small,
        // in-range start of 2 then overflows i64::MAX. The start bound
        // check passes (2 < 100), so this exercises the checked_add arm
        // specifically, not the start guard or the checked_mul arm.
        let v = [ve(0, 2, 1, i64::MAX)];
        let err = validate_view(&v, &[100]).unwrap_err();
        assert!(
            err.to_string().contains("overflow"),
            "expected overflow error, got: {err}"
        );
    }

    /// Minimal `BandRef` stub: only the inputs `is_2d` actually inspects
    /// (`dim_names`, `view`, `raw_source_shape`) carry meaningful values;
    /// every other method returns a placeholder we never read.
    struct StubBand {
        dim_names: Vec<String>,
        source_shape: Vec<u64>,
        shape: Vec<u64>,
        view: Vec<ViewEntry>,
    }

    impl BandRef for StubBand {
        fn ndim(&self) -> usize {
            self.dim_names.len()
        }
        fn dim_names(&self) -> Vec<&str> {
            self.dim_names.iter().map(String::as_str).collect()
        }
        fn shape(&self) -> &[u64] {
            &self.shape
        }
        fn raw_source_shape(&self) -> &[u64] {
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
            unimplemented!("not used in is_2d tests")
        }
        fn contiguous_data(&self) -> Result<Cow<'_, [u8]>, ArrowError> {
            unimplemented!("not used in is_2d tests")
        }
    }

    fn band(dims: &[&str], source_shape: &[u64], view: &[ViewEntry]) -> StubBand {
        let shape = view.iter().map(|v| v.steps as u64).collect();
        StubBand {
            dim_names: dims.iter().map(|s| (*s).to_string()).collect(),
            source_shape: source_shape.to_vec(),
            shape,
            view: view.to_vec(),
        }
    }

    #[test]
    fn is_2d_identity_yx_is_true() {
        let b = band(&["y", "x"], &[4, 5], &[ve(0, 0, 1, 4), ve(1, 0, 1, 5)]);
        assert!(b.is_2d());
    }

    #[test]
    fn is_2d_identity_3d_is_false() {
        let b = band(
            &["time", "y", "x"],
            &[3, 4, 5],
            &[ve(0, 0, 1, 3), ve(1, 0, 1, 4), ve(2, 0, 1, 5)],
        );
        assert!(!b.is_2d());
    }

    #[test]
    fn is_2d_identity_1d_is_false() {
        let b = band(&["x"], &[5], &[ve(0, 0, 1, 5)]);
        assert!(!b.is_2d());
    }

    #[test]
    fn is_2d_yx_with_slice_view_is_false() {
        // Same dim_names but the y-axis is sliced — view is not the identity.
        let b = band(&["y", "x"], &[4, 5], &[ve(0, 1, 1, 2), ve(1, 0, 1, 5)]);
        assert!(!b.is_2d());
    }

    #[test]
    fn is_2d_yx_with_step_two_is_false() {
        let b = band(&["y", "x"], &[4, 5], &[ve(0, 0, 2, 2), ve(1, 0, 1, 5)]);
        assert!(!b.is_2d());
    }

    #[test]
    fn is_2d_yx_with_broadcast_is_false() {
        let b = band(&["y", "x"], &[4, 5], &[ve(0, 0, 0, 4), ve(1, 0, 1, 5)]);
        assert!(!b.is_2d());
    }

    #[test]
    fn is_2d_permuted_xy_is_false() {
        // dim_names are swapped — not the legacy 2D shape, even though the
        // view per-axis is the identity.
        let b = band(&["x", "y"], &[5, 4], &[ve(0, 0, 1, 5), ve(1, 0, 1, 4)]);
        assert!(!b.is_2d());
    }

    #[test]
    fn is_2d_yx_with_transposed_source_axes_is_false() {
        // dim_names are ["y","x"] but the view permutes the source axes,
        // so the band exposes y-then-x out of an x-then-y source.
        let b = band(&["y", "x"], &[5, 4], &[ve(1, 0, 1, 4), ve(0, 0, 1, 5)]);
        assert!(!b.is_2d());
    }

    #[test]
    fn is_2d_yx_other_dim_names_is_false() {
        let b = band(&["lat", "lon"], &[4, 5], &[ve(0, 0, 1, 4), ve(1, 0, 1, 5)]);
        assert!(!b.is_2d());
    }
}
