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

//! The raster crates' domain error type.
//!
//! The raster code used to plumb [`arrow_schema::ArrowError`] everywhere,
//! which forced a `map_err` at every boundary that produced or consumed a
//! raster error. [`RasterError`] mirrors `sedona-geometry`'s
//! `SedonaGeometryError`: a small `thiserror` enum that library accessors
//! return directly, with `From` bridges so callers use `?` instead of a
//! hand-written `map_err`.
//!
//! The one raster-specific addition over the geometry mirror is the
//! [`RasterError::Arrow`] variant with `#[from] ArrowError`: raster code
//! calls Arrow APIs constantly, so `?` over an `ArrowError` needs to resolve
//! to a `RasterError` without a `map_err`.
//!
//! Boundaries:
//! - `From<ArrowError> for RasterError` — `?` over Arrow calls inside the
//!   raster library.
//! - `From<RasterError> for DataFusionError` — `?` at UDF boundaries (the
//!   raster-functions / raster-gdal crates return `DataFusionError`).
//! - `From<RasterError> for ArrowError` — the bridge back for signatures
//!   fixed by Arrow, notably `RecordBatchReader::next`.
//!
//! Internal-assertion errors ("this is a SedonaDB bug") are intentionally
//! *not* a variant here: they stay a boundary concern raised through
//! `sedona_common::sedona_internal_err!`, which produces a `DataFusionError`
//! directly — same as `sedona-geometry`, which has no internal variant.

use arrow_schema::ArrowError;
use datafusion_common::{exec_datafusion_err, DataFusionError};
use thiserror::Error;

/// Error type for the raster crates.
///
/// Mirrors `sedona_geometry::error::SedonaGeometryError`. Construct
/// [`RasterError::Invalid`] for a bad-input/precondition failure and
/// [`RasterError::External`] to carry an arbitrary source error; Arrow
/// errors convert automatically via `?`.
#[derive(Error, Debug)]
pub enum RasterError {
    /// Invalid input or a violated precondition, described by the message.
    #[error("{0}")]
    Invalid(String),
    /// An Arrow error encountered while building or reading raster arrays.
    #[error(transparent)]
    Arrow(#[from] ArrowError),
    /// Any other source error, boxed.
    #[error(transparent)]
    External(Box<dyn std::error::Error + Send + Sync>),
    /// A raster error with no more specific classification.
    #[error("Unknown raster error")]
    Unknown,
}

impl From<RasterError> for DataFusionError {
    fn from(value: RasterError) -> Self {
        match value {
            // Keep Arrow errors typed so DataFusion renders them as Arrow
            // errors rather than an opaque external error. Delegate to
            // DataFusion's own `From` so the backtrace is attached exactly as
            // `arrow_datafusion_err!` would have done at the sites this
            // replaces.
            RasterError::Arrow(e) => DataFusionError::from(e),
            // `Execution` (not `Internal`/`External`) matches the existing
            // `exec_datafusion_err!` sites this replaces: a user-facing
            // execution error without the "this is a DataFusion bug" framing.
            // Go through the macro rather than constructing `Execution`
            // directly — `Execution` has no backtrace slot, so the macro
            // appends the backtrace to the message, and building the variant
            // by hand would silently drop it.
            RasterError::Invalid(msg) => exec_datafusion_err!("{msg}"),
            // `External` has no backtrace slot either, and no macro precedent
            // among the replaced sites — carry the source through unchanged.
            RasterError::External(e) => DataFusionError::External(e),
            RasterError::Unknown => exec_datafusion_err!("Unknown raster error"),
        }
    }
}

impl From<RasterError> for ArrowError {
    fn from(value: RasterError) -> Self {
        match value {
            RasterError::Arrow(e) => e,
            // `InvalidArgumentError` is what `Invalid` is built from on the way
            // in, so map it back rather than collapsing it into
            // `ExternalError`. Functions that still return `ArrowError` (e.g.
            // `Grid::from_raster`) otherwise change variant for the same
            // failure.
            RasterError::Invalid(msg) => ArrowError::InvalidArgumentError(msg),
            RasterError::External(e) => ArrowError::ExternalError(e),
            other => ArrowError::ExternalError(Box::new(other)),
        }
    }
}

impl From<DataFusionError> for RasterError {
    fn from(value: DataFusionError) -> Self {
        // Raster accessors lean on `datafusion_common::cast::as_*`, which
        // return `DataFusionError`. Carry it as a source so `?` works without
        // a `map_err` and the original message survives round-tripping back to
        // `DataFusionError` at the UDF boundary.
        RasterError::External(Box::new(value))
    }
}

/// Attach a message to a fallible result and convert it to a [`RasterError`].
///
/// Replaces the `map_err(|e| exec_datafusion_err!("context: {e}"))` pattern:
/// `foo().context("context")?` formats the message as `"context: {source}"`
/// and, via the `From` bridges above, resolves under `?` in functions that
/// return `RasterError`, `DataFusionError`, or `ArrowError`.
pub trait RasterResultExt<T> {
    /// Prefix the error with an eagerly-evaluated `context` message.
    fn context(self, context: impl std::fmt::Display) -> Result<T, RasterError>;

    /// Prefix the error with a lazily-computed message, only paid on error.
    fn with_context<C, F>(self, context: F) -> Result<T, RasterError>
    where
        C: std::fmt::Display,
        F: FnOnce() -> C;
}

impl<T, E: std::fmt::Display> RasterResultExt<T> for Result<T, E> {
    fn context(self, context: impl std::fmt::Display) -> Result<T, RasterError> {
        self.map_err(|e| RasterError::Invalid(format!("{context}: {e}")))
    }

    fn with_context<C, F>(self, context: F) -> Result<T, RasterError>
    where
        C: std::fmt::Display,
        F: FnOnce() -> C,
    {
        self.map_err(|e| RasterError::Invalid(format!("{}: {e}", context())))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn invalid_and_external_display() {
        let invalid = RasterError::Invalid("bad band".to_string());
        assert_eq!(invalid.to_string(), "bad band");

        let source = Box::new(std::io::Error::other("boom"));
        let external = RasterError::External(source);
        assert_eq!(external.to_string(), "boom");

        assert_eq!(RasterError::Unknown.to_string(), "Unknown raster error");
    }

    #[test]
    fn arrow_round_trips_without_double_wrapping() {
        // `?` over an ArrowError yields RasterError::Arrow ...
        let err: RasterError = ArrowError::ComputeError("nope".to_string()).into();
        assert!(matches!(err, RasterError::Arrow(_)));
        // ... and the bridge back to ArrowError unwraps it rather than
        // boxing it as an ExternalError.
        let back: ArrowError = err.into();
        assert!(matches!(back, ArrowError::ComputeError(_)));
    }

    #[test]
    fn invalid_argument_round_trips_to_the_same_arrow_variant() {
        // Functions that still return `ArrowError` reach it through this
        // bridge (e.g. `Grid::from_raster` calling `RasterRef::width`), so the
        // variant a caller sees must not change just because the value took a
        // detour through `RasterError`.
        let original = ArrowError::InvalidArgumentError("no width".to_string());
        let via_raster: RasterError = original.into();
        let back: ArrowError = via_raster.into();
        assert!(
            matches!(back, ArrowError::InvalidArgumentError(_)),
            "expected InvalidArgumentError, got {back:?}"
        );
        assert_eq!(back.to_string(), "Invalid argument error: no width");

        // A `RasterError::Invalid` raised directly (not from Arrow) maps to the
        // same variant, so both origins are indistinguishable downstream.
        let direct: ArrowError = RasterError::Invalid("no width".to_string()).into();
        assert!(matches!(direct, ArrowError::InvalidArgumentError(_)));

        // `External` unwraps to `ExternalError` rather than double-boxing.
        let external: ArrowError =
            RasterError::External(Box::new(std::io::Error::other("boom"))).into();
        match external {
            ArrowError::ExternalError(e) => assert_eq!(e.to_string(), "boom"),
            other => panic!("expected ExternalError, got {other:?}"),
        }
    }

    #[test]
    fn into_datafusion_preserves_kind() {
        let df: DataFusionError = RasterError::Invalid("bad".to_string()).into();
        assert!(matches!(df, DataFusionError::Execution(_)));

        let df: DataFusionError = RasterError::Arrow(ArrowError::ComputeError("x".into())).into();
        assert!(matches!(df, DataFusionError::ArrowError(_, _)));
    }

    #[test]
    fn context_prefixes_message() {
        // Use a plain-Display source so the assertion reads clearly; the point
        // is that `context` prefixes the message with the source appended.
        let r: Result<(), RasterError> = Err(RasterError::Invalid("underlying".to_string()));
        let err = r.context("while doing the thing").unwrap_err();
        assert_eq!(err.to_string(), "while doing the thing: underlying");

        // Lazy variant is equivalent on the error path.
        let r: Result<(), RasterError> = Err(RasterError::Invalid("underlying".to_string()));
        let err = r.with_context(|| "while doing the thing").unwrap_err();
        assert_eq!(err.to_string(), "while doing the thing: underlying");

        // The source error's own Display is appended verbatim, so an Arrow
        // source keeps its "Compute error:" prefix.
        let r: Result<(), ArrowError> = Err(ArrowError::ComputeError("boom".to_string()));
        let err = r.context("ctx").unwrap_err();
        assert_eq!(err.to_string(), "ctx: Compute error: boom");
    }
}
