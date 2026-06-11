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

//! Zarr backend implementing [`sedona_raster::raster_loader::AsyncRasterLoader`].
//!
//! Resolves a band's OutDb URI back into a Zarr chunk read: the URI is
//! a chunk anchor of the form
//! `<store_uri>#array=<array_path>&chunk=<i0>,<i1>,...` (see
//! [`crate::source_uri::build_chunk_anchor`]). The loader parses the
//! anchor, opens the Zarr store and array, and retrieves the named
//! chunk's bytes via `zarrs` — all wrapped in
//! `tokio::task::spawn_blocking` so the caller's async runtime is not
//! stalled by Zarr's blocking decoder.
//!
//! Registered against the per-session
//! [`RasterLoaderRegistry`](sedona_raster::raster_loader::RasterLoaderRegistry);
//! the loader claims the [`ZARR_FORMAT`] `outdb_format` via
//! `supports_format`. As an out-of-tree plugin, `sedona-raster-zarr` does
//! not depend on `sedona` — callers wire the registration themselves from
//! their `SedonaContext` setup:
//!
//! ```ignore
//! ctx.register_raster_loader(std::sync::Arc::new(sedona_raster_zarr::ZarrLoader::new()));
//! ```

use arrow_buffer::Buffer;
use arrow_schema::ArrowError;
use async_trait::async_trait;
use sedona_common::sedona_internal_datafusion_err;
use sedona_raster::raster_loader::{AsyncRasterLoader, RasterLoadRequest, RasterLoadResult};
use zarrs::array::{Array, ArrayBytes};

use crate::dtype::zarr_to_band_data_type;
use crate::source_uri::{object_store_for_uri, open_storage_from_uri, parse_chunk_anchor};

/// Format key the loader registers under. Keep in sync with
/// `outdb_format` values emitted by the Zarr reader's band builder
/// (see `crate::loader`).
pub const ZARR_FORMAT: &str = "zarr";

/// Async raster byte loader for Zarr-backed bands.
///
/// Stateless: each call builds an `ObjectStore` from the anchor's store URI
/// (via [`object_store_for_uri`]) and opens the array over async storage.
/// Building the store here is the in-process bridge until a credentialed
/// store can be passed in from DataFusion's `ObjectStoreRegistry`; see the
/// loader-FFI follow-up.
#[derive(Debug, Default, Clone, Copy)]
pub struct ZarrLoader;

impl ZarrLoader {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl AsyncRasterLoader for ZarrLoader {
    fn name(&self) -> &str {
        ZARR_FORMAT
    }

    /// Zarr-specific: claims only bands whose `outdb_format` is
    /// [`ZARR_FORMAT`]. Everything else (including the unset `None` that
    /// `RS_FromPath` emits) falls through to the catch-all GDAL loader.
    fn supports_format(&self, format: Option<&str>) -> bool {
        format == Some(ZARR_FORMAT)
    }

    async fn load(&self, reqs: &[&RasterLoadRequest]) -> Result<Vec<RasterLoadResult>, ArrowError> {
        // These requests are currently issued in serial. We may want to increase the
        // concurrency of these requests; however, in many cases we will have multiple
        // partitions issuing these requests at once.
        let mut results = Vec::with_capacity(reqs.len());
        for req in reqs {
            results.push(self.load_one(req).await?);
        }
        Ok(results)
    }
}

impl ZarrLoader {
    async fn load_one(&self, req: &RasterLoadRequest<'_>) -> Result<RasterLoadResult, ArrowError> {
        let anchor = parse_chunk_anchor(req.uri)?;
        // Build the store from the anchor's store URI. Temporary: when a
        // credentialed store can be passed in from DataFusion's
        // ObjectStoreRegistry, this is replaced by a store carried on the
        // request (loader-FFI follow-up).
        let store = object_store_for_uri(&anchor.store_uri)?;
        let storage = open_storage_from_uri(&anchor.store_uri, store)?;

        let array_path = if anchor.array_path.starts_with('/') {
            anchor.array_path.clone()
        } else {
            format!("/{}", anchor.array_path)
        };
        let array = Array::async_open(storage, &array_path).await.map_err(|e| {
            ArrowError::ExternalError(Box::new(sedona_internal_datafusion_err!(
                "failed to open Zarr array {array_path}: {e}"
            )))
        })?;

        // Verify the Zarr array's dtype matches the band metadata's claim
        // before reading. Mismatches surface here rather than letting
        // RS_EnsureLoaded's expected-byte-count check mis-blame the loader.
        let file_dtype = zarr_to_band_data_type(array.data_type())?;
        if file_dtype != req.data_type {
            return Err(ArrowError::ExternalError(Box::new(
                sedona_internal_datafusion_err!(
                    "Zarr OutDb band metadata claims {:?} but array {} is {:?}",
                    req.data_type,
                    array_path,
                    file_dtype
                ),
            )));
        }

        let bytes = array
            .async_retrieve_chunk::<ArrayBytes<'static>>(&anchor.chunk_indices)
            .await
            .map_err(|e| {
                ArrowError::ExternalError(Box::new(sedona_internal_datafusion_err!(
                    "failed to retrieve chunk {:?} from {array_path}: {e}",
                    anchor.chunk_indices
                )))
            })?;
        let raw = bytes.into_fixed().map_err(|_| {
            ArrowError::InvalidArgumentError(format!(
                "array {array_path}: variable-length chunk bytes are not supported"
            ))
        })?;
        Ok(RasterLoadResult::unresolved(
            Buffer::from_vec(raw.into_owned()),
            req,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use sedona_schema::raster::BandDataType;
    use tempfile::TempDir;
    use zarrs::array::ArrayBuilder;
    use zarrs::array::{data_type as zarr_dtype, FillValue};
    use zarrs::group::GroupBuilder;
    use zarrs_filesystem::FilesystemStore;

    use crate::source_uri::build_chunk_anchor;

    /// Build a Zarr group at `<tempdir>/store.zarr` containing one array
    /// `temperature` of UInt8 with shape [2, 3] and chunk shape [2, 3]
    /// (one chunk). Returns the store URI and array path.
    fn build_uint8_zarr(dir: &TempDir) -> (String, &'static str, Vec<u8>) {
        let store_path = dir.path().join("store.zarr");
        let store = Arc::new(FilesystemStore::new(&store_path).unwrap());

        // Root group metadata — Zarr v3 stores need this for
        // `Group::open(store, "/")` to succeed.
        GroupBuilder::new()
            .build(store.clone(), "/")
            .unwrap()
            .store_metadata()
            .unwrap();

        let array = ArrayBuilder::new(
            vec![2, 3],
            vec![2, 3],
            zarr_dtype::uint8(),
            FillValue::from(0u8),
        )
        .build(store.clone(), "/temperature")
        .unwrap();
        array.store_metadata().unwrap();

        let pixels: Vec<u8> = vec![10, 11, 12, 13, 14, 15];
        array.store_chunk(&[0, 0], pixels.clone()).unwrap();

        let store_uri = format!("file://{}", store_path.display());
        (store_uri, "temperature", pixels)
    }

    #[tokio::test]
    async fn zarr_loader_reads_uint8_chunk() {
        let tmp = TempDir::new().unwrap();
        let (store_uri, array_path, expected_pixels) = build_uint8_zarr(&tmp);
        let uri = build_chunk_anchor(&store_uri, array_path, &[0, 0]);

        let loader = ZarrLoader::new();
        let req = RasterLoadRequest {
            uri: &uri,
            dim_names: &["y", "x"],
            source_shape: &[2, 3],
            view: &[],
            data_type: BandDataType::UInt8,
        };
        let result = loader.load(&[&req]).await.unwrap();
        assert_eq!(result[0].bytes.as_slice(), expected_pixels.as_slice());
    }

    #[tokio::test]
    async fn zarr_loader_errors_when_dtype_disagrees_with_array() {
        let tmp = TempDir::new().unwrap();
        let (store_uri, array_path, _) = build_uint8_zarr(&tmp);
        let uri = build_chunk_anchor(&store_uri, array_path, &[0, 0]);

        let loader = ZarrLoader::new();
        let req = RasterLoadRequest {
            uri: &uri,
            dim_names: &["y", "x"],
            source_shape: &[2, 3],
            view: &[], // Array is UInt8 but the band claims Int16.
            data_type: BandDataType::Int16,
        };
        let err = loader.load(&[&req]).await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("metadata claims") && (msg.contains("UInt8") || msg.contains("Int16")),
            "expected dtype-mismatch diagnostic, got: {msg}"
        );
    }

    #[tokio::test]
    async fn zarr_loader_errors_on_malformed_chunk_anchor_uri() {
        let loader = ZarrLoader::new();
        let req = RasterLoadRequest {
            uri: "file:///tmp/foo.zarr", // missing fragment
            dim_names: &["y", "x"],
            source_shape: &[2, 3],
            view: &[],
            data_type: BandDataType::UInt8,
        };
        let err = loader.load(&[&req]).await.unwrap_err();
        assert!(
            err.to_string().contains("missing"),
            "expected missing-fragment diagnostic, got: {err}"
        );
    }

    #[tokio::test]
    async fn zarr_loader_errors_on_missing_array_path() {
        let tmp = TempDir::new().unwrap();
        let (store_uri, _, _) = build_uint8_zarr(&tmp);
        // Anchor a chunk against a non-existent array.
        let uri = build_chunk_anchor(&store_uri, "nonexistent", &[0, 0]);

        let loader = ZarrLoader::new();
        let req = RasterLoadRequest {
            uri: &uri,
            dim_names: &["y", "x"],
            source_shape: &[2, 3],
            view: &[],
            data_type: BandDataType::UInt8,
        };
        let err = loader.load(&[&req]).await.unwrap_err();
        assert!(
            err.to_string().contains("nonexistent")
                || err.to_string().to_lowercase().contains("array"),
            "expected diagnostic to name the missing array path, got: {err}"
        );
    }

    #[tokio::test]
    async fn zarr_loader_errors_on_unsupported_scheme() {
        // `s3`/`http` are supported now; an unsupported scheme (e.g. `gs://`)
        // is rejected at store construction, without touching the network.
        let loader = ZarrLoader::new();
        let uri = build_chunk_anchor("gs://bucket/foo.zarr", "temperature", &[0, 0]);
        let req = RasterLoadRequest {
            uri: &uri,
            dim_names: &["y", "x"],
            source_shape: &[2, 3],
            view: &[],
            data_type: BandDataType::UInt8,
        };
        let err = loader.load(&[&req]).await.unwrap_err();
        assert!(
            err.to_string().contains("unsupported Zarr URI scheme"),
            "expected unsupported-scheme rejection, got: {err}"
        );
    }
}
