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

//! Zarr backend implementing [`sedona_raster::outdb_loader::AsyncByteLoader`].
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
//! [`OutDbLoaderRegistry`](sedona_raster::outdb_loader::OutDbLoaderRegistry)
//! under the format key [`ZARR_FORMAT`]. As an out-of-tree plugin,
//! `sedona-raster-zarr` does not depend on `sedona` — callers wire the
//! registration themselves from their `SedonaContext` setup:
//!
//! ```ignore
//! ctx.register_outdb_loader(
//!     sedona_raster_zarr::ZARR_FORMAT,
//!     std::sync::Arc::new(sedona_raster_zarr::ZarrLoader::new()),
//! );
//! ```

use std::sync::Arc;

use arrow_buffer::Buffer;
use arrow_schema::ArrowError;
use async_trait::async_trait;
use sedona_common::sedona_internal_datafusion_err;
use sedona_raster::outdb_loader::{AsyncByteLoader, OutDbLoadRequest};
use zarrs::group::Group;
use zarrs_filesystem::FilesystemStore;

use crate::dtype::zarr_to_band_data_type;
use crate::loader::retrieve_chunk_bytes;
use crate::source_uri::{group_uri_to_filesystem_path, parse_chunk_anchor};

/// Format key the loader registers under. Keep in sync with
/// `outdb_format` values emitted by the Zarr reader's band builder
/// (see `crate::loader`).
pub const ZARR_FORMAT: &str = "zarr";

/// Async OutDb byte loader for Zarr-backed bands.
///
/// Stateless: dataset opens use a fresh `FilesystemStore` per call.
/// Caching the open store per `(store_uri, array_path)` is a follow-up
/// optimisation that doesn't change the trait surface.
#[derive(Debug, Default, Clone, Copy)]
pub struct ZarrLoader;

impl ZarrLoader {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl AsyncByteLoader for ZarrLoader {
    async fn load(&self, req: &OutDbLoadRequest<'_>) -> Result<Buffer, ArrowError> {
        // Take owned copies for the spawn_blocking closure.
        let uri = req.uri.to_string();
        let expected_dtype = req.data_type;

        let buffer = tokio::task::spawn_blocking(move || -> Result<Buffer, ArrowError> {
            let anchor = parse_chunk_anchor(&uri)?;
            let fs_path = group_uri_to_filesystem_path(&anchor.store_uri)?;
            let store = FilesystemStore::new(&fs_path).map_err(|e| {
                ArrowError::ExternalError(Box::new(sedona_internal_datafusion_err!(
                    "failed to open Zarr store at {}: {e}",
                    fs_path.display()
                )))
            })?;
            let storage: Arc<FilesystemStore> = Arc::new(store);

            // The group itself isn't strictly needed to open an array,
            // but resolving the array path through it gives a clear
            // diagnostic if the group root or the array path is wrong.
            let group = Group::open(storage.clone(), "/").map_err(|e| {
                ArrowError::ExternalError(Box::new(sedona_internal_datafusion_err!(
                    "failed to open Zarr group at {}: {e}",
                    fs_path.display()
                )))
            })?;
            let _ = group; // group handle dropped; array open uses storage directly.

            let array_path = if anchor.array_path.starts_with('/') {
                anchor.array_path.clone()
            } else {
                format!("/{}", anchor.array_path)
            };
            let array = zarrs::array::Array::open(storage, &array_path).map_err(|e| {
                ArrowError::ExternalError(Box::new(sedona_internal_datafusion_err!(
                    "failed to open Zarr array {}: {e}",
                    array_path
                )))
            })?;

            // Verify the Zarr array's dtype matches the band metadata's
            // claim before reading. Mismatches catch silent byte-count
            // surprises here rather than letting RS_EnsureLoaded's
            // expected-byte-count check mis-blame the loader for size.
            let file_dtype = zarr_to_band_data_type(array.data_type())?;
            if file_dtype != expected_dtype {
                return Err(ArrowError::ExternalError(Box::new(
                    sedona_internal_datafusion_err!(
                        "Zarr OutDb band metadata claims {:?} but array {} is {:?}",
                        expected_dtype,
                        array_path,
                        file_dtype
                    ),
                )));
            }

            let bytes = retrieve_chunk_bytes(&array, &anchor.chunk_indices)?;
            Ok(Buffer::from_vec(bytes))
        })
        .await
        .map_err(|e| {
            ArrowError::ExternalError(Box::new(sedona_internal_datafusion_err!(
                "Zarr OutDb loader task panicked or was cancelled: {e}"
            )))
        })??;

        Ok(buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sedona_schema::raster::BandDataType;
    use tempfile::TempDir;
    use zarrs::array::ArrayBuilder;
    use zarrs::array::{data_type as zarr_dtype, FillValue};
    use zarrs::group::GroupBuilder;

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
        let req = OutDbLoadRequest {
            uri: &uri,
            dim_names: &["y", "x"],
            source_shape: &[2, 3],
            data_type: BandDataType::UInt8,
        };
        let buf = loader.load(&req).await.unwrap();
        assert_eq!(buf.as_slice(), expected_pixels.as_slice());
    }

    #[tokio::test]
    async fn zarr_loader_errors_when_dtype_disagrees_with_array() {
        let tmp = TempDir::new().unwrap();
        let (store_uri, array_path, _) = build_uint8_zarr(&tmp);
        let uri = build_chunk_anchor(&store_uri, array_path, &[0, 0]);

        let loader = ZarrLoader::new();
        let req = OutDbLoadRequest {
            uri: &uri,
            dim_names: &["y", "x"],
            source_shape: &[2, 3],
            // Array is UInt8 but the band claims Int16.
            data_type: BandDataType::Int16,
        };
        let err = loader.load(&req).await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("metadata claims") && (msg.contains("UInt8") || msg.contains("Int16")),
            "expected dtype-mismatch diagnostic, got: {msg}"
        );
    }

    #[tokio::test]
    async fn zarr_loader_errors_on_malformed_chunk_anchor_uri() {
        let loader = ZarrLoader::new();
        let req = OutDbLoadRequest {
            uri: "file:///tmp/foo.zarr", // missing fragment
            dim_names: &["y", "x"],
            source_shape: &[2, 3],
            data_type: BandDataType::UInt8,
        };
        let err = loader.load(&req).await.unwrap_err();
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
        let req = OutDbLoadRequest {
            uri: &uri,
            dim_names: &["y", "x"],
            source_shape: &[2, 3],
            data_type: BandDataType::UInt8,
        };
        let err = loader.load(&req).await.unwrap_err();
        assert!(
            err.to_string().contains("nonexistent")
                || err.to_string().to_lowercase().contains("array"),
            "expected diagnostic to name the missing array path, got: {err}"
        );
    }

    #[tokio::test]
    async fn zarr_loader_errors_on_cloud_scheme_until_supported() {
        let loader = ZarrLoader::new();
        let uri = build_chunk_anchor("s3://bucket/foo.zarr", "temperature", &[0, 0]);
        let req = OutDbLoadRequest {
            uri: &uri,
            dim_names: &["y", "x"],
            source_shape: &[2, 3],
            data_type: BandDataType::UInt8,
        };
        let err = loader.load(&req).await.unwrap_err();
        assert!(
            err.to_string().contains("cloud") || err.to_string().contains("s3://"),
            "expected cloud-scheme rejection, got: {err}"
        );
    }
}
