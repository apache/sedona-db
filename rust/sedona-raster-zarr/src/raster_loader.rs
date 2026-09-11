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
//! [`crate::source_uri::build_chunk_anchor`]).
//!
//! `load` treats its request slice as a batch (see the trait docs): the
//! anchors are first reduced to the distinct stores, arrays and chunks
//! they touch ([`LoadPlan`]), then each store is built once, each array is
//! opened once (one metadata round trip per distinct array rather than
//! per request), and each distinct chunk is fetched once with up to
//! [`ZarrLoader::concurrency`] reads in flight. Requests that name the
//! same chunk share the loaded bytes. All I/O goes through `zarrs`'s async
//! API over `object_store`, so nothing blocks the caller's runtime.
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

use std::collections::HashMap;
use std::hash::Hash;

use arrow_buffer::Buffer;
use arrow_schema::ArrowError;
use async_trait::async_trait;
use futures::{StreamExt, TryStreamExt, stream};
use sedona_common::sedona_internal_datafusion_err;
use sedona_raster::raster_loader::{AsyncRasterLoader, RasterLoadRequest, RasterLoadResult};
use zarrs::array::{Array, ArrayBytes};
use zarrs::storage::{AsyncReadableListableStorage, AsyncReadableListableStorageTraits};

use crate::dtype::zarr_to_band_data_type;
use crate::source_uri::{object_store_for_uri, open_storage_from_uri, parse_chunk_anchor};

/// Format key the loader registers under. Keep in sync with
/// `outdb_format` values emitted by the Zarr reader's band builder
/// (see `crate::loader`).
pub const ZARR_FORMAT: &str = "zarr";

/// Default cap on chunk reads in flight per `load` call. Every partition
/// issues its own `load` concurrently, so this bounds the per-partition
/// fan-out, not the process-wide one.
pub const DEFAULT_LOAD_CONCURRENCY: usize = 8;

/// Async raster byte loader for Zarr-backed bands.
///
/// Stateless across calls: each call builds an `ObjectStore` per distinct
/// store URI (via [`object_store_for_uri`]) and opens each distinct array
/// over async storage. Building the store here is the in-process bridge
/// until a credentialed store can be passed in from DataFusion's
/// `ObjectStoreRegistry`; see the loader-FFI follow-up.
#[derive(Debug, Clone, Copy)]
pub struct ZarrLoader {
    concurrency: usize,
}

impl Default for ZarrLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl ZarrLoader {
    pub fn new() -> Self {
        Self {
            concurrency: DEFAULT_LOAD_CONCURRENCY,
        }
    }

    /// Cap on chunk reads in flight per `load` call. Clamped to at least 1.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency.max(1);
        self
    }

    /// The configured cap on in-flight chunk reads per `load` call.
    pub fn concurrency(&self) -> usize {
        self.concurrency
    }
}

/// A batch of chunk anchors reduced to the distinct stores, arrays and
/// chunks it touches (each in first-seen order) plus the request → chunk
/// map. Pure — no I/O — so the dedupe is unit-testable on its own.
#[derive(Debug, PartialEq, Eq)]
struct LoadPlan {
    /// Distinct store URIs.
    stores: Vec<String>,
    /// Distinct `(store index, absolute array path)` pairs.
    arrays: Vec<(usize, String)>,
    /// Distinct `(array index, chunk indices)` pairs.
    chunks: Vec<(usize, Vec<u64>)>,
    /// For each request, in order, the index into `chunks` that serves it.
    request_chunk: Vec<usize>,
}

impl LoadPlan {
    fn build<'a>(uris: impl IntoIterator<Item = &'a str>) -> Result<Self, ArrowError> {
        let mut plan = Self {
            stores: Vec::new(),
            arrays: Vec::new(),
            chunks: Vec::new(),
            request_chunk: Vec::new(),
        };
        let mut store_ids = HashMap::new();
        let mut array_ids = HashMap::new();
        let mut chunk_ids = HashMap::new();

        for uri in uris {
            let anchor = parse_chunk_anchor(uri)?;
            let array_path = if anchor.array_path.starts_with('/') {
                anchor.array_path
            } else {
                format!("/{}", anchor.array_path)
            };
            let store = intern(&mut store_ids, &mut plan.stores, anchor.store_uri);
            let array = intern(&mut array_ids, &mut plan.arrays, (store, array_path));
            let chunk = intern(
                &mut chunk_ids,
                &mut plan.chunks,
                (array, anchor.chunk_indices),
            );
            plan.request_chunk.push(chunk);
        }
        Ok(plan)
    }
}

/// Index of `key` in `items`, appending it (and recording it in `ids`) on
/// first sight.
fn intern<K: Clone + Eq + Hash>(ids: &mut HashMap<K, usize>, items: &mut Vec<K>, key: K) -> usize {
    if let Some(&id) = ids.get(&key) {
        return id;
    }
    let id = items.len();
    items.push(key.clone());
    ids.insert(key, id);
    id
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
        let plan = LoadPlan::build(reqs.iter().map(|req| req.uri))?;

        // One store per distinct store URI.
        let storages: Vec<AsyncReadableListableStorage> = plan
            .stores
            .iter()
            .map(|store_uri| {
                let store = object_store_for_uri(store_uri)?;
                open_storage_from_uri(store_uri, store)
            })
            .collect::<Result<_, _>>()?;

        // One open per distinct array: a metadata round trip each, so this
        // is the dedupe that matters for a batch of rows over the same
        // group.
        let mut arrays: Vec<Array<dyn AsyncReadableListableStorageTraits>> =
            Vec::with_capacity(plan.arrays.len());
        for (store, array_path) in &plan.arrays {
            let store_uri = &plan.stores[*store];
            let array = Array::async_open(storages[*store].clone(), array_path)
                .await
                .map_err(|e| {
                    ArrowError::ExternalError(Box::new(sedona_internal_datafusion_err!(
                        "failed to open Zarr array {array_path} in {store_uri}: {e}"
                    )))
                })?;
            arrays.push(array);
        }

        // Verify every request's claimed dtype against its array before
        // reading anything. Mismatches surface here rather than letting
        // RS_EnsureLoaded's expected-byte-count check mis-blame the loader.
        for (req, &chunk) in reqs.iter().zip(&plan.request_chunk) {
            let (array_idx, _) = plan.chunks[chunk];
            let array_path = &plan.arrays[array_idx].1;
            let file_dtype = zarr_to_band_data_type(arrays[array_idx].data_type())?;
            if file_dtype != req.data_type {
                return Err(ArrowError::ExternalError(Box::new(
                    sedona_internal_datafusion_err!(
                        "Zarr OutDb band metadata claims {:?} but array {} is {:?} ({})",
                        req.data_type,
                        array_path,
                        file_dtype,
                        req.uri
                    ),
                )));
            }
        }

        // Fetch each distinct chunk once, `concurrency` at a time. Results
        // complete out of order, so carry the chunk index and slot them
        // back afterwards. `try_collect` stops at the first error and drops
        // whatever is still in flight.
        // (A plain loop rather than `iter().map(|..| async move {..})`: the
        // closure form trips rustc's "implementation of `FnOnce` is not
        // general enough" check under async_trait's `Send` bound.)
        let mut fetches = Vec::with_capacity(plan.chunks.len());
        for (chunk_idx, (array_idx, chunk_indices)) in plan.chunks.iter().enumerate() {
            let array = &arrays[*array_idx];
            let (store, array_path) = &plan.arrays[*array_idx];
            let store_uri = plan.stores[*store].as_str();
            fetches.push(async move {
                retrieve_chunk(array, store_uri, array_path, chunk_indices)
                    .await
                    .map(|buffer| (chunk_idx, buffer))
            });
        }
        let fetched: Vec<(usize, Buffer)> = stream::iter(fetches)
            .buffer_unordered(self.concurrency)
            .try_collect()
            .await?;
        let mut buffers: Vec<Option<Buffer>> = vec![None; plan.chunks.len()];
        for (chunk_idx, buffer) in fetched {
            buffers[chunk_idx] = Some(buffer);
        }

        // Scatter back in request order. Requests naming the same chunk
        // share one loaded `Buffer` (refcounted clone, no copy).
        reqs.iter()
            .zip(&plan.request_chunk)
            .map(|(req, &chunk)| {
                let buffer = buffers[chunk].clone().ok_or_else(|| {
                    ArrowError::ExternalError(Box::new(sedona_internal_datafusion_err!(
                        "Zarr loader: chunk {chunk} of {} was never fetched",
                        plan.chunks.len()
                    )))
                })?;
                Ok(RasterLoadResult::unresolved(buffer, req))
            })
            .collect()
    }
}

/// Read one chunk's raw bytes out of an already-open array.
async fn retrieve_chunk(
    array: &Array<dyn AsyncReadableListableStorageTraits>,
    store_uri: &str,
    array_path: &str,
    chunk_indices: &[u64],
) -> Result<Buffer, ArrowError> {
    let bytes = array
        .async_retrieve_chunk::<ArrayBytes<'static>>(chunk_indices)
        .await
        .map_err(|e| {
            ArrowError::ExternalError(Box::new(sedona_internal_datafusion_err!(
                "failed to retrieve chunk {chunk_indices:?} from {array_path} in {store_uri}: {e}"
            )))
        })?;
    let raw = bytes.into_fixed().map_err(|_| {
        ArrowError::InvalidArgumentError(format!(
            "array {array_path} in {store_uri}: variable-length chunk bytes are not supported"
        ))
    })?;
    Ok(Buffer::from_vec(raw.into_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sedona_raster::view_entries::ViewEntries;
    use std::sync::Arc;

    use sedona_schema::raster::BandDataType;
    use tempfile::TempDir;
    use zarrs::array::ArrayBuilder;
    use zarrs::array::{FillValue, data_type as zarr_dtype};
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
            view: &ViewEntries::identity_for_shape(&[2, 3]),
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
            view: &ViewEntries::identity_for_shape(&[2, 3]), // Array is UInt8 but the band claims Int16.
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
            view: &ViewEntries::identity_for_shape(&[2, 3]),
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
            view: &ViewEntries::identity_for_shape(&[2, 3]),
            data_type: BandDataType::UInt8,
        };
        let err = loader.load(&[&req]).await.unwrap_err();
        assert!(
            err.to_string().contains("/nonexistent") && err.to_string().contains(&store_uri),
            "expected diagnostic to name the missing array path and its store, got: {err}"
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
            view: &ViewEntries::identity_for_shape(&[2, 3]),
            data_type: BandDataType::UInt8,
        };
        let err = loader.load(&[&req]).await.unwrap_err();
        assert!(
            err.to_string().contains("unsupported Zarr URI scheme"),
            "expected unsupported-scheme rejection, got: {err}"
        );
    }

    /// Two arrays, each shape `[2, 6]` chunked `[2, 3]` (two chunks apiece),
    /// every chunk holding distinct pixels: temperature chunks are 10..16
    /// and 20..26, pressure chunks are 30..36 and 40..46.
    fn build_two_array_zarr(dir: &TempDir) -> String {
        let store_path = dir.path().join("store.zarr");
        let store = Arc::new(FilesystemStore::new(&store_path).unwrap());
        GroupBuilder::new()
            .build(store.clone(), "/")
            .unwrap()
            .store_metadata()
            .unwrap();
        for (name, base) in [("temperature", 10u8), ("pressure", 30u8)] {
            let array = ArrayBuilder::new(
                vec![2, 6],
                vec![2, 3],
                zarr_dtype::uint8(),
                FillValue::from(0u8),
            )
            .build(store.clone(), &format!("/{name}"))
            .unwrap();
            array.store_metadata().unwrap();
            array
                .store_chunk(&[0, 0], (base..base + 6).collect::<Vec<u8>>())
                .unwrap();
            array
                .store_chunk(&[0, 1], (base + 10..base + 16).collect::<Vec<u8>>())
                .unwrap();
        }
        format!("file://{}", store_path.display())
    }

    #[test]
    fn load_plan_dedupes_stores_arrays_and_chunks() {
        let uris = [
            build_chunk_anchor("file:///a.zarr", "t", &[0, 0]),
            build_chunk_anchor("file:///a.zarr", "t", &[0, 1]),
            build_chunk_anchor("file:///a.zarr", "/p", &[0, 0]),
            build_chunk_anchor("file:///b.zarr", "t", &[0, 0]),
            // Duplicate of the first anchor.
            build_chunk_anchor("file:///a.zarr", "t", &[0, 0]),
        ];
        let plan = LoadPlan::build(uris.iter().map(String::as_str)).unwrap();
        assert_eq!(plan.stores, vec!["file:///a.zarr", "file:///b.zarr"]);
        assert_eq!(
            plan.arrays,
            vec![
                (0, "/t".to_string()),
                (0, "/p".to_string()),
                (1, "/t".to_string())
            ]
        );
        assert_eq!(
            plan.chunks,
            vec![
                (0, vec![0, 0]),
                (0, vec![0, 1]),
                (1, vec![0, 0]),
                (2, vec![0, 0])
            ]
        );
        assert_eq!(plan.request_chunk, vec![0, 1, 2, 3, 0]);
    }

    #[test]
    fn load_plan_rejects_malformed_anchor() {
        let err = LoadPlan::build(["file:///a.zarr"]).unwrap_err();
        assert!(err.to_string().contains("missing"), "{err}");
    }

    #[test]
    fn with_concurrency_clamps_to_at_least_one() {
        assert_eq!(ZarrLoader::new().concurrency(), DEFAULT_LOAD_CONCURRENCY);
        assert_eq!(ZarrLoader::new().with_concurrency(0).concurrency(), 1);
        assert_eq!(ZarrLoader::new().with_concurrency(3).concurrency(), 3);
    }

    #[tokio::test]
    async fn zarr_loader_batches_requests_across_arrays_and_chunks() {
        let tmp = TempDir::new().unwrap();
        let store_uri = build_two_array_zarr(&tmp);
        let view = ViewEntries::identity_for_shape(&[2, 3]);
        let uris = [
            build_chunk_anchor(&store_uri, "temperature", &[0, 0]),
            build_chunk_anchor(&store_uri, "pressure", &[0, 1]),
            build_chunk_anchor(&store_uri, "temperature", &[0, 1]),
            build_chunk_anchor(&store_uri, "pressure", &[0, 0]),
            // Duplicate anchor: must come back with the same bytes, shared.
            build_chunk_anchor(&store_uri, "temperature", &[0, 0]),
        ];
        let reqs: Vec<RasterLoadRequest> = uris
            .iter()
            .map(|uri| RasterLoadRequest {
                uri,
                dim_names: &["y", "x"],
                source_shape: &[2, 3],
                view: &view,
                data_type: BandDataType::UInt8,
            })
            .collect();
        let req_refs: Vec<&RasterLoadRequest> = reqs.iter().collect();
        let expected: [Vec<u8>; 5] = [
            (10..16).collect(),
            (40..46).collect(),
            (20..26).collect(),
            (30..36).collect(),
            (10..16).collect(),
        ];

        // Serial and fanned-out must agree, and both must preserve request order.
        for concurrency in [1, DEFAULT_LOAD_CONCURRENCY] {
            let results = ZarrLoader::new()
                .with_concurrency(concurrency)
                .load(&req_refs)
                .await
                .unwrap();
            assert_eq!(results.len(), 5);
            for (i, (result, want)) in results.iter().zip(&expected).enumerate() {
                assert_eq!(
                    result.bytes.as_slice(),
                    want.as_slice(),
                    "request {i} at concurrency {concurrency}"
                );
            }
            assert_eq!(
                results[0].bytes.as_ptr(),
                results[4].bytes.as_ptr(),
                "duplicate anchors must share one loaded buffer"
            );
        }
    }

    #[tokio::test]
    async fn zarr_loader_batch_fails_when_any_request_dtype_disagrees() {
        let tmp = TempDir::new().unwrap();
        let store_uri = build_two_array_zarr(&tmp);
        let view = ViewEntries::identity_for_shape(&[2, 3]);
        let good = build_chunk_anchor(&store_uri, "temperature", &[0, 0]);
        let bad = build_chunk_anchor(&store_uri, "pressure", &[0, 0]);
        let reqs = [
            RasterLoadRequest {
                uri: &good,
                dim_names: &["y", "x"],
                source_shape: &[2, 3],
                view: &view,
                data_type: BandDataType::UInt8,
            },
            RasterLoadRequest {
                uri: &bad,
                dim_names: &["y", "x"],
                source_shape: &[2, 3],
                view: &view,
                // Array is UInt8 but this band claims Int16.
                data_type: BandDataType::Int16,
            },
        ];
        let err = ZarrLoader::new()
            .load(&[&reqs[0], &reqs[1]])
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("metadata claims") && err.contains("/pressure"),
            "expected the dtype-mismatch diagnostic naming the array, got: {err}"
        );
    }
}
