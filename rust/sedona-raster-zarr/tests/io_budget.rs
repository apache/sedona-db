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

//! Measure HTTP requests, not the loader's own permit counters: a codec may
//! fan one subset read out into several concurrent storage operations.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::extract::Request;
use axum::middleware::{Next, from_fn};
use axum::response::Response;
use sedona_raster::raster_loader::{AsyncRasterLoader, RasterLoadRequest};
use sedona_raster::view_entries::{ViewEntries, ViewEntry};
use sedona_raster_zarr::ZarrLoader;
use sedona_schema::raster::BandDataType;
use tempfile::TempDir;
use tower_http::services::ServeDir;
use zarrs::array::{ArrayBuilder, data_type};
use zarrs_filesystem::FilesystemStore;

#[derive(Default)]
struct Reads {
    active: AtomicUsize,
    peak: AtomicUsize,
}

struct ActiveRead(Arc<Reads>);

impl Drop for ActiveRead {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::SeqCst);
    }
}

async fn assert_subset_io_budget(sharded: bool) {
    let tmp = TempDir::new().unwrap();
    let store = Arc::new(FilesystemStore::new(tmp.path()).unwrap());
    let chunk_shape = if sharded { vec![4, 6] } else { vec![2, 3] };
    let mut builder = ArrayBuilder::new(vec![4, 24], chunk_shape, data_type::uint8(), 0u8);
    if sharded {
        builder.subchunk_shape(vec![2, 3]);
    }
    let array = builder.build(store, "/pixels").unwrap();
    array.store_metadata().unwrap();
    let pixels: Vec<u8> = (0..96).collect();
    array
        .store_array_subset(&array.subset_all(), &pixels)
        .unwrap();

    let reads = Arc::new(Reads::default());
    let tracked = reads.clone();
    let router = Router::new()
        .fallback_service(ServeDir::new(tmp.path()))
        .layer(from_fn(move |req: Request, next: Next| {
            let reads = tracked.clone();
            async move {
                if !req.uri().path().contains("/c/") {
                    return next.run(req).await;
                }
                let active = reads.active.fetch_add(1, Ordering::SeqCst) + 1;
                reads.peak.fetch_max(active, Ordering::SeqCst);
                let _active = ActiveRead(reads);
                // Ensure overlapping requests are observable, and keep them
                // counted until their response bodies have been read.
                tokio::time::sleep(Duration::from_millis(2)).await;
                let (parts, body) = next.run(req).await.into_parts();
                let bytes = to_bytes(body, usize::MAX).await.unwrap();
                Response::from_parts(parts, Body::from(bytes))
            }
        }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let uri = format!("http://{}#array=pixels", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    // Also stop the server if an assertion panics.
    let _abort = ServerAbort(server.abort_handle());

    let full = ViewEntries::identity_for_shape(&[4, 24]);
    let crop = ViewEntries::new(vec![
        ViewEntry {
            source_axis: 0,
            start: 1,
            step: 1,
            steps: 2,
        },
        ViewEntry {
            source_axis: 1,
            start: 2,
            step: 1,
            steps: 20,
        },
    ]);
    let reqs: Vec<_> = [&full, &crop]
        .into_iter()
        .map(|view| RasterLoadRequest {
            uri: &uri,
            dim_names: &["y", "x"],
            source_shape: &[4, 24],
            view,
            data_type: BandDataType::UInt8,
        })
        .collect();
    let req_refs: Vec<_> = reqs.iter().collect();
    let cropped: Vec<u8> = pixels[26..46]
        .iter()
        .chain(&pixels[50..70])
        .copied()
        .collect();

    // Small budgets catch zarrs' default minimum of four. Five and seven
    // also catch rounding when it divides work between shards and codecs.
    for concurrency in [1, 2, 3, 5, 7] {
        reads.peak.store(0, Ordering::SeqCst);
        let loader = ZarrLoader::new().with_concurrency(concurrency);
        for _ in 0..2 {
            // Separate calls on clones share a budget, including when the
            // shard indexes are already cached from the previous iteration.
            let clones = [loader.clone(), loader.clone()];
            let results =
                futures::future::join_all(clones.iter().map(|loader| loader.load(&req_refs))).await;
            for results in results {
                let results = results.unwrap();
                assert_eq!(results[0].bytes.as_slice(), pixels);
                assert_eq!(results[1].bytes.as_slice(), cropped);
            }
        }
        assert_eq!(reads.active.load(Ordering::SeqCst), 0);
        let peak = reads.peak.load(Ordering::SeqCst);
        assert!(
            peak > 0 && peak <= concurrency,
            "sharded={sharded}, budget={concurrency}, actual HTTP peak={peak}"
        );
        if concurrency > 1 {
            assert!(peak > 1, "subset reads should still run concurrently");
        }
    }
}

struct ServerAbort(tokio::task::AbortHandle);

impl Drop for ServerAbort {
    fn drop(&mut self) {
        self.0.abort();
    }
}

#[tokio::test]
async fn full_array_and_crop_reads_respect_io_budget() {
    assert_subset_io_budget(false).await;
}

#[tokio::test]
async fn sharded_full_array_and_crop_reads_respect_io_budget() {
    assert_subset_io_budget(true).await;
}
