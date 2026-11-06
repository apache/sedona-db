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

//! The I/O budget of `GdalLoader` (`with_concurrency`): how long a fixed
//! set of COGs takes to load as the budget grows, with one caller and with
//! several callers sharing the budget the way DataFusion partitions do. Two
//! stores: the page-cached local filesystem, where reads are cheap and
//! decoding dominates, and an HTTP server (read through `/vsicurl/`) that
//! sleeps for a fixed time on every request, a stand-in for object storage
//! where the round trips dominate. The knee of the HTTP curve is what the
//! default budget is chosen from; the local curve shows what that budget
//! costs when there is no latency to hide.
//!
//! Every iteration reads the files under fresh names (hard links in a new
//! directory, made outside the timed section), so GDAL's caches never serve
//! a later iteration from an earlier one: each read is a cold open, as it is
//! for a query that reads each COG once.
//!
//! Run with `cargo bench -p sedona-raster-gdal --bench gdal_io_concurrency`.
//! `GDAL_BENCH_LATENCY_MS` sets the HTTP store's per-request latency
//! (default 10).

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use axum::Router;
use axum::extract::Request;
use axum::middleware::{self, Next};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use sedona_gdal::global::with_global_gdal;
use sedona_gdal::raster::types::{Buffer as GdalBuffer, GdalDataType};
use sedona_raster::raster_loader::{AsyncRasterLoader, RasterLoadRequest};
use sedona_raster::view_entries::ViewEntries;
use sedona_raster_gdal::GdalLoader;
use sedona_schema::raster::BandDataType;
use tempfile::TempDir;
use tower_http::services::ServeDir;

const FILES: usize = 64;
const SIDE: usize = 1024;
const CALLERS: [usize; 2] = [1, 8];
const BUDGETS: [usize; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// One single-band uint8 DEFLATE COG of `SIDE × SIDE` (512-pixel tiles),
/// written to `dir/cog.tif`.
fn build_file(dir: &TempDir) {
    let first = dir.path().join("cog.tif");
    with_global_gdal(|gdal| {
        let mem = gdal
            .create_mem_dataset(SIDE, SIDE, 1, GdalDataType::UInt8)
            .unwrap();
        mem.set_geo_transform(&[0.0, 1.0, 0.0, SIDE as f64, 0.0, -1.0])
            .unwrap();
        // A gradient with some noise, so tiles compress like real data
        // rather than to nothing.
        let pixels: Vec<u8> = (0..SIDE * SIDE)
            .map(|i| ((i % SIDE + i / SIDE) as u8) ^ ((i * 2654435761) >> 28) as u8)
            .collect();
        let mut buffer = GdalBuffer::new((SIDE, SIDE), pixels);
        mem.rasterband(1)
            .unwrap()
            .write((0, 0), (SIDE, SIDE), &mut buffer)
            .unwrap();
        let cog = gdal.get_driver_by_name("COG").unwrap();
        mem.create_copy(&cog, &first.to_string_lossy(), &["COMPRESS=DEFLATE"])
            .unwrap();
    })
    .unwrap();
}

/// A fresh directory `g{generation}` under `root` holding `FILES` hard links
/// to `root/cog.tif`, the previous generation's directory removed.
fn fresh_generation(root: &Path, generation: usize) -> String {
    if generation > 0 {
        let _ = std::fs::remove_dir_all(root.join(format!("g{}", generation - 1)));
    }
    let name = format!("g{generation}");
    std::fs::create_dir(root.join(&name)).unwrap();
    for i in 0..FILES {
        std::fs::hard_link(
            root.join("cog.tif"),
            root.join(&name).join(format!("cog{i}.tif")),
        )
        .unwrap();
    }
    name
}

/// `callers` tasks, each loading its share of the files through clones of
/// one loader, so they share its budget like partitions share the
/// registered loader.
async fn load_all(prefix: &str, generation: &str, loader: &GdalLoader, callers: usize) {
    let per_caller = FILES / callers;
    let tasks: Vec<_> = (0..callers)
        .map(|c| {
            let loader = loader.clone();
            let uris: Vec<String> = (c * per_caller..(c + 1) * per_caller)
                .map(|i| format!("{prefix}/{generation}/cog{i}.tif#band=1"))
                .collect();
            tokio::spawn(async move {
                let shape = [SIDE as i64, SIDE as i64];
                let view = ViewEntries::identity_for_shape(&shape);
                let reqs: Vec<RasterLoadRequest> = uris
                    .iter()
                    .map(|uri| RasterLoadRequest {
                        uri,
                        dim_names: &["y", "x"],
                        source_shape: &shape,
                        view: &view,
                        data_type: BandDataType::UInt8,
                    })
                    .collect();
                let refs: Vec<&RasterLoadRequest> = reqs.iter().collect();
                loader.load(&refs).await.unwrap();
            })
        })
        .collect();
    for task in tasks {
        task.await.unwrap();
    }
}

fn sweep(
    c: &mut Criterion,
    group_name: &str,
    root: &Path,
    prefix: &str,
    rt: &tokio::runtime::Runtime,
    generation: &AtomicUsize,
) {
    let mut group = c.benchmark_group(group_name);
    group.sample_size(10);
    for callers in CALLERS {
        for budget in BUDGETS {
            let loader = GdalLoader::new().with_concurrency(budget);
            group.bench_with_input(
                BenchmarkId::new(format!("callers{callers}"), budget),
                &budget,
                |b, _| {
                    b.iter_batched(
                        || fresh_generation(root, generation.fetch_add(1, Ordering::Relaxed)),
                        |g| rt.block_on(load_all(prefix, &g, &loader, callers)),
                        BatchSize::PerIteration,
                    )
                },
            );
        }
    }
    group.finish();
}

/// An HTTP server over `root` that sleeps `latency` before answering every
/// request. [`ServeDir`] serves `GET`, `HEAD` and byte ranges, which
/// `/vsicurl/` relies on, and hyper keeps connections alive, so only the
/// latency is measured. It runs on a runtime of its own, so it never
/// competes with the loads for the bench runtime's workers. Stopped on drop.
struct LatencyServer {
    port: u16,
    runtime: Option<tokio::runtime::Runtime>,
}

impl LatencyServer {
    fn start(root: &Path, latency: Duration) -> Self {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let listener = runtime
            .block_on(tokio::net::TcpListener::bind("127.0.0.1:0"))
            .unwrap();
        let port = listener.local_addr().unwrap().port();
        let app = Router::new()
            .fallback_service(ServeDir::new(root))
            .layer(middleware::from_fn(
                move |req: Request, next: Next| async move {
                    tokio::time::sleep(latency).await;
                    next.run(req).await
                },
            ));
        runtime.spawn(async move { axum::serve(listener, app).await.unwrap() });
        Self {
            port,
            runtime: Some(runtime),
        }
    }
}

impl Drop for LatencyServer {
    fn drop(&mut self) {
        if let Some(runtime) = self.runtime.take() {
            runtime.shutdown_background();
        }
    }
}

fn bench_io_budget(c: &mut Criterion) {
    let tmp = TempDir::new().unwrap();
    build_file(&tmp);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(8)
        .enable_all()
        .build()
        .unwrap();
    // One counter for both sweeps: they share `tmp`, and each leaves its
    // last generation's directory behind.
    let generation = AtomicUsize::new(0);

    sweep(
        c,
        "gdal_io_budget_local",
        tmp.path(),
        &tmp.path().display().to_string(),
        &rt,
        &generation,
    );

    let latency_ms: u64 = std::env::var("GDAL_BENCH_LATENCY_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);
    let server = LatencyServer::start(tmp.path(), Duration::from_millis(latency_ms));
    let prefix = format!("/vsicurl/http://127.0.0.1:{}", server.port);
    sweep(
        c,
        &format!("gdal_io_budget_http_{latency_ms}ms"),
        tmp.path(),
        &prefix,
        &rt,
        &generation,
    );
}

criterion_group!(benches, bench_io_budget);
criterion_main!(benches);
