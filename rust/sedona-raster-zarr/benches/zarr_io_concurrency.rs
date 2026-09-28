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

//! The I/O budget of `ZarrLoader` (`with_concurrency`): how long a fixed
//! set of chunks takes to load as the budget grows, with one caller and
//! with several callers sharing the budget the way DataFusion partitions
//! do. Two stores: the page-cached local filesystem, where reads are cheap
//! and decoding dominates, and an HTTP server that sleeps for a fixed time
//! on every request, a stand-in for object storage where the round trip
//! dominates. The knee of the HTTP curve is what the default budget is
//! chosen from; the local curve shows what that budget costs when there is
//! no latency to hide.
//!
//! Run with `cargo bench -p sedona-raster-zarr --bench zarr_io_concurrency`.
//! `ZARR_BENCH_LATENCY_MS` sets the HTTP store's per-request latency
//! (default 10). The HTTP store needs `python3` on the path; without it that
//! half is skipped.

use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use sedona_raster::raster_loader::{AsyncRasterLoader, RasterLoadRequest};
use sedona_raster::view_entries::ViewEntries;
use sedona_raster_zarr::ZarrLoader;
use sedona_schema::raster::BandDataType;
use tempfile::TempDir;
use zarrs::array::{ArrayBuilder, FillValue, data_type as zarr_dtype};
use zarrs::group::GroupBuilder;
use zarrs_filesystem::FilesystemStore;

const ARRAYS: usize = 4;
const CHUNKS_PER_ARRAY: u64 = 64;
const CHUNK_SIDE: u64 = 128;
const CALLERS: [usize; 2] = [1, 8];
const BUDGETS: [usize; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// A group of `ARRAYS` uint8 arrays, each one row of `CHUNKS_PER_ARRAY`
/// square chunks (256 chunks of 16 KiB), written under `dir/bench.zarr`.
fn build_store(dir: &TempDir) {
    let store_path = dir.path().join("bench.zarr");
    let store = Arc::new(FilesystemStore::new(&store_path).unwrap());
    GroupBuilder::new()
        .build(store.clone(), "/")
        .unwrap()
        .store_metadata()
        .unwrap();
    for a in 0..ARRAYS {
        let array = ArrayBuilder::new(
            vec![CHUNK_SIDE, CHUNK_SIDE * CHUNKS_PER_ARRAY],
            vec![CHUNK_SIDE, CHUNK_SIDE],
            zarr_dtype::uint8(),
            FillValue::from(0u8),
        )
        .build(store.clone(), &format!("/band{a}"))
        .unwrap();
        array.store_metadata().unwrap();
        let chunk = vec![a as u8; (CHUNK_SIDE * CHUNK_SIDE) as usize];
        for j in 0..CHUNKS_PER_ARRAY {
            array.store_chunk(&[0, j], chunk.clone()).unwrap();
        }
    }
}

/// Owned request data, so callers can be spawned as tasks and rebuild the
/// borrowed `RasterLoadRequest`s per call.
struct Owned {
    uris: Vec<String>,
    shape: [i64; 2],
    view: ViewEntries,
}

impl Owned {
    fn new(store_uri: &str) -> Self {
        let shape = [CHUNK_SIDE as i64, CHUNK_SIDE as i64];
        Self {
            uris: (0..ARRAYS)
                .flat_map(|a| (0..CHUNKS_PER_ARRAY).map(move |j| (a, j)))
                .map(|(a, j)| format!("{store_uri}#array=band{a}&chunk=0,{j}"))
                .collect(),
            view: ViewEntries::identity_for_shape(&shape),
            shape,
        }
    }

    async fn load(&self, loader: &ZarrLoader, range: std::ops::Range<usize>) {
        let reqs: Vec<RasterLoadRequest> = self.uris[range]
            .iter()
            .map(|uri| RasterLoadRequest {
                uri,
                dim_names: &["y", "x"],
                source_shape: &self.shape,
                view: &self.view,
                data_type: BandDataType::UInt8,
            })
            .collect();
        let refs: Vec<&RasterLoadRequest> = reqs.iter().collect();
        loader.load(&refs).await.unwrap();
    }
}

/// `callers` tasks, each loading its share of the chunks through clones of
/// one loader, so they share its budget like partitions share the
/// registered loader.
async fn load_all(owned: &Arc<Owned>, loader: &ZarrLoader, callers: usize) {
    let per_caller = owned.uris.len() / callers;
    let tasks: Vec<_> = (0..callers)
        .map(|c| {
            let owned = Arc::clone(owned);
            let loader = loader.clone();
            let range = c * per_caller..(c + 1) * per_caller;
            tokio::spawn(async move { owned.load(&loader, range).await })
        })
        .collect();
    for task in tasks {
        task.await.unwrap();
    }
}

fn sweep(c: &mut Criterion, group_name: &str, store_uri: &str, rt: &tokio::runtime::Runtime) {
    let owned = Arc::new(Owned::new(store_uri));
    let mut group = c.benchmark_group(group_name);
    group.sample_size(10);
    for callers in CALLERS {
        for budget in BUDGETS {
            let loader = ZarrLoader::new().with_concurrency(budget);
            // Open the arrays once so the sweep measures chunk reads only.
            rt.block_on(load_all(&owned, &loader, callers));
            group.bench_with_input(
                BenchmarkId::new(format!("callers{callers}"), budget),
                &budget,
                |b, _| b.iter(|| rt.block_on(load_all(&owned, &loader, callers))),
            );
        }
    }
    group.finish();
}

/// A Python HTTP server over `root` that sleeps `latency` before answering
/// every request, with keep-alive and a deep listen backlog so that only
/// the latency is measured; killed on drop.
struct LatencyServer {
    child: Child,
    port: u16,
}

impl LatencyServer {
    fn start(root: &std::path::Path, latency: Duration) -> Option<Self> {
        let port = TcpListener::bind("127.0.0.1:0")
            .ok()?
            .local_addr()
            .ok()?
            .port();
        let script = r#"
import http.server, sys, time
root, port, latency = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]) / 1000.0
class Handler(http.server.SimpleHTTPRequestHandler):
    # Keep-alive, so the client reuses connections instead of opening one
    # per request; a burst of new connections would overflow the listen
    # backlog and stall on SYN retransmits, which is not what is measured.
    protocol_version = "HTTP/1.1"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=root, **kwargs)
    def do_GET(self):
        time.sleep(latency)
        super().do_GET()
    def do_HEAD(self):
        time.sleep(latency)
        super().do_HEAD()
    def log_message(self, *args):
        pass
class Server(http.server.ThreadingHTTPServer):
    request_queue_size = 1024
Server(("127.0.0.1", port), Handler).serve_forever()
"#;
        let child = Command::new("python3")
            .arg("-c")
            .arg(script)
            .arg(root)
            .arg(port.to_string())
            .arg(latency.as_millis().to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;
        let deadline = Instant::now() + Duration::from_secs(10);
        while TcpStream::connect(("127.0.0.1", port)).is_err() {
            if Instant::now() > deadline {
                return None;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        Some(Self { child, port })
    }
}

impl Drop for LatencyServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn bench_io_budget(c: &mut Criterion) {
    let tmp = TempDir::new().unwrap();
    build_store(&tmp);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(8)
        .enable_all()
        .build()
        .unwrap();

    let local_uri = format!("file://{}", tmp.path().join("bench.zarr").display());
    sweep(c, "zarr_io_budget_local", &local_uri, &rt);

    let latency_ms: u64 = std::env::var("ZARR_BENCH_LATENCY_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);
    match LatencyServer::start(tmp.path(), Duration::from_millis(latency_ms)) {
        Some(server) => {
            let http_uri = format!("http://127.0.0.1:{}/bench.zarr", server.port);
            sweep(
                c,
                &format!("zarr_io_budget_http_{latency_ms}ms"),
                &http_uri,
                &rt,
            );
        }
        None => eprintln!("python3 not available or server did not start; skipping the HTTP store"),
    }
}

criterion_group!(benches, bench_io_budget);
criterion_main!(benches);
