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
//! directory, made outside the timed section), so neither GDAL's caches nor
//! the loader's per-thread dataset cache serves a later iteration from an
//! earlier one: each read is a cold open, as it is for a query that reads
//! each COG once.
//!
//! Run with `cargo bench -p sedona-raster-gdal --bench gdal_io_concurrency`.
//! `GDAL_BENCH_LATENCY_MS` sets the HTTP store's per-request latency
//! (default 10). The HTTP store needs `python3` on the path; without it that
//! half is skipped.

use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use sedona_gdal::global::with_global_gdal;
use sedona_gdal::raster::types::{Buffer as GdalBuffer, GdalDataType};
use sedona_raster::raster_loader::{AsyncRasterLoader, RasterLoadRequest};
use sedona_raster::view_entries::ViewEntries;
use sedona_raster_gdal::GdalLoader;
use sedona_schema::raster::BandDataType;
use tempfile::TempDir;

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

/// A Python HTTP server over `root` that sleeps `latency` before answering
/// every request and serves byte ranges (which `/vsicurl/` relies on), with
/// keep-alive and a deep listen backlog so that only the latency is
/// measured; killed on drop.
struct LatencyServer {
    child: Child,
    port: u16,
}

impl LatencyServer {
    fn start(root: &Path, latency: Duration) -> Option<Self> {
        let port = TcpListener::bind("127.0.0.1:0")
            .ok()?
            .local_addr()
            .ok()?
            .port();
        let script = r#"
import http.server, os, re, sys, time
root, port, latency = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]) / 1000.0
class Handler(http.server.BaseHTTPRequestHandler):
    # Keep-alive, so the client reuses connections instead of opening one
    # per request; a burst of new connections would overflow the listen
    # backlog and stall on SYN retransmits, which is not what is measured.
    protocol_version = "HTTP/1.1"
    def _serve(self, body):
        time.sleep(latency)
        path = os.path.join(root, self.path.split("?")[0].lstrip("/"))
        if not os.path.isfile(path):
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        size = os.path.getsize(path)
        m = re.match(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
        start, end = 0, size - 1
        if m:
            start = int(m.group(1))
            end = min(int(m.group(2)) if m.group(2) else size - 1, size - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        if body:
            with open(path, "rb") as f:
                f.seek(start)
                self.wfile.write(f.read(end - start + 1))
    def do_GET(self):
        self._serve(True)
    def do_HEAD(self):
        self._serve(False)
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
    match LatencyServer::start(tmp.path(), Duration::from_millis(latency_ms)) {
        Some(server) => {
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
        None => eprintln!("python3 not available or server did not start; skipping the HTTP store"),
    }
}

criterion_group!(benches, bench_io_budget);
criterion_main!(benches);
