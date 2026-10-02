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

//! GDAL backend implementing [`sedona_raster::raster_loader::AsyncRasterLoader`].
//!
//! Reads OutDb raster bands identified by a `#band=N` URI fragment via
//! GDAL's blocking API. The blocking work runs inside
//! `tokio::task::spawn_blocking` so the caller's async runtime is not
//! stalled. Each read opens its file, reads the requested bands, and closes
//! it again; the loader keeps no datasets open between calls. Repeat loads
//! are served before they reach the loader, by `RS_EnsureLoaded`'s session
//! chunk cache, and GDAL's own `/vsicurl` cache makes reopening a remote
//! file cheap. A per-thread dataset cache here hit rarely (reads land on
//! whichever blocking thread is free), held open file descriptors on every
//! blocking thread the budget brings, and kept about a gigabyte more
//! resident on a 640-COG load, for no measurable gain in wall time.
//!
//! `load` treats its request slice as a batch: requests are grouped by
//! file, and each file is read by its own blocking task, so a batch of
//! COGs on object storage waits on many round trips at once rather than
//! one after another. A file named by several requests (one per band) is
//! opened once and its bands are read in turn by the same task, since a
//! GDAL dataset handle must not cross threads. Files read at once are
//! bounded by the loader's I/O budget ([`GdalLoader::concurrency`]), which
//! every concurrent `load` call on the loader and its clones shares:
//! DataFusion runs one call per partition at once, and the budget is what
//! the storage sees in total, not per partition.
//!
//! ## Cancellation
//!
//! Reads run as a loop of block-height-aligned strips
//! (`band.block_size().1` rows per iteration), with a cooperative
//! cancellation check between strips. When the outer async future is
//! dropped (e.g. a query is cancelled), a [`CancelOnDrop`] guard flips
//! an [`AtomicBool`] shared by every task of the call; the next iteration
//! of each task's loop observes the flag and returns a cancellation error
//! rather than running to completion. Files still waiting for a share of
//! the budget are never started. The same guard fires when one file fails:
//! the call returns that error and the files still being read stop at
//! their next check. A task holds its share of the budget until its
//! blocking work has actually returned, so abandoned reads still count
//! against the budget while they wind down.
//!
//! Cancellation granularity is the source's natural block height:
//!
//! * Strip GeoTIFF: typically 1–64 rows per check (fine-grained).
//! * Tile GeoTIFF (COG): typically 256 rows per check (fine-grained).
//! * PNG/JPEG and similar whole-image-block formats: the first read
//!   forces full decompression in one call; subsequent in-call rows
//!   hit GDAL's block cache. Effectively whole-image cancellation
//!   granularity for these formats — the byte-cap below is the
//!   primary safety net for them.
//!
//! ## Byte cap
//!
//! Requests are pre-validated against [`MAX_OUTDB_LOAD_BYTES`] (4 GiB)
//! before the blocking task is spawned. This catches runaway requests
//! (corrupt metadata, accidentally-huge bands) at the boundary so they
//! can't tie up a blocking-pool thread.
//!
//! Registered against the per-session
//! [`RasterLoaderRegistry`](sedona_raster::raster_loader::RasterLoaderRegistry)
//! under the format key `"gdal"`. The `sedona` crate constructs a
//! [`GdalLoader`] during `SedonaContext` construction and registers
//! it during session bootstrap.

use std::collections::HashMap;
use std::iter::zip;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use arrow_buffer::Buffer;
use arrow_schema::ArrowError;
use async_trait::async_trait;
use datafusion_common::{DataFusionError, Result as DFResult};
use futures::{StreamExt, TryStreamExt, stream};
use sedona_gdal::dataset::Dataset;
use sedona_gdal::raster::rasterband::RasterBand;
use sedona_raster::raster_loader::{AsyncRasterLoader, RasterLoadRequest, RasterLoadResult};
use sedona_raster::traits::{is_spatial_dim_pair, split_outdb_band_fragment};
use sedona_schema::raster::BandDataType;
use tokio::sync::Semaphore;

use crate::gdal_common::{convert_gdal_err, gdal_to_band_data_type, open_gdal_dataset, with_gdal};

/// Diagnostic name for the GDAL raster loader (reported via
/// [`AsyncRasterLoader::name`]). GDAL is a catch-all loader — it doesn't key
/// off a specific `outdb_format` — so this is an identity label, not a
/// dispatch key.
pub const GDAL_FORMAT: &str = "gdal";

/// Maximum bytes a single OutDb load request will produce.
///
/// Requests with `Π source_shape × data_type.byte_size()` greater than
/// this value are rejected before spawning the blocking read task.
/// 4 GiB is intentionally conservative: typical satellite imagery bands
/// (Landsat, Sentinel-2, MODIS) are under 1 GiB; anything larger usually
/// indicates corrupt metadata or an accidentally-huge band claim. If we
/// ever want a tunable here, [`SedonaOptions`] is the natural home for
/// the override.
pub const MAX_OUTDB_LOAD_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Default I/O budget: files read at once across every concurrent `load`
/// call on a loader (all partitions together), not per call.
///
/// Chosen from `benches/gdal_io_concurrency.rs` (64 COGs of 1024 × 1024
/// uint8, one caller and eight callers sharing the budget, Apple M-series,
/// 12 cores) and from 640 public 1113 × 1113 COGs read from S3 over a home
/// connection. Against an HTTP store adding 10 ms to every request, time
/// halves with every doubling of the budget (3.7 s at 1, 437 ms at 8, 112 ms
/// at 32, 63 ms at 64, with one or eight callers alike); the bench has no
/// knee of its own short of running every file at once. On S3 the knee is
/// real: the load part of the query took 169 s at 1, 23 s at 8, 14 s at 16,
/// 10 s at 32 and 9 s at 64 and 128, so past 32 the store, not the budget,
/// is the limit. Against the page-cached local filesystem a larger budget
/// never hurt (80 ms at 1, 12 ms at 8, 9.7 ms at 32, 8.7 ms at 64). With
/// no measured gain past 32, the smaller budget was taken: each file in
/// flight holds a blocking thread and an open dataset. Memory in flight
/// does not grow with the budget: `EnsureLoadedExec` already bounds the
/// bytes one call returns.
pub const DEFAULT_LOAD_CONCURRENCY: usize = 32;

/// GDAL-backed `AsyncRasterLoader`.
///
/// The only state is the I/O budget, which clones share. Datasets are opened
/// per read and closed when the read returns (see the module docs).
#[derive(Debug, Clone)]
pub struct GdalLoader {
    concurrency: usize,
    /// The I/O budget: one permit per file read in flight, shared by every
    /// concurrent `load` call on this loader and its clones.
    permits: Arc<Semaphore>,
    io: Arc<IoCounters>,
}

/// In-flight file reads, for tests and benchmarks.
#[derive(Debug, Default)]
struct IoCounters {
    in_flight: AtomicUsize,
    peak_in_flight: AtomicUsize,
}

impl IoCounters {
    fn enter(&self) {
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak_in_flight.fetch_max(now, Ordering::SeqCst);
    }

    fn exit(&self) {
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

impl Default for GdalLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl GdalLoader {
    pub fn new() -> Self {
        Self {
            concurrency: DEFAULT_LOAD_CONCURRENCY,
            permits: Arc::new(Semaphore::new(DEFAULT_LOAD_CONCURRENCY)),
            io: Arc::new(IoCounters::default()),
        }
    }

    /// The I/O budget: files read at once, counted across every concurrent
    /// `load` call on this loader and its clones. Clamped to at least 1.
    /// Starts a fresh budget, so configure before the loader is shared.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        debug_assert!(
            Arc::strong_count(&self.permits) == 1,
            "configure the I/O budget before sharing the loader"
        );
        self.concurrency = concurrency.max(1);
        self.permits = Arc::new(Semaphore::new(self.concurrency));
        self
    }

    /// The configured I/O budget: the maximum number of files read at once
    /// across all concurrent calls.
    pub fn concurrency(&self) -> usize {
        self.concurrency
    }

    /// The most file reads this loader has had in flight at once, over its
    /// lifetime. For tests and benchmarks.
    pub fn peak_in_flight(&self) -> usize {
        self.io.peak_in_flight.load(Ordering::SeqCst)
    }
}

/// Drop guard that flips an `AtomicBool` when the outer async future
/// is dropped. Paired with the `spawn_blocking` tasks that poll the same
/// flag between unit-of-work iterations: dropping the outer future
/// signals every blocking task of the call to exit at its next
/// checkpoint.
struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

#[async_trait]
impl AsyncRasterLoader for GdalLoader {
    fn name(&self) -> &str {
        GDAL_FORMAT
    }

    /// GDAL is the catch-all byte loader: it attempts any band format,
    /// including the `None` that `RS_FromPath` emits — it opens the file
    /// with GDAL regardless of the declared format. Registered first so
    /// format-specific loaders (e.g. Zarr) registered later win for the
    /// formats they claim.
    fn supports_format(&self, _format: Option<&str>) -> bool {
        true
    }

    async fn load(&self, reqs: &[&RasterLoadRequest]) -> Result<Vec<RasterLoadResult>, ArrowError> {
        let load_requests = reqs
            .iter()
            .map(|req| self.validate_one(req))
            .collect::<Result<Vec<_>, ArrowError>>()?;

        let buffers = self.load_all(load_requests).await?;
        if buffers.len() != reqs.len() {
            return Err(ArrowError::InvalidArgumentError(format!(
                "GDAL loader returned {} buffer(s) for {} request(s)",
                buffers.len(),
                reqs.len()
            )));
        }

        let results = zip(buffers, reqs)
            .map(|(buf, req)| RasterLoadResult::unresolved(buf, req))
            .collect::<Vec<_>>();
        Ok(results)
    }
}

struct OwnedGdalLoadRequest {
    uri: String,
    height: usize,
    width: usize,
    expected_bytes: usize,
    byte_size: usize,
    expected_dtype: BandDataType,
}

impl GdalLoader {
    /// Validate one request and return the information we'll queue onto the worker
    fn validate_one(&self, req: &RasterLoadRequest) -> Result<OwnedGdalLoadRequest, ArrowError> {
        // Validate request shape synchronously, before spawning a blocking
        // task — these are programming errors, no point queueing them
        // onto a worker.
        if req.source_shape.len() != 2 {
            return Err(ArrowError::NotYetImplemented(format!(
                "GDAL raster loader only supports 2-D bands; got source_shape with {} dims",
                req.source_shape.len()
            )));
        }

        if req.dim_names.len() != 2 || !is_spatial_dim_pair(req.dim_names[0], req.dim_names[1]) {
            return Err(ArrowError::InvalidArgumentError(format!(
                "GDAL raster loader requires a 2-D spatial dim pair \
                 ([\"y\", \"x\"], [\"lat\", \"lon\"], or [\"latitude\", \"longitude\"]); got {:?}",
                req.dim_names
            )));
        }

        // The Y-like (row) axis is source_shape[0], the X-like (column) axis
        // is source_shape[1] — guaranteed by the dim-pair check above.
        let height = usize::try_from(req.source_shape[0]).map_err(|_| {
            ArrowError::InvalidArgumentError(format!(
                "GDAL OutDb source_shape[0]={} exceeds usize::MAX",
                req.source_shape[0]
            ))
        })?;

        let width = usize::try_from(req.source_shape[1]).map_err(|_| {
            ArrowError::InvalidArgumentError(format!(
                "GDAL OutDb source_shape[1]={} exceeds usize::MAX",
                req.source_shape[1]
            ))
        })?;

        let byte_size = req.data_type.byte_size();

        // Byte-cap validation: compute Π source_shape × byte_size in u64
        // with checked arithmetic so a hostile request can't wrap to a
        // small accept-value. Reject before allocating.
        let expected_bytes_u64 = (req.source_shape[0] as u64)
            .checked_mul(req.source_shape[1] as u64)
            .and_then(|elems| elems.checked_mul(byte_size as u64))
            .ok_or_else(|| {
                ArrowError::InvalidArgumentError(format!(
                    "GDAL OutDb request byte count overflows u64 for source_shape {:?} × byte_size {}",
                    req.source_shape, byte_size
                ))
            })?;
        if expected_bytes_u64 > MAX_OUTDB_LOAD_BYTES {
            return Err(ArrowError::InvalidArgumentError(format!(
                "GDAL OutDb request exceeds MAX_OUTDB_LOAD_BYTES ({} > {}); \
                 increase the cap or split the band into smaller reads",
                expected_bytes_u64, MAX_OUTDB_LOAD_BYTES
            )));
        }

        Ok(OwnedGdalLoadRequest {
            uri: req.uri.to_string(),
            height,
            width,
            expected_bytes: expected_bytes_u64 as usize,
            byte_size,
            expected_dtype: req.data_type,
        })
    }

    async fn load_all(&self, reqs: Vec<OwnedGdalLoadRequest>) -> Result<Vec<Buffer>, ArrowError> {
        // Cancellation plumbing: the guard lives in this async fn's
        // frame. On normal completion `_guard` drops after every task has
        // returned, flipping the flag on finished work (no-op). When the
        // outer future is dropped mid-await, or when one file fails and
        // `try_collect` returns early, the guard drops and flips the flag;
        // each blocking task still running observes it at its next strip
        // boundary and returns a cancellation error.
        let cancel: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
        let _guard = CancelOnDrop(Arc::clone(&cancel));

        // Group requests by file, keeping first-appearance order, so that a
        // file named once per band is opened once, by one thread.
        let num_reqs = reqs.len();
        let mut files: Vec<Vec<(usize, OwnedGdalLoadRequest)>> = Vec::new();
        let mut file_index: HashMap<String, usize> = HashMap::new();
        for (req_idx, req) in reqs.into_iter().enumerate() {
            let (path, _) = split_outdb_band_fragment(&req.uri)
                .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
            let idx = *file_index.entry(path).or_insert_with(|| {
                files.push(Vec::new());
                files.len() - 1
            });
            files[idx].push((req_idx, req));
        }

        // Read each file on its own blocking task, each holding one permit
        // of the shared I/O budget; `buffer_unordered` merely keeps one call
        // from queueing more futures than the budget could ever run. The
        // permit moves into the blocking closure, so it is released when
        // the read has actually returned, not when its future is dropped.
        // Results complete out of order, so carry the request index and
        // slot them back afterwards. `try_collect` stops at the first error
        // and drops the rest; files not yet started never start.
        let mut reads = Vec::with_capacity(files.len());
        for file_reqs in files {
            let permits = Arc::clone(&self.permits);
            let io = Arc::clone(&self.io);
            let cancel = Arc::clone(&cancel);
            reads.push(async move {
                let permit = permits.acquire_owned().await.map_err(|_| {
                    ArrowError::ExternalError(Box::new(
                        sedona_common::sedona_internal_datafusion_err!(
                            "GDAL loader: I/O budget closed"
                        ),
                    ))
                })?;
                tokio::task::spawn_blocking(move || {
                    let _permit = permit;
                    io.enter();
                    let result = read_file(file_reqs, &cancel);
                    io.exit();
                    result.map_err(|e| ArrowError::ExternalError(Box::new(e)))
                })
                .await
                .map_err(|e| {
                    ArrowError::ExternalError(Box::new(
                        sedona_common::sedona_internal_datafusion_err!(
                            "GDAL raster loader task panicked or was cancelled: {e}"
                        ),
                    ))
                })?
            });
        }
        let read: Vec<Vec<(usize, Buffer)>> = stream::iter(reads)
            .buffer_unordered(self.concurrency)
            .try_collect()
            .await?;

        let mut buffers: Vec<Option<Buffer>> = vec![None; num_reqs];
        for (req_idx, buffer) in read.into_iter().flatten() {
            buffers[req_idx] = Some(buffer);
        }
        buffers
            .into_iter()
            .enumerate()
            .map(|(req_idx, buffer)| {
                buffer.ok_or_else(|| {
                    ArrowError::ExternalError(Box::new(
                        sedona_common::sedona_internal_datafusion_err!(
                            "GDAL loader: request {req_idx} of {num_reqs} was never read"
                        ),
                    ))
                })
            })
            .collect()
    }
}

/// Read every requested band of one file, in request order, on the calling
/// (blocking) thread. Returns each band's bytes tagged with its request
/// index.
fn read_file(
    reqs: Vec<(usize, OwnedGdalLoadRequest)>,
    cancel: &AtomicBool,
) -> DFResult<Vec<(usize, Buffer)>> {
    with_gdal(|gdal| {
        let mut buffers = Vec::with_capacity(reqs.len());
        // The file's bands share one open dataset, dropped (closed) when this
        // call returns. `load_all` groups requests by file, so this opens once
        // per call; the path check keeps it correct if a group ever mixes files.
        let mut open: Option<(String, Dataset)> = None;
        for (req_idx, req) in reqs {
            if cancel.load(Ordering::Acquire) {
                return Err(cancelled_err(0, req.height));
            }

            // `#band=N` fragment, with N defaulting to 1 if absent.
            let (path, band_num) = split_outdb_band_fragment(&req.uri)?;
            let dataset = match &mut open {
                Some((open_path, dataset)) if *open_path == path => &*dataset,
                slot => {
                    let dataset = open_gdal_dataset(gdal, &path, None)?;
                    &slot.insert((path, dataset)).1
                }
            };
            let band = dataset
                .rasterband(band_num as usize)
                .map_err(convert_gdal_err)?;

            // Verify the file's pixel type matches the band metadata's
            // claim BEFORE reading. The bytes-out path doesn't convert;
            // a mismatch would produce a 2x-or-N/2 byte count and the
            // size check in `RS_EnsureLoaded` would mis-blame the
            // loader for size rather than naming the dtype mismatch.
            // Catch it cleanly here.
            let file_dtype = gdal_to_band_data_type(band.band_type())?;
            if file_dtype != req.expected_dtype {
                return sedona_common::sedona_internal_err!(
                    "GDAL OutDb band metadata claims {:?} but file {} band {} is {:?}",
                    req.expected_dtype,
                    req.uri,
                    band_num,
                    file_dtype
                );
            }

            // Pre-allocate the output buffer once; each strip read
            // writes into a contiguous slice.
            let mut output = vec![0u8; req.expected_bytes];
            read_band_blockwise(
                &band,
                &mut output,
                req.width,
                req.height,
                req.byte_size,
                cancel,
            )?;
            buffers.push((req_idx, Buffer::from_vec(output)));
        }
        Ok(buffers)
    })
}

/// Read a band's full extent into `output` in row-major order, looping
/// over block-height-aligned horizontal strips.
///
/// The cancellation flag is checked between strips. Each iteration
/// reads at most `block_h` rows via [`RasterBand::read_into_bytes`]
/// directly into the appropriate slice of `output`. For strip-layout
/// files, each iteration covers exactly one strip; for tile-layout
/// files, each iteration covers one row of tiles. GDAL's internal
/// block cache amortises decompression cost so the per-iteration
/// overhead is small.
///
/// `output` must have length `width * height * byte_size`; assumed by
/// the caller.
fn read_band_blockwise(
    band: &RasterBand<'_>,
    output: &mut [u8],
    width: usize,
    height: usize,
    byte_size: usize,
    cancel: &AtomicBool,
) -> DFResult<()> {
    let row_bytes = width.saturating_mul(byte_size);
    // `block_size().1` is the band's natural strip / tile height. Edge
    // bands sometimes report `0` for degenerate inputs; clamp to >=1
    // so the loop always makes progress.
    let (_block_w, block_h) = band.block_size();
    let block_h = block_h.max(1);

    let mut y_start: usize = 0;
    while y_start < height {
        if cancel.load(Ordering::Acquire) {
            return Err(cancelled_err(y_start, height));
        }
        let chunk_h = (height - y_start).min(block_h);
        let byte_off = y_start.saturating_mul(row_bytes);
        let byte_end = byte_off.saturating_add(chunk_h.saturating_mul(row_bytes));
        // Sanity: should always hold given the caller's pre-allocated
        // output slice; defensive in case of arithmetic surprises.
        if byte_end > output.len() {
            return sedona_common::sedona_internal_err!(
                "GDAL OutDb read range [{}..{}) exceeds output buffer length {}",
                byte_off,
                byte_end,
                output.len()
            );
        }
        band.read_into_bytes(
            (0, y_start as isize),
            (width, chunk_h),
            (width, chunk_h),
            &mut output[byte_off..byte_end],
            None,
        )
        .map_err(convert_gdal_err)?;
        y_start += chunk_h;
    }
    Ok(())
}

fn cancelled_err(y_start: usize, height: usize) -> DataFusionError {
    sedona_common::sedona_internal_datafusion_err!(
        "GDAL OutDb load cancelled at row {y_start} of {height}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gdal_common::with_gdal;
    use crate::gdal_dataset_provider::thread_local_cache;
    use sedona_gdal::raster::types::Buffer as GdalBuffer;
    use sedona_raster::view_entries::ViewEntries;
    use sedona_schema::raster::BandDataType;
    use tempfile::TempDir;

    /// Write a 2-row × 3-col UInt8 GeoTIFF and return its path. Pixels
    /// `0..6` in row-major C-order.
    fn write_uint8_geotiff(dir: &TempDir, name: &str) -> String {
        let path = dir.path().join(name);
        let path_str = path.to_string_lossy().to_string();
        with_gdal(|gdal| {
            let driver = gdal.get_driver_by_name("GTiff").unwrap();
            let dataset = driver
                .create_with_band_type::<u8>(&path_str, 3, 2, 1)
                .unwrap();
            dataset
                .set_geo_transform(&[0.0, 1.0, 0.0, 2.0, 0.0, -1.0])
                .unwrap();
            let band = dataset.rasterband(1).unwrap();
            let mut buffer = GdalBuffer::new((3, 2), (0..6u8).collect::<Vec<_>>());
            band.write((0, 0), (3, 2), &mut buffer).unwrap();
            Ok(())
        })
        .unwrap();
        path_str
    }

    /// Write a `width × height` UInt8 GeoTIFF where pixel `(x, y)` = `(y * width + x) as u8`.
    /// Used to verify block-aligned strip reads assemble identical bytes
    /// to a single bulk read.
    fn write_pattern_geotiff(dir: &TempDir, name: &str, width: usize, height: usize) -> String {
        let path = dir.path().join(name);
        let path_str = path.to_string_lossy().to_string();
        with_gdal(|gdal| {
            let driver = gdal.get_driver_by_name("GTiff").unwrap();
            let dataset = driver
                .create_with_band_type::<u8>(&path_str, width, height, 1)
                .unwrap();
            dataset
                .set_geo_transform(&[0.0, 1.0, 0.0, height as f64, 0.0, -1.0])
                .unwrap();
            let band = dataset.rasterband(1).unwrap();
            let pixels: Vec<u8> = (0..width * height).map(|i| (i % 251) as u8).collect();
            let mut buffer = GdalBuffer::new((width, height), pixels);
            band.write((0, 0), (width, height), &mut buffer).unwrap();
            Ok(())
        })
        .unwrap();
        path_str
    }

    #[tokio::test]
    async fn gdal_loader_reads_2d_uint8_geotiff() {
        let tmp = TempDir::new().unwrap();
        let path = write_uint8_geotiff(&tmp, "fixture.tif");
        let uri = format!("{path}#band=1");

        let loader = GdalLoader::new();
        let req = RasterLoadRequest {
            uri: &uri,
            dim_names: &["y", "x"],
            source_shape: &[2, 3],
            view: &ViewEntries::identity_for_shape(&[2, 3]),
            data_type: BandDataType::UInt8,
        };

        let result = loader.load(&[&req]).await.unwrap();
        assert_eq!(result[0].bytes.len(), 6);
        assert_eq!(result[0].bytes.as_slice(), &[0u8, 1, 2, 3, 4, 5]);
    }

    #[tokio::test]
    async fn gdal_loader_defaults_to_band_1_when_fragment_missing() {
        let tmp = TempDir::new().unwrap();
        let path = write_uint8_geotiff(&tmp, "no_fragment.tif");
        let uri = path;

        let loader = GdalLoader::new();
        let req = RasterLoadRequest {
            uri: &uri,
            dim_names: &["y", "x"],
            source_shape: &[2, 3],
            view: &ViewEntries::identity_for_shape(&[2, 3]),
            data_type: BandDataType::UInt8,
        };
        let result = loader.load(&[&req]).await.unwrap();
        assert_eq!(result[0].bytes.len(), 6);
    }

    #[tokio::test]
    async fn gdal_loader_rejects_non_2d_source_shape() {
        let loader = GdalLoader::new();
        let req = RasterLoadRequest {
            uri: "ignored",
            dim_names: &["t", "y", "x"],
            source_shape: &[2, 3, 4],
            view: &ViewEntries::identity_for_shape(&[2, 3, 4]),
            data_type: BandDataType::UInt8,
        };
        let err = loader.load(&[&req]).await.unwrap_err();
        assert!(
            err.to_string().contains("2-D"),
            "expected 2-D rejection diagnostic, got: {err}"
        );
    }

    #[tokio::test]
    async fn gdal_loader_rejects_unrecognized_dim_names() {
        let loader = GdalLoader::new();
        let req = RasterLoadRequest {
            uri: "ignored",
            dim_names: &["x", "y"], // transposed — not a recognized (y, x) pair
            source_shape: &[2, 3],
            view: &ViewEntries::identity_for_shape(&[2, 3]),
            data_type: BandDataType::UInt8,
        };
        let err = loader.load(&[&req]).await.unwrap_err();
        assert!(
            err.to_string().contains("spatial dim pair"),
            "expected spatial-dim-pair rejection diagnostic, got: {err}"
        );
    }

    #[tokio::test]
    async fn gdal_loader_accepts_latlon_dim_names() {
        let tmp = TempDir::new().unwrap();
        let path = write_uint8_geotiff(&tmp, "latlon.tif");
        let uri = format!("{path}#band=1");

        let loader = GdalLoader::new();
        let req = RasterLoadRequest {
            uri: &uri,
            dim_names: &["lat", "lon"],
            source_shape: &[2, 3],
            view: &ViewEntries::identity_for_shape(&[2, 3]),
            data_type: BandDataType::UInt8,
        };
        // lat/lon is a recognized spatial pair, so the request is accepted and
        // the GeoTIFF is read just like a ["y", "x"] band.
        let result = loader.load(&[&req]).await.unwrap();
        assert_eq!(result[0].bytes.len(), 6);
    }

    #[tokio::test]
    async fn gdal_loader_errors_when_dtype_disagrees_with_file() {
        let tmp = TempDir::new().unwrap();
        let path = write_uint8_geotiff(&tmp, "dtype_mismatch.tif");
        let uri = format!("{path}#band=1");

        let loader = GdalLoader::new();
        let req = RasterLoadRequest {
            uri: &uri,
            dim_names: &["y", "x"],
            source_shape: &[2, 3],
            view: &ViewEntries::identity_for_shape(&[2, 3]), // File is UInt8 but we claim Int16 — should fail with a
            // clear dtype-mismatch message, not garbled bytes.
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
    async fn gdal_loader_errors_on_missing_file() {
        let loader = GdalLoader::new();
        let req = RasterLoadRequest {
            uri: "/nonexistent/path/to/file.tif#band=1",
            dim_names: &["y", "x"],
            source_shape: &[2, 3],
            view: &ViewEntries::identity_for_shape(&[2, 3]),
            data_type: BandDataType::UInt8,
        };
        let err = loader.load(&[&req]).await.unwrap_err();
        // GDAL's "no such file" error message wraps through our convert.
        assert!(err.to_string().to_lowercase().contains("nonexistent"));
    }

    #[tokio::test]
    async fn gdal_loader_errors_on_band_index_out_of_range() {
        let tmp = TempDir::new().unwrap();
        let path = write_uint8_geotiff(&tmp, "oob_band.tif");
        // File has 1 band; ask for band 5.
        let uri = format!("{path}#band=5");

        let loader = GdalLoader::new();
        let req = RasterLoadRequest {
            uri: &uri,
            dim_names: &["y", "x"],
            source_shape: &[2, 3],
            view: &ViewEntries::identity_for_shape(&[2, 3]),
            data_type: BandDataType::UInt8,
        };
        let err = loader.load(&[&req]).await.unwrap_err();
        let msg = err.to_string();
        // GDAL surfaces this as a band-index error; just verify the
        // dispatch went through and the error was propagated, not the
        // exact GDAL phrasing.
        assert!(
            !msg.contains("dim_names") && !msg.contains("2-D"),
            "expected a GDAL-layer error, not request-validation; got: {msg}"
        );
    }

    #[tokio::test]
    async fn gdal_loader_rejects_request_over_byte_cap() {
        let loader = GdalLoader::new();
        // 2^31 elements × 4 bytes = 8 GiB, well over the 4 GiB cap.
        // (Source shape values are u64, so this fits the request
        // struct; only the cap should reject it.)
        let req = RasterLoadRequest {
            uri: "ignored",
            dim_names: &["y", "x"],
            source_shape: &[1 << 16, 1 << 16],
            view: &ViewEntries::identity_for_shape(&[1 << 16, 1 << 16]),
            data_type: BandDataType::Float32,
        };
        let err = loader.load(&[&req]).await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("MAX_OUTDB_LOAD_BYTES"),
            "expected byte-cap diagnostic, got: {msg}"
        );
    }

    #[tokio::test]
    async fn gdal_loader_multi_strip_read_matches_single_call() {
        // 64-row image: at default GTiff strip layout, this produces
        // multiple strips, exercising the block-iter loop.
        let tmp = TempDir::new().unwrap();
        let path = write_pattern_geotiff(&tmp, "multistrip.tif", 16, 64);
        let uri = format!("{path}#band=1");

        let loader = GdalLoader::new();
        let req = RasterLoadRequest {
            uri: &uri,
            dim_names: &["y", "x"],
            source_shape: &[64, 16],
            view: &ViewEntries::identity_for_shape(&[64, 16]),
            data_type: BandDataType::UInt8,
        };
        let result = loader.load(&[&req]).await.unwrap();
        let expected: Vec<u8> = (0..16 * 64).map(|i| (i % 251) as u8).collect();
        assert_eq!(result[0].bytes.as_slice(), expected.as_slice());
    }

    /// Pre-arm the cancellation flag, then drive `read_band_blockwise`
    /// directly against a real band. The loop should bail before
    /// reading anything.
    #[test]
    fn read_band_blockwise_honours_pre_cancelled_flag() {
        let tmp = TempDir::new().unwrap();
        let path = write_pattern_geotiff(&tmp, "cancel.tif", 16, 64);
        with_gdal(|gdal| {
            let cache = thread_local_cache()?;
            let dataset = cache.get_or_create_outdb_source(gdal, &path, None)?;
            let band = dataset.rasterband(1).map_err(convert_gdal_err)?;
            let cancel = AtomicBool::new(true);
            let mut out = vec![0u8; 16 * 64];
            let err = read_band_blockwise(&band, &mut out, 16, 64, 1, &cancel)
                .expect_err("pre-armed cancel flag should short-circuit the loop");
            let msg = err.to_string();
            assert!(
                msg.contains("cancelled"),
                "expected a cancellation diagnostic, got: {msg}"
            );
            // Output buffer was never written into.
            assert!(out.iter().all(|&b| b == 0));
            Ok(())
        })
        .unwrap();
    }

    /// Write a 2 × 3 UInt8 GeoTIFF with two bands, band `b` holding pixels
    /// `10 * b + (0..6)`.
    fn write_two_band_geotiff(dir: &TempDir, name: &str) -> String {
        let path = dir.path().join(name);
        let path_str = path.to_string_lossy().to_string();
        with_gdal(|gdal| {
            let driver = gdal.get_driver_by_name("GTiff").unwrap();
            let dataset = driver
                .create_with_band_type::<u8>(&path_str, 3, 2, 2)
                .unwrap();
            dataset
                .set_geo_transform(&[0.0, 1.0, 0.0, 2.0, 0.0, -1.0])
                .unwrap();
            for b in 1..=2u8 {
                let band = dataset.rasterband(b as usize).unwrap();
                let pixels = (0..6u8).map(|i| 10 * b + i).collect::<Vec<_>>();
                let mut buffer = GdalBuffer::new((3, 2), pixels);
                band.write((0, 0), (3, 2), &mut buffer).unwrap();
            }
            Ok(())
        })
        .unwrap();
        path_str
    }

    /// `n` distinct 2 × 3 UInt8 GeoTIFFs, returned as `#band=1` URIs.
    fn write_many_geotiffs(dir: &TempDir, n: usize) -> Vec<String> {
        (0..n)
            .map(|i| format!("{}#band=1", write_uint8_geotiff(dir, &format!("f{i}.tif"))))
            .collect()
    }

    /// Load every URI as a 2 × 3 UInt8 band in one call.
    async fn load_uris(
        loader: &GdalLoader,
        uris: &[String],
    ) -> Result<Vec<RasterLoadResult>, ArrowError> {
        let view = ViewEntries::identity_for_shape(&[2, 3]);
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
        let refs: Vec<&RasterLoadRequest> = reqs.iter().collect();
        loader.load(&refs).await
    }

    #[test]
    fn with_concurrency_clamps_to_at_least_one() {
        assert_eq!(GdalLoader::new().concurrency(), DEFAULT_LOAD_CONCURRENCY);
        assert_eq!(GdalLoader::new().with_concurrency(0).concurrency(), 1);
        assert_eq!(GdalLoader::new().with_concurrency(3).concurrency(), 3);
        assert_eq!(
            GdalLoader::new()
                .with_concurrency(3)
                .permits
                .available_permits(),
            3
        );
    }

    #[tokio::test]
    async fn gdal_loader_serves_a_batch_of_requests_in_request_order() {
        // RS_EnsureLoaded issues every OutDb band of a batch in one call:
        // results must come back one per request, in request order, and a
        // file that appears twice is read twice without disturbing the
        // others. Distinct shapes make any reordering observable.
        let tmp = TempDir::new().unwrap();
        let a = write_uint8_geotiff(&tmp, "a.tif"); // 2 × 3, pixels 0..6
        let b = write_pattern_geotiff(&tmp, "b.tif", 4, 2); // 2 × 4, pixels 0..8
        let c = write_two_band_geotiff(&tmp, "c.tif"); // 2 × 3, bands 10.. and 20..
        let a_uri = format!("{a}#band=1");
        let b_uri = format!("{b}#band=1");
        let c1_uri = format!("{c}#band=1");
        let c2_uri = format!("{c}#band=2");
        let view_a = ViewEntries::identity_for_shape(&[2, 3]);
        let view_b = ViewEntries::identity_for_shape(&[2, 4]);
        let req = |uri, shape, view| RasterLoadRequest {
            uri,
            dim_names: &["y", "x"],
            source_shape: shape,
            view,
            data_type: BandDataType::UInt8,
        };
        let req_a = req(&a_uri, &[2, 3], &view_a);
        let req_b = req(&b_uri, &[2, 4], &view_b);
        let req_c1 = req(&c1_uri, &[2, 3], &view_a);
        let req_c2 = req(&c2_uri, &[2, 3], &view_a);

        // Serial and fanned-out must agree, and both must preserve request
        // order, including two bands of one file split around other files.
        for concurrency in [1, DEFAULT_LOAD_CONCURRENCY] {
            let loader = GdalLoader::new().with_concurrency(concurrency);
            let results = loader
                .load(&[&req_c2, &req_a, &req_b, &req_c1, &req_a])
                .await
                .unwrap();
            assert_eq!(results.len(), 5);
            assert_eq!(results[0].bytes.as_slice(), &[20u8, 21, 22, 23, 24, 25]);
            assert_eq!(results[1].bytes.as_slice(), &[0u8, 1, 2, 3, 4, 5]);
            assert_eq!(results[2].bytes.as_slice(), &[0u8, 1, 2, 3, 4, 5, 6, 7]);
            assert_eq!(results[3].bytes.as_slice(), &[10u8, 11, 12, 13, 14, 15]);
            assert_eq!(results[4].bytes.as_slice(), &[0u8, 1, 2, 3, 4, 5]);
            assert_eq!(results[2].source_shape, vec![2, 4]);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn gdal_loader_fan_out_preserves_order_across_many_files() {
        let tmp = TempDir::new().unwrap();
        // Distinct contents per file so a misplaced result is observable.
        let uris: Vec<String> = (0..48)
            .map(|i| {
                let path = write_pattern_geotiff(&tmp, &format!("p{i}.tif"), 3, 2 + i);
                format!("{path}#band=1")
            })
            .collect();
        let views: Vec<ViewEntries> = (0..48)
            .map(|i| ViewEntries::identity_for_shape(&[2 + i as i64, 3]))
            .collect();
        let shapes: Vec<[i64; 2]> = (0..48).map(|i| [2 + i as i64, 3]).collect();
        let reqs: Vec<RasterLoadRequest> = (0..48)
            .map(|i| RasterLoadRequest {
                uri: &uris[i],
                dim_names: &["y", "x"],
                source_shape: &shapes[i],
                view: &views[i],
                data_type: BandDataType::UInt8,
            })
            .collect();
        let refs: Vec<&RasterLoadRequest> = reqs.iter().collect();

        let loader = GdalLoader::new().with_concurrency(8);
        let results = loader.load(&refs).await.unwrap();
        for (i, result) in results.iter().enumerate() {
            let expected: Vec<u8> = (0..3 * (2 + i)).map(|p| (p % 251) as u8).collect();
            assert_eq!(result.bytes.as_slice(), expected.as_slice(), "request {i}");
        }
    }

    /// Eight concurrent calls on clones of one loader with a budget of two
    /// never have more than two files being read between them.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_loads_share_one_io_budget() {
        let tmp = TempDir::new().unwrap();
        let uris = Arc::new(write_many_geotiffs(&tmp, 16));
        let loader = GdalLoader::new().with_concurrency(2);

        let mut tasks = Vec::new();
        for _ in 0..8 {
            let loader = loader.clone();
            let uris = Arc::clone(&uris);
            tasks.push(tokio::spawn(async move {
                load_uris(&loader, &uris).await.unwrap().len()
            }));
        }
        for task in tasks {
            assert_eq!(task.await.unwrap(), 16);
        }

        let peak = loader.peak_in_flight();
        assert!((1..=2).contains(&peak), "peak in flight {peak}");
        assert_eq!(loader.permits.available_permits(), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn gdal_loader_batch_fails_when_any_file_fails() {
        // One unreadable file among good ones fails the whole call with that
        // file's error, at any budget, and leaves the budget intact.
        let tmp = TempDir::new().unwrap();
        let mut uris = write_many_geotiffs(&tmp, 12);
        uris.insert(5, "/nonexistent/path/to/file.tif#band=1".to_string());
        for concurrency in [1, 4, DEFAULT_LOAD_CONCURRENCY] {
            let loader = GdalLoader::new().with_concurrency(concurrency);
            let err = load_uris(&loader, &uris).await.unwrap_err();
            assert!(
                err.to_string().to_lowercase().contains("nonexistent"),
                "concurrency {concurrency}: {err}"
            );
            // Tasks still running when the call returned wind down and hand
            // their permits back.
            for _ in 0..200 {
                if loader.permits.available_permits() == concurrency {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            assert_eq!(loader.permits.available_permits(), concurrency);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropping_the_load_future_abandons_files_not_yet_started() {
        // With the whole budget taken, a call's files wait for a permit.
        // Dropping the call (a cancelled query) must not start any of them,
        // and must not leak or consume budget.
        let tmp = TempDir::new().unwrap();
        let uris = write_many_geotiffs(&tmp, 4);
        let loader = GdalLoader::new().with_concurrency(1);
        let held = Arc::clone(&loader.permits).acquire_owned().await.unwrap();

        let timed_out = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            load_uris(&loader, &uris),
        )
        .await;
        assert!(timed_out.is_err(), "load should wait on the held budget");
        assert_eq!(loader.peak_in_flight(), 0, "no file should have started");

        drop(held);
        assert_eq!(loader.permits.available_permits(), 1);
        // The loader is still usable after the abandoned call.
        assert_eq!(load_uris(&loader, &uris).await.unwrap().len(), 4);
    }
}
