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
//! Reads OutDb raster bands on the `sedona.gdal` container codec: one
//! whole-object chunk per band, decoded by GDAL for the source band named in
//! the codec's configuration. The blocking work runs on the runtime's blocking
//! pool, as `SpawnedTask`s, so the caller's async runtime is not stalled. Each read opens its file, reads the requested bands, and closes
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
//! bounded by the loader's [`RasterIoBudget`], which every concurrent `load`
//! call on the loader and its clones shares: DataFusion runs one call per
//! partition at once, and the budget is what the storage sees in total, not
//! per partition. A session gives its loader the session's budget, which
//! `RS_FromPath`'s file opens draw on as well, so the cap is on pixel reads
//! and header opens together (`sedona.raster.io_concurrency`).
//!
//! Each blocking task waits for its permit on its own thread; the call never
//! waits for the budget as a future (the budget has no async acquire; see
//! `sedona_raster::io_budget`). The call spawns a file's task only once the
//! previous one has its permit, so it keeps at most one blocking thread
//! waiting for the budget. Results are placed by request index as the files
//! finish, and taken in file order, so a failing call reports its first
//! failing file.
//!
//! ## Cancellation
//!
//! Reads run as a loop of block-height-aligned strips
//! (`band.block_size().1` rows per iteration), with a cooperative
//! cancellation check between strips. When the outer async future is
//! dropped (e.g. a query is cancelled), a [`CancelOnDrop`] guard flips
//! an [`AtomicBool`] shared by every task of the call; the next iteration
//! of each task's loop observes the flag and returns a cancellation error
//! rather than running to completion. Files not yet spawned are never
//! started, a spawned task that has not begun is aborted with the call's
//! `SpawnedTask`s, and one still waiting for its permit returns as soon as
//! it gets one, without opening its file. The same guard fires when one file fails:
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
use datafusion_common_runtime::SpawnedTask;
use futures::StreamExt;
use futures::stream::FuturesOrdered;
use sedona_gdal::raster::rasterband::RasterBand;
use sedona_raster::chunks::{GDAL_CODEC, GdalCodec};
use sedona_raster::io_budget::RasterIoBudget;
use sedona_raster::raster_loader::{AsyncRasterLoader, RasterLoadRequest, RasterLoadResult};
use sedona_raster::traits::is_spatial_dim_pair;
use sedona_schema::raster::BandDataType;
use tokio::sync::oneshot;

use crate::gdal_common::{convert_gdal_err, gdal_to_band_data_type, open_gdal_dataset, with_gdal};

/// Diagnostic name for the GDAL raster loader (reported via
/// [`AsyncRasterLoader::name`]). The loader dispatches on the
/// [`GDAL_CODEC`] codec name, so this is an identity label, not a dispatch
/// key.
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

/// GDAL-backed `AsyncRasterLoader`.
///
/// The only state is the I/O budget, which clones share. Datasets are opened
/// per read and closed when the read returns (see the module docs).
#[derive(Debug, Clone)]
pub struct GdalLoader {
    /// The I/O budget: one permit per file read in flight, shared by every
    /// concurrent `load` call on this loader and its clones, and by whatever
    /// else the budget was handed to.
    budget: RasterIoBudget,
    io: Arc<IoCounters>,
}

/// In-flight file reads, for tests and benchmarks.
#[derive(Debug, Default)]
struct IoCounters {
    in_flight: AtomicUsize,
    peak_in_flight: AtomicUsize,
}

impl IoCounters {
    /// Count one read as in flight until the returned guard drops, including
    /// when the read panics.
    fn enter(&self) -> InFlightRead<'_> {
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak_in_flight.fetch_max(now, Ordering::SeqCst);
        InFlightRead(self)
    }
}

struct InFlightRead<'a>(&'a IoCounters);

impl Drop for InFlightRead<'_> {
    fn drop(&mut self) {
        self.0.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

impl Default for GdalLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl GdalLoader {
    /// A loader with a budget of its own, of
    /// [`DEFAULT_RASTER_IO_CONCURRENCY`](sedona_raster::io_budget::DEFAULT_RASTER_IO_CONCURRENCY).
    /// A session uses [`Self::with_budget`] instead.
    pub fn new() -> Self {
        Self {
            budget: RasterIoBudget::default(),
            io: Arc::new(IoCounters::default()),
        }
    }

    /// Draw on `budget`, typically the session's, shared with the session's
    /// other blocking raster I/O.
    pub fn with_budget(mut self, budget: RasterIoBudget) -> Self {
        self.budget = budget;
        self
    }

    /// Draw on a fresh budget of `concurrency` files at once (clamped to at
    /// least 1). For tests and benchmarks.
    pub fn with_concurrency(self, concurrency: usize) -> Self {
        self.with_budget(RasterIoBudget::new(concurrency))
    }

    /// The budget this loader draws on.
    pub fn budget(&self) -> &RasterIoBudget {
        &self.budget
    }

    /// The budget's current limit: the most files read at once across all
    /// of the budget's users.
    pub fn concurrency(&self) -> usize {
        self.budget.limit()
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

    /// GDAL decodes exactly one pipeline: the single [`GDAL_CODEC`] container
    /// codec that `RS_FromPath` writes. Every other pipeline (Zarr's, or a
    /// standard chunk pipeline) belongs to another loader; nothing falls
    /// through to GDAL by default.
    fn supports(&self, codec_names: &[&str]) -> bool {
        codec_names == [GDAL_CODEC]
    }

    async fn load(&self, reqs: &[&RasterLoadRequest]) -> Result<Vec<RasterLoadResult>, ArrowError> {
        let load_requests = reqs
            .iter()
            .map(|req| self.validate_one(req))
            .collect::<Result<Vec<_>, ArrowError>>()?;

        let buffers = self.load_all(load_requests).await?;
        let results = zip(buffers, reqs)
            .map(|(buf, req)| RasterLoadResult::unresolved(buf, req))
            .collect::<Vec<_>>();
        Ok(results)
    }
}

#[derive(Debug)]
struct OwnedGdalLoadRequest {
    /// The file to open: the band's one whole-object chunk.
    path: String,
    /// The 1-based source band, from the codec's configuration.
    band: usize,
    /// GDAL open options from the codec's configuration; empty for none.
    open_options: Vec<String>,
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

        // The file is the band's one whole-object chunk; the source band and
        // open options are the GDAL codec's configuration.
        let path = req.single_chunk_uri()?.to_string();
        let codec = GdalCodec::parse(req.codecs)
            .map_err(|e| ArrowError::ExternalError(Box::new(e)))?
            .ok_or_else(|| {
                ArrowError::InvalidArgumentError(format!(
                    "GDAL raster loader decodes the {GDAL_CODEC} codec only; got codecs {}",
                    req.codecs
                ))
            })?;
        if let Some(subdataset) = &codec.subdataset {
            return Err(ArrowError::NotYetImplemented(format!(
                "the {GDAL_CODEC} codec's subdataset ({subdataset:?}) is not supported yet"
            )));
        }

        Ok(OwnedGdalLoadRequest {
            path,
            band: codec.band as usize,
            open_options: codec.open_options,
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
        // outer future is dropped mid-await, or when a file fails and the
        // call returns early, the guard drops and flips the flag; each
        // blocking task still running observes it at its next strip
        // boundary, and each still waiting for budget when it gets a permit,
        // and returns a cancellation error.
        let cancel: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
        let _guard = CancelOnDrop(Arc::clone(&cancel));

        // Group requests by file (and open options, which change how a file
        // opens), keeping first-appearance order, so that a file named once
        // per band is opened once, by one thread.
        let num_reqs = reqs.len();
        let mut files: Vec<Vec<(usize, OwnedGdalLoadRequest)>> = Vec::new();
        let mut file_index: HashMap<(String, Vec<String>), usize> = HashMap::new();
        for (req_idx, req) in reqs.into_iter().enumerate() {
            let idx = *file_index
                .entry((req.path.clone(), req.open_options.clone()))
                .or_insert_with(|| {
                    files.push(Vec::new());
                    files.len() - 1
                });
            files[idx].push((req_idx, req));
        }

        // Each file is read by its own blocking task, which waits for its
        // permit of the shared I/O budget on its own thread and holds it
        // until the read has returned. Nothing here waits for the budget as
        // a future: the budget has no async acquire (see
        // `sedona_raster::io_budget`), so a task that polls a pending load
        // and then blocks its thread in `RS_FromPath` cannot be in line for
        // a slot it would never take. A task is spawned only once the
        // previous one has its permit, so a call keeps at most one thread
        // waiting for the budget, and as many reading as the budget lets it.
        //
        // Each file's buffers go into their requests' slots as the file
        // finishes. Results are taken in file order, so a failing call
        // reports the failure of its first failing file, whatever order the
        // reads end in. Returning early drops the rest: a task that has not
        // started is aborted with its `SpawnedTask`, reads in flight stop at
        // their next strip, and a task still waiting returns as soon as it
        // gets a permit, without opening anything.
        let mut files = files.into_iter();
        let mut reads = FuturesOrdered::new();
        let mut waiting: Option<oneshot::Receiver<()>> = None;
        let mut buffers: Vec<Option<Buffer>> = vec![None; num_reqs];
        loop {
            if waiting.is_none()
                && let Some(file_reqs) = files.next()
            {
                let (got_permit, permit_signal) = oneshot::channel();
                let budget = self.budget.clone();
                let io = Arc::clone(&self.io);
                let cancel = Arc::clone(&cancel);
                reads.push_back(SpawnedTask::spawn_blocking(move || {
                    let _permit = budget.acquire_blocking();
                    // The call may be gone; it no longer listens.
                    let _ = got_permit.send(());
                    if cancel.load(Ordering::Acquire) {
                        return Err(cancelled_err(0, 0));
                    }
                    let _in_flight = io.enter();
                    read_file(file_reqs, &cancel)
                }));
                waiting = Some(permit_signal);
            }
            tokio::select! {
                read = reads.next() => match read {
                    Some(read) => {
                        let read = read.map_err(|e| {
                            ArrowError::ExternalError(Box::new(
                                sedona_common::sedona_internal_datafusion_err!(
                                    "GDAL raster loader task panicked: {e}"
                                ),
                            ))
                        })?;
                        for (req_idx, buffer) in
                            read.map_err(|e| ArrowError::ExternalError(Box::new(e)))?
                        {
                            buffers[req_idx] = Some(buffer);
                        }
                    }
                    // Every task spawned so far has finished, so the one
                    // that was waiting got its permit: start the next file,
                    // if any.
                    None => {
                        if waiting.take().is_none() {
                            break;
                        }
                    }
                },
                // Resolves when the waiting task gets its permit, or panics.
                _ = async { waiting.as_mut().expect("guarded by the precondition").await },
                    if waiting.is_some() => waiting = None,
            }
        }

        // Every request belongs to exactly one file, and a file's read
        // returns a buffer for each of its requests.
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
    let Some((_, first)) = reqs.first() else {
        return Ok(Vec::new());
    };
    with_gdal(|gdal| {
        // The file's bands share one open dataset, closed when this returns.
        let open_options: Vec<&str> = first.open_options.iter().map(String::as_str).collect();
        let dataset = open_gdal_dataset(
            gdal,
            &first.path,
            (!open_options.is_empty()).then_some(open_options.as_slice()),
        )?;
        let mut buffers = Vec::with_capacity(reqs.len());
        for (req_idx, req) in &reqs {
            if cancel.load(Ordering::Acquire) {
                return Err(cancelled_err(0, req.height));
            }
            let band = dataset.rasterband(req.band).map_err(convert_gdal_err)?;

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
                    req.path,
                    req.band,
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
            buffers.push((*req_idx, Buffer::from_vec(output)));
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
        // Also stop at process exit, which waits for in-flight GDAL calls.
        if cancel.load(Ordering::Acquire) || sedona_gdal::global::is_gdal_exiting() {
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
    use sedona_gdal::raster::types::Buffer as GdalBuffer;
    use sedona_raster::chunks::ChunkRef;
    use sedona_raster::io_budget::DEFAULT_RASTER_IO_CONCURRENCY;
    use sedona_raster::view_entries::ViewEntries;
    use sedona_schema::raster::BandDataType;
    use tempfile::TempDir;

    /// Bind `$req` to a request for one whole-object chunk at `$path` on the
    /// GDAL codec for source band `$band`, over an identity view. The
    /// request's backing storage lands in the enclosing scope, so `$req`
    /// outlives the statement.
    macro_rules! gdal_request {
        ($req:ident, $path:expr, $band:expr, $dims:expr, $shape:expr, $dtype:expr) => {
            let chunk = [Some(ChunkRef::whole($path))];
            let codecs = GdalCodec::new($band).to_json();
            let dims: &[&str] = &$dims;
            let shape: &[i64] = &$shape;
            let view = ViewEntries::identity_for_shape(shape);
            let $req = RasterLoadRequest {
                chunks: &chunk,
                chunk_shape: shape,
                codecs: &codecs,
                dim_names: dims,
                source_shape: shape,
                view: &view,
                data_type: $dtype,
            };
        };
    }

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

        let loader = GdalLoader::new();
        gdal_request!(req, &path, 1, ["y", "x"], [2, 3], BandDataType::UInt8);

        let result = loader.load(&[&req]).await.unwrap();
        assert_eq!(result[0].bytes.len(), 6);
        assert_eq!(result[0].bytes.as_slice(), &[0u8, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn gdal_loader_supports_only_the_gdal_codec() {
        // No catch-all any more: the loader claims exactly the one container
        // codec, so nothing falls through to GDAL by accident.
        let loader = GdalLoader::new();
        assert!(loader.supports(&[GDAL_CODEC]));
        assert!(!loader.supports(&[]));
        assert!(!loader.supports(&["bytes"]));
        assert!(!loader.supports(&["bytes", "zstd"]));
        assert!(!loader.supports(&[GDAL_CODEC, "zstd"]));
        assert!(!loader.supports(&["sedona.zarr"]));
    }

    #[tokio::test]
    async fn gdal_loader_rejects_a_pipeline_that_is_not_the_gdal_codec() {
        // Dispatch should never send such a band here, but a direct caller
        // gets a clear error naming the codec rather than a GDAL open.
        let tmp = TempDir::new().unwrap();
        let path = write_uint8_geotiff(&tmp, "other_codec.tif");
        let chunk = [Some(ChunkRef::whole(&path))];
        let shape = [2i64, 3];
        let view = ViewEntries::identity_for_shape(&shape);
        let req = RasterLoadRequest {
            chunks: &chunk,
            chunk_shape: &shape,
            codecs: r#"[{"name":"bytes","configuration":{"endian":"little"}}]"#,
            dim_names: &["y", "x"],
            source_shape: &shape,
            view: &view,
            data_type: BandDataType::UInt8,
        };
        let err = GdalLoader::new().load(&[&req]).await.unwrap_err();
        assert!(
            err.to_string().contains(GDAL_CODEC),
            "expected the codec name in the diagnostic, got: {err}"
        );
    }

    #[tokio::test]
    async fn gdal_loader_rejects_multi_chunk_and_byte_range_bands() {
        // GDAL decodes whole objects: a band split into several chunks, or a
        // byte range of a file, is the generic loader's business.
        let tmp = TempDir::new().unwrap();
        let path = write_uint8_geotiff(&tmp, "tiles.tif");
        let codecs = GdalCodec::new(1).to_json();
        let shape = [2i64, 3];
        let view = ViewEntries::identity_for_shape(&shape);

        let two = [Some(ChunkRef::whole(&path)), Some(ChunkRef::whole(&path))];
        let req = RasterLoadRequest {
            chunks: &two,
            chunk_shape: &[1, 3],
            codecs: &codecs,
            dim_names: &["y", "x"],
            source_shape: &shape,
            view: &view,
            data_type: BandDataType::UInt8,
        };
        let err = GdalLoader::new().load(&[&req]).await.unwrap_err();
        assert!(
            err.to_string().contains("single-chunk"),
            "expected a single-chunk diagnostic, got: {err}"
        );

        let range = [Some(ChunkRef::range(&path, 0, 6))];
        let req = RasterLoadRequest {
            chunks: &range,
            chunk_shape: &shape,
            codecs: &codecs,
            dim_names: &["y", "x"],
            source_shape: &shape,
            view: &view,
            data_type: BandDataType::UInt8,
        };
        let err = GdalLoader::new().load(&[&req]).await.unwrap_err();
        assert!(
            err.to_string().contains("whole objects only"),
            "expected a whole-object diagnostic, got: {err}"
        );
    }

    #[test]
    fn validate_one_carries_the_codec_configuration() {
        // The source band and open options ride along into the owned
        // request; a subdataset is refused until GDAL container formats are
        // supported.
        let chunk = [Some(ChunkRef::whole("/tmp/x.tif"))];
        let shape = [2i64, 3];
        let view = ViewEntries::identity_for_shape(&shape);
        let mut codec = GdalCodec::new(3).with_driver("GTiff");
        codec.open_options = vec!["NUM_THREADS=2".to_string()];
        let codecs = codec.to_json();
        let req = RasterLoadRequest {
            chunks: &chunk,
            chunk_shape: &shape,
            codecs: &codecs,
            dim_names: &["y", "x"],
            source_shape: &shape,
            view: &view,
            data_type: BandDataType::UInt8,
        };
        let owned = GdalLoader::new().validate_one(&req).unwrap();
        assert_eq!(owned.path, "/tmp/x.tif");
        assert_eq!(owned.band, 3);
        assert_eq!(owned.open_options, vec!["NUM_THREADS=2".to_string()]);

        codec.subdataset = Some("tas".to_string());
        let codecs = codec.to_json();
        let req = RasterLoadRequest {
            codecs: &codecs,
            ..req
        };
        let err = GdalLoader::new().validate_one(&req).unwrap_err();
        assert!(
            err.to_string().contains("subdataset"),
            "expected a subdataset diagnostic, got: {err}"
        );
    }

    #[tokio::test]
    async fn gdal_loader_rejects_non_2d_source_shape() {
        let loader = GdalLoader::new();
        gdal_request!(
            req,
            "ignored",
            1,
            ["t", "y", "x"],
            [2, 3, 4],
            BandDataType::UInt8
        );
        let err = loader.load(&[&req]).await.unwrap_err();
        assert!(
            err.to_string().contains("2-D"),
            "expected 2-D rejection diagnostic, got: {err}"
        );
    }

    #[tokio::test]
    async fn gdal_loader_rejects_unrecognized_dim_names() {
        let loader = GdalLoader::new();
        // transposed — not a recognized (y, x) pair
        gdal_request!(req, "ignored", 1, ["x", "y"], [2, 3], BandDataType::UInt8);
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

        let loader = GdalLoader::new();
        gdal_request!(req, &path, 1, ["lat", "lon"], [2, 3], BandDataType::UInt8);
        // lat/lon is a recognized spatial pair, so the request is accepted and
        // the GeoTIFF is read just like a ["y", "x"] band.
        let result = loader.load(&[&req]).await.unwrap();
        assert_eq!(result[0].bytes.len(), 6);
    }

    #[tokio::test]
    async fn gdal_loader_errors_when_dtype_disagrees_with_file() {
        let tmp = TempDir::new().unwrap();
        let path = write_uint8_geotiff(&tmp, "dtype_mismatch.tif");

        let loader = GdalLoader::new();
        // File is UInt8 but we claim Int16 — should fail with a clear
        // dtype-mismatch message, not garbled bytes.
        gdal_request!(req, &path, 1, ["y", "x"], [2, 3], BandDataType::Int16);
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
        gdal_request!(
            req,
            "/nonexistent/path/to/file.tif",
            1,
            ["y", "x"],
            [2, 3],
            BandDataType::UInt8
        );
        let err = loader.load(&[&req]).await.unwrap_err();
        // GDAL's "no such file" error message wraps through our convert.
        assert!(err.to_string().to_lowercase().contains("nonexistent"));
    }

    #[tokio::test]
    async fn gdal_loader_errors_on_band_index_out_of_range() {
        let tmp = TempDir::new().unwrap();
        let path = write_uint8_geotiff(&tmp, "oob_band.tif");
        // File has 1 band; the codec asks for band 5.
        let loader = GdalLoader::new();
        gdal_request!(req, &path, 5, ["y", "x"], [2, 3], BandDataType::UInt8);
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
        gdal_request!(
            req,
            "ignored",
            1,
            ["y", "x"],
            [1 << 16, 1 << 16],
            BandDataType::Float32
        );
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

        let loader = GdalLoader::new();
        gdal_request!(req, &path, 1, ["y", "x"], [64, 16], BandDataType::UInt8);
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
            let dataset = open_gdal_dataset(gdal, &path, None)?;
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

    /// `n` distinct 2 × 3 UInt8 GeoTIFFs, returned as their paths.
    fn write_many_geotiffs(dir: &TempDir, n: usize) -> Vec<String> {
        (0..n)
            .map(|i| write_uint8_geotiff(dir, &format!("f{i}.tif")))
            .collect()
    }

    /// Load every path's band 1 as a 2 × 3 UInt8 band in one call.
    async fn load_paths(
        loader: &GdalLoader,
        paths: &[String],
    ) -> Result<Vec<RasterLoadResult>, ArrowError> {
        let view = ViewEntries::identity_for_shape(&[2, 3]);
        let codecs = GdalCodec::new(1).to_json();
        let chunks: Vec<[Option<ChunkRef<'_>>; 1]> = paths
            .iter()
            .map(|path| [Some(ChunkRef::whole(path))])
            .collect();
        let reqs: Vec<RasterLoadRequest> = chunks
            .iter()
            .map(|chunk| RasterLoadRequest {
                chunks: chunk,
                chunk_shape: &[2, 3],
                codecs: &codecs,
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
        assert_eq!(
            GdalLoader::new().concurrency(),
            DEFAULT_RASTER_IO_CONCURRENCY
        );
        assert_eq!(GdalLoader::new().with_concurrency(0).concurrency(), 1);
        assert_eq!(GdalLoader::new().with_concurrency(3).concurrency(), 3);
        assert_eq!(
            GdalLoader::new().with_concurrency(3).budget().available(),
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
        let a_chunk = [Some(ChunkRef::whole(&a))];
        let b_chunk = [Some(ChunkRef::whole(&b))];
        let c_chunk = [Some(ChunkRef::whole(&c))];
        let band1 = GdalCodec::new(1).to_json();
        let band2 = GdalCodec::new(2).to_json();
        let view_a = ViewEntries::identity_for_shape(&[2, 3]);
        let view_b = ViewEntries::identity_for_shape(&[2, 4]);
        let req = |chunks, codecs, shape, view| RasterLoadRequest {
            chunks,
            chunk_shape: shape,
            codecs,
            dim_names: &["y", "x"],
            source_shape: shape,
            view,
            data_type: BandDataType::UInt8,
        };
        let req_a = req(&a_chunk, &band1, &[2, 3], &view_a);
        let req_b = req(&b_chunk, &band1, &[2, 4], &view_b);
        let req_c1 = req(&c_chunk, &band1, &[2, 3], &view_a);
        let req_c2 = req(&c_chunk, &band2, &[2, 3], &view_a);

        // Serial and fanned-out must agree, and both must preserve request
        // order, including two bands of one file split around other files.
        for concurrency in [1, DEFAULT_RASTER_IO_CONCURRENCY] {
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
        let paths: Vec<String> = (0..48)
            .map(|i| write_pattern_geotiff(&tmp, &format!("p{i}.tif"), 3, 2 + i))
            .collect();
        let chunks: Vec<[Option<ChunkRef<'_>>; 1]> = paths
            .iter()
            .map(|path| [Some(ChunkRef::whole(path))])
            .collect();
        let codecs = GdalCodec::new(1).to_json();
        let views: Vec<ViewEntries> = (0..48)
            .map(|i| ViewEntries::identity_for_shape(&[2 + i as i64, 3]))
            .collect();
        let shapes: Vec<[i64; 2]> = (0..48).map(|i| [2 + i as i64, 3]).collect();
        let reqs: Vec<RasterLoadRequest> = (0..48)
            .map(|i| RasterLoadRequest {
                chunks: &chunks[i],
                chunk_shape: &shapes[i],
                codecs: &codecs,
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
        let paths = Arc::new(write_many_geotiffs(&tmp, 16));
        let loader = GdalLoader::new().with_concurrency(2);

        let mut tasks = Vec::new();
        for _ in 0..8 {
            let loader = loader.clone();
            let paths = Arc::clone(&paths);
            tasks.push(tokio::spawn(async move {
                load_paths(&loader, &paths).await.unwrap().len()
            }));
        }
        for task in tasks {
            assert_eq!(task.await.unwrap(), 16);
        }

        let peak = loader.peak_in_flight();
        assert!((1..=2).contains(&peak), "peak in flight {peak}");
        assert_eq!(loader.budget().available(), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn gdal_loader_batch_fails_when_any_file_fails() {
        // One unreadable file among good ones fails the whole call with that
        // file's error, at any budget, and leaves the budget intact.
        let tmp = TempDir::new().unwrap();
        let mut paths = write_many_geotiffs(&tmp, 12);
        paths.insert(5, "/nonexistent/path/to/file.tif".to_string());
        for concurrency in [1, 4, DEFAULT_RASTER_IO_CONCURRENCY] {
            let loader = GdalLoader::new().with_concurrency(concurrency);
            let err = load_paths(&loader, &paths).await.unwrap_err();
            assert!(
                err.to_string().to_lowercase().contains("nonexistent"),
                "concurrency {concurrency}: {err}"
            );
            // Tasks still running when the call returned wind down and hand
            // their permits back.
            for _ in 0..200 {
                if loader.budget().available() == concurrency {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            assert_eq!(loader.budget().available(), concurrency);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropping_the_load_future_abandons_files_not_yet_started() {
        // With the whole budget taken, a call's files wait for a permit.
        // Dropping the call (a cancelled query) must not start any of them,
        // and must not leak or consume budget: the one task still waiting
        // takes the permit when it frees, sees the call is gone, and gives
        // it back without opening anything.
        let tmp = TempDir::new().unwrap();
        let paths = write_many_geotiffs(&tmp, 4);
        let loader = GdalLoader::new().with_concurrency(1);
        let held = loader.budget().try_acquire().unwrap();

        let timed_out = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            load_paths(&loader, &paths),
        )
        .await;
        assert!(timed_out.is_err(), "load should wait on the held budget");
        assert_eq!(loader.peak_in_flight(), 0, "no file should have started");

        drop(held);
        for _ in 0..200 {
            if loader.budget().available() == 1 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert_eq!(loader.budget().available(), 1);
        assert_eq!(loader.budget().in_use(), 0);
        assert_eq!(loader.peak_in_flight(), 0, "no file should have started");
        // The loader is still usable after the abandoned call.
        assert_eq!(load_paths(&loader, &paths).await.unwrap().len(), 4);
    }

    /// Pixel reads and `RS_FromPath` opens drawing on one session budget
    /// never have more files in flight between them than the budget allows.
    /// At a budget of 1 this also shows that neither waits for a permit while
    /// it holds one, which would deadlock the two against each other.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn loader_reads_and_frompath_opens_share_one_budget() {
        use crate::rs_frompath::RsFromPath;
        use arrow_array::StringArray;
        use arrow_schema::DataType;
        use datafusion_common::config::ConfigOptions;
        use datafusion_expr::ColumnarValue;
        use sedona_common::option::SedonaOptions;
        use sedona_expr::scalar_udf::SedonaScalarKernel;
        use sedona_raster::raster_loader::{
            RasterLoaderConfig, RasterLoaderRegistry, io_budget_from_config,
        };
        use sedona_schema::datatypes::SedonaType;
        use std::sync::RwLock;

        let tmp = TempDir::new().unwrap();
        let paths = Arc::new(write_many_geotiffs(&tmp, 24));

        for limit in [1, 3] {
            // A session's wiring: one budget, handed to the loader and to
            // the config extension `RS_FromPath` reads it from.
            let budget = RasterIoBudget::default();
            let loader = GdalLoader::new().with_budget(budget.clone());
            let mut config = ConfigOptions::new();
            config.extensions.insert(SedonaOptions::default());
            config.extensions.insert(
                RasterLoaderConfig::from_handle(Arc::new(RwLock::new(RasterLoaderRegistry::new())))
                    .with_io_budget(budget.clone()),
            );
            config
                .set("sedona.raster.io_concurrency", &limit.to_string())
                .unwrap();
            // What `RS_EnsureLoaded` does before it calls the loader.
            io_budget_from_config(&config).unwrap();
            let config = Arc::new(config);
            let kernel = Arc::new(RsFromPath::default());

            let mut tasks = Vec::new();
            for _ in 0..4 {
                let loader = loader.clone();
                let loader_paths = Arc::clone(&paths);
                tasks.push(tokio::spawn(async move {
                    load_paths(&loader, &loader_paths).await.unwrap().len()
                }));

                let kernel = Arc::clone(&kernel);
                let config = Arc::clone(&config);
                let input =
                    ColumnarValue::Array(Arc::new(StringArray::from(paths.as_ref().clone())));
                tasks.push(tokio::spawn(async move {
                    match kernel
                        .invoke_batch_from_args(
                            &[],
                            &[input],
                            &SedonaType::Arrow(DataType::Null),
                            0,
                            Some(&config),
                        )
                        .unwrap()
                    {
                        ColumnarValue::Array(rasters) => rasters.len(),
                        other => panic!("expected an array, got {other:?}"),
                    }
                }));
            }
            let done = tokio::time::timeout(
                std::time::Duration::from_secs(60),
                futures::future::join_all(tasks),
            )
            .await
            .expect("pixel reads and header opens deadlocked on one budget");
            for rows in done {
                assert_eq!(rows.unwrap(), 24);
            }

            assert_eq!(budget.limit(), limit);
            let peak = budget.peak_in_use();
            assert!((1..=limit).contains(&peak), "budget {limit}: peak {peak}");
            assert!(loader.peak_in_flight() <= limit);
            assert_eq!(budget.in_use(), 0);
            assert_eq!(budget.available(), limit);
        }
    }

    /// One task polls a load that is waiting for the budget, then evaluates
    /// `RS_FromPath`, which blocks that task's thread until it gets a permit
    /// (as when `SortPreservingMergeExec` polls several partitions, or
    /// `EnsureLoadedExec` evaluates hoisted `RS_FromPath` arguments). If the
    /// load waited as a future, the semaphore would hand the released permit
    /// to that future, parked inside the blocked task, and neither could
    /// ever run again.
    #[test]
    fn a_task_with_a_pending_load_can_evaluate_rs_frompath() {
        use crate::rs_frompath::RsFromPath;
        use arrow_array::StringArray;
        use arrow_schema::DataType;
        use datafusion_common::config::ConfigOptions;
        use datafusion_expr::ColumnarValue;
        use sedona_common::option::SedonaOptions;
        use sedona_expr::scalar_udf::SedonaScalarKernel;
        use sedona_raster::raster_loader::{RasterLoaderConfig, RasterLoaderRegistry};
        use sedona_schema::datatypes::SedonaType;
        use std::sync::RwLock;
        use std::time::Duration;

        let tmp = TempDir::new().unwrap();
        let paths = write_many_geotiffs(&tmp, 4);
        let budget = RasterIoBudget::new(1);
        let loader = GdalLoader::new().with_budget(budget.clone());
        let mut config = ConfigOptions::new();
        config.extensions.insert(SedonaOptions::default());
        config.extensions.insert(
            RasterLoaderConfig::from_handle(Arc::new(RwLock::new(RasterLoaderRegistry::new())))
                .with_io_budget(budget.clone()),
        );
        config.set("sedona.raster.io_concurrency", "1").unwrap();
        // Held elsewhere, so both the load and `RS_FromPath` have to wait.
        let held = budget.try_acquire().unwrap();

        // A timeout inside the runtime cannot fire while the task's thread
        // is blocked, so the test thread watches the clock, and a deadlocked
        // runtime is left behind on its own thread rather than dropped.
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .build()
                .unwrap();
            let rows = runtime.block_on(async move {
                tokio::spawn(async move {
                    let release = std::thread::spawn(move || {
                        std::thread::sleep(Duration::from_millis(50));
                        drop(held);
                    });
                    let input = ColumnarValue::Array(Arc::new(StringArray::from(paths.clone())));
                    let (loaded, opened) = tokio::join!(load_paths(&loader, &paths), async {
                        RsFromPath::default().invoke_batch_from_args(
                            &[],
                            &[input],
                            &SedonaType::Arrow(DataType::Null),
                            0,
                            Some(&config),
                        )
                    });
                    release.join().unwrap();
                    let ColumnarValue::Array(opened) = opened.unwrap() else {
                        panic!("expected an array");
                    };
                    (loaded.unwrap().len(), opened.len())
                })
                .await
                .unwrap()
            });
            done_tx.send(rows).unwrap();
        });
        let rows = done_rx
            .recv_timeout(Duration::from_secs(60))
            .expect("a pending load and RS_FromPath in one task deadlocked");
        assert_eq!(rows, (4, 4));
        assert_eq!(budget.in_use(), 0);
        assert_eq!(budget.available(), 1);
    }
}
