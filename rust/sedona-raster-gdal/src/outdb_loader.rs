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

//! GDAL backend implementing [`sedona_raster::outdb_loader::AsyncByteLoader`].
//!
//! Reads OutDb raster bands identified by a `#band=N` URI fragment via
//! GDAL's blocking API. The blocking work runs inside
//! `tokio::task::spawn_blocking` so the caller's async runtime is not
//! stalled. Dataset opens are cached per-thread via the existing
//! `GDALDatasetCache` thread-local, so repeated queries against the
//! same file pay one open per worker thread.
//!
//! ## Cancellation
//!
//! Reads run as a loop of block-height-aligned strips
//! (`band.block_size().1` rows per iteration), with a cooperative
//! cancellation check between strips. When the outer async future is
//! dropped (e.g. a query is cancelled), a [`CancelOnDrop`] guard flips
//! a shared [`AtomicBool`]; the next iteration of the loop observes
//! the flag and returns a cancellation error rather than running to
//! completion.
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
//! [`OutDbLoaderRegistry`](sedona_raster::outdb_loader::OutDbLoaderRegistry)
//! under the format key `"gdal"`. The `sedona` crate constructs a
//! [`GdalLoader`] from `SedonaContext::new_from_context` and registers
//! it during session bootstrap.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use arrow_buffer::Buffer;
use arrow_schema::ArrowError;
use async_trait::async_trait;
use datafusion_common::{DataFusionError, Result as DFResult};
use sedona_gdal::raster::rasterband::RasterBand;
use sedona_raster::outdb_loader::{AsyncByteLoader, OutDbLoadRequest};

use crate::gdal_common::{convert_gdal_err, gdal_to_band_data_type, with_gdal};
use crate::gdal_dataset_provider::thread_local_cache;
use crate::source_uri::parse_outdb_source;

/// Format key the loader is registered under. Keep in sync with
/// `SedonaContext::new_from_context` and any band-builder code emitting
/// `outdb_format` values.
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

/// GDAL-backed `AsyncByteLoader`.
///
/// Stateless: the per-thread dataset cache lives in a thread-local owned
/// by `sedona-raster-gdal::gdal_dataset_provider`, so constructing a
/// `GdalLoader` is free and instances are interchangeable.
#[derive(Debug, Default, Clone, Copy)]
pub struct GdalLoader;

impl GdalLoader {
    pub fn new() -> Self {
        Self
    }
}

/// Drop guard that flips an `AtomicBool` when the outer async future
/// is dropped. Paired with a `spawn_blocking` task that polls the same
/// flag between unit-of-work iterations: dropping the outer future
/// signals the blocking task to exit at the next checkpoint.
struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

#[async_trait]
impl AsyncByteLoader for GdalLoader {
    async fn load(&self, req: &OutDbLoadRequest<'_>) -> Result<Buffer, ArrowError> {
        // Validate request shape synchronously, before spawning a blocking
        // task — these are programming errors, no point queueing them
        // onto a worker.
        if req.source_shape.len() != 2 {
            return Err(ArrowError::NotYetImplemented(format!(
                "GDAL OutDb loader only supports 2-D bands; got source_shape with {} dims",
                req.source_shape.len()
            )));
        }
        if req.dim_names != ["y", "x"] {
            return Err(ArrowError::InvalidArgumentError(format!(
                "GDAL OutDb loader requires dim_names=[\"y\", \"x\"]; got {:?}",
                req.dim_names
            )));
        }
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
        let expected_bytes_u64 = (req.source_shape[0])
            .checked_mul(req.source_shape[1])
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
        let expected_bytes = expected_bytes_u64 as usize;

        // Take owned copies for the spawn_blocking closure (the closure
        // must be `'static`).
        let uri = req.uri.to_string();
        let expected_dtype = req.data_type;

        // Cancellation plumbing: the guard lives in this async fn's
        // frame. On normal completion `_guard` drops after the await
        // returns, flipping the flag on an already-finished blocking
        // task (no-op). On cancellation (outer future dropped
        // mid-await), the guard drops first and flips the flag; the
        // blocking task observes it at the next strip boundary and
        // returns a cancellation error.
        let cancel: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
        let _guard = CancelOnDrop(Arc::clone(&cancel));

        let buffer = tokio::task::spawn_blocking({
            let cancel = Arc::clone(&cancel);
            move || -> Result<Buffer, ArrowError> {
                with_gdal(|gdal| {
                    // `#band=N` fragment, with N defaulting to 1 if absent.
                    let (path, band_num) = parse_outdb_source(&uri)?;
                    let cache = thread_local_cache()?;
                    let dataset = cache.get_or_create_outdb_source(gdal, &path, None)?;
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
                    if file_dtype != expected_dtype {
                        return Err(sedona_common::sedona_internal_datafusion_err!(
                            "GDAL OutDb band metadata claims {:?} but file {} band {} is {:?}",
                            expected_dtype,
                            uri,
                            band_num,
                            file_dtype
                        ));
                    }

                    // Pre-allocate the output buffer once; each strip read
                    // writes into a contiguous slice.
                    let mut output = vec![0u8; expected_bytes];
                    read_band_blockwise(&band, &mut output, width, height, byte_size, &cancel)?;
                    Ok(Buffer::from_vec(output))
                })
                .map_err(|e| ArrowError::ExternalError(Box::new(e)))
            }
        })
        .await
        .map_err(|e| {
            ArrowError::ExternalError(Box::new(sedona_common::sedona_internal_datafusion_err!(
                "GDAL OutDb loader task panicked or was cancelled: {e}"
            )))
        })??;

        Ok(buffer)
    }
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
            return Err(sedona_common::sedona_internal_datafusion_err!(
                "GDAL OutDb read range [{}..{}) exceeds output buffer length {}",
                byte_off,
                byte_end,
                output.len()
            ));
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
        let req = OutDbLoadRequest {
            uri: &uri,
            dim_names: &["y", "x"],
            source_shape: &[2, 3],
            data_type: BandDataType::UInt8,
        };

        let buf = loader.load(&req).await.unwrap();
        assert_eq!(buf.len(), 6);
        assert_eq!(buf.as_slice(), &[0u8, 1, 2, 3, 4, 5]);
    }

    #[tokio::test]
    async fn gdal_loader_defaults_to_band_1_when_fragment_missing() {
        let tmp = TempDir::new().unwrap();
        let path = write_uint8_geotiff(&tmp, "no_fragment.tif");
        let uri = path;

        let loader = GdalLoader::new();
        let req = OutDbLoadRequest {
            uri: &uri,
            dim_names: &["y", "x"],
            source_shape: &[2, 3],
            data_type: BandDataType::UInt8,
        };
        let buf = loader.load(&req).await.unwrap();
        assert_eq!(buf.len(), 6);
    }

    #[tokio::test]
    async fn gdal_loader_rejects_non_2d_source_shape() {
        let loader = GdalLoader::new();
        let req = OutDbLoadRequest {
            uri: "ignored",
            dim_names: &["t", "y", "x"],
            source_shape: &[2, 3, 4],
            data_type: BandDataType::UInt8,
        };
        let err = loader.load(&req).await.unwrap_err();
        assert!(
            err.to_string().contains("2-D"),
            "expected 2-D rejection diagnostic, got: {err}"
        );
    }

    #[tokio::test]
    async fn gdal_loader_rejects_non_yx_dim_names() {
        let loader = GdalLoader::new();
        let req = OutDbLoadRequest {
            uri: "ignored",
            dim_names: &["x", "y"], // transposed
            source_shape: &[2, 3],
            data_type: BandDataType::UInt8,
        };
        let err = loader.load(&req).await.unwrap_err();
        assert!(
            err.to_string().contains("dim_names"),
            "expected dim_names rejection diagnostic, got: {err}"
        );
    }

    #[tokio::test]
    async fn gdal_loader_errors_when_dtype_disagrees_with_file() {
        let tmp = TempDir::new().unwrap();
        let path = write_uint8_geotiff(&tmp, "dtype_mismatch.tif");
        let uri = format!("{path}#band=1");

        let loader = GdalLoader::new();
        let req = OutDbLoadRequest {
            uri: &uri,
            dim_names: &["y", "x"],
            source_shape: &[2, 3],
            // File is UInt8 but we claim Int16 — should fail with a
            // clear dtype-mismatch message, not garbled bytes.
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
    async fn gdal_loader_errors_on_missing_file() {
        let loader = GdalLoader::new();
        let req = OutDbLoadRequest {
            uri: "/nonexistent/path/to/file.tif#band=1",
            dim_names: &["y", "x"],
            source_shape: &[2, 3],
            data_type: BandDataType::UInt8,
        };
        let err = loader.load(&req).await.unwrap_err();
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
        let req = OutDbLoadRequest {
            uri: &uri,
            dim_names: &["y", "x"],
            source_shape: &[2, 3],
            data_type: BandDataType::UInt8,
        };
        let err = loader.load(&req).await.unwrap_err();
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
        let req = OutDbLoadRequest {
            uri: "ignored",
            dim_names: &["y", "x"],
            source_shape: &[1 << 16, 1 << 16],
            data_type: BandDataType::Float32,
        };
        let err = loader.load(&req).await.unwrap_err();
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
        let req = OutDbLoadRequest {
            uri: &uri,
            dim_names: &["y", "x"],
            source_shape: &[64, 16],
            data_type: BandDataType::UInt8,
        };
        let buf = loader.load(&req).await.unwrap();
        let expected: Vec<u8> = (0..16 * 64).map(|i| (i % 251) as u8).collect();
        assert_eq!(buf.as_slice(), expected.as_slice());
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
}
