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
//! GDAL's blocking API, wrapped in `tokio::task::spawn_blocking` so the
//! caller's async runtime is not stalled. Dataset opens are cached
//! per-thread via the existing `GDALDatasetCache` thread-local, so
//! repeated queries against the same file pay one open per worker
//! thread.
//!
//! Registered against the per-session
//! [`OutDbLoaderRegistry`](sedona_raster::outdb_loader::OutDbLoaderRegistry)
//! under the format key `"gdal"`. The `sedona` crate constructs a
//! [`GdalLoader`] from `SedonaContext::new_from_context` and registers
//! it during session bootstrap.

use arrow_buffer::Buffer;
use arrow_schema::ArrowError;
use async_trait::async_trait;
use sedona_raster::outdb_loader::{AsyncByteLoader, OutDbLoadRequest};

use crate::gdal_common::{convert_gdal_err, gdal_to_band_data_type, with_gdal};
use crate::gdal_dataset_provider::thread_local_cache;
use crate::source_uri::parse_outdb_source;

/// Format key the loader is registered under. Keep in sync with
/// `SedonaContext::new_from_context` and any band-builder code emitting
/// `outdb_format` values.
pub const GDAL_FORMAT: &str = "gdal";

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

        // Take owned copies for the spawn_blocking closure (the closure
        // must be `'static`).
        let uri = req.uri.to_string();
        let expected_dtype = req.data_type;

        // GDAL is sync and the thread-local cache uses `Rc`. Run on a
        // blocking-pool thread so the async runtime stays responsive
        // and the per-worker thread-local cache is constructed lazily
        // there. Result `Buffer` is `Send`, so it crosses the await
        // back to the caller cleanly.
        let buffer = tokio::task::spawn_blocking(move || -> Result<Buffer, ArrowError> {
            with_gdal(|gdal| {
                // `#band=N` fragment, with N defaulting to 1 if absent.
                let (path, band_num) = parse_outdb_source(&uri)?;
                let cache = thread_local_cache()?;
                let dataset = cache.get_or_create_outdb_source(gdal, &path, None)?;
                let band = dataset
                    .rasterband(band_num as usize)
                    .map_err(convert_gdal_err)?;

                // Verify the file's pixel type matches the band metadata's
                // claim BEFORE reading. `read_as_bytes` returns bytes in
                // the file's native type with no conversion — a mismatch
                // would produce a 2x-or-N/2 byte count and the size
                // check in `RS_EnsureLoaded` would mis-blame the loader
                // for size rather than naming the dtype mismatch. Catch
                // it cleanly here.
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

                // `read_as_bytes` allocates a fresh Vec internally.
                // `Buffer::from_vec` wraps it without copying.
                let bytes = band
                    .read_as_bytes((0, 0), (width, height), (width, height), None)
                    .map_err(convert_gdal_err)?;
                Ok(Buffer::from_vec(bytes))
            })
            .map_err(|e| ArrowError::ExternalError(Box::new(e)))
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
}
