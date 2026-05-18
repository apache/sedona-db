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

//! Process-wide hook that resolves the bytes of a schema-OutDb raster band.
//!
//! `sedona-raster` deliberately knows nothing about GDAL, Zarr, or any
//! other backend. When a session needs to read pixels from a band whose
//! Arrow `data` column is empty, [`BandRefImpl`](crate::array) calls into
//! [`load_outdb`], which dispatches to whatever loader was installed at
//! bootstrap via [`set_outdb_band_loader`]. `sedona-raster-gdal` installs
//! a GDAL-backed loader from `SedonaContext::new()`; other backends (e.g.
//! a future Zarr crate) would install their own.
//!
//! Existing GDAL kernels (`raster_ref_to_gdal_mem`, VRT builder) short-
//! circuit OutDb bands long before reaching the byte-access surface, so
//! the loader is consulted only by non-GDAL consumers (Arrow FFI, numeric
//! kernels, Python bindings).

use std::sync::OnceLock;

use arrow_schema::ArrowError;
use sedona_schema::raster::BandDataType;

/// Everything a backend needs to materialise a single OutDb band's bytes.
///
/// `source_shape` is the **raw** source shape in `dim_names` order — the
/// shape of the buffer the loader must return — *not* the visible shape.
/// View composition stays inside `BandRefImpl`: the band walks the
/// returned buffer using the existing stride machinery.
pub struct OutDbLoadRequest<'a> {
    /// External URI (e.g. `file:///tmp/foo.tif#band=1`,
    /// `s3://bucket/cube.zarr#path=/a&slice=0`). Bare paths are also
    /// allowed; backends are responsible for parsing.
    pub uri: &'a str,
    /// Optional format hint set on the band (`"geotiff"`, `"zarr"`, …).
    /// `None` means "let the backend infer from the URI".
    pub format: Option<&'a str>,
    /// Per-axis names, parallel to `source_shape`. Backends use this to
    /// map their native axis order onto the band's. The GDAL backend
    /// today requires 2-D `["y", "x"]`; other shapes return an error.
    pub dim_names: &'a [&'a str],
    /// Raw source shape in `dim_names` order. The loader returns bytes
    /// sized to exactly `prod(source_shape) * data_type.byte_size()`.
    pub source_shape: &'a [u64],
    /// Pixel type. Loaders must return bytes encoding pixels in this type.
    pub data_type: BandDataType,
}

/// Function-pointer signature for OutDb loaders.
pub type OutDbBandLoader = fn(&OutDbLoadRequest<'_>) -> Result<Vec<u8>, ArrowError>;

static OUTDB_BAND_LOADER: OnceLock<OutDbBandLoader> = OnceLock::new();

/// Install the process-wide OutDb loader. Idempotent — first writer wins.
///
/// Multiple `SedonaContext::new()` calls within one process must not panic;
/// subsequent calls are silently ignored. This is the only way to register
/// a loader; tests dispatch internally on `req.uri` / `req.data_type`.
pub fn set_outdb_band_loader(f: OutDbBandLoader) {
    let _ = OUTDB_BAND_LOADER.set(f);
}

/// Dispatch a load request through the installed loader. Returns
/// `NotYetImplemented` with a clear message if no loader was registered.
pub(crate) fn load_outdb(req: &OutDbLoadRequest<'_>) -> Result<Vec<u8>, ArrowError> {
    match OUTDB_BAND_LOADER.get() {
        Some(loader) => loader(req),
        None => Err(ArrowError::NotYetImplemented(
            "no OutDb loader registered; sedona-raster-gdal (or another \
             backend) must be initialised before reading OutDb bands"
                .to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::RasterStructArray;
    use crate::builder::RasterBuilder;
    use crate::traits::RasterRef;
    use std::sync::Once;

    /// Single process-wide mock loader, dispatched by URI scheme.
    /// `OUTDB_BAND_LOADER` is a `OnceLock` — exactly one registration per
    /// process, so every test that needs OutDb behavior routes through
    /// here and selects a case via the URI it constructs the band with.
    fn mock_load(req: &OutDbLoadRequest<'_>) -> Result<Vec<u8>, ArrowError> {
        let expected: usize =
            req.source_shape.iter().product::<u64>() as usize * req.data_type.byte_size();
        match req.uri {
            "mock://zeros" => Ok(vec![0u8; expected]),
            "mock://pattern" => Ok((0..expected).map(|i| (i & 0xFF) as u8).collect()),
            "mock://too-small" => Ok(vec![0u8; expected.saturating_sub(1)]),
            "mock://error" => Err(ArrowError::ExternalError(Box::new(std::io::Error::other(
                "mock loader simulated failure",
            )))),
            other => Err(ArrowError::NotYetImplemented(format!(
                "mock loader: unrecognised uri `{other}`"
            ))),
        }
    }

    static MOCK_INSTALL: Once = Once::new();
    fn install_mock_loader() {
        MOCK_INSTALL.call_once(|| set_outdb_band_loader(mock_load));
    }

    /// Build a single-raster, single-band StructArray whose band is
    /// schema-OutDb (empty data column) and points at `uri`.
    fn build_outdb_band(
        uri: Option<&str>,
        data_type: BandDataType,
        source_shape: &[u64],
    ) -> arrow_array::StructArray {
        let dim_names: Vec<&str> = ["y", "x", "t", "u", "v"]
            .iter()
            .take(source_shape.len())
            .copied()
            .collect();
        let spatial_shape: Vec<i64> = source_shape.iter().map(|&v| v as i64).collect();

        let mut b = RasterBuilder::new(1);
        b.start_raster_nd(
            &[0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
            &dim_names,
            &spatial_shape,
            None,
        )
        .unwrap();
        b.start_band_nd(
            None,
            &dim_names,
            source_shape,
            data_type,
            None,
            uri,
            uri.map(|_| "geotiff"),
        )
        .unwrap();
        b.band_data_writer().append_value([0u8; 0]); // empty → schema-OutDb
        b.finish_band().unwrap();
        b.finish_raster().unwrap();
        b.finish().unwrap()
    }

    #[test]
    fn nd_buffer_returns_loader_bytes_for_outdb_band() {
        install_mock_loader();
        let arr = build_outdb_band(Some("mock://pattern"), BandDataType::UInt8, &[2, 3]);
        let rasters = RasterStructArray::new(&arr);
        let raster = rasters.get(0).unwrap();
        let band = raster.band(0).unwrap();
        let buf = band.nd_buffer().unwrap();
        assert_eq!(buf.buffer, &[0u8, 1, 2, 3, 4, 5]);
        assert_eq!(buf.shape, vec![2, 3]);
        assert_eq!(buf.data_type, BandDataType::UInt8);
    }

    #[test]
    fn contiguous_data_returns_loader_bytes_for_outdb_band() {
        install_mock_loader();
        let arr = build_outdb_band(Some("mock://pattern"), BandDataType::UInt8, &[2, 3]);
        let rasters = RasterStructArray::new(&arr);
        let raster = rasters.get(0).unwrap();
        let band = raster.band(0).unwrap();
        let cow = band.contiguous_data().unwrap();
        assert_eq!(&*cow, &[0u8, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn data_returns_loader_bytes_for_outdb_band() {
        install_mock_loader();
        let arr = build_outdb_band(Some("mock://pattern"), BandDataType::UInt8, &[2, 3]);
        let rasters = RasterStructArray::new(&arr);
        let raster = rasters.get(0).unwrap();
        let band = raster.band(0).unwrap();
        assert_eq!(band.data(), &[0u8, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn is_indb_stays_false_after_loader_runs() {
        // Pin the schema-vs-runtime reframe: `is_indb()` is the schema
        // discriminator (Arrow column emptiness), not a runtime
        // byte-availability check. It must stay false after the loader
        // has populated `outdb_loaded`.
        install_mock_loader();
        let arr = build_outdb_band(Some("mock://zeros"), BandDataType::UInt8, &[2, 3]);
        let rasters = RasterStructArray::new(&arr);
        let raster = rasters.get(0).unwrap();
        let band = raster.band(0).unwrap();
        assert!(!band.is_indb());
        let _ = band.nd_buffer().unwrap();
        assert!(
            !band.is_indb(),
            "is_indb must remain a schema discriminator after the loader runs"
        );
    }

    #[test]
    fn nd_buffer_is_cached_across_calls() {
        // Smoke test: repeated nd_buffer() over the same band must
        // succeed and return identical contents (the OnceCell hides
        // the second loader call).
        install_mock_loader();
        let arr = build_outdb_band(Some("mock://pattern"), BandDataType::UInt8, &[2, 3]);
        let rasters = RasterStructArray::new(&arr);
        let raster = rasters.get(0).unwrap();
        let band = raster.band(0).unwrap();
        let a = band.nd_buffer().unwrap();
        let buf_a = a.buffer.to_vec();
        let b = band.nd_buffer().unwrap();
        assert_eq!(buf_a, b.buffer);
    }

    #[test]
    fn nd_buffer_errors_when_outdb_uri_is_missing() {
        install_mock_loader();
        // OutDb-shaped band (empty data column) but no outdb_uri set.
        let arr = build_outdb_band(None, BandDataType::UInt8, &[2, 3]);
        let rasters = RasterStructArray::new(&arr);
        let raster = rasters.get(0).unwrap();
        let band = raster.band(0).unwrap();
        let err = band.nd_buffer().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("outdb_uri"),
            "error message should reference the missing uri: {msg}"
        );
    }

    #[test]
    fn nd_buffer_surfaces_loader_failure() {
        install_mock_loader();
        let arr = build_outdb_band(Some("mock://error"), BandDataType::UInt8, &[2, 3]);
        let rasters = RasterStructArray::new(&arr);
        let raster = rasters.get(0).unwrap();
        let band = raster.band(0).unwrap();
        assert!(band.nd_buffer().is_err());
    }

    #[test]
    fn nd_buffer_rejects_undersized_loader_output() {
        install_mock_loader();
        let arr = build_outdb_band(Some("mock://too-small"), BandDataType::UInt8, &[2, 3]);
        let rasters = RasterStructArray::new(&arr);
        let raster = rasters.get(0).unwrap();
        let band = raster.band(0).unwrap();
        let err = band.nd_buffer().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("buffer") || msg.contains("byte"),
            "undersized loader output should trip the view-bounds check: {msg}"
        );
    }

    #[test]
    fn data_returns_empty_on_loader_failure() {
        // `data()` is the legacy infallible accessor — errors collapse
        // to `&[]`.
        install_mock_loader();
        let arr = build_outdb_band(Some("mock://error"), BandDataType::UInt8, &[2, 3]);
        let rasters = RasterStructArray::new(&arr);
        let raster = rasters.get(0).unwrap();
        let band = raster.band(0).unwrap();
        assert_eq!(band.data(), &[] as &[u8]);
    }
}
