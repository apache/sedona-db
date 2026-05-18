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

//! GDAL-backed implementation of the OutDb band loader.
//!
//! Installs the process-wide loader registered with
//! [`sedona_raster::set_outdb_band_loader`]. Reads the band identified by
//! a `#band=N` URI fragment via the thread-local `GDALDatasetCache` so
//! the underlying GDAL dataset is opened at most once per (thread, URI).

use arrow_schema::ArrowError;
use sedona_raster::{OutDbBandLoader, OutDbLoadRequest};

use crate::gdal_common::{convert_gdal_err, gdal_to_band_data_type, with_gdal};
use crate::gdal_dataset_provider::thread_local_cache;
use crate::source_uri::parse_outdb_source;

/// Install the GDAL-backed OutDb loader as the process-wide handler.
///
/// Idempotent — multiple `SedonaContext::new()` calls per process are
/// safe; subsequent registrations are silently ignored by the underlying
/// `OnceLock` in `sedona-raster`.
pub fn register_outdb_loader() {
    let loader: OutDbBandLoader = gdal_load;
    sedona_raster::set_outdb_band_loader(loader);
}

fn gdal_load(req: &OutDbLoadRequest<'_>, scratch: &mut Vec<u8>) -> Result<(), ArrowError> {
    // 2-D `[y, x]` only for v1; higher-rank or transposed OutDb reads
    // need MDArray support and explicit axis mapping, both tracked
    // separately. Matches `raster_ref_to_gdal_mem`'s 2-D constraint in
    // `gdal_common.rs`.
    if req.source_shape.len() != 2 {
        return Err(ArrowError::NotYetImplemented(format!(
            "OutDb GDAL loader only supports 2-D bands; got source_shape with {} dims",
            req.source_shape.len()
        )));
    }
    if req.dim_names != ["y", "x"] {
        return Err(ArrowError::InvalidArgumentError(format!(
            "OutDb GDAL loader requires dim_names=[\"y\", \"x\"]; got {:?}",
            req.dim_names
        )));
    }
    // u64 → usize narrowing. Only the 32-bit target ever exercises the
    // error branch — on 64-bit targets the conversion is infallible.
    let height = usize::try_from(req.source_shape[0]).map_err(|_| {
        ArrowError::InvalidArgumentError(format!(
            "OutDb source_shape[0]={} exceeds usize::MAX",
            req.source_shape[0]
        ))
    })?;
    let width = usize::try_from(req.source_shape[1]).map_err(|_| {
        ArrowError::InvalidArgumentError(format!(
            "OutDb source_shape[1]={} exceeds usize::MAX",
            req.source_shape[1]
        ))
    })?;

    let uri = req.uri;

    with_gdal(|gdal| {
        // `#band=N` fragment, with N defaulting to 1 if absent. The
        // returned path is what GDAL itself will open (VSI translation
        // happens inside `open_gdal_dataset` via the cache key).
        let (path, band_num) = parse_outdb_source(uri)?;
        let cache = thread_local_cache()?;
        let dataset = cache.get_or_create_outdb_source(gdal, &path, None)?;
        let band = dataset
            .rasterband(band_num as usize)
            .map_err(convert_gdal_err)?;
        // Verify the file's pixel type matches the band metadata's
        // claim BEFORE reading. `read_as_bytes` returns bytes in the
        // file's native type with no conversion — a mismatch would
        // produce a 2x-or-N/2 byte count and the size check in
        // `source_bytes()` would mis-blame the loader for size rather
        // than naming the dtype mismatch. Catch it cleanly here.
        let file_dtype = gdal_to_band_data_type(band.band_type())?;
        if file_dtype != req.data_type {
            return Err(sedona_common::sedona_internal_datafusion_err!(
                "OutDb band metadata claims {:?} but file {} band {} is {:?}",
                req.data_type,
                uri,
                band_num,
                file_dtype
            ));
        }
        // Caller (`load_outdb`) cleared scratch on entry; we append the
        // band's bytes into it. `read_as_bytes` allocates a fresh Vec, so
        // this is one heap operation per band — when GDAL exposes a
        // `read_into_bytes(&mut [u8])` we can write directly into scratch
        // and eliminate it. Tracked separately.
        let bytes = band
            .read_as_bytes((0, 0), (width, height), (width, height), None)
            .map_err(convert_gdal_err)?;
        scratch.extend_from_slice(&bytes);
        Ok(())
    })
    .map_err(|e| ArrowError::ExternalError(Box::new(e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sedona_gdal::gdal::Gdal;
    use sedona_gdal::raster::types::Buffer;
    use sedona_raster::array::RasterStructArray;
    use sedona_raster::builder::RasterBuilder;
    use sedona_raster::traits::RasterRef;
    use sedona_schema::raster::BandDataType;
    use std::sync::Once;
    use tempfile::TempDir;

    static REGISTER: Once = Once::new();
    fn ensure_registered() {
        REGISTER.call_once(register_outdb_loader);
    }

    /// 4x3 UInt8 GeoTIFF with deterministic byte pattern. Returns the
    /// raw row-major bytes the file holds so callers can compare.
    fn write_test_tiff(gdal: &Gdal, path: &str, width: usize, height: usize) -> Vec<u8> {
        let driver = gdal.get_driver_by_name("GTiff").unwrap();
        let dataset = driver
            .create_with_band_type::<u8>(path, width, height, 1)
            .unwrap();
        dataset
            .set_geo_transform(&[0.0, 1.0, 0.0, 0.0, 0.0, -1.0])
            .unwrap();
        let data: Vec<u8> = (0..(width * height)).map(|i| (i & 0xFF) as u8).collect();
        {
            let band = dataset.rasterband(1).unwrap();
            let mut buffer = Buffer::new((width, height), data.clone());
            band.write((0, 0), (width, height), &mut buffer).unwrap();
        }
        // Force a flush so reads via a separate dataset see the bytes.
        drop(dataset);
        data
    }

    fn build_outdb_band_array(uri: &str, source_shape: &[u64]) -> arrow_array::StructArray {
        build_outdb_band_array_with(uri, &["y", "x"], source_shape, BandDataType::UInt8)
    }

    fn build_outdb_band_array_with(
        uri: &str,
        dim_names: &[&str],
        source_shape: &[u64],
        data_type: BandDataType,
    ) -> arrow_array::StructArray {
        let mut b = RasterBuilder::new(1);
        let spatial: Vec<i64> = source_shape.iter().map(|&v| v as i64).collect();
        b.start_raster_nd(&[0.0, 1.0, 0.0, 0.0, 0.0, -1.0], dim_names, &spatial, None)
            .unwrap();
        b.start_band_nd(
            None,
            dim_names,
            source_shape,
            data_type,
            None,
            Some(uri),
            Some("geotiff"),
        )
        .unwrap();
        b.band_data_writer().append_value([0u8; 0]); // schema-OutDb
        b.finish_band().unwrap();
        b.finish_raster().unwrap();
        b.finish().unwrap()
    }

    #[test]
    fn loads_outdb_band_bytes_from_geotiff() {
        ensure_registered();
        let tmp = TempDir::new().unwrap();
        let tif_path = tmp.path().join("single.tif");
        let tif_str = tif_path.to_str().unwrap();

        let expected = with_gdal(|gdal| Ok(write_test_tiff(gdal, tif_str, 4, 3))).unwrap();

        let arr = build_outdb_band_array(tif_str, &[3, 4]); // [height, width]
        let rasters = RasterStructArray::new(&arr);
        let raster = rasters.get(0).unwrap();
        let band = raster.band(0).unwrap();

        let mut scratch = Vec::new();
        let buf = band.nd_buffer(&mut scratch).unwrap();
        assert_eq!(buf.buffer, expected.as_slice());
        assert_eq!(buf.shape, vec![3, 4]);
        assert!(
            !band.is_indb(),
            "is_indb must remain false; OutDb is a schema property"
        );
    }

    #[test]
    fn repeated_calls_on_same_band_reuse_thread_local_dataset_cache() {
        ensure_registered();
        let tmp = TempDir::new().unwrap();
        let tif_path = tmp.path().join("cached.tif");
        let tif_str = tif_path.to_str().unwrap();
        let expected = with_gdal(|gdal| Ok(write_test_tiff(gdal, tif_str, 4, 3))).unwrap();

        let arr = build_outdb_band_array(tif_str, &[3, 4]);
        let rasters = RasterStructArray::new(&arr);
        let raster = rasters.get(0).unwrap();
        let band = raster.band(0).unwrap();

        // Both calls go through GDAL. Cross-call cheapness comes from the
        // `GDALDatasetCache` thread-local LRU keeping the source dataset
        // open across calls; the per-band scratch is owned here and reused
        // for both reads (same Vec<u8>, same allocation reused). The
        // returned bytes must match across calls.
        let mut scratch = Vec::new();
        let a = band.nd_buffer(&mut scratch).unwrap().buffer.to_vec();
        let b = band.nd_buffer(&mut scratch).unwrap().buffer.to_vec();
        assert_eq!(a, expected);
        assert_eq!(a, b);
    }

    fn write_two_band_tiff(gdal: &Gdal, path: &str) -> (Vec<u8>, Vec<u8>) {
        let driver = gdal.get_driver_by_name("GTiff").unwrap();
        let dataset = driver.create_with_band_type::<u8>(path, 2, 2, 2).unwrap();
        dataset
            .set_geo_transform(&[0.0, 1.0, 0.0, 0.0, 0.0, -1.0])
            .unwrap();
        let band1_data = vec![10u8, 20, 30, 40];
        let band2_data = vec![100u8, 110, 120, 130];
        {
            let b1 = dataset.rasterband(1).unwrap();
            let mut buf1 = Buffer::new((2, 2), band1_data.clone());
            b1.write((0, 0), (2, 2), &mut buf1).unwrap();
        }
        {
            let b2 = dataset.rasterband(2).unwrap();
            let mut buf2 = Buffer::new((2, 2), band2_data.clone());
            b2.write((0, 0), (2, 2), &mut buf2).unwrap();
        }
        drop(dataset);
        (band1_data, band2_data)
    }

    #[test]
    fn band_fragment_selects_correct_band() {
        ensure_registered();
        let tmp = TempDir::new().unwrap();
        let tif_path = tmp.path().join("multi.tif");
        let tif_str = tif_path.to_str().unwrap();
        let (band1, band2) = with_gdal(|gdal| Ok(write_two_band_tiff(gdal, tif_str))).unwrap();

        let uri_b1 = format!("{tif_str}#band=1");
        let uri_b2 = format!("{tif_str}#band=2");

        let mut scratch = Vec::new();
        let arr1 = build_outdb_band_array(&uri_b1, &[2, 2]);
        let r1 = RasterStructArray::new(&arr1);
        let raster_one = r1.get(0).unwrap();
        let band_one = raster_one.band(0).unwrap();
        assert_eq!(
            band_one.nd_buffer(&mut scratch).unwrap().buffer,
            band1.as_slice()
        );

        let arr2 = build_outdb_band_array(&uri_b2, &[2, 2]);
        let r2 = RasterStructArray::new(&arr2);
        let raster_two = r2.get(0).unwrap();
        let band_two = raster_two.band(0).unwrap();
        assert_eq!(
            band_two.nd_buffer(&mut scratch).unwrap().buffer,
            band2.as_slice()
        );
    }

    #[test]
    fn rejects_non_yx_dim_names() {
        ensure_registered();
        let tmp = TempDir::new().unwrap();
        let tif_path = tmp.path().join("xy.tif");
        let tif_str = tif_path.to_str().unwrap();
        let _ = with_gdal(|gdal| Ok(write_test_tiff(gdal, tif_str, 4, 3))).unwrap();

        // Band metadata says dim order is [x, y] instead of the
        // GDAL-expected [y, x]. Loader must refuse rather than reading
        // with transposed dimensions.
        let arr = build_outdb_band_array_with(tif_str, &["x", "y"], &[4, 3], BandDataType::UInt8);
        let rasters = RasterStructArray::new(&arr);
        let raster = rasters.get(0).unwrap();
        let band = raster.band(0).unwrap();
        let mut scratch = Vec::new();
        let err = band.nd_buffer(&mut scratch).unwrap_err().to_string();
        assert!(
            err.contains("dim_names") && err.contains("\"x\""),
            "expected dim_names rejection naming the offending axes, got: {err}"
        );
    }

    #[test]
    fn rejects_pixel_type_mismatch() {
        ensure_registered();
        let tmp = TempDir::new().unwrap();
        let tif_path = tmp.path().join("u16.tif");
        let tif_str = tif_path.to_str().unwrap();
        // Write a UInt16 GeoTIFF but claim UInt8 in the band metadata.
        with_gdal(|gdal| {
            let driver = gdal.get_driver_by_name("GTiff").unwrap();
            let dataset = driver
                .create_with_band_type::<u16>(tif_str, 2, 2, 1)
                .unwrap();
            dataset
                .set_geo_transform(&[0.0, 1.0, 0.0, 0.0, 0.0, -1.0])
                .unwrap();
            {
                let band = dataset.rasterband(1).unwrap();
                let mut buf = Buffer::new((2, 2), vec![1u16, 2, 3, 4]);
                band.write((0, 0), (2, 2), &mut buf).unwrap();
            }
            drop(dataset);
            Ok(())
        })
        .unwrap();

        let arr = build_outdb_band_array_with(tif_str, &["y", "x"], &[2, 2], BandDataType::UInt8);
        let rasters = RasterStructArray::new(&arr);
        let raster = rasters.get(0).unwrap();
        let band = raster.band(0).unwrap();
        let mut scratch = Vec::new();
        let err = band.nd_buffer(&mut scratch).unwrap_err().to_string();
        assert!(
            err.contains("UInt8") && err.contains("UInt16"),
            "expected dtype mismatch naming both types, got: {err}"
        );
    }

    #[test]
    fn errors_on_missing_file() {
        ensure_registered();
        let tmp = TempDir::new().unwrap();
        let missing = tmp.path().join("does_not_exist.tif");
        let arr = build_outdb_band_array(missing.to_str().unwrap(), &[2, 2]);
        let rasters = RasterStructArray::new(&arr);
        let raster = rasters.get(0).unwrap();
        let band = raster.band(0).unwrap();
        // Loader propagates GDAL's open-failure as an ArrowError; the
        // exact message is GDAL/version dependent, so just assert that
        // *some* error surfaces rather than a panic or silent empty buffer.
        let mut scratch = Vec::new();
        assert!(band.nd_buffer(&mut scratch).is_err());
    }

    #[test]
    fn errors_on_band_index_out_of_range() {
        ensure_registered();
        let tmp = TempDir::new().unwrap();
        let tif_path = tmp.path().join("oneband.tif");
        let tif_str = tif_path.to_str().unwrap();
        let _ = with_gdal(|gdal| Ok(write_test_tiff(gdal, tif_str, 2, 2))).unwrap();

        // File has 1 band; request band 5.
        let uri = format!("{tif_str}#band=5");
        let arr = build_outdb_band_array(&uri, &[2, 2]);
        let rasters = RasterStructArray::new(&arr);
        let raster = rasters.get(0).unwrap();
        let band = raster.band(0).unwrap();
        let mut scratch = Vec::new();
        assert!(band.nd_buffer(&mut scratch).is_err());
    }
}
