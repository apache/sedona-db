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

//! RS_FromGDALRaster UDF - Parse binary content using GDAL driver as in-db raster
//!
//! Similar to PostGIS's ST_FromGDALRaster. Parses binary content using GDAL driver
//! and loads it as an in-db raster with all band data stored inline.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use arrow_array::{cast::AsArray, Array};
use arrow_schema::DataType;
use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_common::{exec_datafusion_err, ScalarValue};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_gdal::gdal::Gdal;
use sedona_gdal::gdal_dyn_bindgen::{GDAL_OF_RASTER, GDAL_OF_READONLY};
use sedona_gdal::raster::types::DatasetOptions;
use sedona_raster::builder::RasterBuilder;
use sedona_schema::datatypes::{SedonaType, RASTER};
use sedona_schema::matchers::ArgMatcher;

use crate::gdal_common::{convert_gdal_err, with_gdal};
use crate::gdal_dataset_provider::configure_thread_local_options;
use crate::utils::append_as_indb_raster;

/// Counter for generating unique VSI memory file names
static VSI_FILE_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// RS_FromGDALRaster() scalar UDF implementation
///
/// Parse binary content using GDAL driver and load it as in-db raster
pub fn rs_from_gdal_raster_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_fromgdalraster",
        vec![Arc::new(RsFromGDALRaster)],
        Volatility::Immutable,
    )
}

/// Kernel implementation for RS_FromGDALRaster
#[derive(Debug)]
pub(crate) struct RsFromGDALRaster;

impl RsFromGDALRaster {
    /// Generate a unique VSI memory file path
    fn generate_vsi_path() -> String {
        let counter = VSI_FILE_COUNTER.fetch_add(1, Ordering::SeqCst);
        let thread_id = std::thread::current().id();
        format!(
            "/vsimem/rs_from_gdal_raster_{:?}_{}.bin",
            thread_id, counter
        )
    }

    /// Write `content` to a temporary `/vsimem` file, open it with GDAL, and
    /// append the decoded raster to `builder` as an in-db raster (all band data
    /// materialised inline). The VSI file is always cleaned up.
    fn append_gdal_raster(gdal: &Gdal, content: &[u8], builder: &mut RasterBuilder) -> Result<()> {
        let vsi_path = Self::generate_vsi_path();
        gdal.create_mem_file(&vsi_path, content)
            .map_err(|e| exec_datafusion_err!("Failed to create VSI memory file: {e}"))?;

        // Open + decode, then always unlink the VSI file (the dataset is dropped
        // at the end of the closure, before the unlink).
        let result = (|| {
            let dataset = gdal
                .open_ex_with_options(
                    &vsi_path,
                    DatasetOptions {
                        open_flags: GDAL_OF_RASTER | GDAL_OF_READONLY,
                        ..Default::default()
                    },
                )
                .map_err(convert_gdal_err)?;
            append_as_indb_raster(&dataset, builder)
        })();
        let _ = gdal.unlink_mem_file(&vsi_path);
        result
    }

    /// Decode each input row into `builder`: a NULL row appends a NULL raster, a
    /// non-null row is decoded to an in-db raster. Generic over the binary array
    /// flavour so `Binary` and `BinaryView` iterate through the same path.
    fn append_rows<'a>(
        gdal: &Gdal,
        rows: impl Iterator<Item = Option<&'a [u8]>>,
        builder: &mut RasterBuilder,
    ) -> Result<()> {
        for row in rows {
            match row {
                None => builder
                    .append_null()
                    .map_err(|e| exec_datafusion_err!("Failed to append null: {e}"))?,
                Some(content) => Self::append_gdal_raster(gdal, content, builder)?,
            }
        }
        Ok(())
    }

    /// Parse binary content into a single in-db raster. Test-only convenience
    /// around [`append_gdal_raster`](Self::append_gdal_raster); the kernel
    /// appends directly into a shared builder.
    #[cfg(test)]
    pub(crate) fn parse_gdal_raster(
        gdal: &Gdal,
        content: &[u8],
    ) -> Result<arrow_array::StructArray> {
        let mut builder = RasterBuilder::new(1);
        Self::append_gdal_raster(gdal, content, &mut builder)?;
        builder
            .finish()
            .map_err(|e| exec_datafusion_err!("Failed to build raster: {e}"))
    }
}

impl SedonaScalarKernel for RsFromGDALRaster {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(vec![ArgMatcher::is_binary()], RASTER);
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        self.invoke_batch_from_args(arg_types, args, &SedonaType::Arrow(DataType::Null), 0, None)
    }

    fn invoke_batch_from_args(
        &self,
        _arg_types: &[SedonaType],
        args: &[ColumnarValue],
        _return_type: &SedonaType,
        _num_rows: usize,
        config_options: Option<&ConfigOptions>,
    ) -> Result<ColumnarValue> {
        with_gdal(|gdal| {
            configure_thread_local_options(gdal, config_options)?;

            let content_array = match &args[0] {
                ColumnarValue::Scalar(scalar) => scalar
                    .to_array()
                    .map_err(|e| exec_datafusion_err!("Failed to convert scalar to array: {e}"))?,
                ColumnarValue::Array(array) => array.clone(),
            };

            // Decode every row into one raster array. The binary matcher accepts
            // both `Binary` and `BinaryView` (`RS_AsGeoTiff` produces the latter),
            // so read each flavour directly rather than narrowing `BinaryView`
            // offsets into `Binary`'s i32 range.
            let mut builder = RasterBuilder::new(content_array.len());
            match content_array.data_type() {
                DataType::Binary => {
                    Self::append_rows(gdal, content_array.as_binary::<i32>().iter(), &mut builder)?
                }
                DataType::BinaryView => {
                    Self::append_rows(gdal, content_array.as_binary_view().iter(), &mut builder)?
                }
                other => {
                    return sedona_internal_err!(
                        "RS_FromGDALRaster expected Binary or BinaryView content, got {other:?}"
                    )
                }
            }
            let result = builder
                .finish()
                .map_err(|e| exec_datafusion_err!("Failed to build raster: {e}"))?;

            match &args[0] {
                ColumnarValue::Scalar(_) => Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
                    &result, 0,
                )?)),
                ColumnarValue::Array(_) => Ok(ColumnarValue::Array(Arc::new(result))),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{ArrayRef, BinaryArray, BinaryViewArray};
    use sedona_gdal::raster::types::Buffer;
    use sedona_raster::array::RasterStructArray;
    use sedona_schema::datatypes::RASTER;
    use sedona_testing::raster_spec::{
        assert_raster_scalar_equals, assert_rasters_equal, RasterSpec,
    };
    use sedona_testing::testers::ScalarUdfTester;

    use crate::rs_as_geotiff::rs_as_geotiff_udf;

    /// Build a small 4x4 single-band GeoTIFF (EPSG:4326) with GDAL and return its
    /// bytes together with the CRS GDAL reads back from them (PROJJSON) — the
    /// fixture-free stand-in for a `.tiff` on disk, so tests exercise the real
    /// decode path without shipping a binary fixture.
    ///
    /// The CRS is read from the fixture itself, not copied from a decode result,
    /// so a [`RasterSpec`] can pin CRS preservation without hard-coding a
    /// PROJJSON blob that drifts across PROJ versions.
    fn make_geotiff_fixture(gdal: &Gdal) -> (Vec<u8>, Option<String>) {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("src.tif");
        let path_str = path.to_string_lossy().to_string();
        {
            let driver = gdal.get_driver_by_name("GTiff").unwrap();
            let dataset = driver
                .create_with_band_type::<u8>(&path_str, 4, 4, 1)
                .unwrap();
            dataset
                .set_geo_transform(&[0.0, 1.0, 0.0, 4.0, 0.0, -1.0])
                .unwrap();
            dataset.set_projection("EPSG:4326").unwrap();
            let band = dataset.rasterband(1).unwrap();
            let mut buffer = Buffer::new((4, 4), (0..16u8).collect::<Vec<_>>());
            band.write((0, 0), (4, 4), &mut buffer).unwrap();
        } // drop flushes the dataset to disk
        let bytes = std::fs::read(&path).unwrap();
        let crs = gdal
            .open_ex_with_options(
                &path_str,
                DatasetOptions {
                    open_flags: GDAL_OF_RASTER | GDAL_OF_READONLY,
                    ..Default::default()
                },
            )
            .unwrap()
            .spatial_ref()
            .ok()
            .and_then(|sr| sr.to_projjson().ok());
        (bytes, crs)
    }

    /// Build a 4x4 two-band UInt8 GeoTIFF (no CRS). Each band's 16 bytes exceed
    /// the inline view threshold, so an in-db decode attaches one shared data
    /// block per band — the property [`decoded_two_band_raster_is_zero_copy`]
    /// pins.
    fn make_two_band_geotiff_bytes(gdal: &Gdal) -> Vec<u8> {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("two_band.tif");
        let path_str = path.to_string_lossy().to_string();
        {
            let driver = gdal.get_driver_by_name("GTiff").unwrap();
            let dataset = driver
                .create_with_band_type::<u8>(&path_str, 4, 4, 2)
                .unwrap();
            dataset
                .set_geo_transform(&[0.0, 1.0, 0.0, 4.0, 0.0, -1.0])
                .unwrap();
            let band1 = dataset.rasterband(1).unwrap();
            let mut b1 = Buffer::new((4, 4), (0..16u8).collect::<Vec<_>>());
            band1.write((0, 0), (4, 4), &mut b1).unwrap();
            let band2 = dataset.rasterband(2).unwrap();
            let mut b2 = Buffer::new((4, 4), (100..116u8).collect::<Vec<_>>());
            band2.write((0, 0), (4, 4), &mut b2).unwrap();
        } // drop flushes the dataset to disk
        std::fs::read(&path).unwrap()
    }

    fn from_gdal_tester() -> ScalarUdfTester {
        ScalarUdfTester::new(
            rs_from_gdal_raster_udf().into(),
            vec![SedonaType::Arrow(DataType::Binary)],
        )
    }

    fn from_gdal_tester_binary_view() -> ScalarUdfTester {
        ScalarUdfTester::new(
            rs_from_gdal_raster_udf().into(),
            vec![SedonaType::Arrow(DataType::BinaryView)],
        )
    }

    /// The single-band fixture's declarative expectation, derived from the
    /// fixture's construction: 4x4 UInt8 spanning the bbox (0, 0)-(4, 4)
    /// (north-up unit pixels), sequential values 0..16, no nodata, in-db.
    fn single_band_spec(crs: Option<&str>) -> RasterSpec {
        RasterSpec::d2(4, 4)
            .bbox(0.0, 0.0, 4.0, 4.0)
            .crs(crs)
            .band_values(&(0..16u8).collect::<Vec<_>>())
    }

    #[test]
    fn test_generate_vsi_path() {
        let path1 = RsFromGDALRaster::generate_vsi_path();
        let path2 = RsFromGDALRaster::generate_vsi_path();

        assert!(path1.starts_with("/vsimem/rs_from_gdal_raster_"));
        assert!(path2.starts_with("/vsimem/rs_from_gdal_raster_"));
        assert_ne!(path1, path2);
    }

    #[test]
    fn udf_from_gdal_raster() {
        let udf: datafusion_expr::ScalarUDF = rs_from_gdal_raster_udf().into();
        assert_eq!(udf.name(), "rs_fromgdalraster");
    }

    #[test]
    fn parse_gdal_raster_builds_indb_raster() {
        // The direct builder path: GeoTIFF bytes decode to an in-db raster
        // matching the fixture's dimensions, transform, CRS and pixel values.
        // The spec's in-db bands assert (via storage_type) that band data was
        // materialised inline rather than left as an out-db reference.
        with_gdal(|gdal| {
            let (bytes, crs) = make_geotiff_fixture(gdal);
            let arr: ArrayRef = Arc::new(RsFromGDALRaster::parse_gdal_raster(gdal, &bytes)?);
            assert_rasters_equal(&arr, &[Some(single_band_spec(crs.as_deref()))]);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn from_gdal_raster_decodes_via_udf() {
        // End-to-end through the UDF's scalar path: GeoTIFF binary in, raster out.
        let (bytes, crs) = with_gdal(|gdal| Ok(make_geotiff_fixture(gdal))).unwrap();
        let result = from_gdal_tester()
            .invoke_scalar(ScalarValue::Binary(Some(bytes)))
            .unwrap();
        assert_raster_scalar_equals(&result, &single_band_spec(crs.as_deref()));
    }

    #[test]
    fn from_gdal_raster_decodes_binary_view_input() {
        // The `is_binary` matcher also accepts `BinaryView`, which is what
        // `RS_AsGeoTiff` produces. The same GeoTIFF bytes wrapped in a
        // `BinaryView` array must decode to the same raster as the `Binary`
        // path — not fail casting `BinaryView` to i32-offset `Binary`.
        let (bytes, crs) = with_gdal(|gdal| Ok(make_geotiff_fixture(gdal))).unwrap();
        let input: ArrayRef = Arc::new(BinaryViewArray::from(vec![Some(bytes.as_slice())]));
        let result = from_gdal_tester_binary_view()
            .invoke_arrays(vec![input])
            .unwrap();
        assert_rasters_equal(&result, &[Some(single_band_spec(crs.as_deref()))]);
    }

    #[test]
    fn as_geotiff_from_gdal_raster_round_trips() {
        // The `RS_AsGeoTiff` -> `RS_FromGDALRaster` round trip used in the docs:
        // encoding yields `BinaryView` bytes that must decode back to the source
        // raster. A north-up single-band UInt8 raster (no CRS, no nodata) is
        // preserved exactly through GeoTIFF.
        let source = RasterSpec::d2(3, 3)
            .crs(None)
            .bbox(0.0, 0.0, 3.0, 3.0)
            .band_values(&[1u8, 2, 3, 4, 5, 6, 7, 8, 9]);

        let encoded = ScalarUdfTester::new(rs_as_geotiff_udf().into(), vec![RASTER])
            .invoke_scalar(&source)
            .unwrap();
        let ScalarValue::BinaryView(Some(bytes)) = encoded else {
            panic!("expected a BinaryView result, got {encoded:?}");
        };

        let decoded = from_gdal_tester_binary_view()
            .invoke_scalar(ScalarValue::BinaryView(Some(bytes)))
            .unwrap();
        assert_raster_scalar_equals(&decoded, &source);
    }

    #[test]
    fn decoded_two_band_raster_is_zero_copy() {
        with_gdal(|gdal| {
            let bytes = make_two_band_geotiff_bytes(gdal);
            let arr = RsFromGDALRaster::parse_gdal_raster(gdal, &bytes)?;

            // Zero-copy: each band's freshly-read allocation is attached as its
            // own shared data block (a refcount bump), so the band-data view
            // has one buffer per band. A copying `append_value` path would
            // consolidate both bands into a single builder-owned buffer.
            let num_data_buffers = RasterStructArray::try_new(&arr)
                .unwrap()
                .band_data_array()
                .data_buffers()
                .len();
            assert_eq!(num_data_buffers, 2);

            // ...and the decoded values/structure match the fixture.
            let arr: ArrayRef = Arc::new(arr);
            assert_rasters_equal(
                &arr,
                &[Some(
                    RasterSpec::d2(4, 4)
                        .crs(None)
                        .bbox(0.0, 0.0, 4.0, 4.0)
                        .band_values(&(0..16u8).collect::<Vec<_>>())
                        .band_values(&(100..116u8).collect::<Vec<_>>()),
                )],
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn null_binary_yields_null_raster() {
        let input: ArrayRef = Arc::new(BinaryArray::from(vec![None::<&[u8]>]));
        let result = from_gdal_tester().invoke_arrays(vec![input]).unwrap();
        assert_rasters_equal(&result, &[None]);
    }

    #[test]
    fn empty_binary_errors() {
        // Non-null but empty bytes: GDAL has nothing to open. A clean error,
        // not a panic.
        let input: ArrayRef = Arc::new(BinaryArray::from(vec![Some(&b""[..])]));
        assert!(from_gdal_tester().invoke_arrays(vec![input]).is_err());
    }

    #[test]
    fn unparseable_bytes_error() {
        // Bytes GDAL cannot identify as any raster format.
        let input: ArrayRef = Arc::new(BinaryArray::from(vec![Some(&b"not a raster at all"[..])]));
        assert!(from_gdal_tester().invoke_arrays(vec![input]).is_err());
    }

    #[test]
    fn truncated_geotiff_errors() {
        // A real GeoTIFF cut down to its 8-byte header: the IFD offset now
        // points past EOF, so GDAL cannot open it. Exercises the malformed
        // header path without a panic.
        let (bytes, _) = with_gdal(|gdal| Ok(make_geotiff_fixture(gdal))).unwrap();
        let truncated = bytes[..8.min(bytes.len())].to_vec();
        let input: ArrayRef = Arc::new(BinaryArray::from(vec![Some(truncated.as_slice())]));
        assert!(from_gdal_tester().invoke_arrays(vec![input]).is_err());
    }
}
