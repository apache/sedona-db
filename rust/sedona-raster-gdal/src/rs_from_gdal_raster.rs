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

use arrow_array::Array;
use arrow_schema::DataType;
use datafusion_common::cast::as_binary_array;
use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_common::{exec_datafusion_err, ScalarValue};
use datafusion_expr::{ColumnarValue, Volatility};
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
            let binary_array = as_binary_array(&content_array)?;
            let len = binary_array.len();

            // Decode every row into one raster array. A NULL input row yields a
            // NULL raster row; a non-null row is decoded to an in-db raster.
            let mut builder = RasterBuilder::new(len);
            for i in 0..len {
                if binary_array.is_null(i) {
                    builder
                        .append_null()
                        .map_err(|e| exec_datafusion_err!("Failed to append null: {e}"))?;
                } else {
                    Self::append_gdal_raster(gdal, binary_array.value(i), &mut builder)?;
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
    use crate::gdal_common::with_gdal;
    use arrow_array::{ArrayRef, BinaryArray};
    use datafusion_common::cast::as_struct_array;
    use sedona_gdal::raster::types::Buffer;
    use sedona_raster::array::RasterStructArray;
    use sedona_raster::traits::RasterRef;
    use sedona_testing::testers::ScalarUdfTester;

    /// Build a small 4x4 single-band GeoTIFF (EPSG:4326) with GDAL and return its
    /// bytes — the fixture-free stand-in for a `.tiff` on disk, so tests exercise
    /// the real decode path without shipping a binary fixture.
    fn make_geotiff_bytes(gdal: &Gdal) -> Vec<u8> {
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
        std::fs::read(&path).unwrap()
    }

    fn from_gdal_tester() -> ScalarUdfTester {
        ScalarUdfTester::new(
            rs_from_gdal_raster_udf().into(),
            vec![SedonaType::Arrow(DataType::Binary)],
        )
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
        // GeoTIFF bytes decode to an in-db raster with the source dimensions/CRS.
        with_gdal(|gdal| {
            let bytes = make_geotiff_bytes(gdal);
            let arr = RsFromGDALRaster::parse_gdal_raster(gdal, &bytes)?;
            let rasters = RasterStructArray::try_new(&arr).unwrap();
            assert_eq!(rasters.len(), 1);
            let raster = rasters.get(0).unwrap();
            assert_eq!(raster.metadata().width(), 4);
            assert_eq!(raster.metadata().height(), 4);
            assert_eq!(raster.num_bands(), 1);
            assert!(raster.crs().is_some(), "EPSG:4326 should survive decode");
            // In-db: band data is materialised inline, not an out-db reference.
            assert!(raster.band_outdb_uri(0).is_none());
            Ok::<_, datafusion_common::DataFusionError>(())
        })
        .unwrap();
    }

    #[test]
    fn from_gdal_raster_decodes_via_udf() {
        // End-to-end through the UDF: GeoTIFF binary in, raster out.
        let bytes = with_gdal(|gdal| Ok(make_geotiff_bytes(gdal))).unwrap();
        let input: ArrayRef = Arc::new(BinaryArray::from(vec![bytes.as_slice()]));
        let result = from_gdal_tester().invoke_arrays(vec![input]).unwrap();
        let rasters = RasterStructArray::try_new(as_struct_array(&result).unwrap()).unwrap();
        let raster = rasters.get(0).unwrap();
        assert_eq!(raster.metadata().width(), 4);
        assert_eq!(raster.metadata().height(), 4);
        assert_eq!(raster.num_bands(), 1);
    }

    #[test]
    fn null_binary_yields_null_raster() {
        let input: ArrayRef = Arc::new(BinaryArray::from(vec![None::<&[u8]>]));
        let result = from_gdal_tester().invoke_arrays(vec![input]).unwrap();
        let struct_arr = as_struct_array(&result).unwrap();
        assert!(
            struct_arr.is_null(0),
            "NULL bytes should yield a NULL raster"
        );
    }
}
