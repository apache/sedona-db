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

//! RS_AsGeoTiff UDF - Export raster as GeoTiff binary
//!
//! Returns a binary DataFrame from a Raster DataFrame with multiple overloads:
//! - RS_AsGeoTiff(raster)
//! - RS_AsGeoTiff(raster, tileSize)
//! - RS_AsGeoTiff(raster, compressionType, imageQuality)
//! - RS_AsGeoTiff(raster, compressionType, imageQuality, tileSize)
//! - RS_AsGeoTiff(raster, compressionType, imageQuality, tileWidth, tileHeight)

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::gdal_common::with_gdal;
use arrow_array::builder::BinaryBuilder;
use arrow_schema::DataType;
use datafusion_common::cast::{as_float64_array, as_string_array, as_uint32_array};
use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_common::{exec_datafusion_err, ScalarValue};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::array::RasterRefImpl;
use sedona_raster_functions::RasterExecutor;
use sedona_schema::datatypes::SedonaType;
use sedona_schema::matchers::ArgMatcher;

// Use thread-local provider to create GDAL datasets from `RasterRef`.
use crate::gdal_dataset_provider::configure_thread_local_options;

/// Counter for generating unique VSI memory file names
static VSI_FILE_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Compression types supported for GeoTiff output
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionType {
    None,
    PackBits,
    Deflate,
    Huffman,
    Lzw,
    Jpeg,
}

impl CompressionType {
    /// Parse compression type from string (case-insensitive)
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "none" => Some(CompressionType::None),
            "packbits" => Some(CompressionType::PackBits),
            "deflate" => Some(CompressionType::Deflate),
            "huffman" => Some(CompressionType::Huffman),
            "lzw" => Some(CompressionType::Lzw),
            "jpeg" => Some(CompressionType::Jpeg),
            _ => None,
        }
    }

    /// Get GDAL compression option value
    pub fn gdal_value(&self) -> &'static str {
        match self {
            CompressionType::None => "NONE",
            CompressionType::PackBits => "PACKBITS",
            CompressionType::Deflate => "DEFLATE",
            CompressionType::Huffman => "CCITTRLE",
            CompressionType::Lzw => "LZW",
            CompressionType::Jpeg => "JPEG",
        }
    }
}

/// RS_AsGeoTiff() scalar UDF implementation
///
/// Returns a binary DataFrame from a Raster DataFrame
pub fn rs_as_geotiff_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_asgeotiff",
        vec![
            Arc::new(RsAsGeoTiff::new(Variant::Basic)), // RS_AsGeoTiff(raster)
            Arc::new(RsAsGeoTiff::new(Variant::WithTileSize)), // RS_AsGeoTiff(raster, tileSize)
            Arc::new(RsAsGeoTiff::new(Variant::WithCompressionQuality)), // RS_AsGeoTiff(raster, compression, quality)
            Arc::new(RsAsGeoTiff::new(Variant::WithCompressionQualityTileSize)), // RS_AsGeoTiff(raster, compression, quality, tileSize)
            Arc::new(RsAsGeoTiff::new(Variant::WithCompressionQualityTileWH)), // RS_AsGeoTiff(raster, compression, quality, tileWidth, tileHeight)
        ],
        Volatility::Immutable,
    )
}

/// Variants for different overloads
#[derive(Debug, Clone, Copy)]
enum Variant {
    Basic,                          // (raster)
    WithTileSize,                   // (raster, tileSize)
    WithCompressionQuality,         // (raster, compression, quality)
    WithCompressionQualityTileSize, // (raster, compression, quality, tileSize)
    WithCompressionQualityTileWH,   // (raster, compression, quality, tileWidth, tileHeight)
}

/// Kernel implementation for RS_AsGeoTiff
#[derive(Debug)]
struct RsAsGeoTiff {
    variant: Variant,
}

impl RsAsGeoTiff {
    fn new(variant: Variant) -> Self {
        Self { variant }
    }

    /// Generate a unique VSI memory file path
    fn generate_vsi_path() -> String {
        let counter = VSI_FILE_COUNTER.fetch_add(1, Ordering::SeqCst);
        let thread_id = std::thread::current().id();
        format!("/vsimem/rs_as_geotiff_{:?}_{}.tif", thread_id, counter)
    }

    /// Convert raster to GeoTiff bytes
    fn raster_to_geotiff(
        gdal: &sedona_gdal::gdal::Gdal,
        raster: &RasterRefImpl,
        compression: Option<CompressionType>,
        quality: Option<f64>,
        tile_width: Option<u32>,
        tile_height: Option<u32>,
    ) -> Result<Vec<u8>> {
        let provider = crate::gdal_dataset_provider::thread_local_provider(gdal)
            .map_err(|e| exec_datafusion_err!("Failed to init GDAL provider: {}", e))?;
        let raster_ds = provider
            .raster_ref_to_gdal(raster)
            .map_err(|e| exec_datafusion_err!("Failed to create GDAL dataset: {}", e))?;
        let source_dataset = raster_ds.as_dataset();

        let driver = gdal
            .get_driver_by_name("GTiff")
            .map_err(|e| exec_datafusion_err!("Failed to get GTiff driver: {}", e))?;

        // Build creation options as string list
        let mut options_list: Vec<String> = Vec::new();

        // Add compression option
        if let Some(comp) = compression {
            options_list.push(format!("COMPRESS={}", comp.gdal_value()));

            // Add quality for JPEG
            if comp == CompressionType::Jpeg {
                if let Some(q) = quality {
                    // JPEG quality is 1-100, we receive 0.0-1.0
                    let jpeg_quality = (q * 100.0).round() as i32;
                    options_list.push(format!("JPEG_QUALITY={}", jpeg_quality.clamp(1, 100)));
                }
            }

            // Add predictor for Deflate/LZW (improves compression)
            if comp == CompressionType::Deflate || comp == CompressionType::Lzw {
                options_list.push("PREDICTOR=2".to_string());
            }
        }

        // Add tiling options
        if let (Some(tw), Some(th)) = (tile_width, tile_height) {
            options_list.push("TILED=YES".to_string());
            options_list.push(format!("BLOCKXSIZE={}", tw));
            options_list.push(format!("BLOCKYSIZE={}", th));
        }

        // Convert to creation options slice
        let options_refs: Vec<&str> = options_list.iter().map(|s| s.as_str()).collect();

        // Generate VSI path for output
        let vsi_path = Self::generate_vsi_path();

        // Create copy to VSI memory file
        let _output_dataset = source_dataset
            .create_copy(&driver, &vsi_path, &options_refs)
            .map_err(|e| exec_datafusion_err!("Failed to create GeoTiff: {}", e))?;

        // Close the output dataset to flush data
        drop(_output_dataset);

        // Read bytes from VSI memory file and clean up
        let bytes = gdal.get_vsi_mem_file_bytes_owned(&vsi_path).map_err(|e| {
            let _ = gdal.unlink_mem_file(&vsi_path);
            exec_datafusion_err!("Failed to read GeoTiff bytes: {}", e)
        })?;

        // Clean up VSI file
        let _ = gdal.unlink_mem_file(&vsi_path);

        Ok(bytes)
    }
}

impl SedonaScalarKernel for RsAsGeoTiff {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matchers = match self.variant {
            Variant::Basic => vec![ArgMatcher::is_raster()],
            Variant::WithTileSize => vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_integer(), // tileSize
            ],
            Variant::WithCompressionQuality => vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_string(),  // compressionType
                ArgMatcher::is_numeric(), // imageQuality
            ],
            Variant::WithCompressionQualityTileSize => vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_string(),  // compressionType
                ArgMatcher::is_numeric(), // imageQuality
                ArgMatcher::is_integer(), // tileSize
            ],
            Variant::WithCompressionQualityTileWH => vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_string(),  // compressionType
                ArgMatcher::is_numeric(), // imageQuality
                ArgMatcher::is_integer(), // tileWidth
                ArgMatcher::is_integer(), // tileHeight
            ],
        };

        let matcher = ArgMatcher::new(matchers, SedonaType::Arrow(DataType::Binary));
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
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
        _return_type: &SedonaType,
        _num_rows: usize,
        config_options: Option<&ConfigOptions>,
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let num_iterations = executor.num_iterations();

        // Convert variant-specific args to arrays upfront via into_array.
        // For variants that don't use a parameter, create null-filled default arrays.
        let (compression_array, quality_array, tile_width_array, tile_height_array) =
            match self.variant {
                Variant::Basic => {
                    // No extra args → all null arrays
                    let compression = ScalarValue::Utf8(None).to_array_of_size(num_iterations)?;
                    let quality = ScalarValue::Float64(None).to_array_of_size(num_iterations)?;
                    let tile_width = ScalarValue::UInt32(None).to_array_of_size(num_iterations)?;
                    let tile_height = ScalarValue::UInt32(None).to_array_of_size(num_iterations)?;
                    (compression, quality, tile_width, tile_height)
                }
                Variant::WithTileSize => {
                    // args[1] → tile_width AND tile_height
                    let compression = ScalarValue::Utf8(None).to_array_of_size(num_iterations)?;
                    let quality = ScalarValue::Float64(None).to_array_of_size(num_iterations)?;
                    let tile_size = args[1]
                        .clone()
                        .cast_to(&DataType::UInt32, None)?
                        .into_array(num_iterations)?;
                    (compression, quality, tile_size.clone(), tile_size)
                }
                Variant::WithCompressionQuality => {
                    // args[1] → compression, args[2] → quality
                    let compression = args[1]
                        .clone()
                        .cast_to(&DataType::Utf8, None)?
                        .into_array(num_iterations)?;
                    let quality = args[2]
                        .clone()
                        .cast_to(&DataType::Float64, None)?
                        .into_array(num_iterations)?;
                    let tile_width = ScalarValue::UInt32(None).to_array_of_size(num_iterations)?;
                    let tile_height = ScalarValue::UInt32(None).to_array_of_size(num_iterations)?;
                    (compression, quality, tile_width, tile_height)
                }
                Variant::WithCompressionQualityTileSize => {
                    // args[1] → compression, args[2] → quality, args[3] → tile_width AND tile_height
                    let compression = args[1]
                        .clone()
                        .cast_to(&DataType::Utf8, None)?
                        .into_array(num_iterations)?;
                    let quality = args[2]
                        .clone()
                        .cast_to(&DataType::Float64, None)?
                        .into_array(num_iterations)?;
                    let tile_size = args[3]
                        .clone()
                        .cast_to(&DataType::UInt32, None)?
                        .into_array(num_iterations)?;
                    (compression, quality, tile_size.clone(), tile_size)
                }
                Variant::WithCompressionQualityTileWH => {
                    // args[1] → compression, args[2] → quality, args[3] → tile_width, args[4] → tile_height
                    let compression = args[1]
                        .clone()
                        .cast_to(&DataType::Utf8, None)?
                        .into_array(num_iterations)?;
                    let quality = args[2]
                        .clone()
                        .cast_to(&DataType::Float64, None)?
                        .into_array(num_iterations)?;
                    let tile_width = args[3]
                        .clone()
                        .cast_to(&DataType::UInt32, None)?
                        .into_array(num_iterations)?;
                    let tile_height = args[4]
                        .clone()
                        .cast_to(&DataType::UInt32, None)?
                        .into_array(num_iterations)?;
                    (compression, quality, tile_width, tile_height)
                }
            };

        // Downcast all parameter arrays once before the loop
        let compression_array = as_string_array(&compression_array)?;
        let quality_array = as_float64_array(&quality_array)?;
        let tile_width_array = as_uint32_array(&tile_width_array)?;
        let tile_height_array = as_uint32_array(&tile_height_array)?;

        // Create iterators for each parameter array
        let mut compression_iter = compression_array.iter();
        let mut quality_iter = quality_array.iter();
        let mut tile_width_iter = tile_width_array.iter();
        let mut tile_height_iter = tile_height_array.iter();

        // Build output binary array
        let mut builder = BinaryBuilder::with_capacity(num_iterations, num_iterations * 1024);

        with_gdal(|gdal| {
            configure_thread_local_options(gdal, config_options)?;
            executor.execute_raster_void(|_i, raster_opt| {
                let compression_opt = compression_iter.next().unwrap();
                let quality_opt = quality_iter.next().unwrap();
                let tile_width_opt = tile_width_iter.next().unwrap();
                let tile_height_opt = tile_height_iter.next().unwrap();

                let raster = match raster_opt {
                    Some(raster) => raster,
                    None => {
                        builder.append_null();
                        return Ok(());
                    }
                };

                let compression = match compression_opt {
                    Some(comp_str) => Some(CompressionType::parse(comp_str).ok_or_else(|| {
                        exec_datafusion_err!(
                            "Unknown compression type: {}. Valid values: None, PackBits, Deflate, Huffman, LZW, JPEG",
                            comp_str
                        )
                    })?),
                    None => None,
                };

                let quality = quality_opt;
                let tile_width = tile_width_opt;
                let tile_height = tile_height_opt;

                let bytes = Self::raster_to_geotiff(
                    gdal,
                    raster,
                    compression,
                    quality,
                    tile_width,
                    tile_height,
                )?;
                builder.append_value(&bytes);

                Ok(())
            })?;

            executor.finish(Arc::new(builder.finish()))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sedona_raster::traits::RasterRef;

    #[test]
    fn test_compression_type_parse() {
        assert_eq!(CompressionType::parse("none"), Some(CompressionType::None));
        assert_eq!(CompressionType::parse("NONE"), Some(CompressionType::None));
        assert_eq!(
            CompressionType::parse("deflate"),
            Some(CompressionType::Deflate)
        );
        assert_eq!(
            CompressionType::parse("DEFLATE"),
            Some(CompressionType::Deflate)
        );
        assert_eq!(CompressionType::parse("lzw"), Some(CompressionType::Lzw));
        assert_eq!(CompressionType::parse("jpeg"), Some(CompressionType::Jpeg));
        assert_eq!(CompressionType::parse("invalid"), None);
    }

    #[test]
    fn test_generate_vsi_path() {
        let path1 = RsAsGeoTiff::generate_vsi_path();
        let path2 = RsAsGeoTiff::generate_vsi_path();

        assert!(path1.starts_with("/vsimem/rs_as_geotiff_"));
        assert!(path1.ends_with(".tif"));
        assert!(path2.starts_with("/vsimem/rs_as_geotiff_"));
        assert_ne!(path1, path2);
    }

    #[test]
    fn udf_as_geotiff() {
        let udf: datafusion_expr::ScalarUDF = rs_as_geotiff_udf().into();
        assert_eq!(udf.name(), "rs_asgeotiff");
    }

    #[test]
    fn test_roundtrip_geotiff() {
        use crate::rs_from_gdal_raster::RsFromGDALRaster;
        use sedona_raster::array::RasterStructArray;
        use sedona_testing::data::test_raster;

        // Load test4.tiff as in-db raster
        let path = test_raster("test4.tiff").expect("test4.tiff should exist");
        with_gdal(|gdal| {
            let raster_arr = crate::utils::load_as_indb_raster(gdal, &path)?;
            let raster_array = RasterStructArray::new(&raster_arr);
            assert_eq!(raster_array.len(), 1);
            let raster = raster_array.get(0).expect("Should get raster");

            let geotiff_bytes =
                RsAsGeoTiff::raster_to_geotiff(gdal, &raster, None, None, None, None)?;
            assert!(geotiff_bytes.len() > 4, "GeoTiff should have content");
            assert!(
                &geotiff_bytes[0..2] == b"II" || &geotiff_bytes[0..2] == b"MM",
                "Should be valid TIFF header"
            );

            let roundtrip_arr = RsFromGDALRaster::parse_gdal_raster(gdal, &geotiff_bytes)?;
            let roundtrip_array = RasterStructArray::new(&roundtrip_arr);
            let roundtrip_raster = roundtrip_array.get(0).expect("Should get roundtrip raster");

            assert_eq!(
                roundtrip_raster.metadata().width(),
                raster.metadata().width()
            );
            assert_eq!(
                roundtrip_raster.metadata().height(),
                raster.metadata().height()
            );
            assert_eq!(roundtrip_raster.bands().len(), raster.bands().len());
            Ok::<_, datafusion_common::DataFusionError>(())
        })
        .expect("Should roundtrip GeoTiff");
    }

    #[test]
    fn test_geotiff_with_compression() {
        use sedona_raster::array::RasterStructArray;
        use sedona_testing::data::test_raster;

        // Load test raster
        let path = test_raster("test4.tiff").expect("test4.tiff should exist");
        with_gdal(|gdal| {
            let raster_arr = crate::utils::load_as_indb_raster(gdal, &path)?;
            let raster_array = RasterStructArray::new(&raster_arr);
            let raster = raster_array.get(0).expect("Should get raster");

            let lzw_bytes = RsAsGeoTiff::raster_to_geotiff(
                gdal,
                &raster,
                Some(CompressionType::Lzw),
                Some(75.0),
                None,
                None,
            )?;
            assert!(
                !lzw_bytes.is_empty(),
                "LZW compressed GeoTiff should have content"
            );

            let deflate_bytes = RsAsGeoTiff::raster_to_geotiff(
                gdal,
                &raster,
                Some(CompressionType::Deflate),
                Some(6.0),
                None,
                None,
            )?;
            assert!(
                !deflate_bytes.is_empty(),
                "DEFLATE compressed GeoTiff should have content"
            );
            assert!(&lzw_bytes[0..2] == b"II" || &lzw_bytes[0..2] == b"MM");
            assert!(&deflate_bytes[0..2] == b"II" || &deflate_bytes[0..2] == b"MM");
            Ok::<_, datafusion_common::DataFusionError>(())
        })
        .expect("Should convert with DEFLATE");
    }
}
