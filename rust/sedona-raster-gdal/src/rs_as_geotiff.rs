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

use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::gdal_common::with_gdal;
use arrow_array::builder::BinaryViewBuilder;
use arrow_buffer::Buffer;
use arrow_schema::DataType;
use datafusion_common::cast::{as_float64_array, as_string_array, as_uint32_array};
use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_common::{exec_datafusion_err, exec_err, ScalarValue};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_gdal::vsi::VSIBuffer;
use sedona_raster::array::RasterRefImpl;
use sedona_raster::traits::RasterRef;
use sedona_raster_functions::RasterExecutor;
use sedona_schema::datatypes::SedonaType;
use sedona_schema::matchers::ArgMatcher;
use sedona_schema::raster::BandDataType;

// Use thread-local provider to create GDAL datasets from `RasterRef`.
use crate::gdal_dataset_provider::{
    configure_thread_local_options, thread_local_provider, GDALDatasetProvider,
};

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

    /// Generate a unique VSI memory file path. `Relaxed` suffices: the counter
    /// only has to hand out distinct values, no ordering with other memory.
    fn generate_vsi_path() -> String {
        let counter = VSI_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let thread_id = std::thread::current().id();
        format!("/vsimem/rs_as_geotiff_{:?}_{}.tif", thread_id, counter)
    }

    /// Convert raster to GeoTiff bytes
    fn raster_to_geotiff(
        gdal: &sedona_gdal::gdal::Gdal,
        provider: &GDALDatasetProvider,
        raster: &RasterRefImpl,
        compression: Option<CompressionType>,
        quality: Option<f64>,
        tile_width: Option<u32>,
        tile_height: Option<u32>,
    ) -> Result<VSIBuffer> {
        let raster_ds = provider
            .raster_ref_to_gdal(raster)
            .map_err(|e| exec_datafusion_err!("Failed to create GDAL dataset: {}", e))?;
        let source_dataset = raster_ds.as_dataset();

        let driver = gdal
            .get_driver_by_name("GTiff")
            .map_err(|e| exec_datafusion_err!("Failed to get GTiff driver: {}", e))?;

        // Validate and map the quality up front so an out-of-range value errors
        // for every codec, not only JPEG (the codecs that ignore quality should
        // not silently accept nonsense either).
        let jpeg_quality = quality.map(jpeg_quality_option).transpose()?;

        // Build creation options as string list
        let mut options_list: Vec<String> = Vec::new();

        // Add compression option
        if let Some(comp) = compression {
            options_list.push(format!("COMPRESS={}", comp.gdal_value()));

            // Add quality for JPEG
            if comp == CompressionType::Jpeg {
                if let Some(q) = jpeg_quality {
                    options_list.push(format!("JPEG_QUALITY={}", q));
                }
            }

            // Add a predictor for Deflate/LZW (improves compression): horizontal
            // differencing (2) for integer samples, floating-point prediction (3)
            // for float samples — predictor 2 on float data is legal but usually
            // hurts the ratio. GTiff requires uniform band types, so the first
            // band's type decides for the whole file.
            if comp == CompressionType::Deflate || comp == CompressionType::Lzw {
                options_list.push(format!("PREDICTOR={}", predictor_for(raster)?));
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

        // Output VSI path, unlinked on every exit path by the guard: without it
        // a failed `create_copy` (invalid creation options, incompatible band
        // layout, ...) can leave a partially written file in process-lifetime
        // vsimem memory, accumulating across failures.
        let vsi_path = Self::generate_vsi_path();
        let guard = VsiMemFileGuard {
            gdal,
            path: &vsi_path,
        };

        // Create the copy in the VSI memory file. The returned dataset is
        // dropped immediately (end of statement), which closes it and flushes
        // the bytes to the vsimem file.
        source_dataset
            .create_copy(&driver, &vsi_path, &options_refs)
            .map_err(|e| exec_datafusion_err!("Failed to create GeoTiff: {}", e))?;

        // Seize the vsimem file's buffer without copying: `VSIBuffer` owns the
        // GDAL allocation (freed on drop) and unlinks the file, so the only
        // byte copy left is the append into the output builder. The guard's
        // unlink becomes a no-op on this path but still cleans up when
        // `create_copy` or the seize fails.
        let bytes = gdal
            .get_vsi_mem_file_buffer_owned(&vsi_path)
            .map_err(|e| exec_datafusion_err!("Failed to read GeoTiff bytes: {}", e))?;

        drop(guard);
        Ok(bytes)
    }
}

/// Unlinks a vsimem file when dropped, so every exit path of
/// [`RsAsGeoTiff::raster_to_geotiff`] — including failed `create_copy` —
/// releases the process-lifetime vsimem allocation.
struct VsiMemFileGuard<'a> {
    gdal: &'a sedona_gdal::gdal::Gdal,
    path: &'a str,
}

impl Drop for VsiMemFileGuard<'_> {
    fn drop(&mut self) {
        // Unlinking a file that create_copy never managed to create is a no-op
        // error, which is fine to ignore.
        let _ = self.gdal.unlink_mem_file(self.path);
    }
}

/// Append one encoded GeoTIFF to the output as a view over the GDAL
/// allocation itself — the bytes are not copied. The [`VSIBuffer`] becomes an
/// external Arrow allocation owned by the output array, so its lifetime (and
/// `VSIFree`) follows the array rather than this call.
fn append_geotiff_view(builder: &mut BinaryViewBuilder, bytes: VSIBuffer) -> Result<()> {
    let len = bytes.len();
    // A binary-view element addresses at most u32::MAX bytes.
    let Ok(view_len) = u32::try_from(len) else {
        return exec_err!("RS_AsGeoTiff: {len}-byte GeoTIFF exceeds the 4 GiB binary output limit");
    };
    // Views this small are stored inline in the view struct; wrapping the
    // allocation would save nothing (and a zero-length buffer has no pointer).
    if len <= 12 {
        builder.append_value(bytes.as_ref());
        return Ok(());
    }
    let Some(ptr) = NonNull::new(bytes.as_ref().as_ptr() as *mut u8) else {
        return exec_err!("RS_AsGeoTiff: GDAL returned a null buffer for a {len}-byte GeoTIFF");
    };
    // SAFETY: `ptr`/`len` describe exactly the allocation owned by `bytes`,
    // which stores a raw pointer (the bytes never move) and stays alive inside
    // the Arc — freeing via `VSIFree` — until the last Arrow reference to
    // `buffer` drops.
    let buffer = unsafe { Buffer::from_custom_allocation(ptr, len, Arc::new(bytes)) };
    let block = builder.append_block(buffer);
    builder
        .try_append_view(block, 0, view_len)
        .map_err(|e| exec_datafusion_err!("RS_AsGeoTiff: failed to append binary view: {e}"))
}

/// Map a quality fraction in `[0.0, 1.0]` to GDAL's 1–100 `JPEG_QUALITY`.
///
/// The fractional scale matches Apache Sedona (GeoTools' `setCompressionQuality`);
/// a value outside the range errors rather than clamping — silently clamping
/// would turn the most likely mistake (passing a 0–100 quality like `75`) into
/// maximum quality with no warning.
fn jpeg_quality_option(quality: f64) -> Result<i32> {
    if !(0.0..=1.0).contains(&quality) {
        return exec_err!(
            "RS_AsGeoTiff: quality must be a fraction between 0.0 and 1.0 (got {quality}); \
             e.g. use 0.75 for JPEG quality 75"
        );
    }
    // Round to 1-100; GDAL rejects 0, so 0.0 maps to the minimum quality 1.
    Ok(((quality * 100.0).round() as i32).max(1))
}

/// TIFF predictor for Deflate/LZW: 3 (floating-point prediction) for float
/// bands, 2 (horizontal differencing) for integer bands. Decided by the first
/// band's sample type; GTiff creation requires uniform band types anyway.
fn predictor_for(raster: &RasterRefImpl) -> Result<i32> {
    let bands = raster.bands();
    if bands.is_empty() {
        return Ok(2);
    }
    let band = bands
        .band(1)
        .map_err(|e| exec_datafusion_err!("RS_AsGeoTiff: {e}"))?;
    let data_type = band
        .metadata()
        .data_type()
        .map_err(|e| exec_datafusion_err!("RS_AsGeoTiff: {e}"))?;
    Ok(match data_type {
        BandDataType::Float32 | BandDataType::Float64 => 3,
        _ => 2,
    })
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

        let matcher = ArgMatcher::new(matchers, SedonaType::Arrow(DataType::BinaryView));
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
        let mut builder = BinaryViewBuilder::with_capacity(num_iterations);

        with_gdal(|gdal| {
            configure_thread_local_options(gdal, config_options)?;
            let provider = thread_local_provider(gdal)
                .map_err(|e| exec_datafusion_err!("Failed to init GDAL provider: {e}"))?;
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
                    &provider,
                    raster,
                    compression,
                    quality,
                    tile_width,
                    tile_height,
                )?;
                append_geotiff_view(&mut builder, bytes)?;

                Ok(())
            })?;

            executor.finish(Arc::new(builder.finish()))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Array;
    use datafusion_expr::ScalarUDF;
    use sedona_gdal::gdal_dyn_bindgen::{GDAL_OF_RASTER, GDAL_OF_READONLY};
    use sedona_gdal::raster::types::DatasetOptions;
    use sedona_raster::array::RasterStructArray;
    use sedona_raster::traits::RasterRef;
    use sedona_schema::datatypes::RASTER;
    use sedona_testing::raster_spec::RasterSpec;
    use sedona_testing::testers::ScalarUdfTester;

    /// A 3x3 single-band UInt8 raster with a CRS (RasterSpec defaults to one),
    /// so GDAL export sets a projection.
    fn test_raster_spec() -> RasterSpec {
        RasterSpec::d2(3, 3)
            .transform([0.0, 1.0, 0.0, 3.0, 0.0, -1.0])
            .band_values(&[1u8, 2, 3, 4, 5, 6, 7, 8, 9])
    }

    /// Build a one-row raster `StructArray` from a spec.
    fn as_raster_array(spec: RasterSpec) -> arrow_array::StructArray {
        spec.build()
    }

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
    fn as_geotiff_produces_valid_tiff() {
        // End-to-end through the UDF: a raster scalar in, GeoTIFF binary out.
        let udf: ScalarUDF = rs_as_geotiff_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);
        let result = tester.invoke_scalar(test_raster_spec()).unwrap();
        let ScalarValue::BinaryView(Some(bytes)) = result else {
            panic!("expected a BinaryView result, got {result:?}");
        };
        assert!(bytes.len() > 4, "GeoTIFF should have content");
        assert!(
            &bytes[0..2] == b"II" || &bytes[0..2] == b"MM",
            "should be a valid TIFF header"
        );
    }

    #[test]
    fn unknown_compression_type_errors_through_udf() {
        // The compression-string parse error surfaces through the UDF itself,
        // not only from CompressionType::parse in isolation.
        let udf: ScalarUDF = rs_as_geotiff_udf().into();
        let tester = ScalarUdfTester::new(
            udf,
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Utf8),
                SedonaType::Arrow(DataType::Float64),
            ],
        );
        let err = tester
            .invoke_scalar_scalar_scalar(test_raster_spec(), "GZIP", 0.5)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("Unknown compression type: GZIP"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn as_geotiff_roundtrips_dimensions() {
        // Export to GeoTIFF, reopen the bytes with GDAL, and confirm the raster
        // dimensions survive the round trip. Uses only merged reader helpers
        // (open + dataset_to_indb_raster) — no dependency on RS_FromGDALRaster.
        with_gdal(|gdal| {
            let arr = as_raster_array(test_raster_spec());
            let rasters = RasterStructArray::try_new(&arr).unwrap();
            let raster = rasters.get(0).unwrap();

            let provider = thread_local_provider(gdal).unwrap();
            let bytes =
                RsAsGeoTiff::raster_to_geotiff(gdal, &provider, &raster, None, None, None, None)?;
            assert!(&bytes[0..2] == b"II" || &bytes[0..2] == b"MM");

            let tmp = tempfile::tempdir().unwrap();
            let path = tmp.path().join("roundtrip.tif");
            std::fs::write(&path, &bytes).unwrap();
            let dataset = gdal
                .open_ex_with_options(
                    path.to_str().unwrap(),
                    DatasetOptions {
                        open_flags: GDAL_OF_RASTER | GDAL_OF_READONLY,
                        ..Default::default()
                    },
                )
                .map_err(crate::gdal_common::convert_gdal_err)?;
            let roundtrip = crate::utils::dataset_to_indb_raster(&dataset)?;
            let rt = RasterStructArray::try_new(&roundtrip).unwrap();
            let rt_raster = rt.get(0).unwrap();

            assert_eq!(rt_raster.metadata().width(), raster.metadata().width());
            assert_eq!(rt_raster.metadata().height(), raster.metadata().height());
            assert_eq!(rt_raster.bands().len(), raster.bands().len());
            // Pixel values must survive too — this would catch predictor or
            // compression corruption that dimension checks cannot.
            assert_eq!(
                rt_raster
                    .bands()
                    .band(1)
                    .unwrap()
                    .nd_buffer()
                    .unwrap()
                    .as_contiguous()
                    .unwrap(),
                raster
                    .bands()
                    .band(1)
                    .unwrap()
                    .nd_buffer()
                    .unwrap()
                    .as_contiguous()
                    .unwrap(),
            );
            Ok::<_, datafusion_common::DataFusionError>(())
        })
        .unwrap();
    }

    #[test]
    fn jpeg_quality_maps_fraction_to_1_100() {
        // The subtle bit of the options plumbing: a 0.0-1.0 fraction (Sedona /
        // GeoTools scale) maps to GDAL's 1-100 JPEG_QUALITY.
        assert_eq!(jpeg_quality_option(0.85).unwrap(), 85);
        assert_eq!(jpeg_quality_option(1.0).unwrap(), 100);
        // GDAL rejects 0, so the bottom of the range maps to minimum quality 1.
        assert_eq!(jpeg_quality_option(0.0).unwrap(), 1);
        assert_eq!(jpeg_quality_option(0.004).unwrap(), 1);
    }

    #[test]
    fn jpeg_quality_out_of_range_errors() {
        // A 0-100 style quality (the likely mistake) errors instead of clamping
        // to maximum quality silently.
        for q in [75.0, -0.1, 1.01, f64::NAN] {
            let err = jpeg_quality_option(q).unwrap_err().to_string();
            assert!(
                err.contains("between 0.0 and 1.0"),
                "unexpected error for {q}: {err}"
            );
        }
    }

    #[test]
    fn failed_create_copy_surfaces_gdal_error() {
        // A creation-option combination GDAL rejects at CreateCopy time —
        // CCITTRLE (Huffman) only accepts 1-bit single-band data, so an
        // ordinary UInt8 raster fails. The error must surface (and the vsimem
        // guard cleans up the partial file rather than leaking it).
        with_gdal(|gdal| {
            let arr = as_raster_array(test_raster_spec());
            let rasters = RasterStructArray::try_new(&arr).unwrap();
            let raster = rasters.get(0).unwrap();
            let provider = thread_local_provider(gdal).unwrap();
            let err = RsAsGeoTiff::raster_to_geotiff(
                gdal,
                &provider,
                &raster,
                Some(CompressionType::Huffman),
                None,
                None,
                None,
            )
            .unwrap_err()
            .to_string();
            assert!(
                err.contains("Failed to create GeoTiff"),
                "unexpected error: {err}"
            );
            Ok::<_, datafusion_common::DataFusionError>(())
        })
        .unwrap();
    }

    #[test]
    fn out_of_range_quality_errors_for_every_codec() {
        // The validation is not JPEG-only: LZW ignores quality, but nonsense
        // still errors rather than being silently dropped.
        with_gdal(|gdal| {
            let arr = as_raster_array(test_raster_spec());
            let rasters = RasterStructArray::try_new(&arr).unwrap();
            let raster = rasters.get(0).unwrap();
            let provider = thread_local_provider(gdal).unwrap();
            let err = RsAsGeoTiff::raster_to_geotiff(
                gdal,
                &provider,
                &raster,
                Some(CompressionType::Lzw),
                Some(75.0),
                None,
                None,
            )
            .unwrap_err()
            .to_string();
            assert!(
                err.contains("between 0.0 and 1.0"),
                "unexpected error: {err}"
            );
            Ok::<_, datafusion_common::DataFusionError>(())
        })
        .unwrap();
    }

    #[test]
    fn predictor_matches_band_type() {
        // Horizontal differencing (2) for integer bands, floating-point
        // prediction (3) for float bands.
        let int_arr = as_raster_array(test_raster_spec());
        let int_rasters = RasterStructArray::try_new(&int_arr).unwrap();
        assert_eq!(predictor_for(&int_rasters.get(0).unwrap()).unwrap(), 2);

        let float_arr = as_raster_array(
            RasterSpec::d2(3, 3)
                .transform([0.0, 1.0, 0.0, 3.0, 0.0, -1.0])
                .band_values(&[1.5f32, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]),
        );
        let float_rasters = RasterStructArray::try_new(&float_arr).unwrap();
        assert_eq!(predictor_for(&float_rasters.get(0).unwrap()).unwrap(), 3);
    }

    #[test]
    fn float_band_exports_with_float_predictor() {
        // End-to-end: a Float32 band under DEFLATE goes out with PREDICTOR=3
        // and the values survive a roundtrip (a wrong predictor that libtiff
        // rejects, or value corruption, would fail here).
        with_gdal(|gdal| {
            let spec = RasterSpec::d2(3, 3)
                .transform([0.0, 1.0, 0.0, 3.0, 0.0, -1.0])
                .band_values(&[1.5f32, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]);
            let arr = as_raster_array(spec);
            let rasters = RasterStructArray::try_new(&arr).unwrap();
            let raster = rasters.get(0).unwrap();
            let provider = thread_local_provider(gdal).unwrap();

            let bytes = RsAsGeoTiff::raster_to_geotiff(
                gdal,
                &provider,
                &raster,
                Some(CompressionType::Deflate),
                None,
                None,
                None,
            )?;

            let tmp = tempfile::tempdir().unwrap();
            let path = tmp.path().join("float_predictor.tif");
            std::fs::write(&path, &bytes).unwrap();
            let dataset = gdal
                .open_ex_with_options(
                    path.to_str().unwrap(),
                    DatasetOptions {
                        open_flags: GDAL_OF_RASTER | GDAL_OF_READONLY,
                        ..Default::default()
                    },
                )
                .map_err(crate::gdal_common::convert_gdal_err)?;
            let roundtrip = crate::utils::dataset_to_indb_raster(&dataset)?;
            let rt = RasterStructArray::try_new(&roundtrip).unwrap();
            let rt_raster = rt.get(0).unwrap();
            assert_eq!(
                rt_raster
                    .bands()
                    .band(1)
                    .unwrap()
                    .nd_buffer()
                    .unwrap()
                    .as_contiguous()
                    .unwrap(),
                raster
                    .bands()
                    .band(1)
                    .unwrap()
                    .nd_buffer()
                    .unwrap()
                    .as_contiguous()
                    .unwrap(),
            );
            Ok::<_, datafusion_common::DataFusionError>(())
        })
        .unwrap();
    }

    /// Reopen exported GeoTIFF bytes and return band 1's (block_x, block_y).
    fn reopened_block_size(gdal: &sedona_gdal::gdal::Gdal, bytes: &[u8]) -> (usize, usize) {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("block_size.tif");
        std::fs::write(&path, bytes).unwrap();
        let dataset = gdal
            .open_ex_with_options(
                path.to_str().unwrap(),
                DatasetOptions {
                    open_flags: GDAL_OF_RASTER | GDAL_OF_READONLY,
                    ..Default::default()
                },
            )
            .unwrap();
        dataset.rasterband(1).unwrap().block_size()
    }

    #[test]
    fn tile_options_survive_export() {
        // The tiling plumbing end to end: TILED=YES + BLOCKXSIZE/BLOCKYSIZE
        // reach GDAL, and a reopened dataset exposes them as the block size.
        // TIFF tile dimensions must be multiples of 16.
        with_gdal(|gdal| {
            let arr = as_raster_array(test_raster_spec());
            let rasters = RasterStructArray::try_new(&arr).unwrap();
            let raster = rasters.get(0).unwrap();
            let provider = thread_local_provider(gdal).unwrap();

            let bytes = RsAsGeoTiff::raster_to_geotiff(
                gdal,
                &provider,
                &raster,
                None,
                None,
                Some(16),
                Some(32),
            )?;
            assert_eq!(reopened_block_size(gdal, &bytes), (16, 32));
            Ok::<_, datafusion_common::DataFusionError>(())
        })
        .unwrap();
    }

    #[test]
    fn tile_size_udf_overload_tiles_squares() {
        // RS_AsGeoTiff(raster, tile_size) writes square tiles of that size.
        let udf: ScalarUDF = rs_as_geotiff_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Int32)]);
        let result = tester
            .invoke_arrays(vec![
                Arc::new(as_raster_array(test_raster_spec())) as arrow_array::ArrayRef,
                Arc::new(arrow_array::Int32Array::from(vec![Some(16)])),
            ])
            .unwrap();
        let binary = result
            .as_any()
            .downcast_ref::<arrow_array::BinaryViewArray>()
            .unwrap();
        with_gdal(|gdal| {
            assert_eq!(reopened_block_size(gdal, binary.value(0)), (16, 16));
            Ok::<_, datafusion_common::DataFusionError>(())
        })
        .unwrap();
    }

    #[test]
    fn null_raster_yields_null_output() {
        let udf: ScalarUDF = rs_as_geotiff_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER]);
        let rasters = sedona_testing::rasters::generate_test_rasters(1, Some(0)).unwrap();
        let result = tester
            .invoke_arrays(vec![Arc::new(rasters) as arrow_array::ArrayRef])
            .unwrap();
        let binary = result
            .as_any()
            .downcast_ref::<arrow_array::BinaryViewArray>()
            .unwrap();
        assert!(binary.is_null(0), "NULL raster should export as NULL");
    }

    #[test]
    fn per_row_options_broadcast() {
        // An array raster with a per-row tile_size column: each row's option
        // reaches its own GDAL export (the broadcast path through the upfront
        // into_array casts).
        let udf: ScalarUDF = rs_as_geotiff_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Int32)]);

        // Two raster rows, tiled 16 and 32 respectively.
        let rasters = sedona_testing::rasters::generate_test_rasters(2, None).unwrap();
        let tiles = arrow_array::Int32Array::from(vec![Some(16), Some(32)]);
        let result = tester
            .invoke_arrays(vec![
                Arc::new(rasters) as arrow_array::ArrayRef,
                Arc::new(tiles),
            ])
            .unwrap();
        let binary = result
            .as_any()
            .downcast_ref::<arrow_array::BinaryViewArray>()
            .unwrap();
        // Each row's bytes are a view over its own GDAL allocation (one
        // variadic buffer per row), not a copy into a shared builder buffer.
        assert_eq!(binary.data_buffers().len(), 2);
        with_gdal(|gdal| {
            assert_eq!(reopened_block_size(gdal, binary.value(0)), (16, 16));
            assert_eq!(reopened_block_size(gdal, binary.value(1)), (32, 32));
            Ok::<_, datafusion_common::DataFusionError>(())
        })
        .unwrap();
    }

    #[test]
    fn as_geotiff_with_compression() {
        // LZW and DEFLATE both produce valid, non-empty GeoTIFFs.
        with_gdal(|gdal| {
            let arr = as_raster_array(test_raster_spec());
            let rasters = RasterStructArray::try_new(&arr).unwrap();
            let raster = rasters.get(0).unwrap();

            let provider = thread_local_provider(gdal).unwrap();
            for comp in [CompressionType::Lzw, CompressionType::Deflate] {
                let bytes = RsAsGeoTiff::raster_to_geotiff(
                    gdal,
                    &provider,
                    &raster,
                    Some(comp),
                    Some(0.75),
                    None,
                    None,
                )?;
                assert!(!bytes.is_empty(), "{comp:?} GeoTIFF should have content");
                assert!(&bytes[0..2] == b"II" || &bytes[0..2] == b"MM");
            }
            Ok::<_, datafusion_common::DataFusionError>(())
        })
        .unwrap();
    }
}
