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

use sedona_gdal::dataset::Dataset;
use sedona_gdal::errors::GdalError;
use sedona_gdal::gdal::Gdal;
use sedona_gdal::gdal_dyn_bindgen::{GDAL_OF_RASTER, GDAL_OF_READONLY, GDAL_OF_VERBOSE_ERROR};
use sedona_gdal::mem::MemDatasetBuilder;
use sedona_gdal::raster::types::DatasetOptions;
use sedona_gdal::raster::types::GdalDataType;

use std::borrow::Cow;

use sedona_raster::traits::{BandRef, RasterRef};
use sedona_schema::raster::BandDataType;

use datafusion_common::{
    arrow_datafusion_err, exec_datafusion_err, exec_err, DataFusionError, Result,
};

/// Execute a closure with a reference to the global [`Gdal`] handle,
/// converting initialization errors to [`DataFusionError`].
pub(crate) fn with_gdal<F, R>(f: F) -> Result<R>
where
    F: FnOnce(&Gdal) -> Result<R>,
{
    match sedona_gdal::global::with_global_gdal(f) {
        Ok(inner_result) => inner_result,
        Err(init_err) => Err(DataFusionError::External(Box::new(init_err))),
    }
}

/// Converts a BandDataType to the corresponding GDAL data type.
pub fn band_data_type_to_gdal(band_type: &BandDataType) -> GdalDataType {
    match band_type {
        BandDataType::UInt8 => GdalDataType::UInt8,
        BandDataType::Int8 => GdalDataType::Int8,
        BandDataType::UInt16 => GdalDataType::UInt16,
        BandDataType::Int16 => GdalDataType::Int16,
        BandDataType::UInt32 => GdalDataType::UInt32,
        BandDataType::Int32 => GdalDataType::Int32,
        BandDataType::UInt64 => GdalDataType::UInt64,
        BandDataType::Int64 => GdalDataType::Int64,
        BandDataType::Float32 => GdalDataType::Float32,
        BandDataType::Float64 => GdalDataType::Float64,
    }
}

/// Converts a GDAL data type to the corresponding BandDataType.
pub fn gdal_to_band_data_type(gdal_type: GdalDataType) -> Result<BandDataType> {
    match gdal_type {
        GdalDataType::UInt8 => Ok(BandDataType::UInt8),
        GdalDataType::Int8 => Ok(BandDataType::Int8),
        GdalDataType::UInt16 => Ok(BandDataType::UInt16),
        GdalDataType::Int16 => Ok(BandDataType::Int16),
        GdalDataType::UInt32 => Ok(BandDataType::UInt32),
        GdalDataType::Int32 => Ok(BandDataType::Int32),
        GdalDataType::UInt64 => Ok(BandDataType::UInt64),
        GdalDataType::Int64 => Ok(BandDataType::Int64),
        GdalDataType::Float32 => Ok(BandDataType::Float32),
        GdalDataType::Float64 => Ok(BandDataType::Float64),
        _ => Err(DataFusionError::NotImplemented(format!(
            "GDAL data type {:?} is not supported",
            gdal_type
        ))),
    }
}

/// Returns the byte size of a GDAL data type.
pub fn gdal_type_byte_size(gdal_type: GdalDataType) -> usize {
    gdal_type.byte_size()
}

/// Interprets bytes according to the band data type and returns the value as `f64`.
///
/// Returns an error if `bytes` does not have the expected length for `band_type`.
pub fn bytes_to_f64(bytes: &[u8], band_type: &BandDataType) -> Result<f64> {
    macro_rules! read_le_f64 {
        ($t:ty, $n:expr) => {{
            let arr: [u8; $n] = bytes.try_into().map_err(|_| {
                exec_datafusion_err!(
                    "Invalid byte slice length for type {}, expected: {}, actual: {}",
                    stringify!($t),
                    $n,
                    bytes.len()
                )
            })?;
            Ok(<$t>::from_le_bytes(arr) as f64)
        }};
    }

    match band_type {
        BandDataType::UInt8 => {
            if bytes.len() != 1 {
                return exec_err!(
                    "Invalid byte length for UInt8: expected 1, got {}",
                    bytes.len()
                );
            }
            Ok(bytes[0] as f64)
        }
        BandDataType::Int8 => {
            if bytes.len() != 1 {
                return exec_err!(
                    "Invalid byte length for Int8: expected 1, got {}",
                    bytes.len()
                );
            }
            Ok(bytes[0] as i8 as f64)
        }
        BandDataType::UInt16 => read_le_f64!(u16, 2),
        BandDataType::Int16 => read_le_f64!(i16, 2),
        BandDataType::UInt32 => read_le_f64!(u32, 4),
        BandDataType::Int32 => read_le_f64!(i32, 4),
        BandDataType::UInt64 => exec_err!(
            "Cannot convert UInt64 nodata value to f64 without potential precision loss; please handle UInt64 specially"
        ),
        BandDataType::Int64 => exec_err!(
            "Cannot convert Int64 nodata value to f64 without potential precision loss; please handle Int64 specially"
        ),
        BandDataType::Float32 => read_le_f64!(f32, 4),
        BandDataType::Float64 => read_le_f64!(f64, 8),
    }
}

/// Convert [GdalError] to [DataFusionError]
pub(crate) fn convert_gdal_err(e: GdalError) -> DataFusionError {
    DataFusionError::External(Box::new(e))
}

/// Resolve a 1-based band index into a `BandRef`, translating to the trait's
/// 0-based API and producing a clear error when out of range or zero. The
/// 1-based convention is preserved at the public boundary because callers
/// already construct band-index lists that way.
fn resolve_band<R: RasterRef + ?Sized>(
    raster: &R,
    one_based_index: usize,
) -> Result<Box<dyn BandRef + '_>> {
    if one_based_index == 0 {
        return exec_err!("Band index must be 1-based; got 0");
    }
    raster.band(one_based_index - 1).ok_or_else(|| {
        exec_datafusion_err!(
            "Band index {one_based_index} is out of range (raster has {} bands)",
            raster.num_bands()
        )
    })
}

/// This function creates a GDAL dataset backed by the MEM driver that directly
/// references the band data stored in the [RasterRef]. No data copying occurs -
/// the GDAL bands point to the same memory as the data buffer held by [RasterRef].
///
/// # Arguments
/// * `raster` - The RasterRef value
/// * `band_indices` - The indices of the bands to include in the GDAL dataset (1-based)
///
/// # Returns
/// A [`Dataset`] that provides access to the GDAL dataset.
///
/// # Errors
/// Returns an error if:
/// - Any band is N-D (not the legacy `["y","x"]` 2-D shape with identity view)
/// - Any band uses OutDb storage
/// - GDAL driver operations fail
/// - Accessing RasterRef fails
pub unsafe fn raster_ref_to_gdal_mem<R: RasterRef + ?Sized>(
    gdal: &Gdal,
    raster: &R,
    band_indices: &[usize],
) -> Result<Dataset> {
    let width = raster
        .width()
        .ok_or_else(|| exec_datafusion_err!("Raster has no width (spatial_shape missing)"))?
        as usize;
    let height = raster
        .height()
        .ok_or_else(|| exec_datafusion_err!("Raster has no height (spatial_shape missing)"))?
        as usize;

    // Create internal MEM dataset via sedona-gdal shim to avoid open dataset list contention.
    let mut mem_ds_builder = MemDatasetBuilder::new(width, height);

    // Add bands with DATAPOINTER option (zero-copy)
    //
    // Note: GDALAddBand always appends a new band, so the destination band index
    // is sequential (1..=band_indices.len()), even if the source band indices are
    // sparse (e.g. [1, 3]).
    for &src_band_index in band_indices.iter() {
        let band = resolve_band(raster, src_band_index)?;

        if !band.is_2d() {
            return exec_err!(
                "GDAL backend requires a 2-dim band; got dim_names={:?}",
                band.dim_names()
            );
        }

        if band.outdb_uri().is_some() {
            return Err(DataFusionError::NotImplemented(
                "OutDb bands are not supported in raster_to_mem_dataset".to_string(),
            ));
        }

        let band_type = band.data_type();
        let gdal_type = band_data_type_to_gdal(&band_type);
        // contiguous_data() is Cow::Borrowed for is_2d identity views; the
        // borrow points at the StructArray's backing buffer, which outlives
        // the dataset (held by the caller). For Cow::Owned the pointer would
        // dangle the moment the Cow drops, so we reject that case loudly.
        let band_data = band
            .contiguous_data()
            .map_err(|e| arrow_datafusion_err!(e))?;
        let bytes: &[u8] = match &band_data {
            Cow::Borrowed(b) => b,
            Cow::Owned(_) => {
                return exec_err!(
                    "Internal: contiguous_data must be borrowed for is_2d bands; got owned"
                );
            }
        };
        let data_ptr = bytes.as_ptr();
        unsafe {
            mem_ds_builder = mem_ds_builder.add_band(gdal_type, data_ptr as *mut u8);
        }
    }

    let dataset = unsafe {
        mem_ds_builder
            .build(gdal)
            .map_err(|e| DataFusionError::External(Box::new(e)))?
    };

    let geotransform: [f64; 6] = raster.transform().try_into().map_err(|_| {
        exec_datafusion_err!(
            "Raster transform must be exactly 6 elements; got {}",
            raster.transform().len()
        )
    })?;

    dataset
        .set_geo_transform(&geotransform)
        .map_err(convert_gdal_err)?;

    // Set projection/CRS if available
    if let Some(crs) = raster.crs() {
        dataset.set_projection(crs).map_err(convert_gdal_err)?;
    }

    for (dst_band_index, &src_band_index) in band_indices.iter().enumerate() {
        let dst_band_index = dst_band_index + 1;
        let band = resolve_band(raster, src_band_index)?;
        let band_type = band.data_type();
        if let Some(nodata_bytes) = band.nodata() {
            let raster_band = dataset
                .rasterband(dst_band_index)
                .map_err(convert_gdal_err)?;
            match band_type {
                BandDataType::UInt64 => {
                    let nodata_bytes: [u8; 8] = nodata_bytes.try_into().map_err(|_| {
                        exec_datafusion_err!("Invalid nodata byte length for UInt64")
                    })?;
                    let nodata = u64::from_le_bytes(nodata_bytes);
                    raster_band
                        .set_no_data_value_u64(Some(nodata))
                        .map_err(convert_gdal_err)?;
                }
                BandDataType::Int64 => {
                    let nodata_bytes: [u8; 8] = nodata_bytes.try_into().map_err(|_| {
                        exec_datafusion_err!("Invalid nodata byte length for Int64")
                    })?;
                    let nodata = i64::from_le_bytes(nodata_bytes);
                    raster_band
                        .set_no_data_value_i64(Some(nodata))
                        .map_err(convert_gdal_err)?;
                }
                _ => {
                    let nodata = bytes_to_f64(nodata_bytes, &band_type)?;
                    raster_band
                        .set_no_data_value(Some(nodata))
                        .map_err(convert_gdal_err)?;
                }
            }
        }
    }

    Ok(dataset)
}

pub fn raster_ref_to_gdal_empty<R: RasterRef + ?Sized>(gdal: &Gdal, raster: &R) -> Result<Dataset> {
    unsafe {
        // SAFETY: raster_ref_to_gdal_mem is safe to call with an empty band list. The
        // returned dataset will have zero bands and references no external memory.
        raster_ref_to_gdal_mem(gdal, raster, &[])
    }
}

/// Interpret optional nodata bytes according to the band data type and return an Option<f64>.
/// Returns `None` if `nodata_bytes` is `None` or cannot be parsed for the given type.
pub fn nodata_bytes_to_f64(nodata_bytes: Option<&[u8]>, band_type: &BandDataType) -> Option<f64> {
    let bytes = nodata_bytes?;
    bytes_to_f64(bytes, band_type).ok()
}

/// Convert a f64 nodata value into a byte vector appropriate for the given band type.
pub fn nodata_f64_to_bytes(nodata: f64, band_type: &BandDataType) -> Vec<u8> {
    match band_type {
        BandDataType::UInt8 => vec![(nodata as u8)],
        BandDataType::Int8 => (nodata as i8).to_le_bytes().to_vec(),
        BandDataType::UInt16 => (nodata as u16).to_le_bytes().to_vec(),
        BandDataType::Int16 => (nodata as i16).to_le_bytes().to_vec(),
        BandDataType::UInt32 => (nodata as u32).to_le_bytes().to_vec(),
        BandDataType::Int32 => (nodata as i32).to_le_bytes().to_vec(),
        BandDataType::UInt64 => (nodata as u64).to_le_bytes().to_vec(),
        BandDataType::Int64 => (nodata as i64).to_le_bytes().to_vec(),
        BandDataType::Float32 => (nodata as f32).to_le_bytes().to_vec(),
        BandDataType::Float64 => nodata.to_le_bytes().to_vec(),
    }
}

/// Open an out-db raster source as a GDAL dataset, normalizing the URL to a GDAL VSI path if needed.
pub fn open_gdal_dataset(gdal: &Gdal, url: &str, open_options: Option<&[&str]>) -> Result<Dataset> {
    let normalized_url = normalize_outdb_source_path(url);
    gdal.open_ex_with_options(
        &normalized_url,
        DatasetOptions {
            open_flags: GDAL_OF_RASTER | GDAL_OF_READONLY | GDAL_OF_VERBOSE_ERROR,
            open_options,
            ..Default::default()
        },
    )
    .map_err(convert_gdal_err)
}

/// Normalize out-db raster URLs to GDAL VSI paths.
///
/// Supported translations:
/// - `s3://bucket/key` -> `/vsis3/bucket/key`
/// - `s3a://bucket/key` -> `/vsis3/bucket/key`
/// - `gs://bucket/key` -> `/vsigs/bucket/key`
/// - `gcs://bucket/key` -> `/vsigs/bucket/key`
/// - `az://container/key` -> `/vsiaz/container/key`
/// - `wasb://container/key` -> `/vsiaz/container/key`
/// - `wasbs://container/key` -> `/vsiaz/container/key`
/// - `abfs://filesystem/path` -> `/vsiadls/filesystem/path`
/// - `abfss://filesystem/path` -> `/vsiadls/filesystem/path`
/// - `http://...` -> `/vsicurl/http://...`
/// - `https://...` -> `/vsicurl/https://...`
///
/// Existing GDAL VSI paths (starting with `/vsi`) and non-matching paths are
/// returned unchanged.
pub(crate) fn normalize_outdb_source_path(path: &str) -> String {
    if path
        .get(..4)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("/vsi"))
    {
        return path.to_string();
    }

    if let Some(rest) = strip_scheme_prefix(path, "s3://") {
        return format!("/vsis3/{rest}");
    }

    if let Some(rest) = strip_scheme_prefix(path, "s3a://") {
        return format!("/vsis3/{rest}");
    }

    if let Some(rest) = strip_scheme_prefix(path, "gs://") {
        return format!("/vsigs/{rest}");
    }

    if let Some(rest) = strip_scheme_prefix(path, "gcs://") {
        return format!("/vsigs/{rest}");
    }

    if let Some(rest) = strip_scheme_prefix(path, "az://") {
        return format!("/vsiaz/{rest}");
    }

    if let Some(rest) = strip_scheme_prefix(path, "wasb://") {
        return format!("/vsiaz/{rest}");
    }

    if let Some(rest) = strip_scheme_prefix(path, "wasbs://") {
        return format!("/vsiaz/{rest}");
    }

    if let Some(rest) = strip_scheme_prefix(path, "abfs://") {
        return format!("/vsiadls/{rest}");
    }

    if let Some(rest) = strip_scheme_prefix(path, "abfss://") {
        return format!("/vsiadls/{rest}");
    }

    if strip_scheme_prefix(path, "http://").is_some()
        || strip_scheme_prefix(path, "https://").is_some()
    {
        return format!("/vsicurl/{path}");
    }

    path.to_string()
}

fn strip_scheme_prefix<'a>(value: &'a str, scheme_prefix: &str) -> Option<&'a str> {
    if value
        .get(..scheme_prefix.len())
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case(scheme_prefix))
    {
        Some(&value[scheme_prefix.len()..])
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::StructArray;
    use sedona_raster::array::RasterStructArray;
    use sedona_raster::builder::RasterBuilder;

    /// Affine: (origin_x, scale_x, skew_x, origin_y, skew_y, scale_y) — same
    /// element order as the canonical 6-element transform.
    type Transform2d = (f64, f64, f64, f64, f64, f64);

    /// 2D in-db test band: `(data_type, nodata_bytes_or_none, raw_pixel_bytes)`.
    type InDbTestBand2d = (BandDataType, Option<Vec<u8>>, Vec<u8>);

    fn build_indb_raster_2d(
        width: u64,
        height: u64,
        transform: Transform2d,
        crs: Option<&str>,
        bands: &[InDbTestBand2d],
    ) -> StructArray {
        let (origin_x, scale_x, skew_x, origin_y, skew_y, scale_y) = transform;
        let mut builder = RasterBuilder::new(1);
        builder
            .start_raster_2d(
                width, height, origin_x, origin_y, scale_x, scale_y, skew_x, skew_y, crs,
            )
            .unwrap();
        for (data_type, nodata, data) in bands {
            builder
                .start_band_2d(*data_type, nodata.as_deref())
                .unwrap();
            builder.band_data_writer().append_value(data);
            builder.finish_band().unwrap();
        }
        builder.finish_raster().unwrap();
        builder.finish().unwrap()
    }

    fn build_outdb_raster_2d(
        width: u64,
        height: u64,
        transform: Transform2d,
        crs: Option<&str>,
        outdb_uri: &str,
        data_type: BandDataType,
        nodata: Option<&[u8]>,
    ) -> StructArray {
        let (origin_x, scale_x, skew_x, origin_y, skew_y, scale_y) = transform;
        let mut builder = RasterBuilder::new(1);
        builder
            .start_raster_2d(
                width, height, origin_x, origin_y, scale_x, scale_y, skew_x, skew_y, crs,
            )
            .unwrap();
        builder
            .start_band(
                None,
                &["y", "x"],
                &[height, width],
                data_type,
                nodata,
                Some(outdb_uri),
                Some("geotiff"),
            )
            .unwrap();
        builder.band_data_writer().append_value([]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        builder.finish().unwrap()
    }

    fn read_band_u64(dataset: &Dataset, band_index: usize, size: (usize, usize)) -> Vec<u64> {
        let band = dataset.rasterband(band_index).unwrap();
        let buffer = band.read_as::<u64>((0, 0), size, size, None).unwrap();
        buffer.data().to_vec()
    }

    fn read_band_i64(dataset: &Dataset, band_index: usize, size: (usize, usize)) -> Vec<i64> {
        let band = dataset.rasterband(band_index).unwrap();
        let buffer = band.read_as::<i64>((0, 0), size, size, None).unwrap();
        buffer.data().to_vec()
    }

    fn assert_wgs84_projection(dataset: &Dataset) {
        assert!(dataset
            .projection()
            .contains("AUTHORITY[\"EPSG\",\"4326\"]"));
    }

    #[test]
    fn test_band_data_type_to_gdal() {
        assert_eq!(
            band_data_type_to_gdal(&BandDataType::UInt8),
            GdalDataType::UInt8
        );
        assert_eq!(
            band_data_type_to_gdal(&BandDataType::Int8),
            GdalDataType::Int8
        );
        assert_eq!(
            band_data_type_to_gdal(&BandDataType::UInt16),
            GdalDataType::UInt16
        );
        assert_eq!(
            band_data_type_to_gdal(&BandDataType::Int16),
            GdalDataType::Int16
        );
        assert_eq!(
            band_data_type_to_gdal(&BandDataType::UInt32),
            GdalDataType::UInt32
        );
        assert_eq!(
            band_data_type_to_gdal(&BandDataType::Int32),
            GdalDataType::Int32
        );
        assert_eq!(
            band_data_type_to_gdal(&BandDataType::UInt64),
            GdalDataType::UInt64
        );
        assert_eq!(
            band_data_type_to_gdal(&BandDataType::Int64),
            GdalDataType::Int64
        );
        assert_eq!(
            band_data_type_to_gdal(&BandDataType::Float32),
            GdalDataType::Float32
        );
        assert_eq!(
            band_data_type_to_gdal(&BandDataType::Float64),
            GdalDataType::Float64
        );
    }

    #[test]
    fn test_bytes_to_f64() {
        // UInt8
        assert_eq!(bytes_to_f64(&[255u8], &BandDataType::UInt8).unwrap(), 255.0);
        assert_eq!(bytes_to_f64(&[0u8], &BandDataType::UInt8).unwrap(), 0.0);

        // Int16
        let val: i16 = -32768;
        assert_eq!(
            bytes_to_f64(&val.to_le_bytes(), &BandDataType::Int16).unwrap(),
            -32768.0
        );

        // Int8
        let val: i8 = -7;
        assert_eq!(
            bytes_to_f64(&val.to_le_bytes(), &BandDataType::Int8).unwrap(),
            -7.0
        );

        // UInt64 should fail explicitly because conversion to f64 is lossy.
        let val: u64 = 42;
        let err = bytes_to_f64(&val.to_le_bytes(), &BandDataType::UInt64)
            .err()
            .unwrap();
        assert!(err.to_string().contains(
            "Cannot convert UInt64 nodata value to f64 without potential precision loss"
        ));

        // Int64 should fail explicitly because conversion to f64 is lossy.
        let val: i64 = -42;
        let err = bytes_to_f64(&val.to_le_bytes(), &BandDataType::Int64)
            .err()
            .unwrap();
        assert!(err
            .to_string()
            .contains("Cannot convert Int64 nodata value to f64 without potential precision loss"));

        // Float32
        let val: f32 = -9999.0;
        assert_eq!(
            bytes_to_f64(&val.to_le_bytes(), &BandDataType::Float32).unwrap(),
            -9999.0
        );

        // Float64
        let val: f64 = f64::NAN;
        let result = bytes_to_f64(&val.to_le_bytes(), &BandDataType::Float64);
        assert!(result.unwrap().is_nan());

        // Wrong length
        assert!(bytes_to_f64(&[1, 2, 3], &BandDataType::UInt8).is_err());
    }

    #[test]
    fn test_normalize_outdb_source_path_local_and_vsi_paths() {
        assert_eq!(
            normalize_outdb_source_path("/tmp/test.tif"),
            "/tmp/test.tif"
        );
        assert_eq!(
            normalize_outdb_source_path("relative/path/test.tif"),
            "relative/path/test.tif"
        );
        assert_eq!(
            normalize_outdb_source_path("/vsis3/my-bucket/test.tif"),
            "/vsis3/my-bucket/test.tif"
        );
        assert_eq!(
            normalize_outdb_source_path("/vsicurl/https://example.com/a.tif"),
            "/vsicurl/https://example.com/a.tif"
        );
        assert_eq!(
            normalize_outdb_source_path("/vsigs/my-bucket/test.tif"),
            "/vsigs/my-bucket/test.tif"
        );
        assert_eq!(
            normalize_outdb_source_path("/vsiaz/my-container/test.tif"),
            "/vsiaz/my-container/test.tif"
        );
        assert_eq!(
            normalize_outdb_source_path("/vsiadls/my-filesystem/test.tif"),
            "/vsiadls/my-filesystem/test.tif"
        );
    }

    #[test]
    fn test_normalize_outdb_source_path_s3_and_s3a() {
        assert_eq!(
            normalize_outdb_source_path("s3://bucket/path/to/test.tif"),
            "/vsis3/bucket/path/to/test.tif"
        );
        assert_eq!(
            normalize_outdb_source_path("s3a://bucket/path/to/test.tif"),
            "/vsis3/bucket/path/to/test.tif"
        );
        assert_eq!(
            normalize_outdb_source_path("S3A://bucket/path/to/test.tif"),
            "/vsis3/bucket/path/to/test.tif"
        );
    }

    #[test]
    fn test_normalize_outdb_source_path_gcs_schemes() {
        assert_eq!(
            normalize_outdb_source_path("gs://bucket/path/to/test.tif"),
            "/vsigs/bucket/path/to/test.tif"
        );
        assert_eq!(
            normalize_outdb_source_path("gcs://bucket/path/to/test.tif"),
            "/vsigs/bucket/path/to/test.tif"
        );
        assert_eq!(
            normalize_outdb_source_path("GCS://bucket/path/to/test.tif"),
            "/vsigs/bucket/path/to/test.tif"
        );
    }

    #[test]
    fn test_normalize_outdb_source_path_azure_schemes() {
        assert_eq!(
            normalize_outdb_source_path("az://container/path/to/test.tif"),
            "/vsiaz/container/path/to/test.tif"
        );
        assert_eq!(
            normalize_outdb_source_path("wasb://container/path/to/test.tif"),
            "/vsiaz/container/path/to/test.tif"
        );
        assert_eq!(
            normalize_outdb_source_path("wasbs://container/path/to/test.tif"),
            "/vsiaz/container/path/to/test.tif"
        );
        assert_eq!(
            normalize_outdb_source_path("abfs://filesystem/path/to/test.tif"),
            "/vsiadls/filesystem/path/to/test.tif"
        );
        assert_eq!(
            normalize_outdb_source_path("abfss://filesystem/path/to/test.tif"),
            "/vsiadls/filesystem/path/to/test.tif"
        );
    }

    #[test]
    fn test_normalize_outdb_source_path_http_and_https() {
        assert_eq!(
            normalize_outdb_source_path("http://example.com/raster.tif"),
            "/vsicurl/http://example.com/raster.tif"
        );
        assert_eq!(
            normalize_outdb_source_path("https://example.com/raster.tif?token=abc#fragment"),
            "/vsicurl/https://example.com/raster.tif?token=abc#fragment"
        );
        assert_eq!(
            normalize_outdb_source_path("HTTPS://example.com/raster.tif"),
            "/vsicurl/HTTPS://example.com/raster.tif"
        );
    }

    #[test]
    fn test_raster_ref_to_gdal_empty_preserves_metadata_and_crs() {
        let raster_array = build_indb_raster_2d(
            3,
            2,
            (10.0, 0.5, 0.1, 20.0, -0.2, -0.5),
            Some("EPSG:4326"),
            &[],
        );
        let raster_struct_array = RasterStructArray::new(&raster_array);
        let raster = raster_struct_array.get(0).unwrap();

        with_gdal(|gdal| {
            let dataset = raster_ref_to_gdal_empty(gdal, &raster)?;
            assert_eq!(dataset.raster_size(), (3, 2));
            assert_eq!(dataset.raster_count(), 0);
            assert_eq!(
                dataset.geo_transform().unwrap(),
                [10.0, 0.5, 0.1, 20.0, -0.2, -0.5]
            );
            assert_wgs84_projection(&dataset);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn test_raster_ref_to_gdal_mem_preserves_band_order_data_and_nodata() {
        let uint64_pixels = [1u64, 2, 3, 4]
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .collect::<Vec<_>>();
        let uint8_pixels = vec![9u8, 8u8, 7u8, 6u8];
        let int64_pixels = [-4i64, -3, -2, -1]
            .into_iter()
            .flat_map(i64::to_le_bytes)
            .collect::<Vec<_>>();
        let uint64_nodata = 9_007_199_254_740_992u64;
        let int64_nodata = -9_007_199_254_740_992i64;
        let raster_array = build_indb_raster_2d(
            2,
            2,
            (5.0, 2.0, 0.0, 8.0, 0.0, -2.0),
            Some("EPSG:4326"),
            &[
                (
                    BandDataType::UInt64,
                    Some(uint64_nodata.to_le_bytes().to_vec()),
                    uint64_pixels,
                ),
                (BandDataType::UInt8, Some(vec![255u8]), uint8_pixels),
                (
                    BandDataType::Int64,
                    Some(int64_nodata.to_le_bytes().to_vec()),
                    int64_pixels,
                ),
            ],
        );
        let raster_struct_array = RasterStructArray::new(&raster_array);
        let raster = raster_struct_array.get(0).unwrap();

        with_gdal(|gdal| {
            let dataset = unsafe { raster_ref_to_gdal_mem(gdal, &raster, &[3, 1])? };
            assert_eq!(dataset.raster_size(), (2, 2));
            assert_eq!(dataset.raster_count(), 2);
            assert_eq!(
                dataset.geo_transform().unwrap(),
                [5.0, 2.0, 0.0, 8.0, 0.0, -2.0]
            );
            assert_wgs84_projection(&dataset);

            let first_band = dataset.rasterband(1).unwrap();
            assert_eq!(first_band.no_data_value(), Some(int64_nodata as f64));
            assert_eq!(read_band_i64(&dataset, 1, (2, 2)), vec![-4, -3, -2, -1]);

            let second_band = dataset.rasterband(2).unwrap();
            assert_eq!(second_band.no_data_value(), Some(uint64_nodata as f64));
            assert_eq!(read_band_u64(&dataset, 2, (2, 2)), vec![1, 2, 3, 4]);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn test_raster_ref_to_gdal_mem_rejects_outdb_bands() {
        let raster_array = build_outdb_raster_2d(
            1,
            1,
            (0.0, 1.0, 0.0, 1.0, 0.0, -1.0),
            None,
            "/tmp/test.tif",
            BandDataType::UInt8,
            Some(&[0u8]),
        );
        let raster_struct_array = RasterStructArray::new(&raster_array);
        let raster = raster_struct_array.get(0).unwrap();

        let err = with_gdal(|gdal| unsafe { raster_ref_to_gdal_mem(gdal, &raster, &[1]) })
            .err()
            .unwrap();
        assert!(err.to_string().contains("OutDb bands are not supported"));
    }

    #[test]
    fn test_raster_ref_to_gdal_mem_rejects_nd_bands() {
        // Build a 3-D in-db band shaped ["time","y","x"] over a 2-D raster.
        // The N-D guard should fire before any GDAL call.
        let mut builder = RasterBuilder::new(1);
        builder
            .start_raster_2d(2, 2, 0.0, 2.0, 1.0, -1.0, 0.0, 0.0, None)
            .unwrap();
        builder
            .start_band(
                None,
                &["time", "y", "x"],
                &[3, 2, 2],
                BandDataType::UInt8,
                None,
                None,
                None,
            )
            .unwrap();
        builder
            .band_data_writer()
            .append_value(vec![0u8; 3 * 2 * 2]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        let raster_array = builder.finish().unwrap();
        let raster_struct_array = RasterStructArray::new(&raster_array);
        let raster = raster_struct_array.get(0).unwrap();

        let err = with_gdal(|gdal| unsafe { raster_ref_to_gdal_mem(gdal, &raster, &[1]) })
            .err()
            .unwrap();
        assert!(
            err.to_string().contains("requires a 2-dim band"),
            "got: {err}"
        );
    }
}
