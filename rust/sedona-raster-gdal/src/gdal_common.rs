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
use sedona_gdal::geo_transform::GeoTransform;
use sedona_gdal::mem::MemDatasetBuilder;
use sedona_gdal::raster::rasterband::RasterBand;
use sedona_gdal::raster::types::DatasetOptions;
use sedona_gdal::raster::types::GdalDataType;

use sedona_raster::traits::{is_spatial_dim_pair, MetadataRef, RasterMetadata, RasterRef};
use sedona_schema::raster::{BandDataType, StorageType};

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

/// Convert raster metadata into GDAL's six-element geo-transform.
///
/// GDAL stores geo-transforms as
/// `[origin_x, pixel_width, rotation_x, origin_y, rotation_y, pixel_height]`.
pub(crate) trait ToGdalGeoTransform {
    fn to_gdal_geotransform(&self) -> GeoTransform;
}

impl<T: MetadataRef + ?Sized> ToGdalGeoTransform for T {
    fn to_gdal_geotransform(&self) -> GeoTransform {
        [
            self.upper_left_x(),
            self.scale_x(),
            self.skew_x(),
            self.upper_left_y(),
            self.skew_y(),
            self.scale_y(),
        ]
    }
}

/// Reconstruct raster metadata from a GDAL six-element geo-transform and raster dimensions.
pub(crate) trait RasterMetadataFromGdalGeoTransform {
    fn to_raster_metadata(&self, width: usize, height: usize) -> RasterMetadata;
}

impl RasterMetadataFromGdalGeoTransform for GeoTransform {
    fn to_raster_metadata(&self, width: usize, height: usize) -> RasterMetadata {
        RasterMetadata {
            width: width as i64,
            height: height as i64,
            upperleft_x: self[0],
            upperleft_y: self[3],
            scale_x: self[1],
            scale_y: self[5],
            skew_x: self[2],
            skew_y: self[4],
        }
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
/// - Any band uses OutDb storage
/// - GDAL driver operations fail
/// - Accessing RasterRef fails
pub unsafe fn raster_ref_to_gdal_mem<R: RasterRef + ?Sized>(
    gdal: &Gdal,
    raster: &R,
    band_indices: &[usize],
) -> Result<Dataset> {
    let metadata = raster.metadata();
    let bands = raster.bands();

    let width = metadata.width() as usize;
    let height = metadata.height() as usize;

    // Create internal MEM dataset via sedona-gdal shim to avoid open dataset list contention.
    let mut mem_ds_builder = MemDatasetBuilder::new(width, height);

    // Add bands with DATAPOINTER option (zero-copy)
    //
    // Note: GDALAddBand always appends a new band, so the destination band index
    // is sequential (1..=band_indices.len()), even if the source band indices are
    // sparse (e.g. [1, 3]).
    for &src_band_index in band_indices.iter() {
        let band = bands
            .band(src_band_index)
            .map_err(|e| arrow_datafusion_err!(e))?;

        // An N-D band's trailing two axes must be the spatial (y, x) pair; the
        // non-spatial axes become a stack of 2-D planes, one GDAL band each. A
        // plain 2-D band is just the single-plane case.
        let dims = band.dim_names();
        let ndim = dims.len();
        if ndim < 2 || !is_spatial_dim_pair(dims[ndim - 2], dims[ndim - 1]) {
            return exec_err!(
                "GDAL backend requires a band whose trailing two dims are a \
                 spatial (y, x) pair; got dim_names={dims:?}"
            );
        }

        // The plane's 2-D extent must equal the MEM dataset's (and the raster's
        // spatial grid); otherwise the per-plane byte slicing below would
        // disagree with the GDAL band size and silently mis-stack the planes.
        // `finish_raster` already enforces this for builder-made rasters, but
        // re-check so the public bridge stays sound for any `RasterRef`.
        let shape = band.shape();
        if shape[ndim - 2] as usize != height || shape[ndim - 1] as usize != width {
            return exec_err!(
                "band spatial extent {}x{} does not match the raster grid \
                 {width}x{height} (dim_names={dims:?})",
                shape[ndim - 1],
                shape[ndim - 2]
            );
        }

        if band.metadata().storage_type()? != StorageType::InDb {
            return Err(DataFusionError::NotImplemented(
                "OutDb bands are not supported by raster_ref_to_gdal_mem".to_string(),
            ));
        }

        let band_metadata = band.metadata();
        let band_type = band_metadata.data_type()?;
        let gdal_type = band_data_type_to_gdal(&band_type);
        // `as_contiguous()` borrows the bytes zero-copy (erroring on a strided
        // view); GDAL holds the pointer, so `raster` must outlive the dataset.
        // Because (y, x) are innermost, each plane is a contiguous sub-range, so
        // the zero-copy DATAPOINTER holds per plane.
        let band_bytes = band.nd_buffer().and_then(|ndb| ndb.as_contiguous())?;
        let plane_bytes = width * height * band_type.byte_size();
        if plane_bytes == 0 || band_bytes.len() % plane_bytes != 0 {
            return exec_err!(
                "band byte length {} is not a multiple of the {width}x{height} \
                 plane size (dim_names={dims:?})",
                band_bytes.len()
            );
        }
        let plane_count = band_bytes.len() / plane_bytes;
        for plane in 0..plane_count {
            let off = plane * plane_bytes;
            let data_ptr: *const u8 = band_bytes[off..off + plane_bytes].as_ptr();
            unsafe {
                mem_ds_builder = mem_ds_builder.add_band(gdal_type, data_ptr as *mut u8);
            }
        }
    }

    let dataset = unsafe {
        mem_ds_builder
            .build(gdal)
            .map_err(|e| DataFusionError::External(Box::new(e)))?
    };

    let geotransform = metadata.to_gdal_geotransform();

    dataset
        .set_geo_transform(&geotransform)
        .map_err(convert_gdal_err)?;

    // Set projection/CRS if available
    if let Some(crs) = raster.crs() {
        dataset.set_projection(crs).map_err(convert_gdal_err)?;
    }

    // Nodata is per source band, shared across all its planes. Walk the dst
    // bands in the same band-major / plane order as the add loop above.
    let mut dst_band_index = 0usize;
    for &src_band_index in band_indices.iter() {
        let band = bands
            .band(src_band_index)
            .map_err(|e| arrow_datafusion_err!(e))?;
        let band_metadata = band.metadata();
        let band_type = band_metadata.data_type()?;
        let plane_bytes = width * height * band_type.byte_size();
        let band_bytes = band.nd_buffer().and_then(|ndb| ndb.as_contiguous())?;
        let plane_count = band_bytes.len() / plane_bytes;
        let nodata = band_metadata.nodata_value();
        for _ in 0..plane_count {
            dst_band_index += 1;
            if let Some(nodata_bytes) = nodata {
                let raster_band = dataset
                    .rasterband(dst_band_index)
                    .map_err(convert_gdal_err)?;
                set_band_nodata_from_bytes(&raster_band, Some(nodata_bytes))?;
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

/// The N-D structure that [`raster_ref_to_gdal_mem`] flattens away when it
/// stacks each band's non-spatial planes into a flat GDAL band list.
///
/// GDAL is 2-D-planar and oblivious to the extra dimensions, so this layout is
/// the out-of-band record needed to regroup a GDAL dataset's bands back into
/// N-D raster bands (see `gdal_dataset_to_nd_raster`). It is derived from the
/// *input* raster and is invariant under spatial-only GDAL ops (warp /
/// reproject / resample), which preserve band count and order and touch only
/// the x/y extent — the spatial extent is read from the output dataset, not
/// from here.
#[derive(Debug, Clone)]
pub struct GdalBandLayout {
    /// One entry per source band, in the order passed to
    /// [`raster_ref_to_gdal_mem`]. GDAL bands are laid out band-major then
    /// plane-major.
    pub bands: Vec<GdalBandPlan>,
}

/// One source band's non-spatial structure within a [`GdalBandLayout`].
#[derive(Debug, Clone)]
pub struct GdalBandPlan {
    pub name: Option<String>,
    /// Full dim-name list, e.g. `["time", "y", "x"]`.
    pub dim_names: Vec<String>,
    /// Sizes of the non-spatial (leading) axes; empty for a plain 2-D band.
    pub nonspatial_shape: Vec<i64>,
    /// Number of 2-D planes (`Π nonspatial_shape`, `1` for a 2-D band) — the
    /// count of consecutive GDAL bands this source band owns.
    pub plane_count: usize,
    pub data_type: BandDataType,
    pub nodata: Option<Vec<u8>>,
}

impl GdalBandLayout {
    /// Derive the layout from `raster`'s selected bands. Order and plane counts
    /// match exactly what [`raster_ref_to_gdal_mem`] emits for the same
    /// `band_indices`.
    pub fn from_raster<R: RasterRef + ?Sized>(raster: &R, band_indices: &[usize]) -> Result<Self> {
        let bands = raster.bands();
        let mut plans = Vec::with_capacity(band_indices.len());
        for &i in band_indices {
            let band = bands.band(i).map_err(|e| arrow_datafusion_err!(e))?;
            let dim_names: Vec<String> = band.dim_names().iter().map(|s| s.to_string()).collect();
            let ndim = dim_names.len();
            if ndim < 2 || !is_spatial_dim_pair(&dim_names[ndim - 2], &dim_names[ndim - 1]) {
                return exec_err!(
                    "GDAL backend requires a band whose trailing two dims are a \
                     spatial (y, x) pair; got dim_names={dim_names:?}"
                );
            }
            let nonspatial_shape: Vec<i64> = band.shape()[..ndim - 2].to_vec();
            let plane_count = nonspatial_shape.iter().product::<i64>() as usize;
            plans.push(GdalBandPlan {
                // `band_indices` are 1-based (the `Bands` wrapper convention used
                // by `raster_ref_to_gdal_mem`), but `band_name` is 0-based.
                name: raster.band_name(i - 1).map(|s| s.to_string()),
                dim_names,
                nonspatial_shape,
                plane_count,
                data_type: band.data_type(),
                nodata: band.nodata().map(|b| b.to_vec()),
            });
        }
        Ok(Self { bands: plans })
    }
}

/// Interpret optional nodata bytes according to the band data type and return an Option<f64>.
/// Returns `None` if `nodata_bytes` is `None` or cannot be parsed for the given type.
pub fn nodata_bytes_to_f64(nodata_bytes: Option<&[u8]>, band_type: &BandDataType) -> Option<f64> {
    let bytes = nodata_bytes?;
    bytes_to_f64(bytes, band_type).ok()
}

/// Read a GDAL band's nodata value into a byte vector using the band's native type.
pub fn band_nodata_to_bytes(band: &RasterBand<'_>) -> Result<Option<Vec<u8>>> {
    let band_type = gdal_to_band_data_type(band.band_type())?;

    Ok(match band_type {
        BandDataType::UInt64 => band
            .no_data_value_u64()
            .map(|nodata| nodata.to_le_bytes().to_vec()),
        BandDataType::Int64 => band
            .no_data_value_i64()
            .map(|nodata| nodata.to_le_bytes().to_vec()),
        _ => band
            .no_data_value()
            .map(|nodata| nodata_f64_to_bytes(nodata, &band_type)),
    })
}

/// Set a GDAL band's nodata value from stored bytes using the band's native type.
pub fn set_band_nodata_from_bytes(
    band: &RasterBand<'_>,
    nodata_bytes: Option<&[u8]>,
) -> Result<()> {
    let band_type = gdal_to_band_data_type(band.band_type())?;

    match (nodata_bytes, band_type) {
        (Some(bytes), BandDataType::UInt64) => {
            let bytes: [u8; 8] = bytes
                .try_into()
                .map_err(|_| exec_datafusion_err!("Invalid nodata byte length for UInt64"))?;
            band.set_no_data_value_u64(Some(u64::from_le_bytes(bytes)))
                .map_err(convert_gdal_err)
        }
        (Some(bytes), BandDataType::Int64) => {
            let bytes: [u8; 8] = bytes
                .try_into()
                .map_err(|_| exec_datafusion_err!("Invalid nodata byte length for Int64"))?;
            band.set_no_data_value_i64(Some(i64::from_le_bytes(bytes)))
                .map_err(convert_gdal_err)
        }
        (Some(bytes), band_type) => band
            .set_no_data_value(Some(bytes_to_f64(bytes, &band_type)?))
            .map_err(convert_gdal_err),
        (None, BandDataType::UInt64) => band.set_no_data_value_u64(None).map_err(convert_gdal_err),
        (None, BandDataType::Int64) => band.set_no_data_value_i64(None).map_err(convert_gdal_err),
        (None, _) => band.set_no_data_value(None).map_err(convert_gdal_err),
    }
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

    use sedona_raster::array::RasterStructArray;
    use sedona_raster::builder::RasterBuilder;
    use sedona_raster::traits::{BandMetadata, RasterMetadata};
    use sedona_schema::raster::StorageType;
    use sedona_testing::rasters::{build_in_db_raster, InDbTestBand};

    fn single_raster<'a>(
        raster_array: &'a arrow_array::StructArray,
    ) -> impl sedona_raster::traits::RasterRef + 'a {
        RasterStructArray::try_new(raster_array)
            .unwrap()
            .get(0)
            .unwrap()
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
    fn test_to_gdal_geotransform() {
        let metadata = RasterMetadata {
            width: 3,
            height: 2,
            upperleft_x: 10.0,
            upperleft_y: 20.0,
            scale_x: 0.5,
            scale_y: -0.5,
            skew_x: 0.1,
            skew_y: -0.2,
        };

        assert_eq!(
            metadata.to_gdal_geotransform(),
            [10.0, 0.5, 0.1, 20.0, -0.2, -0.5]
        );
    }

    #[test]
    fn test_to_raster_metadata() {
        let geotransform: GeoTransform = [12.5, 0.25, 0.75, -8.0, -0.5, -2.0];
        let metadata = geotransform.to_raster_metadata(4, 3);

        assert_eq!(metadata.width, 4);
        assert_eq!(metadata.height, 3);
        assert_eq!(metadata.upperleft_x, 12.5);
        assert_eq!(metadata.upperleft_y, -8.0);
        assert_eq!(metadata.scale_x, 0.25);
        assert_eq!(metadata.scale_y, -2.0);
        assert_eq!(metadata.skew_x, 0.75);
        assert_eq!(metadata.skew_y, -0.5);
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
        let metadata = RasterMetadata {
            width: 3,
            height: 2,
            upperleft_x: 10.0,
            upperleft_y: 20.0,
            scale_x: 0.5,
            scale_y: -0.5,
            skew_x: 0.1,
            skew_y: -0.2,
        };
        let raster_array = build_in_db_raster(metadata, Some("EPSG:4326"), &[]);
        let raster = single_raster(&raster_array);

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
        let metadata = RasterMetadata {
            width: 2,
            height: 2,
            upperleft_x: 5.0,
            upperleft_y: 8.0,
            scale_x: 2.0,
            scale_y: -2.0,
            skew_x: 0.0,
            skew_y: 0.0,
        };
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
        let raster_array = build_in_db_raster(
            metadata,
            Some("EPSG:4326"),
            &[
                InDbTestBand {
                    datatype: BandDataType::UInt64,
                    nodata_value: Some(uint64_nodata.to_le_bytes().to_vec()),
                    data: uint64_pixels,
                },
                InDbTestBand {
                    datatype: BandDataType::UInt8,
                    nodata_value: Some(vec![255u8]),
                    data: uint8_pixels,
                },
                InDbTestBand {
                    datatype: BandDataType::Int64,
                    nodata_value: Some(int64_nodata.to_le_bytes().to_vec()),
                    data: int64_pixels,
                },
            ],
        );
        let raster = single_raster(&raster_array);

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
        let mut builder = RasterBuilder::new(1);
        let metadata = RasterMetadata {
            width: 1,
            height: 1,
            upperleft_x: 0.0,
            upperleft_y: 1.0,
            scale_x: 1.0,
            scale_y: -1.0,
            skew_x: 0.0,
            skew_y: 0.0,
        };
        builder.start_raster(&metadata, None).unwrap();
        builder
            .start_band(BandMetadata {
                datatype: BandDataType::UInt8,
                nodata_value: Some(vec![0u8]),
                storage_type: StorageType::OutDbRef,
                outdb_url: Some("/tmp/test.tif".to_string()),
                outdb_band_id: Some(1),
            })
            .unwrap();
        builder.band_data_writer().append_value([]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        let raster_array = builder.finish().unwrap();
        let raster = single_raster(&raster_array);

        let err = with_gdal(|gdal| unsafe { raster_ref_to_gdal_mem(gdal, &raster, &[1]) })
            .err()
            .unwrap();
        assert!(err.to_string().contains("OutDb bands are not supported"));
    }

    #[test]
    fn test_raster_ref_to_gdal_mem_nd_band_stacks_and_round_trips() {
        // 3-D band ["time","y","x"] shape [3,2,2] over a 2x2 raster; its three
        // time planes flatten into three GDAL bands, then regroup back to the
        // same N-D band via the layout — a byte-exact round trip.
        let data: Vec<u8> = (0u8..12).collect();
        let mut builder = RasterBuilder::new(1);
        builder
            .start_raster_2d(2, 2, 0.0, 2.0, 1.0, -1.0, 0.0, 0.0, None)
            .unwrap();
        builder
            .start_band_nd(
                Some("cube"),
                &["time", "y", "x"],
                &[3, 2, 2],
                BandDataType::UInt8,
                None,
                None,
                None,
            )
            .unwrap();
        builder.band_data_writer().append_value(&data);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        let raster_array = builder.finish().unwrap();
        let raster = single_raster(&raster_array);

        let (band_count, reconstructed) = with_gdal(|gdal| {
            let dataset = unsafe { raster_ref_to_gdal_mem(gdal, &raster, &[1]) }?;
            let band_count = dataset.raster_count();
            let layout = GdalBandLayout::from_raster(&raster, &[1])?;
            let reconstructed = crate::utils::gdal_dataset_to_nd_raster(&dataset, &layout)?;
            Ok((band_count, reconstructed))
        })
        .unwrap();

        // 3 time planes -> 3 GDAL bands.
        assert_eq!(band_count, 3);

        // Round-trip identity: name, dims, shape, bytes.
        let rt = single_raster(&reconstructed);
        assert_eq!(rt.band_name(0), Some("cube"));
        let band = rt.band(0).unwrap();
        assert_eq!(band.dim_names(), vec!["time", "y", "x"]);
        assert_eq!(band.shape(), &[3, 2, 2]);
        let ndb = band.nd_buffer().unwrap();
        assert_eq!(ndb.as_contiguous().unwrap(), &data[..]);
    }

    #[test]
    fn test_raster_ref_to_gdal_mem_mixed_2d_and_nd_round_trips() {
        // One 2-D band (1 plane) + one 3-D band with nodata (3 planes) in the
        // same raster. Exercises heterogeneous plane counts (the regroup
        // run-length split), the 2-D degenerate path, and per-plane nodata
        // expansion across an N-D band's GDAL bands.
        let band0: Vec<u8> = (0u8..4).collect(); // [y,x] = [2,2]
        let band1: Vec<u8> = (10u8..22).collect(); // [time,y,x] = [3,2,2]
        let mut builder = RasterBuilder::new(1);
        builder
            .start_raster_2d(2, 2, 0.0, 2.0, 1.0, -1.0, 0.0, 0.0, None)
            .unwrap();
        builder
            .start_band_nd(
                Some("flat"),
                &["y", "x"],
                &[2, 2],
                BandDataType::UInt8,
                None,
                None,
                None,
            )
            .unwrap();
        builder.band_data_writer().append_value(&band0);
        builder.finish_band().unwrap();
        builder
            .start_band_nd(
                Some("cube"),
                &["time", "y", "x"],
                &[3, 2, 2],
                BandDataType::UInt8,
                Some(&[7u8]),
                None,
                None,
            )
            .unwrap();
        builder.band_data_writer().append_value(&band1);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        let raster_array = builder.finish().unwrap();
        let raster = single_raster(&raster_array);

        let (band_count, plane_nodata, reconstructed) = with_gdal(|gdal| {
            let dataset = unsafe { raster_ref_to_gdal_mem(gdal, &raster, &[1, 2]) }?;
            let band_count = dataset.raster_count();
            // GDAL bands: 1 = flat (no nodata), 2..=4 = cube planes (nodata 7).
            let mut plane_nodata = Vec::new();
            for g in 1..=band_count {
                let band = dataset.rasterband(g).map_err(convert_gdal_err)?;
                plane_nodata.push(band_nodata_to_bytes(&band)?);
            }
            let layout = GdalBandLayout::from_raster(&raster, &[1, 2])?;
            let reconstructed = crate::utils::gdal_dataset_to_nd_raster(&dataset, &layout)?;
            Ok((band_count, plane_nodata, reconstructed))
        })
        .unwrap();

        // 1 plane + 3 planes -> 4 GDAL bands.
        assert_eq!(band_count, 4);
        // Per-plane nodata: the 2-D band carries none; each cube plane carries 7.
        assert_eq!(
            plane_nodata,
            vec![None, Some(vec![7]), Some(vec![7]), Some(vec![7])]
        );

        let rt = single_raster(&reconstructed);
        assert_eq!(rt.num_bands(), 2);

        // Band 0 regroups to the 2-D band.
        assert_eq!(rt.band_name(0), Some("flat"));
        let b0 = rt.band(0).unwrap();
        assert_eq!(b0.dim_names(), vec!["y", "x"]);
        assert_eq!(b0.shape(), &[2, 2]);
        assert_eq!(b0.nodata(), None);
        assert_eq!(b0.nd_buffer().unwrap().as_contiguous().unwrap(), &band0[..]);

        // Band 1 regroups its 3 planes back into the 3-D band, nodata intact.
        assert_eq!(rt.band_name(1), Some("cube"));
        let b1 = rt.band(1).unwrap();
        assert_eq!(b1.dim_names(), vec!["time", "y", "x"]);
        assert_eq!(b1.shape(), &[3, 2, 2]);
        assert_eq!(b1.nodata(), Some(&[7u8][..]));
        assert_eq!(b1.nd_buffer().unwrap().as_contiguous().unwrap(), &band1[..]);
    }

    #[test]
    fn test_raster_ref_to_gdal_mem_multi_nonspatial_dims_round_trip() {
        // Two non-spatial axes ["time","level","y","x"] = [2,2,2,2] -> 4 planes,
        // flattened C-order (time outer, level inner) and regrouped back.
        let data: Vec<u8> = (0u8..16).collect();
        let mut builder = RasterBuilder::new(1);
        builder
            .start_raster_2d(2, 2, 0.0, 2.0, 1.0, -1.0, 0.0, 0.0, None)
            .unwrap();
        builder
            .start_band_nd(
                Some("hypercube"),
                &["time", "level", "y", "x"],
                &[2, 2, 2, 2],
                BandDataType::UInt8,
                None,
                None,
                None,
            )
            .unwrap();
        builder.band_data_writer().append_value(&data);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        let raster_array = builder.finish().unwrap();
        let raster = single_raster(&raster_array);

        let (band_count, reconstructed) = with_gdal(|gdal| {
            let dataset = unsafe { raster_ref_to_gdal_mem(gdal, &raster, &[1]) }?;
            let band_count = dataset.raster_count();
            let layout = GdalBandLayout::from_raster(&raster, &[1])?;
            let reconstructed = crate::utils::gdal_dataset_to_nd_raster(&dataset, &layout)?;
            Ok((band_count, reconstructed))
        })
        .unwrap();

        assert_eq!(band_count, 4); // 2 * 2 planes
        let rt = single_raster(&reconstructed);
        let band = rt.band(0).unwrap();
        assert_eq!(band.dim_names(), vec!["time", "level", "y", "x"]);
        assert_eq!(band.shape(), &[2, 2, 2, 2]);
        assert_eq!(
            band.nd_buffer().unwrap().as_contiguous().unwrap(),
            &data[..]
        );
    }

    #[test]
    fn test_raster_ref_to_gdal_mem_rejects_non_trailing_spatial_pair() {
        // Spatial axes not innermost: ["y","x","time"]. The raster's spatial
        // grid is still satisfied (y, x present at the right sizes), so the
        // builder accepts it, but the GDAL bridge requires the spatial pair to
        // be the trailing two dims.
        let mut builder = RasterBuilder::new(1);
        builder
            .start_raster_2d(2, 2, 0.0, 2.0, 1.0, -1.0, 0.0, 0.0, None)
            .unwrap();
        builder
            .start_band_nd(
                None,
                &["y", "x", "time"],
                &[2, 2, 3],
                BandDataType::UInt8,
                None,
                None,
                None,
            )
            .unwrap();
        builder
            .band_data_writer()
            .append_value(vec![0u8; 2 * 2 * 3]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        let raster_array = builder.finish().unwrap();
        let raster = single_raster(&raster_array);

        let err = with_gdal(|gdal| unsafe { raster_ref_to_gdal_mem(gdal, &raster, &[1]) })
            .err()
            .unwrap();
        assert!(
            err.to_string().contains("trailing two dims are a"),
            "got: {err}"
        );
    }
}
