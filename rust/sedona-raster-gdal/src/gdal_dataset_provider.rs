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

use std::convert::TryInto;
use std::{cell::RefCell, marker::PhantomData, num::NonZeroUsize, rc::Rc};

use datafusion_common::config::ConfigOptions;
use datafusion_common::{exec_datafusion_err, exec_err, DataFusionError, Result};

use sedona_gdal::dataset::Dataset;
use sedona_gdal::gdal::Gdal;
use sedona_gdal::geo_transform::{GeoTransform, GeoTransformEx};
use sedona_gdal::raster::types::GdalDataType;

use sedona_common::SedonaOptions;
use sedona_raster::traits::RasterRef;
use sedona_schema::raster::BandDataType;

use crate::gdal_common::{
    band_data_type_to_gdal, bytes_to_f64, convert_gdal_err, normalize_outdb_source_path,
    open_gdal_dataset, raster_ref_to_gdal_empty, raster_ref_to_gdal_mem,
};
use crate::source_uri::parse_outdb_source;

/// A GDAL dataset constructed from a `RasterRef`.
///
/// This struct is designed to keep any backing GDAL datasets alive for as long as
/// the returned `dataset` might reference them.
///
/// Field semantics by raster storage layout:
///
/// 1) **In-db bands only**
///    - `dataset`: a GDAL **MEM** dataset containing all bands.
///    - `gdal_mem_source`: `None` (the MEM dataset is already stored in `dataset`).
///    - `_gdal_outdb_sources`: empty.
///
/// 2) **Out-db bands only**
///    - `dataset`: a GDAL **VRT** dataset sized like the target raster, with each VRT band
///      sourcing from an external dataset band.
///    - `gdal_mem_source`: `None`.
///    - `_gdal_outdb_sources`: contains the opened external GDAL datasets (kept alive via `Rc`).
///      (There may be duplicates if multiple bands reference the same URL; that is fine.)
///
/// 3) **Mixed in-db + out-db bands**
///    - `dataset`: a GDAL **VRT** dataset with band order matching the target raster.
///      In-db bands source from a MEM dataset; out-db bands source from external datasets.
///    - `gdal_mem_source`: `Some(MEM dataset)` containing only the in-db bands, in the same order
///      as they appear in the target raster.
///    - `_gdal_outdb_sources`: contains the opened external GDAL datasets (kept alive via `Rc`).
pub(crate) struct RasterDataset<'a> {
    /// The dataset to use for further GDAL operations.
    dataset: Rc<Dataset>,
    /// A MEM dataset holding in-db band data when `dataset` is a VRT that references it.
    _gdal_mem_source: Option<Rc<Dataset>>,
    /// External datasets referenced by the VRT; kept alive for the lifetime of this struct.
    _gdal_outdb_sources: Vec<Rc<Dataset>>,
    /// Binds this dataset's lifetime to the borrowed source raster.
    _source_raster: PhantomData<&'a dyn RasterRef>,
}

impl<'a> RasterDataset<'a> {
    /// Return a reference to the underlying GDAL dataset.
    pub(crate) fn as_dataset(&self) -> &Dataset {
        &self.dataset
    }
}

thread_local! {
    /// Thread-local lazily-initialized `GDALDatasetCache`.
    static TL_GDAL_DATASET_CACHE: RefCell<Option<Rc<GDALDatasetCache>>> = const { RefCell::new(None) };
}

const DEFAULT_GDAL_SOURCE_CACHE_SIZE: usize = 32;
const DEFAULT_GDAL_VRT_CACHE_SIZE: usize = 32;

pub(crate) fn configure_thread_local_options(
    gdal: &Gdal,
    config_options: Option<&ConfigOptions>,
) -> Result<()> {
    // Set frequently requested GDAL config options as thread-local options to eliminate the
    // need for acquiring configs from global config or environment variable, which is very
    // likely to result in heavy contention in multi-threaded environments.
    let cpl_debug_enabled = config_options
        .and_then(|c| {
            c.extensions
                .get::<SedonaOptions>()
                .map(|opts| opts.gdal.cpl_debug)
        })
        .unwrap_or(false);
    let cpl_debug_value = if cpl_debug_enabled { "ON" } else { "OFF" };

    let thread_local_options = [
        ("CPL_DEBUG", cpl_debug_value),
        ("OSR_DEFAULT_AXIS_MAPPING_STRATEGY", "AUTHORITY_COMPLIANT"),
        ("GDAL_VALIDATE_CREATION_OPTIONS", "YES"),
        ("CHECK_WITH_INVERT_PROJ", "NO"),
        ("GDAL_FORCE_CACHING", "NO"),
        ("GDAL_ENABLE_READ_WRITE_MUTEX", "YES"),
    ];

    for (key, value) in thread_local_options {
        gdal.set_thread_local_config_option(key, value)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
    }
    Ok(())
}

/// Get or create the thread-local `GDALDatasetCache`.
pub(crate) fn thread_local_cache() -> Result<Rc<GDALDatasetCache>> {
    TL_GDAL_DATASET_CACHE.with(|cell| {
        let mut opt = cell.borrow_mut();
        if let Some(rc) = opt.as_ref() {
            Ok(Rc::clone(rc))
        } else {
            let cache = Rc::new(GDALDatasetCache::try_new(
                DEFAULT_GDAL_SOURCE_CACHE_SIZE,
                DEFAULT_GDAL_VRT_CACHE_SIZE,
            )?);
            *opt = Some(Rc::clone(&cache));
            Ok(cache)
        }
    })
}

/// Build a `GDALDatasetProvider` from an explicit GDAL handle and the thread-local cache.
pub(crate) fn thread_local_provider<'a>(gdal: &'a Gdal) -> Result<GDALDatasetProvider<'a>> {
    Ok(GDALDatasetProvider::new(gdal, thread_local_cache()?))
}

#[derive(Hash, Eq, PartialEq)]
struct OutDbSourceKey {
    path: String,
    open_options: Option<Vec<String>>,
}

impl OutDbSourceKey {
    pub fn new(path: &str, options: Option<&[&str]>) -> Self {
        let normalized_path = normalize_outdb_source_path(path);
        let open_options = options
            .filter(|opts| !opts.is_empty())
            .map(|opts| opts.iter().map(|s| (*s).to_string()).collect());

        Self {
            path: normalized_path,
            open_options,
        }
    }
}

pub(crate) struct GDALDatasetCache {
    cached_sources: RefCell<lru::LruCache<OutDbSourceKey, Rc<Dataset>>>,
    cached_vrts: RefCell<lru::LruCache<VrtKey, Rc<CachedVrt>>>,
}

impl GDALDatasetCache {
    pub fn try_new(cache_capacity: usize, vrt_cache_capacity: usize) -> Result<Self> {
        let Some(cap) = NonZeroUsize::new(cache_capacity) else {
            return Err(DataFusionError::Configuration(
                "Raster source cache size should be greater than 0".to_string(),
            ));
        };
        let Some(vrt_cap) = NonZeroUsize::new(vrt_cache_capacity) else {
            return Err(DataFusionError::Configuration(
                "Raster VRT cache size should be greater than 0".to_string(),
            ));
        };
        let cache = lru::LruCache::new(cap);
        let vrt_cache = lru::LruCache::new(vrt_cap);
        Ok(Self {
            cached_sources: RefCell::new(cache),
            cached_vrts: RefCell::new(vrt_cache),
        })
    }

    fn get_or_create_outdb_source(
        &self,
        gdal: &Gdal,
        path: &str,
        options: Option<&[&str]>,
    ) -> Result<Rc<Dataset>> {
        let cache_key = OutDbSourceKey::new(path, options);
        let mut cache = self.cached_sources.borrow_mut();
        if let Some(cached_source) = cache.get(&cache_key) {
            Ok(Rc::clone(cached_source))
        } else {
            let source_dataset = open_gdal_dataset(gdal, path, options)?;
            let rc_dataset = Rc::new(source_dataset);
            cache.put(cache_key, Rc::clone(&rc_dataset));
            Ok(rc_dataset)
        }
    }

    fn build_vrt_from_sources<R: RasterRef + ?Sized>(
        &self,
        gdal: &Gdal,
        raster: &R,
        gdal_mem_source: Option<&Rc<Dataset>>,
    ) -> Result<(Rc<Dataset>, Vec<Rc<Dataset>>)> {
        let num_bands = raster.num_bands();

        let raster_width = raster
            .width()
            .ok_or_else(|| exec_datafusion_err!("Raster has no width (spatial_shape missing)"))?;
        let raster_height = raster
            .height()
            .ok_or_else(|| exec_datafusion_err!("Raster has no height (spatial_shape missing)"))?;
        let width: i32 = raster_width.try_into().map_err(|_| {
            exec_datafusion_err!(
                "Raster width {} exceeds supported GDAL/i32 limit {}",
                raster_width,
                i32::MAX
            )
        })?;
        let height: i32 = raster_height.try_into().map_err(|_| {
            exec_datafusion_err!(
                "Raster height {} exceeds supported GDAL/i32 limit {}",
                raster_height,
                i32::MAX
            )
        })?;
        let vrt_width: usize = raster_width.try_into().map_err(|_| {
            exec_datafusion_err!(
                "Raster width {} exceeds supported GDAL/usize limit",
                raster_width
            )
        })?;
        let vrt_height: usize = raster_height.try_into().map_err(|_| {
            exec_datafusion_err!(
                "Raster height {} exceeds supported GDAL/usize limit",
                raster_height
            )
        })?;
        let mut vrt = gdal
            .create_vrt(vrt_width, vrt_height)
            .map_err(convert_gdal_err)?;

        let geotransform: [f64; 6] = raster.transform().try_into().map_err(|_| {
            exec_datafusion_err!(
                "Raster transform must be exactly 6 elements; got {}",
                raster.transform().len()
            )
        })?;
        vrt.set_geo_transform(&geotransform)
            .map_err(convert_gdal_err)?;
        if let Some(crs) = raster.crs() {
            vrt.set_projection(crs).map_err(convert_gdal_err)?;
        }

        let mut outdb_sources: Vec<Rc<Dataset>> = Vec::new();
        let mut mem_band_index: usize = 1;

        for i in 0..num_bands {
            let band = raster
                .band(i)
                .ok_or_else(|| exec_datafusion_err!("Band index {} is out of range", i))?;

            if !band.is_2d() {
                return exec_err!(
                    "GDAL backend requires 2-dim bands; got dim_names={:?}",
                    band.dim_names()
                );
            }

            let band_type = band.data_type();
            let gdal_type = band_data_type_to_gdal(&band_type);
            if matches!(gdal_type, GdalDataType::Unknown) {
                return Err(DataFusionError::NotImplemented(format!(
                    "Band data type {:?} is not supported by this GDAL build",
                    band_type
                )));
            }

            // VRT bands are 1-based.
            let vrt_band_num = i + 1;
            vrt.add_band(gdal_type, None).map_err(convert_gdal_err)?;
            let vrt_band = vrt.rasterband(vrt_band_num).map_err(convert_gdal_err)?;

            if let Some(nodata_bytes) = band.nodata() {
                match band_type {
                    BandDataType::UInt64 => {
                        let nodata_bytes: [u8; 8] = nodata_bytes.try_into().map_err(|_| {
                            exec_datafusion_err!("Invalid nodata byte length for UInt64")
                        })?;
                        let nodata = u64::from_le_bytes(nodata_bytes);
                        vrt_band
                            .set_no_data_value_u64(Some(nodata))
                            .map_err(convert_gdal_err)?;
                    }
                    BandDataType::Int64 => {
                        let nodata_bytes: [u8; 8] = nodata_bytes.try_into().map_err(|_| {
                            exec_datafusion_err!("Invalid nodata byte length for Int64")
                        })?;
                        let nodata = i64::from_le_bytes(nodata_bytes);
                        vrt_band
                            .set_no_data_value_i64(Some(nodata))
                            .map_err(convert_gdal_err)?;
                    }
                    _ => {
                        let nodata = bytes_to_f64(nodata_bytes, &band_type)?;
                        vrt_band
                            .set_no_data_value(nodata)
                            .map_err(convert_gdal_err)?;
                    }
                }
            }

            match band.outdb_uri() {
                Some(uri) => {
                    let (gdal_uri, source_band_num_u32) = parse_outdb_source(uri)?;
                    let source_band_num: usize = source_band_num_u32.try_into().map_err(|_| {
                        exec_datafusion_err!("Band {} out-db band index is too large", vrt_band_num)
                    })?;

                    let source_dataset = self.get_or_create_outdb_source(gdal, &gdal_uri, None)?;

                    // If GDALGetGeoTransform(hdsSrc, ogt) fails, we fall back to (0, 1, 0, 0, 0, -1),
                    // which is the identity transform.
                    let src_geo_transform = source_dataset
                        .geo_transform()
                        .unwrap_or([0.0, 1.0, 0.0, 0.0, 0.0, -1.0]);
                    let (src_w, src_h) = source_dataset.raster_size();

                    // Compute source and destination windows for the VRT simple source. The VRT usually only
                    // clip a small portion of the source dataset.
                    let Some((src_window, dst_window)) = compute_vrt_simple_source_windows(
                        &geotransform,
                        (width, height),
                        &src_geo_transform,
                        (src_w as i32, src_h as i32),
                    )?
                    else {
                        // No spatial overlap between the target raster and the source dataset.
                        // Leave the VRT band as nodata.
                        continue;
                    };

                    let source_band = source_dataset
                        .rasterband(source_band_num)
                        .map_err(convert_gdal_err)?;

                    vrt_band
                        // Avoid passing per-source NODATA to VRT simple sources; some GDAL builds
                        // warn that NODATA isn't supported for neighbour-sampled simple sources
                        // on virtual datasources. We set band-level NODATA via set_no_data_value.
                        .add_simple_source(&source_band, src_window, dst_window, None, None)
                        .map_err(convert_gdal_err)?;

                    outdb_sources.push(source_dataset);
                }
                None => {
                    let mem_dataset = gdal_mem_source
                        .as_ref()
                        .expect("in-db dataset should exist");
                    let source_band = mem_dataset
                        .rasterband(mem_band_index)
                        .map_err(convert_gdal_err)?;
                    mem_band_index += 1;

                    vrt_band
                        .add_simple_source(
                            &source_band,
                            (0, 0, width, height),
                            (0, 0, width, height),
                            None,
                            None,
                        )
                        .map_err(convert_gdal_err)?;
                }
            }
        }

        Ok((Rc::new(vrt.as_dataset()), outdb_sources))
    }
}

#[derive(Clone)]
pub(crate) struct GDALDatasetProvider<'a> {
    gdal: &'a Gdal,
    cache: Rc<GDALDatasetCache>,
}

impl<'a> GDALDatasetProvider<'a> {
    pub fn new(gdal: &'a Gdal, cache: Rc<GDALDatasetCache>) -> Self {
        Self { gdal, cache }
    }

    pub fn raster_ref_to_gdal<'b, R: RasterRef + ?Sized>(
        &self,
        raster: &'b R,
    ) -> Result<RasterDataset<'b>> {
        let num_bands = raster.num_bands();

        if num_bands == 0 {
            let dataset = raster_ref_to_gdal_empty(self.gdal, raster)?;
            return Ok(RasterDataset {
                dataset: Rc::new(dataset),
                _gdal_mem_source: None,
                _gdal_outdb_sources: Vec::new(),
                _source_raster: PhantomData,
            });
        }

        let mut indb_band_indices = Vec::with_capacity(num_bands);
        let mut has_outdb = false;
        for i in 0..num_bands {
            let band = raster
                .band(i)
                .ok_or_else(|| exec_datafusion_err!("Band index {} is out of range", i))?;
            if band.outdb_uri().is_some() {
                has_outdb = true;
            } else {
                // raster_ref_to_gdal_mem expects 1-based source band indices.
                indb_band_indices.push(i + 1);
            }
        }

        let mut gdal_mem_source = if !indb_band_indices.is_empty() {
            Some(Rc::new(unsafe {
                raster_ref_to_gdal_mem(self.gdal, raster, &indb_band_indices)?
            }))
        } else {
            None
        };

        if !has_outdb {
            let dataset = gdal_mem_source.take().expect("in-db dataset should exist");
            return Ok(RasterDataset {
                dataset,
                _gdal_mem_source: None,
                _gdal_outdb_sources: Vec::new(),
                _source_raster: PhantomData,
            });
        }

        if indb_band_indices.is_empty() {
            let vrt_key = VrtKey::from_raster(raster)?;
            if let Some(cached) = self.cache.cached_vrts.borrow_mut().get(&vrt_key) {
                return Ok(RasterDataset {
                    dataset: Rc::clone(&cached.dataset),
                    _gdal_mem_source: None,
                    _gdal_outdb_sources: cached.outdb_sources.clone(),
                    _source_raster: PhantomData,
                });
            }

            let (dataset, outdb_sources) =
                self.cache.build_vrt_from_sources(self.gdal, raster, None)?;
            let cached = Rc::new(CachedVrt {
                dataset: Rc::clone(&dataset),
                outdb_sources: outdb_sources.clone(),
            });
            self.cache
                .cached_vrts
                .borrow_mut()
                .put(vrt_key, Rc::clone(&cached));

            return Ok(RasterDataset {
                dataset,
                _gdal_mem_source: None,
                _gdal_outdb_sources: outdb_sources,
                _source_raster: PhantomData,
            });
        }

        let (dataset, outdb_sources) =
            self.cache
                .build_vrt_from_sources(self.gdal, raster, gdal_mem_source.as_ref())?;

        Ok(RasterDataset {
            dataset,
            _gdal_mem_source: gdal_mem_source,
            _gdal_outdb_sources: outdb_sources,
            _source_raster: PhantomData,
        })
    }
}

#[derive(Hash, Eq, PartialEq)]
struct VrtBandKey {
    data_type: BandDataType,
    nodata_bits: Option<u64>,
    /// Normalized GDAL-side URI, with the SedonaDB `#band=N` fragment stripped.
    /// `None` for in-db bands (paired with `outdb_band_id == None`).
    outdb_url: Option<String>,
    /// 1-based source band index parsed from `outdb_uri`. `None` for in-db bands.
    outdb_band_id: Option<u32>,
}

#[derive(Hash, Eq, PartialEq)]
struct VrtKey {
    width: u64,
    height: u64,
    geotransform_bits: [u64; 6],
    crs: Option<String>,
    bands: Vec<VrtBandKey>,
}

impl VrtKey {
    fn from_raster<R: RasterRef + ?Sized>(raster: &R) -> Result<Self> {
        let num_bands = raster.num_bands();
        let width = raster
            .width()
            .ok_or_else(|| exec_datafusion_err!("Raster has no width (spatial_shape missing)"))?;
        let height = raster
            .height()
            .ok_or_else(|| exec_datafusion_err!("Raster has no height (spatial_shape missing)"))?;
        let geotransform: [f64; 6] = raster.transform().try_into().map_err(|_| {
            exec_datafusion_err!(
                "Raster transform must be exactly 6 elements; got {}",
                raster.transform().len()
            )
        })?;
        let geotransform_bits = geotransform.map(f64::to_bits);

        let mut band_keys = Vec::with_capacity(num_bands);
        for i in 0..num_bands {
            let band = raster
                .band(i)
                .ok_or_else(|| exec_datafusion_err!("Band index {} is out of range", i))?;
            let band_type = band.data_type();
            let nodata_bits = match (band.nodata(), band_type) {
                (Some(bytes), BandDataType::UInt64) => {
                    let bytes: [u8; 8] = bytes.try_into().map_err(|_| {
                        exec_datafusion_err!("Invalid nodata byte length for UInt64")
                    })?;
                    Some(u64::from_le_bytes(bytes))
                }
                (Some(bytes), BandDataType::Int64) => {
                    let bytes: [u8; 8] = bytes.try_into().map_err(|_| {
                        exec_datafusion_err!("Invalid nodata byte length for Int64")
                    })?;
                    Some(u64::from_le_bytes(bytes))
                }
                (Some(bytes), _) => Some(bytes_to_f64(bytes, &band_type)?.to_bits()),
                (None, _) => None,
            };
            let (outdb_url, outdb_band_id) = match band.outdb_uri() {
                Some(uri) => {
                    let (gdal_uri, band_id) = parse_outdb_source(uri)?;
                    (Some(normalize_outdb_source_path(&gdal_uri)), Some(band_id))
                }
                None => (None, None),
            };
            band_keys.push(VrtBandKey {
                data_type: band_type,
                nodata_bits,
                outdb_url,
                outdb_band_id,
            });
        }

        Ok(Self {
            width,
            height,
            geotransform_bits,
            crs: raster.crs().map(|s| s.to_string()),
            bands: band_keys,
        })
    }
}

struct CachedVrt {
    dataset: Rc<Dataset>,
    outdb_sources: Vec<Rc<Dataset>>,
}

type PixelWindow = (i32, i32, i32, i32);
type PixelWindowOverlap = Option<(PixelWindow, PixelWindow)>;

fn compute_vrt_simple_source_windows(
    dst_gt: &GeoTransform,
    dst_size: (i32, i32),
    src_gt: &GeoTransform,
    src_size: (i32, i32),
) -> Result<PixelWindowOverlap> {
    let (dst_w, dst_h) = dst_size;
    let (src_w, src_h) = src_size;
    if dst_w <= 0 || dst_h <= 0 || src_w <= 0 || src_h <= 0 {
        return Ok(None);
    }

    // Alignment check (similar intent to rt_raster_same_alignment()).
    // Require equal pixel size + rotation terms.
    let eps = f32::EPSILON as f64;
    if (dst_gt[1] - src_gt[1]).abs() > eps
        || (dst_gt[2] - src_gt[2]).abs() > eps
        || (dst_gt[4] - src_gt[4]).abs() > eps
        || (dst_gt[5] - src_gt[5]).abs() > eps
    {
        return exec_err!(
            "Out-db raster is not aligned with target raster (geotransform mismatch): dst={:?} src={:?}",
            dst_gt, src_gt
        );
    }

    // Compute the pixel/line offset of the destination upper-left in the source grid.
    let inv_src = src_gt
        .invert()
        .map_err(|e| exec_datafusion_err!("Failed to invert source geotransform: {e}"))?;
    let (off_x_f, off_y_f) = inv_src.apply(dst_gt[0], dst_gt[3]);

    let off_x_r: f64 = off_x_f.round();
    let off_y_r: f64 = off_y_f.round();
    if (off_x_f - off_x_r).abs() > eps || (off_y_f - off_y_r).abs() > eps {
        return exec_err!(
            "Out-db raster is not aligned with target raster (non-integer pixel offset): off=({off_x_f},{off_y_f})"
        );
    }

    if off_x_r < (i32::MIN as f64)
        || off_x_r > (i32::MAX as f64)
        || off_y_r < (i32::MIN as f64)
        || off_y_r > (i32::MAX as f64)
    {
        return exec_err!("Out-db raster alignment offset is out of supported range");
    }

    let off_x = off_x_r as i32;
    let off_y = off_y_r as i32;

    // Compute overlapped windows (clipped) while preserving the aligned-grid assumption.
    let dst_xoff = 0.max(-off_x);
    let dst_yoff = 0.max(-off_y);
    let src_xoff = 0.max(off_x);
    let src_yoff = 0.max(off_y);

    let xsize = (dst_w - dst_xoff).min(src_w - src_xoff);
    let ysize = (dst_h - dst_yoff).min(src_h - src_yoff);
    if xsize <= 0 || ysize <= 0 {
        return Ok(None);
    }

    Ok(Some((
        (src_xoff, src_yoff, xsize, ysize),
        (dst_xoff, dst_yoff, xsize, ysize),
    )))
}

#[cfg(test)]
mod tests {
    use super::{GDALDatasetCache, GDALDatasetProvider};
    use std::rc::Rc;

    use arrow_array::StructArray;
    use sedona_gdal::raster::types::Buffer;
    use sedona_raster::array::RasterStructArray;
    use sedona_raster::builder::RasterBuilder;
    use sedona_schema::raster::BandDataType;
    use tempfile::TempDir;

    use crate::gdal_common::with_gdal;

    fn create_source_tiff(temp_dir: &TempDir) -> String {
        let path = temp_dir.path().join("source.tif");
        let path_str = path.to_string_lossy().to_string();

        with_gdal(|gdal| {
            let driver = gdal.get_driver_by_name("GTiff").unwrap();
            let dataset = driver
                .create_with_band_type::<u8>(&path_str, 8, 8, 1)
                .unwrap();
            dataset
                .set_geo_transform(&[0.0, 1.0, 0.0, 8.0, 0.0, -1.0])
                .unwrap();
            let band = dataset.rasterband(1).unwrap();
            band.set_no_data_value(Some(0.0)).unwrap();
            let mut buffer = Buffer::new((8, 8), vec![1u8; 8 * 8]);
            band.write((0, 0), (8, 8), &mut buffer).unwrap();
            Ok(())
        })
        .unwrap();

        path_str
    }

    /// Two-band GeoTIFF on disk: band 1 is filled with `band1_fill`, band 2
    /// with `band2_fill`. Used to exercise `#band=2` selection end-to-end.
    fn create_two_band_source_tiff(temp_dir: &TempDir, band1_fill: u8, band2_fill: u8) -> String {
        let path = temp_dir.path().join("two_band.tif");
        let path_str = path.to_string_lossy().to_string();

        with_gdal(|gdal| {
            let driver = gdal.get_driver_by_name("GTiff").unwrap();
            let dataset = driver
                .create_with_band_type::<u8>(&path_str, 8, 8, 2)
                .unwrap();
            dataset
                .set_geo_transform(&[0.0, 1.0, 0.0, 8.0, 0.0, -1.0])
                .unwrap();
            for (i, fill) in [band1_fill, band2_fill].iter().enumerate() {
                let band = dataset.rasterband(i + 1).unwrap();
                let mut buffer = Buffer::new((8, 8), vec![*fill; 8 * 8]);
                band.write((0, 0), (8, 8), &mut buffer).unwrap();
            }
            Ok(())
        })
        .unwrap();

        path_str
    }

    type IndbBandSpec = (BandDataType, Option<Vec<u8>>, Vec<u8>);

    fn build_indb_raster_8x8(crs: Option<&str>, bands: &[IndbBandSpec]) -> StructArray {
        let mut builder = RasterBuilder::new(1);
        builder
            .start_raster_2d(8, 8, 0.0, 8.0, 1.0, -1.0, 0.0, 0.0, crs)
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

    fn build_outdb_raster(uri: &str) -> StructArray {
        let mut builder = RasterBuilder::new(1);
        builder
            .start_raster_2d(8, 8, 0.0, 8.0, 1.0, -1.0, 0.0, 0.0, None)
            .unwrap();
        builder
            .start_band(
                None,
                &["y", "x"],
                &[8, 8],
                BandDataType::UInt8,
                Some(&[0u8]),
                Some(uri),
                Some("geotiff"),
            )
            .unwrap();
        builder.band_data_writer().append_value([]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        builder.finish().unwrap()
    }

    fn build_mixed_raster(uri: &str) -> StructArray {
        let mut builder = RasterBuilder::new(1);
        builder
            .start_raster_2d(8, 8, 0.0, 8.0, 1.0, -1.0, 0.0, 0.0, Some("EPSG:4326"))
            .unwrap();

        builder
            .start_band_2d(BandDataType::UInt8, Some(&[255u8]))
            .unwrap();
        builder.band_data_writer().append_value(vec![7u8; 8 * 8]);
        builder.finish_band().unwrap();

        builder
            .start_band(
                None,
                &["y", "x"],
                &[8, 8],
                BandDataType::UInt8,
                Some(&[0u8]),
                Some(uri),
                Some("geotiff"),
            )
            .unwrap();
        builder.band_data_writer().append_value([]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        builder.finish().unwrap()
    }

    fn assert_wgs84_projection_wkt(wkt: &str) {
        assert!(wkt.contains("AUTHORITY[\"EPSG\",\"4326\"]"));
    }

    #[test]
    fn test_outdb_source_key_normalizes_s3_variants() {
        let s3_key = super::OutDbSourceKey::new("s3://bucket/path/test.tif", None);
        let s3a_key = super::OutDbSourceKey::new("s3a://bucket/path/test.tif", None);
        assert!(s3_key == s3a_key);
    }

    #[test]
    fn test_outdb_source_key_normalizes_gcs_variants() {
        let gs_key = super::OutDbSourceKey::new("gs://bucket/path/test.tif", None);
        let gcs_key = super::OutDbSourceKey::new("gcs://bucket/path/test.tif", None);
        assert!(gs_key == gcs_key);
    }

    #[test]
    fn test_outdb_source_key_normalizes_azure_variants() {
        let az_blob_key = super::OutDbSourceKey::new("az://container/path/test.tif", None);
        let wasbs_blob_key = super::OutDbSourceKey::new("wasbs://container/path/test.tif", None);
        assert!(az_blob_key == wasbs_blob_key);

        let abfs_adls_key = super::OutDbSourceKey::new("abfs://fs/path/test.tif", None);
        let abfss_adls_key = super::OutDbSourceKey::new("abfss://fs/path/test.tif", None);
        assert!(abfs_adls_key == abfss_adls_key);
    }

    #[test]
    fn test_vrt_key_normalizes_equivalent_outdb_urls() {
        let raster_s3_struct = build_outdb_raster("s3://bucket/path/test.tif");
        let raster_s3a_struct = build_outdb_raster("s3a://bucket/path/test.tif");

        let raster_s3_array = RasterStructArray::new(&raster_s3_struct);
        let raster_s3a_array = RasterStructArray::new(&raster_s3a_struct);

        let raster_s3 = raster_s3_array.get(0).unwrap();
        let raster_s3a = raster_s3a_array.get(0).unwrap();

        let key_s3 = super::VrtKey::from_raster(&raster_s3).unwrap();
        let key_s3a = super::VrtKey::from_raster(&raster_s3a).unwrap();

        assert!(key_s3 == key_s3a);
    }

    #[test]
    fn test_vrt_key_normalizes_gcs_and_azure_equivalent_outdb_urls() {
        let raster_gs_struct = build_outdb_raster("gs://bucket/path/test.tif");
        let raster_gcs_struct = build_outdb_raster("gcs://bucket/path/test.tif");
        let raster_az_struct = build_outdb_raster("az://container/path/test.tif");
        let raster_wasbs_struct = build_outdb_raster("wasbs://container/path/test.tif");
        let raster_abfs_struct = build_outdb_raster("abfs://filesystem/path/test.tif");
        let raster_abfss_struct = build_outdb_raster("abfss://filesystem/path/test.tif");

        let raster_gs_array = RasterStructArray::new(&raster_gs_struct);
        let raster_gcs_array = RasterStructArray::new(&raster_gcs_struct);
        let raster_az_array = RasterStructArray::new(&raster_az_struct);
        let raster_wasbs_array = RasterStructArray::new(&raster_wasbs_struct);
        let raster_abfs_array = RasterStructArray::new(&raster_abfs_struct);
        let raster_abfss_array = RasterStructArray::new(&raster_abfss_struct);

        let key_gs = super::VrtKey::from_raster(&raster_gs_array.get(0).unwrap()).unwrap();
        let key_gcs = super::VrtKey::from_raster(&raster_gcs_array.get(0).unwrap()).unwrap();
        assert!(key_gs == key_gcs);

        let key_az = super::VrtKey::from_raster(&raster_az_array.get(0).unwrap()).unwrap();
        let key_wasbs = super::VrtKey::from_raster(&raster_wasbs_array.get(0).unwrap()).unwrap();
        assert!(key_az == key_wasbs);

        let key_abfs = super::VrtKey::from_raster(&raster_abfs_array.get(0).unwrap()).unwrap();
        let key_abfss = super::VrtKey::from_raster(&raster_abfss_array.get(0).unwrap()).unwrap();
        assert!(key_abfs == key_abfss);
    }

    #[test]
    fn test_outdb_vrt_cache_reuse() {
        let temp_dir = TempDir::new().unwrap();
        let path = create_source_tiff(&temp_dir);
        let raster_struct = build_outdb_raster(&path);
        let raster_array = RasterStructArray::new(&raster_struct);
        let raster = raster_array.get(0).unwrap();

        let cache = Rc::new(GDALDatasetCache::try_new(4, 4).unwrap());
        let (first, second) = with_gdal(|gdal| {
            let provider = GDALDatasetProvider::new(gdal, Rc::clone(&cache));
            let first = provider.raster_ref_to_gdal(&raster)?;
            let second = provider.raster_ref_to_gdal(&raster)?;
            Ok((first, second))
        })
        .unwrap();

        assert_eq!(
            first.as_dataset().c_dataset(),
            second.as_dataset().c_dataset()
        );
    }

    #[test]
    fn test_provider_returns_empty_dataset_for_zero_band_raster() {
        let mut builder = RasterBuilder::new(1);
        builder
            .start_raster_2d(3, 2, 1.0, 4.0, 1.0, -1.0, 0.0, 0.0, Some("EPSG:4326"))
            .unwrap();
        builder.finish_raster().unwrap();
        let raster_struct = builder.finish().unwrap();
        let raster_array = RasterStructArray::new(&raster_struct);
        let raster = raster_array.get(0).unwrap();
        let cache = Rc::new(GDALDatasetCache::try_new(4, 4).unwrap());

        let dataset = with_gdal(|gdal| {
            let provider = GDALDatasetProvider::new(gdal, Rc::clone(&cache));
            provider.raster_ref_to_gdal(&raster)
        })
        .unwrap();

        assert_eq!(dataset.as_dataset().raster_size(), (3, 2));
        assert_eq!(dataset.as_dataset().raster_count(), 0);
        assert_wgs84_projection_wkt(&dataset.as_dataset().projection());
    }

    #[test]
    fn test_provider_returns_mem_dataset_for_indb_raster() {
        let mut builder = RasterBuilder::new(1);
        builder
            .start_raster_2d(2, 2, 0.0, 2.0, 1.0, -1.0, 0.0, 0.0, Some("EPSG:4326"))
            .unwrap();
        builder
            .start_band_2d(BandDataType::UInt8, Some(&[255u8]))
            .unwrap();
        builder
            .band_data_writer()
            .append_value(vec![11u8, 12u8, 13u8, 14u8]);
        builder.finish_band().unwrap();
        builder
            .start_band_2d(BandDataType::UInt8, Some(&[0u8]))
            .unwrap();
        builder
            .band_data_writer()
            .append_value(vec![21u8, 22u8, 23u8, 24u8]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        let raster_struct = builder.finish().unwrap();
        let raster_array = RasterStructArray::new(&raster_struct);
        let raster = raster_array.get(0).unwrap();
        let cache = Rc::new(GDALDatasetCache::try_new(4, 4).unwrap());

        let dataset = with_gdal(|gdal| {
            let provider = GDALDatasetProvider::new(gdal, Rc::clone(&cache));
            provider.raster_ref_to_gdal(&raster)
        })
        .unwrap();

        let band = dataset
            .as_dataset()
            .rasterband(2)
            .unwrap()
            .read_as::<u8>((0, 0), (2, 2), (2, 2), None)
            .unwrap();
        assert_eq!(dataset.as_dataset().raster_count(), 2);
        assert_eq!(band.data().to_vec(), vec![21u8, 22u8, 23u8, 24u8]);
    }

    #[test]
    fn test_provider_preserves_band_order_for_mixed_raster() {
        let temp_dir = TempDir::new().unwrap();
        let path = create_source_tiff(&temp_dir);
        let raster_struct = build_mixed_raster(&path);
        let raster_array = RasterStructArray::new(&raster_struct);
        let raster = raster_array.get(0).unwrap();
        let cache = Rc::new(GDALDatasetCache::try_new(4, 4).unwrap());

        let dataset = with_gdal(|gdal| {
            let provider = GDALDatasetProvider::new(gdal, Rc::clone(&cache));
            provider.raster_ref_to_gdal(&raster)
        })
        .unwrap();

        let band1 = dataset
            .as_dataset()
            .rasterband(1)
            .unwrap()
            .read_as::<u8>((0, 0), (8, 8), (8, 8), None)
            .unwrap();
        let band2 = dataset
            .as_dataset()
            .rasterband(2)
            .unwrap()
            .read_as::<u8>((0, 0), (8, 8), (8, 8), None)
            .unwrap();
        assert_eq!(band1.data().to_vec(), vec![7u8; 8 * 8]);
        assert_eq!(band2.data().to_vec(), vec![1u8; 8 * 8]);
    }

    #[test]
    fn test_vrt_key_distinguishes_lossless_uint64_nodata() {
        fn one_band_uint64(nodata: u64) -> StructArray {
            let mut builder = RasterBuilder::new(1);
            builder
                .start_raster_2d(1, 1, 0.0, 1.0, 1.0, -1.0, 0.0, 0.0, None)
                .unwrap();
            builder
                .start_band_2d(BandDataType::UInt64, Some(&nodata.to_le_bytes()))
                .unwrap();
            builder.band_data_writer().append_value(1u64.to_le_bytes());
            builder.finish_band().unwrap();
            builder.finish_raster().unwrap();
            builder.finish().unwrap()
        }

        let raster_a = one_band_uint64(9_007_199_254_740_992u64);
        let raster_b = one_band_uint64(9_007_199_254_740_993u64);

        let key_a =
            super::VrtKey::from_raster(&RasterStructArray::new(&raster_a).get(0).unwrap()).unwrap();
        let key_b =
            super::VrtKey::from_raster(&RasterStructArray::new(&raster_b).get(0).unwrap()).unwrap();

        assert!(key_a != key_b);
    }

    #[test]
    fn test_provider_rejects_nd_band_in_vrt_path() {
        let temp_dir = TempDir::new().unwrap();
        let path = create_source_tiff(&temp_dir);

        // Build a raster mixing one in-db 3-D band (forces N-D rejection inside
        // build_vrt_from_sources) with one out-db band.
        let mut builder = RasterBuilder::new(1);
        builder
            .start_raster_2d(8, 8, 0.0, 8.0, 1.0, -1.0, 0.0, 0.0, None)
            .unwrap();
        builder
            .start_band(
                None,
                &["time", "y", "x"],
                &[2, 8, 8],
                BandDataType::UInt8,
                None,
                None,
                None,
            )
            .unwrap();
        builder
            .band_data_writer()
            .append_value(vec![0u8; 2 * 8 * 8]);
        builder.finish_band().unwrap();
        builder
            .start_band(
                None,
                &["y", "x"],
                &[8, 8],
                BandDataType::UInt8,
                Some(&[0u8]),
                Some(&path),
                Some("geotiff"),
            )
            .unwrap();
        builder.band_data_writer().append_value([]);
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();
        let raster_struct = builder.finish().unwrap();
        let raster_array = RasterStructArray::new(&raster_struct);
        let raster = raster_array.get(0).unwrap();
        let cache = Rc::new(GDALDatasetCache::try_new(4, 4).unwrap());

        let err = with_gdal(|gdal| {
            let provider = GDALDatasetProvider::new(gdal, Rc::clone(&cache));
            provider.raster_ref_to_gdal(&raster)
        })
        .err()
        .unwrap();
        assert!(err.to_string().contains("2-dim band"), "got: {err}");
    }

    #[test]
    fn test_provider_selects_outdb_band_via_band_fragment() {
        let temp_dir = TempDir::new().unwrap();
        // Source TIFF: band 1 filled with 7s, band 2 filled with 99s.
        let path = create_two_band_source_tiff(&temp_dir, 7u8, 99u8);
        // SedonaDB convention #band=2 → second source band.
        let uri = format!("{path}#band=2");
        let raster_struct = build_outdb_raster(&uri);
        let raster_array = RasterStructArray::new(&raster_struct);
        let raster = raster_array.get(0).unwrap();
        let cache = Rc::new(GDALDatasetCache::try_new(4, 4).unwrap());

        let dataset = with_gdal(|gdal| {
            let provider = GDALDatasetProvider::new(gdal, Rc::clone(&cache));
            provider.raster_ref_to_gdal(&raster)
        })
        .unwrap();

        let band = dataset
            .as_dataset()
            .rasterband(1)
            .unwrap()
            .read_as::<u8>((0, 0), (8, 8), (8, 8), None)
            .unwrap();
        assert_eq!(band.data().to_vec(), vec![99u8; 8 * 8]);
    }
}
