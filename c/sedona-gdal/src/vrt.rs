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

//! GDAL VRT (Virtual Raster) API wrappers.

use std::ffi::CString;
use std::ops::{Deref, DerefMut};
use std::ptr::{null, null_mut};

use crate::cpl::CslStringList;
use crate::dataset::Dataset;
use crate::errors::Result;
use crate::gdal_api::{call_gdal_api, GdalApi};
use crate::raster::rasterband::RasterBand;
use crate::{gdal_dyn_bindgen::*, raster::types::GdalDataType};

/// Special value indicating that nodata is not set for a VRT source.
/// Matches `VRT_NODATA_UNSET` from GDAL's `gdal_vrt.h`.
pub const NODATA_UNSET: f64 = -1234.56;

/// A VRT (Virtual Raster) dataset.
pub struct VrtDataset {
    dataset: Dataset,
}

impl VrtDataset {
    /// Create an empty VRT dataset with the given raster size.
    pub fn create(api: &'static GdalApi, x_size: usize, y_size: usize) -> Result<Self> {
        let x: i32 = x_size.try_into()?;
        let y: i32 = y_size.try_into()?;
        let c_dataset = unsafe { call_gdal_api!(api, VRTCreate, x, y) };

        if c_dataset.is_null() {
            return Err(api.last_null_pointer_err("VRTCreate"));
        }

        Ok(VrtDataset {
            dataset: Dataset::new(api, c_dataset),
        })
    }

    /// Return the underlying `Dataset`, transferring ownership.
    pub fn as_dataset(self) -> Dataset {
        let VrtDataset { dataset } = self;
        dataset
    }

    /// Add a band to this VRT dataset.
    /// Return the 1-based index of the new band.
    pub fn add_band(&mut self, data_type: GdalDataType, options: Option<&[&str]>) -> Result<usize> {
        let csl = CslStringList::try_from_iter(options.unwrap_or(&[]).iter().copied())?;

        // Preserve null semantics: pass null when no options given.
        let opts_ptr = if csl.is_empty() {
            null_mut()
        } else {
            csl.as_ptr()
        };

        let rv = unsafe {
            call_gdal_api!(
                self.dataset.api(),
                GDALAddBand,
                self.dataset.c_dataset(),
                data_type.to_c(),
                opts_ptr
            )
        };

        if rv != CE_None {
            return Err(self.dataset.api().last_cpl_err(rv as u32));
        }

        Ok(self.raster_count())
    }

    /// Fetch a VRT band by 1-indexed band number.
    pub fn rasterband(&self, band_index: usize) -> Result<VrtRasterBand<'_>> {
        let band = self.dataset.rasterband(band_index)?;
        Ok(VrtRasterBand { band })
    }
}

impl Deref for VrtDataset {
    type Target = Dataset;

    fn deref(&self) -> &Self::Target {
        &self.dataset
    }
}

impl DerefMut for VrtDataset {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.dataset
    }
}

impl AsRef<Dataset> for VrtDataset {
    fn as_ref(&self) -> &Dataset {
        &self.dataset
    }
}

/// A raster band within a VRT dataset.
pub struct VrtRasterBand<'a> {
    band: RasterBand<'a>,
}

impl<'a> VrtRasterBand<'a> {
    /// Return the raw GDAL raster band handle.
    pub fn c_rasterband(&self) -> GDALRasterBandH {
        self.band.c_rasterband()
    }

    /// Add a simple source to this VRT band.
    /// Map a source window to a destination window, with optional resampling and nodata.
    pub fn add_simple_source(
        &self,
        source_band: &RasterBand<'_>,
        src_window: (i32, i32, i32, i32),
        dst_window: (i32, i32, i32, i32),
        resampling: Option<&str>,
        nodata: Option<f64>,
    ) -> Result<()> {
        let c_resampling = resampling.map(CString::new).transpose()?;

        let resampling_ptr = c_resampling.as_ref().map(|s| s.as_ptr()).unwrap_or(null());

        let nodata_value = nodata.unwrap_or(NODATA_UNSET);

        let rv = unsafe {
            call_gdal_api!(
                self.band.api(),
                VRTAddSimpleSource,
                self.band.c_rasterband(),
                source_band.c_rasterband(),
                src_window.0,
                src_window.1,
                src_window.2,
                src_window.3,
                dst_window.0,
                dst_window.1,
                dst_window.2,
                dst_window.3,
                resampling_ptr,
                nodata_value
            )
        };

        if rv != CE_None {
            return Err(self.band.api().last_cpl_err(rv as u32));
        }
        Ok(())
    }

    /// Set the nodata value for this VRT band.
    pub fn set_no_data_value(&self, nodata: f64) -> Result<()> {
        let rv = unsafe {
            call_gdal_api!(
                self.band.api(),
                GDALSetRasterNoDataValue,
                self.band.c_rasterband(),
                nodata
            )
        };

        if rv != CE_None {
            return Err(self.band.api().last_cpl_err(rv as u32));
        }
        Ok(())
    }
}

impl<'a> Deref for VrtRasterBand<'a> {
    type Target = RasterBand<'a>;

    fn deref(&self) -> &Self::Target {
        &self.band
    }
}

impl<'a> DerefMut for VrtRasterBand<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.band
    }
}

impl<'a> AsRef<RasterBand<'a>> for VrtRasterBand<'a> {
    fn as_ref(&self) -> &RasterBand<'a> {
        &self.band
    }
}

#[cfg(all(test, feature = "gdal-sys"))]
mod tests {
    use crate::dataset::Dataset;
    use crate::gdal_dyn_bindgen::GDAL_OF_READONLY;
    use crate::global::with_global_gdal_api;
    use crate::raster::types::GdalDataType;
    use crate::vrt::{VrtDataset, NODATA_UNSET};

    fn fixture(name: &str) -> String {
        sedona_testing::data::test_raster(name).unwrap()
    }

    #[test]
    fn test_vrt_create() {
        with_global_gdal_api(|api| {
            let vrt = VrtDataset::create(api, 100, 100).unwrap();
            assert_eq!(vrt.raster_count(), 0);
            assert!(!vrt.c_dataset().is_null());
        })
        .unwrap();
    }

    #[test]
    fn test_vrt_add_band() {
        with_global_gdal_api(|api| {
            let mut vrt = VrtDataset::create(api, 100, 100).unwrap();
            let band_idx = vrt.add_band(GdalDataType::Float32, None).unwrap();
            assert_eq!(band_idx, 1);
            assert_eq!(vrt.raster_count(), 1);

            let band_idx = vrt.add_band(GdalDataType::UInt8, None).unwrap();
            assert_eq!(band_idx, 2);
            assert_eq!(vrt.raster_count(), 2);
        })
        .unwrap();
    }

    #[test]
    fn test_vrt_set_geo_transform() {
        with_global_gdal_api(|api| {
            let vrt = VrtDataset::create(api, 100, 100).unwrap();
            let transform = [0.0, 1.0, 0.0, 100.0, 0.0, -1.0];
            vrt.set_geo_transform(&transform).unwrap();
            assert_eq!(vrt.geo_transform().unwrap(), transform);
        })
        .unwrap();
    }

    #[test]
    fn test_vrt_set_projection() {
        with_global_gdal_api(|api| {
            let vrt = VrtDataset::create(api, 100, 100).unwrap();
            vrt.set_projection("EPSG:4326").unwrap();
            assert!(vrt.projection().contains("4326"));
        })
        .unwrap();
    }

    #[test]
    fn test_vrt_add_simple_source() {
        with_global_gdal_api(|api| {
            let source = Dataset::open_ex(
                api,
                &fixture("tinymarble.tif"),
                GDAL_OF_READONLY,
                None,
                None,
                None,
            )
            .unwrap();
            let source_band_type = source.rasterband(1).unwrap().band_type();

            let mut vrt = VrtDataset::create(api, 1, 1).unwrap();
            vrt.add_band(source_band_type, None).unwrap();

            let source_band = source.rasterband(1).unwrap();
            let vrt_band = vrt.rasterband(1).unwrap();

            vrt_band
                .add_simple_source(&source_band, (0, 0, 1, 1), (0, 0, 1, 1), None, None)
                .unwrap();

            let source_px = source_band
                .read_as::<f64>((0, 0), (1, 1), (1, 1), None)
                .unwrap()
                .data()[0];
            let vrt_px = vrt_band
                .read_as::<f64>((0, 0), (1, 1), (1, 1), None)
                .unwrap()
                .data()[0];

            assert_eq!(vrt_px, source_px);
        })
        .unwrap();
    }

    #[test]
    fn test_vrt_nodata_unset() {
        assert_eq!(NODATA_UNSET, -1234.56);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_vrt_set_no_data_value() {
        with_global_gdal_api(|api| {
            let mut vrt = VrtDataset::create(api, 1, 1).unwrap();
            vrt.add_band(GdalDataType::UInt8, None).unwrap();
            let band = vrt.rasterband(1).unwrap();
            band.set_no_data_value(-9999.0).unwrap();
            assert_eq!(band.no_data_value(), Some(-9999.0));
        })
        .unwrap();
    }
}
