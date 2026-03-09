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

//! Ported (and contains copied code) from georust/gdal:
//! <https://github.com/georust/gdal/blob/v0.19.0/src/dataset.rs>.
//! Original code is licensed under MIT.

use std::ffi::{CStr, CString};
use std::ptr;

use crate::cpl::CslStringList;
use crate::errors::{GdalError, Result};
use crate::gdal_api::{call_gdal_api, GdalApi};
use crate::gdal_dyn_bindgen::*;
use crate::raster::types::{DatasetOptions, GdalDataType as RustGdalDataType};
use crate::raster::RasterBand;
use crate::spatial_ref::SpatialRef;
use crate::vector::Layer;

/// A GDAL dataset.
pub struct Dataset {
    api: &'static GdalApi,
    c_dataset: GDALDatasetH,
    owned: bool,
}

unsafe impl Send for Dataset {}

impl Drop for Dataset {
    fn drop(&mut self) {
        if self.owned && !self.c_dataset.is_null() {
            unsafe { call_gdal_api!(self.api, GDALClose, self.c_dataset) };
        }
    }
}

impl Dataset {
    /// Open a dataset with extended options.
    pub fn open_ex(
        api: &'static GdalApi,
        path: &str,
        open_flags: GDALOpenFlags,
        allowed_drivers: Option<&[&str]>,
        open_options: Option<&[&str]>,
        sibling_files: Option<&[&str]>,
    ) -> Result<Self> {
        let c_path = CString::new(path)?;

        // Build CslStringLists from Option<&[&str]>.
        // None → null pointer (use GDAL default).
        // Some(&[]) → pointer to [null] (explicitly empty list).
        let drivers_csl = allowed_drivers
            .map(|v| CslStringList::try_from_iter(v.iter().copied()))
            .transpose()?;
        let options_csl = open_options
            .map(|v| CslStringList::try_from_iter(v.iter().copied()))
            .transpose()?;
        let siblings_csl = sibling_files
            .map(|v| CslStringList::try_from_iter(v.iter().copied()))
            .transpose()?;

        let c_dataset = unsafe {
            call_gdal_api!(
                api,
                GDALOpenEx,
                c_path.as_ptr(),
                open_flags,
                drivers_csl
                    .as_ref()
                    .map_or(ptr::null(), |csl| csl.as_ptr() as *const *const _),
                options_csl
                    .as_ref()
                    .map_or(ptr::null(), |csl| csl.as_ptr() as *const *const _),
                siblings_csl
                    .as_ref()
                    .map_or(ptr::null(), |csl| csl.as_ptr() as *const *const _)
            )
        };

        if c_dataset.is_null() {
            return Err(api.last_cpl_err(CE_Failure as u32));
        }

        Ok(Self {
            api,
            c_dataset,
            owned: true,
        })
    }

    /// Create a new owned Dataset from a C handle.
    pub(crate) fn new_owned(api: &'static GdalApi, c_dataset: GDALDatasetH) -> Self {
        Self {
            api,
            c_dataset,
            owned: true,
        }
    }

    /// Wrap an existing C dataset handle (non-owning).
    ///
    /// # Safety
    ///
    /// The caller must ensure the handle is valid and outlives this `Dataset`.
    pub unsafe fn from_c_dataset(api: &'static GdalApi, c_dataset: GDALDatasetH) -> Self {
        Self {
            api,
            c_dataset,
            owned: false,
        }
    }

    /// Return the raw C dataset handle.
    pub fn c_dataset(&self) -> GDALDatasetH {
        self.c_dataset
    }

    /// Return raster size as (x_size, y_size).
    pub fn raster_size(&self) -> (usize, usize) {
        let x = unsafe { call_gdal_api!(self.api, GDALGetRasterXSize, self.c_dataset) };
        let y = unsafe { call_gdal_api!(self.api, GDALGetRasterYSize, self.c_dataset) };
        (x as usize, y as usize)
    }

    /// Return the number of raster bands.
    pub fn raster_count(&self) -> usize {
        unsafe { call_gdal_api!(self.api, GDALGetRasterCount, self.c_dataset) as usize }
    }

    /// Get a raster band (1-indexed).
    pub fn rasterband(&self, band_index: usize) -> Result<RasterBand<'_>> {
        let band_index_i32 = i32::try_from(band_index)?;
        let c_band =
            unsafe { call_gdal_api!(self.api, GDALGetRasterBand, self.c_dataset, band_index_i32) };
        if c_band.is_null() {
            return Err(GdalError::NullPointer {
                method_name: "GDALGetRasterBand",
                msg: format!("band index {band_index}"),
            });
        }
        Ok(RasterBand::new(self.api, c_band, self))
    }

    /// Get the geo-transform.
    pub fn geo_transform(&self) -> Result<[f64; 6]> {
        let mut gt = [0.0f64; 6];
        let rv = unsafe {
            call_gdal_api!(
                self.api,
                GDALGetGeoTransform,
                self.c_dataset,
                gt.as_mut_ptr()
            )
        };
        if rv != CE_None {
            return Err(self.api.last_cpl_err(rv as u32));
        }
        Ok(gt)
    }

    /// Set the geo-transform.
    pub fn set_geo_transform(&self, gt: &[f64; 6]) -> Result<()> {
        let rv = unsafe {
            call_gdal_api!(
                self.api,
                GDALSetGeoTransform,
                self.c_dataset,
                gt.as_ptr() as *mut f64
            )
        };
        if rv != CE_None {
            return Err(self.api.last_cpl_err(rv as u32));
        }
        Ok(())
    }

    /// Get the projection string.
    pub fn projection(&self) -> String {
        unsafe {
            let ptr = call_gdal_api!(self.api, GDALGetProjectionRef, self.c_dataset);
            if ptr.is_null() {
                String::new()
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        }
    }

    /// Set the projection string.
    pub fn set_projection(&self, projection: &str) -> Result<()> {
        let c_projection = CString::new(projection)?;
        let rv = unsafe {
            call_gdal_api!(
                self.api,
                GDALSetProjection,
                self.c_dataset,
                c_projection.as_ptr()
            )
        };
        if rv != CE_None {
            return Err(self.api.last_cpl_err(rv as u32));
        }
        Ok(())
    }

    /// Get the spatial reference.
    pub fn spatial_ref(&self) -> Result<SpatialRef> {
        let c_srs = unsafe { call_gdal_api!(self.api, GDALGetSpatialRef, self.c_dataset) };
        if c_srs.is_null() {
            return Err(GdalError::NullPointer {
                method_name: "GDALGetSpatialRef",
                msg: "returned null".to_string(),
            });
        }
        // GDALGetSpatialRef returns a borrowed reference — clone it via OSRClone.
        unsafe { SpatialRef::from_c_srs_clone(self.api, c_srs) }
    }

    /// Set the spatial reference.
    pub fn set_spatial_ref(&self, srs: &SpatialRef) -> Result<()> {
        let rv =
            unsafe { call_gdal_api!(self.api, GDALSetSpatialRef, self.c_dataset, srs.c_srs()) };
        if rv != CE_None {
            return Err(self.api.last_cpl_err(rv as u32));
        }
        Ok(())
    }

    /// Create a copy of this dataset to a new file using the given driver.
    pub fn create_copy(
        &self,
        driver: &crate::driver::Driver,
        filename: &str,
        options: &[&str],
    ) -> Result<Dataset> {
        let c_filename = CString::new(filename)?;
        let csl = CslStringList::try_from_iter(options.iter().copied())?;

        let c_ds = unsafe {
            call_gdal_api!(
                self.api,
                GDALCreateCopy,
                driver.c_driver(),
                c_filename.as_ptr(),
                self.c_dataset,
                0, // bStrict
                csl.as_ptr(),
                ptr::null_mut(),
                ptr::null_mut()
            )
        };
        if c_ds.is_null() {
            return Err(self.api.last_cpl_err(CE_Failure as u32));
        }
        Ok(Dataset {
            api: self.api,
            c_dataset: c_ds,
            owned: true,
        })
    }

    /// Create a new vector layer.
    pub fn create_layer(&self, options: LayerOptions<'_>) -> Result<Layer<'_>> {
        let c_name = CString::new(options.name)?;
        let c_srs = options.srs.map_or(ptr::null_mut(), |s| s.c_srs());

        let csl = CslStringList::try_from_iter(options.options.unwrap_or(&[]).iter().copied())?;

        let c_layer = unsafe {
            call_gdal_api!(
                self.api,
                GDALDatasetCreateLayer,
                self.c_dataset,
                c_name.as_ptr(),
                c_srs,
                options.ty,
                csl.as_ptr()
            )
        };
        if c_layer.is_null() {
            return Err(GdalError::NullPointer {
                method_name: "GDALDatasetCreateLayer",
                msg: format!("failed to create layer '{}'", options.name),
            });
        }
        Ok(Layer::new(self.api, c_layer, self))
    }

    /// Get the GDAL API reference.
    pub fn api(&self) -> &'static GdalApi {
        self.api
    }

    /// Open a dataset using a `DatasetOptions` struct (georust-compatible convenience).
    pub fn open_ex_with_options(
        api: &'static GdalApi,
        path: &str,
        options: DatasetOptions<'_>,
    ) -> Result<Self> {
        Self::open_ex(
            api,
            path,
            options.open_flags,
            options.allowed_drivers,
            options.open_options,
            options.sibling_files,
        )
    }

    /// Add a raster band backed by an existing memory buffer (zero-copy).
    ///
    /// This wraps `GDALAddBand` with the `DATAPOINTER`, `PIXELOFFSET`, and `LINEOFFSET`
    /// options, allowing you to attach existing memory to a MEM dataset without copying.
    ///
    /// # Arguments
    /// * `data_type` - The GDAL data type of the band.
    /// * `data_ptr` - Pointer to the band pixel data.
    /// * `pixel_offset` - Byte offset between consecutive pixels. `None` defaults to the
    ///   byte size of `data_type`.
    /// * `line_offset` - Byte offset between consecutive lines. `None` defaults to
    ///   `pixel_offset * width`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `data_ptr` points to a valid buffer of at least
    /// `height * line_offset` bytes (or `height * width * data_type.byte_size()` when
    /// using defaults), and that the buffer outlives this dataset.
    pub unsafe fn add_band_with_data(
        &self,
        data_type: RustGdalDataType,
        data_ptr: *const u8,
        pixel_offset: Option<i64>,
        line_offset: Option<i64>,
    ) -> Result<()> {
        let data_pointer = format!("DATAPOINTER={data_ptr:p}");

        let mut options = CslStringList::with_capacity(3);
        options.add_string(&data_pointer)?;

        if let Some(pixel) = pixel_offset {
            options.set_name_value("PIXELOFFSET", &pixel.to_string())?;
        }

        if let Some(line) = line_offset {
            options.set_name_value("LINEOFFSET", &line.to_string())?;
        }

        let err = call_gdal_api!(
            self.api,
            GDALAddBand,
            self.c_dataset,
            data_type.to_c(),
            options.as_ptr()
        );
        if err != CE_None {
            return Err(self.api.last_cpl_err(err as u32));
        }
        Ok(())
    }

    /// Mark this dataset as owning its handle (for `Drop`).
    pub fn set_owned(&mut self, owned: bool) {
        self.owned = owned;
    }
}

/// Options for creating a vector layer.
pub struct LayerOptions<'a> {
    pub name: &'a str,
    pub srs: Option<&'a SpatialRef>,
    pub ty: OGRwkbGeometryType,
    /// Additional driver-specific options, in the form `"name=value"`.
    pub options: Option<&'a [&'a str]>,
}

impl Default for LayerOptions<'_> {
    fn default() -> Self {
        Self {
            name: "",
            srs: None,
            ty: OGRwkbGeometryType::wkbUnknown,
            options: None,
        }
    }
}

#[cfg(all(test, feature = "gdal-sys"))]
mod tests {
    use crate::driver::DriverManager;
    use crate::global::with_global_gdal_api;

    #[test]
    fn test_geo_transform_roundtrip() {
        with_global_gdal_api(|api| {
            let driver = DriverManager::get_driver_by_name(api, "MEM").unwrap();
            let ds = driver.create("", 256, 256, 1).unwrap();

            let gt = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
            ds.set_geo_transform(&gt).unwrap();
            let got = ds.geo_transform().unwrap();
            assert_eq!(gt, got);
        })
        .unwrap();
    }

    #[test]
    fn test_geo_transform_unset() {
        with_global_gdal_api(|api| {
            let driver = DriverManager::get_driver_by_name(api, "MEM").unwrap();
            let ds = driver.create("", 256, 256, 1).unwrap();

            // MEM driver without an explicit set_geo_transform returns an error
            assert!(ds.geo_transform().is_err());
        })
        .unwrap();
    }

    #[test]
    fn test_set_projection_roundtrip() {
        with_global_gdal_api(|api| {
            let driver = DriverManager::get_driver_by_name(api, "MEM").unwrap();
            let ds = driver.create("", 256, 256, 1).unwrap();

            let wkt = r#"GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]"#;
            ds.set_projection(wkt).unwrap();
            let got = ds.projection();
            // The returned WKT may be reformatted by GDAL, so just check it contains WGS 84
            assert!(got.contains("WGS 84"), "Expected WGS 84 in: {got}");
        })
        .unwrap();
    }

    #[test]
    fn test_dataset_raster_count() {
        with_global_gdal_api(|api| {
            let driver = DriverManager::get_driver_by_name(api, "MEM").unwrap();

            let ds1 = driver.create("", 64, 64, 1).unwrap();
            assert_eq!(ds1.raster_count(), 1);

            let ds3 = driver.create("", 64, 64, 3).unwrap();
            assert_eq!(ds3.raster_count(), 3);
        })
        .unwrap();
    }

    #[test]
    fn test_dataset_raster_size() {
        with_global_gdal_api(|api| {
            let driver = DriverManager::get_driver_by_name(api, "MEM").unwrap();
            let ds = driver.create("", 123, 456, 1).unwrap();
            assert_eq!(ds.raster_size(), (123, 456));
        })
        .unwrap();
    }
}
