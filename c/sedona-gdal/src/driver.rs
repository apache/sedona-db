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
//! <https://github.com/georust/gdal/blob/v0.19.0/src/driver.rs>.
//! Original code is licensed under MIT.

use std::ffi::CString;
use std::ptr;

use crate::dataset::Dataset;
use crate::errors::Result;
use crate::gdal_api::{call_gdal_api, GdalApi};
use crate::gdal_dyn_bindgen::*;
use crate::raster::types::GdalDataType as RustGdalDataType;
use crate::raster::types::GdalType;

/// A GDAL driver.
pub struct Driver {
    api: &'static GdalApi,
    c_driver: GDALDriverH,
}

impl Driver {
    /// Wrap an existing C driver handle.
    ///
    /// # Safety
    ///
    /// The caller must ensure the handle is valid.
    pub unsafe fn from_c_driver(api: &'static GdalApi, c_driver: GDALDriverH) -> Self {
        Self { api, c_driver }
    }

    /// Return the raw C driver handle.
    pub fn c_driver(&self) -> GDALDriverH {
        self.c_driver
    }

    /// Create a new raster dataset (with u8 band type).
    pub fn create(
        &self,
        filename: &str,
        size_x: usize,
        size_y: usize,
        bands: usize,
    ) -> Result<Dataset> {
        self.create_with_band_type::<u8>(filename, size_x, size_y, bands)
    }

    /// Create a new raster dataset with a specific band type.
    pub fn create_with_band_type<T: GdalType>(
        &self,
        filename: &str,
        size_x: usize,
        size_y: usize,
        bands: usize,
    ) -> Result<Dataset> {
        let c_filename = CString::new(filename)?;
        let x: i32 = size_x.try_into()?;
        let y: i32 = size_y.try_into()?;
        let b: i32 = bands.try_into()?;
        let c_ds = unsafe {
            call_gdal_api!(
                self.api,
                GDALCreate,
                self.c_driver,
                c_filename.as_ptr(),
                x,
                y,
                b,
                T::gdal_ordinal(),
                ptr::null_mut()
            )
        };
        if c_ds.is_null() {
            return Err(self.api.last_cpl_err(CE_Failure as u32));
        }
        Ok(Dataset::new(self.api, c_ds))
    }

    /// Create a new raster dataset with a runtime data type.
    ///
    /// Unlike [`create_with_band_type`](Self::create_with_band_type), this accepts a
    /// [`GdalDataType`](RustGdalDataType) enum value instead of a compile-time generic,
    /// which is useful when the data type is only known at runtime.
    pub fn create_with_data_type(
        &self,
        filename: &str,
        size_x: usize,
        size_y: usize,
        bands: usize,
        data_type: RustGdalDataType,
    ) -> Result<Dataset> {
        let c_filename = CString::new(filename)?;
        let x: i32 = size_x.try_into()?;
        let y: i32 = size_y.try_into()?;
        let b: i32 = bands.try_into()?;
        let c_ds = unsafe {
            call_gdal_api!(
                self.api,
                GDALCreate,
                self.c_driver,
                c_filename.as_ptr(),
                x,
                y,
                b,
                data_type.to_c(),
                ptr::null_mut()
            )
        };
        if c_ds.is_null() {
            return Err(self.api.last_cpl_err(CE_Failure as u32));
        }
        Ok(Dataset::new(self.api, c_ds))
    }

    /// Create a new dataset (vector-only, no raster bands).
    pub fn create_vector_only(&self, filename: &str) -> Result<Dataset> {
        let c_filename = CString::new(filename)?;
        let c_ds = unsafe {
            call_gdal_api!(
                self.api,
                GDALCreate,
                self.c_driver,
                c_filename.as_ptr(),
                1,
                1,
                0,
                GDALDataType::GDT_Unknown,
                ptr::null_mut()
            )
        };
        if c_ds.is_null() {
            return Err(self.api.last_cpl_err(CE_Failure as u32));
        }
        Ok(Dataset::new(self.api, c_ds))
    }
}

/// Driver manager for looking up drivers by name.
pub struct DriverManager;

impl DriverManager {
    pub fn get_driver_by_name(api: &'static GdalApi, name: &str) -> Result<Driver> {
        let c_name = CString::new(name)?;
        let c_driver = unsafe { call_gdal_api!(api, GDALGetDriverByName, c_name.as_ptr()) };
        if c_driver.is_null() {
            return Err(api.last_null_pointer_err("GDALGetDriverByName"));
        }
        Ok(Driver { api, c_driver })
    }
}

#[cfg(all(test, feature = "gdal-sys"))]
mod tests {
    use crate::driver::DriverManager;
    use crate::errors::GdalError;
    use crate::global::with_global_gdal_api;
    use crate::raster::types::GdalDataType;

    #[test]
    fn test_get_driver_by_name() {
        with_global_gdal_api(|api| {
            let gtiff = DriverManager::get_driver_by_name(api, "GTiff").unwrap();
            assert!(!gtiff.c_driver().is_null());
            let mem = DriverManager::get_driver_by_name(api, "MEM").unwrap();
            assert!(!mem.c_driver().is_null());
        })
        .unwrap();
    }

    #[test]
    fn test_get_driver_by_name_invalid() {
        with_global_gdal_api(|api| {
            let err = DriverManager::get_driver_by_name(api, "NO_SUCH_DRIVER");
            assert!(matches!(err, Err(GdalError::NullPointer { .. })));
        })
        .unwrap();
    }

    #[test]
    fn test_driver_create() {
        with_global_gdal_api(|api| {
            let driver = DriverManager::get_driver_by_name(api, "MEM").unwrap();
            let ds = driver.create("", 32, 16, 2).unwrap();
            assert_eq!(ds.raster_size(), (32, 16));
            assert_eq!(ds.raster_count(), 2);
        })
        .unwrap();
    }

    #[test]
    fn test_driver_create_with_band_type() {
        with_global_gdal_api(|api| {
            let driver = DriverManager::get_driver_by_name(api, "MEM").unwrap();
            let ds = driver.create_with_band_type::<u8>("", 10, 20, 1).unwrap();
            assert_eq!(ds.raster_count(), 1);
            let ds = driver.create_with_band_type::<f32>("", 10, 20, 2).unwrap();
            assert_eq!(ds.raster_count(), 2);
            let ds = driver.create_with_band_type::<i16>("", 10, 20, 3).unwrap();
            assert_eq!(ds.raster_count(), 3);
        })
        .unwrap();
    }

    #[test]
    fn test_driver_create_with_data_type() {
        with_global_gdal_api(|api| {
            let driver = DriverManager::get_driver_by_name(api, "MEM").unwrap();
            let ds = driver
                .create_with_data_type("", 8, 8, 1, GdalDataType::UInt16)
                .unwrap();
            assert_eq!(ds.raster_count(), 1);
        })
        .unwrap();
    }

    #[test]
    fn test_driver_create_vector_only() {
        with_global_gdal_api(|api| {
            let driver = DriverManager::get_driver_by_name(api, "MEM").unwrap();
            let ds = driver.create_vector_only("").unwrap();
            assert_eq!(ds.raster_count(), 0);
            assert_eq!(ds.raster_size(), (1, 1));
        })
        .unwrap();
    }
}
