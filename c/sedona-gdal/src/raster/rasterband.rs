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
//! <https://github.com/georust/gdal/blob/v0.19.0/src/raster/rasterband.rs>.
//! Original code is licensed under MIT.

use std::marker::PhantomData;

use crate::dataset::Dataset;
use crate::errors::{GdalError, Result};
use crate::gdal_api::{call_gdal_api, GdalApi};
use crate::raster::types::{Buffer, GdalType, ResampleAlg};
use crate::{gdal_dyn_bindgen::*, raster::types::GdalDataType};

/// A raster band of a dataset.
pub struct RasterBand<'a> {
    api: &'static GdalApi,
    c_rasterband: GDALRasterBandH,
    _dataset: PhantomData<&'a Dataset>,
}

impl<'a> RasterBand<'a> {
    pub(crate) fn new(
        api: &'static GdalApi,
        c_rasterband: GDALRasterBandH,
        _dataset: &'a Dataset,
    ) -> Self {
        Self {
            api,
            c_rasterband,
            _dataset: PhantomData,
        }
    }

    /// Return the raw C raster band handle.
    pub fn c_rasterband(&self) -> GDALRasterBandH {
        self.c_rasterband
    }

    /// Read a window of this band into a typed buffer.
    /// If `e_resample_alg` is `None`, use nearest-neighbour resampling.
    pub fn read_as<T: GdalType + Copy>(
        &self,
        window: (isize, isize),
        window_size: (usize, usize),
        size: (usize, usize),
        e_resample_alg: Option<ResampleAlg>,
    ) -> Result<Buffer<T>> {
        let len = size.0 * size.1;
        // Safety: all GdalType implementations are numeric primitives (u8, i8, u16, ..., f64),
        // for which zeroed memory is a valid bit pattern.
        let mut data: Vec<T> = vec![unsafe { std::mem::zeroed() }; len];

        let resample_alg = e_resample_alg.unwrap_or(ResampleAlg::NearestNeighbour);
        let mut extra_arg = GDALRasterIOExtraArg {
            eResampleAlg: resample_alg.to_gdal(),
            ..GDALRasterIOExtraArg::default()
        };

        let rv = unsafe {
            call_gdal_api!(
                self.api,
                GDALRasterIOEx,
                self.c_rasterband,
                GF_Read,
                i32::try_from(window.0)?,
                i32::try_from(window.1)?,
                i32::try_from(window_size.0)?,
                i32::try_from(window_size.1)?,
                data.as_mut_ptr() as *mut std::ffi::c_void,
                i32::try_from(size.0)?,
                i32::try_from(size.1)?,
                T::gdal_ordinal(),
                0, // nPixelSpace (auto)
                0, // nLineSpace (auto)
                &mut extra_arg
            )
        };
        if rv != CE_None {
            return Err(self.api.last_cpl_err(rv as u32));
        }

        Ok(Buffer::new(size, data))
    }

    /// Write a buffer to this raster band.
    pub fn write<T: GdalType + Copy>(
        &self,
        window: (isize, isize),
        window_size: (usize, usize),
        buffer: &mut Buffer<T>,
    ) -> Result<()> {
        let expected_len = buffer.shape.0 * buffer.shape.1;
        if buffer.data.len() != expected_len {
            return Err(GdalError::BufferSizeMismatch(
                buffer.data.len(),
                buffer.shape,
            ));
        }
        let rv = unsafe {
            call_gdal_api!(
                self.api,
                GDALRasterIO,
                self.c_rasterband,
                GF_Write,
                i32::try_from(window.0)?,
                i32::try_from(window.1)?,
                i32::try_from(window_size.0)?,
                i32::try_from(window_size.1)?,
                buffer.data.as_mut_ptr() as *mut std::ffi::c_void,
                i32::try_from(buffer.shape.0)?,
                i32::try_from(buffer.shape.1)?,
                T::gdal_ordinal(),
                0, // nPixelSpace (auto)
                0  // nLineSpace (auto)
            )
        };
        if rv != CE_None {
            return Err(self.api.last_cpl_err(rv as u32));
        }
        Ok(())
    }

    /// Fetch this band's data type.
    pub fn band_type(&self) -> GdalDataType {
        GdalDataType::from_c(self.c_band_type()).unwrap_or(GdalDataType::Unknown)
    }

    /// Fetch this band's raw GDAL data type.
    pub fn c_band_type(&self) -> GDALDataType {
        unsafe { call_gdal_api!(self.api, GDALGetRasterDataType, self.c_rasterband) }
    }

    /// Fetch band size as `(x_size, y_size)`.
    pub fn size(&self) -> (usize, usize) {
        let x = unsafe { call_gdal_api!(self.api, GDALGetRasterBandXSize, self.c_rasterband) };
        let y = unsafe { call_gdal_api!(self.api, GDALGetRasterBandYSize, self.c_rasterband) };
        (x as usize, y as usize)
    }

    /// Fetch the natural block size as `(x_size, y_size)`.
    pub fn block_size(&self) -> (usize, usize) {
        let mut x: i32 = 0;
        let mut y: i32 = 0;
        unsafe {
            call_gdal_api!(
                self.api,
                GDALGetBlockSize,
                self.c_rasterband,
                &mut x,
                &mut y
            )
        };
        (x as usize, y as usize)
    }

    /// Fetch the band's nodata value.
    /// Return `None` if no nodata value is set.
    pub fn no_data_value(&self) -> Option<f64> {
        let mut success: i32 = 0;
        let value = unsafe {
            call_gdal_api!(
                self.api,
                GDALGetRasterNoDataValue,
                self.c_rasterband,
                &mut success
            )
        };
        if success != 0 {
            Some(value)
        } else {
            None
        }
    }

    /// Set the band's nodata value.
    /// Pass `None` to clear any existing nodata value.
    pub fn set_no_data_value(&self, value: Option<f64>) -> Result<()> {
        let rv = if let Some(val) = value {
            unsafe { call_gdal_api!(self.api, GDALSetRasterNoDataValue, self.c_rasterband, val) }
        } else {
            unsafe { call_gdal_api!(self.api, GDALDeleteRasterNoDataValue, self.c_rasterband) }
        };
        if rv != CE_None {
            return Err(self.api.last_cpl_err(rv as u32));
        }
        Ok(())
    }

    /// Set the band's nodata value as `u64`.
    /// Pass `None` to clear any existing nodata value.
    pub fn set_no_data_value_u64(&self, value: Option<u64>) -> Result<()> {
        let rv = if let Some(val) = value {
            unsafe {
                call_gdal_api!(
                    self.api,
                    GDALSetRasterNoDataValueAsUInt64,
                    self.c_rasterband,
                    val
                )
            }
        } else {
            unsafe { call_gdal_api!(self.api, GDALDeleteRasterNoDataValue, self.c_rasterband) }
        };
        if rv != CE_None {
            return Err(self.api.last_cpl_err(rv as u32));
        }
        Ok(())
    }

    /// Set the band's nodata value as `i64`.
    /// Pass `None` to clear any existing nodata value.
    pub fn set_no_data_value_i64(&self, value: Option<i64>) -> Result<()> {
        let rv = if let Some(val) = value {
            unsafe {
                call_gdal_api!(
                    self.api,
                    GDALSetRasterNoDataValueAsInt64,
                    self.c_rasterband,
                    val
                )
            }
        } else {
            unsafe { call_gdal_api!(self.api, GDALDeleteRasterNoDataValue, self.c_rasterband) }
        };
        if rv != CE_None {
            return Err(self.api.last_cpl_err(rv as u32));
        }
        Ok(())
    }

    /// Get the GDAL API reference.
    pub fn api(&self) -> &'static GdalApi {
        self.api
    }
}

/// Return the actual block size for a block index.
/// Clamp edge blocks to the raster extent.
pub fn actual_block_size(
    band: &RasterBand<'_>,
    block_index: (usize, usize),
) -> Result<(usize, usize)> {
    let (block_x, block_y) = band.block_size();
    let (raster_x, raster_y) = band.size();
    let x_off = block_index.0 * block_x;
    let y_off = block_index.1 * block_y;
    if x_off >= raster_x || y_off >= raster_y {
        return Err(GdalError::BadArgument(format!(
            "block index ({}, {}) is out of bounds for raster size ({}, {})",
            block_index.0, block_index.1, raster_x, raster_y
        )));
    }
    let actual_x = if x_off + block_x > raster_x {
        raster_x - x_off
    } else {
        block_x
    };
    let actual_y = if y_off + block_y > raster_y {
        raster_y - y_off
    } else {
        block_y
    };
    Ok((actual_x, actual_y))
}

#[cfg(all(test, feature = "gdal-sys"))]
mod tests {
    use crate::dataset::Dataset;
    use crate::driver::DriverManager;
    use crate::gdal_dyn_bindgen::*;
    use crate::global::with_global_gdal_api;
    use crate::raster::types::ResampleAlg;

    fn fixture(name: &str) -> String {
        sedona_testing::data::test_raster(name).unwrap()
    }

    #[test]
    fn test_read_raster() {
        with_global_gdal_api(|api| {
            let path = fixture("tinymarble.tif");
            let dataset = Dataset::open_ex(api, &path, GDAL_OF_READONLY, None, None, None).unwrap();
            let rb = dataset.rasterband(1).unwrap();
            let rv = rb.read_as::<u8>((20, 30), (2, 3), (2, 3), None).unwrap();
            assert_eq!(rv.shape, (2, 3));
            assert_eq!(rv.data(), [7, 7, 7, 10, 8, 12]);
        })
        .unwrap();
    }

    #[test]
    fn test_read_raster_with_default_resample() {
        with_global_gdal_api(|api| {
            let path = fixture("tinymarble.tif");
            let dataset = Dataset::open_ex(api, &path, GDAL_OF_READONLY, None, None, None).unwrap();
            let rb = dataset.rasterband(1).unwrap();
            let rv = rb.read_as::<u8>((20, 30), (4, 4), (2, 2), None).unwrap();
            assert_eq!(rv.shape, (2, 2));
            // Default is NearestNeighbour; exact values are GDAL-version-dependent
            // when downsampling from 4x4 to 2x2. Just verify shape and non-emptiness.
            assert_eq!(rv.data().len(), 4);
        })
        .unwrap();
    }

    #[test]
    fn test_read_raster_with_average_resample() {
        with_global_gdal_api(|api| {
            let path = fixture("tinymarble.tif");
            let dataset = Dataset::open_ex(api, &path, GDAL_OF_READONLY, None, None, None).unwrap();
            let rb = dataset.rasterband(1).unwrap();
            let rv = rb
                .read_as::<u8>((20, 30), (4, 4), (2, 2), Some(ResampleAlg::Average))
                .unwrap();
            assert_eq!(rv.shape, (2, 2));
            // Average resampling; exact values are GDAL-version-dependent, so just
            // verify that the downsampled result has the expected shape and length.
            assert_eq!(rv.data().len(), 4);
        })
        .unwrap();
    }

    #[test]
    fn test_get_no_data_value() {
        with_global_gdal_api(|api| {
            // tinymarble.tif has no nodata
            let path = fixture("tinymarble.tif");
            let dataset = Dataset::open_ex(api, &path, GDAL_OF_READONLY, None, None, None).unwrap();
            let rb = dataset.rasterband(1).unwrap();
            assert!(rb.no_data_value().is_none());

            // labels.tif has nodata=255
            let path = fixture("labels.tif");
            let dataset = Dataset::open_ex(api, &path, GDAL_OF_READONLY, None, None, None).unwrap();
            let rb = dataset.rasterband(1).unwrap();
            assert_eq!(rb.no_data_value(), Some(255.0));
        })
        .unwrap();
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_set_no_data_value() {
        with_global_gdal_api(|api| {
            let driver = DriverManager::get_driver_by_name(api, "MEM").unwrap();
            let dataset = driver.create("", 20, 10, 1).unwrap();
            let rasterband = dataset.rasterband(1).unwrap();
            assert_eq!(rasterband.no_data_value(), None);
            assert!(rasterband.set_no_data_value(Some(1.23)).is_ok());
            assert_eq!(rasterband.no_data_value(), Some(1.23));
            assert!(rasterband.set_no_data_value(None).is_ok());
            assert_eq!(rasterband.no_data_value(), None);
        })
        .unwrap();
    }
}
