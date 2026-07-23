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

struct RasterIoRequest {
    window: (isize, isize),
    window_size: (usize, usize),
    size: (usize, usize),
    e_resample_alg: Option<ResampleAlg>,
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

        self.read_impl(
            RasterIoRequest {
                window,
                window_size,
                size,
                e_resample_alg,
            },
            data.as_mut_ptr() as *mut std::ffi::c_void,
            T::gdal_ordinal(),
        )?;

        Ok(Buffer::new(size, data))
    }

    /// Read a window of this band into a byte buffer using the band's native GDAL data type.
    ///
    /// The returned bytes use GDAL's in-memory representation for the current platform.
    /// If `e_resample_alg` is `None`, use nearest-neighbour resampling.
    pub fn read_as_bytes(
        &self,
        window: (isize, isize),
        window_size: (usize, usize),
        size: (usize, usize),
        e_resample_alg: Option<ResampleAlg>,
    ) -> Result<Vec<u8>> {
        let len = self.expected_byte_len(size)?;
        let mut data = vec![0u8; len];
        self.read_into_bytes(window, window_size, size, &mut data, e_resample_alg)?;
        Ok(data)
    }

    /// Read a window of this band into a caller-provided byte buffer using the band's native
    /// GDAL data type.
    ///
    /// The buffer length must equal `size.0 * size.1 * band_type().byte_size()`.
    /// If `e_resample_alg` is `None`, use nearest-neighbour resampling.
    pub fn read_into_bytes(
        &self,
        window: (isize, isize),
        window_size: (usize, usize),
        size: (usize, usize),
        data: &mut [u8],
        e_resample_alg: Option<ResampleAlg>,
    ) -> Result<()> {
        let expected_len = self.expected_byte_len(size)?;
        if data.len() != expected_len {
            return Err(GdalError::BadArgument(format!(
                "byte buffer length {} does not match expected size {} for raster window {:?}",
                data.len(),
                expected_len,
                size
            )));
        }

        self.read_impl(
            RasterIoRequest {
                window,
                window_size,
                size,
                e_resample_alg,
            },
            data.as_mut_ptr() as *mut std::ffi::c_void,
            self.c_band_type(),
        )
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

    /// Fetch the band's nodata value as `u64`.
    /// Return `None` if no nodata value is set.
    pub fn no_data_value_u64(&self) -> Option<u64> {
        let mut success: i32 = 0;
        let value = unsafe {
            call_gdal_api!(
                self.api,
                GDALGetRasterNoDataValueAsUInt64,
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

    /// Fetch the band's nodata value as `i64`.
    /// Return `None` if no nodata value is set.
    pub fn no_data_value_i64(&self) -> Option<i64> {
        let mut success: i32 = 0;
        let value = unsafe {
            call_gdal_api!(
                self.api,
                GDALGetRasterNoDataValueAsInt64,
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

    /// Resample a **fractional** source window into a sub-rectangle of a larger
    /// destination byte buffer, using the band's native GDAL data type.
    ///
    /// This is the same-CRS regridding primitive: unlike [`Self::read_as_bytes`]
    /// (whose source window is whole pixels), `src_window` is a floating
    /// `(x_off, y_off, x_size, y_size)` in source-pixel coordinates, so an output
    /// grid whose origin/pixel size is offset from the source grid by a fraction
    /// of a pixel still samples at the correct sub-pixel position (GDAL's
    /// `bFloatingPointWindowValidity`). The `dst_size` resampled pixels are
    /// written at `dst_off` within a `full_width`-pixel-wide destination buffer
    /// (via GDAL's pixel/line spacing), leaving the rest of `dst` untouched — so
    /// a caller pre-fills `dst` with nodata and only the covered sub-window is
    /// overwritten.
    ///
    /// `src_window` must lie within the band extent; the caller computes the
    /// covered sub-window (output pixels whose centres fall inside the source)
    /// so this holds. The integer read window handed to GDAL is the enclosing
    /// whole-pixel box clamped to the band, and the floating window is clamped
    /// into it, so a boundary output pixel whose extent pokes half a pixel past
    /// the source still reads (nearest samples its centre, which is inside).
    pub fn read_regrid_into(
        &self,
        src_window: (f64, f64, f64, f64),
        dst_off: (usize, usize),
        dst_size: (usize, usize),
        full_width: usize,
        dst: &mut [u8],
        e_resample_alg: ResampleAlg,
    ) -> Result<()> {
        let (src_w, src_h) = self.size();
        let byte_size = self.band_type().byte_size();
        if byte_size == 0 {
            return Err(GdalError::BadArgument(
                "Cannot regrid a band with unknown GDAL data type".to_string(),
            ));
        }
        if src_w == 0 || src_h == 0 {
            return Ok(());
        }

        let (dst_x, dst_y) = dst_off;
        let (buf_w, buf_h) = dst_size;
        if buf_w == 0 || buf_h == 0 {
            return Ok(());
        }
        // The written sub-rectangle must fit inside the destination buffer.
        if dst_x + buf_w > full_width {
            return Err(GdalError::BadArgument(format!(
                "regrid destination window x[{dst_x}..{}] exceeds buffer width {full_width}",
                dst_x + buf_w
            )));
        }
        let line_stride = full_width * byte_size;
        let required = (dst_y + buf_h) * line_stride;
        if dst.len() < required {
            return Err(GdalError::BadArgument(format!(
                "regrid destination buffer of {} bytes is too small (need {required})",
                dst.len()
            )));
        }

        // Integer read window: the whole-pixel box enclosing the floating window,
        // clamped to the band extent (GDAL requires an in-bounds integer window).
        // `ix0`/`iy0` land on a valid pixel (<= size-1) so `ix0 + 1 <= src_w`, and
        // the clamp bounds stay ordered.
        let (fx, fy, fw, fh) = src_window;
        let ix0 = (fx.floor().max(0.0) as usize).min(src_w - 1);
        let iy0 = (fy.floor().max(0.0) as usize).min(src_h - 1);
        let ix1 = ((fx + fw).ceil().max(0.0) as usize).clamp(ix0 + 1, src_w);
        let iy1 = ((fy + fh).ceil().max(0.0) as usize).clamp(iy0 + 1, src_h);
        let nx = ix1 - ix0;
        let ny = iy1 - iy0;

        // Clamp the floating window into the integer window so it never claims
        // pixels the read region does not cover.
        let win_x0 = fx.max(ix0 as f64);
        let win_y0 = fy.max(iy0 as f64);
        let win_x1 = (fx + fw).min(ix1 as f64);
        let win_y1 = (fy + fh).min(iy1 as f64);

        let mut extra_arg = GDALRasterIOExtraArg {
            eResampleAlg: e_resample_alg.to_gdal(),
            bFloatingPointWindowValidity: 1,
            dfXOff: win_x0,
            dfYOff: win_y0,
            dfXSize: (win_x1 - win_x0).max(0.0),
            dfYSize: (win_y1 - win_y0).max(0.0),
            ..GDALRasterIOExtraArg::default()
        };

        // Point at the first byte of the destination sub-rectangle; pixel/line
        // spacing place the resampled pixels into the wider buffer.
        let offset = dst_y * line_stride + dst_x * byte_size;
        let data_ptr = unsafe { dst.as_mut_ptr().add(offset) } as *mut std::ffi::c_void;

        let rv = unsafe {
            call_gdal_api!(
                self.api,
                GDALRasterIOEx,
                self.c_rasterband,
                GF_Read,
                i32::try_from(ix0)?,
                i32::try_from(iy0)?,
                i32::try_from(nx)?,
                i32::try_from(ny)?,
                data_ptr,
                i32::try_from(buf_w)?,
                i32::try_from(buf_h)?,
                self.c_band_type(),
                i64::try_from(byte_size)?,
                i64::try_from(line_stride)?,
                &mut extra_arg
            )
        };
        if rv != CE_None {
            return Err(self.api.last_cpl_err(rv as u32));
        }
        Ok(())
    }

    fn read_impl(
        &self,
        request: RasterIoRequest,
        data_ptr: *mut std::ffi::c_void,
        buf_type: GDALDataType,
    ) -> Result<()> {
        let resample_alg = request
            .e_resample_alg
            .unwrap_or(ResampleAlg::NearestNeighbour);
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
                i32::try_from(request.window.0)?,
                i32::try_from(request.window.1)?,
                i32::try_from(request.window_size.0)?,
                i32::try_from(request.window_size.1)?,
                data_ptr,
                i32::try_from(request.size.0)?,
                i32::try_from(request.size.1)?,
                buf_type,
                0,
                0,
                &mut extra_arg
            )
        };
        if rv != CE_None {
            return Err(self.api.last_cpl_err(rv as u32));
        }

        Ok(())
    }

    fn expected_byte_len(&self, size: (usize, usize)) -> Result<usize> {
        let bytes_per_value = self.band_type().byte_size();
        if bytes_per_value == 0 {
            return Err(GdalError::BadArgument(
                "Cannot read bytes for band with unknown GDAL data type".to_string(),
            ));
        }

        Ok(size.0 * size.1 * bytes_per_value)
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
    use crate::raster::types::{Buffer, ResampleAlg};

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
    fn read_regrid_into_places_fractional_window() {
        with_global_gdal_api(|api| {
            let driver = DriverManager::get_driver_by_name(api, "MEM").unwrap();
            // 2x1 source, values [10, 20] (source col 0 = 10, col 1 = 20).
            let dataset = driver.create_with_band_type::<u8>("", 2, 1, 1).unwrap();
            let band = dataset.rasterband(1).unwrap();
            let mut src = Buffer::new((2, 1), vec![10u8, 20]);
            band.write((0, 0), (2, 1), &mut src).unwrap();

            // A 2x1 destination pre-filled with nodata (255). Regrid the
            // fractional source window [0, 1.5] (covering source col 0 only) into
            // destination col 0: nearest samples the output pixel centre at
            // source 0.75 -> col 0 = 10; destination col 1 is never written, so
            // it keeps the nodata fill.
            let mut dst = vec![255u8, 255];
            band.read_regrid_into(
                (0.0, 0.0, 1.5, 1.0),
                (0, 0),
                (1, 1),
                2,
                &mut dst,
                ResampleAlg::NearestNeighbour,
            )
            .unwrap();
            assert_eq!(dst, vec![10u8, 255]);

            // The complementary window [0.5, 2.0] (covering source col 1) placed
            // at destination col 1 samples source 1.25 -> col 1 = 20.
            let mut dst = vec![255u8, 255];
            band.read_regrid_into(
                (0.5, 0.0, 1.5, 1.0),
                (1, 0),
                (1, 1),
                2,
                &mut dst,
                ResampleAlg::NearestNeighbour,
            )
            .unwrap();
            assert_eq!(dst, vec![255u8, 20]);
        })
        .unwrap();
    }

    #[test]
    fn test_read_raster_as_bytes_u8() {
        with_global_gdal_api(|api| {
            let driver = DriverManager::get_driver_by_name(api, "MEM").unwrap();
            let dataset = driver.create_with_band_type::<u8>("", 2, 3, 1).unwrap();
            let rasterband = dataset.rasterband(1).unwrap();
            let mut buffer = Buffer::new((2, 3), vec![7u8, 7, 7, 10, 8, 12]);
            rasterband.write((0, 0), (2, 3), &mut buffer).unwrap();

            let rv = rasterband
                .read_as_bytes((0, 0), (2, 3), (2, 3), None)
                .unwrap();
            assert_eq!(rv, vec![7u8, 7, 7, 10, 8, 12]);
        })
        .unwrap();
    }

    #[test]
    fn test_read_raster_as_bytes_u16() {
        with_global_gdal_api(|api| {
            let driver = DriverManager::get_driver_by_name(api, "MEM").unwrap();
            let dataset = driver.create_with_band_type::<u16>("", 2, 2, 1).unwrap();
            let rasterband = dataset.rasterband(1).unwrap();
            let mut buffer = Buffer::new((2, 2), vec![1u16, 256, 511, 1024]);
            rasterband.write((0, 0), (2, 2), &mut buffer).unwrap();

            let rv = rasterband
                .read_as_bytes((0, 0), (2, 2), (2, 2), None)
                .unwrap();
            let expected: Vec<u8> = vec![1u16, 256, 511, 1024]
                .into_iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect();
            assert_eq!(rv, expected);
        })
        .unwrap();
    }

    #[test]
    fn test_read_into_bytes_matches_read_as_bytes() {
        with_global_gdal_api(|api| {
            let driver = DriverManager::get_driver_by_name(api, "MEM").unwrap();
            let dataset = driver.create_with_band_type::<u16>("", 2, 2, 1).unwrap();
            let rasterband = dataset.rasterband(1).unwrap();
            let mut buffer = Buffer::new((2, 2), vec![5u16, 6, 7, 8]);
            rasterband.write((0, 0), (2, 2), &mut buffer).unwrap();

            let expected = rasterband
                .read_as_bytes((0, 0), (2, 2), (2, 2), None)
                .unwrap();
            let mut actual = vec![0u8; expected.len()];
            rasterband
                .read_into_bytes((0, 0), (2, 2), (2, 2), &mut actual, None)
                .unwrap();

            assert_eq!(actual, expected);
        })
        .unwrap();
    }

    #[test]
    fn test_read_into_bytes_with_wrong_len() {
        with_global_gdal_api(|api| {
            let driver = DriverManager::get_driver_by_name(api, "MEM").unwrap();
            let dataset = driver.create_with_band_type::<u16>("", 2, 2, 1).unwrap();
            let rasterband = dataset.rasterband(1).unwrap();
            let mut data = vec![0u8; 7];

            let err = rasterband
                .read_into_bytes((0, 0), (2, 2), (2, 2), &mut data, None)
                .unwrap_err();
            assert!(matches!(err, crate::errors::GdalError::BadArgument(_)));
        })
        .unwrap();
    }

    #[test]
    fn test_read_as_bytes_with_average_resample() {
        with_global_gdal_api(|api| {
            let driver = DriverManager::get_driver_by_name(api, "MEM").unwrap();
            let dataset = driver.create_with_band_type::<u8>("", 4, 4, 1).unwrap();
            let rasterband = dataset.rasterband(1).unwrap();
            let mut buffer = Buffer::new(
                (4, 4),
                vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            );
            rasterband.write((0, 0), (4, 4), &mut buffer).unwrap();

            let rv = rasterband
                .read_as_bytes((0, 0), (4, 4), (2, 2), Some(ResampleAlg::Average))
                .unwrap();
            assert_eq!(rv.len(), 4);
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

    #[test]
    fn test_set_no_data_value_u64() {
        with_global_gdal_api(|api| {
            let driver = DriverManager::get_driver_by_name(api, "MEM").unwrap();
            let dataset = driver.create_with_band_type::<u64>("", 20, 10, 1).unwrap();
            let rasterband = dataset.rasterband(1).unwrap();
            let nodata = 9_007_199_254_740_993u64;

            assert_eq!(rasterband.no_data_value_u64(), None);
            assert!(rasterband.set_no_data_value_u64(Some(nodata)).is_ok());
            assert_eq!(rasterband.no_data_value_u64(), Some(nodata));
            assert!(rasterband.set_no_data_value_u64(None).is_ok());
            assert_eq!(rasterband.no_data_value_u64(), None);
        })
        .unwrap();
    }

    #[test]
    fn test_set_no_data_value_i64() {
        with_global_gdal_api(|api| {
            let driver = DriverManager::get_driver_by_name(api, "MEM").unwrap();
            let dataset = driver.create_with_band_type::<i64>("", 20, 10, 1).unwrap();
            let rasterband = dataset.rasterband(1).unwrap();
            let nodata = -9_007_199_254_740_993i64;

            assert_eq!(rasterband.no_data_value_i64(), None);
            assert!(rasterband.set_no_data_value_i64(Some(nodata)).is_ok());
            assert_eq!(rasterband.no_data_value_i64(), Some(nodata));
            assert!(rasterband.set_no_data_value_i64(None).is_ok());
            assert_eq!(rasterband.no_data_value_i64(), None);
        })
        .unwrap();
    }
}
