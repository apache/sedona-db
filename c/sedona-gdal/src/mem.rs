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

//! High-level builder for creating in-memory (MEM) GDAL datasets.
//!
//! [`MemDatasetBuilder`] provides a fluent, type-safe API for constructing GDAL MEM
//! datasets with zero-copy band attachment, optional geo-transform, projection, and
//! per-band nodata values.
//!
//! # Example
//!
//! ```rust,ignore
//! use sedona_gdal::global::with_global_gdal;
//! use sedona_gdal::mem::{MemDatasetBuilder, Nodata};
//! use sedona_gdal::GdalDataType;
//!
//! with_global_gdal(|gdal| {
//!     let data: Vec<u8> = vec![0u8; 256 * 256];
//!     let dataset = unsafe {
//!         MemDatasetBuilder::new(256, 256)
//!             .add_band(GdalDataType::UInt8, data.as_ptr())
//!             .geo_transform([0.0, 1.0, 0.0, 0.0, 0.0, -1.0])
//!             .projection("EPSG:4326")
//!             .build(gdal)
//!             .unwrap()
//!     };
//!     assert_eq!(dataset.raster_count(), 1);
//! }).unwrap();
//! ```

use crate::call_gdal_api;
use crate::dataset::Dataset;
use crate::errors::Result;
use crate::gdal::Gdal;
use crate::gdal_api::GdalApi;
use crate::gdal_dyn_bindgen::CE_Failure;
use crate::raster::types::GdalDataType;

/// Nodata value for a raster band.
///
/// GDAL has three separate APIs for setting nodata depending on the band data type:
/// - [`f64`] for most types (UInt8 through Float64, excluding Int64/UInt64)
/// - [`i64`] for Int64 bands
/// - [`u64`] for UInt64 bands
///
/// This enum encapsulates all three variants so callers don't need to match on
/// the band type when setting nodata.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Nodata {
    F64(f64),
    I64(i64),
    U64(u64),
}

/// A band specification for [`MemDatasetBuilder`].
struct MemBand {
    data_type: GdalDataType,
    data_ptr: *const u8,
    pixel_offset: Option<i64>,
    line_offset: Option<i64>,
    nodata: Option<Nodata>,
}

/// A builder for constructing in-memory (MEM) GDAL datasets.
///
/// This creates datasets using `MEMDataset::Create` (bypassing GDAL's open-dataset-list
/// mutex for better concurrency) and attaches bands via `GDALAddBand` with `DATAPOINTER`
/// options for zero-copy operation.
///
/// # Safety
///
/// All `add_band*` methods are `unsafe` because the caller must ensure that the
/// provided data pointers remain valid for the lifetime of the built [`Dataset`].
pub struct MemDatasetBuilder {
    width: usize,
    height: usize,
    n_owned_bands: usize,
    owned_bands_data_type: Option<GdalDataType>,
    bands: Vec<MemBand>,
    geo_transform: Option<[f64; 6]>,
    projection: Option<String>,
}

impl MemDatasetBuilder {
    /// Create a new builder for a MEM dataset with the given dimensions.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            n_owned_bands: 0,
            owned_bands_data_type: None,
            bands: Vec::new(),
            geo_transform: None,
            projection: None,
        }
    }

    /// Create a new builder for a MEM dataset with the given dimensions and number of owned bands.
    pub fn new_with_owned_bands(
        width: usize,
        height: usize,
        n_owned_bands: usize,
        owned_bands_data_type: GdalDataType,
    ) -> Self {
        Self {
            width,
            height,
            n_owned_bands,
            owned_bands_data_type: Some(owned_bands_data_type),
            bands: Vec::new(),
            geo_transform: None,
            projection: None,
        }
    }

    /// Create a MEM dataset with owned bands.
    ///
    /// This is a convenience shortcut equivalent to
    /// `MemDatasetBuilder::new_with_owned_bands(...).build(gdal)`.
    ///
    /// Unlike [`build`](Self::build), this method is safe because datasets created
    /// with only owned bands do not reference any external memory.
    pub fn create(
        gdal: &Gdal,
        width: usize,
        height: usize,
        n_owned_bands: usize,
        owned_bands_data_type: GdalDataType,
    ) -> Result<Dataset> {
        // SAFETY: `new_with_owned_bands` creates a builder with zero external bands,
        // so no data pointers need to outlive the dataset.
        unsafe {
            Self::new_with_owned_bands(width, height, n_owned_bands, owned_bands_data_type)
                .build(gdal)
        }
    }

    /// Add a zero-copy band from a raw data pointer.
    ///
    /// Uses default pixel and line offsets (contiguous, row-major layout).
    ///
    /// # Safety
    ///
    /// The caller must ensure `data_ptr` points to a valid buffer of at least
    /// `height * width * data_type.byte_size()` bytes, and that the buffer
    /// outlives the built [`Dataset`].
    pub unsafe fn add_band(self, data_type: GdalDataType, data_ptr: *const u8) -> Self {
        self.add_band_with_options(data_type, data_ptr, None, None, None)
    }

    /// Add a zero-copy band with custom offsets and optional nodata.
    ///
    /// # Arguments
    /// * `data_type` - The GDAL data type of the band.
    /// * `data_ptr` - Pointer to the band pixel data.
    /// * `pixel_offset` - Byte offset between consecutive pixels. `None` defaults to
    ///   the byte size of `data_type`.
    /// * `line_offset` - Byte offset between consecutive lines. `None` defaults to
    ///   `pixel_offset * width`.
    /// * `nodata` - Optional nodata value for the band.
    ///
    /// # Safety
    ///
    /// The caller must ensure `data_ptr` points to a valid buffer of sufficient size
    /// for the given dimensions and offsets, and that the buffer outlives the built
    /// [`Dataset`].
    pub unsafe fn add_band_with_options(
        mut self,
        data_type: GdalDataType,
        data_ptr: *const u8,
        pixel_offset: Option<i64>,
        line_offset: Option<i64>,
        nodata: Option<Nodata>,
    ) -> Self {
        self.bands.push(MemBand {
            data_type,
            data_ptr,
            pixel_offset,
            line_offset,
            nodata,
        });
        self
    }

    /// Set the geo-transform for the dataset.
    ///
    /// The array is `[origin_x, pixel_width, rotation_x, origin_y, rotation_y, pixel_height]`.
    pub fn geo_transform(mut self, gt: [f64; 6]) -> Self {
        self.geo_transform = Some(gt);
        self
    }

    /// Set the projection (CRS) for the dataset as a WKT or PROJ string.
    pub fn projection(mut self, wkt: impl Into<String>) -> Self {
        self.projection = Some(wkt.into());
        self
    }

    /// Build the GDAL MEM dataset.
    ///
    /// Creates an empty MEM dataset using [`create_mem_dataset`], then attaches
    /// bands, sets the geo-transform, projection, and per-band nodata values.
    ///
    /// # Safety
    ///
    /// This method is unsafe because the built dataset references memory provided via
    /// the `add_band*` methods. The caller must ensure all data pointers remain valid
    /// for the lifetime of the returned [`Dataset`].
    pub unsafe fn build(self, gdal: &Gdal) -> Result<Dataset> {
        let dataset = gdal.create_mem_dataset(
            self.width,
            self.height,
            self.n_owned_bands,
            self.owned_bands_data_type.unwrap_or(GdalDataType::UInt8),
        )?;

        // Attach bands (zero-copy via DATAPOINTER).
        for band_spec in &self.bands {
            dataset.add_band_with_data(
                band_spec.data_type,
                band_spec.data_ptr,
                band_spec.pixel_offset,
                band_spec.line_offset,
            )?;
        }

        // Set geo-transform.
        if let Some(gt) = &self.geo_transform {
            dataset.set_geo_transform(gt)?;
        }

        // Set projection/CRS.
        if let Some(proj) = &self.projection {
            dataset.set_projection(proj)?;
        }

        // Set per-band nodata values.
        for (i, band_spec) in self.bands.iter().enumerate() {
            if let Some(nodata) = &band_spec.nodata {
                let raster_band = dataset.rasterband(i + 1 + self.n_owned_bands)?;
                match nodata {
                    Nodata::F64(v) => raster_band.set_no_data_value(Some(*v))?,
                    Nodata::I64(v) => raster_band.set_no_data_value_i64(Some(*v))?,
                    Nodata::U64(v) => raster_band.set_no_data_value_u64(Some(*v))?,
                }
            }
        }

        Ok(dataset)
    }
}

/// Create a bare in-memory (MEM) GDAL dataset via `MEMDataset::Create`.
///
/// This bypasses GDAL's open-dataset-list mutex for better concurrency.
/// The returned dataset has `n_owned_bands` bands of type
/// `owned_bands_data_type` whose pixel data is owned by GDAL.
///
/// For a higher-level builder that also attaches zero-copy external bands,
/// geo-transforms, projections, and nodata values, see [`MemDatasetBuilder`].
pub(crate) fn create_mem_dataset(
    api: &'static GdalApi,
    width: usize,
    height: usize,
    n_owned_bands: usize,
    owned_bands_data_type: GdalDataType,
) -> Result<Dataset> {
    let empty_filename = c"";
    let c_data_type = owned_bands_data_type.to_c();
    let handle = unsafe {
        call_gdal_api!(
            api,
            MEMDatasetCreate,
            empty_filename.as_ptr(),
            width.try_into()?,
            height.try_into()?,
            n_owned_bands.try_into()?,
            c_data_type,
            std::ptr::null_mut()
        )
    };

    if handle.is_null() {
        return Err(api.last_cpl_err(CE_Failure as u32));
    }
    Ok(Dataset::new_owned(api, handle))
}

#[cfg(all(test, feature = "gdal-sys"))]
mod tests {
    use crate::global::with_global_gdal;
    use crate::mem::{MemDatasetBuilder, Nodata};
    use crate::raster::types::GdalDataType;

    #[test]
    fn test_mem_builder_single_band() {
        with_global_gdal(|gdal| {
            let data = vec![42u8; 64 * 64];
            let dataset = unsafe {
                MemDatasetBuilder::new(64, 64)
                    .add_band(GdalDataType::UInt8, data.as_ptr())
                    .build(gdal)
                    .unwrap()
            };
            assert_eq!(dataset.raster_size(), (64, 64));
            assert_eq!(dataset.raster_count(), 1);
        })
        .unwrap();
    }

    #[test]
    fn test_mem_builder_multi_band() {
        with_global_gdal(|gdal| {
            let band1 = vec![1u16; 32 * 32];
            let band2 = vec![2u16; 32 * 32];
            let band3 = vec![3u16; 32 * 32];
            let dataset = unsafe {
                MemDatasetBuilder::new(32, 32)
                    .add_band(GdalDataType::UInt16, band1.as_ptr() as *const u8)
                    .add_band(GdalDataType::UInt16, band2.as_ptr() as *const u8)
                    .add_band(GdalDataType::UInt16, band3.as_ptr() as *const u8)
                    .build(gdal)
                    .unwrap()
            };
            assert_eq!(dataset.raster_count(), 3);
        })
        .unwrap();
    }

    #[test]
    fn test_mem_builder_with_geo_transform() {
        with_global_gdal(|gdal| {
            let data = vec![0f32; 10 * 10];
            let gt = [100.0, 0.5, 0.0, 200.0, 0.0, -0.5];
            let dataset = unsafe {
                MemDatasetBuilder::new(10, 10)
                    .add_band(GdalDataType::Float32, data.as_ptr() as *const u8)
                    .geo_transform(gt)
                    .build(gdal)
                    .unwrap()
            };
            let got = dataset.geo_transform().unwrap();
            assert_eq!(gt, got);
        })
        .unwrap();
    }

    #[test]
    fn test_mem_builder_with_projection() {
        with_global_gdal(|gdal| {
            let data = [0u8; 8 * 8];
            let dataset = unsafe {
                MemDatasetBuilder::new(8, 8)
                    .add_band(GdalDataType::UInt8, data.as_ptr())
                    .projection(r#"GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]"#)
                    .build(gdal)
                    .unwrap()
            };
            let proj = dataset.projection();
            assert!(proj.contains("WGS 84"), "Expected WGS 84 in: {proj}");
        })
        .unwrap();
    }

    #[test]
    fn test_mem_builder_with_nodata() {
        with_global_gdal(|gdal| {
            let data = [0f64; 4 * 4];
            let dataset = unsafe {
                MemDatasetBuilder::new(4, 4)
                    .add_band_with_options(
                        GdalDataType::Float64,
                        data.as_ptr() as *const u8,
                        None,
                        None,
                        Some(Nodata::F64(-9999.0)),
                    )
                    .build(gdal)
                    .unwrap()
            };
            let band = dataset.rasterband(1).unwrap();
            let nodata = band.no_data_value();
            assert_eq!(nodata, Some(-9999.0));
        })
        .unwrap();
    }

    #[test]
    fn test_mem_builder_zero_bands() {
        with_global_gdal(|gdal| {
            let dataset = unsafe { MemDatasetBuilder::new(16, 16).build(gdal).unwrap() };
            assert_eq!(dataset.raster_count(), 0);
            assert_eq!(dataset.raster_size(), (16, 16));
        })
        .unwrap();
    }

    #[test]
    fn test_mem_builder_mixed_band_types() {
        with_global_gdal(|gdal| {
            let band_u8 = [0u8; 8 * 8];
            let band_f64 = vec![0f64; 8 * 8];
            let dataset = unsafe {
                MemDatasetBuilder::new(8, 8)
                    .add_band(GdalDataType::UInt8, band_u8.as_ptr())
                    .add_band(GdalDataType::Float64, band_f64.as_ptr() as *const u8)
                    .build(gdal)
                    .unwrap()
            };
            assert_eq!(dataset.raster_count(), 2);
        })
        .unwrap();
    }

    #[test]
    pub fn test_mem_builder_with_owned_bands() {
        with_global_gdal(|gdal| {
            let dataset = unsafe {
                MemDatasetBuilder::new_with_owned_bands(16, 16, 2, GdalDataType::UInt16)
                    .build(gdal)
                    .unwrap()
            };
            assert_eq!(dataset.raster_count(), 2);
            assert_eq!(
                dataset.rasterband(1).unwrap().band_type(),
                GdalDataType::UInt16
            );
            assert_eq!(
                dataset.rasterband(2).unwrap().band_type(),
                GdalDataType::UInt16
            );

            let dataset = MemDatasetBuilder::create(gdal, 10, 8, 1, GdalDataType::Float32).unwrap();
            assert_eq!(dataset.raster_count(), 1);
            assert_eq!(
                dataset.rasterband(1).unwrap().band_type(),
                GdalDataType::Float32
            );
        })
        .unwrap();
    }

    #[test]
    pub fn test_mem_builder_mixed_owned_and_external_bands() {
        with_global_gdal(|gdal| {
            let external_band = [0u8; 8 * 8];
            let dataset = unsafe {
                MemDatasetBuilder::new_with_owned_bands(8, 8, 1, GdalDataType::Float32)
                    .add_band_with_options(
                        GdalDataType::UInt8,
                        external_band.as_ptr(),
                        None,
                        None,
                        Some(Nodata::U64(255)),
                    )
                    .build(gdal)
                    .unwrap()
            };
            assert_eq!(dataset.raster_count(), 2);
            assert_eq!(
                dataset.rasterband(1).unwrap().band_type(),
                GdalDataType::Float32
            );
            assert_eq!(
                dataset.rasterband(2).unwrap().band_type(),
                GdalDataType::UInt8
            );
            let nodata = dataset.rasterband(2).unwrap().no_data_value();
            assert_eq!(nodata, Some(255.0));
        })
        .unwrap();
    }
}
