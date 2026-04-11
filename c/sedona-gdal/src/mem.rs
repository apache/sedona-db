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

use crate::dataset::Dataset;
use crate::errors::Result;
use crate::gdal::Gdal;
use crate::gdal_api::{call_gdal_api, GdalApi};
use crate::gdal_dyn_bindgen::CE_Failure;
use crate::raster::types::GdalDataType;

/// Nodata value for a raster band.
///
/// GDAL has three separate APIs for setting nodata depending on the band data type:
/// - [`f64`] for most types (UInt8 through Float64, excluding Int64/UInt64)
/// - [`i64`] for Int64 bands
/// - [`u64`] for UInt64 bands
///
/// This enum encapsulates the three nodata value representations exposed by GDAL.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Nodata {
    F64(f64),
    I64(i64),
    U64(u64),
}

/// A band specification for [`MemDatasetBuilder`].
struct MemBand {
    data_type: GdalDataType,
    data_ptr: *mut u8,
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
/// provided data pointers remain valid for the lifetime of the built [`Dataset`],
/// satisfy the alignment requirements of the band data type, and refer to writable
/// memory if GDAL may write through the attached `DATAPOINTER` band.
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
    /// Create a builder for a MEM dataset with the given dimensions.
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

    /// Create a builder for a MEM dataset with GDAL-owned bands.
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

    /// Create a MEM dataset with GDAL-owned bands.
    /// This is a safe shortcut for `new_with_owned_bands(...).build(gdal)`.
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
    /// Use default contiguous row-major pixel and line offsets.
    ///
    /// # Safety
    ///
    /// The caller must ensure `data_ptr` points to a valid buffer of at least
    /// `height * width * data_type.byte_size()` bytes, is properly aligned for
    /// `data_type`, and outlives the built [`Dataset`].
    pub unsafe fn add_band(self, data_type: GdalDataType, data_ptr: *mut u8) -> Self {
        self.add_band_with_options(data_type, data_ptr, None, None, None)
    }

    /// Add a zero-copy band with custom offsets and optional nodata.
    ///
    /// # Safety
    ///
    /// The caller must ensure `data_ptr` points to a valid buffer of sufficient size
    /// for the given dimensions and offsets, is properly aligned for `data_type`, and
    /// outlives the built [`Dataset`].
    pub unsafe fn add_band_with_options(
        mut self,
        data_type: GdalDataType,
        data_ptr: *mut u8,
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

    /// Set the dataset geotransform coefficients.
    pub fn geo_transform(mut self, gt: [f64; 6]) -> Self {
        self.geo_transform = Some(gt);
        self
    }

    /// Set the dataset projection definition string.
    pub fn projection(mut self, projection: impl Into<String>) -> Self {
        self.projection = Some(projection.into());
        self
    }

    /// Build the MEM dataset and attach the configured bands and metadata.
    ///
    /// # Safety
    ///
    /// This method is unsafe because the built dataset references memory provided via
    /// the `add_band*` methods. The caller must ensure all data pointers remain valid
    /// for the lifetime of the returned [`Dataset`] and satisfy the alignment
    /// requirements of their band data types.
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

/// Create a bare in-memory MEM dataset with GDAL-owned bands.
/// For a higher-level builder with external bands and metadata, use `MemDatasetBuilder`.
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
    Ok(Dataset::new(api, handle))
}

#[cfg(all(test, feature = "gdal-sys"))]
mod tests {
    use crate::global::with_global_gdal;
    use crate::mem::{MemDatasetBuilder, Nodata};
    use crate::raster::types::GdalDataType;

    #[test]
    fn test_mem_builder_single_band() {
        with_global_gdal(|gdal| {
            let mut data = vec![42u8; 64 * 64];
            let dataset = unsafe {
                MemDatasetBuilder::new(64, 64)
                    .add_band(GdalDataType::UInt8, data.as_mut_ptr())
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
            let mut band1 = vec![1u16; 32 * 32];
            let mut band2 = vec![2u16; 32 * 32];
            let mut band3 = vec![3u16; 32 * 32];
            let dataset = unsafe {
                MemDatasetBuilder::new(32, 32)
                    .add_band(GdalDataType::UInt16, band1.as_mut_ptr() as *mut u8)
                    .add_band(GdalDataType::UInt16, band2.as_mut_ptr() as *mut u8)
                    .add_band(GdalDataType::UInt16, band3.as_mut_ptr() as *mut u8)
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
            let mut data = vec![0f32; 10 * 10];
            let gt = [100.0, 0.5, 0.0, 200.0, 0.0, -0.5];
            let dataset = unsafe {
                MemDatasetBuilder::new(10, 10)
                    .add_band(GdalDataType::Float32, data.as_mut_ptr() as *mut u8)
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
    fn test_mem_builder_with_wkt_projection() {
        let projections = [
            r#"GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]"#,
            r#"{"$schema":"https://proj.org/schemas/v0.7/projjson.schema.json","type":"GeographicCRS","name":"WGS 84 (CRS84)","datum_ensemble":{"name":"World Geodetic System 1984 ensemble","members":[{"name":"World Geodetic System 1984 (Transit)","id":{"authority":"EPSG","code":1166}},{"name":"World Geodetic System 1984 (G730)","id":{"authority":"EPSG","code":1152}},{"name":"World Geodetic System 1984 (G873)","id":{"authority":"EPSG","code":1153}},{"name":"World Geodetic System 1984 (G1150)","id":{"authority":"EPSG","code":1154}},{"name":"World Geodetic System 1984 (G1674)","id":{"authority":"EPSG","code":1155}},{"name":"World Geodetic System 1984 (G1762)","id":{"authority":"EPSG","code":1156}},{"name":"World Geodetic System 1984 (G2139)","id":{"authority":"EPSG","code":1309}},{"name":"World Geodetic System 1984 (G2296)","id":{"authority":"EPSG","code":1383}}],"ellipsoid":{"name":"WGS 84","semi_major_axis":6378137,"inverse_flattening":298.257223563},"accuracy":"2.0","id":{"authority":"EPSG","code":6326}},"coordinate_system":{"subtype":"ellipsoidal","axis":[{"name":"Geodetic longitude","abbreviation":"Lon","direction":"east","unit":"degree"},{"name":"Geodetic latitude","abbreviation":"Lat","direction":"north","unit":"degree"}]},"scope":"Not known.","area":"World.","bbox":{"south_latitude":-90,"west_longitude":-180,"north_latitude":90,"east_longitude":180},"id":{"authority":"OGC","code":"CRS84"}}"#,
            "EPSG:4326",
        ];
        for projection in projections {
            with_global_gdal(|gdal| {
                let mut data = [0u8; 8 * 8];
                let dataset = unsafe {
                    MemDatasetBuilder::new(8, 8)
                        .add_band(GdalDataType::UInt8, data.as_mut_ptr())
                        .projection(projection)
                        .build(gdal)
                        .unwrap()
                };
                let proj = dataset.projection();
                assert!(proj.contains("WGS 84"), "Expected WGS 84 in: {proj}");
            })
            .unwrap();
        }
    }

    #[test]
    fn test_mem_builder_with_epsg_code_projection() {
        with_global_gdal(|gdal| {
            let mut data = [0u8; 8 * 8];
            let dataset = unsafe {
                MemDatasetBuilder::new(8, 8)
                    .add_band(GdalDataType::UInt8, data.as_mut_ptr())
                    .projection("EPSG:4326")
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
            let mut data = [0f64; 4 * 4];
            let dataset = unsafe {
                MemDatasetBuilder::new(4, 4)
                    .add_band_with_options(
                        GdalDataType::Float64,
                        data.as_mut_ptr() as *mut u8,
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
            let mut band_u8 = [0u8; 8 * 8];
            let mut band_f64 = vec![0f64; 8 * 8];
            let dataset = unsafe {
                MemDatasetBuilder::new(8, 8)
                    .add_band(GdalDataType::UInt8, band_u8.as_mut_ptr())
                    .add_band(GdalDataType::Float64, band_f64.as_mut_ptr() as *mut u8)
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
            let mut external_band = [0u8; 8 * 8];
            let dataset = unsafe {
                MemDatasetBuilder::new_with_owned_bands(8, 8, 1, GdalDataType::Float32)
                    .add_band_with_options(
                        GdalDataType::UInt8,
                        external_band.as_mut_ptr(),
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
