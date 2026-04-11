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

//! High-level convenience wrapper around [`GdalApi`].
//!
//! [`Gdal`] bundles a `&'static GdalApi` reference and exposes ergonomic
//! methods that delegate to the lower-level constructors and free functions
//! scattered across the crate, eliminating the need to pass `api` explicitly
//! at every call site.

use crate::config;
use crate::dataset::Dataset;
use crate::driver::{Driver, DriverManager};
use crate::errors::Result;
use crate::gdal_api::GdalApi;
use crate::gdal_dyn_bindgen::{GDALOpenFlags, OGRFieldType};
use crate::mem::create_mem_dataset;
use crate::raster::polygonize::{polygonize, PolygonizeOptions};
use crate::raster::rasterband::RasterBand;
use crate::raster::rasterize::{rasterize, RasterizeOptions};
use crate::raster::rasterize_affine::rasterize_affine;
use crate::raster::types::DatasetOptions;
use crate::raster::types::GdalDataType;
use crate::spatial_ref::SpatialRef;
use crate::vector::feature::FieldDefn;
use crate::vector::geometry::Geometry;
use crate::vector::layer::Layer;
use crate::vrt::VrtDataset;
use crate::vsi;

/// High-level convenience wrapper around [`GdalApi`].
///
/// Stores a `&'static GdalApi` reference and provides ergonomic methods that
/// delegate to the various constructors and free functions in the crate.
pub struct Gdal {
    api: &'static GdalApi,
}

impl Gdal {
    /// Create a `Gdal` wrapper for the given API reference.
    pub(crate) fn new(api: &'static GdalApi) -> Self {
        Self { api }
    }

    // -- Info ----------------------------------------------------------------

    /// Return the name of the loaded GDAL library.
    pub fn name(&self) -> &str {
        self.api.name()
    }

    /// Fetch GDAL version information for a standard request key.
    /// See also [`GdalApi::version_info`].
    pub fn version_info(&self, request: &str) -> String {
        self.api.version_info(request)
    }

    // -- Config --------------------------------------------------------------

    /// Set a thread-local GDAL configuration option.
    /// See also [`config::set_thread_local_config_option`].
    pub fn set_thread_local_config_option(&self, key: &str, value: &str) -> Result<()> {
        config::set_thread_local_config_option(self.api, key, value)
    }

    // -- Driver --------------------------------------------------------------

    /// Fetch a GDAL driver by its short name.
    /// See also [`DriverManager::get_driver_by_name`].
    pub fn get_driver_by_name(&self, name: &str) -> Result<Driver> {
        DriverManager::get_driver_by_name(self.api, name)
    }

    // -- Dataset -------------------------------------------------------------

    /// Open a dataset with extended options.
    /// See also [`Dataset::open_ex`].
    pub fn open_ex(
        &self,
        path: &str,
        open_flags: GDALOpenFlags,
        allowed_drivers: Option<&[&str]>,
        open_options: Option<&[&str]>,
        sibling_files: Option<&[&str]>,
    ) -> Result<Dataset> {
        Dataset::open_ex(
            self.api,
            path,
            open_flags,
            allowed_drivers,
            open_options,
            sibling_files,
        )
    }

    /// Open a dataset using a [`DatasetOptions`] struct.
    /// See also [`Dataset::open_ex_with_options`].
    pub fn open_ex_with_options(&self, path: &str, options: DatasetOptions<'_>) -> Result<Dataset> {
        Dataset::open_ex_with_options(self.api, path, options)
    }

    // -- Spatial Reference ---------------------------------------------------

    /// Create a spatial reference from a WKT string.
    /// See also [`SpatialRef::from_wkt`].
    pub fn spatial_ref_from_wkt(&self, wkt: &str) -> Result<SpatialRef> {
        SpatialRef::from_wkt(self.api, wkt)
    }

    // -- VRT -----------------------------------------------------------------

    /// Create an empty VRT dataset with the given raster size.
    /// See also [`VrtDataset::create`].
    pub fn create_vrt(&self, x_size: usize, y_size: usize) -> Result<VrtDataset> {
        VrtDataset::create(self.api, x_size, y_size)
    }

    // -- Geometry ------------------------------------------------------------

    /// Create a geometry from WKB bytes.
    /// See also [`Geometry::from_wkb`].
    pub fn geometry_from_wkb(&self, wkb: &[u8]) -> Result<Geometry> {
        Geometry::from_wkb(self.api, wkb)
    }

    /// Create a geometry from a WKT string.
    /// See also [`Geometry::from_wkt`].
    pub fn geometry_from_wkt(&self, wkt: &str) -> Result<Geometry> {
        Geometry::from_wkt(self.api, wkt)
    }

    // -- Vector --------------------------------------------------------------

    /// Create an OGR field definition.
    /// See also [`FieldDefn::new`].
    pub fn create_field_defn(&self, name: &str, field_type: OGRFieldType) -> Result<FieldDefn> {
        FieldDefn::new(self.api, name, field_type)
    }

    // -- VSI (Virtual File System) -------------------------------------------

    /// Create a VSI in-memory file from the given bytes.
    /// See also [`vsi::create_mem_file`].
    pub fn create_mem_file(&self, file_name: &str, data: &[u8]) -> Result<()> {
        vsi::create_mem_file(self.api, file_name, data)
    }

    /// Delete a VSI in-memory file.
    /// See also [`vsi::unlink_mem_file`].
    pub fn unlink_mem_file(&self, file_name: &str) -> Result<()> {
        vsi::unlink_mem_file(self.api, file_name)
    }

    /// Copy the bytes of a VSI in-memory file, taking ownership of the GDAL buffer.
    /// See also [`vsi::get_vsi_mem_file_bytes_owned`].
    pub fn get_vsi_mem_file_bytes_owned(&self, file_name: &str) -> Result<Vec<u8>> {
        vsi::get_vsi_mem_file_bytes_owned(self.api, file_name)
    }

    // -- Raster operations ---------------------------------------------------

    /// Create a bare in-memory MEM dataset with GDAL-owned bands.
    /// See also [`crate::mem::MemDatasetBuilder`] for higher-level MEM dataset construction.
    pub fn create_mem_dataset(
        &self,
        width: usize,
        height: usize,
        n_owned_bands: usize,
        owned_bands_data_type: GdalDataType,
    ) -> Result<Dataset> {
        create_mem_dataset(
            self.api,
            width,
            height,
            n_owned_bands,
            owned_bands_data_type,
        )
    }

    /// Rasterize geometries using the dataset geotransform as the transformer.
    /// See also [`rasterize_affine`].
    pub fn rasterize_affine(
        &self,
        dataset: &Dataset,
        bands: &[usize],
        geometries: &[Geometry],
        burn_values: &[f64],
        all_touched: bool,
    ) -> Result<()> {
        rasterize_affine(
            self.api,
            dataset,
            bands,
            geometries,
            burn_values,
            all_touched,
        )
    }

    /// Rasterize geometries into the selected dataset bands.
    /// See also [`rasterize`].
    pub fn rasterize(
        &self,
        dataset: &Dataset,
        band_list: &[i32],
        geometries: &[&Geometry],
        burn_values: &[f64],
        options: Option<RasterizeOptions>,
    ) -> Result<()> {
        rasterize(
            self.api,
            dataset,
            band_list,
            geometries,
            burn_values,
            options,
        )
    }

    /// Polygonize a raster band into a vector layer.
    /// See also [`polygonize`].
    pub fn polygonize(
        &self,
        src_band: &RasterBand<'_>,
        mask_band: Option<&RasterBand<'_>>,
        out_layer: &Layer<'_>,
        pixel_value_field: i32,
        options: &PolygonizeOptions,
    ) -> Result<()> {
        polygonize(
            self.api,
            src_band,
            mask_band,
            out_layer,
            pixel_value_field,
            options,
        )
    }
}
