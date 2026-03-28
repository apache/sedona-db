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
//! <https://github.com/georust/gdal/blob/v0.19.0/src/vector/layer.rs>.
//! Original code is licensed under MIT.

use std::marker::PhantomData;

use crate::dataset::Dataset;
use crate::errors::{GdalError, Result};
use crate::gdal_api::{call_gdal_api, GdalApi};
use crate::gdal_dyn_bindgen::*;
use crate::vector::feature::{Feature, FieldDefn};

/// An OGR layer (borrowed from a Dataset).
pub struct Layer<'a> {
    api: &'static GdalApi,
    c_layer: OGRLayerH,
    _dataset: PhantomData<&'a Dataset>,
}

impl<'a> Layer<'a> {
    pub(crate) fn new(api: &'static GdalApi, c_layer: OGRLayerH, _dataset: &'a Dataset) -> Self {
        Self {
            api,
            c_layer,
            _dataset: PhantomData,
        }
    }

    /// Return the raw C layer handle.
    pub fn c_layer(&self) -> OGRLayerH {
        self.c_layer
    }

    /// Reset reading to the first feature.
    pub fn reset_reading(&self) {
        unsafe { call_gdal_api!(self.api, OGR_L_ResetReading, self.c_layer) };
    }

    /// Fetch the next feature from the current read cursor.
    /// Return `None` when no more features are available.
    pub fn next_feature(&self) -> Option<Feature<'_>> {
        let c_feature = unsafe { call_gdal_api!(self.api, OGR_L_GetNextFeature, self.c_layer) };
        if c_feature.is_null() {
            None
        } else {
            Some(Feature::new(self.api, c_feature))
        }
    }

    /// Create a field in this layer.
    /// Allow the driver to approximate the definition if needed.
    pub fn create_field(&self, field_defn: &FieldDefn) -> Result<()> {
        let rv = unsafe {
            call_gdal_api!(
                self.api,
                OGR_L_CreateField,
                self.c_layer,
                field_defn.c_field_defn(),
                1 // bApproxOK
            )
        };
        if rv != OGRERR_NONE {
            return Err(GdalError::OgrError {
                err: rv,
                method_name: "OGR_L_CreateField",
            });
        }
        Ok(())
    }

    /// Fetch the feature count for this layer.
    /// Return `-1` if the count is unknown and `force` is `false`.
    pub fn feature_count(&self, force: bool) -> i64 {
        unsafe {
            call_gdal_api!(
                self.api,
                OGR_L_GetFeatureCount,
                self.c_layer,
                if force { 1 } else { 0 }
            )
        }
    }

    /// Iterate over features from the start of the layer.
    /// This resets the layer read cursor before iteration.
    pub fn features(&mut self) -> FeatureIterator<'_> {
        self.reset_reading();
        FeatureIterator { layer: self }
    }
}

/// Iterator over features in a layer.
pub struct FeatureIterator<'a> {
    layer: &'a Layer<'a>,
}

impl<'a> Iterator for FeatureIterator<'a> {
    type Item = Feature<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.layer.next_feature()
    }
}

#[cfg(all(test, feature = "gdal-sys"))]
mod tests {
    use gdal_sys::{
        GDALDatasetGetLayer, GDALDatasetGetLayerCount, OGR_F_Create, OGR_F_SetGeometry,
        OGR_L_CreateFeature, OGR_L_GetLayerDefn,
    };

    use super::Layer;
    use crate::dataset::{Dataset, LayerOptions};
    use crate::driver::DriverManager;
    use crate::gdal_dyn_bindgen::OGRwkbGeometryType;
    use crate::global::with_global_gdal_api;
    use crate::vector::geometry::Geometry;
    use crate::vsi::unlink_mem_file;

    #[test]
    fn test_layer_iteration_and_reset() {
        with_global_gdal_api(|api| {
            let path = "/vsimem/test_layer_iteration.gpkg";
            let driver = DriverManager::get_driver_by_name(api, "GPKG").unwrap();
            let dataset = driver.create_vector_only(path).unwrap();

            let layer = dataset
                .create_layer(LayerOptions {
                    name: "features",
                    srs: None,
                    ty: OGRwkbGeometryType::wkbPoint,
                    options: None,
                })
                .unwrap();

            let layer_defn = unsafe { OGR_L_GetLayerDefn(layer.c_layer()) };
            assert!(!layer_defn.is_null());

            for x in [1.0_f64, 2.0, 3.0] {
                let feature = unsafe { OGR_F_Create(layer_defn) };
                assert!(!feature.is_null());

                let geometry = Geometry::from_wkt(api, &format!("POINT ({x} 0)")).unwrap();
                let set_geometry_err = unsafe { OGR_F_SetGeometry(feature, geometry.c_geometry()) };
                assert_eq!(set_geometry_err, gdal_sys::OGRErr::OGRERR_NONE);

                let create_feature_err = unsafe { OGR_L_CreateFeature(layer.c_layer(), feature) };
                assert_eq!(create_feature_err, gdal_sys::OGRErr::OGRERR_NONE);

                unsafe { gdal_sys::OGR_F_Destroy(feature) };
            }

            let write_count = unsafe { GDALDatasetGetLayerCount(dataset.c_dataset()) };
            assert_eq!(write_count, 1);

            let read_dataset = Dataset::open_ex(
                api,
                path,
                crate::gdal_dyn_bindgen::GDAL_OF_VECTOR | crate::gdal_dyn_bindgen::GDAL_OF_READONLY,
                None,
                None,
                None,
            )
            .unwrap();

            let read_count = unsafe { GDALDatasetGetLayerCount(read_dataset.c_dataset()) };
            assert_eq!(read_count, 1);

            let c_layer = unsafe { GDALDatasetGetLayer(read_dataset.c_dataset(), 0) };
            assert!(!c_layer.is_null());
            let mut read_layer = Layer::new(api, c_layer, &read_dataset);

            assert_eq!(read_layer.feature_count(true), 3);

            let mut iter = read_layer.features();
            assert!(iter.next().is_some());
            assert!(iter.next().is_some());
            assert!(iter.next().is_some());
            assert!(iter.next().is_none());

            read_layer.reset_reading();
            assert!(read_layer.next_feature().is_some());

            assert_eq!(read_layer.features().count(), 3);
            assert_eq!(read_layer.features().count(), 3);

            unlink_mem_file(api, path).unwrap();
        })
        .unwrap();
    }
}
