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

    /// Get the next feature (returns None when exhausted).
    pub fn next_feature(&self) -> Option<Feature<'_>> {
        let c_feature = unsafe { call_gdal_api!(self.api, OGR_L_GetNextFeature, self.c_layer) };
        if c_feature.is_null() {
            None
        } else {
            Some(Feature::new(self.api, c_feature))
        }
    }

    /// Create a field on this layer.
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

    /// Get the number of features in this layer.
    ///
    /// If `force` is true, the count will be computed even if it is expensive.
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

    /// Iterate over all features.
    pub fn features(&self) -> FeatureIterator<'_> {
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
