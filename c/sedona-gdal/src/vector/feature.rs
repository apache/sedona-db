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
//! <https://github.com/georust/gdal/blob/v0.19.0/src/vector/feature.rs>.
//! Original code is licensed under MIT.

use std::ffi::CString;
use std::marker::PhantomData;

use crate::errors::{GdalError, Result};
use crate::gdal_api::{call_gdal_api, GdalApi};
use crate::gdal_dyn_bindgen::*;
use crate::vector::geometry::Envelope;

/// An OGR feature.
pub struct Feature<'a> {
    api: &'static GdalApi,
    c_feature: OGRFeatureH,
    _lifetime: PhantomData<&'a ()>,
}

impl Drop for Feature<'_> {
    fn drop(&mut self) {
        if !self.c_feature.is_null() {
            unsafe { call_gdal_api!(self.api, OGR_F_Destroy, self.c_feature) };
        }
    }
}

impl<'a> Feature<'a> {
    pub(crate) fn new(api: &'static GdalApi, c_feature: OGRFeatureH) -> Self {
        Self {
            api,
            c_feature,
            _lifetime: PhantomData,
        }
    }

    /// Fetch the feature geometry.
    /// The returned geometry is borrowed; return `None` if no geometry is set.
    pub fn geometry(&self) -> Option<BorrowedGeometry<'_>> {
        let c_geom = unsafe { call_gdal_api!(self.api, OGR_F_GetGeometryRef, self.c_feature) };
        if c_geom.is_null() {
            None
        } else {
            Some(BorrowedGeometry {
                api: self.api,
                c_geom,
                _lifetime: PhantomData,
            })
        }
    }

    /// Fetch the index of a field by name.
    /// Return an error if the field is not found.
    pub fn field_index(&self, name: &str) -> Result<i32> {
        let c_name = CString::new(name)?;
        let idx = unsafe {
            call_gdal_api!(
                self.api,
                OGR_F_GetFieldIndex,
                self.c_feature,
                c_name.as_ptr()
            )
        };
        if idx < 0 {
            return Err(GdalError::BadArgument(format!("field '{name}' not found")));
        }
        Ok(idx)
    }

    /// Fetch a field value as `f64`.
    pub fn field_as_double(&self, field_index: i32) -> f64 {
        unsafe {
            call_gdal_api!(
                self.api,
                OGR_F_GetFieldAsDouble,
                self.c_feature,
                field_index
            )
        }
    }

    /// Fetch a field value as `i32`.
    /// Return `None` if the field is unset or null.
    pub fn field_as_integer(&self, field_index: i32) -> Option<i32> {
        let is_set = unsafe {
            call_gdal_api!(
                self.api,
                OGR_F_IsFieldSetAndNotNull,
                self.c_feature,
                field_index
            )
        };
        if is_set != 0 {
            Some(unsafe {
                call_gdal_api!(
                    self.api,
                    OGR_F_GetFieldAsInteger,
                    self.c_feature,
                    field_index
                )
            })
        } else {
            None
        }
    }
}

/// A geometry borrowed from a feature (not owned — will NOT be destroyed).
pub struct BorrowedGeometry<'a> {
    api: &'static GdalApi,
    c_geom: OGRGeometryH,
    _lifetime: PhantomData<&'a ()>,
}

impl<'a> BorrowedGeometry<'a> {
    /// Return the raw C geometry handle.
    pub fn c_geometry(&self) -> OGRGeometryH {
        self.c_geom
    }

    /// Export to ISO WKB.
    pub fn wkb(&self) -> Result<Vec<u8>> {
        let size = unsafe { call_gdal_api!(self.api, OGR_G_WkbSize, self.c_geom) };
        if size < 0 {
            return Err(GdalError::BadArgument(format!(
                "OGR_G_WkbSize returned negative size: {size}"
            )));
        }
        let mut buf = vec![0u8; size as usize];
        let rv = unsafe {
            call_gdal_api!(
                self.api,
                OGR_G_ExportToIsoWkb,
                self.c_geom,
                wkbNDR,
                buf.as_mut_ptr()
            )
        };
        if rv != OGRERR_NONE {
            return Err(GdalError::OgrError {
                err: rv,
                method_name: "OGR_G_ExportToIsoWkb",
            });
        }
        Ok(buf)
    }

    /// Fetch the 2D envelope of this geometry.
    pub fn envelope(&self) -> Envelope {
        let mut env = OGREnvelope {
            MinX: 0.0,
            MaxX: 0.0,
            MinY: 0.0,
            MaxY: 0.0,
        };
        unsafe { call_gdal_api!(self.api, OGR_G_GetEnvelope, self.c_geom, &mut env) };
        Envelope {
            MinX: env.MinX,
            MaxX: env.MaxX,
            MinY: env.MinY,
            MaxY: env.MaxY,
        }
    }
}

/// An OGR field definition.
pub struct FieldDefn {
    api: &'static GdalApi,
    c_field_defn: OGRFieldDefnH,
}

impl Drop for FieldDefn {
    fn drop(&mut self) {
        if !self.c_field_defn.is_null() {
            unsafe { call_gdal_api!(self.api, OGR_Fld_Destroy, self.c_field_defn) };
        }
    }
}

impl FieldDefn {
    /// Create a new field definition.
    pub fn new(api: &'static GdalApi, name: &str, field_type: OGRFieldType) -> Result<Self> {
        let c_name = CString::new(name)?;
        let c_field_defn =
            unsafe { call_gdal_api!(api, OGR_Fld_Create, c_name.as_ptr(), field_type) };
        if c_field_defn.is_null() {
            return Err(api.last_null_pointer_err("OGR_Fld_Create"));
        }
        Ok(Self { api, c_field_defn })
    }

    /// Return the raw C handle.
    pub fn c_field_defn(&self) -> OGRFieldDefnH {
        self.c_field_defn
    }
}
