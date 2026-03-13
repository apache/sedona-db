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
//! <https://github.com/georust/gdal/blob/v0.19.0/src/vector/geometry.rs>.
//! Original code is licensed under MIT.

use std::ffi::CString;
use std::ptr;

use crate::errors::{GdalError, Result};
use crate::gdal_api::{call_gdal_api, GdalApi};
use crate::gdal_dyn_bindgen::*;

pub type Envelope = OGREnvelope;

/// An OGR geometry.
pub struct Geometry {
    api: &'static GdalApi,
    c_geom: OGRGeometryH,
}

unsafe impl Send for Geometry {}

impl Drop for Geometry {
    fn drop(&mut self) {
        if !self.c_geom.is_null() {
            unsafe { call_gdal_api!(self.api, OGR_G_DestroyGeometry, self.c_geom) };
        }
    }
}

impl Geometry {
    /// Create a geometry from WKB bytes.
    pub fn from_wkb(api: &'static GdalApi, wkb: &[u8]) -> Result<Self> {
        let wkb_len: i32 = wkb.len().try_into()?;
        let mut c_geom: OGRGeometryH = ptr::null_mut();
        let rv = unsafe {
            call_gdal_api!(
                api,
                OGR_G_CreateFromWkb,
                wkb.as_ptr() as *const std::ffi::c_void,
                ptr::null_mut(), // hSRS
                &mut c_geom,
                wkb_len
            )
        };
        if rv != OGRERR_NONE {
            return Err(GdalError::OgrError {
                err: rv,
                method_name: "OGR_G_CreateFromWkb",
            });
        }
        if c_geom.is_null() {
            return Err(GdalError::NullPointer {
                method_name: "OGR_G_CreateFromWkb",
                msg: "returned null geometry".to_string(),
            });
        }
        Ok(Self { api, c_geom })
    }

    /// Create a geometry from WKT string.
    pub fn from_wkt(api: &'static GdalApi, wkt: &str) -> Result<Self> {
        let c_wkt = CString::new(wkt)?;
        let mut wkt_ptr = c_wkt.as_ptr() as *mut std::os::raw::c_char;
        let mut c_geom: OGRGeometryH = ptr::null_mut();
        let rv = unsafe {
            call_gdal_api!(
                api,
                OGR_G_CreateFromWkt,
                &mut wkt_ptr,
                ptr::null_mut(), // hSRS
                &mut c_geom
            )
        };
        if rv != OGRERR_NONE {
            return Err(GdalError::OgrError {
                err: rv,
                method_name: "OGR_G_CreateFromWkt",
            });
        }
        if c_geom.is_null() {
            return Err(GdalError::NullPointer {
                method_name: "OGR_G_CreateFromWkt",
                msg: "returned null geometry".to_string(),
            });
        }
        Ok(Self { api, c_geom })
    }

    /// Return the raw C geometry handle.
    pub fn c_geometry(&self) -> OGRGeometryH {
        self.c_geom
    }

    /// Get the bounding envelope.
    pub fn envelope(&self) -> Envelope {
        let mut env = OGREnvelope {
            MinX: 0.0,
            MaxX: 0.0,
            MinY: 0.0,
            MaxY: 0.0,
        };
        unsafe { call_gdal_api!(self.api, OGR_G_GetEnvelope, self.c_geom, &mut env) };
        env
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
                wkbNDR, // little-endian
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
}
