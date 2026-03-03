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
//! <https://github.com/georust/gdal/blob/v0.19.0/src/spatial_ref/srs.rs>.
//! Original code is licensed under MIT.

use std::ffi::{CStr, CString};
use std::ptr;

use crate::call_gdal_api;
use crate::errors::{GdalError, Result};
use crate::gdal_api::GdalApi;
use crate::gdal_dyn_bindgen::*;

/// An OGR spatial reference system.
pub struct SpatialRef {
    api: &'static GdalApi,
    c_srs: OGRSpatialReferenceH,
}

unsafe impl Send for SpatialRef {}

impl Drop for SpatialRef {
    fn drop(&mut self) {
        if !self.c_srs.is_null() {
            unsafe { call_gdal_api!(self.api, OSRRelease, self.c_srs) };
        }
    }
}

impl SpatialRef {
    /// Create a new SpatialRef from a WKT string.
    pub fn from_wkt(api: &'static GdalApi, wkt: &str) -> Result<Self> {
        let c_wkt = CString::new(wkt)?;
        let c_srs = unsafe { call_gdal_api!(api, OSRNewSpatialReference, c_wkt.as_ptr()) };
        if c_srs.is_null() {
            return Err(GdalError::NullPointer {
                method_name: "OSRNewSpatialReference",
                msg: "failed to create spatial reference from WKT".to_string(),
            });
        }
        Ok(Self { api, c_srs })
    }

    /// Create a SpatialRef by cloning a borrowed C handle via `OSRClone`.
    ///
    /// # Safety
    ///
    /// The caller must ensure `c_srs` is a valid `OGRSpatialReferenceH`.
    pub unsafe fn from_c_srs_clone(
        api: &'static GdalApi,
        c_srs: OGRSpatialReferenceH,
    ) -> Result<Self> {
        let cloned = call_gdal_api!(api, OSRClone, c_srs);
        if cloned.is_null() {
            return Err(GdalError::NullPointer {
                method_name: "OSRClone",
                msg: "failed to clone spatial reference".to_string(),
            });
        }
        Ok(Self { api, c_srs: cloned })
    }

    /// Return the raw C handle.
    pub fn c_srs(&self) -> OGRSpatialReferenceH {
        self.c_srs
    }

    /// Export to PROJJSON string.
    pub fn to_projjson(&self) -> Result<String> {
        unsafe {
            let mut ptr: *mut std::os::raw::c_char = ptr::null_mut();
            let rv = call_gdal_api!(
                self.api,
                OSRExportToPROJJSON,
                self.c_srs,
                &mut ptr,
                ptr::null()
            );
            if rv != crate::gdal_dyn_bindgen::OGRERR_NONE || ptr.is_null() {
                return Err(GdalError::NullPointer {
                    method_name: "OSRExportToPROJJSON",
                    msg: "returned null".to_string(),
                });
            }
            let result = CStr::from_ptr(ptr).to_string_lossy().into_owned();
            call_gdal_api!(self.api, VSIFree, ptr as *mut std::ffi::c_void);
            Ok(result)
        }
    }
}

#[cfg(all(test, feature = "gdal-sys"))]
mod tests {
    use crate::errors::GdalError;
    use crate::global::get_global_gdal_api;
    use crate::spatial_ref::SpatialRef;

    const WGS84_WKT: &str = r#"GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]"#;

    #[test]
    fn test_from_wkt() {
        let api = get_global_gdal_api().unwrap();
        let srs = SpatialRef::from_wkt(api, WGS84_WKT).unwrap();
        assert!(!srs.c_srs().is_null());
    }

    #[test]
    fn test_from_wkt_invalid() {
        let api = get_global_gdal_api().unwrap();
        let err = SpatialRef::from_wkt(api, "WGS\u{0}84");
        assert!(matches!(err, Err(GdalError::FfiNulError(_))));
    }

    #[test]
    fn test_to_projjson() {
        let api = get_global_gdal_api().unwrap();
        let srs = SpatialRef::from_wkt(api, WGS84_WKT).unwrap();
        let projjson = srs.to_projjson().unwrap();
        assert!(
            projjson.contains("WGS 84"),
            "unexpected projjson: {projjson}"
        );
    }

    #[test]
    fn test_from_c_srs_clone() {
        let api = get_global_gdal_api().unwrap();
        let srs = SpatialRef::from_wkt(api, WGS84_WKT).unwrap();
        let cloned = unsafe { SpatialRef::from_c_srs_clone(api, srs.c_srs()) }.unwrap();
        assert_eq!(srs.to_projjson().unwrap(), cloned.to_projjson().unwrap());
    }
}
