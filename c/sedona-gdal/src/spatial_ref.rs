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

use crate::errors::{GdalError, Result};
use crate::gdal_api::{call_gdal_api, GdalApi};
use crate::gdal_dyn_bindgen::OGRERR_NONE;
use crate::gdal_dyn_bindgen::*;

/// An OGR spatial reference system.
pub struct SpatialRef {
    api: &'static GdalApi,
    c_srs: OGRSpatialReferenceH,
}

// SAFETY: `SpatialRef` has unique ownership of its GDAL handle and only moves that
// ownership between threads. The handle is released exactly once on drop, and this
// wrapper does not provide shared concurrent access, so `Send` is sound while `Sync`
// remains intentionally unimplemented.
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
            return Err(api.last_null_pointer_err("OSRNewSpatialReference"));
        }
        Ok(Self { api, c_srs })
    }

    /// Set spatial reference from various text formats.
    ///
    /// This method will examine the provided input, and try to deduce the format,
    /// and then use it to initialize the spatial reference system. See the [C++ API docs][CPP]
    /// for details on these forms.
    ///
    /// [CPP]: https://gdal.org/api/ogrspatialref.html#_CPPv4N19OGRSpatialReference16SetFromUserInputEPKc
    pub fn from_definition(api: &'static GdalApi, definition: &str) -> Result<SpatialRef> {
        let c_definition = CString::new(definition)?;
        let c_obj = unsafe { call_gdal_api!(api, OSRNewSpatialReference, ptr::null()) };
        if c_obj.is_null() {
            return Err(api.last_null_pointer_err("OSRNewSpatialReference"));
        }
        let rv = unsafe { call_gdal_api!(api, OSRSetFromUserInput, c_obj, c_definition.as_ptr()) };
        if rv != OGRERR_NONE {
            unsafe { call_gdal_api!(api, OSRRelease, c_obj) };
            return Err(GdalError::OgrError {
                err: rv,
                method_name: "OSRSetFromUserInput",
            });
        }
        Ok(SpatialRef { api, c_srs: c_obj })
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
            return Err(api.last_null_pointer_err("OSRClone"));
        }
        Ok(Self { api, c_srs: cloned })
    }

    /// Return the borrowed raw C handle.
    ///
    /// The returned handle is owned by `self` and must not be released or destroyed
    /// by the caller. It is only valid for the lifetime of `&self`.
    pub fn c_srs(&self) -> OGRSpatialReferenceH {
        self.c_srs
    }

    /// Returns whether EPSG defines this CRS with latitude before longitude.
    pub fn epsg_treats_as_lat_long(&self) -> bool {
        unsafe { call_gdal_api!(self.api, OSREPSGTreatsAsLatLong, self.c_srs) != 0 }
    }

    /// Returns the data-axis to SRS-axis mapping as an owned vector.
    pub fn data_axis_to_srs_axis_mapping(&self) -> Result<Vec<i32>> {
        let mut count: i32 = 0;
        let ptr = unsafe {
            call_gdal_api!(
                self.api,
                OSRGetDataAxisToSRSAxisMapping,
                self.c_srs,
                &mut count
            )
        };

        if count < 0 {
            return Err(GdalError::BadArgument(format!(
                "OSRGetDataAxisToSRSAxisMapping returned negative count: {count}"
            )));
        }

        if count == 0 {
            return Ok(Vec::new());
        }

        if ptr.is_null() {
            return Err(self
                .api
                .last_null_pointer_err("OSRGetDataAxisToSRSAxisMapping"));
        }

        let count = usize::try_from(count)?;
        let mapping = unsafe { std::slice::from_raw_parts(ptr, count) }.to_vec();
        Ok(mapping)
    }

    /// Returns the current axis mapping strategy.
    pub fn axis_mapping_strategy(&self) -> OSRAxisMappingStrategy {
        unsafe { call_gdal_api!(self.api, OSRGetAxisMappingStrategy, self.c_srs) }
    }

    /// Sets the axis mapping strategy used by this spatial reference.
    pub fn set_axis_mapping_strategy(&self, strategy: OSRAxisMappingStrategy) {
        unsafe { call_gdal_api!(self.api, OSRSetAxisMappingStrategy, self.c_srs, strategy) };
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
            if rv != OGRERR_NONE {
                if !ptr.is_null() {
                    call_gdal_api!(self.api, VSIFree, ptr as *mut std::ffi::c_void);
                }
                return Err(GdalError::OgrError {
                    err: rv,
                    method_name: "OSRExportToPROJJSON",
                });
            }
            if ptr.is_null() {
                return Err(self.api.last_null_pointer_err("OSRExportToPROJJSON"));
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
    use crate::gdal_dyn_bindgen::{OAMS_AUTHORITY_COMPLIANT, OAMS_TRADITIONAL_GIS_ORDER};
    use crate::global::with_global_gdal_api;
    use crate::spatial_ref::SpatialRef;

    const WGS84_WKT: &str = r#"GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]"#;

    #[test]
    fn test_from_wkt() {
        with_global_gdal_api(|api| {
            let srs = SpatialRef::from_wkt(api, WGS84_WKT).unwrap();
            assert!(!srs.c_srs().is_null());
        })
        .unwrap();
    }

    #[test]
    fn test_from_wkt_invalid() {
        with_global_gdal_api(|api| {
            let err = SpatialRef::from_wkt(api, "WGS\u{0}84");
            assert!(matches!(err, Err(GdalError::FfiNulError(_))));
        })
        .unwrap();
    }

    #[test]
    fn test_from_definition() {
        with_global_gdal_api(|api| {
            let srs = SpatialRef::from_definition(api, WGS84_WKT).unwrap();
            assert!(!srs.c_srs().is_null());

            let srs = SpatialRef::from_definition(api, "EPSG:4326").unwrap();
            assert!(!srs.c_srs().is_null());
        })
        .unwrap();
    }

    #[test]
    fn test_epsg_treats_as_lat_long() {
        with_global_gdal_api(|api| {
            let srs = SpatialRef::from_definition(api, "EPSG:4326").unwrap();
            assert!(srs.epsg_treats_as_lat_long());
            let srs = SpatialRef::from_definition(api, "OGC:CRS84").unwrap();
            assert!(!srs.epsg_treats_as_lat_long());
        })
        .unwrap();
    }

    #[test]
    fn test_axis_mapping_strategy_getter_setter() {
        with_global_gdal_api(|api| {
            let srs = SpatialRef::from_definition(api, "EPSG:4326").unwrap();
            assert_eq!(srs.axis_mapping_strategy(), OAMS_AUTHORITY_COMPLIANT);
            assert_eq!(srs.data_axis_to_srs_axis_mapping().unwrap(), vec![1, 2]);

            // Force lon/lat order by swapping axes in the mapping, and check that the getter reflects that.
            srs.set_axis_mapping_strategy(OAMS_TRADITIONAL_GIS_ORDER);
            assert_eq!(srs.axis_mapping_strategy(), OAMS_TRADITIONAL_GIS_ORDER);
            assert_eq!(srs.data_axis_to_srs_axis_mapping().unwrap(), vec![2, 1]);

            srs.set_axis_mapping_strategy(OAMS_AUTHORITY_COMPLIANT);
            assert_eq!(srs.axis_mapping_strategy(), OAMS_AUTHORITY_COMPLIANT);
            assert_eq!(srs.data_axis_to_srs_axis_mapping().unwrap(), vec![1, 2]);

            let srs = SpatialRef::from_definition(api, "OGC:CRS84").unwrap();
            assert_eq!(srs.axis_mapping_strategy(), OAMS_AUTHORITY_COMPLIANT);
            assert_eq!(srs.data_axis_to_srs_axis_mapping().unwrap(), vec![1, 2]);

            // The original axis order of CRS84 is already lat/lon, so swapping axes in the mapping should
            // have no effect.
            srs.set_axis_mapping_strategy(OAMS_TRADITIONAL_GIS_ORDER);
            assert_eq!(srs.axis_mapping_strategy(), OAMS_TRADITIONAL_GIS_ORDER);
            assert_eq!(srs.data_axis_to_srs_axis_mapping().unwrap(), vec![1, 2]);

            srs.set_axis_mapping_strategy(OAMS_AUTHORITY_COMPLIANT);
            assert_eq!(srs.axis_mapping_strategy(), OAMS_AUTHORITY_COMPLIANT);
            assert_eq!(srs.data_axis_to_srs_axis_mapping().unwrap(), vec![1, 2]);
        })
        .unwrap();
    }

    #[test]
    fn test_to_projjson() {
        with_global_gdal_api(|api| {
            let srs = SpatialRef::from_wkt(api, WGS84_WKT).unwrap();
            let projjson = srs.to_projjson().unwrap();
            assert!(
                projjson.contains("WGS 84"),
                "unexpected projjson: {projjson}"
            );
        })
        .unwrap();
    }

    #[test]
    fn test_from_c_srs_clone() {
        with_global_gdal_api(|api| {
            let srs = SpatialRef::from_wkt(api, WGS84_WKT).unwrap();
            let cloned = unsafe { SpatialRef::from_c_srs_clone(api, srs.c_srs()) }.unwrap();
            assert_eq!(srs.to_projjson().unwrap(), cloned.to_projjson().unwrap());
        })
        .unwrap();
    }
}
