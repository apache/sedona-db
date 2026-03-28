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

// SAFETY: `Geometry` has unique ownership of its GDAL handle and only transfers that
// ownership across threads. The handle is destroyed exactly once on drop, and this
// wrapper does not expose concurrent shared access, so `Send` is sound while `Sync`
// remains intentionally unimplemented.
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
            if !c_geom.is_null() {
                unsafe { call_gdal_api!(api, OGR_G_DestroyGeometry, c_geom) };
            }
            return Err(GdalError::OgrError {
                err: rv,
                method_name: "OGR_G_CreateFromWkb",
            });
        }
        if c_geom.is_null() {
            return Err(api.last_null_pointer_err("OGR_G_CreateFromWkb"));
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
            if !c_geom.is_null() {
                unsafe { call_gdal_api!(api, OGR_G_DestroyGeometry, c_geom) };
            }
            return Err(GdalError::OgrError {
                err: rv,
                method_name: "OGR_G_CreateFromWkt",
            });
        }
        if c_geom.is_null() {
            return Err(api.last_null_pointer_err("OGR_G_CreateFromWkt"));
        }
        Ok(Self { api, c_geom })
    }

    /// Return the borrowed raw C geometry handle.
    ///
    /// The returned handle is owned by `self` and must not be destroyed by the
    /// caller. It is only valid for the lifetime of `&self`.
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

#[cfg(all(test, feature = "gdal-sys"))]
mod tests {
    use super::*;

    use crate::errors::GdalError;
    use crate::global::with_global_gdal_api;

    #[test]
    fn test_from_wkt_envelope() {
        with_global_gdal_api(|api| {
            let geometry = Geometry::from_wkt(api, "POINT (1 2)").unwrap();
            let envelope = geometry.envelope();

            assert_eq!(envelope.MinX, 1.0);
            assert_eq!(envelope.MaxX, 1.0);
            assert_eq!(envelope.MinY, 2.0);
            assert_eq!(envelope.MaxY, 2.0);
        })
        .unwrap();
    }

    #[test]
    fn test_from_wkb() {
        with_global_gdal_api(|api| {
            let geometry = Geometry::from_wkt(api, "POINT (1 2)").unwrap();
            let wkb = geometry.wkb().unwrap();
            let geometry = Geometry::from_wkb(api, &wkb).unwrap();
            assert!(!geometry.c_geometry().is_null());
        })
        .unwrap();
    }

    #[test]
    fn test_wkb_round_trip_preserves_envelope() {
        with_global_gdal_api(|api| {
            let geometry = Geometry::from_wkt(api, "LINESTRING (0 1, 2 3, 4 5)").unwrap();
            let wkb = geometry.wkb().unwrap();
            let round_tripped = Geometry::from_wkb(api, &wkb).unwrap();

            assert!(!wkb.is_empty());

            let envelope = geometry.envelope();
            let round_trip_envelope = round_tripped.envelope();
            assert_eq!(envelope.MinX, round_trip_envelope.MinX);
            assert_eq!(envelope.MaxX, round_trip_envelope.MaxX);
            assert_eq!(envelope.MinY, round_trip_envelope.MinY);
            assert_eq!(envelope.MaxY, round_trip_envelope.MaxY);
        })
        .unwrap();
    }

    #[test]
    fn test_from_wkt_invalid() {
        with_global_gdal_api(|api| {
            let error = Geometry::from_wkt(api, "POINT (").err().unwrap();
            assert!(matches!(
                error,
                GdalError::OgrError {
                    method_name: "OGR_G_CreateFromWkt",
                    ..
                }
            ));
        })
        .unwrap();
    }

    #[test]
    fn test_from_wkb_invalid() {
        with_global_gdal_api(|api| {
            let error = Geometry::from_wkb(api, &[0x01, 0x02, 0x03]).err().unwrap();
            assert!(matches!(
                error,
                GdalError::OgrError {
                    method_name: "OGR_G_CreateFromWkb",
                    ..
                }
            ));
        })
        .unwrap();
    }
}
