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
//! This module is our own reimplemented PROJ binding. We are not using georust/proj directly
//! because:
//!
//! - It creates a dedicated proj context each time we call proj functions, which is inefficient
//! - It leaks memory when an error occurs in some functions.
//! - It will call proj_cleanup when some proj objects are still being used in multi-threaded environment
//! - It requires proj_sys, which means we would need a build-time dependency on PROJ. This
//!   introduces licensing complexities because of restrictions placed on Apache projects
//!   with respect to the EPSG database.
//!
//! Our implementation dynamically can load PROJ from proj-sys or from a shared library
//! using dlopen()/LoadLibrary() at runtime.
//!
//! This wrapper requires PROJ >= 6.2.
//!
//! The original code was copied from
//! https://github.com/georust/proj/blob/95716f6cf90135150e313222bf296eb32e9e4493/src/proj.rs
//! however has undergone significant changes in the course of introducing the dynamic API.
//! The original work has MIT/Apache-2.0 license.
use std::{
    ffi::{c_int, CStr, CString},
    fmt::Debug,
    path::PathBuf,
    ptr,
    rc::Rc,
    sync::Arc,
};

use crate::{error::SedonaProjError, proj_dyn_bindgen};

/// A macro to safely call a function pointer from a ProjApi
///
/// Panics if the function was not available. A function not being available
/// would be a programming error on this library's part (creating the ProjApi
/// should error if a required function can't be found).
///
/// This macro helpfully allows searching for all API usage in the event that
/// we would like to revert the dynamic loading in favour of a simpler approach.
macro_rules! call_proj_api {
    ($api:expr, $func:ident $(, $arg:expr)*) => {
        if let Some(func) = $api.inner.$func {
            func($($arg),*)
        } else {
            panic!("{} function not available", stringify!(func))
        }
    }
}

/// Wrapper around a PROJ Context
///
/// A PROJ context the object used to configure data file locations, networking
/// preferences, and logging.
///
/// PROJ contexts are not thread safe; however, a future improvement may be to enforced
/// serialized access using a mutex rather than force context to be configured as thread
/// locals.
#[derive(Debug)]
pub(crate) struct ProjContext {
    inner: *mut crate::proj_dyn_bindgen::PJ_CONTEXT,
    api: Arc<ProjApi>,
}

impl ProjContext {
    /// Create a context from a shared library path
    ///
    /// This path will be passed verbatim to dlopen()/LoadLibrary(), from which function
    /// pointers will be loaded. This works for most package manager-installed PROJ
    /// distributions (and should error and not crash in the event of missing dependency
    /// libraries and/or insufficient version).
    pub(crate) fn try_from_shared_library(path: PathBuf) -> Result<Self, SedonaProjError> {
        let api = ProjApi::try_from_shared_library(path)?;
        api.check_version()?;

        let inner = unsafe { call_proj_api!(api, proj_context_create) };
        Ok(Self { inner, api })
    }

    /// Create a context from proj_sys
    ///
    /// This will error if the crate was built without the proj-sys feature.
    pub(crate) fn try_from_proj_sys() -> Result<Self, SedonaProjError> {
        let api = ProjApi::try_from_proj_sys()?;
        api.check_version()?;

        let inner = unsafe { call_proj_api!(api, proj_context_create) };
        Ok(Self { inner, api })
    }

    /// Set the path to proj.db
    ///
    /// The path to proj.db is required for PROJ distributions installed from a non-system
    /// package manager (i.e., MacOS, Windows, Conda, or pyproj via PyPI). For proj_sys
    /// or system (e.g., a Linux package manager), this shouldn't need explicit configuration.
    pub(crate) fn set_database_path(&mut self, path: &str) -> Result<(), SedonaProjError> {
        let path_c_string = CString::new(path)
            .map_err(|_| SedonaProjError::Invalid("Embedded nul in rust string".to_string()))?;
        let success = unsafe {
            call_proj_api!(
                self.api,
                proj_context_set_database_path,
                self.inner,
                path_c_string.as_ptr(),
                ptr::null(),
                ptr::null()
            )
        };

        if success != 0 {
            Ok(())
        } else {
            Err(SedonaProjError::LibraryError(format!(
                "Failed to set PROJ database path to '{path}'"
            )))
        }
    }

    /// Set the logging level for PROJ operations
    ///
    /// `level` - Unsigned Integer value representing the log level:
    /// - PJ_LOG_LEVEL_PJ_LOG_NONE (0): No logging
    /// - PJ_LOG_LEVEL_PJ_LOG_ERROR (1): Error messages
    /// - PJ_LOG_LEVEL_PJ_LOG_DEBUG (2): Debug messages
    /// - PJ_LOG_LEVEL_PJ_LOG_TRACE (3): Trace
    /// - PJ_LOG_LEVEL_PJ_LOG_TELL (4): Tell
    pub(crate) fn set_log_level(&self, level: u32) -> Result<(), SedonaProjError> {
        unsafe {
            call_proj_api!(self.api, proj_log_level, self.inner, level as _);
        }
        Ok(())
    }

    /// Set the path in which to look for PROJ data files
    ///
    /// Most PROJ distributions come with a few small data files installed to a /share directory
    /// and on MacOS, Windows, Conda, or pyproj via PyPI it is necessary to explicitly set this
    /// value to the installed data location. For proj_sys or system (e.g., a Linux package manager),
    /// this shouldn't need explicit configuration; however, this can be used for a separate install
    /// of proj-data or individually installed grid files.
    pub(crate) fn set_search_paths(
        &mut self,
        search_paths: &[String],
    ) -> Result<(), SedonaProjError> {
        let path_c_strings = search_paths
            .iter()
            .map(|path| CString::new(path.as_str()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| SedonaProjError::Invalid("embedded nul in Rust string".to_string()))?;
        let path_c_string_pointers = path_c_strings
            .iter()
            .map(|c_string| c_string.as_ptr())
            .collect::<Vec<_>>();

        unsafe {
            call_proj_api!(
                self.api,
                proj_context_set_search_paths,
                self.inner,
                path_c_string_pointers.len().try_into().map_err(|_| {
                    SedonaProjError::LibraryError("Invalid number of search paths".to_string())
                })?,
                path_c_string_pointers.as_ptr()
            )
        };
        Ok(())
    }

    /// Internal helper to get the last errno code
    fn errno(&self) -> c_int {
        unsafe { call_proj_api!(self.api, proj_context_errno, self.inner) }
    }

    /// Internal helper to get a human-readable error from an errno code
    ///
    /// Note that PROJ logs specific errors, so turning on logging may help to get a
    /// better handle on why an operation failed.
    fn errno_string(&self, code: c_int) -> String {
        let ptr = unsafe { call_proj_api!(self.api, proj_context_errno_string, self.inner, code) };

        if ptr.is_null() {
            "<unknown error>".to_string()
        } else {
            let c_str = unsafe { CStr::from_ptr(ptr) };
            c_str.to_string_lossy().to_string()
        }
    }
}

impl Drop for ProjContext {
    fn drop(&mut self) {
        unsafe {
            if let Some(func) = self.api.inner.proj_context_destroy {
                func(self.inner)
            } else {
                panic!("proj_context_destroy function not available");
            }
        }
    }
}

/// A wrapper around PROJ's PJ_AREA structure.
///
/// This represents a bounding box area of use to use as input to constrain
/// the candidate coordinate operations to be potentially more accurate. Note
/// that this structure can also represent an unconstrained area of interest.
#[derive(Debug)]
pub(crate) struct ProjArea {
    inner: *mut proj_dyn_bindgen::PJ_AREA,
    api: Arc<ProjApi>,
}

impl ProjArea {
    /// Create a new ProjArea from an optional Area.
    ///
    /// If an area is provided, sets the bounding box. Otherwise creates
    /// an unconstrained area of interest. Bounds are in the order
    /// west, south, east, north.
    fn new(api: Arc<ProjApi>, area: Option<(f64, f64, f64, f64)>) -> Self {
        let inner = unsafe { call_proj_api!(api, proj_area_create) };
        if let Some(narea) = area {
            unsafe {
                call_proj_api!(
                    api,
                    proj_area_set_bbox,
                    inner,
                    narea.0,
                    narea.1,
                    narea.2,
                    narea.3
                );
            }
        }
        Self { inner, api }
    }
}

impl Drop for ProjArea {
    fn drop(&mut self) {
        unsafe {
            call_proj_api!(self.api, proj_area_destroy, self.inner);
        }
    }
}

/// Coordinate operation or Coordinate Reference System
///
/// This wraps PROJ's PJ structure, which is used in PROJ to represent individual
/// coordinate reference systems ([Self::try_new]) or transformations between them
/// ([Self::try_crs_to_crs]).
///
/// Note that this wrapper always normalizes input and output coordinate reference systems
/// to longitude/easting first when creating transformations.
#[derive(Debug)]
pub(crate) struct Proj {
    inner: *mut proj_dyn_bindgen::PJconsts,
    ctx: Rc<ProjContext>,
}

impl Drop for Proj {
    fn drop(&mut self) {
        unsafe {
            call_proj_api!(self.ctx.api, proj_destroy, self.inner);
        }
    }
}

impl Proj {
    /// Create a new transformation object from a string representing a CRS or pipeline
    ///
    /// The CRS can be in any form accepted by PROJ. Some useful types of strings
    /// accepted are "authority:code", WKT1/2 strings, and PROJJSON.
    ///
    /// Pipelines are typically represented using PROJ-style strings but other types
    /// of input are also possible depending on the version.
    pub(crate) fn try_new(ctx: Rc<ProjContext>, definition: &str) -> Result<Self, SedonaProjError> {
        let c_definition = CString::new(definition)
            .map_err(|_| SedonaProjError::Invalid("embedded nul in Rust string".to_string()))?;
        let inner =
            unsafe { call_proj_api!(ctx.api, proj_create, ctx.inner, c_definition.as_ptr()) };
        if inner.is_null() {
            let err = ctx.errno();
            return Err(SedonaProjError::CreateError(ctx.errno_string(err)));
        }

        Ok(Self { inner, ctx })
    }

    /// Create a transformation between two coordinate reference systems.
    ///
    /// This creates a transformation pipeline that converts coordinates from
    /// the source CRS to the target CRS. The transformation is optimized
    /// for the specified area of use if one is provided.
    ///
    /// ## A Note on Coordinate Order
    /// The required input **and** output coordinate order is **normalised** to
    /// `Longitude, Latitude` / `Easting, Northing`.
    ///
    /// This overrides the expected order of the specified input and/or output CRS if necessary.
    /// For example: per its definition, EPSG:4326 has an axis order of Latitude, Longitude.
    /// Without normalisation, users would have to remember to reverse the coordinates.
    pub(crate) fn try_crs_to_crs(
        ctx: Rc<ProjContext>,
        source_crs: &Proj,
        target_crs: &Proj,
        area: Option<(f64, f64, f64, f64)>,
    ) -> Result<Self, SedonaProjError> {
        let proj_area = ProjArea::new(ctx.api.clone(), area);
        let inner = unsafe {
            call_proj_api!(
                ctx.api,
                proj_create_crs_to_crs_from_pj,
                ctx.inner,
                source_crs.inner,
                target_crs.inner,
                proj_area.inner,
                ptr::null()
            )
        };
        if inner.is_null() {
            let err = ctx.errno();
            return Err(SedonaProjError::CreateError(ctx.errno_string(err)));
        }

        let non_normalized = Self { inner, ctx };
        non_normalized.normalize_for_visualization()
    }

    fn normalize_for_visualization(&self) -> Result<Self, SedonaProjError> {
        let inner = unsafe {
            call_proj_api!(
                self.ctx.api,
                proj_normalize_for_visualization,
                self.ctx.inner,
                self.inner
            )
        };
        if inner.is_null() {
            let err = self.ctx.errno();
            return Err(SedonaProjError::CreateError(self.ctx.errno_string(err)));
        }

        Ok(Self {
            inner,
            ctx: self.ctx.clone(),
        })
    }

    /// Transform XY coordinates
    pub(crate) fn transform_xy(
        &mut self,
        point: (f64, f64),
    ) -> Result<(f64, f64), SedonaProjError> {
        // Filling extra dimensions with zeroes is what PostGIS does
        let xyzt = (point.0, point.1, 0.0, 0.0);
        let xyzt_out = self.transform(xyzt)?;
        Ok((xyzt_out.0, xyzt_out.1))
    }

    /// Transform XYZT coordinates
    pub(crate) fn transform(
        &mut self,
        point: (f64, f64, f64, f64),
    ) -> Result<(f64, f64, f64, f64), SedonaProjError> {
        let xyzt = proj_dyn_bindgen::PJ_XYZT {
            x: point.0,
            y: point.1,
            z: point.2,
            t: point.3,
        };

        let (transformed, err) = unsafe {
            call_proj_api!(self.ctx.api, proj_errno_reset, self.inner);
            let trans = call_proj_api!(
                self.ctx.api,
                proj_trans,
                self.inner,
                proj_dyn_bindgen::PJ_DIRECTION_PJ_FWD,
                proj_dyn_bindgen::PJ_COORD { xyzt }
            );
            let err = call_proj_api!(self.ctx.api, proj_errno, self.inner);
            (
                (trans.xyzt.x, trans.xyzt.y, trans.xyzt.z, trans.xyzt.t),
                err,
            )
        };

        if err != 0 {
            return Err(SedonaProjError::TransformError(self.ctx.errno_string(err)));
        }

        Ok(transformed)
    }
}

/// Dynamically loaded PROJ C API
///
/// Provides a rust wrapper around the set of function pointers required by
/// the Sedona PROJ wrapper. This is not intended to provide the entire API.
/// When loading from proj_sys, the function pointers to the Rust functions
/// are returned such that most of the code paths are the same even when using
/// proj_sys to resolve the PROJ location. Dynamic libraries are currently
/// loaded using C code; however, this could be migrated to Rust which also
/// provides dynamic library loading capabilities.
///
/// This API is thread safe and is marked as such; however, clients must not
/// call the inner release callback. Doing so will set function pointers to
/// null, which will cause subsequent calls to panic.
#[derive(Default)]
struct ProjApi {
    inner: proj_dyn_bindgen::ProjApi,
    name: String,
}

unsafe impl Send for ProjApi {}
unsafe impl Sync for ProjApi {}

impl Drop for ProjApi {
    fn drop(&mut self) {
        if let Some(releaser) = self.inner.release {
            unsafe { releaser(&mut self.inner) }
        }
    }
}

impl Debug for ProjApi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProjApi").field("name", &self.name).finish()
    }
}

impl ProjApi {
    fn try_from_shared_library(shared_library: PathBuf) -> Result<Arc<Self>, SedonaProjError> {
        let mut inner = proj_dyn_bindgen::ProjApi::default();
        let mut err_message = (0..1024).map(|_| 0).collect::<Vec<u8>>();
        let shared_library_c = CString::new(shared_library.to_string_lossy().to_string())
            .map_err(|_| SedonaProjError::Invalid("embedded nul in Rust string".to_string()))?;

        let err = unsafe {
            proj_dyn_bindgen::proj_dyn_api_init(
                &mut inner as _,
                shared_library_c.as_ptr(),
                err_message.as_mut_ptr() as _,
                err_message.len().try_into().unwrap(),
            )
        };

        let c_err_message = CStr::from_bytes_until_nul(&err_message)
            .map_err(|_| SedonaProjError::Invalid("embedded nul in C string".to_string()))?;
        if err != 0 {
            return Err(SedonaProjError::LibraryError(
                c_err_message.to_string_lossy().to_string(),
            ));
        }

        Ok(Arc::new(Self {
            inner,
            name: shared_library.to_string_lossy().to_string(),
        }))
    }

    fn try_from_proj_sys() -> Result<Arc<Self>, SedonaProjError> {
        #[cfg(feature = "proj-sys")]
        return Ok(Arc::new(Self::from_proj_sys()));

        #[cfg(not(feature = "proj-sys"))]
        Err(SedonaProjError::LibraryError(
            "Can't initialize ProjApi from sedona-proj without proj-sys feature".to_string(),
        ))
    }

    fn check_version(&self) -> Result<(), SedonaProjError> {
        if self.inner.proj_info.is_none() {
            return Err(SedonaProjError::LibraryError(
                "Can't check PROJ version: proj_info callable not set".to_string(),
            ));
        }

        let info = unsafe { call_proj_api!(self, proj_info) };
        if info.major > 6 || (info.major == 6 && info.minor >= 2) {
            return Ok(());
        }

        Err(SedonaProjError::LibraryError(format!(
            "PROJ >= 6.2.0 required for sedona-proj (got {}.{}.{}",
            info.major, info.minor, info.patch
        )))
    }

    /// When creating this object from proj_sys, we collect Rust function pointers
    /// such that the code paths taken by higher level code is mostly the same.
    /// This helps ensure our cargo test runs are vaguely realistic of usage from
    /// a binding like R or Python.
    #[cfg(feature = "proj-sys")]
    fn from_proj_sys() -> Self {
        use proj_sys::{
            proj_area_create, proj_area_destroy, proj_area_set_bbox, proj_context_create,
            proj_context_destroy, proj_context_errno, proj_context_errno_string,
            proj_context_set_database_path, proj_context_set_search_paths, proj_create,
            proj_create_crs_to_crs_from_pj, proj_cs_get_axis_count, proj_destroy, proj_errno,
            proj_errno_reset, proj_info, proj_log_level, proj_normalize_for_visualization,
            proj_trans, proj_trans_array,
        };

        let mut inner = proj_dyn_bindgen::ProjApi::default();

        // The target types here are too verbose
        #[allow(clippy::missing_transmute_annotations)]
        unsafe {
            inner.proj_area_create = Some(std::mem::transmute(
                proj_area_create as unsafe extern "C" fn() -> _,
            ));
            inner.proj_area_destroy = Some(std::mem::transmute(
                proj_area_destroy as unsafe extern "C" fn(*mut _) -> _,
            ));
            inner.proj_area_set_bbox = Some(std::mem::transmute(
                proj_area_set_bbox as unsafe extern "C" fn(*mut _, f64, f64, f64, f64) -> _,
            ));
            inner.proj_context_create = Some(std::mem::transmute(
                proj_context_create as unsafe extern "C" fn() -> _,
            ));
            inner.proj_context_destroy = Some(std::mem::transmute(
                proj_context_destroy as unsafe extern "C" fn(*mut _) -> _,
            ));
            inner.proj_context_errno = Some(std::mem::transmute(
                proj_context_errno as unsafe extern "C" fn(*mut _) -> _,
            ));
            inner.proj_context_errno_string = Some(std::mem::transmute(
                proj_context_errno_string as unsafe extern "C" fn(*mut _, i32) -> _,
            ));
            inner.proj_context_set_database_path = Some(std::mem::transmute(
                proj_context_set_database_path
                    as unsafe extern "C" fn(*mut _, *const _, *const _, *const _) -> _,
            ));
            inner.proj_context_set_search_paths = Some(std::mem::transmute(
                proj_context_set_search_paths
                    as unsafe extern "C" fn(*mut _, c_int, *const *const _) -> _,
            ));
            inner.proj_create_crs_to_crs_from_pj = Some(std::mem::transmute(
                proj_create_crs_to_crs_from_pj
                    as unsafe extern "C" fn(*mut _, *const _, *const _, *mut _, *const _) -> _,
            ));
            inner.proj_create = Some(std::mem::transmute(
                proj_create as unsafe extern "C" fn(*mut _, *const _) -> _,
            ));
            inner.proj_cs_get_axis_count = Some(std::mem::transmute(
                proj_cs_get_axis_count as unsafe extern "C" fn(*mut _, *const _) -> _,
            ));
            inner.proj_destroy = Some(std::mem::transmute(
                proj_destroy as unsafe extern "C" fn(*mut _) -> _,
            ));
            inner.proj_errno = Some(std::mem::transmute(
                proj_errno as unsafe extern "C" fn(*const _) -> _,
            ));
            inner.proj_errno_reset = Some(std::mem::transmute(
                proj_errno_reset as unsafe extern "C" fn(*const _) -> _,
            ));
            inner.proj_info = Some(std::mem::transmute(
                proj_info as unsafe extern "C" fn() -> _,
            ));
            inner.proj_log_level = Some(std::mem::transmute(
                proj_log_level as unsafe extern "C" fn(*mut _, _) -> _,
            ));
            inner.proj_normalize_for_visualization = Some(std::mem::transmute(
                proj_normalize_for_visualization as unsafe extern "C" fn(*mut _, *const _) -> _,
            ));
            inner.proj_trans = Some(std::mem::transmute(
                proj_trans as unsafe extern "C" fn(*mut _, _, _) -> _,
            ));
            inner.proj_trans_array = Some(std::mem::transmute(
                proj_trans_array as unsafe extern "C" fn(*mut _, _, usize, *mut _) -> _,
            ));
        }

        Self {
            inner,
            name: "proj_sys".to_string(),
        }
    }
}

// We don't have control over this generated source, so we can't derive the implementation
#[allow(clippy::derivable_impls)]
impl Default for proj_dyn_bindgen::ProjApi {
    fn default() -> Self {
        Self {
            proj_area_create: Default::default(),
            proj_area_destroy: Default::default(),
            proj_area_set_bbox: Default::default(),
            proj_context_create: Default::default(),
            proj_context_destroy: Default::default(),
            proj_context_errno_string: Default::default(),
            proj_context_errno: Default::default(),
            proj_context_set_database_path: Default::default(),
            proj_context_set_search_paths: Default::default(),
            proj_create_crs_to_crs_from_pj: Default::default(),
            proj_create: Default::default(),
            proj_cs_get_axis_count: Default::default(),
            proj_destroy: Default::default(),
            proj_errno_reset: Default::default(),
            proj_errno: Default::default(),
            proj_info: Default::default(),
            proj_log_level: Default::default(),
            proj_normalize_for_visualization: Default::default(),
            proj_trans: Default::default(),
            proj_trans_array: Default::default(),
            release: Default::default(),
            private_data: ptr::null_mut(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_relative_eq;

    /// Test conversion from NAD83 US Survey Feet (EPSG 2230) to NAD83 Metres (EPSG 26946)
    #[test]
    fn test_crs_to_crs_conversion() {
        let ctx = Rc::new(ProjContext::try_from_proj_sys().unwrap());

        let from = Proj::try_new(ctx.clone(), "EPSG:2230").unwrap();
        let to = Proj::try_new(ctx.clone(), "EPSG:26946").unwrap();
        let mut transformer = Proj::try_crs_to_crs(ctx.clone(), &from, &to, None).unwrap();

        let result = transformer
            .transform_xy((4760096.421921, 3744293.729449))
            .unwrap();
        assert_relative_eq!(result.0, 1450880.2910605022, epsilon = 1e-8);
        assert_relative_eq!(result.1, 1141263.0111604782, epsilon = 1e-8);
    }

    #[test]
    fn test_crs_to_crs_conversion_from_shared_library() {
        if let Ok(proj_library) = std::env::var("SEDONA_PROJ_TEST_SHARED_LIBRARY") {
            if !proj_library.is_empty() {
                let ctx =
                    Rc::new(ProjContext::try_from_shared_library(proj_library.into()).unwrap());

                let from = Proj::try_new(ctx.clone(), "EPSG:2230").unwrap();
                let to = Proj::try_new(ctx.clone(), "EPSG:26946").unwrap();
                let mut transformer = Proj::try_crs_to_crs(ctx.clone(), &from, &to, None).unwrap();

                let result = transformer
                    .transform_xy((4760096.421921, 3744293.729449))
                    .unwrap();
                assert_relative_eq!(result.0, 1450880.2910605022, epsilon = 1e-8);
                assert_relative_eq!(result.1, 1141263.0111604782, epsilon = 1e-8);
                return;
            }
        }

        println!("Skipping ProjContext::try_from_shared_library test - SEDONA_PROJ_TEST_SHARED_LIBRARY environment variable not set");
    }

    #[test]
    fn test_shared_library_error() {
        let err = ProjContext::try_from_shared_library("/not a shared library anywhere".into())
            .unwrap_err();
        assert!(matches!(err, SedonaProjError::LibraryError(_)));
        assert!(!err.to_string().is_empty());
    }

    /// Test error handling with invalid CRS definitions
    #[test]
    fn test_invalid_crs_error() {
        let ctx = Rc::new(ProjContext::try_from_proj_sys().unwrap());
        let err = Proj::try_new(ctx, "invalid_crs").unwrap_err();
        assert_eq!(err.to_string(), "Invalid PROJ string syntax");
    }

    /// Test that PROJ pipeline definition works
    #[test]
    fn test_pipeline_conversion() {
        let ctx = Rc::new(ProjContext::try_from_proj_sys().unwrap());

        // Generated by PROJ by specifying "from" and "to" EPSG codes
        let projstring = "
            proj=pipeline step proj=unitconvert xy_in=us-ft
            xy_out=m step inv proj=lcc lat_0=32.1666666666667
            lon_0=-116.25 lat_1=33.8833333333333 lat_2=32.7833333333333
            x_0=2000000.0001016 y_0=500000.0001016 ellps=GRS80 step proj=lcc lat_0=32.1666666666667
            lon_0=-116.25 lat_1=33.8833333333333 lat_2=32.7833333333333 x_0=2000000 y_0=500000
            ellps=GRS80
            ";
        let mut nad83_m = Proj::try_new(ctx, projstring).unwrap();

        // Presidio, San Francisco
        let result = nad83_m
            .transform_xy((4760096.421921, 3744293.729449))
            .unwrap();
        assert_relative_eq!(result.0, 1450880.2910605022, epsilon = 1e-8);
        assert_relative_eq!(result.1, 1141263.0111604782, epsilon = 1e-8);
    }

    /// Test conversion error handling
    #[test]
    fn test_conversion_error() {
        let ctx = Rc::new(ProjContext::try_from_proj_sys().unwrap());

        // GEOS projection expecting lon/lat input but getting large coordinate values
        let mut transformer = Proj::try_new(
            ctx,
            "+proj=geos +lon_0=0.00 +lat_0=0.00 +a=6378169.00 +b=6356583.80 +h=35785831.0",
        )
        .unwrap();

        let err = transformer
            .transform_xy((4760096.421921, 3744293.729449))
            .unwrap_err();

        assert!(matches!(err, SedonaProjError::TransformError(_)));
    }

    /// Test error recovery - subsequent valid conversions should work after an error
    #[test]
    fn test_error_recovery() {
        let ctx = Rc::new(ProjContext::try_from_proj_sys().unwrap());
        let mut transformer = Proj::try_new(
            ctx,
            "+proj=geos +lon_0=0.00 +lat_0=0.00 +a=6378169.00 +b=6356583.80 +h=35785831.0",
        )
        .unwrap();

        // First conversion should fail
        assert!(transformer
            .transform_xy((4760096.421921, 3744293.729449))
            .is_err());

        // But a subsequent valid conversion should still work
        assert!(transformer.transform_xy((0.0, 0.0)).is_ok());
    }

    /// Test coordinate transformation with area of use
    #[test]
    fn test_area_of_use() {
        let ctx = Rc::new(ProjContext::try_from_proj_sys().unwrap());
        let from = Proj::try_new(ctx.clone(), "EPSG:2230").unwrap();
        let to = Proj::try_new(ctx.clone(), "EPSG:4326").unwrap();

        // Define an area of use (California region)
        let area = (-124.0, 32.0, -114.0, 42.0); // west, south, east, north
        let mut transformer = Proj::try_crs_to_crs(ctx.clone(), &from, &to, Some(area)).unwrap();

        let result = transformer
            .transform_xy((4760096.421921, 3744293.729449))
            .unwrap();

        // Should get reasonable latitude/longitude values for California
        assert!(result.0 >= -125.0 && result.0 <= -113.0); // longitude
        assert!(result.1 >= 31.0 && result.1 <= 43.0); // latitude
    }
}
