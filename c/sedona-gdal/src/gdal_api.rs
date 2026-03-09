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

use std::ffi::CStr;
use std::path::PathBuf;

use libloading::Library;

use crate::dyn_load;
use crate::errors::{GdalError, GdalInitLibraryError};
use crate::gdal_dyn_bindgen::SedonaGdalApi;

/// Invoke a function pointer from the `SedonaGdalApi` struct.
///
/// # Panics
///
/// Panics if the function pointer is `None`. This is unreachable in correct usage
/// because all function pointers are guaranteed to be `Some` after successful
/// initialization of [`GdalApi`] via [`GdalApi::try_from_shared_library`] or
/// [`GdalApi::try_from_current_process`], and you cannot obtain a `&GdalApi`
/// without successful initialization.
macro_rules! call_gdal_api {
    ($api:expr, $func:ident $(, $arg:expr)*) => {
        if let Some(func) = $api.inner.$func {
            func($($arg),*)
        } else {
            panic!("{} function not available", stringify!($func))
        }
    };
}

pub(crate) use call_gdal_api;

#[derive(Debug)]
pub struct GdalApi {
    pub(crate) inner: SedonaGdalApi,
    /// The dynamically loaded GDAL library. Kept alive for the lifetime of the
    /// function pointers in `inner`. This is never dropped because the `GdalApi`
    /// lives in a `static OnceLock` (see `global.rs`).
    _lib: Library,
    name: String,
}

impl GdalApi {
    pub fn try_from_shared_library(shared_library: PathBuf) -> Result<Self, GdalInitLibraryError> {
        let (lib, inner) = dyn_load::load_gdal_from_path(&shared_library)?;
        Ok(Self {
            inner,
            _lib: lib,
            name: shared_library.to_string_lossy().into_owned(),
        })
    }

    pub fn try_from_current_process() -> Result<Self, GdalInitLibraryError> {
        let (lib, inner) = dyn_load::load_gdal_from_current_process()?;
        Ok(Self {
            inner,
            _lib: lib,
            name: "current_process".to_string(),
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Query GDAL version information.
    ///
    /// `request` is one of the standard `GDALVersionInfo` keys:
    /// - `"RELEASE_NAME"` — e.g. `"3.8.4"`
    /// - `"VERSION_NUM"` — e.g. `"3080400"`
    /// - `"BUILD_INFO"` — multi-line build details
    pub fn version_info(&self, request: &str) -> String {
        let c_request = std::ffi::CString::new(request).unwrap();
        let ptr = unsafe { call_gdal_api!(self, GDALVersionInfo, c_request.as_ptr()) };
        if ptr.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned()
        }
    }

    /// Check the last CPL error and return a `GdalError`, it always returns an error struct
    /// (even when the error number is 0).
    pub fn last_cpl_err(&self, default_err_class: u32) -> GdalError {
        let err_no = unsafe { call_gdal_api!(self, CPLGetLastErrorNo) };
        let err_msg = unsafe {
            let msg_ptr = call_gdal_api!(self, CPLGetLastErrorMsg);
            if msg_ptr.is_null() {
                String::new()
            } else {
                CStr::from_ptr(msg_ptr).to_string_lossy().into_owned()
            }
        };
        unsafe { call_gdal_api!(self, CPLErrorReset) };
        GdalError::CplError {
            class: default_err_class,
            number: err_no,
            msg: err_msg,
        }
    }
}
