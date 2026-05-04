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

use crate::s2geography_c_bindgen::*;
use std::cell::RefCell;
use std::ptr;
use thiserror::Error;

/// Compute an S2 Cell identifier from a longitude/latitude pair
///
/// If either longitude or latitude are NaN (e.g., an empty point),
/// the sentinel cell (`u64::MAX`) is returned. Lon/Lat pairs are
/// normalized such that invalid lon/lat pairs will still compute
/// a result (even though that result may be difficult to interpret).
pub fn s2_cell_id_from_lnglat(lnglat: (f64, f64)) -> u64 {
    let vertex = S2GeogVertex {
        v: [lnglat.0, lnglat.1, 0.0, 0.0],
    };
    unsafe { S2GeogLngLatToCellId(&vertex) }
}

/// Dependency versions for underlying libraries
pub struct Versions {}

impl Versions {
    /// Return the statically linked s2 version as a string
    pub fn s2geometry() -> String {
        unsafe {
            let raw_c_str = S2GeogS2GeometryVersion();
            let c_str = std::ffi::CStr::from_ptr(raw_c_str);
            c_str.to_string_lossy().into_owned()
        }
    }

    /// Return the linked Abseil version as a string
    ///
    /// Depending on build-time settings, this may have been statically
    /// or dynamically linked.
    pub fn abseil() -> String {
        unsafe {
            let raw_c_str = S2GeogAbseilVersion();
            let c_str = std::ffi::CStr::from_ptr(raw_c_str);
            c_str.to_string_lossy().into_owned()
        }
    }

    /// Return the linked OpenSSL version as a string
    ///
    /// Depending on build-time settings, this may have been statically
    /// or dynamically linked.
    pub fn openssl() -> String {
        unsafe {
            let raw_c_str = S2GeogOpenSSLVersion();
            let c_str = std::ffi::CStr::from_ptr(raw_c_str);
            c_str.to_string_lossy().into_owned()
        }
    }

    /// Return the linked nanoarrow version as a string
    pub fn nanoarrow() -> String {
        unsafe {
            let raw_c_str = S2GeogNanoarrowVersion();
            let c_str = std::ffi::CStr::from_ptr(raw_c_str);
            c_str.to_string_lossy().into_owned()
        }
    }

    /// Return the linked geoarrow version as a string
    pub fn geoarrow() -> String {
        unsafe {
            let raw_c_str = S2GeogGeoArrowVersion();
            let c_str = std::ffi::CStr::from_ptr(raw_c_str);
            c_str.to_string_lossy().into_owned()
        }
    }
}

/// Error returned by s2geography C API functions
#[derive(Debug, Error)]
#[error("{message} (code: {code})")]
pub struct S2GeogCError {
    /// The errno-compatible error code
    pub code: i32,
    /// The error message from the C API
    pub message: String,
}

impl S2GeogCError {
    /// Create a new error with the given code and message
    pub fn new(code: i32, message: String) -> Self {
        Self { code, message }
    }

    /// Create an error with just a code (no message available)
    pub fn from_code(func_name: &str, code: i32) -> Self {
        Self {
            code,
            message: format!("{} failed", func_name),
        }
    }
}

/// Call an s2geography C function that returns an error code and accepts an error pointer.
///
/// Uses a thread-local error object to avoid having to allocate one on each stack.
#[macro_export]
macro_rules! s2geog_call {
    ($func:ident ( $($arg:expr),* $(,)? )) => {{
        $crate::utils::S2GEOG_ERROR.with_borrow_mut(|err| {
            let code = $crate::s2geography_c_bindgen::$func($($arg,)* err.as_mut_ptr());
            if code == $crate::s2geography_c_bindgen::S2GEOGRAPHY_OK {
                Ok(())
            } else {
                let msg = err.message();
                if msg.is_empty() {
                    Err($crate::utils::S2GeogCError::from_code(stringify!($func), code))
                } else {
                    Err($crate::utils::S2GeogCError::new(code, msg))
                }
            }
        })
    }};
}

/// Call an s2geography C function that returns an error code (no error pointer).
///
/// Use this for functions like `S2GeogInitKernels` that only return a code.
#[macro_export]
macro_rules! s2geog_check {
    ($func:ident ( $($arg:expr),* $(,)? )) => {{
        let code = $crate::s2geography_c_bindgen::$func($($arg,)*);
        if code == $crate::s2geography_c_bindgen::S2GEOGRAPHY_OK {
            Ok(())
        } else {
            Err($crate::utils::S2GeogCError::from_code(stringify!($func), code))
        }
    }};
}

thread_local! {
    /// Thread-local error object reused across calls
    pub static S2GEOG_ERROR: RefCell<S2GeogErrorGuard> = RefCell::new(S2GeogErrorGuard::new());
}

/// Safe wrapper around an S2GeogError that ensures proper cleanup
pub struct S2GeogErrorGuard {
    ptr: *mut S2GeogError,
}

impl S2GeogErrorGuard {
    /// Create a new error guard with an allocated error object
    pub fn new() -> Self {
        let mut ptr: *mut S2GeogError = ptr::null_mut();
        unsafe {
            S2GeogErrorCreate(&mut ptr);
        }
        Self { ptr }
    }

    /// Get the raw pointer for passing to C functions
    pub fn as_mut_ptr(&mut self) -> *mut S2GeogError {
        self.ptr
    }

    /// Get the error message, or an empty string if no error
    pub fn message(&self) -> String {
        if self.ptr.is_null() {
            return String::new();
        }
        unsafe {
            let c_str = S2GeogErrorGetMessage(self.ptr);
            if c_str.is_null() {
                String::new()
            } else {
                std::ffi::CStr::from_ptr(c_str)
                    .to_string_lossy()
                    .into_owned()
            }
        }
    }
}

impl Default for S2GeogErrorGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for S2GeogErrorGuard {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                S2GeogErrorDestroy(self.ptr);
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_s2_cell_id_from_lnglat() {
        // Check a single, finite cell
        assert_eq!(s2_cell_id_from_lnglat((0.0, 0.0)), 1152921504606846977);

        // Emptyish cases should return the sentinel cell
        assert_eq!(s2_cell_id_from_lnglat((f64::NAN, 0.0)), u64::MAX);
        assert_eq!(s2_cell_id_from_lnglat((0.0, f64::NAN)), u64::MAX);
        assert_eq!(s2_cell_id_from_lnglat((f64::NAN, f64::NAN)), u64::MAX);

        // These should both return something (even if what it returns is difficult
        // to interpret)
        assert_ne!(s2_cell_id_from_lnglat((181.0, 0.0)), u64::MAX);
        assert_ne!(s2_cell_id_from_lnglat((0.0, 91.0)), u64::MAX);
    }

    #[test]
    fn test_versions() {
        assert!(Versions::s2geometry().contains("."));

        // These all may have been picked up from the environment
        assert!(Versions::abseil().starts_with("20"));
        assert!(Versions::openssl().contains("."));
        assert!(Versions::geoarrow().contains("."));
        assert!(Versions::nanoarrow().contains("."));
    }
}
