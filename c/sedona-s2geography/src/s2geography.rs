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

use std::sync::Arc;

use datafusion_common::Result;
use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::ScalarKernelRef;
use sedona_extension::{extension::SedonaCScalarKernel, scalar_kernel::ImportedScalarKernel};

use crate::geography_glue_bindgen::*;

/// Compute an S2 Cell identifier from a longitude/latitude pair
///
/// If either longitude or latitude are NaN (e.g., an empty point),
/// the sentinel cell (`u64::MAX`) is returned. Lon/Lat pairs are
/// normalized such that invalid lon/lat pairs will still compute
/// a result (even though that result may be difficult to interpret).
pub fn s2_cell_id_from_lnglat(lnglat: (f64, f64)) -> u64 {
    unsafe { SedonaGeographyGlueLngLatToCellId(lnglat.0, lnglat.1) }
}

pub fn s2_scalar_kernels() -> Result<Vec<(String, ScalarKernelRef)>> {
    let mut ffi_scalar_kernels = Vec::<SedonaCScalarKernel>::new();
    ffi_scalar_kernels.resize_with(unsafe { SedonaGeographyGlueNumKernels() }, Default::default);

    let err_code = unsafe {
        SedonaGeographyGlueInitKernels(
            ffi_scalar_kernels.as_mut_ptr() as _,
            size_of::<SedonaCScalarKernel>() * ffi_scalar_kernels.len(),
        )
    };

    if err_code != 0 {
        return sedona_internal_err!("SedonaGeographyGlueInitKernels() failed");
    }

    ffi_scalar_kernels
        .into_iter()
        .map(|c_kernel| {
            let imported_kernel = ImportedScalarKernel::try_from(c_kernel)?;
            Ok((
                imported_kernel.function_name().unwrap().to_string(),
                Arc::new(imported_kernel) as ScalarKernelRef,
            ))
        })
        .collect()
}

/// Dependency versions for underlying libraries
pub struct Versions {}

impl Versions {
    /// Return the statically linked s2 version as a string
    pub fn s2geometry() -> String {
        unsafe {
            let raw_c_str = SedonaGeographyGlueS2GeometryVersion();
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
            let raw_c_str = SedonaGeographyGlueAbseilVersion();
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
            let raw_c_str = SedonaGeographyGlueOpenSSLVersion();
            let c_str = std::ffi::CStr::from_ptr(raw_c_str);
            c_str.to_string_lossy().into_owned()
        }
    }

    /// A simple function that performs a non-trivial operation
    ///
    /// This is needed as a smoke check to ensure required libraries are linked.
    pub fn test_linkage() -> f64 {
        unsafe { SedonaGeographyGlueTestLinkage() }
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
    fn test_s2_scalar_kernels() {
        let kernels = s2_scalar_kernels().unwrap();
        assert!(!kernels.is_empty());
    }

    #[test]
    fn test_versions() {
        assert_eq!(Versions::s2geometry(), "0.11.1");
        assert!(Versions::abseil().starts_with("20"));
        assert!(Versions::openssl().contains("."));
        assert!(Versions::test_linkage() > 0.0);
    }
}
