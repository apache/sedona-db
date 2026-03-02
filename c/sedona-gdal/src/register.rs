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

use crate::errors::GdalInitLibraryError;
use crate::gdal_api::GdalApi;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// Minimum GDAL version required by sedona-gdal.
#[cfg(feature = "gdal-sys")]
const MIN_GDAL_VERSION_MAJOR: i32 = 3;
#[cfg(feature = "gdal-sys")]
const MIN_GDAL_VERSION_MINOR: i32 = 4;

static GDAL_API: OnceLock<GdalApi> = OnceLock::new();
static GDAL_API_INIT_LOCK: Mutex<()> = Mutex::new(());

fn init_gdal_api<F>(init: F) -> Result<&'static GdalApi, GdalInitLibraryError>
where
    F: FnOnce() -> Result<GdalApi, GdalInitLibraryError>,
{
    if let Some(api) = GDAL_API.get() {
        return Ok(api);
    }

    let _guard = GDAL_API_INIT_LOCK
        .lock()
        .map_err(|_| GdalInitLibraryError::Invalid("GDAL API init lock poisoned".to_string()))?;

    if let Some(api) = GDAL_API.get() {
        return Ok(api);
    }

    let api = init()?;

    // Register all GDAL drivers once, immediately after loading symbols.
    // This mirrors georust/gdal's `_register_drivers()` pattern where
    // `GDALAllRegister` is called via `std::sync::Once` before any driver
    // lookup or dataset open. Here the `OnceLock` + `Mutex` already
    // guarantees this runs exactly once.
    unsafe {
        let Some(gdal_all_register) = api.inner.GDALAllRegister else {
            return Err(GdalInitLibraryError::LibraryError(
                "GDALAllRegister symbol not loaded".to_string(),
            ));
        };
        gdal_all_register();
    }

    let _ = GDAL_API.set(api);
    Ok(GDAL_API.get().expect("GDAL API should be set"))
}

pub fn configure_global_gdal_api(shared_library: PathBuf) -> Result<(), GdalInitLibraryError> {
    init_gdal_api(|| GdalApi::try_from_shared_library(shared_library))?;
    Ok(())
}

pub fn is_gdal_api_configured() -> bool {
    GDAL_API.get().is_some()
}

pub fn with_global_gdal_api<F, R>(func: F) -> Result<R, GdalInitLibraryError>
where
    F: FnOnce(&'static GdalApi) -> Result<R, GdalInitLibraryError>,
{
    let api = get_global_gdal_api()?;
    func(api)
}

/// Get a reference to the global GDAL API, initializing from the current process
/// if not already done.
pub fn get_global_gdal_api() -> Result<&'static GdalApi, GdalInitLibraryError> {
    init_gdal_api(|| {
        #[cfg(feature = "gdal-sys")]
        check_gdal_version()?;
        GdalApi::try_from_current_process()
    })
}

/// Verify that the compile-time-linked GDAL library meets the minimum version
/// requirement. Calling into `gdal-sys` also forces the linker to include GDAL
/// symbols, so that `try_from_current_process` (which resolves function pointers
/// via `dlsym` on the current process) can find them at runtime.
///
/// We use `GDALVersionInfo("VERSION_NUM")` instead of `GDALCheckVersion` because
/// the latter performs an **exact** major.minor match and rejects newer versions
/// (e.g. GDAL 3.12 fails a check for 3.4), whereas we need a **minimum** version
/// check (>=).
#[cfg(feature = "gdal-sys")]
fn check_gdal_version() -> Result<(), GdalInitLibraryError> {
    use std::ffi::CStr;

    // Matches the GDAL_COMPUTE_VERSION(maj,min,rev) macro: maj*1000000 + min*10000 + rev*100
    let min_version_num = MIN_GDAL_VERSION_MAJOR * 1_000_000 + MIN_GDAL_VERSION_MINOR * 10_000;

    let version_ptr = unsafe { gdal_sys::GDALVersionInfo(c"VERSION_NUM".as_ptr()) };
    if version_ptr.is_null() {
        return Err(GdalInitLibraryError::LibraryError(
            "GDALVersionInfo(\"VERSION_NUM\") returned null".to_string(),
        ));
    }

    let version_cstr = unsafe { CStr::from_ptr(version_ptr) };
    let version_num: i32 = version_cstr
        .to_str()
        .map_err(|e| {
            GdalInitLibraryError::LibraryError(format!(
                "GDAL version string is not valid UTF-8: {e}"
            ))
        })?
        .trim()
        .parse()
        .map_err(|e| {
            GdalInitLibraryError::LibraryError(format!(
                "Failed to parse GDAL version number {:?}: {e}",
                version_cstr
            ))
        })?;

    if version_num < min_version_num {
        // Get the human-readable release name for the error message.
        let release_ptr = unsafe { gdal_sys::GDALVersionInfo(c"RELEASE_NAME".as_ptr()) };
        let release_name = if release_ptr.is_null() {
            format!("version_num={version_num}")
        } else {
            unsafe { CStr::from_ptr(release_ptr) }
                .to_string_lossy()
                .into_owned()
        };
        return Err(GdalInitLibraryError::LibraryError(format!(
            "GDAL >= {MIN_GDAL_VERSION_MAJOR}.{MIN_GDAL_VERSION_MINOR} required \
             for sedona-gdal (found {release_name})"
        )));
    }

    Ok(())
}
