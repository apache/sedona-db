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

/// Builder for the global [`GdalApi`].
///
/// Provides a way to configure how GDAL is loaded before the first use.
/// Use [`configure_global_gdal_api`] to install a builder, then the actual
/// loading happens lazily on the first call to [`get_global_gdal_api`].
///
/// # Examples
///
/// ```no_run
/// use sedona_gdal::register::{GdalApiBuilder, configure_global_gdal_api};
///
/// // Configure with a specific shared library path
/// let builder = GdalApiBuilder::default()
///     .with_shared_library("/usr/lib/libgdal.so".into());
/// configure_global_gdal_api(builder).unwrap();
/// ```
#[derive(Default)]
pub struct GdalApiBuilder {
    shared_library: Option<PathBuf>,
}

impl GdalApiBuilder {
    /// Set the path to the GDAL shared library.
    ///
    /// If unset, GDAL symbols will be resolved from the current process image
    /// (equivalent to `dlopen(NULL)`), which requires GDAL to already be linked
    /// into the process (e.g. via `gdal-sys`).
    ///
    /// Note that the path is passed directly to `dlopen()`/`LoadLibrary()`,
    /// which takes into account the working directory. As a security measure,
    /// applications may wish to verify that the path is absolute. This should
    /// not be specified from untrusted input.
    pub fn with_shared_library(self, path: PathBuf) -> Self {
        Self {
            shared_library: Some(path),
        }
    }

    /// Build a [`GdalApi`] with the configured options.
    ///
    /// When `shared_library` is set, loads GDAL from that path. Otherwise,
    /// resolves symbols from the current process (with an optional
    /// compile-time version check when the `gdal-sys` feature is enabled).
    pub fn build(&self) -> Result<GdalApi, GdalInitLibraryError> {
        if let Some(shared_library) = &self.shared_library {
            GdalApi::try_from_shared_library(shared_library.clone())
        } else {
            #[cfg(feature = "gdal-sys")]
            check_gdal_version()?;
            GdalApi::try_from_current_process()
        }
    }
}

/// Global builder configuration, protected by a [`Mutex`].
///
/// Set via [`configure_global_gdal_api`] before the first call to
/// [`get_global_gdal_api`]. Multiple calls to `configure_global_gdal_api`
/// are allowed as long as the API has not been initialized yet.
/// The same mutex also serves as the initialization guard for
/// [`get_global_gdal_api`], eliminating the need for a separate lock.
static GDAL_API_BUILDER: Mutex<Option<GdalApiBuilder>> = Mutex::new(None);

static GDAL_API: OnceLock<GdalApi> = OnceLock::new();

/// Get a reference to the global GDAL API, initializing it if not already done.
///
/// On first call, reads the builder set by [`configure_global_gdal_api`] (or uses
/// [`GdalApiBuilder::default()`] if none was configured) and calls its `build()`
/// method to create the [`GdalApi`]. The result is stored in a process-global
/// `OnceLock` and reused for all subsequent calls.
pub fn get_global_gdal_api() -> Result<&'static GdalApi, GdalInitLibraryError> {
    if let Some(api) = GDAL_API.get() {
        return Ok(api);
    }

    let guard = GDAL_API_BUILDER
        .lock()
        .map_err(|_| GdalInitLibraryError::Invalid("GDAL API builder lock poisoned".to_string()))?;

    if let Some(api) = GDAL_API.get() {
        return Ok(api);
    }

    let api = guard
        .as_ref()
        .unwrap_or(&GdalApiBuilder::default())
        .build()?;

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

/// Configure the global GDAL API.
///
/// Stores the given [`GdalApiBuilder`] for use when the global [`GdalApi`] is
/// first initialized (lazily, on the first call to [`get_global_gdal_api`]).
///
/// This can be called multiple times before the API is initialized — each call
/// replaces the previous builder. However, once [`get_global_gdal_api`] has been
/// called and the API has been successfully initialized, subsequent configurations
/// are accepted but will have no effect (the `OnceLock` ensures the API is created
/// only once).
///
/// # Typical usage
///
/// 1. The application (e.g. sedona-db) calls `configure_global_gdal_api` with its
///    default builder early in startup.
/// 2. User code may call `configure_global_gdal_api` again to override the
///    configuration before the first query that uses GDAL.
/// 3. On the first actual GDAL operation, [`get_global_gdal_api`] reads the
///    builder and initializes the API.
pub fn configure_global_gdal_api(builder: GdalApiBuilder) -> Result<(), GdalInitLibraryError> {
    let mut global_builder = GDAL_API_BUILDER.lock().map_err(|_| {
        GdalInitLibraryError::Invalid(
            "Failed to acquire lock for global GDAL configuration".to_string(),
        )
    })?;
    global_builder.replace(builder);
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

#[cfg(test)]
mod test {
    use super::*;

    /// Building with default options (no shared library) should succeed when
    /// `gdal-sys` is linked, loading symbols from the current process.
    #[test]
    fn test_builder_default() {
        let api = GdalApiBuilder::default()
            .build()
            .expect("GdalApiBuilder::default().build() should succeed with gdal-sys");
        assert_eq!(api.name(), "current_process");
    }

    /// Building with an explicit shared library path should use that path.
    /// Gated behind an environment variable; skips gracefully if unset.
    #[test]
    fn test_builder_with_shared_library() {
        if let Ok(gdal_library) = std::env::var("SEDONA_GDAL_TEST_SHARED_LIBRARY") {
            if !gdal_library.is_empty() {
                let api = GdalApiBuilder::default()
                    .with_shared_library(gdal_library.clone().into())
                    .build()
                    .expect("Should build GdalApi from SEDONA_GDAL_TEST_SHARED_LIBRARY");
                assert_eq!(api.name(), gdal_library);
                return;
            }
        }

        println!(
            "Skipping test_builder_with_shared_library - \
             SEDONA_GDAL_TEST_SHARED_LIBRARY environment variable not set"
        );
    }

    /// Building with an invalid shared library path should return an error.
    #[test]
    fn test_builder_invalid_path() {
        let err = GdalApiBuilder::default()
            .with_shared_library("/nonexistent/libgdal.so".into())
            .build()
            .unwrap_err();
        assert!(
            matches!(err, GdalInitLibraryError::LibraryError(_)),
            "Expected LibraryError, got: {err:?}"
        );
    }

    /// `get_global_gdal_api` should succeed and return a valid API reference.
    ///
    /// Note: this test touches the process-global `OnceLock` and cannot be
    /// "undone", so it effectively tests the first-call initialization path.
    /// Subsequent tests in the same process will hit the fast path.
    #[test]
    fn test_get_global_gdal_api() {
        let api = get_global_gdal_api().expect("get_global_gdal_api should succeed");
        assert!(!api.name().is_empty(), "API name should not be empty");
    }

    /// After `get_global_gdal_api` succeeds, `is_gdal_api_configured` should
    /// return true.
    #[test]
    fn test_is_gdal_api_configured() {
        // Ensure the API is initialized.
        let _ = get_global_gdal_api().expect("get_global_gdal_api should succeed");
        assert!(
            is_gdal_api_configured(),
            "is_gdal_api_configured should return true after initialization"
        );
    }

    /// `with_global_gdal_api` should pass a valid reference to the closure.
    #[test]
    fn test_with_global_gdal_api() {
        let name = with_global_gdal_api(|api| Ok(api.name().to_string()))
            .expect("with_global_gdal_api should succeed");
        assert!(!name.is_empty(), "API name should not be empty");
    }
}
