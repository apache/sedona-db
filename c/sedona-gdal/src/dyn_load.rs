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

use std::path::Path;

use libloading::Library;

use crate::errors::GdalInitLibraryError;
use crate::gdal_dyn_bindgen::SedonaGdalApi;

/// Load a single symbol from the library and write it into the given field.
///
/// We load as a raw `*const ()` pointer and transmute to the target function pointer
/// type. This is the standard pattern for dynamic symbol loading where the loaded
/// symbol's signature is known but cannot be expressed as a generic parameter to
/// `Library::get` (because each field has a different signature).
///
/// On failure returns a `GdalInitLibraryError` with the symbol name and the
/// underlying OS error message.
macro_rules! load_fn {
    ($lib:expr, $api:expr, $name:ident) => {
        // The target types here are too verbose to annotate for each call site
        #[allow(clippy::missing_transmute_annotations)]
        {
            $api.$name = Some(unsafe {
                let sym = $lib
                    .get::<*const ()>(concat!(stringify!($name), "\0").as_bytes())
                    .map_err(|e| {
                        GdalInitLibraryError::LibraryError(format!(
                            "Failed to load symbol {}: {}",
                            stringify!($name),
                            e
                        ))
                    })?;
                std::mem::transmute(sym.into_raw().into_raw())
            });
        }
    };
}

/// Try to load a symbol under one of several names (e.g. for C++ mangled symbols).
/// Writes the first successful match into `$api.$field`. Returns an error only if
/// *none* of the names resolve.
macro_rules! load_fn_any {
    ($lib:expr, $api:expr, $field:ident, [$($name:expr),+ $(,)?]) => {{
        let mut found = false;
        $(
            if !found {
                if let Ok(sym) = unsafe { $lib.get::<*const ()>($name) } {
                    // The target types here are too verbose to annotate for each call site
                    #[allow(clippy::missing_transmute_annotations)]
                    {
                        $api.$field = Some(unsafe { std::mem::transmute(sym.into_raw().into_raw()) });
                    }
                    found = true;
                }
            }
        )+
        if !found {
            return Err(GdalInitLibraryError::LibraryError(format!(
                "Failed to resolve {} under any known mangled name",
                stringify!($field),
            )));
        }
    }};
}

/// Populate all function-pointer fields of [`SedonaGdalApi`] from the given
/// [`Library`] handle.
fn load_all_symbols(lib: &Library, api: &mut SedonaGdalApi) -> Result<(), GdalInitLibraryError> {
    // --- Dataset ---
    load_fn!(lib, api, GDALOpenEx);
    load_fn!(lib, api, GDALClose);
    load_fn!(lib, api, GDALGetRasterXSize);
    load_fn!(lib, api, GDALGetRasterYSize);
    load_fn!(lib, api, GDALGetRasterCount);
    load_fn!(lib, api, GDALGetRasterBand);
    load_fn!(lib, api, GDALGetGeoTransform);
    load_fn!(lib, api, GDALSetGeoTransform);
    load_fn!(lib, api, GDALGetProjectionRef);
    load_fn!(lib, api, GDALSetProjection);
    load_fn!(lib, api, GDALGetSpatialRef);
    load_fn!(lib, api, GDALSetSpatialRef);
    load_fn!(lib, api, GDALCreateCopy);
    load_fn!(lib, api, GDALDatasetCreateLayer);

    // --- Driver ---
    load_fn!(lib, api, GDALAllRegister);
    load_fn!(lib, api, GDALGetDriverByName);
    load_fn!(lib, api, GDALCreate);

    // --- Band ---
    load_fn!(lib, api, GDALAddBand);
    load_fn!(lib, api, GDALRasterIO);
    load_fn!(lib, api, GDALRasterIOEx);
    load_fn!(lib, api, GDALGetRasterDataType);
    load_fn!(lib, api, GDALGetRasterBandXSize);
    load_fn!(lib, api, GDALGetRasterBandYSize);
    load_fn!(lib, api, GDALGetBlockSize);
    load_fn!(lib, api, GDALGetRasterNoDataValue);
    load_fn!(lib, api, GDALSetRasterNoDataValue);
    load_fn!(lib, api, GDALDeleteRasterNoDataValue);
    load_fn!(lib, api, GDALSetRasterNoDataValueAsUInt64);
    load_fn!(lib, api, GDALSetRasterNoDataValueAsInt64);

    // --- SpatialRef ---
    load_fn!(lib, api, OSRNewSpatialReference);
    load_fn!(lib, api, OSRDestroySpatialReference);
    load_fn!(lib, api, OSRExportToPROJJSON);
    load_fn!(lib, api, OSRClone);
    load_fn!(lib, api, OSRRelease);

    // --- Geometry ---
    load_fn!(lib, api, OGR_G_CreateFromWkb);
    load_fn!(lib, api, OGR_G_CreateFromWkt);
    load_fn!(lib, api, OGR_G_ExportToIsoWkb);
    load_fn!(lib, api, OGR_G_WkbSize);
    load_fn!(lib, api, OGR_G_GetEnvelope);
    load_fn!(lib, api, OGR_G_DestroyGeometry);

    // --- Vector / Layer ---
    load_fn!(lib, api, OGR_L_ResetReading);
    load_fn!(lib, api, OGR_L_GetNextFeature);
    load_fn!(lib, api, OGR_L_CreateField);
    load_fn!(lib, api, OGR_L_GetFeatureCount);
    load_fn!(lib, api, OGR_F_GetGeometryRef);
    load_fn!(lib, api, OGR_F_GetFieldIndex);
    load_fn!(lib, api, OGR_F_GetFieldAsDouble);
    load_fn!(lib, api, OGR_F_GetFieldAsInteger);
    load_fn!(lib, api, OGR_F_IsFieldSetAndNotNull);
    load_fn!(lib, api, OGR_F_Destroy);
    load_fn!(lib, api, OGR_Fld_Create);
    load_fn!(lib, api, OGR_Fld_Destroy);

    // --- VSI ---
    load_fn!(lib, api, VSIFileFromMemBuffer);
    load_fn!(lib, api, VSIFCloseL);
    load_fn!(lib, api, VSIUnlink);
    load_fn!(lib, api, VSIGetMemFileBuffer);
    load_fn!(lib, api, VSIFree);
    load_fn!(lib, api, VSIMalloc);

    // --- VRT ---
    load_fn!(lib, api, VRTCreate);
    load_fn!(lib, api, VRTAddSimpleSource);

    // --- Rasterize / Polygonize ---
    load_fn!(lib, api, GDALRasterizeGeometries);
    load_fn!(lib, api, GDALFPolygonize);
    load_fn!(lib, api, GDALPolygonize);

    // --- Version ---
    load_fn!(lib, api, GDALVersionInfo);

    // --- Config ---
    load_fn!(lib, api, CPLSetThreadLocalConfigOption);

    // --- Error ---
    load_fn!(lib, api, CPLGetLastErrorNo);
    load_fn!(lib, api, CPLGetLastErrorMsg);
    load_fn!(lib, api, CPLErrorReset);

    // --- Data Type ---
    load_fn!(lib, api, GDALGetDataTypeSizeBytes);

    // --- C++ API: MEMDataset::Create (resolved via mangled symbol names) ---
    // The symbol is mangled differently on Linux, macOS, and MSVC, and the
    // `char**` vs `const char**` parameter also affects the mangling.
    load_fn_any!(
        lib,
        api,
        MEMDatasetCreate,
        [
            // Linux and macOS
            b"_ZN10MEMDataset6CreateEPKciii12GDALDataTypePPc\0",
            // MSVC
            b"?Create@MEMDataset@@SAPEAV1@PEBDHHHW4GDALDataType@@PEAPEAD@Z\0",
        ]
    );

    Ok(())
}

/// Load a GDAL shared library from `path` and populate a [`SedonaGdalApi`] struct.
///
/// Returns the `(Library, SedonaGdalApi)` pair. The caller is responsible for
/// keeping the `Library` alive for the lifetime of the function pointers.
pub(crate) fn load_gdal_from_path(
    path: &Path,
) -> Result<(Library, SedonaGdalApi), GdalInitLibraryError> {
    let lib = unsafe { Library::new(path.as_os_str()) }.map_err(|e| {
        GdalInitLibraryError::LibraryError(format!(
            "Failed to load GDAL library from {}: {}",
            path.display(),
            e
        ))
    })?;

    let mut api = SedonaGdalApi::default();
    load_all_symbols(&lib, &mut api)?;
    Ok((lib, api))
}

/// Load GDAL symbols from the current process image (equivalent to `dlopen(NULL)`).
///
/// Returns the `(Library, SedonaGdalApi)` pair. The caller is responsible for
/// keeping the `Library` alive for the lifetime of the function pointers.
pub(crate) fn load_gdal_from_current_process(
) -> Result<(Library, SedonaGdalApi), GdalInitLibraryError> {
    let lib = current_process_library()?;
    let mut api = SedonaGdalApi::default();
    load_all_symbols(&lib, &mut api)?;
    Ok((lib, api))
}

/// Open a handle to the current process image.
#[cfg(unix)]
fn current_process_library() -> Result<Library, GdalInitLibraryError> {
    Ok(libloading::os::unix::Library::this().into())
}

#[cfg(windows)]
fn current_process_library() -> Result<Library, GdalInitLibraryError> {
    // Safety: loading symbols from the current process is safe.
    Ok(unsafe { libloading::os::windows::Library::this() }
        .map_err(|e| {
            GdalInitLibraryError::LibraryError(format!(
                "Failed to open current process handle: {}",
                e
            ))
        })?
        .into())
}

#[cfg(not(any(unix, windows)))]
fn current_process_library() -> Result<Library, GdalInitLibraryError> {
    Err(GdalInitLibraryError::Invalid(
        "current_process_library() is not implemented for this platform. \
    Only Unix and Windows are supported.",
    ))
}

#[cfg(test)]
mod test {
    use super::*;

    /// Loading from an invalid path should return a `LibraryError`.
    #[test]
    fn test_shared_library_error() {
        let err = load_gdal_from_path(Path::new("/not/a/valid/gdal/library.so")).unwrap_err();
        assert!(
            matches!(err, GdalInitLibraryError::LibraryError(_)),
            "Expected LibraryError, got: {err:?}"
        );
        assert!(
            !err.to_string().is_empty(),
            "Error message should not be empty"
        );
    }

    /// Verify that loading from the current process succeeds when GDAL symbols
    /// are linked in (i.e. the `gdal-sys` feature is enabled).
    #[cfg(feature = "gdal-sys")]
    #[test]
    fn test_load_from_current_process() {
        let (_lib, api) = load_gdal_from_current_process()
            .expect("load_gdal_from_current_process should succeed when gdal-sys is linked");

        // Spot-check that key function pointers were resolved.
        assert!(
            api.GDALAllRegister.is_some(),
            "GDALAllRegister should be loaded"
        );
        assert!(api.GDALOpenEx.is_some(), "GDALOpenEx should be loaded");
        assert!(api.GDALClose.is_some(), "GDALClose should be loaded");
    }

    /// Test loading from an explicit shared library path, gated behind an
    /// environment variable. Skips gracefully when the variable is unset.
    #[test]
    fn test_load_from_shared_library() {
        if let Ok(gdal_library) = std::env::var("SEDONA_GDAL_TEST_SHARED_LIBRARY") {
            if !gdal_library.is_empty() {
                let (_lib, api) = load_gdal_from_path(Path::new(&gdal_library))
                    .expect("Should load GDAL from SEDONA_GDAL_TEST_SHARED_LIBRARY");

                assert!(
                    api.GDALAllRegister.is_some(),
                    "GDALAllRegister should be loaded"
                );
                assert!(api.GDALOpenEx.is_some(), "GDALOpenEx should be loaded");
                return;
            }
        }

        println!(
            "Skipping test_load_from_shared_library - \
             SEDONA_GDAL_TEST_SHARED_LIBRARY environment variable not set"
        );
    }
}
