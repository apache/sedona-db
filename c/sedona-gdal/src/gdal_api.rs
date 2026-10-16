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
use crate::gdal_dyn_bindgen::{GDALTransformerFunc, SedonaGdalApi};

#[cfg(feature = "gdal-sys")]
unsafe extern "C" {
    #[link_name = "VRTCreate"]
    fn gdal_sys_vrt_create(x_size: i32, y_size: i32) -> *mut std::ffi::c_void;
    #[link_name = "VRTAddSimpleSource"]
    fn gdal_sys_vrt_add_simple_source(
        vrt_band: *mut std::ffi::c_void,
        source_band: *mut std::ffi::c_void,
        src_x_off: i32,
        src_y_off: i32,
        src_x_size: i32,
        src_y_size: i32,
        dst_x_off: i32,
        dst_y_off: i32,
        dst_x_size: i32,
        dst_y_size: i32,
        resampling: *const std::ffi::c_char,
        nodata: f64,
    ) -> i32;
}

// GDAL 3.12 and earlier predate the public MEMCreate C API. Unix and MinGW use
// the Itanium C++ ABI, so referencing the private constructor preserves the
// fast path and makes static linkers retain its object from libgdal.a.
#[cfg(all(
    feature = "gdal-sys",
    not(gdal_has_mem_create),
    any(unix, all(windows, target_env = "gnu"))
))]
unsafe extern "C" {
    #[link_name = "_ZN10MEMDataset6CreateEPKciii12GDALDataTypePPc"]
    fn gdal_sys_mem_dataset_create(
        filename: *const std::ffi::c_char,
        x_size: i32,
        y_size: i32,
        bands: i32,
        data_type: crate::gdal_dyn_bindgen::GDALDataType,
        options: *mut *mut std::ffi::c_char,
    ) -> *mut std::ffi::c_void;
}

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
    /// The dynamically loaded GDAL library, when applicable. Kept alive for the
    /// lifetime of the function pointers in `inner`. Statically linked API
    /// tables do not need a library handle.
    _lib: Option<Library>,
    name: String,
}

impl GdalApi {
    pub fn try_from_shared_library(shared_library: PathBuf) -> Result<Self, GdalInitLibraryError> {
        let (lib, inner) = dyn_load::load_gdal_from_path(&shared_library)?;
        Ok(Self {
            inner,
            _lib: Some(lib),
            name: shared_library.to_string_lossy().into_owned(),
        })
    }

    pub fn try_from_current_process() -> Result<Self, GdalInitLibraryError> {
        let (lib, inner) = dyn_load::load_gdal_from_current_process()?;
        Ok(Self {
            inner,
            _lib: Some(lib),
            name: "current_process".to_string(),
        })
    }

    /// Build the API table directly from functions linked by `gdal-sys`.
    ///
    /// Linked GDAL functions are not guaranteed to be exported from the final
    /// executable or DLL for runtime symbol lookup, so retain their addresses
    /// directly when the platform has a supported MEM creator ABI.
    #[cfg(all(feature = "gdal-sys", any(unix, all(windows, target_env = "gnu"))))]
    pub fn from_gdal_sys() -> Self {
        let mut inner = SedonaGdalApi::default();

        macro_rules! set_fn {
            ($name:ident) => {
                // The generated gdal-sys bindings model C enums as integers,
                // while Sedona's dynamic table uses repr(C) enums. Their ABIs
                // are identical, but their Rust function-pointer types differ.
                #[allow(clippy::missing_transmute_annotations)]
                {
                    inner.$name =
                        Some(unsafe { std::mem::transmute(gdal_sys::$name as *const ()) });
                }
            };
        }

        set_fn!(GDALOpenEx);
        set_fn!(GDALClose);
        set_fn!(GDALGetRasterXSize);
        set_fn!(GDALGetRasterYSize);
        set_fn!(GDALGetRasterCount);
        set_fn!(GDALGetRasterBand);
        set_fn!(GDALGetGeoTransform);
        set_fn!(GDALSetGeoTransform);
        set_fn!(GDALGetProjectionRef);
        set_fn!(GDALSetProjection);
        set_fn!(GDALGetSpatialRef);
        set_fn!(GDALSetSpatialRef);
        set_fn!(GDALCreateCopy);
        set_fn!(GDALDatasetCreateLayer);
        set_fn!(GDALAllRegister);
        set_fn!(GDALGetDriverByName);
        set_fn!(GDALCreate);
        set_fn!(GDALAddBand);
        set_fn!(GDALRasterIO);
        set_fn!(GDALRasterIOEx);
        set_fn!(GDALGetRasterDataType);
        set_fn!(GDALGetRasterBandXSize);
        set_fn!(GDALGetRasterBandYSize);
        set_fn!(GDALGetBlockSize);
        set_fn!(GDALGetRasterNoDataValue);
        set_fn!(GDALGetRasterNoDataValueAsUInt64);
        set_fn!(GDALGetRasterNoDataValueAsInt64);
        set_fn!(GDALSetRasterNoDataValue);
        set_fn!(GDALDeleteRasterNoDataValue);
        set_fn!(GDALSetRasterNoDataValueAsUInt64);
        set_fn!(GDALSetRasterNoDataValueAsInt64);
        set_fn!(OSRNewSpatialReference);
        set_fn!(OSRSetFromUserInput);
        set_fn!(OSREPSGTreatsAsLatLong);
        set_fn!(OSRGetDataAxisToSRSAxisMapping);
        set_fn!(OSRGetAxisMappingStrategy);
        set_fn!(OSRSetAxisMappingStrategy);
        set_fn!(OSRDestroySpatialReference);
        set_fn!(OSRExportToPROJJSON);
        set_fn!(OSRExportToWkt);
        set_fn!(OSRClone);
        set_fn!(OSRRelease);
        set_fn!(OGR_G_CreateFromWkb);
        set_fn!(OGR_G_CreateFromWkt);
        set_fn!(OGR_G_ExportToIsoWkb);
        set_fn!(OGR_G_WkbSize);
        set_fn!(OGR_G_GetEnvelope);
        set_fn!(OGR_G_DestroyGeometry);
        set_fn!(OGR_L_ResetReading);
        set_fn!(OGR_L_GetNextFeature);
        set_fn!(OGR_L_CreateField);
        set_fn!(OGR_L_GetFeatureCount);
        set_fn!(OGR_F_GetGeometryRef);
        set_fn!(OGR_F_GetFieldIndex);
        set_fn!(OGR_F_GetFieldAsDouble);
        set_fn!(OGR_F_GetFieldAsInteger);
        set_fn!(OGR_F_IsFieldSetAndNotNull);
        set_fn!(OGR_F_Destroy);
        set_fn!(OGR_Fld_Create);
        set_fn!(OGR_Fld_Destroy);
        set_fn!(VSIFileFromMemBuffer);
        set_fn!(VSIFCloseL);
        set_fn!(VSIUnlink);
        set_fn!(VSIGetMemFileBuffer);
        set_fn!(VSIFree);
        set_fn!(VSIMalloc);
        inner.VRTCreate = Some(gdal_sys_vrt_create);
        inner.VRTAddSimpleSource = Some(gdal_sys_vrt_add_simple_source);
        set_fn!(GDALReprojectImage);
        set_fn!(GDALCreateGenImgProjTransformer);
        set_fn!(GDALDestroyGenImgProjTransformer);
        set_fn!(GDALSuggestedWarpOutput);
        set_fn!(GDALGenImgProjTransform);
        set_fn!(GDALRasterizeGeometries);
        set_fn!(GDALFPolygonize);
        set_fn!(GDALPolygonize);
        set_fn!(GDALVersionInfo);
        set_fn!(CPLSetThreadLocalConfigOption);
        set_fn!(CPLGetLastErrorNo);
        set_fn!(CPLGetLastErrorMsg);
        set_fn!(CPLErrorReset);
        set_fn!(GDALGetDataTypeSizeBytes);

        #[cfg(gdal_has_mem_create)]
        set_fn!(MEMCreate);

        #[cfg(not(gdal_has_mem_create))]
        {
            inner.MEMDatasetCreate = Some(gdal_sys_mem_dataset_create);
        }

        Self {
            inner,
            _lib: None,
            name: "gdal_sys".to_string(),
        }
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

    /// Return the `GDALGenImgProjTransform` callback, passed as the transformer
    /// to `GDALSuggestedWarpOutput`. Errors if the symbol did not resolve.
    pub(crate) fn gen_img_proj_transform_fn(&self) -> Result<GDALTransformerFunc, GdalError> {
        self.inner.GDALGenImgProjTransform.ok_or_else(|| {
            GdalError::BadArgument("GDALGenImgProjTransform symbol is not available".to_string())
        })
    }

    /// Check the last CPL error and return a `GdalError::CplError`, it always returns an error struct
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

    /// Check the last CPL error and return a `GdalError::NullPointer`, it always returns an error struct
    pub fn last_null_pointer_err(&self, method_name: &'static str) -> GdalError {
        let err_msg = unsafe {
            let msg_ptr = call_gdal_api!(self, CPLGetLastErrorMsg);
            if msg_ptr.is_null() {
                String::new()
            } else {
                CStr::from_ptr(msg_ptr).to_string_lossy().into_owned()
            }
        };
        unsafe { call_gdal_api!(self, CPLErrorReset) };
        GdalError::NullPointer {
            method_name,
            msg: err_msg,
        }
    }
}
