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

//! Pure Rust dynamic loader for PROJ symbols using `libloading`.
//!
//! Replaces the C implementation (`proj_dyn.c`) that used `dlopen`/`dlsym`
//! (Unix) or `LoadLibrary`/`GetProcAddress` (Windows).

use std::path::Path;

use libloading::Library;

use crate::error::SedonaProjError;
use crate::proj_dyn_bindgen::ProjApi;

/// Load a single symbol from the library and write it into the given field.
///
/// We load as a raw `*const ()` pointer and transmute to the target function pointer
/// type. This is the standard pattern for dynamic symbol loading where the loaded
/// symbol's signature is known but cannot be expressed as a generic parameter to
/// `Library::get` (because each field has a different signature).
///
/// On failure returns a `SedonaProjError` with the symbol name and the
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
                        SedonaProjError::LibraryError(format!(
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

/// Populate all 21 function-pointer fields of [`ProjApi`] from the given
/// [`Library`] handle.
fn load_all_symbols(lib: &Library, api: &mut ProjApi) -> Result<(), SedonaProjError> {
    load_fn!(lib, api, proj_area_create);
    load_fn!(lib, api, proj_area_destroy);
    load_fn!(lib, api, proj_area_set_bbox);
    load_fn!(lib, api, proj_context_create);
    load_fn!(lib, api, proj_context_destroy);
    load_fn!(lib, api, proj_context_errno_string);
    load_fn!(lib, api, proj_context_errno);
    load_fn!(lib, api, proj_context_set_database_path);
    load_fn!(lib, api, proj_context_set_search_paths);
    load_fn!(lib, api, proj_create_crs_to_crs_from_pj);
    load_fn!(lib, api, proj_create);
    load_fn!(lib, api, proj_cs_get_axis_count);
    load_fn!(lib, api, proj_destroy);
    load_fn!(lib, api, proj_errno_reset);
    load_fn!(lib, api, proj_errno);
    load_fn!(lib, api, proj_info);
    load_fn!(lib, api, proj_log_level);
    load_fn!(lib, api, proj_normalize_for_visualization);
    load_fn!(lib, api, proj_trans);
    load_fn!(lib, api, proj_trans_array);
    load_fn!(lib, api, proj_as_projjson);

    Ok(())
}

/// Load a PROJ shared library from `path` and populate a [`ProjApi`] struct.
///
/// Returns the `(Library, ProjApi)` pair. The caller is responsible for
/// keeping the `Library` alive for the lifetime of the function pointers.
pub(crate) fn load_proj_from_path(path: &Path) -> Result<(Library, ProjApi), SedonaProjError> {
    let lib = unsafe { Library::new(path.as_os_str()) }.map_err(|e| {
        SedonaProjError::LibraryError(format!(
            "Failed to load PROJ library from {}: {}",
            path.display(),
            e
        ))
    })?;

    let mut api = unsafe { std::mem::zeroed::<ProjApi>() };
    load_all_symbols(&lib, &mut api)?;
    Ok((lib, api))
}
