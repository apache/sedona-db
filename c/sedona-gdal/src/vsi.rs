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
//! <https://github.com/georust/gdal/blob/v0.19.0/src/vsi.rs>.
//! Original code is licensed under MIT.
//!
//! GDAL Virtual File System (VSI) wrappers.

use std::ffi::{c_char, CStr, CString};

use crate::call_gdal_api;
use crate::cpl::CslStringList;
use crate::errors::{GdalError, Result};
use crate::gdal_api::GdalApi;
use crate::gdal_dyn_bindgen::{vsi_l_offset, GIntBig, VSIDIREntry, VSIDIR};

#[derive(Debug, Clone)]
/// Owned snapshot of one entry returned by GDAL VSI directory iteration.
pub struct VsiDirEntry {
    pub name: String,
    pub mode: Option<i32>,
    pub size: Option<vsi_l_offset>,
    pub mtime: Option<GIntBig>,
}

/// RAII wrapper around a GDAL `VSIDIR` handle.
pub struct VsiDir {
    api: &'static GdalApi,
    handle: *mut VSIDIR,
}

impl VsiDir {
    /// Returns the next directory entry, or `None` at end-of-stream.
    pub fn next_entry(&mut self) -> Option<VsiDirEntry> {
        let entry = unsafe { call_gdal_api!(self.api, VSIGetNextDirEntry, self.handle) };
        if entry.is_null() {
            return None;
        }

        let entry = unsafe { &*entry };
        Some(clone_dir_entry(entry))
    }
}

impl Iterator for VsiDir {
    type Item = VsiDirEntry;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_entry()
    }
}

impl Drop for VsiDir {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { call_gdal_api!(self.api, VSICloseDir, self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

/// Copies a GDAL-owned `VSIDIREntry` into an owned Rust value.
///
/// GDAL may reuse or invalidate entry pointers across iterator steps, so callers
/// must not retain borrowed references to fields on the original C struct.
fn clone_dir_entry(entry: &VSIDIREntry) -> VsiDirEntry {
    let name = if entry.pszName.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(entry.pszName) }
            .to_string_lossy()
            .into_owned()
    };

    VsiDirEntry {
        name,
        mode: (entry.bModeKnown != 0).then_some(entry.nMode),
        size: (entry.bSizeKnown != 0).then_some(entry.nSize),
        mtime: (entry.bMTimeKnown != 0).then_some(entry.nMTime),
    }
}

/// Opens a VSI directory handle for iteration.
///
/// `recurse_depth` follows GDAL semantics (`0` for direct children, `-1` for
/// recursive traversal). Optional `CslStringList` entries are forwarded as
/// `VSIOpenDir` options.
pub fn open_dir(
    api: &'static GdalApi,
    path: &str,
    recurse_depth: i32,
    options: Option<&CslStringList>,
) -> Result<VsiDir> {
    let c_path = CString::new(path)?;
    let options_ptr: *const *const c_char = options
        .map(|opts| opts.as_ptr() as *const *const c_char)
        .unwrap_or(std::ptr::null());
    let handle =
        unsafe { call_gdal_api!(api, VSIOpenDir, c_path.as_ptr(), recurse_depth, options_ptr) };
    if handle.is_null() {
        return Err(api.last_null_pointer_err("VSIOpenDir"));
    }
    Ok(VsiDir { api, handle })
}

/// Returns the directory separator GDAL expects for the given VSI path.
pub fn get_directory_separator(api: &'static GdalApi, path: &str) -> Result<String> {
    let c_path = CString::new(path)?;
    let separator_ptr = unsafe { call_gdal_api!(api, VSIGetDirectorySeparator, c_path.as_ptr()) };
    if separator_ptr.is_null() {
        return Err(api.last_null_pointer_err("VSIGetDirectorySeparator"));
    }
    Ok(unsafe { CStr::from_ptr(separator_ptr) }
        .to_string_lossy()
        .into_owned())
}

/// Creates a new VSI in-memory file from a given buffer.
///
/// The data is copied into GDAL-allocated memory (via `VSIMalloc`) so that
/// GDAL can safely free it with `VSIFree` when ownership is taken.
pub fn create_mem_file(api: &'static GdalApi, file_name: &str, data: Vec<u8>) -> Result<()> {
    let c_file_name = CString::new(file_name)?;
    let len = data.len();

    // Allocate via GDAL's allocator so GDAL can safely free it.
    let gdal_buf = unsafe { call_gdal_api!(api, VSIMalloc, len) } as *mut u8;
    if gdal_buf.is_null() {
        return Err(GdalError::NullPointer {
            method_name: "VSIMalloc",
            msg: format!("failed to allocate {len} bytes"),
        });
    }

    // Copy data into GDAL-allocated buffer
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), gdal_buf, len);
    }
    // Rust Vec is dropped here, freeing the Rust-allocated memory.

    let handle = unsafe {
        call_gdal_api!(
            api,
            VSIFileFromMemBuffer,
            c_file_name.as_ptr(),
            gdal_buf,
            len as i64,
            1 // bTakeOwnership = true — GDAL will VSIFree gdal_buf
        )
    };

    if handle.is_null() {
        // GDAL did not take ownership, so we must free.
        unsafe { call_gdal_api!(api, VSIFree, gdal_buf as *mut std::ffi::c_void) };
        return Err(GdalError::NullPointer {
            method_name: "VSIFileFromMemBuffer",
            msg: String::new(),
        });
    }

    unsafe {
        call_gdal_api!(api, VSIFCloseL, handle);
    }

    Ok(())
}

/// Unlink (delete) a VSI in-memory file.
pub fn unlink_mem_file(api: &'static GdalApi, file_name: &str) -> Result<()> {
    let c_file_name = CString::new(file_name)?;

    let rv = unsafe { call_gdal_api!(api, VSIUnlink, c_file_name.as_ptr()) };

    if rv != 0 {
        return Err(GdalError::UnlinkMemFile {
            file_name: file_name.to_string(),
        });
    }

    Ok(())
}

/// Copies the bytes of the VSI in-memory file, taking ownership and freeing the GDAL memory.
pub fn get_vsi_mem_file_bytes_owned(api: &'static GdalApi, file_name: &str) -> Result<Vec<u8>> {
    let c_file_name = CString::new(file_name)?;

    let owned_bytes = unsafe {
        let mut length: i64 = 0;
        let bytes = call_gdal_api!(
            api,
            VSIGetMemFileBuffer,
            c_file_name.as_ptr(),
            &mut length,
            1 // bUnlinkAndSeize = true
        );

        if bytes.is_null() {
            return Err(GdalError::NullPointer {
                method_name: "VSIGetMemFileBuffer",
                msg: String::new(),
            });
        }

        if length < 0 {
            call_gdal_api!(api, VSIFree, bytes.cast::<std::ffi::c_void>());
            return Err(GdalError::BadArgument(format!(
                "VSIGetMemFileBuffer returned negative length: {length}"
            )));
        }

        let slice = std::slice::from_raw_parts(bytes, length as usize);
        let vec = slice.to_vec();

        call_gdal_api!(api, VSIFree, bytes.cast::<std::ffi::c_void>());

        vec
    };

    Ok(owned_bytes)
}

#[cfg(all(test, feature = "gdal-sys"))]
mod tests {
    use super::*;
    use crate::global::with_global_gdal_api;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn create_temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("{prefix}_{}_{}", std::process::id(), nanos));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn create_and_retrieve_mem_file() {
        let file_name = "/vsimem/525ebf24-a030-4677-bb4e-a921741cabe0";

        with_global_gdal_api(|api| {
            create_mem_file(api, file_name, vec![1_u8, 2, 3, 4]).unwrap();

            let bytes = get_vsi_mem_file_bytes_owned(api, file_name).unwrap();

            assert_eq!(bytes, vec![1_u8, 2, 3, 4]);

            // mem file must not be there anymore
            assert!(matches!(
                unlink_mem_file(api, file_name).unwrap_err(),
                GdalError::UnlinkMemFile {
                    file_name: err_file_name
                }
                if err_file_name == file_name
            ));
        })
        .unwrap();
    }

    #[test]
    fn create_and_unlink_mem_file() {
        let file_name = "/vsimem/bbf5f1d6-c1e9-4469-a33b-02cd9173132d";

        with_global_gdal_api(|api| {
            create_mem_file(api, file_name, vec![1_u8, 2, 3, 4]).unwrap();

            unlink_mem_file(api, file_name).unwrap();
        })
        .unwrap();
    }

    #[test]
    fn no_mem_file() {
        with_global_gdal_api(|api| {
            assert!(matches!(
                get_vsi_mem_file_bytes_owned(api, "foobar").unwrap_err(),
                GdalError::NullPointer {
                    method_name: "VSIGetMemFileBuffer",
                    msg,
                }
                if msg.is_empty()
            ));
        })
        .unwrap();
    }

    #[test]
    fn open_dir_local_recursive_and_non_recursive() {
        let base = create_temp_dir("sedona_gdal_vsi_read_dir");
        fs::write(base.join("File 1.txt"), b"x").unwrap();
        fs::write(base.join("File 2.txt"), b"x").unwrap();
        fs::create_dir(base.join("folder")).unwrap();
        fs::write(base.join("folder").join("File 3.txt"), b"x").unwrap();
        let path = base.to_string_lossy().to_string();

        with_global_gdal_api(|api| {
            let mut non_recursive = open_dir(api, &path, 0, None).unwrap();
            let mut non_recursive_names = Vec::new();
            for entry in &mut non_recursive {
                non_recursive_names.push(entry.name);
            }
            non_recursive_names.sort();
            assert_eq!(
                non_recursive_names,
                vec!["File 1.txt", "File 2.txt", "folder"]
            );

            let mut options = CslStringList::new();
            options.set_name_value("NAME_AND_TYPE_ONLY", "YES").unwrap();
            let mut recursive = open_dir(api, &path, -1, Some(&options)).unwrap();
            let mut recursive_names = Vec::new();
            for entry in &mut recursive {
                recursive_names.push(entry.name);
            }
            recursive_names.sort();
            assert_eq!(
                recursive_names,
                vec!["File 1.txt", "File 2.txt", "folder", "folder/File 3.txt"]
            );
        })
        .unwrap();

        fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn get_directory_separator_returns_non_empty() {
        with_global_gdal_api(|api| {
            let separator = get_directory_separator(api, "/vsimem/").unwrap();
            assert!(!separator.is_empty());
        })
        .unwrap();
    }
}
