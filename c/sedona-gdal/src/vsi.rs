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

use std::ffi::CString;

use crate::call_gdal_api;
use crate::errors::{GdalError, Result};
use crate::gdal_api::GdalApi;

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
    use crate::global::get_global_gdal_api;

    #[test]
    fn create_and_retrieve_mem_file() {
        let file_name = "/vsimem/525ebf24-a030-4677-bb4e-a921741cabe0";

        let api = get_global_gdal_api().unwrap();
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
    }

    #[test]
    fn create_and_unlink_mem_file() {
        let file_name = "/vsimem/bbf5f1d6-c1e9-4469-a33b-02cd9173132d";

        let api = get_global_gdal_api().unwrap();
        create_mem_file(api, file_name, vec![1_u8, 2, 3, 4]).unwrap();

        unlink_mem_file(api, file_name).unwrap();
    }

    #[test]
    fn no_mem_file() {
        let api = get_global_gdal_api().unwrap();
        assert!(matches!(
            get_vsi_mem_file_bytes_owned(api, "foobar").unwrap_err(),
            GdalError::NullPointer {
                method_name: "VSIGetMemFileBuffer",
                msg,
            }
            if msg.is_empty()
        ));
    }
}
