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
use std::ops::Deref;

use crate::errors::{GdalError, Result};
use crate::gdal_api::{call_gdal_api, GdalApi};

/// An owned GDAL-allocated VSI memory buffer.
pub struct VSIBuffer {
    api: &'static GdalApi,
    ptr: *mut u8,
    len: usize,
}

// SAFETY: `VsiBuffer` uniquely owns the GDAL-allocated buffer it wraps. Ownership may
// move across threads, and the buffer is released exactly once on drop using GDAL's
// allocator.
unsafe impl Send for VSIBuffer {}

// SAFETY: `VsiBuffer` exposes only shared read-only slice access to an immutable
// GDAL-owned byte buffer. Concurrent reads are therefore safe, and the buffer is
// still released exactly once on drop using GDAL's allocator.
unsafe impl Sync for VSIBuffer {}

impl VSIBuffer {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl AsRef<[u8]> for VSIBuffer {
    fn as_ref(&self) -> &[u8] {
        if self.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }
}

impl Deref for VSIBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl Drop for VSIBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { call_gdal_api!(self.api, VSIFree, self.ptr.cast::<std::ffi::c_void>()) };
        }
    }
}

/// Creates a new VSI in-memory file from a given buffer.
///
/// The data is copied into GDAL-allocated memory (via `VSIMalloc`) so that
/// GDAL can safely free it with `VSIFree` when ownership is taken, without
/// crossing allocator boundaries back into Rust.
pub fn create_mem_file(api: &'static GdalApi, file_name: &str, data: &[u8]) -> Result<()> {
    let c_file_name = CString::new(file_name)?;
    let len = data.len();
    let len_i64 = i64::try_from(len)?;

    let gdal_buf = if len == 0 {
        std::ptr::null_mut()
    } else {
        // Allocate via GDAL's allocator so GDAL can safely free it.
        let gdal_buf = unsafe { call_gdal_api!(api, VSIMalloc, len) } as *mut u8;
        if gdal_buf.is_null() {
            return Err(api.last_null_pointer_err("VSIMalloc"));
        }

        // Copy data into GDAL-allocated buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), gdal_buf, len);
        }
        gdal_buf
    };

    let handle = unsafe {
        call_gdal_api!(
            api,
            VSIFileFromMemBuffer,
            c_file_name.as_ptr(),
            gdal_buf,
            len_i64,
            1 // bTakeOwnership = true — GDAL will VSIFree gdal_buf
        )
    };

    if handle.is_null() {
        // GDAL did not take ownership, so we must free.
        if !gdal_buf.is_null() {
            unsafe { call_gdal_api!(api, VSIFree, gdal_buf as *mut std::ffi::c_void) };
        }
        return Err(api.last_null_pointer_err("VSIFileFromMemBuffer"));
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

/// Returns an owned GDAL-allocated buffer containing the bytes of the VSI in-memory
/// file, taking ownership and freeing the GDAL memory on drop.
pub fn get_vsi_mem_file_buffer_owned(api: &'static GdalApi, file_name: &str) -> Result<VSIBuffer> {
    let c_file_name = CString::new(file_name)?;

    let mut length: i64 = 0;
    let bytes = unsafe {
        call_gdal_api!(
            api,
            VSIGetMemFileBuffer,
            c_file_name.as_ptr(),
            &mut length,
            1 // bUnlinkAndSeize = true
        )
    };

    if length < 0 {
        if !bytes.is_null() {
            unsafe { call_gdal_api!(api, VSIFree, bytes.cast::<std::ffi::c_void>()) };
        }
        return Err(GdalError::BadArgument(format!(
            "VSIGetMemFileBuffer returned negative length: {length}"
        )));
    }

    if bytes.is_null() {
        if length == 0 {
            return Ok(VSIBuffer {
                api,
                ptr: std::ptr::null_mut(),
                len: 0,
            });
        }
        return Err(api.last_null_pointer_err("VSIGetMemFileBuffer"));
    }

    let len = usize::try_from(length)?;
    Ok(VSIBuffer {
        api,
        ptr: bytes.cast::<u8>(),
        len,
    })
}

/// Copies the bytes of the VSI in-memory file, taking ownership and freeing the GDAL memory.
pub fn get_vsi_mem_file_bytes_owned(api: &'static GdalApi, file_name: &str) -> Result<Vec<u8>> {
    let buffer = get_vsi_mem_file_buffer_owned(api, file_name)?;
    Ok(buffer.as_ref().to_vec())
}

#[cfg(all(test, feature = "gdal-sys"))]
mod tests {
    use super::*;
    use crate::global::with_global_gdal_api;

    #[test]
    fn create_and_retrieve_mem_file() {
        let file_name = "/vsimem/525ebf24-a030-4677-bb4e-a921741cabe0";

        with_global_gdal_api(|api| {
            create_mem_file(api, file_name, &[1_u8, 2, 3, 4]).unwrap();

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
    fn create_and_retrieve_mem_file_buffer() {
        let file_name = "/vsimem/2e3c48a5-d2ef-4f5c-896d-5467cdca9406";

        with_global_gdal_api(|api| {
            create_mem_file(api, file_name, &[1_u8, 2, 3, 4]).unwrap();

            let buffer = get_vsi_mem_file_buffer_owned(api, file_name).unwrap();

            assert_eq!(buffer.len(), 4);
            assert_eq!(buffer.as_ref(), &[1_u8, 2, 3, 4]);
        })
        .unwrap();
    }

    #[test]
    fn create_and_unlink_mem_file() {
        let file_name = "/vsimem/bbf5f1d6-c1e9-4469-a33b-02cd9173132d";

        with_global_gdal_api(|api| {
            create_mem_file(api, file_name, &[1_u8, 2, 3, 4]).unwrap();

            unlink_mem_file(api, file_name).unwrap();
        })
        .unwrap();
    }

    #[test]
    fn create_and_retrieve_empty_mem_file() {
        let file_name = "/vsimem/3f9e6282-313d-4c51-81ab-f020ff2134d8";

        with_global_gdal_api(|api| {
            create_mem_file(api, file_name, &[]).unwrap();

            let bytes = get_vsi_mem_file_bytes_owned(api, file_name).unwrap();

            assert!(bytes.is_empty());
        })
        .unwrap();
    }

    #[test]
    fn create_and_retrieve_empty_mem_file_buffer() {
        let file_name = "/vsimem/17319db4-775b-4380-a9af-802c160dcb24";

        with_global_gdal_api(|api| {
            create_mem_file(api, file_name, &[]).unwrap();

            let buffer = get_vsi_mem_file_buffer_owned(api, file_name).unwrap();

            assert!(buffer.is_empty());
            assert_eq!(buffer.as_ref(), b"");
        })
        .unwrap();
    }

    #[test]
    fn no_mem_file() {
        with_global_gdal_api(|api| {
            let bytes = get_vsi_mem_file_bytes_owned(api, "foobar").unwrap();
            assert!(bytes.is_empty());
        })
        .unwrap();
    }
}
