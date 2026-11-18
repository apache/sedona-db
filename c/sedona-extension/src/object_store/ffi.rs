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

//! Version 1 C ABI. Keep these declarations in sync with sedona_extension.h.
use crate::extension::{FFI_ArrowAsyncDeviceStreamHandler, SedonaCError};
use std::ffi::{c_int, c_void};

pub const OBJECT_STORE_ABI_VERSION: u64 = 1;
pub const OBJECT_STORE_READ: u64 = 1;
pub const OBJECT_STORE_LIST: u64 = 2;
pub const OBJECT_STORE_LIST_WITH_DELIMITER: u64 = 4;
/// Maximum wire payload bytes in a batch, request, or metadata response.
pub const OBJECT_STORE_MAX_BYTES: usize = 1024 * 1024;
/// Maximum number of listing entries in one batch.
pub const OBJECT_STORE_MAX_ROWS: usize = 128;

/// Borrowed for the duration of `call` only. Providers must not retain this pointer.
#[repr(C)]
pub struct SedonaCObjectStoreCall {
    /// Remaining operation budget; includes provider retries.
    pub timeout_ms: u64,
    /// Thread-safe, nonblocking callback. A nonzero value requests cancellation.
    pub is_cancelled: Option<unsafe extern "C" fn(*const c_void) -> c_int>,
    pub private_data: *const c_void,
}

/// Owned bytes. Release with the allocating library's callback, exactly once.
#[derive(Default)]
#[repr(C)]
pub struct SedonaCObjectStoreBuffer {
    pub data: *const u8,
    pub len: usize,
    pub release: Option<unsafe extern "C" fn(*mut SedonaCObjectStoreBuffer)>,
    pub private_data: *mut c_void,
}

/// A single-use stream. `start` accepts the handler only on success (return 0).
/// The caller releases this handle after start returns, including on error.
#[derive(Default)]
#[repr(C)]
pub struct SedonaCObjectStoreStream {
    pub start: Option<
        unsafe extern "C" fn(
            *mut SedonaCObjectStoreStream,
            *mut FFI_ArrowAsyncDeviceStreamHandler,
            *mut SedonaCError,
        ) -> c_int,
    >,
    pub release: Option<unsafe extern "C" fn(*mut SedonaCObjectStoreStream)>,
    pub private_data: *mut c_void,
}

/// Initialize to zero before a call. Both fields are owned even on failure.
#[derive(Default)]
#[repr(C)]
pub struct SedonaCObjectStoreResponse {
    pub metadata: SedonaCObjectStoreBuffer,
    pub stream: SedonaCObjectStoreStream,
}

/// Thread-safe read-only store. Requests and responses use the documented v1 JSON
/// schemas. A nonzero result is a stable storage error code, not a platform errno.
#[derive(Default)]
#[repr(C)]
pub struct SedonaCObjectStore {
    pub abi_version: u64,
    pub capabilities: u64,
    pub call: Option<
        unsafe extern "C" fn(
            *const SedonaCObjectStore,
            *const SedonaCObjectStoreCall,
            *const u8,
            usize,
            *mut SedonaCObjectStoreResponse,
            *mut SedonaCError,
        ) -> c_int,
    >,
    pub release: Option<unsafe extern "C" fn(*mut SedonaCObjectStore)>,
    pub private_data: *mut c_void,
    pub reserved: [u64; 4],
}

// SAFETY: Accepted stores support concurrent const-self calls and serialized
// ownership transfer/release on arbitrary threads, as required by the ABI.
unsafe impl Send for SedonaCObjectStore {}
unsafe impl Sync for SedonaCObjectStore {}
// SAFETY: These owned results may move between threads; callbacks are serialized.
unsafe impl Send for SedonaCObjectStoreBuffer {}
unsafe impl Send for SedonaCObjectStoreStream {}

impl Drop for SedonaCObjectStoreBuffer {
    fn drop(&mut self) {
        if let Some(release) = self.release {
            unsafe { release(self) };
            self.release = None;
        }
    }
}
impl Drop for SedonaCObjectStoreStream {
    fn drop(&mut self) {
        if let Some(release) = self.release {
            unsafe { release(self) };
            self.release = None;
        }
    }
}
// No Drop on the raw store: the importer schedules its release on the foreign
// executor. C callers must call release explicitly.
