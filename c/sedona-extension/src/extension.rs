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

use std::{
    ffi::c_int,
    os::raw::{c_char, c_void},
    ptr::null_mut,
};

use arrow_array::ffi::{FFI_ArrowArray, FFI_ArrowSchema};

// -----------------------------------------------------------------------------
// Async Result Handler
// -----------------------------------------------------------------------------

/// Raw FFI representation of an async result handler.
///
/// This follows the Arrow Async Stream interface pattern where the consumer
/// provides callback handlers that the producer calls when the async operation
/// completes. This enables high-concurrency scenarios without blocking threads.
///
/// The producer MUST call exactly one of `on_success` or `on_error`, followed
/// by `release`. The handler is invalid after `release` is called.
///
/// Similar to Arrow's `ArrowAsyncDeviceStreamHandler`, but for single operations
/// rather than streams.
#[derive(Default)]
#[repr(C)]
pub struct SedonaCAsyncResultHandler {
    /// Called on successful completion with optional payload data.
    ///
    /// `data` and `data_len` provide an optional byte payload. If no payload,
    /// `data` is NULL and `data_len` is 0. The data is only valid for the
    /// duration of this callback - the consumer must copy if needed longer.
    ///
    /// After this callback returns, the producer MUST call `release`.
    pub on_success: Option<
        unsafe extern "C" fn(
            self_: *mut SedonaCAsyncResultHandler,
            data: *const u8,
            data_len: usize,
        ),
    >,

    /// Called on error with an error code and message.
    ///
    /// `code` is an `errno`-compatible error code.
    /// `message` is a null-terminated error string, valid only for this callback.
    ///
    /// After this callback returns, the producer MUST call `release`.
    pub on_error: Option<
        unsafe extern "C" fn(
            self_: *mut SedonaCAsyncResultHandler,
            code: c_int,
            message: *const c_char,
        ),
    >,

    /// Release callback to clean up the handler's resources.
    ///
    /// The producer MUST call this after calling either `on_success` or `on_error`.
    /// After this is called, the handler is invalid and must not be used.
    pub release: Option<unsafe extern "C" fn(self_: *mut SedonaCAsyncResultHandler)>,

    /// Opaque consumer-specific data
    pub private_data: *mut c_void,
}

unsafe impl Send for SedonaCAsyncResultHandler {}

impl Drop for SedonaCAsyncResultHandler {
    fn drop(&mut self) {
        if let Some(releaser) = self.release {
            unsafe { releaser(self) }
            self.release = None;
            self.private_data = null_mut();
        }
    }
}

// -----------------------------------------------------------------------------
// Object Store
// -----------------------------------------------------------------------------

/// Raw FFI representation of an ObjectStore
///
/// This follows the Arrow C Data Interface pattern with function pointers
/// for callbacks and a `release` callback for cleanup.
///
/// See the ImportedObjectStore and ExportedObjectStore for high-level
/// APIs to import and export implementations using this struct.
#[derive(Default)]
#[repr(C)]
pub struct SedonaCObjectStore {
    /// Callback to get a display string for the object store.
    ///
    /// The returned pointer must remain valid until the next call to any
    /// method on this struct or until `release` is called.
    ///
    /// Return value: pointer to a null-terminated character array.
    pub display: Option<unsafe extern "C" fn(self_: *const SedonaCObjectStore) -> *const c_char>,

    /// Callback to get a debug string for the object store.
    ///
    /// The returned pointer must remain valid until the next call to any
    /// method on this struct or until `release` is called.
    ///
    /// Return value: pointer to a null-terminated character array.
    pub debug: Option<unsafe extern "C" fn(self_: *const SedonaCObjectStore) -> *const c_char>,

    /// Delete the object at the specified location.
    ///
    /// This is an async operation. The implementation MUST eventually call
    /// either `on_success` or `on_error` on the handler, followed by `release`.
    ///
    /// `location` is a null-terminated path string that remains valid for the
    /// duration of this call only.
    ///
    /// Return value: 0 if the operation was successfully initiated,
    /// `errno`-compatible error code if the operation could not be started.
    /// If non-zero is returned, the handler callbacks must NOT be called.
    pub delete: Option<
        unsafe extern "C" fn(
            self_: *const SedonaCObjectStore,
            location: *const c_char,
            handler: *mut SedonaCAsyncResultHandler,
        ) -> c_int,
    >,

    /// Copy an object from one location to another.
    ///
    /// This is an async operation. The implementation MUST eventually call
    /// either `on_success` or `on_error` on the handler, followed by `release`.
    ///
    /// `from` and `to` are null-terminated path strings that remain valid for
    /// the duration of this call only.
    ///
    /// Return value: 0 if the operation was successfully initiated,
    /// `errno`-compatible error code if the operation could not be started.
    /// If non-zero is returned, the handler callbacks must NOT be called.
    pub copy: Option<
        unsafe extern "C" fn(
            self_: *const SedonaCObjectStore,
            from: *const c_char,
            to: *const c_char,
            handler: *mut SedonaCAsyncResultHandler,
        ) -> c_int,
    >,

    /// Copy an object from one location to another, only if the destination
    /// does not already exist.
    ///
    /// This is an async operation. The implementation MUST eventually call
    /// either `on_success` or `on_error` on the handler, followed by `release`.
    ///
    /// `from` and `to` are null-terminated path strings that remain valid for
    /// the duration of this call only.
    ///
    /// Return value: 0 if the operation was successfully initiated,
    /// `errno`-compatible error code if the operation could not be started.
    /// If non-zero is returned, the handler callbacks must NOT be called.
    pub copy_if_not_exists: Option<
        unsafe extern "C" fn(
            self_: *const SedonaCObjectStore,
            from: *const c_char,
            to: *const c_char,
            handler: *mut SedonaCAsyncResultHandler,
        ) -> c_int,
    >,

    /// Release callback: release the object store's own resources.
    pub release: Option<unsafe extern "C" fn(self_: *mut SedonaCObjectStore)>,

    /// Opaque producer-specific data
    pub private_data: *mut c_void,
}

unsafe impl Send for SedonaCObjectStore {}
unsafe impl Sync for SedonaCObjectStore {}

impl Drop for SedonaCObjectStore {
    fn drop(&mut self) {
        if let Some(releaser) = self.release {
            unsafe { releaser(self) }
            self.release = None;
            self.private_data = null_mut();
        }
    }
}

/// Raw FFI representation of the SedonaCScalarKernel
///
/// See the ImportedScalarKernel and ExportedScalarKernel for high-level
/// APIs to import and export implementations using this struct.
#[derive(Default)]
#[repr(C)]
pub struct SedonaCScalarKernel {
    pub function_name:
        Option<unsafe extern "C" fn(self_: *const SedonaCScalarKernel) -> *const c_char>,
    pub new_impl: Option<
        unsafe extern "C" fn(self_: *const SedonaCScalarKernel, out: *mut SedonaCScalarKernelImpl),
    >,

    pub release: Option<unsafe extern "C" fn(self_: *mut SedonaCScalarKernel)>,
    pub private_data: *mut c_void,
}

unsafe impl Send for SedonaCScalarKernel {}
unsafe impl Sync for SedonaCScalarKernel {}

impl Drop for SedonaCScalarKernel {
    fn drop(&mut self) {
        if let Some(releaser) = self.release {
            unsafe { releaser(self) }
            self.release = None;
            self.private_data = null_mut();
        }
    }
}

/// Raw FFI representation of the SedonaCScalarKernelImpl
#[derive(Default)]
#[repr(C)]
pub struct SedonaCScalarKernelImpl {
    pub init: Option<
        unsafe extern "C" fn(
            self_: *mut SedonaCScalarKernelImpl,
            arg_types: *const *const FFI_ArrowSchema,
            scalar_args: *const *mut FFI_ArrowArray,
            n_args: i64,
            out: *mut FFI_ArrowSchema,
        ) -> c_int,
    >,

    pub execute: Option<
        unsafe extern "C" fn(
            self_: *mut SedonaCScalarKernelImpl,
            args: *const *mut FFI_ArrowArray,
            n_args: i64,
            n_rows: i64,
            out: *mut FFI_ArrowArray,
        ) -> c_int,
    >,

    pub get_last_error:
        Option<unsafe extern "C" fn(self_: *mut SedonaCScalarKernelImpl) -> *const c_char>,

    pub release: Option<unsafe extern "C" fn(self_: *mut SedonaCScalarKernelImpl)>,

    pub private_data: *mut c_void,
}

impl Drop for SedonaCScalarKernelImpl {
    fn drop(&mut self) {
        if let Some(releaser) = self.release {
            unsafe { releaser(self) }
            self.release = None;
            self.private_data = null_mut();
        }
    }
}

/// Check if a schema is valid
///
/// The [FFI_ArrowSchema] doesn't have the ability to check for a NULL release callback,
/// so we provide a mechanism to do so here.
pub fn ffi_arrow_schema_is_valid(schema: *const FFI_ArrowSchema) -> bool {
    let schema_internal = schema as *const c_void as *const ArrowSchemaInternal;
    if let Some(schema_ref) = unsafe { schema_internal.as_ref() } {
        schema_ref.release.is_some()
    } else {
        false
    }
}

#[repr(C)]
struct ArrowSchemaInternal {
    format: *const c_char,
    name: *const c_char,
    metadata: *const c_char,
    flags: i64,
    n_children: i64,
    children: *mut *mut ArrowSchemaInternal,
    dictionary: *mut ArrowSchemaInternal,
    release: Option<unsafe extern "C" fn(*mut ArrowSchemaInternal)>,
    private_data: *mut c_void,
}
