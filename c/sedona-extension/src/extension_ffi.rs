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
use arrow_schema::ArrowError;

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

/// Constant for the CPU device
pub const ARROW_DEVICE_CPU: i32 = 1;

/// FFI representation of the ArrowDeviceArray from the Arrow C Device Data Interface
///
/// Defined here because it is not yet defined in arrow-rs and is needed for the async
/// array stream.
///
/// Note that implementing drop() isn't needed here because Rust will drop() the array.
///
/// See: https://arrow.apache.org/docs/format/CDeviceDataInterface.html
#[repr(C)]
pub struct FFI_ArrowDeviceArray {
    pub array: FFI_ArrowArray,
    pub device_id: i64,
    pub device_type: i32,
    pub sync_event: *mut c_void,
}

impl TryFrom<FFI_ArrowDeviceArray> for FFI_ArrowArray {
    type Error = ArrowError;

    fn try_from(value: FFI_ArrowDeviceArray) -> Result<Self, Self::Error> {
        if value.device_id != 1 {
            return Err(ArrowError::CDataInterface(
                "Can't create FFI_ArrowArray from non-CPU FFI_ArrowDeviceArray".to_string(),
            ));
        }

        if !value.sync_event.is_null() {
            return Err(ArrowError::CDataInterface(
                "Can't create FFI_ArrowArray from FFI_ArrowDeviceArray with non-null sync event"
                    .to_string(),
            ));
        }

        Ok(value.array)
    }
}

impl From<FFI_ArrowArray> for FFI_ArrowDeviceArray {
    fn from(value: FFI_ArrowArray) -> Self {
        FFI_ArrowDeviceArray {
            array: value,
            device_id: -1,
            device_type: 1,
            sync_event: std::ptr::null_mut(),
        }
    }
}

/// FFI representation of the ArrowDeviceArrayStream from the Arrow C Device Data Interface
///
/// See: https://arrow.apache.org/docs/format/CDeviceDataInterface.html
#[repr(C)]
pub struct FFI_ArrowDeviceArrayStream {
    pub device_type: i32,
    pub get_schema: Option<
        unsafe extern "C" fn(
            self_: *mut FFI_ArrowDeviceArrayStream,
            out: *mut FFI_ArrowSchema,
        ) -> c_int,
    >,
    pub get_next: Option<
        unsafe extern "C" fn(
            self_: *mut FFI_ArrowDeviceArrayStream,
            out: *mut FFI_ArrowDeviceArray,
        ) -> c_int,
    >,
    pub get_last_error:
        Option<unsafe extern "C" fn(self_: *mut FFI_ArrowDeviceArrayStream) -> *const c_char>,
    pub release: Option<unsafe extern "C" fn(self_: *mut FFI_ArrowDeviceArrayStream)>,
    pub private_data: *mut c_void,
}

impl Drop for FFI_ArrowDeviceArrayStream {
    fn drop(&mut self) {
        if let Some(releaser) = self.release {
            unsafe { releaser(self) };
        }
    }
}

/// FFI representation of the ArrowAsyncDeviceStreamHandler from the Arrow C Device Data Interface
///
/// See: https://arrow.apache.org/docs/format/CDeviceDataInterface.html
#[repr(C)]
pub struct FFI_ArrowAsyncDeviceStreamHandler {
    pub on_schema: Option<
        unsafe extern "C" fn(
            self_: *mut FFI_ArrowAsyncDeviceStreamHandler,
            schema: *mut FFI_ArrowSchema,
        ) -> c_int,
    >,
    pub on_next_task: Option<
        unsafe extern "C" fn(
            self_: *mut FFI_ArrowAsyncDeviceStreamHandler,
            task: *mut FFI_ArrowAsyncTask,
            metadata: *const c_char,
        ) -> c_int,
    >,
    pub on_error: Option<
        unsafe extern "C" fn(
            self_: *mut FFI_ArrowAsyncDeviceStreamHandler,
            code: c_int,
            message: *const c_char,
            metadata: *const c_char,
        ),
    >,
    pub release: Option<unsafe extern "C" fn(self_: *mut FFI_ArrowAsyncDeviceStreamHandler)>,
    pub private_data: *mut c_void,
}

impl Drop for FFI_ArrowAsyncDeviceStreamHandler {
    fn drop(&mut self) {
        if let Some(releaser) = self.release {
            unsafe { releaser(self) };
        }
    }
}

/// FFI representation of the ArrowAsyncTask from the Arrow C Device Data Interface
///
/// See: https://arrow.apache.org/docs/format/CDeviceDataInterface.html
#[repr(C)]
pub struct FFI_ArrowAsyncTask {
    pub extract_data: Option<
        unsafe extern "C" fn(
            self_: *mut FFI_ArrowAsyncTask,
            out: *mut FFI_ArrowDeviceArray,
        ) -> c_int,
    >,
    pub private_data: *mut c_void,
}
