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

use arrow_array::{
    ffi::{FFI_ArrowArray, FFI_ArrowSchema},
    ffi_stream::FFI_ArrowArrayStream,
};

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

/// Raw FFI representation of the SedonaCError
#[derive(Default)]
#[repr(C)]
pub struct SedonaCError {
    pub err: *const c_char,
    pub err_len: u32,
    pub reserved: u32,
    pub release: Option<unsafe extern "C" fn(self_: *mut SedonaCError)>,
}

impl SedonaCError {
    pub fn new(message: &str) -> Self {
        use std::ffi::CString;

        match CString::new(message) {
            Ok(c_string) => {
                let len = message.len() as u32;
                let ptr = c_string.into_raw();

                extern "C" fn release_error(self_: *mut SedonaCError) {
                    unsafe {
                        if !(*self_).err.is_null() {
                            let _ = std::ffi::CString::from_raw((*self_).err as *mut c_char);
                        }
                        (*self_).release = None;
                    }
                }

                SedonaCError {
                    err: ptr,
                    err_len: len,
                    reserved: 0,
                    release: Some(release_error),
                }
            }
            Err(_) => UNKNOWN_SEDONA_C_ERROR,
        }
    }
}

impl std::fmt::Display for SedonaCError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.err.is_null() || self.err_len == 0 {
            return write!(f, "");
        }
        unsafe {
            let bytes = std::slice::from_raw_parts(self.err as *const u8, self.err_len as usize);
            write!(f, "{}", String::from_utf8_lossy(bytes))
        }
    }
}

impl Drop for SedonaCError {
    fn drop(&mut self) {
        if let Some(releaser) = self.release {
            unsafe { releaser(self) }
            self.release = None;
        }
    }
}

extern "C" fn sedona_c_noop_release(self_: *mut SedonaCError) {
    // Must set release to NULL per Arrow C Data Interface contract
    unsafe {
        (*self_).release = None;
    }
}

pub const UNKNOWN_SEDONA_C_ERROR: SedonaCError = SedonaCError {
    err: c"Unknown error".as_ptr(),
    err_len: "Unknown error".len() as u32,
    reserved: 0,
    release: Some(sedona_c_noop_release),
};

/// Write an error to an FFI error pointer safely.
///
/// This uses `ptr::write` instead of assignment to avoid running Drop on
/// potentially uninitialized memory. C consumers may pass uninitialized
/// SedonaCError structs, so we cannot assume the previous value is valid.
///
/// # Safety
///
/// The caller must ensure `err` points to valid memory for a SedonaCError,
/// but the memory does not need to be initialized.
#[inline]
pub unsafe fn write_ffi_error(err: *mut SedonaCError, message: &str) {
    if !err.is_null() {
        std::ptr::write(err, SedonaCError::new(message));
    }
}

/// Macro to write an error to an FFI error pointer with formatting.
///
/// This is a convenience wrapper around `write_ffi_error` that supports
/// format strings like `format!()`.
///
/// # Example
///
/// ```ignore
/// set_ffi_error!(err, "Failed to parse: {}", e);
/// ```
#[macro_export]
macro_rules! set_ffi_error {
    ($err:expr, $msg:expr) => {
        $crate::extension::write_ffi_error($err, $msg)
    };
    ($err:expr, $fmt:expr, $($arg:tt)*) => {
        $crate::extension::write_ffi_error($err, &format!($fmt, $($arg)*))
    };
}

/// Raw FFI representation of the SedonaCExpr
#[derive(Default)]
#[repr(C)]
pub struct SedonaCExpr {
    pub get_property_schema: Option<
        unsafe extern "C" fn(
            self_: *const SedonaCExpr,
            property: *const c_char,
            out: *mut FFI_ArrowSchema,
            err: *mut SedonaCError,
        ) -> c_int,
    >,

    pub get_property: Option<
        unsafe extern "C" fn(
            self_: *const SedonaCExpr,
            property: *const c_char,
            args: *const c_char,
            out: *mut FFI_ArrowArray,
            err: *mut SedonaCError,
        ) -> c_int,
    >,

    pub reserved: *mut c_void,

    pub release: Option<unsafe extern "C" fn(self_: *mut SedonaCExpr)>,

    pub private_data: *mut c_void,
}

unsafe impl Send for SedonaCExpr {}
unsafe impl Sync for SedonaCExpr {}

impl Drop for SedonaCExpr {
    fn drop(&mut self) {
        if let Some(releaser) = self.release {
            unsafe { releaser(self) }
            self.release = None;
            self.private_data = null_mut();
        }
    }
}

/// Raw FFI representation of the SedonaCExecutionPlanArgs
///
/// This structure is passed to methods that need JSON-serialized arguments,
/// optional execution plans, and/or expressions.
#[derive(Default)]
#[repr(C)]
pub struct SedonaCExecutionPlanArgs {
    /// JSON-serialized arguments
    pub args: *const u8,
    pub args_len: usize,
    /// Optional array of execution plans
    pub exec_plans: *const *const SedonaCExecutionPlan,
    pub num_exec_plans: usize,
    /// Optional array of expressions
    pub exprs: *const *const SedonaCExpr,
    pub num_exprs: usize,
    pub reserved: *mut c_void,
}

/// Raw FFI representation of the SedonaCExecutionPlan
#[derive(Default)]
#[repr(C)]
pub struct SedonaCExecutionPlan {
    pub get_schema: Option<
        unsafe extern "C" fn(
            self_: *const SedonaCExecutionPlan,
            out: *mut FFI_ArrowSchema,
            err: *mut SedonaCError,
        ) -> c_int,
    >,

    pub get_property_schema: Option<
        unsafe extern "C" fn(
            self_: *const SedonaCExecutionPlan,
            property: *const c_char,
            out: *mut FFI_ArrowSchema,
            err: *mut SedonaCError,
        ) -> c_int,
    >,

    pub get_property: Option<
        unsafe extern "C" fn(
            self_: *const SedonaCExecutionPlan,
            property: *const c_char,
            args: *mut SedonaCExecutionPlanArgs,
            out: *mut FFI_ArrowArray,
            err: *mut SedonaCError,
        ) -> c_int,
    >,

    pub with_property: Option<
        unsafe extern "C" fn(
            self_: *const SedonaCExecutionPlan,
            property: *const c_char,
            args: *mut SedonaCExecutionPlanArgs,
            out: *mut SedonaCExecutionPlan,
            err: *mut SedonaCError,
        ) -> c_int,
    >,

    pub execute: Option<
        unsafe extern "C" fn(
            self_: *const SedonaCExecutionPlan,
            args: *mut SedonaCExecutionPlanArgs,
            out: *mut FFI_ArrowArrayStream,
            err: *mut SedonaCError,
        ) -> c_int,
    >,

    pub execute_async: Option<
        unsafe extern "C" fn(
            self_: *const SedonaCExecutionPlan,
            args: *mut SedonaCExecutionPlanArgs,
            out: *mut c_void,
            err: *mut SedonaCError,
        ) -> c_int,
    >,

    pub reserved: *mut c_void,

    pub release: Option<unsafe extern "C" fn(self_: *mut SedonaCExecutionPlan)>,

    pub private_data: *mut c_void,
}

unsafe impl Send for SedonaCExecutionPlan {}
unsafe impl Sync for SedonaCExecutionPlan {}

impl Drop for SedonaCExecutionPlan {
    fn drop(&mut self) {
        if let Some(releaser) = self.release {
            unsafe { releaser(self) }
            self.release = None;
            self.private_data = null_mut();
        }
    }
}

/// Raw FFI representation of a TableProvider.
///
/// This provides a minimal interface for importing a TableProvider
/// across an FFI boundary in a version-agnostic manner.
#[derive(Default)]
#[repr(C)]
pub struct SedonaCTableProvider {
    /// Get the schema of this table provider
    pub get_schema: Option<
        unsafe extern "C" fn(
            self_: *const SedonaCTableProvider,
            out: *mut FFI_ArrowSchema,
            err: *mut SedonaCError,
        ) -> c_int,
    >,

    /// Get the schema of a property
    pub get_property_schema: Option<
        unsafe extern "C" fn(
            self_: *const SedonaCTableProvider,
            property: *const c_char,
            out: *mut FFI_ArrowSchema,
            err: *mut SedonaCError,
        ) -> c_int,
    >,

    /// Get a property value as an Arrow array
    pub get_property: Option<
        unsafe extern "C" fn(
            self_: *const SedonaCTableProvider,
            property: *const c_char,
            args: *mut SedonaCExecutionPlanArgs,
            out: *mut FFI_ArrowArray,
            err: *mut SedonaCError,
        ) -> c_int,
    >,

    /// Perform a scan operation and return an execution plan
    ///
    /// The args parameter contains JSON-serialized scan arguments.
    /// Returns an execution plan that can be used to read the data.
    pub scan: Option<
        unsafe extern "C" fn(
            self_: *const SedonaCTableProvider,
            args: *mut SedonaCExecutionPlanArgs,
            out: *mut SedonaCExecutionPlan,
            err: *mut SedonaCError,
        ) -> c_int,
    >,

    /// Perform an insert operation
    ///
    /// The args parameter contains JSON-serialized insert arguments.
    /// The exec_plans field should contain the plan providing rows to insert.
    /// Returns an execution plan that performs the insert.
    pub insert: Option<
        unsafe extern "C" fn(
            self_: *const SedonaCTableProvider,
            args: *mut SedonaCExecutionPlanArgs,
            out: *mut SedonaCExecutionPlan,
            err: *mut SedonaCError,
        ) -> c_int,
    >,

    /// Perform an update operation
    ///
    /// The args parameter contains JSON-serialized update arguments
    /// (filters, column assignments, etc.).
    /// Returns an execution plan that performs the update.
    pub update: Option<
        unsafe extern "C" fn(
            self_: *const SedonaCTableProvider,
            args: *mut SedonaCExecutionPlanArgs,
            out: *mut SedonaCExecutionPlan,
            err: *mut SedonaCError,
        ) -> c_int,
    >,

    /// Perform a delete operation
    ///
    /// The args parameter contains JSON-serialized delete arguments
    /// (filters, etc.).
    /// Returns an execution plan that performs the delete.
    pub delete_rows: Option<
        unsafe extern "C" fn(
            self_: *const SedonaCTableProvider,
            args: *mut SedonaCExecutionPlanArgs,
            out: *mut SedonaCExecutionPlan,
            err: *mut SedonaCError,
        ) -> c_int,
    >,

    pub reserved: *mut c_void,

    pub release: Option<unsafe extern "C" fn(self_: *mut SedonaCTableProvider)>,

    pub private_data: *mut c_void,
}

unsafe impl Send for SedonaCTableProvider {}
unsafe impl Sync for SedonaCTableProvider {}

impl Drop for SedonaCTableProvider {
    fn drop(&mut self) {
        if let Some(releaser) = self.release {
            unsafe { releaser(self) }
            self.release = None;
            self.private_data = null_mut();
        }
    }
}
