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

use std::{ffi::c_int, os::raw::{c_char, c_void}};

use arrow_array::ffi::{FFI_ArrowArray, FFI_ArrowSchema};

#[derive(Default)]
#[repr(C)]
pub struct SedonaCScalarUdfFactory {
    pub new_scalar_udf_impl: Option<
        unsafe extern "C" fn(self_: *const SedonaCScalarUdfFactory, out: *mut SedonaCScalarUdf),
    >,

    release: Option<unsafe extern "C" fn(self_: *mut SedonaCScalarUdfFactory)>,
    private_data: *mut c_void,
}

unsafe impl Send for SedonaCScalarUdfFactory {}
unsafe impl Sync for SedonaCScalarUdfFactory {}

impl Drop for SedonaCScalarUdfFactory {
    fn drop(&mut self) {
        if let Some(releaser) = self.release {
            unsafe { releaser(self) }
        }
    }
}

#[derive(Default)]
#[repr(C)]
pub struct SedonaCScalarUdf {
    pub init: Option<
        unsafe extern "C" fn(
            self_: *mut SedonaCScalarUdf,
            arg_types: *const *const FFI_ArrowSchema,
            scalar_args: *mut *mut FFI_ArrowArray,
            n_args: i64,
            out: *mut FFI_ArrowSchema,
        ) -> c_int,
    >,

    pub execute: Option<
        unsafe extern "C" fn(
            self_: *mut SedonaCScalarUdf,
            args: *mut *mut FFI_ArrowArray,
            n_args: i64,
            n_rows: i64,
            out: *mut FFI_ArrowArray,
        ) -> c_int,
    >,

    pub get_last_error: Option<unsafe extern "C" fn(self_: *mut SedonaCScalarUdf) -> *const c_char>,

    release: Option<unsafe extern "C" fn(self_: *mut SedonaCScalarUdf)>,

    private_data: *mut c_void,
}

impl Drop for SedonaCScalarUdf {
    fn drop(&mut self) {
        if let Some(releaser) = self.release {
            unsafe { releaser(self) }
        }
    }
}
