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

use std::os::raw::{c_char, c_void};

use arrow_array::ffi::{FFI_ArrowArray, FFI_ArrowSchema};

#[repr(C)]
pub struct SedonaCScalarUdf {
    pub init: Option<
        unsafe extern "C" fn(
            self_: *mut SedonaCScalarUdf,
            arg_types: *const *const FFI_ArrowSchema,
            scalar_args: *mut *mut FFI_ArrowArray,
            n_args: i64,
            out: *mut FFI_ArrowSchema,
        ) -> i32,
    >,

    pub execute: Option<
        unsafe extern "C" fn(
            self_: *mut SedonaCScalarUdf,
            args: *mut *mut FFI_ArrowArray,
            n_args: i64,
            n_rows: i64,
            out: *mut FFI_ArrowArray,
        ) -> i32,
    >,

    pub get_last_error: Option<unsafe extern "C" fn(self_: *mut SedonaCScalarUdf) -> *const c_char>,

    pub release: Option<unsafe extern "C" fn(self_: *mut SedonaCScalarUdf)>,

    pub private_data: *mut c_void,
}
