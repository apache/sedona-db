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
    fmt::Debug,
    os::raw::{c_char, c_void},
    ptr::null_mut,
};

use arrow_array::{
    builder::StringBuilder,
    ffi::{FFI_ArrowArray, FFI_ArrowSchema},
    Array,
};
use arrow_schema::{DataType, Field};
use datafusion_expr::Expr;

use crate::extension::{SedonaCError, SedonaCExpr};
use crate::set_ffi_error;
use crate::utils::{cstr_from_ptr_or_empty, ERRNO_OK};

/// Wrapper around a [datafusion_expr::Expr] that can be exported across FFI.
pub struct ExportedExpr {
    expr: Expr,
}

impl Debug for ExportedExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExportedExpr")
            .field("expr", &self.expr)
            .finish()
    }
}

impl ExportedExpr {
    /// Create a new ExportedExpr from a datafusion Expr.
    pub fn new(expr: Expr) -> Self {
        Self { expr }
    }

    /// Get the inner expression reference.
    pub fn expr(&self) -> &Expr {
        &self.expr
    }

    fn get_property(&self, property: &str) -> Result<String, String> {
        match property {
            "debug_string" => Ok(format!("{:?}", self.expr)),
            "display_string" => Ok(format!("{}", self.expr)),
            "variant" => Ok(self.expr.variant_name().to_string()),
            _ => Err(format!("Unknown property: {}", property)),
        }
    }
}

impl From<Expr> for ExportedExpr {
    fn from(expr: Expr) -> Self {
        Self::new(expr)
    }
}

impl From<ExportedExpr> for SedonaCExpr {
    fn from(value: ExportedExpr) -> Self {
        let boxed = Box::new(value);
        Self {
            get_property_schema: Some(c_expr_get_property_schema),
            get_property: Some(c_expr_get_property),
            with_property: None,
            reserved: null_mut(),
            release: Some(c_expr_release),
            private_data: Box::into_raw(boxed) as *mut c_void,
        }
    }
}

unsafe extern "C" fn c_expr_get_property_schema(
    _self_: *const SedonaCExpr,
    _property: *const c_char,
    out: *mut FFI_ArrowSchema,
    err: *mut SedonaCError,
) -> c_int {
    debug_assert!(!out.is_null(), "out pointer is null");
    // All properties are currently returned as Utf8 strings
    let field = Field::new("value", DataType::Utf8, false);
    match FFI_ArrowSchema::try_from(&field) {
        Ok(ffi_schema) => {
            std::ptr::write(out, ffi_schema);
            ERRNO_OK
        }
        Err(e) => {
            set_ffi_error!(err, "Failed to convert field to FFI schema: {}", e);
            libc::EINVAL
        }
    }
}

unsafe extern "C" fn c_expr_get_property(
    self_: *const SedonaCExpr,
    property: *const c_char,
    _args: *const u8,
    _args_len: usize,
    out: *mut FFI_ArrowArray,
    err: *mut SedonaCError,
) -> c_int {
    debug_assert!(!self_.is_null(), "self pointer is null");
    debug_assert!(!out.is_null(), "out pointer is null");
    let self_ref = &*self_;
    debug_assert!(!self_ref.private_data.is_null(), "private_data is null");
    let exported = &*(self_ref.private_data as *const ExportedExpr);
    let property_str = cstr_from_ptr_or_empty(property);

    match exported.get_property(&property_str) {
        Ok(value) => {
            let mut builder = StringBuilder::new();
            builder.append_value(&value);
            let array = builder.finish();
            let ffi_array = FFI_ArrowArray::new(&array.to_data());
            std::ptr::write(out, ffi_array);
            ERRNO_OK
        }
        Err(e) => {
            set_ffi_error!(err, "{}", e);
            libc::EINVAL
        }
    }
}

unsafe extern "C" fn c_expr_release(self_: *mut SedonaCExpr) {
    debug_assert!(!self_.is_null(), "self pointer is null");
    let self_ref = &mut *self_;
    if !self_ref.private_data.is_null() {
        let _ = Box::from_raw(self_ref.private_data as *mut ExportedExpr);
        self_ref.private_data = null_mut();
    }
    self_ref.release = None;
}
