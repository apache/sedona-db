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
    ffi::{c_int, CString},
    fmt::{Debug, Display},
    os::raw::{c_char, c_void},
    ptr::null_mut,
};

use arrow_array::{
    builder::StringBuilder,
    ffi::{FFI_ArrowArray, FFI_ArrowSchema},
    Array,
};
use arrow_schema::{DataType, Field};
use datafusion_common::Result;
use datafusion_expr::Expr;
use sedona_common::{sedona_internal_datafusion_err, sedona_internal_err};

use crate::extension::{SedonaCError, SedonaCExprView};
use crate::set_ffi_error;
use crate::utils::{
    call_get_property_schema_impl, cstr_from_ptr_or_empty, parse_ffi_array_to_bytes, ERRNO_OK,
};

/// Wrapper around a [datafusion_expr::Expr] that can be exported across FFI.
pub struct ExportedExprView<'a> {
    expr: &'a Expr,
}

impl<'a> Debug for ExportedExprView<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExportedExprView")
            .field("expr", &self.expr)
            .finish()
    }
}

impl<'a> ExportedExprView<'a> {
    /// Create a new ExportedExprView from a reference to a datafusion Expr.
    ///
    /// The returned view is valid only as long as the referenced Expr remains alive.
    pub fn new(expr: &'a Expr) -> Self {
        Self { expr }
    }

    /// Get the inner expression reference.
    pub fn expr(&self) -> &Expr {
        self.expr
    }

    /// Get a FFI-compatible view of this expression.
    ///
    /// The returned [SedonaCExprView] is valid only as long as this ExportedExprView
    /// remains alive. The caller must ensure they do not use the FFI view after
    /// this object is dropped.
    pub fn as_ffi_view(&self) -> SedonaCExprView {
        SedonaCExprView {
            get_property_schema: Some(c_expr_get_property_schema),
            get_property: Some(c_expr_get_property),
            reserved: null_mut(),
            private_data: self as *const ExportedExprView as *const c_void,
        }
    }

    fn get_property(&self, property: &str) -> Result<String, String> {
        match property {
            "debug_string" => Ok(format!("{:?}", self.expr)),
            "display_string" => Ok(format!("{}", self.expr)),
            _ => Err(format!("Unknown property: {}", property)),
        }
    }
}

impl<'a> From<&'a Expr> for ExportedExprView<'a> {
    fn from(expr: &'a Expr) -> Self {
        Self::new(expr)
    }
}

unsafe extern "C" fn c_expr_get_property_schema(
    _self_: *const SedonaCExprView,
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
    self_: *const SedonaCExprView,
    property: *const c_char,
    _args: *const u8,
    args_len: usize,
    out: *mut FFI_ArrowArray,
    err: *mut SedonaCError,
) -> c_int {
    debug_assert!(!self_.is_null(), "self pointer is null");
    debug_assert!(!out.is_null(), "out pointer is null");

    if args_len > 0 {
        set_ffi_error!(err, "get_property does not accept arguments");
        return libc::EINVAL;
    }

    let self_ref = &*self_;
    debug_assert!(!self_ref.private_data.is_null(), "private_data is null");
    let exported = &*(self_ref.private_data as *const ExportedExprView);
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

/// A borrowed expression view that can be used across an FFI boundary.
///
/// This wraps a reference to a [SedonaCExprView] and provides `Debug` and `Display`
/// implementations by querying properties from the FFI interface.
///
/// The lifetime `'a` represents the lifetime of the underlying expression that
/// this view references. The view must not be used after the underlying expression
/// is dropped.
pub struct ImportedExprView<'a> {
    inner: &'a SedonaCExprView,
}

impl<'a> Debug for ImportedExprView<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Ok(debug_str) = get_expr_view_string_property(self.inner, "debug_string") {
            f.debug_struct("ImportedExprView")
                .field("inner", &debug_str)
                .finish()
        } else {
            f.debug_struct("ImportedExprView").finish()
        }
    }
}

impl<'a> Display for ImportedExprView<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Ok(display_str) = get_expr_view_string_property(self.inner, "display_string") {
            write!(f, "{}", display_str)
        } else {
            write!(f, "ImportedExprView")
        }
    }
}

impl<'a> ImportedExprView<'a> {
    /// Create a new ImportedExprView from a SedonaCExprView reference.
    ///
    /// Returns an error if the SedonaCExprView does not have a valid private_data pointer.
    pub fn try_new(inner: &'a SedonaCExprView) -> Result<Self> {
        if inner.private_data.is_null() {
            return sedona_internal_err!("SedonaCExprView has null private_data (invalid view)");
        }
        Ok(Self { inner })
    }

    /// Get a string property from this expression view.
    pub fn get_string_property(&self, property: &str) -> Result<String> {
        get_expr_view_string_property(self.inner, property)
    }
}

/// Get a string property from a [SedonaCExprView].
pub fn get_expr_view_string_property(expr: &SedonaCExprView, property: &str) -> Result<String> {
    let Some(get_property) = expr.get_property else {
        return sedona_internal_err!("SedonaCExprView does not have get_property");
    };

    let property_cstr = CString::new(property)
        .map_err(|e| sedona_internal_datafusion_err!("Invalid property name: {}", e))?;

    let mut ffi_array = FFI_ArrowArray::empty();
    let mut err = SedonaCError::default();

    let code = unsafe {
        get_property(
            expr,
            property_cstr.as_ptr(),
            std::ptr::null(),
            0,
            &mut ffi_array,
            &mut err,
        )
    };

    if code != ERRNO_OK {
        return sedona_internal_err!("SedonaCExprView failed to get '{}': {}", property, err);
    }

    let data_type = get_expr_view_property_data_type(expr, property)?;
    let bytes = parse_ffi_array_to_bytes(ffi_array, &data_type)?;
    String::from_utf8(bytes)
        .map_err(|e| sedona_internal_datafusion_err!("Invalid UTF-8 in '{}': {}", property, e))
}

/// Get the data type for a property from a [SedonaCExprView].
fn get_expr_view_property_data_type(expr: &SedonaCExprView, property: &str) -> Result<DataType> {
    let Some(get_property_schema) = expr.get_property_schema else {
        return Ok(DataType::Utf8);
    };

    call_get_property_schema_impl(property, |prop, schema, err| unsafe {
        get_property_schema(expr, prop, schema, err)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_expr::col;

    #[test]
    fn test_expr_view_debug_string() {
        let expr = col("test_column");
        let exported = ExportedExprView::new(&expr);
        let view = exported.as_ffi_view();
        let imported = ImportedExprView::try_new(&view).unwrap();

        let debug_str = imported.get_string_property("debug_string").unwrap();
        assert_eq!(debug_str, format!("{:?}", expr));
    }

    #[test]
    fn test_expr_view_display_string() {
        let expr = col("test_column");
        let exported = ExportedExprView::new(&expr);
        let view = exported.as_ffi_view();
        let imported = ImportedExprView::try_new(&view).unwrap();

        let display_str = imported.get_string_property("display_string").unwrap();
        assert_eq!(display_str, format!("{}", expr));
    }

    #[test]
    fn test_expr_view_debug_impl() {
        let expr = col("my_col");
        let exported = ExportedExprView::new(&expr);
        let view = exported.as_ffi_view();
        let imported = ImportedExprView::try_new(&view).unwrap();

        let debug_output = format!("{:?}", imported);
        assert!(debug_output.contains("ImportedExprView"));
        assert!(debug_output.contains("my_col"));
    }

    #[test]
    fn test_expr_view_display_impl() {
        let expr = col("my_col");
        let exported = ExportedExprView::new(&expr);
        let view = exported.as_ffi_view();
        let imported = ImportedExprView::try_new(&view).unwrap();

        let display_output = format!("{}", imported);
        assert_eq!(display_output, "my_col");
    }

    #[test]
    fn test_expr_view_unknown_property_error() {
        let expr = col("test");
        let exported = ExportedExprView::new(&expr);
        let view = exported.as_ffi_view();
        let imported = ImportedExprView::try_new(&view).unwrap();

        let result = imported.get_string_property("nonexistent_property");
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Unknown property"));
    }

    #[test]
    fn test_expr_view_with_null_private_data_fails() {
        let invalid_view = SedonaCExprView::default();
        let result = ImportedExprView::try_new(&invalid_view);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("null private_data"));
    }
}
