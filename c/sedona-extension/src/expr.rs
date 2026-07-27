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
    sync::Arc,
};

use arrow_array::{
    builder::StringBuilder,
    ffi::{FFI_ArrowArray, FFI_ArrowSchema},
    Array,
};
use arrow_schema::{DataType, Field};
use datafusion_common::Result;
use datafusion_expr::Expr;
use datafusion_physical_expr::PhysicalExpr;
use sedona_common::{sedona_internal_datafusion_err, sedona_internal_err};

use crate::extension::{SedonaCError, SedonaCExpr};
use crate::set_ffi_error;
use crate::utils::{
    call_get_property_schema_impl, cstr_from_ptr_or_empty, parse_ffi_array_to_bytes, ERRNO_OK,
};

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
            _ => Err(format!("Unknown property: {}", property)),
        }
    }

    fn with_property(&self, property: &str) -> Result<ExportedExpr, String> {
        match property {
            "cloned" => Ok(ExportedExpr::new(self.expr.clone())),
            _ => Err(format!("Unknown with_property: {}", property)),
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
            with_property: Some(c_expr_with_property),
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

unsafe extern "C" fn c_expr_with_property(
    self_: *const SedonaCExpr,
    property: *const c_char,
    _args: *const u8,
    args_len: usize,
    out: *mut SedonaCExpr,
    err: *mut SedonaCError,
) -> c_int {
    debug_assert!(!self_.is_null(), "self pointer is null");
    debug_assert!(!out.is_null(), "out pointer is null");

    if args_len > 0 {
        set_ffi_error!(err, "with_property does not accept arguments");
        return libc::EINVAL;
    }

    let self_ref = &*self_;
    debug_assert!(!self_ref.private_data.is_null(), "private_data is null");
    let exported = &*(self_ref.private_data as *const ExportedExpr);
    let property_str = cstr_from_ptr_or_empty(property);

    match exported.with_property(&property_str) {
        Ok(new_expr) => {
            let ffi_expr: SedonaCExpr = new_expr.into();
            std::ptr::write(out, ffi_expr);
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

/// Wrapper around an [Arc<dyn PhysicalExpr>] that can be exported across FFI.
pub struct ExportedPhysicalExpr {
    expr: Arc<dyn PhysicalExpr>,
}

impl Debug for ExportedPhysicalExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExportedPhysicalExpr")
            .field("expr", &self.expr)
            .finish()
    }
}

impl ExportedPhysicalExpr {
    /// Create a new ExportedPhysicalExpr from a PhysicalExpr.
    pub fn new(expr: Arc<dyn PhysicalExpr>) -> Self {
        Self { expr }
    }

    /// Get the inner expression reference.
    pub fn expr(&self) -> &Arc<dyn PhysicalExpr> {
        &self.expr
    }

    fn get_property(&self, property: &str) -> Result<String, String> {
        match property {
            "debug_string" => Ok(format!("{:?}", self.expr)),
            "display_string" => Ok(format!("{}", self.expr)),
            _ => Err(format!("Unknown property: {}", property)),
        }
    }

    fn with_property(&self, property: &str) -> Result<ExportedPhysicalExpr, String> {
        match property {
            "cloned" => Ok(ExportedPhysicalExpr::new(Arc::clone(&self.expr))),
            _ => Err(format!("Unknown with_property: {}", property)),
        }
    }
}

impl From<Arc<dyn PhysicalExpr>> for ExportedPhysicalExpr {
    fn from(expr: Arc<dyn PhysicalExpr>) -> Self {
        Self::new(expr)
    }
}

impl From<ExportedPhysicalExpr> for SedonaCExpr {
    fn from(value: ExportedPhysicalExpr) -> Self {
        let boxed = Box::new(value);
        Self {
            get_property_schema: Some(c_physical_expr_get_property_schema),
            get_property: Some(c_physical_expr_get_property),
            with_property: Some(c_physical_expr_with_property),
            reserved: null_mut(),
            release: Some(c_physical_expr_release),
            private_data: Box::into_raw(boxed) as *mut c_void,
        }
    }
}

unsafe extern "C" fn c_physical_expr_get_property_schema(
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

unsafe extern "C" fn c_physical_expr_get_property(
    self_: *const SedonaCExpr,
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
    let exported = &*(self_ref.private_data as *const ExportedPhysicalExpr);
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

unsafe extern "C" fn c_physical_expr_with_property(
    self_: *const SedonaCExpr,
    property: *const c_char,
    _args: *const u8,
    args_len: usize,
    out: *mut SedonaCExpr,
    err: *mut SedonaCError,
) -> c_int {
    debug_assert!(!self_.is_null(), "self pointer is null");
    debug_assert!(!out.is_null(), "out pointer is null");

    if args_len > 0 {
        set_ffi_error!(err, "with_property does not accept arguments");
        return libc::EINVAL;
    }

    let self_ref = &*self_;
    debug_assert!(!self_ref.private_data.is_null(), "private_data is null");
    let exported = &*(self_ref.private_data as *const ExportedPhysicalExpr);
    let property_str = cstr_from_ptr_or_empty(property);

    match exported.with_property(&property_str) {
        Ok(new_expr) => {
            let ffi_expr: SedonaCExpr = new_expr.into();
            std::ptr::write(out, ffi_expr);
            ERRNO_OK
        }
        Err(e) => {
            set_ffi_error!(err, "{}", e);
            libc::EINVAL
        }
    }
}

unsafe extern "C" fn c_physical_expr_release(self_: *mut SedonaCExpr) {
    debug_assert!(!self_.is_null(), "self pointer is null");
    let self_ref = &mut *self_;
    if !self_ref.private_data.is_null() {
        let _ = Box::from_raw(self_ref.private_data as *mut ExportedPhysicalExpr);
        self_ref.private_data = null_mut();
    }
    self_ref.release = None;
}

/// An expression wrapper that can be imported from across an FFI boundary.
///
/// This wraps a [SedonaCExpr] and provides `Debug` and `Display` implementations
/// by querying properties from the FFI interface.
pub struct ImportedExpr {
    inner: SedonaCExpr,
}

impl Debug for ImportedExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Ok(debug_str) = get_expr_string_property(&self.inner, "debug_string") {
            f.debug_struct("ImportedExpr")
                .field("inner", &debug_str)
                .finish()
        } else {
            f.debug_struct("ImportedExpr").finish()
        }
    }
}

impl Display for ImportedExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Ok(display_str) = get_expr_string_property(&self.inner, "display_string") {
            write!(f, "{}", display_str)
        } else {
            write!(f, "ImportedExpr")
        }
    }
}

impl ImportedExpr {
    /// Create a new ImportedExpr from a SedonaCExpr.
    ///
    /// Returns an error if the SedonaCExpr does not have a valid release callback.
    pub fn try_new(inner: SedonaCExpr) -> Result<Self> {
        if inner.release.is_none() {
            return sedona_internal_err!("SedonaCExpr does not have a release callback");
        }
        Ok(Self { inner })
    }

    /// Get a string property from this expression.
    pub fn get_string_property(&self, property: &str) -> Result<String> {
        get_expr_string_property(&self.inner, property)
    }

    /// Clone this expression, returning a new ImportedExpr.
    pub fn clone_expr(&self) -> Result<ImportedExpr> {
        let new_ffi = call_with_property(&self.inner, "cloned")?;
        ImportedExpr::try_new(new_ffi)
    }
}

/// Call with_property on a [SedonaCExpr] and return a new SedonaCExpr.
fn call_with_property(expr: &SedonaCExpr, property: &str) -> Result<SedonaCExpr> {
    let Some(with_property) = expr.with_property else {
        return sedona_internal_err!("SedonaCExpr does not have with_property");
    };

    let property_cstr = CString::new(property)
        .map_err(|e| sedona_internal_datafusion_err!("Invalid property name: {}", e))?;

    let mut out = SedonaCExpr::default();
    let mut err = SedonaCError::default();

    let code = unsafe {
        with_property(
            expr,
            property_cstr.as_ptr(),
            std::ptr::null(),
            0,
            &mut out,
            &mut err,
        )
    };

    if code != ERRNO_OK {
        return sedona_internal_err!("SedonaCExpr with_property '{}' failed: {}", property, err);
    }

    Ok(out)
}

/// Get a string property from a [SedonaCExpr].
pub fn get_expr_string_property(expr: &SedonaCExpr, property: &str) -> Result<String> {
    let Some(get_property) = expr.get_property else {
        return sedona_internal_err!("SedonaCExpr does not have get_property");
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
        return sedona_internal_err!("SedonaCExpr failed to get '{}': {}", property, err);
    }

    let data_type = get_expr_property_data_type(expr, property)?;
    let bytes = parse_ffi_array_to_bytes(ffi_array, &data_type)?;
    String::from_utf8(bytes)
        .map_err(|e| sedona_internal_datafusion_err!("Invalid UTF-8 in '{}': {}", property, e))
}

/// Get the data type for a property from a [SedonaCExpr].
fn get_expr_property_data_type(expr: &SedonaCExpr, property: &str) -> Result<DataType> {
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

    fn roundtrip_expr(expr: Expr) -> ImportedExpr {
        let exported = ExportedExpr::new(expr);
        let ffi_expr: SedonaCExpr = exported.into();
        ImportedExpr::try_new(ffi_expr).unwrap()
    }

    #[test]
    fn test_roundtrip_debug_string() {
        let expr = col("test_column");
        let imported = roundtrip_expr(expr.clone());

        let debug_str = imported.get_string_property("debug_string").unwrap();
        assert_eq!(debug_str, format!("{:?}", expr));
    }

    #[test]
    fn test_roundtrip_display_string() {
        let expr = col("test_column");
        let imported = roundtrip_expr(expr.clone());

        let display_str = imported.get_string_property("display_string").unwrap();
        assert_eq!(display_str, format!("{}", expr));
    }

    #[test]
    fn test_debug_impl() {
        let expr = col("my_col");
        let imported = roundtrip_expr(expr);

        let debug_output = format!("{:?}", imported);
        assert!(debug_output.contains("ImportedExpr"));
        assert!(debug_output.contains("my_col"));
    }

    #[test]
    fn test_display_impl() {
        let expr = col("my_col");
        let imported = roundtrip_expr(expr);

        let display_output = format!("{}", imported);
        assert_eq!(display_output, "my_col");
    }

    #[test]
    fn test_unknown_property_error() {
        let expr = col("test");
        let imported = roundtrip_expr(expr);

        let result = imported.get_string_property("nonexistent_property");
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Unknown property"));
    }

    #[test]
    fn test_imported_expr_without_release_fails() {
        let invalid_expr = SedonaCExpr::default();
        let result = ImportedExpr::try_new(invalid_expr);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("does not have a release callback"));
    }

    #[test]
    fn test_clone_expr() {
        let expr = col("original");
        let imported = roundtrip_expr(expr.clone());

        // Clone via FFI
        let cloned = imported.clone_expr().unwrap();

        // Verify the clone has the same display string
        let original_display = imported.get_string_property("display_string").unwrap();
        let cloned_display = cloned.get_string_property("display_string").unwrap();
        assert_eq!(original_display, cloned_display);
        assert_eq!(cloned_display, "original");
    }

    #[test]
    fn test_unknown_with_property_error() {
        let expr = col("test");
        let exported = ExportedExpr::new(expr);
        let ffi_expr: SedonaCExpr = exported.into();
        let imported = ImportedExpr::try_new(ffi_expr).unwrap();

        // Try to call an unknown with_property
        let result = call_with_property(&imported.inner, "unknown_property");
        assert!(result.is_err());
        let err_msg = result.err().unwrap().to_string();
        assert!(err_msg.contains("Unknown with_property"));
    }

    fn roundtrip_physical_expr(expr: Arc<dyn PhysicalExpr>) -> ImportedExpr {
        let exported = ExportedPhysicalExpr::new(expr);
        let ffi_expr: SedonaCExpr = exported.into();
        ImportedExpr::try_new(ffi_expr).unwrap()
    }

    #[test]
    fn test_physical_expr_roundtrip_debug_string() {
        use datafusion_physical_expr::expressions::Column;
        let expr: Arc<dyn PhysicalExpr> = Arc::new(Column::new("test_column", 0));
        let imported = roundtrip_physical_expr(Arc::clone(&expr));

        let debug_str = imported.get_string_property("debug_string").unwrap();
        assert_eq!(debug_str, format!("{:?}", expr));
    }

    #[test]
    fn test_physical_expr_roundtrip_display_string() {
        use datafusion_physical_expr::expressions::Column;
        let expr: Arc<dyn PhysicalExpr> = Arc::new(Column::new("test_column", 0));
        let imported = roundtrip_physical_expr(Arc::clone(&expr));

        let display_str = imported.get_string_property("display_string").unwrap();
        assert_eq!(display_str, format!("{}", expr));
    }

    #[test]
    fn test_physical_expr_clone() {
        use datafusion_physical_expr::expressions::Column;
        let expr: Arc<dyn PhysicalExpr> = Arc::new(Column::new("original", 0));
        let imported = roundtrip_physical_expr(expr);

        // Clone via FFI
        let cloned = imported.clone_expr().unwrap();

        // Verify the clone has the same display string
        let original_display = imported.get_string_property("display_string").unwrap();
        let cloned_display = cloned.get_string_property("display_string").unwrap();
        assert_eq!(original_display, cloned_display);
        assert_eq!(cloned_display, "original@0");
    }
}
