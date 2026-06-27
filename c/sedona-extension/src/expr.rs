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
    ffi::{c_char, c_int, c_void, CStr},
    fmt::Debug,
    ptr::null_mut,
    sync::Arc,
};

use arrow_array::{builder::StringBuilder, ffi::FFI_ArrowArray, Array};
use arrow_schema::SchemaRef;
use datafusion_common::{DataFusionError, Result};
use datafusion_physical_expr::PhysicalExpr;
use sedona_common::{sedona_internal_datafusion_err, sedona_internal_err};
use sedona_expr::spatial_filter::SpatialFilterFactory;
use serde::Deserialize;

use crate::extension::{SedonaCError, SedonaCExpr};
use crate::utils::ERRNO_OK;

/// A wrapper around a DataFusion [PhysicalExpr] with its associated schema.
///
/// This struct bundles a physical expression with the schema context needed to
/// interpret it, which is required for operations like type resolution
/// and extracting spatial filter information.
pub struct PhysicalExprWithSchema {
    expr: Arc<dyn PhysicalExpr>,
    schema: SchemaRef,
}

impl Debug for PhysicalExprWithSchema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhysicalExprWithSchema")
            .field("expr", &self.expr.to_string())
            .field("schema", &self.schema)
            .finish()
    }
}

impl PhysicalExprWithSchema {
    /// Create a new PhysicalExprWithSchema
    pub fn new(expr: Arc<dyn PhysicalExpr>, schema: SchemaRef) -> Self {
        Self { expr, schema }
    }

    /// Get a reference to the expression
    pub fn expr(&self) -> &Arc<dyn PhysicalExpr> {
        &self.expr
    }

    /// Get a reference to the schema
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Consume self and return the inner expression and schema
    pub fn into_parts(self) -> (Arc<dyn PhysicalExpr>, SchemaRef) {
        (self.expr, self.schema)
    }

    /// Extract the bounding box for a spatial filter on the given column
    ///
    /// Returns the bounding box as a JSON string, or an error if extraction fails.
    pub fn filter_bbox(&self, column_name: &str) -> Result<String> {
        let factory = SpatialFilterFactory::default();
        let spatial_filter = factory.try_from_expr(&self.expr)?;
        let bbox = spatial_filter.filter_bbox(column_name);
        serde_json::to_string(&bbox)
            .map_err(|e| sedona_internal_datafusion_err!("Failed to serialize bounding box: {}", e))
    }
}

/// Arguments for the bbox property
#[derive(Debug, Deserialize)]
struct BboxArgs {
    column: String,
}

/// Wrapper around a [SedonaCExpr] that can be used to import an expression
/// from a C implementation.
pub struct ImportedExpr {
    inner: SedonaCExpr,
}

impl Debug for ImportedExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImportedExpr")
            .field("inner", &"<SedonaCExpr>")
            .finish()
    }
}

impl TryFrom<SedonaCExpr> for ImportedExpr {
    type Error = DataFusionError;

    fn try_from(value: SedonaCExpr) -> Result<Self> {
        match (&value.get_property, &value.release) {
            (Some(_), Some(_)) => Ok(Self { inner: value }),
            _ => Err(sedona_internal_datafusion_err!(
                "Can't import released or uninitialized SedonaCExpr"
            )),
        }
    }
}

impl ImportedExpr {
    /// Get a property from this expression
    ///
    /// # Safety
    ///
    /// The caller must ensure that the property name and args are valid
    /// null-terminated C strings if provided.
    pub unsafe fn get_property(
        &self,
        property: &CStr,
        args: Option<&CStr>,
    ) -> Result<FFI_ArrowArray> {
        let get_property = self
            .inner
            .get_property
            .ok_or_else(|| sedona_internal_datafusion_err!("get_property callback is null"))?;

        let args_ptr = args.map(|a| a.as_ptr()).unwrap_or(std::ptr::null());
        let mut out = FFI_ArrowArray::empty();
        let mut err = SedonaCError::default();

        let result = get_property(&self.inner, property.as_ptr(), args_ptr, &mut out, &mut err);

        if result != ERRNO_OK {
            return sedona_internal_err!("get_property failed: {}", err);
        }

        Ok(out)
    }
}

/// Export a [PhysicalExprWithSchema] as a [SedonaCExpr] for use across FFI boundaries.
pub struct ExportedExpr {
    inner: Arc<PhysicalExprWithSchema>,
}

impl ExportedExpr {
    /// Create a new ExportedExpr from a PhysicalExprWithSchema
    pub fn new(expr_with_schema: PhysicalExprWithSchema) -> Self {
        Self {
            inner: Arc::new(expr_with_schema),
        }
    }

    /// Export this expression as a SedonaCExpr
    ///
    /// The returned SedonaCExpr takes ownership of the Arc and will release
    /// it when the release callback is called.
    pub fn export(self) -> SedonaCExpr {
        let boxed = Box::new(self.inner);

        SedonaCExpr {
            get_property_schema: Some(exported_expr_get_property_schema),
            get_property: Some(exported_expr_get_property),
            reserved: null_mut(),
            release: Some(exported_expr_release),
            private_data: Box::into_raw(boxed) as *mut c_void,
        }
    }
}

unsafe extern "C" fn exported_expr_get_property_schema(
    _self_: *const SedonaCExpr,
    _property: *const c_char,
    err: *mut SedonaCError,
) -> c_int {
    // TODO: Implement property schema retrieval
    if !err.is_null() {
        *err = SedonaCError::new("get_property_schema not implemented");
    }
    libc::EINVAL
}

unsafe extern "C" fn exported_expr_get_property(
    self_: *const SedonaCExpr,
    property: *const c_char,
    args: *const c_char,
    out: *mut FFI_ArrowArray,
    err: *mut SedonaCError,
) -> c_int {
    if self_.is_null() || (*self_).private_data.is_null() {
        if !err.is_null() {
            *err = SedonaCError::new("null self pointer");
        }
        return libc::EINVAL;
    }

    let expr_arc = &*((*self_).private_data as *const Arc<PhysicalExprWithSchema>);

    if property.is_null() {
        if !err.is_null() {
            *err = SedonaCError::new("null property pointer");
        }
        return libc::EINVAL;
    }

    let property_str = match CStr::from_ptr(property).to_str() {
        Ok(s) => s,
        Err(_) => {
            if !err.is_null() {
                *err = SedonaCError::new("invalid UTF-8 in property name");
            }
            return libc::EINVAL;
        }
    };

    match property_str {
        "bbox" => {
            // Parse args JSON to get column name
            if args.is_null() {
                if !err.is_null() {
                    *err = SedonaCError::new("bbox property requires args with column name");
                }
                return libc::EINVAL;
            }

            let args_str = match CStr::from_ptr(args).to_str() {
                Ok(s) => s,
                Err(_) => {
                    if !err.is_null() {
                        *err = SedonaCError::new("invalid UTF-8 in args");
                    }
                    return libc::EINVAL;
                }
            };

            let bbox_args: BboxArgs = match serde_json::from_str(args_str) {
                Ok(a) => a,
                Err(e) => {
                    if !err.is_null() {
                        *err = SedonaCError::new(&format!("failed to parse args: {}", e));
                    }
                    return libc::EINVAL;
                }
            };

            // Extract bbox using SpatialFilterFactory
            let bbox_json = match expr_arc.filter_bbox(&bbox_args.column) {
                Ok(json) => json,
                Err(e) => {
                    if !err.is_null() {
                        *err = SedonaCError::new(&format!("failed to extract bbox: {}", e));
                    }
                    return libc::EINVAL;
                }
            };

            // Create a length-1 string array with the JSON result
            let mut builder = StringBuilder::new();
            builder.append_value(&bbox_json);
            let array = builder.finish();

            // Export the array via FFI
            std::ptr::write(out, FFI_ArrowArray::new(&array.to_data()));
            ERRNO_OK
        }
        _ => {
            if !err.is_null() {
                *err = SedonaCError::new(&format!("unknown property: {}", property_str));
            }
            libc::EINVAL
        }
    }
}

unsafe extern "C" fn exported_expr_release(self_: *mut SedonaCExpr) {
    if self_.is_null() {
        return;
    }

    let expr = &mut *self_;
    if !expr.private_data.is_null() {
        let _ = Box::from_raw(expr.private_data as *mut Arc<PhysicalExprWithSchema>);
        expr.private_data = null_mut();
    }

    expr.get_property_schema = None;
    expr.get_property = None;
    expr.release = None;
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{DataType, Field, Schema};
    use datafusion_common::ScalarValue;
    use datafusion_physical_expr::expressions::Literal;

    #[test]
    fn test_physical_expr_with_schema_new() {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let expr: Arc<dyn PhysicalExpr> = Arc::new(Literal::new(ScalarValue::Int32(Some(42))));
        let expr_with_schema = PhysicalExprWithSchema::new(expr.clone(), schema.clone());

        assert!(Arc::ptr_eq(&expr_with_schema.schema, &schema));
    }

    #[test]
    fn test_physical_expr_with_schema_into_parts() {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let expr: Arc<dyn PhysicalExpr> = Arc::new(Literal::new(ScalarValue::Int32(Some(42))));
        let expr_with_schema = PhysicalExprWithSchema::new(expr.clone(), schema.clone());

        let (returned_expr, returned_schema) = expr_with_schema.into_parts();
        assert_eq!(returned_expr.to_string(), expr.to_string());
        assert!(Arc::ptr_eq(&returned_schema, &schema));
    }

    #[test]
    fn test_exported_expr_roundtrip() {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let expr: Arc<dyn PhysicalExpr> = Arc::new(Literal::new(ScalarValue::Int32(Some(42))));
        let expr_with_schema = PhysicalExprWithSchema::new(expr, schema);

        let exported = ExportedExpr::new(expr_with_schema);
        let mut c_expr = exported.export();

        // Verify release callback works
        assert!(c_expr.release.is_some());
        unsafe {
            (c_expr.release.unwrap())(&mut c_expr);
        }
        assert!(c_expr.release.is_none());
        assert!(c_expr.private_data.is_null());
    }
}
