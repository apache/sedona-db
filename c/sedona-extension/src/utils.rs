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

//! Utilities for FFI property access and conversion.

use std::borrow::Cow;
use std::ffi::{c_char, c_int, CStr, CString};
use std::ptr::null_mut;

use arrow_array::builder::{BinaryBuilder, StringBuilder};
use arrow_array::ffi::FFI_ArrowArray;
use arrow_array::Array;
use arrow_schema::ffi::FFI_ArrowSchema;
use arrow_schema::{DataType, Field};
use datafusion_common::Result;
use sedona_common::{sedona_internal_datafusion_err, sedona_internal_err};
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::extension::{
    SedonaCError, SedonaCExecutionPlan, SedonaCExecutionPlanArgs, SedonaCTableProvider,
};

/// Success return code for FFI functions.
pub const ERRNO_OK: c_int = 0;

/// The value returned by a property getter.
///
/// This enum represents the two primary data types returned by FFI property accessors:
/// UTF-8 strings and binary data. It provides a type-safe way to handle property values
/// and can be converted to Arrow arrays for FFI transport.
#[derive(Debug, Clone)]
pub enum PropertyValue {
    /// A UTF-8 string value.
    String(String),
    /// A binary (bytes) value.
    Binary(Vec<u8>),
}

impl PropertyValue {
    /// Returns the Arrow [DataType] for this property value.
    pub fn data_type(&self) -> DataType {
        match self {
            PropertyValue::String(_) => DataType::Utf8,
            PropertyValue::Binary(_) => DataType::Binary,
        }
    }

    /// Converts this property value into an FFI-compatible Arrow array.
    ///
    /// This is useful for FFI callbacks that need to return property values
    /// as Arrow arrays across the FFI boundary.
    pub fn into_ffi_array(self) -> FFI_ArrowArray {
        match self {
            PropertyValue::String(value) => {
                let mut builder = StringBuilder::new();
                builder.append_value(&value);
                let array = builder.finish();
                FFI_ArrowArray::new(&array.to_data())
            }
            PropertyValue::Binary(value) => {
                let mut builder = BinaryBuilder::new();
                builder.append_value(&value);
                let array = builder.finish();
                FFI_ArrowArray::new(&array.to_data())
            }
        }
    }
}

/// Safely convert a C string pointer to a Rust string, treating null as empty.
///
/// # Safety
///
/// The pointer, if non-null, must point to a valid null-terminated C string.
/// The string must remain valid for the duration of the returned `Cow`.
pub unsafe fn cstr_from_ptr_or_empty<'a>(ptr: *const c_char) -> Cow<'a, str> {
    if ptr.is_null() {
        Cow::Borrowed("")
    } else {
        CStr::from_ptr(ptr).to_string_lossy()
    }
}

/// Write the schema used by UTF-8 properties to an FFI output pointer.
///
/// # Safety
///
/// `out` must point to writable memory for an [`FFI_ArrowSchema`]. If non-null,
/// `err` must point to writable memory for a [`SedonaCError`].
pub unsafe fn write_utf8_property_schema(
    out: *mut FFI_ArrowSchema,
    err: *mut SedonaCError,
) -> c_int {
    let field = Field::new("value", DataType::Utf8, false);
    match FFI_ArrowSchema::try_from(&field) {
        Ok(schema) => {
            unsafe { std::ptr::write(out, schema) };
            ERRNO_OK
        }
        Err(error) => {
            unsafe { crate::set_ffi_error!(err, "Failed to export property schema: {}", error) };
            libc::EINVAL
        }
    }
}

/// Serialize a property as JSON in a UTF-8 Arrow scalar.
///
/// # Safety
///
/// `out` must point to writable memory for an [`FFI_ArrowArray`]. If non-null,
/// `err` must point to writable memory for a [`SedonaCError`].
pub unsafe fn write_json_property<T: Serialize>(
    value: &T,
    out: *mut FFI_ArrowArray,
    err: *mut SedonaCError,
) -> c_int {
    match serde_json::to_string(value) {
        Ok(value) => {
            unsafe { std::ptr::write(out, PropertyValue::String(value).into_ffi_array()) };
            ERRNO_OK
        }
        Err(error) => {
            unsafe { crate::set_ffi_error!(err, "Failed to serialize property: {}", error) };
            libc::EINVAL
        }
    }
}

/// Deserialize a JSON value from a nullable, null-terminated C string.
///
/// # Safety
///
/// A non-null `args` pointer must refer to a valid null-terminated C string.
pub unsafe fn parse_json_c_args<T: DeserializeOwned>(args: *const c_char) -> Result<T> {
    if args.is_null() {
        return sedona_internal_err!("Property requires JSON arguments");
    }
    serde_json::from_str(&cstr_from_ptr_or_empty(args)).map_err(|error| {
        sedona_internal_datafusion_err!("Failed to parse property arguments: {error}")
    })
}

/// Get a string property from a [SedonaCTableProvider].
pub fn get_table_provider_string_property(
    provider: &SedonaCTableProvider,
    property: &str,
) -> Result<String> {
    let Some(get_property) = provider.get_property else {
        return sedona_internal_err!("SedonaCTableProvider does not have get_property");
    };

    call_get_string_property_impl(
        property,
        "SedonaCTableProvider",
        |prop, args, out, err| unsafe { get_property(provider, prop, args, out, err) },
        || get_table_provider_property_data_type(provider, property),
    )
}

/// Call `get_property` on a [SedonaCExecutionPlan] and deserialize the result.
///
/// This handles the common pattern of:
/// 1. Extracting the get_property function pointer
/// 2. Calling it with a property name and optional serializable args
/// 3. Parsing the binary array result
/// 4. Deserializing from JSON to the target type
///
/// # Arguments
///
/// * `plan` - The execution plan to query
/// * `property` - The property name to retrieve
/// * `args` - Optional arguments to pass (will be serialized to JSON bytes)
///
/// # Errors
///
/// Returns an error if:
/// - The plan does not have a `get_property` callback
/// - The FFI call fails
/// - The result cannot be parsed as a binary array
/// - The JSON deserialization fails
pub fn get_plan_property<T, A>(
    plan: &SedonaCExecutionPlan,
    property: &str,
    args: Option<&A>,
) -> Result<T>
where
    T: DeserializeOwned,
    A: Serialize,
{
    let Some(get_property) = plan.get_property else {
        return sedona_internal_err!("SedonaCExecutionPlan does not have get_property");
    };

    // Serialize args if provided
    let args_bytes = match args {
        Some(a) => serde_json::to_vec(a)
            .map_err(|e| sedona_internal_datafusion_err!("Failed to serialize args: {}", e))?,
        None => Vec::new(),
    };

    let mut ffi_args = SedonaCExecutionPlanArgs {
        args: if args_bytes.is_empty() {
            std::ptr::null()
        } else {
            args_bytes.as_ptr()
        },
        args_len: args_bytes.len(),
        exec_plans: std::ptr::null(),
        num_exec_plans: 0,
        exprs: std::ptr::null(),
        num_exprs: 0,
        reserved: null_mut(),
    };

    let bytes = call_get_property_impl(
        property,
        "SedonaCExecutionPlan",
        |property, out, err| unsafe { get_property(plan, property, &mut ffi_args, out, err) },
        || get_plan_property_data_type(plan, property),
    )?;
    serde_json::from_slice(&bytes)
        .map_err(|e| sedona_internal_datafusion_err!("Failed to deserialize property: {}", e))
}

/// Core implementation for getting property data type via FFI.
///
/// The caller provides a closure that performs the actual FFI call with
/// the correct self pointer type.
pub fn call_get_property_schema_impl<F>(property: &str, call_ffi: F) -> Result<DataType>
where
    F: FnOnce(*const c_char, *mut FFI_ArrowSchema, *mut SedonaCError) -> c_int,
{
    let property_cstr = CString::new(property)
        .map_err(|e| sedona_internal_datafusion_err!("Invalid property name: {}", e))?;

    let mut ffi_schema = FFI_ArrowSchema::empty();
    let mut err = SedonaCError::default();

    let code = call_ffi(property_cstr.as_ptr(), &mut ffi_schema, &mut err);

    if code != ERRNO_OK {
        return sedona_internal_err!("Failed to get property schema for '{}': {}", property, err);
    }

    // Try to convert the FFI schema to a Field
    let field = Field::try_from(&ffi_schema).map_err(|e| {
        sedona_internal_datafusion_err!("Failed to parse property schema for '{}': {}", property, e)
    })?;
    Ok(field.data_type().clone())
}

/// Core implementation for retrieving the bytes of an FFI property.
///
/// The caller adapts its provider-specific callback to `call_get_property` and
/// supplies the property's data type through `get_data_type`.
pub fn call_get_property_impl<F, G>(
    property: &str,
    type_name: &str,
    call_get_property: F,
    get_data_type: G,
) -> Result<Vec<u8>>
where
    F: FnOnce(*const c_char, *mut FFI_ArrowArray, *mut SedonaCError) -> c_int,
    G: FnOnce() -> Result<DataType>,
{
    let property_cstr = CString::new(property)
        .map_err(|e| sedona_internal_datafusion_err!("Invalid property name: {}", e))?;
    let mut ffi_array = FFI_ArrowArray::empty();
    let mut err = SedonaCError::default();

    let code = call_get_property(property_cstr.as_ptr(), &mut ffi_array, &mut err);
    if code != ERRNO_OK {
        return sedona_internal_err!("{} failed to get '{}': {}", type_name, property, err);
    }

    let data_type = get_data_type()?;
    parse_ffi_array_to_bytes(ffi_array, &data_type)
}

/// Retrieve and deserialize a JSON property whose optional arguments are
/// passed as a null-terminated JSON string.
pub fn call_get_json_property_impl<T, A, F, G>(
    property: &str,
    type_name: &str,
    args: Option<&A>,
    get_property_schema: G,
    get_property: F,
) -> Result<T>
where
    T: DeserializeOwned,
    A: Serialize,
    F: FnOnce(*const c_char, *const c_char, *mut FFI_ArrowArray, *mut SedonaCError) -> c_int,
    G: FnOnce(*const c_char, *mut FFI_ArrowSchema, *mut SedonaCError) -> c_int,
{
    let args_json = args
        .map(serde_json::to_string)
        .transpose()
        .map_err(|error| sedona_internal_datafusion_err!("Failed to serialize args: {error}"))?
        .map(CString::new)
        .transpose()
        .map_err(|error| sedona_internal_datafusion_err!("Invalid property args: {error}"))?;
    let args = args_json
        .as_ref()
        .map_or(std::ptr::null(), |args| args.as_ptr());
    let bytes = call_get_property_impl(
        property,
        type_name,
        |property, out, error| get_property(property, args, out, error),
        || call_get_property_schema_impl(property, get_property_schema),
    )?;
    serde_json::from_slice(&bytes).map_err(|error| {
        sedona_internal_datafusion_err!("Failed to deserialize property '{}': {}", property, error)
    })
}

/// Core implementation for getting a string property via FFI.
///
/// The caller provides closures for the FFI call and data type lookup.
fn call_get_string_property_impl<F, G>(
    property: &str,
    type_name: &str,
    call_get_property: F,
    get_data_type: G,
) -> Result<String>
where
    F: FnOnce(
        *const c_char,
        *mut SedonaCExecutionPlanArgs,
        *mut arrow_array::ffi::FFI_ArrowArray,
        *mut SedonaCError,
    ) -> c_int,
    G: FnOnce() -> Result<DataType>,
{
    let mut ffi_args = SedonaCExecutionPlanArgs {
        args: std::ptr::null(),
        args_len: 0,
        exec_plans: std::ptr::null(),
        num_exec_plans: 0,
        exprs: std::ptr::null(),
        num_exprs: 0,
        reserved: null_mut(),
    };

    let bytes = call_get_property_impl(
        property,
        type_name,
        |property, out, err| call_get_property(property, &mut ffi_args, out, err),
        get_data_type,
    )?;
    String::from_utf8(bytes)
        .map_err(|e| sedona_internal_datafusion_err!("Invalid UTF-8 in '{}': {}", property, e))
}

/// Get the schema for a property from a [SedonaCExecutionPlan].
///
/// Returns the DataType describing the property's data type.
/// If `get_property_schema` is not implemented, defaults to Binary.
fn get_plan_property_data_type(plan: &SedonaCExecutionPlan, property: &str) -> Result<DataType> {
    let Some(get_property_schema) = plan.get_property_schema else {
        return Ok(DataType::Binary);
    };

    call_get_property_schema_impl(property, |prop, schema, err| unsafe {
        get_property_schema(plan, prop, schema, err)
    })
}

/// Parse an FFI array and return the raw bytes.
pub fn parse_ffi_array_to_bytes(
    ffi_array: arrow_array::ffi::FFI_ArrowArray,
    data_type: &DataType,
) -> Result<Vec<u8>> {
    let data = unsafe { arrow_array::ffi::from_ffi_and_data_type(ffi_array, data_type.clone())? };
    let array = arrow_array::make_array(data);

    if array.len() != 1 || array.null_count() != 0 {
        return sedona_internal_err!(
            "Expected get_property() to return non-null array of length 1"
        );
    }

    // Handle different array types
    match data_type {
        DataType::Binary => {
            let binary_array = array
                .as_any()
                .downcast_ref::<arrow_array::BinaryArray>()
                .ok_or_else(|| {
                    sedona_internal_datafusion_err!("Expected binary array from get_property")
                })?;
            Ok(binary_array.value(0).to_vec())
        }
        DataType::LargeBinary => {
            let binary_array = array
                .as_any()
                .downcast_ref::<arrow_array::LargeBinaryArray>()
                .ok_or_else(|| {
                    sedona_internal_datafusion_err!("Expected large binary array from get_property")
                })?;
            Ok(binary_array.value(0).to_vec())
        }
        DataType::Utf8 => {
            let string_array = array
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .ok_or_else(|| {
                    sedona_internal_datafusion_err!("Expected string array from get_property")
                })?;
            Ok(string_array.value(0).as_bytes().to_vec())
        }
        DataType::LargeUtf8 => {
            let string_array = array
                .as_any()
                .downcast_ref::<arrow_array::LargeStringArray>()
                .ok_or_else(|| {
                    sedona_internal_datafusion_err!("Expected large string array from get_property")
                })?;
            Ok(string_array.value(0).as_bytes().to_vec())
        }
        _ => sedona_internal_err!("Unsupported data type for property: {:?}", data_type),
    }
}

/// Get a string property from a [SedonaCExecutionPlan].
///
/// This is a convenience wrapper around [get_plan_property] for string values.
pub fn get_plan_string_property(plan: &SedonaCExecutionPlan, property: &str) -> Result<String> {
    let Some(get_property) = plan.get_property else {
        return sedona_internal_err!("SedonaCExecutionPlan does not have get_property");
    };

    call_get_string_property_impl(
        property,
        "SedonaCExecutionPlan",
        |prop, args, out, err| unsafe { get_property(plan, prop, args, out, err) },
        || get_plan_property_data_type(plan, property),
    )
}

/// Get the schema for a property from a [SedonaCTableProvider].
///
/// Returns the DataType describing the property's data type.
/// If `get_property_schema` is not implemented, defaults to Binary.
fn get_table_provider_property_data_type(
    provider: &SedonaCTableProvider,
    property: &str,
) -> Result<DataType> {
    let Some(get_property_schema) = provider.get_property_schema else {
        return Ok(DataType::Binary);
    };

    call_get_property_schema_impl(property, |prop, schema, err| unsafe {
        get_property_schema(provider, prop, schema, err)
    })
}
