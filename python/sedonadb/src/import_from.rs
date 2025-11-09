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
    ffi::{c_void, CString},
    sync::Arc,
};

use arrow_array::{
    ffi::{FFI_ArrowArray, FFI_ArrowSchema},
    ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream},
    make_array, ArrayRef, RecordBatchReader,
};
use arrow_schema::{Field, Schema};
use datafusion::catalog::TableProvider;
use datafusion_expr::ScalarUDF;
use datafusion_ffi::{
    table_provider::{FFI_TableProvider, ForeignTableProvider},
    udf::{FFI_ScalarUDF, ForeignScalarUDF},
};
use pyo3::{
    types::{PyAnyMethods, PyCapsule, PyCapsuleMethods},
    Bound, PyAny, Python,
};
use sedona::record_batch_reader_provider::RecordBatchReaderProvider;
use sedona_schema::{
    datatypes::SedonaType,
    matchers::{ArgMatcher, TypeMatcher},
};

use crate::error::PySedonaError;

pub fn import_table_provider_from_any<'py>(
    py: Python<'py>,
    obj: &Bound<PyAny>,
    requested_schema: Option<&Bound<PyAny>>,
) -> Result<Arc<dyn TableProvider>, PySedonaError> {
    if obj.hasattr("__datafusion_table_provider__")? {
        let provider = import_ffi_table_provider(obj)?;
        Ok(provider)
    } else if obj.hasattr("__arrow_c_stream__")? {
        let reader = import_arrow_array_stream(py, obj, requested_schema)?;
        Ok(Arc::new(RecordBatchReaderProvider::new(reader)))
    } else {
        Err(PySedonaError::SedonaPython(
            "Can't create SedonaDB table from object".to_string(),
        ))
    }
}

pub fn import_ffi_table_provider(
    obj: &Bound<PyAny>,
) -> Result<Arc<dyn TableProvider>, PySedonaError> {
    let capsule = obj.getattr("__datafusion_table_provider__")?.call0()?;
    let contents =
        check_pycapsule(&capsule, "datafusion_table_provider")? as *mut FFI_TableProvider;
    let provider = ForeignTableProvider::from(unsafe { contents.as_ref().unwrap() });
    Ok(Arc::new(provider))
}

pub fn import_ffi_scalar_udf(obj: &Bound<PyAny>) -> Result<ScalarUDF, PySedonaError> {
    let capsule = obj.getattr("__datafusion_scalar_udf__")?.call0()?;
    let udf_ptr = check_pycapsule(&capsule, "datafusion_scalar_udf")? as *mut FFI_ScalarUDF;
    let udf: ForeignScalarUDF = unsafe { udf_ptr.as_ref().unwrap().try_into()? };
    Ok(udf.into())
}

pub fn import_arrow_array_stream<'py>(
    py: Python<'py>,
    obj: &Bound<PyAny>,
    requested_schema: Option<&Bound<PyAny>>,
) -> Result<Box<dyn RecordBatchReader + Send>, PySedonaError> {
    let capsule = if let Some(requested_schema) = requested_schema {
        let schema = import_arrow_schema(requested_schema)?;
        let ffi_schema = FFI_ArrowSchema::try_from(schema)?;
        let ffi_schema_capsule =
            PyCapsule::new(py, ffi_schema, Some(CString::new("arrow_schema").unwrap()))?;

        obj.getattr("__arrow_c_stream__")?
            .call1((ffi_schema_capsule,))?
    } else {
        obj.getattr("__arrow_c_stream__")?.call0()?
    };

    let stream = unsafe {
        FFI_ArrowArrayStream::from_raw(check_pycapsule(&capsule, "arrow_array_stream")? as _)
    };

    let stream_reader = ArrowArrayStreamReader::try_new(stream)?;
    Ok(Box::new(stream_reader))
}

pub fn import_arrow_array(obj: &Bound<PyAny>) -> Result<(Field, ArrayRef), PySedonaError> {
    let schema_and_array = obj.getattr("__arrow_c_array__")?.call0()?;
    let (schema_capsule, array_capsule): (Bound<PyCapsule>, Bound<PyCapsule>) =
        schema_and_array.extract()?;

    let ffi_schema = unsafe {
        FFI_ArrowSchema::from_raw(check_pycapsule(&schema_capsule, "arrow_schema")? as _)
    };
    let ffi_array =
        unsafe { FFI_ArrowArray::from_raw(check_pycapsule(&array_capsule, "arrow_array")? as _) };

    let result_field = Field::try_from(&ffi_schema)?;
    let result_array_data = unsafe { arrow_array::ffi::from_ffi(ffi_array, &ffi_schema)? };

    Ok((result_field, make_array(result_array_data)))
}

pub fn import_arg_matcher(
    obj: &Bound<PyAny>,
) -> Result<Arc<dyn TypeMatcher + Send + Sync>, PySedonaError> {
    if let Ok(string_value) = obj.extract::<String>() {
        match string_value.as_str() {
            "geometry" => return Ok(ArgMatcher::is_geometry()),
            "geography" => return Ok(ArgMatcher::is_geography()),
            "numeric" => return Ok(ArgMatcher::is_numeric()),
            "string" => return Ok(ArgMatcher::is_string()),
            "binary" => return Ok(ArgMatcher::is_binary()),
            "boolean" => return Ok(ArgMatcher::is_boolean()),
            v => {
                return Err(PySedonaError::SedonaPython(format!(
                    "Can't interpret literal string '{v}' as ArgMatcher"
                )))
            }
        }
    }

    let sedona_type = import_sedona_type(obj)?;
    Ok(ArgMatcher::is_exact(sedona_type))
}

pub fn import_sedona_type(obj: &Bound<PyAny>) -> Result<SedonaType, PySedonaError> {
    let field = import_arrow_field(obj)?;
    Ok(SedonaType::from_storage_field(&field)?)
}

pub fn import_arrow_field(obj: &Bound<PyAny>) -> Result<Field, PySedonaError> {
    let capsule = obj.getattr("__arrow_c_schema__")?.call0()?;
    let schema =
        unsafe { FFI_ArrowSchema::from_raw(check_pycapsule(&capsule, "arrow_schema")? as _) };

    Ok(Field::try_from(&schema)?)
}

pub fn import_arrow_schema(obj: &Bound<PyAny>) -> Result<Schema, PySedonaError> {
    let capsule = obj.getattr("__arrow_c_schema__")?.call0()?;
    let schema =
        unsafe { FFI_ArrowSchema::from_raw(check_pycapsule(&capsule, "arrow_schema")? as _) };

    Ok(Schema::try_from(&schema)?)
}

pub fn check_pycapsule(obj: &Bound<PyAny>, name: &str) -> Result<*mut c_void, PySedonaError> {
    let capsule = obj
        .downcast::<PyCapsule>()
        .map_err(|e| PySedonaError::SedonaPython(e.to_string()))?;

    let actual_name = capsule
        .name()?
        .map(|obj| obj.to_string_lossy().to_string())
        .unwrap_or("<unnamed>".to_string());
    if actual_name != name {
        return Err(PySedonaError::SedonaPython(format!(
            "Expected PyCapsule with name '{name}' but got PyCapsule with name '{actual_name}'"
        )));
    }

    if capsule.pointer().is_null() {
        return Err(PySedonaError::SedonaPython(format!(
            "PyCapsule with name '{name}' is NULL"
        )));
    }

    Ok(capsule.pointer())
}
