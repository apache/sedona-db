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

use std::{ffi::CString, iter::zip, sync::Arc};

use arrow_array::{
    ffi::{FFI_ArrowArray, FFI_ArrowSchema},
    ArrayRef,
};
use arrow_schema::{DataType, FieldRef};
use datafusion_common::{not_impl_datafusion_err, Result, ScalarValue};
use datafusion_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
    TypeSignature, Volatility,
};
use datafusion_ffi::udf::FFI_ScalarUDF;
use pyo3::{
    pyclass, pyfunction, pymethods,
    types::{PyCapsule, PyTuple},
    Bound, PyObject, Python,
};
use sedona_schema::datatypes::SedonaType;

use crate::{
    error::PySedonaError,
    import_from::{import_arrow_array, import_arrow_field},
    schema::{PySedonaField, PySedonaType},
};

#[pyfunction]
pub fn sedona_scalar_udf<'py>(
    py: Python<'py>,
    name: &str,
    py_return_field: PyObject,
    py_invoke_batch: PyObject,
    volatility: &str,
) -> Result<Bound<'py, PyCapsule>, PySedonaError> {
    let volatility = match volatility {
        "immutable" => Volatility::Immutable,
        "stable" => Volatility::Stable,
        "volatile" => Volatility::Volatile,
        v => {
            return Err(PySedonaError::SedonaPython(format!(
                "Expected one of 'immutable', 'stable', or 'volatile' but got '{v}'"
            )));
        }
    };

    let udf_impl = PySedonaScalarUdf {
        name: name.to_string(),
        signature: Signature::new(TypeSignature::UserDefined, volatility),
        py_return_field,
        py_invoke_batch,
    };

    let name = cr"datafusion_scalar_udf".into();
    let udf = ScalarUDF::from(udf_impl);
    let ffi_udf = FFI_ScalarUDF::from(Arc::new(udf));
    Ok(PyCapsule::new(py, ffi_udf, Some(name))?)
}

#[derive(Debug)]
struct PySedonaScalarUdf {
    name: String,
    signature: Signature,
    py_return_field: PyObject,
    py_invoke_batch: PyObject,
}

impl ScalarUDFImpl for PySedonaScalarUdf {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Err(PySedonaError::SedonaPython("Unexpected call to return_type()".to_string()).into())
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        Ok(eval_return_field(&self.py_return_field, args)?)
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        Ok(arg_types.to_vec())
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        Ok(eval_invoke_batch(&self.py_invoke_batch, args)?)
    }
}

fn eval_return_field(func: &PyObject, args: ReturnFieldArgs) -> Result<FieldRef, PySedonaError> {
    let return_field = Python::with_gil(|py| -> Result<FieldRef, PySedonaError> {
        let py_arg_fields = args
            .arg_fields
            .iter()
            .map(|f| PySedonaField::new(f.as_ref().clone()))
            .collect::<Vec<_>>();
        let py_scalar_values = zip(args.arg_fields, args.scalar_arguments)
            .map(|(field, maybe_arg)| {
                maybe_arg.map(|arg| PySedonaValue {
                    field: field.clone(),
                    value: ColumnarValue::Scalar(arg.clone()),
                    num_rows: 1,
                })
            })
            .collect::<Vec<_>>();

        let py_return_field = func.call(py, (py_arg_fields, py_scalar_values), None)?;
        if py_return_field.is_none(py) {
            return Err(PySedonaError::DF(Box::new(not_impl_datafusion_err!(
                "Python Udf does not apply to arguments"
            ))));
        }

        let return_field = import_arrow_field(py_return_field.bind(py))?;
        Ok(Arc::new(return_field))
    })?;

    Ok(return_field)
}

fn eval_invoke_batch(
    func: &PyObject,
    args: ScalarFunctionArgs,
) -> Result<ColumnarValue, PySedonaError> {
    let result = Python::with_gil(|py| -> Result<ArrayRef, PySedonaError> {
        let py_values = zip(&args.arg_fields, &args.args)
            .map(|(f, arg)| PySedonaValue {
                field: f.clone(),
                value: arg.clone(),
                num_rows: args.number_rows,
            })
            .collect::<Vec<_>>();

        let py_return_field = PySedonaField::new(args.return_field.as_ref().clone());
        let py_args = PyTuple::new(py, py_values)?;

        let result = func.call(py, (py_args, py_return_field, args.number_rows), None)?;

        let (result_field, result_array) = import_arrow_array(result.bind(py))?;
        let result_sedona_type = SedonaType::from_storage_field(&result_field)?;

        let expected_result_sedona_type = SedonaType::from_storage_field(&args.return_field)?;
        if expected_result_sedona_type != result_sedona_type {
            return Err(PySedonaError::SedonaPython(format!(
                "Expected {expected_result_sedona_type} but got {result_sedona_type}"
            )));
        }

        Ok(result_array)
    })?;

    if args.args.is_empty() {
        return Ok(ColumnarValue::Array(result));
    }

    for arg in &args.args {
        match arg {
            ColumnarValue::Array(_) => return Ok(ColumnarValue::Array(result)),
            ColumnarValue::Scalar(_) => {}
        }
    }

    Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
        &result, 0,
    )?))
}

#[pyclass]
#[derive(Debug)]
pub struct PySedonaValue {
    pub field: FieldRef,
    pub value: ColumnarValue,
    pub num_rows: usize,
}

#[pymethods]
impl PySedonaValue {
    #[getter]
    fn r#type(&self) -> Result<PySedonaType, PySedonaError> {
        Ok(PySedonaType::new(SedonaType::from_storage_field(
            &self.field,
        )?))
    }

    fn is_scalar(&self) -> bool {
        matches!(&self.value, ColumnarValue::Scalar(_))
    }

    fn to_array(&self) -> Result<Self, PySedonaError> {
        Ok(PySedonaValue {
            field: self.field.clone(),
            value: ColumnarValue::Array(self.value.to_array(self.num_rows)?),
            num_rows: self.num_rows,
        })
    }

    fn __arrow_c_schema__<'py>(
        &self,
        py: Python<'py>,
    ) -> Result<Bound<'py, PyCapsule>, PySedonaError> {
        let schema_capsule_name = CString::new("arrow_schema").unwrap();
        let ffi_schema = FFI_ArrowSchema::try_from(self.field.as_ref().clone())?;
        Ok(PyCapsule::new(py, ffi_schema, Some(schema_capsule_name))?)
    }

    fn __arrow_c_array__<'py>(
        &self,
        py: Python<'py>,
    ) -> Result<Bound<'py, PyCapsule>, PySedonaError> {
        let schema_capsule_name = CString::new("arrow_array").unwrap();
        let out_size = match &self.value {
            ColumnarValue::Array(array) => array.len(),
            ColumnarValue::Scalar(_) => 1,
        };
        let array = self.value.to_array(out_size)?;
        let ffi_array = FFI_ArrowArray::new(&array.to_data());
        Ok(PyCapsule::new(py, ffi_array, Some(schema_capsule_name))?)
    }

    fn __repr__(&self) -> String {
        format!("PySedonaValue {self:?}")
    }
}
