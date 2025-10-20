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
use datafusion_common::{Result, ScalarValue};
use datafusion_expr::{ColumnarValue, ScalarUDF, Volatility};
use datafusion_ffi::udf::FFI_ScalarUDF;
use pyo3::{
    pyclass, pyfunction, pymethods,
    types::{PyCapsule, PyNone, PyTuple},
    Bound, PyObject, Python,
};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::{
    error::PySedonaError,
    import_from::{import_arg_matcher, import_arrow_array, import_sedona_type},
    schema::PySedonaType,
};

#[pyfunction]
pub fn sedona_scalar_udf<'py>(
    py: Python<'py>,
    py_invoke_batch: PyObject,
    py_return_type: PyObject,
    py_input_types: Option<Vec<PyObject>>,
    volatility: &str,
    name: &str,
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

    let scalar_kernel = sedona_scalar_kernel(py, py_input_types, py_return_type, py_invoke_batch)?;
    let sedona_scalar_udf =
        SedonaScalarUDF::new(name, vec![Arc::new(scalar_kernel)], volatility, None);
    let scalar_udf: ScalarUDF = sedona_scalar_udf.into();

    let name = cr"datafusion_scalar_udf".into();
    let ffi_udf = FFI_ScalarUDF::from(Arc::new(scalar_udf));
    Ok(PyCapsule::new(py, ffi_udf, Some(name))?)
}

fn sedona_scalar_kernel<'py>(
    py: Python<'py>,
    input_types: Option<Vec<PyObject>>,
    py_return_field: PyObject,
    py_invoke_batch: PyObject,
) -> Result<PySedonaScalarKernel, PySedonaError> {
    let matcher = if let Some(input_types) = input_types {
        let arg_matchers = input_types
            .iter()
            .map(|obj| import_arg_matcher(obj.bind(py)))
            .collect::<Result<Vec<_>, _>>()?;
        let return_type = import_sedona_type(py_return_field.bind(py))?;
        Some(ArgMatcher::new(arg_matchers, return_type))
    } else {
        None
    };

    let kernel_impl = PySedonaScalarKernel {
        matcher,
        py_return_field,
        py_invoke_batch,
    };

    Ok(kernel_impl)
}

#[derive(Debug)]
struct PySedonaScalarKernel {
    matcher: Option<ArgMatcher>,
    py_return_field: PyObject,
    py_invoke_batch: PyObject,
}

impl SedonaScalarKernel for PySedonaScalarKernel {
    fn return_type(&self, _args: &[SedonaType]) -> Result<Option<SedonaType>> {
        Err(PySedonaError::SedonaPython("Unexpected call to return_type()".to_string()).into())
    }

    fn return_type_from_args_and_scalars(
        &self,
        args: &[SedonaType],
        scalar_args: &[Option<&ScalarValue>],
    ) -> Result<Option<SedonaType>> {
        if let Some(matcher) = &self.matcher {
            let return_type = matcher.match_args(args)?;
            return Ok(return_type);
        }

        let return_type = Python::with_gil(|py| -> Result<Option<SedonaType>, PySedonaError> {
            let py_sedona_types = args
                .iter()
                .map(|arg| -> Result<_, PySedonaError> { Ok(PySedonaType::new(arg.clone())) })
                .collect::<Result<Vec<_>, _>>()?;
            let py_scalar_values = zip(&py_sedona_types, scalar_args)
                .map(|(sedona_type, maybe_arg)| {
                    maybe_arg.map(|arg| PySedonaValue {
                        sedona_type: sedona_type.clone(),
                        value: ColumnarValue::Scalar(arg.clone()),
                        num_rows: 1,
                    })
                })
                .collect::<Vec<_>>();

            let py_return_field =
                self.py_return_field
                    .call(py, (py_sedona_types, py_scalar_values), None)?;
            if py_return_field.is_none(py) {
                return Ok(None);
            }

            let return_type = import_sedona_type(py_return_field.bind(py))?;
            Ok(Some(return_type))
        })?;

        Ok(return_type)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let result = Python::with_gil(|py| -> Result<ArrayRef, PySedonaError> {
            let py_values = zip(arg_types, args)
                .map(|(sedona_type, arg)| PySedonaValue {
                    sedona_type: PySedonaType::new(sedona_type.clone()),
                    value: arg.clone(),
                    num_rows: 0,
                })
                .collect::<Vec<_>>();

            // let expected_return_type = SedonaType::from_storage_field(&args.return_field)?;
            // let py_return_type = PySedonaType::new(expected_return_type.clone());
            let py_return_type = PyNone::get(py);
            let py_args = PyTuple::new(py, py_values)?;

            let result = self
                .py_invoke_batch
                .call(py, (py_args, py_return_type, 0), None)?;

            let (_, result_array) = import_arrow_array(result.bind(py))?;
            // let result_sedona_type = SedonaType::from_storage_field(&result_field)?;

            // if expected_return_type != result_sedona_type {
            //     return Err(PySedonaError::SedonaPython(format!(
            //         "Expected {expected_return_type} but got {result_sedona_type}"
            //     )));
            // }

            Ok(result_array)
        })?;

        if args.is_empty() {
            return Ok(ColumnarValue::Array(result));
        }

        for arg in args {
            match arg {
                ColumnarValue::Array(_) => return Ok(ColumnarValue::Array(result)),
                ColumnarValue::Scalar(_) => {}
            }
        }

        Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
            &result, 0,
        )?))
    }
}

#[pyclass]
#[derive(Debug)]
pub struct PySedonaValue {
    pub sedona_type: PySedonaType,
    pub value: ColumnarValue,
    pub num_rows: usize,
}

#[pymethods]
impl PySedonaValue {
    #[getter]
    fn r#type(&self) -> Result<PySedonaType, PySedonaError> {
        Ok(self.sedona_type.clone())
    }

    fn is_scalar(&self) -> bool {
        matches!(&self.value, ColumnarValue::Scalar(_))
    }

    fn to_array(&self) -> Result<Self, PySedonaError> {
        Ok(PySedonaValue {
            sedona_type: self.sedona_type.clone(),
            value: ColumnarValue::Array(self.value.to_array(self.num_rows)?),
            num_rows: self.num_rows,
        })
    }

    fn __arrow_c_schema__<'py>(
        &self,
        py: Python<'py>,
    ) -> Result<Bound<'py, PyCapsule>, PySedonaError> {
        let schema_capsule_name = CString::new("arrow_schema").unwrap();
        let storage_field = self.sedona_type.inner.to_storage_field("", true)?;
        let ffi_schema = FFI_ArrowSchema::try_from(storage_field)?;
        Ok(PyCapsule::new(py, ffi_schema, Some(schema_capsule_name))?)
    }

    fn __arrow_c_array__<'py>(
        &self,
        py: Python<'py>,
        requsted_schema: PyObject
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
        format!(
            "PySedonaValue {}[{}]",
            self.sedona_type.inner, self.num_rows
        )
    }
}
