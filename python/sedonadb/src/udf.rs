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

use std::{iter::zip, sync::Arc};

use arrow_array::{
    ffi::{FFI_ArrowArray, FFI_ArrowSchema},
    ArrayRef,
};
use arrow_schema::{Field, FieldRef};
use datafusion_common::config::ConfigOptions;
use datafusion_common::{Result, ScalarValue};
use datafusion_expr::{
    Accumulator, AggregateUDF, AggregateUDFImpl, ColumnarValue, ScalarUDF, ScalarUDFImpl,
    Volatility,
};
use datafusion_ffi::udf::FFI_ScalarUDF;
use pyo3::{
    pyclass, pyfunction, pymethods,
    types::{PyAnyMethods, PyCapsule, PyTuple, PyTupleMethods},
    Bound, Py, PyAny, Python,
};
use sedona_expr::aggregate_udf::{SedonaAccumulator, SedonaAccumulatorRef, SedonaAggregateUDF};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::{
    error::PySedonaError,
    expr::PyExpr,
    import_from::{check_pycapsule, import_arg_matcher, import_arrow_array, import_sedona_type},
    schema::PySedonaType,
};

/// General [ScalarUDF] wrapper
///
/// This wrapper wraps a de-Sedona-fied scalar function. Sedona scalar functions
/// implement overloading such that registration adds to existing overloads instead
/// of replacing a function with a given name. This struct is primarily used to
/// generate an Expr scalar function call.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct PyScalarUdf {
    pub inner: Arc<ScalarUDF>,
}

#[pymethods]
impl PyScalarUdf {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn __repr__(&self) -> String {
        self.inner.name().to_string()
    }

    fn call(&self, args: Vec<PyExpr>) -> Result<PyExpr, PySedonaError> {
        let expr_args = args.iter().map(|arg| arg.inner.clone()).collect();
        let result = self.inner.call(expr_args);
        Ok(PyExpr::new(result))
    }

    fn __datafusion_scalar_udf__<'py>(
        &self,
        py: Python<'py>,
    ) -> Result<Bound<'py, PyCapsule>, PySedonaError> {
        let ffi_scalar_udf = FFI_ScalarUDF::from(self.inner.clone());
        Ok(PyCapsule::new_with_value(
            py,
            ffi_scalar_udf,
            c"datafusion_scalar_udf",
        )?)
    }
}

/// General [AggregateUDF] wrapper
///
/// This wrapper wraps a de-Sedona-fied aggregate function. Sedona aggregate functions
/// implement overloading such that registration adds to existing overloads instead
/// of replacing a function with a given name. This struct is primarily used to
/// generate an Expr aggregate function call.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct PyAggregateUdf {
    pub inner: Arc<AggregateUDF>,
}

#[pymethods]
impl PyAggregateUdf {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn __repr__(&self) -> String {
        self.inner.name().to_string()
    }

    fn call(&self, args: Vec<PyExpr>) -> Result<PyExpr, PySedonaError> {
        let expr_args = args.iter().map(|arg| arg.inner.clone()).collect();
        let result = self.inner.call(expr_args);
        Ok(PyExpr::new(result))
    }
}

/// SedonaScalarUdf wrapper
///
/// This wrapper wraps something that is very specifically a *SedonaScalarUDF*. These
/// UDFs implement overloading such that multiple implementations of a single function
/// can be combined instead of replaced. This struct is primarily used to represent
/// a Python-implemented user-defined function before registration.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PySedonaScalarUdf {
    pub inner: SedonaScalarUDF,
}

#[pymethods]
impl PySedonaScalarUdf {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn __repr__(&self) -> String {
        self.inner.name().to_string()
    }

    fn call(&self, args: Vec<PyExpr>) -> Result<PyExpr, PySedonaError> {
        let expr_args = args.iter().map(|arg| arg.inner.clone()).collect();
        let scalar_udf: ScalarUDF = self.inner.clone().into();
        let result = scalar_udf.call(expr_args);
        Ok(PyExpr::new(result))
    }

    fn __datafusion_scalar_udf__<'py>(
        &self,
        py: Python<'py>,
    ) -> Result<Bound<'py, PyCapsule>, PySedonaError> {
        let scalar_udf: ScalarUDF = self.inner.clone().into();
        let ffi_scalar_udf = FFI_ScalarUDF::from(Arc::new(scalar_udf));
        Ok(PyCapsule::new_with_value(
            py,
            ffi_scalar_udf,
            c"datafusion_scalar_udf",
        )?)
    }
}

/// Parse a Python-supplied volatility string into a [Volatility].
fn parse_volatility(volatility: &str) -> Result<Volatility, PySedonaError> {
    match volatility {
        "immutable" => Ok(Volatility::Immutable),
        "stable" => Ok(Volatility::Stable),
        "volatile" => Ok(Volatility::Volatile),
        v => Err(PySedonaError::SedonaPython(format!(
            "Expected one of 'immutable', 'stable', or 'volatile' but got '{v}'"
        ))),
    }
}

/// Validate that an array imported from a Python UDF has the expected logical
/// type, accepting either the exact [SedonaType] or its bare storage type.
fn validate_imported_type(
    expected: &SedonaType,
    actual: &SedonaType,
    context: &str,
) -> Result<(), PySedonaError> {
    if expected != actual {
        let expected_storage = SedonaType::Arrow(expected.storage_type().clone());
        if &expected_storage != actual {
            return Err(PySedonaError::SedonaPython(format!(
                "Expected {context} to return array of type {expected} or its storage but got {actual}"
            )));
        }
    }
    Ok(())
}

#[pyfunction]
pub fn sedona_scalar_udf<'py>(
    py: Python<'py>,
    py_invoke_batch: Py<PyAny>,
    py_return_type: Py<PyAny>,
    py_input_types: Option<Vec<Py<PyAny>>>,
    volatility: &str,
    name: &str,
) -> Result<PySedonaScalarUdf, PySedonaError> {
    let volatility = parse_volatility(volatility)?;

    let scalar_kernel = sedona_scalar_kernel(py, py_input_types, py_return_type, py_invoke_batch)?;
    let sedona_scalar_udf = SedonaScalarUDF::new(name, vec![Arc::new(scalar_kernel)], volatility);

    Ok(PySedonaScalarUdf {
        inner: sedona_scalar_udf,
    })
}

fn sedona_scalar_kernel<'py>(
    py: Python<'py>,
    input_types: Option<Vec<Py<PyAny>>>,
    py_return_field: Py<PyAny>,
    py_invoke_batch: Py<PyAny>,
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
    py_return_field: Py<PyAny>,
    py_invoke_batch: Py<PyAny>,
}

impl SedonaScalarKernel for PySedonaScalarKernel {
    fn return_type(&self, _args: &[SedonaType]) -> Result<Option<SedonaType>> {
        Err(PySedonaError::SedonaPython("Unexpected call to return_type()".to_string()).into())
    }

    fn invoke_batch(
        &self,
        _arg_types: &[SedonaType],
        _args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        Err(PySedonaError::SedonaPython("Unexpected call to invoke_batch()".to_string()).into())
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

        let return_type = Python::attach(|py| -> Result<Option<SedonaType>, PySedonaError> {
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

    fn invoke_batch_from_args(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
        return_type: &SedonaType,
        num_rows: usize,
        _config_options: Option<&ConfigOptions>,
    ) -> Result<ColumnarValue> {
        let result = Python::attach(|py| -> Result<ArrayRef, PySedonaError> {
            let py_values = zip(arg_types, args)
                .map(|(sedona_type, arg)| PySedonaValue {
                    sedona_type: PySedonaType::new(sedona_type.clone()),
                    value: arg.clone(),
                    num_rows,
                })
                .collect::<Vec<_>>();

            let py_return_type = PySedonaType::new(return_type.clone());
            let py_args = PyTuple::new(py, py_values)?;

            let result =
                self.py_invoke_batch
                    .call(py, (py_args, py_return_type, num_rows), None)?;
            let result_bound = result.bind(py);
            if !result_bound.hasattr("__arrow_c_array__")? {
                return Err(
                    PySedonaError::SedonaPython(
                        "Expected result of user-defined function to return an object implementing __arrow_c_array__()".to_string()
                    )
                );
            }

            let (result_field, result_array) = import_arrow_array(result_bound)?;
            let result_sedona_type = SedonaType::from_storage_field(&result_field)?;

            validate_imported_type(
                return_type,
                &result_sedona_type,
                "result of user-defined function",
            )?;

            if result_array.len() != num_rows {
                return Err(PySedonaError::SedonaPython(format!(
                    "Expected result of user-defined function to return array of length {num_rows} but got {}",
                    result_array.len()
                )));
            }

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

    #[getter]
    fn storage(&self) -> Result<Self, PySedonaError> {
        Ok(PySedonaValue {
            sedona_type: PySedonaType {
                inner: SedonaType::Arrow(self.sedona_type.inner.storage_type().clone()),
            },
            value: self.value.clone(),
            num_rows: self.num_rows,
        })
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
        let storage_field = self.sedona_type.inner.to_storage_field("", true)?;
        let ffi_schema = FFI_ArrowSchema::try_from(storage_field)?;
        Ok(PyCapsule::new_with_value(py, ffi_schema, c"arrow_schema")?)
    }

    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_array__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<PyCapsule>>,
    ) -> Result<(Bound<'py, PyCapsule>, Bound<'py, PyCapsule>), PySedonaError> {
        if let Some(requested_schema) = requested_schema {
            let ffi_requested_schema = unsafe {
                FFI_ArrowSchema::from_raw(check_pycapsule(&requested_schema, "arrow_schema")? as _)
            };
            let requested_type =
                SedonaType::from_storage_field(&Field::try_from(&ffi_requested_schema)?)?;
            if requested_type != self.sedona_type.inner {
                return Err(PySedonaError::SedonaPython(
                    "requested type is not implemented for PySedonaValue".to_string(),
                ));
            }
        }

        let field = self.sedona_type.inner.to_storage_field("", true)?;
        let ffi_schema = FFI_ArrowSchema::try_from(&field)?;

        let out_size = match &self.value {
            ColumnarValue::Array(array) => array.len(),
            ColumnarValue::Scalar(_) => 1,
        };
        let array = self.value.to_array(out_size)?;
        let ffi_array = FFI_ArrowArray::new(&array.to_data());

        Ok((
            PyCapsule::new_with_value(py, ffi_schema, c"arrow_schema")?,
            PyCapsule::new_with_value(py, ffi_array, c"arrow_array")?,
        ))
    }

    fn __repr__(&self) -> String {
        let label = match &self.value {
            ColumnarValue::Array(_) => "Array",
            ColumnarValue::Scalar(_) => "Scalar",
        };

        format!(
            "PySedonaValue {label} {}[{}]",
            self.sedona_type.inner, self.num_rows
        )
    }
}

/// SedonaAggregateUdf wrapper for Python-implemented aggregate UDFs.
///
/// Parallel to [PySedonaScalarUdf]: holds a SedonaAggregateUDF that wraps a
/// Python class which produces accumulator instances. Registration goes
/// through `__sedona_internal_udf__()` on the Python side; `InternalContext`
/// recognizes the capsule and routes to `insert_aggregate_udf`.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PySedonaAggregateUdf {
    pub inner: SedonaAggregateUDF,
}

#[pymethods]
impl PySedonaAggregateUdf {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn __repr__(&self) -> String {
        self.inner.name().to_string()
    }

    fn call(&self, args: Vec<PyExpr>) -> Result<PyExpr, PySedonaError> {
        let expr_args = args.iter().map(|arg| arg.inner.clone()).collect();
        let aggregate_udf: AggregateUDF = self.inner.clone().into();
        let result = aggregate_udf.call(expr_args);
        Ok(PyExpr::new(result))
    }
}

#[pyfunction]
pub fn sedona_aggregate_udf<'py>(
    py: Python<'py>,
    py_factory: Py<PyAny>,
    py_return_type: Py<PyAny>,
    py_input_types: Vec<Py<PyAny>>,
    py_state_types: Vec<Py<PyAny>>,
    volatility: &str,
    name: &str,
) -> Result<PySedonaAggregateUdf, PySedonaError> {
    let volatility = parse_volatility(volatility)?;

    let arg_matchers = py_input_types
        .iter()
        .map(|obj| import_arg_matcher(obj.bind(py)))
        .collect::<Result<Vec<_>, _>>()?;
    let return_type = import_sedona_type(py_return_type.bind(py))?;
    let matcher = ArgMatcher::new(arg_matchers, return_type);

    let state_types = py_state_types
        .iter()
        .map(|obj| import_sedona_type(obj.bind(py)))
        .collect::<Result<Vec<_>, _>>()?;

    let kernel = PySedonaAggregateKernel {
        matcher,
        state_types,
        py_factory,
    };
    let kernel_ref: SedonaAccumulatorRef = Arc::new(kernel);
    let sedona_aggregate_udf = SedonaAggregateUDF::new(name, vec![kernel_ref], volatility);

    Ok(PySedonaAggregateUdf {
        inner: sedona_aggregate_udf,
    })
}

#[derive(Debug)]
struct PySedonaAggregateKernel {
    matcher: ArgMatcher,
    state_types: Vec<SedonaType>,
    py_factory: Py<PyAny>,
}

impl SedonaAccumulator for PySedonaAggregateKernel {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        self.matcher.match_args(args)
    }

    fn accumulator(
        &self,
        args: &[SedonaType],
        output_type: &SedonaType,
    ) -> Result<Box<dyn Accumulator>> {
        let instance = Python::attach(|py| -> Result<Py<PyAny>, PySedonaError> {
            Ok(self.py_factory.call0(py)?)
        })?;
        Ok(Box::new(PySedonaAccumulator {
            instance,
            input_types: args.to_vec(),
            state_types: self.state_types.clone(),
            output_type: output_type.clone(),
        }))
    }

    fn state_fields(&self, _args: &[SedonaType]) -> Result<Vec<FieldRef>> {
        self.state_types
            .iter()
            .enumerate()
            .map(|(i, t)| Ok(Arc::new(t.to_storage_field(&format!("state_{i}"), true)?)))
            .collect()
    }
}

#[derive(Debug)]
struct PySedonaAccumulator {
    instance: Py<PyAny>,
    input_types: Vec<SedonaType>,
    state_types: Vec<SedonaType>,
    output_type: SedonaType,
}

impl PySedonaAccumulator {
    /// Build a tuple of PySedonaValue (Array variant) from raw Arrow arrays
    /// for handoff into a Python method call.
    fn arrays_to_py_values<'py>(
        py: Python<'py>,
        types: &[SedonaType],
        arrays: &[ArrayRef],
    ) -> Result<Bound<'py, PyTuple>, PySedonaError> {
        if types.len() != arrays.len() {
            return Err(PySedonaError::SedonaPython(format!(
                "Internal aggregate UDF bridge: expected {} arrays, got {}",
                types.len(),
                arrays.len()
            )));
        }
        let values: Vec<_> = zip(types, arrays)
            .map(|(t, a)| PySedonaValue {
                sedona_type: PySedonaType::new(t.clone()),
                value: ColumnarValue::Array(a.clone()),
                num_rows: a.len(),
            })
            .collect();
        Ok(PyTuple::new(py, values)?)
    }

    /// Pull a ScalarValue out of a Python return value that implements
    /// `__arrow_c_array__()` (a one-element Arrow array). The expected
    /// SedonaType is used to validate the array's logical type.
    fn import_scalar(
        py: Python<'_>,
        result: Py<PyAny>,
        expected: &SedonaType,
        context: &str,
    ) -> Result<ScalarValue, PySedonaError> {
        let result_bound = result.bind(py);
        if !result_bound.hasattr("__arrow_c_array__")? {
            return Err(PySedonaError::SedonaPython(format!(
                "Expected {context} to return an object implementing __arrow_c_array__()"
            )));
        }
        let (field, array) = import_arrow_array(result_bound)?;
        let actual = SedonaType::from_storage_field(&field)?;
        validate_imported_type(expected, &actual, context)?;
        if array.len() != 1 {
            return Err(PySedonaError::SedonaPython(format!(
                "Expected {context} to be a 1-element array; got length {}",
                array.len()
            )));
        }
        Ok(ScalarValue::try_from_array(&array, 0)?)
    }
}

impl Accumulator for PySedonaAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        Python::attach(|py| -> Result<(), PySedonaError> {
            let args = Self::arrays_to_py_values(py, &self.input_types, values)?;
            self.instance.call_method1(py, "update", (args,))?;
            Ok(())
        })?;
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let scalar = Python::attach(|py| -> Result<ScalarValue, PySedonaError> {
            let result = self.instance.call_method0(py, "evaluate")?;
            Self::import_scalar(py, result, &self.output_type, "aggregate UDF evaluate()")
        })?;
        Ok(scalar)
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let scalars = Python::attach(|py| -> Result<Vec<ScalarValue>, PySedonaError> {
            let result = self.instance.call_method0(py, "state")?;
            let result_bound = result.into_bound(py);
            let tuple = result_bound.cast::<PyTuple>().map_err(|_| {
                PySedonaError::SedonaPython(
                    "Expected aggregate UDF state() to return a tuple".to_string(),
                )
            })?;
            if tuple.len() != self.state_types.len() {
                return Err(PySedonaError::SedonaPython(format!(
                    "Expected aggregate UDF state() to return {} elements; got {}",
                    self.state_types.len(),
                    tuple.len()
                )));
            }
            let mut out = Vec::with_capacity(tuple.len());
            for (i, item) in tuple.iter().enumerate() {
                let scalar = Self::import_scalar(
                    py,
                    item.unbind(),
                    &self.state_types[i],
                    "aggregate UDF state() element",
                )?;
                out.push(scalar);
            }
            Ok(out)
        })?;
        Ok(scalars)
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        Python::attach(|py| -> Result<(), PySedonaError> {
            let args = Self::arrays_to_py_values(py, &self.state_types, states)?;
            self.instance.call_method1(py, "merge", (args,))?;
            Ok(())
        })?;
        Ok(())
    }

    fn size(&self) -> usize {
        // Only the Rust-side struct is accounted for; the Python accumulator's
        // heap footprint is opaque to DataFusion's memory limiter. Acceptable
        // for v1 (a conservative under-estimate). A future `mem_used()` hook on
        // the Python class could report the true size.
        std::mem::size_of::<Self>()
    }
}
