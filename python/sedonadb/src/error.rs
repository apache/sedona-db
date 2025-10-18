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
use arrow_schema::ArrowError;
use datafusion_common::DataFusionError;
use pyo3::{create_exception, PyErr};

use thiserror::Error;

create_exception!(sedonadb._lib, SedonaError, pyo3::exceptions::PyException);

#[derive(Error, Debug)]
pub enum PySedonaError {
    #[error("{0}")]
    Py(PyErr),
    #[error("{0}")]
    DF(Box<DataFusionError>),
    #[error("{0}")]
    SedonaPython(String),
}

impl From<PySedonaError> for PyErr {
    fn from(e: PySedonaError) -> Self {
        match e {
            PySedonaError::Py(py_err) => py_err,
            PySedonaError::DF(e) => {
                let msg = e.message().to_string();
                SedonaError::new_err(msg)
            }
            PySedonaError::SedonaPython(msg) => SedonaError::new_err(msg),
        }
    }
}

impl From<DataFusionError> for PySedonaError {
    fn from(other: DataFusionError) -> Self {
        PySedonaError::DF(Box::new(other))
    }
}

impl From<PySedonaError> for DataFusionError {
    fn from(other: PySedonaError) -> Self {
        DataFusionError::External(Box::new(other))
    }
}

impl From<PyErr> for PySedonaError {
    fn from(other: PyErr) -> Self {
        PySedonaError::Py(other)
    }
}

impl From<ArrowError> for PySedonaError {
    fn from(other: ArrowError) -> Self {
        PySedonaError::DF(Box::new(DataFusionError::ArrowError(Box::new(other), None)))
    }
}
