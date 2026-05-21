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

//! `sedonadb-zarr` — Python plugin exposing Zarr groups to SedonaDB
//! via the `ExternalFormatSpec` + Arrow C Stream contract.

use std::ffi::CString;
use std::sync::Mutex;

use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use sedona_raster_zarr::ZarrChunkReader;

/// Single-use `__arrow_c_stream__` wrapper around `ZarrChunkReader`.
#[pyclass]
pub struct PyZarrChunkReader {
    inner: Mutex<Option<ZarrChunkReader>>,
}

#[pymethods]
impl PyZarrChunkReader {
    #[new]
    #[pyo3(signature = (uri, arrays=None, batch_size=8192))]
    fn new(uri: &str, arrays: Option<Vec<String>>, batch_size: usize) -> PyResult<Self> {
        let reader = ZarrChunkReader::try_new(uri, arrays.as_deref(), batch_size)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Mutex::new(Some(reader)),
        })
    }

    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        #[allow(unused_variables)] requested_schema: Option<Bound<'py, PyCapsule>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let reader = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("PyZarrChunkReader mutex poisoned"))?
            .take()
            .ok_or_else(|| {
                PyRuntimeError::new_err(
                    "PyZarrChunkReader has already been consumed; \
                     a RecordBatchReader can only be exported once.",
                )
            })?;
        let ffi_stream = FFI_ArrowArrayStream::new(Box::new(reader));
        let capsule_name = CString::new("arrow_array_stream")
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        PyCapsule::new(py, ffi_stream, Some(capsule_name))
    }
}

#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyZarrChunkReader>()?;
    Ok(())
}
