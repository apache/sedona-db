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

//! `sedonadb-zarr` — Python plugin package wiring Zarr support into a
//! `sedonadb` session.
//!
//! Two PyO3-exposed surfaces:
//! - `register_udtf(internal_ctx)` — attaches the `sd_read_zarr` SQL
//!   UDTF to a sedonadb session.
//! - `PyZarrChunkReader` — a streaming reader producible from Python,
//!   exposing `__arrow_c_stream__` so it plugs into
//!   `ExternalFormatSpec.open_reader` (the `con.read_format(spec, uri)`
//!   surface).
//!
//! The Python `ZarrFormatSpec(ExternalFormatSpec)` class in
//! `sedonadb_zarr/__init__.py` wraps `PyZarrChunkReader`; the Rust side
//! here is intentionally thin.

use std::ffi::CString;
use std::sync::Mutex;

use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
// `python/sedonadb` compiles its rlib as `_lib` (the lib.name override
// maturin needs). Our own crate also has `lib.name = "_lib"`, which
// would collide with the pymodule below — the leading `::` resolves to
// the extern crate, not our local module.
use ::_lib::context::InternalContext;
use sedona_raster_zarr::ZarrChunkReader;

/// Attach the `sd_read_zarr` SQL UDTF to a sedonadb session.
///
/// Called from `sedonadb_zarr.register(con)`. After this,
/// `con.sql("SELECT * FROM sd_read_zarr(...)")` works.
#[pyfunction]
fn register_udtf(internal_ctx: &Bound<'_, InternalContext>) -> PyResult<()> {
    let ctx = internal_ctx.borrow();
    sedona_raster_zarr::register_udtf(&ctx.inner.ctx);
    Ok(())
}

/// Python-callable wrapper around `ZarrChunkReader` that exposes
/// `__arrow_c_stream__`. Consumed on first call (subsequent calls
/// error) — matching pyarrow's `RecordBatchReader` convention.
///
/// The Python `ZarrFormatSpec.open_reader` returns one of these; the
/// `ExternalFormatSpec` framework then drives the chunk grid via the
/// Arrow C stream protocol.
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
    m.add_function(wrap_pyfunction!(register_udtf, m)?)?;
    m.add_class::<PyZarrChunkReader>()?;
    Ok(())
}
