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
//! - [`ZarrTableFunction`] — Python class with
//!   `__datafusion_table_function__(session)` that returns an
//!   `FFI_TableFunction` PyCapsule. sedonadb's
//!   `InternalContext.register_udtf_capsule` attaches it under
//!   `sd_read_zarr`.
//! - [`PyZarrChunkReader`] — streaming reader producible from Python,
//!   exposing `__arrow_c_stream__` so it plugs into
//!   `ExternalFormatSpec.open_reader` (the `con.read_format(spec, uri)`
//!   surface).
//!
//! The Rust side carries no dependency on `sedonadb`'s host extension
//! — UDTF handoff is via `datafusion-ffi`, which gives an ABI-stable
//! C-level interface so plugin and host cdylibs don't need to share
//! anything beyond Arrow C Data/Stream and the FFI structs.

use std::ffi::CString;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use datafusion_catalog::TableFunctionImpl;
use datafusion_ffi::proto::logical_extension_codec::FFI_LogicalExtensionCodec;
use datafusion_ffi::udtf::FFI_TableFunction;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use sedona_raster_zarr::{ZarrChunkReader, ZarrReadFunction};

/// `c"datafusion_table_function"` capsule name expected by sedonadb's
/// (and datafusion-python's) host-side UDTF registration.
const UDTF_CAPSULE_NAME: &std::ffi::CStr = c"datafusion_table_function";

/// Codec capsule name the host passes to us via the `session` argument.
const CODEC_CAPSULE_NAME: &std::ffi::CStr = c"datafusion_logical_extension_codec";

/// Zarr UDTF surface. The Python plugin instantiates one of these and
/// hands the instance to sedonadb's `con._impl.register_udtf` (Python
/// wrapper). Sedonadb then calls `__datafusion_table_function__(session)`
/// on it to obtain the FFI struct.
#[pyclass(name = "ZarrTableFunction", module = "sedonadb_zarr")]
#[derive(Default, Debug, Clone)]
pub struct ZarrTableFunction;

#[pymethods]
impl ZarrTableFunction {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __datafusion_table_function__<'py>(
        &self,
        py: Python<'py>,
        session: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let codec = ffi_logical_codec_from_pycapsule(&session)?;
        let udtf: Arc<dyn TableFunctionImpl> = Arc::new(ZarrReadFunction::default());
        let provider = FFI_TableFunction::new_with_ffi_codec(udtf, None, codec);
        let name = CString::new(UDTF_CAPSULE_NAME.to_bytes()).unwrap();
        PyCapsule::new(py, provider, Some(name))
    }
}

/// Extract an [`FFI_LogicalExtensionCodec`] from a session capsule the
/// host hands us. Accepts either a raw capsule named
/// `"datafusion_logical_extension_codec"` or any object exposing
/// `__datafusion_logical_extension_codec__()` that returns one.
fn ffi_logical_codec_from_pycapsule(obj: &Bound<'_, PyAny>) -> PyResult<FFI_LogicalExtensionCodec> {
    let attr_name = "__datafusion_logical_extension_codec__";
    let capsule = if obj.hasattr(attr_name)? {
        obj.getattr(attr_name)?.call0()?
    } else {
        obj.clone()
    };
    let capsule = capsule.downcast::<PyCapsule>().map_err(|e| {
        PyValueError::new_err(format!(
            "session capsule must be a PyCapsule (or expose {attr_name}), got {e}"
        ))
    })?;
    let name = capsule
        .name()?
        .ok_or_else(|| PyValueError::new_err("session capsule has no name"))?;
    if name != CODEC_CAPSULE_NAME {
        return Err(PyValueError::new_err(format!(
            "session capsule name mismatch: expected {CODEC_CAPSULE_NAME:?}, got {name:?}"
        )));
    }
    let ptr = capsule.pointer() as *mut FFI_LogicalExtensionCodec;
    let ptr = NonNull::new(ptr)
        .ok_or_else(|| PyValueError::new_err("session capsule pointer is null"))?;
    // SAFETY: capsule name was verified above; PyO3 keeps the value
    // alive for the lifetime of the capsule. `clone()` runs the FFI
    // release-aware clone hook so we end up with an independent codec
    // that outlives this scope.
    let codec = unsafe { ptr.as_ref().clone() };
    Ok(codec)
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
    m.add_class::<ZarrTableFunction>()?;
    m.add_class::<PyZarrChunkReader>()?;
    Ok(())
}
