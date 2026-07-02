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

use std::sync::Arc;

use arrow_buffer::Buffer;
use arrow_schema::ArrowError;
use async_trait::async_trait;
use bytes::Bytes;
use pyo3::{
    buffer::PyBuffer,
    pyclass, pyfunction, pymethods,
    types::{PyAnyMethods, PyList, PyListMethods},
    Py, PyAny, Python,
};
use sedona_raster::{
    raster_loader::{AsyncRasterLoader, RasterLoadRequest, RasterLoadResult},
    view_entries::ViewEntry,
};
use sedona_schema::raster::BandDataType;

/// Python-backed raster loader.
///
/// This struct wraps Python callables that implement the raster loading logic.
/// The Python class should provide:
/// - `name() -> str`: Returns the loader name for diagnostics
/// - `supports_format(format: Optional[str]) -> bool`: Whether this loader handles the format
/// - `load(requests: List[PyRasterLoadRequest]) -> List[PyRasterLoadResult]`: Load the raster data
#[derive(Debug)]
pub struct PyRasterLoader {
    /// Cached loader name (called once at construction)
    name: String,
    /// Python callable that checks if a format is supported
    py_supports_format: Py<PyAny>,
    /// Python callable that loads raster data
    py_load: Py<PyAny>,
}

impl PyRasterLoader {
    pub fn new(py_name: Py<PyAny>, py_supports_format: Py<PyAny>, py_load: Py<PyAny>) -> Self {
        // Call py_name() once and cache the result
        let name = Python::attach(|py| {
            py_name
                .call(py, (), None)
                .and_then(|obj| obj.extract::<String>(py))
                .unwrap_or_else(|_| "python".to_string())
        });

        Self {
            name,
            py_supports_format,
            py_load,
        }
    }
}

#[async_trait]
impl AsyncRasterLoader for PyRasterLoader {
    fn name(&self) -> &str {
        &self.name
    }

    fn supports_format(&self, format: Option<&str>) -> bool {
        Python::attach(|py| {
            let result = self
                .py_supports_format
                .call(py, (format,), None)
                .and_then(|obj| obj.extract::<bool>(py));
            result.unwrap_or(false)
        })
    }

    async fn load(&self, reqs: &[&RasterLoadRequest]) -> Result<Vec<RasterLoadResult>, ArrowError> {
        // Convert requests to owned data for the blocking task
        let owned_requests: Vec<OwnedPyLoadRequest> = reqs
            .iter()
            .map(|req| OwnedPyLoadRequest {
                uri: req.uri.to_string(),
                dim_names: req.dim_names.iter().map(|s| s.to_string()).collect(),
                source_shape: req.source_shape.to_vec(),
                view: req.view.to_vec(),
                data_type: req.data_type,
            })
            .collect();

        let py_load = Python::attach(|py| self.py_load.clone_ref(py));

        // Spawn a blocking task to hold the GIL and call Python
        let results = tokio::task::spawn_blocking(move || {
            Python::attach(|py| -> Result<Vec<PyRasterLoadResultData>, ArrowError> {
                // Helper to convert pyo3 errors to ArrowError
                fn py_err(e: impl std::fmt::Display) -> ArrowError {
                    ArrowError::InvalidArgumentError(format!("Python raster loader error: {e}"))
                }

                // Convert owned requests to Python objects
                let py_requests: Vec<PyRasterLoadRequest> = owned_requests
                    .iter()
                    .map(|req| PyRasterLoadRequest {
                        uri: req.uri.clone(),
                        dim_names: req.dim_names.clone(),
                        source_shape: req.source_shape.clone(),
                        view: req.view.iter().map(|v| PyViewEntry::from(*v)).collect(),
                        data_type: PyBandDataType(req.data_type),
                    })
                    .collect();

                let py_requests_list = PyList::new(py, py_requests).map_err(py_err)?;

                // Call the Python load function
                let result = py_load
                    .call(py, (py_requests_list,), None)
                    .map_err(py_err)?;
                let result_list: &pyo3::Bound<'_, PyList> =
                    result.bind(py).cast().map_err(py_err)?;

                // Extract results
                let mut results = Vec::new();
                for item in result_list.iter() {
                    let bytes_obj = item.getattr("bytes").map_err(py_err)?;

                    // Zero-copy: wrap the Python buffer with a wrapper that keeps it alive
                    // Accepts any object implementing the buffer protocol (bytes, memoryview, numpy array, etc.)
                    let py_buffer = PyBufferWrapper::new(&bytes_obj)?;
                    let bytes = Bytes::from_owner(py_buffer);

                    let source_shape: Vec<i64> = item
                        .getattr("source_shape")
                        .map_err(py_err)?
                        .extract()
                        .map_err(py_err)?;
                    let view_attr = item.getattr("view").map_err(py_err)?;
                    let view_list: &pyo3::Bound<'_, PyList> = view_attr.cast().map_err(py_err)?;

                    let mut view = Vec::new();
                    for v in view_list.iter() {
                        let source_axis: i64 = v
                            .getattr("source_axis")
                            .map_err(py_err)?
                            .extract()
                            .map_err(py_err)?;
                        let start: i64 = v
                            .getattr("start")
                            .map_err(py_err)?
                            .extract()
                            .map_err(py_err)?;
                        let step: i64 = v
                            .getattr("step")
                            .map_err(py_err)?
                            .extract()
                            .map_err(py_err)?;
                        let steps: i64 = v
                            .getattr("steps")
                            .map_err(py_err)?
                            .extract()
                            .map_err(py_err)?;
                        view.push(ViewEntry {
                            source_axis,
                            start,
                            step,
                            steps,
                        });
                    }

                    results.push(PyRasterLoadResultData {
                        bytes,
                        source_shape,
                        view,
                    });
                }

                Ok(results)
            })
        })
        .await
        .map_err(|e| ArrowError::ExternalError(Box::new(e)))??;

        if results.len() != reqs.len() {
            return Err(ArrowError::InvalidArgumentError(format!(
                "Python raster loader '{}' returned {} result(s) for {} request(s)",
                self.name(),
                results.len(),
                reqs.len()
            )));
        }

        // Convert to RasterLoadResult
        let load_results = results
            .into_iter()
            .map(|r| RasterLoadResult {
                bytes: Buffer::from(r.bytes),
                source_shape: r.source_shape,
                view: r.view,
            })
            .collect();

        Ok(load_results)
    }
}

/// Owned version of RasterLoadRequest for passing to blocking task
struct OwnedPyLoadRequest {
    uri: String,
    dim_names: Vec<String>,
    source_shape: Vec<i64>,
    view: Vec<ViewEntry>,
    data_type: BandDataType,
}

/// Intermediate result data extracted from Python (zero-copy for bytes)
struct PyRasterLoadResultData {
    bytes: Bytes,
    source_shape: Vec<i64>,
    view: Vec<ViewEntry>,
}

/// Zero-copy wrapper around Python buffer protocol objects.
///
/// This struct keeps the Python buffer alive and provides access to its
/// memory without copying. The pointer and length are cached at construction
/// time so that `AsRef<[u8]>` doesn't require the GIL.
///
/// # Safety
/// - Only accepts read-only buffers to ensure the memory won't be mutated
/// - The pointer remains valid as long as the `PyBuffer<u8>` is alive
/// - `PyBuffer<u8>` is already Send + Sync and handles GIL acquisition on Drop
struct PyBufferWrapper {
    /// The Python buffer handle - keeps the source object alive
    _buffer: PyBuffer<u8>,
    /// Cached pointer to the buffer data
    ptr: *const u8,
    /// Cached length of the buffer data
    len: usize,
}

impl PyBufferWrapper {
    /// Create a new PyBufferWrapper from any Python object implementing the buffer protocol.
    ///
    /// The Python GIL must be held when calling this function.
    /// Returns an error if the buffer is not C-contiguous.
    fn new(obj: &pyo3::Bound<'_, pyo3::types::PyAny>) -> Result<Self, ArrowError> {
        let buffer: PyBuffer<u8> = PyBuffer::get(obj).map_err(|e| {
            ArrowError::InvalidArgumentError(format!(
                "Failed to get buffer from Python object: {e}"
            ))
        })?;

        if !buffer.is_c_contiguous() {
            return Err(ArrowError::InvalidArgumentError(
                "Buffer must be C-contiguous".to_string(),
            ));
        }

        // Cache the pointer and length while we have the GIL
        let ptr = buffer.buf_ptr() as *const u8;
        let len = buffer.len_bytes();

        Ok(Self {
            _buffer: buffer,
            ptr,
            len,
        })
    }
}

// Safety: PyBuffer<u8> is already Send + Sync, and we only read from immutable data
unsafe impl Send for PyBufferWrapper {}
unsafe impl Sync for PyBufferWrapper {}

impl AsRef<[u8]> for PyBufferWrapper {
    fn as_ref(&self) -> &[u8] {
        // Safety: ptr and len are valid as long as self.buffer is alive
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// Python-visible raster load request
#[pyclass]
pub struct PyRasterLoadRequest {
    #[pyo3(get)]
    pub uri: String,
    #[pyo3(get)]
    pub dim_names: Vec<String>,
    #[pyo3(get)]
    pub source_shape: Vec<i64>,
    #[pyo3(get)]
    pub view: Vec<PyViewEntry>,
    #[pyo3(get)]
    pub data_type: PyBandDataType,
}

#[pymethods]
impl PyRasterLoadRequest {
    fn __repr__(&self) -> String {
        format!(
            "PyRasterLoadRequest(uri={:?}, source_shape={:?})",
            self.uri, self.source_shape
        )
    }
}

/// Python-visible view entry
#[pyclass(from_py_object)]
#[derive(Clone, Copy)]
pub struct PyViewEntry {
    #[pyo3(get)]
    pub source_axis: i64,
    #[pyo3(get)]
    pub start: i64,
    #[pyo3(get)]
    pub step: i64,
    #[pyo3(get)]
    pub steps: i64,
}

#[pymethods]
impl PyViewEntry {
    #[new]
    fn new(source_axis: i64, start: i64, step: i64, steps: i64) -> Self {
        Self {
            source_axis,
            start,
            step,
            steps,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PyViewEntry(source_axis={}, start={}, step={}, steps={})",
            self.source_axis, self.start, self.step, self.steps
        )
    }
}

impl From<ViewEntry> for PyViewEntry {
    fn from(v: ViewEntry) -> Self {
        Self {
            source_axis: v.source_axis,
            start: v.start,
            step: v.step,
            steps: v.steps,
        }
    }
}

impl From<PyViewEntry> for ViewEntry {
    fn from(v: PyViewEntry) -> Self {
        Self {
            source_axis: v.source_axis,
            start: v.start,
            step: v.step,
            steps: v.steps,
        }
    }
}

/// Python-visible band data type wrapper
#[pyclass(skip_from_py_object)]
#[derive(Clone, Copy)]
pub struct PyBandDataType(pub BandDataType);

#[pymethods]
impl PyBandDataType {
    #[getter]
    fn name(&self) -> &str {
        match self.0 {
            BandDataType::Int8 => "int8",
            BandDataType::UInt8 => "uint8",
            BandDataType::Int16 => "int16",
            BandDataType::UInt16 => "uint16",
            BandDataType::Int32 => "int32",
            BandDataType::UInt32 => "uint32",
            BandDataType::Int64 => "int64",
            BandDataType::UInt64 => "uint64",
            BandDataType::Float32 => "float32",
            BandDataType::Float64 => "float64",
        }
    }

    #[getter]
    fn byte_size(&self) -> usize {
        self.0.byte_size()
    }

    fn __repr__(&self) -> String {
        format!("PyBandDataType({})", self.name())
    }
}

/// Python-visible raster load result
#[pyclass]
pub struct PyRasterLoadResult {
    #[pyo3(get, set)]
    pub bytes: Vec<u8>,
    #[pyo3(get, set)]
    pub source_shape: Vec<i64>,
    #[pyo3(get, set)]
    pub view: Vec<PyViewEntry>,
}

#[pymethods]
impl PyRasterLoadResult {
    #[new]
    fn new(bytes: Vec<u8>, source_shape: Vec<i64>, view: Vec<PyViewEntry>) -> Self {
        Self {
            bytes,
            source_shape,
            view,
        }
    }

    /// Create an unresolved result (full source, view left as-is)
    #[staticmethod]
    fn unresolved(bytes: Vec<u8>, request: &PyRasterLoadRequest) -> Self {
        Self {
            bytes,
            source_shape: request.source_shape.clone(),
            view: request.view.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PyRasterLoadResult(bytes_len={}, source_shape={:?})",
            self.bytes.len(),
            self.source_shape
        )
    }
}

/// Create a Python-backed raster loader
#[pyfunction]
pub fn py_raster_loader(
    py_name: Py<PyAny>,
    py_supports_format: Py<PyAny>,
    py_load: Py<PyAny>,
) -> PyRasterLoaderWrapper {
    PyRasterLoaderWrapper {
        inner: Arc::new(PyRasterLoader::new(py_name, py_supports_format, py_load)),
    }
}

/// Wrapper to expose PyRasterLoader to Python for registration
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PyRasterLoaderWrapper {
    pub inner: Arc<PyRasterLoader>,
}

#[pymethods]
impl PyRasterLoaderWrapper {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn supports_format(&self, format: Option<&str>) -> bool {
        self.inner.supports_format(format)
    }

    fn __repr__(&self) -> String {
        format!("PyRasterLoaderWrapper(name={})", self.inner.name())
    }
}
