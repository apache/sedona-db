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

//! PyO3 shim around `sedona-raster-zarr`. Exposes:
//! - `PyZarrChunkReader`: a single-use `__arrow_c_stream__` wrapper consumed by
//!    the Python-side `ZarrFormatSpec`.
//! - `ZarrRasterLoader`: a Python class implementing the raster loader interface
//!   that can be registered with `sedonadb` via `py_raster_loader`.

use std::ffi::CString;
use std::sync::{Mutex, OnceLock};

use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use sedona_raster_zarr::{object_store_for_uri, open_storage_from_uri, ZarrChunkReader, ZarrLoader};
use tokio::runtime::{Builder, Runtime};

/// Process-wide tokio runtime backing the sync Python FFI bridge.
///
/// `ZarrChunkReader::try_new` is async, but the Python constructor is sync
/// by contract, so we `block_on` here. Building a runtime per reader is
/// wasteful; one shared runtime serves every open in the package. `next()`
/// on the returned reader is pure CPU and never touches this runtime.
fn shared_runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("build sedonadb-zarr tokio runtime")
    })
}

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
        let store = object_store_for_uri(uri).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let storage =
            open_storage_from_uri(uri, store).map_err(|e| PyValueError::new_err(e.to_string()))?;
        // The crate's async-native loader exposes `try_new` as an
        // `async fn`; the Python FFI is a sync constructor by contract,
        // so we bridge by blocking on the shared package runtime.
        // `next()` on the returned reader is pure CPU and stays synchronous.
        let arrays_ref = arrays.as_deref();
        let reader = shared_runtime()
            .block_on(ZarrChunkReader::try_new(
                storage, uri, arrays_ref, batch_size,
            ))
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

/// Result from a Zarr raster load operation.
///
/// This type matches the interface expected by `py_raster_loader` from `sedonadb`:
/// - `bytes`: raw pixel data
/// - `source_shape`: shape of the returned data
/// - `view`: view entries (identity when returning full source)
#[pyclass]
#[derive(Clone)]
pub struct ZarrLoadResult {
    #[pyo3(get)]
    pub bytes: Vec<u8>,
    #[pyo3(get)]
    pub source_shape: Vec<u64>,
    #[pyo3(get)]
    pub view: Vec<ZarrViewEntry>,
}

#[pymethods]
impl ZarrLoadResult {
    fn __repr__(&self) -> String {
        format!(
            "ZarrLoadResult(bytes_len={}, source_shape={:?})",
            self.bytes.len(),
            self.source_shape
        )
    }
}

/// View entry for Zarr load results.
#[pyclass]
#[derive(Clone, Copy)]
pub struct ZarrViewEntry {
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
impl ZarrViewEntry {
    fn __repr__(&self) -> String {
        format!(
            "ZarrViewEntry(source_axis={}, start={}, step={}, steps={})",
            self.source_axis, self.start, self.step, self.steps
        )
    }
}

/// Zarr-backed raster loader for use with `sedonadb.py_raster_loader`.
///
/// This class implements the raster loader interface that `py_raster_loader`
/// expects. Use it with sedonadb like:
///
/// ```python
/// from sedonadb._lib import py_raster_loader, InternalContext
/// from sedonadb_zarr import ZarrRasterLoader
///
/// loader = ZarrRasterLoader()
/// wrapper = py_raster_loader(loader.name, loader.supports_format, loader.load)
/// ctx = InternalContext({})
/// ctx.register_raster_loader(wrapper)
/// ```
#[pyclass]
pub struct ZarrRasterLoader {
    loader: ZarrLoader,
}

#[pymethods]
impl ZarrRasterLoader {
    #[new]
    fn new() -> Self {
        Self {
            loader: ZarrLoader::new(),
        }
    }

    /// Returns the loader name ("zarr").
    fn name(&self) -> &str {
        use sedona_raster::raster_loader::AsyncRasterLoader;
        self.loader.name()
    }

    /// Check if this loader supports a given format.
    ///
    /// Returns True only for "zarr" format; None and other formats
    /// fall through to GDAL.
    fn supports_format(&self, format: Option<&str>) -> bool {
        use sedona_raster::raster_loader::AsyncRasterLoader;
        self.loader.supports_format(format)
    }

    /// Load raster data from Zarr stores.
    ///
    /// Takes a list of request objects with attributes:
    /// - uri: str
    /// - dim_names: list[str]
    /// - source_shape: list[int]
    /// - view: list[ViewEntry-like]
    /// - data_type: BandDataType-like with .name and .byte_size
    ///
    /// Returns a list of ZarrLoadResult objects.
    fn load(&self, py: Python<'_>, requests: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<ZarrLoadResult>> {
        use sedona_raster::raster_loader::AsyncRasterLoader;
        use sedona_raster::traits::ViewEntry;
        use sedona_schema::raster::BandDataType;

        // Extract data from Python request objects into owned structures
        let mut owned_requests: Vec<OwnedLoadRequest> = Vec::with_capacity(requests.len());
        for req in &requests {
            let uri: String = req.getattr("uri")?.extract()?;
            let dim_names: Vec<String> = req.getattr("dim_names")?.extract()?;
            let source_shape: Vec<u64> = req.getattr("source_shape")?.extract()?;

            // Extract view entries
            let view_list = req.getattr("view")?;
            let mut view = Vec::new();
            for v in view_list.try_iter()? {
                let v = v?;
                let source_axis: i64 = v.getattr("source_axis")?.extract()?;
                let start: i64 = v.getattr("start")?.extract()?;
                let step: i64 = v.getattr("step")?.extract()?;
                let steps: i64 = v.getattr("steps")?.extract()?;
                view.push(ViewEntry {
                    source_axis,
                    start,
                    step,
                    steps,
                });
            }

            // Extract data type
            let data_type_obj = req.getattr("data_type")?;
            let dtype_name: String = data_type_obj.getattr("name")?.extract()?;
            let data_type = match dtype_name.as_str() {
                "int8" => BandDataType::Int8,
                "uint8" => BandDataType::UInt8,
                "int16" => BandDataType::Int16,
                "uint16" => BandDataType::UInt16,
                "int32" => BandDataType::Int32,
                "uint32" => BandDataType::UInt32,
                "int64" => BandDataType::Int64,
                "uint64" => BandDataType::UInt64,
                "float32" => BandDataType::Float32,
                "float64" => BandDataType::Float64,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "Unknown data type: {other}"
                    )))
                }
            };

            owned_requests.push(OwnedLoadRequest {
                uri,
                dim_names,
                source_shape,
                view,
                data_type,
            });
        }

        // Build RasterLoadRequest references
        let dim_names_refs: Vec<Vec<&str>> = owned_requests
            .iter()
            .map(|r| r.dim_names.iter().map(|s| s.as_str()).collect())
            .collect();
        let rust_requests: Vec<sedona_raster::raster_loader::RasterLoadRequest<'_>> =
            owned_requests
                .iter()
                .zip(dim_names_refs.iter())
                .map(|(req, dim_refs)| sedona_raster::raster_loader::RasterLoadRequest {
                    uri: &req.uri,
                    dim_names: dim_refs,
                    source_shape: &req.source_shape,
                    view: &req.view,
                    data_type: req.data_type,
                })
                .collect();

        let request_refs: Vec<&sedona_raster::raster_loader::RasterLoadRequest<'_>> =
            rust_requests.iter().collect();

        // Run the async load on the shared runtime
        let results = py.allow_threads(|| {
            shared_runtime().block_on(async { self.loader.load(&request_refs).await })
        });

        let results = results.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // Convert results to Python objects
        let py_results: Vec<ZarrLoadResult> = results
            .into_iter()
            .map(|r| ZarrLoadResult {
                bytes: r.bytes.to_vec(),
                source_shape: r.source_shape,
                view: r
                    .view
                    .into_iter()
                    .map(|v| ZarrViewEntry {
                        source_axis: v.source_axis,
                        start: v.start,
                        step: v.step,
                        steps: v.steps,
                    })
                    .collect(),
            })
            .collect();

        Ok(py_results)
    }

    fn __repr__(&self) -> String {
        format!("ZarrRasterLoader(name='{}')", self.name())
    }
}

/// Owned version of RasterLoadRequest for temporary storage.
struct OwnedLoadRequest {
    uri: String,
    dim_names: Vec<String>,
    source_shape: Vec<u64>,
    view: Vec<sedona_raster::traits::ViewEntry>,
    data_type: sedona_schema::raster::BandDataType,
}

#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyZarrChunkReader>()?;
    m.add_class::<ZarrRasterLoader>()?;
    m.add_class::<ZarrLoadResult>()?;
    m.add_class::<ZarrViewEntry>()?;
    Ok(())
}
