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

//! PyO3 shim around `sedona-raster-zarr`. Exposes `PyZarrChunkReader`,
//! a single-use `__arrow_c_stream__` wrapper consumed by the
//! Python-side `ZarrFormatSpec`.

use std::ffi::CString;
use std::sync::{Arc, Mutex, OnceLock};

use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use object_store::ObjectStore;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use sedona_raster_zarr::{open_storage_from_uri, ZarrChunkReader};
use tokio::runtime::{Builder, Runtime};
use url::Url;

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
        let store =
            default_object_store_for_uri(uri).map_err(|e| PyValueError::new_err(e.to_string()))?;
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

/// Build an `Arc<dyn ObjectStore>` for `uri` using env-var credential
/// discovery, per scheme. This is the temporary bridge that lets the
/// Python FFI work without going through `con.read_format`; it
/// reaches into `AWS_*` environment variables via the `object_store`
/// per-backend `Builder::from_env` helpers.
///
/// Only the schemes exercised by the crate's tests are wired up (`s3`,
/// `http`/`https`); `gcs`/`azure` are intentionally omitted until there
/// is coverage for them.
///
/// **Slated for removal.** When the host wheel surfaces
/// `args.src.store` to `open_reader` via the `FFI_ObjectStore`
/// capsule machinery (in progress — see
/// <https://github.com/apache/sedona-db/pull/890>), this function (and
/// the `object_store` `aws`/`http` features that back it)
/// gets deleted — the store will arrive credentialed from the host's
/// `ObjectStoreRegistry` and `PyZarrChunkReader::new` will extract it
/// from the capsule instead. For `file://` and bare paths the
/// returned store is a no-op placeholder; the loader uses
/// `FilesystemStore` directly for the local case.
fn default_object_store_for_uri(uri: &str) -> Result<Arc<dyn ObjectStore>, PyErr> {
    // file:// and bare paths use a LocalFileSystem rooted at `/`;
    // open_storage_from_uri prefixes it at the group's path. This
    // matches the store the host's ObjectStoreRegistry yields for
    // file://, so the loader treats local and cloud uniformly.
    if uri.starts_with("file://") || !uri.contains("://") {
        return Ok(Arc::new(object_store::local::LocalFileSystem::new()));
    }
    let url = Url::parse(uri)
        .map_err(|e| PyValueError::new_err(format!("group URI {uri:?} is not a valid URL: {e}")))?;
    match url.scheme().to_ascii_lowercase().as_str() {
        "s3" => {
            use object_store::aws::AmazonS3Builder;
            let store = AmazonS3Builder::from_env()
                .with_url(uri)
                .build()
                .map_err(|e| PyValueError::new_err(build_err("s3", uri, e)))?;
            Ok(Arc::new(store))
        }
        "http" | "https" => {
            use object_store::http::HttpBuilder;
            // open_storage_from_uri applies the path as a PrefixStore,
            // so the HttpStore must be rooted at scheme+authority only
            // — unlike S3, HttpBuilder roots at whatever URL it's given,
            // so hand it the authority without the path.
            let authority = format!("{}://{}", url.scheme(), url.authority());
            let store = HttpBuilder::new()
                .with_url(authority)
                .build()
                .map_err(|e| PyValueError::new_err(build_err("http", uri, e)))?;
            Ok(Arc::new(store))
        }
        other => Err(PyValueError::new_err(format!(
            "unsupported Zarr URI scheme {other:?}; expected one of: \
             file, s3, http, https"
        ))),
    }
}

fn build_err(backend: &str, uri: &str, err: object_store::Error) -> String {
    format!(
        "failed to build {backend} object_store for {uri}: {err}. \
         Provide credentials via standard environment variables \
         (e.g. AWS_ACCESS_KEY_ID/AWS_REGION for s3)."
    )
}

#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyZarrChunkReader>()?;
    Ok(())
}
