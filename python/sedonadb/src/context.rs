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
use std::{collections::HashMap, sync::Arc};

use datafusion_expr::ScalarUDFImpl;
use pyo3::prelude::*;
use sedona::context::SedonaContext;
use sedona::context_builder::SedonaContextBuilder;
use tokio::runtime::Runtime;

use crate::{
    dataframe::InternalDataFrame,
    datasource::PyExternalFormat,
    error::PySedonaError,
    import_from::{import_ffi_scalar_udf, import_table_provider_from_any},
    runtime::wait_for_future,
    udf::PySedonaScalarUdf,
};

#[pyclass]
pub struct InternalContext {
    pub inner: SedonaContext,
    pub runtime: Arc<Runtime>,
}

#[pymethods]
impl InternalContext {
    #[new]
    #[pyo3(signature = (options=HashMap::new()))]
    fn new(py: Python, options: HashMap<String, String>) -> Result<Self, PySedonaError> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                PySedonaError::SedonaPython(format!("Failed to build multithreaded runtime: {e}"))
            })?;

        let builder = SedonaContextBuilder::from_options(&options)
            .map_err(|e| PySedonaError::SedonaPython(e.to_string()))?;

        let inner = wait_for_future(py, &runtime, builder.build())??;

        Ok(Self {
            inner,
            runtime: Arc::new(runtime),
        })
    }

    pub fn view<'py>(
        &self,
        py: Python<'py>,
        name: &str,
    ) -> Result<InternalDataFrame, PySedonaError> {
        let df = wait_for_future(py, &self.runtime, self.inner.ctx.table(name))??;
        Ok(InternalDataFrame::new(df, self.runtime.clone()))
    }

    pub fn create_data_frame<'py>(
        &self,
        py: Python<'py>,
        obj: &Bound<PyAny>,
        requested_schema: Option<&Bound<PyAny>>,
    ) -> Result<InternalDataFrame, PySedonaError> {
        let provider = import_table_provider_from_any(py, obj, requested_schema)?;
        let df = self.inner.ctx.read_table(provider)?;
        Ok(InternalDataFrame::new(df, self.runtime.clone()))
    }

    pub fn read_parquet<'py>(
        &self,
        py: Python<'py>,
        table_paths: Vec<String>,
        options: HashMap<String, PyObject>,
        geometry_columns: Option<String>,
        validate: bool,
    ) -> Result<InternalDataFrame, PySedonaError> {
        // Convert Python options to strings, filtering out None values
        let rust_options: HashMap<String, String> = options
            .into_iter()
            .filter_map(|(k, v)| {
                if v.is_none(py) {
                    None
                } else {
                    v.bind(py)
                        .str()
                        .and_then(|s| s.extract())
                        .map(|s: String| (k, s))
                        .ok()
                }
            })
            .collect();

        let mut geo_options =
            sedona_geoparquet::provider::GeoParquetReadOptions::from_table_options(rust_options)
                .map_err(|e| PySedonaError::SedonaPython(format!("Invalid table options: {e}")))?;
        if let Some(geometry_columns) = geometry_columns {
            geo_options = geo_options
                .with_geometry_columns_json(&geometry_columns)
                .map_err(|e| {
                    PySedonaError::SedonaPython(format!("Invalid geometry_columns JSON: {e}"))
                })?;
        }
        geo_options = geo_options.with_validate(validate);

        let df = wait_for_future(
            py,
            &self.runtime,
            self.inner.read_parquet(table_paths, geo_options),
        )??;
        Ok(InternalDataFrame::new(df, self.runtime.clone()))
    }

    pub fn read_external_format<'py>(
        &self,
        py: Python<'py>,
        format_spec: Bound<PyAny>,
        table_paths: Vec<String>,
        check_extension: bool,
    ) -> Result<InternalDataFrame, PySedonaError> {
        let spec = format_spec
            .call_method0("__sedona_external_format__")?
            .extract::<PyExternalFormat>()?;
        let df = wait_for_future(
            py,
            &self.runtime,
            self.inner
                .read_external_format(Arc::new(spec), table_paths, None, check_extension),
        )??;

        Ok(InternalDataFrame::new(df, self.runtime.clone()))
    }

    pub fn sql<'py>(
        &self,
        py: Python<'py>,
        query: &str,
    ) -> Result<InternalDataFrame, PySedonaError> {
        let df = wait_for_future(py, &self.runtime, self.inner.sql(query))??;
        Ok(InternalDataFrame::new(df, self.runtime.clone()))
    }

    pub fn drop_view(&self, table_ref: &str) -> Result<(), PySedonaError> {
        self.inner.ctx.deregister_table(table_ref)?;
        Ok(())
    }

    pub fn scalar_udf(&self, name: &str) -> Result<PySedonaScalarUdf, PySedonaError> {
        if let Some(sedona_scalar_udf) = self.inner.functions.scalar_udf(name) {
            Ok(PySedonaScalarUdf {
                inner: sedona_scalar_udf.clone(),
            })
        } else {
            Err(PySedonaError::SedonaPython(format!(
                "Sedona scalar UDF with name {name} was not found"
            )))
        }
    }

    pub fn register_udf(&mut self, udf: Bound<PyAny>) -> Result<(), PySedonaError> {
        if udf.hasattr("__sedona_internal_udf__")? {
            let py_scalar_udf = udf
                .getattr("__sedona_internal_udf__")?
                .call0()?
                .extract::<PySedonaScalarUdf>()?;
            let name = py_scalar_udf.inner.name();
            self.inner
                .functions
                .insert_scalar_udf(py_scalar_udf.inner.clone());
            self.inner.ctx.register_udf(
                self.inner
                    .functions
                    .scalar_udf(name)
                    .unwrap()
                    .clone()
                    .into(),
            );
            return Ok(());
        } else if udf.hasattr("__datafusion_scalar_udf__")? {
            let scalar_udf = import_ffi_scalar_udf(&udf)?;
            self.inner.ctx.register_udf(scalar_udf);
            return Ok(());
        }

        Err(PySedonaError::SedonaPython(
            "Expected an object implementing __sedona_internal_udf__ or __datafusion_scalar_udf__"
                .to_string(),
        ))
    }

    /// Register a UDTF (table function) whose implementation is passed
    /// through a `PyCapsule` containing an
    /// `Arc<dyn datafusion::catalog::TableFunctionImpl>`.
    ///
    /// Plugin-handoff API: format-specific Python packages (e.g.
    /// `sedonadb-zarr`) build their UDTF in their own PyO3 extension
    /// and pass it across as an opaque capsule, sidestepping the
    /// cross-extension `#[pyclass]` type-id mismatch.
    ///
    /// The capsule must be named `"sedonadb.udtf"` and store an
    /// `Arc<dyn TableFunctionImpl>` as its value. We clone the Arc to
    /// take our own refcount; the capsule retains its own copy until
    /// Python GC drops it.
    pub fn register_udtf_capsule(
        &self,
        name: &str,
        capsule: &Bound<'_, pyo3::types::PyCapsule>,
    ) -> Result<(), PySedonaError> {
        use std::sync::Arc;

        let expected = c"sedonadb.udtf";
        let actual = capsule
            .name()?
            .ok_or_else(|| PySedonaError::SedonaPython("UDTF capsule has no name".to_string()))?;
        if actual != expected {
            return Err(PySedonaError::SedonaPython(format!(
                "UDTF capsule name mismatch: expected {expected:?}, got {actual:?}"
            )));
        }
        let ptr = capsule.pointer() as *const Arc<dyn datafusion::catalog::TableFunctionImpl>;
        if ptr.is_null() {
            return Err(PySedonaError::SedonaPython(
                "UDTF capsule pointer is null".to_string(),
            ));
        }
        // SAFETY: the capsule's payload is an `Arc<dyn TableFunctionImpl>`
        // (validated by name above). PyO3 keeps the value alive for the
        // lifetime of the capsule; we clone the Arc to obtain an
        // independent refcount that outlives this scope.
        let udtf = unsafe { (*ptr).clone() };
        self.inner.ctx.register_udtf(name, udtf);
        Ok(())
    }
}
