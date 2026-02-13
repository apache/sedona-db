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
use std::{collections::HashMap, num::NonZeroUsize, path::PathBuf, sync::Arc};

use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::execution::{
    disk_manager::{DiskManagerBuilder, DiskManagerMode},
    memory_pool::TrackConsumersPool,
};
use datafusion_expr::ScalarUDFImpl;
use pyo3::prelude::*;
use sedona::context::SedonaContext;
use sedona::memory_pool::{SedonaFairSpillPool, DEFAULT_UNSPILLABLE_RESERVE_RATIO};
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
    #[pyo3(signature = (memory_limit=None, temp_dir=None, memory_pool_type=None, unspillable_reserve_ratio=None))]
    fn new(
        py: Python,
        memory_limit: Option<usize>,
        temp_dir: Option<String>,
        memory_pool_type: Option<String>,
        unspillable_reserve_ratio: Option<f64>,
    ) -> Result<Self, PySedonaError> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                PySedonaError::SedonaPython(format!("Failed to build multithreaded runtime: {e}"))
            })?;

        let mut rt_builder = RuntimeEnvBuilder::new();
        if let Some(memory_limit) = memory_limit {
            let pool_type = memory_pool_type.as_deref().unwrap_or("fair");
            let pool: Arc<dyn MemoryPool> = match pool_type {
                "fair" => {
                    let unspillable_reserve =
                        unspillable_reserve_ratio.unwrap_or(DEFAULT_UNSPILLABLE_RESERVE_RATIO);
                    let pool = SedonaFairSpillPool::new(memory_limit, unspillable_reserve);
                    Arc::new(TrackConsumersPool::new(
                        pool,
                        NonZeroUsize::new(10).unwrap(),
                    ))
                }
                "greedy" => {
                    let pool = GreedyMemoryPool::new(memory_limit);
                    Arc::new(TrackConsumersPool::new(
                        pool,
                        NonZeroUsize::new(10).unwrap(),
                    ))
                }
                _ => {
                    return Err(PySedonaError::SedonaPython(format!(
                        "Invalid memory pool type: {}",
                        pool_type
                    )))
                }
            };
            rt_builder = rt_builder.with_memory_pool(pool);
        }
        if let Some(temp_dir) = temp_dir {
            let dm_builder = DiskManagerBuilder::default()
                .with_mode(DiskManagerMode::Directories(vec![PathBuf::from(temp_dir)]));
            rt_builder = rt_builder.with_disk_manager_builder(dm_builder);
        }
        let runtime_env = rt_builder.build_arc().map_err(|e| {
            PySedonaError::SedonaPython(format!("Failed to build runtime env: {e}"))
        })?;

        let inner = wait_for_future(
            py,
            &runtime,
            SedonaContext::new_local_interactive_with_runtime_env(runtime_env),
        )??;

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
}
