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

use pyo3::prelude::*;
use sedona::context::SedonaContext;
use tokio::runtime::Runtime;

use crate::{
    dataframe::InternalDataFrame, error::PySedonaError,
    import_from::import_table_provider_from_any, runtime::wait_for_future,
};

#[pyclass]
pub struct InternalContext {
    pub inner: SedonaContext,
    pub runtime: Arc<Runtime>,
}

#[pymethods]
impl InternalContext {
    #[new]
    fn new(py: Python) -> Result<Self, PySedonaError> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                PySedonaError::SedonaPython(format!("Failed to build multithreaded runtime: {e}"))
            })?;

        let inner = wait_for_future(py, &runtime, SedonaContext::new_local_interactive())??;

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

        let geo_options =
            sedona_geoparquet::provider::GeoParquetReadOptions::from_table_options(rust_options)
                .map_err(|e| PySedonaError::SedonaPython(format!("Invalid table options: {e}")))?;
        let df = wait_for_future(
            py,
            &self.runtime,
            self.inner.read_parquet(table_paths, geo_options),
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
}
