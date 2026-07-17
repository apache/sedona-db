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

use arrow_schema::DataType;
use pyo3::prelude::*;
use sedona::context::SedonaContext;
use sedona::context_builder::SedonaContextBuilder;
use sedona_datasource::format::ExternalFormatFactory;
use tokio::runtime::Runtime;

use crate::{
    dataframe::InternalDataFrame,
    datasource::PyExternalFormat,
    error::PySedonaError,
    import_from::import_table_provider_from_any,
    raster_loader::PyRasterLoaderWrapper,
    runtime::wait_for_future,
    udf::{PyAggregateUdf, PyScalarUdf, PySedonaAggregateUdf, PySedonaScalarUdf},
};

#[pyclass]
pub struct InternalContext {
    pub inner: SedonaContext,
    pub runtime: Arc<Runtime>,
}

/// Convert a Python options dict to a string map for the reader/object-store
/// layer, dropping `None` values (so callers can pass `None` to mean "unset").
fn stringify_options(
    py: Python<'_>,
    options: HashMap<String, Py<PyAny>>,
) -> HashMap<String, String> {
    options
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
        .collect()
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
        options: HashMap<String, Py<PyAny>>,
        geometry_columns: Option<String>,
        validate: bool,
        partitioning: Option<Vec<String>>,
    ) -> Result<InternalDataFrame, PySedonaError> {
        // Convert Python options to strings, filtering out None values
        let rust_options = stringify_options(py, options);

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
        if let Some(partitioning) = partitioning {
            geo_options = geo_options.with_table_partition_cols(
                partitioning
                    .iter()
                    .map(|name| (name.clone(), DataType::Utf8View))
                    .collect(),
            );
        }

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
        partitioning: Option<Vec<String>>,
    ) -> Result<InternalDataFrame, PySedonaError> {
        let spec = format_spec
            .call_method0("__sedonadb_external_format__")?
            .extract::<PyExternalFormat>()?;
        let df = wait_for_future(
            py,
            &self.runtime,
            self.inner.read_external_format(
                Arc::new(spec),
                table_paths,
                None,
                check_extension,
                partitioning.map(|cols| {
                    cols.iter()
                        .map(|name| (name.clone(), DataType::Utf8View))
                        .collect()
                }),
            ),
        )??;

        Ok(InternalDataFrame::new(df, self.runtime.clone()))
    }

    pub fn read_csv<'py>(
        &self,
        py: Python<'py>,
        table_paths: Vec<String>,
        options: HashMap<String, Py<PyAny>>,
        has_header: bool,
        delimiter: &str,
    ) -> Result<InternalDataFrame, PySedonaError> {
        let rust_options = stringify_options(py, options);
        let df = wait_for_future(
            py,
            &self.runtime,
            self.inner
                .read_csv(table_paths, &rust_options, has_header, delimiter),
        )??;
        Ok(InternalDataFrame::new(df, self.runtime.clone()))
    }

    pub fn read_json<'py>(
        &self,
        py: Python<'py>,
        table_paths: Vec<String>,
        options: HashMap<String, Py<PyAny>>,
    ) -> Result<InternalDataFrame, PySedonaError> {
        let rust_options = stringify_options(py, options);
        let df = wait_for_future(
            py,
            &self.runtime,
            self.inner.read_json(table_paths, &rust_options),
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

    pub fn scalar_udf<'py>(
        &self,
        py: Python<'py>,
        name: &str,
    ) -> Result<Bound<'py, PyAny>, PySedonaError> {
        let name_lower = name.to_lowercase();
        let inner = if let Some(sedona_scalar_udf) = self.inner.scalar_udf(&name_lower)? {
            Some(ScalarUdfLookup::Sedona(sedona_scalar_udf))
        } else {
            self.inner
                .ctx
                .state()
                .scalar_functions()
                .get(&name_lower)
                .cloned()
                .map(ScalarUdfLookup::DataFusion)
        };

        if let Some(ScalarUdfLookup::Sedona(sedona_scalar_udf)) = inner {
            Ok(Bound::new(
                py,
                PySedonaScalarUdf {
                    inner: sedona_scalar_udf,
                },
            )?
            .into_any())
        } else if let Some(ScalarUdfLookup::DataFusion(scalar_udf)) = inner {
            Ok(Bound::new(py, PyScalarUdf { inner: scalar_udf })?.into_any())
        } else {
            Err(PySedonaError::SedonaPython(format!(
                "Scalar UDF with name {name} was not found"
            )))
        }
    }

    pub fn aggregate_udf<'py>(
        &self,
        py: Python<'py>,
        name: &str,
    ) -> Result<Bound<'py, PyAny>, PySedonaError> {
        let name_lower = name.to_lowercase();
        let aggregate_udf = self
            .inner
            .ctx
            .state()
            .aggregate_functions()
            .get(&name_lower)
            .cloned();

        if let Some(aggregate_udf) = aggregate_udf {
            Ok(Bound::new(
                py,
                PyAggregateUdf {
                    inner: aggregate_udf,
                },
            )?
            .into_any())
        } else {
            Err(PySedonaError::SedonaPython(format!(
                "Aggregate UDF with name {name} was not found"
            )))
        }
    }

    pub fn list_scalar_udfs(&self) -> Result<Vec<String>, PySedonaError> {
        Ok(self.inner.scalar_udf_names()?)
    }

    pub fn list_aggregate_udfs(&self) -> Result<Vec<String>, PySedonaError> {
        Ok(self
            .inner
            .ctx
            .state()
            .aggregate_functions()
            .keys()
            .cloned()
            .collect())
    }

    pub fn register_component(&self, component: Bound<PyAny>) -> Result<(), PySedonaError> {
        if component.hasattr("__sedonadb_internal_udf__")? {
            let py_scalar_udf = component
                .getattr("__sedonadb_internal_udf__")?
                .call0()?
                .extract::<PySedonaScalarUdf>()?;
            self.inner
                .register_sedona_scalar_udf(py_scalar_udf.inner.clone())?;
            return Ok(());
        } else if component.hasattr("__sedonadb_internal_aggregate_udf__")? {
            let py_agg_udf = component
                .getattr("__sedonadb_internal_aggregate_udf__")?
                .call0()?
                .extract::<PySedonaAggregateUdf>()?;
            self.inner
                .register_sedona_aggregate_udf(py_agg_udf.inner.clone())?;
            return Ok(());
        } else if component.hasattr("__sedonadb_external_format__")? {
            let spec = component
                .call_method0("__sedonadb_external_format__")?
                .extract::<PyExternalFormat>()?;
            let state_ref = self.inner.ctx.state_ref();
            let mut writable = state_ref.write();
            writable
                .register_file_format(Arc::new(ExternalFormatFactory::new(Arc::new(spec))), true)?;
            return Ok(());
        } else if component.hasattr("__sedonadb_raster_loader__")? {
            let wrapper = component
                .call_method0("__sedonadb_raster_loader__")?
                .extract::<PyRasterLoaderWrapper>()?;
            self.inner.register_raster_loader(wrapper.inner);
            return Ok(());
        } else if let Ok(py_raster_loader) = component.extract::<PyRasterLoaderWrapper>() {
            self.inner.register_raster_loader(py_raster_loader.inner);
            return Ok(());
        }

        // A better error is raised in Python before this point
        Err(PySedonaError::SedonaPython(
            "Unsupported object".to_string(),
        ))
    }
}

enum ScalarUdfLookup {
    Sedona(sedona_expr::scalar_udf::SedonaScalarUDF),
    DataFusion(Arc<datafusion_expr::ScalarUDF>),
}
