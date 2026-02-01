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
use pyo3::types::{PyDict, PyInt, PyString};
use sedona::context::SedonaContext;
use sedona_geoparquet::{GeoParquetColumnEncoding, GeoParquetColumnMetadata};
use sedona_schema::crs::deserialize_crs_from_obj;
use serde_json::Value as JsonValue;
use tokio::runtime::Runtime;

use crate::{
    dataframe::InternalDataFrame,
    datasource::PyExternalFormat,
    error::PySedonaError,
    import_from::{import_ffi_scalar_udf, import_table_provider_from_any},
    runtime::wait_for_future,
    udf::PySedonaScalarUdf,
};

fn parse_geometry_columns<'py>(
    py: Python<'py>,
    geometry_columns: HashMap<String, PyObject>,
) -> Result<HashMap<String, GeoParquetColumnMetadata>, PySedonaError> {
    let mut out = HashMap::new();

    for (column_name, spec_obj) in geometry_columns {
        let spec = spec_obj.bind(py);
        let mut column_metadata = GeoParquetColumnMetadata::default();
        let mut encoding: Option<String> = None;
        let mut edges: Option<String> = None;
        let mut crs: Option<JsonValue> = None;

        if spec.is_instance_of::<PyDict>() {
            let spec_dict = spec.downcast::<PyDict>().map_err(|_| {
                PySedonaError::SedonaPython(format!(
                    "geometry_columns['{column_name}'] must be a dict"
                ))
            })?;
            for (key, value) in spec_dict.iter() {
                let key_str: String = key.extract()?;
                match key_str.as_str() {
                    "encoding" => {
                        if value.is_none() {
                            return Err(PySedonaError::SedonaPython(format!(
                                "Geometry column '{column_name}' missing required key 'encoding'"
                            )));
                        }
                        encoding = Some(value.extract::<String>()?);
                    }
                    "edges" => {
                        if !value.is_none() {
                            edges = Some(value.extract::<String>()?);
                        }
                    }
                    "crs" => {
                        if !value.is_none() {
                            let crs_value = if value.is_instance_of::<PyString>() {
                                JsonValue::String(value.extract::<String>()?)
                            } else if value.is_instance_of::<PyInt>() {
                                let crs_int: i64 = value.extract()?;
                                JsonValue::Number(crs_int.into())
                            } else {
                                return Err(PySedonaError::SedonaPython(format!(
                                    "Unsupported CRS value for column '{column_name}' (expected string or integer)"
                                )));
                            };

                            let parsed_crs = deserialize_crs_from_obj(&crs_value).map_err(|e| {
                                PySedonaError::SedonaPython(format!(
                                    "Invalid CRS for column '{column_name}': {e}"
                                ))
                            })?;

                            if let Some(parsed_crs) = parsed_crs {
                                let normalized = serde_json::from_str(&parsed_crs.to_json())
                                    .map_err(|e| {
                                        PySedonaError::SedonaPython(format!(
                                            "Invalid CRS for column '{column_name}': {e}"
                                        ))
                                    })?;
                                crs = Some(normalized);
                            }
                        }
                    }
                    _ => {
                        return Err(PySedonaError::SedonaPython(format!(
                            "Unexpected key '{key_str}' in geometry_columns['{column_name}']"
                        )));
                    }
                }
            }
        } else if spec.is_instance_of::<PyString>() {
            encoding = Some(spec.extract::<String>()?);
        } else {
            return Err(PySedonaError::SedonaPython(format!(
                "geometry_columns['{column_name}'] must be a dict or string"
            )));
        }

        let encoding = encoding.ok_or_else(|| {
            PySedonaError::SedonaPython(format!(
                "Geometry column '{column_name}' missing required key 'encoding'"
            ))
        })?;

        column_metadata.encoding = match encoding.to_lowercase().as_str() {
            "wkb" => GeoParquetColumnEncoding::WKB,
            _ => {
                return Err(PySedonaError::SedonaPython(format!(
                    "Unsupported geometry encoding '{encoding}' for column '{column_name}' (only 'WKB' is supported)"
                )));
            }
        };

        if let Some(edges) = edges {
            match edges.to_lowercase().as_str() {
                "planar" => {}
                "spherical" => {
                    column_metadata.edges = Some("spherical".to_string());
                }
                _ => {
                    return Err(PySedonaError::SedonaPython(format!(
                        "Unsupported edges value '{edges}' for column '{column_name}' (expected 'planar' or 'spherical')"
                    )));
                }
            }
        }

        if let Some(crs) = crs {
            column_metadata.crs = Some(crs);
        }

        out.insert(column_name, column_metadata);
    }

    Ok(out)
}

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
        geometry_columns: Option<HashMap<String, PyObject>>,
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
            let geometry_columns = parse_geometry_columns(py, geometry_columns)?;
            geo_options = geo_options.with_geometry_columns(geometry_columns);
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
