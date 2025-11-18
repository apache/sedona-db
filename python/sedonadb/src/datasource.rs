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

use std::{collections::HashMap, ffi::CString, sync::Arc};

use arrow_array::{ffi_stream::FFI_ArrowArrayStream, RecordBatch, RecordBatchReader};
use arrow_schema::{ArrowError, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::{physical_expr::conjunction, physical_plan::PhysicalExpr};
use datafusion_common::{DataFusionError, Result};
use pyo3::{
    exceptions::PyNotImplementedError, pyclass, pymethods, types::PyCapsule, Bound, PyObject,
    Python,
};
use sedona_datasource::{
    spec::{ExternalFormatSpec, Object, OpenReaderArgs},
    utility::ProjectedRecordBatchReader,
};
use sedona_expr::spatial_filter::SpatialFilter;
use sedona_geometry::interval::IntervalTrait;

use crate::{
    error::PySedonaError,
    import_from::{import_arrow_array_stream, import_arrow_schema},
    schema::PySedonaSchema,
};

/// Python object that calls the methods of Python-level ExternalFormatSpec
///
/// The main purpose of this object is to implement [ExternalFormatSpec] such
/// that it can be used by SedonaDB/DataFusion internals.
#[pyclass]
#[derive(Debug)]
pub struct PyExternalFormat {
    extension: String,
    py_spec: PyObject,
}

impl Clone for PyExternalFormat {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            extension: self.extension.clone(),
            py_spec: self.py_spec.clone_ref(py),
        })
    }
}

impl PyExternalFormat {
    fn with_options_impl<'py>(
        &self,
        py: Python<'py>,
        options: &HashMap<String, String>,
    ) -> Result<Self, PySedonaError> {
        let new_py_spec = self
            .py_spec
            .call_method(py, "with_options", (options.clone(),), None)?;
        let new_extension = new_py_spec
            .getattr(py, "extension")?
            .extract::<String>(py)?;
        Ok(Self {
            extension: new_extension,
            py_spec: new_py_spec,
        })
    }

    fn infer_schema_impl<'py>(
        &self,
        py: Python<'py>,
        object: &Object,
    ) -> Result<Schema, PySedonaError> {
        let maybe_schema = self.py_spec.call_method(
            py,
            "infer_schema",
            (PyDataSourceObject {
                inner: object.clone(),
            },),
            None,
        );

        match maybe_schema {
            Ok(py_schema) => import_arrow_schema(py_schema.bind(py)),
            Err(e) => {
                if e.is_instance_of::<PyNotImplementedError>(py) {
                    // Fall back on the open_reader implementation, as for some
                    // external formats there is no other mechanism to infer a schema
                    // other than to open a reader and query the schema at that point.
                    let reader_args = OpenReaderArgs {
                        src: object.clone(),
                        batch_size: None,
                        file_schema: None,
                        file_projection: None,
                        filters: vec![],
                    };

                    let reader = self.open_reader_impl(py, &reader_args)?;
                    Ok(reader.schema().as_ref().clone())
                } else {
                    Err(PySedonaError::from(e))
                }
            }
        }
    }

    fn open_reader_impl<'py>(
        &self,
        py: Python<'py>,
        args: &OpenReaderArgs,
    ) -> Result<Box<dyn RecordBatchReader + Send>, PySedonaError> {
        let reader_obj = self.py_spec.call_method(
            py,
            "open_reader",
            (PyOpenReaderArgs {
                inner: args.clone(),
            },),
            None,
        )?;

        let reader = import_arrow_array_stream(py, reader_obj.bind(py), None)?;
        let wrapped_reader = WrappedRecordBatchReader {
            inner: reader,
            shelter: Some(reader_obj),
        };
        Ok(Box::new(wrapped_reader))
    }
}

#[pymethods]
impl PyExternalFormat {
    #[new]
    fn new<'py>(py: Python<'py>, py_spec: PyObject) -> Result<Self, PySedonaError> {
        let extension = py_spec.getattr(py, "extension")?.extract::<String>(py)?;
        Ok(Self { extension, py_spec })
    }
}

#[async_trait]
impl ExternalFormatSpec for PyExternalFormat {
    fn extension(&self) -> &str {
        &self.extension
    }

    fn with_options(
        &self,
        options: &HashMap<String, String>,
    ) -> Result<Arc<dyn ExternalFormatSpec>> {
        let new_external_format = Python::with_gil(|py| self.with_options_impl(py, options))
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        Ok(Arc::new(new_external_format))
    }

    async fn infer_schema(&self, location: &Object) -> Result<Schema> {
        let schema = Python::with_gil(|py| self.infer_schema_impl(py, location))
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        Ok(schema)
    }

    async fn open_reader(
        &self,
        args: &OpenReaderArgs,
    ) -> Result<Box<dyn RecordBatchReader + Send>> {
        let reader = Python::with_gil(|py| self.open_reader_impl(py, args))
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        Ok(reader)
    }
}

/// Wrapper around the [Object] such that the [PyExternalFormatSpec] can pass
/// required information into Python method calls
///
/// Currently this only exposes `to_url()`; however, we can and should expose
/// the ability to read portions of files using the underlying object_store.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyDataSourceObject {
    pub inner: Object,
}

#[pymethods]
impl PyDataSourceObject {
    fn to_url(&self) -> Option<String> {
        self.inner.to_url_string()
    }
}

/// Wrapper around the [OpenReaderArgs] such that the [PyExternalFormatSpec] can pass
/// required information into Python method calls
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyOpenReaderArgs {
    pub inner: OpenReaderArgs,
}

#[pymethods]
impl PyOpenReaderArgs {
    #[getter]
    fn src(&self) -> PyDataSourceObject {
        PyDataSourceObject {
            inner: self.inner.src.clone(),
        }
    }

    #[getter]
    fn batch_size(&self) -> Option<usize> {
        self.inner.batch_size
    }

    #[getter]
    fn file_schema(&self) -> Option<PySedonaSchema> {
        self.inner
            .file_schema
            .as_ref()
            .map(|schema| PySedonaSchema::new(schema.as_ref().clone()))
    }

    #[getter]
    fn file_projection(&self) -> Option<Vec<usize>> {
        self.inner.file_projection.clone()
    }

    #[getter]
    fn filters(&self) -> Vec<PyFilter> {
        self.inner
            .filters
            .iter()
            .map(|f| PyFilter { inner: f.clone() })
            .collect()
    }

    #[getter]
    fn filter(&self) -> Option<PyFilter> {
        if self.inner.filters.is_empty() {
            None
        } else {
            Some(PyFilter {
                inner: conjunction(self.inner.filters.iter().cloned()),
            })
        }
    }

    fn is_projected(&self) -> Result<bool, PySedonaError> {
        match (&self.inner.file_projection, &self.inner.file_schema) {
            (None, None) | (None, Some(_)) => Ok(false),
            (Some(projection), Some(schema)) => {
                let seq_along_schema = (0..schema.fields().len()).collect::<Vec<_>>();
                Ok(&seq_along_schema != projection)
            }
            (Some(_), None) => Err(PySedonaError::SedonaPython(
                "Can't check projection for OpenReaderArgs with no schema".to_string(),
            )),
        }
    }
}

/// Wrapper around a PhysicalExpr such that the [PyExternalFormatSpec] can pass
/// required information into Python method calls
///
/// This currently only exposes `bounding_box()`, but in the future could expose
/// various ways to serialize the expression (SQL, DataFusion ProtoBuf, Substrait).
#[pyclass]
#[derive(Debug)]
pub struct PyFilter {
    inner: Arc<dyn PhysicalExpr>,
}

#[pymethods]
impl PyFilter {
    fn bounding_box(
        &self,
        column_index: usize,
    ) -> Result<Option<(f64, f64, f64, f64)>, PySedonaError> {
        let filter = SpatialFilter::try_from_expr(&self.inner)?;
        let filter_bbox = filter.filter_bbox(column_index);
        if filter_bbox.x().is_full() || filter_bbox.y().is_full() {
            Ok(None)
        } else {
            Ok(Some((
                filter_bbox.x().lo(),
                filter_bbox.y().lo(),
                filter_bbox.x().hi(),
                filter_bbox.y().hi(),
            )))
        }
    }

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

/// RecordBatchReader utility that helps ensure projected output
///
/// Because the output of `open_reader()` is required to take into account
/// the projection, we need to provide a utility to ensure this is take into account.
/// This wrapper is a thin wrapper around the [ProjectedRecordBatchReader] that allows
/// it to be constructed from Python using either a set of indices or a set of names.
#[pyclass]
pub struct PyProjectedRecordBatchReader {
    inner_object: PyObject,
    projection_indices: Option<Vec<usize>>,
    projection_names: Option<Vec<String>>,
}

#[pymethods]
impl PyProjectedRecordBatchReader {
    #[new]
    fn new(
        inner_object: PyObject,
        projection_indices: Option<Vec<usize>>,
        projection_names: Option<Vec<String>>,
    ) -> Self {
        Self {
            inner_object,
            projection_indices,
            projection_names,
        }
    }

    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        #[allow(unused_variables)] requested_schema: Option<Bound<'py, PyCapsule>>,
    ) -> Result<Bound<'py, PyCapsule>, PySedonaError> {
        let inner = import_arrow_array_stream(py, self.inner_object.bind(py), None)?;

        let reader = match (&self.projection_indices, &self.projection_names) {
            (None, None) | (Some(_), Some(_)) => {
                return Err(PySedonaError::SedonaPython("PyProjectedRecordBatchReader must be specified by one of projection_indices or projection_names".to_string()))
            }
            (Some(indices), None) => {
                ProjectedRecordBatchReader::from_projection(inner, indices.clone())?
            }
            (None, Some(names)) => {
                ProjectedRecordBatchReader::from_output_names(inner, &names.iter().map(|s| s.as_str()).collect::<Vec<&str>>())?
            }
        };

        let ffi_stream = FFI_ArrowArrayStream::new(Box::new(reader));
        let stream_capsule_name = CString::new("arrow_array_stream").unwrap();
        Ok(PyCapsule::new(py, ffi_stream, Some(stream_capsule_name))?)
    }
}

/// Helper to ensure a Python object stays in scope for the duration of a
/// [RecordBatchReader]'s output.
///
/// Some Python frameworks require that some parent object outlive a returned
/// ArrowArrayStream/RecordBatchReader (e.g., the pyogrio context manager, or
/// an ADBC statement/cursor).
struct WrappedRecordBatchReader {
    pub inner: Box<dyn RecordBatchReader + Send>,
    pub shelter: Option<PyObject>,
}

impl RecordBatchReader for WrappedRecordBatchReader {
    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }
}

impl Iterator for WrappedRecordBatchReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(item) = self.inner.next() {
            Some(item)
        } else {
            self.shelter = None;
            None
        }
    }
}
