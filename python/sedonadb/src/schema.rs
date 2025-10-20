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
use std::ffi::CString;

use arrow_array::ffi::FFI_ArrowSchema;
use arrow_schema::{Field, Schema};
use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use sedona_schema::datatypes::SedonaType;

use crate::error::PySedonaError;

#[pyclass]
pub struct PySedonaSchema {
    pub inner: Schema,
}

impl PySedonaSchema {
    pub fn new(inner: Schema) -> Self {
        Self { inner }
    }

    pub fn schema_capsule<'py>(
        &self,
        py: Python<'py>,
    ) -> Result<Bound<'py, PyCapsule>, PySedonaError> {
        let schema_capsule_name = CString::new("arrow_schema").unwrap();
        let ffi_schema = FFI_ArrowSchema::try_from(self.inner.clone())?;
        Ok(PyCapsule::new(py, ffi_schema, Some(schema_capsule_name))?)
    }

    fn field_by_index(&self, i: usize) -> PySedonaField {
        PySedonaField::new(self.inner.field(i).clone())
    }

    fn field_by_name(&self, name: &str) -> Option<PySedonaField> {
        if let Ok(field) = self.inner.field_with_name(name) {
            Some(PySedonaField::new(field.clone()))
        } else {
            None
        }
    }
}

#[pymethods]
impl PySedonaSchema {
    fn field<'py>(
        &self,
        py: Python<'py>,
        index_or_name: PyObject,
    ) -> Result<PySedonaField, PySedonaError> {
        if let Ok(index) = index_or_name.extract::<usize>(py) {
            if index < self.inner.fields().len() {
                Ok(self.field_by_index(index))
            } else {
                Err(PyIndexError::new_err(format!("Field index {index} out of range")).into())
            }
        } else if let Ok(name) = index_or_name.extract::<&str>(py) {
            match self.field_by_name(name) {
                Some(field) => Ok(field),
                None => Err(PyKeyError::new_err(format!("No field named '{name}'")).into()),
            }
        } else {
            Err(
                PyTypeError::new_err("Field accessor must be an integer index or string name")
                    .into(),
            )
        }
    }

    fn __arrow_c_schema__<'py>(
        &self,
        py: Python<'py>,
    ) -> Result<Bound<'py, PyCapsule>, PySedonaError> {
        self.schema_capsule(py)
    }

    fn __repr__(&self) -> String {
        let fields = (0..self.inner.fields().len())
            .map(|i| self.field_by_index(i).repr())
            .collect::<Vec<String>>();
        let maybe_s = if fields.len() != 1 { "s" } else { "" };

        // Try not to overwhelm the console in the event of extremely wide schemas
        if fields.len() > 100 {
            format!(
                "SedonaSchema with {} field{maybe_s}:\n  {}\n...with {} more fields",
                fields.len(),
                fields[..100].join("\n  "),
                fields.len() - 100
            )
        } else {
            format!(
                "SedonaSchema with {} field{maybe_s}:\n  {}",
                fields.len(),
                fields.join("\n  ")
            )
        }
    }
}

#[pyclass]
pub struct PySedonaField {
    pub inner: Field,
}

impl PySedonaField {
    pub fn new(inner: Field) -> Self {
        Self { inner }
    }

    pub fn repr(&self) -> String {
        let name = self.inner.name();
        let maybe_nullable_prefix = if self.inner.is_nullable() {
            ""
        } else {
            "non-nullable "
        };

        // Reprs that error can be difficult to debug, so if this fails for some
        // reason, just print the debug output of the Field
        if let Ok(py_sedona_type) = self.r#type() {
            format!("{name}: {maybe_nullable_prefix}{}", py_sedona_type.repr())
        } else {
            format!("{:?}", self.inner)
        }
    }
}

#[pymethods]
impl PySedonaField {
    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    #[getter]
    fn r#type(&self) -> Result<PySedonaType, PySedonaError> {
        Ok(PySedonaType::new(SedonaType::from_storage_field(
            &self.inner,
        )?))
    }

    fn __arrow_c_schema__<'py>(
        &self,
        py: Python<'py>,
    ) -> Result<Bound<'py, PyCapsule>, PySedonaError> {
        let schema_capsule_name = CString::new("arrow_schema").unwrap();
        let ffi_schema = FFI_ArrowSchema::try_from(self.inner.clone())?;
        Ok(PyCapsule::new(py, ffi_schema, Some(schema_capsule_name))?)
    }

    fn __repr__(&self) -> String {
        format!("SedonaField {}", self.repr())
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct PySedonaType {
    pub inner: SedonaType,
}

impl PySedonaType {
    pub fn new(inner: SedonaType) -> Self {
        Self { inner }
    }

    pub fn repr(&self) -> String {
        format!("{}<{}>", self.inner.logical_type_name(), self.inner)
    }
}

#[pymethods]
impl PySedonaType {
    #[getter]
    fn crs<'py>(&self, py: Python<'py>) -> Result<Option<PyObject>, PySedonaError> {
        match &self.inner {
            SedonaType::Wkb(_, crs) | SedonaType::WkbView(_, crs) => {
                if let Some(crs) = crs {
                    let json = py.import("json")?;
                    let geoarrow_types_crs = py.import("geoarrow.types.crs")?;

                    // Use geoarrow.types.crs.create() so that we don't have to do that here
                    // Parse the JSON into a Python object first, because this is what create()
                    // expects
                    let crs_string = crs.to_json();
                    let crs_py = json.getattr("loads")?.call1((crs_string,))?;
                    let crs_py_obj = geoarrow_types_crs.getattr("create")?.call1((crs_py,))?;
                    Ok(Some(crs_py_obj.into()))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    #[getter]
    fn edge_type<'py>(&self, py: Python<'py>) -> Result<Option<PyObject>, PySedonaError> {
        match &self.inner {
            SedonaType::Wkb(edges, _) | SedonaType::WkbView(edges, _) => {
                let geoarrow_types = py.import("geoarrow.types")?;
                let py_edge_type_cls = geoarrow_types.getattr("EdgeType")?;
                let py_edge_type = py_edge_type_cls
                    .getattr("create")?
                    .call1((format!("{edges:?}"),))?;
                Ok(Some(py_edge_type.into()))
            }
            _ => Ok(None),
        }
    }

    fn __arrow_c_schema__<'py>(
        &self,
        py: Python<'py>,
    ) -> Result<Bound<'py, PyCapsule>, PySedonaError> {
        let schema_capsule_name = CString::new("arrow_schema").unwrap();
        let field = self.inner.to_storage_field("", true)?;
        let ffi_schema = FFI_ArrowSchema::try_from(field)?;
        Ok(PyCapsule::new(py, ffi_schema, Some(schema_capsule_name))?)
    }

    fn __repr__(&self) -> String {
        format!("SedonaType {}", self.repr())
    }
}
