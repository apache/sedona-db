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

use datafusion_common::{Column, ScalarValue};
use datafusion_expr::{
    expr::{FieldMetadata, InList},
    Cast, Expr,
};
use pyo3::prelude::*;

use crate::error::PySedonaError;
use crate::import_from::{import_arrow_array, import_arrow_field};

#[pyclass(name = "InternalExpr")]
#[derive(Clone)]
pub struct PyExpr {
    pub inner: Expr,
}

impl PyExpr {
    pub fn new(inner: Expr) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyExpr {
    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }

    fn debug_string(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn variant_name(&self) -> String {
        self.inner.variant_name().to_string()
    }

    fn alias(&self, name: &str) -> Result<Self, PySedonaError> {
        let inner = self.inner.clone().alias_if_changed(name.to_string())?;
        Ok(Self { inner })
    }

    fn cast(&self, target: Bound<'_, PyAny>) -> Result<Self, PySedonaError> {
        let field = import_arrow_field(&target)?;
        if let Some(type_name) = field.extension_type_name() {
            return Err(PySedonaError::SedonaPython(format!(
                "Can't cast to Arrow extension type '{type_name}'"
            )));
        }
        let inner = Expr::Cast(Cast::new(
            Box::new(self.inner.clone()),
            field.data_type().clone(),
        ));
        Ok(Self { inner })
    }

    fn is_null(&self) -> Self {
        Self {
            inner: Expr::IsNull(Box::new(self.inner.clone())),
        }
    }

    fn is_not_null(&self) -> Self {
        Self {
            inner: Expr::IsNotNull(Box::new(self.inner.clone())),
        }
    }

    #[pyo3(signature = (values, negated=false))]
    fn isin(&self, values: Vec<PyRef<'_, PyExpr>>, negated: bool) -> Self {
        let list = values.iter().map(|e| e.inner.clone()).collect();
        Self {
            inner: Expr::InList(InList::new(Box::new(self.inner.clone()), list, negated)),
        }
    }

    fn negate(&self) -> Self {
        Self {
            inner: Expr::Negative(Box::new(self.inner.clone())),
        }
    }
}

#[pyfunction]
pub fn expr_col(name: &str) -> PyExpr {
    PyExpr {
        inner: Expr::Column(Column::new_unqualified(name)),
    }
}

#[pyfunction]
pub fn expr_lit(obj: Bound<'_, PyAny>) -> Result<PyExpr, PySedonaError> {
    let (field, array) = import_arrow_array(&obj)?;
    if array.len() != 1 {
        return Err(PySedonaError::SedonaPython(format!(
            "Expected literal Arrow array of length 1, got length {}",
            array.len()
        )));
    }
    let scalar_value = ScalarValue::try_from_array(&array, 0)?;
    let metadata = if field.metadata().is_empty() {
        None
    } else {
        Some(FieldMetadata::new_from_field(&field))
    };
    let inner = Expr::Literal(scalar_value, metadata);
    Ok(PyExpr { inner })
}
