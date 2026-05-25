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

//! Python-facing expression wrapper.
//!
//! This module is the Rust half of `sedonadb.expr.Expr`. It exposes a thin
//! PyO3 class (`PyExpr`, exported to Python as `_lib.InternalExpr`) that owns
//! a single `datafusion_expr::Expr` and a small set of factory `#[pyfunction]`s
//! used by the Python wrapper to construct columns and literals.
//!
//! The high-level shape:
//!
//! - The Python side (`sedonadb.expr.Expr`) holds a handle to a `PyExpr` and
//!   provides operator overloading, docstrings, and Pythonic ergonomics.
//! - This Rust side stays minimal: build the right `Expr` variant, do the
//!   minimum amount of validation that benefits from native types (e.g.
//!   inspecting Arrow extension metadata for `cast`), and bubble up errors
//!   via `PySedonaError`.
//!
//! The design mirrors the equivalent layer in the R bindings
//! (`r/sedonadb/src/rust/src/expression.rs`), where the Python `Expr` plays the
//! role of `SedonaDBExpr` and `expr_col` / `expr_lit` mirror
//! `SedonaDBExprFactory::column` / `::literal`.

use datafusion_common::Column;
use datafusion_expr::{expr::InList, BinaryExpr, Cast, Expr, Operator, SortExpr};
use pyo3::prelude::*;

use crate::error::PySedonaError;
use crate::import_from::{import_arrow_field, import_arrow_scalar};

/// PyO3 wrapper around a single DataFusion logical expression.
///
/// `PyExpr` is exposed to Python as `_lib.InternalExpr` and is the value
/// type behind the user-facing `sedonadb.expr.Expr` Python class. It is
/// `Clone` so that PyO3 method signatures can take `Vec<PyExpr>` by value;
/// each clone is cheap because `Expr` itself owns its children behind `Box`,
/// and cloning copies only the small wrapper plus a few `Arc`s in the
/// underlying tree.
///
/// The `inner` field is intentionally `pub` so that other PyO3 modules in
/// this crate (e.g. `dataframe.rs`) can take `Vec<PyExpr>` arguments and
/// move the inner `Expr` out without going through accessor methods.
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
    /// Pretty-print using DataFusion's `Display` impl (e.g. `"x AS y"`).
    /// Used by Python's `repr()` / `str()` paths.
    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }

    /// Debug-print the full Rust `Expr` tree. Useful for tests and
    /// troubleshooting; not part of the user-facing surface.
    fn debug_string(&self) -> String {
        format!("{:?}", self.inner)
    }

    /// Return the name of the `Expr` enum variant (e.g. `"Column"`,
    /// `"Literal"`, `"BinaryExpr"`). This is a stable structural property
    /// that tests can assert on without depending on Display formatting.
    fn variant_name(&self) -> String {
        self.inner.variant_name().to_string()
    }

    /// Wrap this expression in `Expr::Alias { name }`.
    ///
    /// We use DataFusion's `alias_if_changed` helper so that aliasing an
    /// expression to its existing name is a no-op. This avoids producing
    /// e.g. `Expr::Alias("x", Expr::Column("x"))` which is redundant but
    /// otherwise legal.
    fn alias(&self, name: &str) -> Result<Self, PySedonaError> {
        let inner = self.inner.clone().alias_if_changed(name.to_string())?;
        Ok(Self { inner })
    }

    /// Wrap this expression in `Expr::Cast` to the storage type of the
    /// provided Arrow field-like object.
    ///
    /// Steps:
    ///
    /// 1. Pull an `arrow_schema::Field` out of `target` via the Arrow
    ///    PyCapsule schema interface (any object exposing
    ///    `__arrow_c_schema__` works — `pyarrow.DataType`,
    ///    `pyarrow.Field`, etc.).
    /// 2. Reject Arrow extension types up front. SedonaDB's spatial types
    ///    are extension types over WKB; users who reach for `cast` on
    ///    those almost certainly want a different operation, so we surface
    ///    a clear error rather than silently dropping the extension.
    /// 3. Build `Expr::Cast { expr, data_type }` using only the storage
    ///    type. We don't carry field metadata through the cast.
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

    /// Build `Expr::IsNull(self)`. SQL semantics — matches NULL only,
    /// not floating-point NaN. The Pythonic NaN-aware accessor will live
    /// on the future `Series` type.
    fn is_null(&self) -> Self {
        Self {
            inner: Expr::IsNull(Box::new(self.inner.clone())),
        }
    }

    /// Build `Expr::IsNotNull(self)`. SQL semantics, mirror of `is_null`.
    fn is_not_null(&self) -> Self {
        Self {
            inner: Expr::IsNotNull(Box::new(self.inner.clone())),
        }
    }

    /// Build `Expr::InList(self IN (values), negated)`.
    ///
    /// The Python side already coerces every element of the user's list
    /// to a `PyExpr` (using `lit()` for non-Expr scalars), so the Rust
    /// side just needs to clone each `Expr` out of its handle and hand
    /// the resulting `Vec<Expr>` to `InList::new`. `negated=true`
    /// produces `NOT IN`, exposed through the optional kwarg below.
    #[pyo3(signature = (values, negated=false))]
    fn isin(&self, values: Vec<PyRef<'_, PyExpr>>, negated: bool) -> Self {
        let list = values.iter().map(|e| e.inner.clone()).collect();
        Self {
            inner: Expr::InList(InList::new(Box::new(self.inner.clone()), list, negated)),
        }
    }

    /// Build `Expr::Negative(self)` — arithmetic negation, the unary `-`.
    fn negate(&self) -> Self {
        Self {
            inner: Expr::Negative(Box::new(self.inner.clone())),
        }
    }

    /// Wrap this expression as an ascending `SortExpr` sort key.
    ///
    /// `nulls_first` controls where null values land in the sorted output.
    /// The Python wrapper defaults this to `false` (nulls last), matching
    /// the default we ship for both directions; users wanting SQL-style
    /// "nulls last on asc, nulls first on desc" can pass an explicit
    /// argument or use `sort_expr()` for full control.
    #[pyo3(signature = (nulls_first=false))]
    fn asc(&self, nulls_first: bool) -> PySortExpr {
        PySortExpr {
            inner: SortExpr::new(self.inner.clone(), /* asc */ true, nulls_first),
        }
    }

    /// Wrap this expression as a descending `SortExpr` sort key. See
    /// `asc` for the meaning of `nulls_first`.
    #[pyo3(signature = (nulls_first=false))]
    fn desc(&self, nulls_first: bool) -> PySortExpr {
        PySortExpr {
            inner: SortExpr::new(self.inner.clone(), /* asc */ false, nulls_first),
        }
    }
}

/// PyO3 wrapper around `datafusion_expr::SortExpr` (a `(expr, asc,
/// nulls_first)` triple). Exposed to Python as `_lib.InternalSortExpr`.
///
/// `SortExpr` values are produced either via `Expr.asc()` / `Expr.desc()`
/// (the common case, with `nulls_first` defaulting to false), or via
/// `sedonadb.expr.sort_expr(expr, asc=..., nulls_first=...)` for full
/// control. They are consumed by `DataFrame.sort(*keys)`.
#[pyclass(name = "InternalSortExpr")]
#[derive(Clone)]
pub struct PySortExpr {
    pub inner: SortExpr,
}

impl PySortExpr {
    pub fn new(inner: SortExpr) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySortExpr {
    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }
}

/// Construct a column reference: `Expr::Column("x")` or `Expr::Column("t.x")`.
///
/// `qualifier` is the optional table name (e.g. the `t` in `t.x`). It maps to
/// `Some(table_ref)` on `Column::new` so that disambiguating a column across
/// multiple input tables in a join is possible without parsing the name as
/// SQL. When `qualifier` is `None` we use `Column::new_unqualified`, which
/// behaves like a bare identifier in SQL and resolves against whichever
/// table's schema introduced the name first.
///
/// Mirrors `SedonaDBExprFactory::column(name, qualifier)` in the R bindings.
#[pyfunction]
#[pyo3(signature = (name, qualifier=None))]
pub fn expr_col(name: &str, qualifier: Option<&str>) -> PyExpr {
    let column = match qualifier {
        Some(q) => Column::new(Some(q), name),
        None => Column::new_unqualified(name),
    };
    PyExpr {
        inner: Expr::Column(column),
    }
}

/// Wrap a Python value (already coerced to an Arrow array on the Python
/// side) as an `Expr::Literal`.
///
/// Steps:
///
/// 1. Use `import_arrow_scalar` from `import_from.rs`, which handles the
///    Arrow C array PyCapsule import, validates that the array has
///    exactly one element, extracts the `ScalarValue`, and preserves any
///    field metadata (e.g. extension type info needed for spatial
///    literals). Centralising this here means the literal path,
///    `with_params`, and any future scalar consumers all share one set
///    of validation and error messages.
/// 2. Decompose the resulting `ScalarAndMetadata` into its `(value,
///    metadata)` parts via `into_inner` and feed them straight to
///    `Expr::Literal`.
#[pyfunction]
pub fn expr_lit(obj: Bound<'_, PyAny>) -> Result<PyExpr, PySedonaError> {
    let (scalar_value, metadata) = import_arrow_scalar(&obj)?.into_inner();
    let inner = Expr::Literal(scalar_value, metadata);
    Ok(PyExpr { inner })
}

/// Build a binary `Expr` from a string operator and two operands.
///
/// All operator dispatch lives in Python (see `_binary` in `expression.py`),
/// which calls into this single Rust factory with the operator name as a
/// string. Centralising the operator-to-`Operator` mapping in one place
/// keeps both Rust and Python code small: adding an operator is a single
/// `match` arm and a new dunder, rather than a separate factory per op.
///
/// Mirrors `SedonaDBExprFactory::binary` in the R bindings.
#[pyfunction]
pub fn expr_binary(op: &str, lhs: &PyExpr, rhs: &PyExpr) -> Result<PyExpr, PySedonaError> {
    let operator = match op {
        "+" => Operator::Plus,
        "-" => Operator::Minus,
        "*" => Operator::Multiply,
        "/" => Operator::Divide,
        "==" => Operator::Eq,
        "!=" => Operator::NotEq,
        ">" => Operator::Gt,
        ">=" => Operator::GtEq,
        "<" => Operator::Lt,
        "<=" => Operator::LtEq,
        "&" => Operator::And,
        "|" => Operator::Or,
        other => {
            return Err(PySedonaError::SedonaPython(format!(
                "Unsupported binary operator '{other}'"
            )))
        }
    };

    let inner = Expr::BinaryExpr(BinaryExpr::new(
        Box::new(lhs.inner.clone()),
        operator,
        Box::new(rhs.inner.clone()),
    ));
    Ok(PyExpr { inner })
}

/// Build a logical-NOT `Expr::Not` over the given expression. Used by the
/// Python `~expr` (`__invert__`) operator.
#[pyfunction]
pub fn expr_not(expr: &PyExpr) -> PyExpr {
    PyExpr {
        inner: Expr::Not(Box::new(expr.inner.clone())),
    }
}

/// Construct a `SortExpr` with explicit direction and null-placement
/// arguments. Lower-level than `Expr.asc()` / `Expr.desc()` — used when
/// the caller wants to set both knobs deliberately, e.g. `nulls_first=true`
/// on an ascending sort.
#[pyfunction]
#[pyo3(signature = (expr, asc=true, nulls_first=false))]
pub fn expr_sort_expr(expr: &PyExpr, asc: bool, nulls_first: bool) -> PySortExpr {
    PySortExpr {
        inner: SortExpr::new(expr.inner.clone(), asc, nulls_first),
    }
}
