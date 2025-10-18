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
use crate::{error::PySedonaError, udf::sedona_scalar_udf};
use pyo3::{ffi::Py_uintptr_t, prelude::*};
use sedona_adbc::AdbcSedonadbDriverInit;
use sedona_proj::register::{configure_global_proj_engine, ProjCrsEngineBuilder};
use std::ffi::c_void;

mod context;
mod dataframe;
mod error;
mod import_from;
mod reader;
mod runtime;
mod schema;
mod udf;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(feature = "mimalloc")]
use mimalloc::MiMalloc;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[cfg(feature = "mimalloc")]
fn configure_tg_allocator() {
    use libmimalloc_sys::{mi_free, mi_malloc, mi_realloc};
    use sedona_tg::tg::set_allocator;

    // Configure tg to use mimalloc
    unsafe { set_allocator(mi_malloc, mi_realloc, mi_free) }.expect("Failed to set tg allocator");
}

#[pyfunction]
fn sedona_python_version() -> PyResult<String> {
    Ok(VERSION.to_string())
}

#[pyfunction]
fn sedona_adbc_driver_init() -> PyResult<Py_uintptr_t> {
    let driver_init_void = AdbcSedonadbDriverInit as *const c_void;
    Ok(driver_init_void as Py_uintptr_t)
}

#[pyfunction]
fn configure_proj_shared(
    shared_library_path: Option<String>,
    database_path: Option<String>,
    search_path: Option<String>,
) -> Result<(), PySedonaError> {
    let mut builder = ProjCrsEngineBuilder::default();

    if let Some(shared_library_path) = shared_library_path {
        builder = builder.with_shared_library(shared_library_path.into());
    }

    if let Some(database_path) = database_path {
        builder = builder.with_database_path(database_path.into());
    }

    if let Some(search_path) = search_path {
        builder = builder.with_search_paths(vec![search_path.into()]);
    }

    configure_global_proj_engine(builder)?;
    Ok(())
}

#[pymodule]
fn _lib(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "mimalloc")]
    configure_tg_allocator();

    m.add_function(wrap_pyfunction!(configure_proj_shared, m)?)?;
    m.add_function(wrap_pyfunction!(sedona_adbc_driver_init, m)?)?;
    m.add_function(wrap_pyfunction!(sedona_python_version, m)?)?;
    m.add_function(wrap_pyfunction!(sedona_scalar_udf, m)?)?;

    m.add_class::<context::InternalContext>()?;
    m.add_class::<dataframe::InternalDataFrame>()?;
    m.add("SedonaError", py.get_type::<error::SedonaError>())?;
    m.add_class::<schema::PySedonaSchema>()?;
    m.add_class::<schema::PySedonaField>()?;
    m.add_class::<schema::PySedonaType>()?;

    Ok(())
}
