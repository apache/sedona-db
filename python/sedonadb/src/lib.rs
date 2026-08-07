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

use crate::{
    error::PySedonaError,
    raster_loader::py_raster_loader,
    udf::{sedona_aggregate_udf, sedona_scalar_udf},
};
use pyo3::{exceptions::PyValueError, ffi::Py_uintptr_t, prelude::*};
use sedona_adbc::AdbcSedonadbDriverInit;
use sedona_gdal::global::{configure_global_gdal_api, with_global_gdal, GdalApiBuilder};
use sedona_proj::register::{configure_global_proj_engine, ProjCrsEngineBuilder};
use sedona_raster::geo_transform::geotransform_from_bbox_and_spatial_shape;
use std::ffi::c_void;

mod context;
mod dataframe;
mod datasource;
mod error;
mod expr;
mod import_from;
mod raster_loader;
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
fn sedona_python_features() -> PyResult<Vec<String>> {
    Ok(vec![
        #[cfg(feature = "s2geography")]
        "s2geography".to_string(),
        #[cfg(feature = "gpu")]
        "gpu".to_string(),
    ])
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

    configure_global_proj_engine(builder)
        .map_err(|e| PySedonaError::SedonaPython(e.to_string()))?;
    Ok(())
}

#[pyfunction]
fn configure_gdal_shared(shared_library_path: String) -> Result<(), PySedonaError> {
    let builder = GdalApiBuilder::default().with_shared_library(shared_library_path.into());
    configure_global_gdal_api(builder).map_err(|e| {
        PySedonaError::SedonaPython(format!("Failed to configure GDAL shared library: {e}"))
    })?;
    Ok(())
}

/// Derive a north-up, GDAL-order geotransform from a spatial bounding box and
/// grid shape. Thin wrapper over `geotransform_from_bbox_and_spatial_shape` so
/// the bbox-to-transform math stays a single Rust source of truth.
///
/// `bbox` is `[xmin, ymin, xmax, ymax]`; `registration` is `"pixel"` (the
/// default when `None`) or `"node"`. Returns the six coefficients
/// `[origin_x, scale_x, skew_x, origin_y, skew_y, scale_y]`.
#[pyfunction]
#[pyo3(signature = (bbox, height, width, registration=None))]
fn geotransform_from_bbox(
    bbox: [f64; 4],
    height: u64,
    width: u64,
    registration: Option<String>,
) -> PyResult<Vec<f64>> {
    geotransform_from_bbox_and_spatial_shape(bbox, height, width, registration.as_deref())
        .map(|gt| gt.to_vec())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn gdal_version() -> Result<Option<String>, PySedonaError> {
    match with_global_gdal(|gdal| gdal.version_info("RELEASE_NAME")) {
        Ok(version) if !version.is_empty() => Ok(Some(version)),
        _ => Ok(None),
    }
}

/// Signal that GDAL is shutting down so datasets are left open during
/// interpreter/library teardown. Registered as a Python `atexit` callback to
/// avoid a Windows process-exit abort (`0xC0000409`) when a GDAL dataset is
/// closed while the library is being unloaded.
#[pyfunction]
fn begin_gdal_shutdown() {
    sedona_gdal::global::begin_gdal_shutdown();
}

#[pymodule(gil_used = false)]
fn _lib(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "mimalloc")]
    configure_tg_allocator();

    m.add_function(wrap_pyfunction!(begin_gdal_shutdown, m)?)?;
    m.add_function(wrap_pyfunction!(configure_gdal_shared, m)?)?;
    m.add_function(wrap_pyfunction!(configure_proj_shared, m)?)?;
    m.add_function(wrap_pyfunction!(expr::expr_binary, m)?)?;
    m.add_function(wrap_pyfunction!(expr::expr_col, m)?)?;
    m.add_function(wrap_pyfunction!(expr::expr_lit, m)?)?;
    m.add_function(wrap_pyfunction!(expr::expr_not, m)?)?;
    m.add_function(wrap_pyfunction!(expr::expr_sort_expr, m)?)?;
    m.add_function(wrap_pyfunction!(gdal_version, m)?)?;
    m.add_function(wrap_pyfunction!(geotransform_from_bbox, m)?)?;
    m.add_function(wrap_pyfunction!(py_raster_loader, m)?)?;
    m.add_function(wrap_pyfunction!(schema::raster_type, m)?)?;
    m.add_function(wrap_pyfunction!(sedona_adbc_driver_init, m)?)?;
    m.add_function(wrap_pyfunction!(sedona_aggregate_udf, m)?)?;
    m.add_function(wrap_pyfunction!(sedona_python_features, m)?)?;
    m.add_function(wrap_pyfunction!(sedona_python_version, m)?)?;
    m.add_function(wrap_pyfunction!(sedona_scalar_udf, m)?)?;

    m.add_class::<context::InternalContext>()?;
    m.add_class::<dataframe::InternalDataFrame>()?;
    m.add_class::<datasource::PyExternalFormat>()?;
    m.add_class::<datasource::PyProjectedRecordBatchReader>()?;
    m.add_class::<expr::PyExpr>()?;
    m.add_class::<expr::PySortExpr>()?;
    m.add_class::<raster_loader::PyBandDataType>()?;
    m.add_class::<raster_loader::PyRasterLoaderWrapper>()?;
    m.add_class::<raster_loader::PyRasterLoadRequest>()?;
    m.add_class::<raster_loader::PyRasterLoadResult>()?;
    m.add_class::<raster_loader::PyViewEntry>()?;
    m.add_class::<schema::PySedonaField>()?;
    m.add_class::<schema::PySedonaSchema>()?;
    m.add_class::<schema::PySedonaType>()?;

    m.add("SedonaError", py.get_type::<error::SedonaError>())?;

    Ok(())
}
