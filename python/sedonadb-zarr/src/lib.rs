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

//! `sedonadb-zarr` — Python plugin package wiring Zarr support into a
//! `sedonadb` session.
//!
//! The Rust side here is intentionally thin: it borrows the sedonadb
//! Python `InternalContext` (an `rlib`-exposed PyO3 class) and calls
//! `sedona_raster_zarr::register_udtf` on its inner `SessionContext`.
//! Everything else (the `ZarrFormatSpec(ExternalFormatSpec)` Python
//! class, the `register(con)` helper) lives on the Python side.

use pyo3::prelude::*;
// The `python/sedonadb` Cargo package compiles its rlib as `_lib`
// (the lib.name override that maturin needs for the wheel's
// `sedonadb._lib` import path). Our own crate also has `lib.name =
// "_lib"`, which would collide with the pymodule below — the leading
// `::` disambiguates: it refers to the extern crate, not our local
// module.
use ::_lib::context::InternalContext;

/// Attach the `sd_read_zarr` SQL UDTF to a sedonadb session.
///
/// Called from the Python `sedonadb_zarr.register(con)` helper. After
/// this, `con.sql("SELECT * FROM sd_read_zarr(...)")` works.
#[pyfunction]
fn register_udtf(internal_ctx: &Bound<'_, InternalContext>) -> PyResult<()> {
    let ctx = internal_ctx.borrow();
    sedona_raster_zarr::register_udtf(&ctx.inner.ctx);
    Ok(())
}

#[pymodule]
fn _lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(register_udtf, m)?)?;
    Ok(())
}
