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
// Example functions

use std::ffi::c_void;

use savvy::savvy;

use savvy_ffi::R_NilValue;
use sedona_adbc::AdbcSedonadbDriverInit;
use sedona_proj::register::{configure_global_proj_engine, ProjCrsEngineBuilder};

mod context;
mod dataframe;
mod error;
mod ffi;
mod runtime;

#[savvy]
fn sedonadb_adbc_init_func() -> savvy::Result<savvy::Sexp> {
    let driver_init_void = AdbcSedonadbDriverInit as *mut c_void;

    unsafe {
        Ok(savvy::Sexp(savvy_ffi::R_MakeExternalPtr(
            driver_init_void,
            R_NilValue,
            R_NilValue,
        )))
    }
}

#[savvy]
fn configure_proj_shared(
    shared_library_path: Option<&str>,
    database_path: Option<&str>,
    search_path: Option<&str>,
) -> savvy::Result<()> {
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
