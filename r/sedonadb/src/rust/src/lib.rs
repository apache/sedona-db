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

#[savvy]
fn parse_crs_metadata(crs_json: &str) -> savvy::Result<savvy::Sexp> {
    use sedona_schema::crs::deserialize_crs_from_obj;

    // The input is GeoArrow extension metadata, which is a JSON object like:
    // {"crs": <PROJJSON or string>}
    // We need to extract the "crs" field first.
    let metadata: serde_json::Value = serde_json::from_str(crs_json)
        .map_err(|e| savvy::Error::new(format!("Failed to parse metadata JSON: {e}")))?;

    let crs_value = metadata.get("crs");
    if crs_value.is_none() || crs_value.unwrap().is_null() {
        return Ok(savvy::NullSexp.into());
    }

    let crs = deserialize_crs_from_obj(crs_value.unwrap())?;
    match crs {
        Some(crs_obj) => {
            let auth_code = crs_obj.to_authority_code().ok().flatten();
            let srid = crs_obj.srid().ok().flatten();
            let name = crs_value.unwrap().get("name").and_then(|v| v.as_str());
            let proj_string = crs_obj.to_crs_string();

            let mut out = savvy::OwnedListSexp::new(4, true)?;
            out.set_name(0, "authority_code")?;
            out.set_name(1, "srid")?;
            out.set_name(2, "name")?;
            out.set_name(3, "proj_string")?;

            if let Some(auth_code) = auth_code {
                out.set_value(0, savvy::Sexp::try_from(auth_code.as_str())?)?;
            } else {
                out.set_value(0, savvy::NullSexp)?;
            }

            if let Some(srid) = srid {
                out.set_value(1, savvy::Sexp::try_from(srid as i32)?)?;
            } else {
                out.set_value(1, savvy::NullSexp)?;
            }

            if let Some(name) = name {
                out.set_value(2, savvy::Sexp::try_from(name)?)?;
            } else {
                out.set_value(2, savvy::NullSexp)?;
            }
            
            out.set_value(3, savvy::Sexp::try_from(proj_string.as_str())?)?;

            Ok(out.into())
        }
        None => Ok(savvy::NullSexp.into()),
    }
}
