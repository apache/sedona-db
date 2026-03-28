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

//! Ported (and contains copied code) from georust/gdal:
//! <https://github.com/georust/gdal/blob/v0.19.0/src/config.rs>.
//! Original code is licensed under MIT.
//!
//! GDAL configuration option wrappers.

use std::ffi::CString;

use crate::errors::Result;
use crate::gdal_api::{call_gdal_api, GdalApi};

/// Set a GDAL library configuration option with **thread-local** scope.
pub fn set_thread_local_config_option(api: &'static GdalApi, key: &str, value: &str) -> Result<()> {
    let c_key = CString::new(key)?;
    let c_val = CString::new(value)?;
    unsafe {
        call_gdal_api!(
            api,
            CPLSetThreadLocalConfigOption,
            c_key.as_ptr(),
            c_val.as_ptr()
        );
    }
    Ok(())
}
