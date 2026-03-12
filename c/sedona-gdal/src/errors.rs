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
//! <https://github.com/georust/gdal/blob/v0.19.0/src/errors.rs>.
//! Original code is licensed under MIT.

use thiserror::Error;

/// Error type for the sedona-gdal crate initialization and library loading.
#[derive(Error, Debug)]
pub enum GdalInitLibraryError {
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    LibraryError(String),
}

/// Error type compatible with the georust/gdal error variants used in this codebase.
#[derive(Clone, Debug, Error)]
pub enum GdalError {
    #[error("CPL error class: '{class:?}', error number: '{number}', error msg: '{msg}'")]
    CplError {
        class: u32,
        number: i32,
        msg: String,
    },
}
