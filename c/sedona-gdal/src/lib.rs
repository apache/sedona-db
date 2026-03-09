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

// --- FFI layer ---
pub(crate) mod dyn_load;
pub mod gdal_dyn_bindgen;

// --- Error types ---
pub mod errors;

// --- Core API ---
pub mod gdal_api;
pub mod global;

// --- High-level wrappers ---
pub mod config;
pub mod cpl;
pub mod dataset;
pub mod driver;
pub mod geo_transform;
pub mod raster;
pub mod spatial_ref;
pub mod vector;
pub mod vrt;
pub mod vsi;
