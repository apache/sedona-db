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

//! GDAL integration foundations for Apache SedonaDB raster types.
//!
//! This crate provides the lower-level utilities used by future GDAL-backed
//! raster functions:
//!
//! - in-db raster to GDAL MEM dataset conversion
//! - out-db and mixed raster to GDAL VRT dataset conversion
//! - GDAL datatype and nodata conversion helpers
//! - path normalization for GDAL VSI-backed raster sources

// Temporary until https://github.com/apache/sedona-db/issues/804 is resolved.
#[allow(dead_code)]
mod gdal_common;
// Temporary until https://github.com/apache/sedona-db/issues/804 is resolved.
#[allow(dead_code)]
mod gdal_dataset_provider;
#[cfg(test)]
mod source_uri;

// Re-export main dataset conversion functions
pub use gdal_common::{
    band_data_type_to_gdal, bytes_to_f64, gdal_to_band_data_type, gdal_type_byte_size,
    nodata_bytes_to_f64, nodata_f64_to_bytes,
};
