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

use std::str::FromStr;

use datafusion::config::TableParquetOptions;
use datafusion_common::{plan_err, DataFusionError};

/// [TableParquetOptions] wrapper with GeoParquet-specific options
#[derive(Debug, Default, Clone)]
pub struct TableGeoParquetOptions {
    /// Inner [TableParquetOptions]
    pub inner: TableParquetOptions,
    /// [GeoParquetVersion] to use when writing GeoParquet files
    pub geoparquet_version: GeoParquetVersion,
    /// When writing [GeoParquetVersion::V1_1], use `true` to overwrite existing
    /// bounding box columns.
    pub overwrite_bbox_columns: bool,
}

impl TableGeoParquetOptions {
    pub fn new() -> Self {
        Self::default()
    }
}

impl From<TableParquetOptions> for TableGeoParquetOptions {
    fn from(value: TableParquetOptions) -> Self {
        Self {
            inner: value,
            ..Default::default()
        }
    }
}

/// The GeoParquet Version to write for output with spatial columns
#[derive(Debug, Clone, Copy, Default)]
pub enum GeoParquetVersion {
    /// Write GeoParquet 1.0 metadata
    ///
    /// GeoParquet 1.0 has the widest support among readers and writers; however
    /// it does not include row-group level statistics.
    #[default]
    V1_0,

    /// Write GeoParquet 1.1 metadata and optional bounding box column
    ///
    /// A bbox column will be included for any column where the Parquet options would
    /// have otherwise written statistics (which it will by default).
    /// This option may be more computationally expensive; however, will result in
    /// row-group level statistics that some readers (e.g., SedonaDB) can use to prune
    /// row groups on read.
    V1_1,

    /// Write GeoParquet 2.0
    ///
    /// The GeoParquet 2.0 options is identical to GeoParquet 1.0 except the underlying storage
    /// of spatial columns is Parquet native geometry, where the Parquet writer will include
    /// native statistics according to the underlying Parquet options. Some readers
    /// (e.g., SedonaDB) can use these statistics to prune row groups on read.
    V2_0,

    /// Do not write GeoParquet metadata
    ///
    /// This option suppresses GeoParquet metadata; however, spatial types will be written as
    /// Parquet native Geometry/Geography when this is supported by the underlying writer.
    Omitted,
}

impl FromStr for GeoParquetVersion {
    type Err = DataFusionError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "1.0" => Ok(GeoParquetVersion::V1_0),
            "1.1" => Ok(GeoParquetVersion::V1_1),
            "2.0" => Ok(GeoParquetVersion::V2_0),
            "none" => Ok(GeoParquetVersion::Omitted),
            _ => plan_err!(
                "Unexpected GeoParquet version string (expected '1.0', '1.1', '2.0', or 'none')"
            ),
        }
    }
}
