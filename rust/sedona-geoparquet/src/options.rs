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

use std::{collections::HashMap, str::FromStr};

use datafusion::config::{ConfigField, TableParquetOptions, Visit};
use datafusion_common::{plan_err, DataFusionError, Result};

use crate::metadata::GeoParquetColumnMetadata;

/// [TableParquetOptions] wrapper with GeoParquet-specific options
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TableGeoParquetOptions {
    /// Inner [TableParquetOptions]
    pub inner: TableParquetOptions,
    /// [GeoParquetVersion] to use when writing GeoParquet files
    pub geoparquet_version: GeoParquetVersion,
    /// When writing [GeoParquetVersion::V1_1], use `true` to overwrite existing
    /// bounding box columns.
    pub overwrite_bbox_columns: bool,
    /// Optional geometry column metadata overrides for schema inference.
    pub geometry_columns: GeometryColumns,
    /// Validate geometry column contents against metadata when reading.
    pub validate: bool,
}

impl TableGeoParquetOptions {
    pub const TABLE_OPTIONS_KEYS: [&str; 4] = [
        "format.geoparquet_version",
        "format.geometry_columns",
        "format.validate",
        "format.overwrite_bbox_columns",
    ];
}

impl ConfigField for TableGeoParquetOptions {
    fn visit<V: Visit>(&self, v: &mut V, key_prefix: &str, _description: &'static str) {
        // Visit inner TableParquetOptions fields
        self.inner.visit(v, key_prefix, "");

        // Visit GeoParquet-specific fields
        self.geoparquet_version.visit(
            v,
            &format!("{key_prefix}.geoparquet_version"),
            "GeoParquet version to use when writing",
        );
        self.overwrite_bbox_columns.visit(
            v,
            &format!("{key_prefix}.overwrite_bbox_columns"),
            "Overwrite existing bounding box columns when writing GeoParquet 1.1",
        );
        self.geometry_columns.visit(
            v,
            &format!("{key_prefix}.geometry_columns"),
            "Optional geometry column metadata overrides for schema inference",
        );
        self.validate.visit(
            v,
            &format!("{key_prefix}.validate"),
            "Validate geometry column contents against metadata when reading",
        );
    }

    fn set(&mut self, key: &str, value: &str) -> Result<()> {
        // Try GeoParquet-specific keys first
        match key {
            "geoparquet_version" => {
                self.geoparquet_version.set(key, value)?;
            }
            "overwrite_bbox_columns" => {
                self.overwrite_bbox_columns.set(key, value)?;
            }
            "geometry_columns" => {
                self.geometry_columns.set(key, value)?;
            }
            "validate" => {
                self.validate.set(key, value)?;
            }
            // Forward all other keys to inner TableParquetOptions
            _ => {
                self.inner.set(key, value)?;
            }
        }
        Ok(())
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
#[derive(Debug, Clone, Copy, Default, PartialEq)]
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

impl ConfigField for GeoParquetVersion {
    fn visit<V: Visit>(&self, v: &mut V, key: &str, description: &'static str) {
        let value = match self {
            GeoParquetVersion::V1_0 => "1.0",
            GeoParquetVersion::V1_1 => "1.1",
            GeoParquetVersion::V2_0 => "2.0",
            GeoParquetVersion::Omitted => "none",
        };
        v.some(key, value, description);
    }

    fn set(&mut self, _key: &str, value: &str) -> Result<()> {
        *self = value.parse()?;
        Ok(())
    }
}

/// Wrapper for geometry column metadata configuration
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GeometryColumns {
    columns: Option<HashMap<String, GeoParquetColumnMetadata>>,
}

impl GeometryColumns {
    /// Create empty geometry columns
    pub fn new() -> Self {
        Self { columns: None }
    }

    /// Create from a HashMap
    pub fn from_map(columns: HashMap<String, GeoParquetColumnMetadata>) -> Self {
        Self {
            columns: Some(columns),
        }
    }

    /// Get the inner HashMap
    pub fn inner(&self) -> Option<&HashMap<String, GeoParquetColumnMetadata>> {
        self.columns.as_ref()
    }

    /// Set from JSON string
    pub fn from_json(json: &str) -> Result<Self> {
        let columns: HashMap<String, GeoParquetColumnMetadata> = serde_json::from_str(json)
            .map_err(|e| {
                DataFusionError::Configuration(format!("geometry_columns must be valid JSON: {e}"))
            })?;
        Ok(Self {
            columns: Some(columns),
        })
    }

    /// Convert to JSON string
    pub fn to_json(&self) -> Result<String> {
        match &self.columns {
            Some(cols) => serde_json::to_string(cols).map_err(|e| {
                DataFusionError::Configuration(format!("Failed to serialize geometry_columns: {e}"))
            }),
            None => Ok(String::new()),
        }
    }
}

impl ConfigField for GeometryColumns {
    fn visit<V: Visit>(&self, v: &mut V, key: &str, description: &'static str) {
        match self.to_json() {
            Ok(json) if !json.is_empty() => v.some(key, json, description),
            _ => {}
        }
    }

    fn set(&mut self, _key: &str, value: &str) -> Result<()> {
        if value.is_empty() {
            self.columns = None;
        } else {
            *self = Self::from_json(value)?;
        }
        Ok(())
    }
}
