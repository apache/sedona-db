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

//! Native FlatGeobuf FileFormat with generic ObjectStore range reads.
//!
//! Enable Sedona's `fgb` feature for SQL file queries; Python enables it by default.
//! The local external datasource adapter below is also available.
//!
//! See the crate README for the partitioning contract and draft limitations.
//!
//! The native format can also be registered with `SessionState::register_file_format`.

mod format;
mod geometry;
mod metadata;
mod object_io;
mod opener;
mod source;
mod store_metadata;
pub use format::{FlatGeobufFormat, FlatGeobufFormatFactory, NativeFlatGeobufFormatFactory};
mod reader;

use arrow_array::RecordBatchReader;
use arrow_schema::Schema;
use async_trait::async_trait;
use datafusion_common::{Result, exec_err};
use metadata::{FileMetadata, local_path};
use sedona_datasource::spec::{ExternalFormatSpec, Object, OpenReaderArgs, SupportsRepartition};
use std::{collections::HashMap, sync::Arc};

/// Native reader for local FlatGeobuf files, without a private metadata cache.
/// Each open reader owns an independent file stream.
#[derive(Debug, Clone)]
pub struct FlatGeobufFormatSpec {
    geometry_column_name: String,
}
impl Default for FlatGeobufFormatSpec {
    fn default() -> Self {
        Self {
            geometry_column_name: "wkb_geometry".into(),
        }
    }
}
impl FlatGeobufFormatSpec {
    async fn metadata(&self, object: &Object) -> Result<Arc<FileMetadata>> {
        let path = local_path(object)?;
        let stamp = metadata::file_stamp(&path).await?;
        Ok(Arc::new(FileMetadata::read(&path, stamp).await?))
    }
}
#[async_trait]
impl ExternalFormatSpec for FlatGeobufFormatSpec {
    async fn infer_schema(&self, object: &Object) -> Result<Schema> {
        Ok(self
            .metadata(object)
            .await?
            .infer_schema(&self.geometry_column_name)?
            .as_ref()
            .clone())
    }

    async fn open_reader(
        &self,
        args: &OpenReaderArgs,
    ) -> Result<Box<dyn RecordBatchReader + Send>> {
        let meta = self.metadata(&args.src).await?;
        reader::open(meta, args, &self.geometry_column_name).await
    }

    fn with_options(
        &self,
        options: &HashMap<String, String>,
    ) -> Result<Arc<dyn ExternalFormatSpec>> {
        let mut format = self.clone();
        for (name, value) in options {
            match name.as_str() {
                "geometry_column_name" if !value.is_empty() => {
                    format.geometry_column_name = value.clone();
                }
                _ => return exec_err!("Unknown or invalid FlatGeobuf option: {name}"),
            }
        }
        Ok(Arc::new(format))
    }

    fn extension(&self) -> &str {
        "fgb"
    }

    fn supports_repartition(&self) -> SupportsRepartition {
        SupportsRepartition::ByRange
    }
}
