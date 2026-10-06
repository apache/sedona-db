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

//! Native local FlatGeobuf reading through SedonaDB's external datasource API.
//!
//! See the crate README for the partitioning contract and draft limitations.
//!
//! Register explicitly in a DataFusion/SedonaDB session:
//!
//! ```no_run
//! use std::sync::Arc;
//! use datafusion::{prelude::SessionContext, datasource::listing::ListingTableUrl};
//! use sedona_datasource::provider::external_table;
//! use sedona_flatgeobuf::FlatGeobufFormatSpec;
//! # async fn example() -> datafusion_common::Result<()> {
//! let context = SessionContext::new();
//! let table = external_table(
//!     Arc::new(FlatGeobufFormatSpec::default()),
//!     &context,
//!     vec![ListingTableUrl::parse("file:///data/roads.fgb")?],
//!     true,
//!     Some(vec![]),
//! ).await?;
//! context.register_table("roads", table)?;
//! # Ok(())
//! # }
//! ```
mod geometry;
mod metadata;
mod reader;

use arrow_array::RecordBatchReader;
use arrow_schema::Schema;
use async_trait::async_trait;
use datafusion_common::{Result, exec_err};
use metadata::{FileMetadata, local_path};
use sedona_datasource::spec::{ExternalFormatSpec, Object, OpenReaderArgs, SupportsRepartition};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

/// Native reader for local FlatGeobuf files.
///
/// Clones share a bounded metadata cache. Each open reader owns an independent
/// file stream. Files must remain unchanged for the duration of a scan.
#[derive(Debug, Clone)]
pub struct FlatGeobufFormatSpec {
    geometry_column_name: String,
    cache: Arc<Mutex<HashMap<std::path::PathBuf, Arc<FileMetadata>>>>,
}
impl Default for FlatGeobufFormatSpec {
    fn default() -> Self {
        Self {
            geometry_column_name: "geometry".into(),
            cache: Default::default(),
        }
    }
}
impl FlatGeobufFormatSpec {
    async fn metadata(&self, object: &Object) -> Result<Arc<FileMetadata>> {
        let path = local_path(object)?;
        let this = self.clone();
        tokio::task::spawn_blocking(move || {
            let stamp = metadata::file_stamp(&path)?;
            let mut cache = this.cache.lock().map_err(|_| {
                datafusion_common::exec_datafusion_err!("FlatGeobuf metadata cache lock poisoned")
            })?;
            if let Some(meta) = cache.get(&path)
                && meta.stamp == stamp
            {
                return Ok(meta.clone());
            }
            let meta = Arc::new(FileMetadata::read(
                &path,
                stamp,
                &this.geometry_column_name,
            )?);
            // Bound retained metadata. Readers keep their own Arc when evicted.
            if cache.len() >= 16 {
                cache.clear();
            }
            cache.insert(path, meta.clone());
            Ok(meta)
        })
        .await
        .map_err(|e| {
            datafusion_common::exec_datafusion_err!("FlatGeobuf metadata task failed: {e}")
        })?
    }
}
#[async_trait]
impl ExternalFormatSpec for FlatGeobufFormatSpec {
    async fn infer_schema(&self, object: &Object) -> Result<Schema> {
        Ok(self.metadata(object).await?.schema.as_ref().clone())
    }
    async fn open_reader(
        &self,
        args: &OpenReaderArgs,
    ) -> Result<Box<dyn RecordBatchReader + Send>> {
        let meta = self.metadata(&args.src).await?;
        let args = args.clone();
        tokio::task::spawn_blocking(move || reader::open(meta, &args))
            .await
            .map_err(|e| {
                datafusion_common::exec_datafusion_err!("FlatGeobuf reader task failed: {e}")
            })?
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
                    // Cached schemas depend on options; keep configurations isolated.
                    format.cache = Default::default();
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
