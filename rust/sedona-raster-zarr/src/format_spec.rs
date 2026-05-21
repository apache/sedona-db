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

//! `ExternalFormatSpec` impl for Zarr groups.
//!
//! Wraps [`ZarrChunkReader`] with the standard SedonaDB datasource API
//! so users can read Zarr groups via `con.read_format(spec, uri)` or
//! through `ListingTable`-style file discovery.

use std::{collections::HashMap, sync::Arc};

use arrow_array::RecordBatchReader;
use arrow_schema::Schema;
use async_trait::async_trait;
use datafusion_common::{DataFusionError, Result};
use sedona_datasource::spec::{ExternalFormatSpec, Object, OpenReaderArgs};
use sedona_schema::datatypes::SedonaType;

use crate::loader::ZarrChunkReader;

/// `ExternalFormatSpec` implementation for Zarr groups.
///
/// Configurable via [`with_options`](ExternalFormatSpec::with_options):
/// - `load_eager`: boolean. `false` (default) emits chunk-anchor URIs
///   only. `true` currently errors — pixel-byte materialisation is
///   pending the async `RS_EnsureLoaded` resolver.
/// - `arrays`: JSON array of strings, e.g. `'["temperature","pressure"]'`.
///   Names a subset of arrays in the group to read; defaults to every
///   multi-dimensional array (1-D coord variables auto-skipped).
#[derive(Debug, Clone, Default)]
pub struct ZarrFormatSpec {
    load_eager: bool,
    arrays: Option<Vec<String>>,
}

impl ZarrFormatSpec {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl ExternalFormatSpec for ZarrFormatSpec {
    async fn infer_schema(&self, _location: &Object) -> Result<Schema> {
        // The Zarr loader always produces the canonical single-column
        // Raster schema. No I/O needed to know that.
        let field = SedonaType::Raster
            .to_storage_field("raster", true)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        Ok(Schema::new(vec![field]))
    }

    async fn open_reader(
        &self,
        args: &OpenReaderArgs,
    ) -> Result<Box<dyn RecordBatchReader + Send>> {
        if self.load_eager {
            return Err(DataFusionError::Plan(
                "ZarrFormatSpec: load_eager = true is not yet supported. \
                 Pixel-byte materialisation will be wired up when the async \
                 RS_EnsureLoaded resolver lands."
                    .into(),
            ));
        }
        let uri = args.src.to_url_string().ok_or_else(|| {
            DataFusionError::Plan(
                "ZarrFormatSpec: could not resolve a URL string from the source object".into(),
            )
        })?;
        let batch_size = args.batch_size.unwrap_or(8192);
        let reader = ZarrChunkReader::try_new(&uri, self.arrays.as_deref(), batch_size)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        Ok(Box::new(reader))
    }

    fn with_options(
        &self,
        options: &HashMap<String, String>,
    ) -> Result<Arc<dyn ExternalFormatSpec>> {
        let mut next = self.clone();
        for (k, v) in options {
            match k.as_str() {
                "load_eager" => {
                    next.load_eager = v.parse().map_err(|_| {
                        DataFusionError::Plan(format!(
                            "ZarrFormatSpec: load_eager must be a boolean; got {v:?}"
                        ))
                    })?;
                }
                "arrays" => {
                    next.arrays = Some(serde_json::from_str::<Vec<String>>(v).map_err(|e| {
                        DataFusionError::Plan(format!(
                            "ZarrFormatSpec: arrays must be a JSON array of strings; \
                                 got {v:?} ({e})"
                        ))
                    })?);
                }
                other => {
                    return Err(DataFusionError::Plan(format!(
                        "ZarrFormatSpec: unknown option {other:?}"
                    )));
                }
            }
        }
        Ok(Arc::new(next))
    }

    fn extension(&self) -> &str {
        ".zarr"
    }

    fn list_single_object(&self) -> bool {
        // A Zarr group is a directory, not a file. The DataFusion
        // listing layer can't enumerate it as a single object — it
        // would return the directory's contents (zarr.json, chunk
        // shards, ...), none of which carry the `.zarr` extension.
        // Routing through the single-object provider keeps the URI
        // intact and hands it to `open_reader` directly.
        true
    }
}
