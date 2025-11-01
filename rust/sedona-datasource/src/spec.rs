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

use std::{collections::HashMap, fmt::Debug, sync::Arc};

use arrow_array::RecordBatchReader;
use arrow_schema::{Schema, SchemaRef};
use async_trait::async_trait;

use datafusion::config::TableOptions;
use datafusion_common::{Result, Statistics};
use datafusion_execution::object_store::ObjectStoreUrl;
use datafusion_physical_expr::PhysicalExpr;
use object_store::{ObjectMeta, ObjectStore};

#[async_trait]
pub trait RecordBatchReaderFormatSpec: Debug + Send + Sync {
    fn extension(&self) -> &str;
    fn with_options(
        &self,
        options: &HashMap<String, String>,
    ) -> Result<Arc<dyn RecordBatchReaderFormatSpec>>;
    fn with_table_options(
        &self,
        table_options: &TableOptions,
    ) -> Arc<dyn RecordBatchReaderFormatSpec>;
    async fn infer_schema(&self, location: &Object) -> Result<Schema>;
    async fn infer_stats(&self, _location: &Object, table_schema: &Schema) -> Result<Statistics> {
        Ok(Statistics::new_unknown(table_schema))
    }
    async fn open_reader(&self, args: &OpenReaderArgs)
        -> Result<Box<dyn RecordBatchReader + Send>>;
}

#[derive(Debug, Clone)]
pub struct OpenReaderArgs {
    pub src: Object,
    pub batch_size: Option<usize>,
    pub file_schema: Option<SchemaRef>,
    pub file_projection: Option<Vec<usize>>,
    pub filters: Option<Vec<Arc<dyn PhysicalExpr>>>,
}

#[derive(Debug, Clone)]
pub struct Object {
    pub store: Arc<dyn ObjectStore>,
    pub url: Option<ObjectStoreUrl>,
    pub meta: Option<ObjectMeta>,
}

impl Object {
    pub fn to_url_string(&self) -> String {
        match (&self.url, &self.meta) {
            (None, None) => format!("{:?}", self.store),
            (None, Some(meta)) => {
                // There's no great way to map an object_store to a url prefix if we're not
                // provided the `url`; however, this is what we have access to on occasion.
                // This is a heuristic that should work for https and a local filesystem,
                // which is what we might be able to expect a non-DataFusion system like
                // GDAL to be able to translate.
                let object_store_debug = format!("{:?}", self.store).to_lowercase();
                if object_store_debug.contains("http") {
                    format!("https://{}", meta.location)
                } else if object_store_debug.contains("local") {
                    format!("file://{}", meta.location)
                } else {
                    format!("{object_store_debug}: {}", meta.location)
                }
            }
            (Some(url), None) => url.to_string(),
            (Some(url), Some(meta)) => format!("{url}/{}", meta.location),
        }
    }
}
