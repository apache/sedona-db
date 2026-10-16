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

use datafusion_common::Result;
use datafusion_datasource::file_groups::FileGroupPartitioner;
use datafusion_datasource::{
    TableSchema,
    file::FileSource,
    file_scan_config::FileScanConfig,
    file_stream::FileOpener,
    projection::{ProjectionOpener, SplitProjection},
};
use datafusion_execution::cache::cache_manager::FileMetadataCache;
use datafusion_physical_expr::LexOrdering;
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use object_store::ObjectStore;
use std::sync::Arc;
#[derive(Debug, Clone)]
pub(crate) struct FlatGeobufSource {
    table_schema: TableSchema,
    format: crate::format::FlatGeobufFormat,
    batch_size: Option<usize>,
    projection: ProjectionExprs,
    metrics: ExecutionPlanMetricsSet,
    pub(crate) cache: Option<Arc<dyn FileMetadataCache>>,
}
impl FlatGeobufSource {
    pub(crate) fn new(table_schema: TableSchema, format: crate::format::FlatGeobufFormat) -> Self {
        let projection = SplitProjection::unprojected(&table_schema).source;
        Self {
            table_schema,
            format,
            batch_size: None,
            projection,
            metrics: Default::default(),
            cache: None,
        }
    }
}
impl FileSource for FlatGeobufSource {
    fn create_file_opener(
        &self,
        store: Arc<dyn ObjectStore>,
        config: &FileScanConfig,
        _partition: usize,
    ) -> Result<Arc<dyn FileOpener>> {
        let schema = config.file_schema();
        let split = SplitProjection::new(schema, &self.projection);
        let opener = Arc::new(crate::opener::FlatGeobufOpener {
            store,
            cache: self.cache.clone(),
            format: self.format.clone(),
            projection: split.file_indices.clone(),
            batch_size: self.batch_size.unwrap_or(8192),
            schema: schema.clone(),
        });
        ProjectionOpener::try_new(split, opener, schema)
    }

    fn repartitioned(
        &self,
        target: usize,
        min_size: usize,
        ordering: Option<LexOrdering>,
        config: &FileScanConfig,
    ) -> Result<Option<FileScanConfig>> {
        if target < 2 || ordering.is_some() {
            return Ok(None);
        }
        let partitioner = FileGroupPartitioner::new()
            .with_target_partitions(target)
            .with_repartition_file_min_size(min_size);
        Ok(partitioner
            .repartition_file_groups(&config.file_groups)
            .map(|groups| {
                let mut conf = config.clone();
                conf.file_groups = groups;
                conf
            }))
    }

    fn table_schema(&self) -> &TableSchema {
        &self.table_schema
    }

    fn with_batch_size(&self, size: usize) -> Arc<dyn FileSource> {
        let mut s = self.clone();
        s.batch_size = Some(size);
        Arc::new(s)
    }

    fn projection(&self) -> Option<&ProjectionExprs> {
        Some(&self.projection)
    }

    fn metrics(&self) -> &ExecutionPlanMetricsSet {
        &self.metrics
    }

    fn file_type(&self) -> &str {
        "fgb"
    }

    fn try_pushdown_projection(
        &self,
        projection: &ProjectionExprs,
    ) -> Result<Option<Arc<dyn FileSource>>> {
        let mut s = self.clone();
        s.projection = self.projection.try_merge(projection)?;
        Ok(Some(Arc::new(s)))
    }
}
