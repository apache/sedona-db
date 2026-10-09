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
use datafusion_datasource::{FileRange, file_groups::FileGroup};
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
use std::{collections::HashMap, sync::Arc};
#[derive(Debug, Clone)]
pub(crate) struct FlatGeobufSource {
    table_schema: TableSchema,
    format: crate::format::FlatGeobufFormat,
    batch_size: Option<usize>,
    projection: ProjectionExprs,
    metrics: ExecutionPlanMetricsSet,
    pub(crate) cache: Option<Arc<dyn FileMetadataCache>>,
    pub(crate) planned: Arc<HashMap<object_store::path::Path, Arc<crate::metadata::Metadata>>>,
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
            planned: Default::default(),
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
        let mut groups = vec![vec![]; target];
        let mut group_index = 0;
        for group in &config.file_groups {
            for file in group.iter() {
                let data = self
                    .planned
                    .get(&file.object_meta.location)
                    .ok_or_else(|| {
                        datafusion_common::exec_datafusion_err!(
                            "Missing FlatGeobuf planning metadata"
                        )
                    })?;
                let count = data.offsets.len() - 1;
                if !data.indexed
                    || count == 0
                    || file.range.is_some()
                    || file.object_meta.size < (min_size as u64)
                {
                    groups[group_index % target].push(file.clone());
                    group_index += 1;
                    continue;
                }
                let parts = target.min(count);
                // Split at feature boundaries so every worker owns at least one feature.
                for i in 0..parts {
                    let mut part = file.clone();
                    part.range = Some(FileRange {
                        start: i64::try_from(
                            data.offsets[(count as u128 * i as u128 / parts as u128) as usize],
                        )
                        .map_err(|_| {
                            datafusion_common::exec_datafusion_err!("FlatGeobuf range overflow")
                        })?,
                        end: i64::try_from(
                            data.offsets
                                [(count as u128 * (i + 1) as u128 / parts as u128) as usize],
                        )
                        .map_err(|_| {
                            datafusion_common::exec_datafusion_err!("FlatGeobuf range overflow")
                        })?,
                    });
                    groups[group_index % target].push(part);
                    group_index += 1;
                }
            }
        }
        let mut conf = config.clone();
        conf.file_groups = groups
            .into_iter()
            .filter(|g| !g.is_empty())
            .map(FileGroup::new)
            .collect();
        Ok(Some(conf))
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
