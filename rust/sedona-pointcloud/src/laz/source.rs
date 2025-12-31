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

use std::{any::Any, sync::Arc};

use arrow_schema::Schema;
use datafusion_common::{config::ConfigOptions, error::DataFusionError, internal_err, Statistics};
use datafusion_datasource::{
    file::FileSource, file_scan_config::FileScanConfig, file_stream::FileOpener,
};
use datafusion_physical_expr::{conjunction, PhysicalExpr};
use datafusion_physical_plan::{
    filter_pushdown::{FilterPushdownPropagation, PushedDown},
    metrics::ExecutionPlanMetricsSet,
};
use object_store::ObjectStore;

use crate::laz::{opener::LazOpener, options::LasExtraBytes, reader::LazFileReaderFactory};

#[derive(Clone, Default, Debug)]
pub struct LazSource {
    /// Optional metrics
    metrics: ExecutionPlanMetricsSet,
    /// The schema of the file.
    pub(crate) table_schema: Option<Arc<Schema>>,
    /// Optional predicate for row filtering during parquet scan
    pub(crate) predicate: Option<Arc<dyn PhysicalExpr>>,
    /// Laz file reader factory
    pub(crate) laz_file_reader_factory: Option<Arc<LazFileReaderFactory>>,
    /// Batch size configuration
    pub(crate) batch_size: Option<usize>,
    pub(crate) projected_statistics: Option<Statistics>,
    pub(crate) extra_bytes: LasExtraBytes,
}

impl LazSource {
    pub fn with_extra_bytes(mut self, extra_bytes: LasExtraBytes) -> Self {
        self.extra_bytes = extra_bytes;
        self
    }
}

impl FileSource for LazSource {
    fn create_file_opener(
        &self,
        object_store: Arc<dyn ObjectStore>,
        base_config: &FileScanConfig,
        _partition: usize,
    ) -> Arc<dyn FileOpener> {
        let projection = base_config
            .file_column_projection_indices()
            .unwrap_or_else(|| (0..base_config.projected_file_schema().fields().len()).collect());

        let laz_file_reader_factory = self
            .laz_file_reader_factory
            .clone()
            .unwrap_or_else(|| Arc::new(LazFileReaderFactory::new(object_store, None)));

        Arc::new(LazOpener {
            projection: Arc::from(projection),
            limit: base_config.limit,
            predicate: self.predicate.clone(),
            laz_file_reader_factory,
            extra_bytes: self.extra_bytes,
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn with_batch_size(&self, batch_size: usize) -> Arc<dyn FileSource> {
        let mut conf = self.clone();
        conf.batch_size = Some(batch_size);
        Arc::new(conf)
    }

    fn with_schema(&self, schema: Arc<Schema>) -> Arc<dyn FileSource> {
        let mut conf = self.clone();
        conf.table_schema = Some(schema);
        Arc::new(conf)
    }

    fn with_projection(&self, _config: &FileScanConfig) -> Arc<dyn FileSource> {
        Arc::new(Self { ..self.clone() })
    }

    fn with_statistics(&self, statistics: Statistics) -> Arc<dyn FileSource> {
        let mut conf = self.clone();
        conf.projected_statistics = Some(statistics);
        Arc::new(conf)
    }

    fn metrics(&self) -> &ExecutionPlanMetricsSet {
        &self.metrics
    }

    fn statistics(&self) -> Result<Statistics, DataFusionError> {
        let Some(statistics) = &self.projected_statistics else {
            return internal_err!("projected_statistics must be set");
        };

        if self.filter().is_some() {
            Ok(statistics.clone().to_inexact())
        } else {
            Ok(statistics.clone())
        }
    }

    fn file_type(&self) -> &str {
        "laz"
    }

    fn try_pushdown_filters(
        &self,
        filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &ConfigOptions,
    ) -> Result<FilterPushdownPropagation<Arc<dyn FileSource>>, DataFusionError> {
        let mut source = self.clone();

        let predicate = match source.predicate {
            Some(predicate) => conjunction(std::iter::once(predicate).chain(filters.clone())),
            None => conjunction(filters.clone()),
        };

        source.predicate = Some(predicate);
        let source = Arc::new(source);

        // Tell our parents that they still have to handle the filters (they will only be used for stats pruning).
        Ok(FilterPushdownPropagation::with_parent_pushdown_result(vec![
            PushedDown::No;
            filters.len()
        ])
        .with_updated_node(source))
    }
}
