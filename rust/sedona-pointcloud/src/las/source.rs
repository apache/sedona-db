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

use std::{any::Any, iter, sync::Arc};

use datafusion_common::{config::ConfigOptions, error::DataFusionError, Statistics};
use datafusion_datasource::{
    file::FileSource, file_groups::FileGroupPartitioner, file_scan_config::FileScanConfig,
    file_stream::FileOpener, source::DataSource, TableSchema,
};
use datafusion_physical_expr::{conjunction, LexOrdering, PhysicalExpr};
use datafusion_physical_plan::{
    filter_pushdown::{FilterPushdownPropagation, PushedDown},
    metrics::ExecutionPlanMetricsSet,
};
use object_store::ObjectStore;

use crate::{
    las::{format::Extension, opener::LasOpener, reader::LasFileReaderFactory},
    options::PointcloudOptions,
};

#[derive(Clone, Debug)]
pub struct LasSource {
    /// Optional metrics
    metrics: ExecutionPlanMetricsSet,
    /// The schema of the file.
    pub(crate) table_schema: Option<TableSchema>,
    /// Optional predicate for row filtering during parquet scan
    pub(crate) predicate: Option<Arc<dyn PhysicalExpr>>,
    /// LAS/LAZ file reader factory
    pub(crate) reader_factory: Option<Arc<LasFileReaderFactory>>,
    /// Batch size configuration
    pub(crate) batch_size: Option<usize>,
    pub(crate) projected_statistics: Option<Statistics>,
    pub(crate) options: PointcloudOptions,
    pub(crate) extension: Extension,
}

impl LasSource {
    pub fn new(extension: Extension) -> Self {
        Self {
            metrics: Default::default(),
            table_schema: Default::default(),
            predicate: Default::default(),
            reader_factory: Default::default(),
            batch_size: Default::default(),
            projected_statistics: Default::default(),
            options: Default::default(),
            extension,
        }
    }

    pub fn with_options(mut self, options: PointcloudOptions) -> Self {
        self.options = options;
        self
    }

    pub fn with_reader_factory(mut self, reader_factory: Arc<LasFileReaderFactory>) -> Self {
        self.reader_factory = Some(reader_factory);
        self
    }
}

impl FileSource for LasSource {
    fn create_file_opener(
        &self,
        object_store: Arc<dyn ObjectStore>,
        base_config: &FileScanConfig,
        partition: usize,
    ) -> Arc<dyn FileOpener> {
        let projection = base_config
            .file_column_projection_indices()
            .unwrap_or_else(|| (0..base_config.projected_file_schema().fields().len()).collect());

        let file_reader_factory = self
            .reader_factory
            .clone()
            .unwrap_or_else(|| Arc::new(LasFileReaderFactory::new(object_store, None)));

        Arc::new(LasOpener {
            projection: Arc::from(projection),
            batch_size: self.batch_size.expect("Must be set"),
            limit: base_config.limit,
            predicate: self.predicate.clone(),
            file_reader_factory,
            options: self.options.clone(),
            partition_count: base_config.output_partitioning().partition_count(),
            partition,
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

    fn with_schema(&self, schema: TableSchema) -> Arc<dyn FileSource> {
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
            return Err(DataFusionError::External(
                "projected_statistics must be set".into(),
            ));
        };

        if self.filter().is_some() {
            Ok(statistics.clone().to_inexact())
        } else {
            Ok(statistics.clone())
        }
    }

    fn file_type(&self) -> &str {
        self.extension.as_str()
    }

    fn repartitioned(
        &self,
        target_partitions: usize,
        repartition_file_min_size: usize,
        output_ordering: Option<LexOrdering>,
        config: &FileScanConfig,
    ) -> Result<Option<FileScanConfig>, DataFusionError> {
        if false | output_ordering.is_none() & self.options.round_robin_partitioning {
            // Custom round robin repartitioning
            let mut config = config.clone();
            config.file_groups = config
                .file_groups
                .into_iter()
                .flat_map(|fg| iter::repeat_n(fg, target_partitions))
                .collect();
            return Ok(Some(config));
        } else {
            // Default byte range repartitioning
            let repartitioned_file_groups_option = FileGroupPartitioner::new()
                .with_target_partitions(target_partitions)
                .with_repartition_file_min_size(repartition_file_min_size)
                .with_preserve_order_within_groups(output_ordering.is_some())
                .repartition_file_groups(&config.file_groups);

            if let Some(repartitioned_file_groups) = repartitioned_file_groups_option {
                let mut source = config.clone();
                source.file_groups = repartitioned_file_groups;
                return Ok(Some(source));
            }
        }

        Ok(None)
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
