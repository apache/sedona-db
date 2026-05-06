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

use datafusion_common::{config::ConfigOptions, error::DataFusionError};
use datafusion_datasource::{
    file::FileSource,
    file_groups::FileGroupPartitioner,
    file_scan_config::FileScanConfig,
    file_stream::FileOpener,
    projection::{ProjectionOpener, SplitProjection},
    source::DataSource,
    TableSchema,
};
use datafusion_physical_expr::{
    conjunction, projection::ProjectionExprs, LexOrdering, PhysicalExpr,
};
use datafusion_physical_plan::{
    filter_pushdown::{FilterPushdownPropagation, PushedDown},
    metrics::ExecutionPlanMetricsSet,
};
use object_store::ObjectStore;

use crate::las::{
    format::Extension, opener::LasOpener, options::LasOptions, reader::LasFileReaderFactory,
};

#[derive(Clone, Debug)]
pub struct LasSource {
    /// Optional metrics
    metrics: ExecutionPlanMetricsSet,
    /// The schema of the file.
    pub(crate) table_schema: TableSchema,
    /// Optional predicate for row filtering during parquet scan
    pub(crate) predicate: Option<Arc<dyn PhysicalExpr>>,
    /// LAS/LAZ file reader factory
    pub(crate) reader_factory: Option<Arc<LasFileReaderFactory>>,
    /// Batch size configuration
    pub(crate) batch_size: Option<usize>,
    pub(crate) options: LasOptions,
    pub(crate) extension: Extension,
    /// Projection pushdown
    pub(crate) split_projection: Option<SplitProjection>,
}

impl LasSource {
    pub fn new(extension: Extension, table_schema: TableSchema) -> Self {
        Self {
            metrics: Default::default(),
            table_schema,
            predicate: Default::default(),
            reader_factory: Default::default(),
            batch_size: Default::default(),
            options: Default::default(),
            extension,
            split_projection: None,
        }
    }

    pub fn with_options(mut self, options: LasOptions) -> Self {
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
    ) -> Result<Arc<dyn FileOpener>, DataFusionError> {
        let file_reader_factory = self
            .reader_factory
            .clone()
            .unwrap_or_else(|| Arc::new(LasFileReaderFactory::new(object_store, None)));

        let inner_opener: Arc<dyn FileOpener> = Arc::new(LasOpener {
            batch_size: self.batch_size.expect("Must be set"),
            limit: base_config.limit,
            predicate: self.predicate.clone(),
            file_reader_factory,
            options: self.options.clone(),
            partition_count: base_config.output_partitioning().partition_count(),
            partition,
        });

        // Wrap with ProjectionOpener to handle reordering/expressions
        if let Some(split_projection) = &self.split_projection {
            ProjectionOpener::try_new(
                split_projection.clone(),
                inner_opener,
                self.table_schema.file_schema(),
            )
        } else {
            Ok(inner_opener)
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn with_batch_size(&self, batch_size: usize) -> Arc<dyn FileSource> {
        let mut conf = self.clone();
        conf.batch_size = Some(batch_size);
        Arc::new(conf)
    }

    fn try_pushdown_projection(
        &self,
        projection: &ProjectionExprs,
    ) -> Result<Option<Arc<dyn FileSource>>, DataFusionError> {
        // Use SplitProjection to handle any projection:
        // - file_indices provides column pruning (always works)
        // - ProjectionOpener handles reordering/expressions/renames after reading
        let split_projection = SplitProjection::new(self.table_schema.file_schema(), projection);

        Ok(Some(Arc::new(Self {
            split_projection: Some(split_projection),
            ..self.clone()
        })))
    }

    fn projection(&self) -> Option<&ProjectionExprs> {
        self.split_projection.as_ref().map(|sp| &sp.source)
    }

    fn metrics(&self) -> &ExecutionPlanMetricsSet {
        &self.metrics
    }

    fn table_schema(&self) -> &TableSchema {
        &self.table_schema
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
        if output_ordering.is_none() & self.options.round_robin_partitioning {
            // Custom round robin repartitioning
            //
            // The default way to partition a dataset to enable parallel reading
            // by DataFusion is through splitting files by byte ranges into the
            // number of target partitions. For selective queries on (partially)
            // ordered datasets that support pruning, this can result in unequal
            // resource use, as all the work is done on one partition while the
            // rest is pruned. Additionally, this breaks the existing locality
            // in the input when it is converted, as data from all partitions
            // ends up in each output row group. This approach addresses these
            // issues by partitioning the dataset using a round-robin scheme
            // across sequential chunks. This improves selective query performance
            // by more than half.
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
