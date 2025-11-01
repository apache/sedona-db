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

use std::{any::Any, collections::HashMap, fmt::Debug, sync::Arc};

use arrow_schema::{Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::{
    config::ConfigOptions,
    datasource::{
        file_format::{file_compression_type::FileCompressionType, FileFormat, FileFormatFactory},
        listing::PartitionedFile,
        physical_plan::{
            FileMeta, FileOpenFuture, FileOpener, FileScanConfig, FileSinkConfig, FileSource,
        },
    },
};
use datafusion_catalog::{memory::DataSourceExec, Session};
use datafusion_common::{not_impl_err, DataFusionError, GetExt, Result, Statistics};
use datafusion_physical_expr::{LexOrdering, LexRequirement, PhysicalExpr};
use datafusion_physical_plan::{
    filter_pushdown::{FilterPushdownPropagation, PushedDown},
    metrics::ExecutionPlanMetricsSet,
    ExecutionPlan,
};
use futures::{StreamExt, TryStreamExt};
use object_store::{ObjectMeta, ObjectStore};
use sedona_common::sedona_internal_err;

use crate::spec::{Object, OpenReaderArgs, RecordBatchReaderFormatSpec};

#[derive(Debug)]
pub struct RecordBatchReaderFormatFactory {
    spec: Arc<dyn RecordBatchReaderFormatSpec>,
}

impl RecordBatchReaderFormatFactory {
    pub fn new(spec: Arc<dyn RecordBatchReaderFormatSpec>) -> Self {
        Self { spec }
    }
}

impl FileFormatFactory for RecordBatchReaderFormatFactory {
    fn create(
        &self,
        _state: &dyn Session,
        format_options: &HashMap<String, String>,
    ) -> Result<Arc<dyn FileFormat>> {
        Ok(Arc::new(RecordBatchReaderFormat {
            spec: self.spec.with_options(format_options)?,
        }))
    }

    fn default(&self) -> Arc<dyn FileFormat> {
        Arc::new(RecordBatchReaderFormat {
            spec: self.spec.clone(),
        })
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl GetExt for RecordBatchReaderFormatFactory {
    fn get_ext(&self) -> String {
        self.spec.extension().to_string()
    }
}

#[derive(Debug)]
pub struct RecordBatchReaderFormat {
    spec: Arc<dyn RecordBatchReaderFormatSpec>,
}

impl RecordBatchReaderFormat {
    pub fn new(spec: Arc<dyn RecordBatchReaderFormatSpec>) -> Self {
        Self { spec }
    }
}

#[async_trait]
impl FileFormat for RecordBatchReaderFormat {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_ext(&self) -> String {
        self.spec.extension().to_string()
    }

    fn get_ext_with_compression(
        &self,
        _file_compression_type: &FileCompressionType,
    ) -> Result<String> {
        not_impl_err!("extension with compression type")
    }

    fn compression_type(&self) -> Option<FileCompressionType> {
        None
    }

    async fn infer_schema(
        &self,
        state: &dyn Session,
        store: &Arc<dyn ObjectStore>,
        objects: &[ObjectMeta],
    ) -> Result<SchemaRef> {
        let mut schemas: Vec<_> = futures::stream::iter(objects)
            .map(|object| async move {
                let schema = self
                    .spec
                    .infer_schema(&Object {
                        store: store.clone(),
                        url: None,
                        meta: Some(object.clone()),
                    })
                    .await?;
                Ok::<_, DataFusionError>((object.location.clone(), schema))
            })
            .boxed() // Workaround https://github.com/rust-lang/rust/issues/64552
            .buffered(state.config_options().execution.meta_fetch_concurrency)
            .try_collect()
            .await?;

        schemas.sort_by(|(location1, _), (location2, _)| location1.cmp(location2));

        let schemas = schemas
            .into_iter()
            .map(|(_, schema)| schema)
            .collect::<Vec<_>>();

        let schema = Schema::try_merge(schemas)?;
        Ok(Arc::new(schema))
    }

    async fn infer_stats(
        &self,
        _state: &dyn Session,
        store: &Arc<dyn ObjectStore>,
        table_schema: SchemaRef,
        object: &ObjectMeta,
    ) -> Result<Statistics> {
        self.spec
            .infer_stats(
                &Object {
                    store: store.clone(),
                    url: None,
                    meta: Some(object.clone()),
                },
                &table_schema,
            )
            .await
    }

    async fn create_physical_plan(
        &self,
        _state: &dyn Session,
        config: FileScanConfig,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(DataSourceExec::from_data_source(config))
    }

    async fn create_writer_physical_plan(
        &self,
        _input: Arc<dyn ExecutionPlan>,
        _state: &dyn Session,
        _conf: FileSinkConfig,
        _order_requirements: Option<LexRequirement>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        not_impl_err!("writing not yet supported for SimpleSedonaFormat")
    }

    fn file_source(&self) -> Arc<dyn FileSource> {
        Arc::new(SedonaFileSource::new(self.spec.clone()))
    }
}

#[derive(Debug, Clone)]
struct SedonaFileSource {
    spec: Arc<dyn RecordBatchReaderFormatSpec>,
    batch_size: Option<usize>,
    file_schema: Option<SchemaRef>,
    file_projection: Option<Vec<usize>>,
    filters: Option<Vec<Arc<dyn PhysicalExpr>>>,
    metrics: ExecutionPlanMetricsSet,
    projected_statistics: Option<Statistics>,
}

impl SedonaFileSource {
    pub fn new(spec: Arc<dyn RecordBatchReaderFormatSpec>) -> Self {
        Self {
            spec,
            batch_size: None,
            file_schema: None,
            file_projection: None,
            filters: None,
            metrics: ExecutionPlanMetricsSet::default(),
            projected_statistics: None,
        }
    }
}

impl FileSource for SedonaFileSource {
    fn create_file_opener(
        &self,
        store: Arc<dyn ObjectStore>,
        base_config: &FileScanConfig,
        partition: usize,
    ) -> Arc<dyn FileOpener> {
        let args = OpenReaderArgs {
            src: Object {
                store: store.clone(),
                url: Some(base_config.object_store_url.clone()),
                meta: None,
            },
            batch_size: self.batch_size,
            file_schema: self.file_schema.clone(),
            file_projection: self.file_projection.clone(),
            filters: self.filters.clone(),
        };

        Arc::new(SimpleOpener {
            spec: self.spec.clone(),
            args,
            partition,
        })
    }

    fn try_pushdown_filters(
        &self,
        filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &ConfigOptions,
    ) -> Result<FilterPushdownPropagation<Arc<dyn FileSource>>> {
        let source = Self {
            filters: Some(filters.clone()),
            ..self.clone()
        };

        Ok(FilterPushdownPropagation::with_parent_pushdown_result(vec![
            PushedDown::No;
            filters.len()
        ])
        .with_updated_node(Arc::new(source)))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn with_batch_size(&self, batch_size: usize) -> Arc<dyn FileSource> {
        Arc::new(Self {
            batch_size: Some(batch_size),
            ..self.clone()
        })
    }

    fn with_schema(&self, schema: SchemaRef) -> Arc<dyn FileSource> {
        Arc::new(Self {
            file_schema: Some(schema),
            ..self.clone()
        })
    }

    fn with_projection(&self, config: &FileScanConfig) -> Arc<dyn FileSource> {
        Arc::new(Self {
            file_projection: config.file_column_projection_indices(),
            ..self.clone()
        })
    }

    fn with_statistics(&self, statistics: Statistics) -> Arc<dyn FileSource> {
        Arc::new(Self {
            projected_statistics: Some(statistics),
            ..self.clone()
        })
    }

    fn metrics(&self) -> &ExecutionPlanMetricsSet {
        &self.metrics
    }

    fn statistics(&self) -> Result<Statistics> {
        let statistics = &self.projected_statistics;
        Ok(statistics
            .clone()
            .expect("projected_statistics must be set"))
    }

    fn file_type(&self) -> &str {
        self.spec.extension()
    }

    // File formats implemented in this way can't be repartitioned. File formats that
    // benefit from this need their own FileFormat implementation.
    fn repartitioned(
        &self,
        _target_partitions: usize,
        _repartition_file_min_size: usize,
        _output_ordering: Option<LexOrdering>,
        _config: &FileScanConfig,
    ) -> Result<Option<FileScanConfig>> {
        Ok(None)
    }
}

#[derive(Debug, Clone)]
struct SimpleOpener {
    spec: Arc<dyn RecordBatchReaderFormatSpec>,
    args: OpenReaderArgs,
    partition: usize,
}

impl FileOpener for SimpleOpener {
    fn open(&self, file_meta: FileMeta, _file: PartitionedFile) -> Result<FileOpenFuture> {
        if self.partition != 0 {
            return sedona_internal_err!("Expected SimpleOpener to open a single partition");
        }

        let mut self_clone = self.clone();
        Ok(Box::pin(async move {
            self_clone.args.src.meta.replace(file_meta.object_meta);
            let reader = self_clone.spec.open_reader(&self_clone.args).await?;
            let stream =
                futures::stream::iter(reader.into_iter().map(|batch| batch.map_err(Into::into)));
            Ok(stream.boxed())
        }))
    }
}

#[cfg(test)]
mod test {

    use arrow_array::{
        Int32Array, Int64Array, RecordBatch, RecordBatchIterator, RecordBatchReader, StringArray,
    };
    use arrow_schema::{DataType, Field};
    use datafusion::{
        config::TableOptions, execution::SessionStateBuilder, prelude::SessionContext,
    };
    use datafusion_common::plan_err;
    use std::io::Write;
    use tempfile::TempDir;

    use super::*;

    #[derive(Debug, Default, Clone)]
    struct EchoSpec {
        option_value: Option<String>,
    }

    #[async_trait]
    impl RecordBatchReaderFormatSpec for EchoSpec {
        fn extension(&self) -> &str {
            "echospec"
        }

        fn with_options(
            &self,
            options: &HashMap<String, String>,
        ) -> Result<Arc<dyn RecordBatchReaderFormatSpec>> {
            let mut self_clone = self.clone();
            for (k, v) in options {
                if k == "option_value" {
                    self_clone.option_value = Some(v.to_string());
                } else {
                    return plan_err!("Unsupported option for EchoSpec: '{k}'");
                }
            }

            Ok(Arc::new(self_clone))
        }

        fn with_table_options(
            &self,
            _table_options: &TableOptions,
        ) -> Arc<dyn RecordBatchReaderFormatSpec> {
            Arc::new(self.clone())
        }

        async fn infer_schema(&self, _location: &Object) -> Result<Schema> {
            Ok(Schema::new(vec![
                Field::new("src", DataType::Utf8, true),
                Field::new("batch_size", DataType::Int64, true),
                Field::new("filter_count", DataType::Int32, true),
                Field::new("option_value", DataType::Utf8, true),
            ]))
        }

        async fn infer_stats(
            &self,
            _location: &Object,
            table_schema: &Schema,
        ) -> Result<Statistics> {
            Ok(Statistics::new_unknown(table_schema))
        }

        async fn open_reader(
            &self,
            args: &OpenReaderArgs,
        ) -> Result<Box<dyn RecordBatchReader + Send>> {
            let src: StringArray = [args.src.clone()]
                .iter()
                .map(|item| Some(item.to_url_string()))
                .collect();
            let batch_size: Int64Array = [args.batch_size]
                .iter()
                .map(|item| item.map(|i| i as i64))
                .collect();
            let filter_count: Int32Array = [args.filters.clone().map(|f| f.len() as i32)]
                .iter()
                .collect();
            let option_value: StringArray = [self.option_value.clone()].iter().collect();

            let schema = Arc::new(self.infer_schema(&args.src).await?);
            let mut batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(src),
                    Arc::new(batch_size),
                    Arc::new(filter_count),
                    Arc::new(option_value),
                ],
            )?;

            if let Some(projection) = &args.file_projection {
                batch = batch.project(projection)?;
            }

            Ok(Box::new(RecordBatchIterator::new([Ok(batch)], schema)))
        }
    }

    #[tokio::test]
    async fn spec_format() {
        let spec = Arc::new(EchoSpec::default());
        let factory = RecordBatchReaderFormatFactory::new(spec);

        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path();
        let file0 = temp_path.join("item0.echospec");
        std::fs::File::create(&file0)
            .unwrap()
            .write_all(b"not empty")
            .unwrap();
        let file1 = temp_path.join("item1.echospec");
        std::fs::File::create(&file1)
            .unwrap()
            .write_all(b"not empty")
            .unwrap();

        let mut state = SessionStateBuilder::new().build();
        state.register_file_format(Arc::new(factory), true).unwrap();
        let ctx = SessionContext::new_with_state(state).enable_url_table();

        let batches_item0 = ctx
            .table(file0.to_string_lossy().to_string())
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(batches_item0.len(), 1);
        assert_eq!(batches_item0[0].num_rows(), 1);
    }
}
