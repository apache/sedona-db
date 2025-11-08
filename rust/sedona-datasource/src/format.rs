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
            FileGroupPartitioner, FileMeta, FileOpenFuture, FileOpener, FileScanConfig,
            FileSinkConfig, FileSource,
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

use crate::spec::{ExternalFormatSpec, Object, OpenReaderArgs, SupportsRepartition};

/// Create a [FileFormatFactory] from a [ExternalFormatSpec]
///
/// The FileFormatFactory is the object that may be registered with a
/// SessionStateBuilder to allow SQL queries to access this format.
#[derive(Debug)]
pub struct ExternalFormatFactory {
    spec: Arc<dyn ExternalFormatSpec>,
}

impl ExternalFormatFactory {
    pub fn new(spec: Arc<dyn ExternalFormatSpec>) -> Self {
        Self { spec }
    }
}

impl FileFormatFactory for ExternalFormatFactory {
    fn create(
        &self,
        _state: &dyn Session,
        format_options: &HashMap<String, String>,
    ) -> Result<Arc<dyn FileFormat>> {
        Ok(Arc::new(ExternalFileFormat {
            spec: self.spec.with_options(format_options)?,
        }))
    }

    fn default(&self) -> Arc<dyn FileFormat> {
        Arc::new(ExternalFileFormat {
            spec: self.spec.clone(),
        })
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl GetExt for ExternalFormatFactory {
    fn get_ext(&self) -> String {
        self.spec.extension().to_string()
    }
}

#[derive(Debug)]
pub(crate) struct ExternalFileFormat {
    spec: Arc<dyn ExternalFormatSpec>,
}

impl ExternalFileFormat {
    pub fn new(spec: Arc<dyn ExternalFormatSpec>) -> Self {
        Self { spec }
    }
}

#[async_trait]
impl FileFormat for ExternalFileFormat {
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
                        store: Some(store.clone()),
                        url: None,
                        meta: Some(object.clone()),
                        range: None,
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
                    store: Some(store.clone()),
                    url: None,
                    meta: Some(object.clone()),
                    range: None,
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
        not_impl_err!("writing not yet supported for ExternalFileFormat")
    }

    fn file_source(&self) -> Arc<dyn FileSource> {
        Arc::new(ExternalFileSource::new(self.spec.clone()))
    }
}

#[derive(Debug, Clone)]
struct ExternalFileSource {
    spec: Arc<dyn ExternalFormatSpec>,
    batch_size: Option<usize>,
    file_schema: Option<SchemaRef>,
    file_projection: Option<Vec<usize>>,
    filters: Vec<Arc<dyn PhysicalExpr>>,
    metrics: ExecutionPlanMetricsSet,
    projected_statistics: Option<Statistics>,
}

impl ExternalFileSource {
    pub fn new(spec: Arc<dyn ExternalFormatSpec>) -> Self {
        Self {
            spec,
            batch_size: None,
            file_schema: None,
            file_projection: None,
            filters: Vec::new(),
            metrics: ExecutionPlanMetricsSet::default(),
            projected_statistics: None,
        }
    }
}

impl FileSource for ExternalFileSource {
    fn create_file_opener(
        &self,
        store: Arc<dyn ObjectStore>,
        base_config: &FileScanConfig,
        _partition: usize,
    ) -> Arc<dyn FileOpener> {
        let args = OpenReaderArgs {
            src: Object {
                store: Some(store.clone()),
                url: Some(base_config.object_store_url.clone()),
                meta: None,
                range: None,
            },
            batch_size: self.batch_size,
            file_schema: self.file_schema.clone(),
            file_projection: self.file_projection.clone(),
            filters: self.filters.clone(),
        };

        Arc::new(ExternalFileOpener {
            spec: self.spec.clone(),
            args,
        })
    }

    fn try_pushdown_filters(
        &self,
        filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &ConfigOptions,
    ) -> Result<FilterPushdownPropagation<Arc<dyn FileSource>>> {
        // Record any new filters
        let num_filters = filters.len();
        let mut new_filters = self.filters.clone();
        new_filters.extend(filters);
        let source = Self {
            filters: new_filters,
            ..self.clone()
        };

        // ...but don't indicate that we handled them so that the filters are
        // applied by the other node.
        Ok(FilterPushdownPropagation::with_parent_pushdown_result(vec![
            PushedDown::No;
            num_filters
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

    fn repartitioned(
        &self,
        target_partitions: usize,
        repartition_file_min_size: usize,
        output_ordering: Option<LexOrdering>,
        config: &FileScanConfig,
    ) -> Result<Option<FileScanConfig>> {
        match self.spec.supports_repartition() {
            SupportsRepartition::None => Ok(None),
            SupportsRepartition::ByRange => {
                // Default implementation
                if config.file_compression_type.is_compressed() || config.new_lines_in_values {
                    return Ok(None);
                }

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
                Ok(None)
            }
        }
    }
}

#[derive(Debug, Clone)]
struct ExternalFileOpener {
    spec: Arc<dyn ExternalFormatSpec>,
    args: OpenReaderArgs,
}

impl FileOpener for ExternalFileOpener {
    fn open(&self, file_meta: FileMeta, _file: PartitionedFile) -> Result<FileOpenFuture> {
        let mut self_clone = self.clone();
        Ok(Box::pin(async move {
            self_clone.args.src.meta.replace(file_meta.object_meta);
            self_clone.args.src.range = file_meta.range;
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
        assert_batches_eq,
        datasource::listing::ListingTableUrl,
        execution::SessionStateBuilder,
        prelude::{col, lit, SessionContext},
    };
    use datafusion_common::plan_err;
    use std::{
        io::{Read, Write},
        path::PathBuf,
    };
    use tempfile::TempDir;
    use url::Url;

    use crate::provider::external_listing_table;

    use super::*;

    fn create_echo_spec_ctx() -> SessionContext {
        let spec = Arc::new(EchoSpec::default());
        let factory = ExternalFormatFactory::new(spec.clone());

        // Register the format
        let mut state = SessionStateBuilder::new().build();
        state.register_file_format(Arc::new(factory), true).unwrap();
        SessionContext::new_with_state(state).enable_url_table()
    }

    fn create_echo_spec_temp_dir() -> (TempDir, Vec<PathBuf>) {
        // Create a temporary directory with a few files with the declared extension
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
        (temp_dir, vec![file0, file1])
    }

    fn check_object_is_readable_file(location: &Object) {
        let url = Url::parse(&location.to_url_string().unwrap()).expect("valid uri");
        assert_eq!(url.scheme(), "file");
        let path = url.to_file_path().expect("can extract file path");

        let mut content = String::new();
        std::fs::File::open(path)
            .expect("url can't be opened")
            .read_to_string(&mut content)
            .expect("failed to read");
        if content.is_empty() {
            panic!("empty file at url {url}");
        }
    }

    #[derive(Debug, Default, Clone)]
    struct EchoSpec {
        option_value: Option<String>,
    }

    #[async_trait]
    impl ExternalFormatSpec for EchoSpec {
        fn extension(&self) -> &str {
            "echospec"
        }

        fn with_options(
            &self,
            options: &HashMap<String, String>,
        ) -> Result<Arc<dyn ExternalFormatSpec>> {
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

        async fn infer_schema(&self, location: &Object) -> Result<Schema> {
            check_object_is_readable_file(location);
            Ok(Schema::new(vec![
                Field::new("src", DataType::Utf8, true),
                Field::new("batch_size", DataType::Int64, true),
                Field::new("filter_count", DataType::Int32, true),
                Field::new("option_value", DataType::Utf8, true),
            ]))
        }

        async fn infer_stats(
            &self,
            location: &Object,
            table_schema: &Schema,
        ) -> Result<Statistics> {
            check_object_is_readable_file(location);
            Ok(Statistics::new_unknown(table_schema))
        }

        async fn open_reader(
            &self,
            args: &OpenReaderArgs,
        ) -> Result<Box<dyn RecordBatchReader + Send>> {
            check_object_is_readable_file(&args.src);

            let src: StringArray = [args.src.clone()]
                .iter()
                .map(|item| Some(item.to_url_string().unwrap()))
                .collect();
            let batch_size: Int64Array = [args.batch_size]
                .iter()
                .map(|item| item.map(|i| i as i64))
                .collect();
            let filter_count: Int32Array = [args.filters.len() as i32].into_iter().collect();
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
        let ctx = create_echo_spec_ctx();
        let (temp_dir, files) = create_echo_spec_temp_dir();

        // Select using just the filename and ensure we get a result
        let batches_item0 = ctx
            .table(files[0].to_string_lossy().to_string())
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(batches_item0.len(), 1);
        assert_eq!(batches_item0[0].num_rows(), 1);

        // With a glob we should get all the files
        let batches = ctx
            .table(format!("{}/*.echospec", temp_dir.path().to_string_lossy()))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        // We should get one value per partition
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].num_rows(), 1);
        assert_eq!(batches[1].num_rows(), 1);
    }

    #[tokio::test]
    async fn spec_format_project_filter() {
        let ctx = create_echo_spec_ctx();
        let (temp_dir, _files) = create_echo_spec_temp_dir();

        // Ensure that if we pass
        let batches = ctx
            .table(format!("{}/*.echospec", temp_dir.path().to_string_lossy()))
            .await
            .unwrap()
            .filter(col("src").like(lit("%item0%")))
            .unwrap()
            .select(vec![col("batch_size"), col("filter_count")])
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_batches_eq!(
            [
                "+------------+--------------+",
                "| batch_size | filter_count |",
                "+------------+--------------+",
                "| 8192       | 1            |",
                "+------------+--------------+",
            ],
            &batches
        );
    }

    #[tokio::test]
    async fn spec_listing_table() {
        let spec = Arc::new(EchoSpec::default());
        let ctx = SessionContext::new();
        let (_temp_dir, files) = create_echo_spec_temp_dir();

        // Select using a listing table and ensure we get a result
        let provider = external_listing_table(
            spec,
            &ctx,
            files
                .iter()
                .map(|f| ListingTableUrl::parse(f.to_string_lossy()).unwrap())
                .collect(),
            true,
        )
        .await
        .unwrap();

        let batches = ctx
            .read_table(Arc::new(provider))
            .unwrap()
            .collect()
            .await
            .unwrap();

        // We should get one value per partition
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].num_rows(), 1);
        assert_eq!(batches[1].num_rows(), 1);
    }

    #[tokio::test]
    async fn spec_listing_table_options() {
        let spec = Arc::new(EchoSpec::default())
            .with_options(&[("option_value".to_string(), "foofy".to_string())].into())
            .unwrap();

        let ctx = SessionContext::new();
        let (_temp_dir, files) = create_echo_spec_temp_dir();

        // Select using a listing table and ensure we get a result with the option passed
        let provider = external_listing_table(
            spec,
            &ctx,
            files
                .iter()
                .map(|f| ListingTableUrl::parse(f.to_string_lossy()).unwrap())
                .collect(),
            true,
        )
        .await
        .unwrap();

        let batches = ctx
            .read_table(Arc::new(provider))
            .unwrap()
            .select(vec![col("batch_size"), col("option_value")])
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_batches_eq!(
            [
                "+------------+--------------+",
                "| batch_size | option_value |",
                "+------------+--------------+",
                "| 8192       | foofy        |",
                "| 8192       | foofy        |",
                "+------------+--------------+",
            ],
            &batches
        );
    }

    #[tokio::test]
    async fn spec_listing_table_errors() {
        let spec = Arc::new(EchoSpec::default())
            .with_options(&[("option_value".to_string(), "foofy".to_string())].into())
            .unwrap();

        let ctx = SessionContext::new();
        let (temp_dir, mut files) = create_echo_spec_temp_dir();

        // Listing table with no files should error
        let err = external_listing_table(spec.clone(), &ctx, vec![], true)
            .await
            .unwrap_err();
        assert_eq!(err.message(), "No table paths were provided");

        // Create a file with a different extension
        let file2 = temp_dir.path().join("item2.echospecNOT");
        std::fs::File::create(&file2)
            .unwrap()
            .write_all(b"not empty")
            .unwrap();
        files.push(file2);

        // With check_extension as true we should get an error
        let err = external_listing_table(
            spec.clone(),
            &ctx,
            files
                .iter()
                .map(|f| ListingTableUrl::parse(f.to_string_lossy()).unwrap())
                .collect(),
            true,
        )
        .await
        .unwrap_err();

        assert!(err
            .message()
            .ends_with("does not match the expected extension 'echospec'"));

        // ...but we should be able to turn off the error
        external_listing_table(
            spec,
            &ctx,
            files
                .iter()
                .map(|f| ListingTableUrl::parse(f.to_string_lossy()).unwrap())
                .collect(),
            false,
        )
        .await
        .unwrap();
    }
}
