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

//! GDAL-backed raster file formats.

use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{Schema, SchemaRef};
use async_trait::async_trait;
use datafusion_catalog::{Session, memory::DataSourceExec};
use datafusion_common::{
    DataFusionError, GetExt, Result, Statistics, config::ConfigOptions, exec_datafusion_err,
    not_impl_err, plan_err, tree_node::TreeNodeRecursion,
};
use datafusion_datasource::{
    PartitionedFile, TableSchema,
    file::FileSource,
    file_compression_type::FileCompressionType,
    file_format::{FileFormat, FileFormatFactory},
    file_scan_config::{FileScanConfig, FileScanConfigBuilder},
    file_stream::{FileOpenFuture, FileOpener},
    projection::{ProjectionOpener, SplitProjection},
};
use datafusion_physical_expr::{LexOrdering, PhysicalExpr, projection::ProjectionExprs};
use datafusion_physical_plan::{
    ExecutionPlan,
    filter_pushdown::{FilterPushdownPropagation, PushedDown},
    metrics::ExecutionPlanMetricsSet,
};
use futures::StreamExt;
use object_store::{ObjectMeta, ObjectStore};
use sedona_common::sedona_internal_datafusion_err;
use sedona_raster::builder::RasterBuilder;
use sedona_raster::{io_budget::RasterIoBudget, raster_loader::io_budget_from_config};
use sedona_schema::datatypes::SedonaType;

use crate::{
    gdal_common::{gdal_path_from_object_store_location, with_gdal},
    raster_loader::CancelOnDrop,
    utils::read_outdb_header,
};

/// The fixed schema returned by every GDAL raster file format.
pub fn gdal_raster_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        SedonaType::Raster
            .to_storage_field("raster", false)
            .expect("raster storage field"),
    ]))
}

/// Factory for a GDAL-backed raster extension such as `tif` or `tiff`.
#[derive(Clone, Debug)]
pub struct GdalRasterFileFormatFactory {
    extension: String,
}

impl GdalRasterFileFormatFactory {
    pub fn new(extension: impl Into<String>) -> Self {
        Self {
            extension: extension.into().trim_start_matches('.').to_lowercase(),
        }
    }
}

impl GetExt for GdalRasterFileFormatFactory {
    fn get_ext(&self) -> String {
        self.extension.clone()
    }
}

impl FileFormatFactory for GdalRasterFileFormatFactory {
    fn create(
        &self,
        state: &dyn Session,
        format_options: &HashMap<String, String>,
    ) -> Result<Arc<dyn FileFormat>> {
        let mut tile = false;
        for (key, value) in format_options {
            // The format. prefix is a DataFusion convention that ensures SQL format options make
            // it through this function.
            let key = key.strip_prefix("format.").unwrap_or(key);
            if key == "tile" {
                tile = value.parse::<bool>().map_err(|_| {
                    DataFusionError::Plan(format!(
                        "Invalid value for GDAL raster option 'tile': '{value}' (expected true or false)"
                    ))
                })?;
            } else if !key.contains('.') {
                return plan_err!("Unsupported GDAL raster format option: '{key}'");
            }
        }

        let budget = io_budget_from_config(state.config_options()).unwrap_or_default();
        Ok(Arc::new(
            GdalRasterFileFormat::new(self.extension.clone(), tile).with_budget(budget),
        ))
    }

    fn default(&self) -> Arc<dyn FileFormat> {
        Arc::new(GdalRasterFileFormat::new(self.extension.clone(), false))
    }
}

/// A GDAL-backed raster reader configured for one registered extension.
#[derive(Debug)]
pub struct GdalRasterFileFormat {
    extension: String,
    tile: bool,
    /// The session-wide cap shared with GDAL pixel reads and `RS_FromPath`
    /// header opens. It reaches every `FileOpener`, including those created
    /// for different DataFusion partitions.
    budget: RasterIoBudget,
}

impl GdalRasterFileFormat {
    pub fn new(extension: impl Into<String>, tile: bool) -> Self {
        Self {
            extension: extension.into(),
            tile,
            budget: RasterIoBudget::default(),
        }
    }

    /// Use a shared budget for the blocking metadata reads made by this
    /// format's file openers.
    fn with_budget(mut self, budget: RasterIoBudget) -> Self {
        self.budget = budget;
        self
    }

    pub fn tile(&self) -> bool {
        self.tile
    }
}

#[async_trait]
impl FileFormat for GdalRasterFileFormat {
    fn get_ext(&self) -> String {
        self.extension.clone()
    }

    fn get_ext_with_compression(
        &self,
        _file_compression_type: &FileCompressionType,
    ) -> Result<String> {
        not_impl_err!("compression is not supported for GDAL raster files")
    }

    fn compression_type(&self) -> Option<FileCompressionType> {
        None
    }

    async fn infer_schema(
        &self,
        _state: &dyn Session,
        _store: &Arc<dyn ObjectStore>,
        objects: &[ObjectMeta],
    ) -> Result<SchemaRef> {
        if objects.is_empty() {
            return plan_err!("Can't infer schema for zero objects. Does the input path exist?");
        }
        Ok(gdal_raster_schema())
    }

    async fn infer_stats(
        &self,
        _state: &dyn Session,
        _store: &Arc<dyn ObjectStore>,
        table_schema: SchemaRef,
        _object: &ObjectMeta,
    ) -> Result<Statistics> {
        Ok(Statistics::new_unknown(&table_schema))
    }

    async fn create_physical_plan(
        &self,
        state: &dyn Session,
        conf: FileScanConfig,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // A ListingTable keeps its format and file source after creation. The
        // query may instead be planned by a different session (for example,
        // through a shared catalog), so replace the source's creation-session
        // budget with the planning session's handle. This also makes a later
        // `SET sedona.raster.io_concurrency` effective for existing tables.
        let conf = if let Some(budget) = io_budget_from_config(state.config_options()) {
            let source = conf
                .file_source()
                .downcast_ref::<GdalRasterFileSource>()
                .cloned()
                .ok_or_else(|| sedona_internal_datafusion_err!("Expected GdalRasterFileSource"))?
                .with_budget(budget);
            FileScanConfigBuilder::from(conf)
                .with_source(Arc::new(source))
                .build()
        } else {
            conf
        };
        Ok(DataSourceExec::from_data_source(conf))
    }

    fn file_source(&self, table_schema: TableSchema) -> Arc<dyn FileSource> {
        Arc::new(GdalRasterFileSource::new(
            self.extension.clone(),
            self.tile,
            self.budget.clone(),
            table_schema,
        ))
    }
}

#[derive(Debug, Clone)]
struct GdalRasterFileSource {
    extension: String,
    tile: bool,
    budget: RasterIoBudget,
    table_schema: TableSchema,
    batch_size: Option<usize>,
    projection: ProjectionExprs,
    metrics: ExecutionPlanMetricsSet,
}

impl GdalRasterFileSource {
    fn new(
        extension: String,
        tile: bool,
        budget: RasterIoBudget,
        table_schema: TableSchema,
    ) -> Self {
        let projection = SplitProjection::unprojected(&table_schema).source;
        Self {
            extension,
            tile,
            budget,
            table_schema,
            batch_size: None,
            projection,
            metrics: ExecutionPlanMetricsSet::default(),
        }
    }

    fn with_budget(mut self, budget: RasterIoBudget) -> Self {
        self.budget = budget;
        self
    }
}

impl FileSource for GdalRasterFileSource {
    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        for expr in &self.projection {
            if f(&expr.expr)? == TreeNodeRecursion::Stop {
                return Ok(TreeNodeRecursion::Stop);
            }
        }
        Ok(TreeNodeRecursion::Continue)
    }

    fn create_file_opener(
        &self,
        _store: Arc<dyn ObjectStore>,
        base_config: &FileScanConfig,
        _partition: usize,
    ) -> Result<Arc<dyn FileOpener>> {
        let split_projection =
            SplitProjection::new(self.table_schema.file_schema(), &self.projection);
        let opener: Arc<dyn FileOpener> = Arc::new(GdalRasterFileOpener {
            base_url: base_config.object_store_url.to_string(),
            tile: self.tile,
            budget: self.budget.clone(),
            batch_size: self.batch_size.unwrap_or(usize::MAX),
            file_projection: split_projection.file_indices.clone(),
        });

        ProjectionOpener::try_new(split_projection, opener, self.table_schema.file_schema())
    }

    fn try_pushdown_filters(
        &self,
        filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &ConfigOptions,
    ) -> Result<FilterPushdownPropagation<Arc<dyn FileSource>>> {
        Ok(FilterPushdownPropagation::with_parent_pushdown_result(
            vec![PushedDown::No; filters.len()],
        ))
    }

    fn try_pushdown_projection(
        &self,
        projection: &ProjectionExprs,
    ) -> Result<Option<Arc<dyn FileSource>>> {
        Ok(Some(Arc::new(Self {
            projection: self.projection.try_merge(projection)?,
            ..self.clone()
        })))
    }

    fn projection(&self) -> Option<&ProjectionExprs> {
        Some(&self.projection)
    }

    fn with_batch_size(&self, batch_size: usize) -> Arc<dyn FileSource> {
        Arc::new(Self {
            batch_size: Some(batch_size),
            ..self.clone()
        })
    }

    fn metrics(&self) -> &ExecutionPlanMetricsSet {
        &self.metrics
    }

    fn table_schema(&self) -> &TableSchema {
        &self.table_schema
    }

    fn file_type(&self) -> &str {
        &self.extension
    }

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
struct GdalRasterFileOpener {
    base_url: String,
    tile: bool,
    budget: RasterIoBudget,
    batch_size: usize,
    file_projection: Vec<usize>,
}

impl FileOpener for GdalRasterFileOpener {
    fn open(&self, file: PartitionedFile) -> Result<FileOpenFuture> {
        let path = gdal_path_from_object_store_location(
            &self.base_url,
            file.object_meta.location.as_ref(),
        )?;
        let tile = self.tile;
        let budget = self.budget.clone();
        let batch_size = self.batch_size;
        let file_projection = self.file_projection.clone();

        Ok(Box::pin(async move {
            let cancel = Arc::new(AtomicBool::new(false));
            let _guard = CancelOnDrop(Arc::clone(&cancel));
            // Opening and reading the header is mandatory even though the schema is
            // fixed: this populates the raster's grid, CRS, bands, and nodata. GDAL
            // is synchronous, so keep the open off DataFusion's async worker. The
            // budget is session-wide, rather than per opener or partition, so its
            // limit remains the configured total when DataFusion scans file groups
            // concurrently.
            let batches = tokio::task::spawn_blocking(move || {
                let header = {
                    let _permit = budget.acquire_blocking();
                    // Dropping the future cannot stop a running blocking task.
                    // Check after waiting so cancelled opens never start I/O.
                    if cancel.load(Ordering::Acquire) {
                        return Err(exec_datafusion_err!("GDAL raster header open cancelled"));
                    }
                    with_gdal(|gdal| read_outdb_header(gdal, &path))?
                };
                let capacity = if tile { header.tile_count()? } else { 1 };
                let mut builder = RasterBuilder::new(capacity);
                if tile {
                    header.append_tiles_to(&path, &mut builder)?;
                } else {
                    header.append_to(&path, &mut builder)?;
                }

                let raster: ArrayRef = Arc::new(builder.finish()?);
                let mut batch = RecordBatch::try_new(gdal_raster_schema(), vec![raster])?;
                batch = batch.project(&file_projection)?;

                let batches = if batch_size == 0 || batch.num_rows() <= batch_size {
                    vec![batch]
                } else {
                    (0..batch.num_rows())
                        .step_by(batch_size)
                        .map(|offset| {
                            batch.slice(offset, batch_size.min(batch.num_rows() - offset))
                        })
                        .collect()
                };
                Ok::<_, DataFusionError>(batches)
            })
            .await
            .map_err(|error| exec_datafusion_err!("GDAL raster reader task failed: {error}"))??;

            Ok(futures::stream::iter(batches.into_iter().map(Ok)).boxed())
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::prelude::SessionContext;
    use datafusion::{
        datasource::physical_plan::FileScanConfigBuilder,
        execution::{context::SessionConfig, object_store::ObjectStoreUrl},
    };
    use sedona_raster::raster_loader::{RasterLoaderConfig, RasterLoaderRegistry};
    use std::sync::RwLock;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropping_queued_header_opens_sends_no_http_requests() {
        use axum::{Router, extract::Request, middleware, middleware::Next};
        use std::{
            sync::atomic::{AtomicUsize, Ordering},
            time::Duration,
        };
        use tower_http::services::ServeDir;

        let raster = sedona_testing::data::test_raster("test4.tiff").unwrap();
        let requests = Arc::new(AtomicUsize::new(0));
        let request_count = Arc::clone(&requests);
        let app = Router::new()
            .fallback_service(ServeDir::new(
                std::path::Path::new(&raster).parent().unwrap(),
            ))
            .layer(middleware::from_fn(move |req: Request, next: Next| {
                let request_count = Arc::clone(&request_count);
                async move {
                    request_count.fetch_add(1, Ordering::SeqCst);
                    next.run(req).await
                }
            }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}/", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let budget = RasterIoBudget::new(1);
        let held = budget.try_acquire().unwrap();
        let opener = GdalRasterFileOpener {
            base_url,
            tile: false,
            budget: budget.clone(),
            batch_size: 1024,
            file_projection: vec![0],
        };
        let mut opens =
            Box::pin(futures::future::join_all((0..4).map(|_| {
                opener.open(PartitionedFile::new("test4.tiff", 0)).unwrap()
            })));
        // Wait for every blocking task to actually queue, rather than relying
        // on a delay to guess whether the futures have started.
        tokio::select! {
            _ = &mut opens => panic!("opens should wait on the held budget"),
            result = tokio::time::timeout(Duration::from_secs(5), async {
                while budget.waiting() != 4 {
                    tokio::task::yield_now().await;
                }
            }) => result.expect("all four header opens should queue"),
        }
        assert_eq!(requests.load(Ordering::SeqCst), 0);
        drop(opens);
        drop(held);

        // Wait until every cancelled task has acquired and returned its
        // permit, so the request count cannot pass before the tasks run.
        tokio::time::timeout(Duration::from_secs(5), async {
            while budget.waiting() != 0 || budget.in_use() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("cancelled opens should release the budget");
        assert_eq!(requests.load(Ordering::SeqCst), 0);
        assert_eq!(budget.available(), 1);

        // Live work still opens the file and proves the HTTP counter works.
        let mut stream = opener
            .open(PartitionedFile::new("test4.tiff", 0))
            .unwrap()
            .await
            .unwrap();
        assert_eq!(stream.next().await.unwrap().unwrap().num_rows(), 1);
        assert!(requests.load(Ordering::SeqCst) > 0);
        assert_eq!(budget.available(), 1);
        server.abort();
    }

    #[test]
    fn fixed_schema_is_one_raster_column() {
        let schema = gdal_raster_schema();
        assert_eq!(schema.fields().len(), 1);
        assert_eq!(schema.field(0).name(), "raster");
        assert_eq!(
            SedonaType::from_storage_field(schema.field(0)).unwrap(),
            SedonaType::Raster
        );
    }

    #[test]
    fn factory_parses_tile_option() {
        let context = SessionContext::new();
        let factory = GdalRasterFileFormatFactory::new(".TIF");
        assert_eq!(factory.get_ext(), "tif");

        let format = factory
            .create(
                &context.state(),
                &HashMap::from([("format.tile".to_string(), "true".to_string())]),
            )
            .unwrap();
        let format = format.downcast_ref::<GdalRasterFileFormat>().unwrap();
        assert!(format.tile());
    }

    #[test]
    fn factory_passes_the_session_io_budget_to_every_file_source() {
        let budget = RasterIoBudget::new(3);
        let mut config = ConfigOptions::new();
        config.extensions.insert(
            RasterLoaderConfig::from_handle(Arc::new(RwLock::new(RasterLoaderRegistry::new())))
                .with_io_budget(budget.clone()),
        );
        let context = SessionContext::new_with_config(SessionConfig::from(config));

        let format = GdalRasterFileFormatFactory::new("tif")
            .create(&context.state(), &HashMap::new())
            .unwrap();
        let format = format.downcast_ref::<GdalRasterFileFormat>().unwrap();
        assert!(format.budget.ptr_eq(&budget));

        let source = format.file_source(TableSchema::from(gdal_raster_schema()));
        let source = source.downcast_ref::<GdalRasterFileSource>().unwrap();
        assert!(source.budget.ptr_eq(&budget));
    }

    fn context_with_io_budget(budget: RasterIoBudget) -> SessionContext {
        let mut config = ConfigOptions::new();
        config.extensions.insert(
            RasterLoaderConfig::from_handle(Arc::new(RwLock::new(RasterLoaderRegistry::new())))
                .with_io_budget(budget),
        );
        SessionContext::new_with_config(SessionConfig::from(config))
    }

    #[tokio::test]
    async fn shared_catalog_scan_uses_the_planning_sessions_io_budget() {
        // A ListingTable retains this format and file source after the
        // creating session registers it in a catalog.
        let creating_budget = RasterIoBudget::new(4);
        let creating_context = context_with_io_budget(creating_budget.clone());
        let format = GdalRasterFileFormatFactory::new("tif")
            .create(&creating_context.state(), &HashMap::new())
            .unwrap();
        let source = format.file_source(TableSchema::from(gdal_raster_schema()));
        assert!(
            source
                .downcast_ref::<GdalRasterFileSource>()
                .unwrap()
                .budget
                .ptr_eq(&creating_budget)
        );

        // A second session reads the same catalog entry with its own cap.
        let planning_budget = RasterIoBudget::new(1);
        let planning_context = context_with_io_budget(planning_budget.clone());
        let conf = FileScanConfigBuilder::new(ObjectStoreUrl::local_filesystem(), source).build();
        let plan = format
            .create_physical_plan(&planning_context.state(), conf)
            .await
            .unwrap();

        let exec = plan.downcast_ref::<DataSourceExec>().unwrap();
        let conf = exec.data_source().downcast_ref::<FileScanConfig>().unwrap();
        let source = conf
            .file_source()
            .downcast_ref::<GdalRasterFileSource>()
            .unwrap();
        assert!(source.budget.ptr_eq(&planning_budget));
    }

    #[test]
    fn object_store_paths_escape_remote_url_delimiters() {
        assert_eq!(
            gdal_path_from_object_store_location("http://host/", "a#b?.tif").unwrap(),
            "http://host/a%23b%3F.tif"
        );
        assert_eq!(
            gdal_path_from_object_store_location("s3://bucket", "dir/a b%.tif").unwrap(),
            "s3://bucket/dir/a%20b%25.tif"
        );
    }

    #[test]
    fn object_store_paths_preserve_local_filesystem_handling() {
        assert_eq!(
            gdal_path_from_object_store_location("file://", "/tmp/a#b.tif").unwrap(),
            "file:///tmp/a#b.tif"
        );
    }
}
