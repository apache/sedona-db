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

use std::{collections::HashMap, fmt, sync::Arc};

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
    file_scan_config::FileScanConfig,
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
use sedona_raster::builder::RasterBuilder;
use sedona_schema::datatypes::SedonaType;

use crate::{gdal_common::with_gdal, utils::read_outdb_header};

/// The fixed schema returned by every GDAL raster file format.
pub fn gdal_raster_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        SedonaType::Raster
            .to_storage_field("raster", false)
            .expect("raster storage field"),
    ]))
}

/// Factory for a GDAL-backed raster extension such as `tif` or `tiff`.
#[derive(Clone)]
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

impl fmt::Debug for GdalRasterFileFormatFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GdalRasterFileFormatFactory")
            .field("extension", &self.extension)
            .finish()
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
        _state: &dyn Session,
        format_options: &HashMap<String, String>,
    ) -> Result<Arc<dyn FileFormat>> {
        let mut tile = false;
        for (key, value) in format_options {
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

        Ok(Arc::new(GdalRasterFileFormat::new(
            self.extension.clone(),
            tile,
        )))
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
}

impl GdalRasterFileFormat {
    pub fn new(extension: impl Into<String>, tile: bool) -> Self {
        Self {
            extension: extension.into(),
            tile,
        }
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
        _state: &dyn Session,
        conf: FileScanConfig,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(DataSourceExec::from_data_source(conf))
    }

    fn file_source(&self, table_schema: TableSchema) -> Arc<dyn FileSource> {
        Arc::new(GdalRasterFileSource::new(
            self.extension.clone(),
            self.tile,
            table_schema,
        ))
    }
}

#[derive(Debug, Clone)]
struct GdalRasterFileSource {
    extension: String,
    tile: bool,
    table_schema: TableSchema,
    batch_size: Option<usize>,
    projection: ProjectionExprs,
    metrics: ExecutionPlanMetricsSet,
}

impl GdalRasterFileSource {
    fn new(extension: String, tile: bool, table_schema: TableSchema) -> Self {
        let projection = SplitProjection::unprojected(&table_schema).source;
        Self {
            extension,
            tile,
            table_schema,
            batch_size: None,
            projection,
            metrics: ExecutionPlanMetricsSet::default(),
        }
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
    batch_size: usize,
    file_projection: Vec<usize>,
}

impl FileOpener for GdalRasterFileOpener {
    fn open(&self, file: PartitionedFile) -> Result<FileOpenFuture> {
        let path = format!("{}{}", self.base_url, file.object_meta.location);
        let tile = self.tile;
        let batch_size = self.batch_size;
        let file_projection = self.file_projection.clone();

        Ok(Box::pin(async move {
            // Opening and reading the header is mandatory even though the schema is
            // fixed: this populates the raster's grid, CRS, bands, and nodata. GDAL
            // is synchronous, so keep the open off DataFusion's async worker.
            let batches = tokio::task::spawn_blocking(move || {
                let header = with_gdal(|gdal| read_outdb_header(gdal, &path))?;
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
}
