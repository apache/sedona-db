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

use arrow_array::{ArrayRef, RecordBatch, RecordBatchIterator, RecordBatchReader};
use arrow_schema::{Schema, SchemaRef};
use async_trait::async_trait;
use datafusion_catalog::Session;
use datafusion_common::{
    DataFusionError, GetExt, Result, Statistics, exec_datafusion_err, plan_err,
};
use datafusion_datasource::{
    TableSchema,
    file::FileSource,
    file_compression_type::FileCompressionType,
    file_format::{FileFormat, FileFormatFactory},
    file_scan_config::FileScanConfig,
};
use datafusion_physical_plan::ExecutionPlan;
use object_store::{ObjectMeta, ObjectStore};
use sedona_datasource::{
    format::ExternalFileFormat,
    spec::{ExternalFormatSpec, Object, OpenReaderArgs},
};
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
    inner: ExternalFileFormat,
}

impl GdalRasterFileFormat {
    pub fn new(extension: impl Into<String>, tile: bool) -> Self {
        let extension = extension.into();
        let spec = Arc::new(GdalRasterFormatSpec {
            extension: extension.clone(),
            tile,
        });
        Self {
            extension,
            tile,
            inner: ExternalFileFormat::new(spec),
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
        file_compression_type: &FileCompressionType,
    ) -> Result<String> {
        self.inner.get_ext_with_compression(file_compression_type)
    }

    fn compression_type(&self) -> Option<FileCompressionType> {
        self.inner.compression_type()
    }

    async fn infer_schema(
        &self,
        state: &dyn Session,
        store: &Arc<dyn ObjectStore>,
        objects: &[ObjectMeta],
    ) -> Result<SchemaRef> {
        self.inner.infer_schema(state, store, objects).await
    }

    async fn infer_stats(
        &self,
        state: &dyn Session,
        store: &Arc<dyn ObjectStore>,
        table_schema: SchemaRef,
        object: &ObjectMeta,
    ) -> Result<Statistics> {
        self.inner
            .infer_stats(state, store, table_schema, object)
            .await
    }

    async fn create_physical_plan(
        &self,
        state: &dyn Session,
        conf: FileScanConfig,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.inner.create_physical_plan(state, conf).await
    }

    fn file_source(&self, table_schema: TableSchema) -> Arc<dyn FileSource> {
        self.inner.file_source(table_schema)
    }
}

#[derive(Debug, Clone)]
struct GdalRasterFormatSpec {
    extension: String,
    tile: bool,
}

#[async_trait]
impl ExternalFormatSpec for GdalRasterFormatSpec {
    async fn infer_schema(&self, _location: &Object) -> Result<Schema> {
        Ok((*gdal_raster_schema()).clone())
    }

    async fn open_reader(
        &self,
        args: &OpenReaderArgs,
    ) -> Result<Box<dyn RecordBatchReader + Send>> {
        let path = args.src.to_url_string().ok_or_else(|| {
            exec_datafusion_err!(
                "Cannot determine location for GDAL raster source {}",
                args.src
            )
        })?;

        // Opening and reading the header is mandatory even though the schema is
        // fixed: this populates the raster's grid, CRS, bands, and nodata. GDAL
        // is synchronous, so keep the open off DataFusion's async worker.
        let tile = self.tile;
        let raster: ArrayRef = tokio::task::spawn_blocking(move || {
            let header = with_gdal(|gdal| read_outdb_header(gdal, &path))?;
            let capacity = if tile { header.tile_count()? } else { 1 };
            let mut builder = RasterBuilder::new(capacity);
            if tile {
                header.append_tiles_to(&path, &mut builder)?;
            } else {
                header.append_to(&path, &mut builder)?;
            }
            Ok::<ArrayRef, DataFusionError>(Arc::new(builder.finish()?))
        })
        .await
        .map_err(|error| exec_datafusion_err!("GDAL raster reader task failed: {error}"))??;
        let schema = gdal_raster_schema();
        let mut batch = RecordBatch::try_new(schema.clone(), vec![raster])?;
        if let Some(projection) = &args.file_projection {
            batch = batch.project(projection)?;
        }

        let batch_schema = batch.schema();
        let batch_size = args.batch_size.unwrap_or(usize::MAX);
        let batches = if batch_size == 0 || batch.num_rows() <= batch_size {
            vec![batch]
        } else {
            (0..batch.num_rows())
                .step_by(batch_size)
                .map(|offset| batch.slice(offset, batch_size.min(batch.num_rows() - offset)))
                .collect()
        };

        Ok(Box::new(RecordBatchIterator::new(
            batches.into_iter().map(Ok),
            batch_schema,
        )))
    }

    fn with_options(
        &self,
        options: &HashMap<String, String>,
    ) -> Result<Arc<dyn ExternalFormatSpec>> {
        let mut tile = self.tile;
        for (key, value) in options {
            if key == "tile" {
                tile = value.parse::<bool>().map_err(|_| {
                    DataFusionError::Plan(format!(
                        "Invalid value for GDAL raster option 'tile': '{value}' (expected true or false)"
                    ))
                })?;
            } else {
                return plan_err!("Unsupported GDAL raster format option: '{key}'");
            }
        }
        Ok(Arc::new(Self {
            extension: self.extension.clone(),
            tile,
        }))
    }

    fn extension(&self) -> &str {
        &self.extension
    }

    async fn infer_stats(&self, _location: &Object, schema: &Schema) -> Result<Statistics> {
        Ok(Statistics::new_unknown(schema))
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
