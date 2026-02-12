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

use std::{any::Any, collections::HashMap, fmt, sync::Arc};

use arrow_schema::{Schema, SchemaRef};
use datafusion_catalog::{memory::DataSourceExec, Session};
use datafusion_common::{
    config::ExtensionOptions, error::DataFusionError, parsers::CompressionTypeVariant, GetExt,
    Statistics,
};
use datafusion_datasource::{
    file::FileSource,
    file_compression_type::FileCompressionType,
    file_format::{FileFormat, FileFormatFactory},
    file_scan_config::{FileScanConfig, FileScanConfigBuilder},
};
use datafusion_physical_plan::ExecutionPlan;
use futures::{StreamExt, TryStreamExt};
use object_store::{ObjectMeta, ObjectStore};

use crate::{
    laz::{metadata::LazMetadataReader, reader::LazFileReaderFactory, source::LazSource},
    options::PointcloudOptions,
};

const DEFAULT_LAZ_EXTENSION: &str = ".laz";

/// Factory struct used to create [LazFormat]
#[derive(Default)]
pub struct LazFormatFactory {
    // inner options for LAZ
    pub options: Option<PointcloudOptions>,
}

impl LazFormatFactory {
    /// Creates an instance of [LazFormatFactory]
    pub fn new() -> Self {
        Self { options: None }
    }

    /// Creates an instance of [LazFormatFactory] with customized default options
    pub fn new_with(options: PointcloudOptions) -> Self {
        Self {
            options: Some(options),
        }
    }
}

impl FileFormatFactory for LazFormatFactory {
    fn create(
        &self,
        state: &dyn Session,
        format_options: &HashMap<String, String>,
    ) -> Result<Arc<dyn FileFormat>, DataFusionError> {
        let mut options = state
            .config_options()
            .extensions
            .get::<PointcloudOptions>()
            .or_else(|| state.table_options().extensions.get::<PointcloudOptions>())
            .cloned()
            .or(self.options.clone())
            .unwrap_or_default();

        for (k, v) in format_options {
            options.set(k, v)?;
        }

        Ok(Arc::new(LazFormat::default().with_options(options)))
    }

    fn default(&self) -> Arc<dyn FileFormat> {
        Arc::new(LazFormat::default())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl GetExt for LazFormatFactory {
    fn get_ext(&self) -> String {
        // Removes the dot, i.e. ".laz" -> "laz"
        DEFAULT_LAZ_EXTENSION[1..].to_string()
    }
}

impl fmt::Debug for LazFormatFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LazFormatFactory")
            .field("LazFormatFactory", &self.options)
            .finish()
    }
}

/// The LAZ `FileFormat` implementation
#[derive(Debug, Default)]
pub struct LazFormat {
    pub options: PointcloudOptions,
}

impl LazFormat {
    pub fn with_options(mut self, options: PointcloudOptions) -> Self {
        self.options = options;
        self
    }
}

#[async_trait::async_trait]
impl FileFormat for LazFormat {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_ext(&self) -> String {
        LazFormatFactory::new().get_ext()
    }

    fn get_ext_with_compression(
        &self,
        file_compression_type: &FileCompressionType,
    ) -> Result<String, DataFusionError> {
        let ext = self.get_ext();
        match file_compression_type.get_variant() {
            CompressionTypeVariant::UNCOMPRESSED => Ok(ext),
            _ => Err(DataFusionError::External(
                "Laz FileFormat does not support compression.".into(),
            )),
        }
    }

    fn compression_type(&self) -> Option<FileCompressionType> {
        Some(FileCompressionType::UNCOMPRESSED)
    }

    async fn infer_schema(
        &self,
        state: &dyn Session,
        store: &Arc<dyn ObjectStore>,
        objects: &[ObjectMeta],
    ) -> Result<SchemaRef, DataFusionError> {
        let file_metadata_cache = state.runtime_env().cache_manager.get_file_metadata_cache();

        let mut schemas: Vec<_> = futures::stream::iter(objects)
            .map(|object_meta| async {
                let loc_path = object_meta.location.clone();

                let schema = LazMetadataReader::new(store, object_meta)
                    .with_file_metadata_cache(Some(Arc::clone(&file_metadata_cache)))
                    .with_options(self.options.clone())
                    .fetch_schema()
                    .await?;

                Ok::<_, DataFusionError>((loc_path, schema))
            })
            .boxed() // Workaround https://github.com/rust-lang/rust/issues/64552
            // fetch schemas concurrently, if requested
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
        state: &dyn Session,
        store: &Arc<dyn ObjectStore>,
        table_schema: SchemaRef,
        object: &ObjectMeta,
    ) -> Result<Statistics, DataFusionError> {
        let file_metadata_cache = state.runtime_env().cache_manager.get_file_metadata_cache();
        LazMetadataReader::new(store, object)
            .with_options(self.options.clone())
            .with_file_metadata_cache(Some(Arc::clone(&file_metadata_cache)))
            .fetch_statistics(&table_schema)
            .await
    }

    async fn create_physical_plan(
        &self,
        state: &dyn Session,
        conf: FileScanConfig,
    ) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        let mut source = conf
            .file_source()
            .as_any()
            .downcast_ref::<LazSource>()
            .cloned()
            .ok_or_else(|| DataFusionError::External("Expected LazSource".into()))?;
        source = source.with_options(self.options.clone());

        let metadata_cache = state.runtime_env().cache_manager.get_file_metadata_cache();
        let store = state
            .runtime_env()
            .object_store(conf.object_store_url.clone())?;
        let laz_reader_factory = Arc::new(LazFileReaderFactory::new(store, Some(metadata_cache)));
        let source = source.with_reader_factory(laz_reader_factory);

        let conf = FileScanConfigBuilder::from(conf)
            .with_source(Arc::new(source))
            .build();

        Ok(DataSourceExec::from_data_source(conf))
    }

    fn file_source(&self) -> Arc<dyn FileSource> {
        Arc::new(LazSource::default().with_options(self.options.clone()))
    }
}

#[cfg(test)]
mod test {
    use std::{collections::HashMap, fs::File, sync::Arc};

    use datafusion::{execution::SessionStateBuilder, prelude::SessionContext};
    use datafusion_datasource::file_format::FileFormatFactory;
    use las::{point::Format, Builder, Writer};

    use crate::laz::format::{LazFormat, LazFormatFactory};

    fn setup_context() -> SessionContext {
        let file_format = Arc::new(LazFormatFactory::new());

        let mut state = SessionStateBuilder::new().build();
        state.register_file_format(file_format, true).unwrap();

        SessionContext::new_with_state(state).enable_url_table()
    }

    #[tokio::test]
    async fn laz_format_factory() {
        let ctx = SessionContext::new();
        let format_factory = Arc::new(LazFormatFactory::new());
        let dyn_format = format_factory
            .create(&ctx.state(), &HashMap::new())
            .unwrap();
        assert!(dyn_format.as_any().downcast_ref::<LazFormat>().is_some());
    }

    #[tokio::test]
    async fn projection() {
        let ctx = setup_context();
        let df = ctx
            .sql("SELECT x, y, z FROM 'tests/data/extra.laz'")
            .await
            .unwrap();

        assert_eq!(df.schema().fields().len(), 3);
    }

    #[tokio::test]
    async fn multiple_files() {
        let tmpdir = tempfile::tempdir().unwrap();

        for i in 0..4 {
            let tmp_path = tmpdir.path().join(format!("tmp{i}.laz"));
            let tmp_file = File::create(&tmp_path).unwrap();

            // create laz file with one point
            let mut builder = Builder::from((1, 4));
            builder.point_format = Format::new(0).unwrap();
            builder.point_format.is_compressed = true;
            let header = builder.into_header().unwrap();
            let mut writer = Writer::new(tmp_file, header).unwrap();
            writer.write_point(Default::default()).unwrap();
            writer.close().unwrap();
        }

        let ctx = setup_context();
        let table = tmpdir.path().to_str().unwrap();
        let df = ctx.sql(&format!("SELECT * FROM '{table}'",)).await.unwrap();

        assert_eq!(df.count().await.unwrap(), 4);
    }

    #[tokio::test]
    async fn file_that_does_not_exist() {
        let ctx = setup_context();
        let err = ctx
            .sql("SELECT * FROM 'nonexisting.laz'")
            .await
            .unwrap_err();
        assert_eq!(
            err.message(),
            "Error during planning: table 'datafusion.public.nonexisting.laz' not found"
        );
    }
}
