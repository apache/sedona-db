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

use arrow_schema::{Schema, SchemaRef};
use async_trait::async_trait;
use datafusion_catalog::{Session, memory::DataSourceExec};
use datafusion_common::{GetExt, Result, Statistics, exec_err, parsers::CompressionTypeVariant};
use datafusion_datasource::{
    TableSchema,
    file::FileSource,
    file_compression_type::FileCompressionType,
    file_format::{FileFormat, FileFormatFactory},
    file_scan_config::{FileScanConfig, FileScanConfigBuilder},
};
use datafusion_physical_plan::ExecutionPlan;
use futures::{StreamExt, TryStreamExt};
use object_store::{ObjectMeta, ObjectStore};
use std::{collections::HashMap, sync::Arc};

/// Native DataFusion FlatGeobuf format using generic object-store range I/O.
#[derive(Debug, Clone)]
pub struct FlatGeobufFormat {
    pub(crate) geometry_column_name: String,
    pub(crate) metadata_size_hint: usize,
}
impl Default for FlatGeobufFormat {
    fn default() -> Self {
        Self {
            geometry_column_name: "wkb_geometry".into(),
            metadata_size_hint: 65536,
        }
    }
}
impl FlatGeobufFormat {
    pub fn with_options(mut self, options: &HashMap<String, String>) -> Result<Self> {
        for (key, value) in options {
            match key.strip_prefix("format.").unwrap_or(key) {
                "geometry_column_name" if !value.is_empty() => {
                    self.geometry_column_name = value.clone()
                }
                "metadata_size_hint" => {
                    self.metadata_size_hint = value.parse().map_err(|_| {
                        datafusion_common::exec_datafusion_err!("Invalid metadata_size_hint")
                    })?
                }
                _ => return exec_err!("Unknown or invalid FlatGeobuf option: {key}"),
            }
        }
        Ok(self)
    }
}
#[derive(Debug, Default)]
pub struct FlatGeobufFormatFactory;
impl GetExt for FlatGeobufFormatFactory {
    fn get_ext(&self) -> String {
        "fgb".into()
    }
}
impl FileFormatFactory for FlatGeobufFormatFactory {
    fn create(
        &self,
        _state: &dyn Session,
        options: &HashMap<String, String>,
    ) -> Result<Arc<dyn FileFormat>> {
        Ok(Arc::new(FlatGeobufFormat::default().with_options(options)?))
    }

    fn default(&self) -> Arc<dyn FileFormat> {
        Arc::new(FlatGeobufFormat::default())
    }
}
#[async_trait]
impl FileFormat for FlatGeobufFormat {
    fn get_ext(&self) -> String {
        "fgb".into()
    }

    fn get_ext_with_compression(&self, compression: &FileCompressionType) -> Result<String> {
        match compression.get_variant() {
            CompressionTypeVariant::UNCOMPRESSED => Ok("fgb".into()),
            _ => exec_err!("FlatGeobuf does not support outer compression"),
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
    ) -> Result<SchemaRef> {
        let cache = state.runtime_env().cache_manager.get_file_metadata_cache();
        let mut objects = objects.to_vec();
        objects.sort_by(|a, b| a.location.cmp(&b.location));
        let schemas: Vec<_> = futures::stream::iter(&objects)
            .map(|object| async {
                crate::store_metadata::metadata(
                    store.clone(),
                    object,
                    self.metadata_size_hint,
                    Some(&cache),
                )
                .await?
                .infer_schema(&self.geometry_column_name)
                .map(|s| s.as_ref().clone())
            })
            .boxed()
            .buffered(
                state
                    .config_options()
                    .execution
                    .meta_fetch_concurrency
                    .max(1),
            )
            .try_collect()
            .await?;
        Ok(Arc::new(Schema::try_merge(schemas)?))
    }

    async fn infer_stats(
        &self,
        _state: &dyn Session,
        _store: &Arc<dyn ObjectStore>,
        table_schema: SchemaRef,
        _object: &ObjectMeta,
    ) -> Result<Statistics> {
        Ok(Statistics::new_unknown(table_schema.as_ref()))
    }

    async fn create_physical_plan(
        &self,
        state: &dyn Session,
        conf: FileScanConfig,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let store = state
            .runtime_env()
            .object_store(conf.object_store_url.clone())?;
        let cache = state.runtime_env().cache_manager.get_file_metadata_cache();
        let mut source = conf
            .file_source()
            .downcast_ref::<crate::source::FlatGeobufSource>()
            .cloned()
            .ok_or_else(|| datafusion_common::exec_datafusion_err!("Expected FlatGeobufSource"))?;
        source.cache = Some(cache.clone());
        let mut planned = HashMap::new();
        for group in &conf.file_groups {
            for file in group.iter() {
                let data = crate::store_metadata::metadata(
                    store.clone(),
                    &file.object_meta,
                    self.metadata_size_hint,
                    Some(&cache),
                )
                .await?;
                planned.insert(file.object_meta.location.clone(), data);
            }
        }
        source.planned = Arc::new(planned);
        let conf = FileScanConfigBuilder::from(conf)
            .with_source(Arc::new(source))
            .build();
        Ok(DataSourceExec::from_data_source(conf))
    }

    fn file_source(&self, table_schema: TableSchema) -> Arc<dyn FileSource> {
        Arc::new(crate::source::FlatGeobufSource::new(
            table_schema,
            self.clone(),
        ))
    }
}

/// Opt-in native reader alias for sessions whose `fgb` default uses GDAL.
#[derive(Debug, Default)]
pub struct NativeFlatGeobufFormatFactory;
impl GetExt for NativeFlatGeobufFormatFactory {
    fn get_ext(&self) -> String {
        "fgb_native".into()
    }
}
impl FileFormatFactory for NativeFlatGeobufFormatFactory {
    fn create(
        &self,
        state: &dyn Session,
        options: &HashMap<String, String>,
    ) -> Result<Arc<dyn FileFormat>> {
        FlatGeobufFormatFactory.create(state, options)
    }
    fn default(&self) -> Arc<dyn FileFormat> {
        FlatGeobufFormatFactory.default()
    }
}
