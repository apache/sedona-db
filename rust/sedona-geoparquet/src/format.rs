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

use std::{any::Any, collections::HashMap, sync::Arc};

use arrow_schema::{Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::{
    config::ConfigOptions,
    datasource::{
        file_format::{
            file_compression_type::FileCompressionType,
            parquet::{fetch_parquet_metadata, ParquetFormat, ParquetFormatFactory},
            FileFormat, FileFormatFactory,
        },
        physical_plan::{
            FileOpener, FileScanConfig, FileScanConfigBuilder, FileSinkConfig, FileSource,
        },
    },
};
use datafusion_catalog::{memory::DataSourceExec, Session};
use datafusion_common::{plan_err, GetExt, Result, Statistics};
use datafusion_physical_expr::{LexRequirement, PhysicalExpr};
use datafusion_physical_plan::{
    filter_pushdown::FilterPushdownPropagation, metrics::ExecutionPlanMetricsSet, ExecutionPlan,
};
use futures::{StreamExt, TryStreamExt};
use object_store::{ObjectMeta, ObjectStore};

use sedona_common::sedona_internal_err;

use sedona_schema::extension_type::ExtensionType;

use crate::{
    file_opener::{storage_schema_contains_geo, GeoParquetFileOpener},
    metadata::{GeoParquetColumnEncoding, GeoParquetMetadata},
    options::{GeoParquetVersion, TableGeoParquetOptions},
    writer::create_geoparquet_writer_physical_plan,
};
use datafusion::datasource::physical_plan::ParquetSource;
use datafusion::datasource::schema_adapter::SchemaAdapterFactory;

/// GeoParquet FormatFactory
///
/// A DataFusion FormatFactory provides a means to allow creating a table
/// or referencing one from a SQL context like COPY TO.
#[derive(Debug, Default)]
pub struct GeoParquetFormatFactory {
    inner: ParquetFormatFactory,
    options: Option<TableGeoParquetOptions>,
}

impl GeoParquetFormatFactory {
    /// Creates an instance of [GeoParquetFormatFactory]
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            inner: ParquetFormatFactory::new(),
            options: None,
        }
    }

    /// Creates an instance of [GeoParquetFormatFactory] with customized default options
    pub fn new_with_options(options: TableGeoParquetOptions) -> Self {
        Self {
            inner: ParquetFormatFactory::new_with_options(options.inner.clone()),
            options: Some(options),
        }
    }
}

impl FileFormatFactory for GeoParquetFormatFactory {
    fn create(
        &self,
        state: &dyn Session,
        format_options: &HashMap<String, String>,
    ) -> Result<Arc<dyn FileFormat>> {
        let mut options_mut = self.options.clone().unwrap_or_default();
        let mut format_options_mut = format_options.clone();
        options_mut.geoparquet_version =
            if let Some(version_string) = format_options_mut.remove("geoparquet_version") {
                match version_string.as_str() {
                    "1.0" => GeoParquetVersion::V1_0,
                    "1.1" => GeoParquetVersion::V1_1,
                    "2.0" => GeoParquetVersion::V2_0,
                    _ => GeoParquetVersion::default(),
                }
            } else {
                GeoParquetVersion::default()
            };

        let inner_format = self.inner.create(state, &format_options_mut)?;
        if let Some(parquet_format) = inner_format.as_any().downcast_ref::<ParquetFormat>() {
            options_mut.inner = parquet_format.options().clone();
            Ok(Arc::new(GeoParquetFormat::new(options_mut)))
        } else {
            sedona_internal_err!(
                "Unexpected format from ParquetFormatFactory: {:?}",
                inner_format
            )
        }
    }

    fn default(&self) -> std::sync::Arc<dyn FileFormat> {
        Arc::new(ParquetFormat::default())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl GetExt for GeoParquetFormatFactory {
    fn get_ext(&self) -> String {
        self.inner.get_ext()
    }
}

/// GeoParquet FileFormat
///
/// This [FileFormat] wraps the [ParquetFormat]. The primary purpose of the
/// FileFormat is to be able to be used in a ListingTable (i.e., multi file table).
/// Here we also use it to implement a basic `TableProvider` that give us most if
/// not all of the features of the underlying Parquet reader.
#[derive(Debug, Default)]
pub struct GeoParquetFormat {
    options: TableGeoParquetOptions,
}

impl GeoParquetFormat {
    /// Create a new instance of the file format
    pub fn new(options: TableGeoParquetOptions) -> Self {
        Self { options }
    }

    fn inner(&self) -> ParquetFormat {
        ParquetFormat::new().with_options(self.options.inner.clone())
    }
}

#[async_trait]
impl FileFormat for GeoParquetFormat {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_ext(&self) -> String {
        ParquetFormatFactory::new().get_ext()
    }

    fn get_ext_with_compression(
        &self,
        file_compression_type: &FileCompressionType,
    ) -> Result<String> {
        self.inner().get_ext_with_compression(file_compression_type)
    }

    fn compression_type(&self) -> Option<FileCompressionType> {
        self.inner().compression_type()
    }

    async fn infer_schema(
        &self,
        state: &dyn Session,
        store: &Arc<dyn ObjectStore>,
        objects: &[ObjectMeta],
    ) -> Result<SchemaRef> {
        // First, try the underlying format without schema metadata. This should work
        // for regular Parquet reads and will at least ensure that the underlying schemas
        // are compatible.
        let inner_schema_without_metadata =
            self.inner().infer_schema(state, store, objects).await?;

        // Collect metadata separately. We can in theory do our own schema
        // inference too to save an extra server request, but then we have to
        // copy more ParquetFormat code. It may be that caching at the object
        // store level is the way to go here.
        let metadatas: Vec<_> = futures::stream::iter(objects)
            .map(|object| {
                fetch_parquet_metadata(
                    store.as_ref(),
                    object,
                    self.inner().metadata_size_hint(),
                    None,
                )
            })
            .boxed() // Workaround https://github.com/rust-lang/rust/issues/64552
            .buffered(state.config_options().execution.meta_fetch_concurrency)
            .try_collect()
            .await?;

        let mut geoparquet_metadata: Option<GeoParquetMetadata> = None;
        for metadata in &metadatas {
            if let Some(kv) = metadata.file_metadata().key_value_metadata() {
                for item in kv {
                    if item.key == "geo" && item.value.is_some() {
                        let this_geoparquet_metadata =
                            GeoParquetMetadata::try_new(item.value.as_ref().unwrap())?;

                        match geoparquet_metadata.as_mut() {
                            Some(existing) => {
                                existing.try_update(&this_geoparquet_metadata)?;
                            }
                            None => geoparquet_metadata = Some(this_geoparquet_metadata),
                        }
                    }
                }
            }
        }

        if let Some(geo_metadata) = geoparquet_metadata {
            let new_fields: Result<Vec<_>> = inner_schema_without_metadata
                .fields()
                .iter()
                .map(|field| {
                    if let Some(geo_column) = geo_metadata.columns.get(field.name()) {
                        match geo_column.encoding {
                            GeoParquetColumnEncoding::WKB => {
                                let extension = ExtensionType::new(
                                    "geoarrow.wkb",
                                    field.data_type().clone(),
                                    Some(geo_column.to_geoarrow_metadata()?),
                                );
                                Ok(Arc::new(
                                    extension.to_field(field.name(), field.is_nullable()),
                                ))
                            }
                            _ => plan_err!(
                                "Unsupported GeoParquet encoding: {}",
                                geo_column.encoding
                            ),
                        }
                    } else {
                        Ok(field.clone())
                    }
                })
                .collect();

            Ok(Arc::new(Schema::new(new_fields?)))
        } else {
            Ok(inner_schema_without_metadata)
        }
    }

    async fn infer_stats(
        &self,
        state: &dyn Session,
        store: &Arc<dyn ObjectStore>,
        table_schema: SchemaRef,
        object: &ObjectMeta,
    ) -> Result<Statistics> {
        // We don't do anything special here to insert GeoStatistics because pruning
        // happens elsewhere. These might be useful for a future optimizer or analyzer
        // pass that can insert optimizations based on geometry type.
        self.inner()
            .infer_stats(state, store, table_schema, object)
            .await
    }

    async fn create_physical_plan(
        &self,
        _state: &dyn Session,
        config: FileScanConfig,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // A copy of ParquetSource::create_physical_plan() that ensures the underlying
        // DataSourceExec is backed by a GeoParquetFileSource instead of a ParquetFileSource
        let mut metadata_size_hint = None;

        if let Some(metadata) = self.inner().metadata_size_hint() {
            metadata_size_hint = Some(metadata);
        }

        let mut source = GeoParquetFileSource::new(self.options.clone());

        if let Some(metadata_size_hint) = metadata_size_hint {
            source = source.with_metadata_size_hint(metadata_size_hint)
        }

        let conf = FileScanConfigBuilder::from(config)
            .with_source(Arc::new(source))
            .build();

        // Build the inner plan
        let inner_plan = DataSourceExec::from_data_source(conf);
        Ok(inner_plan)
    }

    async fn create_writer_physical_plan(
        &self,
        input: Arc<dyn ExecutionPlan>,
        _state: &dyn Session,
        conf: FileSinkConfig,
        order_requirements: Option<LexRequirement>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        create_geoparquet_writer_physical_plan(input, conf, order_requirements, &self.options)
    }

    fn file_source(&self) -> Arc<dyn FileSource> {
        Arc::new(
            GeoParquetFileSource::try_from_file_source(self.inner().file_source(), None, None)
                .unwrap(),
        )
    }
}

/// Geo aware wrapper around a [ParquetSource]
///
/// The primary reason for this is to (1) ensure that the schema we pass
/// to the Parquet file opener is the raw/unwrapped schema and (2) provide a
/// custom [FileOpener] that implements pruning based on a predicate and
/// column statistics. We have to keep a copy of `metadata_size_hint` and
/// `predicate` because the [ParquetSource] marks these fields as private
/// and we need them for our custom file opener.
#[derive(Debug, Clone)]
pub struct GeoParquetFileSource {
    inner: ParquetSource,
    metadata_size_hint: Option<usize>,
    predicate: Option<Arc<dyn PhysicalExpr>>,
}

impl GeoParquetFileSource {
    /// Create a new file source based on [TableParquetOptions]
    pub fn new(options: TableGeoParquetOptions) -> Self {
        Self {
            inner: ParquetSource::new(options.inner.clone()),
            metadata_size_hint: None,
            predicate: None,
        }
    }

    /// Create a new file source based on an arbitrary file source
    ///
    /// Panics if the provided [FileSource] is not a [ParquetSource].
    /// This is needed because some functions from which this needs to be
    /// called do not return errors.
    pub fn from_file_source(
        inner: Arc<dyn FileSource>,
        metadata_size_hint: Option<usize>,
        predicate: Option<Arc<dyn PhysicalExpr>>,
    ) -> Self {
        Self::try_from_file_source(inner, metadata_size_hint, predicate).unwrap()
    }

    /// Create a new file source based on an arbitrary file source
    ///
    /// Returns an error if the provided [FileSource] is not a [ParquetSource].
    pub fn try_from_file_source(
        inner: Arc<dyn FileSource>,
        metadata_size_hint: Option<usize>,
        predicate: Option<Arc<dyn PhysicalExpr>>,
    ) -> Result<Self> {
        if let Some(parquet_source) = inner.as_any().downcast_ref::<ParquetSource>() {
            let mut parquet_source = parquet_source.clone();
            // Extract the predicate from the existing source if it exists so we can keep a copy of it
            let new_predicate = match (parquet_source.predicate().cloned(), predicate) {
                (None, None) => None,
                (None, Some(specified_predicate)) => Some(specified_predicate),
                (Some(inner_predicate), None) => Some(inner_predicate),
                (Some(_), Some(specified_predicate)) => {
                    parquet_source = parquet_source.with_predicate(specified_predicate.clone());
                    Some(specified_predicate)
                }
            };

            Ok(Self {
                inner: parquet_source.clone(),
                metadata_size_hint,
                predicate: new_predicate,
            })
        } else {
            sedona_internal_err!("GeoParquetFileSource constructed from non-ParquetSource")
        }
    }

    /// Apply a predicate to the [FileSource]
    pub fn with_predicate(&self, predicate: Arc<dyn PhysicalExpr>) -> Self {
        Self {
            inner: self.inner.with_predicate(predicate.clone()),
            metadata_size_hint: self.metadata_size_hint,
            predicate: Some(predicate),
        }
    }

    /// Apply a [SchemaAdapterFactory] to the inner [ParquetSource]
    pub fn with_schema_adapter_factory(
        &self,
        schema_adapter_factory: Arc<dyn SchemaAdapterFactory>,
    ) -> Self {
        let inner = self
            .inner
            .clone()
            .with_schema_adapter_factory(schema_adapter_factory)
            .expect("with_schema_adapter_factory failed");

        let parquet_source = inner
            .as_any()
            .downcast_ref::<ParquetSource>()
            .expect("GeoParquetFileSource constructed with non ParquetSource inner")
            .clone();

        Self {
            inner: parquet_source,
            metadata_size_hint: self.metadata_size_hint,
            predicate: self.predicate.clone(),
        }
    }

    /// Apply a metadata size hint to the inner [ParquetSource]
    pub fn with_metadata_size_hint(&self, hint: usize) -> Self {
        Self {
            inner: self.inner.clone().with_metadata_size_hint(hint),
            metadata_size_hint: Some(hint),
            predicate: self.predicate.clone(),
        }
    }
}

impl FileSource for GeoParquetFileSource {
    fn create_file_opener(
        &self,
        object_store: Arc<dyn ObjectStore>,
        base_config: &FileScanConfig,
        partition: usize,
    ) -> Arc<dyn FileOpener> {
        let inner_opener =
            self.inner
                .create_file_opener(object_store.clone(), base_config, partition);

        // If there are no geo columns or no pruning predicate, just return the inner opener
        if self.predicate.is_none() || !storage_schema_contains_geo(&base_config.file_schema) {
            return inner_opener;
        }

        Arc::new(GeoParquetFileOpener::new(
            inner_opener,
            object_store,
            self.metadata_size_hint,
            self.predicate.clone().unwrap(),
            base_config.file_schema.clone(),
            self.inner.table_parquet_options().global.pruning,
        ))
    }

    fn try_pushdown_filters(
        &self,
        filters: Vec<Arc<dyn PhysicalExpr>>,
        config: &ConfigOptions,
    ) -> Result<FilterPushdownPropagation<Arc<dyn FileSource>>> {
        let inner_result = self.inner.try_pushdown_filters(filters.clone(), config)?;
        match &inner_result.updated_node {
            Some(updated_node) => {
                let updated_inner = Self::try_from_file_source(
                    updated_node.clone(),
                    self.metadata_size_hint,
                    None,
                )?;
                Ok(inner_result.with_updated_node(Arc::new(updated_inner)))
            }
            None => Ok(inner_result),
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn with_batch_size(&self, batch_size: usize) -> Arc<dyn FileSource> {
        Arc::new(Self::from_file_source(
            self.inner.with_batch_size(batch_size),
            self.metadata_size_hint,
            self.predicate.clone(),
        ))
    }

    fn with_schema(&self, schema: SchemaRef) -> Arc<dyn FileSource> {
        Arc::new(Self::from_file_source(
            self.inner.with_schema(schema),
            self.metadata_size_hint,
            self.predicate.clone(),
        ))
    }

    fn with_projection(&self, config: &FileScanConfig) -> Arc<dyn FileSource> {
        Arc::new(Self::from_file_source(
            self.inner.with_projection(config),
            self.metadata_size_hint,
            self.predicate.clone(),
        ))
    }

    fn with_statistics(&self, statistics: Statistics) -> Arc<dyn FileSource> {
        Arc::new(Self::from_file_source(
            self.inner.with_statistics(statistics),
            self.metadata_size_hint,
            self.predicate.clone(),
        ))
    }

    fn metrics(&self) -> &ExecutionPlanMetricsSet {
        // We can and probably should insert some of our own metrics here for easier
        // debugging (e.g., time spent fetching our metadata and number of files/
        // row groups pruned).
        self.inner.metrics()
    }

    fn statistics(&self) -> Result<Statistics> {
        self.inner.statistics()
    }

    fn file_type(&self) -> &str {
        self.inner.file_type()
    }
}

#[cfg(test)]
mod test {
    use std::fmt::Debug;
    use std::ops::Deref;

    use arrow_array::RecordBatch;
    use arrow_schema::DataType;
    use datafusion::config::TableParquetOptions;
    use datafusion::datasource::physical_plan::ParquetSource;
    use datafusion::datasource::schema_adapter::{SchemaAdapter, SchemaAdapterFactory};
    use datafusion::{
        execution::SessionStateBuilder,
        prelude::{col, ParquetReadOptions, SessionContext},
    };
    use datafusion_common::ScalarValue;
    use datafusion_expr::{Expr, Operator, ScalarUDF, Signature, SimpleScalarUDF, Volatility};
    use datafusion_physical_expr::expressions::{BinaryExpr, Column, Literal};
    use datafusion_physical_expr::PhysicalExpr;

    use rstest::rstest;
    use sedona_schema::crs::lnglat;
    use sedona_schema::datatypes::{Edges, SedonaType, WKB_GEOMETRY};
    use sedona_schema::schema::SedonaSchema;
    use sedona_testing::create::create_scalar;
    use sedona_testing::data::{geoarrow_data_dir, test_geoparquet};

    use super::*;

    fn setup_context() -> SessionContext {
        let mut state = SessionStateBuilder::new().build();
        state
            .register_file_format(Arc::new(GeoParquetFormatFactory::new()), true)
            .unwrap();
        SessionContext::new_with_state(state).enable_url_table()
    }

    #[tokio::test]
    async fn format_from_url_table() {
        let ctx = setup_context();
        let example = test_geoparquet("example", "geometry").unwrap();
        let df = ctx.table(&example).await.unwrap();

        // Check that the logical plan resulting from a read has the correct schema
        assert_eq!(
            df.schema().clone().strip_qualifiers().field_names(),
            ["wkt", "geometry"]
        );

        let sedona_types = df
            .schema()
            .sedona_types()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(sedona_types.len(), 2);
        assert_eq!(sedona_types[0], SedonaType::Arrow(DataType::Utf8View));
        assert_eq!(
            sedona_types[1],
            SedonaType::WkbView(Edges::Planar, lnglat())
        );

        // Check that the batches resulting from the actual execution also have
        // the correct schema
        let batches = df.collect().await.unwrap();
        assert_eq!(batches.len(), 1);
        let sedona_types = batches[0]
            .schema()
            .sedona_types()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(sedona_types.len(), 2);
        assert_eq!(sedona_types[0], SedonaType::Arrow(DataType::Utf8View));
        assert_eq!(
            sedona_types[1],
            SedonaType::WkbView(Edges::Planar, lnglat())
        );

        // Check that the content is the same as if it were read by the normal reader
        let unwrapped_batches: Vec<_> = batches
            .into_iter()
            .map(|batch| {
                let fields_without_metadata: Vec<_> = batch
                    .schema()
                    .clone()
                    .fields()
                    .iter()
                    .map(|f| f.deref().clone().with_metadata(HashMap::new()))
                    .collect();
                let schema = Schema::new_with_metadata(fields_without_metadata, HashMap::new());
                RecordBatch::try_new(Arc::new(schema), batch.columns().to_vec()).unwrap()
            })
            .collect();

        let batches_from_parquet = ctx
            .read_parquet(example, ParquetReadOptions::default())
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(unwrapped_batches, batches_from_parquet)
    }

    #[tokio::test]
    async fn projection_without_spatial() {
        let example = test_geoparquet("example", "geometry").unwrap();
        let ctx = setup_context();

        // Completely deselect all geometry columns
        let df = ctx
            .table(&example)
            .await
            .unwrap()
            .select(vec![col("wkt")])
            .unwrap();

        let sedona_types = df
            .schema()
            .sedona_types()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(sedona_types.len(), 1);
        assert_eq!(sedona_types[0], SedonaType::Arrow(DataType::Utf8View));
    }

    #[tokio::test]
    async fn multiple_files() {
        let ctx = setup_context();
        let data_dir = geoarrow_data_dir().unwrap();
        let df = ctx
            .table(format!("{data_dir}/example/files/*_geo.parquet"))
            .await
            .unwrap();

        let sedona_types = df
            .schema()
            .sedona_types()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(sedona_types.len(), 2);
        assert_eq!(sedona_types[0], SedonaType::Arrow(DataType::Utf8View));
        assert_eq!(
            sedona_types[1],
            SedonaType::WkbView(Edges::Planar, lnglat())
        );

        // Make sure all the rows show up!
        let batches = df.collect().await.unwrap();
        let mut total_size = 0;
        for batch in batches {
            total_size += batch.num_rows();
        }
        assert_eq!(total_size, 244);
    }

    #[rstest]
    #[tokio::test]
    async fn pruning_geoparquet_metadata(#[values("st_intersects", "st_contains")] udf_name: &str) {
        let data_dir = geoarrow_data_dir().unwrap();
        let ctx = setup_context();

        let udf: ScalarUDF = SimpleScalarUDF::new_with_signature(
            udf_name,
            Signature::any(2, Volatility::Immutable),
            DataType::Boolean,
            Arc::new(|_args| Ok(ScalarValue::Boolean(Some(true)).into())),
        )
        .into();

        let definitely_non_intersecting_scalar =
            create_scalar(Some("POINT (100 200)"), &WKB_GEOMETRY);
        let storage_field = WKB_GEOMETRY.to_storage_field("", true).unwrap();

        let df = ctx
            .table(format!("{data_dir}/example/files/*_geo.parquet"))
            .await
            .unwrap()
            .filter(udf.call(vec![
                col("geometry"),
                Expr::Literal(
                    definitely_non_intersecting_scalar,
                    Some(storage_field.metadata().into()),
                ),
            ]))
            .unwrap();

        let batches_out = df.collect().await.unwrap();
        assert!(batches_out.is_empty());

        let definitely_intersecting_scalar = create_scalar(Some("POINT (30 10)"), &WKB_GEOMETRY);
        let df = ctx
            .table(format!("{data_dir}/example/files/*_geo.parquet"))
            .await
            .unwrap()
            .filter(udf.call(vec![
                col("geometry"),
                Expr::Literal(
                    definitely_intersecting_scalar,
                    Some(storage_field.metadata().into()),
                ),
            ]))
            .unwrap();

        let batches_out = df.collect().await.unwrap();
        assert!(!batches_out.is_empty());
    }

    #[tokio::test]
    async fn should_not_prune_geoparquet_metadata_after_disabling_pruning() {
        let data_dir = geoarrow_data_dir().unwrap();
        let ctx = setup_context();
        ctx.sql("SET datafusion.execution.parquet.pruning TO false")
            .await
            .expect("Disabling parquet pruning failed");

        let udf: ScalarUDF = SimpleScalarUDF::new_with_signature(
            "st_intersects",
            Signature::any(2, Volatility::Immutable),
            DataType::Boolean,
            Arc::new(|_args| Ok(ScalarValue::Boolean(Some(true)).into())),
        )
        .into();

        let definitely_non_intersecting_scalar =
            create_scalar(Some("POINT (100 200)"), &WKB_GEOMETRY);
        let storage_field = WKB_GEOMETRY.to_storage_field("", true).unwrap();

        let df = ctx
            .table(format!("{data_dir}/example/files/*_geo.parquet"))
            .await
            .unwrap()
            .filter(udf.call(vec![
                col("geometry"),
                Expr::Literal(
                    definitely_non_intersecting_scalar,
                    Some(storage_field.metadata().into()),
                ),
            ]))
            .unwrap();

        // Even if the query window does not intersect with the data, we should not prune
        // any files because pruning has been disabled. We can retrieve the data here
        // because the dummy UDF always returns true.
        let batches_out = df.collect().await.unwrap();
        assert!(!batches_out.is_empty());
    }

    #[tokio::test]
    async fn geoparquet_format_factory() {
        let ctx = SessionContext::new();
        let format_factory = Arc::new(GeoParquetFormatFactory::new());
        let dyn_format = format_factory
            .create(&ctx.state(), &HashMap::new())
            .unwrap();
        assert!(dyn_format
            .as_any()
            .downcast_ref::<GeoParquetFormat>()
            .is_some());
    }

    #[tokio::test]
    async fn test_with_predicate() {
        // Create a parquet source with the correct constructor signature
        let parquet_source = ParquetSource::new(TableParquetOptions::default());

        // Create a simple predicate (column > 0)
        let column = Arc::new(Column::new("test", 0));
        let value = Arc::new(Literal::new(ScalarValue::Int32(Some(0))));
        let predicate: Arc<dyn PhysicalExpr> =
            Arc::new(BinaryExpr::new(column, Operator::Gt, value));

        // Create GeoParquetFileSource and apply predicate
        let geo_source =
            GeoParquetFileSource::try_from_file_source(Arc::new(parquet_source), None, None)
                .unwrap();
        let geo_source_with_predicate = geo_source.with_predicate(predicate);
        assert!(geo_source_with_predicate.inner.predicate().is_some());
    }

    #[tokio::test]
    async fn test_with_schema_adapter_factory() {
        // Create a parquet source with the correct constructor signature
        let parquet_source = ParquetSource::new(TableParquetOptions::default());

        // Create a simple schema adapter factory
        #[derive(Debug)]
        struct MockSchemaAdapterFactory;

        impl SchemaAdapterFactory for MockSchemaAdapterFactory {
            fn create(
                &self,
                _projected_table_schema: SchemaRef,
                _table_schema: SchemaRef,
            ) -> Box<dyn SchemaAdapter> {
                // Since we never use the result, we can return unimplemented
                // We only want to test that the method chains properly
                unimplemented!()
            }
        }

        let schema_adapter_factory: Arc<dyn SchemaAdapterFactory> =
            Arc::new(MockSchemaAdapterFactory);

        // Create GeoParquetFileSource and apply schema adapter factory
        let geo_source =
            GeoParquetFileSource::try_from_file_source(Arc::new(parquet_source), None, None)
                .unwrap();
        let geo_source_with_adapter =
            geo_source.with_schema_adapter_factory(schema_adapter_factory);
        assert!(geo_source_with_adapter
            .inner
            .schema_adapter_factory()
            .is_some());
    }
}
