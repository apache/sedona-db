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

use std::any::Any;
use std::sync::Arc;

use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::{
    catalog::TableProvider,
    config::TableOptions,
    datasource::{
        file_format::FileFormat,
        listing::{ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl},
        physical_plan::FileScanConfig,
        TableType,
    },
    execution::{options::ReadOptions, SessionState},
    logical_expr::Expr,
    physical_plan::ExecutionPlan,
    prelude::{SessionConfig, SessionContext},
};
use datafusion_catalog::{memory::DataSourceExec, Session};
use datafusion_common::{exec_err, Result};
use datafusion_datasource::{
    file_groups::FileGroup, file_scan_config::FileScanConfigBuilder, table_schema::TableSchema,
    PartitionedFile,
};
use datafusion_execution::object_store::ObjectStoreUrl;
use object_store::{path::Path as ObjectPath, ObjectMeta};

use crate::{
    format::ExternalFileFormat,
    spec::{ExternalFormatSpec, Object},
};

/// Resolve an [ExternalFormatSpec] + URLs into a [TableProvider].
///
/// Dispatches on [`ExternalFormatSpec::list_single_object`]:
/// - `false` (default): builds a [`ListingTable`] that lists files at
///   the URL prefix matching the spec's extension. Best for formats
///   whose unit of work is a single file (Parquet, FlatGeobuf, ...).
/// - `true`: builds a [`SingleObjectExternalTable`] that treats each
///   URI as one opaque object, skipping listing entirely. Required
///   for directory-shaped formats like Zarr.
pub async fn external_table(
    spec: Arc<dyn ExternalFormatSpec>,
    context: &SessionContext,
    table_paths: Vec<ListingTableUrl>,
    check_extension: bool,
) -> Result<Arc<dyn TableProvider>> {
    if table_paths.is_empty() {
        return exec_err!("No table paths were provided");
    }

    if spec.list_single_object() {
        let provider = SingleObjectExternalTable::try_new(spec, table_paths).await?;
        Ok(Arc::new(provider) as Arc<dyn TableProvider>)
    } else {
        let provider = external_listing_table(spec, context, table_paths, check_extension).await?;
        Ok(Arc::new(provider) as Arc<dyn TableProvider>)
    }
}

/// Create a [ListingTable] from an [ExternalFormatSpec] and one or more URLs
///
/// This can be used to resolve a format specification into a TableProvider that
/// may be registered with a [SessionContext].
pub async fn external_listing_table(
    spec: Arc<dyn ExternalFormatSpec>,
    context: &SessionContext,
    table_paths: Vec<ListingTableUrl>,
    check_extension: bool,
) -> Result<ListingTable> {
    let session_config = context.copied_config();
    let options = RecordBatchReaderTableOptions {
        spec,
        check_extension,
    };
    let listing_options =
        options.to_listing_options(&session_config, context.copied_table_options());

    let option_extension = listing_options.file_extension.clone();

    if table_paths.is_empty() {
        return exec_err!("No table paths were provided");
    }

    // check if the file extension matches the expected extension if one is provided
    if !option_extension.is_empty() && options.check_extension {
        for path in &table_paths {
            let file_path = path.as_str();
            if !file_path.ends_with(option_extension.clone().as_str()) && !path.is_collection() {
                return exec_err!(
                        "File path '{file_path}' does not match the expected extension '{option_extension}'"
                    );
            }
        }
    }

    let resolved_schema = options
        .get_resolved_schema(&session_config, context.state(), table_paths[0].clone())
        .await?;
    let config = ListingTableConfig::new_with_multi_paths(table_paths)
        .with_listing_options(listing_options)
        .with_schema(resolved_schema);

    ListingTable::try_new(config)
}

#[derive(Debug, Clone)]
struct RecordBatchReaderTableOptions {
    spec: Arc<dyn ExternalFormatSpec>,
    check_extension: bool,
}

#[async_trait]
impl ReadOptions<'_> for RecordBatchReaderTableOptions {
    fn to_listing_options(
        &self,
        config: &SessionConfig,
        table_options: TableOptions,
    ) -> ListingOptions {
        let format = if let Some(modified) = self.spec.with_table_options(&table_options) {
            ExternalFileFormat::new(modified)
        } else {
            ExternalFileFormat::new(self.spec.clone())
        };

        ListingOptions::new(Arc::new(format))
            .with_file_extension(self.spec.extension())
            .with_session_config_options(config)
    }

    async fn get_resolved_schema(
        &self,
        config: &SessionConfig,
        state: SessionState,
        table_path: ListingTableUrl,
    ) -> Result<SchemaRef> {
        self.to_listing_options(config, state.default_table_options())
            .infer_schema(&state, &table_path)
            .await
    }
}

/// [`TableProvider`] that treats each input URI as one opaque object.
///
/// Built when [`ExternalFormatSpec::list_single_object`] is `true`. The
/// listing layer is bypassed entirely: each URI is synthesised into a
/// single-element [`PartitionedFile`] whose `object_meta.location` is
/// the URL's path within the [`ObjectStoreUrl`]. The format's
/// [`ExternalFileFormat::file_source`] then drives the same scan path
/// used by [`ListingTable`] — projections, filter pushdown, and
/// streaming behaviour are identical.
///
/// Required for directory-shaped formats like Zarr where the
/// "object" is the directory itself.
#[derive(Debug)]
pub struct SingleObjectExternalTable {
    spec: Arc<dyn ExternalFormatSpec>,
    schema: SchemaRef,
    /// Each input URI as (scheme-level object store URL, path within store).
    /// All entries must share the same object store URL — mixed schemes
    /// (e.g. `file://` + `s3://`) are rejected.
    files: Vec<(ObjectStoreUrl, ObjectPath)>,
}

impl SingleObjectExternalTable {
    async fn try_new(
        spec: Arc<dyn ExternalFormatSpec>,
        table_paths: Vec<ListingTableUrl>,
    ) -> Result<Self> {
        let files: Vec<(ObjectStoreUrl, ObjectPath)> = table_paths
            .iter()
            .map(|p| (p.object_store(), p.prefix().clone()))
            .collect();

        // All URIs must resolve to the same object store. Mixing schemes
        // (e.g. one file://, one s3://) would force the scan to dispatch
        // to multiple stores — not supported here.
        let first_store = files[0].0.clone();
        for (store, _) in &files[1..] {
            if store != &first_store {
                return exec_err!(
                    "external_table: all URIs in a single-object scan must share the same \
                     object store; got both '{first_store}' and '{store}'"
                );
            }
        }

        // Resolve the schema from the first object. Most directory-format
        // specs (e.g. Zarr) infer a fixed schema irrespective of the
        // input; the few that hit the store will receive a synthesised
        // ObjectMeta they can use.
        let probe = Object {
            store: None,
            url: Some(first_store.clone()),
            meta: Some(synthetic_object_meta(&files[0].1)),
            range: None,
        };
        let schema = Arc::new(spec.infer_schema(&probe).await?);

        Ok(Self {
            spec,
            schema,
            files,
        })
    }
}

#[async_trait]
impl TableProvider for SingleObjectExternalTable {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let (object_store_url, _) = &self.files[0];

        let table_schema = TableSchema::new(self.schema.clone(), vec![]);
        let format = ExternalFileFormat::new(self.spec.clone());
        let file_source = format.file_source(table_schema);

        let partitioned_files: Vec<PartitionedFile> = self
            .files
            .iter()
            .map(|(_, location)| PartitionedFile {
                object_meta: synthetic_object_meta(location),
                partition_values: vec![],
                range: None,
                extensions: None,
                statistics: None,
                metadata_size_hint: None,
            })
            .collect();

        let mut builder = FileScanConfigBuilder::new(object_store_url.clone(), file_source)
            .with_file_group(FileGroup::new(partitioned_files))
            .with_limit(limit);
        if let Some(indices) = projection {
            builder = builder.with_projection_indices(Some(indices.clone()))?;
        }
        let config: FileScanConfig = builder.build();
        Ok(DataSourceExec::from_data_source(config))
    }
}

/// Synthesise an [`ObjectMeta`] for a URI we haven't `head`'d.
///
/// `size: 0` and a zeroed `last_modified` are intentional: this table
/// provider never lists or stats objects, so DataFusion's downstream
/// machinery only uses the `location` field. Specs that need real
/// stats can override via [`ExternalFormatSpec::infer_stats`].
fn synthetic_object_meta(location: &ObjectPath) -> ObjectMeta {
    ObjectMeta {
        location: location.clone(),
        last_modified: Default::default(),
        size: 0,
        e_tag: None,
        version: None,
    }
}
