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

//! URL-as-table resolution for directory-shaped external formats.
//!
//! **Experimental**: the shape of this hook may change.
//!
//! DataFusion's [`enable_url_table`](SessionContext::enable_url_table)
//! installs a resolver (`DynamicListTableFactory`) that always builds a
//! `ListingTable` for a bare `FROM '<url>'`: it lists the files under the
//! URL prefix, takes the first object it finds, and picks a file format by
//! *that object's* extension. For a directory-shaped format like Zarr —
//! where the "table" is the `.zarr` directory itself, not the files within
//! it — this lists the directory contents and tries to parse an inner chunk
//! (e.g. `zarr.json` or a raw binary chunk) as the wrong format, which fails.
//!
//! [`enable_sedona_url_table`] installs [`SedonaUrlTableFactory`] instead. It
//! matches the URL's extension against the session's registered
//! [`ExternalFormatSpec`](sedona_datasource::spec::ExternalFormatSpec)s: when
//! the match is a directory-shaped format
//! ([`list_single_object`](sedona_datasource::spec::ExternalFormatSpec::list_single_object)
//! `== true`), it builds a
//! [`SingleObjectExternalTable`](sedona_datasource::provider::SingleObjectExternalTable)
//! (via [`external_table`]) that passes the URL through untouched. Everything
//! else — the file-shaped formats DataFusion already handles (GeoParquet,
//! CSV, ...) — delegates to DataFusion's default resolver unchanged.

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::{
    catalog::TableProvider,
    datasource::{dynamic_file::DynamicListTableFactory, listing::ListingTableUrl},
    execution::SessionState,
    prelude::SessionContext,
};
use datafusion_catalog::{DynamicFileCatalog, UrlTableFactory};
use datafusion_common::Result;
use datafusion_session::SessionStore;

use sedona_datasource::{format::ExternalFormatFactory, provider::external_table};

/// Install SedonaDB's URL-as-table resolver on `ctx`.
///
/// Drop-in replacement for DataFusion's
/// [`SessionContext::enable_url_table`] that additionally routes
/// directory-shaped external formats through the single-object table
/// path. Mirrors `enable_url_table`'s wiring: it wraps the current catalog
/// list in a [`DynamicFileCatalog`] backed by a [`SedonaUrlTableFactory`],
/// then points the factory's session store at the (unchanged) session
/// state so it can resolve registered file formats at query time.
///
/// **Experimental.**
pub fn enable_sedona_url_table(ctx: SessionContext) -> SessionContext {
    let factory = Arc::new(SedonaUrlTableFactory::new());
    let current_catalog_list = ctx.state().catalog_list().clone();
    let catalog_list = Arc::new(DynamicFileCatalog::new(
        current_catalog_list,
        Arc::clone(&factory) as Arc<dyn UrlTableFactory>,
    ));
    ctx.register_catalog_list(catalog_list);
    factory.session_store().with_state(ctx.state_weak_ref());
    ctx
}

/// [`UrlTableFactory`] that pre-routes directory-shaped external formats to
/// the single-object table path and delegates everything else to
/// DataFusion's default [`DynamicListTableFactory`].
///
/// **Experimental.**
#[derive(Debug)]
pub struct SedonaUrlTableFactory {
    /// DataFusion's default resolver, used for every URL that does not
    /// resolve to a registered directory-shaped format. Owns the
    /// [`SessionStore`] that both it and our routing logic read the live
    /// [`SessionState`] from.
    inner: DynamicListTableFactory,
}

impl SedonaUrlTableFactory {
    /// Create a factory with a fresh [`SessionStore`]. Wire the store to a
    /// session with [`SessionStore::with_state`] (done for you by
    /// [`enable_sedona_url_table`]) before resolving any URL.
    pub fn new() -> Self {
        Self {
            inner: DynamicListTableFactory::new(SessionStore::new()),
        }
    }

    /// The [`SessionStore`] shared by the routing logic and the delegated
    /// [`DynamicListTableFactory`].
    pub fn session_store(&self) -> &SessionStore {
        self.inner.session_store()
    }

    /// Resolve the current [`SessionState`] from the session store, or
    /// `None` if the session has gone away (in which case the caller falls
    /// back to the default resolver, which surfaces the canonical error).
    fn session_state(&self) -> Option<SessionState> {
        self.session_store()
            .get_session()
            .upgrade()
            .and_then(|session| {
                session
                    .read()
                    .as_any()
                    .downcast_ref::<SessionState>()
                    .cloned()
            })
    }

    /// Build a single-object table for `url` if its extension resolves to a
    /// registered directory-shaped [`ExternalFormatFactory`]; otherwise
    /// `None` so the caller delegates to DataFusion's default resolver.
    async fn try_single_object(&self, url: &str) -> Result<Option<Arc<dyn TableProvider>>> {
        let Ok(table_url) = ListingTableUrl::parse(url) else {
            return Ok(None);
        };
        let Some(extension) = url_extension(url) else {
            return Ok(None);
        };
        let Some(state) = self.session_state() else {
            return Ok(None);
        };
        let Some(factory) = state.get_file_format_factory(&extension) else {
            return Ok(None);
        };
        let Some(external) = factory.downcast_ref::<ExternalFormatFactory>() else {
            return Ok(None);
        };
        if !external.spec().list_single_object() {
            return Ok(None);
        }

        let spec = external.spec().clone();
        // `SingleObjectExternalTable` only needs the runtime environment
        // (for the object store registry), which the reconstructed context
        // shares with the live session via its `Arc<RuntimeEnv>`.
        let ctx = SessionContext::new_with_state(state);
        let provider = external_table(spec, &ctx, vec![table_url], false, None).await?;
        Ok(Some(provider))
    }
}

impl Default for SedonaUrlTableFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl UrlTableFactory for SedonaUrlTableFactory {
    async fn try_new(&self, url: &str) -> Result<Option<Arc<dyn TableProvider>>> {
        if let Some(provider) = self.try_single_object(url).await? {
            return Ok(Some(provider));
        }
        self.inner.try_new(url).await
    }
}

/// The lower-cased file extension of the last path segment of `url`, or
/// `None` if there is no extension. Matches DataFusion's own extension
/// lookup key (no leading dot) so a format registered under `"zarr"` is
/// found for `.../foo.zarr`.
fn url_extension(url: &str) -> Option<String> {
    let path = url.split(['?', '#']).next().unwrap_or(url);
    let segment = path.trim_end_matches('/').rsplit('/').next()?;
    let (stem, extension) = segment.rsplit_once('.')?;
    if stem.is_empty() {
        // A leading-dot segment like `.hidden` has no extension.
        return None;
    }
    Some(extension.to_lowercase())
}

#[cfg(test)]
mod test {
    use std::{
        collections::HashMap,
        io::{Read, Write},
        path::PathBuf,
        sync::Arc,
    };

    use arrow_array::{
        Int32Array, Int64Array, RecordBatch, RecordBatchIterator, RecordBatchReader, StringArray,
    };
    use arrow_schema::{DataType, Field, Schema};
    use async_trait::async_trait;
    use datafusion::{
        assert_batches_eq,
        execution::SessionStateBuilder,
        prelude::{col, lit, SessionContext},
    };
    use datafusion_common::{plan_err, Result, Statistics};
    use sedona_datasource::spec::{ExternalFormatSpec, Object, OpenReaderArgs};
    use tempfile::TempDir;
    use url::Url;

    use super::*;

    #[test]
    fn extracts_url_extension() {
        assert_eq!(
            url_extension("file:///a/b/foo.zarr").as_deref(),
            Some("zarr")
        );
        // Trailing slash on a directory-shaped URL.
        assert_eq!(
            url_extension("file:///a/b/foo.zarr/").as_deref(),
            Some("zarr")
        );
        // Only the last segment matters; dots in parent dirs are ignored.
        assert_eq!(
            url_extension("file:///a.b/c/foo.TIF").as_deref(),
            Some("tif")
        );
        // Query strings are stripped before extracting the extension.
        assert_eq!(
            url_extension("https://host/foo.zarr?x=1.2").as_deref(),
            Some("zarr")
        );
        assert_eq!(url_extension("file:///a/b/noext"), None);
        assert_eq!(url_extension("file:///a/b/.hidden"), None);
    }

    /// Register `spec` as a file format and return a context with the
    /// SedonaDB URL-as-table resolver installed, so `SELECT * FROM '<url>'`
    /// routes through [`enable_sedona_url_table`].
    fn create_spec_ctx(spec: Arc<dyn ExternalFormatSpec>) -> SessionContext {
        let factory = ExternalFormatFactory::new(spec);

        // Register the format - use new_with_default_features to get default catalogs
        let mut state = SessionStateBuilder::new_with_default_features().build();
        state.register_file_format(Arc::new(factory), true).unwrap();
        enable_sedona_url_table(SessionContext::new_with_state(state))
    }

    fn create_echo_spec_ctx() -> SessionContext {
        create_spec_ctx(Arc::new(EchoSpec::default()))
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

    /// Spec for a directory-shaped format whose "object" is the
    /// directory itself. Used to exercise the
    /// [`SingleObjectExternalTable`](sedona_datasource::provider::SingleObjectExternalTable)
    /// path through
    /// [`external_table`](sedona_datasource::provider::external_table).
    #[derive(Debug, Default, Clone)]
    struct DirectorySpec;

    #[async_trait]
    impl ExternalFormatSpec for DirectorySpec {
        fn extension(&self) -> &str {
            // No leading dot: file formats register under this key
            // lower-cased, and both DataFusion's listing resolver and the
            // URL-as-table resolver look them up by the dot-free extension
            // of the path (`foo.dirfmt` -> `dirfmt`).
            "dirfmt"
        }

        fn list_single_object(&self) -> bool {
            true
        }

        fn with_options(
            &self,
            _options: &HashMap<String, String>,
        ) -> Result<Arc<dyn ExternalFormatSpec>> {
            Ok(Arc::new(self.clone()))
        }

        async fn infer_schema(&self, location: &Object) -> Result<Schema> {
            // The single-object provider must synthesise an ObjectMeta
            // before calling us; assert that contract here.
            assert!(
                location.meta.is_some(),
                "single-object scan must synthesise an ObjectMeta",
            );
            Ok(Schema::new(vec![
                Field::new("uri_path", DataType::Utf8, false),
                Field::new("row_idx", DataType::Int32, false),
            ]))
        }

        async fn open_reader(
            &self,
            args: &OpenReaderArgs,
        ) -> Result<Box<dyn RecordBatchReader + Send>> {
            let meta = args
                .src
                .meta
                .as_ref()
                .expect("single-object scan must synthesise an ObjectMeta");
            let path = meta.location.to_string();
            let schema = Arc::new(self.infer_schema(&args.src).await?);
            let mut batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(StringArray::from(vec![path])),
                    Arc::new(Int32Array::from(vec![0])),
                ],
            )?;
            if let Some(projection) = &args.file_projection {
                batch = batch.project(projection)?;
            }
            Ok(Box::new(RecordBatchIterator::new([Ok(batch)], schema)))
        }
    }

    /// Build a `file://` URL for a local filesystem path.
    fn file_url(path: &std::path::Path) -> String {
        url::Url::from_file_path(path).unwrap().to_string()
    }

    #[tokio::test]
    async fn spec_format() {
        let ctx = create_echo_spec_ctx();
        let (temp_dir, files) = create_echo_spec_temp_dir();

        // Select using just the filename and ensure we get a result
        // Quote the path to prevent it from being parsed as a multi-part identifier
        let batches_item0 = ctx
            .table(format!("\"{}\"", files[0].to_string_lossy()))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(batches_item0.len(), 1);
        assert_eq!(batches_item0[0].num_rows(), 1);

        // With a glob we should get all the files
        let batches = ctx
            .table(format!(
                "\"{}/*.echospec\"",
                temp_dir.path().to_string_lossy()
            ))
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
        // Quote the path to prevent it from being parsed as a multi-part identifier
        let batches = ctx
            .table(format!(
                "\"{}/*.echospec\"",
                temp_dir.path().to_string_lossy()
            ))
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
    async fn url_table_routes_directory_shape_to_single_object() {
        // `SELECT * FROM '<dir-url>'` for a directory-shaped format must
        // resolve through the single-object path, not DataFusion's listing
        // path. The `group.dirfmt` directory contains a `metadata.json`
        // file: the listing path would list that inner file, key on its
        // `json` extension, and read it as the wrong format — returning the
        // wrong rows (or erroring). The single-object path passes the
        // directory URL straight to the spec, so we get its one synthetic
        // row describing the directory itself.
        let ctx = create_spec_ctx(Arc::new(DirectorySpec));
        let temp_dir = TempDir::new().unwrap();
        let dir_path = temp_dir.path().join("group.dirfmt");
        std::fs::create_dir(&dir_path).unwrap();
        std::fs::File::create(dir_path.join("metadata.json"))
            .unwrap()
            .write_all(b"{}")
            .unwrap();

        // Quote the URL so it resolves as a single table reference rather
        // than a multi-part identifier — the same catalog path SQL's
        // `FROM '<url>'` takes.
        let batches = ctx
            .table(format!("\"{}\"", file_url(&dir_path)))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 1);
        let uri_path = batches[0]
            .column_by_name("uri_path")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        // The single-object path passed the directory URL through; the
        // listing path would instead have surfaced `metadata.json`.
        assert!(uri_path.value(0).ends_with("group.dirfmt"));
        let row_idx = batches[0]
            .column_by_name("row_idx")
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(row_idx.value(0), 0);
    }

    #[tokio::test]
    async fn url_table_file_shape_still_lists() {
        // A file-shaped format (`list_single_object = false`) must keep
        // resolving through DataFusion's listing path unchanged: the glob
        // fans out to one row per matching file rather than treating the
        // directory as a single object.
        let ctx = create_echo_spec_ctx();
        let (temp_dir, _files) = create_echo_spec_temp_dir();

        let glob = format!("{}/*.echospec", temp_dir.path().to_string_lossy());
        let batches = ctx
            .table(format!("\"{glob}\""))
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 2);
    }
}
