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
//! [`ExternalFormatSpec`](crate::spec::ExternalFormatSpec)s: when the match
//! is a directory-shaped format
//! ([`list_single_object`](crate::spec::ExternalFormatSpec::list_single_object)
//! `== true`), it builds a
//! [`SingleObjectExternalTable`](crate::provider::SingleObjectExternalTable)
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

use crate::{format::ExternalFormatFactory, provider::external_table};

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
        let Some(external) = factory.as_any().downcast_ref::<ExternalFormatFactory>() else {
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
}
