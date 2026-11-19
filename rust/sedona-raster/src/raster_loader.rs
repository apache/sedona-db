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

//! Async byte-loading for schema-OutDb raster bands.
//!
//! `sedona-raster` deliberately knows nothing about GDAL, Zarr, or any
//! other backend. Backends implement [`AsyncRasterLoader`] — declaring which
//! codec pipelines they decode, by codec name — and register themselves
//! against an [`RasterLoaderRegistry`]. The
//! `RS_EnsureLoaded` UDF in the `sedona` crate consumes the registry to
//! materialise OutDb bands at query time; band accessors
//! (`BandRef::nd_buffer()` / `contiguous_data()`) do **not** invoke the
//! loader transparently — they return whatever is in the `data` column
//! verbatim, surfacing a clear error when the column is empty.

use std::sync::{Arc, RwLock};

use arrow_buffer::Buffer;
use arrow_schema::ArrowError;
use datafusion_common::config::{
    ConfigEntry, ConfigExtension, ConfigField, ConfigOptions, ExtensionOptions, Visit,
};
use datafusion_common::{Result as DFResult, config_err};
use sedona_common::option::SedonaOptions;
use sedona_schema::raster::BandDataType;

use crate::chunk_cache::{NoChunkCache, RasterChunkCache};
use crate::chunks::ChunkRef;
use crate::io_budget::RasterIoBudget;
use crate::view_entries::ViewEntries;

/// Everything a backend needs to materialise a single OutDb band's bytes.
///
/// Constructed by `RS_EnsureLoaded` once per row from the band's schema
/// metadata. The lifetime is the request's, not the loader's — borrowed
/// fields point into the input `RecordBatch` and stay valid for the
/// duration of the [`AsyncRasterLoader::load`] future.
///
/// The request carries the band's `view`, but a loader may ignore it: it
/// returns a [`RasterLoadResult`] that *describes* the bytes it produced
/// (their `source_shape` and `view`), so the caller never needs a separate
/// "did you resolve the view?" flag. A loader that doesn't prune I/O returns
/// the full `source_shape` with the view unresolved
/// ([`RasterLoadResult::unresolved`]); one that range-reads a sub-window
/// returns the visible shape with an identity view. Actually pruning by the
/// view is tracked in <https://github.com/apache/sedona-db/issues/897>.
#[derive(Debug, Clone, Copy)]
pub struct RasterLoadRequest<'a> {
    /// The band's `chunks` list in grid (row-major) order; `None` entries
    /// are missing chunks. A container codec such as
    /// [`GDAL_CODEC`](crate::chunks::GDAL_CODEC) reads bands of exactly one
    /// whole-object chunk ([`Self::single_chunk`]); stitching several chunks
    /// into one band is the generic loader's job.
    pub chunks: &'a [Option<ChunkRef<'a>>],
    /// Pixels per chunk per dimension, parallel to `source_shape`.
    pub chunk_shape: &'a [i64],
    /// The band's codec pipeline as JSON, shared by every chunk. Loaders
    /// parse it with [`crate::chunks::CodecPipeline::parse`] for their
    /// configuration; the registry already matched it by name.
    pub codecs: &'a str,
    /// Per-axis names parallel to `source_shape`. Backends use this to
    /// map their native axis order onto the band's.
    pub dim_names: &'a [&'a str],
    /// Raw source shape in `dim_names` order. A loader that returns the full
    /// source produces a `Buffer` of `Π source_shape × byte_size` bytes in
    /// C-order over `dim_names`.
    pub source_shape: &'a [i64],
    /// The band's view over `source_shape` (the visible region). A loader
    /// may ignore it and return the full source (reporting the view
    /// unresolved), or honor it and return only the visible region; the
    /// returned [`RasterLoadResult`] says which.
    pub view: &'a ViewEntries,
    /// Pixel type the band claims. The loader returns bytes encoding this
    /// type and errors if the source disagrees (e.g. file's dtype differs).
    pub data_type: BandDataType,
}

impl<'a> RasterLoadRequest<'a> {
    /// The one whole-object chunk of a band on a container codec: `chunks`
    /// has exactly one entry, it is present, and it is not a byte range.
    /// Loaders that decode whole objects (GDAL, the transitional Zarr codec)
    /// call this; a band of several chunks is an error for them.
    pub fn single_chunk(&self) -> Result<ChunkRef<'a>, ArrowError> {
        let chunk = match self.chunks {
            [Some(chunk)] => *chunk,
            [None] => {
                return Err(ArrowError::InvalidArgumentError(
                    "the band's only chunk is missing".to_string(),
                ));
            }
            _ => {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "this loader reads single-chunk bands only, got {} chunks (codecs {})",
                    self.chunks.len(),
                    self.codecs
                )));
            }
        };
        if !chunk.is_whole_object() {
            return Err(ArrowError::InvalidArgumentError(format!(
                "this loader reads whole objects only, got a byte range of {} (codecs {})",
                chunk.uri.unwrap_or("<no uri>"),
                self.codecs
            )));
        }
        Ok(chunk)
    }

    /// [`Self::single_chunk`]'s URI.
    pub fn single_chunk_uri(&self) -> Result<&'a str, ArrowError> {
        self.single_chunk()?
            .uri
            .ok_or_else(|| ArrowError::InvalidArgumentError("chunk has no uri".to_string()))
    }
}

/// What [`AsyncRasterLoader::load`] returns: the bytes plus the layout they
/// are in. Self-describing, so the caller builds the output band from it
/// without a separate capability flag.
///
/// - **Unresolved** (loader ignored the view): `bytes` is the full source,
///   `source_shape` == request's, `view` == request's. See
///   [`RasterLoadResult::unresolved`].
/// - **Resolved** (loader cropped to the view): `bytes` is the visible
///   region, `source_shape` is the visible shape, `view` is identity.
#[derive(Debug, Clone)]
pub struct RasterLoadResult {
    /// Loaded bytes, C-order over `dim_names`, length
    /// `Π source_shape × data_type.byte_size()`.
    pub bytes: Buffer,
    /// The shape `bytes` are laid out in.
    pub source_shape: Vec<i64>,
    /// View to apply to `bytes` to obtain the visible region (identity when
    /// the loader already resolved the crop).
    pub view: ViewEntries,
}

impl RasterLoadResult {
    /// Full source, view left unresolved — echoes the request. The default
    /// a loader that doesn't prune I/O returns.
    pub fn unresolved(bytes: Buffer, req: &RasterLoadRequest<'_>) -> Self {
        Self {
            bytes,
            source_shape: req.source_shape.to_vec(),
            view: req.view.clone(),
        }
    }
}

/// Backend trait. Implementers live in format-specific crates
/// (`sedona-raster-gdal`, `sedona-raster-zarr`, …) and are registered
/// against an [`RasterLoaderRegistry`]; each declares the codec pipelines
/// it decodes, by codec name, via [`AsyncRasterLoader::supports`].
///
/// Synchronous backends (e.g. GDAL) wrap their I/O in
/// `tokio::task::spawn_blocking` inside the impl — the trait itself stays
/// async-only so the dispatcher (`RS_EnsureLoaded`) can `buffer_unordered`
/// over many in-flight loads. The result type is
/// [`arrow_buffer::Buffer`] (not `Vec<u8>`) so backends that already
/// produce reference-counted bytes (e.g. `object_store` returning
/// `bytes::Bytes`) hand them off zero-copy, and so the dispatcher can
/// build the output `BinaryViewArray` directly from collected Buffers
/// without an extra copy through a `BinaryViewBuilder` block buffer.
#[async_trait::async_trait]
pub trait AsyncRasterLoader: Send + Sync + std::fmt::Debug {
    /// Short name for diagnostics and error messages (e.g. `"gdal"`,
    /// `"zarr"`).
    fn name(&self) -> &str;

    /// Whether this loader decodes a band whose `codecs` pipeline has these
    /// codec names, in pipeline order. A container-codec backend returns
    /// `true` for exactly its one name (GDAL for `["sedona.gdal"]`); a
    /// generic backend returns `true` when every name is in its decoder
    /// table. Nothing guesses: a pipeline no loader claims is an error.
    fn supports(&self, codec_names: &[&str]) -> bool;

    /// Fetch the band's bytes, returning a [`RasterLoadResult`] that
    /// describes the layout produced. A loader that doesn't prune by the
    /// view returns [`RasterLoadResult::unresolved`] (full source, length
    /// `Π source_shape × data_type.byte_size()` in C-order over
    /// `dim_names`). Errors propagate to the caller of `RS_EnsureLoaded`.
    async fn load(&self, req: &[&RasterLoadRequest]) -> Result<Vec<RasterLoadResult>, ArrowError>;
}

/// Process-side registry of raster byte loaders.
///
/// One registry instance per `SedonaContext`. The owning context wraps it
/// in `Arc<RwLock<…>>` so extension crates (`sedona-raster-zarr`, future COG /
/// Icechunk / …) can register their loaders post-construction via a
/// public `SedonaContext::register_raster_loader` API.
///
/// Loaders are held in registration order. A band's codec names are
/// matched against each loader's [`AsyncRasterLoader::supports`], scanning
/// **most-recently-registered first** — so a loader registered later wins
/// for any pipeline both claim (an extension can take over a codec from a
/// compiled-in loader).
#[derive(Debug, Default)]
pub struct RasterLoaderRegistry {
    loaders: Vec<Arc<dyn AsyncRasterLoader>>,
}

impl RasterLoaderRegistry {
    /// Construct an empty registry. Compiled-in backends (`sedona-raster-gdal`
    /// under the `gdal` feature) register themselves from `SedonaContext::new`;
    /// extension backends register via `SedonaContext::register_raster_loader`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a loader. Each loader declares the codec names it decodes
    /// via [`AsyncRasterLoader::supports`]; lookup scans in reverse, so a
    /// loader registered later takes precedence over an earlier one for any
    /// pipeline both accept.
    pub fn register(&mut self, loader: Arc<dyn AsyncRasterLoader>) {
        self.loaders.push(loader);
    }

    /// Find the loader for a band whose `codecs` pipeline has these codec
    /// names. Scans most-recently-registered first and returns the first
    /// loader whose [`AsyncRasterLoader::supports`] accepts them, or `None`
    /// when nothing does.
    pub fn get(&self, codec_names: &[&str]) -> Option<Arc<dyn AsyncRasterLoader>> {
        self.loaders
            .iter()
            .rev()
            .find(|l| l.supports(codec_names))
            .cloned()
    }

    /// Like [`Self::get`], but returns a planner error naming the codecs and
    /// the registered loaders when nothing accepts them. `RS_EnsureLoaded`
    /// dispatches through this so the diagnostic points users at the
    /// install/register step.
    pub fn get_or_error(
        &self,
        codec_names: &[&str],
    ) -> datafusion_common::Result<Arc<dyn AsyncRasterLoader>> {
        if let Some(loader) = self.get(codec_names) {
            return Ok(loader);
        }
        let registered = self.loader_names();
        let registered_msg = if registered.is_empty() {
            "no raster loaders are registered in this session".to_string()
        } else {
            format!("registered loaders: {}", registered.join(", "))
        };
        datafusion_common::plan_err!(
            "no raster loader decodes the codec pipeline {codec_names:?} — {registered_msg}"
        )
    }

    /// Names of registered loaders, most-recently-registered first. Useful
    /// for diagnostics.
    pub fn loader_names(&self) -> Vec<&str> {
        self.loaders.iter().rev().map(|l| l.name()).collect()
    }

    /// True if no loader is registered.
    pub fn is_empty(&self) -> bool {
        self.loaders.is_empty()
    }
}

/// `ConfigField`-shaped wrapper around the shared registry handle.
///
/// Mirrors `sedona_common::option::SedonaRuntime` so the registry
/// can live inside a `ConfigOptions` extension and stay reachable from
/// `AsyncScalarUDFImpl::invoke_async_with_args` (which only receives
/// `Arc<ConfigOptions>`). The inner `Arc<RwLock<...>>` is cloned
/// between the `SedonaContext` (mutable register API) and the config
/// extension (read at UDF dispatch time); both observe the same
/// underlying lock.
#[derive(Debug, Clone)]
pub struct RasterLoaderRegistryOption(Arc<RwLock<RasterLoaderRegistry>>);

impl RasterLoaderRegistryOption {
    /// Wrap an existing shared registry handle.
    pub fn new(inner: Arc<RwLock<RasterLoaderRegistry>>) -> Self {
        Self(inner)
    }

    /// Clone the inner Arc for callers that need their own owning handle
    /// (e.g. `SedonaContext::register_raster_loader` needs to write
    /// through the same lock that the config extension exposes for
    /// reads).
    pub fn handle(&self) -> Arc<RwLock<RasterLoaderRegistry>> {
        Arc::clone(&self.0)
    }
}

impl Default for RasterLoaderRegistryOption {
    fn default() -> Self {
        Self(Arc::new(RwLock::new(RasterLoaderRegistry::new())))
    }
}

impl PartialEq for RasterLoaderRegistryOption {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl ConfigField for RasterLoaderRegistryOption {
    fn visit<V: Visit>(&self, v: &mut V, key: &str, description: &'static str) {
        let snapshot = match self.0.read() {
            Ok(g) => g
                .loader_names()
                .into_iter()
                .map(String::from)
                .collect::<Vec<_>>(),
            Err(p) => p
                .into_inner()
                .loader_names()
                .into_iter()
                .map(String::from)
                .collect::<Vec<_>>(),
        };
        v.some(
            key,
            format!("RasterLoaderRegistry {{ loaders: {snapshot:?} }}"),
            description,
        );
    }

    fn set(&mut self, key: &str, _value: &str) -> DFResult<()> {
        config_err!("Can't set {key} from SQL")
    }
}

/// `ConfigExtension` that stashes the per-session
/// [`RasterLoaderRegistry`] inside a `ConfigOptions`. Registered into
/// the session's `ConfigOptions` during `SedonaContext` construction;
/// consumed by the `RS_EnsureLoaded` async UDF at dispatch time
/// via `args.config_options.extensions.get::<RasterLoaderConfig>()`.
///
/// The PREFIX namespace is `sedona.raster_loader` — kept separate from
/// `sedona`'s main `SedonaOptions` extension because this lives in
/// `sedona-raster` (which is upstream of `sedona-common` in the
/// dependency graph) and adding a field to `SedonaOptions` would
/// require an undesirable circular dep.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct RasterLoaderConfig {
    pub registry: RasterLoaderRegistryOption,
    /// The session's [`RasterChunkCache`], a [`NoChunkCache`] until one is
    /// attached. Read by `RS_EnsureLoaded` before dispatching to a loader.
    pub cache: RasterChunkCacheOption,
    /// The session's [`RasterIoBudget`], shared with the session's GDAL
    /// loader and read by `RS_FromPath`.
    pub io_budget: RasterIoBudgetOption,
}

impl RasterLoaderConfig {
    /// Build a config extension that closes over an existing shared
    /// registry handle; the cache keeps nothing until [`Self::with_cache`].
    pub fn from_handle(registry: Arc<RwLock<RasterLoaderRegistry>>) -> Self {
        Self {
            registry: RasterLoaderRegistryOption::new(registry),
            cache: RasterChunkCacheOption::default(),
            io_budget: RasterIoBudgetOption::default(),
        }
    }

    /// Attach the session's chunk cache.
    pub fn with_cache(mut self, cache: Arc<dyn RasterChunkCache>) -> Self {
        self.cache = RasterChunkCacheOption::new(cache);
        self
    }

    /// The session's chunk cache.
    pub fn cache(&self) -> Arc<dyn RasterChunkCache> {
        Arc::clone(&self.cache.0)
    }

    /// Attach the session's I/O budget. Pass the same budget to every
    /// blocking raster I/O user of the session (the GDAL loader).
    pub fn with_io_budget(mut self, budget: RasterIoBudget) -> Self {
        self.io_budget = RasterIoBudgetOption(budget);
        self
    }

    /// The session's I/O budget.
    pub fn io_budget(&self) -> RasterIoBudget {
        self.io_budget.0.clone()
    }
}

/// The session's [`RasterIoBudget`], with `sedona.raster.io_concurrency`
/// applied so a `SET` takes effect on the next call, or `None` when the
/// session carries no [`RasterLoaderConfig`] (a bare DataFusion context).
/// Every user of the budget calls this before taking a permit.
pub fn io_budget_from_config(config: &ConfigOptions) -> Option<RasterIoBudget> {
    let budget = config.extensions.get::<RasterLoaderConfig>()?.io_budget();
    if let Some(opts) = config.extensions.get::<SedonaOptions>() {
        let limit = opts.raster.io_concurrency.max(1);
        if budget.limit() != limit {
            budget.set_limit(limit);
        }
    }
    Some(budget)
}

/// `ConfigField`-shaped handle to the session's [`RasterIoBudget`], so it
/// rides in the same `ConfigOptions` extension as the loader registry.
/// Compares by identity; SQL cannot set it (the limit is
/// `sedona.raster.io_concurrency`, applied by [`io_budget_from_config`]).
#[derive(Debug, Clone, Default)]
pub struct RasterIoBudgetOption(RasterIoBudget);

impl PartialEq for RasterIoBudgetOption {
    fn eq(&self, other: &Self) -> bool {
        self.0.ptr_eq(&other.0)
    }
}

impl ConfigField for RasterIoBudgetOption {
    fn visit<V: Visit>(&self, v: &mut V, key: &str, description: &'static str) {
        v.some(
            key,
            format!(
                "RasterIoBudget {{ limit: {}, in_use: {} }}",
                self.0.limit(),
                self.0.in_use()
            ),
            description,
        )
    }

    fn set(&mut self, key: &str, _value: &str) -> DFResult<()> {
        config_err!("Can't set {key} from SQL")
    }
}

/// `ConfigField`-shaped handle to the session's [`RasterChunkCache`], so it
/// rides in the same `ConfigOptions` extension as the loader registry and
/// is reachable from `RS_EnsureLoaded`. Defaults to a [`NoChunkCache`].
/// Compares by identity; SQL cannot set it (the budget is
/// `sedona.raster.cache_max_bytes`, applied to the cache at dispatch time).
#[derive(Debug, Clone)]
pub struct RasterChunkCacheOption(Arc<dyn RasterChunkCache>);

impl RasterChunkCacheOption {
    /// Wrap the session's cache.
    pub fn new(cache: Arc<dyn RasterChunkCache>) -> Self {
        Self(cache)
    }
}

impl Default for RasterChunkCacheOption {
    fn default() -> Self {
        Self(Arc::new(NoChunkCache))
    }
}

impl PartialEq for RasterChunkCacheOption {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl ConfigField for RasterChunkCacheOption {
    fn visit<V: Visit>(&self, v: &mut V, key: &str, description: &'static str) {
        let stats = self.0.stats();
        v.some(
            key,
            format!(
                "RasterChunkCache {{ max_bytes: {}, entries: {}, idle_bytes: {}, \
                 shared_bytes: {} }}",
                self.0.max_bytes(),
                stats.entries,
                stats.idle_bytes,
                stats.shared_bytes
            ),
            description,
        )
    }

    fn set(&mut self, key: &str, _value: &str) -> DFResult<()> {
        config_err!("Can't set {key} from SQL")
    }
}

impl ConfigExtension for RasterLoaderConfig {
    const PREFIX: &'static str = "sedona.raster_loader";
}

impl ConfigField for RasterLoaderConfig {
    fn visit<V: Visit>(&self, v: &mut V, key_prefix: &str, _description: &'static str) {
        let key = if key_prefix.is_empty() {
            "registry".to_string()
        } else {
            format!("{key_prefix}.registry")
        };
        self.registry
            .visit(v, &key, "Registered raster byte loaders");
        let cache_key = if key_prefix.is_empty() {
            "cache".to_string()
        } else {
            format!("{key_prefix}.cache")
        };
        self.cache
            .visit(v, &cache_key, "Session cache of loaded OutDb band bytes");
        let io_budget_key = if key_prefix.is_empty() {
            "io_budget".to_string()
        } else {
            format!("{key_prefix}.io_budget")
        };
        self.io_budget.visit(
            v,
            &io_budget_key,
            "Session budget of blocking raster file operations in flight",
        );
    }

    fn set(&mut self, key: &str, _value: &str) -> DFResult<()> {
        config_err!("Can't set {key} from SQL")
    }
}

impl ExtensionOptions for RasterLoaderConfig {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn cloned(&self) -> Box<dyn ExtensionOptions> {
        Box::new(self.clone())
    }

    fn set(&mut self, key: &str, value: &str) -> DFResult<()> {
        <Self as ConfigField>::set(self, key, value)
    }

    fn entries(&self) -> Vec<ConfigEntry> {
        struct EntryCollector(Vec<ConfigEntry>);
        impl Visit for EntryCollector {
            fn some<V: std::fmt::Display>(
                &mut self,
                key: &str,
                value: V,
                description: &'static str,
            ) {
                self.0.push(ConfigEntry {
                    key: key.to_string(),
                    value: Some(value.to_string()),
                    description,
                });
            }
            fn none(&mut self, key: &str, description: &'static str) {
                self.0.push(ConfigEntry {
                    key: key.to_string(),
                    value: None,
                    description,
                });
            }
        }
        let mut collector = EntryCollector(vec![]);
        self.visit(&mut collector, Self::PREFIX, "");
        collector.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view_entries::ViewEntries;
    use std::sync::Mutex;

    /// Minimal in-test loader: records the request and returns a buffer
    /// of `Π source_shape × byte_size` zeros. Carries a name and the set
    /// of codec names it decodes so registry dispatch can be exercised.
    #[derive(Debug, Default)]
    struct MockLoader {
        name: String,
        codecs: Vec<String>,
        seen: Mutex<Vec<(String, Vec<i64>)>>,
    }

    impl MockLoader {
        fn new(name: &str, codecs: &[&str]) -> Self {
            Self {
                name: name.to_string(),
                codecs: codecs.iter().map(|s| s.to_string()).collect(),
                seen: Mutex::default(),
            }
        }
    }

    #[async_trait::async_trait]
    impl AsyncRasterLoader for MockLoader {
        fn name(&self) -> &str {
            &self.name
        }

        fn supports(&self, codec_names: &[&str]) -> bool {
            !codec_names.is_empty()
                && codec_names
                    .iter()
                    .all(|name| self.codecs.iter().any(|s| s == name))
        }

        async fn load(
            &self,
            req: &[&RasterLoadRequest],
        ) -> Result<Vec<RasterLoadResult>, ArrowError> {
            // This mock loader only supports a single request at a time
            assert_eq!(req.len(), 1);
            self.seen.lock().unwrap().push((
                req[0].single_chunk_uri()?.to_string(),
                req[0].source_shape.to_vec(),
            ));
            let elements: i64 = req[0].source_shape.iter().copied().product();
            let len = elements as usize * req[0].data_type.byte_size();
            Ok(vec![RasterLoadResult::unresolved(
                Buffer::from_vec(vec![0u8; len]),
                req[0],
            )])
        }
    }

    #[test]
    fn registry_starts_empty_and_reports_no_loaders() {
        let r = RasterLoaderRegistry::new();
        assert!(r.is_empty());
        assert!(r.get(&["sedona.gdal"]).is_none());
        assert!(r.loader_names().is_empty());
    }

    #[test]
    fn registry_get_returns_loader_that_supports_the_codecs() {
        let mut r = RasterLoaderRegistry::new();
        r.register(Arc::new(MockLoader::new("mock", &["mock", "bytes"])));
        assert!(!r.is_empty());
        assert!(r.get(&["mock"]).is_some());
        // Every name in the pipeline has to be claimed.
        assert!(r.get(&["bytes", "mock"]).is_some());
        assert!(r.get(&["bytes", "zstd"]).is_none());
        assert!(r.get(&["sedona.gdal"]).is_none());
        assert!(r.get(&[]).is_none());
    }

    #[test]
    fn registry_get_returns_most_recently_registered_matching_loader() {
        let mut r = RasterLoaderRegistry::new();
        let first = Arc::new(MockLoader::new("first", &["mock"]));
        let second = Arc::new(MockLoader::new("second", &["mock"]));
        r.register(first.clone());
        r.register(second.clone());
        // Both claim "mock"; the reverse scan returns the later registration.
        let resolved = r.get(&["mock"]).unwrap();
        assert!(Arc::ptr_eq(
            &(resolved as Arc<dyn AsyncRasterLoader>),
            &(second as Arc<dyn AsyncRasterLoader>)
        ));
    }

    #[test]
    fn registry_catch_all_yields_to_later_specific_loader() {
        // Catch-all registered first (accepts every pipeline); a specific
        // loader registered after wins for the codecs it claims.
        #[derive(Debug)]
        struct CatchAll;
        #[async_trait::async_trait]
        impl AsyncRasterLoader for CatchAll {
            fn name(&self) -> &str {
                "catch-all"
            }
            fn supports(&self, _codec_names: &[&str]) -> bool {
                true
            }
            async fn load(
                &self,
                reqs: &[&RasterLoadRequest],
            ) -> Result<Vec<RasterLoadResult>, ArrowError> {
                Ok(reqs
                    .iter()
                    .map(|req| {
                        RasterLoadResult::unresolved(Buffer::from_vec(Vec::<u8>::new()), req)
                    })
                    .collect())
            }
        }

        let mut r = RasterLoaderRegistry::new();
        r.register(Arc::new(CatchAll));
        r.register(Arc::new(MockLoader::new("zarr", &["zarr"])));
        assert_eq!(r.get(&["zarr"]).unwrap().name(), "zarr");
        assert_eq!(r.get(&["bytes", "zstd"]).unwrap().name(), "catch-all");
        assert_eq!(r.get(&[]).unwrap().name(), "catch-all");
    }

    #[test]
    fn registry_loader_names_lists_registered_most_recent_first() {
        let mut r = RasterLoaderRegistry::new();
        r.register(Arc::new(MockLoader::new("gdal", &[])));
        r.register(Arc::new(MockLoader::new("zarr", &["zarr"])));
        assert_eq!(r.loader_names(), vec!["zarr", "gdal"]);
    }

    #[tokio::test]
    async fn loader_load_returns_buffer_of_expected_size() {
        let loader = MockLoader::default();
        let req = RasterLoadRequest {
            chunks: &[Some(ChunkRef::whole("file:///tmp/foo.tif"))],
            chunk_shape: &[3, 4],
            codecs: r#"[{"name":"mock"}]"#,
            dim_names: &["y", "x"],
            source_shape: &[3, 4],
            view: &ViewEntries::identity_for_shape(&[3, 4]),
            data_type: BandDataType::UInt8,
        };
        let result = loader.load(&[&req]).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].bytes.len(), 12); // 3 × 4 × 1 byte
        let seen = loader.seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].0, "file:///tmp/foo.tif");
        assert_eq!(seen[0].1, vec![3, 4]);
    }

    #[tokio::test]
    async fn loader_load_through_registry_dispatches_to_correct_backend() {
        let mut r = RasterLoaderRegistry::new();
        let gdal = Arc::new(MockLoader::new("gdal", &["gdal"]));
        let zarr = Arc::new(MockLoader::new("zarr", &["zarr"]));
        r.register(gdal.clone());
        r.register(zarr.clone());

        let req = RasterLoadRequest {
            chunks: &[Some(ChunkRef::whole("s3://bucket/cube.zarr/a/c/0/0/0"))],
            chunk_shape: &[2, 3, 4],
            codecs: r#"[{"name":"zarr"}]"#,
            dim_names: &["t", "y", "x"],
            source_shape: &[2, 3, 4],
            view: &ViewEntries::identity_for_shape(&[2, 3, 4]),
            data_type: BandDataType::Float32,
        };
        let loader = r.get(&["zarr"]).unwrap();
        let result = loader.load(&[&req]).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].bytes.len(), 2 * 3 * 4 * 4); // Float32 = 4 bytes

        // Dispatched to zarr, not gdal.
        assert_eq!(zarr.seen.lock().unwrap().len(), 1);
        assert_eq!(gdal.seen.lock().unwrap().len(), 0);
    }

    #[test]
    fn registry_get_missing_codec_returns_none_for_diagnostic_message() {
        let r = RasterLoaderRegistry::new();
        // Caller (RS_EnsureLoaded) sees None and can build a diagnostic
        // listing the registered loaders.
        assert!(r.get(&["nonexistent"]).is_none());
        let err = r.get_or_error(&["nonexistent"]).unwrap_err().to_string();
        assert!(err.contains("nonexistent"), "{err}");
        assert!(err.contains("no raster loaders are registered"), "{err}");
    }

    #[test]
    fn single_chunk_accepts_one_whole_object_only() {
        fn request<'a>(
            chunks: &'a [Option<ChunkRef<'a>>],
            view: &'a ViewEntries,
        ) -> RasterLoadRequest<'a> {
            RasterLoadRequest {
                chunks,
                chunk_shape: &[3, 4],
                codecs: r#"[{"name":"mock"}]"#,
                dim_names: &["y", "x"],
                source_shape: &[3, 4],
                view,
                data_type: BandDataType::UInt8,
            }
        }
        let view = ViewEntries::identity_for_shape(&[3, 4]);

        let whole = [Some(ChunkRef::whole("file:///tmp/foo.tif"))];
        assert_eq!(
            request(&whole, &view).single_chunk_uri().unwrap(),
            "file:///tmp/foo.tif"
        );

        let missing = [None];
        assert!(request(&missing, &view).single_chunk().is_err());

        let range = [Some(ChunkRef::range("file:///tmp/foo.tif", 0, 12))];
        let err = request(&range, &view)
            .single_chunk()
            .unwrap_err()
            .to_string();
        assert!(err.contains("whole objects only"), "{err}");

        let two = [
            Some(ChunkRef::whole("file:///tmp/a.tif")),
            Some(ChunkRef::whole("file:///tmp/b.tif")),
        ];
        let err = request(&two, &view).single_chunk().unwrap_err().to_string();
        assert!(err.contains("got 2 chunks"), "{err}");
    }

    #[test]
    fn io_budget_from_config_applies_the_session_setting() {
        let mut config = ConfigOptions::new();
        assert!(io_budget_from_config(&config).is_none());

        let budget = RasterIoBudget::new(32);
        config.extensions.insert(
            RasterLoaderConfig::from_handle(Arc::new(RwLock::new(RasterLoaderRegistry::new())))
                .with_io_budget(budget.clone()),
        );
        // Without SedonaOptions the budget keeps its size.
        let found = io_budget_from_config(&config).unwrap();
        assert!(found.ptr_eq(&budget));
        assert_eq!(budget.limit(), 32);

        config.extensions.insert(SedonaOptions::default());
        config.set("sedona.raster.io_concurrency", "5").unwrap();
        assert!(io_budget_from_config(&config).unwrap().ptr_eq(&budget));
        assert_eq!(budget.limit(), 5);
        assert_eq!(budget.available(), 5);

        config.set("sedona.raster.io_concurrency", "0").unwrap();
        io_budget_from_config(&config).unwrap();
        assert_eq!(budget.limit(), 1);

        // The budget shows up among the extension's entries.
        let entries = config
            .extensions
            .get::<RasterLoaderConfig>()
            .unwrap()
            .entries();
        assert!(
            entries
                .iter()
                .any(|e| e.key == "sedona.raster_loader.io_budget"
                    && e.value.as_deref() == Some("RasterIoBudget { limit: 1, in_use: 0 }"))
        );
    }
}
