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
//! `outdb_format` values they handle — and register themselves against an
//! [`RasterLoaderRegistry`]. The
//! `RS_EnsureLoaded` UDF in the `sedona` crate consumes the registry to
//! materialise OutDb bands at query time; band accessors
//! (`BandRef::nd_buffer()` / `contiguous_data()`) do **not** invoke the
//! loader transparently — they return whatever is in the `data` column
//! verbatim, surfacing a clear error when the column is empty.

use std::sync::{Arc, RwLock};

use arrow_buffer::Buffer;
use arrow_schema::ArrowError;
use datafusion_common::config::{
    ConfigEntry, ConfigExtension, ConfigField, ExtensionOptions, Visit,
};
use datafusion_common::{config_err, Result as DFResult};
use sedona_schema::raster::BandDataType;

use crate::view_entries::ViewEntry;

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
    /// Anchor URI from the band's `outdb_uri` column. Bare paths and
    /// scheme'd URIs both allowed; backend is responsible for parsing.
    pub uri: &'a str,
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
    pub view: &'a [ViewEntry],
    /// Pixel type the band claims. The loader returns bytes encoding this
    /// type and errors if the source disagrees (e.g. file's dtype differs).
    pub data_type: BandDataType,
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
    pub view: Vec<ViewEntry>,
}

impl RasterLoadResult {
    /// Full source, view left unresolved — echoes the request. The default
    /// a loader that doesn't prune I/O returns.
    pub fn unresolved(bytes: Buffer, req: &RasterLoadRequest<'_>) -> Self {
        Self {
            bytes,
            source_shape: req.source_shape.to_vec(),
            view: req.view.to_vec(),
        }
    }
}

/// Backend trait. Implementers live in format-specific crates
/// (`sedona-raster-gdal`, `sedona-raster-zarr`, …) and are registered
/// against an [`RasterLoaderRegistry`]; each declares the band
/// `outdb_format` values it handles via [`AsyncRasterLoader::supports_format`].
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

    /// Whether this loader will attempt a band with the given
    /// `outdb_format`. `format` is the band's `outdb_format` column value —
    /// `None` when unset (as `RS_FromPath` emits). A catch-all backend
    /// (e.g. GDAL) returns `true` for any input; a format-specific backend
    /// (e.g. Zarr) returns `true` only for its own format.
    fn supports_format(&self, format: Option<&str>) -> bool;

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
/// Loaders are held in registration order. A band's `outdb_format` is
/// matched against each loader's [`AsyncRasterLoader::supports_format`],
/// scanning **most-recently-registered first** — so a format-specific
/// loader registered after a catch-all (GDAL) wins for the formats it
/// claims, while everything else falls through to the catch-all.
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

    /// Register a loader. Each loader declares the formats it handles via
    /// [`AsyncRasterLoader::supports_format`]; lookup scans in reverse, so a
    /// loader registered later takes precedence over an earlier one for any
    /// format both accept. The catch-all (GDAL) is registered first so
    /// specific backends registered afterward win for their own formats.
    pub fn register(&mut self, loader: Arc<dyn AsyncRasterLoader>) {
        self.loaders.push(loader);
    }

    /// Find the loader for a band `outdb_format` (`None` when unset, as
    /// `RS_FromPath` emits). Scans most-recently-registered first and
    /// returns the first loader whose [`AsyncRasterLoader::supports_format`]
    /// accepts it, or `None` when nothing does.
    pub fn get(&self, format: Option<&str>) -> Option<Arc<dyn AsyncRasterLoader>> {
        self.loaders
            .iter()
            .rev()
            .find(|l| l.supports_format(format))
            .cloned()
    }

    /// Like [`Self::get`], but returns a planner error naming the format and
    /// the registered loaders when nothing accepts it. `RS_EnsureLoaded`
    /// dispatches through this so the diagnostic points users at the
    /// install/register step.
    pub fn get_or_error(
        &self,
        format: Option<&str>,
    ) -> datafusion_common::Result<Arc<dyn AsyncRasterLoader>> {
        if let Some(loader) = self.get(format) {
            return Ok(loader);
        }
        let registered = self.loader_names();
        let registered_msg = if registered.is_empty() {
            "no raster loaders are registered in this session".to_string()
        } else {
            format!("registered loaders: {}", registered.join(", "))
        };
        datafusion_common::plan_err!(
            "no raster loader accepts raster format {format:?} — {registered_msg}"
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
/// the session's `ConfigOptions` at `SedonaContext::new_from_context`
/// time; consumed by the `RS_EnsureLoaded` async UDF at dispatch time
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
}

impl RasterLoaderConfig {
    /// Build a config extension that closes over an existing shared
    /// registry handle. Use this rather than `default()` when wiring
    /// from `SedonaContext::new_from_context` so the context's mutable
    /// `register_raster_loader` API writes to the same `RwLock` the
    /// config extension exposes for reads.
    pub fn from_handle(registry: Arc<RwLock<RasterLoaderRegistry>>) -> Self {
        Self {
            registry: RasterLoaderRegistryOption::new(registry),
        }
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
    use std::sync::Mutex;

    /// Minimal in-test loader: records the request and returns a buffer
    /// of `Π source_shape × byte_size` zeros. Carries a name and the set
    /// of formats it claims so registry dispatch can be exercised.
    #[derive(Debug, Default)]
    struct MockLoader {
        name: String,
        formats: Vec<String>,
        seen: Mutex<Vec<(String, Vec<i64>)>>,
    }

    impl MockLoader {
        fn new(name: &str, formats: &[&str]) -> Self {
            Self {
                name: name.to_string(),
                formats: formats.iter().map(|s| s.to_string()).collect(),
                seen: Mutex::default(),
            }
        }
    }

    #[async_trait::async_trait]
    impl AsyncRasterLoader for MockLoader {
        fn name(&self) -> &str {
            &self.name
        }

        fn supports_format(&self, format: Option<&str>) -> bool {
            matches!(format, Some(f) if self.formats.iter().any(|s| s == f))
        }

        async fn load(
            &self,
            req: &[&RasterLoadRequest],
        ) -> Result<Vec<RasterLoadResult>, ArrowError> {
            // This mock loader only supports a single request at a time
            assert_eq!(req.len(), 1);
            self.seen
                .lock()
                .unwrap()
                .push((req[0].uri.to_string(), req[0].source_shape.to_vec()));
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
        assert!(r.get(Some("gdal")).is_none());
        assert!(r.loader_names().is_empty());
    }

    #[test]
    fn registry_get_returns_loader_that_supports_the_format() {
        let mut r = RasterLoaderRegistry::new();
        r.register(Arc::new(MockLoader::new("mock", &["mock"])));
        assert!(!r.is_empty());
        assert!(r.get(Some("mock")).is_some());
        assert!(r.get(Some("gdal")).is_none());
        assert!(r.get(None).is_none());
    }

    #[test]
    fn registry_get_returns_most_recently_registered_matching_loader() {
        let mut r = RasterLoaderRegistry::new();
        let first = Arc::new(MockLoader::new("first", &["mock"]));
        let second = Arc::new(MockLoader::new("second", &["mock"]));
        r.register(first.clone());
        r.register(second.clone());
        // Both claim "mock"; the reverse scan returns the later registration.
        let resolved = r.get(Some("mock")).unwrap();
        assert!(Arc::ptr_eq(
            &(resolved as Arc<dyn AsyncRasterLoader>),
            &(second as Arc<dyn AsyncRasterLoader>)
        ));
    }

    #[test]
    fn registry_catch_all_yields_to_later_specific_loader() {
        // Catch-all registered first (accepts everything, including None);
        // a specific loader registered after wins for its own format.
        #[derive(Debug)]
        struct CatchAll;
        #[async_trait::async_trait]
        impl AsyncRasterLoader for CatchAll {
            fn name(&self) -> &str {
                "catch-all"
            }
            fn supports_format(&self, _format: Option<&str>) -> bool {
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
        assert_eq!(r.get(Some("zarr")).unwrap().name(), "zarr");
        assert_eq!(r.get(Some("geotiff")).unwrap().name(), "catch-all");
        assert_eq!(r.get(None).unwrap().name(), "catch-all");
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
            uri: "file:///tmp/foo.tif",
            dim_names: &["y", "x"],
            source_shape: &[3, 4],
            view: &[],
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
            uri: "s3://bucket/cube.zarr",
            dim_names: &["t", "y", "x"],
            source_shape: &[2, 3, 4],
            view: &[],
            data_type: BandDataType::Float32,
        };
        let loader = r.get(Some("zarr")).unwrap();
        let result = loader.load(&[&req]).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].bytes.len(), 2 * 3 * 4 * 4); // Float32 = 4 bytes

        // Dispatched to zarr, not gdal.
        assert_eq!(zarr.seen.lock().unwrap().len(), 1);
        assert_eq!(gdal.seen.lock().unwrap().len(), 0);
    }

    #[test]
    fn registry_get_missing_format_returns_none_for_diagnostic_message() {
        let r = RasterLoaderRegistry::new();
        // Caller (RS_EnsureLoaded) sees None and can build a diagnostic
        // listing the registered loaders.
        assert!(r.get(Some("nonexistent")).is_none());
    }
}
