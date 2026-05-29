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
//! other backend. Backends implement [`AsyncByteLoader`] and register
//! themselves with a format key against an [`OutDbLoaderRegistry`]. The
//! `RS_EnsureLoaded` UDF in the `sedona` crate consumes the registry to
//! materialise OutDb bands at query time; band accessors
//! (`BandRef::nd_buffer()` / `contiguous_data()`) do **not** invoke the
//! loader transparently — they return whatever is in the `data` column
//! verbatim, surfacing a clear error when the column is empty.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use arrow_buffer::Buffer;
use arrow_schema::ArrowError;
use datafusion_common::config::{
    ConfigEntry, ConfigExtension, ConfigField, ExtensionOptions, Visit,
};
use datafusion_common::{config_err, Result as DFResult};
use sedona_schema::raster::BandDataType;

/// Everything a backend needs to materialise a single OutDb band's bytes.
///
/// Constructed by `RS_EnsureLoaded` once per row from the band's schema
/// metadata. The lifetime is the request's, not the loader's — borrowed
/// fields point into the input `RecordBatch` and stay valid for the
/// duration of the [`AsyncByteLoader::load`] future.
#[derive(Debug, Clone, Copy)]
pub struct OutDbLoadRequest<'a> {
    /// Anchor URI from the band's `outdb_uri` column. Bare paths and
    /// scheme'd URIs both allowed; backend is responsible for parsing.
    pub uri: &'a str,
    /// Per-axis names parallel to `source_shape`. Backends use this to
    /// map their native axis order onto the band's.
    pub dim_names: &'a [&'a str],
    /// Raw source shape in `dim_names` order. The loader returns a
    /// `Buffer` whose length equals `Π source_shape × data_type.byte_size()`
    /// bytes, encoding pixels in C-order over `dim_names`.
    pub source_shape: &'a [u64],
    /// Pixel type the band claims. The loader returns bytes encoding this
    /// type and errors if the source disagrees (e.g. file's dtype differs).
    pub data_type: BandDataType,
}

/// Backend trait. Implementers live in format-specific crates
/// (`sedona-raster-gdal`, `sedona-raster-zarr`, …) and are registered
/// against an [`OutDbLoaderRegistry`] under a format key matching the
/// band's `outdb_format` column.
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
pub trait AsyncByteLoader: Send + Sync {
    /// Fetch the band's bytes. The returned `Buffer` must contain exactly
    /// `Π source_shape × data_type.byte_size()` bytes in C-order over
    /// `dim_names`. Errors propagate to the caller of `RS_EnsureLoaded`.
    async fn load(&self, req: &OutDbLoadRequest<'_>) -> Result<Buffer, ArrowError>;
}

/// Process-side registry mapping `outdb_format` keys to loader instances.
///
/// One registry instance per `SedonaContext`. The owning context wraps it
/// in `Arc<RwLock<…>>` so plugin crates (`sedona-raster-zarr`, future COG /
/// Icechunk / …) can register their loaders post-construction via a
/// public `SedonaContext::register_outdb_loader` API.
#[derive(Default)]
pub struct OutDbLoaderRegistry {
    loaders: HashMap<String, Arc<dyn AsyncByteLoader>>,
}

impl std::fmt::Debug for OutDbLoaderRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Show registered format keys but hide the opaque loader impls
        // (which don't carry a meaningful Debug surface).
        let mut keys: Vec<&str> = self.loaders.keys().map(String::as_str).collect();
        keys.sort();
        f.debug_struct("OutDbLoaderRegistry")
            .field("formats", &keys)
            .finish()
    }
}

impl OutDbLoaderRegistry {
    /// Construct an empty registry. Compiled-in backends (`sedona-raster-gdal`
    /// under the `gdal` feature) register themselves from `SedonaContext::new`;
    /// plugin backends register via `SedonaContext::register_outdb_loader`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a loader under a format key. Later registrations for the
    /// same key overwrite — registries are mutable for the lifetime of
    /// the session and there's no value in locking down after first
    /// registration (a process running plugins may legitimately swap
    /// implementations during setup).
    pub fn register(&mut self, format: impl Into<String>, loader: Arc<dyn AsyncByteLoader>) {
        self.loaders.insert(format.into(), loader);
    }

    /// Look up a loader by format key. Returns `None` for keys with no
    /// registered backend. `RS_EnsureLoaded` surfaces the `None` case as a
    /// query-time error that names the missing format and points users at
    /// the install/register step.
    pub fn get(&self, format: &str) -> Option<Arc<dyn AsyncByteLoader>> {
        self.loaders.get(format).cloned()
    }

    /// Iterate registered format keys. Useful for diagnostics ("no loader
    /// for 'zarr'; registered formats are: gdal").
    pub fn formats(&self) -> impl Iterator<Item = &str> {
        self.loaders.keys().map(String::as_str)
    }

    /// True if any loader is registered.
    pub fn is_empty(&self) -> bool {
        self.loaders.is_empty()
    }
}

/// `ConfigField`-shaped wrapper around the shared registry handle.
///
/// Mirrors `sedona_common::option::CrsProviderOption` so the registry
/// can live inside a `ConfigOptions` extension and stay reachable from
/// `AsyncScalarUDFImpl::invoke_async_with_args` (which only receives
/// `Arc<ConfigOptions>`). The inner `Arc<RwLock<...>>` is cloned
/// between the `SedonaContext` (mutable register API) and the config
/// extension (read at UDF dispatch time); both observe the same
/// underlying lock.
#[derive(Debug, Clone)]
pub struct OutDbLoaderRegistryOption(Arc<RwLock<OutDbLoaderRegistry>>);

impl OutDbLoaderRegistryOption {
    /// Wrap an existing shared registry handle.
    pub fn new(inner: Arc<RwLock<OutDbLoaderRegistry>>) -> Self {
        Self(inner)
    }

    /// Clone the inner Arc for callers that need their own owning handle
    /// (e.g. `SedonaContext::register_outdb_loader` needs to write
    /// through the same lock that the config extension exposes for
    /// reads).
    pub fn handle(&self) -> Arc<RwLock<OutDbLoaderRegistry>> {
        Arc::clone(&self.0)
    }
}

impl Default for OutDbLoaderRegistryOption {
    fn default() -> Self {
        Self(Arc::new(RwLock::new(OutDbLoaderRegistry::new())))
    }
}

impl PartialEq for OutDbLoaderRegistryOption {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl ConfigField for OutDbLoaderRegistryOption {
    fn visit<V: Visit>(&self, v: &mut V, key: &str, description: &'static str) {
        let snapshot = match self.0.read() {
            Ok(g) => g.formats().map(String::from).collect::<Vec<_>>(),
            Err(p) => p
                .into_inner()
                .formats()
                .map(String::from)
                .collect::<Vec<_>>(),
        };
        v.some(
            key,
            format!("OutDbLoaderRegistry {{ formats: {snapshot:?} }}"),
            description,
        );
    }

    fn set(&mut self, key: &str, _value: &str) -> DFResult<()> {
        config_err!("Can't set {key} from SQL")
    }
}

/// `ConfigExtension` that stashes the per-session
/// [`OutDbLoaderRegistry`] inside a `ConfigOptions`. Registered into
/// the session's `ConfigOptions` at `SedonaContext::new_from_context`
/// time; consumed by the `RS_EnsureLoaded` async UDF at dispatch time
/// via `args.config_options.extensions.get::<OutDbLoaderConfig>()`.
///
/// The PREFIX namespace is `sedona.outdb_loader` — kept separate from
/// `sedona`'s main `SedonaOptions` extension because this lives in
/// `sedona-raster` (which is upstream of `sedona-common` in the
/// dependency graph) and adding a field to `SedonaOptions` would
/// require an undesirable circular dep.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct OutDbLoaderConfig {
    pub registry: OutDbLoaderRegistryOption,
}

impl OutDbLoaderConfig {
    /// Build a config extension that closes over an existing shared
    /// registry handle. Use this rather than `default()` when wiring
    /// from `SedonaContext::new_from_context` so the context's mutable
    /// `register_outdb_loader` API writes to the same `RwLock` the
    /// config extension exposes for reads.
    pub fn from_handle(registry: Arc<RwLock<OutDbLoaderRegistry>>) -> Self {
        Self {
            registry: OutDbLoaderRegistryOption::new(registry),
        }
    }
}

impl ConfigExtension for OutDbLoaderConfig {
    const PREFIX: &'static str = "sedona.outdb_loader";
}

impl ConfigField for OutDbLoaderConfig {
    fn visit<V: Visit>(&self, v: &mut V, key_prefix: &str, _description: &'static str) {
        let key = if key_prefix.is_empty() {
            "registry".to_string()
        } else {
            format!("{key_prefix}.registry")
        };
        self.registry
            .visit(v, &key, "Registered OutDb byte loaders");
    }

    fn set(&mut self, key: &str, _value: &str) -> DFResult<()> {
        config_err!("Can't set {key} from SQL")
    }
}

impl ExtensionOptions for OutDbLoaderConfig {
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
    /// of `Π source_shape × byte_size` zeros.
    #[derive(Default)]
    struct MockLoader {
        seen: Mutex<Vec<(String, Vec<u64>)>>,
    }

    #[async_trait::async_trait]
    impl AsyncByteLoader for MockLoader {
        async fn load(&self, req: &OutDbLoadRequest<'_>) -> Result<Buffer, ArrowError> {
            self.seen
                .lock()
                .unwrap()
                .push((req.uri.to_string(), req.source_shape.to_vec()));
            let elements: u64 = req.source_shape.iter().copied().product();
            let len = elements as usize * req.data_type.byte_size();
            Ok(Buffer::from_vec(vec![0u8; len]))
        }
    }

    #[test]
    fn registry_starts_empty_and_reports_no_formats() {
        let r = OutDbLoaderRegistry::new();
        assert!(r.is_empty());
        assert!(r.get("gdal").is_none());
        assert_eq!(r.formats().count(), 0);
    }

    #[test]
    fn registry_get_returns_registered_loader() {
        let mut r = OutDbLoaderRegistry::new();
        r.register("mock", Arc::new(MockLoader::default()));
        assert!(!r.is_empty());
        assert!(r.get("mock").is_some());
        assert!(r.get("gdal").is_none());
    }

    #[test]
    fn registry_register_overwrites_existing_key() {
        let mut r = OutDbLoaderRegistry::new();
        let first = Arc::new(MockLoader::default());
        let second = Arc::new(MockLoader::default());
        r.register("mock", first.clone());
        r.register("mock", second.clone());
        // Two distinct loaders pushed under the same key; the second wins.
        let resolved = r.get("mock").unwrap();
        assert!(Arc::ptr_eq(
            &(resolved as Arc<dyn AsyncByteLoader>),
            &(second as Arc<dyn AsyncByteLoader>)
        ));
    }

    #[test]
    fn registry_formats_lists_registered_keys() {
        let mut r = OutDbLoaderRegistry::new();
        r.register("gdal", Arc::new(MockLoader::default()));
        r.register("zarr", Arc::new(MockLoader::default()));
        let mut formats: Vec<&str> = r.formats().collect();
        formats.sort();
        assert_eq!(formats, vec!["gdal", "zarr"]);
    }

    #[tokio::test]
    async fn loader_load_returns_buffer_of_expected_size() {
        let loader = MockLoader::default();
        let req = OutDbLoadRequest {
            uri: "file:///tmp/foo.tif",
            dim_names: &["y", "x"],
            source_shape: &[3, 4],
            data_type: BandDataType::UInt8,
        };
        let buf = loader.load(&req).await.unwrap();
        assert_eq!(buf.len(), 12); // 3 × 4 × 1 byte
        let seen = loader.seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].0, "file:///tmp/foo.tif");
        assert_eq!(seen[0].1, vec![3, 4]);
    }

    #[tokio::test]
    async fn loader_load_through_registry_dispatches_to_correct_backend() {
        let mut r = OutDbLoaderRegistry::new();
        let gdal = Arc::new(MockLoader::default());
        let zarr = Arc::new(MockLoader::default());
        r.register("gdal", gdal.clone());
        r.register("zarr", zarr.clone());

        let req = OutDbLoadRequest {
            uri: "s3://bucket/cube.zarr",
            dim_names: &["t", "y", "x"],
            source_shape: &[2, 3, 4],
            data_type: BandDataType::Float32,
        };
        let loader = r.get("zarr").unwrap();
        let buf = loader.load(&req).await.unwrap();
        assert_eq!(buf.len(), 2 * 3 * 4 * 4); // Float32 = 4 bytes

        // Dispatched to zarr, not gdal.
        assert_eq!(zarr.seen.lock().unwrap().len(), 1);
        assert_eq!(gdal.seen.lock().unwrap().len(), 0);
    }

    #[test]
    fn registry_get_missing_format_returns_none_for_diagnostic_message() {
        let r = OutDbLoaderRegistry::new();
        // Caller (RS_EnsureLoaded) sees None and can build a diagnostic
        // listing the registered formats.
        assert!(r.get("nonexistent").is_none());
    }
}
