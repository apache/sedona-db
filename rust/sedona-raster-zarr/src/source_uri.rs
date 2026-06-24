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

//! Zarr URI helpers — store resolution and per-chunk OutDb anchors.
//!
//! Two URI shapes flow through the loader:
//!
//! 1. **Group URI** (the loader entry-point argument). Identifies a Zarr
//!    group on a backend. Supported schemes: `file://`, bare local path,
//!    `s3://`, `http://`, `https://` (the `object_store` `aws`/`http`
//!    features). Other schemes (e.g. `gs://`, `az://`) error as
//!    unsupported until their backends are wired in.
//! 2. **Chunk anchor URI** (written into a band's `outdb_uri`). Addresses
//!    one chunk in one array within a group:
//!    `<store-uri>#array=<array-path>&chunk=<i0>,<i1>,...`. The store URI
//!    is the original group URI verbatim (whatever scheme that uses);
//!    array path and chunk indices both live in the fragment so the
//!    store URI is unambiguous even when both contain `/` (e.g.
//!    `s3://bucket/foo.zarr/2024` + `subgroup/B01`).
//!
//!    The "this is a zarr anchor" signal lives in the band's
//!    `outdb_format = "zarr"` field, not in a URI scheme prefix — matches
//!    the GDAL OutDb convention and lets the format-keyed dispatcher
//!    route without parsing URI schemes.

use std::sync::Arc;

use arrow_schema::ArrowError;
use object_store::path::Path as ObjectPath;
use object_store::prefix::PrefixStore;
use object_store::ObjectStore;
use url::Url;
use zarrs::storage::AsyncReadableListableStorage;
use zarrs_object_store::AsyncObjectStore;

/// Parts of a chunk-anchor URI. The async raster byte loader parses
/// `outdb_uri` values back into this struct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkAnchor {
    pub store_uri: String,
    pub array_path: String,
    pub chunk_indices: Vec<u64>,
}

/// Build a chunk anchor URI from its parts.
///
/// Format: `<store_uri>#array=<array_path>&chunk=<i0>,<i1>,...`. The
/// store URI is the original group URI verbatim; array path and chunk
/// indices live in the fragment so the store URI is unambiguous even
/// when path components themselves contain `/`. "This is zarr" lives
/// in the band's `outdb_format` field rather than in a URI scheme.
pub fn build_chunk_anchor(store_uri: &str, array_path: &str, chunk_indices: &[u64]) -> String {
    let indices = chunk_indices
        .iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let array = array_path.trim_start_matches('/');
    format!("{store_uri}#array={array}&chunk={indices}")
}

/// Parse a chunk-anchor URI back into its parts.
///
/// Strict: rejects URIs that don't carry both `array=` and `chunk=`
/// fragment parameters with valid values.
pub fn parse_chunk_anchor(uri: &str) -> Result<ChunkAnchor, ArrowError> {
    let (store_uri, fragment) = uri.split_once('#').ok_or_else(|| {
        ArrowError::InvalidArgumentError(format!(
            "chunk anchor URI missing `#array=...&chunk=...` fragment: {uri}"
        ))
    })?;

    let mut array_path: Option<&str> = None;
    let mut chunk_str: Option<&str> = None;
    for pair in fragment.split('&') {
        if let Some(v) = pair.strip_prefix("array=") {
            array_path = Some(v);
        } else if let Some(v) = pair.strip_prefix("chunk=") {
            chunk_str = Some(v);
        }
        // Unknown fragment params are ignored — forward-compatible for
        // future per-anchor extensions (e.g. version pins).
    }

    let array_path = array_path
        .ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!(
                "chunk anchor URI fragment missing `array=` parameter: {uri}"
            ))
        })?
        .to_string();
    let indices_str = chunk_str.ok_or_else(|| {
        ArrowError::InvalidArgumentError(format!(
            "chunk anchor URI fragment missing `chunk=` parameter: {uri}"
        ))
    })?;
    if indices_str.is_empty() {
        return Err(ArrowError::InvalidArgumentError(format!(
            "chunk anchor URI has empty index list: {uri}"
        )));
    }
    let chunk_indices = indices_str
        .split(',')
        .map(|s| {
            s.parse::<u64>().map_err(|_| {
                ArrowError::InvalidArgumentError(format!(
                    "chunk index `{s}` is not a non-negative integer in {uri}"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(ChunkAnchor {
        store_uri: store_uri.to_string(),
        array_path,
        chunk_indices,
    })
}

/// Resolve a group URI into an async zarrs storage handle rooted at
/// the group, given a credentialed [`object_store::ObjectStore`] for
/// the URI.
///
/// **Contract:** the caller passes a `store` rooted at the URI's
/// scheme + authority — exactly what
/// [`datafusion_execution::object_store::ObjectStoreRegistry::get_store`]
/// returns (e.g. for `s3://bucket/path/foo.zarr` the store is rooted at
/// `s3://bucket`; for `file:///data/foo.zarr` it's a
/// `LocalFileSystem` rooted at `/`). This helper then wraps it in a
/// [`PrefixStore`] at the URL's path so the returned storage is rooted
/// at the zarr group. Bare paths are treated as `file://`.
///
/// The scheme itself is opaque here: any backend whose `store` honours
/// the rooting contract works without a per-scheme branch. Schemes the
/// caller can't produce a store for never reach this function.
///
/// Returns storage rooted at the group: callers invoke
/// `Group::async_open(storage, "/").await`.
/// Build an `Arc<dyn ObjectStore>` for `uri` using env-var credential
/// discovery, per scheme: `file://` / bare paths → `LocalFileSystem`,
/// `s3://` → `AmazonS3Builder`, `http(s)://` → `HttpBuilder`.
///
/// This is the in-process bridge that lets the byte loader (and the Python
/// reader shim) resolve a store from a URI when no host-provided store is
/// available. It is **temporary**: once a credentialed `ObjectStore` can be
/// passed in from DataFusion's `ObjectStoreRegistry` (the obstore FFI; see
/// the loader-FFI follow-up), callers receive the store instead of building
/// it here, and this function — with the `object_store` `aws`/`http`
/// features behind it — goes away.
pub fn object_store_for_uri(uri: &str) -> Result<Arc<dyn ObjectStore>, ArrowError> {
    // file:// and bare paths use a LocalFileSystem rooted at `/`;
    // open_storage_from_uri prefixes it at the group's path.
    if uri.starts_with("file://") || !uri.contains("://") {
        return Ok(Arc::new(object_store::local::LocalFileSystem::new()));
    }
    let url = Url::parse(uri).map_err(|e| {
        ArrowError::InvalidArgumentError(format!("group URI {uri:?} is not a valid URL: {e}"))
    })?;
    let build_err = |backend: &str, e: object_store::Error| {
        ArrowError::ExternalError(Box::new(sedona_common::sedona_internal_datafusion_err!(
            "failed to build {backend} object_store for {uri}: {e}. Provide credentials via \
             standard environment variables (e.g. AWS_ACCESS_KEY_ID/AWS_REGION for s3)."
        )))
    };
    match url.scheme().to_ascii_lowercase().as_str() {
        "s3" => {
            use object_store::aws::AmazonS3Builder;
            let store = AmazonS3Builder::from_env()
                .with_url(uri)
                .build()
                .map_err(|e| build_err("s3", e))?;
            Ok(Arc::new(store))
        }
        "http" | "https" => {
            use object_store::http::HttpBuilder;
            // open_storage_from_uri applies the path as a PrefixStore, so the
            // HttpStore must be rooted at scheme+authority only — unlike S3,
            // HttpBuilder roots at whatever URL it's given.
            let authority = format!("{}://{}", url.scheme(), url.authority());
            let store = HttpBuilder::new()
                .with_url(authority)
                .build()
                .map_err(|e| build_err("http", e))?;
            Ok(Arc::new(store))
        }
        other => Err(ArrowError::NotYetImplemented(format!(
            "unsupported Zarr URI scheme {other:?}; expected one of: file, s3, http, https"
        ))),
    }
}

pub fn open_storage_from_uri(
    uri: &str,
    store: Arc<dyn ObjectStore>,
) -> Result<AsyncReadableListableStorage, ArrowError> {
    // Bare local paths aren't valid URLs; coerce them to `file://` so
    // the path lands in `url.path()` like every other scheme.
    let url = Url::parse(uri)
        .or_else(|_| Url::parse(&format!("file://{uri}")))
        .map_err(|e| {
            ArrowError::InvalidArgumentError(format!("group URI {uri:?} is not a valid URL: {e}"))
        })?;

    let prefix = url.path().trim_start_matches('/').to_string();
    let prefixed: Arc<dyn ObjectStore> = if prefix.is_empty() {
        store
    } else {
        Arc::new(PrefixStore::new(store, ObjectPath::from(prefix)))
    };
    Ok(Arc::new(AsyncObjectStore::new(prefixed)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_anchor_basic() {
        let uri = build_chunk_anchor("file:///tmp/datacube.zarr", "temperature", &[47, 2, 3]);
        assert_eq!(
            uri,
            "file:///tmp/datacube.zarr#array=temperature&chunk=47,2,3"
        );
    }

    #[test]
    fn build_anchor_strips_leading_slash_on_array_path() {
        let uri = build_chunk_anchor("file:///tmp/foo.zarr", "/array", &[0]);
        assert_eq!(uri, "file:///tmp/foo.zarr#array=array&chunk=0");
    }

    #[test]
    fn build_anchor_with_nested_array_path() {
        let uri = build_chunk_anchor("s3://bucket/foo.zarr/2024", "subgroup/B01", &[1, 5]);
        assert_eq!(
            uri,
            "s3://bucket/foo.zarr/2024#array=subgroup/B01&chunk=1,5"
        );
    }

    #[test]
    fn parse_anchor_basic() {
        let anchor =
            parse_chunk_anchor("file:///tmp/datacube.zarr#array=temperature&chunk=47,2,3").unwrap();
        assert_eq!(anchor.store_uri, "file:///tmp/datacube.zarr");
        assert_eq!(anchor.array_path, "temperature");
        assert_eq!(anchor.chunk_indices, vec![47, 2, 3]);
    }

    #[test]
    fn parse_anchor_with_s3_store_and_nested_array() {
        let anchor =
            parse_chunk_anchor("s3://bucket/foo.zarr/2024#array=subgroup/B01&chunk=0,0,0").unwrap();
        assert_eq!(anchor.store_uri, "s3://bucket/foo.zarr/2024");
        assert_eq!(anchor.array_path, "subgroup/B01");
        assert_eq!(anchor.chunk_indices, vec![0, 0, 0]);
    }

    #[test]
    fn parse_anchor_rejects_missing_fragment() {
        let err = parse_chunk_anchor("file:///foo.zarr")
            .unwrap_err()
            .to_string();
        assert!(err.contains("fragment"), "{err}");
    }

    #[test]
    fn parse_anchor_rejects_missing_array_param() {
        let err = parse_chunk_anchor("file:///foo.zarr#chunk=0")
            .unwrap_err()
            .to_string();
        assert!(err.contains("array="), "{err}");
    }

    #[test]
    fn parse_anchor_rejects_missing_chunk_param() {
        let err = parse_chunk_anchor("file:///foo.zarr#array=a")
            .unwrap_err()
            .to_string();
        assert!(err.contains("chunk="), "{err}");
    }

    #[test]
    fn parse_anchor_rejects_empty_indices() {
        let err = parse_chunk_anchor("file:///foo.zarr#array=a&chunk=")
            .unwrap_err()
            .to_string();
        assert!(err.contains("empty"), "{err}");
    }

    #[test]
    fn parse_anchor_rejects_non_integer_index() {
        let err = parse_chunk_anchor("file:///foo.zarr#array=a&chunk=0,abc,2")
            .unwrap_err()
            .to_string();
        assert!(err.contains("abc"), "{err}");
    }

    #[test]
    fn parse_anchor_ignores_unknown_fragment_params() {
        let anchor = parse_chunk_anchor("file:///foo.zarr#array=t&chunk=0,0&version=v3").unwrap();
        assert_eq!(anchor.array_path, "t");
        assert_eq!(anchor.chunk_indices, vec![0, 0]);
    }

    #[test]
    fn anchor_roundtrip() {
        let original = ChunkAnchor {
            store_uri: "file:///tmp/foo.zarr".into(),
            array_path: "subgroup/B01".into(),
            chunk_indices: vec![10, 20, 30],
        };
        let uri = build_chunk_anchor(
            &original.store_uri,
            &original.array_path,
            &original.chunk_indices,
        );
        let parsed = parse_chunk_anchor(&uri).unwrap();
        assert_eq!(parsed, original);
    }

    fn placeholder_store() -> Arc<dyn ObjectStore> {
        Arc::new(object_store::memory::InMemory::new())
    }

    #[test]
    fn open_storage_wraps_cloud_store_with_prefix() {
        // An InMemory store stands in for a bucket-rooted cloud store;
        // the s3:// path component should be applied as a PrefixStore.
        let storage = open_storage_from_uri("s3://bucket/path/foo.zarr", placeholder_store())
            .expect("cloud URI builds storage from a passed-in store");
        assert!(Arc::strong_count(&storage) >= 1);
    }

    #[test]
    fn open_storage_accepts_file_scheme() {
        let storage = open_storage_from_uri("file:///nonexistent/foo.zarr", placeholder_store())
            .expect("file:// URI builds storage");
        assert!(Arc::strong_count(&storage) >= 1);
    }

    #[test]
    fn open_storage_accepts_bare_path() {
        // Bare paths are coerced to file:// rather than rejected.
        let storage = open_storage_from_uri("/nonexistent/foo.zarr", placeholder_store())
            .expect("bare path builds storage");
        assert!(Arc::strong_count(&storage) >= 1);
    }
}
