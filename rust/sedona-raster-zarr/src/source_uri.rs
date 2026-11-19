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

//! Zarr URI helpers — store resolution and the transitional per-chunk codec.
//!
//! Two kinds of reference flow through the loader:
//!
//! 1. **Group URI** (the loader entry-point argument). Identifies a Zarr
//!    group on a backend. Supported schemes: `file://`, bare local path,
//!    `s3://`, `http://`, `https://` (the `object_store` `aws`/`http`
//!    features). Other schemes (e.g. `gs://`, `az://`) error as
//!    unsupported until their backends are wired in.
//! 2. **Chunk reference** (a band's one `chunks` entry plus its `codecs`).
//!    The chunk's `uri` is the chunk object's URL, `<group-uri>/<store
//!    key>` ([`chunk_object_uri`]); for an array stored with the
//!    `sharding_indexed` codec it is the shard object, the object actually
//!    read. The band's `codecs` is the one-entry [`ZARR_CODEC`] pipeline
//!    whose configuration names the group (`store`, the group URI
//!    verbatim), the `array` path and the `chunk` indices — on the
//!    inner-chunk grid for a sharded array — which is what the byte loader
//!    opens and reads. The store URI stays unambiguous even when it and
//!    the array path both contain `/` (e.g. `s3://bucket/foo.zarr/2024` +
//!    `subgroup/B01`), because they are separate configuration fields.
//!
//!    This codec is transitional: it keeps today's zarrs-based loader
//!    working behind the codec dispatch. The generic chunk loader replaces
//!    it with real Zarr v3 pipelines per chunk object, at which point the
//!    `uri` alone is what gets read and this codec goes away.

use std::sync::Arc;

use arrow_schema::ArrowError;
use object_store::ObjectStore;
use object_store::path::Path as ObjectPath;
use object_store::prefix::PrefixStore;
use sedona_raster::chunks::{CodecPipeline, CodecSpec};
use serde_json::{Map, Value, json};
use url::Url;
use zarrs::storage::AsyncReadableListableStorage;
use zarrs_object_store::AsyncObjectStore;

/// Name of the transitional container codec that reads one Zarr chunk
/// through zarrs. Configuration: `{"store": …, "array": …, "chunk": [...]}`,
/// see [`ZarrCodec`]. Loaders dispatch on this name.
pub const ZARR_CODEC: &str = "sedona.zarr";

/// Configuration of the [`ZARR_CODEC`] codec: which chunk of which array
/// in which group a band reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZarrCodec {
    /// The group URI verbatim, as the reader was opened with.
    pub store: String,
    /// The array path within the group, without a leading slash.
    pub array: String,
    /// Chunk indices on the grid the reader walks: the inner-chunk grid
    /// of a sharded array, the chunk grid of any other.
    pub chunk: Vec<u64>,
}

impl ZarrCodec {
    pub fn new(store: &str, array: &str, chunk: &[u64]) -> Self {
        Self {
            store: store.to_string(),
            array: array.trim_start_matches('/').to_string(),
            chunk: chunk.to_vec(),
        }
    }

    /// The array path with its leading slash, as `Array::async_open` takes it.
    pub fn array_path(&self) -> String {
        format!("/{}", self.array)
    }

    /// This codec as a one-entry pipeline.
    pub fn to_pipeline(&self) -> CodecPipeline {
        let mut configuration = Map::new();
        configuration.insert("store".to_string(), Value::String(self.store.clone()));
        configuration.insert("array".to_string(), Value::String(self.array.clone()));
        configuration.insert("chunk".to_string(), json!(self.chunk));
        CodecPipeline::new(vec![CodecSpec::new(ZARR_CODEC, Some(configuration))])
    }

    /// The JSON a band stores in `codecs` for this codec.
    pub fn to_json(&self) -> String {
        self.to_pipeline().to_json()
    }

    /// Read this codec out of a pipeline. `Ok(None)` when the pipeline is
    /// not a single [`ZARR_CODEC`]; an error when it is but its
    /// configuration is malformed.
    pub fn from_pipeline(pipeline: &CodecPipeline) -> Result<Option<Self>, ArrowError> {
        let Some(codec) = pipeline.single() else {
            return Ok(None);
        };
        if codec.name != ZARR_CODEC {
            return Ok(None);
        }
        let configuration = codec.configuration.as_ref().ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!(
                "{ZARR_CODEC} needs a configuration with \"store\", \"array\" and \"chunk\""
            ))
        })?;
        let string = |key: &str| -> Result<String, ArrowError> {
            match configuration.get(key) {
                Some(Value::String(s)) if !s.is_empty() => Ok(s.clone()),
                _ => Err(ArrowError::InvalidArgumentError(format!(
                    "{ZARR_CODEC}: \"{key}\" must be a non-empty string"
                ))),
            }
        };
        let store = string("store")?;
        let array = string("array")?.trim_start_matches('/').to_string();
        let chunk = match configuration.get("chunk") {
            Some(Value::Array(items)) if !items.is_empty() => items
                .iter()
                .map(|item| {
                    item.as_u64().ok_or_else(|| {
                        ArrowError::InvalidArgumentError(format!(
                            "{ZARR_CODEC}: chunk index {item} is not a non-negative integer"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
            _ => {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "{ZARR_CODEC}: \"chunk\" must be a non-empty array of chunk indices"
                )));
            }
        };
        Ok(Some(Self {
            store,
            array,
            chunk,
        }))
    }

    /// [`Self::from_pipeline`] on a band's `codecs` JSON.
    pub fn parse(codecs: &str) -> Result<Option<Self>, ArrowError> {
        let pipeline = CodecPipeline::parse(codecs)
            .map_err(|e| ArrowError::InvalidArgumentError(e.to_string()))?;
        Self::from_pipeline(&pipeline)
    }

    /// Parse a band's `codecs`, failing when they are not this codec.
    pub fn parse_required(codecs: &str) -> Result<Self, ArrowError> {
        Self::parse(codecs)?.ok_or_else(|| {
            ArrowError::InvalidArgumentError(format!(
                "band codecs {codecs} are not the {ZARR_CODEC} codec; the Zarr loader reads \
                 only bands written by the Zarr reader"
            ))
        })
    }
}

/// The URL of the object holding `store_key` (an array's chunk or shard
/// key, as `Array::chunk_key` returns it) in the group at `group_uri`:
/// `<group_uri>/<store_key>`.
pub fn chunk_object_uri(group_uri: &str, store_key: &str) -> String {
    format!(
        "{}/{}",
        group_uri.trim_end_matches('/'),
        store_key.trim_start_matches('/')
    )
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
            use object_store::ClientOptions;
            use object_store::http::HttpBuilder;
            // open_storage_from_uri applies the path as a PrefixStore, so the
            // HttpStore must be rooted at scheme+authority only — unlike S3,
            // HttpBuilder roots at whatever URL it's given.
            let authority = format!("{}://{}", url.scheme(), url.authority());
            // object_store refuses plain `http://` unless asked; an explicit
            // `http://` URI is the caller asking (local servers, test
            // fixtures, private networks).
            let options =
                ClientOptions::new().with_allow_http(url.scheme().eq_ignore_ascii_case("http"));
            let store = HttpBuilder::new()
                .with_url(authority)
                .with_client_options(options)
                .build()
                .map_err(|e| build_err("http", e))?;
            Ok(Arc::new(store))
        }
        other => Err(ArrowError::NotYetImplemented(format!(
            "unsupported Zarr URI scheme {other:?}; expected one of: file, s3, http, https"
        ))),
    }
}

/// Identity of the `ObjectStore` client [`object_store_for_uri`] builds for
/// `uri`: its scheme and authority. The client never depends on the path
/// ([`open_storage_from_uri`] applies that as a prefix), so every group in
/// one bucket or on one host shares a client and its connection pool.
pub fn store_client_key(uri: &str) -> Result<String, ArrowError> {
    if uri.starts_with("file://") || !uri.contains("://") {
        return Ok("file://".to_string());
    }
    let url = Url::parse(uri).map_err(|e| {
        ArrowError::InvalidArgumentError(format!("group URI {uri:?} is not a valid URL: {e}"))
    })?;
    Ok(format!(
        "{}://{}",
        url.scheme().to_ascii_lowercase(),
        url.authority()
    ))
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
    fn zarr_codec_round_trips_through_json() {
        let codec = ZarrCodec::new("file:///tmp/datacube.zarr", "temperature", &[47, 2, 3]);
        let json = codec.to_json();
        // Compared as values: serde_json's key order depends on whether the
        // build unifies its `preserve_order` feature.
        let value: Value = serde_json::from_str(&json).unwrap();
        assert_eq!(
            value,
            json!([{
                "name": "sedona.zarr",
                "configuration": {
                    "store": "file:///tmp/datacube.zarr",
                    "array": "temperature",
                    "chunk": [47, 2, 3],
                },
            }])
        );
        assert_eq!(ZarrCodec::parse(&json).unwrap(), Some(codec));
    }

    #[test]
    fn zarr_codec_strips_leading_slash_on_array_path() {
        let codec = ZarrCodec::new("file:///tmp/foo.zarr", "/array", &[0]);
        assert_eq!(codec.array, "array");
        assert_eq!(codec.array_path(), "/array");
        // A leading slash in stored JSON is tolerated on read too.
        let parsed = ZarrCodec::parse(
            r#"[{"name":"sedona.zarr","configuration":{"store":"s3://b/x.zarr","array":"/sub/B01","chunk":[1,5]}}]"#,
        )
        .unwrap()
        .unwrap();
        assert_eq!(parsed.array, "sub/B01");
        assert_eq!(parsed.store, "s3://b/x.zarr");
        assert_eq!(parsed.chunk, vec![1, 5]);
    }

    #[test]
    fn zarr_codec_is_none_for_other_pipelines() {
        assert_eq!(ZarrCodec::parse(r#"[{"name":"bytes"}]"#).unwrap(), None);
        assert_eq!(
            ZarrCodec::parse(r#"[{"name":"sedona.gdal","configuration":{"band":1}}]"#).unwrap(),
            None
        );
        // Two codecs are never this container codec, whatever their names.
        assert_eq!(
            ZarrCodec::parse(r#"[{"name":"sedona.zarr"},{"name":"zstd"}]"#).unwrap(),
            None
        );
        let err = ZarrCodec::parse_required(r#"[{"name":"bytes"}]"#)
            .unwrap_err()
            .to_string();
        assert!(err.contains("sedona.zarr"), "{err}");
    }

    #[test]
    fn zarr_codec_rejects_malformed_configuration() {
        for bad in [
            r#"[{"name":"sedona.zarr"}]"#,
            r#"[{"name":"sedona.zarr","configuration":{"array":"a","chunk":[0]}}]"#,
            r#"[{"name":"sedona.zarr","configuration":{"store":"s3://b","chunk":[0]}}]"#,
            r#"[{"name":"sedona.zarr","configuration":{"store":"s3://b","array":"a"}}]"#,
            r#"[{"name":"sedona.zarr","configuration":{"store":"s3://b","array":"a","chunk":[]}}]"#,
            r#"[{"name":"sedona.zarr","configuration":{"store":"s3://b","array":"a","chunk":[0,-1]}}]"#,
            r#"[{"name":"sedona.zarr","configuration":{"store":"s3://b","array":"a","chunk":"0,1"}}]"#,
            "not json",
        ] {
            assert!(ZarrCodec::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn chunk_object_uri_joins_group_and_store_key() {
        assert_eq!(
            chunk_object_uri("file:///tmp/foo.zarr", "temperature/c/0/0"),
            "file:///tmp/foo.zarr/temperature/c/0/0"
        );
        assert_eq!(
            chunk_object_uri("s3://bucket/foo.zarr/2024/", "/subgroup/B01/c/1/5"),
            "s3://bucket/foo.zarr/2024/subgroup/B01/c/1/5"
        );
        assert_eq!(
            chunk_object_uri("/tmp/foo.zarr", "a/0.0"),
            "/tmp/foo.zarr/a/0.0"
        );
    }

    #[test]
    fn store_client_key_is_scheme_and_authority() {
        assert_eq!(store_client_key("file:///tmp/a.zarr").unwrap(), "file://");
        assert_eq!(store_client_key("/tmp/a.zarr").unwrap(), "file://");
        assert_eq!(
            store_client_key("s3://bucket/a/b.zarr").unwrap(),
            "s3://bucket"
        );
        assert_eq!(
            store_client_key("s3://bucket/other.zarr").unwrap(),
            "s3://bucket"
        );
        assert_eq!(
            store_client_key("HTTPS://Host.example:8443/x/y.zarr").unwrap(),
            "https://host.example:8443"
        );
        assert_ne!(
            store_client_key("s3://a/x.zarr").unwrap(),
            store_client_key("s3://b/x.zarr").unwrap()
        );
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
