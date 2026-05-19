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

//! Zarr URI helpers — group locators and per-chunk OutDb anchors.
//!
//! Two URI shapes flow through the loader:
//!
//! 1. **Group URI** (the loader entry-point argument). Identifies a Zarr
//!    group on a backend, e.g. `file:///tmp/datacube.zarr`,
//!    `s3://bucket/datacube.zarr/2024`, or a bare local path. Phase 1
//!    accepts `file://` and bare-path; cloud schemes ship with the
//!    resolver.
//! 2. **Chunk anchor URI** (written into a band's `outdb_uri`). Addresses
//!    one chunk in one array within a group:
//!    `<store-uri>#array=<array-path>&chunk=<i0>,<i1>,...`. The store URI
//!    is the original group URI verbatim (whatever scheme that uses);
//!    array path and chunk indices both live in the fragment so the
//!    store URI is unambiguous even when both contain `/` (e.g.
//!    `s3://bucket/foo.zarr/2024` + `subgroup/B01`). Inner-chunk indices
//!    always — sharding is a storage detail the resolver handles.
//!
//!    The "this is a zarr anchor" signal lives in the band's
//!    `outdb_format = "zarr"` field, not in a URI scheme prefix — matches
//!    the GDAL OutDb convention and lets the format-keyed dispatcher
//!    route without parsing URI schemes.

use arrow_schema::ArrowError;

/// Parts of a chunk-anchor URI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkAnchor {
    /// Original store URI for the *group* (e.g. `file:///tmp/foo.zarr`,
    /// `s3://bucket/foo.zarr/2024`). Same value across every chunk of every
    /// array in a group.
    pub store_uri: String,
    /// Array's path within the store (e.g. `temperature`, `subgroup/B01`).
    pub array_path: String,
    /// Chunk's position in the array's inner chunk grid (one index per
    /// array dimension).
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

/// Normalize a user-supplied group URI into a local filesystem path.
///
/// Phase 1 supports `file://` and bare-path URIs only. Cloud schemes
/// (`s3://`, `gs://`, `az://`, `https://`) error with a clear message
/// pointing at the resolver work that adds them.
pub fn group_uri_to_filesystem_path(uri: &str) -> Result<std::path::PathBuf, ArrowError> {
    if let Some(rest) = uri.strip_prefix("file://") {
        return Ok(std::path::PathBuf::from(rest));
    }
    for scheme in ["s3://", "gs://", "az://", "https://", "http://"] {
        if uri.starts_with(scheme) {
            return Err(ArrowError::NotYetImplemented(format!(
                "cloud Zarr stores ({scheme}…) are not supported in this phase; \
                 use a local filesystem path or `file://` URI for now"
            )));
        }
    }
    // Bare path.
    Ok(std::path::PathBuf::from(uri))
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
        // No `#` means no fragment, so neither `array=` nor `chunk=` is
        // present. The parser surfaces "missing fragment".
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
        // Forward-compatible: extra `&key=value` pairs in the fragment
        // shouldn't break parsing — they're reserved for future use.
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

    #[test]
    fn group_uri_file_scheme() {
        let path = group_uri_to_filesystem_path("file:///tmp/foo.zarr").unwrap();
        assert_eq!(path, std::path::PathBuf::from("/tmp/foo.zarr"));
    }

    #[test]
    fn group_uri_bare_path() {
        let path = group_uri_to_filesystem_path("/tmp/foo.zarr").unwrap();
        assert_eq!(path, std::path::PathBuf::from("/tmp/foo.zarr"));
    }

    #[test]
    fn group_uri_cloud_scheme_errors() {
        let err = group_uri_to_filesystem_path("s3://bucket/foo.zarr")
            .unwrap_err()
            .to_string();
        assert!(err.contains("s3://"), "{err}");
        assert!(err.contains("not supported"), "{err}");
    }
}
