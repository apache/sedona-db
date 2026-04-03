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

/// Parsed components of an outdb_uri.
///
/// The outdb_uri format is `scheme://path#fragment`, e.g.:
/// - `geotiff://s3://bucket/file.tif#band=1`
/// - `zarr://s3://bucket/store#temperature/0.0.0`
///
/// The scheme determines which loader to dispatch to.
/// The path is the external resource location (what RS_BandPath returns to users).
/// The fragment encodes loader-specific details (band id, chunk coords, etc.).
/// Each loader defines its own fragment convention.
///
/// TODO: For formats like Zarr that may need complex metadata (array path, chunk
/// coordinates, byte ranges), a simple key-value fragment may not be sufficient.
/// If this becomes a limitation, consider switching the fragment to a JSON object
/// or making the entire outdb_uri a JSON string for those formats.
#[derive(Debug, PartialEq)]
pub struct OutDbUri<'a> {
    /// Loader scheme (e.g., "geotiff", "zarr")
    pub scheme: &'a str,
    /// External resource path (e.g., "s3://bucket/file.tif")
    pub path: &'a str,
    /// Loader-specific fragment (e.g., "band=1"), or None if absent
    pub fragment: Option<&'a str>,
}

/// Parse an outdb_uri into its components.
///
/// Returns `None` if the URI doesn't contain `://` (not a valid outdb_uri).
///
/// # Examples
/// ```
/// use sedona_raster::outdb_uri::parse_outdb_uri;
///
/// let parsed = parse_outdb_uri("geotiff://s3://bucket/file.tif#band=1").unwrap();
/// assert_eq!(parsed.scheme, "geotiff");
/// assert_eq!(parsed.path, "s3://bucket/file.tif");
/// assert_eq!(parsed.fragment, Some("band=1"));
///
/// let parsed = parse_outdb_uri("zarr://s3://bucket/store").unwrap();
/// assert_eq!(parsed.scheme, "zarr");
/// assert_eq!(parsed.path, "s3://bucket/store");
/// assert_eq!(parsed.fragment, None);
/// ```
pub fn parse_outdb_uri(uri: &str) -> Option<OutDbUri<'_>> {
    let scheme_end = uri.find("://")?;
    let scheme = &uri[..scheme_end];
    let rest = &uri[scheme_end + 3..];

    let (path, fragment) = match rest.rfind('#') {
        Some(hash_pos) => (&rest[..hash_pos], Some(&rest[hash_pos + 1..])),
        None => (rest, None),
    };

    Some(OutDbUri {
        scheme,
        path,
        fragment,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geotiff_with_fragment() {
        let parsed = parse_outdb_uri("geotiff://s3://bucket/file.tif#band=1").unwrap();
        assert_eq!(parsed.scheme, "geotiff");
        assert_eq!(parsed.path, "s3://bucket/file.tif");
        assert_eq!(parsed.fragment, Some("band=1"));
    }

    #[test]
    fn test_zarr_with_fragment() {
        let parsed = parse_outdb_uri("zarr://s3://bucket/store#temperature/0.0.0").unwrap();
        assert_eq!(parsed.scheme, "zarr");
        assert_eq!(parsed.path, "s3://bucket/store");
        assert_eq!(parsed.fragment, Some("temperature/0.0.0"));
    }

    #[test]
    fn test_no_fragment() {
        let parsed = parse_outdb_uri("zarr://s3://bucket/store").unwrap();
        assert_eq!(parsed.scheme, "zarr");
        assert_eq!(parsed.path, "s3://bucket/store");
        assert_eq!(parsed.fragment, None);
    }

    #[test]
    fn test_local_path() {
        let parsed = parse_outdb_uri("geotiff:///data/rasters/dem.tif#band=1").unwrap();
        assert_eq!(parsed.scheme, "geotiff");
        assert_eq!(parsed.path, "/data/rasters/dem.tif");
        assert_eq!(parsed.fragment, Some("band=1"));
    }

    #[test]
    fn test_plain_s3_url_parsed_as_scheme() {
        // A plain s3:// URL is technically parseable — s3 becomes the scheme
        let parsed = parse_outdb_uri("s3://bucket/file.tif").unwrap();
        assert_eq!(parsed.scheme, "s3");
        assert_eq!(parsed.path, "bucket/file.tif");
        assert_eq!(parsed.fragment, None);
    }

    #[test]
    fn test_invalid_no_scheme() {
        assert!(parse_outdb_uri("/local/path/file.tif").is_none());
        assert!(parse_outdb_uri("just-a-string").is_none());
    }

    #[test]
    fn test_invalid_empty() {
        assert!(parse_outdb_uri("").is_none());
    }
}
