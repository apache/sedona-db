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

//! GDAL-format-driver-internal parser for out-db raster source URIs.
//!
//! When a band's `outdb_format` dispatches to the GDAL driver, the loader
//! uses this helper to extract a 1-based source band index from `outdb_uri`
//! via the SedonaDB convention `<uri>#band=N`. The convention is private to
//! the GDAL driver — the schema and format-agnostic surfaces (e.g.
//! `RS_BandPath`) treat `outdb_uri` as opaque. Other format drivers handle
//! their own URIs however they like.

use datafusion_common::{error::Result, exec_err};

/// Parse a SedonaDB out-db source URI into the GDAL-side URI and 1-based
/// source band index.
///
/// Behaviour:
///
/// - `<uri>#band=N` where `N` parses as a `u32` in `1..=u32::MAX`: strips
///   the fragment and returns `(<uri>, N)`.
/// - `<uri>#band=...` with a value that is not a positive `u32` (zero,
///   negative, non-numeric, empty, or overflowing `u32`): returns an
///   `Execution` error. The user explicitly asked for a band; we refuse to
///   silently substitute a default.
/// - GDAL-native subdataset URIs (e.g. `HDF5:"x.h5":/var`,
///   `NETCDF:"x.nc":var`, `GTIFF_DIR:1:multi.tif`) and any URI whose
///   fragment is not `band=...`: pass through verbatim with default band
///   index 1.
/// - Plain URIs without any fragment: pass through verbatim with default
///   band index 1.
pub(crate) fn parse_outdb_source(uri: &str) -> Result<(String, u32)> {
    // rsplit lets a trailing `#band=N` win over any earlier `#anchor` in the
    // URI — useful for users who append the SedonaDB convention to a URI
    // that already carries a fragment.
    if let Some((prefix, fragment)) = uri.rsplit_once('#') {
        if let Some(band_str) = fragment.strip_prefix("band=") {
            return match band_str.parse::<u32>() {
                Ok(band) if band >= 1 => Ok((prefix.to_string(), band)),
                _ => exec_err!(
                    "Invalid band index in outdb URI fragment '#band={band_str}': expected a positive integer in 1..=u32::MAX"
                ),
            };
        }
    }
    Ok((uri.to_string(), 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_fragment_defaults_to_band_one() {
        assert_eq!(
            parse_outdb_source("s3://bucket/file.tif").unwrap(),
            ("s3://bucket/file.tif".to_string(), 1),
        );
    }

    #[test]
    fn band_fragment_extracts_index() {
        assert_eq!(
            parse_outdb_source("s3://bucket/file.tif#band=42").unwrap(),
            ("s3://bucket/file.tif".to_string(), 42),
        );
    }

    #[test]
    fn band_one_fragment_round_trips() {
        assert_eq!(
            parse_outdb_source("s3://bucket/file.tif#band=1").unwrap(),
            ("s3://bucket/file.tif".to_string(), 1),
        );
    }

    #[test]
    fn band_max_u32_accepted() {
        let max = u32::MAX;
        let uri = format!("s3://bucket/file.tif#band={max}");
        assert_eq!(
            parse_outdb_source(&uri).unwrap(),
            ("s3://bucket/file.tif".to_string(), max),
        );
    }

    #[test]
    fn band_zero_errors() {
        let msg = parse_outdb_source("s3://bucket/file.tif#band=0")
            .unwrap_err()
            .to_string();
        assert!(msg.contains("band=0"), "msg was: {msg}");
        assert!(msg.contains("positive integer"), "msg was: {msg}");
    }

    #[test]
    fn negative_band_errors() {
        let msg = parse_outdb_source("s3://bucket/file.tif#band=-2")
            .unwrap_err()
            .to_string();
        assert!(msg.contains("band=-2"), "msg was: {msg}");
    }

    #[test]
    fn band_overflow_errors() {
        // 4294967296 = u32::MAX + 1
        let msg = parse_outdb_source("s3://bucket/file.tif#band=4294967296")
            .unwrap_err()
            .to_string();
        assert!(msg.contains("band=4294967296"), "msg was: {msg}");
    }

    #[test]
    fn non_numeric_band_errors() {
        let msg = parse_outdb_source("s3://bucket/file.tif#band=abc")
            .unwrap_err()
            .to_string();
        assert!(msg.contains("band=abc"), "msg was: {msg}");
    }

    #[test]
    fn empty_band_value_errors() {
        let msg = parse_outdb_source("s3://bucket/file.tif#band=")
            .unwrap_err()
            .to_string();
        assert!(msg.contains("band="), "msg was: {msg}");
    }

    #[test]
    fn non_band_fragment_passes_through() {
        let uri = "s3://bucket/file.tif#section";
        assert_eq!(parse_outdb_source(uri).unwrap(), (uri.to_string(), 1));
    }

    #[test]
    fn empty_fragment_passes_through() {
        let uri = "s3://bucket/file.tif#";
        assert_eq!(parse_outdb_source(uri).unwrap(), (uri.to_string(), 1));
    }

    #[test]
    fn url_query_string_preserved_with_band_fragment() {
        assert_eq!(
            parse_outdb_source("https://example.com/r.tif?token=abc#band=3").unwrap(),
            ("https://example.com/r.tif?token=abc".to_string(), 3),
        );
    }

    #[test]
    fn url_query_string_with_non_band_fragment_passes_through() {
        let uri = "https://example.com/r.tif?token=abc#anchor";
        assert_eq!(parse_outdb_source(uri).unwrap(), (uri.to_string(), 1));
    }

    #[test]
    fn local_path_with_band_fragment() {
        assert_eq!(
            parse_outdb_source("/tmp/file.tif#band=5").unwrap(),
            ("/tmp/file.tif".to_string(), 5),
        );
    }

    #[test]
    fn local_path_without_fragment() {
        assert_eq!(
            parse_outdb_source("/tmp/file.tif").unwrap(),
            ("/tmp/file.tif".to_string(), 1),
        );
    }

    #[test]
    fn gdal_subdataset_hdf5_passthrough() {
        let uri = r#"HDF5:"/path/x.h5"://temperature"#;
        assert_eq!(parse_outdb_source(uri).unwrap(), (uri.to_string(), 1));
    }

    #[test]
    fn gdal_subdataset_netcdf_passthrough() {
        let uri = r#"NETCDF:"/path/file.nc":variable"#;
        assert_eq!(parse_outdb_source(uri).unwrap(), (uri.to_string(), 1));
    }

    #[test]
    fn gdal_subdataset_gtiff_dir_passthrough() {
        let uri = "GTIFF_DIR:1:/path/multi.tif";
        assert_eq!(parse_outdb_source(uri).unwrap(), (uri.to_string(), 1));
    }

    #[test]
    fn gdal_subdataset_with_band_fragment_extracts_band() {
        let uri = r#"HDF5:"/path/x.h5":/var#band=3"#;
        let (gdal_uri, band) = parse_outdb_source(uri).unwrap();
        assert_eq!(gdal_uri, r#"HDF5:"/path/x.h5":/var"#);
        assert_eq!(band, 3);
    }

    #[test]
    fn trailing_band_wins_over_earlier_anchor() {
        assert_eq!(
            parse_outdb_source("https://example.com/r.tif#anchor#band=7").unwrap(),
            ("https://example.com/r.tif#anchor".to_string(), 7),
        );
    }

    #[test]
    fn empty_uri() {
        assert_eq!(parse_outdb_source("").unwrap(), (String::new(), 1));
    }
}
