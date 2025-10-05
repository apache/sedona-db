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
use std::{env, fs};

use datafusion_common::Result;
use sedona_common::sedona_internal_err;

/// Find the most likely path to the test GeoParquet file
///
/// See <https://github.com/geoarrow/geooarrow-data> for available files. Most files
/// are available from a naive submodule checkout; however, some must be downloaded
/// (e.g., for benchmarks).
pub fn test_geoparquet(group: &str, name: &str) -> Result<String> {
    let geoarrow_data = geoarrow_data_dir()?;
    let path = format!("{geoarrow_data}/{group}/files/{group}_{name}_geo.parquet");
    if let Ok(exists) = fs::exists(&path) {
        if exists {
            return Ok(path);
        }
    }

    sedona_internal_err!(
        "geoarrow-data test file '{path}' does not exist.\n{}\n{}",
        "You may need to check the value of the SEDONA_GEOARROW_DATA_DIR environment variable,",
        "run submodules/download-assets.py, or check the name of the file you requested"
    )
}

/// Find the most likely path to the geoarrow-data testing directory if it exists
///
/// This looks for a geoarrow-data checkout using the value of SEDONA_GEOARROW_DATA_DIR,
/// the directory that would be valid if running cargo run from the repository root,
/// or the directory that would be valid if running cargo test (in that order).
pub fn geoarrow_data_dir() -> Result<String> {
    // Always use env-specified and error if it doesn't exist
    if let Ok(from_env) = env::var("SEDONA_GEOARROW_DATA_DIR") {
        if fs::exists(&from_env)? {
            return Ok(from_env);
        } else {
            return sedona_internal_err!(
                "{}\n{}{}{}",
                "Can't resolve geoarrow-data from the current working directory because",
                "the value of the SEDONA_GEOARROW_DATA_DIR (",
                from_env,
                ") does not exist"
            );
        }
    }

    let likely_possibilities = [
        // Because we're in a cargo test from rust/some-crate
        "../../submodules/geoarrow-data".to_string(),
        // Because we're in the cli from cargo run
        "submodules/geoarrow-data".to_string(),
    ];

    for possibility in likely_possibilities.into_iter().rev() {
        if let Ok(exists) = fs::exists(&possibility) {
            if exists {
                return Ok(possibility);
            }
        }
    }

    sedona_internal_err!(
        "{}\n{}\n{}",
        "Can't resolve geoarrow-data from the current working directory",
        "You may need to run `git submodule init && git submodule update --recursive` or",
        "set the SEDONA_GEOARROW_DATA_DIR environment variable"
    )
}

/// Find the most likely path to the sedona-testing directory if it exists
///
/// This mirrors [`geoarrow_data_dir`] but for the sedona-testing submodule.
/// It checks the `SEDONA_TESTING_DIR` environment variable first, then
/// falls back to the typical repository-relative locations.
pub fn sedona_testing_dir() -> Result<String> {
    if let Ok(from_env) = env::var("SEDONA_TESTING_DIR") {
        if fs::exists(&from_env)? {
            return Ok(from_env);
        } else {
            return sedona_internal_err!(
                "{}\n{}{}{}",
                "Can't resolve sedona-testing directory because",
                "the value of the SEDONA_TESTING_DIR (",
                from_env,
                ") does not exist"
            );
        }
    }

    let likely_possibilities = [
        "../../submodules/sedona-testing".to_string(),
        "submodules/sedona-testing".to_string(),
    ];

    for possibility in likely_possibilities.into_iter().rev() {
        if let Ok(exists) = fs::exists(&possibility) {
            if exists {
                return Ok(possibility);
            }
        }
    }

    sedona_internal_err!(
        "{}\n{}\n{}",
        "Can't resolve sedona-testing directory from the current working directory",
        "You may need to run `git submodule init && git submodule update --recursive` or",
        "set the SEDONA_TESTING_DIR environment variable"
    )
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn example_files() {
        // By default this should resolve, since we are in a test!
        assert!(geoarrow_data_dir().is_ok());
        assert!(test_geoparquet("natural-earth", "countries").is_ok());

        // Check a good data dir but a bad file
        let err = test_geoparquet("invalid group", "invalid name").unwrap_err();
        assert!(err.message().contains("geoarrow-data test file"));

        // Check a bad data dir
        env::set_var("SEDONA_GEOARROW_DATA_DIR", "this_directory_does_not_exist");
        let err = geoarrow_data_dir();
        env::remove_var("SEDONA_GEOARROW_DATA_DIR");
        assert!(err
            .unwrap_err()
            .message()
            .contains("the value of the SEDONA_GEOARROW_DATA_DIR"));

        // Check a good but explicitly specified data dir
        env::set_var("SEDONA_GEOARROW_DATA_DIR", geoarrow_data_dir().unwrap());
        let maybe_file = test_geoparquet("natural-earth", "countries");
        env::remove_var("SEDONA_GEOARROW_DATA_DIR");
        assert!(maybe_file.is_ok());
    }

    #[test]
    fn sedona_testing_dir_resolves() {
        assert!(sedona_testing_dir().is_ok());

        env::set_var("SEDONA_TESTING_DIR", "this_directory_does_not_exist");
        let err = sedona_testing_dir();
        env::remove_var("SEDONA_TESTING_DIR");
        assert!(err
            .unwrap_err()
            .message()
            .contains("the value of the SEDONA_TESTING_DIR"));

        env::set_var("SEDONA_TESTING_DIR", sedona_testing_dir().unwrap());
        let maybe_dir = sedona_testing_dir();
        env::remove_var("SEDONA_TESTING_DIR");
        assert!(maybe_dir.is_ok());
    }
}
