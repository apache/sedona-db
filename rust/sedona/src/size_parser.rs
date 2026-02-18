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

//! Parse human-readable size strings (e.g., `"4gb"`, `"512m"`, `"1.5g"`) into
//! byte counts.
//!
//! This is used by the CLI (`--memory-limit 4g`), the Python bindings
//! (`sd.options.memory_limit = "4gb"`), and
//! [`SedonaContextBuilder::from_options`](crate::context_builder::SedonaContextBuilder::from_options).

use std::collections::HashMap;
use std::sync::LazyLock;

use datafusion::error::{DataFusionError, Result};

#[derive(Debug, Clone, Copy)]
enum ByteUnit {
    Byte,
    KiB,
    MiB,
    GiB,
    TiB,
}

impl ByteUnit {
    fn multiplier(&self) -> u64 {
        match self {
            ByteUnit::Byte => 1,
            ByteUnit::KiB => 1 << 10,
            ByteUnit::MiB => 1 << 20,
            ByteUnit::GiB => 1 << 30,
            ByteUnit::TiB => 1 << 40,
        }
    }
}

static BYTE_SUFFIXES: LazyLock<HashMap<&'static str, ByteUnit>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert("b", ByteUnit::Byte);
    m.insert("k", ByteUnit::KiB);
    m.insert("kb", ByteUnit::KiB);
    m.insert("m", ByteUnit::MiB);
    m.insert("mb", ByteUnit::MiB);
    m.insert("g", ByteUnit::GiB);
    m.insert("gb", ByteUnit::GiB);
    m.insert("t", ByteUnit::TiB);
    m.insert("tb", ByteUnit::TiB);
    m
});

static SUFFIX_REGEX: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"^([0-9.]+)\s*([a-z]+)?$").unwrap());

/// Parse a human-readable size string into a byte count.
///
/// Accepts formats like `"4gb"`, `"512m"`, `"1.5g"`, `"4096"`, `"100 mb"`.
/// The suffix is case-insensitive. When no suffix is provided, the value is
/// treated as bytes.
///
/// # Supported suffixes
///
/// | Suffix    | Unit     |
/// |-----------|----------|
/// | `b`       | Bytes    |
/// | `k`, `kb` | KiB      |
/// | `m`, `mb` | MiB      |
/// | `g`, `gb` | GiB      |
/// | `t`, `tb` | TiB      |
///
/// # Examples
///
/// ```
/// use sedona::size_parser::parse_size_string;
///
/// assert_eq!(parse_size_string("4gb").unwrap(), 4 * 1024 * 1024 * 1024);
/// assert_eq!(parse_size_string("512m").unwrap(), 512 * 1024 * 1024);
/// assert_eq!(parse_size_string("4096").unwrap(), 4096);
/// ```
pub fn parse_size_string(size: &str) -> Result<usize> {
    let lower = size.to_lowercase();
    if let Some(caps) = SUFFIX_REGEX.captures(&lower) {
        let num_str = caps.get(1).unwrap().as_str();
        let num = num_str
            .parse::<f64>()
            .map_err(|_| DataFusionError::Configuration(format!("Invalid size string '{size}'")))?;

        let suffix = caps.get(2).map(|m| m.as_str()).unwrap_or("b");
        let unit = BYTE_SUFFIXES.get(suffix).ok_or_else(|| {
            DataFusionError::Configuration(format!("Invalid size string '{size}'"))
        })?;
        let total_bytes = num * unit.multiplier() as f64;
        if !total_bytes.is_finite() || total_bytes > usize::MAX as f64 {
            return Err(DataFusionError::Configuration(format!(
                "Size string '{size}' is too large"
            )));
        }

        Ok(total_bytes as usize)
    } else {
        Err(DataFusionError::Configuration(format!(
            "Invalid size string '{size}'"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_numbers_are_bytes() {
        assert_eq!(parse_size_string("5").unwrap(), 5);
        assert_eq!(parse_size_string("100").unwrap(), 100);
    }

    #[test]
    fn byte_suffix() {
        assert_eq!(parse_size_string("5b").unwrap(), 5);
    }

    #[test]
    fn kib_suffixes() {
        assert_eq!(parse_size_string("4k").unwrap(), 4 * 1024);
        assert_eq!(parse_size_string("4kb").unwrap(), 4 * 1024);
    }

    #[test]
    fn mib_suffixes() {
        assert_eq!(parse_size_string("20m").unwrap(), 20 * 1024 * 1024);
        assert_eq!(parse_size_string("20mb").unwrap(), 20 * 1024 * 1024);
    }

    #[test]
    fn gib_suffixes() {
        assert_eq!(parse_size_string("2g").unwrap(), 2 * 1024 * 1024 * 1024);
        assert_eq!(parse_size_string("2gb").unwrap(), 2 * 1024 * 1024 * 1024);
    }

    #[test]
    fn tib_suffixes() {
        assert_eq!(
            parse_size_string("3t").unwrap(),
            3 * 1024 * 1024 * 1024 * 1024
        );
        assert_eq!(
            parse_size_string("4tb").unwrap(),
            4 * 1024 * 1024 * 1024 * 1024
        );
    }

    #[test]
    fn case_insensitive() {
        assert_eq!(parse_size_string("4K").unwrap(), 4 * 1024);
        assert_eq!(parse_size_string("4KB").unwrap(), 4 * 1024);
        assert_eq!(parse_size_string("20M").unwrap(), 20 * 1024 * 1024);
        assert_eq!(parse_size_string("20MB").unwrap(), 20 * 1024 * 1024);
        assert_eq!(parse_size_string("2G").unwrap(), 2 * 1024 * 1024 * 1024);
        assert_eq!(parse_size_string("2GB").unwrap(), 2 * 1024 * 1024 * 1024);
        assert_eq!(
            parse_size_string("2T").unwrap(),
            2 * 1024 * 1024 * 1024 * 1024
        );
    }

    #[test]
    fn decimal_values() {
        assert_eq!(
            parse_size_string("1.5g").unwrap(),
            (1.5 * 1024.0 * 1024.0 * 1024.0) as usize
        );
        assert_eq!(
            parse_size_string("0.5m").unwrap(),
            (0.5 * 1024.0 * 1024.0) as usize
        );
        assert_eq!(
            parse_size_string("9.5 gb").unwrap(),
            (9.5 * 1024.0 * 1024.0 * 1024.0) as usize
        );
    }

    #[test]
    fn spaces_between_number_and_suffix() {
        assert_eq!(parse_size_string("4 k").unwrap(), 4 * 1024);
        assert_eq!(parse_size_string("20 mb").unwrap(), 20 * 1024 * 1024);
    }

    #[test]
    fn invalid_input() {
        assert!(parse_size_string("invalid").is_err());
        assert!(parse_size_string("4kbx").is_err());
        assert!(parse_size_string("-20mb").is_err());
        assert!(parse_size_string("-100").is_err());
        assert!(parse_size_string("12k12k").is_err());
    }

    #[test]
    fn overflow() {
        assert!(parse_size_string("99999999t").is_err());
    }
}
