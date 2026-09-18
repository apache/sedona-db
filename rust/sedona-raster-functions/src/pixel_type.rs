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

//! Pixel (band data) type names, shared by every function that takes one as
//! a SQL string argument (`RS_MakeEmptyRaster`, `RS_AsRaster`, ...).

use datafusion_common::{error::Result, exec_err};
use sedona_schema::raster::BandDataType;

/// Parse a pixel type name into a [`BandDataType`].
///
/// Three spellings are accepted, all case-insensitively:
///
/// - Sedona Spark's letter codes: `D`, `F`, `I`, `UI`, `S`, `US`, `B`, `I8`,
///   `U64`, `I64`.
/// - Width-and-signedness names: `int8`, `uint8`, `int16`, `uint16`, `int32`,
///   `uint32`, `int64`, `uint64`, `float32`, `float64`.
/// - The names `RS_BandPixelType` returns (`UNSIGNED_8BITS`, `REAL_64BITS`,
///   ...), so a band's type can be fed straight back into a constructor.
///
/// Anything else is an error rather than a silent default.
pub fn parse_pixel_type(value: &str) -> Result<BandDataType> {
    match value.trim().to_ascii_lowercase().as_str() {
        "d" | "float64" | "real_64bits" => Ok(BandDataType::Float64),
        "f" | "float32" | "real_32bits" => Ok(BandDataType::Float32),
        "i" | "int32" | "signed_32bits" => Ok(BandDataType::Int32),
        "ui" | "uint32" | "unsigned_32bits" => Ok(BandDataType::UInt32),
        "s" | "int16" | "signed_16bits" => Ok(BandDataType::Int16),
        "us" | "uint16" | "unsigned_16bits" => Ok(BandDataType::UInt16),
        "b" | "uint8" | "unsigned_8bits" => Ok(BandDataType::UInt8),
        "i8" | "int8" | "signed_8bits" => Ok(BandDataType::Int8),
        "u64" | "uint64" | "unsigned_64bits" => Ok(BandDataType::UInt64),
        "i64" | "int64" | "signed_64bits" => Ok(BandDataType::Int64),
        other => exec_err!(
            "Unsupported pixelType: {} (expected one of D/F/I/UI/S/US/B/I8/U64/I64, \
             int8/uint8/int16/uint16/int32/uint32/int64/uint64/float32/float64, \
             or an RS_BandPixelType name such as UNSIGNED_8BITS)",
            other
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spark_letter_codes() {
        assert_eq!(parse_pixel_type("D").unwrap(), BandDataType::Float64);
        assert_eq!(parse_pixel_type("F").unwrap(), BandDataType::Float32);
        assert_eq!(parse_pixel_type("I").unwrap(), BandDataType::Int32);
        assert_eq!(parse_pixel_type("UI").unwrap(), BandDataType::UInt32);
        assert_eq!(parse_pixel_type("S").unwrap(), BandDataType::Int16);
        assert_eq!(parse_pixel_type("US").unwrap(), BandDataType::UInt16);
        assert_eq!(parse_pixel_type("B").unwrap(), BandDataType::UInt8);
        assert_eq!(parse_pixel_type("I8").unwrap(), BandDataType::Int8);
        assert_eq!(parse_pixel_type("U64").unwrap(), BandDataType::UInt64);
        assert_eq!(parse_pixel_type("I64").unwrap(), BandDataType::Int64);
    }

    #[test]
    fn width_names_case_insensitive_and_trimmed() {
        assert_eq!(parse_pixel_type("float64").unwrap(), BandDataType::Float64);
        assert_eq!(
            parse_pixel_type(" Float32 ").unwrap(),
            BandDataType::Float32
        );
        assert_eq!(parse_pixel_type("INT32").unwrap(), BandDataType::Int32);
        assert_eq!(parse_pixel_type("uint32").unwrap(), BandDataType::UInt32);
        assert_eq!(parse_pixel_type("int16").unwrap(), BandDataType::Int16);
        assert_eq!(parse_pixel_type("uint16").unwrap(), BandDataType::UInt16);
        assert_eq!(parse_pixel_type("uint8").unwrap(), BandDataType::UInt8);
        assert_eq!(parse_pixel_type("int8").unwrap(), BandDataType::Int8);
        assert_eq!(parse_pixel_type("uint64").unwrap(), BandDataType::UInt64);
        assert_eq!(parse_pixel_type("int64").unwrap(), BandDataType::Int64);
    }

    #[test]
    fn rs_bandpixeltype_names_round_trip() {
        for data_type in [
            BandDataType::UInt8,
            BandDataType::UInt16,
            BandDataType::Int16,
            BandDataType::UInt32,
            BandDataType::Int32,
            BandDataType::Float32,
            BandDataType::Float64,
            BandDataType::UInt64,
            BandDataType::Int64,
            BandDataType::Int8,
        ] {
            assert_eq!(
                parse_pixel_type(data_type.pixel_type_name()).unwrap(),
                data_type
            );
        }
    }

    #[test]
    fn unknown_name_errors() {
        let err = parse_pixel_type("complex128").unwrap_err().to_string();
        assert!(err.contains("Unsupported pixelType: complex128"), "{err}");
    }
}
