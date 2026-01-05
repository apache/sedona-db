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

use sedona_schema::raster::BandDataType;

/// Get the number of bytes per pixel for a given BandDataType
pub fn bytes_per_pixel(datatype: BandDataType) -> usize {
    match datatype {
        BandDataType::UInt8 => 1,
        BandDataType::Int16 | BandDataType::UInt16 => 2,
        BandDataType::Int32 | BandDataType::UInt32 | BandDataType::Float32 => 4,
        BandDataType::Float64 => 8,
    }
}

/// Convert an f64 nodata value to bytes for the given BandDataType
pub fn f64_to_bandtype_bytes(value: f64, datatype: BandDataType) -> Vec<u8> {
    match datatype {
        BandDataType::UInt8 => vec![value as u8],
        BandDataType::UInt16 => (value as u16).to_le_bytes().to_vec(),
        BandDataType::Int16 => (value as i16).to_le_bytes().to_vec(),
        BandDataType::UInt32 => (value as u32).to_le_bytes().to_vec(),
        BandDataType::Int32 => (value as i32).to_le_bytes().to_vec(),
        BandDataType::Float32 => (value as f32).to_le_bytes().to_vec(),
        BandDataType::Float64 => value.to_le_bytes().to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_bytes_per_pixel() {
        assert_eq!(bytes_per_pixel(BandDataType::UInt8), 1);
        assert_eq!(bytes_per_pixel(BandDataType::Int16), 2);
        assert_eq!(bytes_per_pixel(BandDataType::UInt16), 2);
        assert_eq!(bytes_per_pixel(BandDataType::Int32), 4);
        assert_eq!(bytes_per_pixel(BandDataType::UInt32), 4);
        assert_eq!(bytes_per_pixel(BandDataType::Float32), 4);
        assert_eq!(bytes_per_pixel(BandDataType::Float64), 8);
    }

    #[test]
    fn test_f64_to_bandtype_bytes() {
        assert_eq!(
            f64_to_bandtype_bytes(255.0, BandDataType::UInt8),
            vec![255u8]
        );
        assert_eq!(
            f64_to_bandtype_bytes(65535.0, BandDataType::UInt16),
            65535u16.to_le_bytes().to_vec()
        );
        assert_eq!(
            f64_to_bandtype_bytes(-32768.0, BandDataType::Int16),
            (-32768i16).to_le_bytes().to_vec()
        );
        assert_eq!(
            f64_to_bandtype_bytes(4294967295.0, BandDataType::UInt32),
            4294967295u32.to_le_bytes().to_vec()
        );
        assert_eq!(
            f64_to_bandtype_bytes(-2147483648.0, BandDataType::Int32),
            (-2147483648i32).to_le_bytes().to_vec()
        );
        assert_eq!(
            f64_to_bandtype_bytes(std::f32::consts::PI.into(), BandDataType::Float32),
            (std::f32::consts::PI).to_le_bytes().to_vec()
        );
        assert_eq!(
            f64_to_bandtype_bytes(std::f64::consts::E, BandDataType::Float64),
            (std::f64::consts::E).to_le_bytes().to_vec()
        );
    }
}
