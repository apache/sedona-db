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

use arrow_schema::ArrowError;
use sedona_schema::raster::BandDataType;

/// Get the number of bytes per pixel for a given BandDataType
pub fn bytes_per_pixel(datatype: BandDataType) -> Result<usize, ArrowError> {
    match datatype {
        BandDataType::UInt8 => Ok(1),
        BandDataType::Int16 | BandDataType::UInt16 => Ok(2),
        BandDataType::Int32 | BandDataType::UInt32 | BandDataType::Float32 => Ok(4),
        BandDataType::Float64 => Ok(8),
    }
}

/// Convert an f64 nodata value to bytes for the given BandDataType
pub fn f64_to_bandtype_bytes(value: f64, datatype: BandDataType) -> Result<Vec<u8>, ArrowError> {
    match datatype {
        BandDataType::UInt8 => Ok(vec![value as u8]),
        BandDataType::UInt16 => Ok((value as u16).to_le_bytes().to_vec()),
        BandDataType::Int16 => Ok((value as i16).to_le_bytes().to_vec()),
        BandDataType::UInt32 => Ok((value as u32).to_le_bytes().to_vec()),
        BandDataType::Int32 => Ok((value as i32).to_le_bytes().to_vec()),
        BandDataType::Float32 => Ok((value as f32).to_le_bytes().to_vec()),
        BandDataType::Float64 => Ok(value.to_le_bytes().to_vec()),
    }
}
