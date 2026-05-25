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

//! Mapping between Zarr datatypes and SedonaDB's `BandDataType`.
//!
//! zarrs 0.23 models `DataType` as a wrapper around `Arc<dyn DataTypeTraits>`,
//! so we discriminate via the type-erased `is::<T>()` check rather than
//! pattern-matching an enum.

use arrow_schema::ArrowError;
use sedona_schema::raster::BandDataType;
use zarrs::array::data_type::{
    BoolDataType, Float32DataType, Float64DataType, Int16DataType, Int32DataType, Int64DataType,
    Int8DataType, UInt16DataType, UInt32DataType, UInt64DataType, UInt8DataType,
};
use zarrs::array::DataType as ZarrDataType;

/// Map a Zarr `DataType` to a SedonaDB `BandDataType`.
///
/// Bool maps to UInt8 losslessly (Zarr packs bools to one byte each, matching
/// our representation). Variable-length, complex, and extended-precision
/// types error with `NotYetImplemented` — they have no numeric counterpart
/// in `BandDataType` today.
pub fn zarr_to_band_data_type(dt: &ZarrDataType) -> Result<BandDataType, ArrowError> {
    if dt.is::<BoolDataType>() {
        Ok(BandDataType::UInt8)
    } else if dt.is::<Int8DataType>() {
        Ok(BandDataType::Int8)
    } else if dt.is::<UInt8DataType>() {
        Ok(BandDataType::UInt8)
    } else if dt.is::<Int16DataType>() {
        Ok(BandDataType::Int16)
    } else if dt.is::<UInt16DataType>() {
        Ok(BandDataType::UInt16)
    } else if dt.is::<Int32DataType>() {
        Ok(BandDataType::Int32)
    } else if dt.is::<UInt32DataType>() {
        Ok(BandDataType::UInt32)
    } else if dt.is::<Int64DataType>() {
        Ok(BandDataType::Int64)
    } else if dt.is::<UInt64DataType>() {
        Ok(BandDataType::UInt64)
    } else if dt.is::<Float32DataType>() {
        Ok(BandDataType::Float32)
    } else if dt.is::<Float64DataType>() {
        Ok(BandDataType::Float64)
    } else {
        Err(ArrowError::NotYetImplemented(format!(
            "Zarr datatype {dt:?} has no BandDataType mapping yet"
        )))
    }
}
