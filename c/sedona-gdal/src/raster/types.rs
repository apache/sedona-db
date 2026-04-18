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

//! Ported (and contains copied code) from georust/gdal:
//! <https://github.com/georust/gdal/blob/v0.19.0/src/raster/types.rs>.
//! Original code is licensed under MIT.

use crate::gdal_dyn_bindgen::{
    self, GDALDataType, GDALOpenFlags, GDALRIOResampleAlg, GDAL_OF_READONLY, GDAL_OF_VERBOSE_ERROR,
};

/// A Rust-friendly enum mirroring the georust/gdal `GdalDataType` names.
///
/// This maps 1-to-1 with [`GDALDataType`] but uses Rust-idiomatic names like `UInt8`
/// instead of `GDT_Byte`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GdalDataType {
    Unknown,
    UInt8,
    Int8,
    UInt16,
    Int16,
    UInt32,
    Int32,
    UInt64,
    Int64,
    Float32,
    Float64,
}

impl GdalDataType {
    /// Convert from the C-level `GDALDataType` enum.
    ///
    /// Returns `None` for complex types and `GDT_TypeCount`.
    pub fn from_c(c_type: GDALDataType) -> Option<Self> {
        match c_type {
            GDALDataType::GDT_Unknown => Some(Self::Unknown),
            GDALDataType::GDT_Byte => Some(Self::UInt8),
            GDALDataType::GDT_Int8 => Some(Self::Int8),
            GDALDataType::GDT_UInt16 => Some(Self::UInt16),
            GDALDataType::GDT_Int16 => Some(Self::Int16),
            GDALDataType::GDT_UInt32 => Some(Self::UInt32),
            GDALDataType::GDT_Int32 => Some(Self::Int32),
            GDALDataType::GDT_UInt64 => Some(Self::UInt64),
            GDALDataType::GDT_Int64 => Some(Self::Int64),
            GDALDataType::GDT_Float32 => Some(Self::Float32),
            GDALDataType::GDT_Float64 => Some(Self::Float64),
            _ => None, // Complex types, Float16, TypeCount
        }
    }

    /// Convert to the C-level `GDALDataType` enum.
    pub fn to_c(self) -> GDALDataType {
        match self {
            Self::Unknown => GDALDataType::GDT_Unknown,
            Self::UInt8 => GDALDataType::GDT_Byte,
            Self::Int8 => GDALDataType::GDT_Int8,
            Self::UInt16 => GDALDataType::GDT_UInt16,
            Self::Int16 => GDALDataType::GDT_Int16,
            Self::UInt32 => GDALDataType::GDT_UInt32,
            Self::Int32 => GDALDataType::GDT_Int32,
            Self::UInt64 => GDALDataType::GDT_UInt64,
            Self::Int64 => GDALDataType::GDT_Int64,
            Self::Float32 => GDALDataType::GDT_Float32,
            Self::Float64 => GDALDataType::GDT_Float64,
        }
    }

    /// Return the ordinal value compatible with the C API (same as `self.to_c() as i32`).
    pub fn ordinal(self) -> i32 {
        self.to_c() as i32
    }

    /// Return the byte size of this data type (0 for Unknown).
    pub fn byte_size(self) -> usize {
        match self {
            Self::Unknown => 0,
            Self::UInt8 | Self::Int8 => 1,
            Self::UInt16 | Self::Int16 => 2,
            Self::UInt32 | Self::Int32 | Self::Float32 => 4,
            Self::UInt64 | Self::Int64 | Self::Float64 => 8,
        }
    }
}

/// Trait mapping Rust primitive types to GDAL data types.
pub trait GdalType {
    fn gdal_ordinal() -> GDALDataType;
}

macro_rules! impl_gdal_type {
    ($($ty:ty => $variant:ident),+ $(,)?) => {
        $(
            impl GdalType for $ty {
                fn gdal_ordinal() -> GDALDataType {
                    GDALDataType::$variant
                }
            }
        )+
    };
}

impl_gdal_type! {
    u8 => GDT_Byte,
    i8 => GDT_Int8,
    u16 => GDT_UInt16,
    i16 => GDT_Int16,
    u32 => GDT_UInt32,
    i32 => GDT_Int32,
    u64 => GDT_UInt64,
    i64 => GDT_Int64,
    f32 => GDT_Float32,
    f64 => GDT_Float64,
}

/// A 2D raster buffer.
#[derive(Debug, Clone)]
pub struct Buffer<T: GdalType> {
    /// Shape as (cols, rows) — matches georust/gdal convention.
    pub shape: (usize, usize),
    pub data: Vec<T>,
}

impl<T: GdalType + Copy> Buffer<T> {
    pub fn new(shape: (usize, usize), data: Vec<T>) -> Self {
        Self { shape, data }
    }

    /// Return the buffer data as a slice (georust compatibility).
    pub fn data(&self) -> &[T] {
        &self.data
    }
}

/// Options for opening a dataset.
pub struct DatasetOptions<'a> {
    pub open_flags: GDALOpenFlags,
    pub allowed_drivers: Option<&'a [&'a str]>,
    pub open_options: Option<&'a [&'a str]>,
    pub sibling_files: Option<&'a [&'a str]>,
}

impl<'a> Default for DatasetOptions<'a> {
    fn default() -> Self {
        Self {
            open_flags: GDAL_OF_READONLY | GDAL_OF_VERBOSE_ERROR,
            allowed_drivers: None,
            open_options: None,
            sibling_files: None,
        }
    }
}

/// Raster creation options (list of "KEY=VALUE" strings).
pub type RasterCreationOptions<'a> = &'a [&'a str];

/// GDAL resample algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResampleAlg {
    NearestNeighbour,
    Bilinear,
    Cubic,
    CubicSpline,
    Lanczos,
    Average,
    Mode,
    Gauss,
}

impl ResampleAlg {
    /// Convert to the numeric `GDALRIOResampleAlg` value used by `GDALRasterIOExtraArg`.
    pub fn to_gdal(self) -> GDALRIOResampleAlg {
        match self {
            ResampleAlg::NearestNeighbour => gdal_dyn_bindgen::GRIORA_NearestNeighbour,
            ResampleAlg::Bilinear => gdal_dyn_bindgen::GRIORA_Bilinear,
            ResampleAlg::Cubic => gdal_dyn_bindgen::GRIORA_Cubic,
            ResampleAlg::CubicSpline => gdal_dyn_bindgen::GRIORA_CubicSpline,
            ResampleAlg::Lanczos => gdal_dyn_bindgen::GRIORA_Lanczos,
            ResampleAlg::Average => gdal_dyn_bindgen::GRIORA_Average,
            ResampleAlg::Mode => gdal_dyn_bindgen::GRIORA_Mode,
            ResampleAlg::Gauss => gdal_dyn_bindgen::GRIORA_Gauss,
        }
    }

    /// Return the string name for use in overview building and VRT resampling options.
    pub fn to_gdal_str(self) -> &'static str {
        match self {
            ResampleAlg::NearestNeighbour => "NearestNeighbour",
            ResampleAlg::Bilinear => "Bilinear",
            ResampleAlg::Cubic => "Cubic",
            ResampleAlg::CubicSpline => "CubicSpline",
            ResampleAlg::Lanczos => "Lanczos",
            ResampleAlg::Average => "Average",
            ResampleAlg::Mode => "Mode",
            ResampleAlg::Gauss => "Gauss",
        }
    }
}
