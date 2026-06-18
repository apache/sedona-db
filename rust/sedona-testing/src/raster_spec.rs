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

//! Terse raster fixtures for tests.
//!
//! [RasterSpec] is to rasters what WKT is to geometries: a compact way to
//! declare a test input without hand-driving [RasterBuilder]. A spec is a
//! plain value — build it once, clone it, pass it to the `invoke_raster_*`
//! helpers on `ScalarUdfTester`, or use it as the expected side of an
//! assertion.
//!
//! ```
//! use sedona_schema::raster::BandDataType;
//! use sedona_testing::raster_spec::RasterSpec;
//!
//! // A 4x5 raster with one Float32 band of sequential pixel values:
//! let raster = RasterSpec::d2(4, 5).band(BandDataType::Float32).build();
//!
//! // A 3-D datacube band with explicit pixel values:
//! let cube = RasterSpec::nd(&["time", "y", "x"], &[2, 2, 2])
//!     .band_values(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
//!     .build();
//! ```

use arrow_array::{
    Array, ArrayRef, Int64Array, ListArray, StringArray, StringViewArray, StructArray,
};
use datafusion_common::ScalarValue;
use datafusion_expr::{Expr, Literal};
use sedona_raster::array::RasterStructArray;
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::is_spatial_dim_pair;
use sedona_schema::crs::lnglat;
use sedona_schema::raster::BandDataType;
use std::sync::Arc;

use crate::rasters::assert_raster_arrays_equal;

/// A pixel type that can fill a test band.
///
/// Implemented for the ten Rust types matching [BandDataType]. Bytes are
/// little-endian, matching the raster byte convention.
pub trait PixelValue: sealed::Sealed + Copy {
    /// The [BandDataType] corresponding to this Rust type.
    const DATA_TYPE: BandDataType;

    /// Append this value's little-endian bytes to `out`.
    fn extend_le_bytes(&self, out: &mut Vec<u8>);

    /// Decode one value from its little-endian bytes.
    ///
    /// Panics if `bytes` is not exactly this type's width (test code).
    fn from_le_slice(bytes: &[u8]) -> Self;
}

mod sealed {
    pub trait Sealed {}
}

macro_rules! impl_pixel_value {
    ($($ty:ty => $data_type:expr),* $(,)?) => {
        $(
            impl sealed::Sealed for $ty {}

            impl PixelValue for $ty {
                const DATA_TYPE: BandDataType = $data_type;

                fn extend_le_bytes(&self, out: &mut Vec<u8>) {
                    out.extend_from_slice(&self.to_le_bytes());
                }

                fn from_le_slice(bytes: &[u8]) -> Self {
                    Self::from_le_bytes(bytes.try_into().expect("pixel byte width"))
                }
            }
        )*
    };
}

impl_pixel_value!(
    u8 => BandDataType::UInt8,
    i8 => BandDataType::Int8,
    u16 => BandDataType::UInt16,
    i16 => BandDataType::Int16,
    u32 => BandDataType::UInt32,
    i32 => BandDataType::Int32,
    u64 => BandDataType::UInt64,
    i64 => BandDataType::Int64,
    f32 => BandDataType::Float32,
    f64 => BandDataType::Float64,
);

/// Sequential fill (0, 1, 2, …) for a band's default pixel data,
/// little-endian, wrapping where the value range is narrower than the
/// pixel count.
fn sequential_band_bytes(data_type: BandDataType, pixel_count: usize) -> Vec<u8> {
    macro_rules! fill {
        ($ty:ty) => {{
            let mut out = Vec::with_capacity(pixel_count * std::mem::size_of::<$ty>());
            for i in 0..pixel_count {
                out.extend_from_slice(&(i as $ty).to_le_bytes());
            }
            out
        }};
    }

    match data_type {
        BandDataType::UInt8 => fill!(u8),
        BandDataType::Int8 => fill!(i8),
        BandDataType::UInt16 => fill!(u16),
        BandDataType::Int16 => fill!(i16),
        BandDataType::UInt32 => fill!(u32),
        BandDataType::Int32 => fill!(i32),
        BandDataType::UInt64 => fill!(u64),
        BandDataType::Int64 => fill!(i64),
        BandDataType::Float32 => fill!(f32),
        BandDataType::Float64 => fill!(f64),
    }
}

#[derive(Clone, Debug)]
struct BandSpec {
    name: Option<String>,
    dims: Vec<String>,
    shape: Vec<i64>,
    data_type: BandDataType,
    nodata: Option<Vec<u8>>,
    /// `None` means "fill with sequential values at build time". OutDb bands
    /// set `Some(vec![])` — the empty-`data` OutDb convention.
    data: Option<Vec<u8>>,
    outdb_uri: Option<String>,
    outdb_format: Option<String>,
}

/// A compact, declarative raster fixture.
///
/// Construct with [RasterSpec::d2] or [RasterSpec::nd], add bands with the
/// `band*` methods, then either [build](RasterSpec::build) a one-row
/// `StructArray`, take a [scalar](RasterSpec::scalar), or combine several
/// specs (with nulls) via [raster_array].
///
/// Defaults: transform `[0, 1, 0, 0, 0, -1]` (unit pixels, north-up, origin
/// 0,0), lng/lat CRS, sequential pixel values per band.
#[derive(Clone, Debug)]
pub struct RasterSpec {
    transform: [f64; 6],
    /// Raster-level spatial dims, X-first (e.g. `["x", "y"]`).
    spatial_dims: Vec<String>,
    /// Sizes matching `spatial_dims` (`[width, height]`).
    spatial_shape: Vec<i64>,
    crs: Option<String>,
    /// Default band layout, C-order (slowest dim first, e.g. `["y", "x"]`).
    default_band_dims: Vec<String>,
    default_band_shape: Vec<i64>,
    bands: Vec<BandSpec>,
}

impl RasterSpec {
    /// A 2-D raster of `width` x `height` pixels. Bands default to
    /// `dims=["y","x"]`, `shape=[height, width]`.
    pub fn d2(width: i64, height: i64) -> Self {
        Self {
            transform: [0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
            spatial_dims: vec!["x".to_string(), "y".to_string()],
            spatial_shape: vec![width, height],
            crs: Some(lnglat().unwrap().to_crs_string()),
            default_band_dims: vec!["y".to_string(), "x".to_string()],
            default_band_shape: vec![height, width],
            bands: Vec::new(),
        }
    }

    /// An N-D raster from a default band layout in C order (slowest dim
    /// first, e.g. `["time", "y", "x"]` with shape `[t, h, w]`).
    ///
    /// The trailing two dims must be a recognized spatial pair (`y`/`x`,
    /// `lat`/`lon`, `latitude`/`longitude`); they become the raster-level
    /// `spatial_dims`/`spatial_shape` in X-first order, matching what the
    /// readers emit.
    pub fn nd(band_dims: &[&str], band_shape: &[i64]) -> Self {
        assert!(
            band_dims.len() >= 2 && band_dims.len() == band_shape.len(),
            "RasterSpec::nd requires >= 2 dims and dims/shape of equal length, \
             got dims {band_dims:?} and shape {band_shape:?}"
        );
        let n = band_dims.len();
        assert!(
            is_spatial_dim_pair(band_dims[n - 2], band_dims[n - 1]),
            "RasterSpec::nd requires the trailing two dims to be a spatial pair \
             (e.g. [\"y\", \"x\"]), got {band_dims:?}"
        );
        Self {
            transform: [0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
            spatial_dims: vec![band_dims[n - 1].to_string(), band_dims[n - 2].to_string()],
            spatial_shape: vec![band_shape[n - 1], band_shape[n - 2]],
            crs: Some(lnglat().unwrap().to_crs_string()),
            default_band_dims: band_dims.iter().map(|d| d.to_string()).collect(),
            default_band_shape: band_shape.to_vec(),
            bands: Vec::new(),
        }
    }

    /// Override the CRS (`None` clears it; the default is lng/lat).
    pub fn crs(mut self, crs: Option<&str>) -> Self {
        self.crs = crs.map(|c| c.to_string());
        self
    }

    /// Override the GDAL geotransform
    /// `[origin_x, scale_x, skew_x, origin_y, skew_y, scale_y]`.
    pub fn transform(mut self, transform: [f64; 6]) -> Self {
        self.transform = transform;
        self
    }

    /// Add a band with the spec's default layout and sequential pixel values
    /// (0, 1, 2, … in `data_type`).
    pub fn band(self, data_type: BandDataType) -> Self {
        let dims = self.default_band_dims.clone();
        let shape = self.default_band_shape.clone();
        self.push_band(dims, shape, data_type, None)
    }

    /// Add a band with the spec's default layout and explicit pixel values
    /// (C order). The band data type is inferred from `T`.
    pub fn band_values<T: PixelValue>(self, values: &[T]) -> Self {
        let dims = self.default_band_dims.clone();
        let shape = self.default_band_shape.clone();
        self.push_band_values(dims, shape, values)
    }

    /// Add a band with an explicit layout (C order) and sequential pixel
    /// values. Use this for bands that differ from the spec's default —
    /// e.g. mixed-dimensionality rasters or bands that disagree on a
    /// non-spatial dim size.
    pub fn band_nd(self, dims: &[&str], shape: &[i64], data_type: BandDataType) -> Self {
        let dims = dims.iter().map(|d| d.to_string()).collect();
        self.push_band(dims, shape.to_vec(), data_type, None)
    }

    /// Add a band with an explicit layout (C order) and explicit pixel
    /// values. The band data type is inferred from `T`.
    pub fn band_values_nd<T: PixelValue>(self, dims: &[&str], shape: &[i64], values: &[T]) -> Self {
        let dims = dims.iter().map(|d| d.to_string()).collect();
        self.push_band_values(dims, shape.to_vec(), values)
    }

    /// Set the nodata value of the most recently added band. `T` must match
    /// the band's data type.
    pub fn nodata<T: PixelValue>(mut self, value: T) -> Self {
        let band = self.last_band_mut("nodata");
        assert_eq!(
            band.data_type,
            T::DATA_TYPE,
            "nodata value type does not match the band data type"
        );
        let mut bytes = Vec::new();
        value.extend_le_bytes(&mut bytes);
        band.nodata = Some(bytes);
        self
    }

    /// Set the name of the most recently added band.
    pub fn name(mut self, name: &str) -> Self {
        self.last_band_mut("name").name = Some(name.to_string());
        self
    }

    /// Make the most recently added band OutDb: its bytes live at `uri`
    /// (using the SedonaDB `<url>#band=N` convention where applicable) and
    /// its `data` buffer is the empty OutDb sentinel.
    pub fn outdb(mut self, uri: &str, format: Option<&str>) -> Self {
        let band = self.last_band_mut("outdb");
        band.outdb_uri = Some(uri.to_string());
        band.outdb_format = format.map(|f| f.to_string());
        band.data = Some(Vec::new());
        self
    }

    /// Build a one-row raster `StructArray` from this spec.
    pub fn build(&self) -> StructArray {
        let mut builder = RasterBuilder::new(1);
        self.append_to(&mut builder);
        builder.finish().expect("finish raster")
    }

    /// This spec as a raster `ScalarValue` (a one-row struct).
    pub fn scalar(&self) -> ScalarValue {
        ScalarValue::Struct(Arc::new(self.build()))
    }

    fn push_band(
        mut self,
        dims: Vec<String>,
        shape: Vec<i64>,
        data_type: BandDataType,
        data: Option<Vec<u8>>,
    ) -> Self {
        self.bands.push(BandSpec {
            name: None,
            dims,
            shape,
            data_type,
            nodata: None,
            data,
            outdb_uri: None,
            outdb_format: None,
        });
        self
    }

    fn push_band_values<T: PixelValue>(
        self,
        dims: Vec<String>,
        shape: Vec<i64>,
        values: &[T],
    ) -> Self {
        let pixel_count = shape.iter().product::<i64>().max(0) as usize;
        assert_eq!(
            values.len(),
            pixel_count,
            "band_values: expected {pixel_count} values for shape {shape:?}, got {}",
            values.len()
        );
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
        for value in values {
            value.extend_le_bytes(&mut bytes);
        }
        self.push_band(dims, shape, T::DATA_TYPE, Some(bytes))
    }

    fn last_band_mut(&mut self, modifier: &str) -> &mut BandSpec {
        self.bands
            .last_mut()
            .unwrap_or_else(|| panic!("RasterSpec::{modifier} requires a band — add one first"))
    }

    fn append_to(&self, builder: &mut RasterBuilder) {
        let spatial_dims: Vec<&str> = self.spatial_dims.iter().map(|d| d.as_str()).collect();
        builder
            .start_raster_nd(
                &self.transform,
                &spatial_dims,
                &self.spatial_shape,
                self.crs.as_deref(),
            )
            .expect("start raster");
        for band in &self.bands {
            let dims: Vec<&str> = band.dims.iter().map(|d| d.as_str()).collect();
            builder
                .start_band_nd(
                    band.name.as_deref(),
                    &dims,
                    &band.shape,
                    band.data_type,
                    band.nodata.as_deref(),
                    band.outdb_uri.as_deref(),
                    band.outdb_format.as_deref(),
                )
                .expect("start band");
            let bytes = match &band.data {
                Some(bytes) => bytes.clone(),
                None => sequential_band_bytes(
                    band.data_type,
                    band.shape.iter().product::<i64>().max(0) as usize,
                ),
            };
            builder.band_data_writer().append_value(&bytes);
            builder.finish_band().expect("finish band");
        }
        builder.finish_raster().expect("finish raster");
    }
}

impl Literal for RasterSpec {
    fn lit(&self) -> Expr {
        Expr::Literal(self.scalar(), None)
    }
}

impl Literal for &RasterSpec {
    fn lit(&self) -> Expr {
        Expr::Literal(self.scalar(), None)
    }
}

/// Build a multi-row raster `StructArray` from specs; `None` entries become
/// null rasters.
pub fn raster_array<I: IntoIterator<Item = Option<RasterSpec>>>(specs: I) -> StructArray {
    let specs: Vec<_> = specs.into_iter().collect();
    let mut builder = RasterBuilder::new(specs.len());
    for spec in &specs {
        match spec {
            Some(spec) => spec.append_to(&mut builder),
            None => builder.append_null().expect("append null raster"),
        }
    }
    builder.finish().expect("finish raster array")
}

/// Assert that a raster-typed result array equals the rasters described by
/// `expected` (with `None` for expected-null rows).
pub fn assert_rasters_equal(actual: &ArrayRef, expected: &[Option<RasterSpec>]) {
    let actual_struct = actual
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("expected a raster StructArray result");
    let expected_struct = raster_array(expected.iter().cloned());
    assert_raster_arrays_equal(
        &RasterStructArray::try_new(actual_struct).unwrap(),
        &RasterStructArray::try_new(&expected_struct).unwrap(),
    );
}

/// Assert that a raster-typed scalar result equals the raster described by
/// `expected`.
pub fn assert_raster_scalar_equals(actual: &ScalarValue, expected: &RasterSpec) {
    match actual {
        ScalarValue::Struct(actual_struct) => {
            let expected_struct = expected.build();
            assert_raster_arrays_equal(
                &RasterStructArray::try_new(actual_struct).unwrap(),
                &RasterStructArray::try_new(&expected_struct).unwrap(),
            );
        }
        other => panic!("expected a raster struct scalar, got {other:?}"),
    }
}

/// Extract one row of a `List<Utf8>` result (e.g. dimension names) as
/// strings. Returns `None` when the row is null.
pub fn list_utf8_row(result: &ArrayRef, row: usize) -> Option<Vec<String>> {
    let list = result
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("expected a ListArray result");
    if list.is_null(row) {
        return None;
    }
    let values = list.value(row);
    // Accept either Utf8 or Utf8View list items.
    if let Some(strings) = values.as_any().downcast_ref::<StringArray>() {
        Some(
            (0..strings.len())
                .map(|i| strings.value(i).to_string())
                .collect(),
        )
    } else if let Some(strings) = values.as_any().downcast_ref::<StringViewArray>() {
        Some(
            (0..strings.len())
                .map(|i| strings.value(i).to_string())
                .collect(),
        )
    } else {
        panic!("expected Utf8 or Utf8View list items")
    }
}

/// Extract one row of a `List<Int64>` result (e.g. a shape) as `i64`s.
/// Returns `None` when the row is null.
pub fn list_i64_row(result: &ArrayRef, row: usize) -> Option<Vec<i64>> {
    let list = result
        .as_any()
        .downcast_ref::<ListArray>()
        .expect("expected a ListArray result");
    if list.is_null(row) {
        return None;
    }
    let values = list.value(row);
    let ints = values
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("expected Int64 list items");
    Some((0..ints.len()).map(|i| ints.value(i)).collect())
}

/// Decode the pixel values of one band (1-based `band_number`) of one row of
/// a raster `StructArray`. `T` must match the band's data type.
pub fn band_pixels<T: PixelValue>(rasters: &StructArray, row: usize, band_number: usize) -> Vec<T> {
    use sedona_raster::traits::RasterRef;

    let array = RasterStructArray::try_new(rasters).unwrap();
    assert!(!array.is_null(row), "raster row {row} is null");
    let raster = array.get(row).expect("raster row");
    let bands = raster.bands();
    let band = bands.band(band_number).expect("band number (1-based)");
    assert_eq!(
        band.data_type(),
        T::DATA_TYPE,
        "band pixel type does not match the requested Rust type"
    );
    let buffer = band.nd_buffer().expect("band nd_buffer");
    let bytes = buffer.as_contiguous().expect("contiguous band bytes");
    bytes
        .chunks_exact(std::mem::size_of::<T>())
        .map(T::from_le_slice)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sedona_raster::traits::RasterRef;

    #[test]
    fn d2_defaults() {
        let raster = RasterSpec::d2(4, 5).band(BandDataType::Float32).build();
        let array = RasterStructArray::try_new(&raster).unwrap();
        assert_eq!(array.len(), 1);
        let raster_ref = array.get(0).unwrap();
        let metadata = raster_ref.metadata();
        assert_eq!(metadata.width(), 4);
        assert_eq!(metadata.height(), 5);
        assert_eq!(metadata.scale_x(), 1.0);
        assert_eq!(metadata.scale_y(), -1.0);

        let bands = raster_ref.bands();
        assert_eq!(bands.len(), 1);
        let band = bands.band(1).unwrap();
        assert_eq!(band.dim_names(), vec!["y", "x"]);
        assert_eq!(band.shape(), &[5, 4]);

        // Sequential default fill.
        assert_eq!(
            band_pixels::<f32>(&raster, 0, 1),
            (0..20).map(|i| i as f32).collect::<Vec<_>>()
        );
    }

    #[test]
    fn nd_spatial_inference() {
        let raster = RasterSpec::nd(&["time", "y", "x"], &[3, 4, 5])
            .band(BandDataType::Float32)
            .build();
        let array = RasterStructArray::try_new(&raster).unwrap();
        let raster_ref = array.get(0).unwrap();

        // Raster-level spatial metadata is X-first, like the readers emit.
        assert_eq!(raster_ref.metadata().width(), 5);
        assert_eq!(raster_ref.metadata().height(), 4);

        let bands = raster_ref.bands();
        let band = bands.band(1).unwrap();
        assert_eq!(band.dim_names(), vec!["time", "y", "x"]);
        assert_eq!(band.shape(), &[3, 4, 5]);
    }

    #[test]
    #[should_panic(expected = "trailing two dims to be a spatial pair")]
    fn nd_rejects_non_spatial_trailing_pair() {
        RasterSpec::nd(&["x", "y", "time"], &[5, 4, 3]);
    }

    #[test]
    fn mixed_and_conflicting_band_dims_build() {
        // One 2-D band next to one 3-D band.
        let mixed = RasterSpec::nd(&["time", "y", "x"], &[3, 4, 5])
            .band_nd(&["y", "x"], &[4, 5], BandDataType::Float32)
            .band(BandDataType::Float32)
            .build();
        let array = RasterStructArray::try_new(&mixed).unwrap();
        let raster_ref = array.get(0).unwrap();
        assert_eq!(raster_ref.bands().len(), 2);

        // Two bands disagreeing on a non-spatial dim size.
        let conflicting = RasterSpec::nd(&["time", "y", "x"], &[3, 4, 5])
            .band(BandDataType::Float32)
            .band_nd(&["time", "y", "x"], &[7, 4, 5], BandDataType::Float32)
            .build();
        let array = RasterStructArray::try_new(&conflicting).unwrap();
        let bands = array.get(0).unwrap();
        let bands = bands.bands();
        assert_eq!(bands.band(1).unwrap().shape(), &[3, 4, 5]);
        assert_eq!(bands.band(2).unwrap().shape(), &[7, 4, 5]);
    }

    #[test]
    fn band_values_and_nodata() {
        let raster = RasterSpec::d2(2, 2)
            .band_values(&[10u16, 20, 30, 40])
            .nodata(0u16)
            .name("temperature")
            .build();
        let array = RasterStructArray::try_new(&raster).unwrap();
        let raster_ref = array.get(0).unwrap();
        let bands = raster_ref.bands();
        let band = bands.band(1).unwrap();
        assert_eq!(band.data_type(), BandDataType::UInt16);
        assert_eq!(band.nodata(), Some(&[0u8, 0u8][..]));
        assert_eq!(band_pixels::<u16>(&raster, 0, 1), vec![10, 20, 30, 40]);
    }

    #[test]
    #[should_panic(expected = "expected 4 values for shape")]
    fn band_values_length_checked() {
        RasterSpec::d2(2, 2).band_values(&[1.0f32, 2.0]);
    }

    #[test]
    fn outdb_band() {
        let raster = RasterSpec::d2(4, 4)
            .band(BandDataType::Float32)
            .outdb("s3://bucket/raster.tif#band=2", None)
            .build();
        let array = RasterStructArray::try_new(&raster).unwrap();
        let raster_ref = array.get(0).unwrap();
        let bands = raster_ref.bands();
        let band = bands.band(1).unwrap();
        assert!(!band.is_indb());
        let metadata = band.metadata();
        assert_eq!(metadata.outdb_url(), Some("s3://bucket/raster.tif"));
        assert_eq!(metadata.outdb_band_id(), Some(2));
    }

    #[test]
    fn raster_array_with_nulls() {
        let array = raster_array(vec![
            Some(RasterSpec::d2(2, 2).band(BandDataType::UInt8)),
            None,
            Some(RasterSpec::d2(3, 3).band(BandDataType::UInt8)),
        ]);
        let rasters = RasterStructArray::try_new(&array).unwrap();
        assert_eq!(rasters.len(), 3);
        assert!(!rasters.is_null(0));
        assert!(rasters.is_null(1));
        assert!(!rasters.is_null(2));
    }

    #[test]
    fn rasters_equal_assertion() {
        let specs = vec![Some(RasterSpec::d2(2, 2).band(BandDataType::UInt8)), None];
        let actual: ArrayRef = Arc::new(raster_array(specs.clone()));
        assert_rasters_equal(&actual, &specs);

        let spec = RasterSpec::d2(2, 2).band(BandDataType::UInt8);
        assert_raster_scalar_equals(&spec.scalar(), &spec);
    }

    #[test]
    #[should_panic(expected = "Band data does not match")]
    fn rasters_equal_detects_pixel_difference() {
        let actual: ArrayRef = Arc::new(raster_array(vec![Some(
            RasterSpec::d2(2, 2).band_values(&[1u8, 2, 3, 4]),
        )]));
        assert_rasters_equal(
            &actual,
            &[Some(RasterSpec::d2(2, 2).band_values(&[1u8, 2, 3, 5]))],
        );
    }

    #[test]
    #[should_panic(expected = "null-ness does not match")]
    fn rasters_equal_detects_null_mismatch() {
        let actual: ArrayRef = Arc::new(raster_array(vec![Some(
            RasterSpec::d2(2, 2).band(BandDataType::UInt8),
        )]));
        assert_rasters_equal(&actual, &[None]);
    }

    #[test]
    fn list_extractors() {
        use arrow_array::builder::{Int64Builder, ListBuilder, StringBuilder};

        let mut names = ListBuilder::new(StringBuilder::new());
        names.values().append_value("time");
        names.values().append_value("y");
        names.append(true);
        names.append_null();
        let names: ArrayRef = Arc::new(names.finish());
        assert_eq!(
            list_utf8_row(&names, 0),
            Some(vec!["time".to_string(), "y".to_string()])
        );
        assert_eq!(list_utf8_row(&names, 1), None);

        let mut shape = ListBuilder::new(Int64Builder::new());
        shape.values().append_value(3);
        shape.values().append_value(4);
        shape.append(true);
        let shape: ArrayRef = Arc::new(shape.finish());
        assert_eq!(list_i64_row(&shape, 0), Some(vec![3, 4]));
    }
}
