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

//! `RS_Value` — sample a raster's pixel value at a point or grid cell.
//!
//! ```text
//! RS_Value(raster, point)              -> Double  -- band defaults to 1
//! RS_Value(raster, point, band)        -> Double
//! RS_Value(raster, colX, rowY)         -> Double  -- 1-based grid coords, band 1
//! RS_Value(raster, colX, rowY, band)   -> Double
//! ```
//!
//! Returns the value of the pixel the point falls in (nearest pixel, no
//! resampling), or the value at the given 1-based grid cell. The result is
//! `NULL` when the raster/arguments are null, the point/cell is out of bounds,
//! or the value equals the band's nodata.
//!
//! The function is tagged [`NEEDS_PIXELS_METADATA_KEY`], so the planner wraps
//! its raster argument in `RS_EnsureLoaded`; by the time a kernel runs the band
//! bytes are materialised InDb and a value is read directly from the band's
//! [`NdBuffer`](sedona_raster::traits::NdBuffer) — no GDAL involved. Only 2-D
//! rasters are supported; a band with extra (non-spatial) dimensions errors.

use std::sync::Arc;

use arrow_array::builder::Float64Builder;
use arrow_schema::DataType;
use datafusion_common::cast::as_int32_array;
use datafusion_common::{exec_datafusion_err, exec_err, DataFusionError, Result};
use datafusion_expr::{ColumnarValue, Volatility};
use geo_traits::{CoordTrait, GeometryTrait, GeometryType, PointTrait};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_proj::transform::with_global_proj_engine;
use sedona_raster::affine_transformation::to_raster_coordinate;
use sedona_raster::traits::{nodata_bytes_to_f64_lossless, RasterRef};
use sedona_schema::crs::CrsRef;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};
use wkb::reader::read_wkb;

use crate::crs_utils::{crs_transform_wkb, resolve_crs};
use crate::executor::RasterExecutor;
use crate::rs_ensure_loaded::NEEDS_PIXELS_METADATA_KEY;

/// `RS_Value()` scalar UDF — sample a pixel value at a point or grid cell.
pub fn rs_value_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_value",
        vec![
            Arc::new(RsValuePoint { with_band: false }), // RS_Value(raster, point)
            Arc::new(RsValuePoint { with_band: true }),  // RS_Value(raster, point, band)
            Arc::new(RsValueGrid { with_band: false }),  // RS_Value(raster, colX, rowY)
            Arc::new(RsValueGrid { with_band: true }),   // RS_Value(raster, colX, rowY, band)
        ],
        Volatility::Immutable,
    )
    // The kernels read pixel bytes, so the raster argument must be materialised
    // InDb first; the planner injects RS_EnsureLoaded based on this flag.
    .with_metadata(NEEDS_PIXELS_METADATA_KEY, "true")
}

/// Kernel for `RS_Value(raster, point[, band])`.
#[derive(Debug)]
struct RsValuePoint {
    with_band: bool,
}

impl SedonaScalarKernel for RsValuePoint {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let mut matchers = vec![
            ArgMatcher::is_raster(),
            ArgMatcher::is_geometry_or_geography(),
        ];
        if self.with_band {
            matchers.push(ArgMatcher::is_integer());
        }
        let matcher = ArgMatcher::new(matchers, SedonaType::Arrow(DataType::Float64));
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let num_iterations = executor.num_iterations();
        let mut builder = Float64Builder::with_capacity(num_iterations);

        // The optional band argument, materialised once as an Int32 array.
        let band_array = if self.with_band {
            Some(
                as_int32_array(
                    &args[2]
                        .clone()
                        .cast_to(&DataType::Int32, None)?
                        .into_array(num_iterations)?,
                )?
                .clone(),
            )
        } else {
            None
        };
        let mut band_iter = band_array.as_ref().map(|a| a.iter());

        // Reprojecting the point into the raster CRS needs a PROJ engine.
        with_global_proj_engine(|engine| {
            executor.execute_raster_wkb_crs_void(|raster_opt, wkb_opt, point_crs| {
                let band_num = next_band(&mut band_iter);

                let (raster, point_wkb) = match (raster_opt, wkb_opt) {
                    (Some(raster), Some(point_wkb)) => (raster, point_wkb),
                    _ => {
                        builder.append_null();
                        return Ok(());
                    }
                };

                // Bring the point into the raster's CRS. A reprojection only
                // happens when both sides carry a (differing) CRS; otherwise
                // the original WKB is sampled directly.
                let raster_crs = resolve_crs(raster.crs())?;
                let reprojected =
                    reproject_point(point_wkb, point_crs, raster_crs.as_deref(), engine)?;
                let wkb = reprojected.as_deref().unwrap_or(point_wkb);

                let (x, y) = read_point_xy(wkb)?;
                let (col, row) = to_raster_coordinate(raster, x, y)
                    .map_err(|e| exec_datafusion_err!("RS_Value: {e}"))?;

                match sample_pixel(raster, col, row, band_num)? {
                    Some(value) => builder.append_value(value),
                    None => builder.append_null(),
                }
                Ok(())
            })
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

/// Kernel for `RS_Value(raster, colX, rowY[, band])` with **1-based** grid
/// coordinates.
#[derive(Debug)]
struct RsValueGrid {
    with_band: bool,
}

impl SedonaScalarKernel for RsValueGrid {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let mut matchers = vec![
            ArgMatcher::is_raster(),
            ArgMatcher::is_integer(),
            ArgMatcher::is_integer(),
        ];
        if self.with_band {
            matchers.push(ArgMatcher::is_integer());
        }
        let matcher = ArgMatcher::new(matchers, SedonaType::Arrow(DataType::Float64));
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let num_iterations = executor.num_iterations();
        let mut builder = Float64Builder::with_capacity(num_iterations);

        let col_array = as_int32_array(
            &args[1]
                .clone()
                .cast_to(&DataType::Int32, None)?
                .into_array(num_iterations)?,
        )?
        .clone();
        let row_array = as_int32_array(
            &args[2]
                .clone()
                .cast_to(&DataType::Int32, None)?
                .into_array(num_iterations)?,
        )?
        .clone();
        let band_array = if self.with_band {
            Some(
                as_int32_array(
                    &args[3]
                        .clone()
                        .cast_to(&DataType::Int32, None)?
                        .into_array(num_iterations)?,
                )?
                .clone(),
            )
        } else {
            None
        };

        let mut col_iter = col_array.iter();
        let mut row_iter = row_array.iter();
        let mut band_iter = band_array.as_ref().map(|a| a.iter());

        executor.execute_raster_void(|_, raster_opt| {
            let col = col_iter.next().flatten();
            let row = row_iter.next().flatten();
            let band_num = next_band(&mut band_iter);

            match (raster_opt, col, row) {
                (Some(raster), Some(col), Some(row)) => {
                    // 1-based grid coordinates -> 0-based pixel indices.
                    match sample_pixel(raster, col as i64 - 1, row as i64 - 1, band_num)? {
                        Some(value) => builder.append_value(value),
                        None => builder.append_null(),
                    }
                }
                _ => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

/// Advance a band-number iterator one row, returning a clamped 1-based band
/// number. Defaults to band 1 when the argument is absent or null and clamps
/// non-positive inputs up to 1, matching the legacy behaviour.
fn next_band(
    band_iter: &mut Option<arrow_array::iterator::ArrayIter<&arrow_array::Int32Array>>,
) -> usize {
    match band_iter.as_mut() {
        Some(iter) => iter.next().flatten().unwrap_or(1).max(1) as usize,
        None => 1,
    }
}

/// Reproject `point_wkb` from its CRS into the raster CRS, returning the
/// transformed WKB only when a reprojection actually happened (so the caller
/// can sample the original bytes otherwise — no allocation in the common case).
///
/// Errors if exactly one of the point / raster carries a CRS: sampling across a
/// known and an unknown CRS would silently mislocate the point.
fn reproject_point(
    point_wkb: &[u8],
    point_crs: CrsRef<'_>,
    raster_crs: CrsRef<'_>,
    engine: &dyn sedona_geometry::transform::CrsEngine,
) -> Result<Option<Vec<u8>>> {
    match (point_crs, raster_crs) {
        (Some(point_crs), Some(raster_crs)) => {
            if point_crs.crs_equals(raster_crs) {
                Ok(None)
            } else {
                Ok(Some(crs_transform_wkb(
                    point_wkb, point_crs, raster_crs, engine,
                )?))
            }
        }
        (None, None) => Ok(None),
        (Some(_), None) => {
            exec_err!("RS_Value: point has a CRS but the raster does not")
        }
        (None, Some(_)) => {
            exec_err!("RS_Value: raster has a CRS but the point does not")
        }
    }
}

/// Read the (x, y) coordinates of a WKB Point geometry.
fn read_point_xy(wkb: &[u8]) -> Result<(f64, f64)> {
    let geom = read_wkb(wkb).map_err(|e| DataFusionError::External(Box::new(e)))?;
    match geom.as_type() {
        GeometryType::Point(point) => {
            let coord = point
                .coord()
                .ok_or_else(|| exec_datafusion_err!("RS_Value: empty point geometry"))?;
            Ok((coord.x(), coord.y()))
        }
        _ => exec_err!("RS_Value expects a Point geometry"),
    }
}

/// Sample band `band_num` (1-based) at 0-based pixel `(col, row)` as `f64`.
///
/// Returns `None` when the pixel is out of bounds or equals the band's nodata.
/// Reads exactly one pixel by computing its byte offset from the band's
/// [`NdBuffer`](sedona_raster::traits::NdBuffer) strides — zero-copy and O(1),
/// no whole-band materialisation. Errors if the band index is out of range or
/// the band is not 2-D.
fn sample_pixel(
    raster: &dyn RasterRef,
    col: i64,
    row: i64,
    band_num: usize,
) -> Result<Option<f64>> {
    let band = raster
        .bands()
        .band(band_num)
        .map_err(|e| exec_datafusion_err!("RS_Value: {e}"))?;
    let buffer = band
        .nd_buffer()
        .map_err(|e| exec_datafusion_err!("RS_Value: {e}"))?;

    if buffer.shape.len() != 2 {
        return exec_err!(
            "RS_Value supports 2-D rasters only; band has {} dimensions",
            buffer.shape.len()
        );
    }
    let (height, width) = (buffer.shape[0], buffer.shape[1]);
    if row < 0 || row >= height || col < 0 || col >= width {
        return Ok(None);
    }

    // Byte offset of the (row, col) pixel via the band's own strides, so the
    // read stays correct for any layout the producer hands us.
    let byte_offset = buffer.offset as i64 + row * buffer.strides[0] + col * buffer.strides[1];
    let size = buffer.data_type.byte_size() as i64;
    let start = usize::try_from(byte_offset)
        .map_err(|_| exec_datafusion_err!("RS_Value: negative pixel byte offset"))?;
    let end = usize::try_from(byte_offset + size)
        .map_err(|_| exec_datafusion_err!("RS_Value: pixel byte offset overflow"))?;
    let bytes = buffer.buffer.get(start..end).ok_or_else(|| {
        exec_datafusion_err!("RS_Value: pixel is out of the band's buffer bounds")
    })?;

    let value = nodata_bytes_to_f64_lossless(bytes, &buffer.data_type)
        .map_err(|e| exec_datafusion_err!("RS_Value: {e}"))?;

    if let Some(nodata) = band
        .nodata_as_f64()
        .map_err(|e| exec_datafusion_err!("RS_Value: {e}"))?
    {
        if value == nodata || (value.is_nan() && nodata.is_nan()) {
            return Ok(None);
        }
    }

    Ok(Some(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_expr::ScalarUDF;
    use sedona_raster::array::RasterStructArray;
    use sedona_schema::datatypes::RASTER;
    use sedona_schema::raster::BandDataType;
    use sedona_testing::raster_spec::RasterSpec;

    /// Resolve a single `RasterRefImpl` from a one-row spec for direct
    /// `sample_pixel` exercise.
    fn sample(spec: RasterSpec, col: i64, row: i64, band: usize) -> Result<Option<f64>> {
        let array = spec.build();
        let rasters = RasterStructArray::try_new(&array).unwrap();
        let raster = rasters.get(0).unwrap();
        sample_pixel(&raster, col, row, band)
    }

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = rs_value_udf().into();
        assert_eq!(udf.name(), "rs_value");
    }

    #[test]
    fn udf_marks_needs_pixels() {
        assert_eq!(
            rs_value_udf()
                .metadata()
                .get(NEEDS_PIXELS_METADATA_KEY)
                .map(String::as_str),
            Some("true")
        );
    }

    #[test]
    fn return_type_is_float64() {
        // Grid (raster, int, int) resolves to a Float64 output.
        let return_type = RsValueGrid { with_band: false }
            .return_type(&[
                RASTER,
                SedonaType::Arrow(DataType::Int32),
                SedonaType::Arrow(DataType::Int32),
            ])
            .unwrap();
        assert_eq!(return_type, Some(SedonaType::Arrow(DataType::Float64)));
    }

    #[test]
    fn samples_2d_pixels_row_major() {
        // 3x2 raster, row-major pixels:
        //   row0 = [10, 20, 30], row1 = [40, 50, 60]
        let spec = || RasterSpec::d2(3, 2).band_values(&[10u8, 20, 30, 40, 50, 60]);
        assert_eq!(sample(spec(), 0, 0, 1).unwrap(), Some(10.0)); // top-left
        assert_eq!(sample(spec(), 2, 0, 1).unwrap(), Some(30.0)); // top-right
        assert_eq!(sample(spec(), 0, 1, 1).unwrap(), Some(40.0)); // bottom-left
        assert_eq!(sample(spec(), 2, 1, 1).unwrap(), Some(60.0)); // bottom-right
    }

    #[test]
    fn out_of_bounds_pixel_is_none() {
        let spec = || RasterSpec::d2(3, 2).band_values(&[10u8, 20, 30, 40, 50, 60]);
        assert_eq!(sample(spec(), 3, 0, 1).unwrap(), None); // col == width
        assert_eq!(sample(spec(), 0, 2, 1).unwrap(), None); // row == height
        assert_eq!(sample(spec(), -1, 0, 1).unwrap(), None); // negative
    }

    #[test]
    fn nodata_pixel_is_none() {
        let spec = RasterSpec::d2(2, 1).band_values(&[7u8, 9]).nodata(9u8);
        assert_eq!(sample(spec.clone(), 0, 0, 1).unwrap(), Some(7.0));
        assert_eq!(sample(spec, 1, 0, 1).unwrap(), None);
    }

    #[test]
    fn second_band_is_addressable() {
        let spec = RasterSpec::d2(2, 1)
            .band_values(&[1u8, 2])
            .band_values(&[30u8, 40]);
        assert_eq!(sample(spec.clone(), 1, 0, 1).unwrap(), Some(2.0));
        assert_eq!(sample(spec, 1, 0, 2).unwrap(), Some(40.0));
    }

    #[test]
    fn float_band_values_round_trip() {
        let spec = RasterSpec::d2(2, 1).band_values(&[1.5f32, -2.5]);
        assert_eq!(sample(spec.clone(), 0, 0, 1).unwrap(), Some(1.5));
        assert_eq!(sample(spec, 1, 0, 1).unwrap(), Some(-2.5));
    }

    #[test]
    fn band_out_of_range_errors() {
        let spec = RasterSpec::d2(2, 1).band_values(&[1u8, 2]);
        let err = sample(spec, 0, 0, 2).unwrap_err().to_string();
        assert!(err.contains("RS_Value"), "unexpected error: {err}");
    }

    #[test]
    fn non_2d_band_errors() {
        // A band with a leading non-spatial dimension is rejected.
        let spec = RasterSpec::nd(&["time", "y", "x"], &[2, 2, 1]).band(BandDataType::UInt8);
        let err = sample(spec, 0, 0, 1).unwrap_err().to_string();
        assert!(err.contains("2-D"), "unexpected error: {err}");
    }
}
