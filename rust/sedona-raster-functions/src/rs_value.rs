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

//! `RS_Value` — sample a raster's pixel value at a point.
//!
//! ```text
//! RS_Value(raster, point)        -> Double  -- band defaults to 1
//! RS_Value(raster, point, band)  -> Double
//! ```
//!
//! Returns the value of the pixel that contains the point (no resampling). The
//! result is `NULL` when the raster/arguments are null, the point is empty, the
//! point is out of bounds, or the value equals the band's nodata.
//!
//! The function is tagged [`NEEDS_PIXELS_METADATA_KEY`], so the planner wraps
//! its raster argument in `RS_EnsureLoaded`; by the time a kernel runs the band
//! bytes are materialised InDb and a value is read directly from the band's
//! [`NdBuffer`](sedona_raster::traits::NdBuffer) — no GDAL involved. Only 2-D
//! rasters are supported; a band with extra (non-spatial) dimensions errors.

use std::{borrow::Cow, sync::Arc};

use arrow_array::{builder::Float64Builder, Array, ArrayRef, Float64Array, StructArray};
use arrow_schema::DataType;
use datafusion_common::cast::as_int32_array;
use datafusion_common::{exec_datafusion_err, exec_err, Result, ScalarValue};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::wkb_header::read_point_xy;
use sedona_proj::transform::with_global_proj_engine;
use sedona_raster::affine_transformation::AffineMatrix;
use sedona_raster::array::RasterStructArray;
use sedona_raster::traits::{nodata_bytes_to_f64_lossless, NdBuffer, RasterRef};
use sedona_schema::crs::CrsRef;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::crs_utils::{align_wkb_to_crs, crs_transform_required, resolve_crs};
use crate::executor::RasterExecutor;
use crate::rs_ensure_loaded::NEEDS_PIXELS_METADATA_KEY;

/// `RS_Value()` scalar UDF — sample a pixel value at a point.
pub fn rs_value_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_value",
        vec![
            Arc::new(RsValuePoint { with_band: false }), // RS_Value(raster, point)
            Arc::new(RsValuePoint { with_band: true }),  // RS_Value(raster, point, band)
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
        // Fast path: a constant (scalar) raster lets us resolve the affine
        // transform and CRS once for the whole batch instead of per point — the
        // common RS_Value(raster_expr, point_column[, band]) shape. The band
        // argument does not change this: a constant band also hoists its buffer,
        // and only a band *column* falls back to per-row band resolution.
        if let ColumnarValue::Scalar(ScalarValue::Struct(raster_struct)) = &args[0] {
            return self.invoke_scalar_raster(arg_types, args, raster_struct.as_ref());
        }

        let executor = RasterExecutor::new(arg_types, args);
        let num_iterations = executor.num_iterations();
        let mut builder = Float64Builder::with_capacity(num_iterations);

        // The optional band argument, materialised once as an Int32 array. Held
        // as an `ArrayRef` so the typed view below borrows it instead of cloning
        // the typed `Int32Array`.
        let band_arr = if self.with_band {
            Some(int32_array_arg(&args[2], num_iterations)?)
        } else {
            None
        };
        let band_array = band_arr.as_ref().map(|a| as_int32_array(a)).transpose()?;
        let mut band_iter = band_array.map(|a| a.iter());

        // Reprojecting the point into the raster CRS needs a PROJ engine.
        with_global_proj_engine(|engine| {
            executor.execute_raster_wkb_crs_void(|raster_opt, wkb_opt, point_crs| {
                let (raster, point_wkb, band_num) =
                    match (raster_opt, wkb_opt, next_band(&mut band_iter)) {
                        (Some(raster), Some(point_wkb), Some(band_num)) => {
                            (raster, point_wkb, band_num)
                        }
                        _ => {
                            builder.append_null();
                            return Ok(());
                        }
                    };

                // Parse the point and bring it into the raster's CRS. Null/empty
                // points (and non-finite coordinates) have no location to sample.
                let raster_crs = resolve_crs(raster.crs())?;
                let Some((x, y)) =
                    resolve_point_xy(point_wkb, point_crs, raster_crs.as_deref(), false, engine)?
                else {
                    builder.append_null();
                    return Ok(());
                };

                let affine = AffineMatrix::from_metadata(&raster.metadata());
                match xy_to_pixel(&affine, x, y)? {
                    Some((col, row)) => match sample_pixel(raster, col, row, band_num)? {
                        Some(value) => builder.append_value(value),
                        None => builder.append_null(),
                    },
                    None => builder.append_null(),
                }
                Ok(())
            })
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

impl RsValuePoint {
    /// Optimized path for a constant (scalar) raster: the affine transform and
    /// raster CRS are resolved once for the whole batch, then a selection vector
    /// drives a tight sample loop. This serves every band shape:
    /// - no band argument or a constant band → the band buffer is hoisted too;
    /// - a band *column* → the band buffer is resolved per row (its `NdBuffer`
    ///   borrows from the band, which can't be cached across distinct bands),
    ///   but the affine/CRS/reproject work is still hoisted.
    ///
    /// Sampling behaviour matches the general path: it uses the same
    /// [`resolve_point_xy`]/[`xy_to_pixel`] helpers, so null/empty/non-finite
    /// points yield NULL identically. Band resolution (and its 2-D check) is
    /// deferred until at least one point needs sampling, so an all-null/all-empty
    /// batch returns NULL without touching the band — as the general path does.
    fn invoke_scalar_raster(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
        raster_struct: &StructArray,
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let n = executor.num_iterations();

        let rasters = RasterStructArray::try_new(raster_struct)?;
        if rasters.is_null(0) {
            // A NULL raster makes every output NULL.
            return executor.finish(Arc::new(Float64Array::from(vec![None; n])));
        }
        let raster = rasters.get(0)?;

        // Band selection: a missing band argument (default 1) or a scalar band is
        // constant for the batch and lets us hoist the band buffer; a band column
        // is resolved per row. A NULL scalar band makes every output NULL.
        let mut const_band: Option<usize> = None;
        let mut band_values: Option<ArrayRef> = None;
        if self.with_band {
            match &args[2] {
                ColumnarValue::Scalar(scalar) => {
                    let arr = ColumnarValue::Scalar(scalar.clone())
                        .cast_to(&DataType::Int32, None)?
                        .into_array(1)?;
                    let arr = as_int32_array(&arr)?;
                    if arr.is_null(0) {
                        return executor.finish(Arc::new(Float64Array::from(vec![None; n])));
                    }
                    // Match `next_band`: clamp to 0 so band 0/negative surface as a
                    // not-1-based error from `Bands::band` rather than being coerced.
                    const_band = Some(arr.value(0).max(0) as usize);
                }
                other => band_values = Some(int32_array_arg(other, n)?),
            }
        } else {
            const_band = Some(1);
        }
        let band_array = band_values
            .as_ref()
            .map(|a| as_int32_array(a))
            .transpose()?;

        // Affine transform and raster CRS are shared by every point in this batch.
        let affine = AffineMatrix::from_metadata(&raster.metadata());
        let raster_crs = resolve_crs(raster.crs())?;
        // A type-level CRS is constant for this batch, so avoid resolving and
        // comparing it for every point when alignment cannot change coordinates.
        let skip_reproject =
            column_reproject_decision(&arg_types[1], raster_crs.as_deref())? == Some(false);

        let mut geom = executor.make_geom_wkb_crs_accessor(1)?;

        // Phase 1 — selection vector: collect (row, x, y, band) for the points
        // worth sampling (non-null, non-empty Point with a non-null band),
        // reprojected into the raster CRS. Skipped rows stay NULL in the output.
        let mut selection: Vec<(usize, f64, f64, usize)> = Vec::with_capacity(n);
        let mut band_iter = band_array.map(|a| a.iter());
        with_global_proj_engine(|engine| {
            for i in 0..n {
                // Advance the band column in lockstep with the row index; a NULL
                // band element leaves the row NULL (matching the general path).
                let band_num = match const_band {
                    Some(b) => b,
                    None => match next_band(&mut band_iter) {
                        Some(b) => b,
                        None => continue,
                    },
                };
                let (wkb_opt, point_crs) = geom.get(i)?;
                let Some(point_wkb) = wkb_opt else {
                    continue;
                };
                if let Some((x, y)) = resolve_point_xy(
                    point_wkb,
                    point_crs,
                    raster_crs.as_deref(),
                    skip_reproject,
                    engine,
                )? {
                    selection.push((i, x, y, band_num));
                }
            }
            Ok(())
        })?;

        let mut out: Vec<Option<f64>> = vec![None; n];
        if selection.is_empty() {
            return executor.finish(Arc::new(Float64Array::from(out)));
        }

        // Phase 2 — sample. A constant band resolves its buffer/nodata once (now
        // that we know a point needs sampling); a band column resolves per row.
        match const_band {
            Some(band_num) => {
                let band = raster
                    .bands()
                    .band(band_num)
                    .map_err(|e| exec_datafusion_err!("RS_Value: {e}"))?;
                if !band.is_spatial_2d() {
                    return exec_err!(
                        "RS_Value supports 2-D rasters only; band is not a 2-D (y, x) grid"
                    );
                }
                let buffer = band
                    .nd_buffer()
                    .map_err(|e| exec_datafusion_err!("RS_Value: {e}"))?;
                let nodata = band
                    .nodata_as_f64()
                    .map_err(|e| exec_datafusion_err!("RS_Value: {e}"))?;
                for (i, x, y, _band) in selection {
                    if let Some((col, row)) = xy_to_pixel(&affine, x, y)? {
                        out[i] = read_pixel(&buffer, nodata, col, row)?;
                    }
                }
            }
            None => {
                for (i, x, y, band_num) in selection {
                    if let Some((col, row)) = xy_to_pixel(&affine, x, y)? {
                        out[i] = sample_pixel(&raster, col, row, band_num)?;
                    }
                }
            }
        }

        executor.finish(Arc::new(Float64Array::from(out)))
    }
}

/// Materialise an integer argument as an owned `Int32` [`ArrayRef`] for the
/// batch. Callers keep the returned `ArrayRef` alive and borrow a typed
/// `&Int32Array` view from it (via `as_int32_array`) rather than cloning the
/// typed array.
fn int32_array_arg(arg: &ColumnarValue, num_iterations: usize) -> Result<ArrayRef> {
    arg.clone()
        .cast_to(&DataType::Int32, None)?
        .into_array(num_iterations)
}

/// Advance the optional band-number iterator one row, yielding the 1-based band
/// to sample. A missing band argument defaults to band 1; a NULL band element
/// returns `None`, which the caller propagates to a NULL result. Band 0 and
/// negative values map to 0 so [`Bands::band`](sedona_raster::traits::Bands::band)
/// rejects them as not 1-based rather than being silently coerced.
fn next_band(
    band_iter: &mut Option<arrow_array::iterator::ArrayIter<&arrow_array::Int32Array>>,
) -> Option<usize> {
    match band_iter.as_mut() {
        None => Some(1),
        Some(iter) => iter.next().flatten().map(|b| b.max(0) as usize),
    }
}

/// Parse a Point WKB and return its `(x, y)` in the raster's CRS, or `None` when
/// there is nothing to sample (the point is empty — both ordinates NaN).
///
/// `skip_reproject` is the scalar-raster fast path's column-level decision. Item
/// CRS values always pass `false` because their CRS can vary by row.
fn resolve_point_xy(
    point_wkb: &[u8],
    point_crs: CrsRef<'_>,
    raster_crs: CrsRef<'_>,
    skip_reproject: bool,
    engine: &dyn sedona_geometry::transform::CrsEngine,
) -> Result<Option<(f64, f64)>> {
    let Some((px, py)) =
        read_point_xy(point_wkb).map_err(|e| exec_datafusion_err!("RS_Value: {e}"))?
    else {
        return Ok(None);
    };

    if skip_reproject {
        return Ok(Some((px, py)));
    }

    match align_wkb_to_crs(point_wkb, point_crs, raster_crs, "point", "raster", engine)? {
        Cow::Borrowed(_) => Ok(Some((px, py))),
        Cow::Owned(reprojected) => read_point_xy(&reprojected)
            .map_err(|e| exec_datafusion_err!("RS_Value: {e}"))?
            .ok_or_else(|| exec_datafusion_err!("RS_Value: reprojected point is empty"))
            .map(Some),
    }
}

/// For a column-level geometry CRS, decide once whether every point can skip
/// CRS alignment. Item-level CRS values vary by row and must be handled by the
/// shared alignment helper for each point.
fn column_reproject_decision(
    point_type: &SedonaType,
    raster_crs: CrsRef<'_>,
) -> Result<Option<bool>> {
    let point_crs = match point_type {
        SedonaType::Wkb(_, crs) | SedonaType::WkbView(_, crs) => crs.as_deref(),
        _ => return Ok(None),
    };

    crs_transform_required(point_crs, raster_crs, "point", "raster").map(Some)
}

/// Map a coordinate in the raster's CRS to a 0-based `(col, row)` pixel index,
/// or `None` when the point has no location to sample.
///
/// A non-finite coordinate (a Point with a NaN/Inf ordinate, e.g. `POINT(NaN 5)`)
/// returns `None`: without this guard a NaN would survive `inv_transform` and the
/// saturating `f64 -> i64` cast would turn it into 0 (in bounds), silently
/// sampling pixel column 0 rather than yielding NULL.
fn xy_to_pixel(affine: &AffineMatrix, x: f64, y: f64) -> Result<Option<(i64, i64)>> {
    if !x.is_finite() || !y.is_finite() {
        return Ok(None);
    }
    let (raster_x, raster_y) = affine
        .inv_transform(x, y)
        .map_err(|e| exec_datafusion_err!("RS_Value: {e}"))?;
    // Floor (not truncate toward zero) so a point just outside the top/left edge
    // maps to a negative index and is rejected as out of bounds, rather than
    // truncating to 0 and sampling an edge pixel. The `f64 -> i64` cast saturates
    // (never panics); the bounds check in `read_pixel`/`sample_pixel` rejects an
    // out-of-range index as NULL.
    Ok(Some((raster_x.floor() as i64, raster_y.floor() as i64)))
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

    // 2-D only: the band must be a recognized spatial (y, x) grid, not just any
    // two-axis band (e.g. (time, band) would have len 2 but no spatial meaning).
    if !band.is_spatial_2d() {
        return exec_err!("RS_Value supports 2-D rasters only; band is not a 2-D (y, x) grid");
    }
    let buffer = band
        .nd_buffer()
        .map_err(|e| exec_datafusion_err!("RS_Value: {e}"))?;
    let nodata = band
        .nodata_as_f64()
        .map_err(|e| exec_datafusion_err!("RS_Value: {e}"))?;
    read_pixel(&buffer, nodata, col, row)
}

/// Read pixel `(col, row)` from an already-resolved band buffer and nodata
/// value. Returns `None` for an out-of-bounds pixel or one that equals nodata.
/// Hoisting the band resolution out of this function lets callers sample many
/// points from one buffer without re-resolving per point.
fn read_pixel(buffer: &NdBuffer, nodata: Option<f64>, col: i64, row: i64) -> Result<Option<f64>> {
    let (height, width) = (buffer.shape[0], buffer.shape[1]);
    if row < 0 || row >= height || col < 0 || col >= width {
        return Ok(None);
    }

    // Byte offset of the (row, col) pixel via the band's own strides, so the
    // read stays correct for any layout the producer hands us. Checked
    // arithmetic throughout: `row`/`col` are already in bounds, but a corrupt
    // stride or offset must surface as an error, never an i64 overflow panic.
    let size = buffer.data_type.byte_size() as i64;
    let byte_offset = row
        .checked_mul(buffer.strides[0])
        .zip(col.checked_mul(buffer.strides[1]))
        .and_then(|(r, c)| r.checked_add(c))
        .and_then(|rc| rc.checked_add(buffer.offset as i64))
        .ok_or_else(|| exec_datafusion_err!("RS_Value: pixel byte offset overflow"))?;
    let end_offset = byte_offset
        .checked_add(size)
        .ok_or_else(|| exec_datafusion_err!("RS_Value: pixel byte offset overflow"))?;
    let start = usize::try_from(byte_offset)
        .map_err(|_| exec_datafusion_err!("RS_Value: negative pixel byte offset"))?;
    let end = usize::try_from(end_offset)
        .map_err(|_| exec_datafusion_err!("RS_Value: pixel byte offset overflow"))?;
    let bytes = buffer.buffer.get(start..end).ok_or_else(|| {
        exec_datafusion_err!("RS_Value: pixel is out of the band's buffer bounds")
    })?;

    // Decode the pixel to f64. The lossless converter errors (rather than
    // silently rounding) on Int64/UInt64 values beyond f64's exact-integer
    // range (2^53) — RS_Value returns a Double, so such a pixel can't be
    // represented faithfully; failing loudly is preferred over a wrong value.
    let value = nodata_bytes_to_f64_lossless(bytes, &buffer.data_type)
        .map_err(|e| exec_datafusion_err!("RS_Value: {e}"))?;

    if let Some(nodata) = nodata {
        if value == nodata || (value.is_nan() && nodata.is_nan()) {
            return Ok(None);
        }
    }

    Ok(Some(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, Float64Array, Int32Array};
    use datafusion_expr::ScalarUDF;
    use sedona_raster::array::RasterStructArray;
    use sedona_schema::crs::lnglat;
    use sedona_schema::datatypes::{Edges, RASTER};
    use sedona_schema::raster::BandDataType;
    use sedona_testing::create::create_array as create_geom_array;
    use sedona_testing::raster_spec::RasterSpec;
    use sedona_testing::rasters::generate_test_rasters;
    use sedona_testing::testers::ScalarUdfTester;

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
        // (raster, point) resolves to a Float64 output.
        let return_type = RsValuePoint { with_band: false }
            .return_type(&[RASTER, SedonaType::Wkb(Edges::Planar, lnglat())])
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
    fn band_zero_errors() {
        // Band 0 is not coerced to band 1 — it surfaces as a 1-based error.
        let spec = RasterSpec::d2(2, 1).band_values(&[1u8, 2]);
        let err = sample(spec, 0, 0, 0).unwrap_err().to_string();
        assert!(err.contains("1-based"), "unexpected error: {err}");
    }

    #[test]
    fn nan_nodata_pixel_is_none() {
        // A float band whose nodata is NaN: a NaN pixel reads as NULL (NaN != NaN
        // makes the `==` check insufficient), a normal pixel reads as its value.
        let spec = RasterSpec::d2(2, 1)
            .band_values(&[f64::NAN, 1.0])
            .nodata(f64::NAN);
        assert_eq!(sample(spec.clone(), 0, 0, 1).unwrap(), None);
        assert_eq!(sample(spec, 1, 0, 1).unwrap(), Some(1.0));
    }

    #[test]
    fn non_2d_band_errors() {
        // A band with a leading non-spatial dimension is rejected.
        let spec = RasterSpec::nd(&["time", "y", "x"], &[2, 2, 1]).band(BandDataType::UInt8);
        let err = sample(spec, 0, 0, 1).unwrap_err().to_string();
        assert!(err.contains("2-D"), "unexpected error: {err}");
    }

    #[test]
    fn point_crs_mismatch_errors() {
        let udf: ScalarUDF = rs_value_udf().into();

        // Raster has a CRS (generate_test_rasters sets OGC:CRS84), point does not.
        let geom_type = SedonaType::Wkb(Edges::Planar, None);
        let tester = ScalarUdfTester::new(udf.clone(), vec![RASTER, geom_type.clone()]);
        let rasters = generate_test_rasters(1, None).unwrap();
        let geoms = create_geom_array(&[Some("POINT (2.1 2.6)")], &geom_type);
        let err = tester
            .invoke_arrays(vec![Arc::new(rasters), geoms])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("raster has a CRS but the point does not"),
            "unexpected error: {err}"
        );

        // Point has a CRS, raster does not.
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(udf, vec![RASTER, geom_type.clone()]);
        let rasters = RasterSpec::d2(2, 2)
            .band(BandDataType::UInt8)
            .crs(None)
            .build();
        let geoms = create_geom_array(&[Some("POINT (0 0)")], &geom_type);
        let err = tester
            .invoke_arrays(vec![Arc::new(rasters), geoms])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("point has a CRS but the raster does not"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn non_point_geometry_errors() {
        let udf: ScalarUDF = rs_value_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(udf, vec![RASTER, geom_type.clone()]);
        let rasters = generate_test_rasters(1, None).unwrap();
        let geoms = create_geom_array(&[Some("LINESTRING (0 0, 1 1)")], &geom_type);
        let err = tester
            .invoke_arrays(vec![Arc::new(rasters), geoms])
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected a Point"), "unexpected error: {err}");
    }

    #[test]
    fn empty_point_is_none() {
        // POINT EMPTY has no location to sample, so the result is NULL rather
        // than an error. The empty check short-circuits before CRS resolution,
        // so a missing/again-mismatched point CRS does not matter here.
        let udf: ScalarUDF = rs_value_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(udf, vec![RASTER, geom_type.clone()]);
        let rasters = generate_test_rasters(1, None).unwrap();
        let geoms = create_geom_array(&[Some("POINT EMPTY")], &geom_type);
        let result = tester
            .invoke_arrays(vec![Arc::new(rasters), geoms])
            .unwrap();
        let arr = result.as_any().downcast_ref::<Float64Array>().unwrap();
        assert!(arr.is_null(0), "POINT EMPTY should sample to NULL");
    }

    #[test]
    fn point_just_outside_top_edge_is_none() {
        // North-up raster: origin (0, 10), 1x1 pixels (geotransform
        // [c, a, b, f, d, e] = [0, 1, 0, 10, 0, -1]), so world y decreases down
        // the rows. A point at y = 10.5 is just *above* the top edge: its pixel
        // row is -0.5, which must floor to -1 (out of bounds -> NULL), not
        // truncate toward zero to 0 (the top row).
        let udf: ScalarUDF = rs_value_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(udf, vec![RASTER, geom_type.clone()]);
        let raster = || {
            RasterSpec::d2(2, 2)
                .band_values(&[1u8, 2, 3, 4])
                .transform([0.0, 1.0, 0.0, 10.0, 0.0, -1.0])
                .build()
        };

        // Just above the top edge -> NULL.
        let geoms = create_geom_array(&[Some("POINT (0.5 10.5)")], &geom_type);
        let result = tester
            .invoke_arrays(vec![Arc::new(raster()), geoms])
            .unwrap();
        let arr = result.as_any().downcast_ref::<Float64Array>().unwrap();
        assert!(arr.is_null(0), "point above the top edge should be NULL");

        // Just inside the top row -> the top-left value (1).
        let geoms = create_geom_array(&[Some("POINT (0.5 9.5)")], &geom_type);
        let result = tester
            .invoke_arrays(vec![Arc::new(raster()), geoms])
            .unwrap();
        let arr = result.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(arr.value(0), 1.0);
    }

    #[test]
    fn scalar_raster_samples_via_fast_path() {
        // A constant (scalar) raster with the default band takes the optimized
        // hoisted path; verify it samples the right pixel and rejects
        // out-of-bounds, matching the general (array-raster) path.
        let udf: ScalarUDF = rs_value_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(udf, vec![RASTER, geom_type.clone()]);

        // North-up 2x2 raster, origin (0, 10), 1x1 pixels; row-major [1,2 / 3,4].
        let raster = RasterSpec::d2(2, 2)
            .band_values(&[1u8, 2, 3, 4])
            .transform([0.0, 1.0, 0.0, 10.0, 0.0, -1.0])
            .build();

        let sample = |wkt: &str| -> Option<f64> {
            let geoms = create_geom_array(&[Some(wkt)], &geom_type);
            let result = tester
                .invoke(vec![
                    ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(raster.clone()))),
                    ColumnarValue::Array(geoms),
                ])
                .unwrap();
            let arr = result.into_array(1).unwrap();
            let arr = arr.as_any().downcast_ref::<Float64Array>().unwrap();
            (!arr.is_null(0)).then(|| arr.value(0))
        };

        assert_eq!(sample("POINT (0.5 9.5)"), Some(1.0)); // pixel (0, 0)
        assert_eq!(sample("POINT (1.5 8.5)"), Some(4.0)); // pixel (1, 1)
        assert_eq!(sample("POINT (100 100)"), None); // outside the footprint
    }

    #[test]
    fn scalar_raster_constant_band_uses_fast_path() {
        // A scalar raster with a constant (scalar) band argument takes the same
        // hoisted fast path as the 2-arg default-band case, just on that band.
        let udf: ScalarUDF = rs_value_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(
            udf,
            vec![
                RASTER,
                geom_type.clone(),
                SedonaType::Arrow(DataType::Int32),
            ],
        );
        // North-up 2x2, two bands: band 1 [1,2,3,4], band 2 [10,20,30,40].
        let raster = RasterSpec::d2(2, 2)
            .band_values(&[1u8, 2, 3, 4])
            .band_values(&[10u8, 20, 30, 40])
            .transform([0.0, 1.0, 0.0, 10.0, 0.0, -1.0])
            .build();
        let geoms = create_geom_array(&[Some("POINT (0.5 9.5)")], &geom_type); // pixel (0, 0)
        let result = tester
            .invoke(vec![
                ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(raster))),
                ColumnarValue::Array(geoms),
                ColumnarValue::Scalar(ScalarValue::Int32(Some(2))),
            ])
            .unwrap();
        let arr = result.into_array(1).unwrap();
        let arr = arr.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(arr.value(0), 10.0); // band 2, pixel (0, 0)
    }

    #[test]
    fn scalar_raster_band_column_resolves_per_row() {
        // A scalar raster with a band *column* still hoists affine/CRS/reproject,
        // but resolves the band buffer per row. A NULL band element -> NULL.
        let udf: ScalarUDF = rs_value_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(
            udf,
            vec![
                RASTER,
                geom_type.clone(),
                SedonaType::Arrow(DataType::Int32),
            ],
        );
        let raster = RasterSpec::d2(2, 2)
            .band_values(&[1u8, 2, 3, 4])
            .band_values(&[10u8, 20, 30, 40])
            .transform([0.0, 1.0, 0.0, 10.0, 0.0, -1.0])
            .build();
        // Three points at pixel (0, 0), sampling band 1, band 2, then a NULL band.
        let geoms = create_geom_array(
            &[
                Some("POINT (0.5 9.5)"),
                Some("POINT (0.5 9.5)"),
                Some("POINT (0.5 9.5)"),
            ],
            &geom_type,
        );
        let bands = Arc::new(Int32Array::from(vec![Some(1), Some(2), None]));
        let result = tester
            .invoke(vec![
                ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(raster))),
                ColumnarValue::Array(geoms),
                ColumnarValue::Array(bands),
            ])
            .unwrap();
        let arr = result.into_array(3).unwrap();
        let arr = arr.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(arr.value(0), 1.0); // band 1
        assert_eq!(arr.value(1), 10.0); // band 2
        assert!(arr.is_null(2), "NULL band element should be NULL");
    }

    #[test]
    fn scalar_raster_null_scalar_band_is_all_null() {
        // A NULL scalar band makes every output NULL without touching the band.
        let udf: ScalarUDF = rs_value_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(
            udf,
            vec![
                RASTER,
                geom_type.clone(),
                SedonaType::Arrow(DataType::Int32),
            ],
        );
        let raster = RasterSpec::d2(2, 2).band_values(&[1u8, 2, 3, 4]).build();
        let geoms = create_geom_array(&[Some("POINT (0.5 0.5)")], &geom_type);
        let result = tester
            .invoke(vec![
                ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(raster))),
                ColumnarValue::Array(geoms),
                ColumnarValue::Scalar(ScalarValue::Int32(None)),
            ])
            .unwrap();
        let arr = result.into_array(1).unwrap();
        let arr = arr.as_any().downcast_ref::<Float64Array>().unwrap();
        assert!(arr.is_null(0), "NULL scalar band should sample to NULL");
    }

    #[test]
    fn non_finite_point_ordinate_is_none() {
        // A point with a NaN/Inf ordinate (e.g. POINT(NaN 5)) has no location to
        // sample. Without the finite guard a NaN survives `inv_transform` and the
        // saturating cast turns it into pixel column 0, silently sampling a value.
        let raster = RasterSpec::d2(2, 2)
            .band_values(&[1u8, 2, 3, 4])
            .transform([0.0, 1.0, 0.0, 10.0, 0.0, -1.0])
            .build();
        let rasters = RasterStructArray::try_new(&raster).unwrap();
        let affine = AffineMatrix::from_metadata(&rasters.get(0).unwrap().metadata());

        assert_eq!(xy_to_pixel(&affine, f64::NAN, 5.0).unwrap(), None);
        assert_eq!(xy_to_pixel(&affine, 5.0, f64::NAN).unwrap(), None);
        assert_eq!(xy_to_pixel(&affine, f64::INFINITY, 5.0).unwrap(), None);
        // A finite in-bounds point still maps to a real pixel.
        assert_eq!(xy_to_pixel(&affine, 0.5, 9.5).unwrap(), Some((0, 0)));
    }

    #[test]
    fn scalar_all_null_points_defer_band_resolution() {
        // The scalar fast path defers band/2-D validation until a point needs
        // sampling: an all-null point column over an unsupported (non-2-D) raster
        // returns all-NULL rather than erroring, matching the general path. A
        // valid point over the same raster still surfaces the 2-D error.
        let udf: ScalarUDF = rs_value_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, None);
        let tester = ScalarUdfTester::new(udf, vec![RASTER, geom_type.clone()]);
        let raster = RasterSpec::nd(&["time", "y", "x"], &[2, 2, 1])
            .band(BandDataType::UInt8)
            .crs(None)
            .build();

        let invoke = |wkt: Option<&str>| {
            let geoms = create_geom_array(&[wkt], &geom_type);
            tester.invoke(vec![
                ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(raster.clone()))),
                ColumnarValue::Array(geoms),
            ])
        };

        // All-null point -> NULL, no band resolution, no error.
        let result = invoke(None).unwrap();
        let arr = result.into_array(1).unwrap();
        let arr = arr.as_any().downcast_ref::<Float64Array>().unwrap();
        assert!(
            arr.is_null(0),
            "all-null point over a non-2-D raster should be NULL"
        );

        // A real point forces band resolution, which rejects the non-2-D raster.
        let err = invoke(Some("POINT (0 0)")).unwrap_err().to_string();
        assert!(err.contains("2-D"), "unexpected error: {err}");
    }

    #[test]
    fn crs_decision_equal_crs_skips_reproject() {
        // The common case: a lng/lat point CRS and a lng/lat raster are detected
        // as equal, so the per-point reproject (and its crs_equals) is skipped.
        // This is what the B1 hoist relies on — if it returned Some(true) the
        // optimization would silently no-op.
        let raster = RasterSpec::d2(2, 2).band(BandDataType::UInt8).build(); // default lng/lat
        let rasters = RasterStructArray::try_new(&raster).unwrap();
        let raster_crs = resolve_crs(rasters.get(0).unwrap().crs()).unwrap();
        let point_type = SedonaType::Wkb(Edges::Planar, lnglat());
        assert_eq!(
            column_reproject_decision(&point_type, raster_crs.as_deref()).unwrap(),
            Some(false),
            "lng/lat point + lng/lat raster must be detected as equal (skip reproject)"
        );
    }

    #[test]
    fn crs_decision_differing_crs_reprojects() {
        use sedona_schema::crs::deserialize_crs;
        let raster = RasterSpec::d2(2, 2)
            .crs(Some("EPSG:4326"))
            .band(BandDataType::UInt8)
            .build();
        let rasters = RasterStructArray::try_new(&raster).unwrap();
        let raster_crs = resolve_crs(rasters.get(0).unwrap().crs()).unwrap();
        let point_type = SedonaType::Wkb(Edges::Planar, deserialize_crs("EPSG:3857").unwrap());
        assert_eq!(
            column_reproject_decision(&point_type, raster_crs.as_deref()).unwrap(),
            Some(true),
            "EPSG:3857 point + EPSG:4326 raster must require reprojection"
        );
    }

    #[test]
    fn crs_decision_one_sided_crs_errors() {
        let raster = RasterSpec::d2(2, 2)
            .crs(None)
            .band(BandDataType::UInt8)
            .build();
        let rasters = RasterStructArray::try_new(&raster).unwrap();
        let raster_crs = resolve_crs(rasters.get(0).unwrap().crs()).unwrap();
        let point_type = SedonaType::Wkb(Edges::Planar, lnglat());
        assert!(column_reproject_decision(&point_type, raster_crs.as_deref()).is_err());
    }
}
