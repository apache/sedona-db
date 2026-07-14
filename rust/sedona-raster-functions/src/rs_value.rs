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

use std::sync::Arc;

use arrow_array::{builder::Float64Builder, Array, ArrayRef, Float64Array, StructArray};
use arrow_schema::DataType;
use datafusion_common::cast::as_int32_array;
use datafusion_common::config::ConfigOptions;
use datafusion_common::{exec_datafusion_err, exec_err, Result, ScalarValue};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::transform::CrsTransform;
use sedona_geometry::wkb_header::read_point_xy;
use sedona_raster::affine_transformation::AffineMatrix;
use sedona_raster::array::RasterStructArray;
use sedona_raster::traits::RasterRef;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::crs_utils::{resolve_crs, with_crs_engine};
use crate::executor::RasterExecutor;
use crate::rs_ensure_loaded::NEEDS_PIXELS_METADATA_KEY;
use crate::sampling::{
    column_point_crs_transform, default_band, int32_array_arg, next_band, point_crs_transform,
    read_pixel, xy_to_pixel,
};

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
        self.invoke(arg_types, args, None)
    }

    fn invoke_batch_from_args(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
        _return_type: &SedonaType,
        _num_rows: usize,
        config_options: Option<&ConfigOptions>,
    ) -> Result<ColumnarValue> {
        self.invoke(arg_types, args, config_options)
    }
}

impl RsValuePoint {
    fn invoke(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
        config_options: Option<&ConfigOptions>,
    ) -> Result<ColumnarValue> {
        // Fast path: a constant (scalar) raster lets us resolve the affine
        // transform and CRS once for the whole batch instead of per point — the
        // common RS_Value(raster_expr, point_column[, band]) shape. The band
        // argument does not change this: a constant band also hoists its buffer,
        // and only a band *column* falls back to per-row band resolution.
        if let ColumnarValue::Scalar(ScalarValue::Struct(raster_struct)) = &args[0] {
            return self.invoke_scalar_raster(arg_types, args, config_options, raster_struct);
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

        // Reprojecting the point into the raster CRS needs a CRS engine.
        with_crs_engine(config_options, |engine| {
            executor.execute_raster_wkb_crs_void(|raster_opt, wkb_opt, point_crs| {
                // Advance the band column every row so it stays in lockstep with
                // the row index (a no-op when there is no band argument).
                let band_arg = next_band(&mut band_iter);
                let (raster, point_wkb) = match (raster_opt, wkb_opt) {
                    (Some(raster), Some(point_wkb)) => (raster, point_wkb),
                    _ => {
                        builder.append_null();
                        return Ok(());
                    }
                };
                // An explicit band column drives the band (a NULL element yields a
                // NULL row); with no band argument it defaults to band 1, but only
                // for a single-band raster.
                let band_num = if self.with_band {
                    match band_arg {
                        Some(band_num) => band_num,
                        None => {
                            builder.append_null();
                            return Ok(());
                        }
                    }
                } else {
                    default_band("RS_Value", raster.num_bands())?
                };

                // Parse the point and bring it into the raster's CRS. Null/empty
                // points (and non-finite coordinates) have no location to sample.
                let raster_crs = resolve_crs(raster.crs())?;
                let trans =
                    point_crs_transform("RS_Value", point_crs, raster_crs.as_deref(), engine)?;
                let Some((x, y)) = resolve_point_xy(point_wkb, trans.as_deref())? else {
                    builder.append_null();
                    return Ok(());
                };

                let affine = AffineMatrix::from_metadata(&raster.metadata());
                match xy_to_pixel("RS_Value", &affine, x, y)? {
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
    /// points yield NULL identically. Band resolution — including the
    /// default-band ambiguity check and the 2-D check — is deferred until at
    /// least one point needs sampling, so an all-null/all-empty batch returns
    /// NULL without touching the band — as the general path does.
    fn invoke_scalar_raster(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
        config_options: Option<&ConfigOptions>,
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

        // Band selection: a missing band argument (default band 1, single-band
        // rasters only) or a scalar band is constant for the batch and lets us
        // hoist the band buffer; a band column is resolved per row. A NULL scalar
        // band makes every output NULL. With no band argument the default-band
        // ambiguity check is deferred until a point needs sampling (below), so an
        // all-null/all-empty batch over a multiband raster stays NULL rather than
        // erroring — matching the general path, which only resolves the band on
        // rows that actually have a point.
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
        }
        let band_array = band_values
            .as_ref()
            .map(|a| as_int32_array(a))
            .transpose()?;

        // Affine transform and raster CRS, resolved once for all points.
        let affine = AffineMatrix::from_metadata(&raster.metadata());
        let raster_crs = resolve_crs(raster.crs())?;

        let mut geom = executor.make_geom_wkb_crs_accessor(1)?;

        // Phase 1 — selection vector: collect (row, x, y, band) for the points
        // worth sampling (non-null, non-empty Point with a non-null band),
        // reprojected into the raster CRS. Skipped rows stay NULL in the output.
        let mut selection: Vec<(usize, f64, f64, usize)> = Vec::with_capacity(n);
        let mut band_iter = band_array.map(|a| a.iter());
        with_crs_engine(config_options, |engine| {
            // Resolve the raster-CRS transform once when the point CRS is
            // column-level (the common case). This skips both a per-point
            // `crs_equals` (which allocates a String, ~15 ns/point) and a
            // per-point transform lookup.
            let hoisted_trans = column_point_crs_transform(
                "RS_Value",
                &arg_types[1],
                raster_crs.as_deref(),
                engine,
            )?;
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
                // A per-item point CRS (hoisted_trans == None) resolves per row.
                let trans = match &hoisted_trans {
                    Some(trans) => trans.clone(),
                    None => {
                        point_crs_transform("RS_Value", point_crs, raster_crs.as_deref(), engine)?
                    }
                };
                if let Some((x, y)) = resolve_point_xy(point_wkb, trans.as_deref())? {
                    selection.push((i, x, y, band_num));
                }
            }
            Ok(())
        })?;

        let mut out: Vec<Option<f64>> = vec![None; n];
        if selection.is_empty() {
            return executor.finish(Arc::new(Float64Array::from(out)));
        }

        // At least one point needs sampling, so the deferred default-band
        // resolution (band 1, single-band rasters only) can now run — and error
        // on a multiband raster, exactly as the general path would for this row.
        let const_band = if self.with_band {
            const_band
        } else {
            Some(default_band("RS_Value", raster.num_bands())?)
        };

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
                    if let Some((col, row)) = xy_to_pixel("RS_Value", &affine, x, y)? {
                        out[i] = read_pixel("RS_Value", &buffer, nodata, col, row)?;
                    }
                }
            }
            None => {
                for (i, x, y, band_num) in selection {
                    if let Some((col, row)) = xy_to_pixel("RS_Value", &affine, x, y)? {
                        out[i] = sample_pixel(&raster, col, row, band_num)?;
                    }
                }
            }
        }

        executor.finish(Arc::new(Float64Array::from(out)))
    }
}

/// Parse a Point WKB and return its `(x, y)` in the raster's CRS, or `None` when
/// there is nothing to sample (the point is empty — both ordinates NaN).
///
/// `trans` is the transform into the raster CRS resolved by the caller (via
/// [`point_crs_transform`]/[`column_point_crs_transform`]); `None` samples the
/// original coordinates. The coordinate is transformed in place — no
/// reprojected-WKB materialisation.
fn resolve_point_xy(
    point_wkb: &[u8],
    trans: Option<&dyn CrsTransform>,
) -> Result<Option<(f64, f64)>> {
    let Some(mut xy) =
        read_point_xy(point_wkb).map_err(|e| exec_datafusion_err!("RS_Value: {e}"))?
    else {
        return Ok(None);
    };
    if let Some(trans) = trans {
        trans
            .transform_coord(&mut xy)
            .map_err(|e| exec_datafusion_err!("RS_Value: {e}"))?;
    }
    Ok(Some(xy))
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
    read_pixel("RS_Value", &buffer, nodata, col, row)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, Float64Array, Int32Array};
    use datafusion_expr::ScalarUDF;
    use sedona_proj::transform::with_global_proj_engine;
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
            err.contains("raster has a CRS but the geometry does not"),
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
            err.contains("geometry has a CRS but the raster does not"),
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
                .bbox(0.0, 8.0, 2.0, 10.0)
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

    /// North-up 2x2 raster spanning world bbox (0, 8)–(2, 10) with 1x1 pixels,
    /// band values row-major `[1, 2, 3, 4]`: pixel (0, 0) covers x∈[0,1),
    /// y∈[9,10) and holds 1; pixel (1, 1) holds 4.
    fn raster_2x2_spec() -> RasterSpec {
        RasterSpec::d2(2, 2)
            .band_values(&[1u8, 2, 3, 4])
            .bbox(0.0, 8.0, 2.0, 10.0)
    }

    /// Two-band variant of [`raster_2x2_spec`] (band 2 holds `[10, 20, 30, 40]`),
    /// used to exercise the default-band ambiguity error.
    fn two_band_raster_spec() -> RasterSpec {
        raster_2x2_spec().band_values(&[10u8, 20, 30, 40])
    }

    #[test]
    fn scalar_raster_samples_via_fast_path() {
        // A constant (scalar) raster with the default band takes the optimized
        // hoisted path; verify it samples the right pixel and rejects
        // out-of-bounds, matching the general (array-raster) path.
        let udf: ScalarUDF = rs_value_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(udf, vec![RASTER, geom_type]);

        let sample = |wkt: &str| tester.invoke_scalar_scalar(raster_2x2_spec(), wkt).unwrap();
        tester.assert_scalar_result_equals(sample("POINT (0.5 9.5)"), 1.0); // pixel (0, 0)
        tester.assert_scalar_result_equals(sample("POINT (1.5 8.5)"), 4.0); // pixel (1, 1)
        tester.assert_scalar_result_equals(sample("POINT (100 100)"), ScalarValue::Float64(None));
    }

    #[test]
    fn default_band_requires_single_band_raster_array_path() {
        // No band argument + multiband raster is ambiguous -> error (general,
        // array-raster path) rather than silently sampling band 1.
        let udf: ScalarUDF = rs_value_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(udf, vec![RASTER, geom_type.clone()]);
        let geoms = create_geom_array(&[Some("POINT (0.5 9.5)")], &geom_type);
        let err = tester
            .invoke_arrays(vec![Arc::new(two_band_raster_spec().build()), geoms])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("specify which band"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn default_band_requires_single_band_raster_scalar_path() {
        // Same ambiguity on the scalar-raster fast path.
        let udf: ScalarUDF = rs_value_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(udf, vec![RASTER, geom_type]);
        let err = tester
            .invoke_scalar_scalar(two_band_raster_spec(), "POINT (0.5 9.5)")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("specify which band"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn scalar_all_null_points_defer_default_band_check() {
        // The scalar fast path defers the default-band ambiguity check until a
        // point needs sampling: an all-NULL point column over a multiband raster
        // returns NULL rather than erroring — matching the general path, which
        // only resolves the band on rows that actually have a point.
        let udf: ScalarUDF = rs_value_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(udf, vec![RASTER, geom_type.clone()]);
        let geoms = create_geom_array(&[None, None], &geom_type);
        let result = tester
            .invoke(vec![
                ColumnarValue::Scalar(two_band_raster_spec().scalar()),
                ColumnarValue::Array(geoms),
            ])
            .unwrap();
        let arr = result.into_array(2).unwrap();
        let arr = arr.as_any().downcast_ref::<Float64Array>().unwrap();
        assert!(arr.is_null(0) && arr.is_null(1));
    }

    #[test]
    fn scalar_raster_constant_band_uses_fast_path() {
        // A scalar raster with a constant (scalar) band argument takes the same
        // hoisted fast path as the 2-arg default-band case, just on that band.
        let udf: ScalarUDF = rs_value_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(
            udf,
            vec![RASTER, geom_type, SedonaType::Arrow(DataType::Int32)],
        );
        let result = tester
            .invoke_scalar_scalar_scalar(two_band_raster_spec(), "POINT (0.5 9.5)", 2)
            .unwrap();
        tester.assert_scalar_result_equals(result, 10.0); // band 2, pixel (0, 0)
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
            .bbox(0.0, 8.0, 2.0, 10.0)
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
            .bbox(0.0, 8.0, 2.0, 10.0)
            .build();
        let rasters = RasterStructArray::try_new(&raster).unwrap();
        let affine = AffineMatrix::from_metadata(&rasters.get(0).unwrap().metadata());

        assert_eq!(
            xy_to_pixel("RS_Value", &affine, f64::NAN, 5.0).unwrap(),
            None
        );
        assert_eq!(
            xy_to_pixel("RS_Value", &affine, 5.0, f64::NAN).unwrap(),
            None
        );
        assert_eq!(
            xy_to_pixel("RS_Value", &affine, f64::INFINITY, 5.0).unwrap(),
            None
        );
        // A finite in-bounds point still maps to a real pixel.
        assert_eq!(
            xy_to_pixel("RS_Value", &affine, 0.5, 9.5).unwrap(),
            Some((0, 0))
        );
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
        // as equal, so no per-point transform is applied. This is what the
        // column-level hoist relies on — if it resolved a transform here the
        // optimization would silently no-op.
        let raster = RasterSpec::d2(2, 2).band(BandDataType::UInt8).build(); // default lng/lat
        let rasters = RasterStructArray::try_new(&raster).unwrap();
        let raster_crs = resolve_crs(rasters.get(0).unwrap().crs()).unwrap();
        let point_type = SedonaType::Wkb(Edges::Planar, lnglat());
        with_global_proj_engine(|engine| {
            let hoisted =
                column_point_crs_transform("RS_Value", &point_type, raster_crs.as_deref(), engine)
                    .unwrap();
            assert!(
                matches!(hoisted, Some(None)),
                "lng/lat point + lng/lat raster must be detected as equal (no transform)"
            );
            Ok(())
        })
        .unwrap();
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
        with_global_proj_engine(|engine| {
            let hoisted =
                column_point_crs_transform("RS_Value", &point_type, raster_crs.as_deref(), engine)
                    .unwrap();
            assert!(
                matches!(hoisted, Some(Some(_))),
                "EPSG:3857 point + EPSG:4326 raster must resolve a transform"
            );
            Ok(())
        })
        .unwrap();
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
        with_global_proj_engine(|engine| {
            assert!(column_point_crs_transform(
                "RS_Value",
                &point_type,
                raster_crs.as_deref(),
                engine
            )
            .is_err());
            Ok(())
        })
        .unwrap();
    }
}
