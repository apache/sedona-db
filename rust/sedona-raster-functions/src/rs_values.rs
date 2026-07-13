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

//! `RS_Values` — sample a raster's pixel value at each point of a MultiPoint.
//!
//! ```text
//! RS_Values(raster, points)        -> List<Double>  -- single-band rasters only
//! RS_Values(raster, points, band)  -> List<Double>
//! ```
//!
//! The plural companion of [`RS_Value`](crate::rs_value): where `RS_Value` takes
//! one Point and returns one Double, `RS_Values` takes a MultiPoint (a single
//! Point is also accepted) and returns a `List<Double>` — one element per
//! sub-point, in input order. Each element is `NULL` when its sub-point is empty,
//! out of bounds, or reads the band's nodata; the whole list is `NULL` when the
//! raster, geometry, or band is `NULL`. An empty MultiPoint yields an empty list.
//!
//! Sampling, CRS handling, and pixel decoding are shared with `RS_Value` via
//! [`crate::sampling`]; this module only adds the per-sub-point iteration and the
//! list-shaped output.
//!
//! Like `RS_Value`, the function is tagged [`NEEDS_PIXELS_METADATA_KEY`] so the
//! planner materialises the raster InDb before a kernel runs, and only 2-D
//! rasters are supported.

use std::sync::Arc;

use arrow_array::builder::{Float64Builder, ListBuilder};
use arrow_array::{Array, ArrayRef, StructArray};
use arrow_schema::{DataType, Field};
use datafusion_common::cast::as_int32_array;
use datafusion_common::{exec_datafusion_err, exec_err, Result, ScalarValue};
use datafusion_expr::{ColumnarValue, Volatility};
use geo_traits::{CoordTrait, GeometryTrait, GeometryType, MultiPointTrait, PointTrait};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_proj::transform::with_global_proj_engine;
use sedona_raster::affine_transformation::AffineMatrix;
use sedona_raster::array::RasterStructArray;
use sedona_raster::traits::{BandRef, NdBuffer, RasterRef};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};
use wkb::reader::read_wkb;

use crate::crs_utils::resolve_crs;
use crate::executor::RasterExecutor;
use crate::rs_ensure_loaded::NEEDS_PIXELS_METADATA_KEY;
use crate::sampling::{
    column_reproject_decision, default_band, int32_array_arg, next_band, read_pixel, reproject_wkb,
    xy_to_pixel,
};

/// The `List<Float64>` output type, matching what a default
/// `ListBuilder<Float64Builder>` produces (field "item", nullable).
fn list_float64_type() -> DataType {
    DataType::List(Arc::new(Field::new("item", DataType::Float64, true)))
}

/// `RS_Values()` scalar UDF — sample pixel values at each point of a MultiPoint.
pub fn rs_values_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_values",
        vec![
            Arc::new(RsValues { with_band: false }), // RS_Values(raster, points)
            Arc::new(RsValues { with_band: true }),  // RS_Values(raster, points, band)
        ],
        Volatility::Immutable,
    )
    // The kernels read pixel bytes, so the raster argument must be materialised
    // InDb first; the planner injects RS_EnsureLoaded based on this flag.
    .with_metadata(NEEDS_PIXELS_METADATA_KEY, "true")
}

/// Kernel for `RS_Values(raster, points[, band])`.
#[derive(Debug)]
struct RsValues {
    with_band: bool,
}

impl SedonaScalarKernel for RsValues {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let mut matchers = vec![
            ArgMatcher::is_raster(),
            ArgMatcher::is_geometry_or_geography(),
        ];
        if self.with_band {
            matchers.push(ArgMatcher::is_integer());
        }
        let matcher = ArgMatcher::new(matchers, SedonaType::Arrow(list_float64_type()));
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        // Fast path: a constant (scalar) raster lets us resolve the affine
        // transform, CRS, and band buffer once for the whole batch instead of
        // per row — the common RS_Values(raster_expr, points_column[, band])
        // shape. Only a band *column* falls back to per-row band resolution.
        if let ColumnarValue::Scalar(ScalarValue::Struct(raster_struct)) = &args[0] {
            return self.invoke_scalar_raster(arg_types, args, raster_struct.as_ref());
        }

        let executor = RasterExecutor::new(arg_types, args);
        let num_iterations = executor.num_iterations();
        let mut list_builder = ListBuilder::new(Float64Builder::new());
        // Per-row scratch for the parsed sub-point coordinates, reused across
        // rows so the parse does not allocate per row.
        let mut points_scratch: Vec<Option<(f64, f64)>> = Vec::new();

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

        // Reprojecting the points into the raster CRS needs a PROJ engine.
        with_global_proj_engine(|engine| {
            executor.execute_raster_wkb_crs_void(|raster_opt, wkb_opt, geom_crs| {
                // Advance the band column every row so it stays in lockstep with
                // the row index (a no-op when there is no band argument).
                let band_arg = next_band(&mut band_iter);
                let (raster, geom_wkb) = match (raster_opt, wkb_opt) {
                    (Some(raster), Some(geom_wkb)) => (raster, geom_wkb),
                    // A NULL raster or geometry yields a NULL row.
                    _ => {
                        list_builder.append_null();
                        return Ok(());
                    }
                };

                // Resolve the band to sample. An explicit band column drives it
                // (a NULL element yields a NULL row); with no band argument it
                // defaults to band 1, but only for a single-band raster — sampling
                // an unspecified band of a multiband raster is ambiguous, so it
                // errors rather than silently picking band 1.
                let band_num = if self.with_band {
                    match band_arg {
                        Some(band_num) => band_num,
                        None => {
                            list_builder.append_null();
                            return Ok(());
                        }
                    }
                } else {
                    default_band("RS_Values", raster.num_bands())?
                };

                // Resolve the band buffer, nodata, and affine transform once for
                // this row, then sample every sub-point against them.
                let raster_crs = resolve_crs(raster.crs())?;
                let band = resolve_band_2d(raster, band_num)?;
                let buffer = band
                    .nd_buffer()
                    .map_err(|e| exec_datafusion_err!("RS_Values: {e}"))?;
                let nodata = band
                    .nodata_as_f64()
                    .map_err(|e| exec_datafusion_err!("RS_Values: {e}"))?;
                let affine = AffineMatrix::from_metadata(&raster.metadata());

                // Reproject the whole geometry into the raster CRS (a no-op that
                // borrows the original bytes when the CRSes match or are absent).
                let reprojected = reproject_wkb(
                    "RS_Values",
                    geom_wkb,
                    geom_crs,
                    raster_crs.as_deref(),
                    engine,
                )?;
                let effective = reprojected.as_deref().unwrap_or(geom_wkb);

                points_scratch.clear();
                collect_point_xys(effective, &mut points_scratch)?;
                for xy in &points_scratch {
                    append_sample(*xy, &affine, &buffer, nodata, &mut list_builder)?;
                }
                list_builder.append(true);
                Ok(())
            })
        })?;

        executor.finish(Arc::new(list_builder.finish()))
    }
}

impl RsValues {
    /// Optimized path for a constant (scalar) raster: the affine transform and
    /// raster CRS are resolved once for the whole batch, and the per-row work
    /// reduces to parsing the points and reading pixels. This serves every band
    /// shape:
    /// - no band argument or a constant band → the band buffer is hoisted too;
    /// - a band *column* → the band buffer is resolved per row (its `NdBuffer`
    ///   borrows from the band, which can't be cached across distinct bands),
    ///   but the affine/CRS/reproject work is still hoisted.
    ///
    /// Sampling behaviour matches the general path: it uses the same
    /// [`collect_point_xys`]/[`append_sample`] helpers, so per-element and
    /// per-row NULL semantics are identical. Band resolution — including the
    /// default-band ambiguity check and the 2-D check — is deferred until at
    /// least one row has a geometry to sample, so an all-null batch returns
    /// NULL without touching the band — as the general path does.
    fn invoke_scalar_raster(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
        raster_struct: &StructArray,
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let n = executor.num_iterations();

        let all_null = |executor: &RasterExecutor| {
            let mut builder = ListBuilder::new(Float64Builder::new());
            for _ in 0..n {
                builder.append_null();
            }
            executor.finish(Arc::new(builder.finish()))
        };

        let rasters = RasterStructArray::try_new(raster_struct)?;
        if rasters.is_null(0) {
            // A NULL raster makes every output NULL.
            return all_null(&executor);
        }
        let raster = rasters.get(0)?;

        // Band selection: a missing band argument (default band 1, single-band
        // rasters only) or a scalar band is constant for the batch and lets us
        // hoist the band buffer; a band column is resolved per row. A NULL scalar
        // band makes every output NULL. With no band argument the default-band
        // ambiguity check is deferred until a row needs sampling (below), so an
        // all-null batch over a multiband raster stays NULL rather than erroring
        // — matching the general path, which only resolves the band on rows that
        // actually have a geometry.
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
                        return all_null(&executor);
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

        // Affine transform and raster CRS, resolved once for all rows. Decide
        // reprojection once when the geometry CRS is column-level (the common
        // case), skipping a per-row `crs_equals` and its String allocation.
        let affine = AffineMatrix::from_metadata(&raster.metadata());
        let raster_crs = resolve_crs(raster.crs())?;
        let needs_reproject =
            column_reproject_decision("RS_Values", &arg_types[1], raster_crs.as_deref())?;
        let skip_reproject = needs_reproject == Some(false);

        let mut geom = executor.make_geom_wkb_crs_accessor(1)?;

        // Phase 1 — parse each row's Point/MultiPoint (reprojected into the
        // raster CRS) into owned sub-point coordinates. All rows share one
        // coordinate arena with per-row (start, len, band) spans; a `None` row
        // is NULL output (NULL geometry or band element).
        let mut coords: Vec<Option<(f64, f64)>> = Vec::new();
        let mut rows: Vec<Option<(usize, usize, usize)>> = Vec::with_capacity(n);
        let mut band_iter = band_array.map(|a| a.iter());
        with_global_proj_engine(|engine| {
            for i in 0..n {
                // Advance the band column in lockstep with the row index; with
                // no band argument this yields a placeholder that the hoisted
                // constant band supersedes below.
                let band_num = match const_band {
                    Some(b) => Some(b),
                    None => next_band(&mut band_iter),
                };
                let (wkb_opt, geom_crs) = geom.get(i)?;
                let (Some(geom_wkb), Some(band_num)) = (wkb_opt, band_num) else {
                    rows.push(None);
                    continue;
                };
                let reprojected = if skip_reproject {
                    None
                } else {
                    reproject_wkb(
                        "RS_Values",
                        geom_wkb,
                        geom_crs,
                        raster_crs.as_deref(),
                        engine,
                    )?
                };
                let effective = reprojected.as_deref().unwrap_or(geom_wkb);
                let start = coords.len();
                collect_point_xys(effective, &mut coords)?;
                rows.push(Some((start, coords.len() - start, band_num)));
            }
            Ok(())
        })?;

        let mut list_builder = ListBuilder::new(Float64Builder::new());
        if rows.iter().all(|r| r.is_none()) {
            // No row has anything to sample, so band resolution (and the
            // default-band ambiguity check) never runs — as in the general path.
            return all_null(&executor);
        }

        // At least one row samples, so the deferred default-band resolution
        // (band 1, single-band rasters only) can now run — and error on a
        // multiband raster, exactly as the general path would for that row.
        let const_band = if self.with_band {
            const_band
        } else {
            Some(default_band("RS_Values", raster.num_bands())?)
        };

        // Phase 2 — sample. A constant band resolves its buffer/nodata/2-D
        // check once; a band column resolves them per row.
        match const_band {
            Some(band_num) => {
                let band = resolve_band_2d(&raster, band_num)?;
                let buffer = band
                    .nd_buffer()
                    .map_err(|e| exec_datafusion_err!("RS_Values: {e}"))?;
                let nodata = band
                    .nodata_as_f64()
                    .map_err(|e| exec_datafusion_err!("RS_Values: {e}"))?;
                for row in &rows {
                    let Some((start, len, _band)) = row else {
                        list_builder.append_null();
                        continue;
                    };
                    for xy in &coords[*start..*start + *len] {
                        append_sample(*xy, &affine, &buffer, nodata, &mut list_builder)?;
                    }
                    list_builder.append(true);
                }
            }
            None => {
                for row in &rows {
                    let Some((start, len, band_num)) = row else {
                        list_builder.append_null();
                        continue;
                    };
                    let band = resolve_band_2d(&raster, *band_num)?;
                    let buffer = band
                        .nd_buffer()
                        .map_err(|e| exec_datafusion_err!("RS_Values: {e}"))?;
                    let nodata = band
                        .nodata_as_f64()
                        .map_err(|e| exec_datafusion_err!("RS_Values: {e}"))?;
                    for xy in &coords[*start..*start + *len] {
                        append_sample(*xy, &affine, &buffer, nodata, &mut list_builder)?;
                    }
                    list_builder.append(true);
                }
            }
        }

        executor.finish(Arc::new(list_builder.finish()))
    }
}

/// Collect each sub-point of a Point/MultiPoint WKB as an owned `(x, y)`,
/// appending to `out` in input order. An empty sub-point contributes `None`. A
/// non-Point/MultiPoint geometry is an error.
fn collect_point_xys(wkb: &[u8], out: &mut Vec<Option<(f64, f64)>>) -> Result<()> {
    let geom = read_wkb(wkb).map_err(|e| exec_datafusion_err!("RS_Values: {e}"))?;
    match geom.as_type() {
        GeometryType::Point(point) => {
            out.push(point.coord().map(|c| (c.x(), c.y())));
        }
        GeometryType::MultiPoint(multi_point) => {
            for point in multi_point.points() {
                out.push(point.coord().map(|c| (c.x(), c.y())));
            }
        }
        _ => return exec_err!("RS_Values expects a Point or MultiPoint geometry"),
    }
    Ok(())
}

/// Resolve 1-based band `band_num` of `raster`, requiring a spatial 2-D
/// (y, x) grid — the only shape the sampling affine is defined for.
fn resolve_band_2d<'a>(
    raster: &'a dyn RasterRef,
    band_num: usize,
) -> Result<Box<dyn BandRef + 'a>> {
    let band = raster
        .bands()
        .band(band_num)
        .map_err(|e| exec_datafusion_err!("RS_Values: {e}"))?;
    if !band.is_spatial_2d() {
        return exec_err!("RS_Values supports 2-D rasters only; band is not a 2-D (y, x) grid");
    }
    Ok(band)
}

/// Append one sampled value (or NULL) to the current list row for a sub-point's
/// `(x, y)` in the raster CRS. `None` coordinates (an empty sub-point) and
/// out-of-bounds/nodata pixels both append a NULL element.
fn append_sample(
    xy: Option<(f64, f64)>,
    affine: &AffineMatrix,
    buffer: &NdBuffer,
    nodata: Option<f64>,
    list_builder: &mut ListBuilder<Float64Builder>,
) -> Result<()> {
    let sample = match xy {
        Some((x, y)) => match xy_to_pixel("RS_Values", affine, x, y)? {
            Some((col, row)) => read_pixel("RS_Values", buffer, nodata, col, row)?,
            None => None,
        },
        None => None,
    };
    match sample {
        Some(value) => list_builder.values().append_value(value),
        None => list_builder.values().append_null(),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, Float64Array, Int32Array, ListArray};
    use datafusion_expr::ScalarUDF;
    use sedona_schema::crs::lnglat;
    use sedona_schema::datatypes::{Edges, RASTER};
    use sedona_schema::raster::BandDataType;
    use sedona_testing::create::create_array as create_geom_array;
    use sedona_testing::raster_spec::RasterSpec;
    use sedona_testing::rasters::generate_test_rasters;
    use sedona_testing::testers::ScalarUdfTester;

    /// North-up 2x2 raster, origin (0, 10), 1x1 pixels, band values row-major
    /// `[1, 2, 3, 4]` (row0 = [1, 2], row1 = [3, 4]). Pixel (0, 0) covers world
    /// x∈[0,1), y∈[9,10) and holds 1; pixel (1, 1) holds 4.
    fn raster_2x2() -> RasterSpec {
        RasterSpec::d2(2, 2)
            .band_values(&[1u8, 2, 3, 4])
            .transform([0.0, 1.0, 0.0, 10.0, 0.0, -1.0])
    }

    /// Build MultiPoint WKB by hand: the test WKT parser cannot express an
    /// EMPTY sub-point, which WKB encodes as a point with NaN ordinates.
    fn multipoint_wkb(points: &[Option<(f64, f64)>]) -> Vec<u8> {
        let mut out = vec![1u8]; // little-endian
        out.extend(4u32.to_le_bytes()); // MultiPoint
        out.extend((points.len() as u32).to_le_bytes());
        for point in points {
            out.push(1);
            out.extend(1u32.to_le_bytes()); // Point
            let (x, y) = point.unwrap_or((f64::NAN, f64::NAN));
            out.extend(x.to_le_bytes());
            out.extend(y.to_le_bytes());
        }
        out
    }

    /// Extract one list row as a `Vec<Option<f64>>`.
    fn row(result: &dyn Array, i: usize) -> Vec<Option<f64>> {
        let list = result.as_any().downcast_ref::<ListArray>().unwrap();
        let values = list.value(i);
        let values = values.as_any().downcast_ref::<Float64Array>().unwrap();
        (0..values.len())
            .map(|j| (!values.is_null(j)).then(|| values.value(j)))
            .collect()
    }

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = rs_values_udf().into();
        assert_eq!(udf.name(), "rs_values");
    }

    #[test]
    fn udf_marks_needs_pixels() {
        assert_eq!(
            rs_values_udf()
                .metadata()
                .get(NEEDS_PIXELS_METADATA_KEY)
                .map(String::as_str),
            Some("true")
        );
    }

    #[test]
    fn return_type_is_list_float64() {
        let return_type = RsValues { with_band: false }
            .return_type(&[RASTER, SedonaType::Wkb(Edges::Planar, lnglat())])
            .unwrap();
        assert_eq!(return_type, Some(SedonaType::Arrow(list_float64_type())));
    }

    #[test]
    fn multipoint_samples_each_point_in_order() {
        let udf: ScalarUDF = rs_values_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(udf, vec![RASTER, geom_type.clone()]);

        // Pixel (0,0)=1, pixel (1,1)=4, and a point far outside -> NULL element.
        let geoms = create_geom_array(
            &[Some("MULTIPOINT (0.5 9.5, 1.5 8.5, 100 100)")],
            &geom_type,
        );
        let result = tester
            .invoke_arrays(vec![Arc::new(raster_2x2().build()), geoms])
            .unwrap();
        assert_eq!(row(&result, 0), vec![Some(1.0), Some(4.0), None]);
    }

    #[test]
    fn single_point_yields_one_element_list() {
        // A plain Point is accepted and produces a one-element list.
        let udf: ScalarUDF = rs_values_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(udf, vec![RASTER, geom_type.clone()]);

        let geoms = create_geom_array(&[Some("POINT (1.5 8.5)")], &geom_type);
        let result = tester
            .invoke_arrays(vec![Arc::new(raster_2x2().build()), geoms])
            .unwrap();
        assert_eq!(row(&result, 0), vec![Some(4.0)]);
    }

    #[test]
    fn empty_multipoint_yields_empty_list() {
        let udf: ScalarUDF = rs_values_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(udf, vec![RASTER, geom_type.clone()]);

        let geoms = create_geom_array(&[Some("MULTIPOINT EMPTY")], &geom_type);
        let result = tester
            .invoke_arrays(vec![Arc::new(raster_2x2().build()), geoms])
            .unwrap();
        let list = result.as_any().downcast_ref::<ListArray>().unwrap();
        assert!(
            !list.is_null(0),
            "empty MultiPoint is an empty list, not NULL"
        );
        assert_eq!(row(&result, 0), Vec::<Option<f64>>::new());
    }

    #[test]
    fn nodata_element_is_null() {
        let udf: ScalarUDF = rs_values_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(udf, vec![RASTER, geom_type.clone()]);

        // Band [1, 2, 3, 4] with nodata=4: sampling pixel (1,1) reads nodata.
        let raster = RasterSpec::d2(2, 2)
            .band_values(&[1u8, 2, 3, 4])
            .nodata(4u8)
            .transform([0.0, 1.0, 0.0, 10.0, 0.0, -1.0])
            .build();
        let geoms = create_geom_array(&[Some("MULTIPOINT (0.5 9.5, 1.5 8.5)")], &geom_type);
        let result = tester.invoke_arrays(vec![Arc::new(raster), geoms]).unwrap();
        assert_eq!(row(&result, 0), vec![Some(1.0), None]);
    }

    #[test]
    fn null_geometry_yields_null_list() {
        let udf: ScalarUDF = rs_values_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(udf, vec![RASTER, geom_type.clone()]);

        let geoms = create_geom_array(&[None], &geom_type);
        let result = tester
            .invoke_arrays(vec![Arc::new(raster_2x2().build()), geoms])
            .unwrap();
        let list = result.as_any().downcast_ref::<ListArray>().unwrap();
        assert!(list.is_null(0));
    }

    #[test]
    fn null_band_element_yields_null_list() {
        let udf: ScalarUDF = rs_values_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(
            udf,
            vec![
                RASTER,
                geom_type.clone(),
                SedonaType::Arrow(DataType::Int32),
            ],
        );

        let geoms = create_geom_array(&[Some("MULTIPOINT (0.5 9.5)")], &geom_type);
        let bands: arrow_array::ArrayRef = Arc::new(Int32Array::from(vec![None::<i32>]));
        let result = tester
            .invoke_arrays(vec![Arc::new(raster_2x2().build()), geoms, bands])
            .unwrap();
        let list = result.as_any().downcast_ref::<ListArray>().unwrap();
        assert!(list.is_null(0));
    }

    #[test]
    fn second_band_is_addressable() {
        let udf: ScalarUDF = rs_values_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(
            udf,
            vec![
                RASTER,
                geom_type.clone(),
                SedonaType::Arrow(DataType::Int32),
            ],
        );

        // Band 1 [1,2,3,4], band 2 [10,20,30,40]; sample band 2 at pixel (0,0).
        let raster = RasterSpec::d2(2, 2)
            .band_values(&[1u8, 2, 3, 4])
            .band_values(&[10u8, 20, 30, 40])
            .transform([0.0, 1.0, 0.0, 10.0, 0.0, -1.0])
            .build();
        let geoms = create_geom_array(&[Some("MULTIPOINT (0.5 9.5)")], &geom_type);
        let bands: arrow_array::ArrayRef = Arc::new(Int32Array::from(vec![Some(2)]));
        let result = tester
            .invoke_arrays(vec![Arc::new(raster), geoms, bands])
            .unwrap();
        assert_eq!(row(&result, 0), vec![Some(10.0)]);
    }

    #[test]
    fn default_band_requires_single_band_raster() {
        // With no band argument, a multiband raster is ambiguous and errors
        // rather than silently sampling band 1 (matches RS_SetBandNoDataValue).
        let udf: ScalarUDF = rs_values_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(udf, vec![RASTER, geom_type.clone()]);
        let raster = RasterSpec::d2(2, 2)
            .band_values(&[1u8, 2, 3, 4])
            .band_values(&[10u8, 20, 30, 40])
            .transform([0.0, 1.0, 0.0, 10.0, 0.0, -1.0])
            .build();
        let geoms = create_geom_array(&[Some("MULTIPOINT (0.5 9.5)")], &geom_type);
        let err = tester
            .invoke_arrays(vec![Arc::new(raster), geoms])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("specify which band"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn non_point_geometry_errors() {
        let udf: ScalarUDF = rs_values_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(udf, vec![RASTER, geom_type.clone()]);

        let geoms = create_geom_array(&[Some("LINESTRING (0 0, 1 1)")], &geom_type);
        let err = tester
            .invoke_arrays(vec![
                Arc::new(generate_test_rasters(1, None).unwrap()),
                geoms,
            ])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("expects a Point or MultiPoint"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn crs_mismatch_errors() {
        // Raster has a CRS (generate_test_rasters sets OGC:CRS84), points do not.
        let udf: ScalarUDF = rs_values_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, None);
        let tester = ScalarUdfTester::new(udf, vec![RASTER, geom_type.clone()]);
        let rasters = generate_test_rasters(1, None).unwrap();
        let geoms = create_geom_array(&[Some("MULTIPOINT (2.1 2.6)")], &geom_type);
        let err = tester
            .invoke_arrays(vec![Arc::new(rasters), geoms])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("raster has a CRS but the geometry does not"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn non_2d_band_errors() {
        let udf: ScalarUDF = rs_values_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, None);
        let tester = ScalarUdfTester::new(udf, vec![RASTER, geom_type.clone()]);
        let raster = RasterSpec::nd(&["time", "y", "x"], &[2, 2, 1])
            .band(BandDataType::UInt8)
            .crs(None)
            .build();
        let geoms = create_geom_array(&[Some("MULTIPOINT (0 0)")], &geom_type);
        let err = tester
            .invoke_arrays(vec![Arc::new(raster), geoms])
            .unwrap_err()
            .to_string();
        assert!(err.contains("2-D"), "unexpected error: {err}");
    }

    #[test]
    fn empty_sub_point_yields_null_element() {
        // A MultiPoint containing an EMPTY sub-point (`MULTIPOINT (0.5 9.5,
        // EMPTY, 1.5 8.5)`): the empty sub-point has no location to sample, so
        // it contributes a NULL element while its neighbours sample normally.
        let udf: ScalarUDF = rs_values_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(udf, vec![RASTER, geom_type.clone()]);

        let wkb = multipoint_wkb(&[Some((0.5, 9.5)), None, Some((1.5, 8.5))]);
        let geoms: arrow_array::ArrayRef =
            Arc::new(arrow_array::BinaryArray::from(vec![Some(wkb.as_slice())]));
        let result = tester
            .invoke_arrays(vec![Arc::new(raster_2x2().build()), geoms])
            .unwrap();
        assert_eq!(row(&result, 0), vec![Some(1.0), None, Some(4.0)]);
    }

    #[test]
    fn scalar_raster_samples_via_fast_path() {
        // A constant (scalar) raster takes the optimized hoisted path; verify
        // the per-element semantics match the general (array-raster) path:
        // in-bounds points sample, out-of-bounds and empty sub-points are NULL.
        let udf: ScalarUDF = rs_values_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(udf, vec![RASTER, geom_type.clone()]);

        // MULTIPOINT (0.5 9.5, 1.5 8.5, 100 100, EMPTY), NULL, MULTIPOINT EMPTY
        let full = multipoint_wkb(&[
            Some((0.5, 9.5)),
            Some((1.5, 8.5)),
            Some((100.0, 100.0)),
            None,
        ]);
        let empty = multipoint_wkb(&[]);
        let geoms: arrow_array::ArrayRef = Arc::new(arrow_array::BinaryArray::from(vec![
            Some(full.as_slice()),
            None,
            Some(empty.as_slice()),
        ]));
        let result = tester
            .invoke(vec![
                ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(raster_2x2().build()))),
                ColumnarValue::Array(geoms),
            ])
            .unwrap();
        let result = result.into_array(3).unwrap();
        assert_eq!(
            row(&result, 0),
            vec![Some(1.0), Some(4.0), None, None],
            "in-bounds points sample; out-of-bounds and EMPTY are NULL elements"
        );
        let list = result.as_any().downcast_ref::<ListArray>().unwrap();
        assert!(list.is_null(1), "NULL geometry is a NULL list");
        assert!(!list.is_null(2), "empty MultiPoint is an empty list");
        assert_eq!(row(&result, 2), Vec::<Option<f64>>::new());
    }

    #[test]
    fn scalar_raster_constant_band_uses_fast_path() {
        // A scalar raster with a constant (scalar) band hoists that band's
        // buffer for the whole batch.
        let udf: ScalarUDF = rs_values_udf().into();
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
        let geoms = create_geom_array(&[Some("MULTIPOINT (0.5 9.5, 1.5 8.5)")], &geom_type);
        let result = tester
            .invoke(vec![
                ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(raster))),
                ColumnarValue::Array(geoms),
                ColumnarValue::Scalar(ScalarValue::Int32(Some(2))),
            ])
            .unwrap();
        let result = result.into_array(1).unwrap();
        assert_eq!(row(&result, 0), vec![Some(10.0), Some(40.0)]);
    }

    #[test]
    fn scalar_raster_band_column_resolves_per_row() {
        // A scalar raster with a band *column* still hoists affine/CRS/reproject
        // but resolves the band buffer per row; a NULL band element -> NULL row.
        let udf: ScalarUDF = rs_values_udf().into();
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
        let geoms = create_geom_array(
            &[
                Some("MULTIPOINT (0.5 9.5)"),
                Some("MULTIPOINT (0.5 9.5)"),
                Some("MULTIPOINT (0.5 9.5)"),
            ],
            &geom_type,
        );
        let bands: arrow_array::ArrayRef = Arc::new(Int32Array::from(vec![Some(1), Some(2), None]));
        let result = tester
            .invoke(vec![
                ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(raster))),
                ColumnarValue::Array(geoms),
                ColumnarValue::Array(bands),
            ])
            .unwrap();
        let result = result.into_array(3).unwrap();
        assert_eq!(row(&result, 0), vec![Some(1.0)]); // band 1
        assert_eq!(row(&result, 1), vec![Some(10.0)]); // band 2
        let list = result.as_any().downcast_ref::<ListArray>().unwrap();
        assert!(list.is_null(2), "NULL band element is a NULL list");
    }

    #[test]
    fn scalar_raster_null_scalar_band_is_all_null() {
        // A NULL scalar band makes every output NULL without touching the band.
        let udf: ScalarUDF = rs_values_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(
            udf,
            vec![
                RASTER,
                geom_type.clone(),
                SedonaType::Arrow(DataType::Int32),
            ],
        );
        let geoms = create_geom_array(&[Some("MULTIPOINT (0.5 9.5)")], &geom_type);
        let result = tester
            .invoke(vec![
                ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(raster_2x2().build()))),
                ColumnarValue::Array(geoms),
                ColumnarValue::Scalar(ScalarValue::Int32(None)),
            ])
            .unwrap();
        let result = result.into_array(1).unwrap();
        let list = result.as_any().downcast_ref::<ListArray>().unwrap();
        assert!(list.is_null(0), "NULL scalar band should yield a NULL list");
    }

    #[test]
    fn default_band_requires_single_band_raster_scalar_path() {
        // Same default-band ambiguity error on the scalar-raster fast path.
        let udf: ScalarUDF = rs_values_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(udf, vec![RASTER, geom_type.clone()]);
        let raster = RasterSpec::d2(2, 2)
            .band_values(&[1u8, 2, 3, 4])
            .band_values(&[10u8, 20, 30, 40])
            .transform([0.0, 1.0, 0.0, 10.0, 0.0, -1.0])
            .build();
        let geoms = create_geom_array(&[Some("MULTIPOINT (0.5 9.5)")], &geom_type);
        let err = tester
            .invoke(vec![
                ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(raster))),
                ColumnarValue::Array(geoms),
            ])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("specify which band"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn scalar_all_null_geometry_defers_band_resolution() {
        // The scalar fast path defers the default-band ambiguity check until a
        // row has a geometry to sample: an all-NULL geometry column over a
        // multiband raster returns NULL rather than erroring — matching the
        // general path, which only resolves the band on rows with a geometry.
        let udf: ScalarUDF = rs_values_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(udf, vec![RASTER, geom_type.clone()]);
        let raster = RasterSpec::d2(2, 2)
            .band_values(&[1u8, 2, 3, 4])
            .band_values(&[10u8, 20, 30, 40])
            .transform([0.0, 1.0, 0.0, 10.0, 0.0, -1.0])
            .build();
        let geoms = create_geom_array(&[None, None], &geom_type);
        let result = tester
            .invoke(vec![
                ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(raster))),
                ColumnarValue::Array(geoms),
            ])
            .unwrap();
        let result = result.into_array(2).unwrap();
        let list = result.as_any().downcast_ref::<ListArray>().unwrap();
        assert!(list.is_null(0) && list.is_null(1));
    }

    #[test]
    fn scalar_null_raster_is_all_null() {
        // A NULL scalar raster makes every row NULL.
        let udf: ScalarUDF = rs_values_udf().into();
        let geom_type = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(udf, vec![RASTER, geom_type.clone()]);
        let raster = generate_test_rasters(1, Some(0)).unwrap();
        let geoms = create_geom_array(&[Some("MULTIPOINT (0.5 9.5)")], &geom_type);
        let result = tester
            .invoke(vec![
                ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(raster))),
                ColumnarValue::Array(geoms),
            ])
            .unwrap();
        let result = result.into_array(1).unwrap();
        let list = result.as_any().downcast_ref::<ListArray>().unwrap();
        assert!(list.is_null(0));
    }
}
