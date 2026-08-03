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

//! RS_Resample UDF - Resample a raster to a new resolution/grid.
//!
//! Changes a raster's pixel grid — its width/height or its pixel size — and,
//! optionally, snaps the output origin to a grid (`gridX`/`gridY`). Band
//! count/order and per-band nodata are preserved. Pixel values are recomputed by
//! GDAL: a plain resolution change reads the source into the new grid with
//! RasterIO resampling, while a scale change that grows the extent or an origin
//! snap go through GDAL's warp (which maps every output pixel centre exactly),
//! leaving the grown/shifted border at nodata.
//!
//! `RS_Resample` is a **same-CRS** operation — it never reprojects. The
//! reference-raster overload requires the reference to share the input's CRS and
//! errors otherwise (matching Apache Sedona (Spark)).
//!
//! Dimension mode preserves the world extent (the pixel size is derived from
//! it). Scale mode keeps the requested pixel size exact and grows the extent up
//! to one pixel so it tiles into whole pixels — matching Sedona Spark's
//! `RS_Resample`.
//!
//! The argument surface mirrors Sedona Spark's positional overloads verbatim, so
//! Spark SQL tends to run unchanged: a reference-raster form, a `(widthOrScale,
//! heightOrScale)` form, and a `(widthOrScale, heightOrScale, gridX, gridY)`
//! form, each ending in `(useScale, algorithm)`.

use std::sync::Arc;

use arrow_array::ArrayRef;
use arrow_schema::DataType;
use datafusion_common::cast::{as_boolean_array, as_float64_array, as_string_array};
use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_common::exec_err;
use datafusion_expr::{ColumnarValue, Volatility};

use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_gdal::gdal::Gdal;
use sedona_gdal::raster::types::ResampleAlg;
use sedona_raster::array::RasterRefImpl;
use sedona_raster::builder::RasterBuilder;
use sedona_raster::geo_transform::{GeoTransform, GeoTransformEx};
use sedona_raster::traits::RasterRef;
use sedona_raster_functions::crs_utils::{crs_transform_required, resolve_crs};
use sedona_raster_functions::rs_ensure_loaded::{
    NEEDS_PIXELS_METADATA_KEY, RETURNS_BYTES_METADATA_KEY,
};
use sedona_raster_functions::RasterExecutor;
use sedona_schema::datatypes::{SedonaType, RASTER};
use sedona_schema::matchers::ArgMatcher;

use crate::gdal_common::{raster_ref_to_gdal_mem, with_gdal, GdalBandLayout};
use crate::gdal_dataset_provider::configure_thread_local_options;
use crate::utils::{
    append_resampled_nd_from_dataset, append_warped_nd_from_dataset, parse_resample_algorithm,
    reject_lossy_resample_dtypes, Grid, OutputGrid,
};

/// RS_Resample() scalar UDF implementation.
///
/// Resamples a raster onto a new pixel grid in the same CRS. The positional
/// overloads match Apache Sedona (Spark):
/// - `RS_Resample(raster, referenceRaster, useScale, algorithm)` — 4 args
/// - `RS_Resample(raster, widthOrScale, heightOrScale, useScale, algorithm)` — 5 args
/// - `RS_Resample(raster, widthOrScale, heightOrScale, gridX, gridY, useScale, algorithm)` — 7 args
pub fn rs_resample_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_resample",
        vec![
            Arc::new(RsResample { arg_count: 4 }),
            Arc::new(RsResample { arg_count: 5 }),
            Arc::new(RsResample { arg_count: 7 }),
        ],
        Volatility::Immutable,
    )
    // Reads band pixels (so the planner materializes OutDb rasters via
    // RS_EnsureLoaded first) and emits a fresh InDb raster (so its output is
    // already loaded and isn't wrapped again).
    .with_metadata(NEEDS_PIXELS_METADATA_KEY, "true")
    .with_metadata(RETURNS_BYTES_METADATA_KEY, "true")
}

/// Kernel implementation for RS_Resample.
///
/// The `useScale` boolean and the 4/5/7-argument overload ladder below are
/// Sedona Spark's `RS_Resample` signature reproduced verbatim, so Spark SQL
/// runs unchanged. That parity is why the modes are one function switching on
/// `arg_count` / `useScale` rather than separately-named functions: the shared
/// SQL surface is deliberate, not an accident of implementation.
#[derive(Debug)]
struct RsResample {
    /// Number of arguments in the matched signature (4, 5, or 7).
    arg_count: usize,
}

impl SedonaScalarKernel for RsResample {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matchers = match self.arg_count {
            // (raster, referenceRaster, useScale, algorithm)
            4 => vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_raster(),
                ArgMatcher::is_boolean(),
                ArgMatcher::is_string(),
            ],
            // (raster, widthOrScale, heightOrScale, useScale, algorithm)
            5 => vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_numeric(),
                ArgMatcher::is_numeric(),
                ArgMatcher::is_boolean(),
                ArgMatcher::is_string(),
            ],
            // (raster, widthOrScale, heightOrScale, gridX, gridY, useScale, algorithm)
            7 => vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_numeric(),
                ArgMatcher::is_numeric(),
                ArgMatcher::is_numeric(),
                ArgMatcher::is_numeric(),
                ArgMatcher::is_boolean(),
                ArgMatcher::is_string(),
            ],
            _ => {
                return sedona_internal_err!(
                    "RS_Resample: unexpected arg_count {}",
                    self.arg_count
                );
            }
        };
        ArgMatcher::new(matchers, RASTER).match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        self.invoke_batch_from_args(arg_types, args, &SedonaType::Arrow(DataType::Null), 0, None)
    }

    fn invoke_batch_from_args(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
        _return_type: &SedonaType,
        _num_rows: usize,
        config_options: Option<&ConfigOptions>,
    ) -> Result<ColumnarValue> {
        match self.arg_count {
            4 => self.invoke_reference(arg_types, args, config_options),
            5 | 7 => self.invoke_dims_or_scale(arg_types, args, config_options),
            other => sedona_internal_err!("RS_Resample: unexpected arg_count {other}"),
        }
    }
}

impl RsResample {
    /// `RS_Resample(raster, referenceRaster, useScale, algorithm)`: the target
    /// grid (dimensions or pixel size) and origin come from the reference raster,
    /// which must share the input's CRS.
    fn invoke_reference(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
        config_options: Option<&ConfigOptions>,
    ) -> Result<ColumnarValue> {
        let num_iterations = RasterExecutor::num_iterations_over(args);

        let use_scale_array = boolean_column(&args[2], num_iterations)?;
        let mut use_scale_iter = use_scale_array.iter();
        let algorithm_array = string_column(&args[3], num_iterations)?;
        let mut algorithm_iter = algorithm_array.iter();

        let mut builder = RasterBuilder::new(num_iterations);

        // The executor iterates the (raster, referenceRaster) pair; useScale and
        // algorithm are advanced in lockstep below.
        let exec_arg_types = vec![arg_types[0].clone(), arg_types[1].clone()];
        let exec_args = vec![args[0].clone(), args[1].clone()];
        let executor =
            RasterExecutor::new_with_num_iterations(&exec_arg_types, &exec_args, num_iterations);

        with_gdal(|gdal| {
            configure_thread_local_options(gdal, config_options)?;
            executor.execute_raster_raster_void(|_i, raster_opt, reference_opt| {
                let use_scale = use_scale_iter.next().flatten();
                let algorithm = algorithm_iter.next().flatten();
                let (Some(raster), Some(reference), Some(use_scale), Some(algorithm)) =
                    (raster_opt, reference_opt, use_scale, algorithm)
                else {
                    builder.append_null()?;
                    return Ok(());
                };

                // Same-CRS only: RS_Resample never reprojects (matching Spark,
                // which throws when the reference SRID differs). Compare CRSes
                // semantically rather than by raw string, so two equivalent
                // encodings of the same CRS are not spuriously rejected; a CRS on
                // exactly one side is likewise an error (via crs_transform_required).
                let raster_crs = resolve_crs(raster.crs())?;
                let reference_crs = resolve_crs(reference.crs())?;
                if crs_transform_required(
                    raster_crs.as_deref(),
                    reference_crs.as_deref(),
                    "input raster",
                    "referenceRaster",
                )? {
                    return exec_err!(
                        "RS_Resample: referenceRaster CRS differs from the input raster CRS; \
                         RS_Resample does not reproject"
                    );
                }

                let plan = plan_from_reference(reference, use_scale, algorithm)?;
                resample_raster(gdal, raster, &plan, &mut builder)?;
                Ok(())
            })?;

            let out: ArrayRef = Arc::new(builder.finish()?);
            RasterExecutor::finish_over(args, out)
        })
    }

    /// The 5- and 7-argument overloads: the target grid comes from
    /// `(widthOrScale, heightOrScale)`, with an optional origin snap to
    /// `(gridX, gridY)` in the 7-argument form.
    fn invoke_dims_or_scale(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
        config_options: Option<&ConfigOptions>,
    ) -> Result<ColumnarValue> {
        let num_iterations = RasterExecutor::num_iterations_over(args);

        // Column indices differ between the 5- and 7-argument overloads.
        let has_grid = self.arg_count == 7;
        let (use_scale_idx, algorithm_idx) = if has_grid { (5, 6) } else { (3, 4) };

        let width_array = float_column(&args[1], num_iterations)?;
        let height_array = float_column(&args[2], num_iterations)?;
        let mut width_iter = width_array.iter();
        let mut height_iter = height_array.iter();

        let grid_x_array = if has_grid {
            Some(float_column(&args[3], num_iterations)?)
        } else {
            None
        };
        let grid_y_array = if has_grid {
            Some(float_column(&args[4], num_iterations)?)
        } else {
            None
        };
        let mut grid_x_iter = grid_x_array.as_ref().map(|a| a.iter());
        let mut grid_y_iter = grid_y_array.as_ref().map(|a| a.iter());

        let use_scale_array = boolean_column(&args[use_scale_idx], num_iterations)?;
        let mut use_scale_iter = use_scale_array.iter();
        let algorithm_array = string_column(&args[algorithm_idx], num_iterations)?;
        let mut algorithm_iter = algorithm_array.iter();

        let mut builder = RasterBuilder::new(num_iterations);

        let exec_arg_types = vec![arg_types[0].clone()];
        let exec_args = vec![args[0].clone()];
        let executor =
            RasterExecutor::new_with_num_iterations(&exec_arg_types, &exec_args, num_iterations);

        with_gdal(|gdal| {
            configure_thread_local_options(gdal, config_options)?;
            executor.execute_raster_void(|_i, raster_opt| {
                let width_or_scale = width_iter.next().flatten();
                let height_or_scale = height_iter.next().flatten();
                let grid_x = grid_x_iter.as_mut().and_then(|it| it.next().flatten());
                let grid_y = grid_y_iter.as_mut().and_then(|it| it.next().flatten());
                let use_scale = use_scale_iter.next().flatten();
                let algorithm = algorithm_iter.next().flatten();

                let (
                    Some(raster),
                    Some(width_or_scale),
                    Some(height_or_scale),
                    Some(use_scale),
                    Some(algorithm),
                ) = (
                    raster_opt,
                    width_or_scale,
                    height_or_scale,
                    use_scale,
                    algorithm,
                )
                else {
                    builder.append_null()?;
                    return Ok(());
                };

                let grid = match (has_grid, grid_x, grid_y) {
                    (false, _, _) => None,
                    (true, Some(gx), Some(gy)) => Some((gx, gy)),
                    // A NULL gridX/gridY in the 7-arg form yields a NULL row.
                    (true, _, _) => {
                        builder.append_null()?;
                        return Ok(());
                    }
                };

                let plan = plan_from_dims_or_scale(
                    width_or_scale,
                    height_or_scale,
                    grid,
                    use_scale,
                    algorithm,
                )?;
                resample_raster(gdal, raster, &plan, &mut builder)?;
                Ok(())
            })?;

            let out: ArrayRef = Arc::new(builder.finish()?);
            RasterExecutor::finish_over(args, out)
        })
    }
}

/// Cast one argument to a `Float64` array (expanding a scalar to `num_iterations`
/// rows), so a per-row column and a scalar are iterated the same way.
fn float_column(arg: &ColumnarValue, num_iterations: usize) -> Result<arrow_array::Float64Array> {
    let array = arg
        .clone()
        .cast_to(&DataType::Float64, None)?
        .into_array(num_iterations)?;
    Ok(as_float64_array(&array)?.clone())
}

/// Cast one argument to a `Boolean` array (expanding a scalar to `num_iterations`
/// rows).
fn boolean_column(arg: &ColumnarValue, num_iterations: usize) -> Result<arrow_array::BooleanArray> {
    let array = arg
        .clone()
        .cast_to(&DataType::Boolean, None)?
        .into_array(num_iterations)?;
    Ok(as_boolean_array(&array)?.clone())
}

/// Cast one argument to a `Utf8` array (expanding a scalar to `num_iterations`
/// rows).
fn string_column(arg: &ColumnarValue, num_iterations: usize) -> Result<arrow_array::StringArray> {
    let array = arg
        .clone()
        .cast_to(&DataType::Utf8, None)?
        .into_array(num_iterations)?;
    Ok(as_string_array(&array)?.clone())
}

/// Which axis the output grid is pinned on.
#[derive(Debug, Clone, Copy, PartialEq)]
enum TargetGrid {
    /// Fixed output dimensions; the pixel size is derived so the world extent
    /// is preserved.
    Dims { width: i64, height: i64 },
    /// Fixed target pixel size; the output dimensions are `ceil(extent / scale)`
    /// and the extent grows up to one pixel so it tiles into whole pixels
    /// (matching Sedona Spark).
    Scale { scale_x: f64, scale_y: f64 },
}

/// A validated resample request: the target grid, an optional origin snap, and
/// the resampling algorithm.
#[derive(Debug, Clone, PartialEq)]
struct ResamplePlan {
    target: TargetGrid,
    /// `(grid_x, grid_y)` if the output origin is snapped to a grid.
    grid_snap: Option<(f64, f64)>,
    algorithm: ResampleAlg,
}

/// Build a plan from a `(widthOrScale, heightOrScale)` pair and optional origin
/// snap. When `use_scale` is false the pair is target dimensions (rejecting a
/// fractional width/height); when true it is the target pixel size.
fn plan_from_dims_or_scale(
    width_or_scale: f64,
    height_or_scale: f64,
    grid: Option<(f64, f64)>,
    use_scale: bool,
    algorithm: &str,
) -> Result<ResamplePlan> {
    let algorithm = parse_resample_algorithm(algorithm, "RS_Resample")?;

    let target = if use_scale {
        if !width_or_scale.is_finite()
            || !height_or_scale.is_finite()
            || width_or_scale == 0.0
            || height_or_scale == 0.0
        {
            return exec_err!(
                "RS_Resample: scaleX and scaleY must be finite and non-zero \
                 (got {width_or_scale}, {height_or_scale})"
            );
        }
        TargetGrid::Scale {
            scale_x: width_or_scale,
            scale_y: height_or_scale,
        }
    } else {
        let width = whole_dimension(width_or_scale, "width")?;
        let height = whole_dimension(height_or_scale, "height")?;
        TargetGrid::Dims { width, height }
    };

    let grid_snap = match grid {
        Some((grid_x, grid_y)) => {
            if !grid_x.is_finite() || !grid_y.is_finite() {
                return exec_err!("RS_Resample: gridX and gridY must be finite");
            }
            Some((grid_x, grid_y))
        }
        None => None,
    };

    Ok(ResamplePlan {
        target,
        grid_snap,
        algorithm,
    })
}

/// Build a plan from a reference raster: its dimensions (or pixel size, when
/// `use_scale`) and its upper-left origin as the snap grid. The core grid math is
/// shared with [`plan_from_dims_or_scale`].
fn plan_from_reference(
    reference: &RasterRefImpl<'_>,
    use_scale: bool,
    algorithm: &str,
) -> Result<ResamplePlan> {
    let t = reference.transform();
    let (width_or_scale, height_or_scale) = if use_scale {
        (t[1], t[5])
    } else {
        (reference.width()? as f64, reference.height()? as f64)
    };
    // The reference's upper-left corner is the grid the output origin snaps to.
    let grid = Some((t[0], t[3]));
    plan_from_dims_or_scale(width_or_scale, height_or_scale, grid, use_scale, algorithm)
}

/// A whole-number dimension from a `Double`: rejects a non-finite, fractional, or
/// non-positive value rather than silently truncating it.
fn whole_dimension(value: f64, name: &str) -> Result<i64> {
    if !value.is_finite() {
        return exec_err!("RS_Resample: {name} must be finite (got {value})");
    }
    if value.fract() != 0.0 {
        return exec_err!(
            "RS_Resample: {name} must be a whole number when useScale is false (got {value})"
        );
    }
    let dim = value as i64;
    if dim <= 0 {
        return exec_err!("RS_Resample: {name} must be positive (got {dim})");
    }
    if dim > i32::MAX as i64 {
        return exec_err!("RS_Resample: {name} {dim} exceeds the maximum raster dimension");
    }
    Ok(dim)
}

/// Spark's `approximateEquals` tolerance for deciding whether the grid snap
/// actually moved the origin. This is a world-coordinate distance, so on a
/// lon/lat raster `1e-6` degrees is ~10 cm at the equator — below the accuracy
/// anyone resampling in lon/lat is likely to care about; for projected rasters
/// it is well below one pixel.
const SNAP_EPS: f64 = 1e-6;

/// Resample one raster into `builder` according to `plan`.
///
/// Band count/order and per-band nodata are preserved. A plain dimension change
/// takes the extent-preserving RasterIO path; a scale change (which grows the
/// extent) or an origin snap go through GDAL's warp, which maps every output
/// pixel centre exactly and fills any output cells outside the source footprint
/// with nodata. The resample is a 2-D `(y, x)` operation broadcast across every
/// non-spatial
/// plane of an N-D band via [`GdalBandLayout`].
fn resample_raster(
    gdal: &Gdal,
    raster: &RasterRefImpl<'_>,
    plan: &ResamplePlan,
    builder: &mut RasterBuilder,
) -> Result<()> {
    let src = Grid::from_raster(raster)?;
    if src.width <= 0 || src.height <= 0 {
        return exec_err!(
            "RS_Resample: source raster has non-positive dimensions {}x{}",
            src.width,
            src.height
        );
    }

    let band_count = raster.bands().len();

    // The regrid path — a scale change, origin snap, or reference grid — moves
    // the output off the source grid and/or grows the extent past the source
    // footprint, which a single axis-aligned RasterIO read cannot express, so it
    // goes through GDAL's warp (which maps every output pixel centre exactly).
    // A plain dimension change stays on the extent-preserving RasterIO
    // decimation.
    let is_regrid = matches!(plan.target, TargetGrid::Scale { .. }) || plan.grid_snap.is_some();

    // Int64/UInt64 are preserved only on the nearest-neighbour RasterIO fast path
    // (a plain dimension change), where each output pixel is a copied source
    // sample — a value selection, not a computation — so it is bit-exact for any
    // dtype whether up- or down-sampling (pinned by the
    // nearest_fast_path_preserves_* tests). Every other case routes pixels through
    // a floating working type — the warp path always, and RasterIO for any
    // interpolating algorithm — so those dtypes are rejected (Int32/UInt32 + Mode
    // only on GDAL < 3.13). RS_ReprojectMatch shares this check.
    let routes_through_float = is_regrid || plan.algorithm != ResampleAlg::NearestNeighbour;
    reject_lossy_resample_dtypes(
        gdal,
        raster,
        band_count,
        plan.algorithm,
        routes_through_float,
        "RS_Resample",
    )?;

    // A scale change or origin snap on a skewed (non-north-up) raster is an
    // explicit unsupported case: the extent-preserving dimension path carries the
    // shear through unchanged, but growing the extent or snapping the origin
    // against a sheared grid is not supported.
    if is_regrid && (src.transform[2] != 0.0 || src.transform[4] != 0.0) {
        return exec_err!(
            "RS_Resample: a scale change or grid snap on a skewed (non-north-up) raster is not \
             supported (only an extent-preserving width/height change is)"
        );
    }

    let band_indices: Vec<usize> = (1..=band_count).collect();
    let layout = GdalBandLayout::from_raster(raster, &band_indices)?;
    // SAFETY: the returned dataset references `raster`'s band bytes zero-copy;
    // it is only read below and dropped before `resample_raster` returns, so
    // `raster` outlives it.
    let src_dataset = unsafe { raster_ref_to_gdal_mem(gdal, raster, &band_indices)? };

    // Output geotransform matching Sedona Spark: the skew is carried through
    // unchanged and the dimensions / pixel size follow the requested mode. The
    // output stays in the source CRS — RS_Resample never reprojects.
    let output = OutputGrid {
        grid: compute_output_grid(&src, plan),
        crs: raster.crs(),
        alg: plan.algorithm,
    };

    if !is_regrid {
        // Extent-preserving RasterIO decimation supplies the pixels (bit-exact
        // for nearest; interpolating algorithms blend as GDAL RasterIO does).
        return append_resampled_nd_from_dataset(&src_dataset, &layout, builder, &output);
    }

    // Warp path: scale-mode grow-extent, origin snap, and reference-grid
    // resampling. Same CRS on both sides (RS_Resample never reprojects), so this
    // is a pure regrid onto `output` — GDAL maps each output pixel centre exactly
    // (no covered-window clamp) and fills the grown/shifted border, uncovered by
    // the source, with each band's nodata. `0.0` uses GDAL's own default
    // working-memory cache size.
    append_warped_nd_from_dataset(gdal, &src_dataset, &layout, builder, &output, 0.0)
}

/// Resolve the output grid, mirroring Apache Sedona (Spark)'s
/// `RasterEditors.resample`: dimension mode preserves the extent, scale mode
/// keeps the pixel size exact and grows the extent, and `gridX`/`gridY` snap the
/// origin outward to the grid. The CRS is unchanged from the source.
fn compute_output_grid(src: &Grid, plan: &ResamplePlan) -> Grid {
    let env = src.envelope();
    let [src_ul_x, _, src_skew_x, src_ul_y, src_skew_y, src_scale_y] = src.transform;

    // Dimensions and pixel size per the requested mode. Skew is carried through
    // unchanged (Spark keeps the original shear in both modes).
    let (mut width, mut height, mut scale_x, mut scale_y) = match plan.target {
        TargetGrid::Dims { width, height } => {
            let sx = (env.max_x - env.min_x) / width as f64;
            let sy = src_scale_y.signum() * (env.max_y - env.min_y) / height as f64;
            (width, height, sx, sy)
        }
        TargetGrid::Scale { scale_x, scale_y } => {
            let width = ((env.max_x - env.min_x) / scale_x.abs()).ceil().max(1.0) as i64;
            let height = ((env.max_y - env.min_y) / scale_y.abs()).ceil().max(1.0) as i64;
            (width, height, scale_x, scale_y)
        }
    };
    let skew_x = src_skew_x;
    let skew_y = src_skew_y;
    let mut upper_left_x = src_ul_x;
    let mut upper_left_y = src_ul_y;

    // Origin snap: move the origin to the grid-cell corner containing it
    // (expanding outward so the original extent stays covered), then regrow the
    // dimension (scale mode) or re-fit the pixel size (dimension mode).
    if let Some((grid_x, grid_y)) = plan.grid_snap {
        let current = [upper_left_x, scale_x, skew_x, upper_left_y, skew_y, scale_y];
        let (snap_x, snap_y) = snap_origin(grid_x, grid_y, &current);
        if (snap_x - upper_left_x).abs() > SNAP_EPS {
            match plan.target {
                TargetGrid::Dims { .. } => scale_x = (env.max_x - snap_x).abs() / width as f64,
                _ => {
                    width = ((env.max_x - snap_x).abs() / scale_x.abs()).ceil().max(1.0) as i64;
                }
            }
            upper_left_x = snap_x;
        }
        if (snap_y - upper_left_y).abs() > SNAP_EPS {
            match plan.target {
                TargetGrid::Dims { .. } => {
                    scale_y = scale_y.signum() * (env.min_y - snap_y).abs() / height as f64;
                }
                _ => {
                    height = ((env.min_y - snap_y).abs() / scale_y.abs()).ceil().max(1.0) as i64;
                }
            }
            upper_left_y = snap_y;
        }
    }

    Grid {
        transform: [upper_left_x, scale_x, skew_x, upper_left_y, skew_y, scale_y],
        width,
        height,
    }
}

/// Snap the origin of `transform` (its `[0]`/`[3]` upper-left) to the grid
/// anchored at `(grid_x, grid_y)` with `transform`'s pixel size and skew: the
/// returned point is the grid-cell corner that contains the origin
/// (inverse-affine then floor, matching Spark's `getGridCoordinatesFromWorld` +
/// `getWorldCornerCoordinates`).
///
/// The math is the shared affine machinery: build the grid's own geotransform
/// (the source pixel size/skew anchored at `(grid_x, grid_y)`), invert it to
/// find the origin's fractional cell, floor to the containing corner, and map
/// that corner back. A degenerate (uninvertible) transform leaves the origin
/// unmoved.
fn snap_origin(grid_x: f64, grid_y: f64, transform: &GeoTransform) -> (f64, f64) {
    let [ul_x, scale_x, skew_x, ul_y, skew_y, scale_y] = *transform;
    let anchored: GeoTransform = [grid_x, scale_x, skew_x, grid_y, skew_y, scale_y];
    let Ok(inverse) = anchored.invert() else {
        return (ul_x, ul_y);
    };
    let (col, row) = inverse.apply(ul_x, ul_y);
    anchored.apply(col.floor(), row.floor())
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_common::ScalarValue;
    use sedona_raster::array::RasterStructArray;
    use sedona_testing::raster_spec::{
        assert_raster_scalar_equals, assert_rasters_equal, raster_array, RasterSpec,
    };
    use sedona_testing::testers::ScalarUdfTester;

    /// Tester for the 5-argument `(raster, widthOrScale, heightOrScale, useScale,
    /// algorithm)` overload.
    fn tester5() -> ScalarUdfTester {
        ScalarUdfTester::new(
            rs_resample_udf().into(),
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Boolean),
                SedonaType::Arrow(DataType::Utf8),
            ],
        )
    }

    /// Tester for the 7-argument `(..., gridX, gridY, useScale, algorithm)`
    /// overload.
    fn tester7() -> ScalarUdfTester {
        ScalarUdfTester::new(
            rs_resample_udf().into(),
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Boolean),
                SedonaType::Arrow(DataType::Utf8),
            ],
        )
    }

    /// Tester for the 4-argument `(raster, referenceRaster, useScale, algorithm)`
    /// overload.
    fn tester_ref() -> ScalarUdfTester {
        ScalarUdfTester::new(
            rs_resample_udf().into(),
            vec![
                RASTER,
                RASTER,
                SedonaType::Arrow(DataType::Boolean),
                SedonaType::Arrow(DataType::Utf8),
            ],
        )
    }

    fn raster_scalar(spec: &RasterSpec) -> ColumnarValue {
        ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(spec.build())))
    }

    fn f64_scalar(v: f64) -> ColumnarValue {
        ColumnarValue::Scalar(ScalarValue::Float64(Some(v)))
    }

    fn bool_scalar(v: bool) -> ColumnarValue {
        ColumnarValue::Scalar(ScalarValue::Boolean(Some(v)))
    }

    fn str_scalar(v: &str) -> ColumnarValue {
        ColumnarValue::Scalar(ScalarValue::Utf8(Some(v.to_string())))
    }

    fn unwrap_scalar(result: Result<ColumnarValue>) -> ScalarValue {
        match result.unwrap() {
            ColumnarValue::Scalar(sv) => sv,
            other => panic!("expected a scalar result, got {other:?}"),
        }
    }

    /// Invoke the 5-argument dimension form `(width, height)` with nearest.
    fn resample_dims(source: &RasterSpec, width: f64, height: f64) -> ScalarValue {
        unwrap_scalar(tester5().invoke(vec![
            raster_scalar(source),
            f64_scalar(width),
            f64_scalar(height),
            bool_scalar(false),
            str_scalar("NearestNeighbor"),
        ]))
    }

    /// Invoke the 5-argument scale form `(scaleX, scaleY)` with nearest.
    fn resample_scale(source: &RasterSpec, scale_x: f64, scale_y: f64) -> ScalarValue {
        unwrap_scalar(tester5().invoke(vec![
            raster_scalar(source),
            f64_scalar(scale_x),
            f64_scalar(scale_y),
            bool_scalar(true),
            str_scalar("NearestNeighbor"),
        ]))
    }

    #[test]
    fn resample_to_same_grid_is_identity() {
        // Resampling to the source dimensions is a straight copy: pixels,
        // transform, CRS, and nodata are all preserved exactly.
        let source = RasterSpec::d2(2, 1)
            .band_values(&[10u8, 20])
            .nodata(255u8)
            .bbox(0.0, 0.0, 4.0, 2.0)
            .crs(Some("EPSG:4326"));

        let result = resample_dims(&source, 2.0, 1.0);
        assert_raster_scalar_equals(&result, &source);
    }

    #[test]
    fn nearest_upsample_replicates_pixels() {
        // A 2x1 -> 4x2 nearest upsample by an integer factor replicates each
        // source pixel into a 2x2 block (block replication is unambiguous, so
        // this is bit-exact and stable across GDAL versions). The world extent
        // x[0,4], y[0,2] is preserved, halving the pixel size.
        let source = RasterSpec::d2(2, 1)
            .band_values(&[10u8, 20])
            .bbox(0.0, 0.0, 4.0, 2.0)
            .crs(Some("EPSG:4326"));

        let result = resample_dims(&source, 4.0, 2.0);

        let expected = RasterSpec::d2(4, 2)
            .band_values(&[10u8, 10, 20, 20, 10, 10, 20, 20])
            .bbox(0.0, 0.0, 4.0, 2.0)
            .crs(Some("EPSG:4326"));
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn nearest_downsample_block_constant_is_stable() {
        // Nearest downsampling picks source pixels; which column of a 2:1
        // decimation it selects is GDAL-version-dependent, so the source here
        // is block-constant per output cell (cols 0,1 -> 10; cols 2,3 -> 20;
        // both rows identical) making the selected value unambiguous: [10, 20].
        let source = RasterSpec::d2(4, 2)
            .band_values(&[10u8, 10, 20, 20, 10, 10, 20, 20])
            .bbox(0.0, 0.0, 4.0, 2.0)
            .crs(Some("EPSG:4326"));

        let result = resample_dims(&source, 2.0, 1.0);

        let expected = RasterSpec::d2(2, 1)
            .band_values(&[10u8, 20])
            .bbox(0.0, 0.0, 4.0, 2.0)
            .crs(Some("EPSG:4326"));
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn scale_mode_derives_dimensions() {
        // Source pixel size is 2 (extent 4 over 2 columns, extent 2 over 1 row);
        // requesting pixel size 1 gives dims ceil(4/1)=4 x ceil(2/1)=2. The extent
        // divides evenly, so the extent does not grow: the output pixel size is
        // exactly the requested 1 / -1 and the bbox is unchanged.
        let source = RasterSpec::d2(2, 1)
            .band_values(&[10u8, 20])
            .bbox(0.0, 0.0, 4.0, 2.0)
            .crs(Some("EPSG:4326"));

        let result = resample_scale(&source, 1.0, -1.0);

        let expected = RasterSpec::d2(4, 2)
            .band_values(&[10u8, 10, 20, 20, 10, 10, 20, 20])
            .bbox(0.0, 0.0, 4.0, 2.0)
            .crs(Some("EPSG:4326"));
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn scale_mode_grows_extent_to_whole_pixels() {
        // Extent 4 wide with pixel size 3 does not tile evenly: Spark keeps the
        // scale exact (3) and grows to ceil(4/3)=2 columns, so the output spans
        // x[0,6] (2*3) — one grown column past the source right edge (x=4) — and
        // that grown column, uncovered by the source, is nodata. Height: extent 2
        // with scale -3 gives ceil(2/3)=1 row spanning y[-1,2].
        //
        // Nearest maps each output column center to a source column: col 0 center
        // x=1.5 -> source col 0 (10); col 1 center x=4.5 is past the source, so it
        // falls to the nodata fill. Row 0 center y=0.5 -> source row 0.
        let source = RasterSpec::d2(2, 1)
            .band_values(&[10u8, 20])
            .nodata(255u8)
            .bbox(0.0, 0.0, 4.0, 2.0)
            .crs(Some("EPSG:4326"));

        let result = resample_scale(&source, 3.0, -3.0);

        // Origin unchanged (0, 2); output 2x1 at pixel size 3/-3 -> bbox x[0,6],
        // y[-1,2]. The source only covers x[0,4], so the second column is nodata.
        let expected = RasterSpec::d2(2, 1)
            .band_values(&[10u8, 255])
            .nodata(255u8)
            .bbox(0.0, -1.0, 6.0, 2.0)
            .crs(Some("EPSG:4326"));
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn scale_mode_grows_extent_without_crs() {
        // The regrid path must also work on a CRS-less raster (a same-pixel-space
        // regrid) — the Python parity fixtures write CRS-less GeoTIFFs. Same grow
        // arithmetic as the CRS'd case above.
        let source = RasterSpec::d2(2, 1)
            .band_values(&[10u8, 20])
            .nodata(255u8)
            .bbox(0.0, 0.0, 4.0, 2.0)
            .crs(None);

        let result = resample_scale(&source, 3.0, -3.0);

        let expected = RasterSpec::d2(2, 1)
            .band_values(&[10u8, 255])
            .nodata(255u8)
            .bbox(0.0, -1.0, 6.0, 2.0)
            .crs(None);
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn grid_snap_shifts_origin_and_regrows() {
        // Scale-mode pixel size 3/-3 snapped to a grid anchored at (1, 3): the
        // source origin (0, 2) snaps outward to the grid corner containing it —
        // x: floor((0-1)/3) = -1 -> 1 + (-1)*3 = -2; y: floor((2-3)/-3) = 0 ->
        // 3 + 0*(-3) = 3. New origin (-2, 3) (a corner of the (1,3)/3 grid), and
        // the source extent stays covered. Width regrows to reach the source
        // right edge x=4: ceil(|4-(-2)|/3) = 2; height ceil(|0-3|/3) = 1.
        let source = RasterSpec::d2(2, 1)
            .band_values(&[10u8, 20])
            .nodata(255u8)
            .bbox(0.0, 0.0, 4.0, 2.0)
            .crs(Some("EPSG:4326"));

        // 7-arg overload: (raster, scaleX=3, scaleY=-3, gridX=1, gridY=3,
        // useScale=true, algorithm).
        let result = unwrap_scalar(tester7().invoke(vec![
            raster_scalar(&source),
            f64_scalar(3.0),
            f64_scalar(-3.0),
            f64_scalar(1.0),
            f64_scalar(3.0),
            bool_scalar(true),
            str_scalar("NearestNeighbor"),
        ]));

        // Origin (-2, 3), 2x1 at 3/-3 -> bbox x[-2,4], y[0,3]. Cell centers are
        // (-0.5, 1.5) — left of the source (x<0), nodata — and (2.5, 1.5) —
        // inside source column 1, value 20.
        let expected = RasterSpec::d2(2, 1)
            .band_values(&[255u8, 20])
            .nodata(255u8)
            .bbox(-2.0, 0.0, 4.0, 3.0)
            .crs(Some("EPSG:4326"));
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn reference_raster_matches_its_grid() {
        // The 4-arg reference overload with useScale=false takes the reference's
        // dimensions and origin. Here the reference is a 4x2 grid over the same
        // extent as the 2x1 source, so the result is the 2x upsample.
        let source = RasterSpec::d2(2, 1)
            .band_values(&[10u8, 20])
            .bbox(0.0, 0.0, 4.0, 2.0)
            .crs(Some("EPSG:4326"));
        let reference = RasterSpec::d2(4, 2)
            .band_values(&[0u8; 8])
            .bbox(0.0, 0.0, 4.0, 2.0)
            .crs(Some("EPSG:4326"));

        let result = unwrap_scalar(tester_ref().invoke(vec![
            raster_scalar(&source),
            raster_scalar(&reference),
            bool_scalar(false),
            str_scalar("NearestNeighbor"),
        ]));

        let expected = RasterSpec::d2(4, 2)
            .band_values(&[10u8, 10, 20, 20, 10, 10, 20, 20])
            .bbox(0.0, 0.0, 4.0, 2.0)
            .crs(Some("EPSG:4326"));
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn reference_raster_crs_mismatch_errors() {
        // A reference raster in a different CRS must error — RS_Resample never
        // reprojects.
        let source = RasterSpec::d2(2, 1)
            .band_values(&[10u8, 20])
            .bbox(0.0, 0.0, 4.0, 2.0)
            .crs(Some("EPSG:4326"));
        let reference = RasterSpec::d2(4, 2)
            .band_values(&[0u8; 8])
            .bbox(0.0, 0.0, 4.0, 2.0)
            .crs(Some("EPSG:3857"));

        let err = tester_ref()
            .invoke(vec![
                raster_scalar(&source),
                raster_scalar(&reference),
                bool_scalar(false),
                str_scalar("NearestNeighbor"),
            ])
            .unwrap_err()
            .to_string();
        assert!(err.contains("does not reproject"), "got: {err}");
    }

    #[test]
    fn multiband_nearest_upsample() {
        // Every band is resampled and the band order/count preserved.
        let source = RasterSpec::d2(2, 1)
            .band_values(&[10u8, 20])
            .nodata(0u8)
            .band_values(&[1u8, 2])
            .bbox(0.0, 0.0, 4.0, 2.0)
            .crs(Some("EPSG:4326"));

        let result = resample_dims(&source, 4.0, 2.0);

        let expected = RasterSpec::d2(4, 2)
            .band_values(&[10u8, 10, 20, 20, 10, 10, 20, 20])
            .nodata(0u8)
            .band_values(&[1u8, 1, 2, 2, 1, 1, 2, 2])
            .bbox(0.0, 0.0, 4.0, 2.0)
            .crs(Some("EPSG:4326"));
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn nd_band_broadcasts_across_planes() {
        // A [time=2, y=1, x=2] band upsampled to (y=2, x=4): the time dimension
        // is preserved and each time plane is resampled independently. Time 0 is
        // [10, 20], time 1 is [30, 40].
        let source = RasterSpec::nd(&["time", "y", "x"], &[2, 1, 2])
            .band_values(&[10u8, 20, 30, 40])
            .transform([0.0, 2.0, 0.0, 2.0, 0.0, -2.0])
            .crs(Some("EPSG:4326"));

        let result = resample_dims(&source, 4.0, 2.0);

        let expected = RasterSpec::nd(&["time", "y", "x"], &[2, 2, 4])
            .band_values(&[
                10u8, 10, 20, 20, 10, 10, 20, 20, // time 0
                30, 30, 40, 40, 30, 30, 40, 40, // time 1
            ])
            .transform([0.0, 1.0, 0.0, 2.0, 0.0, -1.0])
            .crs(Some("EPSG:4326"));
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn null_raster_yields_null() {
        let result = tester5()
            .invoke(vec![
                ColumnarValue::Array(Arc::new(raster_array(vec![
                    Some(
                        RasterSpec::d2(2, 1)
                            .band_values(&[10u8, 20])
                            .bbox(0.0, 0.0, 4.0, 2.0)
                            .crs(Some("EPSG:4326")),
                    ),
                    None,
                ]))),
                f64_scalar(4.0),
                f64_scalar(2.0),
                bool_scalar(false),
                str_scalar("NearestNeighbor"),
            ])
            .unwrap();
        let ColumnarValue::Array(arr) = result else {
            panic!("expected an array result");
        };

        let expected = vec![
            Some(
                RasterSpec::d2(4, 2)
                    .band_values(&[10u8, 10, 20, 20, 10, 10, 20, 20])
                    .bbox(0.0, 0.0, 4.0, 2.0)
                    .crs(Some("EPSG:4326")),
            ),
            None,
        ];
        assert_rasters_equal(&arr, &expected);
    }

    #[test]
    fn skewed_raster_keeps_skew_and_derives_scale_from_envelope() {
        // A skewed (non-north-up) raster resamples with the shear carried through
        // UNCHANGED and the new pixel size derived from the axis-aligned envelope
        // divided across the new dimensions, matching Sedona Spark (which keeps
        // the original skew and derives the scale from the envelope, ignoring the
        // skew for the scale). Source transform [ulx=0, scale_x=2, skew_x=0.5,
        // uly=2, skew_y=0.5, scale_y=-2] over 2x1 has envelope x in [0, 4.5],
        // y in [0, 3], so upsampling to 4x2 gives scale_x = 4.5/4 = 1.125,
        // scale_y = -3/2 = -1.5, with skew unchanged at 0.5. The pixels are the
        // extent-preserving RasterIO decimation.
        let source = RasterSpec::d2(2, 1)
            .band_values(&[10u8, 20])
            .transform([0.0, 2.0, 0.5, 2.0, 0.5, -2.0])
            .crs(Some("EPSG:4326"));

        let result = resample_dims(&source, 4.0, 2.0);

        let expected = RasterSpec::d2(4, 2)
            .band_values(&[10u8, 10, 20, 20, 10, 10, 20, 20])
            .transform([0.0, 1.125, 0.5, 2.0, 0.5, -1.5])
            .crs(Some("EPSG:4326"));
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn skewed_scale_change_errors() {
        // A skewed raster in the grow/snap regrid path (scale mode) is an
        // explicit unsupported case and errors, rather than silently producing a
        // wrong grid.
        let source = RasterSpec::d2(2, 1)
            .band_values(&[10u8, 20])
            .transform([0.0, 2.0, 0.5, 2.0, 0.5, -2.0])
            .crs(Some("EPSG:4326"));

        let err = tester5()
            .invoke(vec![
                raster_scalar(&source),
                f64_scalar(3.0),
                f64_scalar(-3.0),
                bool_scalar(true),
                str_scalar("NearestNeighbor"),
            ])
            .unwrap_err()
            .to_string();
        assert!(err.contains("skewed"), "got: {err}");
    }

    #[test]
    fn int64_allowed_only_on_the_nearest_fast_path() {
        // 64-bit ints are exact only on the nearest RasterIO fast path (a plain
        // dimension change). Any warp-bound case — an interpolating algorithm, or
        // a scale change / origin snap / reference grid even with nearest —
        // routes pixels through a floating working type that cannot represent
        // Int64/UInt64 exactly, so RS_Resample rejects it (matching
        // RS_ReprojectMatch).
        let source = RasterSpec::d2(2, 1)
            .band_values(&[10i64, 20])
            .bbox(0.0, 0.0, 4.0, 2.0)
            .crs(Some("EPSG:4326"));

        // Non-nearest dimension change: warp-bound, rejected.
        let err = tester5()
            .invoke(vec![
                raster_scalar(&source),
                f64_scalar(4.0),
                f64_scalar(2.0),
                bool_scalar(false),
                str_scalar("Bilinear"),
            ])
            .unwrap_err()
            .to_string();
        assert!(err.contains("Int64/UInt64"), "got: {err}");

        // Nearest *scale* change also warps (grow-extent / origin snap), so it is
        // rejected too despite the nearest algorithm.
        let err = tester5()
            .invoke(vec![
                raster_scalar(&source),
                f64_scalar(3.0),
                f64_scalar(-3.0),
                bool_scalar(true),
                str_scalar("NearestNeighbor"),
            ])
            .unwrap_err()
            .to_string();
        assert!(err.contains("Int64/UInt64"), "got: {err}");

        // Nearest dimension change is the bit-exact fast path and succeeds.
        tester5()
            .invoke(vec![
                raster_scalar(&source),
                f64_scalar(4.0),
                f64_scalar(2.0),
                bool_scalar(false),
                str_scalar("NearestNeighbor"),
            ])
            .unwrap();
    }

    #[test]
    fn nearest_fast_path_preserves_int64_exactly() {
        // The premise for allowing Int64/UInt64 only on the nearest fast path is
        // that RasterIO nearest copies a source sample rather than computing it, so
        // it survives values that f64 cannot represent (2^53+1 rounds to 2^53, and
        // 2^53+3 to 2^53+4, if routed through a floating working type). Verified for
        // both up- and down-sampling.
        let a: i64 = (1i64 << 53) + 1;
        let b: i64 = (1i64 << 53) + 3;

        // 2x1 -> 4x2 nearest upsample replicates each pixel.
        let up_src = RasterSpec::d2(2, 1)
            .band_values(&[a, b])
            .bbox(0.0, 0.0, 4.0, 2.0)
            .crs(Some("EPSG:4326"));
        let up_expected = RasterSpec::d2(4, 2)
            .band_values(&[a, a, b, b, a, a, b, b])
            .bbox(0.0, 0.0, 4.0, 2.0)
            .crs(Some("EPSG:4326"));
        assert_raster_scalar_equals(&resample_dims(&up_src, 4.0, 2.0), &up_expected);

        // 4x1 -> 2x1 nearest decimation (block-constant so the selected column is
        // unambiguous, as in nearest_downsample_block_constant_is_stable).
        let down_src = RasterSpec::d2(4, 1)
            .band_values(&[a, a, b, b])
            .bbox(0.0, 0.0, 4.0, 1.0)
            .crs(Some("EPSG:4326"));
        let down_expected = RasterSpec::d2(2, 1)
            .band_values(&[a, b])
            .bbox(0.0, 0.0, 4.0, 1.0)
            .crs(Some("EPSG:4326"));
        assert_raster_scalar_equals(&resample_dims(&down_src, 2.0, 1.0), &down_expected);
    }

    #[test]
    fn nearest_fast_path_preserves_uint64_exactly() {
        // UInt64 above i64::MAX and above 2^63 (neither representable in f64) also
        // survives the nearest fast path unchanged.
        let a: u64 = u64::MAX - 1;
        let b: u64 = (1u64 << 63) + 5;
        let source = RasterSpec::d2(2, 1)
            .band_values(&[a, b])
            .bbox(0.0, 0.0, 4.0, 2.0)
            .crs(Some("EPSG:4326"));
        let expected = RasterSpec::d2(4, 2)
            .band_values(&[a, a, b, b, a, a, b, b])
            .bbox(0.0, 0.0, 4.0, 2.0)
            .crs(Some("EPSG:4326"));
        assert_raster_scalar_equals(&resample_dims(&source, 4.0, 2.0), &expected);
    }

    #[test]
    fn int32_mode_gated_on_gdal_version() {
        // Mode selects the most-common source value, so it should return an exact
        // source value — but GDAL routes Mode through a 32-bit float working type
        // on GDAL < 3.13 (a double from 3.13 on), and a 32-bit float cannot
        // represent Int32 above 2^24 (2^24+1 -> 2^24, 2^24+3 -> 2^24+4). So on older
        // GDAL RS_Resample rejects Int32/UInt32 + Mode rather than silently
        // corrupting a category code; on GDAL >= 3.13 the value survives exactly.
        let a: i32 = (1i32 << 24) + 1;
        let b: i32 = (1i32 << 24) + 3;
        // 4x1 -> 2x1 block-constant windows so Mode's selection is unambiguous.
        let source = RasterSpec::d2(4, 1)
            .band_values(&[a, a, b, b])
            .bbox(0.0, 0.0, 4.0, 1.0)
            .crs(Some("EPSG:4326"));
        let invoke = || {
            tester5().invoke(vec![
                raster_scalar(&source),
                f64_scalar(2.0),
                f64_scalar(1.0),
                bool_scalar(false),
                str_scalar("Mode"),
            ])
        };

        let gdal_version = with_gdal(|gdal| Ok(gdal.version_num())).unwrap();
        if gdal_version >= 3_130_000 {
            let expected = RasterSpec::d2(2, 1)
                .band_values(&[a, b])
                .bbox(0.0, 0.0, 4.0, 1.0)
                .crs(Some("EPSG:4326"));
            assert_raster_scalar_equals(&unwrap_scalar(invoke()), &expected);
        } else {
            let err = invoke().unwrap_err().to_string();
            assert!(err.contains("Int32/UInt32"), "got: {err}");
            assert!(err.contains("Mode"), "got: {err}");
        }
    }

    #[test]
    fn bilinear_scale_change_of_constant_is_constant_via_warp() {
        // A non-nearest algorithm on the regrid (warp) path — here a scale change.
        // Interpolating a constant band yields that constant everywhere — exact
        // and GDAL-version-independent — which pins the warp path end-to-end (the
        // blended values near edges are pinned against rasterio in the Python
        // parity tests). Pixel size 1 over the 4x4 extent divides evenly (no grown
        // border), so the whole 4x4 output is the constant.
        let source = RasterSpec::d2(2, 2)
            .band_values(&[7.0f64, 7.0, 7.0, 7.0])
            .bbox(0.0, 0.0, 4.0, 4.0)
            .crs(Some("EPSG:4326"));

        // 5-arg overload: (raster, scaleX=1, scaleY=-1, useScale=true, algorithm).
        let result = unwrap_scalar(tester5().invoke(vec![
            raster_scalar(&source),
            f64_scalar(1.0),
            f64_scalar(-1.0),
            bool_scalar(true),
            str_scalar("Bilinear"),
        ]));

        let expected = RasterSpec::d2(4, 4)
            .band_values(&[7.0f64; 16])
            .bbox(0.0, 0.0, 4.0, 4.0)
            .crs(Some("EPSG:4326"));
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn fractional_dimension_errors() {
        // useScale=false with a fractional width is rejected rather than
        // truncated.
        let source = RasterSpec::d2(2, 1)
            .band_values(&[10u8, 20])
            .bbox(0.0, 0.0, 4.0, 2.0)
            .crs(Some("EPSG:4326"));

        let err = tester5()
            .invoke(vec![
                raster_scalar(&source),
                f64_scalar(3.5),
                f64_scalar(2.0),
                bool_scalar(false),
                str_scalar("NearestNeighbor"),
            ])
            .unwrap_err()
            .to_string();
        assert!(err.contains("whole number"), "got: {err}");
    }

    #[test]
    fn unknown_algorithm_errors_through_udf() {
        let source = RasterSpec::d2(2, 1)
            .band_values(&[10u8, 20])
            .bbox(0.0, 0.0, 4.0, 2.0)
            .crs(Some("EPSG:4326"));

        let err = tester5()
            .invoke(vec![
                raster_scalar(&source),
                f64_scalar(4.0),
                f64_scalar(2.0),
                bool_scalar(false),
                str_scalar("sinc"),
            ])
            .unwrap_err()
            .to_string();
        assert!(err.contains("unknown algorithm"), "got: {err}");
    }

    #[test]
    fn output_keeps_input_crs_and_origin() {
        // Structural pin: the output CRS always equals the input CRS (RS_Resample
        // never reprojects). A 4x4 EPSG:4326 raster resampled to 8x8 keeps its
        // CRS and origin.
        let source = RasterSpec::d2(4, 4)
            .band_values(&(0u8..16).collect::<Vec<_>>())
            .bbox(10.0, 40.0, 14.0, 44.0)
            .crs(Some("EPSG:4326"));

        let result = resample_dims(&source, 8.0, 8.0);

        let ScalarValue::Struct(struct_array) = &result else {
            panic!("expected a raster struct scalar, got {result:?}");
        };
        let rasters = RasterStructArray::try_new(struct_array).unwrap();
        let raster = rasters.get(0).unwrap();
        assert_eq!(raster.crs(), Some("EPSG:4326"));
        assert_eq!(raster.transform()[0], 10.0);
        assert_eq!(raster.transform()[3], 44.0);
    }
}
