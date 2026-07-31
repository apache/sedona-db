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

//! RS_ReprojectMatch UDF - Reproject a raster onto a reference raster's grid.
//!
//! Reprojects the input raster onto the reference raster's CRS, pixel grid, and
//! envelope: the output always has the *same* extent, resolution, dimensions,
//! and CRS as the reference (in the spirit of `rioxarray`'s `reproject_match`).
//! The input's band count/order, per-band data type, and nodata are preserved.
//! Pixel values are recomputed by GDAL's warp (`GDALReprojectImage`); output
//! cells the reprojected input footprint does not cover are filled with nodata.
//!
//! The reference raster contributes only its grid — transform, dimensions, and
//! CRS — never its pixels.
//!
//! Int64/UInt64 input rasters are not supported: GDAL's warp routes 64-bit
//! integer pixels through a floating working type (a `double` for
//! nearest/interpolation, a 32-bit `float` for mode on GDAL < 3.13), which
//! cannot represent them exactly. Cast to a supported type first.

use std::sync::Arc;

use arrow_array::ArrayRef;
use arrow_schema::DataType;
use datafusion_common::cast::as_string_view_array;
use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_common::{exec_err, ScalarValue};
use datafusion_expr::{ColumnarValue, Volatility};

use sedona_common::{sedona_internal_err, SedonaOptions};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_gdal::geo_transform::GeoTransform;
use sedona_gdal::raster::types::ResampleAlg;
use sedona_raster::array::RasterRefImpl;
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::RasterRef;
use sedona_raster_functions::rs_ensure_loaded::{
    NEEDS_PIXELS_METADATA_KEY, RETURNS_BYTES_METADATA_KEY,
};
use sedona_raster_functions::RasterExecutor;
use sedona_schema::datatypes::{SedonaType, RASTER};
use sedona_schema::matchers::ArgMatcher;

use crate::gdal_common::{raster_ref_to_gdal_mem, with_gdal, GdalBandLayout};
use crate::gdal_dataset_provider::configure_thread_local_options;
use crate::utils::{
    append_warped_nd_from_dataset, parse_resample_algorithm, reject_lossy_resample_dtypes, Grid,
    OutputGrid,
};

/// RS_ReprojectMatch() scalar UDF implementation.
///
/// Reprojects `raster` onto `reference`'s CRS + grid + envelope.
///
/// Signatures (matching Apache Sedona (Spark)):
/// - `RS_ReprojectMatch(raster, reference)` — 2 args (algorithm defaults to
///   `NearestNeighbor`)
/// - `RS_ReprojectMatch(raster, reference, algorithm)` — 3 args
pub fn rs_reproject_match_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_reprojectmatch",
        vec![
            Arc::new(RsReprojectMatch { arg_count: 2 }),
            Arc::new(RsReprojectMatch { arg_count: 3 }),
        ],
        Volatility::Immutable,
    )
    // Reads band pixels (so the planner materializes OutDb rasters via
    // RS_EnsureLoaded first) and emits a fresh InDb raster (so its output is
    // already loaded and isn't wrapped again).
    .with_metadata(NEEDS_PIXELS_METADATA_KEY, "true")
    .with_metadata(RETURNS_BYTES_METADATA_KEY, "true")
}

/// Kernel implementation for RS_ReprojectMatch.
#[derive(Debug)]
struct RsReprojectMatch {
    /// Number of arguments in the matched signature (2 or 3).
    arg_count: usize,
}

impl SedonaScalarKernel for RsReprojectMatch {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matchers = match self.arg_count {
            2 => vec![ArgMatcher::is_raster(), ArgMatcher::is_raster()],
            3 => vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_raster(),
                ArgMatcher::is_string(),
            ],
            _ => {
                return sedona_internal_err!(
                    "RS_ReprojectMatch: unexpected arg_count {}",
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
        let num_iterations = RasterExecutor::num_iterations_over(args);

        // Algorithm string at index 2 (when arg_count == 3); otherwise the
        // Spark default `NearestNeighbor`. Expand to a `Utf8View` array so a
        // per-row column and a scalar are handled identically (the view avoids
        // re-buffering the string in the scalar-to-array case).
        let warp_memory_limit_bytes = warp_memory_limit_bytes_from_config(config_options);
        let algorithm_array = if self.arg_count >= 3 {
            args[2]
                .clone()
                .cast_to(&DataType::Utf8View, None)?
                .into_array(num_iterations)?
        } else {
            ScalarValue::Utf8View(Some("NearestNeighbor".to_string()))
                .to_array_of_size(num_iterations)?
        };
        let algorithm_array = as_string_view_array(&algorithm_array)?.clone();
        let mut algorithm_iter = algorithm_array.iter();

        let mut builder = RasterBuilder::new(num_iterations);

        // The executor iterates the (raster, reference) pair; the algorithm
        // column is advanced in lockstep below.
        let exec_arg_types = vec![arg_types[0].clone(), arg_types[1].clone()];
        let exec_args = vec![args[0].clone(), args[1].clone()];
        let executor =
            RasterExecutor::new_with_num_iterations(&exec_arg_types, &exec_args, num_iterations);

        with_gdal(|gdal| {
            configure_thread_local_options(gdal, config_options)?;
            executor.execute_raster_raster_void(|_i, raster_opt, reference_opt| {
                let algorithm = algorithm_iter.next().flatten();
                // A NULL input, NULL reference, or NULL algorithm yields a NULL
                // output row (SQL semantics).
                let (Some(raster), Some(reference), Some(algorithm)) =
                    (raster_opt, reference_opt, algorithm)
                else {
                    builder.append_null()?;
                    return Ok(());
                };

                let alg = parse_resample_algorithm(algorithm, "RS_ReprojectMatch")?;
                reproject_match(
                    gdal,
                    raster,
                    reference,
                    alg,
                    warp_memory_limit_bytes,
                    &mut builder,
                )?;
                Ok(())
            })?;

            let out: ArrayRef = Arc::new(builder.finish()?);
            RasterExecutor::finish_over(args, out)
        })
    }
}

/// The GDAL warp memory limit in bytes from `sedona.gdal.warp_memory_limit_mb`,
/// or `0.0` (GDAL's own default) when unset. Megabytes match `gdalwarp -wm`.
fn warp_memory_limit_bytes_from_config(config_options: Option<&ConfigOptions>) -> f64 {
    config_options
        .and_then(|c| c.extensions.get::<SedonaOptions>())
        .and_then(|opts| opts.gdal.warp_memory_limit_mb)
        .map(|mb| mb as f64 * 1_000_000.0)
        .unwrap_or(0.0)
}

/// Reproject `raster` onto `reference`'s grid, appending the result to `builder`.
///
/// The output grid — transform, dimensions, and CRS — is the reference's; only
/// the input's pixels (warped) and band structure survive.
fn reproject_match(
    gdal: &sedona_gdal::gdal::Gdal,
    raster: &RasterRefImpl<'_>,
    reference: &RasterRefImpl<'_>,
    alg: ResampleAlg,
    warp_memory_limit_bytes: f64,
    builder: &mut RasterBuilder,
) -> Result<()> {
    let src_width = raster.width()?;
    let src_height = raster.height()?;
    if src_width <= 0 || src_height <= 0 {
        return exec_err!(
            "RS_ReprojectMatch: input raster has non-positive dimensions {src_width}x{src_height}"
        );
    }

    let ref_width = reference.width()?;
    let ref_height = reference.height()?;
    if ref_width <= 0 || ref_height <= 0 {
        return exec_err!(
            "RS_ReprojectMatch: reference raster has non-positive dimensions \
             {ref_width}x{ref_height}"
        );
    }

    // Reprojecting between an unknown CRS and a real one is undefined; require
    // that either both carry a CRS or neither does (a same-space regrid).
    match (raster.crs(), reference.crs()) {
        (Some(_), None) => {
            return exec_err!(
                "RS_ReprojectMatch: input raster has a CRS but the reference does not; \
                 cannot reproject onto a CRS-less grid"
            );
        }
        (None, Some(_)) => {
            return exec_err!(
                "RS_ReprojectMatch: reference raster has a CRS but the input does not; \
                 cannot reproject a CRS-less raster onto it"
            );
        }
        _ => {}
    }

    let ref_transform = reference.transform();
    if ref_transform.len() != 6 {
        return sedona_internal_err!(
            "RS_ReprojectMatch: reference transform has {} elements, expected 6",
            ref_transform.len()
        );
    }
    let mut transform: GeoTransform = [0.0; 6];
    transform.copy_from_slice(ref_transform);

    let band_count = raster.bands().len();

    // A warp always routes pixels through a floating working type, so no
    // resampling method preserves Int64/UInt64 exactly (nor Int32/UInt32 + Mode on
    // GDAL < 3.13). Reject those up front rather than mis-warping silently.
    // RS_Resample shares this check.
    reject_lossy_resample_dtypes(gdal, raster, band_count, alg, true, "RS_ReprojectMatch")?;

    let band_indices: Vec<usize> = (1..=band_count).collect();
    let layout = GdalBandLayout::from_raster(raster, &band_indices)?;
    // SAFETY: the returned dataset references `raster`'s band bytes zero-copy;
    // it is only read below (via warp) and dropped before `reproject_match`
    // returns, so `raster` outlives it.
    let src_dataset = unsafe { raster_ref_to_gdal_mem(gdal, raster, &band_indices)? };

    let output = OutputGrid {
        grid: Grid {
            transform,
            width: ref_width,
            height: ref_height,
        },
        crs: reference.crs(),
        alg,
    };
    append_warped_nd_from_dataset(
        gdal,
        &src_dataset,
        &layout,
        builder,
        &output,
        warp_memory_limit_bytes,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use sedona_testing::raster_spec::{
        assert_raster_scalar_equals, assert_rasters_equal, raster_array, RasterSpec,
    };
    use sedona_testing::testers::ScalarUdfTester;

    fn tester() -> ScalarUdfTester {
        ScalarUdfTester::new(
            rs_reproject_match_udf().into(),
            vec![RASTER, RASTER, SedonaType::Arrow(DataType::Utf8)],
        )
    }

    fn tester2() -> ScalarUdfTester {
        ScalarUdfTester::new(rs_reproject_match_udf().into(), vec![RASTER, RASTER])
    }

    /// Invoke `RS_ReprojectMatch(raster, reference, algorithm)` on two scalar
    /// rasters and return the resulting scalar.
    fn reproject(source: &RasterSpec, reference: &RasterSpec, algorithm: &str) -> ScalarValue {
        tester()
            .invoke_scalar_scalar_scalar(source, reference, algorithm)
            .unwrap()
    }

    #[test]
    fn match_same_grid_and_crs_is_identity() {
        // Reprojecting onto a reference that already has the input's grid and CRS
        // exercises the full warp path but round-trips the raster unchanged.
        let source = RasterSpec::d2(2, 2)
            .band_values(&[1u8, 2, 3, 4])
            .nodata(0u8)
            .bbox(0.0, 0.0, 2.0, 2.0)
            .crs(Some("EPSG:4326"));
        // Reference: same grid, different (irrelevant) pixel values.
        let reference = RasterSpec::d2(2, 2)
            .band_values(&[9u8, 9, 9, 9])
            .bbox(0.0, 0.0, 2.0, 2.0)
            .crs(Some("EPSG:4326"));

        let result = reproject(&source, &reference, "NearestNeighbor");
        assert_raster_scalar_equals(&result, &source);
    }

    #[test]
    fn match_reference_grid_upsamples_onto_finer_grid() {
        // Same CRS, but the reference is a finer 4x4 grid over the same extent.
        // A nearest 2x upsample replicates each source pixel into a 2x2 block —
        // unambiguous, so bit-exact — and the output takes the reference's grid.
        let source = RasterSpec::d2(2, 2)
            .band_values(&[10u8, 20, 30, 40])
            .bbox(0.0, 0.0, 4.0, 4.0)
            .crs(Some("EPSG:4326"));
        let reference = RasterSpec::d2(4, 4)
            .band_values(&(0u8..16).collect::<Vec<_>>())
            .bbox(0.0, 0.0, 4.0, 4.0)
            .crs(Some("EPSG:4326"));

        let result = reproject(&source, &reference, "NearestNeighbor");

        let expected = RasterSpec::d2(4, 4)
            .band_values(&[
                10u8, 10, 20, 20, //
                10, 10, 20, 20, //
                30, 30, 40, 40, //
                30, 30, 40, 40, //
            ])
            .bbox(0.0, 0.0, 4.0, 4.0)
            .crs(Some("EPSG:4326"));
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn match_reprojects_onto_reference_crs_and_grid() {
        // The input is EPSG:4326; the reference defines an EPSG:3857 grid. The
        // exact reprojected pixels are checked against rasterio in the Python
        // parity tests; here we pin the structural result — the output takes the
        // reference's CRS, transform, and dimensions exactly.
        let source = RasterSpec::d2(4, 4)
            .band_values(&(0u8..16).collect::<Vec<_>>())
            .nodata(255u8)
            .bbox(10.0, 40.0, 14.0, 44.0)
            .crs(Some("EPSG:4326"));
        // A north-up Web Mercator grid covering roughly the same footprint.
        let reference = RasterSpec::d2(4, 4)
            .band_values(&[0u8; 16])
            .transform([1_113_194.9, 111_319.5, 0.0, 5_465_442.2, 0.0, -111_319.5])
            .crs(Some("EPSG:3857"));

        let result = reproject(&source, &reference, "NearestNeighbor");

        let ScalarValue::Struct(struct_array) = &result else {
            panic!("expected a raster struct scalar, got {result:?}");
        };
        let rasters = sedona_raster::array::RasterStructArray::try_new(struct_array).unwrap();
        let out = rasters.get(0).unwrap();

        assert_eq!(out.crs(), Some("EPSG:3857"));
        assert_eq!(out.width().unwrap(), 4);
        assert_eq!(out.height().unwrap(), 4);
        // The output grid is the reference's grid, verbatim.
        let transform = out.transform();
        assert_eq!(transform[0], 1_113_194.9);
        assert_eq!(transform[3], 5_465_442.2);
        assert_eq!(transform[1], 111_319.5);
        assert_eq!(transform[5], -111_319.5);
    }

    #[test]
    fn uncovered_reference_cells_become_nodata() {
        // The reference grid extends past the input footprint on the right: the
        // input covers x[0,2] but the reference spans x[0,4] at the input's
        // resolution and CRS. The uncovered right column reads back as nodata.
        let source = RasterSpec::d2(2, 1)
            .band_values(&[10u8, 20])
            .nodata(255u8)
            .bbox(0.0, 0.0, 2.0, 1.0)
            .crs(Some("EPSG:4326"));
        let reference = RasterSpec::d2(4, 1)
            .band_values(&[0u8, 0, 0, 0])
            .bbox(0.0, 0.0, 4.0, 1.0)
            .crs(Some("EPSG:4326"));

        let result = reproject(&source, &reference, "NearestNeighbor");

        // Reference cell centers x = 0.5, 1.5, 2.5, 3.5; the last two are past
        // the input's right edge (x = 2), so they fill with the input's nodata.
        let expected = RasterSpec::d2(4, 1)
            .band_values(&[10u8, 20, 255, 255])
            .nodata(255u8)
            .bbox(0.0, 0.0, 4.0, 1.0)
            .crs(Some("EPSG:4326"));
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn multiband_reproject_preserves_band_order() {
        // Every input band is warped onto the reference grid and band
        // order/count/nodata preserved. Two-band 2x2 -> 4x4 upsample.
        let source = RasterSpec::d2(2, 2)
            .band_values(&[10u8, 20, 30, 40])
            .nodata(0u8)
            .band_values(&[1u8, 2, 3, 4])
            .bbox(0.0, 0.0, 4.0, 4.0)
            .crs(Some("EPSG:4326"));
        let reference = RasterSpec::d2(4, 4)
            .band_values(&[0u8; 16])
            .bbox(0.0, 0.0, 4.0, 4.0)
            .crs(Some("EPSG:4326"));

        let result = reproject(&source, &reference, "NearestNeighbor");

        let expected = RasterSpec::d2(4, 4)
            .band_values(&[
                10u8, 10, 20, 20, 10, 10, 20, 20, 30, 30, 40, 40, 30, 30, 40, 40,
            ])
            .nodata(0u8)
            .band_values(&[1u8, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4])
            .bbox(0.0, 0.0, 4.0, 4.0)
            .crs(Some("EPSG:4326"));
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn two_arg_overload_defaults_to_nearest() {
        // RS_ReprojectMatch(raster, reference) — algorithm defaults to nearest.
        let source = RasterSpec::d2(2, 2)
            .band_values(&[10u8, 20, 30, 40])
            .bbox(0.0, 0.0, 4.0, 4.0)
            .crs(Some("EPSG:4326"));
        let reference = RasterSpec::d2(4, 4)
            .band_values(&[0u8; 16])
            .bbox(0.0, 0.0, 4.0, 4.0)
            .crs(Some("EPSG:4326"));

        let result = tester2().invoke_scalar_scalar(&source, &reference).unwrap();

        let expected = RasterSpec::d2(4, 4)
            .band_values(&[
                10u8, 10, 20, 20, 10, 10, 20, 20, 30, 30, 40, 40, 30, 30, 40, 40,
            ])
            .bbox(0.0, 0.0, 4.0, 4.0)
            .crs(Some("EPSG:4326"));
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn null_input_or_reference_yields_null() {
        let source = RasterSpec::d2(2, 2)
            .band_values(&[10u8, 20, 30, 40])
            .bbox(0.0, 0.0, 4.0, 4.0)
            .crs(Some("EPSG:4326"));
        let reference = RasterSpec::d2(4, 4)
            .band_values(&[0u8; 16])
            .bbox(0.0, 0.0, 4.0, 4.0)
            .crs(Some("EPSG:4326"));

        // A per-row input column (row 1 NULL) over a scalar reference +
        // algorithm: the NULL input row must produce a NULL output row.
        let result = tester()
            .invoke_array_scalar_scalar(
                Arc::new(raster_array(vec![Some(source.clone()), None])),
                &reference,
                "NearestNeighbor",
            )
            .unwrap();

        let expected = vec![
            Some(
                RasterSpec::d2(4, 4)
                    .band_values(&[
                        10u8, 10, 20, 20, 10, 10, 20, 20, 30, 30, 40, 40, 30, 30, 40, 40,
                    ])
                    .bbox(0.0, 0.0, 4.0, 4.0)
                    .crs(Some("EPSG:4326")),
            ),
            None,
        ];
        assert_rasters_equal(&result, &expected);
    }

    #[test]
    fn one_sided_crs_errors() {
        // Input has a CRS, reference does not: reprojecting onto a CRS-less grid
        // is undefined and must error rather than silently mis-warp.
        let source = RasterSpec::d2(2, 2)
            .band_values(&[10u8, 20, 30, 40])
            .bbox(0.0, 0.0, 4.0, 4.0)
            .crs(Some("EPSG:4326"));
        let reference = RasterSpec::d2(4, 4)
            .band_values(&[0u8; 16])
            .bbox(0.0, 0.0, 4.0, 4.0)
            .crs(None);

        let err = tester()
            .invoke_scalar_scalar_scalar(&source, &reference, "NearestNeighbor")
            .unwrap_err()
            .to_string();
        assert!(err.contains("reference does not"), "got: {err}");
    }

    #[test]
    fn unknown_algorithm_errors() {
        let source = RasterSpec::d2(2, 2)
            .band_values(&[10u8, 20, 30, 40])
            .bbox(0.0, 0.0, 4.0, 4.0)
            .crs(Some("EPSG:4326"));
        let reference = RasterSpec::d2(4, 4)
            .band_values(&[0u8; 16])
            .bbox(0.0, 0.0, 4.0, 4.0)
            .crs(Some("EPSG:4326"));

        let err = tester()
            .invoke_scalar_scalar_scalar(&source, &reference, "sinc")
            .unwrap_err()
            .to_string();
        assert!(err.contains("unknown algorithm"), "got: {err}");
    }

    #[test]
    fn int64_uint64_rejected() {
        // GDAL's warp routes 64-bit integers through a floating working type, so
        // no resampling method preserves them exactly. Both dtypes are rejected
        // up front regardless of algorithm — nearest and bilinear both error.
        let reference = RasterSpec::d2(2, 2)
            .band_values(&[0u8; 4])
            .bbox(0.0, 0.0, 2.0, 2.0)
            .crs(Some("EPSG:4326"));
        let int64_source = RasterSpec::d2(2, 2)
            .band_values(&[1i64, 2, 3, 4])
            .bbox(0.0, 0.0, 2.0, 2.0)
            .crs(Some("EPSG:4326"));
        let uint64_source = RasterSpec::d2(2, 2)
            .band_values(&[1u64, 2, 3, 4])
            .bbox(0.0, 0.0, 2.0, 2.0)
            .crs(Some("EPSG:4326"));

        for source in [&int64_source, &uint64_source] {
            for alg in ["NearestNeighbor", "Bilinear"] {
                let err = tester()
                    .invoke_scalar_scalar_scalar(source, &reference, alg)
                    .unwrap_err()
                    .to_string();
                assert!(
                    err.contains("does not support Int64/UInt64 rasters"),
                    "got: {err}"
                );
            }
        }
    }

    #[test]
    fn int32_mode_gated_on_gdal_version() {
        // Mode uses a 32-bit float working type on GDAL < 3.13 (a double from 3.13
        // on), which cannot represent Int32/UInt32 above 2^24. Mode is a selection,
        // so silently rounding a category code is wrong: reject Int32 + Mode on
        // older GDAL, accept it (double working type) on GDAL >= 3.13. Other
        // algorithms are unaffected — a double is exact for 32-bit ints.
        let reference = RasterSpec::d2(2, 2)
            .band_values(&[0u8; 4])
            .bbox(0.0, 0.0, 2.0, 2.0)
            .crs(Some("EPSG:4326"));
        let big: i32 = (1i32 << 24) + 1;
        let int32_source = RasterSpec::d2(2, 2)
            .band_values(&[big, big, big, big])
            .bbox(0.0, 0.0, 2.0, 2.0)
            .crs(Some("EPSG:4326"));

        // Bilinear routes through a double (exact for 32-bit ints), so Int32 is
        // accepted on every GDAL version — the gate is Mode-specific.
        tester()
            .invoke_scalar_scalar_scalar(&int32_source, &reference, "Bilinear")
            .unwrap();

        let gdal_version = with_gdal(|gdal| Ok(gdal.version_num())).unwrap();
        let mode = tester().invoke_scalar_scalar_scalar(&int32_source, &reference, "Mode");
        if gdal_version >= 3_130_000 {
            mode.unwrap();
        } else {
            let err = mode.unwrap_err().to_string();
            assert!(
                err.contains("Mode resampling of Int32/UInt32"),
                "got: {err}"
            );
        }
    }
}
