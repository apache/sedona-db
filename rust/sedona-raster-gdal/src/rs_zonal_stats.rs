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

//! RS_ZonalStats and RS_ZonalStatsAll UDFs - Compute statistics for pixels within a geometry
//!
//! RS_ZonalStats computes a single statistic (count, sum, mean, median, mode, stddev, variance, min, max)
//! for all pixels within a geometry boundary.
//!
//! RS_ZonalStatsAll computes all statistics and returns them as a struct.
//!
//! Signatures (matching Apache Sedona docs):
//!
//! ## RS_ZonalStats
//! - `RS_ZonalStats(raster, zone, statType)` — 3 args
//! - `RS_ZonalStats(raster, zone, statType, allTouched)` — 4 args
//! - `RS_ZonalStats(raster, zone, band, statType, allTouched)` — 5 args
//! - `RS_ZonalStats(raster, zone, band, statType, allTouched, excludeNoData)` — 6 args
//! - `RS_ZonalStats(raster, zone, band, statType, allTouched, excludeNoData, lenient)` — 7 args
//!
//! ## RS_ZonalStatsAll
//! - `RS_ZonalStatsAll(raster, zone)` — 2 args
//! - `RS_ZonalStatsAll(raster, zone, band)` — 3 args
//! - `RS_ZonalStatsAll(raster, zone, band, allTouched)` — 4 args
//! - `RS_ZonalStatsAll(raster, zone, band, allTouched, excludeNoData)` — 5 args
//! - `RS_ZonalStatsAll(raster, zone, band, allTouched, excludeNoData, lenient)` — 6 args

use std::collections::HashMap;
use std::convert::TryInto;
use std::sync::Arc;

use arrow_array::builder::{Float64Builder, Int64Builder, StructBuilder};
use arrow_array::{Array, ArrayRef};
use arrow_array::{BooleanArray, Int32Array};
use arrow_schema::{DataType, Field, Fields};
use datafusion_common::cast::{as_boolean_array, as_int32_array, as_string_array};
use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_common::{exec_datafusion_err, exec_err, ScalarValue};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_common::sedona_internal_err;
use sedona_gdal::gdal::Gdal;
use sedona_gdal::mem::MemDatasetBuilder;
use sedona_gdal::raster::Buffer;
use sedona_gdal::raster::GdalDataType;
use sedona_proj::transform::with_global_proj_engine;

use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::affine_transformation::to_world_coordinate;
use sedona_raster::array::RasterRefImpl;
use sedona_raster::traits::RasterRef;
use sedona_raster_functions::crs_utils::{crs_transform_wkb, resolve_crs};
use sedona_raster_functions::RasterExecutor;
use sedona_schema::datatypes::SedonaType;
use sedona_schema::matchers::ArgMatcher;

use crate::gdal_common::{nodata_bytes_to_f64, with_gdal};
use crate::gdal_dataset_provider::configure_thread_local_options;
use crate::raster_band_reader::RasterBandReader;

/// Statistics types supported by RS_ZonalStats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatType {
    Count,
    Sum,
    Mean,
    Median,
    Mode,
    StdDev,
    Variance,
    Min,
    Max,
}

impl StatType {
    /// Parse stat type from string (case-insensitive)
    fn from_str(s: &str) -> Option<StatType> {
        match s.to_lowercase().as_str() {
            "count" => Some(StatType::Count),
            "sum" => Some(StatType::Sum),
            "mean" | "avg" | "average" => Some(StatType::Mean),
            "median" => Some(StatType::Median),
            "mode" => Some(StatType::Mode),
            "stddev" | "std" | "standarddeviation" => Some(StatType::StdDev),
            "variance" | "var" => Some(StatType::Variance),
            "min" | "minimum" => Some(StatType::Min),
            "max" | "maximum" => Some(StatType::Max),
            _ => None,
        }
    }
}

/// Computed statistics for a zone
#[derive(Debug, Default)]
pub struct ZonalStatistics {
    pub count: i64,
    pub sum: f64,
    pub mean: f64,
    pub median: f64,
    pub mode: f64,
    pub stddev: f64,
    pub variance: f64,
    pub min: f64,
    pub max: f64,
}

impl ZonalStatistics {
    /// Get a specific statistic value
    fn get(&self, stat_type: StatType) -> f64 {
        match stat_type {
            StatType::Count => self.count as f64,
            StatType::Sum => self.sum,
            StatType::Mean => self.mean,
            StatType::Median => self.median,
            StatType::Mode => self.mode,
            StatType::StdDev => self.stddev,
            StatType::Variance => self.variance,
            StatType::Min => self.min,
            StatType::Max => self.max,
        }
    }
}

// =============================================================================
// RS_ZonalStats UDF
// =============================================================================

/// RS_ZonalStats() scalar UDF implementation
///
/// Computes a single statistic for pixels within a geometry.
///
/// Signatures:
/// - `RS_ZonalStats(raster, zone, statType)` — 3 args
/// - `RS_ZonalStats(raster, zone, statType, allTouched)` — 4 args
/// - `RS_ZonalStats(raster, zone, band, statType, allTouched)` — 5 args
/// - `RS_ZonalStats(raster, zone, band, statType, allTouched, excludeNoData)` — 6 args
/// - `RS_ZonalStats(raster, zone, band, statType, allTouched, excludeNoData, lenient)` — 7 args
pub fn rs_zonal_stats_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_zonalstats",
        vec![
            Arc::new(RsZonalStats { arg_count: 3 }),
            Arc::new(RsZonalStats { arg_count: 4 }),
            Arc::new(RsZonalStats { arg_count: 5 }),
            Arc::new(RsZonalStats { arg_count: 6 }),
            Arc::new(RsZonalStats { arg_count: 7 }),
        ],
        Volatility::Immutable,
    )
}

/// Kernel implementation for RS_ZonalStats
#[derive(Debug)]
struct RsZonalStats {
    /// Number of arguments in the matched signature (3..=7)
    arg_count: usize,
}

impl SedonaScalarKernel for RsZonalStats {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matchers = match self.arg_count {
            3 => vec![
                // RS_ZonalStats(raster, zone, statType)
                ArgMatcher::is_raster(),
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_string(),
            ],
            4 => vec![
                // RS_ZonalStats(raster, zone, statType, allTouched)
                ArgMatcher::is_raster(),
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_string(),
                ArgMatcher::is_boolean(),
            ],
            5 => vec![
                // RS_ZonalStats(raster, zone, band, statType, allTouched)
                ArgMatcher::is_raster(),
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_integer(),
                ArgMatcher::is_string(),
                ArgMatcher::is_boolean(),
            ],
            6 => vec![
                // RS_ZonalStats(raster, zone, band, statType, allTouched, excludeNoData)
                ArgMatcher::is_raster(),
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_integer(),
                ArgMatcher::is_string(),
                ArgMatcher::is_boolean(),
                ArgMatcher::is_boolean(),
            ],
            7 => vec![
                // RS_ZonalStats(raster, zone, band, statType, allTouched, excludeNoData, lenient)
                ArgMatcher::is_raster(),
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_integer(),
                ArgMatcher::is_string(),
                ArgMatcher::is_boolean(),
                ArgMatcher::is_boolean(),
                ArgMatcher::is_boolean(),
            ],
            _ => {
                return sedona_internal_err!(
                    "RS_ZonalStats: unexpected arg_count {}",
                    self.arg_count
                );
            }
        };

        let matcher = ArgMatcher::new(matchers, SedonaType::Arrow(DataType::Float64));
        matcher.match_args(args)
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
        let num_iterations = calc_num_iterations(args);

        // Geometry is always at index 1 (zone).
        let geom_arg_idx: usize = 1;

        // Determine arg indices based on arg_count.
        //   3-arg: (raster, zone, statType)                         → stat=2
        //   4-arg: (raster, zone, statType, allTouched)             → stat=2, allTouched=3
        //   5-arg: (raster, zone, band, statType, allTouched)       → band=2, stat=3, allTouched=4
        //   6-arg: (raster, zone, band, statType, allTouched, excludeNoData) → band=2, stat=3, allTouched=4, excludeNoData=5
        //   7-arg: + lenient=6
        let (
            stat_arg_idx,
            band_arg_idx,
            all_touched_arg_idx,
            exclude_nodata_arg_idx,
            lenient_arg_idx,
        ) = match self.arg_count {
            3 => (2, None, None, None, None),
            4 => (2, None, Some(3), None, None),
            5 => (3, Some(2), Some(4), None, None),
            6 => (3, Some(2), Some(4), Some(5), None),
            7 => (3, Some(2), Some(4), Some(5), Some(6)),
            _ => unreachable!(),
        };

        // Get stat type array
        let stat_array = args[stat_arg_idx]
            .clone()
            .cast_to(&DataType::Utf8, None)?
            .into_array(num_iterations)?;
        let stat_array = as_string_array(&stat_array)?.clone();
        let mut stat_iter = stat_array.iter();

        let mut builder = Float64Builder::with_capacity(num_iterations);

        // Expand option args to arrays so they can vary row-by-row.
        let band_array = expand_int32_arg(args, band_arg_idx, 1, num_iterations)?;
        let all_touched_array =
            expand_boolean_arg(args, all_touched_arg_idx, false, num_iterations)?;
        let exclude_nodata_array =
            expand_boolean_arg(args, exclude_nodata_arg_idx, true, num_iterations)?;
        let lenient_array = expand_boolean_arg(args, lenient_arg_idx, true, num_iterations)?;

        let mut band_iter = band_array.iter();
        let mut all_touched_iter = all_touched_array.iter();
        let mut exclude_nodata_iter = exclude_nodata_array.iter();
        let mut lenient_iter = lenient_array.iter();

        let exec_arg_types = vec![arg_types[0].clone(), arg_types[geom_arg_idx].clone()];
        let exec_args = vec![args[0].clone(), args[geom_arg_idx].clone()];
        let executor =
            RasterExecutor::new_with_num_iterations(&exec_arg_types, &exec_args, num_iterations);

        with_gdal(|gdal| {
            configure_thread_local_options(gdal, config_options)?;
            with_global_proj_engine(|engine| {
                executor.execute_raster_wkb_crs_void(|raster_opt, wkb_opt, geom_crs| {
                let stat_str = stat_iter
                    .next()
                    .flatten()
                    .ok_or_else(|| exec_datafusion_err!("Stat type is required"))?;
                let stat_type = StatType::from_str(stat_str)
                    .ok_or_else(|| exec_datafusion_err!("Unknown stat type: {}", stat_str))?;
                let band = band_iter
                    .next()
                    .flatten()
                    .unwrap_or(1)
                    .max(1)
                    .try_into()
                    .unwrap_or(1);
                let all_touched = all_touched_iter.next().flatten().unwrap_or(false);
                let exclude_nodata = exclude_nodata_iter.next().flatten().unwrap_or(true);
                let lenient = lenient_iter.next().flatten().unwrap_or(true);

                let (raster, geom_wkb) = match (raster_opt, wkb_opt) {
                    (Some(r), Some(w)) => (r, w),
                    _ => {
                        builder.append_null();
                        return Ok(());
                    }
                };

                let raster_crs = resolve_crs(raster.crs())?;

                let geom_wkb = match (geom_crs, raster_crs.as_deref()) {
                    (Some(geom_crs), Some(raster_crs)) => {
                        crs_transform_wkb(geom_wkb, geom_crs, raster_crs, engine)?
                    }
                    (None, None) => geom_wkb.to_vec(),
                    (Some(_), None) => {
                        return exec_err!(
                            "Cannot operate on geometry and raster: raster has no CRS but geometry does"
                        )
                    }
                    (None, Some(_)) => {
                        return exec_err!(
                            "Cannot operate on geometry and raster: geometry has no CRS but raster does"
                        )
                    }
                };

                match compute_zonal_stats(
                    gdal,
                    raster,
                    &geom_wkb,
                    band,
                    all_touched,
                    exclude_nodata,
                ) {
                    Ok(stats) => builder.append_value(stats.get(stat_type)),
                    Err(e) => {
                        if lenient {
                            eprintln!("RS_ZonalStats error: {}", e);
                            builder.append_null();
                        } else {
                            return Err(e);
                        }
                    }
                }

                Ok(())
            })
            })?;

            executor.finish(Arc::new(builder.finish()))
        })
    }
}

// =============================================================================
// RS_ZonalStatsAll UDF
// =============================================================================

/// RS_ZonalStatsAll() scalar UDF implementation
///
/// Computes all statistics for pixels within a geometry and returns a struct.
///
/// Signatures:
/// - `RS_ZonalStatsAll(raster, zone)` — 2 args
/// - `RS_ZonalStatsAll(raster, zone, band)` — 3 args
/// - `RS_ZonalStatsAll(raster, zone, band, allTouched)` — 4 args
/// - `RS_ZonalStatsAll(raster, zone, band, allTouched, excludeNoData)` — 5 args
/// - `RS_ZonalStatsAll(raster, zone, band, allTouched, excludeNoData, lenient)` — 6 args
pub fn rs_zonal_stats_all_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_zonalstatsall",
        vec![
            Arc::new(RsZonalStatsAll { arg_count: 2 }),
            Arc::new(RsZonalStatsAll { arg_count: 3 }),
            Arc::new(RsZonalStatsAll { arg_count: 4 }),
            Arc::new(RsZonalStatsAll { arg_count: 5 }),
            Arc::new(RsZonalStatsAll { arg_count: 6 }),
        ],
        Volatility::Immutable,
    )
}

/// Kernel implementation for RS_ZonalStatsAll
#[derive(Debug)]
struct RsZonalStatsAll {
    /// Number of arguments in the matched signature (2..=6)
    arg_count: usize,
}

impl SedonaScalarKernel for RsZonalStatsAll {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matchers = match self.arg_count {
            2 => vec![
                // RS_ZonalStatsAll(raster, zone)
                ArgMatcher::is_raster(),
                ArgMatcher::is_geometry_or_geography(),
            ],
            3 => vec![
                // RS_ZonalStatsAll(raster, zone, band)
                ArgMatcher::is_raster(),
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_integer(),
            ],
            4 => vec![
                // RS_ZonalStatsAll(raster, zone, band, allTouched)
                ArgMatcher::is_raster(),
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_integer(),
                ArgMatcher::is_boolean(),
            ],
            5 => vec![
                // RS_ZonalStatsAll(raster, zone, band, allTouched, excludeNoData)
                ArgMatcher::is_raster(),
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_integer(),
                ArgMatcher::is_boolean(),
                ArgMatcher::is_boolean(),
            ],
            6 => vec![
                // RS_ZonalStatsAll(raster, zone, band, allTouched, excludeNoData, lenient)
                ArgMatcher::is_raster(),
                ArgMatcher::is_geometry_or_geography(),
                ArgMatcher::is_integer(),
                ArgMatcher::is_boolean(),
                ArgMatcher::is_boolean(),
                ArgMatcher::is_boolean(),
            ],
            _ => {
                return sedona_internal_err!(
                    "RS_ZonalStatsAll: unexpected arg_count {}",
                    self.arg_count
                );
            }
        };

        let matcher = ArgMatcher::new(matchers, SedonaType::Arrow(zonal_stats_struct_type()));
        matcher.match_args(args)
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
        let num_iterations = calc_num_iterations(args);

        // Geometry is always at index 1 (zone).
        let geom_arg_idx: usize = 1;

        // Determine arg indices based on arg_count.
        //   2-arg: (raster, zone)
        //   3-arg: (raster, zone, band)                                    → band=2
        //   4-arg: (raster, zone, band, allTouched)                        → band=2, allTouched=3
        //   5-arg: (raster, zone, band, allTouched, excludeNoData)         → band=2, allTouched=3, excludeNoData=4
        //   6-arg: + lenient=5
        let (band_arg_idx, all_touched_arg_idx, exclude_nodata_arg_idx, lenient_arg_idx) =
            match self.arg_count {
                2 => (None, None, None, None),
                3 => (Some(2), None, None, None),
                4 => (Some(2), Some(3), None, None),
                5 => (Some(2), Some(3), Some(4), None),
                6 => (Some(2), Some(3), Some(4), Some(5)),
                _ => unreachable!(),
            };

        // Build struct result
        let fields = zonal_stats_struct_fields();
        let mut builder = StructBuilder::from_fields(fields, num_iterations);

        // Expand option args to arrays so they can vary row-by-row.
        let band_array = expand_int32_arg(args, band_arg_idx, 1, num_iterations)?;
        let all_touched_array =
            expand_boolean_arg(args, all_touched_arg_idx, false, num_iterations)?;
        let exclude_nodata_array =
            expand_boolean_arg(args, exclude_nodata_arg_idx, true, num_iterations)?;
        let lenient_array = expand_boolean_arg(args, lenient_arg_idx, true, num_iterations)?;

        let mut band_iter = band_array.iter();
        let mut all_touched_iter = all_touched_array.iter();
        let mut exclude_nodata_iter = exclude_nodata_array.iter();
        let mut lenient_iter = lenient_array.iter();

        let append_null = |builder: &mut StructBuilder| {
            builder
                .field_builder::<Int64Builder>(0)
                .unwrap()
                .append_null();
            for j in 1..9 {
                builder
                    .field_builder::<Float64Builder>(j)
                    .unwrap()
                    .append_null();
            }
            builder.append_null();
        };

        let append_stats = |builder: &mut StructBuilder, stats: ZonalStatistics| {
            builder
                .field_builder::<Int64Builder>(0)
                .unwrap()
                .append_value(stats.count);
            builder
                .field_builder::<Float64Builder>(1)
                .unwrap()
                .append_value(stats.sum);
            builder
                .field_builder::<Float64Builder>(2)
                .unwrap()
                .append_value(stats.mean);
            builder
                .field_builder::<Float64Builder>(3)
                .unwrap()
                .append_value(stats.median);
            builder
                .field_builder::<Float64Builder>(4)
                .unwrap()
                .append_value(stats.mode);
            builder
                .field_builder::<Float64Builder>(5)
                .unwrap()
                .append_value(stats.stddev);
            builder
                .field_builder::<Float64Builder>(6)
                .unwrap()
                .append_value(stats.variance);
            builder
                .field_builder::<Float64Builder>(7)
                .unwrap()
                .append_value(stats.min);
            builder
                .field_builder::<Float64Builder>(8)
                .unwrap()
                .append_value(stats.max);
            builder.append(true);
        };

        let exec_arg_types = vec![arg_types[0].clone(), arg_types[geom_arg_idx].clone()];
        let exec_args = vec![args[0].clone(), args[geom_arg_idx].clone()];
        let executor =
            RasterExecutor::new_with_num_iterations(&exec_arg_types, &exec_args, num_iterations);

        with_gdal(|gdal| {
            configure_thread_local_options(gdal, config_options)?;
            with_global_proj_engine(|engine| {
                executor.execute_raster_wkb_crs_void(|raster_opt, wkb_opt, geom_crs| {
                let band = band_iter
                    .next()
                    .flatten()
                    .unwrap_or(1)
                    .max(1)
                    .try_into()
                    .unwrap_or(1);
                let all_touched = all_touched_iter.next().flatten().unwrap_or(false);
                let exclude_nodata = exclude_nodata_iter.next().flatten().unwrap_or(true);
                let lenient = lenient_iter.next().flatten().unwrap_or(true);

                let (raster, geom_wkb) = match (raster_opt, wkb_opt) {
                    (Some(r), Some(w)) => (r, w),
                    _ => {
                        append_null(&mut builder);
                        return Ok(());
                    }
                };

                let raster_crs = resolve_crs(raster.crs())?;

                let geom_wkb = match (geom_crs, raster_crs.as_deref()) {
                    (Some(geom_crs), Some(raster_crs)) => {
                        crs_transform_wkb(geom_wkb, geom_crs, raster_crs, engine)?
                    }
                    (None, None) => geom_wkb.to_vec(),
                    (Some(_), None) => {
                        return exec_err!(
                            "Cannot operate on geometry and raster: raster has no CRS but geometry does"
                        )
                    }
                    (None, Some(_)) => {
                        return exec_err!(
                            "Cannot operate on geometry and raster: geometry has no CRS but raster does"
                        )
                    }
                };

                match compute_zonal_stats(
                    gdal,
                    raster,
                    &geom_wkb,
                    band,
                    all_touched,
                    exclude_nodata,
                ) {
                    Ok(stats) => append_stats(&mut builder, stats),
                    Err(e) => {
                        if lenient {
                            eprintln!("RS_ZonalStatsAll error: {}", e);
                            append_null(&mut builder);
                        } else {
                            return Err(e);
                        }
                    }
                }

                Ok(())
            })
            })?;

            executor.finish(Arc::new(builder.finish()) as ArrayRef)
        })
    }
}

/// Return type for ZonalStatsAll struct
fn zonal_stats_struct_type() -> DataType {
    DataType::Struct(zonal_stats_struct_fields())
}

/// Fields for the ZonalStatsAll struct
fn zonal_stats_struct_fields() -> Fields {
    Fields::from(vec![
        Field::new("count", DataType::Int64, true),
        Field::new("sum", DataType::Float64, true),
        Field::new("mean", DataType::Float64, true),
        Field::new("median", DataType::Float64, true),
        Field::new("mode", DataType::Float64, true),
        Field::new("stddev", DataType::Float64, true),
        Field::new("variance", DataType::Float64, true),
        Field::new("min", DataType::Float64, true),
        Field::new("max", DataType::Float64, true),
    ])
}

// =============================================================================
// Core Statistics Computation
// =============================================================================

/// Compute zonal statistics for a raster within a geometry
fn compute_zonal_stats(
    gdal: &Gdal,
    raster: &RasterRefImpl<'_>,
    geom_wkb: &[u8],
    band_num: usize,
    all_touched: bool,
    exclude_nodata: bool,
) -> Result<ZonalStatistics> {
    let metadata = raster.metadata();

    let mut band_reader = RasterBandReader::new(gdal, raster);

    // Parse geometry from WKB
    let geometry = gdal
        .geometry_from_wkb(geom_wkb)
        .map_err(|e| exec_datafusion_err!("Failed to parse geometry from WKB: {}", e))?;

    let geom_bounds = bounds_from_envelope(geometry.envelope());
    let raster_bounds = raster_bounds(raster);
    let intersection = match geom_bounds.intersection(raster_bounds) {
        Some(bounds) => bounds,
        None => return compute_statistics(&[]),
    };

    let window = match bounds_to_window(raster, intersection)? {
        Some(window) => window,
        None => return compute_statistics(&[]),
    };

    // Create a mask raster
    let mask_dataset =
        MemDatasetBuilder::create(gdal, window.width, window.height, 1, GdalDataType::UInt8)
            .map_err(|e| exec_datafusion_err!("Failed to create mask dataset: {}", e))?;

    // Set geotransform
    let start_col = window.xoff as f64;
    let start_row = window.yoff as f64;
    let geotransform = [
        metadata.upper_left_x() + start_col * metadata.scale_x() + start_row * metadata.skew_x(),
        metadata.scale_x(),
        metadata.skew_x(),
        metadata.upper_left_y() + start_col * metadata.skew_y() + start_row * metadata.scale_y(),
        metadata.skew_y(),
        metadata.scale_y(),
    ];
    mask_dataset
        .set_geo_transform(&geotransform)
        .map_err(|e| exec_datafusion_err!("Failed to set geotransform: {}", e))?;

    // Initialize mask to 0
    let mask_band = mask_dataset
        .rasterband(1)
        .map_err(|e| exec_datafusion_err!("Failed to get mask band: {}", e))?;
    let zeros = vec![0u8; window.width * window.height];
    let mut buffer = Buffer::new((window.width, window.height), zeros);
    mask_band
        .write((0, 0), (window.width, window.height), &mut buffer)
        .map_err(|e| exec_datafusion_err!("Failed to initialize mask: {}", e))?;

    gdal.rasterize_affine(&mask_dataset, &[1], &[geometry], &[1.0], all_touched)
        .map_err(|e| exec_datafusion_err!("Failed to rasterize geometry: {}", e))?;

    // Read mask
    let mask_band = mask_dataset
        .rasterband(1)
        .map_err(|e| exec_datafusion_err!("Failed to get mask band: {}", e))?;
    let mask_buffer = mask_band
        .read_as::<u8>(
            (0, 0),
            (window.width, window.height),
            (window.width, window.height),
            None,
        )
        .map_err(|e| exec_datafusion_err!("Failed to read mask: {}", e))?;
    let mask = mask_buffer.data();

    let band = raster
        .bands()
        .band(band_num)
        .map_err(|e| exec_datafusion_err!("Failed to get band: {}", e))?;
    let band_metadata = band.metadata();
    let data_type = band_metadata.data_type()?;
    let nodata = nodata_bytes_to_f64(band_metadata.nodata_value(), &data_type);

    // Collect pixel values within the geometry
    let mut values: Vec<f64> = Vec::new();

    let band_values = band_reader.read_window_f64(
        band_num,
        (window.xoff, window.yoff),
        (window.width, window.height),
    )?;

    for (pixel_idx, &mask_val) in mask.iter().enumerate().take(window.width * window.height) {
        if mask_val == 1 {
            let value = band_values[pixel_idx];

            // Check for nodata
            if exclude_nodata {
                if let Some(no_data) = nodata {
                    if (value - no_data).abs() < f64::EPSILON || value.is_nan() {
                        continue;
                    }
                }
            }

            values.push(value);
        }
    }

    // Compute statistics
    compute_statistics(&values)
}

#[derive(Clone, Copy, Debug)]
struct Bounds {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
}

impl Bounds {
    fn intersection(self, other: Bounds) -> Option<Bounds> {
        let min_x = self.min_x.max(other.min_x);
        let max_x = self.max_x.min(other.max_x);
        let min_y = self.min_y.max(other.min_y);
        let max_y = self.max_y.min(other.max_y);

        if min_x > max_x || min_y > max_y {
            None
        } else {
            Some(Bounds {
                min_x,
                max_x,
                min_y,
                max_y,
            })
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct RasterWindow {
    xoff: usize,
    yoff: usize,
    width: usize,
    height: usize,
}

fn bounds_from_envelope(env: sedona_gdal::vector::Envelope) -> Bounds {
    Bounds {
        min_x: env.MinX,
        max_x: env.MaxX,
        min_y: env.MinY,
        max_y: env.MaxY,
    }
}

fn raster_bounds(raster: &RasterRefImpl<'_>) -> Bounds {
    let metadata = raster.metadata();
    let width = metadata.width() as i64;
    let height = metadata.height() as i64;
    let corners = [
        to_world_coordinate(raster, 0, 0),
        to_world_coordinate(raster, width, 0),
        to_world_coordinate(raster, 0, height),
        to_world_coordinate(raster, width, height),
    ];

    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for (x, y) in corners {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }

    Bounds {
        min_x,
        max_x,
        min_y,
        max_y,
    }
}

fn world_to_pixel_f64(
    raster: &RasterRefImpl<'_>,
    world_x: f64,
    world_y: f64,
) -> Result<(f64, f64)> {
    let metadata = raster.metadata();
    let det = metadata.scale_x() * metadata.scale_y() - metadata.skew_x() * metadata.skew_y();

    if det.abs() < f64::EPSILON {
        return exec_err!("Cannot compute coordinate: determinant is zero.");
    }

    let inv_scale_x = metadata.scale_y() / det;
    let inv_scale_y = metadata.scale_x() / det;
    let inv_skew_x = -metadata.skew_x() / det;
    let inv_skew_y = -metadata.skew_y() / det;

    let dx = world_x - metadata.upper_left_x();
    let dy = world_y - metadata.upper_left_y();

    let col = inv_scale_x * dx + inv_skew_x * dy;
    let row = inv_skew_y * dx + inv_scale_y * dy;

    Ok((col, row))
}

fn bounds_to_window(raster: &RasterRefImpl<'_>, bounds: Bounds) -> Result<Option<RasterWindow>> {
    let metadata = raster.metadata();
    let raster_w = metadata.width() as isize;
    let raster_h = metadata.height() as isize;

    let corners = [
        (bounds.min_x, bounds.min_y),
        (bounds.min_x, bounds.max_y),
        (bounds.max_x, bounds.min_y),
        (bounds.max_x, bounds.max_y),
    ];

    let mut min_col = f64::INFINITY;
    let mut max_col = f64::NEG_INFINITY;
    let mut min_row = f64::INFINITY;
    let mut max_row = f64::NEG_INFINITY;

    for (x, y) in corners {
        let (col, row) = world_to_pixel_f64(raster, x, y)?;
        min_col = min_col.min(col);
        max_col = max_col.max(col);
        min_row = min_row.min(row);
        max_row = max_row.max(row);
    }

    let mut start_col = min_col.floor() as isize - 1;
    let mut end_col = max_col.ceil() as isize + 1;
    let mut start_row = min_row.floor() as isize - 1;
    let mut end_row = max_row.ceil() as isize + 1;

    start_col = start_col.max(0).min(raster_w);
    end_col = end_col.max(0).min(raster_w);
    start_row = start_row.max(0).min(raster_h);
    end_row = end_row.max(0).min(raster_h);

    if end_col <= start_col || end_row <= start_row {
        return Ok(None);
    }

    Ok(Some(RasterWindow {
        xoff: start_col as usize,
        yoff: start_row as usize,
        width: (end_col - start_col) as usize,
        height: (end_row - start_row) as usize,
    }))
}

/// Compute all statistics from a vector of values
fn compute_statistics(values: &[f64]) -> Result<ZonalStatistics> {
    if values.is_empty() {
        return Ok(ZonalStatistics {
            count: 0,
            sum: 0.0,
            mean: f64::NAN,
            median: f64::NAN,
            mode: f64::NAN,
            stddev: f64::NAN,
            variance: f64::NAN,
            min: f64::NAN,
            max: f64::NAN,
        });
    }

    let count = values.len() as i64;
    let sum: f64 = values.iter().sum();
    let mean = sum / count as f64;

    // Min and max
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Variance and standard deviation
    let variance = if count > 1 {
        let sum_sq_diff: f64 = values.iter().map(|&v| (v - mean).powi(2)).sum();
        sum_sq_diff / (count as f64 - 1.0) // Sample variance
    } else {
        0.0
    };
    let stddev = variance.sqrt();

    // Median
    let median = {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = sorted.len() / 2;
        if sorted.len().is_multiple_of(2) {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    };

    // Mode (most frequent value)
    let mode = {
        let mut counts: HashMap<i64, usize> = HashMap::new();
        for &v in values {
            // Quantize to avoid floating point comparison issues
            let key = (v * 1_000_000.0).round() as i64;
            *counts.entry(key).or_insert(0) += 1;
        }
        let (mode_key, _) = counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .unwrap_or((0, 0));
        mode_key as f64 / 1_000_000.0
    };

    Ok(ZonalStatistics {
        count,
        sum,
        mean,
        median,
        mode,
        stddev,
        variance,
        min,
        max,
    })
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Expand an optional Int32 argument to an array. If `arg_idx` is `None`, returns
/// a constant array filled with `default_val`.
fn expand_int32_arg(
    args: &[ColumnarValue],
    arg_idx: Option<usize>,
    default_val: i32,
    num_iterations: usize,
) -> Result<Int32Array> {
    let array = match arg_idx {
        Some(idx) => args[idx]
            .clone()
            .cast_to(&DataType::Int32, None)?
            .into_array(num_iterations)?,
        None => ScalarValue::Int32(Some(default_val)).to_array_of_size(num_iterations)?,
    };
    Ok(as_int32_array(&array)?.clone())
}

/// Expand an optional Boolean argument to an array. If `arg_idx` is `None`, returns
/// a constant array filled with `default_val`.
fn expand_boolean_arg(
    args: &[ColumnarValue],
    arg_idx: Option<usize>,
    default_val: bool,
    num_iterations: usize,
) -> Result<BooleanArray> {
    let array = match arg_idx {
        Some(idx) => args[idx]
            .clone()
            .cast_to(&DataType::Boolean, None)?
            .into_array(num_iterations)?,
        None => ScalarValue::Boolean(Some(default_val)).to_array_of_size(num_iterations)?,
    };
    Ok(as_boolean_array(&array)?.clone())
}

/// Calculate number of iterations
fn calc_num_iterations(args: &[ColumnarValue]) -> usize {
    for arg in args {
        if let ColumnarValue::Array(array) = arg {
            return array.len();
        }
    }
    1
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_common::cast::as_struct_array;
    use sedona_gdal::mem::MemDatasetBuilder;
    use sedona_gdal::raster::GdalDataType;
    use sedona_raster::affine_transformation::to_world_coordinate;
    use sedona_raster::array::RasterStructArray;
    use sedona_raster_functions::crs_utils::crs_transform_coord;
    use sedona_schema::crs::deserialize_crs;
    use sedona_schema::datatypes::Edges;
    use sedona_schema::datatypes::RASTER;
    use sedona_testing::create::make_wkb;

    #[test]
    fn test_stat_type_from_str() {
        assert_eq!(StatType::from_str("count"), Some(StatType::Count));
        assert_eq!(StatType::from_str("COUNT"), Some(StatType::Count));
        assert_eq!(StatType::from_str("mean"), Some(StatType::Mean));
        assert_eq!(StatType::from_str("avg"), Some(StatType::Mean));
        assert_eq!(StatType::from_str("stddev"), Some(StatType::StdDev));
        assert_eq!(StatType::from_str("invalid"), None);
    }

    #[test]
    fn test_compute_statistics() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = compute_statistics(&values).unwrap();

        assert_eq!(stats.count, 5);
        assert!((stats.sum - 15.0).abs() < f64::EPSILON);
        assert!((stats.mean - 3.0).abs() < f64::EPSILON);
        assert!((stats.median - 3.0).abs() < f64::EPSILON);
        assert!((stats.min - 1.0).abs() < f64::EPSILON);
        assert!((stats.max - 5.0).abs() < f64::EPSILON);
        // Variance = ((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 4 = 10/4 = 2.5
        assert!((stats.variance - 2.5).abs() < 0.001);
        assert!((stats.stddev - 2.5_f64.sqrt()).abs() < 0.001);
    }

    #[test]
    fn test_compute_statistics_empty() {
        let values: Vec<f64> = vec![];
        let stats = compute_statistics(&values).unwrap();

        assert_eq!(stats.count, 0);
        assert_eq!(stats.sum, 0.0);
        assert!(stats.mean.is_nan());
        assert!(stats.min.is_nan());
        assert!(stats.max.is_nan());
    }

    #[test]
    fn test_compute_statistics_single() {
        let values = vec![42.0];
        let stats = compute_statistics(&values).unwrap();

        assert_eq!(stats.count, 1);
        assert!((stats.sum - 42.0).abs() < f64::EPSILON);
        assert!((stats.mean - 42.0).abs() < f64::EPSILON);
        assert!((stats.median - 42.0).abs() < f64::EPSILON);
        assert!((stats.min - 42.0).abs() < f64::EPSILON);
        assert!((stats.max - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rs_zonal_stats_with_test_raster() {
        use sedona_raster::array::RasterStructArray;

        let test_file = sedona_testing::data::test_raster("test4.tiff").unwrap();
        with_gdal(|gdal| {
            let raster_array = crate::utils::load_as_indb_raster(gdal, &test_file)?;
            let raster_struct = RasterStructArray::new(&raster_array);
            let raster = raster_struct.get(0).unwrap();

            let metadata = raster.metadata();
            let min_x = metadata.upper_left_x();
            let max_y = metadata.upper_left_y();
            let max_x = min_x + (metadata.width() as f64 * metadata.scale_x());
            let min_y = max_y + (metadata.height() as f64 * metadata.scale_y());

            let wkt = format!(
                "POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))",
                min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y, min_x, min_y
            );

            let geometry = gdal.geometry_from_wkt(&wkt).unwrap();
            let geom_wkb = geometry.wkb().map_err(|e| exec_datafusion_err!("{e}"))?;
            let stats = compute_zonal_stats(gdal, &raster, &geom_wkb, 1, false, true)?;
            assert!(stats.count > 0, "Should have some pixels");
            assert!(stats.min <= stats.max, "Min should be <= max");
            assert!(
                stats.min <= stats.mean && stats.mean <= stats.max,
                "Mean should be between min and max"
            );
            Ok::<_, datafusion_common::DataFusionError>(())
        })
        .unwrap();
    }

    #[test]
    fn test_rs_zonal_stats_crs_mismatch() {
        use sedona_expr::scalar_udf::SedonaScalarKernel;

        let test_file = sedona_testing::data::test_raster("test4.tiff").unwrap();
        let raster_array =
            with_gdal(|gdal| crate::utils::load_as_indb_raster(gdal, &test_file)).unwrap();

        let raster_struct = RasterStructArray::new(&raster_array);
        let raster = raster_struct.get(0).unwrap();
        let width = raster.metadata().width() as i64;
        let height = raster.metadata().height() as i64;
        let col = width / 2;
        let row = height / 2;
        let (lon, lat) = to_world_coordinate(&raster, col, row);

        let point_wkt = format!("POINT ({} {})", lon, lat);
        let point_wkb = make_wkb(&point_wkt);
        let (x_merc, y_merc) = with_global_proj_engine(|engine| {
            crs_transform_coord(engine, (lon, lat), "OGC:CRS84", "EPSG:3857")
        })
        .unwrap();
        let point_merc_wkt = format!("POINT ({} {})", x_merc, y_merc);
        let point_merc_wkb = make_wkb(&point_merc_wkt);

        let raster_scalar = ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(raster_array)));
        let geom_type_4326 = SedonaType::Wkb(Edges::Planar, deserialize_crs("EPSG:4326").unwrap());
        let geom_type_3857 = SedonaType::Wkb(Edges::Planar, deserialize_crs("EPSG:3857").unwrap());

        let zonal_kernel = RsZonalStats { arg_count: 3 };

        let stat_type = ColumnarValue::Scalar(ScalarValue::Utf8(Some("count".to_string())));

        let result_4326 = match zonal_kernel.invoke_batch(
            &[RASTER, geom_type_4326, SedonaType::Arrow(DataType::Utf8)],
            &[
                raster_scalar.clone(),
                ColumnarValue::Scalar(ScalarValue::Binary(Some(point_wkb))),
                stat_type.clone(),
            ],
        ) {
            Ok(value) => value,
            Err(err) => {
                let message = err.to_string();
                if message.contains("proj-sys") {
                    return;
                }
                panic!("Unexpected RS_ZonalStats error: {message}");
            }
        };

        let result_3857 = match zonal_kernel.invoke_batch(
            &[RASTER, geom_type_3857, SedonaType::Arrow(DataType::Utf8)],
            &[
                raster_scalar,
                ColumnarValue::Scalar(ScalarValue::Binary(Some(point_merc_wkb))),
                stat_type,
            ],
        ) {
            Ok(value) => value,
            Err(err) => {
                let message = err.to_string();
                if message.contains("proj-sys") {
                    return;
                }
                panic!("Unexpected RS_ZonalStats error: {message}");
            }
        };

        let value_4326 = match result_4326 {
            ColumnarValue::Scalar(ScalarValue::Float64(Some(value))) => value,
            _ => panic!("Expected Float64 scalar result"),
        };
        let value_3857 = match result_3857 {
            ColumnarValue::Scalar(ScalarValue::Float64(Some(value))) => value,
            _ => panic!("Expected Float64 scalar result"),
        };

        assert_eq!(value_4326, value_3857);
    }

    #[test]
    fn test_rs_zonal_stats_outdb_raster() {
        use arrow_schema::DataType;
        use sedona_expr::scalar_udf::SedonaScalarKernel;
        use sedona_schema::datatypes::SedonaType;
        use sedona_testing::create::make_wkb;

        let test_file = sedona_testing::data::test_raster("test4.tiff").unwrap();
        let in_db_array =
            with_gdal(|gdal| crate::utils::load_as_indb_raster(gdal, &test_file)).unwrap();

        let outdb_kernel = crate::rs_from_path::RsFromPath::new(false);
        let outdb_value = outdb_kernel
            .invoke_batch(
                &[SedonaType::Arrow(DataType::Utf8)],
                &[ColumnarValue::Scalar(ScalarValue::Utf8(Some(
                    test_file.clone(),
                )))],
            )
            .unwrap();

        let raster_struct = RasterStructArray::new(&in_db_array);
        let raster = raster_struct.get(0).unwrap();
        let metadata = raster.metadata();
        let min_x = metadata.upper_left_x();
        let max_y = metadata.upper_left_y();
        let max_x = min_x + (metadata.width() as f64 * metadata.scale_x());
        let min_y = max_y + (metadata.height() as f64 * metadata.scale_y());

        let wkt = format!(
            "POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))",
            min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y, min_x, min_y
        );
        let geom_wkb = make_wkb(&wkt);

        let zonal_kernel = RsZonalStats { arg_count: 3 };
        let geom_type = SedonaType::Wkb(Edges::Planar, deserialize_crs("EPSG:4326").unwrap());

        let args = vec![
            ColumnarValue::Scalar(ScalarValue::Struct(Arc::new(in_db_array.clone()))),
            ColumnarValue::Scalar(ScalarValue::Binary(Some(geom_wkb.clone()))),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("count".to_string()))),
        ];
        let outdb_args = vec![
            outdb_value,
            ColumnarValue::Scalar(ScalarValue::Binary(Some(geom_wkb))),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some("count".to_string()))),
        ];

        let in_db_result = match zonal_kernel.invoke_batch(
            &[RASTER, geom_type.clone(), SedonaType::Arrow(DataType::Utf8)],
            &args,
        ) {
            Ok(value) => value,
            Err(err) => {
                let message = err.to_string();
                if message.contains("proj-sys") {
                    return;
                }
                panic!("Unexpected RS_ZonalStats error: {message}");
            }
        };
        let outdb_result = match zonal_kernel.invoke_batch(
            &[RASTER, geom_type, SedonaType::Arrow(DataType::Utf8)],
            &outdb_args,
        ) {
            Ok(value) => value,
            Err(err) => {
                let message = err.to_string();
                if message.contains("proj-sys") {
                    return;
                }
                panic!("Unexpected RS_ZonalStats error: {message}");
            }
        };

        let in_db_value = match in_db_result {
            ColumnarValue::Scalar(ScalarValue::Float64(Some(value))) => value,
            _ => panic!("Expected Float64 scalar result"),
        };
        let outdb_value = match outdb_result {
            ColumnarValue::Scalar(ScalarValue::Float64(Some(value))) => value,
            _ => panic!("Expected Float64 scalar result"),
        };

        assert_eq!(in_db_value, outdb_value);
    }

    #[test]
    fn test_rs_zonal_stats_outdb_tile_from_rs_geotiff_tiles() {
        use arrow_schema::{DataType, Field, Schema, SchemaRef};
        use sedona_raster::array::RasterStructArray;
        use tempfile::tempdir;

        let tmp = tempdir().unwrap();
        let dst = tmp.path().join("test4.tiff");
        let src = sedona_testing::data::test_raster("test4.tiff").unwrap();
        std::fs::copy(&src, &dst).unwrap();

        // Build a record batch the same way rs_geotiff_tiles does.
        let rast_field = sedona_schema::datatypes::RASTER
            .to_storage_field("rast", false)
            .unwrap();
        let schema: SchemaRef = Arc::new(Schema::new(vec![
            Field::new("path", DataType::Utf8, false),
            Field::new("x", DataType::UInt32, false),
            Field::new("y", DataType::UInt32, false),
            rast_field,
        ]));

        let batch = crate::rs_geotiff_tiles::build_batch_for_file(dst, schema)
            .unwrap()
            .unwrap();
        assert!(batch.num_rows() > 0);

        let rast_array = batch.column(3).clone();
        let rast_struct_array =
            as_struct_array(&rast_array).expect("rast column should be a StructArray");
        let rast_struct = RasterStructArray::new(rast_struct_array);
        let raster = rast_struct.get(0).unwrap();

        // Polygon covering the whole tile.
        let metadata = raster.metadata();
        let min_x = metadata.upper_left_x();
        let max_y = metadata.upper_left_y();
        let max_x = min_x + (metadata.width() as f64 * metadata.scale_x());
        let min_y = max_y + (metadata.height() as f64 * metadata.scale_y());
        let wkt = format!(
            "POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))",
            min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y, min_x, min_y
        );
        let geom_wkb = make_wkb(&wkt);

        let result =
            with_gdal(|gdal| compute_zonal_stats(gdal, &raster, &geom_wkb, 1, false, true));
        assert!(
            result.is_ok(),
            "Zonal stats should succeed on out-db tiles: {:?}",
            result.err()
        );

        let stats = result.unwrap();
        assert!(stats.count > 0);
        assert!(stats.min <= stats.max);
        assert!(stats.min <= stats.mean && stats.mean <= stats.max);
    }

    #[test]
    fn test_rs_zonal_stats_outdb_tile_exclude_nodata() {
        use arrow_schema::{Schema, SchemaRef};
        use std::path::Path;
        use tempfile::tempdir;

        fn write_tiled_geotiff_f32(
            gdal: &sedona_gdal::gdal::Gdal,
            path: &Path,
            w: usize,
            h: usize,
            block: u32,
            nodata: f64,
        ) {
            let mem_ds = MemDatasetBuilder::create(gdal, w, h, 1, GdalDataType::Float32).unwrap();
            mem_ds
                .set_geo_transform(&[0.0, 1.0, 0.0, 0.0, 0.0, -1.0])
                .unwrap();

            let band = mem_ds.rasterband(1).unwrap();
            band.set_no_data_value(Some(nodata)).unwrap();

            let mut data: Vec<f32> = (0..(w * h)).map(|v| v as f32).collect();
            // Put a few nodata pixels in the upper-left block so tile (0,0) contains them.
            for (col, row) in [(0usize, 0usize), (1, 2), (3, 3)] {
                data[row * w + col] = nodata as f32;
            }
            let mut buffer = Buffer::new((w, h), data);
            band.write((0, 0), (w, h), &mut buffer).unwrap();

            let gtiff_driver = gdal.get_driver_by_name("GTiff").unwrap();
            let options_list = [
                "TILED=YES".to_string(),
                format!("BLOCKXSIZE={}", block),
                format!("BLOCKYSIZE={}", block),
            ];
            let options_refs: Vec<&str> = options_list.iter().map(|s| s.as_str()).collect();
            let _out = mem_ds
                .create_copy(&gtiff_driver, path.to_str().unwrap(), &options_refs)
                .unwrap();
        }

        with_gdal(|gdal| {
            let tmp = tempdir().unwrap();
            let dst = tmp.path().join("nodata_tiles.tif");
            let nodata = -9999.0;
            write_tiled_geotiff_f32(gdal, &dst, 32, 32, 16, nodata);

            let rast_field = sedona_schema::datatypes::RASTER
                .to_storage_field("rast", false)
                .unwrap();
            let schema: SchemaRef = Arc::new(Schema::new(vec![
                Field::new("path", DataType::Utf8, false),
                Field::new("x", DataType::UInt32, false),
                Field::new("y", DataType::UInt32, false),
                rast_field,
            ]));

            let batch = crate::rs_geotiff_tiles::build_batch_for_file(dst, schema)?
                .expect("expected at least one tile");
            assert!(batch.num_rows() > 0);

            let rast_array = batch.column(3).clone();
            let rast_struct_array =
                as_struct_array(&rast_array).expect("rast column should be a StructArray");
            let rast_struct = RasterStructArray::new(rast_struct_array);
            let raster = rast_struct.get(0).unwrap();

            let band = raster.bands().band(1).unwrap();
            let band_meta = band.metadata();
            let nodata_meta =
                nodata_bytes_to_f64(band_meta.nodata_value(), &band_meta.data_type().unwrap());

            let metadata = raster.metadata();
            let min_x = metadata.upper_left_x();
            let max_y = metadata.upper_left_y();
            let max_x = min_x + (metadata.width() as f64 * metadata.scale_x());
            let min_y = max_y + (metadata.height() as f64 * metadata.scale_y());
            let wkt = format!(
                "POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))",
                min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y, min_x, min_y
            );
            let geom_wkb = make_wkb(&wkt);

            let include = compute_zonal_stats(gdal, &raster, &geom_wkb, 1, false, false)?;
            let exclude = compute_zonal_stats(gdal, &raster, &geom_wkb, 1, false, true)?;
            assert_eq!(include.count, 256);
            assert_eq!(exclude.count, 253);
            assert_eq!(nodata_meta, Some(-9999.0));
            assert!(exclude.min >= 0.0);
            assert!((include.sum - exclude.sum - 3.0 * nodata).abs() < 1e-6);
            Ok::<_, datafusion_common::DataFusionError>(())
        })
        .unwrap();
    }
}
