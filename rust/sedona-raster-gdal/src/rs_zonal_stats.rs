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

//! RS_ZonalStats / RS_ZonalStatsAll UDFs — summary statistics of the raster
//! pixels covered by a zone geometry.
//!
//! `RS_ZonalStats(raster, zone, stat[, options])` returns one statistic as a
//! `Float64`; `RS_ZonalStatsAll(raster, zone[, options])` returns every
//! statistic as a struct. Both compute over the pixels of a single band whose
//! centre falls inside the zone (or that the zone merely touches, with
//! `all_touched`), optionally excluding the band's nodata value.
//!
//! The optional trailing `options` argument is a JSON object rather than the
//! positional `band, all_touched, exclude_nodata, lenient` overload ladder of
//! Sedona Spark, so the surface stays a single `(input, options)` shape:
//!
//! ```json
//! {"band": 1, "all_touched": false, "exclude_nodata": true, "lenient": true}
//! ```
//!
//! These functions operate on 2-D `(y, x)` bands. A band that is not a 2-D
//! spatial grid is rejected; computing a statistic per non-spatial plane of an
//! N-D band is not supported.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::builder::{Float64Builder, Int64Builder, StructBuilder};
use arrow_array::{ArrayRef, StringArray};
use arrow_schema::{DataType, Field, Fields};
use datafusion_common::config::ConfigOptions;
use datafusion_common::error::Result;
use datafusion_common::{exec_datafusion_err, exec_err};
use datafusion_expr::{ColumnarValue, Volatility};
use serde::Deserialize;

use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_gdal::gdal::Gdal;
use sedona_gdal::geo_transform::{GeoTransform, GeoTransformEx};
use sedona_gdal::mem::MemDatasetBuilder;
use sedona_gdal::raster::types::GdalDataType;
use sedona_gdal::vector::geometry::Geometry;
use sedona_raster::array::RasterRefImpl;
use sedona_raster::traits::RasterRef;
use sedona_raster_functions::crs_utils::{crs_transform_wkb, resolve_crs, with_crs_engine};
use sedona_raster_functions::rs_ensure_loaded::NEEDS_PIXELS_METADATA_KEY;
use sedona_raster_functions::RasterExecutor;
use sedona_schema::datatypes::SedonaType;
use sedona_schema::matchers::ArgMatcher;
use sedona_schema::raster::BandDataType;

use crate::gdal_common::with_gdal;
use crate::gdal_dataset_provider::configure_thread_local_options;

/// The statistics RS_ZonalStatsAll returns, in the order Sedona Spark reports
/// them. RS_ZonalStats selects one of these by name.
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
    /// Parse a statistic name (case-insensitive). Aliases match Sedona Spark
    /// (`avg`/`average` for mean, `sd` for stddev).
    fn from_str(s: &str) -> Option<StatType> {
        match s.to_lowercase().as_str() {
            "count" => Some(StatType::Count),
            "sum" => Some(StatType::Sum),
            "mean" | "avg" | "average" => Some(StatType::Mean),
            "median" => Some(StatType::Median),
            "mode" => Some(StatType::Mode),
            "stddev" | "sd" => Some(StatType::StdDev),
            "variance" => Some(StatType::Variance),
            "min" => Some(StatType::Min),
            "max" => Some(StatType::Max),
            _ => None,
        }
    }
}

/// Optional `(input, options)` JSON payload. Missing fields take their default;
/// unknown fields are rejected so a typo surfaces rather than being ignored.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "snake_case")]
struct ZonalStatsOptions {
    /// 1-based band to compute over. `None` means "resolve the default": band 1
    /// for a single-band raster, an error for a multiband raster (naming the
    /// band is required rather than silently getting band 1).
    band: Option<i64>,
    /// Include every pixel the zone touches, not only those whose centre it
    /// covers.
    all_touched: bool,
    /// Skip pixels equal to the band's nodata value.
    exclude_nodata: bool,
    /// Return NULL when the zone does not intersect the raster, rather than
    /// erroring. Only the no-intersection case is softened; malformed geometry
    /// or an unreadable band always errors.
    lenient: bool,
}

impl Default for ZonalStatsOptions {
    fn default() -> Self {
        Self {
            band: None,
            all_touched: false,
            exclude_nodata: true,
            lenient: true,
        }
    }
}

impl ZonalStatsOptions {
    /// Parse the JSON options string, or return the defaults when no options
    /// argument was supplied (`None`) or it is SQL NULL.
    fn parse(json: Option<&str>) -> Result<Self> {
        match json {
            None => Ok(Self::default()),
            Some(s) => serde_json::from_str(s)
                .map_err(|e| exec_datafusion_err!("RS_ZonalStats: invalid options JSON: {e}")),
        }
    }
}

/// Every statistic for a zone. `count` is always present (0 when the zone
/// selects no pixels); the remaining fields are `None` in exactly that
/// no-pixel case and `Some` otherwise, mirroring Sedona Spark (which returns
/// `count = 0` and NULL for the rest).
#[derive(Debug, Clone, PartialEq)]
struct ZonalStatistics {
    count: i64,
    sum: Option<f64>,
    mean: Option<f64>,
    median: Option<f64>,
    mode: Option<f64>,
    stddev: Option<f64>,
    variance: Option<f64>,
    min: Option<f64>,
    max: Option<f64>,
}

impl ZonalStatistics {
    /// The value RS_ZonalStats returns for a single statistic. `count` is never
    /// NULL (it is 0 for an empty zone); the others are NULL for an empty zone.
    fn get(&self, stat_type: StatType) -> Option<f64> {
        match stat_type {
            StatType::Count => Some(self.count as f64),
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
// RS_ZonalStats
// =============================================================================

/// `RS_ZonalStats(raster, zone, stat[, options])` — one statistic as a
/// `Float64`. `stat` is a statistic name (`count`, `sum`, `mean`, `median`,
/// `mode`, `stddev`, `variance`, `min`, `max`).
pub fn rs_zonal_stats_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_zonalstats",
        vec![
            Arc::new(RsZonalStats {
                with_options: false,
            }),
            Arc::new(RsZonalStats { with_options: true }),
        ],
        Volatility::Immutable,
    )
    // Reads band pixels, so the planner materializes OutDb rasters via
    // RS_EnsureLoaded first.
    .with_metadata(NEEDS_PIXELS_METADATA_KEY, "true")
}

#[derive(Debug)]
struct RsZonalStats {
    /// Whether this kernel matches the trailing JSON `options` argument.
    with_options: bool,
}

impl SedonaScalarKernel for RsZonalStats {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let mut matchers = vec![
            ArgMatcher::is_raster(),
            ArgMatcher::is_geometry_or_geography(),
            ArgMatcher::is_string(),
        ];
        if self.with_options {
            matchers.push(ArgMatcher::is_string());
        }
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
        let num_iterations = RasterExecutor::num_iterations_over(args);

        // stat is at index 2, options (when present) at index 3.
        let stat_array = expand_string_arg(&args[2], num_iterations)?;
        let mut stat_iter = stat_array.iter();
        let options_array = self
            .with_options
            .then(|| expand_string_arg(&args[3], num_iterations))
            .transpose()?;
        let mut options_iter = options_array.as_ref().map(|a| a.iter());

        let mut builder = Float64Builder::with_capacity(num_iterations);
        let mut scratch: Vec<f64> = Vec::new();

        // The executor only sees (raster, zone); the stat/options columns are
        // advanced in lockstep below.
        let exec_arg_types = [arg_types[0].clone(), arg_types[1].clone()];
        let exec_args = [args[0].clone(), args[1].clone()];
        let executor =
            RasterExecutor::new_with_num_iterations(&exec_arg_types, &exec_args, num_iterations);

        with_gdal(|gdal| {
            configure_thread_local_options(gdal, config_options)?;
            with_crs_engine(config_options, |engine| {
                executor.execute_raster_wkb_crs_void(|raster_opt, wkb_opt, geom_crs| {
                    let stat_str = stat_iter.next().flatten();
                    let options_str = options_iter.as_mut().and_then(|i| i.next().flatten());
                    let options = ZonalStatsOptions::parse(options_str)?;

                    // A NULL stat, raster, or zone propagates to a NULL row.
                    let (Some(stat_str), Some(raster), Some(wkb)) = (stat_str, raster_opt, wkb_opt)
                    else {
                        builder.append_null();
                        return Ok(());
                    };
                    let stat_type = StatType::from_str(stat_str).ok_or_else(|| {
                        exec_datafusion_err!("RS_ZonalStats: unknown statistic {stat_str:?}")
                    })?;

                    // Reproject the zone into the raster's CRS; a known/unknown
                    // CRS mismatch on either side would mislocate it.
                    let raster_crs = resolve_crs(raster.crs())?;
                    let geom_wkb = match (geom_crs, raster_crs.as_deref()) {
                        (Some(geom_crs), Some(raster_crs)) => {
                            crs_transform_wkb(wkb, geom_crs, raster_crs, engine)?
                        }
                        (None, None) => wkb.to_vec(),
                        (Some(_), None) => return exec_err!(
                            "Cannot operate on geometry and raster: raster has no CRS but geometry does"
                        ),
                        (None, Some(_)) => return exec_err!(
                            "Cannot operate on geometry and raster: geometry has no CRS but raster does"
                        ),
                    };
                    match compute_zonal_stats(gdal, raster, &geom_wkb, &options, &mut scratch)? {
                        Some(stats) => match stats.get(stat_type) {
                            Some(value) => builder.append_value(value),
                            None => builder.append_null(),
                        },
                        // The zone does not intersect the raster: NULL when
                        // lenient (the default), an error otherwise.
                        None if options.lenient => builder.append_null(),
                        None => return no_intersection_err(),
                    }
                    Ok(())
                })
            })?;

            let out: ArrayRef = Arc::new(builder.finish());
            RasterExecutor::finish_over(args, out)
        })
    }
}

// =============================================================================
// RS_ZonalStatsAll
// =============================================================================

/// `RS_ZonalStatsAll(raster, zone[, options])` — every statistic as a struct
/// with fields `count, sum, mean, median, mode, stddev, variance, min, max`.
pub fn rs_zonal_stats_all_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_zonalstatsall",
        vec![
            Arc::new(RsZonalStatsAll {
                with_options: false,
            }),
            Arc::new(RsZonalStatsAll { with_options: true }),
        ],
        Volatility::Immutable,
    )
    .with_metadata(NEEDS_PIXELS_METADATA_KEY, "true")
}

#[derive(Debug)]
struct RsZonalStatsAll {
    with_options: bool,
}

impl SedonaScalarKernel for RsZonalStatsAll {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let mut matchers = vec![
            ArgMatcher::is_raster(),
            ArgMatcher::is_geometry_or_geography(),
        ];
        if self.with_options {
            matchers.push(ArgMatcher::is_string());
        }
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
        let num_iterations = RasterExecutor::num_iterations_over(args);

        // options (when present) is at index 2.
        let options_array = self
            .with_options
            .then(|| expand_string_arg(&args[2], num_iterations))
            .transpose()?;
        let mut options_iter = options_array.as_ref().map(|a| a.iter());

        let mut builder = StructBuilder::from_fields(zonal_stats_struct_fields(), num_iterations);
        let mut scratch: Vec<f64> = Vec::new();

        let exec_arg_types = [arg_types[0].clone(), arg_types[1].clone()];
        let exec_args = [args[0].clone(), args[1].clone()];
        let executor =
            RasterExecutor::new_with_num_iterations(&exec_arg_types, &exec_args, num_iterations);

        with_gdal(|gdal| {
            configure_thread_local_options(gdal, config_options)?;
            with_crs_engine(config_options, |engine| {
                executor.execute_raster_wkb_crs_void(|raster_opt, wkb_opt, geom_crs| {
                    let options_str = options_iter.as_mut().and_then(|i| i.next().flatten());
                    let options = ZonalStatsOptions::parse(options_str)?;

                    let (Some(raster), Some(wkb)) = (raster_opt, wkb_opt) else {
                        append_struct_null(&mut builder);
                        return Ok(());
                    };

                    let raster_crs = resolve_crs(raster.crs())?;
                    let geom_wkb = match (geom_crs, raster_crs.as_deref()) {
                        (Some(geom_crs), Some(raster_crs)) => {
                            crs_transform_wkb(wkb, geom_crs, raster_crs, engine)?
                        }
                        (None, None) => wkb.to_vec(),
                        (Some(_), None) => return exec_err!(
                            "Cannot operate on geometry and raster: raster has no CRS but geometry does"
                        ),
                        (None, Some(_)) => return exec_err!(
                            "Cannot operate on geometry and raster: geometry has no CRS but raster does"
                        ),
                    };
                    match compute_zonal_stats(gdal, raster, &geom_wkb, &options, &mut scratch)? {
                        Some(stats) => append_struct_stats(&mut builder, &stats),
                        None if options.lenient => append_struct_null(&mut builder),
                        None => return no_intersection_err(),
                    }
                    Ok(())
                })
            })?;

            let out: ArrayRef = Arc::new(builder.finish());
            RasterExecutor::finish_over(args, out)
        })
    }
}

/// Struct data type RS_ZonalStatsAll returns.
fn zonal_stats_struct_type() -> DataType {
    DataType::Struct(zonal_stats_struct_fields())
}

/// Fields of the RS_ZonalStatsAll struct, in Sedona Spark order. `count` is an
/// `Int64` (a whole pixel count); every other statistic is a `Float64`.
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

/// Append a fully-NULL struct row (the zone does not intersect the raster and
/// `lenient` is set).
fn append_struct_null(builder: &mut StructBuilder) {
    builder
        .field_builder::<Int64Builder>(0)
        .unwrap()
        .append_null();
    for i in 1..=8 {
        builder
            .field_builder::<Float64Builder>(i)
            .unwrap()
            .append_null();
    }
    builder.append(false);
}

/// Append one computed-stats row. The float fields carry through the `Option`
/// so an empty zone records `count = 0` with the rest NULL.
fn append_struct_stats(builder: &mut StructBuilder, stats: &ZonalStatistics) {
    builder
        .field_builder::<Int64Builder>(0)
        .unwrap()
        .append_value(stats.count);
    for (i, value) in [
        stats.sum,
        stats.mean,
        stats.median,
        stats.mode,
        stats.stddev,
        stats.variance,
        stats.min,
        stats.max,
    ]
    .into_iter()
    .enumerate()
    {
        builder
            .field_builder::<Float64Builder>(i + 1)
            .unwrap()
            .append_option(value);
    }
    builder.append(true);
}

// =============================================================================
// Core computation
// =============================================================================

/// A rectangular pixel window (offset + size) into the raster grid.
#[derive(Clone, Copy, Debug)]
struct PixelWindow {
    col_off: usize,
    row_off: usize,
    width: usize,
    height: usize,
}

/// Compute the statistics of the pixels a zone geometry selects on one band.
///
/// Returns `Ok(None)` when the zone's envelope does not intersect the raster
/// extent (the caller turns this into NULL when `lenient`, an error otherwise).
/// A zone that intersects the extent but whose selected pixels are all outside
/// the geometry or all nodata yields `Ok(Some(..))` with `count = 0`.
///
/// `scratch` is a reused buffer for the selected pixel values so the per-row
/// collection does not allocate a fresh `Vec` each call.
fn compute_zonal_stats(
    gdal: &Gdal,
    raster: &RasterRefImpl<'_>,
    geom_wkb: &[u8],
    options: &ZonalStatsOptions,
    scratch: &mut Vec<f64>,
) -> Result<Option<ZonalStatistics>> {
    let num_bands = raster.num_bands();
    let band_num = resolve_band(options.band, num_bands)?;

    let band = raster
        .bands()
        .band(band_num)
        .map_err(|e| exec_datafusion_err!("RS_ZonalStats: failed to read band {band_num}: {e}"))?;
    if !band.is_spatial_2d() {
        return exec_err!(
            "RS_ZonalStats supports 2-D rasters only; band {band_num} is not a 2-D (y, x) grid"
        );
    }
    let data_type = band.data_type();
    let byte_size = data_type.byte_size();

    let metadata = raster.metadata();
    let transform = raster_transform(raster)?;
    let width = usize::try_from(metadata.width())
        .map_err(|_| exec_datafusion_err!("RS_ZonalStats: negative raster width"))?;
    let height = usize::try_from(metadata.height())
        .map_err(|_| exec_datafusion_err!("RS_ZonalStats: negative raster height"))?;

    // Parse the zone and clamp its envelope to the raster grid. A disjoint
    // envelope is the no-intersection case.
    let geometry = gdal
        .geometry_from_wkb(geom_wkb)
        .map_err(|e| exec_datafusion_err!("RS_ZonalStats: failed to parse geometry: {e}"))?;
    let Some(window) = envelope_window(&geometry, &transform, width, height)? else {
        return Ok(None);
    };

    // Rasterize the zone into a window-sized 0/1 mask (moves `geometry`, whose
    // only remaining use is the burn).
    let mask = rasterize_zone_mask(gdal, geometry, &transform, &window, options.all_touched)?;

    // Read the band once (zero-copy borrow) and collect the selected values.
    let nd_buffer = band
        .nd_buffer()
        .map_err(|e| exec_datafusion_err!("RS_ZonalStats: failed to read band {band_num}: {e}"))?;
    let band_bytes = nd_buffer.as_contiguous().map_err(|e| {
        exec_datafusion_err!("RS_ZonalStats: band {band_num} is not contiguous: {e}")
    })?;
    let expected = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(byte_size))
        .ok_or_else(|| exec_datafusion_err!("RS_ZonalStats: raster dimensions overflow"))?;
    if band_bytes.len() != expected {
        return sedona_internal_err!(
            "RS_ZonalStats: band {band_num} byte length {} does not match {width}x{height} of {data_type:?}",
            band_bytes.len()
        );
    }

    // Nodata is compared in the band's own byte representation, never through
    // f64 — an Int64/UInt64 nodata beyond 2^53 must not alias a nearby pixel.
    let nodata = if options.exclude_nodata {
        band.nodata()
    } else {
        None
    };

    scratch.clear();
    collect_masked_values(
        band_bytes, data_type, width, &window, &mask, nodata, scratch,
    );

    Ok(Some(compute_statistics(scratch)))
}

/// Resolve the 1-based band to use. `Some(b)` must be a valid 1-based index;
/// `None` defaults to band 1 for a single-band raster and errors for a
/// multiband raster (matching the codebase's `default_band` convention, which
/// refuses to silently pick band 1 when the choice is ambiguous).
fn resolve_band(band: Option<i64>, num_bands: usize) -> Result<usize> {
    match band {
        Some(b) => {
            if b < 1 {
                return exec_err!("RS_ZonalStats: band must be >= 1, got {b}");
            }
            let b = b as usize;
            if b > num_bands {
                return exec_err!("RS_ZonalStats: band {b} is out of range (1-{num_bands})");
            }
            Ok(b)
        }
        None => {
            if num_bands == 1 {
                Ok(1)
            } else {
                exec_err!(
                    "RS_ZonalStats: raster has {num_bands} bands; set the \"band\" option to \
                     choose one (only a single-band raster may omit it)"
                )
            }
        }
    }
}

/// The raster's 6-coefficient GDAL geotransform as a fixed array.
fn raster_transform(raster: &RasterRefImpl<'_>) -> Result<GeoTransform> {
    let t = raster.transform();
    <[f64; 6]>::try_from(t)
        .map_err(|_| exec_datafusion_err!("RS_ZonalStats: expected a 6-element geotransform"))
}

/// The zone's envelope intersected with the raster extent, snapped outward to
/// whole pixels. `None` when the envelope is disjoint from the raster. Mirrors
/// the window RS_Clip / PostGIS ST_Clip use: all four corners are mapped
/// through the inverse geotransform so a skewed raster still gets a correct
/// superset window.
fn envelope_window(
    geometry: &Geometry,
    transform: &GeoTransform,
    width: usize,
    height: usize,
) -> Result<Option<PixelWindow>> {
    let env = geometry.envelope();
    let inverse = transform
        .invert()
        .map_err(|e| exec_datafusion_err!("RS_ZonalStats: geotransform is not invertible: {e}"))?;

    let corners = [
        (env.MinX, env.MinY),
        (env.MinX, env.MaxY),
        (env.MaxX, env.MinY),
        (env.MaxX, env.MaxY),
    ];
    let mut min_col = f64::INFINITY;
    let mut max_col = f64::NEG_INFINITY;
    let mut min_row = f64::INFINITY;
    let mut max_row = f64::NEG_INFINITY;
    for (x, y) in corners {
        let (col, row) = inverse.apply(x, y);
        min_col = min_col.min(col);
        max_col = max_col.max(col);
        min_row = min_row.min(row);
        max_row = max_row.max(row);
    }

    let col0 = min_col.floor();
    let row0 = min_row.floor();
    let col1 = max_col.ceil().max(col0 + 1.0);
    let row1 = max_row.ceil().max(row0 + 1.0);

    // Intersect with the raster extent. `>=` also rejects the NaN envelope of
    // an empty geometry.
    let col0 = col0.max(0.0);
    let row0 = row0.max(0.0);
    let col1 = col1.min(width as f64);
    let row1 = row1.min(height as f64);
    if !(col0 < col1 && row0 < row1) {
        return Ok(None);
    }

    Ok(Some(PixelWindow {
        col_off: col0 as usize,
        row_off: row0 as usize,
        width: (col1 - col0) as usize,
        height: (row1 - row0) as usize,
    }))
}

/// Rasterize the zone into a window-sized `u8` mask: 1 where the zone covers a
/// pixel, 0 elsewhere. The MEM band is zero-filled on creation, so only the
/// burn (value 1, inside the geometry) has to be written.
fn rasterize_zone_mask(
    gdal: &Gdal,
    geometry: Geometry,
    transform: &GeoTransform,
    window: &PixelWindow,
    all_touched: bool,
) -> Result<Vec<u8>> {
    let mask_dataset =
        MemDatasetBuilder::create(gdal, window.width, window.height, 1, GdalDataType::UInt8)
            .map_err(|e| exec_datafusion_err!("RS_ZonalStats: failed to create mask: {e}"))?;
    let (window_ulx, window_uly) = transform.apply(window.col_off as f64, window.row_off as f64);
    let mask_transform = [
        window_ulx,
        transform[1],
        transform[2],
        window_uly,
        transform[4],
        transform[5],
    ];
    mask_dataset
        .set_geo_transform(&mask_transform)
        .map_err(|e| exec_datafusion_err!("RS_ZonalStats: failed to set mask geotransform: {e}"))?;

    gdal.rasterize_affine(&mask_dataset, &[1], &[geometry], &[1.0], all_touched)
        .map_err(|e| exec_datafusion_err!("RS_ZonalStats: failed to rasterize zone: {e}"))?;

    let mask_band = mask_dataset
        .rasterband(1)
        .map_err(|e| exec_datafusion_err!("RS_ZonalStats: failed to read mask band: {e}"))?;
    let mask_buffer = mask_band
        .read_as::<u8>(
            (0, 0),
            (window.width, window.height),
            (window.width, window.height),
            None,
        )
        .map_err(|e| exec_datafusion_err!("RS_ZonalStats: failed to read mask: {e}"))?;
    Ok(mask_buffer.data().to_vec())
}

/// Append every selected pixel value (masked in, and — when `nodata` is set —
/// not byte-equal to the nodata sentinel) to `out` as `f64`.
///
/// The data type is dispatched once, outside the loop, so the per-pixel body is
/// a fixed-width little-endian read plus the mask/nodata comparisons rather
/// than a per-pixel type match.
fn collect_masked_values(
    band_bytes: &[u8],
    data_type: BandDataType,
    width: usize,
    window: &PixelWindow,
    mask: &[u8],
    nodata: Option<&[u8]>,
    out: &mut Vec<f64>,
) {
    macro_rules! collect {
        ($t:ty, $n:literal) => {{
            for row in 0..window.height {
                let src_row = window.row_off + row;
                let mask_row = row * window.width;
                for col in 0..window.width {
                    if mask[mask_row + col] == 0 {
                        continue;
                    }
                    let idx = (src_row * width + window.col_off + col) * $n;
                    let px = &band_bytes[idx..idx + $n];
                    if let Some(nd) = nodata {
                        if px == nd {
                            continue;
                        }
                    }
                    let mut arr = [0u8; $n];
                    arr.copy_from_slice(px);
                    out.push(<$t>::from_le_bytes(arr) as f64);
                }
            }
        }};
    }

    match data_type {
        BandDataType::UInt8 => collect!(u8, 1),
        BandDataType::Int8 => collect!(i8, 1),
        BandDataType::UInt16 => collect!(u16, 2),
        BandDataType::Int16 => collect!(i16, 2),
        BandDataType::UInt32 => collect!(u32, 4),
        BandDataType::Int32 => collect!(i32, 4),
        BandDataType::UInt64 => collect!(u64, 8),
        BandDataType::Int64 => collect!(i64, 8),
        BandDataType::Float32 => collect!(f32, 4),
        BandDataType::Float64 => collect!(f64, 8),
    }
}

/// Compute every statistic from the selected pixel values.
///
/// An empty slice yields `count = 0` and NULL for the rest (Sedona Spark's
/// empty-zone shortcut). Variance is the sample (n-1) variance, matching Spark;
/// for a single pixel it is 0. Median is the linear-interpolated 50th
/// percentile, which for the median reduces to the middle element (odd n) or
/// the mean of the two central elements (even n). Mode is the most frequent
/// value, breaking ties toward the larger value.
///
/// `values` is sorted in place (for the median); the caller owns it as reusable
/// scratch.
fn compute_statistics(values: &mut [f64]) -> ZonalStatistics {
    let count = values.len() as i64;
    if values.is_empty() {
        return ZonalStatistics {
            count: 0,
            sum: None,
            mean: None,
            median: None,
            mode: None,
            stddev: None,
            variance: None,
            min: None,
            max: None,
        };
    }

    let n = values.len();
    let sum: f64 = values.iter().sum();
    let mean = sum / n as f64;
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let variance = if n > 1 {
        let sum_sq: f64 = values.iter().map(|&v| (v - mean).powi(2)).sum();
        sum_sq / (n as f64 - 1.0)
    } else {
        0.0
    };
    let stddev = variance.sqrt();

    let mode = compute_mode(values);

    // Median needs the values sorted; do it in place on the scratch buffer.
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = n / 2;
    let median = if n.is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    };

    ZonalStatistics {
        count,
        sum: Some(sum),
        mean: Some(mean),
        median: Some(median),
        mode: Some(mode),
        stddev: Some(stddev),
        variance: Some(variance),
        min: Some(min),
        max: Some(max),
    }
}

/// The most frequent value, breaking ties toward the larger value (matching
/// Sedona Spark's `StatUtils.mode`, which returns the largest of the tied
/// modes). Values are keyed by their exact bit pattern, so integer-valued
/// pixels compare exactly.
fn compute_mode(values: &[f64]) -> f64 {
    let mut counts: HashMap<u64, usize> = HashMap::new();
    for &v in values {
        *counts.entry(v.to_bits()).or_insert(0) += 1;
    }
    let best_count = counts.values().copied().max().unwrap_or(0);
    counts
        .into_iter()
        .filter(|&(_, c)| c == best_count)
        .map(|(bits, _)| f64::from_bits(bits))
        .fold(f64::NEG_INFINITY, f64::max)
}

// =============================================================================
// Argument helpers
// =============================================================================

/// The error returned for a non-intersecting zone when `lenient` is off.
fn no_intersection_err<T>() -> Result<T> {
    exec_err!(
        "RS_ZonalStats: the zone geometry does not intersect the raster; \
         set the \"lenient\" option to return NULL instead"
    )
}

/// Cast a column to `Utf8` and materialize it to a `StringArray` so its values
/// can be iterated in lockstep with the raster/zone rows.
fn expand_string_arg(arg: &ColumnarValue, num_iterations: usize) -> Result<StringArray> {
    let array = arg
        .clone()
        .cast_to(&DataType::Utf8, None)?
        .into_array(num_iterations)?;
    Ok(datafusion_common::cast::as_string_array(&array)?.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stat_type_from_str_matches_spark_aliases() {
        assert_eq!(StatType::from_str("count"), Some(StatType::Count));
        assert_eq!(StatType::from_str("COUNT"), Some(StatType::Count));
        assert_eq!(StatType::from_str("mean"), Some(StatType::Mean));
        assert_eq!(StatType::from_str("avg"), Some(StatType::Mean));
        assert_eq!(StatType::from_str("average"), Some(StatType::Mean));
        assert_eq!(StatType::from_str("stddev"), Some(StatType::StdDev));
        assert_eq!(StatType::from_str("sd"), Some(StatType::StdDev));
        assert_eq!(StatType::from_str("variance"), Some(StatType::Variance));
        assert_eq!(StatType::from_str("min"), Some(StatType::Min));
        assert_eq!(StatType::from_str("max"), Some(StatType::Max));
        assert_eq!(StatType::from_str("nonsense"), None);
    }

    #[test]
    fn options_parse_defaults_and_overrides() {
        let d = ZonalStatsOptions::parse(None).unwrap();
        assert_eq!(d.band, None);
        assert!(!d.all_touched);
        assert!(d.exclude_nodata);
        assert!(d.lenient);

        let o = ZonalStatsOptions::parse(Some(
            r#"{"band": 2, "all_touched": true, "exclude_nodata": false, "lenient": false}"#,
        ))
        .unwrap();
        assert_eq!(o.band, Some(2));
        assert!(o.all_touched);
        assert!(!o.exclude_nodata);
        assert!(!o.lenient);

        // A partial object keeps the defaults for the omitted fields.
        let p = ZonalStatsOptions::parse(Some(r#"{"band": 3}"#)).unwrap();
        assert_eq!(p.band, Some(3));
        assert!(p.exclude_nodata);
    }

    #[test]
    fn options_parse_rejects_unknown_field() {
        let err = ZonalStatsOptions::parse(Some(r#"{"bnad": 1}"#)).unwrap_err();
        assert!(err.to_string().contains("invalid options JSON"), "{err}");
    }

    #[test]
    fn resolve_band_defaults_and_bounds() {
        // Single-band raster may omit the band.
        assert_eq!(resolve_band(None, 1).unwrap(), 1);
        // Multiband raster must name the band.
        let err = resolve_band(None, 3).unwrap_err().to_string();
        assert!(err.contains("has 3 bands"), "{err}");
        // Explicit band is range-checked (1-based).
        assert_eq!(resolve_band(Some(2), 3).unwrap(), 2);
        assert!(resolve_band(Some(0), 3)
            .unwrap_err()
            .to_string()
            .contains(">= 1"));
        assert!(resolve_band(Some(4), 3)
            .unwrap_err()
            .to_string()
            .contains("out of range"));
    }

    #[test]
    fn statistics_of_one_to_five() {
        // count, sum, mean, min, max, median are exact; variance/stddev are the
        // sample (n-1) values: ((1-3)^2+..+(5-3)^2)/4 = 10/4 = 2.5.
        let mut values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = compute_statistics(&mut values);
        assert_eq!(s.count, 5);
        assert_eq!(s.sum, Some(15.0));
        assert_eq!(s.mean, Some(3.0));
        assert_eq!(s.min, Some(1.0));
        assert_eq!(s.max, Some(5.0));
        assert_eq!(s.median, Some(3.0));
        assert_eq!(s.variance, Some(2.5));
        assert_eq!(s.stddev, Some(2.5_f64.sqrt()));
    }

    #[test]
    fn statistics_empty_is_zero_count_and_nulls() {
        let mut values: Vec<f64> = vec![];
        let s = compute_statistics(&mut values);
        assert_eq!(s.count, 0);
        assert_eq!(s.sum, None);
        assert_eq!(s.mean, None);
        assert_eq!(s.median, None);
        assert_eq!(s.min, None);
        assert_eq!(s.max, None);
        assert_eq!(s.variance, None);
        // A single-stat lookup returns 0 for count, NULL for the rest.
        assert_eq!(s.get(StatType::Count), Some(0.0));
        assert_eq!(s.get(StatType::Sum), None);
        assert_eq!(s.get(StatType::Mean), None);
    }

    #[test]
    fn statistics_single_value_has_zero_variance() {
        let mut values = vec![42.0];
        let s = compute_statistics(&mut values);
        assert_eq!(s.count, 1);
        assert_eq!(s.mean, Some(42.0));
        assert_eq!(s.median, Some(42.0));
        assert_eq!(s.variance, Some(0.0));
        assert_eq!(s.stddev, Some(0.0));
        assert_eq!(s.mode, Some(42.0));
    }

    #[test]
    fn median_even_count_averages_the_middle_pair() {
        let mut values = vec![4.0, 1.0, 3.0, 2.0];
        let s = compute_statistics(&mut values);
        assert_eq!(s.median, Some(2.5));
    }

    #[test]
    fn mode_breaks_ties_toward_the_larger_value() {
        // 1 and 3 each appear twice; the tie resolves to the larger, 3.
        let mut values = vec![1.0, 1.0, 3.0, 3.0, 2.0];
        assert_eq!(compute_statistics(&mut values).mode, Some(3.0));
        // A clear winner is returned as-is.
        let mut values = vec![7.0, 7.0, 7.0, 1.0, 2.0];
        assert_eq!(compute_statistics(&mut values).mode, Some(7.0));
    }
}

/// UDF-level tests: exercise the kernels end to end and pin the numbers against
/// values computed by hand (which agree with numpy — see the Python parity
/// tests for the rasterio/numpy cross-check).
#[cfg(test)]
mod udf_tests {
    use super::*;

    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::types::{Float64Type, Int64Type};
    use arrow_array::{Array, StructArray};
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use sedona_proj::transform::{with_global_proj_engine, LazyProjEngine};
    use sedona_schema::crs::deserialize_crs;
    use sedona_schema::datatypes::{Edges, RASTER};
    use sedona_testing::create::make_wkb;
    use sedona_testing::raster_spec::RasterSpec;
    use sedona_testing::testers::ScalarUdfTester;

    // Struct field positions (Sedona Spark order).
    const COUNT: usize = 0;
    const SUM: usize = 1;
    const MEAN: usize = 2;
    const MEDIAN: usize = 3;
    const MODE: usize = 4;
    const STDDEV: usize = 5;
    const VARIANCE: usize = 6;
    const MIN: usize = 7;
    const MAX: usize = 8;

    /// A 4×2 UInt8 raster with pixel values 1..=8 (row-major), world extent
    /// x ∈ [0, 4], y ∈ [0, 2] with 1×1 north-up pixels. Pixel centres:
    /// row y=1.5 → 1,2,3,4 at x=0.5,1.5,2.5,3.5; row y=0.5 → 5,6,7,8.
    fn small_raster() -> RasterSpec {
        RasterSpec::d2(4, 2)
            .band_values(&[1u8, 2, 3, 4, 5, 6, 7, 8])
            .crs(None)
            .transform([0.0, 1.0, 0.0, 2.0, 0.0, -1.0])
    }

    /// The left half of `small_raster` (x ∈ [0, 2]) selects the four pixels
    /// {1, 2, 5, 6}.
    const LEFT_HALF: &str = "POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))";

    fn stats_arg_types(with_options: bool) -> Vec<SedonaType> {
        let mut t = vec![
            RASTER,
            SedonaType::Wkb(Edges::Planar, None),
            SedonaType::Arrow(DataType::Utf8),
        ];
        if with_options {
            t.push(SedonaType::Arrow(DataType::Utf8));
        }
        t
    }

    /// Invoke RS_ZonalStats on a scalar raster + zone, returning the raw result
    /// so both the value and error paths are testable.
    fn call_stats(
        spec: &RasterSpec,
        wkt: &str,
        stat: &str,
        options: Option<&str>,
    ) -> Result<ColumnarValue> {
        let kernel = RsZonalStats {
            with_options: options.is_some(),
        };
        let mut args = vec![
            ColumnarValue::Scalar(spec.scalar()),
            ColumnarValue::Scalar(ScalarValue::Binary(Some(make_wkb(wkt)))),
            ColumnarValue::Scalar(ScalarValue::Utf8(Some(stat.to_string()))),
        ];
        if let Some(o) = options {
            args.push(ColumnarValue::Scalar(ScalarValue::Utf8(Some(
                o.to_string(),
            ))));
        }
        kernel.invoke_batch(&stats_arg_types(options.is_some()), &args)
    }

    /// Invoke RS_ZonalStatsAll on a scalar raster + zone.
    fn call_all(spec: &RasterSpec, wkt: &str, options: Option<&str>) -> Result<ColumnarValue> {
        let kernel = RsZonalStatsAll {
            with_options: options.is_some(),
        };
        let mut arg_types = vec![RASTER, SedonaType::Wkb(Edges::Planar, None)];
        let mut args = vec![
            ColumnarValue::Scalar(spec.scalar()),
            ColumnarValue::Scalar(ScalarValue::Binary(Some(make_wkb(wkt)))),
        ];
        if let Some(o) = options {
            arg_types.push(SedonaType::Arrow(DataType::Utf8));
            args.push(ColumnarValue::Scalar(ScalarValue::Utf8(Some(
                o.to_string(),
            ))));
        }
        kernel.invoke_batch(&arg_types, &args)
    }

    fn cv_f64(cv: ColumnarValue) -> Option<f64> {
        match cv {
            ColumnarValue::Scalar(ScalarValue::Float64(v)) => v,
            other => panic!("expected a Float64 scalar, got {other:?}"),
        }
    }

    fn cv_struct(cv: ColumnarValue) -> Arc<StructArray> {
        match cv {
            ColumnarValue::Scalar(ScalarValue::Struct(s)) => s,
            other => panic!("expected a struct scalar, got {other:?}"),
        }
    }

    fn f64_field(s: &StructArray, col: usize) -> Option<f64> {
        let c = s.column(col);
        (!c.is_null(0)).then(|| c.as_primitive::<Float64Type>().value(0))
    }

    fn i64_field(s: &StructArray, col: usize) -> Option<i64> {
        let c = s.column(col);
        (!c.is_null(0)).then(|| c.as_primitive::<Int64Type>().value(0))
    }

    #[test]
    fn single_stats_match_hand_computed_values() {
        let spec = small_raster();
        // Selected pixels {1, 2, 5, 6}: exact for the integer-selection stats.
        assert_eq!(
            cv_f64(call_stats(&spec, LEFT_HALF, "count", None).unwrap()),
            Some(4.0)
        );
        assert_eq!(
            cv_f64(call_stats(&spec, LEFT_HALF, "sum", None).unwrap()),
            Some(14.0)
        );
        assert_eq!(
            cv_f64(call_stats(&spec, LEFT_HALF, "mean", None).unwrap()),
            Some(3.5)
        );
        assert_eq!(
            cv_f64(call_stats(&spec, LEFT_HALF, "min", None).unwrap()),
            Some(1.0)
        );
        assert_eq!(
            cv_f64(call_stats(&spec, LEFT_HALF, "max", None).unwrap()),
            Some(6.0)
        );
        assert_eq!(
            cv_f64(call_stats(&spec, LEFT_HALF, "median", None).unwrap()),
            Some(3.5)
        );
        // All four values are unique, so every one is a mode; the tie resolves
        // to the largest (6), matching Sedona Spark.
        assert_eq!(
            cv_f64(call_stats(&spec, LEFT_HALF, "mode", None).unwrap()),
            Some(6.0)
        );

        // Sample (n-1) variance / stddev: float accumulation, so approximate.
        let var = cv_f64(call_stats(&spec, LEFT_HALF, "variance", None).unwrap()).unwrap();
        assert!((var - 17.0 / 3.0).abs() < 1e-9, "variance was {var}");
        let sd = cv_f64(call_stats(&spec, LEFT_HALF, "stddev", None).unwrap()).unwrap();
        assert!(
            (sd - (17.0_f64 / 3.0).sqrt()).abs() < 1e-9,
            "stddev was {sd}"
        );
    }

    #[test]
    fn all_returns_full_struct() {
        let s = cv_struct(call_all(&small_raster(), LEFT_HALF, None).unwrap());
        assert!(!s.is_null(0), "the struct itself is valid");
        assert_eq!(i64_field(&s, COUNT), Some(4));
        assert_eq!(f64_field(&s, SUM), Some(14.0));
        assert_eq!(f64_field(&s, MEAN), Some(3.5));
        assert_eq!(f64_field(&s, MEDIAN), Some(3.5));
        assert_eq!(f64_field(&s, MODE), Some(6.0));
        assert_eq!(f64_field(&s, MIN), Some(1.0));
        assert_eq!(f64_field(&s, MAX), Some(6.0));
        assert!((f64_field(&s, VARIANCE).unwrap() - 17.0 / 3.0).abs() < 1e-9);
        assert!((f64_field(&s, STDDEV).unwrap() - (17.0_f64 / 3.0).sqrt()).abs() < 1e-9);
    }

    #[test]
    fn zone_that_selects_no_pixel_centre_is_count_zero_not_null() {
        // A tiny zone inside the top-left pixel (centre 0.5, 1.5) but not
        // covering that centre: with all_touched off, no pixel is selected. The
        // zone still overlaps the raster extent, so count is 0 (not NULL).
        let tiny = "POLYGON((0.1 1.6, 0.4 1.6, 0.4 1.9, 0.1 1.9, 0.1 1.6))";
        assert_eq!(
            cv_f64(call_stats(&small_raster(), tiny, "count", None).unwrap()),
            Some(0.0)
        );
        assert_eq!(
            cv_f64(call_stats(&small_raster(), tiny, "sum", None).unwrap()),
            None
        );
        assert_eq!(
            cv_f64(call_stats(&small_raster(), tiny, "mean", None).unwrap()),
            None
        );

        let s = cv_struct(call_all(&small_raster(), tiny, None).unwrap());
        assert!(
            !s.is_null(0),
            "an intersecting-but-empty zone is a valid row"
        );
        assert_eq!(i64_field(&s, COUNT), Some(0));
        assert_eq!(f64_field(&s, SUM), None);
        assert_eq!(f64_field(&s, MEAN), None);
    }

    #[test]
    fn all_touched_selects_the_touched_pixel() {
        // The same tiny zone, with all_touched, burns the pixel it lies inside
        // (value 1) even though it misses the centre.
        let tiny = "POLYGON((0.1 1.6, 0.4 1.6, 0.4 1.9, 0.1 1.9, 0.1 1.6))";
        let opts = r#"{"all_touched": true}"#;
        assert_eq!(
            cv_f64(call_stats(&small_raster(), tiny, "count", Some(opts)).unwrap()),
            Some(1.0)
        );
        assert_eq!(
            cv_f64(call_stats(&small_raster(), tiny, "sum", Some(opts)).unwrap()),
            Some(1.0)
        );
    }

    #[test]
    fn no_intersection_is_null_when_lenient_and_errors_when_strict() {
        let far = "POLYGON((100 100, 101 100, 101 101, 100 101, 100 100))";
        // Lenient (default): the whole value is NULL, including count.
        assert_eq!(
            cv_f64(call_stats(&small_raster(), far, "count", None).unwrap()),
            None
        );
        assert!(cv_struct(call_all(&small_raster(), far, None).unwrap()).is_null(0));

        // Strict: both functions error.
        let err = call_stats(&small_raster(), far, "count", Some(r#"{"lenient": false}"#))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("does not intersect"),
            "unexpected error: {err}"
        );
        let err = call_all(&small_raster(), far, Some(r#"{"lenient": false}"#))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("does not intersect"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn nodata_pixels_are_excluded_by_default_and_kept_when_asked() {
        // A 2×2 UInt8 raster [10, 255, 20, 30] with nodata 255, world extent
        // x ∈ [0, 2], y ∈ [0, 2]; the zone covers all four pixels.
        let spec = RasterSpec::d2(2, 2)
            .band_values(&[10u8, 255, 20, 30])
            .nodata(255u8)
            .crs(None)
            .transform([0.0, 1.0, 0.0, 2.0, 0.0, -1.0]);
        // Default excludes the nodata pixel: {10, 20, 30}.
        assert_eq!(
            cv_f64(call_stats(&spec, LEFT_HALF_FULL, "count", None).unwrap()),
            Some(3.0)
        );
        assert_eq!(
            cv_f64(call_stats(&spec, LEFT_HALF_FULL, "sum", None).unwrap()),
            Some(60.0)
        );
        // exclude_nodata=false keeps it: {10, 255, 20, 30}.
        let keep = r#"{"exclude_nodata": false}"#;
        assert_eq!(
            cv_f64(call_stats(&spec, LEFT_HALF_FULL, "count", Some(keep)).unwrap()),
            Some(4.0)
        );
        assert_eq!(
            cv_f64(call_stats(&spec, LEFT_HALF_FULL, "sum", Some(keep)).unwrap()),
            Some(315.0)
        );
    }

    /// A zone covering the whole 2×2 nodata raster above.
    const LEFT_HALF_FULL: &str = "POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))";

    #[test]
    fn multiband_raster_requires_the_band_option() {
        let spec = RasterSpec::d2(2, 2)
            .band_values(&[1u8, 2, 3, 4])
            .band_values(&[10u8, 20, 30, 40])
            .crs(None)
            .transform([0.0, 1.0, 0.0, 2.0, 0.0, -1.0]);
        // Omitting the band on a multiband raster errors rather than defaulting.
        let err = call_stats(&spec, LEFT_HALF_FULL, "sum", None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("2 bands"), "unexpected error: {err}");
        // Naming the band selects it (band 1 sums to 10, band 2 to 100).
        assert_eq!(
            cv_f64(call_stats(&spec, LEFT_HALF_FULL, "sum", Some(r#"{"band": 1}"#)).unwrap()),
            Some(10.0)
        );
        assert_eq!(
            cv_f64(call_stats(&spec, LEFT_HALF_FULL, "sum", Some(r#"{"band": 2}"#)).unwrap()),
            Some(100.0)
        );
        // An out-of-range band errors.
        let err = call_stats(&spec, LEFT_HALF_FULL, "sum", Some(r#"{"band": 3}"#))
            .unwrap_err()
            .to_string();
        assert!(err.contains("out of range"), "unexpected error: {err}");
    }

    #[test]
    fn unknown_statistic_errors() {
        let err = call_stats(&small_raster(), LEFT_HALF, "bogus", None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("unknown statistic"), "unexpected error: {err}");
    }

    #[test]
    fn null_raster_or_zone_yields_null() {
        // A NULL zone geometry propagates to a NULL result.
        let kernel = RsZonalStats {
            with_options: false,
        };
        let result = kernel
            .invoke_batch(
                &stats_arg_types(false),
                &[
                    ColumnarValue::Scalar(small_raster().scalar()),
                    ColumnarValue::Scalar(ScalarValue::Binary(None)),
                    ColumnarValue::Scalar(ScalarValue::Utf8(Some("count".to_string()))),
                ],
            )
            .unwrap();
        assert_eq!(cv_f64(result), None);

        // A NULL statistic name also yields NULL.
        let result = kernel
            .invoke_batch(
                &stats_arg_types(false),
                &[
                    ColumnarValue::Scalar(small_raster().scalar()),
                    ColumnarValue::Scalar(ScalarValue::Binary(Some(make_wkb(LEFT_HALF)))),
                    ColumnarValue::Scalar(ScalarValue::Utf8(None)),
                ],
            )
            .unwrap();
        assert_eq!(cv_f64(result), None);
    }

    #[test]
    fn reprojects_the_zone_into_the_raster_crs() {
        // The raster is EPSG:4326; the zone is supplied in EPSG:3857 (the
        // reprojected LEFT_HALF polygon). Reprojecting it back to the raster CRS
        // must recover the same four-pixel selection.
        let spec = small_raster().crs(Some("EPSG:4326"));
        let crs_4326 = deserialize_crs("EPSG:4326").unwrap().unwrap();
        let crs_3857 = deserialize_crs("EPSG:3857").unwrap().unwrap();
        let wkb_4326 = make_wkb(LEFT_HALF);
        let wkb_3857 = with_global_proj_engine(|engine| {
            crs_transform_wkb(&wkb_4326, crs_4326.as_ref(), crs_3857.as_ref(), engine)
        })
        .unwrap();

        let udf: ScalarUDF = rs_zonal_stats_udf().into();
        let arg_types = vec![
            RASTER,
            SedonaType::Wkb(Edges::Planar, Some(crs_3857)),
            SedonaType::Arrow(DataType::Utf8),
        ];
        let tester = ScalarUdfTester::new(udf, arg_types).with_crs_engine(Arc::new(LazyProjEngine));
        let result = tester
            .invoke_scalar_scalar_scalar(&spec, ScalarValue::Binary(Some(wkb_3857)), "count")
            .unwrap();
        assert_eq!(result, ScalarValue::Float64(Some(4.0)));
    }
}
