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

//! `RS_SummaryStats` and `RS_SummaryStatsAll` — summary statistics of a
//! band's pixel values.
//!
//! ```text
//! RS_SummaryStats(raster, statType)                              -> Double  -- band 1
//! RS_SummaryStats(raster, statType, band)                        -> Double
//! RS_SummaryStats(raster, statType, band, excludeNoDataValue)    -> Double
//! RS_SummaryStatsAll(raster)                                     -> Struct  -- band 1
//! RS_SummaryStatsAll(raster, band)                               -> Struct
//! RS_SummaryStatsAll(raster, band, excludeNoDataValue)           -> Struct
//! ```
//!
//! `statType` is one of `count`, `sum`, `mean`, `stddev`, `min` or `max`
//! (case-insensitive); `stddev` is the population standard deviation.
//! `RS_SummaryStatsAll` returns all six as the fields of one struct.
//! `excludeNoDataValue` defaults to true, leaving nodata pixels out of the
//! statistics. Over no pixels `count` and `sum` are 0 and the others are NaN.

use std::ops::ControlFlow;
use std::sync::Arc;

use arrow_array::builder::Float64Builder;
use arrow_array::{Array, ArrayRef, StructArray};
use arrow_buffer::NullBufferBuilder;
use arrow_schema::{DataType, Field, Fields};
use datafusion_common::cast::{as_boolean_array, as_int32_array, as_string_array};
use datafusion_common::{Result, exec_err};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::traits::{NdBuffer, RasterRef, nodata_bytes_to_f64_lossless};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::executor::RasterExecutor;
use crate::pixel_scan::{NodataMatcher, scan_pixels, spatial_2d_buffer};
use crate::rs_ensure_loaded::NEEDS_PIXELS_METADATA_KEY;
use crate::sampling::{int32_array_arg, resolve_band};

/// `RS_SummaryStats()` scalar UDF — a summary statistic of a band.
pub fn rs_summarystats_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_summarystats",
        vec![
            Arc::new(RsSummaryStats { num_args: 2 }), // (raster, statType)
            Arc::new(RsSummaryStats { num_args: 3 }), // (raster, statType, band)
            Arc::new(RsSummaryStats { num_args: 4 }), // (..., band, excludeNoDataValue)
        ],
        Volatility::Immutable,
    )
    // The kernel reads pixel bytes, so the raster argument must be materialised
    // InDb first; the planner injects RS_EnsureLoaded based on this flag.
    .with_metadata(NEEDS_PIXELS_METADATA_KEY, "true")
}

#[derive(Debug)]
struct RsSummaryStats {
    num_args: usize,
}

/// `RS_SummaryStatsAll()` scalar UDF — every summary statistic of a band.
pub fn rs_summarystatsall_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_summarystatsall",
        vec![
            Arc::new(RsSummaryStatsAll { num_args: 1 }), // (raster)
            Arc::new(RsSummaryStatsAll { num_args: 2 }), // (raster, band)
            Arc::new(RsSummaryStatsAll { num_args: 3 }), // (raster, band, excludeNoDataValue)
        ],
        Volatility::Immutable,
    )
    // The kernel reads pixel bytes, so the raster argument must be materialised
    // InDb first; the planner injects RS_EnsureLoaded based on this flag.
    .with_metadata(NEEDS_PIXELS_METADATA_KEY, "true")
}

/// `RS_SummaryStatsAll`'s struct: Sedona Spark's field names and order, each
/// a non-nullable double (a NULL input nulls the whole struct instead).
fn summary_fields() -> Fields {
    ["count", "sum", "mean", "stddev", "min", "max"]
        .into_iter()
        .map(|name| Field::new(name, DataType::Float64, false))
        .collect()
}

#[derive(Debug)]
struct RsSummaryStatsAll {
    num_args: usize,
}

impl SedonaScalarKernel for RsSummaryStatsAll {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matchers = [
            ArgMatcher::is_raster(),
            ArgMatcher::is_integer(),
            ArgMatcher::is_boolean(),
        ];
        let matcher = ArgMatcher::new(
            matchers[..self.num_args].to_vec(),
            SedonaType::Arrow(DataType::Struct(summary_fields())),
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let num_iterations = executor.num_iterations();
        let mut builders: Vec<Float64Builder> = (0..6)
            .map(|_| Float64Builder::with_capacity(num_iterations))
            .collect();
        let mut validity = NullBufferBuilder::new(num_iterations);

        let band_arr = args
            .get(1)
            .map(|arg| int32_array_arg(arg, num_iterations))
            .transpose()?;
        let exclude_arr = args
            .get(2)
            .map(|arg| {
                arg.cast_to(&DataType::Boolean, None)?
                    .into_array(num_iterations)
            })
            .transpose()?;
        let band = band_arr.as_ref().map(|a| as_int32_array(a)).transpose()?;
        let exclude = exclude_arr
            .as_ref()
            .map(|a| as_boolean_array(a))
            .transpose()?;

        executor.execute_raster_void(|i, raster_opt| {
            let stats = match raster_opt {
                Some(raster)
                    if !band.is_some_and(|b| b.is_null(i))
                        && !exclude.is_some_and(|e| e.is_null(i)) =>
                {
                    // Clamp a negative band to 0 so resolve_band rejects it as
                    // not 1-based rather than wrapping it into a huge usize.
                    let band_num = band.map_or(1, |b| b.value(i).max(0) as usize);
                    let exclude_nodata = exclude.is_none_or(|e| e.value(i));
                    Some(with_band_values(
                        "RS_SummaryStatsAll",
                        raster,
                        band_num,
                        exclude_nodata,
                        summarize,
                    )?)
                }
                _ => None,
            };
            match stats {
                Some(stats) => {
                    validity.append_non_null();
                    for (builder, stat) in builders.iter_mut().zip(stats) {
                        builder.append_value(stat);
                    }
                }
                None => {
                    validity.append_null();
                    // The fields are non-nullable, so a null row carries a
                    // placeholder in every child under a null struct slot.
                    for builder in builders.iter_mut() {
                        builder.append_value(0.0);
                    }
                }
            }
            Ok(())
        })?;

        let arrays: Vec<ArrayRef> = builders
            .iter_mut()
            .map(|builder| Arc::new(builder.finish()) as ArrayRef)
            .collect();
        let struct_array = StructArray::try_new(summary_fields(), arrays, validity.finish())?;
        executor.finish(Arc::new(struct_array))
    }
}

impl SedonaScalarKernel for RsSummaryStats {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matchers = [
            ArgMatcher::is_raster(),
            ArgMatcher::is_string(),
            ArgMatcher::is_integer(),
            ArgMatcher::is_boolean(),
        ];
        let matcher = ArgMatcher::new(
            matchers[..self.num_args].to_vec(),
            SedonaType::Arrow(DataType::Float64),
        );
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

        let arg_array = |index: usize, data_type: &DataType| -> Result<Option<ArrayRef>> {
            args.get(index)
                .map(|arg| arg.cast_to(data_type, None)?.into_array(num_iterations))
                .transpose()
        };
        let stat_arr = arg_array(1, &DataType::Utf8)?.unwrap();
        let band_arr = args
            .get(2)
            .map(|arg| int32_array_arg(arg, num_iterations))
            .transpose()?;
        let exclude_arr = arg_array(3, &DataType::Boolean)?;
        let stat = as_string_array(&stat_arr)?;
        let band = band_arr.as_ref().map(|a| as_int32_array(a)).transpose()?;
        let exclude = exclude_arr
            .as_ref()
            .map(|a| as_boolean_array(a))
            .transpose()?;

        executor.execute_raster_void(|i, raster_opt| {
            let Some(raster) = raster_opt else {
                builder.append_null();
                return Ok(());
            };
            if stat.is_null(i)
                || band.is_some_and(|b| b.is_null(i))
                || exclude.is_some_and(|e| e.is_null(i))
            {
                builder.append_null();
                return Ok(());
            }
            let stat_type = StatType::parse(stat.value(i))?;
            // Clamp a negative band to 0 so resolve_band rejects it as not
            // 1-based rather than wrapping it into a huge usize.
            let band_num = band.map_or(1, |b| b.value(i).max(0) as usize);
            let exclude_nodata = exclude.is_none_or(|e| e.value(i));
            builder.append_value(summary_stat(raster, band_num, exclude_nodata, stat_type)?);
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

const FUNC: &str = "RS_SummaryStats";

/// `stat_type` over the 1-based band `band_num`, leaving out nodata pixels
/// when `exclude_nodata` is set.
fn summary_stat(
    raster: &dyn RasterRef,
    band_num: usize,
    exclude_nodata: bool,
    stat_type: StatType,
) -> Result<f64> {
    with_band_values(FUNC, raster, band_num, exclude_nodata, |values| {
        stat_type.compute(values)
    })
}

/// Run `f` over the values of the 1-based band `band_num`, leaving out nodata
/// pixels when `exclude_nodata` is set. `func` names the calling UDF for the
/// error messages.
fn with_band_values<T>(
    func: &'static str,
    raster: &dyn RasterRef,
    band_num: usize,
    exclude_nodata: bool,
    f: impl FnOnce(&BandValues) -> Result<T>,
) -> Result<T> {
    let band = resolve_band(func, raster, band_num)?;
    let values = BandValues {
        func,
        buffer: spatial_2d_buffer(func, band.as_ref())?,
        nodata: if exclude_nodata {
            NodataMatcher::for_band(func, band.as_ref())?
        } else {
            None
        },
    };
    f(&values)
}

/// The values of one band, in row-major order.
///
/// A statistic reads them by re-scanning the band's pixels once per pass rather
/// than collecting them: the band bytes are already in memory, and a copy would
/// be 8 bytes per pixel (8x a UInt8 band) outside the query's memory accounting.
struct BandValues<'a> {
    func: &'static str,
    buffer: NdBuffer<'a>,
    /// Pixels this matches are left out; `None` keeps every pixel.
    nodata: Option<NodataMatcher<'a>>,
}

impl BandValues<'_> {
    fn is_nodata(&self, pixel: &[u8]) -> bool {
        self.nodata
            .as_ref()
            .is_some_and(|nodata| nodata.matches(pixel))
    }

    /// The number of values, counted without decoding any of them.
    fn count(&self) -> Result<usize> {
        let mut count = 0;
        scan_pixels(self.func, &self.buffer, |_, _, pixel| {
            if !self.is_nodata(pixel) {
                count += 1;
            }
            ControlFlow::Continue(())
        })?;
        Ok(count)
    }

    /// Visit each value in order.
    fn for_each(&self, mut visit: impl FnMut(f64)) -> Result<()> {
        let data_type = self.buffer.data_type;
        let mut inexact = None;
        scan_pixels(self.func, &self.buffer, |col, row, pixel| {
            if self.is_nodata(pixel) {
                return ControlFlow::Continue(());
            }
            // Decoding fails only for a 64-bit integer beyond 2^53, which an f64
            // statistic cannot represent exactly; error rather than round it.
            match nodata_bytes_to_f64_lossless(pixel, &data_type) {
                Ok(value) => {
                    visit(value);
                    ControlFlow::Continue(())
                }
                Err(_) => {
                    inexact = Some((col, row));
                    ControlFlow::Break(())
                }
            }
        })?;
        match inexact {
            Some((col, row)) => exec_err!(
                "{}: the {data_type:?} pixel at column {col}, row {row} exceeds 2^53 in \
                 magnitude, so a Float64 statistic cannot represent it exactly",
                self.func
            ),
            None => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StatType {
    Count,
    Sum,
    Mean,
    StdDev,
    Min,
    Max,
}

impl StatType {
    fn parse(name: &str) -> Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "count" => Ok(Self::Count),
            "sum" => Ok(Self::Sum),
            "mean" => Ok(Self::Mean),
            "stddev" => Ok(Self::StdDev),
            "min" => Ok(Self::Min),
            "max" => Ok(Self::Max),
            _ => exec_err!(
                "{FUNC}: invalid statType '{name}'; expected one of 'count', 'sum', 'mean', \
                 'stddev', 'min', 'max'"
            ),
        }
    }

    /// The statistic over `values`.
    ///
    /// Each statistic repeats Sedona Spark's arithmetic operation for operation
    /// — Apache Commons Math's `DescriptiveStatistics` and a population
    /// `StandardDeviation` — so the two engines agree to the last bit rather
    /// than merely to within rounding: summation order, the mean's correction
    /// pass, and the variance's `accum2` term all change the low bits.
    fn compute(self, values: &BandValues) -> Result<f64> {
        match self {
            Self::Count => Ok(values.count()? as f64),
            Self::Sum => Ok(first_pass(values)?.sum),
            Self::Mean => mean(values, &first_pass(values)?),
            Self::StdDev => Ok(summarize(values)?[3]),
            Self::Min => Ok(first_pass(values)?.min),
            Self::Max => Ok(first_pass(values)?.max),
        }
    }
}

/// All six statistics, in Sedona Spark's order: count, sum, mean, stddev, min
/// and max. Three passes over the band: the first pass, the mean's correction
/// pass, and the variance pass.
fn summarize(values: &BandValues) -> Result<[f64; 6]> {
    let first = first_pass(values)?;
    let mean = mean(values, &first)?;
    let stddev = population_stddev(values, first.count, mean)?;
    Ok([
        first.count as f64,
        first.sum,
        mean,
        stddev,
        first.min,
        first.max,
    ])
}

/// What one scan of the values yields: their number, Commons Math `Sum` (a
/// left-to-right sum, 0 over no values), and Commons Math `Min` and `Max`
/// (NaN over no values).
struct FirstPass {
    count: usize,
    sum: f64,
    min: f64,
    max: f64,
}

fn first_pass(values: &BandValues) -> Result<FirstPass> {
    let (mut count, mut sum) = (0usize, 0.0);
    let (mut min, mut max) = (None, None);
    values.for_each(|v| {
        count += 1;
        sum += v;
        min = Some(keep_extreme(min, v, |kept, v| kept < v));
        max = Some(keep_extreme(max, v, |kept, v| kept > v));
    })?;
    Ok(FirstPass {
        count,
        sum,
        min: min.unwrap_or(f64::NAN),
        max: max.unwrap_or(f64::NAN),
    })
}

/// One step of Commons Math `Min`/`Max`: start from the first value and keep
/// it over each later non-NaN value unless `keep(kept, value)` fails, so NaN
/// values are skipped (unless every value is NaN).
fn keep_extreme(kept: Option<f64>, v: f64, keep: impl Fn(f64, f64) -> bool) -> f64 {
    match kept {
        None => v,
        Some(k) if v.is_nan() || keep(k, v) => k,
        Some(_) => v,
    }
}

/// Commons Math `Mean`: the definitional mean plus a second-pass correction
/// for the rounding error of the first; NaN over no values.
fn mean(values: &BandValues, first: &FirstPass) -> Result<f64> {
    if first.count == 0 {
        return Ok(f64::NAN);
    }
    let n = first.count as f64;
    let xbar = first.sum / n;
    let mut correction = 0.0;
    values.for_each(|v| correction += v - xbar)?;
    Ok(xbar + correction / n)
}

/// Commons Math `StandardDeviation` without bias correction: the square root
/// of the corrected two-pass population variance around the Commons `mean`
/// of `count` values; 0 over one value and NaN over none.
fn population_stddev(values: &BandValues, count: usize, mean: f64) -> Result<f64> {
    let variance = match count {
        0 => f64::NAN,
        1 => 0.0,
        len => {
            let (mut accum, mut accum2) = (0.0, 0.0);
            values.for_each(|v| {
                let dev = v - mean;
                accum += dev * dev;
                accum2 += dev;
            })?;
            let len = len as f64;
            (accum - (accum2 * accum2 / len)) / len
        }
    };
    Ok(variance.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{BooleanArray, Float64Array, Int32Array, StringArray};
    use datafusion_expr::ScalarUDF;
    use sedona_raster::array::RasterStructArray;
    use sedona_schema::datatypes::RASTER;
    use sedona_testing::raster_spec::{RasterSpec, raster_array};
    use sedona_testing::testers::ScalarUdfTester;

    fn stat(spec: RasterSpec, stat_type: &str, band: usize, exclude: bool) -> Result<f64> {
        let array = spec.build();
        let rasters = RasterStructArray::try_new(&array).unwrap();
        summary_stat(
            &rasters.get(0).unwrap(),
            band,
            exclude,
            StatType::parse(stat_type)?,
        )
    }

    /// Six values near 1e8 with fractional parts, row-major. Commons Math's
    /// arithmetic and each shortcut a port might take round differently here,
    /// so the anchors below pin every step (see
    /// `fixture_separates_the_shortcuts`).
    const CANCELLATION: [f64; 6] = [
        100000001.3,
        100000000.2,
        100000002.5,
        100000001.1,
        100000001.1,
        100000000.7,
    ];

    fn float_band() -> RasterSpec {
        RasterSpec::d2(3, 2).band_values(&CANCELLATION)
    }

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = rs_summarystats_udf().into();
        assert_eq!(udf.name(), "rs_summarystats");
        assert_eq!(
            rs_summarystats_udf()
                .metadata()
                .get(NEEDS_PIXELS_METADATA_KEY)
                .map(String::as_str),
            Some("true")
        );
    }

    #[test]
    fn statistics_match_sedona_spark_bit_for_bit() {
        // Anchors are Sedona Spark 1.9.1's output for the same six pixels
        // (RS_SummaryStats over a float64 GeoTIFF), which Commons Math computes
        // with the operation order replicated above.
        let cases = [
            ("count", 6.0),
            ("sum", 600000006.9000001),
            ("mean", 100000001.15),
            ("stddev", 0.7017834414254126),
            ("min", 100000000.2),
            ("max", 100000002.5),
        ];
        for (stat_type, expected) in cases {
            assert_eq!(
                stat(float_band(), stat_type, 1, true).unwrap(),
                expected,
                "{stat_type}"
            );
        }
    }

    /// Guards the fixture: every shortcut from Commons Math's arithmetic lands
    /// on a different value than the anchors above, so the anchors fail if the
    /// kernel ever takes one.
    #[test]
    fn fixture_separates_the_shortcuts() {
        let v = CANCELLATION;
        let n = v.len() as f64;
        let rev_sum = v.iter().rev().fold(0.0, |acc, x| acc + x);
        let col_major_sum = [0, 3, 1, 4, 2, 5].iter().fold(0.0, |acc, &i| acc + v[i]);
        assert_ne!(rev_sum, 600000006.9000001, "summation order");
        assert_ne!(col_major_sum, 600000006.9000001, "summation order");

        let naive_mean = v.iter().sum::<f64>() / n;
        assert_ne!(naive_mean, 100000001.15, "mean correction pass");

        let variance = |mean: f64, with_accum2: bool| {
            let accum: f64 = v.iter().map(|x| (x - mean) * (x - mean)).sum();
            let accum2: f64 = v.iter().map(|x| x - mean).sum();
            if with_accum2 {
                (accum - accum2 * accum2 / n) / n
            } else {
                accum / n
            }
        };
        let anchor = 0.7017834414254126;
        assert_ne!(variance(100000001.15, false).sqrt(), anchor, "accum2 term");
        assert_ne!(
            variance(naive_mean, true).sqrt(),
            anchor,
            "variance around naive mean"
        );
        assert_ne!(
            variance(naive_mean, false).sqrt(),
            anchor,
            "naive two-pass stddev"
        );
    }

    #[test]
    fn stat_type_is_case_insensitive() {
        assert_eq!(stat(float_band(), "MeAn", 1, true).unwrap(), 100000001.15);
    }

    #[test]
    fn invalid_stat_type_errors() {
        let err = stat(float_band(), "median", 1, true)
            .unwrap_err()
            .to_string();
        assert!(err.contains("invalid statType 'median'"), "{err}");
    }

    #[test]
    fn nodata_is_excluded_unless_asked_not_to() {
        let spec = RasterSpec::d2(4, 1)
            .band_values(&[1u8, 2, 3, 250])
            .nodata(250u8);
        assert_eq!(stat(spec.clone(), "count", 1, true).unwrap(), 3.0);
        assert_eq!(stat(spec.clone(), "max", 1, true).unwrap(), 3.0);
        assert_eq!(stat(spec.clone(), "count", 1, false).unwrap(), 4.0);
        assert_eq!(stat(spec, "max", 1, false).unwrap(), 250.0);
    }

    #[test]
    fn nan_nodata_excludes_nan_pixels() {
        let spec = RasterSpec::d2(3, 1)
            .band_values(&[f32::NAN, 1.5, f32::NAN])
            .nodata(f32::NAN);
        assert_eq!(stat(spec.clone(), "count", 1, true).unwrap(), 1.0);
        assert_eq!(stat(spec, "sum", 1, true).unwrap(), 1.5);
    }

    #[test]
    fn no_values_gives_zero_count_and_sum_and_nan_otherwise() {
        let spec = RasterSpec::d2(2, 1).band_values(&[7u8, 7]).nodata(7u8);
        assert_eq!(stat(spec.clone(), "count", 1, true).unwrap(), 0.0);
        assert_eq!(stat(spec.clone(), "sum", 1, true).unwrap(), 0.0);
        for stat_type in ["mean", "stddev", "min", "max"] {
            assert!(
                stat(spec.clone(), stat_type, 1, true).unwrap().is_nan(),
                "{stat_type}"
            );
        }
    }

    #[test]
    fn single_value_has_zero_stddev() {
        let spec = RasterSpec::d2(1, 1).band_values(&[4.5f64]);
        assert_eq!(stat(spec, "stddev", 1, true).unwrap(), 0.0);
    }

    #[test]
    fn min_and_max_skip_nan_values() {
        let spec = RasterSpec::d2(3, 1).band_values(&[f64::NAN, 2.0, -1.0]);
        assert_eq!(stat(spec.clone(), "min", 1, true).unwrap(), -1.0);
        assert_eq!(stat(spec.clone(), "max", 1, true).unwrap(), 2.0);
        // A NaN in the data (not declared nodata) still poisons the sum.
        assert!(stat(spec, "sum", 1, true).unwrap().is_nan());
    }

    #[test]
    fn band_out_of_range_errors() {
        let err = stat(float_band(), "mean", 2, true).unwrap_err().to_string();
        assert!(err.contains("RS_SummaryStats"), "{err}");
        let err = stat(float_band(), "mean", 0, true).unwrap_err().to_string();
        assert!(err.contains("1-based"), "{err}");
    }

    #[test]
    fn inexact_64_bit_pixel_errors() {
        let spec = RasterSpec::d2(2, 1).band_values(&[0u64, u64::MAX]);
        let err = stat(spec, "sum", 1, true).unwrap_err().to_string();
        assert!(err.contains("pixel at column 1, row 0"), "{err}");
        assert!(err.contains("2^53"), "{err}");
    }

    #[test]
    fn count_needs_no_values() {
        // Counting decodes nothing, so a 64-bit sentinel fill (kept here) that
        // no f64 statistic could represent still counts.
        let spec = RasterSpec::d2(2, 1)
            .band_values(&[1u64, u64::MAX])
            .nodata(u64::MAX);
        assert_eq!(stat(spec.clone(), "count", 1, true).unwrap(), 1.0);
        assert_eq!(stat(spec, "count", 1, false).unwrap(), 2.0);
    }

    #[test]
    fn udf_invoke_every_arity() {
        let two_bands = || {
            RasterSpec::d2(2, 1)
                .band_values(&[1u8, 3])
                .band_values(&[10u8, 250])
                .nodata(250u8)
        };
        let rasters = raster_array([Some(two_bands()), Some(two_bands()), None]);
        let utf8 = SedonaType::Arrow(DataType::Utf8);
        let int32 = SedonaType::Arrow(DataType::Int32);
        let boolean = SedonaType::Arrow(DataType::Boolean);
        let stats = Arc::new(StringArray::from(vec![Some("sum"), None, Some("sum")]));
        let udf: ScalarUDF = rs_summarystats_udf().into();

        // Band 1 by default; a NULL statType or raster is NULL.
        let tester = ScalarUdfTester::new(udf.clone(), vec![RASTER, utf8.clone()]);
        let result = tester
            .invoke_arrays(vec![Arc::new(rasters.clone()), stats.clone()])
            .unwrap();
        assert_eq!(
            result.as_any().downcast_ref::<Float64Array>().unwrap(),
            &Float64Array::from(vec![Some(4.0), None, None])
        );

        // Band 2 leaves out its nodata pixel; a NULL band is NULL.
        let tester = ScalarUdfTester::new(udf.clone(), vec![RASTER, utf8.clone(), int32.clone()]);
        let result = tester
            .invoke_arrays(vec![
                Arc::new(raster_array([
                    Some(two_bands()),
                    Some(two_bands()),
                    Some(two_bands()),
                ])),
                stats.clone(),
                Arc::new(Int32Array::from(vec![Some(2), Some(2), None])),
            ])
            .unwrap();
        assert_eq!(
            result.as_any().downcast_ref::<Float64Array>().unwrap(),
            &Float64Array::from(vec![Some(10.0), None, None])
        );

        // ...unless told to keep it; a NULL flag is NULL.
        let tester = ScalarUdfTester::new(udf, vec![RASTER, utf8, int32, boolean]);
        let result = tester
            .invoke_arrays(vec![
                Arc::new(raster_array([Some(two_bands()), Some(two_bands())])),
                Arc::new(StringArray::from(vec!["sum", "sum"])),
                Arc::new(Int32Array::from(vec![2, 2])),
                Arc::new(BooleanArray::from(vec![Some(false), None])),
            ])
            .unwrap();
        assert_eq!(
            result.as_any().downcast_ref::<Float64Array>().unwrap(),
            &Float64Array::from(vec![Some(260.0), None])
        );
    }

    #[test]
    fn return_type_is_float64() {
        let kernel = RsSummaryStats { num_args: 2 };
        assert_eq!(
            kernel
                .return_type(&[RASTER, SedonaType::Arrow(DataType::Utf8)])
                .unwrap(),
            Some(SedonaType::Arrow(DataType::Float64))
        );
    }

    /// `RS_SummaryStatsAll` over the first row of `spec`.
    fn all(spec: RasterSpec, band: usize, exclude: bool) -> Result<[f64; 6]> {
        let array = spec.build();
        let rasters = RasterStructArray::try_new(&array).unwrap();
        with_band_values(
            "RS_SummaryStatsAll",
            &rasters.get(0).unwrap(),
            band,
            exclude,
            summarize,
        )
    }

    #[test]
    fn all_matches_sedona_spark_bit_for_bit() {
        // The same Sedona Spark 1.9.1 anchors as the single statistics, in
        // RS_SummaryStatsAll's field order.
        assert_eq!(
            all(float_band(), 1, true).unwrap(),
            [
                6.0,
                600000006.9000001,
                100000001.15,
                0.7017834414254126,
                100000000.2,
                100000002.5,
            ]
        );
    }

    #[test]
    fn all_agrees_with_each_single_statistic() {
        // One struct and six separate calls go through the same arithmetic.
        let spec = RasterSpec::d2(3, 2)
            .band_values(&[f64::NAN, 2.5, -1.0, 7.0, 250.0, 0.125])
            .nodata(250f64);
        for exclude in [true, false] {
            let got = all(spec.clone(), 1, exclude).unwrap();
            for (i, stat_type) in ["count", "sum", "mean", "stddev", "min", "max"]
                .into_iter()
                .enumerate()
            {
                let single = stat(spec.clone(), stat_type, 1, exclude).unwrap();
                assert_eq!(
                    got[i].to_bits(),
                    single.to_bits(),
                    "{stat_type}, exclude={exclude}"
                );
            }
        }
    }

    #[test]
    fn all_over_no_values() {
        let spec = RasterSpec::d2(2, 1).band_values(&[7u8, 7]).nodata(7u8);
        let [count, sum, rest @ ..] = all(spec, 1, true).unwrap();
        assert_eq!((count, sum), (0.0, 0.0));
        assert!(rest.iter().all(|v| v.is_nan()), "{rest:?}");
    }

    #[test]
    fn all_errors_name_the_function() {
        let err = all(float_band(), 2, true).unwrap_err().to_string();
        assert!(err.contains("RS_SummaryStatsAll"), "{err}");
    }

    #[test]
    fn all_udf_every_arity() {
        let two_bands = || {
            RasterSpec::d2(2, 1)
                .band_values(&[1u8, 3])
                .band_values(&[10u8, 250])
                .nodata(250u8)
        };
        let int32 = SedonaType::Arrow(DataType::Int32);
        let boolean = SedonaType::Arrow(DataType::Boolean);
        let udf: ScalarUDF = rs_summarystatsall_udf().into();
        let sums = |result: &ArrayRef| {
            let result = result.as_any().downcast_ref::<StructArray>().unwrap();
            let sum = result
                .column_by_name("sum")
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            (0..result.len())
                .map(|i| (!result.is_null(i)).then(|| sum.value(i)))
                .collect::<Vec<_>>()
        };

        // Band 1 by default; a NULL raster is a NULL struct.
        let tester = ScalarUdfTester::new(udf.clone(), vec![RASTER]);
        tester.assert_return_type(DataType::Struct(summary_fields()));
        let result = tester
            .invoke_array(Arc::new(raster_array([Some(two_bands()), None])))
            .unwrap();
        assert_eq!(sums(&result), vec![Some(4.0), None]);

        // Band 2 leaves out its nodata pixel; a NULL band is a NULL struct.
        let tester = ScalarUdfTester::new(udf.clone(), vec![RASTER, int32.clone()]);
        let result = tester
            .invoke_arrays(vec![
                Arc::new(raster_array([Some(two_bands()), Some(two_bands())])),
                Arc::new(Int32Array::from(vec![Some(2), None])),
            ])
            .unwrap();
        assert_eq!(sums(&result), vec![Some(10.0), None]);

        // ...unless told to keep it; a NULL flag is a NULL struct.
        let tester = ScalarUdfTester::new(udf, vec![RASTER, int32, boolean]);
        let result = tester
            .invoke_arrays(vec![
                Arc::new(raster_array([Some(two_bands()), Some(two_bands())])),
                Arc::new(Int32Array::from(vec![2, 2])),
                Arc::new(BooleanArray::from(vec![Some(false), None])),
            ])
            .unwrap();
        assert_eq!(sums(&result), vec![Some(260.0), None]);
    }
}
