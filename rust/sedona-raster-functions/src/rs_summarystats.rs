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

//! `RS_SummaryStats` — one summary statistic of a band's pixel values.
//!
//! ```text
//! RS_SummaryStats(raster, statType)                              -> Double  -- band 1
//! RS_SummaryStats(raster, statType, band)                        -> Double
//! RS_SummaryStats(raster, statType, band, excludeNoDataValue)    -> Double
//! ```
//!
//! `statType` is one of `count`, `sum`, `mean`, `stddev`, `min` or `max`
//! (case-insensitive); `stddev` is the population standard deviation.
//! `excludeNoDataValue` defaults to true, leaving nodata pixels out of the
//! statistic. Over no pixels `count` and `sum` are 0 and the others are NaN.

use std::ops::ControlFlow;
use std::sync::Arc;

use arrow_array::builder::Float64Builder;
use arrow_array::{Array, ArrayRef};
use arrow_schema::DataType;
use datafusion_common::cast::{as_boolean_array, as_int32_array, as_string_array};
use datafusion_common::{Result, exec_err};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::error::RasterResultExt;
use sedona_raster::traits::{RasterRef, nodata_bytes_to_f64_lossless};
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

        // Scratch buffer for a band's values, reused across rows.
        let mut values = Vec::new();
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
            band_values(raster, band_num, exclude_nodata, &mut values)?;
            builder.append_value(stat_type.compute(&values));
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

/// Collect the values of the 1-based band `band_num` into `values` in
/// row-major order, leaving out nodata pixels when `exclude_nodata` is set.
fn band_values(
    raster: &dyn RasterRef,
    band_num: usize,
    exclude_nodata: bool,
    values: &mut Vec<f64>,
) -> Result<()> {
    const FUNC: &str = "RS_SummaryStats";
    values.clear();
    let band = resolve_band(FUNC, raster, band_num)?;
    let buffer = spatial_2d_buffer(FUNC, band.as_ref())?;
    let nodata = if exclude_nodata {
        NodataMatcher::for_band(FUNC, band.as_ref())?
    } else {
        None
    };
    let mut decode_err = None;
    scan_pixels(FUNC, &buffer, |_, _, pixel| {
        if nodata.as_ref().is_some_and(|nodata| nodata.matches(pixel)) {
            return ControlFlow::Continue(());
        }
        // Errors, rather than rounding, on a 64-bit integer pixel beyond 2^53,
        // which an f64 statistic cannot represent exactly.
        match nodata_bytes_to_f64_lossless(pixel, &buffer.data_type).context(FUNC) {
            Ok(value) => {
                values.push(value);
                ControlFlow::Continue(())
            }
            Err(e) => {
                decode_err = Some(e.into());
                ControlFlow::Break(())
            }
        }
    })?;
    decode_err.map_or(Ok(()), Err)
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
                "RS_SummaryStats: invalid statType '{name}'; expected one of 'count', 'sum', \
                 'mean', 'stddev', 'min', 'max'"
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
    fn compute(self, values: &[f64]) -> f64 {
        match self {
            Self::Count => values.len() as f64,
            Self::Sum => sum(values),
            Self::Mean => mean(values),
            Self::StdDev => population_stddev(values, mean(values)),
            Self::Min => extreme(values, |kept, v| kept < v),
            Self::Max => extreme(values, |kept, v| kept > v),
        }
    }
}

/// Commons Math `Sum`: a left-to-right sum, 0 over no values.
fn sum(values: &[f64]) -> f64 {
    values.iter().fold(0.0, |acc, v| acc + v)
}

/// Commons Math `Mean`: the definitional mean plus a second-pass correction
/// for the rounding error of the first; NaN over no values.
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let n = values.len() as f64;
    let xbar = sum(values) / n;
    let correction = values.iter().fold(0.0, |acc, v| acc + (v - xbar));
    xbar + correction / n
}

/// Commons Math `StandardDeviation` without bias correction: the square root
/// of the corrected two-pass population variance around `mean`; 0 over one
/// value and NaN over none.
fn population_stddev(values: &[f64], mean: f64) -> f64 {
    let variance = match values.len() {
        0 => f64::NAN,
        1 => 0.0,
        len => {
            let (accum, accum2) = values.iter().fold((0.0, 0.0), |(accum, accum2), v| {
                let dev = v - mean;
                (accum + dev * dev, accum2 + dev)
            });
            let len = len as f64;
            (accum - (accum2 * accum2 / len)) / len
        }
    };
    variance.sqrt()
}

/// Commons Math `Min`/`Max`: starts from the first value and keeps it over
/// each later non-NaN value unless `keep(kept, value)` fails, so NaN values are
/// skipped (unless every value is NaN); NaN over no values.
fn extreme(values: &[f64], keep: impl Fn(f64, f64) -> bool) -> f64 {
    let Some(&first) = values.first() else {
        return f64::NAN;
    };
    values.iter().fold(
        first,
        |kept, &v| {
            if v.is_nan() || keep(kept, v) { kept } else { v }
        },
    )
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
        let mut values = Vec::new();
        band_values(&rasters.get(0).unwrap(), band, exclude, &mut values)?;
        Ok(StatType::parse(stat_type)?.compute(&values))
    }

    /// A 3x2 float band whose statistics exercise every rounding step: a large
    /// magnitude beside small fractions, so the naive formulas lose low bits.
    fn float_band() -> RasterSpec {
        RasterSpec::d2(3, 2).band_values(&[0.1f64, 0.2, 0.3, 1e10, -3.5, 7.25])
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
            ("sum", 1.000000000435e10),
            ("mean", 1666666667.3916667),
            ("stddev", 3726779962.17542),
            ("min", -3.5),
            ("max", 1e10),
        ];
        for (stat_type, expected) in cases {
            assert_eq!(
                stat(float_band(), stat_type, 1, true).unwrap(),
                expected,
                "{stat_type}"
            );
        }
    }

    #[test]
    fn stat_type_is_case_insensitive() {
        assert_eq!(
            stat(float_band(), "MeAn", 1, true).unwrap(),
            1666666667.3916667
        );
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
        let spec = RasterSpec::d2(1, 1).band_values(&[u64::MAX]);
        let err = stat(spec, "sum", 1, true).unwrap_err().to_string();
        assert!(err.contains("2^53"), "{err}");
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

        // Band 2 leaves out its nodata pixel.
        let tester = ScalarUdfTester::new(udf.clone(), vec![RASTER, utf8.clone(), int32.clone()]);
        let result = tester
            .invoke_arrays(vec![
                Arc::new(rasters.clone()),
                stats.clone(),
                Arc::new(Int32Array::from(vec![2, 2, 2])),
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
}
