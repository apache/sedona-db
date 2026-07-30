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
use std::sync::Arc;

use datafusion_common::{error::Result, exec_datafusion_err};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::builder::RasterBuilder;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::executor::RasterExecutor;
use crate::raster_min_max::{append_combined_band, BandIndices, MinMaxOp};

/// RS_Min() scalar UDF implementation
///
/// **Experimental.** Computes the row-wise pixel-wise minimum of one band of
/// two rasters, e.g. `LEAST` for rasters. See
/// [rs_min_agg_udf](crate::rs_min_max_agg::rs_min_agg_udf) for the
/// equivalent aggregate across rows.
pub fn rs_min_udf() -> SedonaScalarUDF {
    build_min_max_udf("rs_min", MinMaxOp::Min)
}

/// RS_Max() scalar UDF implementation
///
/// **Experimental.** Computes the row-wise pixel-wise maximum of one band of
/// two rasters, e.g. `GREATEST` for rasters. See
/// [rs_max_agg_udf](crate::rs_min_max_agg::rs_max_agg_udf) for the
/// equivalent aggregate across rows.
pub fn rs_max_udf() -> SedonaScalarUDF {
    build_min_max_udf("rs_max", MinMaxOp::Max)
}

fn build_min_max_udf(name: &str, op: MinMaxOp) -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        name,
        vec![
            Arc::new(RsMinMax::new(op)),
            Arc::new(RsMinMaxWithBand::new(op)),
        ],
        Volatility::Immutable,
    )
}

fn invoke_min_max(
    op: MinMaxOp,
    arg_types: &[SedonaType],
    args: &[ColumnarValue],
) -> Result<ColumnarValue> {
    let executor = RasterExecutor::new(arg_types, args);
    let band_indices = BandIndices::resolve(args.get(2), executor.num_iterations())?;
    let mut builder = RasterBuilder::new(executor.num_iterations());

    executor.execute_raster_raster_void(|row, a_opt, b_opt| match (a_opt, b_opt) {
        (Some(a), Some(b)) => append_combined_band(op, &mut builder, a, b, band_indices.get(row)),
        _ => builder
            .append_null()
            .map_err(|e| exec_datafusion_err!("{e}")),
    })?;

    executor.finish(Arc::new(
        builder.finish().map_err(|e| exec_datafusion_err!("{e}"))?,
    ))
}

/// `RS_Min(raster, raster)`/`RS_Max(raster, raster)` — operates on band 1.
#[derive(Debug)]
struct RsMinMax {
    op: MinMaxOp,
    matcher: ArgMatcher,
}

impl RsMinMax {
    fn new(op: MinMaxOp) -> Self {
        Self {
            op,
            matcher: ArgMatcher::new(
                vec![ArgMatcher::is_raster(), ArgMatcher::is_raster()],
                SedonaType::Raster,
            ),
        }
    }
}

impl SedonaScalarKernel for RsMinMax {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        self.matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        invoke_min_max(self.op, arg_types, args)
    }
}

/// `RS_Min(raster, raster, band_index)`/`RS_Max(raster, raster, band_index)` —
/// 1-based, same convention as `RS_BandPixelType`.
#[derive(Debug)]
struct RsMinMaxWithBand {
    op: MinMaxOp,
    matcher: ArgMatcher,
}

impl RsMinMaxWithBand {
    fn new(op: MinMaxOp) -> Self {
        Self {
            op,
            matcher: ArgMatcher::new(
                vec![
                    ArgMatcher::is_raster(),
                    ArgMatcher::is_raster(),
                    ArgMatcher::is_integer(),
                ],
                SedonaType::Raster,
            ),
        }
    }
}

impl SedonaScalarKernel for RsMinMaxWithBand {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        self.matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        invoke_min_max(self.op, arg_types, args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::ArrayRef;
    use datafusion_expr::ScalarUDF;
    use sedona_testing::raster_spec::{assert_rasters_equal, raster_array, RasterSpec};

    fn invoke(kernel: &RsMinMax, a: RasterSpec, b: RasterSpec) -> ColumnarValue {
        let arg_types = vec![SedonaType::Raster, SedonaType::Raster];
        let args = vec![
            ColumnarValue::Array(Arc::new(raster_array([Some(a)])) as ArrayRef),
            ColumnarValue::Array(Arc::new(raster_array([Some(b)])) as ArrayRef),
        ];
        kernel.invoke_batch(&arg_types, &args).unwrap()
    }

    #[test]
    fn udf_names() {
        let min_udf: ScalarUDF = rs_min_udf().into();
        let max_udf: ScalarUDF = rs_max_udf().into();
        assert_eq!(min_udf.name(), "rs_min");
        assert_eq!(max_udf.name(), "rs_max");
    }

    #[test]
    fn min_basic() {
        let kernel = RsMinMax::new(MinMaxOp::Min);
        let a = RasterSpec::d2(2, 1).band_values(&[5u8, 2u8]);
        let b = RasterSpec::d2(2, 1).band_values(&[3u8, 9u8]);

        let ColumnarValue::Array(result) = invoke(&kernel, a, b) else {
            panic!("expected array result");
        };

        let expected = RasterSpec::d2(2, 1).band_values(&[3u8, 2u8]);
        assert_rasters_equal(&result, &[Some(expected)]);
    }

    #[test]
    fn max_basic() {
        let kernel = RsMinMax::new(MinMaxOp::Max);
        let a = RasterSpec::d2(2, 1).band_values(&[5u8, 2u8]);
        let b = RasterSpec::d2(2, 1).band_values(&[3u8, 9u8]);

        let ColumnarValue::Array(result) = invoke(&kernel, a, b) else {
            panic!("expected array result");
        };

        let expected = RasterSpec::d2(2, 1).band_values(&[5u8, 9u8]);
        assert_rasters_equal(&result, &[Some(expected)]);
    }

    #[test]
    fn nodata_is_skipped() {
        let kernel = RsMinMax::new(MinMaxOp::Min);
        let a = RasterSpec::d2(2, 1)
            .band_values(&[255u8, 255u8])
            .nodata(255u8);
        let b = RasterSpec::d2(2, 1)
            .band_values(&[3u8, 255u8])
            .nodata(255u8);

        let ColumnarValue::Array(result) = invoke(&kernel, a, b) else {
            panic!("expected array result");
        };

        let expected = RasterSpec::d2(2, 1)
            .band_values(&[3u8, 255u8])
            .nodata(255u8);
        assert_rasters_equal(&result, &[Some(expected)]);
    }

    #[test]
    fn nan_does_not_discard_a_valid_value_regardless_of_side() {
        // Neither a's NaN nor b's NaN should unconditionally "win" --
        // PartialOrd comparisons against NaN are always false, which
        // previously made the second operand win outright whenever it held
        // NaN (silently discarding a valid first-operand value for both Min
        // and Max).
        let min_kernel = RsMinMax::new(MinMaxOp::Min);
        let max_kernel = RsMinMax::new(MinMaxOp::Max);

        // NaN in b.
        let a = RasterSpec::d2(1, 1).band_values(&[3.0f64]);
        let b = RasterSpec::d2(1, 1).band_values(&[f64::NAN]);
        let ColumnarValue::Array(result) = invoke(&min_kernel, a.clone(), b.clone()) else {
            panic!("expected array result");
        };
        assert_rasters_equal(
            &result,
            &[Some(RasterSpec::d2(1, 1).band_values(&[3.0f64]))],
        );
        let ColumnarValue::Array(result) = invoke(&max_kernel, a, b) else {
            panic!("expected array result");
        };
        assert_rasters_equal(
            &result,
            &[Some(RasterSpec::d2(1, 1).band_values(&[3.0f64]))],
        );

        // NaN in a.
        let a = RasterSpec::d2(1, 1).band_values(&[f64::NAN]);
        let b = RasterSpec::d2(1, 1).band_values(&[3.0f64]);
        let ColumnarValue::Array(result) = invoke(&min_kernel, a.clone(), b.clone()) else {
            panic!("expected array result");
        };
        assert_rasters_equal(
            &result,
            &[Some(RasterSpec::d2(1, 1).band_values(&[3.0f64]))],
        );
        let ColumnarValue::Array(result) = invoke(&max_kernel, a, b) else {
            panic!("expected array result");
        };
        assert_rasters_equal(
            &result,
            &[Some(RasterSpec::d2(1, 1).band_values(&[3.0f64]))],
        );
    }

    #[test]
    fn mismatched_shape_errors() {
        let kernel = RsMinMax::new(MinMaxOp::Min);
        let a = RasterSpec::d2(2, 1).band_values(&[5u8, 2u8]);
        let b = RasterSpec::d2(1, 1).band_values(&[3u8]);

        let arg_types = vec![SedonaType::Raster, SedonaType::Raster];
        let args = vec![
            ColumnarValue::Array(Arc::new(raster_array([Some(a)])) as ArrayRef),
            ColumnarValue::Array(Arc::new(raster_array([Some(b)])) as ArrayRef),
        ];
        let err = kernel.invoke_batch(&arg_types, &args).unwrap_err();
        assert!(err.to_string().contains("different shapes"));
    }

    #[test]
    fn explicit_band_selects_second_band() {
        let kernel = RsMinMaxWithBand::new(MinMaxOp::Min);
        let a = RasterSpec::d2(1, 1)
            .band_values(&[5u8])
            .band_values(&[50u8]);
        let b = RasterSpec::d2(1, 1)
            .band_values(&[3u8])
            .band_values(&[90u8]);

        let arg_types = vec![
            SedonaType::Raster,
            SedonaType::Raster,
            SedonaType::Arrow(arrow_schema::DataType::Int32),
        ];
        let args = vec![
            ColumnarValue::Array(Arc::new(raster_array([Some(a)])) as ArrayRef),
            ColumnarValue::Array(Arc::new(raster_array([Some(b)])) as ArrayRef),
            ColumnarValue::Array(Arc::new(arrow_array::Int32Array::from(vec![2])) as ArrayRef),
        ];
        let ColumnarValue::Array(result) = kernel.invoke_batch(&arg_types, &args).unwrap() else {
            panic!("expected array result");
        };

        let expected = RasterSpec::d2(1, 1).band_values(&[50u8]);
        assert_rasters_equal(&result, &[Some(expected)]);
    }

    #[test]
    fn out_of_range_band_is_null() {
        let kernel = RsMinMaxWithBand::new(MinMaxOp::Min);
        let a = RasterSpec::d2(1, 1).band_values(&[5u8]);
        let b = RasterSpec::d2(1, 1).band_values(&[3u8]);

        let arg_types = vec![
            SedonaType::Raster,
            SedonaType::Raster,
            SedonaType::Arrow(arrow_schema::DataType::Int32),
        ];
        let args = vec![
            ColumnarValue::Array(Arc::new(raster_array([Some(a)])) as ArrayRef),
            ColumnarValue::Array(Arc::new(raster_array([Some(b)])) as ArrayRef),
            ColumnarValue::Array(Arc::new(arrow_array::Int32Array::from(vec![2])) as ArrayRef),
        ];
        let ColumnarValue::Array(result) = kernel.invoke_batch(&arg_types, &args).unwrap() else {
            panic!("expected array result");
        };

        assert_rasters_equal(&result, &[None]);
    }
}
