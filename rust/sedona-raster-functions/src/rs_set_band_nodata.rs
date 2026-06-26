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

//! `RS_SetBandNoDataValue` — set a band's nodata sentinel.
//!
//! ```text
//! RS_SetBandNoDataValue(raster, nodata)        -> Raster  -- single-band rasters only
//! RS_SetBandNoDataValue(raster, band, nodata)  -> Raster
//! ```
//!
//! The setter companion to the `RS_BandNoDataValue` getter. `nodata` is a double
//! packed into the band's native data type. A null raster, band, or nodata value
//! yields a null raster (matching `RS_SetCRS`/`RS_SetSRID`).
//!
//! An out-of-range band index is an error — unlike the getter, which returns
//! NULL for a missing band. The asymmetry is deliberate: this op rewrites the
//! whole raster, so silently nulling it on a typo'd band index would lose data,
//! whereas the getter's NULL is a harmless scalar miss.
//!
//! `nodata` is taken as `f64`, so an integer sentinel beyond 2^53 cannot be
//! represented exactly; pass nodata within `f64`'s exact-integer range.
//!
//! The raster is rebuilt with [`BandRef::copy_into`], which carries each band's
//! pixel data over by sharing the backing buffers (zero-copy) — only the
//! addressed band's nodata is overridden.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int64Type};
use arrow_array::Array;
use arrow_schema::DataType;
use datafusion_common::error::Result;
use datafusion_common::{arrow_datafusion_err, exec_err};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::builder::{RasterBuilder, RasterOverrides};
use sedona_raster::traits::{nodata_f64_to_bytes, BandOverrides, RasterRef};
use sedona_schema::datatypes::SedonaType;
use sedona_schema::matchers::ArgMatcher;

use crate::executor::RasterExecutor;

/// RS_SetBandNoDataValue() scalar UDF implementation
pub fn rs_set_band_nodata_value_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_setbandnodatavalue",
        vec![
            Arc::new(RsSetBandNoDataValue { with_band: false }),
            Arc::new(RsSetBandNoDataValue { with_band: true }),
        ],
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct RsSetBandNoDataValue {
    with_band: bool,
}

impl SedonaScalarKernel for RsSetBandNoDataValue {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let mut matchers = vec![ArgMatcher::is_raster()];
        if self.with_band {
            matchers.push(ArgMatcher::is_integer());
        }
        matchers.push(ArgMatcher::is_numeric());
        let matcher = ArgMatcher::new(matchers, SedonaType::Raster);
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let n = executor.num_iterations();

        // The band column (1-based; absent for the 2-arg form, where the band is
        // resolved in `set_band_nodata`) and the nodata value column, read inline
        // per row.
        let band_array = if self.with_band {
            Some(
                args[1]
                    .clone()
                    .cast_to(&DataType::Int64, None)?
                    .into_array(n)?,
            )
        } else {
            None
        };
        let band_values = band_array.as_ref().map(|a| a.as_primitive::<Int64Type>());

        // nodata is taken as f64. An integer sentinel is cast losslessly here so
        // long as it is within f64's exact-integer range (±2^53); a larger
        // i64/u64 would lose precision in this cast. Whether the value fits the
        // *band's* data type is validated later by `nodata_f64_to_bytes`, which
        // errors rather than truncating. (Dedicated integer kernels could be
        // registered if exact large-integer sentinels are ever needed.)
        let value_arg = if self.with_band { &args[2] } else { &args[1] };
        let value_array = value_arg
            .clone()
            .cast_to(&DataType::Float64, None)?
            .into_array(n)?;
        let value_values = value_array.as_primitive::<Float64Type>();

        let mut builder = RasterBuilder::new(n);
        executor.execute_raster_void(|i, raster_opt| {
            let null_out =
                |b: &mut RasterBuilder| b.append_null().map_err(|e| arrow_datafusion_err!(e));

            // A null raster, band, or value yields a null raster.
            let Some(raster) = raster_opt else {
                return null_out(&mut builder);
            };
            let band = match band_values {
                Some(bands) if bands.is_null(i) => return null_out(&mut builder),
                Some(bands) => Some(bands.value(i)),
                None => None,
            };
            if value_values.is_null(i) {
                return null_out(&mut builder);
            }
            set_band_nodata(&mut builder, raster, band, value_values.value(i))
        })?;

        executor.finish(Arc::new(builder.finish()?))
    }
}

/// Copy `raster` into `builder`, overriding the addressed (1-based) band's
/// nodata with `value` packed into that band's data type. Every other band is
/// copied with its data shared and metadata inherited.
///
/// `band` is `None` for the 2-argument form (no band given): it defaults to band
/// 1 only when the raster is single-band, and errors on a multiband raster so a
/// caller can't silently set nodata on just band 1.
fn set_band_nodata(
    builder: &mut RasterBuilder,
    raster: &dyn RasterRef,
    band: Option<i64>,
    value: f64,
) -> Result<()> {
    let num_bands = raster.num_bands();
    let band = match band {
        Some(band) => band,
        None if num_bands == 1 => 1,
        None => {
            return exec_err!(
                "RS_SetBandNoDataValue: raster has {num_bands} bands; specify which band to set \
                 (the 2-argument form is only allowed for a single-band raster)"
            )
        }
    };
    if band < 1 || band as usize > num_bands {
        return exec_err!(
            "RS_SetBandNoDataValue: band {band} out of range (raster has {num_bands} band(s))"
        );
    }

    // Copy the raster header (transform/dims/crs) verbatim; we rebuild the bands
    // below so we can override the addressed band's nodata.
    builder.start_raster_from(raster, RasterOverrides::default())?;

    for band_idx in 0..num_bands {
        let band_ref = raster.band(band_idx)?;
        // Override the nodata only on the addressed band; others inherit.
        let nodata_bytes = if band_idx + 1 == band as usize {
            Some(nodata_f64_to_bytes(value, &band_ref.data_type())?)
        } else {
            None
        };
        band_ref.copy_into(
            builder,
            BandOverrides {
                nodata: nodata_bytes.as_deref(),
                ..Default::default()
            },
        )?;
        builder.finish_band()?;
    }

    builder.finish_raster()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::DataType;
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use sedona_schema::datatypes::RASTER;
    use sedona_testing::raster_spec::{assert_rasters_equal, RasterSpec};
    use sedona_testing::rasters::generate_test_rasters;
    use sedona_testing::testers::ScalarUdfTester;

    /// Two UInt8 bands, no nodata.
    fn two_band() -> RasterSpec {
        RasterSpec::d2(2, 1)
            .band_values(&[1u8, 2])
            .band_values(&[3u8, 4])
    }

    fn tester_2arg() -> ScalarUdfTester {
        let udf: ScalarUDF = rs_set_band_nodata_value_udf().into();
        ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Float64)])
    }

    fn tester_3arg() -> ScalarUdfTester {
        let udf: ScalarUDF = rs_set_band_nodata_value_udf().into();
        ScalarUdfTester::new(
            udf,
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Int32),
                SedonaType::Arrow(DataType::Float64),
            ],
        )
    }

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = rs_set_band_nodata_value_udf().into();
        assert_eq!(udf.name(), "rs_setbandnodatavalue");
    }

    #[test]
    fn sets_default_band_nodata_preserving_everything_else() {
        // Two-arg form sets the sole band's nodata on a single-band raster;
        // pixels, transform, CRS are all preserved (the whole-raster comparison
        // proves it).
        let tester = tester_2arg();
        tester.assert_return_type(RASTER);

        let one_band = RasterSpec::d2(2, 1).band_values(&[1u8, 2]);
        let result = tester
            .invoke_array_scalar(Arc::new(one_band.build()), 5.0)
            .unwrap();
        let expected = RasterSpec::d2(2, 1).band_values(&[1u8, 2]).nodata(5u8);
        assert_rasters_equal(&result, &[Some(expected)]);
    }

    #[test]
    fn two_arg_form_on_multiband_errors() {
        // The 2-arg form is ambiguous on a multiband raster — require an
        // explicit band rather than silently setting only band 1.
        let err = tester_2arg()
            .invoke_array_scalar(Arc::new(two_band().build()), 5.0)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("specify which band"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn sets_specific_band_nodata() {
        let result = tester_3arg()
            .invoke_array_scalar_scalar(Arc::new(two_band().build()), 2_i32, 9.0)
            .unwrap();
        let expected = RasterSpec::d2(2, 1)
            .band_values(&[1u8, 2])
            .band_values(&[3u8, 4])
            .nodata(9u8);
        assert_rasters_equal(&result, &[Some(expected)]);
    }

    #[test]
    fn null_value_nulls_raster() {
        let result = tester_2arg()
            .invoke_array_scalar(Arc::new(two_band().build()), ScalarValue::Float64(None))
            .unwrap();
        assert_rasters_equal(&result, &[None]);
    }

    #[test]
    fn null_raster_stays_null() {
        let rasters = generate_test_rasters(1, Some(0)).unwrap();
        let result = tester_2arg()
            .invoke_array_scalar(Arc::new(rasters), 1.0)
            .unwrap();
        assert_rasters_equal(&result, &[None]);
    }

    #[test]
    fn band_out_of_range_errors() {
        let err = tester_3arg()
            .invoke_array_scalar_scalar(Arc::new(two_band().build()), 5_i32, 1.0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("out of range"), "unexpected error: {err}");
    }

    #[test]
    fn value_out_of_dtype_range_errors() {
        // 300 doesn't fit in a UInt8 band. Use a single-band raster so the 2-arg
        // form resolves a band and reaches the dtype range check.
        let one_band = RasterSpec::d2(2, 1).band_values(&[1u8, 2]);
        let err = tester_2arg()
            .invoke_array_scalar(Arc::new(one_band.build()), 300.0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("UInt8"), "unexpected error: {err}");
    }
}
