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

//! `RS_BandIsNoData` — whether every pixel of a band holds its nodata value.
//!
//! ```text
//! RS_BandIsNoData(raster)        -> Boolean  -- band 1
//! RS_BandIsNoData(raster, band)  -> Boolean
//! ```
//!
//! True when the band declares a nodata value and every pixel holds it; false
//! when any pixel holds data, or when the band has no nodata value at all. A
//! NaN pixel matches a NaN nodata. The scan stops at the first data pixel.

use std::ops::ControlFlow;
use std::sync::Arc;

use arrow_array::Array;
use arrow_array::builder::BooleanBuilder;
use arrow_schema::DataType;
use datafusion_common::Result;
use datafusion_common::cast::as_int32_array;
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::traits::RasterRef;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::executor::RasterExecutor;
use crate::pixel_scan::{NodataMatcher, scan_pixels, spatial_2d_buffer};
use crate::rs_ensure_loaded::NEEDS_PIXELS_METADATA_KEY;
use crate::sampling::{int32_array_arg, resolve_band};

/// `RS_BandIsNoData()` scalar UDF — whether a band is entirely nodata.
pub fn rs_bandisnodata_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_bandisnodata",
        vec![
            Arc::new(RsBandIsNoData { with_band: false }), // RS_BandIsNoData(raster)
            Arc::new(RsBandIsNoData { with_band: true }),  // RS_BandIsNoData(raster, band)
        ],
        Volatility::Immutable,
    )
    // The kernel reads pixel bytes, so the raster argument must be materialised
    // InDb first; the planner injects RS_EnsureLoaded based on this flag.
    .with_metadata(NEEDS_PIXELS_METADATA_KEY, "true")
}

#[derive(Debug)]
struct RsBandIsNoData {
    with_band: bool,
}

impl SedonaScalarKernel for RsBandIsNoData {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let mut matchers = vec![ArgMatcher::is_raster()];
        if self.with_band {
            matchers.push(ArgMatcher::is_integer());
        }
        let matcher = ArgMatcher::new(matchers, SedonaType::Arrow(DataType::Boolean));
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let num_iterations = executor.num_iterations();
        let mut builder = BooleanBuilder::with_capacity(num_iterations);

        let band_arr = if self.with_band {
            Some(int32_array_arg(&args[1], num_iterations)?)
        } else {
            None
        };
        let band = band_arr.as_ref().map(|a| as_int32_array(a)).transpose()?;

        executor.execute_raster_void(|i, raster_opt| {
            let Some(raster) = raster_opt else {
                builder.append_null();
                return Ok(());
            };
            let band_num = match band {
                None => 1,
                Some(band) if band.is_null(i) => {
                    builder.append_null();
                    return Ok(());
                }
                // Clamp a negative band to 0 so resolve_band rejects it as not
                // 1-based rather than wrapping it into a huge usize.
                Some(band) => band.value(i).max(0) as usize,
            };
            builder.append_value(band_is_nodata(raster, band_num)?);
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

/// Whether every pixel of the 1-based band `band_num` holds its nodata value.
fn band_is_nodata(raster: &dyn RasterRef, band_num: usize) -> Result<bool> {
    let band = resolve_band("RS_BandIsNoData", raster, band_num)?;
    // Check the shape first so a non-2-D band errors whether or not it has a
    // nodata value.
    let buffer = spatial_2d_buffer("RS_BandIsNoData", band.as_ref())?;
    let Some(nodata) = NodataMatcher::for_band("RS_BandIsNoData", band.as_ref())? else {
        return Ok(false);
    };
    let mut all_nodata = true;
    scan_pixels("RS_BandIsNoData", &buffer, |_, _, pixel| {
        if nodata.matches(pixel) {
            ControlFlow::Continue(())
        } else {
            all_nodata = false;
            ControlFlow::Break(())
        }
    })?;
    Ok(all_nodata)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, BooleanArray, Int32Array};
    use datafusion_expr::ScalarUDF;
    use sedona_raster::array::RasterStructArray;
    use sedona_schema::datatypes::RASTER;
    use sedona_testing::raster_spec::{RasterSpec, raster_array};
    use sedona_testing::testers::ScalarUdfTester;

    fn is_nodata(spec: RasterSpec, band: usize) -> Result<bool> {
        let array = spec.build();
        let rasters = RasterStructArray::try_new(&array).unwrap();
        band_is_nodata(&rasters.get(0).unwrap(), band)
    }

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = rs_bandisnodata_udf().into();
        assert_eq!(udf.name(), "rs_bandisnodata");
        assert_eq!(
            rs_bandisnodata_udf()
                .metadata()
                .get(NEEDS_PIXELS_METADATA_KEY)
                .map(String::as_str),
            Some("true")
        );
    }

    #[test]
    fn all_nodata_band_is_true() {
        let spec = RasterSpec::d2(2, 2).band_values(&[9u8; 4]).nodata(9u8);
        assert!(is_nodata(spec, 1).unwrap());
    }

    #[test]
    fn one_data_pixel_is_false() {
        // The data pixel is the last one, so the whole band is scanned.
        let spec = RasterSpec::d2(2, 2)
            .band_values(&[9u8, 9, 9, 1])
            .nodata(9u8);
        assert!(!is_nodata(spec, 1).unwrap());
    }

    #[test]
    fn band_without_nodata_is_false() {
        // Even a band of zeros is data when no nodata value is declared.
        let spec = RasterSpec::d2(2, 2).band_values(&[0u8; 4]);
        assert!(!is_nodata(spec, 1).unwrap());
    }

    #[test]
    fn nan_nodata_matches_nan_pixels() {
        let spec = RasterSpec::d2(2, 1)
            .band_values(&[f64::NAN, f64::NAN])
            .nodata(f64::NAN);
        assert!(is_nodata(spec, 1).unwrap());
        let spec = RasterSpec::d2(2, 1)
            .band_values(&[f64::NAN, 0.5])
            .nodata(f64::NAN);
        assert!(!is_nodata(spec, 1).unwrap());
    }

    #[test]
    fn addresses_the_requested_band() {
        let spec = RasterSpec::d2(2, 1)
            .band_values(&[1u8, 2])
            .nodata(0u8)
            .band_values(&[0u8, 0])
            .nodata(0u8);
        assert!(!is_nodata(spec.clone(), 1).unwrap());
        assert!(is_nodata(spec, 2).unwrap());
    }

    #[test]
    fn band_out_of_range_errors() {
        let spec = RasterSpec::d2(2, 1).band_values(&[1u8, 2]).nodata(0u8);
        let err = is_nodata(spec.clone(), 2).unwrap_err().to_string();
        assert!(err.contains("RS_BandIsNoData"), "{err}");
        let err = is_nodata(spec, 0).unwrap_err().to_string();
        assert!(err.contains("1-based"), "{err}");
    }

    #[test]
    fn non_2d_band_errors() {
        // With or without a nodata value.
        let spec = RasterSpec::d2(2, 1).band_values_nd(&["time", "y", "x"], &[1, 1, 2], &[0u8, 0]);
        let err = is_nodata(spec.clone(), 1).unwrap_err().to_string();
        assert!(err.contains("2-D"), "{err}");
        let err = is_nodata(spec.nodata(0u8), 1).unwrap_err().to_string();
        assert!(err.contains("2-D"), "{err}");
    }

    #[test]
    fn udf_invoke_defaults_to_band_one_and_propagates_nulls() {
        let rasters = raster_array([
            Some(RasterSpec::d2(1, 1).band_values(&[0u8]).nodata(0u8)),
            None,
            Some(
                RasterSpec::d2(1, 1)
                    .band_values(&[1u8])
                    .nodata(0u8)
                    .band_values(&[0u8])
                    .nodata(0u8),
            ),
        ]);

        let udf: ScalarUDF = rs_bandisnodata_udf().into();
        let tester = ScalarUdfTester::new(udf.clone(), vec![RASTER]);
        let result = tester.invoke_array(Arc::new(rasters.clone())).unwrap();
        let result = result.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(
            result.iter().collect::<Vec<_>>(),
            vec![Some(true), None, Some(false)]
        );

        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Int32)]);
        let result = tester
            .invoke_arrays(vec![
                Arc::new(rasters),
                Arc::new(Int32Array::from(vec![Some(1), Some(1), None])),
            ])
            .unwrap();
        let result = result.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(
            result.iter().collect::<Vec<_>>(),
            vec![Some(true), None, None]
        );
    }

    #[test]
    fn udf_invoke_second_band() {
        let rasters = raster_array([Some(
            RasterSpec::d2(1, 1)
                .band_values(&[1u8])
                .nodata(0u8)
                .band_values(&[0u8])
                .nodata(0u8),
        )]);
        let udf: ScalarUDF = rs_bandisnodata_udf().into();
        let tester = ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Int32)]);
        let result = tester
            .invoke_arrays(vec![Arc::new(rasters), Arc::new(Int32Array::from(vec![2]))])
            .unwrap();
        let result = result.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(result.value(0));
        assert!(!result.is_null(0));
    }
}
