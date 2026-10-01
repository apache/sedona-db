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

//! `RS_ReplaceBandNoDataValue` — move a band's nodata to a new value, carrying
//! the nodata pixels with it.
//!
//! ```text
//! RS_ReplaceBandNoDataValue(raster, band, nodata)  -> Raster
//! ```
//!
//! Every pixel of the addressed band that reads as nodata today is rewritten to
//! `nodata`, which then becomes the band's nodata value, so the same pixels
//! read as nodata before and after the call. `RS_SetBandNoDataValue` only
//! changes the declared value and leaves the pixels alone.
//!
//! A pixel reads as nodata by the same rules sampling uses
//! ([`NodataMatcher`]): exact for integer bands, numeric for float bands, so
//! `-0.0` matches a `0.0` nodata and any NaN matches a NaN nodata.
//!
//! The band must already have a nodata value; with none there is nothing to
//! replace, so that is an error. A null raster, band or `nodata` yields a null
//! raster. Other bands are carried over untouched and zero-copy. Only 2-D
//! bands are supported.
//!
//! There is deliberately no form without a band: Sedona Spark's 2-argument
//! forms default to band 1 where SedonaDB's reject a multiband raster, and a
//! new function need not inherit that divergence.

use std::ops::ControlFlow;
use std::sync::Arc;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int64Type};
use arrow_buffer::Buffer;
use arrow_schema::DataType;
use datafusion_common::{Result, exec_datafusion_err, exec_err};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::band_builder::check_band_data_len;
use sedona_raster::builder::{RasterBuilder, RasterOverrides};
use sedona_raster::traits::{BandOverrides, BandRef, Override, RasterRef, nodata_f64_to_bytes};
use sedona_schema::datatypes::SedonaType;
use sedona_schema::matchers::ArgMatcher;

use crate::executor::RasterExecutor;
use crate::pixel_scan::{NodataMatcher, scan_pixels, spatial_2d_buffer};
use crate::rs_ensure_loaded::{NEEDS_PIXELS_METADATA_KEY, RETURNS_BYTES_METADATA_KEY};
use crate::sampling::resolve_band;

const FUNC: &str = "RS_ReplaceBandNoDataValue";

/// `RS_ReplaceBandNoDataValue()` scalar UDF.
pub fn rs_replace_band_nodata_value_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_replacebandnodatavalue",
        vec![Arc::new(RsReplaceBandNoDataValue)],
        Volatility::Immutable,
    )
    // The kernel reads and rewrites pixel bytes, so the raster argument must be
    // materialised InDb first; the planner injects RS_EnsureLoaded on this flag.
    .with_metadata(NEEDS_PIXELS_METADATA_KEY, "true")
    // The output is InDb too (the addressed band is fresh bytes or its loaded
    // bytes, the others are copied from the loaded input), so a consumer must
    // not wrap it in another RS_EnsureLoaded.
    .with_metadata(RETURNS_BYTES_METADATA_KEY, "true")
}

#[derive(Debug)]
struct RsReplaceBandNoDataValue;

impl SedonaScalarKernel for RsReplaceBandNoDataValue {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        ArgMatcher::new(
            vec![
                ArgMatcher::is_raster(),
                ArgMatcher::is_integer(),
                ArgMatcher::is_numeric(),
            ],
            SedonaType::Raster,
        )
        .match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let n = executor.num_iterations();

        let band_array = args[1]
            .clone()
            .cast_to(&DataType::Int64, None)?
            .into_array(n)?;
        let band = band_array.as_primitive::<Int64Type>();
        let nodata_array = args[2]
            .clone()
            .cast_to(&DataType::Float64, None)?
            .into_array(n)?;
        let nodata = nodata_array.as_primitive::<Float64Type>();

        let mut builder = RasterBuilder::new(n);
        executor.execute_raster_void(|i, raster_opt| {
            let Some(raster) = raster_opt else {
                return Ok(builder.append_null()?);
            };
            if band.is_null(i) || nodata.is_null(i) {
                return Ok(builder.append_null()?);
            }
            // Clamp a negative band to 0 so resolve_band rejects it as not
            // 1-based rather than wrapping it into a huge usize.
            let band_num = band.value(i).max(0) as usize;
            replace_band_nodata(&mut builder, raster, band_num, nodata.value(i))
        })?;

        executor.finish(Arc::new(builder.finish()?))
    }
}

/// Copy `raster` into `builder` with band `band_num` (1-based) moved to the
/// nodata value `nodata`: its nodata pixels are rewritten to the new value
/// first, so they keep reading as nodata.
fn replace_band_nodata(
    builder: &mut RasterBuilder,
    raster: &dyn RasterRef,
    band_num: usize,
    nodata: f64,
) -> Result<()> {
    let target = resolve_band(FUNC, raster, band_num)?;
    let new_nodata = nodata_f64_to_bytes(nodata, &target.data_type())
        .map_err(|e| exec_datafusion_err!("{FUNC}: {e}"))?;
    let data = replace_nodata_pixels(target.as_ref(), &new_nodata)?;

    builder.start_raster_from(raster, RasterOverrides::default())?;
    for band_idx in 0..raster.num_bands() {
        let band = raster.band(band_idx)?;
        let overrides = match &data {
            // The new bytes are the band's visible pixels, packed row-major, so
            // they carry an identity view over the visible shape.
            Some(data) if band_idx + 1 == band_num => BandOverrides {
                data: Override::Set(data),
                view: Override::Clear,
                source_shape: Some(band.shape()),
                nodata: Override::Set(&new_nodata),
                ..Default::default()
            },
            // No pixel held the old nodata, so only the declared value changes.
            None if band_idx + 1 == band_num => BandOverrides {
                nodata: Override::Set(&new_nodata),
                ..Default::default()
            },
            _ => BandOverrides::default(),
        };
        band.copy_into(builder, overrides)?;
        builder.finish_band()?;
    }
    builder.finish_raster()?;
    Ok(())
}

/// The band's visible pixels, packed row-major, with every pixel that reads as
/// nodata rewritten to `new_nodata`, or `None` when no pixel reads as nodata
/// and the band's bytes can be kept as they are.
fn replace_nodata_pixels(band: &dyn BandRef, new_nodata: &[u8]) -> Result<Option<Buffer>> {
    let Some(matcher) = NodataMatcher::for_band(FUNC, band)? else {
        return exec_err!(
            "{FUNC}: the band has no nodata value to replace; use RS_SetBandNoDataValue to set one"
        );
    };
    let buffer = spatial_2d_buffer(FUNC, band)?;
    let (height, width) = (buffer.shape[0], buffer.shape[1]);

    // Look for a nodata pixel before copying anything: a band without one
    // keeps its bytes, at the cost of a scan up to the first match.
    let mut any_nodata = false;
    scan_pixels(FUNC, &buffer, |_, _, pixel| {
        if matcher.matches(pixel) {
            any_nodata = true;
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    })?;
    if !any_nodata {
        return Ok(None);
    }

    // A broadcast view can describe far more pixels than its source holds, so
    // size the packed output with checked arithmetic and reject it before
    // allocating rather than after.
    let len = usize::try_from(width)
        .ok()
        .and_then(|w| w.checked_mul(usize::try_from(height).ok()?))
        .and_then(|n| n.checked_mul(new_nodata.len()))
        .ok_or_else(|| {
            exec_datafusion_err!("{FUNC}: a {width} x {height} band is too large to materialise")
        })?;
    check_band_data_len(len).map_err(|e| exec_datafusion_err!("{FUNC}: {e}"))?;

    let mut data = Vec::with_capacity(len);
    scan_pixels(FUNC, &buffer, |_, _, pixel| {
        data.extend_from_slice(if matcher.matches(pixel) {
            new_nodata
        } else {
            pixel
        });
        ControlFlow::Continue(())
    })?;
    Ok(Some(Buffer::from_vec(data)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{
        ArrayRef, BinaryViewArray, Float64Array, Int64Array, ListArray, StructArray,
    };
    use datafusion_expr::ScalarUDF;
    use sedona_schema::datatypes::RASTER;
    use sedona_schema::raster::{band_indices, raster_indices};
    use sedona_testing::raster_spec::{
        RasterSpec, assert_raster_scalar_equals, assert_rasters_equal, raster_array,
    };
    use sedona_testing::testers::ScalarUdfTester;

    #[test]
    fn udf_metadata() {
        let udf = rs_replace_band_nodata_value_udf();
        let scalar: ScalarUDF = udf.clone().into();
        assert_eq!(scalar.name(), "rs_replacebandnodatavalue");
        for key in [NEEDS_PIXELS_METADATA_KEY, RETURNS_BYTES_METADATA_KEY] {
            assert_eq!(udf.metadata().get(key).map(String::as_str), Some("true"));
        }
    }

    #[test]
    fn rewrites_nodata_pixels_and_moves_the_sentinel() {
        // Band 1 (nodata 0) has two nodata pixels; band 2 is untouched.
        let input = RasterSpec::d2(3, 1)
            .band_values(&[0u8, 5, 0])
            .nodata(0u8)
            .band_values(&[0u8, 9, 0])
            .nodata(0u8);
        let tester = ScalarUdfTester::new(
            rs_replace_band_nodata_value_udf().into(),
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Int64),
                SedonaType::Arrow(DataType::Float64),
            ],
        );
        let result = tester
            .invoke_scalar_scalar_scalar(&input, 1, 255.0)
            .unwrap();
        let expected = RasterSpec::d2(3, 1)
            .band_values(&[255u8, 5, 255])
            .nodata(255u8)
            .band_values(&[0u8, 9, 0])
            .nodata(0u8);
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn a_band_without_nodata_pixels_keeps_its_bytes() {
        // No pixel holds the old nodata 0, so only the declared value moves
        // and the band's 16 bytes (past the 12 a view holds inline) are carried
        // over without a copy.
        let input = RasterSpec::d2(4, 4).band_values(&[1u8; 16]).nodata(0u8);
        let input: ArrayRef = Arc::new(raster_array(vec![Some(input)]));
        let tester = ScalarUdfTester::new(
            rs_replace_band_nodata_value_udf().into(),
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Int64),
                SedonaType::Arrow(DataType::Float64),
            ],
        );
        let result = tester
            .invoke_arrays(vec![
                input.clone(),
                Arc::new(Int64Array::from(vec![1])),
                Arc::new(Float64Array::from(vec![255.0])),
            ])
            .unwrap();
        let expected = RasterSpec::d2(4, 4).band_values(&[1u8; 16]).nodata(255u8);
        assert_rasters_equal(&result, &[Some(expected)]);

        let data_buffer = |rasters: &ArrayRef| {
            let bands = rasters
                .as_any()
                .downcast_ref::<StructArray>()
                .unwrap()
                .column(raster_indices::BANDS)
                .as_any()
                .downcast_ref::<ListArray>()
                .unwrap()
                .values()
                .as_any()
                .downcast_ref::<StructArray>()
                .unwrap()
                .clone();
            let data = bands
                .column(band_indices::DATA)
                .as_any()
                .downcast_ref::<BinaryViewArray>()
                .unwrap()
                .clone();
            data.data_buffers()[0].as_ptr()
        };
        assert_eq!(data_buffer(&result), data_buffer(&input));
    }

    #[test]
    fn float_nodata_matches_like_sampling() {
        // -0.0 matches a 0.0 nodata, and any NaN matches a NaN nodata, as they
        // do when sampling, so every pixel that read as nodata is carried over.
        let input = RasterSpec::d2(3, 1)
            .band_values(&[-0.0f64, 0.0, 1.5])
            .nodata(0.0f64);
        let tester = ScalarUdfTester::new(
            rs_replace_band_nodata_value_udf().into(),
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Int64),
                SedonaType::Arrow(DataType::Float64),
            ],
        );
        let result = tester
            .invoke_scalar_scalar_scalar(&input, 1, -9999.0)
            .unwrap();
        let expected = RasterSpec::d2(3, 1)
            .band_values(&[-9999.0f64, -9999.0, 1.5])
            .nodata(-9999.0f64);
        assert_raster_scalar_equals(&result, &expected);

        let other_nan = f32::from_bits(f32::NAN.to_bits() | 1);
        let input = RasterSpec::d2(3, 1)
            .band_values(&[f32::NAN, other_nan, 2.0])
            .nodata(f32::NAN);
        let result = tester.invoke_scalar_scalar_scalar(&input, 1, -1.0).unwrap();
        let expected = RasterSpec::d2(3, 1)
            .band_values(&[-1.0f32, -1.0, 2.0])
            .nodata(-1.0f32);
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn a_band_without_nodata_errors() {
        let input = RasterSpec::d2(2, 1).band_values(&[0u8, 3]);
        let tester = ScalarUdfTester::new(
            rs_replace_band_nodata_value_udf().into(),
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Int64),
                SedonaType::Arrow(DataType::Float64),
            ],
        );
        let err = tester
            .invoke_scalar_scalar_scalar(&input, 1, 7.0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("no nodata value to replace"), "{err}");
    }

    #[test]
    fn a_nodata_the_band_type_cannot_hold_errors() {
        let input = RasterSpec::d2(2, 1).band_values(&[0u8, 3]).nodata(0u8);
        let tester = ScalarUdfTester::new(
            rs_replace_band_nodata_value_udf().into(),
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Int64),
                SedonaType::Arrow(DataType::Float64),
            ],
        );
        for value in [256.0, 1.5, -1.0] {
            let err = tester
                .invoke_scalar_scalar_scalar(&input, 1, value)
                .unwrap_err()
                .to_string();
            assert!(err.contains(FUNC), "{value}: {err}");
        }
    }

    #[test]
    fn band_out_of_range_errors() {
        let input = RasterSpec::d2(2, 1).band_values(&[0u8, 3]).nodata(0u8);
        let tester = ScalarUdfTester::new(
            rs_replace_band_nodata_value_udf().into(),
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Int64),
                SedonaType::Arrow(DataType::Float64),
            ],
        );
        for (band, expected) in [(0, "1-based"), (-1, "1-based"), (2, "out of range")] {
            let err = tester
                .invoke_scalar_scalar_scalar(&input, band, 7.0)
                .unwrap_err()
                .to_string();
            assert!(err.contains(expected), "{band}: {err}");
        }
    }

    #[test]
    fn null_arguments_give_a_null_raster() {
        let tester = ScalarUdfTester::new(
            rs_replace_band_nodata_value_udf().into(),
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Int64),
                SedonaType::Arrow(DataType::Float64),
            ],
        );
        let spec = || Some(RasterSpec::d2(2, 1).band_values(&[0u8, 3]).nodata(0u8));
        let result = tester
            .invoke_arrays(vec![
                Arc::new(raster_array(vec![None, spec(), spec()])),
                Arc::new(Int64Array::from(vec![Some(1), None, Some(1)])),
                Arc::new(Float64Array::from(vec![Some(7.0), Some(7.0), None])),
            ])
            .unwrap();
        assert_eq!(result.null_count(), 3);
    }

    #[test]
    fn a_non_2d_band_errors() {
        let input = RasterSpec::d2(1, 1)
            .band_values_nd(&["time", "y", "x"], &[2, 1, 1], &[0u8, 3])
            .nodata(0u8);
        let tester = ScalarUdfTester::new(
            rs_replace_band_nodata_value_udf().into(),
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Int64),
                SedonaType::Arrow(DataType::Float64),
            ],
        );
        let err = tester
            .invoke_scalar_scalar_scalar(&input, 1, 7.0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("2-D"), "{err}");
    }
}
