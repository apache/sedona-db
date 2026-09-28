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

//! `RS_SetValue` — overwrite one pixel of a band.
//!
//! ```text
//! RS_SetValue(raster, colX, rowY, newValue)        -> Raster  -- single-band rasters only
//! RS_SetValue(raster, band, colX, rowY, newValue)  -> Raster
//! ```
//!
//! `colX` and `rowY` are 1-based, as in PostGIS and Sedona Spark; a pixel
//! outside the grid is an error. The pixel is overwritten whether or not it holds the band's nodata
//! value, and the band's nodata value itself is unchanged.
//!
//! `newValue` is stored the way Sedona Spark stores it: truncated toward zero in
//! an integer band and rounded to the nearest `f32` in a Float32 band. A value
//! outside an integer band's range, or NaN, is an error rather than wrapping
//! around as Spark's Java cast does.
//!
//! Only the addressed band's bytes are copied (into a packed buffer with the one
//! pixel changed); every other band is carried over zero-copy.

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
use sedona_raster::traits::{BandOverrides, BandRef, Override, RasterRef, pixel_f64_to_bytes};
use sedona_schema::datatypes::SedonaType;
use sedona_schema::matchers::ArgMatcher;
use sedona_schema::raster::BandDataType;

use crate::executor::RasterExecutor;
use crate::pixel_scan::{scan_pixels, spatial_2d_buffer};
use crate::rs_ensure_loaded::{NEEDS_PIXELS_METADATA_KEY, RETURNS_BYTES_METADATA_KEY};
use crate::sampling::{default_band, resolve_band};

const FUNC: &str = "RS_SetValue";

/// `RS_SetValue()` scalar UDF — overwrite one pixel of a band.
pub fn rs_setvalue_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_setvalue",
        vec![
            Arc::new(RsSetValue { with_band: false }), // (raster, colX, rowY, newValue)
            Arc::new(RsSetValue { with_band: true }),  // (raster, band, colX, rowY, newValue)
        ],
        Volatility::Immutable,
    )
    // The kernel reads and rewrites pixel bytes, so the raster argument must be
    // materialised InDb first; the planner injects RS_EnsureLoaded on this flag.
    .with_metadata(NEEDS_PIXELS_METADATA_KEY, "true")
    // The output is InDb too (the edited band is fresh bytes, the others are
    // copied from the loaded input), so a consumer of RS_SetValue — including a
    // nested RS_SetValue — must not wrap it in another RS_EnsureLoaded.
    .with_metadata(RETURNS_BYTES_METADATA_KEY, "true")
}

#[derive(Debug)]
struct RsSetValue {
    with_band: bool,
}

impl SedonaScalarKernel for RsSetValue {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let mut matchers = vec![ArgMatcher::is_raster()];
        if self.with_band {
            matchers.push(ArgMatcher::is_integer());
        }
        matchers.extend([
            ArgMatcher::is_integer(),
            ArgMatcher::is_integer(),
            ArgMatcher::is_numeric(),
        ]);
        ArgMatcher::new(matchers, SedonaType::Raster).match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = RasterExecutor::new(arg_types, args);
        let n = executor.num_iterations();

        let int_column = |arg: &ColumnarValue| -> Result<_> {
            arg.clone().cast_to(&DataType::Int64, None)?.into_array(n)
        };
        let first = if self.with_band { 2 } else { 1 };
        let band_array = self.with_band.then(|| int_column(&args[1])).transpose()?;
        let col_array = int_column(&args[first])?;
        let row_array = int_column(&args[first + 1])?;
        let value_array = args[first + 2]
            .clone()
            .cast_to(&DataType::Float64, None)?
            .into_array(n)?;
        let band = band_array.as_ref().map(|a| a.as_primitive::<Int64Type>());
        let col = col_array.as_primitive::<Int64Type>();
        let row = row_array.as_primitive::<Int64Type>();
        let value = value_array.as_primitive::<Float64Type>();

        let mut builder = RasterBuilder::new(n);
        executor.execute_raster_void(|i, raster_opt| {
            let Some(raster) = raster_opt else {
                return Ok(builder.append_null()?);
            };
            if band.is_some_and(|b| b.is_null(i))
                || col.is_null(i)
                || row.is_null(i)
                || value.is_null(i)
            {
                return Ok(builder.append_null()?);
            }
            let band_num = match band {
                // Clamp a negative band to 0 so resolve_band rejects it as not
                // 1-based rather than wrapping it into a huge usize.
                Some(band) => band.value(i).max(0) as usize,
                None => default_band(FUNC, "4-argument", raster.num_bands())?,
            };
            set_value(
                &mut builder,
                raster,
                band_num,
                col.value(i),
                row.value(i),
                value.value(i),
            )
        })?;

        executor.finish(Arc::new(builder.finish()?))
    }
}

/// Copy `raster` into `builder` with the 1-based pixel (`col`, `row`) of the
/// 1-based band `band_num` set to `value`.
fn set_value(
    builder: &mut RasterBuilder,
    raster: &dyn RasterRef,
    band_num: usize,
    col: i64,
    row: i64,
    value: f64,
) -> Result<()> {
    let target = resolve_band(FUNC, raster, band_num)?;
    let data = set_pixel(target.as_ref(), col, row, value)?;

    builder.start_raster_from(raster, RasterOverrides::default())?;
    for band_idx in 0..raster.num_bands() {
        let band = raster.band(band_idx)?;
        let overrides = if band_idx + 1 == band_num {
            // The new bytes are the band's visible pixels, packed row-major, so
            // they carry an identity view over the visible shape.
            BandOverrides {
                data: Override::Set(&data),
                view: Override::Clear,
                source_shape: Some(band.shape()),
                ..Default::default()
            }
        } else {
            BandOverrides::default()
        };
        band.copy_into(builder, overrides)?;
        builder.finish_band()?;
    }
    builder.finish_raster()?;
    Ok(())
}

/// The band's visible pixels, packed row-major, with the 1-based pixel
/// (`col`, `row`) set to `value`.
fn set_pixel(band: &dyn BandRef, col: i64, row: i64, value: f64) -> Result<Buffer> {
    let buffer = spatial_2d_buffer(FUNC, band)?;
    let (height, width) = (buffer.shape[0], buffer.shape[1]);
    if !(1..=width).contains(&col) || !(1..=height).contains(&row) {
        return exec_err!(
            "{FUNC}: pixel ({col}, {row}) is outside the {width} x {height} grid \
             (colX and rowY are 1-based)"
        );
    }
    let pixel = pixel_bytes(value, &buffer.data_type)?;

    // A broadcast view can describe far more pixels than its source holds, so
    // size the packed output with checked arithmetic and reject it before
    // allocating rather than after.
    let len = usize::try_from(width)
        .ok()
        .and_then(|w| w.checked_mul(usize::try_from(height).ok()?))
        .and_then(|n| n.checked_mul(pixel.len()))
        .ok_or_else(|| {
            exec_datafusion_err!("{FUNC}: a {width} x {height} band is too large to materialise")
        })?;
    check_band_data_len(len).map_err(|e| exec_datafusion_err!("{FUNC}: {e}"))?;
    let mut data = Vec::with_capacity(len);
    match buffer.as_contiguous() {
        Ok(bytes) => data.extend_from_slice(bytes),
        // A strided, reversed or broadcast view: gather its visible pixels.
        Err(_) => scan_pixels(FUNC, &buffer, |_, _, bytes| {
            data.extend_from_slice(bytes);
            std::ops::ControlFlow::Continue(())
        })?,
    }
    let size = pixel.len();
    let start = ((row - 1) * width + (col - 1)) as usize * size;
    data[start..start + size].copy_from_slice(&pixel);
    Ok(Buffer::from_vec(data))
}

/// `value` packed as a pixel of `data_type`, via the shared
/// [`pixel_f64_to_bytes`] (Spark's truncation toward zero for integer bands).
/// A value that does not fit is an error, where Spark's Java cast would wrap
/// around instead.
fn pixel_bytes(value: f64, data_type: &BandDataType) -> Result<Vec<u8>> {
    pixel_f64_to_bytes(value, data_type)
        .map_err(|_| exec_datafusion_err!("{FUNC}: {value} does not fit a {data_type:?} pixel"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Float64Array, Int64Array};
    use datafusion_expr::ScalarUDF;
    use sedona_raster::array::RasterStructArray;
    use sedona_schema::datatypes::RASTER;
    use sedona_testing::raster_spec::{RasterSpec, assert_rasters_equal, raster_array};
    use sedona_testing::testers::ScalarUdfTester;

    fn i64_t() -> SedonaType {
        SedonaType::Arrow(DataType::Int64)
    }
    fn f64_t() -> SedonaType {
        SedonaType::Arrow(DataType::Float64)
    }

    /// Run the 5-argument form over one row per spec.
    fn set(
        specs: Vec<Option<RasterSpec>>,
        band: i64,
        col: i64,
        row: i64,
        value: f64,
    ) -> Result<arrow_array::ArrayRef> {
        let n = specs.len();
        let tester = ScalarUdfTester::new(
            rs_setvalue_udf().into(),
            vec![RASTER, i64_t(), i64_t(), i64_t(), f64_t()],
        );
        tester.invoke_arrays(vec![
            Arc::new(raster_array(specs)),
            Arc::new(Int64Array::from(vec![band; n])),
            Arc::new(Int64Array::from(vec![col; n])),
            Arc::new(Int64Array::from(vec![row; n])),
            Arc::new(Float64Array::from(vec![value; n])),
        ])
    }

    /// A 3x2 two-band raster: UInt8 band 1 (nodata 0), Float32 band 2.
    fn two_band() -> RasterSpec {
        RasterSpec::d2(3, 2)
            .band_values(&[1u8, 2, 3, 4, 5, 6])
            .nodata(0u8)
            .band_values(&[0.5f32, 1.5, 2.5, 3.5, 4.5, 5.5])
    }

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = rs_setvalue_udf().into();
        assert_eq!(udf.name(), "rs_setvalue");
        assert_eq!(
            rs_setvalue_udf()
                .metadata()
                .get(NEEDS_PIXELS_METADATA_KEY)
                .map(String::as_str),
            Some("true")
        );
    }

    #[test]
    fn sets_one_pixel_of_the_addressed_band() {
        // (col 3, row 2) is the last pixel, 1-based.
        let result = set(vec![Some(two_band())], 1, 3, 2, 42.0).unwrap();
        let expected = RasterSpec::d2(3, 2)
            .band_values(&[1u8, 2, 3, 4, 5, 42])
            .nodata(0u8)
            .band_values(&[0.5f32, 1.5, 2.5, 3.5, 4.5, 5.5]);
        assert_rasters_equal(&result, &[Some(expected)]);

        let result = set(vec![Some(two_band())], 2, 1, 1, -7.25).unwrap();
        let expected = RasterSpec::d2(3, 2)
            .band_values(&[1u8, 2, 3, 4, 5, 6])
            .nodata(0u8)
            .band_values(&[-7.25f32, 1.5, 2.5, 3.5, 4.5, 5.5]);
        assert_rasters_equal(&result, &[Some(expected)]);
    }

    #[test]
    fn overwrites_a_nodata_pixel_and_keeps_the_nodata_value() {
        let spec = RasterSpec::d2(2, 1).band_values(&[0u8, 9]).nodata(0u8);
        let result = set(vec![Some(spec)], 1, 1, 1, 5.0).unwrap();
        let expected = RasterSpec::d2(2, 1).band_values(&[5u8, 9]).nodata(0u8);
        assert_rasters_equal(&result, &[Some(expected)]);
    }

    #[test]
    fn untouched_bands_share_their_buffers() {
        let input = raster_array([Some(two_band())]);
        let tester = ScalarUdfTester::new(
            rs_setvalue_udf().into(),
            vec![RASTER, i64_t(), i64_t(), i64_t(), f64_t()],
        );
        let result = tester
            .invoke_arrays(vec![
                Arc::new(input.clone()),
                Arc::new(Int64Array::from(vec![1])),
                Arc::new(Int64Array::from(vec![1])),
                Arc::new(Int64Array::from(vec![1])),
                Arc::new(Float64Array::from(vec![9.0])),
            ])
            .unwrap();
        let result = result
            .as_any()
            .downcast_ref::<arrow_array::StructArray>()
            .unwrap();
        let before = RasterStructArray::try_new(&input).unwrap();
        let after = RasterStructArray::try_new(result).unwrap();
        let (before, after) = (before.get(0).unwrap(), after.get(0).unwrap());
        let band_ptr = |raster: &dyn RasterRef, i| {
            raster.band(i).unwrap().nd_buffer().unwrap().buffer.as_ptr()
        };
        assert_eq!(band_ptr(&before, 1), band_ptr(&after, 1));
        assert_ne!(band_ptr(&before, 0), band_ptr(&after, 0));
    }

    #[test]
    fn integer_values_truncate_toward_zero() {
        let spec = || RasterSpec::d2(1, 1).band_values(&[0i16]);
        let result = set(vec![Some(spec())], 1, 1, 1, 3.9).unwrap();
        assert_rasters_equal(&result, &[Some(RasterSpec::d2(1, 1).band_values(&[3i16]))]);
        let result = set(vec![Some(spec())], 1, 1, 1, -3.9).unwrap();
        assert_rasters_equal(&result, &[Some(RasterSpec::d2(1, 1).band_values(&[-3i16]))]);
        // A value just past the range truncates back into it.
        let result = set(vec![Some(spec())], 1, 1, 1, 32767.9).unwrap();
        assert_rasters_equal(
            &result,
            &[Some(RasterSpec::d2(1, 1).band_values(&[32767i16]))],
        );
    }

    #[test]
    fn values_outside_an_integer_type_error() {
        let spec = || Some(RasterSpec::d2(1, 1).band_values(&[0u8]));
        for value in [256.0, -1.0, f64::NAN, f64::INFINITY] {
            let err = set(vec![spec()], 1, 1, 1, value).unwrap_err().to_string();
            assert!(err.contains("does not fit a UInt8 pixel"), "{value}: {err}");
        }
    }

    #[test]
    fn int64_values_stop_at_the_exact_double_range() {
        // A 64-bit band takes integers up to ±2^53, the largest a double holds
        // exactly; beyond that the value is rejected rather than stored lossily.
        let spec = || Some(RasterSpec::d2(1, 1).band_values(&[0i64]));
        let limit = (1i64 << 53) as f64;
        let result = set(vec![spec()], 1, 1, 1, -limit).unwrap();
        assert_rasters_equal(
            &result,
            &[Some(RasterSpec::d2(1, 1).band_values(&[-(1i64 << 53)]))],
        );
        let err = set(vec![spec()], 1, 1, 1, 1e18).unwrap_err().to_string();
        assert!(err.contains("does not fit a Int64 pixel"), "{err}");
    }

    #[test]
    fn oversized_broadcast_band_errors_before_allocating() {
        // A 1x1 UInt16 source broadcast to 2^62 columns describes far more
        // bytes than one band can hold; it must error, not panic in the
        // allocator.
        use sedona_raster::builder::StartBandArgs;
        use sedona_raster::view_entries::{ViewEntries, ViewEntry};

        let width = 1i64 << 62;
        let view = ViewEntries::try_new(
            vec![
                ViewEntry {
                    source_axis: 0,
                    start: 0,
                    step: 0,
                    steps: 1,
                },
                ViewEntry {
                    source_axis: 1,
                    start: 0,
                    step: 0,
                    steps: width,
                },
            ],
            &[1, 1],
        )
        .unwrap();
        let mut builder = RasterBuilder::new(1);
        builder
            .start_raster_2d(width, 1, 0., 0., 1., -1., 0., 0., None)
            .unwrap();
        builder
            .start_band(StartBandArgs {
                view: Some(&view),
                ..StartBandArgs::new(&["y", "x"], &[1, 1], BandDataType::UInt16)
            })
            .unwrap();
        builder.band_data_writer().append_value(7u16.to_le_bytes());
        builder.finish_band().unwrap();
        builder.finish_raster().unwrap();

        let tester = ScalarUdfTester::new(
            rs_setvalue_udf().into(),
            vec![RASTER, i64_t(), i64_t(), f64_t()],
        );
        let err = tester
            .invoke_arrays(vec![
                Arc::new(builder.finish().unwrap()),
                Arc::new(Int64Array::from(vec![1])),
                Arc::new(Int64Array::from(vec![1])),
                Arc::new(Float64Array::from(vec![9.0])),
            ])
            .unwrap_err()
            .to_string();
        assert!(err.contains("RS_SetValue"), "{err}");
    }

    #[test]
    fn float32_rounds_to_nearest() {
        let spec = Some(RasterSpec::d2(1, 1).band_values(&[0f32]));
        let result = set(vec![spec], 1, 1, 1, 0.1).unwrap();
        assert_rasters_equal(
            &result,
            &[Some(RasterSpec::d2(1, 1).band_values(&[0.1f32]))],
        );
    }

    #[test]
    fn pixel_outside_the_grid_errors() {
        for (col, row) in [(0, 1), (1, 0), (4, 1), (1, 3)] {
            let err = set(vec![Some(two_band())], 1, col, row, 1.0)
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("outside the 3 x 2 grid"),
                "({col}, {row}): {err}"
            );
        }
    }

    #[test]
    fn band_out_of_range_errors() {
        for (band, expected) in [(0, "1-based"), (-1, "1-based"), (3, "out of range")] {
            let err = set(vec![Some(two_band())], band, 1, 1, 1.0)
                .unwrap_err()
                .to_string();
            assert!(err.contains(expected), "{band}: {err}");
        }
    }

    #[test]
    fn elided_band_only_on_a_single_band_raster() {
        let tester = ScalarUdfTester::new(
            rs_setvalue_udf().into(),
            vec![RASTER, i64_t(), i64_t(), f64_t()],
        );
        let args = |spec: RasterSpec| -> Vec<arrow_array::ArrayRef> {
            vec![
                Arc::new(raster_array([Some(spec)])),
                Arc::new(Int64Array::from(vec![2])),
                Arc::new(Int64Array::from(vec![1])),
                Arc::new(Float64Array::from(vec![7.0])),
            ]
        };
        let single = RasterSpec::d2(2, 1).band_values(&[1u8, 2]);
        let result = tester.invoke_arrays(args(single)).unwrap();
        assert_rasters_equal(
            &result,
            &[Some(RasterSpec::d2(2, 1).band_values(&[1u8, 7]))],
        );

        let err = tester
            .invoke_arrays(args(two_band()))
            .unwrap_err()
            .to_string();
        assert!(err.contains("specify which band"), "{err}");
    }

    #[test]
    fn null_arguments_give_a_null_raster() {
        let tester = ScalarUdfTester::new(
            rs_setvalue_udf().into(),
            vec![RASTER, i64_t(), i64_t(), i64_t(), f64_t()],
        );
        let result = tester
            .invoke_arrays(vec![
                Arc::new(raster_array([
                    None,
                    Some(two_band()),
                    Some(two_band()),
                    Some(two_band()),
                    Some(two_band()),
                ])),
                Arc::new(Int64Array::from(vec![
                    Some(1),
                    None,
                    Some(1),
                    Some(1),
                    Some(1),
                ])),
                Arc::new(Int64Array::from(vec![
                    Some(1),
                    Some(1),
                    None,
                    Some(1),
                    Some(1),
                ])),
                Arc::new(Int64Array::from(vec![
                    Some(1),
                    Some(1),
                    Some(1),
                    None,
                    Some(1),
                ])),
                Arc::new(Float64Array::from(vec![
                    Some(1.0),
                    Some(1.0),
                    Some(1.0),
                    Some(1.0),
                    None,
                ])),
            ])
            .unwrap();
        assert_rasters_equal(&result, &[None, None, None, None, None]);
    }

    #[test]
    fn non_2d_band_errors() {
        let spec = RasterSpec::d2(2, 1).band_values_nd(&["time", "y", "x"], &[1, 1, 2], &[0u8, 0]);
        let err = set(vec![Some(spec)], 1, 1, 1, 1.0).unwrap_err().to_string();
        assert!(err.contains("2-D"), "{err}");
    }
}
