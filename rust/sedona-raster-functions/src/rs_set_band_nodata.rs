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
//! RS_SetBandNoDataValue(raster, nodata)                 -> Raster  -- single-band rasters only
//! RS_SetBandNoDataValue(raster, band, nodata)           -> Raster
//! RS_SetBandNoDataValue(raster, band, nodata, replace)  -> Raster
//! ```
//!
//! With `replace` true, every pixel in the addressed band currently equal to
//! that band's nodata is rewritten to the new sentinel before it is declared —
//! so the pixels that read as nodata before the call still read as nodata
//! after it. The band must already have a nodata value to replace; without one
//! there is nothing to match, and Sedona Spark raises there too. Other bands
//! are untouched. A null `nodata` clears the band as it does in the 3-argument
//! form and `replace` is moot (there is no new sentinel to rewrite pixels to),
//! so no pixel is touched.
//!
//! `replace` rewrites pixels, so unlike the metadata-only forms it cannot share
//! the source buffer: the addressed band is materialized. That needs a
//! contiguous view — a strided band errors, pointing at `RS_EnsureContiguous`.
//!
//! The setter companion to the `RS_BandNoDataValue` getter. `nodata` is a double
//! packed into the band's native data type. A null raster or band yields a null
//! raster (matching `RS_SetCRS`/`RS_SetSRID`); a null `nodata` value instead
//! **clears** the addressed band's nodata — the raster is rebuilt with that band
//! carrying no nodata sentinel, rather than nulling the whole row.
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

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int64Type};
use arrow_schema::DataType;
use datafusion_common::error::Result;
use datafusion_common::exec_err;
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::builder::{RasterBuilder, RasterOverrides, StartBandArgs};
use sedona_raster::traits::{BandOverrides, BandRef, Override, RasterRef, nodata_f64_to_bytes};
use sedona_schema::datatypes::SedonaType;
use sedona_schema::matchers::ArgMatcher;

use crate::executor::RasterExecutor;
use crate::rs_ensure_loaded::NEEDS_PIXELS_METADATA_KEY;

/// RS_SetBandNoDataValue() scalar UDF implementation
pub fn rs_set_band_nodata_value_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_setbandnodatavalue",
        vec![
            Arc::new(RsSetBandNoDataValue {
                with_band: false,
                with_replace: false,
            }),
            Arc::new(RsSetBandNoDataValue {
                with_band: true,
                with_replace: false,
            }),
            Arc::new(RsSetBandNoDataValue {
                with_band: true,
                with_replace: true,
            }),
        ],
        Volatility::Immutable,
    )
    // The replace kernel rewrites pixel bytes, so the raster argument must be
    // materialised InDb first; the planner injects RS_EnsureLoaded on this flag.
    .with_metadata(NEEDS_PIXELS_METADATA_KEY, "true")
}

#[derive(Debug)]
struct RsSetBandNoDataValue {
    with_band: bool,
    /// Matches the 4-argument form, whose trailing boolean asks for the old
    /// nodata pixels to be rewritten. Only ever set together with `with_band`:
    /// Sedona Spark spells this overload with an explicit band.
    with_replace: bool,
}

impl SedonaScalarKernel for RsSetBandNoDataValue {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let mut matchers = vec![ArgMatcher::is_raster()];
        if self.with_band {
            matchers.push(ArgMatcher::is_integer());
        }
        matchers.push(ArgMatcher::is_numeric());
        if self.with_replace {
            matchers.push(ArgMatcher::is_boolean());
        }
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

        // The replace flag, read per row like the others so it may be a column.
        let replace_array = if self.with_replace {
            Some(
                args[3]
                    .clone()
                    .cast_to(&DataType::Boolean, None)?
                    .into_array(n)?,
            )
        } else {
            None
        };
        let replace_values = replace_array.as_ref().map(|a| a.as_boolean());

        let mut builder = RasterBuilder::new(n);
        executor.execute_raster_void(|i, raster_opt| {
            let null_out = |b: &mut RasterBuilder| b.append_null().map_err(Into::into);

            // A null raster, band, or value yields a null raster.
            let Some(raster) = raster_opt else {
                return null_out(&mut builder);
            };
            let band = match band_values {
                Some(bands) if bands.is_null(i) => return null_out(&mut builder),
                Some(bands) => Some(bands.value(i)),
                None => None,
            };
            // A null `nodata` value is not a null output: it flows through as
            // `None` and clears the addressed band's nodata.
            let value = if value_values.is_null(i) {
                None
            } else {
                Some(value_values.value(i))
            };
            let replace = match replace_values {
                Some(flags) if flags.is_null(i) => return null_out(&mut builder),
                Some(flags) => flags.value(i),
                None => false,
            };
            set_band_nodata(&mut builder, raster, band, value, replace)
        })?;

        executor.finish(Arc::new(builder.finish()?))
    }
}

/// Copy `raster` into `builder`, overriding the addressed (1-based) band's
/// nodata: `Some(value)` packs `value` into that band's data type, `None`
/// clears the band's nodata. Every other band is copied with its data shared
/// and metadata (including its own nodata) inherited.
///
/// `band` is `None` for the 2-argument form (no band given): it defaults to band
/// 1 only when the raster is single-band, and errors on a multiband raster so a
/// caller can't silently set nodata on just band 1.
fn set_band_nodata(
    builder: &mut RasterBuilder,
    raster: &dyn RasterRef,
    band: Option<i64>,
    value: Option<f64>,
    replace: bool,
) -> Result<()> {
    let num_bands = raster.num_bands();
    let band = match band {
        Some(band) => band,
        None if num_bands == 1 => 1,
        None => {
            return exec_err!(
                "RS_SetBandNoDataValue: raster has {num_bands} bands; specify which band to set \
                 (the 2-argument form is only allowed for a single-band raster)"
            );
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
        let addressed = band_idx + 1 == band as usize;
        // Pack the addressed band's new nodata bytes into a scratch binding the
        // override borrows from. A null value leaves this `None` so the override
        // clears the band's nodata rather than setting it.
        let new_nodata: Option<Vec<u8>> = match value {
            Some(v) if addressed => Some(nodata_f64_to_bytes(v, &band_ref.data_type())?),
            _ => None,
        };
        // `replace` rewrites pixels, so the addressed band is rebuilt from
        // materialized bytes instead of sharing the source buffer. A null
        // `value` falls through to the metadata-only path below: there is no
        // new sentinel to rewrite pixels to, so the band is simply cleared.
        let rewrite = new_nodata.as_deref().filter(|_| addressed && replace);
        if let Some(new_bytes) = rewrite {
            let pixels = replace_nodata_pixels(band_ref.as_ref(), new_bytes)?;
            let dim_names = band_ref.dim_names();
            let shape = band_ref.shape().to_vec();
            builder.start_band(StartBandArgs {
                name: band_ref.name(),
                nodata: Some(new_bytes),
                ..StartBandArgs::new(&dim_names, &shape, band_ref.data_type())
            })?;
            builder.band_data_writer().append_value(&pixels);
            builder.finish_band()?;
            continue;
        }

        // Only the addressed band's nodata is touched (`Set` or `Clear`); every
        // other band keeps its own.
        let nodata = if addressed {
            match new_nodata.as_deref() {
                Some(bytes) => Override::Set(bytes),
                None => Override::Clear,
            }
        } else {
            Override::Keep
        };
        band_ref.copy_into(
            builder,
            BandOverrides {
                nodata,
                ..Default::default()
            },
        )?;
        builder.finish_band()?;
    }

    builder.finish_raster()?;
    Ok(())
}

/// `band`'s visible bytes with every pixel equal to its current nodata
/// rewritten to `new_nodata`.
///
/// The comparison is on the raw little-endian bytes rather than on decoded
/// values, which makes it exact for every band data type without a dispatch
/// over them — `new_nodata` arrives already packed into this band's type by
/// `nodata_f64_to_bytes`, so both sides are the same width by construction.
/// The one behavior this inherits from byte equality is that a NaN sentinel
/// matches only pixels carrying the identical NaN bit pattern; Sedona Spark
/// compares numerically, where NaN never equals itself and so never matches.
fn replace_nodata_pixels(band: &dyn BandRef, new_nodata: &[u8]) -> Result<Vec<u8>> {
    let Some(old_nodata) = band.nodata() else {
        return exec_err!(
            "RS_SetBandNoDataValue: replace requires the band to already have a nodata value \
             to replace, but this band has none"
        );
    };

    let width = band.data_type().byte_size();
    if old_nodata.len() != width || new_nodata.len() != width {
        return exec_err!(
            "RS_SetBandNoDataValue: nodata width mismatch for a {width}-byte band \
             (existing {} bytes, new {} bytes)",
            old_nodata.len(),
            new_nodata.len()
        );
    }

    let buffer = band.nd_buffer()?;
    let mut pixels = buffer.as_contiguous()?.to_vec();
    for pixel in pixels.chunks_exact_mut(width) {
        if pixel == old_nodata {
            pixel.copy_from_slice(new_nodata);
        }
    }
    Ok(pixels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::ArrayRef;
    use arrow_schema::DataType;
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use sedona_schema::datatypes::RASTER;
    use sedona_testing::raster_spec::{RasterSpec, assert_rasters_equal};
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

    fn tester_4arg() -> ScalarUdfTester {
        let udf: ScalarUDF = rs_set_band_nodata_value_udf().into();
        ScalarUdfTester::new(
            udf,
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Int32),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Boolean),
            ],
        )
    }

    /// `tester_4arg` takes the raster as an array and the rest as scalars.
    fn invoke_4arg(spec: RasterSpec, band: i32, value: f64, replace: bool) -> Result<ArrayRef> {
        let result = tester_4arg().invoke(vec![
            ColumnarValue::Array(Arc::new(spec.build())),
            ColumnarValue::Scalar(ScalarValue::Int32(Some(band))),
            ColumnarValue::Scalar(ScalarValue::Float64(Some(value))),
            ColumnarValue::Scalar(ScalarValue::Boolean(Some(replace))),
        ])?;
        match result {
            ColumnarValue::Array(array) => Ok(array),
            _ => unreachable!("array input yields an array result"),
        }
    }

    #[test]
    fn replace_rewrites_old_nodata_pixels_in_the_addressed_band_only() {
        // Band 1 carries nodata 2 on a pixel holding 2; band 2 carries nodata 3
        // on a pixel holding 3. Replacing band 1's sentinel with 9 rewrites its
        // 2 to 9 and leaves band 2 — pixels and sentinel both — untouched.
        let input = RasterSpec::d2(2, 1)
            .band_values(&[1u8, 2])
            .nodata(2u8)
            .band_values(&[3u8, 4])
            .nodata(3u8);
        let result = invoke_4arg(input, 1, 9.0, true).unwrap();
        let expected = RasterSpec::d2(2, 1)
            .band_values(&[1u8, 9])
            .nodata(9u8)
            .band_values(&[3u8, 4])
            .nodata(3u8);
        assert_rasters_equal(&result, &[Some(expected)]);
    }

    #[test]
    fn replace_false_leaves_pixels_alone() {
        // The same call with replace=false is the 3-arg behavior: the sentinel
        // moves to 9 but the pixel holding the old sentinel keeps its value, so
        // it stops reading as nodata.
        let input = RasterSpec::d2(2, 1).band_values(&[1u8, 2]).nodata(2u8);
        let result = invoke_4arg(input, 1, 9.0, false).unwrap();
        let expected = RasterSpec::d2(2, 1).band_values(&[1u8, 2]).nodata(9u8);
        assert_rasters_equal(&result, &[Some(expected)]);
    }

    #[test]
    fn replace_without_an_existing_nodata_errors() {
        // Nothing to match against — Sedona Spark raises here too.
        let err = invoke_4arg(two_band(), 1, 9.0, true).unwrap_err();
        assert!(
            err.message().contains("nodata value"),
            "unexpected message: {err}"
        );
    }

    #[test]
    fn replace_rewrites_every_matching_pixel() {
        // More than one pixel carries the sentinel, and a pixel that merely
        // looks similar (1) is left alone.
        let input = RasterSpec::d2(4, 1)
            .band_values(&[2u8, 1, 2, 5])
            .nodata(2u8);
        let result = invoke_4arg(input, 1, 9.0, true).unwrap();
        let expected = RasterSpec::d2(4, 1)
            .band_values(&[9u8, 1, 9, 5])
            .nodata(9u8);
        assert_rasters_equal(&result, &[Some(expected)]);
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
    fn null_value_clears_band_nodata() {
        // A null nodata value clears the addressed band's nodata rather than
        // nulling the whole raster: the single band starts with nodata 5 and
        // ends with none, every other field preserved.
        let one_band = RasterSpec::d2(2, 1).band_values(&[1u8, 2]).nodata(5u8);
        let result = tester_2arg()
            .invoke_array_scalar(Arc::new(one_band.build()), ScalarValue::Float64(None))
            .unwrap();
        let expected = RasterSpec::d2(2, 1).band_values(&[1u8, 2]);
        assert_rasters_equal(&result, &[Some(expected)]);
    }

    #[test]
    fn null_value_clears_only_the_addressed_band_nodata() {
        // 3-arg form on a multiband raster: clearing band 2's nodata leaves
        // band 1's nodata untouched.
        let input = RasterSpec::d2(2, 1)
            .band_values(&[1u8, 2])
            .nodata(7u8)
            .band_values(&[3u8, 4])
            .nodata(8u8);
        let result = tester_3arg()
            .invoke_array_scalar_scalar(Arc::new(input.build()), 2_i32, ScalarValue::Float64(None))
            .unwrap();
        let expected = RasterSpec::d2(2, 1)
            .band_values(&[1u8, 2])
            .nodata(7u8)
            .band_values(&[3u8, 4]);
        assert_rasters_equal(&result, &[Some(expected)]);
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
