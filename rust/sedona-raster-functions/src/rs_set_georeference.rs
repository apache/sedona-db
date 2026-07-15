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

//! `RS_SetGeoReference` — set a raster's affine geotransform.
//!
//! ```text
//! RS_SetGeoReference(raster, georef)          -> Raster  -- GDAL format
//! RS_SetGeoReference(raster, georef, format)  -> Raster
//! ```
//!
//! The setter companion to [`rs_georeference`](crate::rs_georeference): `georef`
//! is a world-file string of six whitespace-separated numbers in the same order
//! the getter emits — `scaleX skewY skewX scaleY upperLeftX upperLeftY`. The
//! `format` is `GDAL` (default, alias `PIXEL`) or `ESRI` (alias `NODE`); in the
//! ESRI/center convention the upper-left coordinates refer to the *center* of
//! the upper-left pixel and are shifted back to the pixel corner on the way in.
//! The shift uses the full affine (scale and skew halves), so it is exact for
//! skewed rasters and round-trips with the getter.
//!
//! The raster is rebuilt with [`RasterBuilder::copy_raster_from`] overriding the
//! transform; each band's pixel buffers are shared zero-copy, and the CRS is
//! carried over unchanged.

use std::sync::Arc;

use arrow_array::Array;
use arrow_schema::DataType;
use datafusion_common::cast::as_string_array;
use datafusion_common::error::Result;
use datafusion_common::{arrow_datafusion_err, exec_datafusion_err, exec_err};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_raster::builder::{RasterBuilder, RasterOverrides};
use sedona_schema::datatypes::SedonaType;
use sedona_schema::matchers::ArgMatcher;

use crate::executor::RasterExecutor;
// Shared with the getter so the two agree on accepted format strings and the
// ESRI center-shift convention (see RS_GeoReference).
use crate::rs_georeference::GeoReferenceFormat;

/// RS_SetGeoReference() scalar UDF implementation
pub fn rs_set_georeference_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "rs_setgeoreference",
        vec![
            Arc::new(RsSetGeoReference { with_format: false }),
            Arc::new(RsSetGeoReference { with_format: true }),
        ],
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct RsSetGeoReference {
    with_format: bool,
}

impl SedonaScalarKernel for RsSetGeoReference {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let mut matchers = vec![ArgMatcher::is_raster(), ArgMatcher::is_string()];
        if self.with_format {
            matchers.push(ArgMatcher::is_string());
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

        // Expand the georeference string (and optional format) to row arrays.
        let georef_array = args[1]
            .clone()
            .cast_to(&DataType::Utf8, None)?
            .into_array(n)?;
        let georef_array = as_string_array(&georef_array)?;

        let format_array = if self.with_format {
            Some(
                args[2]
                    .clone()
                    .cast_to(&DataType::Utf8, None)?
                    .into_array(n)?,
            )
        } else {
            None
        };
        let format_array = format_array
            .as_ref()
            .map(|a| as_string_array(a))
            .transpose()?;

        let mut builder = RasterBuilder::new(n);
        executor.execute_raster_void(|i, raster_opt| {
            let null_out =
                |b: &mut RasterBuilder| b.append_null().map_err(|e| arrow_datafusion_err!(e));

            // A null georef or format is a null *input* and yields a null raster
            // (matching RS_SetCRS); check those first so a null-driven row never
            // raises an error.
            if georef_array.is_null(i) {
                return null_out(&mut builder);
            }
            let format = match &format_array {
                Some(fmt) if fmt.is_null(i) => return null_out(&mut builder),
                Some(fmt) => GeoReferenceFormat::from_str(fmt.value(i))?,
                None => GeoReferenceFormat::Gdal,
            };

            // Parse and validate the georef up front — before the null-raster
            // check — so an invalid (non-null) georef errors for every row,
            // rather than being masked on the rows that happen to align with a
            // null raster.
            let transform = parse_georeference(georef_array.value(i), format)?;

            let Some(raster) = raster_opt else {
                return null_out(&mut builder);
            };
            builder
                .copy_raster_from(
                    raster,
                    RasterOverrides {
                        transform: Some(transform),
                    },
                )
                .map_err(|e| arrow_datafusion_err!(e))
        })?;

        executor.finish(Arc::new(
            builder.finish().map_err(|e| arrow_datafusion_err!(e))?,
        ))
    }
}

/// Parse a world-file georeference string into a GDAL-order transform
/// `[upper_left_x, scale_x, skew_x, upper_left_y, skew_y, scale_y]`.
///
/// The input lists six whitespace-separated numbers in world-file order
/// (`scaleX skewY skewX scaleY upperLeftX upperLeftY`); for ESRI the upper-left
/// coordinates are pixel-center and shifted back to the corner.
///
/// The stringly-typed six-value format is a bit unwieldy; a future dedicated
/// affine-matrix type (cf. `sedona_raster::affine_transformation::AffineMatrix`,
/// which could also back `ST_Affine`) would be a cleaner input.
fn parse_georeference(georef: &str, format: GeoReferenceFormat) -> Result<[f64; 6]> {
    let values: Vec<f64> = georef
        .split_whitespace()
        .map(|token| {
            token
                .parse::<f64>()
                .map_err(|_| exec_datafusion_err!("RS_SetGeoReference: invalid number '{token}'"))
        })
        .collect::<Result<_>>()?;

    if values.len() != 6 {
        return exec_err!(
            "RS_SetGeoReference: expected 6 whitespace-separated values \
             (scaleX skewY skewX scaleY upperLeftX upperLeftY), got {}",
            values.len()
        );
    }

    let [scale_x, skew_y, skew_x, scale_y, mut upper_left_x, mut upper_left_y] = [
        values[0], values[1], values[2], values[3], values[4], values[5],
    ];

    if format == GeoReferenceFormat::Esri {
        // ESRI reports the upper-left *pixel center*, i.e. the world coordinates
        // of pixel-space (0.5, 0.5); shift back to the corner through the full
        // affine so skewed rasters stay exact (the inverse of the getter's shift).
        upper_left_x -= (scale_x + skew_x) * 0.5;
        upper_left_y -= (skew_y + scale_y) * 0.5;
    }

    // GDAL geotransform order.
    Ok([upper_left_x, scale_x, skew_x, upper_left_y, skew_y, scale_y])
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::ArrayRef;
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use sedona_schema::datatypes::RASTER;
    use sedona_testing::raster_spec::{assert_rasters_equal, raster_array, RasterSpec};
    use sedona_testing::testers::ScalarUdfTester;

    /// A 2x2 raster with a CRS — so the comparisons confirm the CRS (and pixels)
    /// survive the transform swap.
    fn base() -> RasterSpec {
        RasterSpec::d2(2, 2)
            .band_values(&[1u8, 2, 3, 4])
            .crs(Some("OGC:CRS84"))
    }

    fn tester_2arg() -> ScalarUdfTester {
        let udf: ScalarUDF = rs_set_georeference_udf().into();
        ScalarUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Utf8)])
    }

    fn tester_3arg() -> ScalarUdfTester {
        let udf: ScalarUDF = rs_set_georeference_udf().into();
        ScalarUdfTester::new(
            udf,
            vec![
                RASTER,
                SedonaType::Arrow(DataType::Utf8),
                SedonaType::Arrow(DataType::Utf8),
            ],
        )
    }

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = rs_set_georeference_udf().into();
        assert_eq!(udf.name(), "rs_setgeoreference");
    }

    #[test]
    fn sets_transform_preserving_crs_and_bands() {
        // World-file order: scaleX skewY skewX scaleY upperLeftX upperLeftY.
        let tester = tester_2arg();
        tester.assert_return_type(RASTER);

        let result = tester
            .invoke_array_scalar(Arc::new(base().build()), "2 0.5 0.25 -3 100 200")
            .unwrap();
        // GDAL-order transform: [upperLeftX, scaleX, skewX, upperLeftY, skewY, scaleY].
        // CRS and pixels must be untouched — assert_rasters_equal checks all of it.
        let expected = base().transform([100.0, 2.0, 0.25, 200.0, 0.5, -3.0]);
        assert_rasters_equal(&result, &[Some(expected)]);
    }

    #[test]
    fn esri_shifts_upper_left_to_corner() {
        // ESRI upper-left is the pixel center; the stored (GDAL) corner is
        // center - scale*0.5: x = 101 - 2*0.5 = 100, y = 198.5 - (-3)*0.5 = 200.
        let result = tester_3arg()
            .invoke_array_scalar_scalar(Arc::new(base().build()), "2 0 0 -3 101 198.5", "ESRI")
            .unwrap();
        let expected = base().transform([100.0, 2.0, 0.0, 200.0, 0.0, -3.0]);
        assert_rasters_equal(&result, &[Some(expected)]);
    }

    #[test]
    fn null_georef_nulls_raster() {
        let result = tester_2arg()
            .invoke_array_scalar(Arc::new(base().build()), ScalarValue::Utf8(None))
            .unwrap();
        assert_rasters_equal(&result, &[None]);
    }

    #[test]
    fn invalid_georef_errors_even_for_null_raster() {
        // A malformed (non-null) georef errors for the whole batch even when its
        // row's raster is null — invalid georeferences are always reported, not
        // masked by a null raster (only a null *georef* yields a null raster).
        let rasters = raster_array([None, Some(base())]);
        let georefs: ArrayRef = Arc::new(arrow_array::StringArray::from(vec![
            Some("not a georeference"),
            Some("2 0 0 -3 100 200"),
        ]));

        let err = tester_2arg()
            .invoke_array_array(Arc::new(rasters), georefs)
            .unwrap_err()
            .to_string();
        assert!(err.contains("invalid number"), "unexpected error: {err}");
    }

    #[test]
    fn esri_skewed_shifts_through_full_affine() {
        // For a skewed georeference the center-to-corner shift includes the skew
        // halves: x = 100 - (2 + 0.25)*0.5 = 98.875, y = 200 - (0.5 - 3)*0.5 = 201.25.
        let result = tester_3arg()
            .invoke_array_scalar_scalar(Arc::new(base().build()), "2 0.5 0.25 -3 100 200", "ESRI")
            .unwrap();
        let expected = base().transform([98.875, 2.0, 0.25, 201.25, 0.5, -3.0]);
        assert_rasters_equal(&result, &[Some(expected)]);
    }

    #[test]
    fn esri_round_trips_skewed_raster_with_getter() {
        // RS_GeoReference(r, 'ESRI') fed back through RS_SetGeoReference(..., 'ESRI')
        // must reproduce the original transform exactly, skew included.
        let transform = [100.0, 2.0, 0.25, 200.0, 0.5, -3.0];
        let skewed = Arc::new(raster_array([Some(base().transform(transform))]));

        let getter: ScalarUDF = crate::rs_georeference::rs_georeference_udf().into();
        let getter_tester =
            ScalarUdfTester::new(getter, vec![RASTER, SedonaType::Arrow(DataType::Utf8)]);
        let georef = getter_tester
            .invoke_array_scalar(skewed.clone(), "ESRI")
            .unwrap();
        let georef = as_string_array(&georef).unwrap().value(0).to_string();

        let result = tester_3arg()
            .invoke_array_scalar_scalar(skewed, georef, "ESRI")
            .unwrap();
        assert_rasters_equal(&result, &[Some(base().transform(transform))]);
    }

    #[test]
    fn node_alias_sets_transform_like_esri() {
        // "NODE" is an alias for the ESRI/center convention.
        let result = tester_3arg()
            .invoke_array_scalar_scalar(Arc::new(base().build()), "2 0 0 -3 101 198.5", "NODE")
            .unwrap();
        let expected = base().transform([100.0, 2.0, 0.0, 200.0, 0.0, -3.0]);
        assert_rasters_equal(&result, &[Some(expected)]);
    }

    #[test]
    fn wrong_value_count_errors() {
        let err = tester_2arg()
            .invoke_array_scalar(Arc::new(base().build()), "1 2 3")
            .unwrap_err()
            .to_string();
        assert!(err.contains("6"), "unexpected error: {err}");
    }

    #[test]
    fn invalid_format_errors() {
        let err = tester_3arg()
            .invoke_array_scalar_scalar(Arc::new(base().build()), "2 0 0 -3 100 200", "NOPE")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("GeoReference format"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn null_georef_with_invalid_format_is_null_not_error() {
        // A null georef short-circuits to a null raster before the (invalid)
        // format string is validated — a null-driving input never errors.
        let result = tester_3arg()
            .invoke_array_scalar_scalar(Arc::new(base().build()), ScalarValue::Utf8(None), "NOPE")
            .unwrap();
        assert_rasters_equal(&result, &[None]);
    }
}
