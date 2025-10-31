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

use arrow_array::builder::BinaryBuilder;
use arrow_schema::DataType;
use datafusion_common::cast::as_float64_array;
use datafusion_common::error::Result;
use datafusion_common::{DataFusionError, ScalarValue};
use datafusion_expr::ColumnarValue;
use geos::{BufferParams, CapStyle, Geom, JoinStyle};
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_geometry::wkb_factory::WKB_MIN_PROBABLE_BYTES;
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

use crate::executor::GeosExecutor;

/// ST_Buffer() implementation using the geos crate
///
/// Supports two signatures:
/// - ST_Buffer(geometry: Geometry, distance: Double)
/// - ST_Buffer(geometry: Geometry, distance: Double, bufferStyleParameters: String)
///
/// Buffer style parameters format: "key1=value1 key2=value2 ..."
/// Supported parameters:
/// - endcap: round, flat/butt, square
/// - join: round, mitre/miter, bevel
/// - side: both, left, right
/// - mitre_limit/miter_limit: numeric value
/// - quad_segs/quadrant_segments: integer value
pub fn st_buffer_impl() -> ScalarKernelRef {
    Arc::new(STBuffer {})
}

#[derive(Debug)]
struct STBuffer {}

impl SedonaScalarKernel for STBuffer {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry(), ArgMatcher::is_numeric()],
            WKB_GEOMETRY,
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        invoke_batch_impl(arg_types, args)
    }
}

pub fn st_buffer_style_impl() -> ScalarKernelRef {
    Arc::new(STBufferStyle {})
}
#[derive(Debug)]
struct STBufferStyle {}

impl SedonaScalarKernel for STBufferStyle {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_geometry(),
                ArgMatcher::is_numeric(),
                ArgMatcher::is_string(),
            ],
            WKB_GEOMETRY,
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        invoke_batch_impl(arg_types, args)
    }
}

fn invoke_batch_impl(arg_types: &[SedonaType], args: &[ColumnarValue]) -> Result<ColumnarValue> {
    let executor = GeosExecutor::new(arg_types, args);
    let mut builder = BinaryBuilder::with_capacity(
        executor.num_iterations(),
        WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
    );

    // Extract Args
    let distance_value = args[1]
        .cast_to(&DataType::Float64, None)?
        .to_array(executor.num_iterations())?;
    let distance_array = as_float64_array(&distance_value)?;
    let mut distance_iter = distance_array.iter();

    let buffer_style_params = extract_optional_string(args.get(2))?;

    // Build BufferParams based on style parameters
    let params = parse_buffer_params(buffer_style_params.as_deref())?;

    // Parse 'side' from the style parameters
    let (is_left, is_right) = parse_buffer_side_style(buffer_style_params.as_deref());

    executor.execute_wkb_void(|wkb| {
        match (wkb, distance_iter.next().unwrap()) {
            (Some(wkb), Some(mut distance)) => {
                if (is_left && distance < 0.0) || (is_right && distance > 0.0) {
                    distance = -distance;
                }
                invoke_scalar(&wkb, distance, &params, &mut builder)?;
                builder.append_value([]);
            }
            _ => builder.append_null(),
        }
        Ok(())
    })?;

    executor.finish(Arc::new(builder.finish()))
}

fn invoke_scalar(
    geos_geom: &geos::Geometry,
    distance: f64,
    params: &BufferParams,
    writer: &mut impl std::io::Write,
) -> Result<()> {
    let geometry = geos_geom
        .buffer_with_params(distance, params)
        .map_err(|e| DataFusionError::External(Box::new(e)))?;

    let wkb = geometry
        .to_wkb()
        .map_err(|e| DataFusionError::Execution(format!("Failed to convert to wkb: {e}")))?;

    writer.write_all(wkb.as_ref())?;
    Ok(())
}

fn extract_optional_string(arg: Option<&ColumnarValue>) -> Result<Option<String>> {
    let Some(arg) = arg else { return Ok(None) };
    let casted = arg.cast_to(&DataType::Utf8, None)?;
    match &casted {
        ColumnarValue::Scalar(ScalarValue::Utf8(Some(s)) | ScalarValue::LargeUtf8(Some(s))) => {
            Ok(Some(s.clone()))
        }
        ColumnarValue::Scalar(scalar) if scalar.is_null() => Ok(None),
        ColumnarValue::Scalar(_) => Ok(None),
        _ => Err(DataFusionError::Execution(format!(
            "Expected scalar bufferStyleParameters, got: {arg:?}",
        ))),
    }
}

fn parse_buffer_side_style(params: Option<&str>) -> (bool, bool) {
    params
        .map(|s| {
            let mut left = false;
            let mut right = false;
            for tok in s.split_whitespace() {
                if let Some((k, v)) = tok.split_once('=') {
                    if k.eq_ignore_ascii_case("side") {
                        if v.eq_ignore_ascii_case("left") {
                            left = true;
                            right = false;
                        } else if v.eq_ignore_ascii_case("right") {
                            right = true;
                            left = false;
                        }
                    }
                }
            }
            (left, right)
        })
        .unwrap_or((false, false))
}

fn parse_buffer_params(params_str: Option<&str>) -> Result<BufferParams> {
    let Some(params_str) = params_str else {
        return BufferParams::builder()
            .build()
            .map_err(|e| DataFusionError::External(Box::new(e)));
    };

    let mut params_builder = BufferParams::builder();
    let mut end_cap_specified = false;

    for param in params_str.split_whitespace() {
        let Some((key, value)) = param.split_once('=') else {
            return Err(DataFusionError::Execution(format!(
                "Missing value for buffer parameter: {param}",
            )));
        };

        if key.eq_ignore_ascii_case("endcap") {
            params_builder = params_builder.end_cap_style(parse_cap_style(value)?);
            end_cap_specified = true;
        } else if key.eq_ignore_ascii_case("join") {
            params_builder = params_builder.join_style(parse_join_style(value)?);
        } else if key.eq_ignore_ascii_case("side") {
            let single_sided = is_single_sided(value)?;
            if single_sided && !end_cap_specified {
                params_builder = params_builder.end_cap_style(CapStyle::Square);
            }
            params_builder = params_builder.single_sided(single_sided);
        } else if key.eq_ignore_ascii_case("mitre_limit") || key.eq_ignore_ascii_case("miter_limit")
        {
            let limit: f64 = parse_number(value, "mitre_limit")?;
            params_builder = params_builder.mitre_limit(limit);
        } else if key.eq_ignore_ascii_case("quad_segs")
            || key.eq_ignore_ascii_case("quadrant_segments")
        {
            let segs: i32 = parse_number(value, "quadrant_segments")?;
            params_builder = params_builder.quadrant_segments(segs);
        } else {
            return Err(DataFusionError::Execution(format!(
                "Invalid buffer parameter: {key} \
                (accept: 'endcap', 'join', 'mitre_limit', 'miter_limit', 'quad_segs', 'quadrant_segments' and 'side')",
            )));
        }
    }

    params_builder
        .build()
        .map_err(|e| DataFusionError::External(Box::new(e)))
}

fn parse_cap_style(value: &str) -> Result<CapStyle> {
    if value.eq_ignore_ascii_case("round") {
        Ok(CapStyle::Round)
    } else if value.eq_ignore_ascii_case("flat") || value.eq_ignore_ascii_case("butt") {
        Ok(CapStyle::Flat)
    } else if value.eq_ignore_ascii_case("square") {
        Ok(CapStyle::Square)
    } else {
        Err(DataFusionError::Execution(format!(
            "Invalid endcap style: '{value}'. Valid options: round, flat, butt, square",
        )))
    }
}

fn parse_join_style(value: &str) -> Result<JoinStyle> {
    if value.eq_ignore_ascii_case("round") {
        Ok(JoinStyle::Round)
    } else if value.eq_ignore_ascii_case("mitre") || value.eq_ignore_ascii_case("miter") {
        Ok(JoinStyle::Mitre)
    } else if value.eq_ignore_ascii_case("bevel") {
        Ok(JoinStyle::Bevel)
    } else {
        Err(DataFusionError::Execution(format!(
            "Invalid join style: '{value}'. Valid options: round, mitre, miter, bevel",
        )))
    }
}

fn is_single_sided(value: &str) -> Result<bool> {
    if value.eq_ignore_ascii_case("both") {
        Ok(false)
    } else if value.eq_ignore_ascii_case("left") || value.eq_ignore_ascii_case("right") {
        Ok(true)
    } else {
        Err(DataFusionError::Execution(format!(
            "Invalid side: '{value}'. Valid options: both, left, right",
        )))
    }
}

fn parse_number<T: std::str::FromStr>(value: &str, param_name: &str) -> Result<T> {
    value.parse().map_err(|_| {
        DataFusionError::Execution(format!(
            "Invalid {param_name} value: '{value}'. Expected a valid number",
        ))
    })
}

#[cfg(test)]
mod tests {
    use arrow_array::ArrayRef;
    use datafusion_common::ScalarValue;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::create::create_array;
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_buffer", st_buffer_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![sedona_type.clone(), SedonaType::Arrow(DataType::Float64)],
        );
        tester.assert_return_type(WKB_GEOMETRY);

        // Check the envelope of the buffers
        let envelope_udf = sedona_functions::st_envelope::st_envelope_udf();
        let envelope_tester = ScalarUdfTester::new(envelope_udf.into(), vec![WKB_GEOMETRY]);

        let buffer_result = tester.invoke_scalar_scalar("POINT (1 2)", 2.0).unwrap();
        let envelope_result = envelope_tester.invoke_scalar(buffer_result).unwrap();
        let expected_envelope = "POLYGON((-1 0, -1 4, 3 4, 3 0, -1 0))";
        tester.assert_scalar_result_equals(envelope_result, expected_envelope);

        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());

        let input_wkt = vec![None, Some("POINT (0 0)")];
        let input_dist = 1;
        let expected_envelope: ArrayRef = create_array(
            &[None, Some("POLYGON((-1 -1, -1 1, 1 1, 1 -1, -1 -1))")],
            &WKB_GEOMETRY,
        );
        let buffer_result = tester
            .invoke_wkb_array_scalar(input_wkt, input_dist)
            .unwrap();
        let envelope_result = envelope_tester.invoke_array(buffer_result).unwrap();
        assert_array_equal(&envelope_result, &expected_envelope);
    }

    #[rstest]
    fn udf_with_buffer_params(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_buffer", st_buffer_style_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![
                sedona_type.clone(),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Utf8),
            ],
        );
        tester.assert_return_type(WKB_GEOMETRY);

        // Envelope checks result in different values for different GEOS versions.
        // This test at least ensures that the buffer parameters are plugged in.
        let buffer_result_flat = tester
            .invoke_scalar_scalar_scalar("LINESTRING (0 0, 10 0)", 2.0, "endcap=flat".to_string())
            .unwrap();

        let buffer_result_square = tester
            .invoke_scalar_scalar_scalar("LINESTRING (0 0, 10 0)", 1.0, "endcap=square".to_string())
            .unwrap();

        assert_ne!(buffer_result_flat, buffer_result_square);
    }

    #[rstest]
    fn udf_with_quad_segs(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_buffer", st_buffer_style_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![
                sedona_type.clone(),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Utf8),
            ],
        );
        tester.assert_return_type(WKB_GEOMETRY);

        let envelope_udf = sedona_functions::st_envelope::st_envelope_udf();
        let envelope_tester = ScalarUdfTester::new(envelope_udf.into(), vec![WKB_GEOMETRY]);
        let input_wkt = "POINT (5 5)";
        let buffer_dist = 3.0;

        let buffer_result_default = tester
            .invoke_scalar_scalar_scalar(input_wkt, buffer_dist, "endcap=round".to_string())
            .unwrap();
        let envelope_result_default = envelope_tester
            .invoke_scalar(buffer_result_default)
            .unwrap();

        let expected_envelope = "POLYGON((2 2, 2 8, 8 8, 8 2, 2 2))";
        tester.assert_scalar_result_equals(envelope_result_default, expected_envelope);

        let buffer_result_low_segs = tester
            .invoke_scalar_scalar_scalar(
                input_wkt,
                buffer_dist,
                "quad_segs=1 endcap=round".to_string(),
            )
            .unwrap();
        let envelope_result_low_segs = envelope_tester
            .invoke_scalar(buffer_result_low_segs)
            .unwrap();
        tester.assert_scalar_result_equals(envelope_result_low_segs, expected_envelope);
    }

    #[test]
    fn test_parse_buffer_params_invalid_endcap() {
        let err = parse_buffer_params(Some("endcap=invalid")).err().unwrap();
        assert_eq!(
            err.message(),
            "Invalid endcap style: 'invalid'. Valid options: round, flat, butt, square"
        );
    }

    #[test]
    fn test_parse_buffer_params_invalid_join() {
        let err = parse_buffer_params(Some("join=invalid")).err().unwrap();
        assert_eq!(
            err.message(),
            "Invalid join style: 'invalid'. Valid options: round, mitre, miter, bevel"
        );
    }

    #[test]
    fn test_parse_buffer_params_invalid_side() {
        let err = parse_buffer_params(Some("side=invalid")).err().unwrap();
        assert_eq!(
            err.message(),
            "Invalid side: 'invalid'. Valid options: both, left, right"
        );
    }

    #[test]
    fn test_parse_buffer_params_invalid_mitre_limit() {
        let err = parse_buffer_params(Some("mitre_limit=not_a_number"))
            .err()
            .unwrap();
        assert_eq!(
            err.message(),
            "Invalid mitre_limit value: 'not_a_number'. Expected a valid number"
        );
    }

    #[test]
    fn test_parse_buffer_params_invalid_miter_limit() {
        let err = parse_buffer_params(Some("miter_limit=abc")).err().unwrap();
        assert_eq!(
            err.message(),
            "Invalid mitre_limit value: 'abc'. Expected a valid number"
        );
    }

    #[test]
    fn test_parse_buffer_params_invalid_quad_segs() {
        let err = parse_buffer_params(Some("quad_segs=not_an_int"))
            .err()
            .unwrap();
        assert_eq!(
            err.message(),
            "Invalid quadrant_segments value: 'not_an_int'. Expected a valid number"
        );
    }

    #[test]
    fn test_parse_buffer_params_invalid_quadrant_segments() {
        let err = parse_buffer_params(Some("quadrant_segments=xyz"))
            .err()
            .unwrap();
        assert_eq!(
            err.message(),
            "Invalid quadrant_segments value: 'xyz'. Expected a valid number"
        );
    }

    #[test]
    fn test_parse_buffer_params_multiple_invalid_params() {
        // Test that the first invalid parameter is caught
        let err = parse_buffer_params(Some("endcap=wrong join=mitre"))
            .err()
            .unwrap();
        assert_eq!(
            err.message(),
            "Invalid endcap style: 'wrong'. Valid options: round, flat, butt, square"
        );
    }

    #[test]
    fn test_parse_buffer_params_invalid_mixed_with_valid() {
        // Test invalid parameter after valid ones
        let err = parse_buffer_params(Some("endcap=round join=invalid"))
            .err()
            .unwrap();
        assert_eq!(
            err.message(),
            "Invalid join style: 'invalid'. Valid options: round, mitre, miter, bevel"
        );
    }

    #[test]
    fn test_parse_buffer_params_invalid_param_name() {
        let err = parse_buffer_params(Some("unknown_param=value"))
            .err()
            .unwrap();
        assert_eq!(
            err.message(),
            "Invalid buffer parameter: unknown_param (accept: 'endcap', 'join', 'mitre_limit', 'miter_limit', 'quad_segs', 'quadrant_segments' and 'side')"
        );
    }

    #[test]
    fn test_parse_buffer_params_missing_value() {
        let err = parse_buffer_params(Some("endcap=round bare_param join=mitre"))
            .err()
            .unwrap();
        assert_eq!(
            err.message(),
            "Missing value for buffer parameter: bare_param"
        );
    }

    #[test]
    fn test_parse_buffer_params_duplicate_params_no_error() {
        let result = parse_buffer_params(Some("endcap=round endcap=flat"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_buffer_params_quad_segs_out_of_range() {
        let result = parse_buffer_params(Some("quad_segs=-5"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_buffer_params_side_functional() {
        let wkt_line = "LINESTRING(50 50, 150 150 ,150 50)";
        let line = geos::Geometry::new_from_wkt(wkt_line).unwrap();
        let buffer_distance = 100.0;

        // BufferParams don't implement types that makes them testable
        let result_params = parse_buffer_params(Some("side=right")).unwrap();
        let expected_params = BufferParams::builder()
            .end_cap_style(CapStyle::Square)
            .single_sided(true)
            .build()
            .unwrap();

        // Testing via behavior here
        let result_buffer = line
            .buffer_with_params(buffer_distance, &result_params)
            .unwrap();
        let expected_buffer = line
            .buffer_with_params(buffer_distance, &expected_params)
            .unwrap();

        // Assert: Compare the resulting Geometry
        assert!(result_buffer.equals_exact(&expected_buffer, 0.1).unwrap());
    }

    #[test]
    fn test_parse_buffer_params_non_default_cap_with_side() {
        let wkt_line = "LINESTRING(50 50, 150 150 ,150 50)";
        let line = geos::Geometry::new_from_wkt(wkt_line).unwrap();
        let result_params = parse_buffer_params(Some("side=right endcap=round")).unwrap();

        // Assert (Expected): The cap should be Flat, and it should be single-sided
        let expected_params = BufferParams::builder()
            .end_cap_style(CapStyle::Round)
            .single_sided(true)
            .build()
            .unwrap();

        // Check functional equivalence by generating and comparing geometries
        let buffer_distance = 84.3;

        let result_buffer = line
            .buffer_with_params(buffer_distance, &result_params)
            .unwrap();
        let expected_buffer = line
            .buffer_with_params(buffer_distance, &expected_params)
            .unwrap();

        assert!(result_buffer.equals_exact(&expected_buffer, 0.1).unwrap());
    }

    #[test]
    fn test_parse_buffer_params_explicit_default_side() {
        let wkt_line = "LINESTRING (0 0, 1 0)";
        let line = geos::Geometry::new_from_wkt(wkt_line).unwrap();
        let buffer_distance = 0.1;

        let result_params = parse_buffer_params(Some("side=both")).unwrap();
        let expected_params = BufferParams::builder().build().unwrap();

        let result_buffer = line
            .buffer_with_params(buffer_distance, &result_params)
            .unwrap();
        let expected_buffer = line
            .buffer_with_params(buffer_distance, &expected_params)
            .unwrap();

        assert!(result_buffer.equals_exact(&expected_buffer, 0.1).unwrap());
    }

    #[test]
    fn test_side_right_geos_3_13() {
        let wkt = "LINESTRING(50 50, 150 150, 150 50)";
        let line = geos::Geometry::new_from_wkt(wkt).unwrap();
        let distance = 100.0;

        // Test single-sided buffer (GEOS 3.13+ removes artifacts, giving 12713.61)
        // PostGIS with GEOS 3.9 returns 16285.08 due to including geometric artifacts
        // GEOS 3.12+ improvements: https://github.com/libgeos/geos/commit/091f6d99
        let params_single = BufferParams::builder().single_sided(true).build().unwrap();

        let buffer_right = line.buffer_with_params(-distance, &params_single).unwrap();
        let area_right = buffer_right.area().unwrap();

        // Expected area with GEOS 3.13 (improved algorithm without artifacts)
        assert!(
            (area_right - 12713.605978550266).abs() < 0.1,
            "Expected GEOS 3.13+ area ~12713.61, got {area_right}"
        );
    }

    #[test]
    fn test_empty_and_invalid_input() {
        assert_eq!(
            parse_buffer_side_style(None),
            (false, false),
            "Should return (false, false) for None."
        );
        assert_eq!(
            parse_buffer_side_style(Some("")),
            (false, false),
            "Should return (false, false) for an empty string."
        );
        assert_eq!(
            parse_buffer_side_style(Some("mitre_limit=5.0")),
            (false, false),
            "Should return (false, false) for an invalid key."
        );
    }

    #[test]
    fn test_single_side_and_case_insensitivity() {
        assert_eq!(
            parse_buffer_side_style(Some("side=left")),
            (true, false),
            "Should detect 'left'."
        );
        assert_eq!(
            parse_buffer_side_style(Some("side=RIGHT")),
            (false, true),
            "Should detect 'RIGHT' case-insensitively."
        );
        assert_eq!(
            parse_buffer_side_style(Some("SiDe=LeFt")),
            (true, false),
            "Should handle mixed case key and value."
        );
        assert_eq!(
            parse_buffer_side_style(Some("join=mitre SIDE=RIGHT mitre_limit=5.0")),
            (false, true),
            "Should ignore other params and detect 'RIGHT'."
        );
        assert_eq!(
            parse_buffer_side_style(Some("side=center")),
            (false, false),
            "Should ignore invalid side values."
        );
    }

    #[test]
    fn test_both_sides_present() {
        assert_eq!(
            parse_buffer_side_style(Some("side=left side=right")),
            (false, true),
            "Should detect both left and right."
        );
        assert_eq!(
            parse_buffer_side_style(Some("side=right side=left join=round")),
            (true, false),
            "Should detect both regardless of order."
        );
        assert_eq!(
            parse_buffer_side_style(Some("SIDE=RIGHT endcap=round side=left")),
            (true, false),
            "Should handle complex string with both sides."
        );
    }
}
