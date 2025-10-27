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

    executor.execute_wkb_void(|wkb| {
        match (wkb, distance_iter.next().unwrap()) {
            (Some(wkb), Some(distance)) => {
                builder.append_value(invoke_scalar(&wkb, distance, &params)?);
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
) -> Result<Vec<u8>> {
    let geometry = geos_geom
        .buffer_with_params(distance, params)
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
    let wkb = geometry
        .to_wkb()
        .map_err(|e| DataFusionError::Execution(format!("Failed to convert to wkb: {e}")))?;

    Ok(wkb.into())
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
            "Expected scalar bufferStyleParameters, got: {:?}",
            arg
        ))),
    }
}

fn parse_buffer_params(params_str: Option<&str>) -> Result<BufferParams> {
    let Some(params_str) = params_str else {
        return BufferParams::builder()
            .build()
            .map_err(|e| DataFusionError::External(Box::new(e)));
    };

    let mut params_builder = BufferParams::builder();

    for param in params_str.split_whitespace() {
        let Some((key, value)) = param.split_once('=') else {
            return Err(DataFusionError::Execution(format!(
                "Missing value for buffer parameter: {}",
                param
            )));
        };

        if key.eq_ignore_ascii_case("endcap") {
            params_builder = params_builder.end_cap_style(parse_cap_style(value)?);
        } else if key.eq_ignore_ascii_case("join") {
            params_builder = params_builder.join_style(parse_join_style(value)?);
        } else if key.eq_ignore_ascii_case("side") {
            params_builder = params_builder.single_sided(is_single_sided(value)?);
        } else if key.eq_ignore_ascii_case("mitre_limit") || key.eq_ignore_ascii_case("miter_limit")
        {
            let limit: f64 = parse_number(value, "mitre_limit")?;
            params_builder = params_builder.mitre_limit(limit);
        } else if key.eq_ignore_ascii_case("quad_segs")
            || key.eq_ignore_ascii_case("quadrant_segments")
        {
            let segs = parse_number(value, "quadrant_segments")?;
            params_builder = params_builder.quadrant_segments(segs);
        } else {
            return Err(DataFusionError::Execution(format!(
                "Invalid buffer parameter: {} \
                (accept: 'endcap', 'join', 'mitre_limit', 'miter_limit', 'quad_segs', 'quadrant_segments' and 'side')",
                key
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
    } else if value.eq_ignore_ascii_case("flat") {
        Ok(CapStyle::Flat)
    } else if value.eq_ignore_ascii_case("square") {
        Ok(CapStyle::Square)
    } else {
        Err(DataFusionError::Execution(format!(
            "Invalid endcap style: '{}'. Valid options: round, flat, butt, square",
            value
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
            "Invalid join style: '{}'. Valid options: round, mitre, miter, bevel",
            value
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
            "Invalid side: '{}'. Valid options: both, left, right",
            value
        )))
    }
}

fn parse_number<T: std::str::FromStr>(value: &str, param_name: &str) -> Result<T> {
    value.parse().map_err(|_| {
        DataFusionError::Execution(format!(
            "Invalid {} value: '{}'. Expected a valid number",
            param_name, value
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

        let envelope_udf = sedona_functions::st_envelope::st_envelope_udf();
        let envelope_tester = ScalarUdfTester::new(envelope_udf.into(), vec![WKB_GEOMETRY]);

        let buffer_result = tester
            .invoke_scalar_scalar_scalar(
                "LINESTRING (0 0, 10 0)",
                2.0,
                "endcap=flat join=mitre quad_segs=2".to_string(),
            )
            .unwrap();
        let envelope_result = envelope_tester.invoke_scalar(buffer_result).unwrap();
        let expected_envelope = "POLYGON((0 -2, 0 2, 10 2, 10 -2, 0 -2))";
        tester.assert_scalar_result_equals(envelope_result, expected_envelope);

        let buffer_result = tester
            .invoke_scalar_scalar_scalar(
                "LINESTRING (0 0, 10 0)",
                1.0,
                "endcap=square join=bevel".to_string(),
            )
            .unwrap();
        let envelope_result = envelope_tester.invoke_scalar(buffer_result).unwrap();
        let expected_envelope = "POLYGON((-1 -1, -1 1, 11 1, 11 -1, -1 -1))";
        tester.assert_scalar_result_equals(envelope_result, expected_envelope);
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
}
