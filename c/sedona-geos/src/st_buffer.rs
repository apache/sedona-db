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
/// Supports three signatures:
/// - ST_Buffer(geometry: Geometry, distance: Double)
/// - ST_Buffer(geometry: Geometry, distance: Double, useSpheroid: Boolean)
/// - ST_Buffer(geometry: Geometry, distance: Double, useSpheroid: Boolean, bufferStyleParameters: String)
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
        if !(2..=4).contains(&args.len()) {
            return Err(DataFusionError::Plan(format!(
                "ST_Buffer expects 2-4 arguments, got {}",
                args.len()
            )));
        }

        let mut matchers = vec![ArgMatcher::is_geometry(), ArgMatcher::is_numeric()];

        if args.len() >= 3 {
            matchers.push(ArgMatcher::is_boolean());
        }
        if args.len() == 4 {
            matchers.push(ArgMatcher::is_string());
        }

        ArgMatcher::new(matchers, WKB_GEOMETRY).match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        // Extract Args
        let distance: Option<f64> = extract_optional_f64(&args[1])?;
        let _use_spheroid = extract_optional_bool(args.get(2))?;
        let buffer_style_params = extract_optional_string(args.get(3))?;

        // Build BufferParams based on style parameters
        let params = parse_buffer_params(buffer_style_params.as_deref())?;

        let executor = GeosExecutor::new(arg_types, args);
        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
        );
        executor.execute_wkb_void(|wkb| {
            match (wkb, distance) {
                (Some(wkb), Some(distance)) => {
                    invoke_scalar(&wkb, distance, &params, &mut builder)?;
                    builder.append_value([]);
                }
                _ => builder.append_null(),
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
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

fn extract_optional_f64(arg: &ColumnarValue) -> Result<Option<f64>> {
    let casted = arg.cast_to(&DataType::Float64, None)?;
    match &casted {
        ColumnarValue::Scalar(scalar) if !scalar.is_null() => {
            Ok(Some(f64::try_from(scalar.clone())?))
        }
        ColumnarValue::Scalar(_) => Ok(None),
        _ => Err(DataFusionError::Execution(format!(
            "Expected scalar distance, got: {:?}",
            arg
        ))),
    }
}

fn extract_optional_bool(arg: Option<&ColumnarValue>) -> Result<Option<bool>> {
    let Some(arg) = arg else { return Ok(None) };
    let casted = arg.cast_to(&DataType::Boolean, None)?;
    match &casted {
        ColumnarValue::Scalar(scalar) if !scalar.is_null() => {
            Ok(Some(bool::try_from(scalar.clone())?))
        }
        ColumnarValue::Scalar(_) => Ok(None),
        _ => Err(DataFusionError::Execution(format!(
            "Expected scalar useSpheroid parameter, got: {:?}",
            arg
        ))),
    }
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
            continue;
        };

        match key.to_lowercase().as_str() {
            "endcap" => {
                params_builder = params_builder.end_cap_style(parse_cap_style(value)?);
            }
            "join" => {
                params_builder = params_builder.join_style(parse_join_style(value)?);
            }
            "side" => {
                params_builder = params_builder.single_sided(parse_side(value)?);
            }
            "mitre_limit" | "miter_limit" => {
                let limit = parse_number(value, "mitre_limit")?;
                params_builder = params_builder.mitre_limit(limit);
            }
            "quad_segs" | "quadrant_segments" => {
                let segs = parse_number(value, "quadrant_segments")?;
                params_builder = params_builder.quadrant_segments(segs);
            }
            _ => {}
        }
    }

    params_builder
        .build()
        .map_err(|e| DataFusionError::External(Box::new(e)))
}

fn parse_cap_style(value: &str) -> Result<CapStyle> {
    match value.to_lowercase().as_str() {
        "round" => Ok(CapStyle::Round),
        "flat" | "butt" => Ok(CapStyle::Flat),
        "square" => Ok(CapStyle::Square),
        _ => Err(DataFusionError::Execution(format!(
            "Invalid endcap style: '{}'. Valid options: round, flat, butt, square",
            value
        ))),
    }
}

fn parse_join_style(value: &str) -> Result<JoinStyle> {
    match value.to_lowercase().as_str() {
        "round" => Ok(JoinStyle::Round),
        "mitre" | "miter" => Ok(JoinStyle::Mitre),
        "bevel" => Ok(JoinStyle::Bevel),
        _ => Err(DataFusionError::Execution(format!(
            "Invalid join style: '{}'. Valid options: round, mitre, miter, bevel",
            value
        ))),
    }
}

fn parse_side(value: &str) -> Result<bool> {
    match value.to_lowercase().as_str() {
        "both" => Ok(false),
        "left" | "right" => Ok(true),
        _ => Err(DataFusionError::Execution(format!(
            "Invalid side: '{}'. Valid options: both, left, right",
            value
        ))),
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
        let udf = SedonaScalarUDF::from_kernel("st_buffer", st_buffer_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![
                sedona_type.clone(),
                SedonaType::Arrow(DataType::Float64),
                SedonaType::Arrow(DataType::Boolean),
                SedonaType::Arrow(DataType::Utf8),
            ],
        );
        tester.assert_return_type(WKB_GEOMETRY);

        let buffer_result = tester
            .invoke(vec![
                ColumnarValue::Scalar(ScalarValue::Binary(Some(sedona_testing::create::make_wkb(
                    "LINESTRING (0 0, 10 0)",
                )))),
                ColumnarValue::Scalar(ScalarValue::Float64(Some(1.0))),
                ColumnarValue::Scalar(ScalarValue::Boolean(Some(false))),
                ColumnarValue::Scalar(ScalarValue::Utf8(Some(
                    "endcap=flat join=mitre quad_segs=2".to_string(),
                ))),
            ])
            .unwrap();

        assert!(matches!(buffer_result, ColumnarValue::Scalar(_)));

        let buffer_result2 = tester
            .invoke(vec![
                ColumnarValue::Scalar(ScalarValue::Binary(Some(sedona_testing::create::make_wkb(
                    "LINESTRING (0 0, 10 0)",
                )))),
                ColumnarValue::Scalar(ScalarValue::Float64(Some(1.0))),
                ColumnarValue::Scalar(ScalarValue::Boolean(Some(false))),
                ColumnarValue::Scalar(ScalarValue::Utf8(Some(
                    "endcap=square join=bevel".to_string(),
                ))),
            ])
            .unwrap();

        assert!(matches!(buffer_result2, ColumnarValue::Scalar(_)));
    }
}
