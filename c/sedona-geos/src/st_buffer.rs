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
use datafusion_common::DataFusionError;
use datafusion_expr::ColumnarValue;
use geos::{BufferParams, Geom};
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_geometry::wkb_factory::WKB_MIN_PROBABLE_BYTES;
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

use crate::executor::GeosExecutor;

/// ST_Buffer() implementation using the geos crate
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
        // Default params
        let params_builder = BufferParams::builder();

        let params = params_builder
            .build()
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        // Extract the constant scalar value before looping over the input geometries
        let distance: Option<f64>;
        let arg1 = args[1].cast_to(&DataType::Float64, None)?;
        if let ColumnarValue::Scalar(scalar_arg) = &arg1 {
            if scalar_arg.is_null() {
                distance = None;
            } else {
                distance = Some(f64::try_from(scalar_arg.clone())?);
            }
        } else {
            return Err(DataFusionError::Execution(format!(
                "Invalid distance: {:?}",
                args[1]
            )));
        }

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
}
