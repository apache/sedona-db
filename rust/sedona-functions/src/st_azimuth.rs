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
use arrow_array::builder::Float64Builder;
use arrow_schema::DataType;
use datafusion_common::{error::Result, exec_err};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{SedonaScalarKernel, SedonaScalarUDF},
};
use sedona_geometry::wkb_header::read_point_xy;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};
use std::sync::Arc;

use crate::executor::WkbBytesExecutor;

/// ST_Azimuth() scalar UDF
///
/// Stub function for azimuth calculation between two points.
pub fn st_azimuth_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_azimuth",
        ItemCrsKernel::wrap_impl(vec![Arc::new(STAzimuth {})]),
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct STAzimuth {}

impl SedonaScalarKernel for STAzimuth {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry(), ArgMatcher::is_geometry()],
            SedonaType::Arrow(DataType::Float64),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        // ST_Azimuth is defined only for two non-empty POINTs, so it never needs
        // the full geometry: read each operand's (x, y) straight from the WKB
        // offset via `read_point_xy` (no parse, no enum). POINT EMPTY or any
        // non-Point maps to the same error the general path raised.
        let executor = WkbBytesExecutor::new(arg_types, args);
        let mut builder = Float64Builder::with_capacity(executor.num_iterations());
        executor.execute_wkb_wkb_void(|maybe_start, maybe_end| {
            match (maybe_start, maybe_end) {
                (Some(start), Some(end)) => {
                    builder.append_option(azimuth_from_bytes(start, end)?);
                }
                _ => builder.append_null(),
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

/// Azimuth between two operands read as raw WKB. Both must be non-empty POINTs;
/// `read_point_xy` returns `Ok(None)` for POINT EMPTY and `Err` for non-Point
/// input, both of which map to the same error the general parser raised.
fn azimuth_from_bytes(start: &[u8], end: &[u8]) -> Result<Option<f64>> {
    match (read_point_xy(start), read_point_xy(end)) {
        (Ok(Some((start_x, start_y))), Ok(Some((end_x, end_y)))) => {
            Ok(calc_azimuth(start_x, start_y, end_x, end_y))
        }
        _ => exec_err!("ST_Azimuth expects both arguments to be non-empty POINT geometries"),
    }
}

fn calc_azimuth(start_x: f64, start_y: f64, end_x: f64, end_y: f64) -> Option<f64> {
    let dx = end_x - start_x;
    let dy = end_y - start_y;

    if dx == 0.0 && dy == 0.0 {
        return None;
    }

    let mut angle = dx.atan2(dy);
    if angle < 0.0 {
        angle += 2.0 * std::f64::consts::PI;
    }

    Some(angle)
}

#[cfg(test)]
mod tests {
    use datafusion_common::scalar::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS, WKB_VIEW_GEOMETRY};
    use sedona_testing::create::create_scalar;
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_azimuth_udf().into();
        assert_eq!(udf.name(), "st_azimuth");
    }

    #[rstest]
    fn udf(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY, WKB_GEOMETRY_ITEM_CRS.clone())]
        start_type: SedonaType,
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY, WKB_GEOMETRY_ITEM_CRS.clone())]
        end_type: SedonaType,
    ) {
        let tester = ScalarUdfTester::new(
            st_azimuth_udf().into(),
            vec![start_type.clone(), end_type.clone()],
        );

        assert_eq!(
            tester.return_type().unwrap(),
            SedonaType::Arrow(DataType::Float64)
        );

        let start = create_scalar(Some("POINT (0 0)"), &start_type);
        let north = create_scalar(Some("POINT (0 1)"), &end_type);
        let east = create_scalar(Some("POINT (1 0)"), &end_type);
        let south = create_scalar(Some("POINT (0 -1)"), &end_type);
        let west = create_scalar(Some("POINT (-1 0)"), &end_type);
        let same = create_scalar(Some("POINT (0 0)"), &end_type);
        let empty = create_scalar(Some("POINT EMPTY"), &end_type);

        let result = tester
            .invoke_scalar_scalar(start.clone(), north.clone())
            .unwrap();
        assert!(matches!(
            result,
            ScalarValue::Float64(Some(val)) if (val - 0.0).abs() < 1e-12
        ));

        let result = tester
            .invoke_scalar_scalar(start.clone(), east.clone())
            .unwrap();
        assert!(matches!(
            result,
            ScalarValue::Float64(Some(val)) if (val - std::f64::consts::FRAC_PI_2).abs() < 1e-12
        ));

        let result = tester
            .invoke_scalar_scalar(start.clone(), south.clone())
            .unwrap();
        assert!(matches!(
            result,
            ScalarValue::Float64(Some(val)) if (val - std::f64::consts::PI).abs() < 1e-12
        ));

        let result = tester
            .invoke_scalar_scalar(start.clone(), west.clone())
            .unwrap();
        assert!(matches!(
            result,
            ScalarValue::Float64(Some(val)) if (val - (3.0 * std::f64::consts::FRAC_PI_2)).abs() < 1e-12
        ));

        // If two points are the same, return NULL
        let result = tester
            .invoke_scalar_scalar(start.clone(), same.clone())
            .unwrap();
        assert!(result.is_null());

        // If either one of the points is empty, return NULL
        let result = tester.invoke_scalar_scalar(start.clone(), empty.clone());
        assert!(
            result.is_err()
                && result
                    .unwrap_err()
                    .to_string()
                    .contains("ST_Azimuth expects both arguments to be non-empty POINT geometries")
        );

        // If either one of the points is NULL, return NULL
        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, north.clone())
            .unwrap();
        assert!(result.is_null());
    }
}
