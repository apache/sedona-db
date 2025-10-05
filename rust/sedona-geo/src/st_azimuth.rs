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

use arrow_array::builder::Float64Builder;
use arrow_schema::DataType;
use datafusion_common::error::Result;
use datafusion_expr::ColumnarValue;
use geo_traits::{CoordTrait, GeometryTrait, GeometryType, PointTrait};
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_functions::executor::WkbExecutor;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};
use wkb::reader::Wkb;

/// ST_Azimuth() implementation
pub fn st_azimuth_impl() -> ScalarKernelRef {
    Arc::new(STAzimuth {})
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
        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = Float64Builder::with_capacity(executor.num_iterations());
        executor.execute_wkb_wkb_void(|maybe_start, maybe_end| {
            match (maybe_start, maybe_end) {
                (Some(start), Some(end)) => match invoke_scalar(start, end)? {
                    Some(angle) => builder.append_value(angle),
                    None => builder.append_null(),
                },
                _ => builder.append_null(),
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(start: &Wkb, end: &Wkb) -> Result<Option<f64>> {
    match (start.as_type(), end.as_type()) {
        (GeometryType::Point(start_point), GeometryType::Point(end_point)) => {
            match (start_point.coord(), end_point.coord()) {
                // If both geometries are non-empty points, calculate the angle
                (Some(start_coord), Some(end_coord)) => Ok(calc_azimuth(
                    start_coord.x(),
                    start_coord.y(),
                    end_coord.x(),
                    end_coord.y(),
                )),
                // If either of the points is empty, the result is NULL
                _ => Ok(None),
            }
        }
        _ => Err(datafusion_common::error::DataFusionError::Execution(
            "ST_Azimuth expects both arguments to be POINT geometries".into(),
        )),
    }
}

// Note: When the two points are completely coincident, PostGIS's ST_Azimuth()
//       returns NULL. However, this returns 0.0.
fn calc_azimuth(start_x: f64, start_y: f64, end_x: f64, end_y: f64) -> Option<f64> {
    let dx = end_x - start_x;
    let dy = end_y - start_y;

    let mut angle = dx.atan2(dy);
    if angle < 0.0 {
        angle += 2.0 * std::f64::consts::PI;
    }

    Some(angle)
}

#[cfg(test)]
mod tests {
    use datafusion_common::scalar::ScalarValue;
    use rstest::rstest;
    use sedona_functions::register::stubs::st_azimuth_udf;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::create::create_scalar;
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn udf(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] start_type: SedonaType,
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] end_type: SedonaType,
    ) {
        let mut udf = st_azimuth_udf();
        udf.add_kernel(st_azimuth_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![start_type.clone(), end_type.clone()]);

        assert_eq!(
            tester.return_type().unwrap(),
            SedonaType::Arrow(DataType::Float64)
        );

        let start = create_scalar(Some("POINT (0 0)"), &start_type);
        let north = create_scalar(Some("POINT (0 1)"), &end_type);
        let east = create_scalar(Some("POINT (1 0)"), &end_type);
        let south = create_scalar(Some("POINT (0 -1)"), &end_type);
        let west = create_scalar(Some("POINT (-1 0)"), &end_type);

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

        let result = tester
            .invoke_scalar_scalar(ScalarValue::Null, north.clone())
            .unwrap();
        assert!(result.is_null());
    }
}
