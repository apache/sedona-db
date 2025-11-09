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
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use geo_traits::{CoordTrait, GeometryTrait, GeometryType, PointTrait};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};
use std::sync::Arc;
use wkb::reader::Wkb;

use crate::executor::WkbExecutor;

/// ST_Azimuth() scalar UDF
///
/// Stub function for azimuth calculation between two points.
pub fn st_azimuth_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_azimuth",
        vec![Arc::new(STAzimuth {})],
        Volatility::Immutable,
        Some(st_azimuth_doc()),
    )
}

fn st_azimuth_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns the azimuth (a clockwise angle measured from north) in radians from geomA to geomB",
        "ST_Azimuth (A: Geometry, B: Geometry)",
    )
    .with_argument("geomA", "geometry: Start point geometry")
    .with_argument("geomB", "geometry: End point geometry")
    .with_sql_example(
        "SELECT degrees(ST_Azimuth(ST_Point(0, 0), ST_Point(1, 1)))",
    )
    .build()
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
                // If either of the points is empty, raise an error.
                _ => {
                    exec_err!("ST_Azimuth expects both arguments to be non-empty POINT geometries")
                }
            }
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
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::create::create_scalar;
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_azimuth_udf().into();
        assert_eq!(udf.name(), "st_azimuth");
        assert!(udf.documentation().is_some());
    }

    #[rstest]
    fn udf(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] start_type: SedonaType,
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] end_type: SedonaType,
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
