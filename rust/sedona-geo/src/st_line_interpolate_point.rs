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
use datafusion_common::{
    cast::as_float64_array, error::Result, exec_datafusion_err, DataFusionError,
};
use datafusion_expr::ColumnarValue;
use geo::{algorithm::line_measures::InterpolatableLine, Euclidean};
use geo_traits::GeometryTrait;
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_geometry::wkb_factory::write_wkb_point;
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

use crate::to_geo::GeoTypesExecutor;

/// ST_LineInterpolatePoint() implementation using [InterpolatableLine]
pub fn st_line_interpolate_point_impl() -> ScalarKernelRef {
    Arc::new(STLineInterpolatePoint {})
}

#[derive(Debug)]
struct STLineInterpolatePoint {}

impl SedonaScalarKernel for STLineInterpolatePoint {
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
        let executor = GeoTypesExecutor::new(arg_types, args);
        let mut builder =
            BinaryBuilder::with_capacity(executor.num_iterations(), 25 * executor.num_iterations());

        let arg1_array = args[1]
            .cast_to(&DataType::Float64, None)?
            .to_array(executor.num_iterations())?;
        let arg1_float = as_float64_array(&arg1_array)?;

        let mut arg1_iter = arg1_float.into_iter();
        executor.execute_wkb_void(|maybe_geom| {
            let maybe_ratio = arg1_iter.next().unwrap();
            match (maybe_geom, maybe_ratio) {
                (Some(geom), Some(ratio)) => {
                    let (x, y) = invoke_scalar(&geom, ratio)?;
                    write_wkb_point(&mut builder, (x, y))
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                    builder.append_value([]);
                }
                _ => builder.append_null(),
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

/// Invokes [InterpolatableLine::point_at_ratio_from_start]
fn invoke_scalar(geom: &geo_types::Geometry<f64>, ratio: f64) -> Result<(f64, f64)> {
    match geom.as_type() {
        geo_traits::GeometryType::LineString(line) => {
            if let Some(pt) = line.point_at_ratio_from_start(&Euclidean, ratio) {
                Ok((pt.x(), pt.y()))
            } else {
                Err(exec_datafusion_err!("Failed to interpolate point"))
            }
        }
        _ => Err(exec_datafusion_err!(
            "Input to STLineInterpolatePoint must be linestring"
        )),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use sedona_functions::register::stubs::st_area_udf;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::{compare::assert_scalar_equal_wkb_geometry, testers::ScalarUdfTester};

    use super::*;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let mut udf = st_area_udf();
        udf.add_kernel(st_line_interpolate_point_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![sedona_type, SedonaType::Arrow(DataType::Float64)],
        );

        assert_eq!(tester.return_type().unwrap(), WKB_GEOMETRY);

        assert_scalar_equal_wkb_geometry(
            &tester
                .invoke_scalar_scalar("LINESTRING (0 0, 1 0)", 0.5)
                .unwrap(),
            Some("POINT (0.5 0)"),
        );
    }
}
