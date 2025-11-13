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
use datafusion_common::{cast::as_float64_array, DataFusionError, Result};
use datafusion_expr::ColumnarValue;
use geos::Geom;
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_geometry::wkb_factory::WKB_MIN_PROBABLE_BYTES;
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

use crate::executor::GeosExecutor;

/// ST_Snap() implementation using the geos crate
pub fn st_snap_impl() -> ScalarKernelRef {
    Arc::new(STSnap {})
}

#[derive(Debug)]
struct STSnap {}

impl SedonaScalarKernel for STSnap {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![
                ArgMatcher::is_geometry(),
                ArgMatcher::is_geometry(),
                ArgMatcher::is_numeric(),
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
        let executor = GeosExecutor::new(arg_types, args);

        let tolerance_value = args[2]
            .cast_to(&DataType::Float64, None)?
            .to_array(executor.num_iterations())?;
        let tolerance_array = as_float64_array(&tolerance_value)?;

        let mut tolerance_iter = tolerance_array.iter();

        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
        );

        executor.execute_wkb_wkb_void(|wkb_input, wkb_ref| {
            match (wkb_input, wkb_ref, tolerance_iter.next().unwrap()) {
                (Some(wkb_i), Some(wkb_r), Some(tolerance)) => {
                    invoke_scalar(wkb_i, wkb_r, tolerance, &mut builder)?;
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
    geom_input: &geos::Geometry,
    geom_reference: &geos::Geometry,
    tolerance: f64,
    writer: &mut impl std::io::Write,
) -> Result<()> {
    let geometry = geom_input
        .snap(geom_reference, tolerance)
        .map_err(|e| DataFusionError::Execution(format!("Failed to snap geometry: {e}")))?;

    let wkb = geometry
        .to_wkb()
        .map_err(|e| DataFusionError::Execution(format!("Failed to convert to wkb: {e}")))?;

    writer.write_all(wkb.as_ref())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{SedonaType, WKB_VIEW_GEOMETRY};
    use sedona_testing::{
        compare::assert_array_equal, create::create_array, testers::ScalarUdfTester,
    };

    use super::*;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_snap", st_snap_impl());
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![
                sedona_type.clone(),
                sedona_type,
                SedonaType::Arrow(DataType::Float64),
            ],
        );
        tester.assert_return_type(WKB_GEOMETRY);

        let input_geometries = create_array(
            &[
                Some("MULTIPOLYGON(((26 125, 26 200, 126 200, 126 125, 26 125 ),( 51 150, 101 150, 76 175, 51 150 )),(( 151 100, 151 200, 176 175, 151 100 )))"),
                Some("MULTIPOLYGON((( 26 125, 26 200, 126 200, 126 125, 26 125 ),( 51 150, 101 150, 76 175, 51 150 )),(( 151 100, 151 200, 176 175, 151 100 )))"),
                Some("LINESTRING (5 107, 54 84, 101 100)"),
                Some("LINESTRING (5 107, 54 84, 101 100)"),
                Some("POINT (1.1 2.1)"),
                Some("POINT (5.9 6.9)"),
                Some("LINESTRING (0.9 0.9, 2.1 2.1, 4.9 4.9)"),
                Some("LINESTRING (10.1 10.1, 12 12)"),
                Some("POLYGON ((0.9 0.9, 0.9 3.1, 3.1 3.1, 3.1 0.9, 0.9 0.9))"),
                Some("POLYGON ((5 5, 5 8, 8 8, 8 5, 5 5))"),
                Some("MULTILINESTRING ((0.9 0.9, 2 2), (3.1 3.1, 4 4))"),
                Some("MULTIPOINT (0.9 0.9, 2.1 2.1, 3.9 3.9)"),
                Some("POINT (1.1 2.1)"), // Within tolerance
                Some("POINT (1.6 2.6)"), // Outside tolerance
                Some("LINESTRING (0 0, 10 10)"), // No snapping needed
                Some("POINT (5 5)"), // Exact match
                Some("POLYGON ((0.9 0.9, 0.9 5.1, 5.1 5.1, 5.1 0.9, 0.9 0.9), (1.9 1.9, 1.9 4.1, 4.1 4.1, 4.1 1.9, 1.9 1.9))"),
                Some("LINESTRING (0.1 0.1, 0.2 0.2, 0.3 0.3, 0.4 0.4, 0.5 0.5, 0.6 0.6, 0.7 0.7, 0.8 0.8, 0.9 0.9)"),
                Some("POINT (1 2)"),
                None, // NULL input
                Some("POINT EMPTY"), // Empty geometry
            ],
            &WKB_GEOMETRY,
        );

        let reference_geometries = create_array(
            &[
                Some("LINESTRING (5 107, 54 84, 101 100)"),
                Some("LINESTRING (5 107, 54 84, 101 100)"),
                Some("MULTIPOLYGON(((26 125, 26 200, 126 200, 126 125, 26 125),(51 150, 101 150, 76 175, 51 150 )),((151 100, 151 200, 176 175, 151 100)))"),
                Some("MULTIPOLYGON(((26 125, 26 200, 126 200, 126 125, 26 125),(51 150, 101 150, 76 175, 51 150 )),((151 100, 151 200, 176 175, 151 100)))"),
                Some("POINT (1 2)"),
                Some("POINT (6 7)"),
                Some("POINT (1 1)"),
                Some("MULTIPOINT ((5 5), (10 10))"),
                Some("LINESTRING (1 1, 1 3, 3 3, 3 1, 1 1)"),
                Some("LINESTRING (4.9 4.9, 4.9 8.1, 8.1 8.1, 8.1 4.9, 4.9 4.9)"),
                Some("MULTIPOINT ((1 1), (3 3))"),
                Some("LINESTRING (1 1, 2 2, 3 3, 4 4)"),
                Some("POINT (1 2)"),
                Some("POINT (1 2)"),
                Some("POINT (5 5)"),
                Some("POINT (5 5)"),
                Some("POLYGON ((1 1, 1 5, 5 5, 5 1, 1 1), (2 2, 2 4, 4 4, 4 2, 2 2))"),
                Some("LINESTRING (0 0, 1 1)"),
                Some("POINT (3 4)"),
                Some("POINT (1 1)"),
                Some("POINT (1 1)"),
            ],
            &WKB_GEOMETRY,
        );

        let expected_geometries = create_array(
            &[
                Some("MULTIPOLYGON(((26 125,26 200,126 200,126 125,101 100,26 125),(51 150,101 150,76 175,51 150)),((151 100,151 200,176 175,151 100)))"),
                Some("MULTIPOLYGON(((5 107,26 200,126 200,126 125,101 100,54 84,5 107),(51 150,101 150,76 175,51 150)),((151 100,151 200,176 175,151 100)))"),
                Some("LINESTRING(5 107,26 125,54 84,101 100)"),
                Some("LINESTRING(26 125,54 84,101 100)"),
                Some("POINT (1 2)"), // Should snap to reference
                Some("POINT (6 7)"), // Should snap to reference
                Some("LINESTRING (1 1, 2.1 2.1, 4.9 4.9)"), // First and last vertices snap
                Some("LINESTRING (10 10, 12 12)"), // First vertex snaps
                Some("POLYGON ((1 1, 1 3, 3 3, 3 1, 1 1))"), // All vertices snap
                Some("POLYGON((4.9 4.9,4.9 8.1,8.1 8.1,8.1 4.9,4.9 4.9))"), // Partial snapping
                Some("MULTILINESTRING ((1 1, 2 2), (3 3, 4 4))"), // Endpoints snap
                Some("MULTIPOINT (1 1, 2 2, 4 4)"), // Points snap to line
                Some("POINT (1 2)"), // Snaps within tolerance
                Some("POINT (1.6 2.6)"), // No snap (outside tolerance)
                Some("LINESTRING (0 0, 5 5, 10 10)"), // No snap (line doesn't get vertex added)
                Some("POINT (5 5)"), // Exact match, no change
                Some("POLYGON ((1 1, 1 5, 5 5, 5 1, 1 1), (2 2, 2 4, 4 4, 4 2, 2 2))"),
                Some("LINESTRING (0 0, 0.2 0.2, 0.3 0.3, 0.4 0.4, 0.5 0.5, 0.6 0.6, 0.7 0.7, 0.8 0.8, 1 1)"),
                Some("POINT (1 2)"), // No snap (outside tolerance)
                None, // NULL result
                Some("POINT EMPTY"), // Empty result
            ],
            &WKB_GEOMETRY,
        );

        let tolerance_array = arrow_array::create_array!(
            Float64,
            [
                Some(25.0 * 1.01),
                Some(25.0 * 1.25),
                Some(25.0 * 1.01),
                Some(25.0 * 1.25),
                Some(0.5),
                Some(0.5),
                Some(0.5),
                Some(0.5),
                Some(0.5),
                Some(0.5),
                Some(0.5),
                Some(0.5),
                Some(0.5),
                Some(0.5),
                Some(1.0),
                Some(0.0),
                Some(0.5),
                Some(0.5),
                Some(0.5),
                Some(0.5),
                Some(0.15)
            ]
        );

        assert_array_equal(
            &tester
                .invoke_arrays(vec![
                    input_geometries,
                    reference_geometries,
                    tolerance_array,
                ])
                .unwrap(),
            &expected_geometries,
        );
    }
}
