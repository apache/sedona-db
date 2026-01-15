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
use datafusion_common::{error::Result, DataFusionError};
use datafusion_expr::ColumnarValue;
use geos::{Geom, Geometry, GeometryTypes};
use sedona_expr::scalar_udf::{ScalarKernelRef, SedonaScalarKernel};
use sedona_geometry::wkb_factory::WKB_MIN_PROBABLE_BYTES;
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

use crate::executor::GeosExecutor;
use crate::geos_to_wkb::write_geos_geometry;

pub fn st_exterior_ring_impl() -> ScalarKernelRef {
    Arc::new(STExteriorRing {})
}

#[derive(Debug)]
struct STExteriorRing {}

impl SedonaScalarKernel for STExteriorRing {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOMETRY);
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = GeosExecutor::new(arg_types, args);
        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
        );

        executor.execute_wkb_void(|maybe_geom| {
            match maybe_geom {
                Some(geom) => {
                    invoke_scalar(&geom, &mut builder)?;
                    builder.append_value([]);
                }
                None => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(geom: &Geometry, writer: &mut BinaryBuilder) -> Result<()> {
    let result = exterior_ring(geom)?;
    write_geos_geometry(&result, writer)?;
    Ok(())
}

/// Core exterior ring logic
fn exterior_ring(geom: &Geometry) -> Result<Geometry> {
    match geom.geometry_type() {
        GeometryTypes::Polygon => {
            let ring = geom.get_exterior_ring().map_err(|e| {
                DataFusionError::Execution(format!("Failed to get exterior ring: {e}"))
            })?;
            Ok(Geom::clone(&ring))
        }

        GeometryTypes::MultiPolygon => {
            let n = geom.get_num_geometries().map_err(|e| {
                DataFusionError::Execution(format!("Failed to get polygon count: {e}"))
            })?;

            let mut rings: Vec<Geometry> = Vec::new();
            for i in 0..n {
                let poly = geom.get_geometry_n(i).map_err(|e| {
                    DataFusionError::Execution(format!("Failed to get polygon {i}: {e}"))
                })?;
                let ring = poly.get_exterior_ring().map_err(|e| {
                    DataFusionError::Execution(format!("Failed to get exterior ring: {e}"))
                })?;
                rings.push(Geom::clone(&ring));
            }

            if rings.is_empty() {
                Geometry::create_geometry_collection(Vec::new()).map_err(|e| {
                    DataFusionError::Execution(format!("Failed to create empty collection: {e}"))
                })
            } else if rings.len() == 1 {
                Ok(rings.into_iter().next().unwrap())
            } else {
                Geometry::create_multiline_string(rings).map_err(|e| {
                    DataFusionError::Execution(format!("Failed to create multilinestring: {e}"))
                })
            }
        }

        GeometryTypes::GeometryCollection => {
            let n = geom.get_num_geometries().map_err(|e| {
                DataFusionError::Execution(format!("Failed to get geometry count: {e}"))
            })?;

            let mut components: Vec<Geometry> = Vec::new();
            for i in 0..n {
                let child = geom.get_geometry_n(i).map_err(|e| {
                    DataFusionError::Execution(format!("Failed to get geometry {i}: {e}"))
                })?;
                components.push(exterior_ring(&Geom::clone(&child))?);
            }

            Geometry::create_geometry_collection(components).map_err(|e| {
                DataFusionError::Execution(format!("Failed to create geometry collection: {e}"))
            })
        }

        // Non-area geometries → empty
        _ => Geometry::create_geometry_collection(Vec::new()).map_err(|e| {
            DataFusionError::Execution(format!("Failed to create empty geometry: {e}"))
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{SedonaType, WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::compare::assert_array_equal;
    use sedona_testing::create::create_array;
    use sedona_testing::testers::ScalarUdfTester;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_kernel("st_exterior_ring", st_exteriorring_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type]);

        tester.assert_return_type(WKB_GEOMETRY);

        let result = tester
            .invoke_scalar("POLYGON((0 0, 1 0, 1 1, 0 0))")
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING (0 0, 1 0, 1 1, 0 0)");

        let result = tester
            .invoke_scalar(datafusion_common::ScalarValue::Null)
            .unwrap();
        assert!(result.is_null());

        let input_wkt = vec![
            Some("POINT(1 2)"),
            Some("LINESTRING (0 0, 1 0, 0 1)"),
            Some("MULTIPOINT (1 1, 2 2)"),
            None,
            Some("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))"),
            Some("POLYGON Z ((0 0 1, 10 0 1, 10 10 1, 0 10 1, 0 0 1))"),
            Some("POLYGON M ((0 0 5, 10 0 5, 10 10 5, 0 10 5, 0 0 5))"),
            Some("POLYGON ZM ((0 0 1 9, 10 0 1 9, 10 10 1 9, 0 10 1 9, 0 0 1 9))"),
            Some(
                "MULTIPOLYGON(((0 0,0 1,1 1,1 0,0 0)),((10 10,10 11,11 11,11 10,10 10)))",
            ),
            Some(
                "MULTIPOLYGON Z (((0 0 3,0 1 3,1 1 3,1 0 3,0 0 3)),((10 10 4,10 11 4,11 11 4,11 10 4,10 10 4)))",
            ),
        ];

        let expected = create_array(
            &[
                None,
                None,
                None,
                None,
                Some("LINESTRING (0 0, 10 0, 10 10, 0 10, 0 0)"),
                Some("LINESTRING Z (0 0 1, 10 0 1, 10 10 1, 0 10 1, 0 0 1)"),
                Some("LINESTRING M (0 0 5, 10 0 5, 10 10 5, 0 10 5, 0 0 5)"),
                Some("LINESTRING ZM (0 0 1 9, 10 0 1 9, 10 10 1 9, 0 10 1 9, 0 0 1 9)"),
                Some(
                    "MULTILINESTRING((0 0,0 1,1 1,1 0,0 0),(10 10,10 11,11 11,11 10,10 10))",
                ),
                Some(
                    "MULTILINESTRING Z ((0 0 3,0 1 3,1 1 3,1 0 3,0 0 3),(10 10 4,10 11 4,11 11 4,11 10 4,10 10 4))",
                ),
            ],
            &WKB_GEOMETRY,
        );

        assert_array_equal(&tester.invoke_wkb_array(input_wkt).unwrap(), &expected);
    }
}
