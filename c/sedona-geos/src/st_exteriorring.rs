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
use datafusion_common::{error::Result, exec_datafusion_err};
use datafusion_expr::ColumnarValue;
use geos::{Geom, Geometry, GeometryTypes};
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{ScalarKernelRef, SedonaScalarKernel},
};
use sedona_geometry::wkb_factory::WKB_MIN_PROBABLE_BYTES;
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOGRAPHY, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

use crate::executor::GeosExecutor;
use crate::geos_to_wkb::write_geos_geometry;

/// ST_ExteriorRing() implementation using the geos crate
///
/// Returns the exterior ring of a Polygon, or NULL for non-polygon geometries.
pub fn st_exterior_ring_impl() -> Vec<ScalarKernelRef> {
    ItemCrsKernel::wrap_impl(vec![
        Arc::new(STExteriorRing {
            matcher: ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOMETRY),
        }),
        Arc::new(STExteriorRing {
            matcher: ArgMatcher::new(vec![ArgMatcher::is_geography()], WKB_GEOGRAPHY),
        }),
    ])
}

#[derive(Debug)]
struct STExteriorRing {
    matcher: ArgMatcher,
}

impl SedonaScalarKernel for STExteriorRing {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        self.matcher.match_args(args)
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
                    if invoke_scalar(geom, &mut builder)? {
                        builder.append_value([]);
                    } else {
                        builder.append_null();
                    }
                }
                _ => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(geom: &Geometry, writer: &mut impl std::io::Write) -> Result<bool> {
    let geom_type = geom
        .geometry_type()
        .map_err(|e| exec_datafusion_err!("Failed to get geometry type: {e}"))?;

    if geom_type != GeometryTypes::Polygon {
        return Ok(false);
    }

    let ring = geom
        .get_exterior_ring()
        .map_err(|e| exec_datafusion_err!("ST_ExteriorRing failed: {e}"))?;
    let line = Geometry::create_line_string(
        ring.get_coord_seq()
            .map_err(|e| exec_datafusion_err!("Failed to get ring coordinates: {e}"))?,
    )
    .map_err(|e| exec_datafusion_err!("Failed to create exterior linestring: {e}"))?;

    write_geos_geometry(&line, writer)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use datafusion_common::ScalarValue;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;
    use sedona_schema::datatypes::{
        WKB_GEOGRAPHY, WKB_GEOGRAPHY_ITEM_CRS, WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS,
    };
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_GEOGRAPHY)] sedona_type: SedonaType) {
        let udf = SedonaScalarUDF::from_impl("st_exteriorring", st_exterior_ring_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type.clone()]);

        tester.assert_return_type(sedona_type.clone());

        let result = tester
            .invoke_scalar("POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))")
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING (0 0, 4 0, 4 4, 0 4, 0 0)");

        // non-polygon returns null
        let result = tester.invoke_scalar("POINT (1 2)").unwrap();
        assert!(result.is_null());

        let result = tester.invoke_scalar(ScalarValue::Null).unwrap();
        assert!(result.is_null());
    }

    #[rstest]
    fn udf_invoke_item_crs(
        #[values(WKB_GEOMETRY_ITEM_CRS.clone(), WKB_GEOGRAPHY_ITEM_CRS.clone())]
        sedona_type: SedonaType,
    ) {
        let udf = SedonaScalarUDF::from_impl("st_exteriorring", st_exterior_ring_impl());
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type.clone()]);
        tester.assert_return_type(sedona_type);

        let result = tester
            .invoke_scalar("POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))")
            .unwrap();
        tester.assert_scalar_result_equals(result, "LINESTRING (0 0, 4 0, 4 4, 0 4, 0 0)");
    }
}
