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

use arrow_array::builder::BooleanBuilder;
use arrow_schema::DataType;
use datafusion_common::error::Result;
use datafusion_expr::Volatility;
use geo_traits::{
    CoordTrait, GeometryCollectionTrait, GeometryTrait, LineStringTrait, MultiPolygonTrait,
    PolygonTrait,
};
use sedona_common::sedona_internal_err;
use sedona_expr::item_crs::ItemCrsKernel;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};
use wkb::reader::Wkb;

use crate::executor::WkbExecutor;

pub fn st_ispolygoncw_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_ispolygoncw",
        ItemCrsKernel::wrap_impl(vec![Arc::new(STIsPolygonOriented {
            exterior_ccw: false,
        })]),
        Volatility::Immutable,
    )
}

pub fn st_ispolygonccw_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_ispolygonccw",
        ItemCrsKernel::wrap_impl(vec![Arc::new(STIsPolygonOriented { exterior_ccw: true })]),
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct STIsPolygonOriented {
    exterior_ccw: bool,
}

impl SedonaScalarKernel for STIsPolygonOriented {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry()],
            SedonaType::Arrow(DataType::Boolean),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[datafusion_expr::ColumnarValue],
    ) -> Result<datafusion_expr::ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = BooleanBuilder::with_capacity(executor.num_iterations());

        executor.execute_wkb_void(|maybe_item| {
            match maybe_item {
                Some(item) => {
                    builder.append_value(has_orientation(item, self.exterior_ccw)?);
                }
                None => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn has_orientation(item: &Wkb, exterior_ccw: bool) -> Result<bool> {
    match item.as_type() {
        geo_traits::GeometryType::Polygon(polygon) => {
            Ok(polygon_has_orientation(polygon, exterior_ccw))
        }
        geo_traits::GeometryType::MultiPolygon(multipolygon) => Ok(multipolygon
            .polygons()
            .all(|polygon| polygon_has_orientation(&polygon, exterior_ccw))),
        geo_traits::GeometryType::GeometryCollection(collection) => {
            collection.geometries().try_fold(true, |acc, item| {
                has_orientation(item, exterior_ccw).map(|oriented| acc && oriented)
            })
        }
        geo_traits::GeometryType::Point(_)
        | geo_traits::GeometryType::LineString(_)
        | geo_traits::GeometryType::MultiPoint(_)
        | geo_traits::GeometryType::MultiLineString(_) => Ok(true),
        _ => sedona_internal_err!("Invalid geometry type"),
    }
}

fn polygon_has_orientation(polygon: &impl PolygonTrait<T = f64>, exterior_ccw: bool) -> bool {
    let Some(exterior) = polygon.exterior() else {
        return true;
    };
    if exterior.num_coords() == 0 {
        return true;
    }
    is_ring_ccw(&exterior) == exterior_ccw
        && polygon
            .interiors()
            .all(|interior| is_ring_ccw(&interior) != exterior_ccw)
}

fn is_ring_ccw(ring: &impl LineStringTrait<T = f64>) -> bool {
    let mut twice_area = 0.0;
    let mut coords = ring.coords();
    let Some(first) = coords.next() else {
        return false;
    };
    let origin = (first.x(), first.y());
    let mut previous = (0.0, 0.0);
    for coord in coords {
        let current = (coord.x() - origin.0, coord.y() - origin.1);
        twice_area += previous.0 * current.1 - current.0 * previous.1;
        previous = current;
    }
    twice_area > 0.0
}

#[cfg(test)]
mod tests {
    use arrow_array::{ArrayRef, create_array as arrow_array};
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS, WKB_VIEW_GEOMETRY};
    use sedona_testing::{compare::assert_array_equal, testers::ScalarUdfTester};

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_ispolygoncw_udf().into();
        assert_eq!(udf.name(), "st_ispolygoncw");
        let udf: ScalarUDF = st_ispolygonccw_udf().into();
        assert_eq!(udf.name(), "st_ispolygonccw");
    }

    const INPUTS: [Option<&str>; 14] = [
        None,
        Some("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
        Some("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"),
        Some("POLYGON ((0 0, 0 4, 4 4, 4 0, 0 0), (1 1, 2 1, 2 2, 1 2, 1 1))"),
        Some("POLYGON ((0 0, 0 4, 4 4, 4 0, 0 0), (1 1, 1 2, 2 2, 2 1, 1 1))"),
        Some("MULTIPOLYGON (((0 0, 0 1, 1 1, 1 0, 0 0)), ((5 5, 6 5, 6 6, 5 6, 5 5)))"),
        Some(
            "GEOMETRYCOLLECTION (POINT (1 1), GEOMETRYCOLLECTION (POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))))",
        ),
        Some("LINESTRING (0 0, 1 1)"),
        Some("POLYGON EMPTY"),
        Some("GEOMETRYCOLLECTION EMPTY"),
        Some("POLYGON ((0 0, 1 1, 2 2, 0 0))"),
        Some("MULTIPOLYGON EMPTY"),
        Some(
            "POLYGON ((10000000 10000000, 10000000.1 10000000, 10000000.1 10000000.1, 10000000 10000000.1, 10000000 10000000))",
        ),
        Some(
            "POLYGON ((10000000 10000000, 10000000 10000000.1, 10000000.1 10000000.1, 10000000.1 10000000, 10000000 10000000))",
        ),
    ];

    #[rstest]
    fn udf_cw(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY, WKB_GEOMETRY_ITEM_CRS.clone())]
        sedona_type: SedonaType,
    ) {
        let tester = ScalarUdfTester::new(st_ispolygoncw_udf().into(), vec![sedona_type]);
        tester.assert_return_type(DataType::Boolean);

        let result = tester.invoke_wkb_scalar(None).unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Null);

        let expected: ArrayRef = arrow_array!(
            Boolean,
            [
                None,
                Some(true),
                Some(false),
                Some(true),
                Some(false),
                Some(false),
                Some(true),
                Some(true),
                Some(true),
                Some(true),
                Some(true),
                Some(true),
                Some(false),
                Some(true)
            ]
        );
        assert_array_equal(
            &tester.invoke_wkb_array(INPUTS.to_vec()).unwrap(),
            &expected,
        );
    }

    #[rstest]
    fn udf_ccw(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY, WKB_GEOMETRY_ITEM_CRS.clone())]
        sedona_type: SedonaType,
    ) {
        let tester = ScalarUdfTester::new(st_ispolygonccw_udf().into(), vec![sedona_type]);
        tester.assert_return_type(DataType::Boolean);

        let expected: ArrayRef = arrow_array!(
            Boolean,
            [
                None,
                Some(false),
                Some(true),
                Some(false),
                Some(false),
                Some(false),
                Some(false),
                Some(true),
                Some(true),
                Some(true),
                Some(false),
                Some(true),
                Some(true),
                Some(false)
            ]
        );
        assert_array_equal(
            &tester.invoke_wkb_array(INPUTS.to_vec()).unwrap(),
            &expected,
        );
    }
}
