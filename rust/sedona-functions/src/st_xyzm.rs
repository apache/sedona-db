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
use std::{sync::Arc, vec};

use crate::executor::WkbExecutor;
use arrow_array::builder::Float64Builder;
use arrow_schema::DataType;
use datafusion_common::error::{DataFusionError, Result};
use datafusion_expr::{ColumnarValue, Volatility};
use geo_traits::{
    CoordTrait, Dimensions, GeometryCollectionTrait, GeometryTrait, LineStringTrait,
    MultiLineStringTrait, MultiPointTrait, MultiPolygonTrait, PointTrait, PolygonTrait,
};
use sedona_common::{sedona_internal_datafusion_err, sedona_internal_err};
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{SedonaScalarKernel, SedonaScalarUDF},
};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};
use wkb::reader::Wkb;

/// ST_X() scalar UDF implementation
///
/// Extract the X coordinate from Point geometries or geographies
pub fn st_x_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_x",
        ItemCrsKernel::wrap_impl(vec![Arc::new(STXyzm { dim: "x" })]),
        Volatility::Immutable,
    )
}

/// ST_Y() scalar UDF implementation
///
/// Extract the Y coordinate from point geometries or geographies
pub fn st_y_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_y",
        ItemCrsKernel::wrap_impl(vec![Arc::new(STXyzm { dim: "y" })]),
        Volatility::Immutable,
    )
}

/// ST_Z() scalar UDF implementation
///
/// Extract the Z coordinate from Point geometries or geographies
pub fn st_z_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_z",
        ItemCrsKernel::wrap_impl(vec![Arc::new(STXyzm { dim: "z" })]),
        Volatility::Immutable,
    )
}

/// ST_M() scalar UDF implementation
///
/// Extract the M coordinate from point geometries or geographies
pub fn st_m_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_m",
        ItemCrsKernel::wrap_impl(vec![Arc::new(STXyzm { dim: "m" })]),
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct STXyzm {
    dim: &'static str,
}

impl SedonaScalarKernel for STXyzm {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry_or_geography()],
            SedonaType::Arrow(DataType::Float64),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let dim_index = match self.dim {
            "x" => 0,
            "y" => 1,
            "z" => 2,
            "m" => 3,
            _ => sedona_internal_err!("unexpected dimension")?,
        };

        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = Float64Builder::with_capacity(executor.num_iterations());

        executor.execute_wkb_void(|maybe_item| {
            match maybe_item {
                Some(item) => {
                    builder.append_option(invoke_scalar(&item, dim_index)?);
                }
                None => builder.append_null(),
            }
            Ok(())
        })?;

        // Create the output array
        executor.finish(Arc::new(builder.finish()))
    }
}

// Extracts the 0th or 1st dimension from any point-like or EMPTY GeometryTrait
//
// Note that PostGIS will fail for anything that is not POINT (whereas we succeed for any
// EMPTY).
fn invoke_scalar(item: &Wkb, dim_index: usize) -> Result<Option<f64>> {
    match item.as_type() {
        geo_traits::GeometryType::Point(point) => {
            let coord_dim = point.dim();
            let coord = PointTrait::coord(point);
            return get_coord(coord_dim, coord, dim_index);
        }
        geo_traits::GeometryType::LineString(linestring) => {
            if LineStringTrait::num_coords(linestring) == 0 {
                return Ok(None);
            }
        }
        geo_traits::GeometryType::Polygon(polygon) => {
            if PolygonTrait::exterior(polygon).is_none() {
                return Ok(None);
            }
        }
        geo_traits::GeometryType::MultiPoint(multipoint) => {
            match MultiPointTrait::num_points(multipoint) {
                0 => return Ok(None),
                1 => {
                    let coord_dim = multipoint.dim();
                    let point = MultiPointTrait::point(multipoint, 0)
                        .ok_or(sedona_internal_datafusion_err!("Missing point"))?;
                    let coord = PointTrait::coord(&point);
                    return get_coord(coord_dim, coord, dim_index);
                }
                _ => {}
            }
        }
        geo_traits::GeometryType::MultiLineString(multilinestring) => {
            if MultiLineStringTrait::num_line_strings(multilinestring) == 0 {
                return Ok(None);
            }
        }
        geo_traits::GeometryType::MultiPolygon(multipolygon) => {
            if MultiPolygonTrait::num_polygons(multipolygon) == 0 {
                return Ok(None);
            }
        }
        geo_traits::GeometryType::GeometryCollection(geometrycollection) => {
            if geometrycollection.num_geometries() == 0 {
                return Ok(None);
            }
        }
        _ => {}
    };

    Err(DataFusionError::Execution("Expected POINT".to_string()))
}

// Extracts the coordinate value from a coord object
pub fn get_coord<C>(
    coord_dim: Dimensions,
    coord: Option<C>,
    dim_index: usize,
) -> Result<Option<f64>>
where
    C: CoordTrait<T = f64>,
{
    match dim_index {
        0 | 1 => return Ok(coord.map(|c| c.nth_or_panic(dim_index))),
        2 => {
            if matches!(coord_dim, Dimensions::Xyz | Dimensions::Xyzm) {
                return Ok(coord.map(|c| c.nth_or_panic(dim_index)));
            }
        }
        3 => match coord_dim {
            Dimensions::Xym => {
                return Ok(coord.map(|c| c.nth_or_panic(2)));
            }
            Dimensions::Xyzm => {
                return Ok(coord.map(|c| c.nth_or_panic(3)));
            }
            _ => {}
        },
        _ => sedona_internal_err!("unexpected dimension index")?,
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{create_array, ArrayRef};
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{
        WKB_GEOGRAPHY, WKB_GEOGRAPHY_ITEM_CRS, WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS,
        WKB_VIEW_GEOGRAPHY, WKB_VIEW_GEOMETRY,
    };
    use sedona_testing::{
        create::create_array, fixtures::MULTIPOINT_WITH_EMPTY_CHILD_WKB, testers::ScalarUdfTester,
    };

    #[test]
    fn udf_metadata() {
        let udf_x: ScalarUDF = st_x_udf().into();
        assert_eq!(udf_x.name(), "st_x");
        assert!(udf_x.documentation().is_none());

        let udf_y: ScalarUDF = st_y_udf().into();
        assert_eq!(udf_y.name(), "st_y");
        assert!(udf_y.documentation().is_none());

        let udf_z: ScalarUDF = st_z_udf().into();
        assert_eq!(udf_z.name(), "st_z");
        assert!(udf_z.documentation().is_none());

        let udf_m: ScalarUDF = st_m_udf().into();
        assert_eq!(udf_m.name(), "st_m");
        assert!(udf_m.documentation().is_none());
    }

    #[rstest]
    fn udf_invoke(
        #[values(WKB_GEOMETRY, WKB_GEOGRAPHY, WKB_VIEW_GEOMETRY, WKB_VIEW_GEOGRAPHY, WKB_GEOMETRY_ITEM_CRS.clone(), WKB_GEOGRAPHY_ITEM_CRS.clone())]
        sedona_type: SedonaType,
    ) {
        let x_tester = ScalarUdfTester::new(st_x_udf().into(), vec![sedona_type.clone()]);
        let y_tester = ScalarUdfTester::new(st_y_udf().into(), vec![sedona_type.clone()]);
        let z_tester = ScalarUdfTester::new(st_z_udf().into(), vec![sedona_type.clone()]);
        let m_tester = ScalarUdfTester::new(st_m_udf().into(), vec![sedona_type.clone()]);

        x_tester.assert_return_type(DataType::Float64);
        y_tester.assert_return_type(DataType::Float64);
        z_tester.assert_return_type(DataType::Float64);
        m_tester.assert_return_type(DataType::Float64);

        x_tester.assert_scalar_result_equals(
            x_tester.invoke_wkb_scalar(Some("POINT (1 2)")).unwrap(),
            ScalarValue::Float64(Some(1.0)),
        );
        y_tester.assert_scalar_result_equals(
            y_tester.invoke_wkb_scalar(Some("POINT (1 2)")).unwrap(),
            ScalarValue::Float64(Some(2.0)),
        );
        z_tester.assert_scalar_result_equals(
            z_tester
                .invoke_wkb_scalar(Some("POINT ZM (1 2 3 4)"))
                .unwrap(),
            ScalarValue::Float64(Some(3.0)),
        );
        m_tester.assert_scalar_result_equals(
            m_tester
                .invoke_wkb_scalar(Some("POINT ZM (1 2 3 4)"))
                .unwrap(),
            ScalarValue::Float64(Some(4.0)),
        );

        let wkb_array = create_array(
            &[
                Some("POINT (1 2)"),
                None,
                Some("MULTIPOINT (3 4)"),
                Some("POINT ZM (1 2 3 4)"),
                Some("POINT M (1 2 4)"),
                Some("POINT Z (1 2 3)"),
                Some("MULTIPOINT Z (1 2 3)"),
                Some("MULTIPOINT M (1 2 3)"),
                Some("MULTIPOINT ZM (1 2 3 4)"),
            ],
            &sedona_type,
        );
        let expected_x: ArrayRef = create_array!(
            Float64,
            [
                Some(1.0),
                None,
                Some(3.0),
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(1.0)
            ]
        );
        let expected_y: ArrayRef = create_array!(
            Float64,
            [
                Some(2.0),
                None,
                Some(4.0),
                Some(2.0),
                Some(2.0),
                Some(2.0),
                Some(2.0),
                Some(2.0),
                Some(2.0)
            ]
        );
        let expected_z: ArrayRef = create_array!(
            Float64,
            [
                None,
                None,
                None,
                Some(3.0),
                None,
                Some(3.0),
                Some(3.0),
                None,
                Some(3.0)
            ]
        );
        let expected_m: ArrayRef = create_array!(
            Float64,
            [
                None,
                None,
                None,
                Some(4.0),
                Some(4.0),
                None,
                None,
                Some(3.0),
                Some(4.0)
            ]
        );
        assert_eq!(
            &x_tester.invoke_array(wkb_array.clone()).unwrap(),
            &expected_x
        );
        assert_eq!(
            &y_tester.invoke_array(wkb_array.clone()).unwrap(),
            &expected_y
        );
        assert_eq!(
            &z_tester.invoke_array(wkb_array.clone()).unwrap(),
            &expected_z
        );
        assert_eq!(
            &m_tester.invoke_array(wkb_array.clone()).unwrap(),
            &expected_m
        );
    }

    #[rstest]
    fn udf_empties(
        #[values(
            "POINT",
            "LINESTRING",
            "POLYGON",
            "MULTIPOINT",
            "MULTILINESTRING",
            "MULTIPOLYGON",
            "GEOMETRYCOLLECTION"
        )]
        geom_type: &str,
    ) {
        let x_tester = ScalarUdfTester::new(st_x_udf().into(), vec![WKB_GEOMETRY]);
        let y_tester = ScalarUdfTester::new(st_y_udf().into(), vec![WKB_GEOMETRY]);
        let z_tester = ScalarUdfTester::new(st_z_udf().into(), vec![WKB_GEOMETRY]);
        let m_tester = ScalarUdfTester::new(st_m_udf().into(), vec![WKB_GEOMETRY]);

        let wkt_empty = format!("{geom_type} EMPTY");
        assert_eq!(
            x_tester.invoke_wkb_scalar(Some(&wkt_empty)).unwrap(),
            ScalarValue::Float64(None)
        );
        assert_eq!(
            y_tester.invoke_wkb_scalar(Some(&wkt_empty)).unwrap(),
            ScalarValue::Float64(None)
        );
        assert_eq!(
            z_tester.invoke_wkb_scalar(Some(&wkt_empty)).unwrap(),
            ScalarValue::Float64(None)
        );
        assert_eq!(
            m_tester.invoke_wkb_scalar(Some(&wkt_empty)).unwrap(),
            ScalarValue::Float64(None)
        );
    }

    #[rstest]
    fn wrong_geometry_type(
        #[values(
            "LINESTRING (0 1, 2 3)",
            "POLYGON ((0 0, 0 1, 1 0, 0 0))",
            "MULTIPOINT ((0 1), (2 3))",
            "MULTILINESTRING ((0 1, 2 3))",
            "MULTIPOLYGON (((0 0, 0 1, 1 0, 0 0)))",
            "GEOMETRYCOLLECTION (POINT (0 1))"
        )]
        bad_wkt: &str,
    ) {
        let x_tester = ScalarUdfTester::new(st_x_udf().into(), vec![WKB_GEOMETRY]);

        let err = x_tester.invoke_wkb_scalar(Some(bad_wkt)).unwrap_err();
        assert_eq!(err.message(), "Expected POINT");
    }

    #[test]
    fn multipoint_with_empty_child() {
        let x_tester = ScalarUdfTester::new(st_x_udf().into(), vec![WKB_GEOMETRY]);
        let y_tester = ScalarUdfTester::new(st_y_udf().into(), vec![WKB_GEOMETRY]);
        let z_tester = ScalarUdfTester::new(st_z_udf().into(), vec![WKB_GEOMETRY]);
        let m_tester = ScalarUdfTester::new(st_m_udf().into(), vec![WKB_GEOMETRY]);

        let scalar = ScalarValue::Binary(Some(MULTIPOINT_WITH_EMPTY_CHILD_WKB.to_vec()));
        assert_eq!(
            x_tester.invoke_scalar(scalar.clone()).unwrap(),
            ScalarValue::Float64(None)
        );
        assert_eq!(
            y_tester.invoke_scalar(scalar.clone()).unwrap(),
            ScalarValue::Float64(None)
        );
        assert_eq!(
            z_tester.invoke_scalar(scalar.clone()).unwrap(),
            ScalarValue::Float64(None)
        );
        assert_eq!(
            m_tester.invoke_scalar(scalar.clone()).unwrap(),
            ScalarValue::Float64(None)
        );
    }
}
