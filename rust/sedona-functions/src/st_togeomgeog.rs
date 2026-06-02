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

use datafusion_common::{config::ConfigOptions, Result};
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_common::sedona_internal_err;
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{SedonaScalarKernel, SedonaScalarUDF},
};
use sedona_schema::{
    datatypes::{Edges, SedonaType, WKB_GEOGRAPHY, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

use crate::st_setsrid::validate_crs_for_type;

/// ST_ToGeometry() scalar UDF
///
/// Converts a geography to a geometry by changing the edge interpretation from
/// spherical to planar. This is a metadata-only operation that does not modify
/// the underlying coordinate data.
pub fn st_togeometry_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_togeometry",
        ItemCrsKernel::wrap_impl(vec![Arc::new(STToGeomGeog {
            matcher: ArgMatcher::new(vec![ArgMatcher::is_geometry_or_geography()], WKB_GEOMETRY),
            target_edges: Edges::Planar,
        })]),
        Volatility::Immutable,
    )
}

/// ST_ToGeography() scalar UDF
///
/// Converts a geometry to a geography by changing the edge interpretation from
/// planar to spherical. This is a metadata-only operation that does not modify
/// the underlying coordinate data.
pub fn st_togeography_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_togeography",
        ItemCrsKernel::wrap_impl(vec![Arc::new(STToGeomGeog {
            matcher: ArgMatcher::new(vec![ArgMatcher::is_geometry_or_geography()], WKB_GEOGRAPHY),
            target_edges: Edges::Spherical,
        })]),
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct STToGeomGeog {
    matcher: ArgMatcher,
    target_edges: Edges,
}

impl SedonaScalarKernel for STToGeomGeog {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        // First check if the matcher matches
        let matched = self.matcher.match_args(args)?;
        if matched.is_none() {
            return Ok(None);
        }

        // Return the input type with the target edges. NULL returns the target edge
        // type with no CRS for the purposes of parameter binding.
        let input_type = &args[0];
        let (output_type, crs) = match input_type {
            SedonaType::Wkb(_, crs) => (SedonaType::Wkb(self.target_edges, crs.clone()), crs),
            SedonaType::WkbView(_, crs) => {
                (SedonaType::WkbView(self.target_edges, crs.clone()), crs)
            }
            _ => (SedonaType::Wkb(self.target_edges, None), &None),
        };

        // Check that the CRS is valid for the output type
        validate_crs_for_type(crs, &output_type)?;

        Ok(Some(output_type))
    }

    fn invoke_batch(
        &self,
        _arg_types: &[SedonaType],
        _args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        sedona_internal_err!("invoke_batch() should not be called")
    }

    fn invoke_batch_from_args(
        &self,
        _arg_types: &[SedonaType],
        args: &[ColumnarValue],
        return_type: &SedonaType,
        _num_rows: usize,
        _config_options: Option<&ConfigOptions>,
    ) -> Result<ColumnarValue> {
        // This is usually a shallow copy but can cast Null into a Binary
        // for Null inputs
        args[0].cast_to(return_type.storage_type(), None)
    }
}

#[cfg(test)]
mod tests {
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_expr::item_crs::parse_item_crs_arg_type;
    use sedona_schema::{
        crs::{deserialize_crs, lnglat},
        datatypes::{
            Edges, WKB_GEOGRAPHY, WKB_GEOGRAPHY_ITEM_CRS, WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS,
            WKB_VIEW_GEOGRAPHY, WKB_VIEW_GEOMETRY,
        },
    };
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    #[test]
    fn st_togeometry_metadata() {
        let udf: ScalarUDF = st_togeometry_udf().into();
        assert_eq!(udf.name(), "st_togeometry");
    }

    #[test]
    fn st_togeography_metadata() {
        let udf: ScalarUDF = st_togeography_udf().into();
        assert_eq!(udf.name(), "st_togeography");
    }

    #[rstest]
    fn st_togeometry_from_geometry(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let tester = ScalarUdfTester::new(st_togeometry_udf().into(), vec![sedona_type.clone()]);

        // Geometry input -> Geometry output (same edges)
        tester.assert_return_type(sedona_type.clone());

        let result = tester.invoke_wkb_scalar(Some("POINT (1 2)")).unwrap();
        tester.assert_scalar_result_equals(result, "POINT (1 2)");

        let result = tester.invoke_wkb_scalar(None).unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Null);
    }

    #[rstest]
    fn st_togeometry_from_geography(
        #[values(WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY)] sedona_type: SedonaType,
    ) {
        let tester = ScalarUdfTester::new(st_togeometry_udf().into(), vec![sedona_type.clone()]);

        // Geography input -> Geometry output (edges changed)
        let expected_type = match sedona_type {
            SedonaType::Wkb(_, _) => WKB_GEOMETRY,
            SedonaType::WkbView(_, _) => WKB_VIEW_GEOMETRY,
            _ => panic!("Unexpected type"),
        };
        tester.assert_return_type(expected_type);

        let result = tester.invoke_wkb_scalar(Some("POINT (1 2)")).unwrap();
        tester.assert_scalar_result_equals(result, "POINT (1 2)");
    }

    #[rstest]
    fn st_togeography_from_geography(
        #[values(WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY)] sedona_type: SedonaType,
    ) {
        let tester = ScalarUdfTester::new(st_togeography_udf().into(), vec![sedona_type.clone()]);

        // Geography input -> Geography output (same edges)
        tester.assert_return_type(sedona_type.clone());

        let result = tester.invoke_wkb_scalar(Some("POINT (1 2)")).unwrap();
        tester.assert_scalar_result_equals(result, "POINT (1 2)");

        let result = tester.invoke_wkb_scalar(None).unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Null);
    }

    #[rstest]
    fn st_togeography_from_geometry(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let tester = ScalarUdfTester::new(st_togeography_udf().into(), vec![sedona_type.clone()]);

        // Geometry input -> Geography output (edges changed)
        let expected_type = match sedona_type {
            SedonaType::Wkb(_, _) => WKB_GEOGRAPHY,
            SedonaType::WkbView(_, _) => WKB_VIEW_GEOGRAPHY,
            _ => panic!("Unexpected type"),
        };
        tester.assert_return_type(expected_type);

        let result = tester.invoke_wkb_scalar(Some("POINT (1 2)")).unwrap();
        tester.assert_scalar_result_equals(result, "POINT (1 2)");
    }

    #[rstest]
    fn st_togeometry_preserves_crs() {
        let geog_with_crs = SedonaType::Wkb(Edges::Spherical, lnglat());
        let tester = ScalarUdfTester::new(st_togeometry_udf().into(), vec![geog_with_crs.clone()]);

        // CRS should be preserved when converting
        let expected_type = SedonaType::Wkb(Edges::Planar, lnglat());
        tester.assert_return_type(expected_type);
    }

    #[rstest]
    fn st_togeography_preserves_crs() {
        let geom_with_crs = SedonaType::Wkb(Edges::Planar, lnglat());
        let tester = ScalarUdfTester::new(st_togeography_udf().into(), vec![geom_with_crs.clone()]);

        // CRS should be preserved when converting
        let expected_type = SedonaType::Wkb(Edges::Spherical, lnglat());
        tester.assert_return_type(expected_type);
    }

    #[rstest]
    fn st_togeography_invalid_crs() {
        let geom_with_crs = SedonaType::Wkb(Edges::Planar, deserialize_crs("EPSG:3857").unwrap());
        let tester = ScalarUdfTester::new(st_togeography_udf().into(), vec![geom_with_crs.clone()]);

        // CRS should be preserved when converting
        let err = tester.return_type().unwrap_err();
        assert_eq!(
            err.message(),
            "Can't assign non-geographic CRS epsg:3857 to column of type Wkb(epsg:3857, Spherical)"
        );
    }

    #[rstest]
    fn st_togeometry_item_crs(
        #[values(WKB_GEOMETRY_ITEM_CRS.clone(), WKB_GEOGRAPHY_ITEM_CRS.clone())]
        sedona_type: SedonaType,
    ) {
        let tester = ScalarUdfTester::new(st_togeometry_udf().into(), vec![sedona_type.clone()]);

        // Item CRS types should work and return item CRS types with Planar edges
        let return_type = tester.return_type().unwrap();
        assert!(return_type.is_item_crs());
        let (item_type, _) = parse_item_crs_arg_type(&return_type).unwrap();
        assert!(
            matches!(
                item_type,
                SedonaType::Wkb(Edges::Planar, _) | SedonaType::WkbView(Edges::Planar, _)
            ),
            "Expected Planar edges, got {:?}",
            item_type
        );
    }

    #[rstest]
    fn st_togeography_item_crs(
        #[values(WKB_GEOMETRY_ITEM_CRS.clone(), WKB_GEOGRAPHY_ITEM_CRS.clone())]
        sedona_type: SedonaType,
    ) {
        let tester = ScalarUdfTester::new(st_togeography_udf().into(), vec![sedona_type.clone()]);

        // Item CRS types should work and return item CRS types with Spherical edges
        let return_type = tester.return_type().unwrap();
        assert!(return_type.is_item_crs());
        let (item_type, _) = parse_item_crs_arg_type(&return_type).unwrap();
        assert!(
            matches!(
                item_type,
                SedonaType::Wkb(Edges::Spherical, _) | SedonaType::WkbView(Edges::Spherical, _)
            ),
            "Expected Spherical edges, got {:?}",
            item_type
        );
    }
}
