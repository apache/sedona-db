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

use sedona_expr::{item_crs::ItemCrsKernel, scalar_udf::ScalarKernelRef};
use sedona_functions::st_envelope::STEnvelope;
use sedona_schema::{datatypes::WKB_GEOMETRY, matchers::ArgMatcher};

use crate::rect_bounder::WkbGeographyBounder;

/// Returns a vector of (function_name, kernel) tuples for the ST_Envelope function
/// Includes both the base kernel and ItemCrs-wrapped version
pub fn st_envelope_kernels() -> Vec<(String, ScalarKernelRef)> {
    ItemCrsKernel::wrap_impl(vec![
        Arc::new(STEnvelope::<WkbGeographyBounder>::new(ArgMatcher::new(
            vec![ArgMatcher::is_geography()],
            WKB_GEOMETRY,
        ))) as ScalarKernelRef,
    ])
    .into_iter()
    .map(|kernel| ("st_envelope".to_string(), kernel))
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_expr::scalar_udf::SedonaScalarUDF;

    use sedona_schema::{
        crs::lnglat,
        datatypes::{
            Edges, SedonaType, WKB_GEOGRAPHY, WKB_GEOGRAPHY_ITEM_CRS, WKB_GEOMETRY,
            WKB_GEOMETRY_ITEM_CRS, WKB_VIEW_GEOGRAPHY,
        },
    };
    use sedona_testing::{
        compare::{assert_array_equal, assert_scalar_wkb_bounds_approx_equal},
        create::create_array,
        testers::ScalarUdfTester,
    };

    fn create_udf() -> SedonaScalarUDF {
        let kernels = st_envelope_kernels();
        SedonaScalarUDF::new(
            "st_envelope",
            kernels.into_iter().map(|(_, k)| k).collect(),
            datafusion_expr::Volatility::Immutable,
        )
    }

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = create_udf().into();
        assert_eq!(udf.name(), "st_envelope");
    }

    #[test]
    fn udf_invoke_scalar() {
        let tester = ScalarUdfTester::new(create_udf().into(), vec![WKB_GEOGRAPHY]);

        // Test with a polygon
        let result = tester
            .invoke_scalar("POLYGON ((1 2, 1 22, 11 22, 11 2, 1 2))")
            .unwrap();
        assert_scalar_wkb_bounds_approx_equal(
            &result,
            1.0,
            1.9999999999999747,
            11.0,
            22.0759758928044,
            f64::EPSILON,
        );

        // Test with a linestring crossing the antimeridian - should return MULTIPOLYGON
        let result = tester
            .invoke_scalar("LINESTRING (170 10, -170 20)")
            .unwrap();
        assert_scalar_wkb_bounds_approx_equal(
            &result,
            -180.0,
            9.999999999999975,
            180.0,
            20.000000000000025,
            f64::EPSILON,
        );
    }

    #[rstest]
    fn udf_invoke_array(#[values(WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY)] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(create_udf().into(), vec![sedona_type.clone()]);
        tester.assert_return_type(WKB_GEOMETRY);

        // We only test empties here because of the floating point expansion used in the bounder
        let input_wkt = vec![
            None,
            Some("POINT EMPTY"),
            Some("LINESTRING EMPTY"),
            Some("POLYGON EMPTY"),
            Some("MULTIPOINT EMPTY"),
            Some("MULTILINESTRING EMPTY"),
            Some("MULTIPOLYGON EMPTY"),
            Some("GEOMETRYCOLLECTION EMPTY"),
        ];
        let expected = create_array(
            &[
                None,
                Some("POINT EMPTY"),
                Some("LINESTRING EMPTY"),
                Some("POLYGON EMPTY"),
                Some("MULTIPOINT EMPTY"),
                Some("MULTILINESTRING EMPTY"),
                Some("MULTIPOLYGON EMPTY"),
                Some("GEOMETRYCOLLECTION EMPTY"),
            ],
            &WKB_GEOMETRY,
        );
        assert_array_equal(&tester.invoke_wkb_array(input_wkt).unwrap(), &expected);
    }

    #[rstest]
    fn udf_propagate_crs(
        #[values(
            SedonaType::Wkb(Edges::Spherical, lnglat()),
            SedonaType::WkbView(Edges::Spherical, lnglat())
        )]
        sedona_type: SedonaType,
    ) {
        let tester = ScalarUdfTester::new(create_udf().into(), vec![sedona_type.clone()]);
        tester.assert_return_type(SedonaType::Wkb(Edges::Planar, lnglat()));
    }

    #[rstest]
    fn udf_invoke_item_crs(#[values(WKB_GEOGRAPHY_ITEM_CRS.clone())] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(create_udf().into(), vec![sedona_type.clone()]);
        // ST_Envelope returns geometry (planar), not geography, even for geography input
        tester.assert_return_type(WKB_GEOMETRY_ITEM_CRS.clone());

        let result = tester
            .invoke_scalar("POLYGON ((1 2, 1 22, 11 22, 11 2, 1 2))")
            .unwrap();
        assert_scalar_wkb_bounds_approx_equal(
            &result,
            1.0,
            1.9999999999999747,
            11.0,
            22.0759758928044,
            f64::EPSILON,
        );
    }
}
