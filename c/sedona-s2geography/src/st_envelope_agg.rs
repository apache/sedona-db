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

use sedona_expr::{aggregate_udf::SedonaAccumulatorRef, item_crs::ItemCrsSedonaAccumulator};
use sedona_functions::st_envelope_agg::STEnvelopeAgg;
use sedona_schema::{datatypes::WKB_GEOMETRY, matchers::ArgMatcher};

use crate::rect_bounder::WkbGeographyBounder;

/// ST_Envelope_Agg() aggregate implementation for geography types
///
/// This uses geodesic calculations via S2 to compute the bounding rectangle,
/// which may return a MULTIPOLYGON for geographies that wrap around the antimeridian.
pub fn st_envelope_agg_impl() -> Vec<SedonaAccumulatorRef> {
    ItemCrsSedonaAccumulator::wrap_impl(vec![Arc::new(STEnvelopeAgg::<WkbGeographyBounder>::new(
        ArgMatcher::new(vec![ArgMatcher::is_geography()], WKB_GEOMETRY),
        false, // TODO: groups_accumulator not yet supported for geography
    ))])
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_expr::{AggregateUDF, Volatility};
    use rstest::rstest;
    use sedona_expr::aggregate_udf::SedonaAggregateUDF;
    use sedona_schema::datatypes::{
        SedonaType, WKB_GEOGRAPHY, WKB_GEOGRAPHY_ITEM_CRS, WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS,
        WKB_VIEW_GEOGRAPHY,
    };
    use sedona_testing::{compare::assert_scalar_equal_wkb_geometry, testers::AggregateUdfTester};

    fn create_udf() -> SedonaAggregateUDF {
        SedonaAggregateUDF::new(
            "st_envelope_agg",
            st_envelope_agg_impl(),
            Volatility::Immutable,
        )
    }

    #[test]
    fn udf_metadata() {
        let udf: AggregateUDF = create_udf().into();
        assert_eq!(udf.name(), "st_envelope_agg");
    }

    #[rstest]
    fn udf_aggregate(#[values(WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY)] sedona_type: SedonaType) {
        let tester = AggregateUdfTester::new(create_udf().into(), vec![sedona_type.clone()]);
        assert_eq!(tester.return_type().unwrap(), WKB_GEOMETRY);

        // Finite input with nulls
        let batches = vec![
            vec![Some("POINT (0 1)"), None, Some("POINT (2 3)")],
            vec![Some("POINT (4 5)"), None, Some("POINT (6 7)")],
        ];
        assert_scalar_equal_wkb_geometry(
            &tester.aggregate_wkt(batches).unwrap(),
            Some("POLYGON((0 1, 0 7, 6 7, 6 1, 0 1))"),
        );

        // Empty input
        assert_scalar_equal_wkb_geometry(&tester.aggregate_wkt(vec![]).unwrap(), None);

        // Degenerate output: point
        assert_scalar_equal_wkb_geometry(
            &tester
                .aggregate_wkt(vec![vec![Some("POINT (0 1)")]])
                .unwrap(),
            Some("POINT (0 1)"),
        );
    }

    #[rstest]
    fn udf_grouped_accumulate(
        #[values(WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY)] sedona_type: SedonaType,
    ) {
        let tester = AggregateUdfTester::new(create_udf().into(), vec![sedona_type.clone()]);
        assert_eq!(tester.return_type().unwrap(), WKB_GEOMETRY);

        // Test grouped aggregation using the tester's grouped method
        let batches = vec![
            vec![Some("POINT (0 1)"), None, Some("POINT (2 3)")],
            vec![Some("POINT (4 5)"), None, Some("POINT (6 7)")],
        ];
        // Just test that it works - the actual grouped_accumulate is disabled for geography
        assert_scalar_equal_wkb_geometry(
            &tester.aggregate_wkt(batches).unwrap(),
            Some("POLYGON((0 1, 0 7, 6 7, 6 1, 0 1))"),
        );
    }

    #[rstest]
    fn udf_invoke_item_crs(#[values(WKB_GEOGRAPHY_ITEM_CRS.clone())] sedona_type: SedonaType) {
        let tester = AggregateUdfTester::new(create_udf().into(), vec![sedona_type.clone()]);
        // ST_Envelope_Agg returns geometry, not geography, even for geography input
        assert_eq!(tester.return_type().unwrap(), WKB_GEOMETRY_ITEM_CRS.clone());
    }
}
