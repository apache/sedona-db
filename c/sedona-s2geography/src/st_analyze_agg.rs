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

use arrow_schema::DataType;
use sedona_expr::{aggregate_udf::SedonaAccumulatorRef, item_crs::ItemCrsSedonaAccumulator};
use sedona_functions::st_analyze_agg::STAnalyzeAgg;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::rect_bounder::WkbGeographyBounder;

/// ST_Analyze_Agg() aggregate implementation for geography types
///
/// This uses geodesic calculations via S2 to compute the bounding rectangle,
/// which handles wraparound correctly for geographies near the antimeridian.
pub fn st_analyze_agg_impl() -> Vec<SedonaAccumulatorRef> {
    let output_fields = STAnalyzeAgg::<WkbGeographyBounder>::output_fields();
    let output_type = SedonaType::Arrow(DataType::Struct(output_fields.into()));

    ItemCrsSedonaAccumulator::wrap_impl(vec![Arc::new(STAnalyzeAgg::<WkbGeographyBounder>::new(
        ArgMatcher::new(vec![ArgMatcher::is_geography()], output_type),
    ))])
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::DataType;
    use datafusion_common::ScalarValue;
    use datafusion_expr::{AggregateUDF, Volatility};
    use rstest::rstest;
    use sedona_expr::aggregate_udf::SedonaAggregateUDF;
    use sedona_schema::datatypes::{
        SedonaType, WKB_GEOGRAPHY, WKB_GEOGRAPHY_ITEM_CRS, WKB_VIEW_GEOGRAPHY,
    };
    use sedona_testing::testers::AggregateUdfTester;

    fn create_udf() -> SedonaAggregateUDF {
        SedonaAggregateUDF::new(
            "st_analyze_agg",
            st_analyze_agg_impl(),
            Volatility::Immutable,
        )
    }

    fn output_type() -> SedonaType {
        let output_fields = STAnalyzeAgg::<WkbGeographyBounder>::output_fields();
        SedonaType::Arrow(DataType::Struct(output_fields.into()))
    }

    #[test]
    fn udf_metadata() {
        let udf: AggregateUDF = create_udf().into();
        assert_eq!(udf.name(), "st_analyze_agg");
    }

    #[rstest]
    fn udf_aggregate(#[values(WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY)] sedona_type: SedonaType) {
        let tester = AggregateUdfTester::new(create_udf().into(), vec![sedona_type.clone()]);
        assert_eq!(tester.return_type().unwrap(), output_type());

        // Basic point analysis
        let result = tester
            .aggregate_wkt(vec![vec![Some("POINT(179 0)"), Some("POINT(-179 1)")]])
            .unwrap();
        assert!(matches!(result, ScalarValue::Struct(_)));

        // Check that count is correct and bbox width is reasonable
        if let ScalarValue::Struct(struct_array) = result {
            let count_array = struct_array.column_by_name("count").unwrap();
            let count_arr = count_array
                .as_any()
                .downcast_ref::<arrow_array::Int64Array>()
                .unwrap();
            assert_eq!(count_arr.value(0), 2);

            // Check bbox width - for points at 179 and -179 longitude,
            // a geodesic bbox should wrap around the antimeridian with small width (~2 degrees)
            let minx = struct_array
                .column_by_name("minx")
                .unwrap()
                .as_any()
                .downcast_ref::<arrow_array::Float64Array>()
                .unwrap()
                .value(0);
            let maxx = struct_array
                .column_by_name("maxx")
                .unwrap()
                .as_any()
                .downcast_ref::<arrow_array::Float64Array>()
                .unwrap()
                .value(0);
            let bbox_width = maxx - minx;
            // Width should be small (crossing antimeridian), not ~358 degrees
            assert!(
                bbox_width < 10.0,
                "bbox width should be small for antimeridian crossing, got {}",
                bbox_width
            );
        }

        // Empty input
        let empty_result = tester.aggregate_wkt(vec![]).unwrap();
        if let ScalarValue::Struct(struct_array) = empty_result {
            let count_array = struct_array.column_by_name("count").unwrap();
            let count_arr = count_array
                .as_any()
                .downcast_ref::<arrow_array::Int64Array>()
                .unwrap();
            assert_eq!(count_arr.value(0), 0);
        }
    }

    #[rstest]
    fn udf_invoke_item_crs(#[values(WKB_GEOGRAPHY_ITEM_CRS.clone())] sedona_type: SedonaType) {
        let tester = AggregateUdfTester::new(create_udf().into(), vec![sedona_type.clone()]);
        // ST_Analyze_Agg returns struct type. Since it's not a geometry/geography type,
        // the item_crs wrapper doesn't apply - output is the plain struct type.
        assert_eq!(tester.return_type().unwrap(), output_type());
    }
}
