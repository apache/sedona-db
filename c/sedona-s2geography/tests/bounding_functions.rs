// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the License for the
// specific language governing permissions and limitations
// under the License.

use std::sync::Arc;

use datafusion_expr::Volatility;
use sedona_expr::{aggregate_udf::SedonaAggregateUDF, scalar_udf::SedonaScalarUDF};
use sedona_geometry::types::Edges;
use sedona_schema::datatypes::SedonaType;
use sedona_testing::testers::ScalarUdfTester;

use sedona_s2geography::rect_bounder::WkbGeographyBounder;

fn scalar_tester(udf: SedonaScalarUDF, arg_type: SedonaType) -> ScalarUdfTester {
    let mut tester = ScalarUdfTester::new(udf.into(), vec![arg_type]);
    let options = tester.sedona_options_mut();
    options.runtime = options
        .runtime
        .with_bounder(Edges::Spherical, Arc::new(WkbGeographyBounder::default()))
        .unwrap();
    tester
}

fn aggregate_udf(name: &str) -> SedonaAggregateUDF {
    let kernels = sedona_s2geography::register::aggregate_kernels()
        .into_iter()
        .find_map(|(kernel_name, kernels)| (kernel_name == name).then_some(kernels))
        .unwrap_or_else(|| panic!("aggregate kernel not found: {name}"));
    SedonaAggregateUDF::new(name, kernels, Volatility::Immutable)
}

mod st_envelope {
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_functions::st_envelope::st_envelope_udf;
    use sedona_geometry::types::Edges;
    use sedona_schema::{
        crs::lnglat,
        datatypes::{
            SedonaType, WKB_GEOGRAPHY, WKB_GEOGRAPHY_ITEM_CRS, WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS,
            WKB_VIEW_GEOGRAPHY,
        },
    };
    use sedona_testing::{
        compare::{assert_array_equal, assert_scalar_wkb_bounds_approx_equal},
        create::create_array,
    };

    use super::scalar_tester;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_envelope_udf().into();
        assert_eq!(udf.name(), "st_envelope");
    }

    #[test]
    fn udf_invoke_scalar() {
        let tester = scalar_tester(st_envelope_udf(), WKB_GEOGRAPHY);
        let result = tester
            .invoke_scalar("POLYGON ((1 2, 1 22, 11 22, 11 2, 1 2))")
            .unwrap();
        assert_scalar_wkb_bounds_approx_equal(
            &result,
            1.0,
            1.9999999999999747,
            11.0,
            22.0759758928044,
            1e-14,
        );

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
    fn udf_invoke_array(#[values(WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY)] arg_type: SedonaType) {
        let tester = scalar_tester(st_envelope_udf(), arg_type);
        tester.assert_return_type(WKB_GEOMETRY);
        let input = vec![
            None,
            Some("POINT EMPTY"),
            Some("LINESTRING EMPTY"),
            Some("POLYGON EMPTY"),
            Some("MULTIPOINT EMPTY"),
            Some("MULTILINESTRING EMPTY"),
            Some("MULTIPOLYGON EMPTY"),
            Some("GEOMETRYCOLLECTION EMPTY"),
        ];
        let expected = create_array(&input, &WKB_GEOMETRY);
        assert_array_equal(&tester.invoke_wkb_array(input).unwrap(), &expected);
    }

    #[rstest]
    fn udf_propagates_crs(
        #[values(
            SedonaType::Wkb(Edges::Spherical, lnglat()),
            SedonaType::WkbView(Edges::Spherical, lnglat())
        )]
        arg_type: SedonaType,
    ) {
        let tester = scalar_tester(st_envelope_udf(), arg_type);
        tester.assert_return_type(SedonaType::Wkb(Edges::Planar, lnglat()));
    }

    #[test]
    fn udf_invoke_item_crs() {
        let tester = scalar_tester(st_envelope_udf(), WKB_GEOGRAPHY_ITEM_CRS.clone());
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
            1e-14,
        );
    }
}

mod st_xy_minmax {
    use arrow_array::{create_array, ArrayRef};
    use arrow_schema::DataType;
    use datafusion_common::ScalarValue;
    use rstest::rstest;
    use sedona_functions::register::default_function_set;
    use sedona_schema::datatypes::{SedonaType, WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY};
    use sedona_testing::compare::assert_array_equal;

    use super::scalar_tester;

    fn tester(name: &str, arg_type: SedonaType) -> sedona_testing::testers::ScalarUdfTester {
        let function_set = default_function_set();
        let udf = function_set.scalar_udf(name).unwrap();
        scalar_tester(udf.clone(), arg_type)
    }

    fn assert_approx_equal(actual: ScalarValue, expected: f64) {
        match actual {
            ScalarValue::Float64(Some(value)) => {
                assert!(
                    (value - expected).abs() < 1e-10,
                    "expected {expected}, got {value}"
                )
            }
            _ => panic!("expected Float64, got {actual:?}"),
        }
    }

    #[rstest]
    fn udf(#[values(WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY)] arg_type: SedonaType) {
        let xmin = tester("st_xmin", arg_type.clone());
        let ymin = tester("st_ymin", arg_type.clone());
        let xmax = tester("st_xmax", arg_type.clone());
        let ymax = tester("st_ymax", arg_type);
        for tester in [&xmin, &ymin, &xmax, &ymax] {
            tester.assert_return_type(DataType::Float64);
        }

        let polygon = "POLYGON ((-1 0, 0 -2, 3 1, 0 4, -1 0))";
        assert_approx_equal(xmin.invoke_scalar(polygon).unwrap(), -1.0);
        assert_approx_equal(ymin.invoke_scalar(polygon).unwrap(), -2.0);
        assert_approx_equal(xmax.invoke_scalar(polygon).unwrap(), 3.0);
        assert_approx_equal(ymax.invoke_scalar(polygon).unwrap(), 4.0);

        let expected: ArrayRef = create_array!(Float64, [None, None, None]);
        let input = vec![None, Some("POINT EMPTY"), Some("GEOMETRYCOLLECTION EMPTY")];
        for tester in [&xmin, &ymin, &xmax, &ymax] {
            assert_array_equal(&tester.invoke_wkb_array(input.clone()).unwrap(), &expected);
        }
    }

    #[test]
    fn point_and_null_scalar() {
        let xmin = tester("st_xmin", WKB_GEOGRAPHY);
        assert_approx_equal(xmin.invoke_scalar("POINT (10 20)").unwrap(), 10.0);
        assert_eq!(
            xmin.invoke_scalar(ScalarValue::Null).unwrap(),
            ScalarValue::Float64(None)
        );
    }
}

mod st_envelope_agg {
    use arrow_array::Array;
    use datafusion_common::ScalarValue;
    use rstest::rstest;
    use sedona_schema::datatypes::{
        SedonaType, WKB_GEOGRAPHY, WKB_GEOGRAPHY_ITEM_CRS, WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS,
        WKB_VIEW_GEOGRAPHY,
    };
    use sedona_testing::{
        compare::{assert_scalar_equal_wkb_geometry, assert_scalar_wkb_bounds_approx_equal},
        create::create_array,
        testers::AggregateUdfTester,
    };

    use super::aggregate_udf;

    fn tester(arg_type: SedonaType) -> AggregateUdfTester {
        AggregateUdfTester::new(aggregate_udf("st_envelope_agg").into(), vec![arg_type])
    }

    #[rstest]
    fn udf_aggregate(#[values(WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY)] arg_type: SedonaType) {
        let tester = tester(arg_type);
        assert_eq!(tester.return_type().unwrap(), WKB_GEOMETRY);
        let batches = vec![
            vec![Some("POINT (0 1)"), None, Some("POINT (2 3)")],
            vec![Some("POINT (4 5)"), None, Some("POINT (6 7)")],
        ];
        assert_scalar_wkb_bounds_approx_equal(
            &tester.aggregate_wkt(batches).unwrap(),
            0.0,
            1.0,
            6.0,
            7.0,
            1e-13,
        );
        assert_scalar_equal_wkb_geometry(&tester.aggregate_wkt(vec![]).unwrap(), None);
    }

    #[rstest]
    fn udf_grouped_accumulate(#[values(WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY)] arg_type: SedonaType) {
        let tester = tester(arg_type.clone());
        let groups = vec![0, 3, 1, 1, 0, 2];
        let batches = vec![
            create_array(&[Some("POINT (0 1)"), None, Some("POINT (2 3)")], &arg_type),
            create_array(&[Some("POINT (4 5)"), None, Some("POINT (6 7)")], &arg_type),
        ];
        let result = tester
            .aggregate_groups(&batches, groups, None, vec![])
            .unwrap();
        assert_eq!(result.len(), 4);
        for (index, expected) in [
            Some((0.0, 1.0, 0.0, 1.0)),
            Some((2.0, 3.0, 4.0, 5.0)),
            Some((6.0, 7.0, 6.0, 7.0)),
            None,
        ]
        .into_iter()
        .enumerate()
        {
            let scalar = ScalarValue::try_from_array(&result, index).unwrap();
            match expected {
                Some((xmin, ymin, xmax, ymax)) => {
                    assert_scalar_wkb_bounds_approx_equal(&scalar, xmin, ymin, xmax, ymax, 1e-13)
                }
                None => assert!(scalar.is_null()),
            }
        }
    }

    #[test]
    fn udf_invoke_item_crs() {
        let tester = tester(WKB_GEOGRAPHY_ITEM_CRS.clone());
        assert_eq!(tester.return_type().unwrap(), WKB_GEOMETRY_ITEM_CRS.clone());
    }
}

mod st_analyze_agg {
    use arrow_array::{Float64Array, Int64Array};
    use datafusion_common::ScalarValue;
    use rstest::rstest;
    use sedona_functions::st_analyze_agg::output_sedona_type;
    use sedona_schema::datatypes::{
        SedonaType, WKB_GEOGRAPHY, WKB_GEOGRAPHY_ITEM_CRS, WKB_VIEW_GEOGRAPHY,
    };
    use sedona_testing::testers::AggregateUdfTester;

    use super::aggregate_udf;

    fn tester(arg_type: SedonaType) -> AggregateUdfTester {
        AggregateUdfTester::new(aggregate_udf("st_analyze_agg").into(), vec![arg_type])
    }

    #[rstest]
    fn udf_aggregate(#[values(WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY)] arg_type: SedonaType) {
        let tester = tester(arg_type);
        assert_eq!(tester.return_type().unwrap(), output_sedona_type());
        let result = tester
            .aggregate_wkt(vec![vec![Some("POINT(179 0)"), Some("POINT(-179 1)")]])
            .unwrap();
        let ScalarValue::Struct(struct_array) = result else {
            panic!("expected struct result")
        };
        let count = struct_array
            .column_by_name("count")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(count.value(0), 2);
        let minx = struct_array
            .column_by_name("minx")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(0);
        let maxx = struct_array
            .column_by_name("maxx")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(0);
        assert!(maxx - minx < 10.0);
    }

    #[test]
    fn udf_invoke_item_crs() {
        let tester = tester(WKB_GEOGRAPHY_ITEM_CRS.clone());
        assert_eq!(tester.return_type().unwrap(), output_sedona_type());
    }
}
