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

use arrow_schema::DataType;
use datafusion_common::error::Result;
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY, WKB_VIEW_GEOGRAPHY, WKB_VIEW_GEOMETRY},
    matchers::ArgMatcher,
};

use crate::{executor::WkbExecutor, st_setsrid::SRIDifiedKernel};

/// ST_GeomFromWKB() scalar UDF implementation
///
/// An implementation of WKB reading using GeoRust's wkb crate.
pub fn st_geomfromwkb_udf() -> SedonaScalarUDF {
    let kernel = Arc::new(STGeomFromWKB {
        validate: true,
        out_type: WKB_VIEW_GEOMETRY,
    });
    let sridified_kernel = Arc::new(SRIDifiedKernel::new(kernel.clone()));
    SedonaScalarUDF::new(
        "st_geomfromwkb",
        vec![sridified_kernel, kernel],
        Volatility::Immutable,
        None,
    )
}

/// ST_GeomFromWKBUnchecked() scalar UDF implementation
///
/// An implementation of WKB reading using GeoRust's wkb crate without validation.
pub fn st_geomfromwkbunchecked_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_geomfromwkbunchecked",
        vec![Arc::new(STGeomFromWKB {
            validate: false,
            out_type: WKB_VIEW_GEOMETRY,
        })],
        Volatility::Immutable,
        None,
    )
}

/// ST_GeogFromWKB() scalar UDF implementation
///
/// An implementation of WKB reading using GeoRust's wkb crate.
pub fn st_geogfromwkb_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_geogfromwkb",
        vec![Arc::new(STGeomFromWKB {
            validate: true,
            out_type: WKB_VIEW_GEOGRAPHY,
        })],
        Volatility::Immutable,
        None,
    )
}

/// for `ST_GeomFromWKBUnchecked()`.
///
/// Parameterized for reuse if `ST_GeogFromWKBUnchecked()` is implemented in the future.
#[derive(Debug)]
struct STGeomFromWKB {
    validate: bool,
    out_type: SedonaType,
}

impl SedonaScalarKernel for STGeomFromWKB {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(vec![ArgMatcher::is_binary()], self.out_type.clone());
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        if self.validate {
            let iter_type = match &arg_types[0] {
                SedonaType::Arrow(data_type) => match data_type {
                    DataType::Binary => WKB_GEOMETRY,
                    DataType::BinaryView => WKB_VIEW_GEOGRAPHY,
                    _ => unreachable!(),
                },
                _ => {
                    unreachable!()
                }
            };

            let temp_args = [iter_type];
            let executor = WkbExecutor::new(&temp_args, args);
            executor.execute_wkb_void(|_maybe_item| Ok(()))?;
        }

        args[0].cast_to(self.out_type.storage_type(), None)
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{create_array, ArrayRef, BinaryArray, BinaryViewArray};
    use datafusion_common::scalar::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_testing::{
        compare::{assert_array_equal, assert_scalar_equal},
        create::{create_array, create_array_item_crs, create_array_storage, create_scalar},
        testers::ScalarUdfTester,
    };

    use super::*;

    const POINT12: [u8; 21] = [
        0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x40,
    ];
    const INVALID_WKB_LEN: [u8; 0] = [];
    const INVALID_WKB_CONTENT: [u8; 5] = [0x01, 0x00, 0x00, 0x00, 0x00];
    const INVALID_WKBS: [&[u8]; 2] = [&INVALID_WKB_LEN, &INVALID_WKB_CONTENT];

    #[test]
    fn udf_metadata() {
        let geog_from_wkb: ScalarUDF = st_geogfromwkb_udf().into();
        assert_eq!(geog_from_wkb.name(), "st_geogfromwkb");
        assert!(geog_from_wkb.documentation().is_none());

        let geom_from_wkb: ScalarUDF = st_geomfromwkb_udf().into();
        assert_eq!(geom_from_wkb.name(), "st_geomfromwkb");
        assert!(geom_from_wkb.documentation().is_none());

        let geom_from_wkb_unchecked: ScalarUDF = st_geomfromwkbunchecked_udf().into();
        assert_eq!(geom_from_wkb_unchecked.name(), "st_geomfromwkbunchecked");
        assert!(geom_from_wkb_unchecked.documentation().is_none());
    }

    #[rstest]
    fn udf(#[values(DataType::Binary, DataType::BinaryView)] data_type: DataType) {
        let udf = st_geomfromwkb_udf();
        let tester = ScalarUdfTester::new(
            udf.clone().into(),
            vec![SedonaType::Arrow(data_type.clone())],
        );

        assert_eq!(tester.return_type().unwrap(), WKB_VIEW_GEOMETRY);

        assert_scalar_equal(
            &tester.invoke_scalar(POINT12.to_vec()).unwrap(),
            &create_scalar(Some("POINT (1 2)"), &WKB_VIEW_GEOMETRY),
        );

        assert_scalar_equal(
            &tester.invoke_scalar(ScalarValue::Null).unwrap(),
            &create_scalar(None, &WKB_VIEW_GEOMETRY),
        );

        let binary_array: BinaryArray = [Some(POINT12), None, Some(POINT12)].iter().collect();
        assert_array_equal(
            &tester.invoke_array(Arc::new(binary_array)).unwrap(),
            &create_array(
                &[Some("POINT (1 2)"), None, Some("POINT (1 2)")],
                &WKB_VIEW_GEOMETRY,
            ),
        );
    }

    #[rstest]
    fn udf_unchecked(#[values(DataType::Binary, DataType::BinaryView)] data_type: DataType) {
        let udf = st_geomfromwkbunchecked_udf();
        let tester = ScalarUdfTester::new(
            udf.clone().into(),
            vec![SedonaType::Arrow(data_type.clone())],
        );

        assert_eq!(tester.return_type().unwrap(), WKB_VIEW_GEOMETRY);

        assert_scalar_equal(
            &tester.invoke_scalar(POINT12.to_vec()).unwrap(),
            &create_scalar(Some("POINT (1 2)"), &WKB_VIEW_GEOMETRY),
        );

        assert_scalar_equal(
            &tester.invoke_scalar(ScalarValue::Null).unwrap(),
            &create_scalar(None, &WKB_VIEW_GEOMETRY),
        );

        let binary_array: BinaryArray = [Some(POINT12), None, Some(POINT12)].iter().collect();
        assert_array_equal(
            &tester.invoke_array(Arc::new(binary_array)).unwrap(),
            &create_array(
                &[Some("POINT (1 2)"), None, Some("POINT (1 2)")],
                &WKB_VIEW_GEOMETRY,
            ),
        );
    }

    #[test]
    fn invalid_wkb() {
        let udf = st_geomfromwkb_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![SedonaType::Arrow(DataType::Binary)]);

        for invalid in INVALID_WKBS {
            let _err = tester
                .invoke_scalar(ScalarValue::Binary(Some(invalid.to_vec())))
                .unwrap_err();
        }
    }

    #[rstest]
    fn unchecked_invalid_wkb(
        #[values(DataType::Binary, DataType::BinaryView)] data_type: DataType,
    ) {
        let udf = st_geomfromwkbunchecked_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![SedonaType::Arrow(data_type.clone())]);

        for invalid in INVALID_WKBS {
            let invalid_scalar = match data_type {
                DataType::Binary => ScalarValue::Binary(Some(invalid.to_vec())),
                DataType::BinaryView => ScalarValue::BinaryView(Some(invalid.to_vec())),
                _ => unreachable!(),
            };

            assert_scalar_equal(
                &tester.invoke_scalar(invalid_scalar).unwrap(),
                &ScalarValue::BinaryView(Some(invalid.to_vec())),
            );

            let input_array: ArrayRef = match data_type {
                DataType::Binary => Arc::new(
                    [Some(invalid), None, Some(invalid)]
                        .iter()
                        .collect::<BinaryArray>(),
                ),
                DataType::BinaryView => Arc::new(
                    [Some(invalid), None, Some(invalid)]
                        .iter()
                        .collect::<BinaryViewArray>(),
                ),
                _ => unreachable!(),
            };

            let expected_array: BinaryViewArray =
                [Some(invalid), None, Some(invalid)].iter().collect();

            assert_array_equal(
                &tester.invoke_array(input_array).unwrap(),
                &(Arc::new(expected_array) as ArrayRef),
            );
        }
    }

    #[test]
    fn udf_invoke_with_array_crs() {
        let udf = st_geomfromwkb_udf();
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![
                SedonaType::Arrow(DataType::Binary),
                SedonaType::Arrow(DataType::Utf8),
            ],
        );

        let return_type = tester.return_type().unwrap();
        assert_eq!(
            return_type,
            SedonaType::new_item_crs(&WKB_VIEW_GEOMETRY).unwrap()
        );

        let sedona_type = WKB_GEOMETRY;
        let wkb_array = create_array_storage(
            &[
                Some("POINT (0 1)"),
                Some("POINT (2 3)"),
                Some("POINT (4 5)"),
                Some("POINT (6 7)"),
                None,
            ],
            &sedona_type,
        );
        let crs_array = create_array!(
            Utf8,
            [
                Some("EPSG:4326"),
                Some("EPSG:3857"),
                Some("EPSG:3857"),
                Some("0"),
                None
            ]
        ) as ArrayRef;

        let result = tester.invoke_arrays(vec![wkb_array, crs_array]).unwrap();
        assert_eq!(
            &result,
            &create_array_item_crs(
                &[
                    Some("POINT (0 1)"),
                    Some("POINT (2 3)"),
                    Some("POINT (4 5)"),
                    Some("POINT (6 7)"),
                    None
                ],
                [
                    Some("OGC:CRS84"),
                    Some("EPSG:3857"),
                    Some("EPSG:3857"),
                    None,
                    None
                ],
                &WKB_VIEW_GEOMETRY
            )
        );
    }

    #[test]
    fn geog() {
        let udf = st_geogfromwkb_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![SedonaType::Arrow(DataType::Binary)]);

        assert_eq!(tester.return_type().unwrap(), WKB_VIEW_GEOGRAPHY);

        assert_scalar_equal(
            &tester.invoke_scalar(POINT12.to_vec()).unwrap(),
            &create_scalar(Some("POINT (1 2)"), &WKB_VIEW_GEOGRAPHY),
        );
    }
}
