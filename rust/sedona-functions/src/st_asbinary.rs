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
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{SedonaScalarKernel, SedonaScalarUDF},
};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

/// ST_AsBinary() scalar UDF implementation
///
/// An implementation of WKB writing using GeoRust's wkt crate.
pub fn st_asbinary_udf() -> SedonaScalarUDF {
    let udf = SedonaScalarUDF::new(
        "st_asbinary",
        ItemCrsKernel::wrap_impl(vec![Arc::new(STAsBinary {})]),
        Volatility::Immutable,
    );
    udf.with_aliases(vec!["st_aswkb".to_string()])
}

#[derive(Debug)]
struct STAsBinary {}

impl SedonaScalarKernel for STAsBinary {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry_or_geography()],
            SedonaType::Arrow(DataType::Binary),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        if matches!(arg_types[0], SedonaType::Wkb(_, _)) {
            Ok(args[0].clone())
        } else {
            args[0].clone().cast_to(&DataType::Binary, None)
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{ArrayRef, BinaryArray};
    use datafusion_common::scalar::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{
        WKB_GEOGRAPHY, WKB_GEOGRAPHY_ITEM_CRS, WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS,
        WKB_VIEW_GEOGRAPHY, WKB_VIEW_GEOMETRY,
    };
    use sedona_testing::testers::ScalarUdfTester;

    use super::*;

    const POINT12: [u8; 21] = [
        0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x40,
    ];

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_asbinary_udf().into();
        assert_eq!(udf.name(), "st_asbinary");
    }

    #[rstest]
    fn udf_geometry_input(
        #[values(
            WKB_GEOMETRY,
            WKB_GEOGRAPHY,
            WKB_GEOMETRY_ITEM_CRS.clone(),
            WKB_GEOGRAPHY_ITEM_CRS.clone(),
        )]
        sedona_type: SedonaType,
    ) {
        let udf = st_asbinary_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type]);

        assert_eq!(
            tester.invoke_wkb_scalar(Some("POINT (1 2)")).unwrap(),
            ScalarValue::Binary(Some(POINT12.to_vec()))
        );

        assert_eq!(
            tester.invoke_wkb_scalar(None).unwrap(),
            ScalarValue::Binary(None)
        );

        let expected_array: BinaryArray = [Some(POINT12), None, Some(POINT12)].iter().collect();
        assert_eq!(
            &tester
                .invoke_wkb_array(vec![Some("POINT (1 2)"), None, Some("POINT (1 2)")])
                .unwrap(),
            &(Arc::new(expected_array) as ArrayRef)
        );
    }

    #[rstest]
    fn udf_geometry_view_input(
        #[values(WKB_VIEW_GEOMETRY, WKB_VIEW_GEOGRAPHY)] sedona_type: SedonaType,
    ) {
        let udf = st_asbinary_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type]);

        assert_eq!(
            tester.invoke_wkb_scalar(Some("POINT (1 2)")).unwrap(),
            ScalarValue::Binary(Some(POINT12.to_vec()))
        );

        assert_eq!(
            tester.invoke_wkb_scalar(None).unwrap(),
            ScalarValue::Binary(None)
        );

        let expected_array: BinaryArray = [Some(POINT12), None, Some(POINT12)].iter().collect();
        assert_eq!(
            &tester
                .invoke_wkb_array(vec![Some("POINT (1 2)"), None, Some("POINT (1 2)")])
                .unwrap(),
            &(Arc::new(expected_array) as ArrayRef)
        );
    }

    #[test]
    fn aliases() {
        let udf: ScalarUDF = st_asbinary_udf().into();
        assert!(udf.aliases().contains(&"st_aswkb".to_string()));
    }
}
