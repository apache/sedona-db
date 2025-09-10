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
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY, WKB_VIEW_GEOGRAPHY, WKB_VIEW_GEOMETRY},
    matchers::ArgMatcher,
};

use crate::executor::WkbExecutor;

/// ST_GeomFromWKB() scalar UDF implementation
///
/// An implementation of WKB reading using GeoRust's wkb crate.
pub fn st_geomfromwkb_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_geomfromwkb",
        vec![Arc::new(STGeomFromWKB {
            validate: true,
            out_type: WKB_VIEW_GEOMETRY,
        })],
        Volatility::Immutable,
        Some(doc("ST_GeomFromWKB", "Geometry")),
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
        Some(doc("ST_GeogFromWKB", "Geography")),
    )
}

fn doc(name: &str, out_type_name: &str) -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        format!("Construct a {out_type_name} from WKB"),
        format!("{name} (Wkb: Binary)"),
    )
    .with_argument(
        "WKB",
        format!(
            "binary: Well-known binary representation of the {}",
            out_type_name.to_lowercase()
        ),
    )
    .with_sql_example(format!("SELECT {name}([01 02 00 00 00 02 00 00 00 00 00 00 00 84 D6 00 C0 00 00 00 00 80 B5 D6 BF 00 00 00 60 E1 EF F7 BF 00 00 00 80 07 5D E5 BF])"))
    .with_related_udf("ST_AsText")
    .build()
}

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
    use arrow_array::BinaryArray;
    use datafusion_common::scalar::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_testing::{
        compare::{assert_array_equal, assert_scalar_equal},
        create::create_array,
        create::create_scalar,
        testers::ScalarUdfTester,
    };

    use super::*;

    const POINT12: [u8; 21] = [
        0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x40,
    ];

    #[test]
    fn udf_metadata() {
        let geog_from_wkb: ScalarUDF = st_geogfromwkb_udf().into();
        assert_eq!(geog_from_wkb.name(), "st_geogfromwkb");
        assert!(geog_from_wkb.documentation().is_some());

        let geom_from_wkb: ScalarUDF = st_geomfromwkb_udf().into();
        assert_eq!(geom_from_wkb.name(), "st_geomfromwkb");
        assert!(geom_from_wkb.documentation().is_some());
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

    #[test]
    fn invalid_wkb() {
        let udf = st_geomfromwkb_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![SedonaType::Arrow(DataType::Binary)]);

        let err = tester
            .invoke_scalar(ScalarValue::Binary(Some(vec![])))
            .unwrap_err();

        assert_eq!(err.message(), "failed to fill whole buffer");
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
