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

use arrow_array::builder::{BinaryBuilder, StringViewBuilder};
use arrow_schema::DataType;
use datafusion_common::{error::Result, exec_datafusion_err, ScalarValue};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use sedona_common::sedona_internal_err;
use sedona_expr::{
    item_crs::make_item_crs,
    scalar_udf::{SedonaScalarKernel, SedonaScalarUDF},
};
use sedona_geometry::{wkb_factory::WKB_MIN_PROBABLE_BYTES, wkb_header::WkbHeader};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS, WKB_VIEW_GEOGRAPHY},
    matchers::ArgMatcher,
};

use crate::executor::WkbExecutor;

/// ST_GeomFromEWKB() scalar UDF implementation
///
/// An implementation of EWKB reading using GeoRust's wkb crate and our internal
/// WkbHeader utility.
pub fn st_geomfromewkb_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_geomfromewkb",
        vec![Arc::new(STGeomFromEWKB {})],
        Volatility::Immutable,
        Some(doc()),
    )
}

fn doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Construct a geometry from EWKB".to_string(),
        "ST_GeomFromEWKB (Wkb: Binary)".to_string(),
    )
    .with_argument(
        "EWKB",
        "binary: Extended well-known binary (EWKB) representation of the geometry".to_string(),
    )
    .with_sql_example("SELECT ST_GeomFromEWKB([01 02 00 00 00 02 00 00 00 00 00 00 00 84 D6 00 C0 00 00 00 00 80 B5 D6 BF 00 00 00 60 E1 EF F7 BF 00 00 00 80 07 5D E5 BF])")
    .build()
}

#[derive(Debug)]
struct STGeomFromEWKB {}

impl SedonaScalarKernel for STGeomFromEWKB {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(vec![ArgMatcher::is_binary()], WKB_GEOMETRY_ITEM_CRS.clone());
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let iter_type = match &arg_types[0] {
            SedonaType::Arrow(data_type) => match data_type {
                DataType::Binary => WKB_GEOMETRY,
                DataType::BinaryView => WKB_VIEW_GEOGRAPHY,
                DataType::Null => SedonaType::Arrow(DataType::Null),
                _ => {
                    return sedona_internal_err!(
                        "Unexpected arguments to invoke_batch: {arg_types:?}"
                    )
                }
            },
            _ => {
                return sedona_internal_err!("Unexpected arguments to invoke_batch: {arg_types:?}")
            }
        };

        let temp_args = [iter_type];
        let executor = WkbExecutor::new(&temp_args, args);
        let mut geom_builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
        );
        let mut srid_builder = StringViewBuilder::with_capacity(executor.num_iterations());

        executor.execute_wkb_void(|maybe_item| {
            match maybe_item {
                Some(item) => {
                    let header =
                        WkbHeader::try_new(item.buf()).map_err(|e| exec_datafusion_err!("{e}"))?;
                    let maybe_crs = match header.srid() {
                        0 => None,
                        valid_srid => Some(format!("EPSG:{valid_srid}")),
                    };

                    wkb::writer::write_geometry(&mut geom_builder, &item, &Default::default())
                        .map_err(|e| exec_datafusion_err!("{e}"))?;
                    geom_builder.append_value([]);
                    srid_builder.append_option(maybe_crs);
                }
                None => {
                    geom_builder.append_null();
                    srid_builder.append_null();
                }
            }

            Ok(())
        })?;

        let new_geom_array = geom_builder.finish();
        let item_result = executor.finish(Arc::new(new_geom_array))?;

        let new_srid_array = srid_builder.finish();
        let crs_value = if matches!(&item_result, ColumnarValue::Scalar(_)) {
            ColumnarValue::Scalar(ScalarValue::try_from_array(&new_srid_array, 0)?)
        } else {
            ColumnarValue::Array(Arc::new(new_srid_array))
        };

        make_item_crs(&WKB_GEOMETRY, item_result, &crs_value, None)
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
        create::{create_array_item_crs, create_scalar, create_scalar_item_crs},
        fixtures::POINT_WITH_SRID_4326_EWKB,
        testers::ScalarUdfTester,
    };

    use super::*;

    const POINT12: [u8; 21] = [
        0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x40,
    ];

    #[test]
    fn udf_metadata() {
        let geog_from_wkb: ScalarUDF = st_geomfromewkb_udf().into();
        assert_eq!(geog_from_wkb.name(), "st_geomfromewkb");
        assert!(geog_from_wkb.documentation().is_some());
    }

    #[rstest]
    fn udf(#[values(DataType::Binary, DataType::BinaryView)] data_type: DataType) {
        let udf = st_geomfromewkb_udf();
        let tester = ScalarUdfTester::new(
            udf.clone().into(),
            vec![SedonaType::Arrow(data_type.clone())],
        );

        assert_eq!(tester.return_type().unwrap(), WKB_GEOMETRY_ITEM_CRS.clone());

        assert_scalar_equal(
            &tester
                .invoke_scalar(POINT_WITH_SRID_4326_EWKB.to_vec())
                .unwrap(),
            &create_scalar_item_crs(
                Some("POINT (1 2)"),
                Some("EPSG:4326"),
                &WKB_GEOMETRY_ITEM_CRS,
            ),
        );

        assert_scalar_equal(
            &tester.invoke_scalar(ScalarValue::Null).unwrap(),
            &create_scalar(None, &WKB_GEOMETRY_ITEM_CRS),
        );

        let binary_array: BinaryArray = [
            Some(POINT12.to_vec()),
            None,
            Some(POINT_WITH_SRID_4326_EWKB.to_vec()),
        ]
        .iter()
        .collect();
        assert_array_equal(
            &tester.invoke_array(Arc::new(binary_array)).unwrap(),
            &create_array_item_crs(
                &[Some("POINT (1 2)"), None, Some("POINT (1 2)")],
                [None, None, Some("EPSG:4326")],
                &WKB_GEOMETRY,
            ),
        );
    }
}
