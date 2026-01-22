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

use arrow_array::builder::BinaryBuilder;
use arrow_schema::DataType;
use datafusion_common::{
    cast::{as_string_view_array, as_struct_array},
    error::Result,
    exec_datafusion_err, exec_err, ScalarValue,
};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::{ewkb_factory::write_ewkb_geometry, wkb_factory::WKB_MIN_PROBABLE_BYTES};
use sedona_schema::{crs::deserialize_crs, datatypes::SedonaType, matchers::ArgMatcher};

use crate::executor::WkbExecutor;

/// ST_AsEWKB() scalar UDF implementation
///
/// An implementation of EWKB writing using Sedona's geometry EWKB/WKB facilities.
pub fn st_asewkb_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_asewkb",
        vec![Arc::new(STAsEWKBItemCrs {}), Arc::new(STAsEWKB {})],
        Volatility::Immutable,
        Some(st_asewkb_doc()),
    )
}

fn st_asewkb_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        r#"Return the Extended Well-Known Binary (EWKB) representation of a geometry or geography.

Compared to ST_AsBinary(), this function embeds an integer SRID derived from the type or derived
from the item-level CRS for item CRS types. This is particularly useful for integration with
PostGIS"#,
        "ST_AsEWKB (A: Geometry)",
    )
    .with_argument("geom", "geometry: Input geometry or geography")
    .with_sql_example("SELECT ST_AsEWKB(ST_Point(1.0, 2.0, 4326))")
    .build()
}

#[derive(Debug)]
struct STAsEWKB {}

impl SedonaScalarKernel for STAsEWKB {
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
        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
        );

        let maybe_srid = match &arg_types[0] {
            SedonaType::Wkb(_, crs) | SedonaType::WkbView(_, crs) => match crs {
                Some(crs) => match crs.srid()? {
                    Some(0) => None,
                    Some(srid) => Some(srid),
                    _ => return exec_err!("CRS {crs} cannot be represented by a single SRID"),
                },
                None => None,
            },
            SedonaType::Arrow(DataType::Null) => None,
            _ => return sedona_internal_err!("Unexpected input to invoke_batch in ST_AsEWKB"),
        };

        executor.execute_wkb_void(|maybe_wkb| {
            match maybe_wkb {
                Some(wkb) => {
                    write_ewkb_geometry(&mut builder, &wkb, maybe_srid)
                        .map_err(|e| exec_datafusion_err!("EWKB writer error {e}"))?;
                    builder.append_value([]);
                }
                None => builder.append_null(),
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

#[derive(Debug)]
struct STAsEWKBItemCrs {}

impl SedonaScalarKernel for STAsEWKBItemCrs {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_item_crs()],
            SedonaType::Arrow(DataType::Binary),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
        );

        let crs_array_ref = match &args[0] {
            ColumnarValue::Array(array) => {
                let struct_array = as_struct_array(array)?;
                struct_array.column(1).clone()
            }
            ColumnarValue::Scalar(ScalarValue::Struct(struct_array)) => {
                struct_array.column(1).clone()
            }
            _ => return sedona_internal_err!("Unexpected item_crs type"),
        };

        let crs_array = as_string_view_array(&crs_array_ref)?;
        let mut srid_iter = crs_array
            .into_iter()
            .map(|maybe_crs_str| match maybe_crs_str {
                None => Ok(None),
                Some(crs_str) => {
                    match deserialize_crs(crs_str)
                        .map_err(|e| exec_datafusion_err!("{}", e.message()))?
                    {
                        None => Ok(None),
                        Some(crs) => match crs.srid()? {
                            Some(0) => Ok(None),
                            Some(srid) => Ok(Some(srid)),
                            _ => {
                                exec_err!("CRS {crs} cannot be represented by a single SRID")
                            }
                        },
                    }
                }
            });

        executor.execute_wkb_void(|maybe_wkb| {
            let maybe_srid = srid_iter.next().unwrap()?;
            match maybe_wkb {
                Some(wkb) => {
                    write_ewkb_geometry(&mut builder, &wkb, maybe_srid)
                        .map_err(|e| exec_datafusion_err!("EWKB writer error {e}"))?;
                    builder.append_value([]);
                }
                None => builder.append_null(),
            }

            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{ArrayRef, BinaryArray};
    use datafusion_common::scalar::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::{
        crs::lnglat,
        datatypes::{
            Edges, WKB_GEOGRAPHY, WKB_GEOGRAPHY_ITEM_CRS, WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS,
            WKB_VIEW_GEOGRAPHY, WKB_VIEW_GEOMETRY,
        },
    };
    use sedona_testing::{
        create::{create_array_item_crs, create_scalar_item_crs},
        testers::ScalarUdfTester,
    };

    use super::*;

    const POINT12: [u8; 21] = [
        0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x40,
    ];

    const POINT12_LNGLAT: [u8; 25] = [
        0x01, 0x01, 0x00, 0x00, 0x20, 0xe6, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xf0, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40,
    ];

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_asewkb_udf().into();
        assert_eq!(udf.name(), "st_asewkb");
        assert!(udf.documentation().is_some())
    }

    #[rstest]
    fn udf_no_srid(
        #[values(
            WKB_GEOMETRY,
            WKB_GEOGRAPHY,
            WKB_VIEW_GEOMETRY,
            WKB_VIEW_GEOGRAPHY,
            WKB_GEOMETRY_ITEM_CRS.clone(),
            WKB_GEOGRAPHY_ITEM_CRS.clone(),
        )]
        sedona_type: SedonaType,
    ) {
        let udf = st_asewkb_udf();
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
    fn udf_srid_from_type(
        #[values(
            SedonaType::Wkb(Edges::Planar, lnglat()),
            SedonaType::Wkb(Edges::Spherical, lnglat())
        )]
        sedona_type: SedonaType,
    ) {
        let udf = st_asewkb_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type]);

        assert_eq!(
            tester.invoke_wkb_scalar(Some("POINT (1 2)")).unwrap(),
            ScalarValue::Binary(Some(POINT12_LNGLAT.to_vec()))
        );

        assert_eq!(
            tester.invoke_wkb_scalar(None).unwrap(),
            ScalarValue::Binary(None)
        );

        let expected_array: BinaryArray = [Some(POINT12_LNGLAT), None, Some(POINT12_LNGLAT)]
            .iter()
            .collect();
        assert_eq!(
            &tester
                .invoke_wkb_array(vec![Some("POINT (1 2)"), None, Some("POINT (1 2)")])
                .unwrap(),
            &(Arc::new(expected_array) as ArrayRef)
        );
    }

    #[test]
    fn udf_srid_from_item_crs() {
        let udf = st_asewkb_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOMETRY_ITEM_CRS.clone()]);

        let scalar_with_srid =
            create_scalar_item_crs(Some("POINT (1 2)"), Some("EPSG:4326"), &WKB_GEOMETRY);
        assert_eq!(
            tester.invoke_scalar(scalar_with_srid).unwrap(),
            ScalarValue::Binary(Some(POINT12_LNGLAT.to_vec()))
        );

        let array_with_srid = create_array_item_crs(
            &[Some("POINT (1 2)"), None, Some("POINT (1 2)")],
            [Some("EPSG:4326"), None, Some("EPSG:4326")],
            &WKB_GEOMETRY,
        );
        let expected_array: BinaryArray = [Some(POINT12_LNGLAT), None, Some(POINT12_LNGLAT)]
            .iter()
            .collect();
        assert_eq!(
            &tester.invoke_array(array_with_srid).unwrap(),
            &(Arc::new(expected_array) as ArrayRef)
        );
    }

    #[test]
    fn udf_invalid_type_crs() {
        let udf = st_asewkb_udf();

        let crs_where_srid_returns_none = deserialize_crs("EPSG:9999999999").unwrap();
        let sedona_type = SedonaType::Wkb(Edges::Planar, crs_where_srid_returns_none);

        let tester = ScalarUdfTester::new(udf.into(), vec![sedona_type]);
        let err = tester.invoke_wkb_scalar(Some("POINT (1 2)")).unwrap_err();
        assert_eq!(
            err.message(),
            "CRS epsg:9999999999 cannot be represented by a single SRID"
        );
    }

    #[test]
    fn udf_invalid_item_crs() {
        let udf = st_asewkb_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![WKB_GEOMETRY_ITEM_CRS.clone()]);

        // Very large SRID
        let scalar_with_srid_outside_u32 = create_scalar_item_crs(
            Some("POINT (1 2)"),
            Some("EPSG:999999999999"),
            &WKB_GEOMETRY,
        );
        let err = tester
            .invoke_scalar(scalar_with_srid_outside_u32)
            .unwrap_err();
        assert_eq!(
            err.message(),
            "CRS epsg:999999999999 cannot be represented by a single SRID"
        );

        // CRS that fails to parse in deserialize_crs()
        let scalar_with_unparsable_crs = create_scalar_item_crs(
            Some("POINT (1 2)"),
            Some("This is invalid JSON and also not auth:code"),
            &WKB_GEOMETRY,
        );
        let err = tester
            .invoke_scalar(scalar_with_unparsable_crs)
            .unwrap_err();
        assert_eq!(
            err.message(),
            "Error deserializing PROJJSON Crs: expected value at line 1 column 1"
        );
    }
}
