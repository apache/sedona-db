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
use std::{str::FromStr, sync::Arc, vec};

use arrow_array::builder::{BinaryBuilder, StringViewBuilder};
use arrow_schema::DataType;
use datafusion_common::cast::as_string_view_array;
use datafusion_common::error::Result;
use datafusion_common::exec_datafusion_err;
use datafusion_common::scalar::ScalarValue;
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_common::sedona_internal_datafusion_err;
use sedona_expr::item_crs::make_item_crs;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::wkb_factory::WKB_MIN_PROBABLE_BYTES;
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOGRAPHY, WKB_GEOMETRY},
    matchers::ArgMatcher,
};
use wkb::writer::{write_geometry, WriteOptions};
use wkb::Endianness;
use wkt::Wkt;

use crate::executor::WkbExecutor;
use crate::st_setsrid::SRIDifiedKernel;

/// ST_GeomFromWKT() UDF implementation
///
/// An implementation of WKT reading using GeoRust's wkt crate.
/// See [`st_geogfromwkt_udf`] for the corresponding geography function.
pub fn st_geomfromwkt_udf() -> SedonaScalarUDF {
    let kernel = Arc::new(STGeoFromWKT {
        out_type: WKB_GEOMETRY,
    });
    let sridified_kernel = Arc::new(SRIDifiedKernel::new(kernel.clone()));

    let udf = SedonaScalarUDF::new(
        "st_geomfromwkt",
        vec![sridified_kernel, kernel],
        Volatility::Immutable,
        None,
    );
    udf.with_aliases(vec![
        "st_geomfromtext".to_string(),
        "st_geometryfromtext".to_string(),
    ])
}

/// ST_GeogFromWKT() UDF implementation
///
/// An implementation of WKT reading using GeoRust's wkt crate.
/// See [`st_geomfromwkt_udf`] for the corresponding geometry function.
pub fn st_geogfromwkt_udf() -> SedonaScalarUDF {
    let udf = SedonaScalarUDF::new(
        "st_geogfromwkt",
        vec![Arc::new(STGeoFromWKT {
            out_type: WKB_GEOGRAPHY,
        })],
        Volatility::Immutable,
        None,
    );
    udf.with_aliases(vec!["st_geogfromtext".to_string()])
}

#[derive(Debug)]
struct STGeoFromWKT {
    out_type: SedonaType,
}

impl SedonaScalarKernel for STGeoFromWKT {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(vec![ArgMatcher::is_string()], self.out_type.clone());
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let arg_array = args[0]
            .cast_to(&DataType::Utf8View, None)?
            .to_array(executor.num_iterations())?;

        let mut builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
        );

        for item in as_string_view_array(&arg_array)? {
            if let Some(wkt_bytes) = item {
                invoke_scalar(wkt_bytes, &mut builder)?;
                builder.append_value(vec![]);
            } else {
                builder.append_null();
            }
        }

        let new_array = builder.finish();
        executor.finish(Arc::new(new_array))
    }
}

fn invoke_scalar(wkt_bytes: &str, builder: &mut BinaryBuilder) -> Result<()> {
    let geometry: Wkt<f64> =
        Wkt::from_str(wkt_bytes).map_err(|err| exec_datafusion_err!("WKT parse error: {err}"))?;

    write_geometry(
        builder,
        &geometry,
        &WriteOptions {
            endianness: Endianness::LittleEndian,
        },
    )
    .map_err(|err| sedona_internal_datafusion_err!("WKB write error: {err}"))
}

/// ST_GeomFromEWKT() UDF implementation
///
/// An implementation of EWKT reading using GeoRust's wkt crate.
pub fn st_geomfromewkt_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_geomfromewkt",
        vec![Arc::new(STGeoFromEWKT {})],
        Volatility::Immutable,
        None,
    )
}

#[derive(Debug)]
struct STGeoFromEWKT {}

impl SedonaScalarKernel for STGeoFromEWKT {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_string()],
            SedonaType::new_item_crs(&WKB_GEOMETRY)?,
        );
        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbExecutor::new(arg_types, args);
        let arg_array = args[0]
            .cast_to(&DataType::Utf8View, None)?
            .to_array(executor.num_iterations())?;

        let mut geom_builder = BinaryBuilder::with_capacity(
            executor.num_iterations(),
            WKB_MIN_PROBABLE_BYTES * executor.num_iterations(),
        );
        let mut srid_builder = StringViewBuilder::with_capacity(executor.num_iterations());

        for item in as_string_view_array(&arg_array)? {
            if let Some(ewkt_bytes) = item {
                match ewkt_bytes.split_once(";") {
                    Some((maybe_srid, wkt_bytes)) => {
                        let srid = parse_maybe_srid(maybe_srid);
                        invoke_scalar_with_srid(
                            wkt_bytes,
                            srid,
                            &mut geom_builder,
                            &mut srid_builder,
                        )?;
                    }
                    None => {
                        invoke_scalar_with_srid(
                            ewkt_bytes,
                            None,
                            &mut geom_builder,
                            &mut srid_builder,
                        )?;
                    }
                }
                geom_builder.append_value(vec![]);
            } else {
                geom_builder.append_null();
                srid_builder.append_null();
            }
        }

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

fn parse_maybe_srid(maybe_srid: &str) -> Option<String> {
    if !maybe_srid.starts_with("SRID=") {
        return None;
    }
    let srid_str = &maybe_srid[5..];
    let auth_code = match srid_str.parse::<u32>() {
        Ok(0) => return None,
        Ok(4326) => "OGC:CRS84".to_string(),
        Ok(srid) => format!("EPSG:{srid}"),
        Err(_) => return None,
    };

    // CRS could be validated here
    // https://github.com/apache/sedona-db/issues/501

    Some(auth_code)
}

fn invoke_scalar_with_srid(
    wkt_bytes: &str,
    srid: Option<String>,
    geom_builder: &mut BinaryBuilder,
    srid_builder: &mut StringViewBuilder,
) -> Result<()> {
    invoke_scalar(wkt_bytes, geom_builder)?;
    srid_builder.append_option(srid);
    Ok(())
}

#[cfg(test)]
mod tests {
    use arrow_array::{create_array, ArrayRef};
    use arrow_schema::DataType;
    use datafusion_common::scalar::ScalarValue;
    use datafusion_expr::{Literal, ScalarUDF};
    use rstest::rstest;
    use sedona_schema::crs::lnglat;
    use sedona_schema::datatypes::Edges;
    use sedona_testing::{
        compare::{assert_array_equal, assert_scalar_equal, assert_scalar_equal_wkb_geometry},
        create::{create_array, create_array_item_crs, create_scalar, create_scalar_item_crs},
        testers::ScalarUdfTester,
    };

    use super::*;

    #[test]
    fn udf_metadata() {
        let geog_from_wkt: ScalarUDF = st_geogfromwkt_udf().into();
        assert_eq!(geog_from_wkt.name(), "st_geogfromwkt");
        assert!(geog_from_wkt.documentation().is_none());

        let geom_from_wkt: ScalarUDF = st_geomfromwkt_udf().into();
        assert_eq!(geom_from_wkt.name(), "st_geomfromwkt");
        assert!(geom_from_wkt.documentation().is_none());
    }

    #[rstest]
    fn udf(#[values(DataType::Utf8, DataType::Utf8View)] data_type: DataType) {
        let udf = st_geomfromwkt_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![SedonaType::Arrow(data_type.clone())]);
        assert_eq!(tester.return_type().unwrap(), WKB_GEOMETRY);

        // Scalar non-null
        assert_scalar_equal_wkb_geometry(
            &tester.invoke_scalar("POINT (1 2)").unwrap(),
            Some("POINT (1 2)"),
        );

        // Scalar null
        assert_scalar_equal_wkb_geometry(&tester.invoke_scalar(ScalarValue::Null).unwrap(), None);

        // Array
        let array_in =
            arrow_array::create_array!(Utf8, [Some("POINT (1 2)"), None, Some("POINT (3 4)")]);
        assert_array_equal(
            &tester.invoke_array(array_in).unwrap(),
            &create_array(
                &[Some("POINT (1 2)"), None, Some("POINT (3 4)")],
                &WKB_GEOMETRY,
            ),
        );
    }

    #[rstest(
        data_type => [DataType::Utf8, DataType::Utf8View],
        srid => [
            (DataType::UInt32, 4326),
            (DataType::Int32, 4326),
            (DataType::Utf8, "4326"),
            (DataType::Utf8, "EPSG:4326"),
        ]
    )]
    fn udf_with_srid(data_type: DataType, srid: (DataType, impl Literal + Copy)) {
        let (srid_type, srid_value) = srid;

        let udf = st_geomfromwkt_udf();
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![SedonaType::Arrow(data_type), SedonaType::Arrow(srid_type)],
        );

        let return_type = tester
            .return_type_with_scalar_scalar(Some("POINT (1 2)"), Some(srid_value))
            .unwrap();
        assert_eq!(return_type, SedonaType::Wkb(Edges::Planar, lnglat()));

        // Scalar non-null
        assert_scalar_equal_wkb_geometry(
            &tester
                .invoke_scalar_scalar("POINT (1 2)", srid_value)
                .unwrap(),
            Some("POINT (1 2)"),
        );
    }

    #[rstest]
    fn ewkt(#[values(DataType::Utf8, DataType::Utf8View)] data_type: DataType) {
        let udf = st_geomfromewkt_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![SedonaType::Arrow(data_type.clone())]);
        tester.assert_return_type(SedonaType::new_item_crs(&WKB_GEOMETRY).unwrap());

        assert_scalar_equal(
            &tester.invoke_scalar("SRID=4326;POINT (1 2)").unwrap(),
            &create_scalar_item_crs(Some("POINT (1 2)"), Some("OGC:CRS84"), &WKB_GEOMETRY),
        );

        assert_scalar_equal(
            &tester.invoke_scalar("SRID=3857;POINT (1 2)").unwrap(),
            &create_scalar_item_crs(Some("POINT (1 2)"), Some("EPSG:3857"), &WKB_GEOMETRY),
        );

        assert_scalar_equal(
            &tester.invoke_scalar("POINT (3 4)").unwrap(),
            &create_scalar_item_crs(Some("POINT (3 4)"), None, &WKB_GEOMETRY),
        );

        let array_in: ArrayRef = arrow_array::create_array!(
            Utf8,
            [
                Some("SRID=4326;POINT (1 2)"),
                Some("SRID=3857;POINT (1 2)"),
                None,
                Some("POINT (3 4)"),
                Some("SRID=0;POINT (5 6)")
            ]
        );
        let expected = create_array_item_crs(
            &[
                Some("POINT (1 2)"),
                Some("POINT (1 2)"),
                None,
                Some("POINT (3 4)"),
                Some("POINT (5 6)"),
            ],
            [Some("OGC:CRS84"), Some("EPSG:3857"), None, None, None],
            &WKB_GEOMETRY,
        );
        assert_array_equal(&tester.invoke_array(array_in).unwrap(), &expected);
    }

    #[test]
    fn udf_invoke_with_array_crs() {
        let udf = st_geomfromwkt_udf();
        let tester = ScalarUdfTester::new(
            udf.into(),
            vec![
                SedonaType::Arrow(DataType::Utf8),
                SedonaType::Arrow(DataType::Utf8),
            ],
        );

        let return_type = tester.return_type().unwrap();
        assert_eq!(
            return_type,
            SedonaType::new_item_crs(&WKB_GEOMETRY).unwrap()
        );

        let wkt_array: ArrayRef = create_array!(
            Utf8,
            [
                Some("POINT (0 1)"),
                Some("POINT (2 3)"),
                Some("POINT (4 5)"),
                Some("POINT (6 7)"),
                None
            ]
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

        let result = tester.invoke_arrays(vec![wkt_array, crs_array]).unwrap();
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
                &WKB_GEOMETRY
            )
        );
    }

    #[test]
    fn invalid_wkt() {
        let udf = st_geomfromwkt_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![SedonaType::Arrow(DataType::Utf8)]);
        let err = tester.invoke_scalar("this is not valid wkt").unwrap_err();

        assert!(err.message().starts_with("WKT parse error"));
    }

    #[test]
    fn geog() {
        let udf = st_geogfromwkt_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![SedonaType::Arrow(DataType::Utf8)]);
        assert_eq!(tester.return_type().unwrap(), WKB_GEOGRAPHY);
        assert_scalar_equal(
            &tester.invoke_scalar("POINT (1 2)").unwrap(),
            &create_scalar(Some("POINT (1 2)"), &WKB_GEOGRAPHY),
        );
    }

    #[test]
    fn aliases() {
        let udf: ScalarUDF = st_geomfromwkt_udf().into();
        assert!(udf.aliases().contains(&"st_geomfromtext".to_string()));
        assert!(udf.aliases().contains(&"st_geometryfromtext".to_string()));

        let udf: ScalarUDF = st_geogfromwkt_udf().into();
        assert!(udf.aliases().contains(&"st_geogfromtext".to_string()));
    }
}
