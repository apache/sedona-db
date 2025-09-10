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

use arrow_array::builder::BinaryBuilder;
use arrow_schema::DataType;
use datafusion_common::cast::as_string_view_array;
use datafusion_common::error::{DataFusionError, Result};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::wkb_factory::WKB_MIN_PROBABLE_BYTES;
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOGRAPHY, WKB_GEOMETRY},
    matchers::ArgMatcher,
};
use wkb::writer::write_geometry;
use wkt::Wkt;

use crate::executor::WkbExecutor;

/// ST_GeomFromWKT() UDF implementation
///
/// An implementation of WKT reading using GeoRust's wkt crate.
/// See [`st_geogfromwkt_udf`] for the corresponding geography function.
pub fn st_geomfromwkt_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new_with_aliases(
        "st_geomfromwkt",
        vec![Arc::new(STGeoFromWKT {
            out_type: WKB_GEOMETRY,
        })],
        Volatility::Immutable,
        Some(doc("ST_GeomFromWKT", "Geometry")),
        vec!["st_geomfromtext".to_string()],
    )
}

/// ST_GeogFromWKT() UDF implementation
///
/// An implementation of WKT reading using GeoRust's wkt crate.
/// See [`st_geomfromwkt_udf`] for the corresponding geometry function.
pub fn st_geogfromwkt_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new_with_aliases(
        "st_geogfromwkt",
        vec![Arc::new(STGeoFromWKT {
            out_type: WKB_GEOGRAPHY,
        })],
        Volatility::Immutable,
        Some(doc("ST_GeogFromWKT", "Geography")),
        vec!["st_geogfromtext".to_string()],
    )
}

fn doc(name: &str, out_type_name: &str) -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        format!("Construct a {out_type_name} from WKT"),
        format!("{name} (Wkt: String)"),
    )
    .with_argument(
        "WKT",
        format!(
            "string: Well-known text representation of the {}",
            out_type_name.to_lowercase()
        ),
    )
    .with_sql_example(format!("SELECT {name}('POINT(40.7128 -74.0060)')"))
    .with_related_udf("ST_AsText")
    .build()
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
    let geometry: Wkt<f64> = Wkt::from_str(wkt_bytes)
        .map_err(|err| DataFusionError::Internal(format!("WKT parse error: {err}")))?;

    write_geometry(builder, &geometry, wkb::Endianness::LittleEndian)
        .map_err(|err| DataFusionError::Internal(format!("WKB write error: {err}")))
}

#[cfg(test)]
mod tests {
    use arrow_schema::DataType;
    use datafusion_common::scalar::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_testing::{
        compare::{assert_array_equal, assert_scalar_equal, assert_scalar_equal_wkb_geometry},
        create::{create_array, create_scalar},
        testers::ScalarUdfTester,
    };

    use super::*;

    #[test]
    fn udf_metadata() {
        let geog_from_wkt: ScalarUDF = st_geogfromwkt_udf().into();
        assert_eq!(geog_from_wkt.name(), "st_geogfromwkt");
        assert!(geog_from_wkt.documentation().is_some());

        let geom_from_wkt: ScalarUDF = st_geomfromwkt_udf().into();
        assert_eq!(geom_from_wkt.name(), "st_geomfromwkt");
        assert!(geom_from_wkt.documentation().is_some());
    }

    #[rstest]
    fn udf(#[values(DataType::Utf8, DataType::Utf8View)] data_type: DataType) {
        let udf = st_geomfromwkt_udf();
        let tester = ScalarUdfTester::new(udf.into(), vec![SedonaType::Arrow(data_type)]);
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

        let udf: ScalarUDF = st_geogfromwkt_udf().into();
        assert!(udf.aliases().contains(&"st_geogfromtext".to_string()));
    }
}
