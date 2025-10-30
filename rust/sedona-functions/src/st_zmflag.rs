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

use crate::executor::WkbBytesExecutor;
use arrow_array::builder::Int8Builder;
use arrow_schema::DataType;
use datafusion_common::{error::Result, DataFusionError};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use geo_traits::Dimensions;
use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::wkb_header::WkbHeader;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

pub fn st_zmflag_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_zmflag",
        vec![Arc::new(STZmFlag {})],
        Volatility::Immutable,
        Some(st_zmflag_doc()),
    )
}

fn st_zmflag_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Returns a code indicating the ZM coordinate dimension of a geometry. Values are 0 for 2D, 1 for 3D-M, 2 for 3D-Z, and 3 for 4D.".to_string(),
        "ST_ZmFlag (A: Geometry)".to_string(),
    )
    .with_argument("geom", "geometry: Input geometry")
    .with_sql_example("SELECT ST_ZmFlag(ST_GeomFromWKT('POLYGON ((0 0, 1 0, 0 1, 0 0))'))")
    .build()
}

#[derive(Debug)]
struct STZmFlag {}

impl SedonaScalarKernel for STZmFlag {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry()],
            SedonaType::Arrow(DataType::Int8),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let executor = WkbBytesExecutor::new(arg_types, args);
        let mut builder = Int8Builder::with_capacity(executor.num_iterations());

        executor.execute_wkb_void(|maybe_item| {
            match maybe_item {
                Some(item) => {
                    builder.append_value(invoke_scalar(item)?);
                }
                None => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(buf: &[u8]) -> Result<i8> {
    let header = WkbHeader::try_new(buf).map_err(|e| DataFusionError::External(Box::new(e)))?;
    let top_level_dimensions = header
        .dimensions()
        .map_err(|e| DataFusionError::External(Box::new(e)))?;

    // Infer dimension based on first coordinate dimension for cases where it differs from top-level
    // e.g GEOMETRYCOLLECTION (POINT Z (1 2 3))
    let dimensions;
    if let Some(first_geom_dimensions) = header.first_geom_dimensions() {
        dimensions = first_geom_dimensions;
    } else {
        dimensions = top_level_dimensions;
    }

    match dimensions {
        Dimensions::Xy => Ok(0),
        Dimensions::Xym => Ok(1),
        Dimensions::Xyz => Ok(2),
        Dimensions::Xyzm => Ok(3),
        _ => sedona_internal_err!("Invalid dimensions: {:?}", dimensions),
    }
}

#[cfg(test)]
mod tests {
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::{
        fixtures::MULTIPOINT_WITH_INFERRED_Z_DIMENSION_WKB, testers::ScalarUdfTester,
    };

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_zmflag_udf().into();
        assert_eq!(udf.name(), "st_zmflag");
        assert!(udf.documentation().is_some());
    }

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = ScalarUdfTester::new(st_zmflag_udf().into(), vec![sedona_type.clone()]);

        tester.assert_return_type(DataType::Int8);

        let result = tester.invoke_scalar("POINT ZM (1 2 3 4)").unwrap();
        tester.assert_scalar_result_equals(result, 3);

        let result = tester.invoke_scalar("POINT (1 2)").unwrap();
        tester.assert_scalar_result_equals(result, 0);

        let result = tester.invoke_scalar("POINT Z (1 2 3)").unwrap();
        tester.assert_scalar_result_equals(result, 2);

        let result = tester.invoke_wkb_scalar(None).unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Null);

        // Z-dimension specified only in the nested geometry, but not the geom collection level
        let result = tester
            .invoke_wkb_scalar(Some("GEOMETRYCOLLECTION (POINT Z (1 2 3))"))
            .unwrap();
        tester.assert_scalar_result_equals(result, 2);

        // Z-dimension specified on both the geom collection and nested geometry level
        // Geometry collection with Z dimension both on the geom collection and nested geometry level
        let result = tester
            .invoke_wkb_scalar(Some("GEOMETRYCOLLECTION Z (POINT Z (1 2 3))"))
            .unwrap();
        tester.assert_scalar_result_equals(result, 2);

        let result = tester
            .invoke_wkb_scalar(Some("GEOMETRYCOLLECTION (POINT M (1 2 3))"))
            .unwrap();
        tester.assert_scalar_result_equals(result, 1);

        let result = tester
            .invoke_wkb_scalar(Some("GEOMETRYCOLLECTION EMPTY"))
            .unwrap();
        tester.assert_scalar_result_equals(result, 0);

        // Empty geometry collections with Z or M dimensions
        let result = tester
            .invoke_wkb_scalar(Some("GEOMETRYCOLLECTION Z EMPTY"))
            .unwrap();
        tester.assert_scalar_result_equals(result, 2);

        let result = tester
            .invoke_wkb_scalar(Some("GEOMETRYCOLLECTION M EMPTY"))
            .unwrap();
        tester.assert_scalar_result_equals(result, 1);
    }

    #[test]
    fn multipoint_with_inferred_z_dimension() {
        let tester = ScalarUdfTester::new(st_zmflag_udf().into(), vec![WKB_GEOMETRY]);

        let scalar = ScalarValue::Binary(Some(MULTIPOINT_WITH_INFERRED_Z_DIMENSION_WKB.to_vec()));
        assert_eq!(
            tester.invoke_scalar(scalar.clone()).unwrap(),
            ScalarValue::Int8(Some(2))
        );
    }
}
