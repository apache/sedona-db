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
use arrow_array::builder::BooleanBuilder;
use arrow_schema::DataType;
use datafusion_common::{error::Result, DataFusionError};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use geo_traits::Dimensions;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_geometry::wkb_header::WkbHeader;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

pub fn st_hasz_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_hasz",
        vec![Arc::new(STHasZm { dim: "z" })],
        Volatility::Immutable,
        Some(st_geometry_type_doc("z")),
    )
}

pub fn st_hasm_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_hasm",
        vec![Arc::new(STHasZm { dim: "m" })],
        Volatility::Immutable,
        Some(st_geometry_type_doc("m")),
    )
}

fn st_geometry_type_doc(dim: &str) -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        format!(
            "Return true if the geometry has a {} dimension",
            dim.to_uppercase()
        ),
        format!("ST_Has{} (A: Geometry)", dim.to_uppercase()),
    )
    .with_argument("geom", "geometry: Input geometry")
    .with_sql_example(format!(
        "SELECT ST_Has{}(ST_GeomFromWKT('POLYGON ((0 0, 1 0, 0 1, 0 0))'))",
        dim.to_uppercase()
    ))
    .build()
}

#[derive(Debug)]
struct STHasZm {
    dim: &'static str,
}

impl SedonaScalarKernel for STHasZm {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry()],
            SedonaType::Arrow(DataType::Boolean),
        );

        matcher.match_args(args)
    }

    fn invoke_batch(
        &self,
        arg_types: &[SedonaType],
        args: &[ColumnarValue],
    ) -> Result<ColumnarValue> {
        let dim_index = match self.dim {
            "z" => 2,
            "m" => 3,
            _ => unreachable!(),
        };

        let executor = WkbBytesExecutor::new(arg_types, args);
        let mut builder = BooleanBuilder::with_capacity(executor.num_iterations());

        executor.execute_wkb_void(|maybe_item| {
            match maybe_item {
                Some(item) => {
                    builder.append_option(invoke_scalar(item, dim_index)?);
                }
                None => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(buf: &[u8], dim_index: usize) -> Result<Option<bool>> {
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

    if dim_index == 2 {
        return Ok(Some(matches!(
            dimensions,
            Dimensions::Xyz | Dimensions::Xyzm
        )));
    }
    if dim_index == 3 {
        return Ok(Some(matches!(
            dimensions,
            Dimensions::Xym | Dimensions::Xyzm
        )));
    }
    Ok(Some(false))
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
        let udf: ScalarUDF = st_hasz_udf().into();
        assert_eq!(udf.name(), "st_hasz");
        assert!(udf.documentation().is_some());

        let udf: ScalarUDF = st_hasm_udf().into();
        assert_eq!(udf.name(), "st_hasm");
        assert!(udf.documentation().is_some())
    }

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let z_tester = ScalarUdfTester::new(st_hasz_udf().into(), vec![sedona_type.clone()]);
        let m_tester = ScalarUdfTester::new(st_hasm_udf().into(), vec![sedona_type.clone()]);

        z_tester.assert_return_type(DataType::Boolean);
        m_tester.assert_return_type(DataType::Boolean);

        let result = z_tester.invoke_scalar("POINT ZM (1 2 3 4)").unwrap();
        z_tester.assert_scalar_result_equals(result, true);

        let result = m_tester.invoke_scalar("POINT ZM (1 2 3 4)").unwrap();
        m_tester.assert_scalar_result_equals(result, true);

        let result = z_tester.invoke_scalar("POINT (1 2)").unwrap();
        z_tester.assert_scalar_result_equals(result, false);

        let result = m_tester.invoke_scalar("POINT (1 2)").unwrap();
        m_tester.assert_scalar_result_equals(result, false);

        let result = z_tester.invoke_scalar("POINT M (1 2 3)").unwrap();
        z_tester.assert_scalar_result_equals(result, false);

        let result = m_tester.invoke_scalar("POINT Z (1 2 3)").unwrap();
        m_tester.assert_scalar_result_equals(result, false);

        let result = z_tester.invoke_wkb_scalar(None).unwrap();
        z_tester.assert_scalar_result_equals(result, ScalarValue::Null);

        let result = m_tester.invoke_wkb_scalar(None).unwrap();
        m_tester.assert_scalar_result_equals(result, ScalarValue::Null);

        // Z-dimension specified only in the nested geometry, but not the geom collection level
        let result = z_tester
            .invoke_wkb_scalar(Some("GEOMETRYCOLLECTION (POINT Z (1 2 3))"))
            .unwrap();
        z_tester.assert_scalar_result_equals(result, ScalarValue::Boolean(Some(true)));

        // Z-dimension specified on both the geom collection and nested geometry level
        // Geometry collection with Z dimension both on the geom collection and nested geometry level
        let result = z_tester
            .invoke_wkb_scalar(Some("GEOMETRYCOLLECTION Z (POINT Z (1 2 3))"))
            .unwrap();
        z_tester.assert_scalar_result_equals(result, ScalarValue::Boolean(Some(true)));

        let result = m_tester
            .invoke_wkb_scalar(Some("GEOMETRYCOLLECTION (POINT M (1 2 3))"))
            .unwrap();
        m_tester.assert_scalar_result_equals(result, ScalarValue::Boolean(Some(true)));

        let result = z_tester
            .invoke_wkb_scalar(Some("GEOMETRYCOLLECTION EMPTY"))
            .unwrap();
        z_tester.assert_scalar_result_equals(result, ScalarValue::Boolean(Some(false)));

        let result = m_tester
            .invoke_wkb_scalar(Some("GEOMETRYCOLLECTION EMPTY"))
            .unwrap();
        m_tester.assert_scalar_result_equals(result, ScalarValue::Boolean(Some(false)));

        // Empty geometry collections with Z or M dimensions
        let result = z_tester
            .invoke_wkb_scalar(Some("GEOMETRYCOLLECTION Z EMPTY"))
            .unwrap();
        z_tester.assert_scalar_result_equals(result, ScalarValue::Boolean(Some(true)));

        let result = m_tester
            .invoke_wkb_scalar(Some("GEOMETRYCOLLECTION M EMPTY"))
            .unwrap();
        m_tester.assert_scalar_result_equals(result, ScalarValue::Boolean(Some(true)));
    }

    #[test]
    fn multipoint_with_inferred_z_dimension() {
        let z_tester = ScalarUdfTester::new(st_hasz_udf().into(), vec![WKB_GEOMETRY]);
        let m_tester = ScalarUdfTester::new(st_hasm_udf().into(), vec![WKB_GEOMETRY]);

        let scalar = ScalarValue::Binary(Some(MULTIPOINT_WITH_INFERRED_Z_DIMENSION_WKB.to_vec()));
        assert_eq!(
            z_tester.invoke_scalar(scalar.clone()).unwrap(),
            ScalarValue::Boolean(Some(true))
        );
        assert_eq!(
            m_tester.invoke_scalar(scalar.clone()).unwrap(),
            ScalarValue::Boolean(Some(false))
        );
    }
}
