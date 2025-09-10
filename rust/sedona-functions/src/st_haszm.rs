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

use crate::executor::WkbExecutor;
use arrow_array::builder::BooleanBuilder;
use arrow_schema::DataType;
use datafusion_common::error::Result;
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, ColumnarValue, Documentation, Volatility,
};
use geo_traits::{Dimensions, GeometryTrait};
use sedona_common::sedona_internal_err;
use sedona_expr::scalar_udf::{SedonaScalarKernel, SedonaScalarUDF};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};
use wkb::reader::Wkb;

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

        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = BooleanBuilder::with_capacity(executor.num_iterations());

        executor.execute_wkb_void(|maybe_item| {
            match maybe_item {
                Some(item) => {
                    builder.append_option(invoke_scalar(&item, dim_index)?);
                }
                None => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

fn invoke_scalar(item: &Wkb, dim_index: usize) -> Result<Option<bool>> {
    match item.as_type() {
        geo_traits::GeometryType::GeometryCollection(collection) => {
            use geo_traits::GeometryCollectionTrait;
            if collection.num_geometries() == 0 {
                Ok(Some(false))
            } else {
                // PostGIS doesn't allow creating a GeometryCollection with geometries of different dimensions
                // so we can just check the dimension of the first one
                let first_geom = unsafe { collection.geometry_unchecked(0) };
                invoke_scalar(first_geom, dim_index)
            }
        }
        _ => {
            let geom_dim = item.dim();
            match dim_index {
                2 => Ok(Some(matches!(geom_dim, Dimensions::Xyz | Dimensions::Xyzm))),
                3 => Ok(Some(matches!(geom_dim, Dimensions::Xym | Dimensions::Xyzm))),
                _ => sedona_internal_err!("unexpected dim_index"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::testers::ScalarUdfTester;

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

        let result = z_tester
            .invoke_wkb_scalar(Some("GEOMETRYCOLLECTION (POINT Z (1 2 3))"))
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
    }
}
