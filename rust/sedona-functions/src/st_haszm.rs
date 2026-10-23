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
use datafusion_common::{DataFusionError, error::Result};
use datafusion_expr::{ColumnarValue, Volatility};
use geo_traits::{Dimensions, GeometryCollectionTrait, GeometryTrait, GeometryType};
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{SedonaScalarKernel, SedonaScalarUDF},
};
use sedona_geometry::{types::GeometryTypeId, wkb_header::WkbHeader};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};
use wkb::reader::read_wkb;

pub fn st_hasz_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_hasz",
        ItemCrsKernel::wrap_impl(vec![Arc::new(STHasZm { dim: "z" })]),
        Volatility::Immutable,
    )
}

pub fn st_hasm_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_hasm",
        ItemCrsKernel::wrap_impl(vec![Arc::new(STHasZm { dim: "m" })]),
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct STHasZm {
    dim: &'static str,
}

impl SedonaScalarKernel for STHasZm {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(
            vec![ArgMatcher::is_geometry_or_geography()],
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
    Ok(Some(has_dimension(buf, dim_index)?))
}

/// Whether the geometry in `buf`, or any member of it, has Z (`dim_index` 2)
/// or M (`dim_index` 3) values.
///
/// As in GEOS, a geometry collection has Z (or M) when any of its members
/// does, empty or not, and its members need not share its dimensions, e.g.
/// GEOMETRYCOLLECTION (POINT (1 2), POINT M (3 4 5)).
fn has_dimension(buf: &[u8], dim_index: usize) -> Result<bool> {
    let header = WkbHeader::try_new(buf).map_err(|e| DataFusionError::External(Box::new(e)))?;
    let top_level_dimensions = header
        .dimensions()
        .map_err(|e| DataFusionError::External(Box::new(e)))?;

    // Infer dimension based on first coordinate dimension for cases where it differs from top-level
    // e.g GEOMETRYCOLLECTION (POINT Z (1 2 3))
    let dimensions = header
        .first_geom_dimensions()
        .unwrap_or(top_level_dimensions);

    let found = match dim_index {
        2 => matches!(dimensions, Dimensions::Xyz | Dimensions::Xyzm),
        3 => matches!(dimensions, Dimensions::Xym | Dimensions::Xyzm),
        _ => false,
    };
    if found {
        return Ok(true);
    }

    // A later member of a collection may have the dimension the first lacks
    let geometry_type = header
        .geometry_type_id()
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
    if geometry_type != GeometryTypeId::GeometryCollection {
        return Ok(false);
    }
    let wkb = read_wkb(buf).map_err(|e| DataFusionError::External(Box::new(e)))?;
    if let GeometryType::GeometryCollection(collection) = wkb.as_type() {
        for member in collection.geometries() {
            if has_dimension(member.buf(), dim_index)? {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{
        WKB_GEOGRAPHY, WKB_GEOGRAPHY_ITEM_CRS, WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS,
        WKB_VIEW_GEOGRAPHY, WKB_VIEW_GEOMETRY,
    };
    use sedona_testing::{
        fixtures::MULTIPOINT_WITH_INFERRED_Z_DIMENSION_WKB, testers::ScalarUdfTester,
    };

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_hasz_udf().into();
        assert_eq!(udf.name(), "st_hasz");

        let udf: ScalarUDF = st_hasm_udf().into();
        assert_eq!(udf.name(), "st_hasm");
    }

    #[rstest]
    fn udf(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY, WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY, WKB_GEOMETRY_ITEM_CRS.clone(), WKB_GEOGRAPHY_ITEM_CRS.clone())]
        sedona_type: SedonaType,
    ) {
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

        // Z or M in a later member only, empty or nested, as in GEOS
        for (wkt, has_z, has_m) in [
            (
                "GEOMETRYCOLLECTION (POINT (1 2), POINT Z (3 4 5))",
                true,
                false,
            ),
            (
                "GEOMETRYCOLLECTION (POINT (1 2), POINT M (3 4 5))",
                false,
                true,
            ),
            (
                "GEOMETRYCOLLECTION (POINT (1 2), POINT ZM (3 4 5 6))",
                true,
                true,
            ),
            (
                "GEOMETRYCOLLECTION (POINT (1 2), POINT M EMPTY)",
                false,
                true,
            ),
            (
                "GEOMETRYCOLLECTION (POINT (1 2), LINESTRING Z EMPTY)",
                true,
                false,
            ),
            (
                "GEOMETRYCOLLECTION (POINT (1 2), GEOMETRYCOLLECTION (POINT (3 4), POINT M (5 6 7)))",
                false,
                true,
            ),
            (
                "GEOMETRYCOLLECTION (POINT (1 2), POINT (3 4))",
                false,
                false,
            ),
        ] {
            let result = z_tester.invoke_wkb_scalar(Some(wkt)).unwrap();
            z_tester.assert_scalar_result_equals(result, ScalarValue::Boolean(Some(has_z)));
            let result = m_tester.invoke_wkb_scalar(Some(wkt)).unwrap();
            m_tester.assert_scalar_result_equals(result, ScalarValue::Boolean(Some(has_m)));
        }

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
