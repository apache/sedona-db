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

use arrow_array::builder::BooleanBuilder;
use arrow_schema::DataType;
use datafusion_common::error::{DataFusionError, Result};
use datafusion_expr::Volatility;
use sedona_common::sedona_internal_err;
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{SedonaScalarKernel, SedonaScalarUDF},
};
use sedona_geometry::types::GeometryTypeId;
use sedona_geometry::wkb_header::WkbHeader;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::executor::WkbBytesExecutor;

pub fn st_iscollection_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_iscollection",
        ItemCrsKernel::wrap_impl(vec![Arc::new(STIsCollection {})]),
        Volatility::Immutable,
    )
}

#[derive(Debug)]
struct STIsCollection {}

impl SedonaScalarKernel for STIsCollection {
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
        args: &[datafusion_expr::ColumnarValue],
    ) -> Result<datafusion_expr::ColumnarValue> {
        let executor = WkbBytesExecutor::new(arg_types, args);
        let mut builder = BooleanBuilder::with_capacity(executor.num_iterations());

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

fn invoke_scalar(buf: &[u8]) -> Result<bool> {
    let header = WkbHeader::try_new(buf).map_err(|e| DataFusionError::External(Box::new(e)))?;
    let geometry_type = header
        .geometry_type_id()
        .map_err(|e| DataFusionError::External(Box::new(e)))?;
    match geometry_type {
        GeometryTypeId::MultiPoint
        | GeometryTypeId::MultiLineString
        | GeometryTypeId::MultiPolygon
        | GeometryTypeId::GeometryCollection => Ok(true),
        GeometryTypeId::Point | GeometryTypeId::LineString | GeometryTypeId::Polygon => Ok(false),
        _ => sedona_internal_err!("Invalid geometry type"),
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{create_array as arrow_array, ArrayRef};
    use datafusion_common::ScalarValue;
    use datafusion_expr::ScalarUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{
        WKB_GEOGRAPHY, WKB_GEOGRAPHY_ITEM_CRS, WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS,
        WKB_VIEW_GEOGRAPHY, WKB_VIEW_GEOMETRY,
    };
    use sedona_testing::{compare::assert_array_equal, testers::ScalarUdfTester};

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: ScalarUDF = st_iscollection_udf().into();
        assert_eq!(udf.name(), "st_iscollection");
    }

    #[rstest]
    fn udf(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY, WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY, WKB_GEOMETRY_ITEM_CRS.clone(), WKB_GEOGRAPHY_ITEM_CRS.clone())]
        sedona_type: SedonaType,
    ) {
        let tester = ScalarUdfTester::new(st_iscollection_udf().into(), vec![sedona_type.clone()]);

        tester.assert_return_type(DataType::Boolean);

        let result = tester
            .invoke_wkb_scalar(Some("MULTIPOINT((0 0), (1 1))"))
            .unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Boolean(Some(true)));

        let result = tester.invoke_wkb_scalar(None).unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Null);

        let input_wkt = vec![
            None,
            Some("POINT(0 0)"),
            Some("LINESTRING(0 0, 1 1)"),
            Some("POLYGON((0 0, 1 0, 0 1, 0 0))"),
            Some("MULTIPOINT((0 0), (1 1))"),
            Some("MULTILINESTRING((0 0, 0 1, 1 1, 0 0),(0 0, 1 1))"),
            Some("MULTIPOLYGON(((0 0, 1 0, 1 1, 0 1, 0 0)))"),
            Some("GEOMETRYCOLLECTION(POINT(0 0), LINESTRING(0 0, 1 1))"),
        ];
        let expected: ArrayRef = arrow_array!(
            Boolean,
            [
                None,
                Some(false),
                Some(false),
                Some(false),
                Some(true),
                Some(true),
                Some(true),
                Some(true)
            ]
        );
        assert_array_equal(&tester.invoke_wkb_array(input_wkt).unwrap(), &expected);
    }
}
