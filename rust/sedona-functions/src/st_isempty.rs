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
use datafusion_expr::{ColumnarValue, Volatility};
use sedona_expr::{
    item_crs::ItemCrsKernel,
    scalar_udf::{SedonaScalarKernel, SedonaScalarUDF},
};
use sedona_geometry::is_empty::is_geometry_empty;
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};
use wkb::reader::Wkb;

pub fn st_isempty_udf() -> SedonaScalarUDF {
    SedonaScalarUDF::new(
        "st_isempty",
        ItemCrsKernel::wrap_impl(vec![Arc::new(STIsEmpty {})]),
        Volatility::Immutable,
        None,
    )
}

#[derive(Debug)]
struct STIsEmpty {}

impl SedonaScalarKernel for STIsEmpty {
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
        let executor = WkbExecutor::new(arg_types, args);
        let mut builder = BooleanBuilder::with_capacity(executor.num_iterations());

        executor.execute_wkb_void(|maybe_item| {
            match maybe_item {
                Some(item) => {
                    builder.append_value(invoke_scalar(&item)?);
                }
                None => builder.append_null(),
            }
            Ok(())
        })?;

        executor.finish(Arc::new(builder.finish()))
    }
}

pub fn is_wkb_empty(item: &Wkb) -> Result<bool> {
    invoke_scalar(item)
}

fn invoke_scalar(item: &Wkb) -> Result<bool> {
    is_geometry_empty(item).map_err(|e| {
        datafusion_common::error::DataFusionError::Execution(format!(
            "Failed to check if geometry is empty: {e}"
        ))
    })
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
        let udf: ScalarUDF = st_isempty_udf().into();
        assert_eq!(udf.name(), "st_isempty");
    }

    #[rstest]
    fn udf(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY, WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY, WKB_GEOMETRY_ITEM_CRS.clone(), WKB_GEOGRAPHY_ITEM_CRS.clone())]
        sedona_type: SedonaType,
    ) {
        let tester = ScalarUdfTester::new(st_isempty_udf().into(), vec![sedona_type.clone()]);

        tester.assert_return_type(DataType::Boolean);

        let result = tester.invoke_wkb_scalar(Some("POINT EMPTY")).unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Boolean(Some(true)));

        let result = tester.invoke_wkb_scalar(None).unwrap();
        tester.assert_scalar_result_equals(result, ScalarValue::Null);

        let input_wkt = vec![
            None,
            Some("MULTIPOINT EMPTY"),
            Some("LINESTRING EMPTY"),
            Some("MULTILINESTRING EMPTY"),
            Some("POLYGON EMPTY"),
            Some("MULTIPOLYGON EMPTY"),
            Some("GEOMETRYCOLLECTION EMPTY"),
            Some("POINT (1 2)"),
            Some("MULTIPOINT ((0 0))"),
            Some("LINESTRING (1 2, 2 2)"),
            Some("MULTILINESTRING ((0 0, 1 0), (1 1, 0 1))"),
            Some("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"),
            Some("MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)))"),
            Some("GEOMETRYCOLLECTION (POINT (1 2), LINESTRING (1 2, 2 2))"),
        ];
        let expected: ArrayRef = arrow_array!(
            Boolean,
            [
                None,
                Some(true),
                Some(true),
                Some(true),
                Some(true),
                Some(true),
                Some(true),
                Some(false),
                Some(false),
                Some(false),
                Some(false),
                Some(false),
                Some(false),
                Some(false)
            ]
        );
        assert_array_equal(&tester.invoke_wkb_array(input_wkt).unwrap(), &expected);
    }
}
