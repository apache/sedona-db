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

use crate::wkb_to_geos::GEOSWkbFactory;
use arrow_array::{cast::AsArray, types::UInt64Type, Array, ArrayRef};
use arrow_schema::{DataType, Field, FieldRef};
use datafusion_common::{cast::as_binary_array, error::Result, DataFusionError, ScalarValue};
use datafusion_expr::{Accumulator, ColumnarValue};
use geo_traits::Dimensions;
use geos::Geom;
use sedona_common::sedona_internal_err;
use sedona_expr::{
    aggregate_udf::{SedonaAccumulator, SedonaAccumulatorRef},
    item_crs::ItemCrsSedonaAccumulator,
};
use sedona_geometry::wkb_factory::write_wkb_geometrycollection_header;
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};
use wkb::reader::read_wkb;

/// ST_Union_Agg() aggregate implementation using GEOS
pub fn st_union_agg_impl() -> Vec<SedonaAccumulatorRef> {
    ItemCrsSedonaAccumulator::wrap_impl(STUnionAgg {})
}

#[derive(Debug)]
struct STUnionAgg {}

impl SedonaAccumulator for STUnionAgg {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOMETRY);
        matcher.match_args(args)
    }

    fn accumulator(
        &self,
        args: &[SedonaType],
        _output_type: &SedonaType,
    ) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(UnionAccumulator::new(args[0].clone())))
    }

    fn state_fields(&self, _args: &[SedonaType]) -> Result<Vec<FieldRef>> {
        Ok(vec![
            Arc::new(Field::new("count", DataType::UInt64, false)),
            Arc::new(Field::new("item", DataType::Binary, true)),
        ])
    }
}

#[derive(Debug)]
struct UnionAccumulator {
    input_type: SedonaType,
    item: Vec<u8>,
    count: usize,
}

const WKB_HEADER_SIZE: usize = 1 + 4 + 4;

fn empty_collection() -> Vec<u8> {
    let mut item = Vec::new();
    write_wkb_geometrycollection_header(&mut item, Dimensions::Xy, 0)
        .expect("Failed to write initial GeometryCollection header");
    item
}

impl UnionAccumulator {
    fn new(input_type: SedonaType) -> Self {
        Self {
            input_type,
            item: empty_collection(),
            count: 0,
        }
    }

    fn make_wkb_result(&mut self) -> Result<Option<Vec<u8>>> {
        if self.count == 0 {
            return Ok(None);
        }

        let mut header = Vec::new();
        write_wkb_geometrycollection_header(&mut header, Dimensions::Xy, self.count)
            .map_err(|e| DataFusionError::Execution(format!("Failed to write header: {e}")))?;
        self.item[0..WKB_HEADER_SIZE].copy_from_slice(&header);

        let wkb = read_wkb(&self.item)
            .map_err(|e| DataFusionError::Execution(format!("Failed to read WKB: {e}")))?;
        let collection = GEOSWkbFactory::new().create(&wkb).map_err(|e| {
            DataFusionError::Execution(format!("Failed to create geometry from WKB: {e}"))
        })?;
        let union = collection
            .unary_union()
            .map_err(|e| DataFusionError::Execution(format!("Failed to compute union: {e}")))?;
        let wkb = union.to_wkb().map_err(|e| {
            DataFusionError::Execution(format!("Failed to convert result to WKB: {e}"))
        })?;

        Ok(Some(wkb))
    }
}

impl Accumulator for UnionAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.is_empty() {
            return sedona_internal_err!("No input arrays provided to accumulator in update_batch");
        }

        let arg_types = std::slice::from_ref(&self.input_type);
        let args = [ColumnarValue::Array(values[0].clone())];
        let executor = sedona_functions::executor::WkbExecutor::new(arg_types, &args);

        executor.execute_wkb_void(|maybe_item| {
            if let Some(item) = maybe_item {
                self.item.extend_from_slice(item.buf());
                self.count += 1;
            }
            Ok(())
        })?;

        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        Ok(ScalarValue::Binary(self.make_wkb_result()?))
    }

    fn size(&self) -> usize {
        size_of::<UnionAccumulator>() + self.item.capacity()
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let serialized_count = ScalarValue::UInt64(Some(self.count as u64));
        let serialized_item =
            ScalarValue::Binary(Some(std::mem::replace(&mut self.item, empty_collection())));
        self.count = 0;
        Ok(vec![serialized_count, serialized_item])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        if states.len() != 2 {
            return sedona_internal_err!(
                "Unexpected number of state fields for st_union_agg() (expected 2, got {})",
                states.len()
            );
        }

        let count_array = states[0].as_primitive::<UInt64Type>();
        let item_array = as_binary_array(&states[1])?;

        for i in 0..count_array.len() {
            let count = count_array.value(i) as usize;
            if count > 0 && !item_array.is_null(i) {
                self.item
                    .extend_from_slice(&item_array.value(i)[WKB_HEADER_SIZE..]);
                self.count += count;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use datafusion_expr::AggregateUDF;
    use rstest::rstest;
    use sedona_expr::aggregate_udf::SedonaAggregateUDF;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS, WKB_VIEW_GEOMETRY};
    use sedona_testing::{compare::assert_scalar_equal_wkb_geometry, testers::AggregateUdfTester};

    use super::*;

    fn create_udf() -> SedonaAggregateUDF {
        SedonaAggregateUDF::new(
            "st_union_agg",
            st_union_agg_impl(),
            datafusion_expr::Volatility::Immutable,
        )
    }

    #[test]
    fn udf_metadata() {
        let aggregate_udf: AggregateUDF = create_udf().into();
        assert_eq!(aggregate_udf.name(), "st_union_agg");
    }

    #[rstest]
    fn polygons(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = AggregateUdfTester::new(create_udf().into(), vec![sedona_type]);
        assert_eq!(tester.return_type().unwrap(), WKB_GEOMETRY);

        assert_scalar_equal_wkb_geometry(
            &tester
                .aggregate_wkt(vec![
                    vec![Some("POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))")],
                    vec![Some("POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))")],
                ])
                .unwrap(),
            Some("POLYGON ((2 1, 2 0, 0 0, 0 2, 1 2, 1 3, 3 3, 3 1, 2 1))"),
        );

        assert_scalar_equal_wkb_geometry(
            &tester
                .aggregate_wkt(vec![
                    vec![Some("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"), None],
                    vec![Some("MULTIPOLYGON (((2 2, 3 2, 3 3, 2 3, 2 2)))"), None],
                ])
                .unwrap(),
            Some("MULTIPOLYGON (((0 0, 0 1, 1 1, 1 0, 0 0)), ((2 2, 2 3, 3 3, 3 2, 2 2)))"),
        );
    }

    #[rstest]
    fn non_polygonal_input_in_any_order(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let tester = AggregateUdfTester::new(create_udf().into(), vec![sedona_type]);
        let expected =
            Some("GEOMETRYCOLLECTION (POINT (5 5), POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0)))");

        for batches in [
            vec![vec![
                Some("POINT (5 5)"),
                Some("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"),
            ]],
            vec![vec![
                Some("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"),
                Some("POINT (5 5)"),
            ]],
        ] {
            assert_scalar_equal_wkb_geometry(&tester.aggregate_wkt(batches).unwrap(), expected);
        }

        assert_scalar_equal_wkb_geometry(
            &tester
                .aggregate_wkt(vec![vec![
                    Some("LINESTRING (0 0, 1 1)"),
                    Some("LINESTRING (1 1, 2 2)"),
                ]])
                .unwrap(),
            Some("LINESTRING (0 0, 1 1, 2 2)"),
        );
    }

    #[rstest]
    fn no_input(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = AggregateUdfTester::new(create_udf().into(), vec![sedona_type]);
        assert_scalar_equal_wkb_geometry(&tester.aggregate(&vec![]).unwrap(), None);
        assert_scalar_equal_wkb_geometry(
            &tester.aggregate_wkt(vec![vec![None, None]]).unwrap(),
            None,
        );
    }

    #[test]
    fn item_crs() {
        let tester =
            AggregateUdfTester::new(create_udf().into(), vec![WKB_GEOMETRY_ITEM_CRS.clone()]);
        assert_eq!(tester.return_type().unwrap(), WKB_GEOMETRY_ITEM_CRS.clone());
    }
}
