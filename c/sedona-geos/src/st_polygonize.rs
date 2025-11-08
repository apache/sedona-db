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

use arrow_array::{
    builder::OffsetBufferBuilder, cast::as_list_array, Array, ArrayRef, BinaryArray,
};
use arrow_schema::{DataType, Field, FieldRef};
use datafusion_common::{cast::as_binary_array, error::Result, DataFusionError, ScalarValue};
use datafusion_expr::{Accumulator, ColumnarValue};
use geos::Geom;
use sedona_expr::aggregate_udf::{SedonaAccumulator, SedonaAccumulatorRef};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

/// ST_Polygonize() aggregate implementation using GEOS
pub fn st_polygonize_impl() -> SedonaAccumulatorRef {
    Arc::new(STPolygonize {})
}

#[derive(Debug)]
struct STPolygonize {}

impl SedonaAccumulator for STPolygonize {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOMETRY);
        matcher.match_args(args)
    }

    fn accumulator(
        &self,
        args: &[SedonaType],
        _output_type: &SedonaType,
    ) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(PolygonizeAccumulator::new(args[0].clone())))
    }

    fn state_fields(&self, _args: &[SedonaType]) -> Result<Vec<FieldRef>> {
        Ok(vec![Arc::new(Field::new(
            "geometries",
            DataType::List(Arc::new(Field::new("item", DataType::Binary, true))),
            false,
        ))])
    }
}

#[derive(Debug)]
struct PolygonizeAccumulator {
    input_type: SedonaType,
    geometries: Vec<Vec<u8>>,
}

impl PolygonizeAccumulator {
    pub fn new(input_type: SedonaType) -> Self {
        Self {
            input_type,
            geometries: Vec::new(),
        }
    }

    fn make_wkb_result(&self) -> Result<Option<Vec<u8>>> {
        if self.geometries.is_empty() {
            return Ok(None);
        }

        let mut geos_geoms = Vec::with_capacity(self.geometries.len());
        for wkb in &self.geometries {
            let geom = geos::Geometry::new_from_wkb(wkb).map_err(|e| {
                DataFusionError::Execution(format!("Failed to convert WKB to GEOS: {e}"))
            })?;
            geos_geoms.push(geom);
        }

        let result = geos::Geometry::polygonize(&geos_geoms)
            .map_err(|e| DataFusionError::Execution(format!("Failed to polygonize: {e}")))?;

        let wkb = result.to_wkb().map_err(|e| {
            DataFusionError::Execution(format!("Failed to convert result to WKB: {e}"))
        })?;

        Ok(Some(wkb.into()))
    }
}

impl Accumulator for PolygonizeAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.is_empty() {
            return Err(DataFusionError::Internal(
                "No input arrays provided to accumulator in update_batch".to_string(),
            ));
        }

        let arg_types = [self.input_type.clone()];
        let args = [ColumnarValue::Array(values[0].clone())];
        let executor = sedona_functions::executor::WkbExecutor::new(&arg_types, &args);

        executor.execute_wkb_void(|maybe_item| {
            if let Some(item) = maybe_item {
                self.geometries.push(item.buf().to_vec());
            }
            Ok(())
        })?;

        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let wkb = self.make_wkb_result()?;
        Ok(ScalarValue::Binary(wkb))
    }

    fn size(&self) -> usize {
        std::mem::size_of::<Self>() + self.geometries.iter().map(|g| g.capacity()).sum::<usize>()
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let binary_array =
            BinaryArray::from_iter(self.geometries.iter().map(|g| Some(g.as_slice())));
        let mut offsets_builder = OffsetBufferBuilder::new(1);
        offsets_builder.push_length(binary_array.len());
        let offsets = offsets_builder.finish();

        let list_array = arrow_array::ListArray::new(
            Arc::new(Field::new("item", DataType::Binary, true)),
            offsets,
            Arc::new(binary_array),
            None,
        );

        Ok(vec![ScalarValue::List(Arc::new(list_array))])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        if states.is_empty() {
            return Err(DataFusionError::Internal(
                "No input arrays provided to accumulator in merge_batch".to_string(),
            ));
        }

        let list_array = as_list_array(&states[0]);

        for i in 0..list_array.len() {
            if list_array.is_null(i) {
                continue;
            }

            let value_ref = list_array.value(i);
            let binary_array = as_binary_array(&value_ref)?;
            for j in 0..binary_array.len() {
                if !binary_array.is_null(j) {
                    self.geometries.push(binary_array.value(j).to_vec());
                }
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
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::{compare::assert_scalar_equal_wkb_geometry, testers::AggregateUdfTester};

    use super::*;

    fn create_udf() -> SedonaAggregateUDF {
        SedonaAggregateUDF::new(
            "st_polygonize",
            vec![st_polygonize_impl()],
            datafusion_expr::Volatility::Immutable,
            None,
        )
    }

    #[test]
    fn udf_metadata() {
        let udf = create_udf();
        let aggregate_udf: AggregateUDF = udf.into();
        assert_eq!(aggregate_udf.name(), "st_polygonize");
    }

    #[rstest]
    fn basic_triangle(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = create_udf();
        let tester = AggregateUdfTester::new(udf.into(), vec![sedona_type.clone()]);
        assert_eq!(tester.return_type().unwrap(), WKB_GEOMETRY);

        let batches = vec![vec![
            Some("LINESTRING (0 0, 10 0)"),
            Some("LINESTRING (10 0, 10 10)"),
            Some("LINESTRING (10 10, 0 0)"),
        ]];

        let result = tester.aggregate_wkt(batches).unwrap();
        assert_scalar_equal_wkb_geometry(
            &result,
            Some("GEOMETRYCOLLECTION (POLYGON ((10 0, 0 0, 10 10, 10 0)))"),
        );
    }

    #[rstest]
    fn polygonize_with_nulls(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = create_udf();
        let tester = AggregateUdfTester::new(udf.into(), vec![sedona_type.clone()]);

        let batches = vec![vec![
            Some("LINESTRING (0 0, 10 0)"),
            None,
            Some("LINESTRING (10 0, 10 10)"),
            None,
            Some("LINESTRING (10 10, 0 0)"),
        ]];

        let result = tester.aggregate_wkt(batches).unwrap();
        assert_scalar_equal_wkb_geometry(
            &result,
            Some("GEOMETRYCOLLECTION (POLYGON ((10 0, 0 0, 10 10, 10 0)))"),
        );
    }

    #[rstest]
    fn polygonize_empty_input(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = create_udf();
        let tester = AggregateUdfTester::new(udf.into(), vec![sedona_type.clone()]);

        let batches: Vec<Vec<Option<&str>>> = vec![];
        assert_scalar_equal_wkb_geometry(&tester.aggregate_wkt(batches).unwrap(), None);
    }

    #[rstest]
    fn polygonize_no_polygons_formed(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let udf = create_udf();
        let tester = AggregateUdfTester::new(udf.into(), vec![sedona_type.clone()]);

        let batches = vec![vec![
            Some("LINESTRING (0 0, 10 0)"),
            Some("LINESTRING (20 0, 30 0)"),
        ]];
        assert_scalar_equal_wkb_geometry(
            &tester.aggregate_wkt(batches).unwrap(),
            Some("GEOMETRYCOLLECTION EMPTY"),
        );
    }

    #[rstest]
    fn polygonize_multiple_polygons(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let udf = create_udf();
        let tester = AggregateUdfTester::new(udf.into(), vec![sedona_type.clone()]);

        let batches = vec![vec![
            Some("LINESTRING (0 0, 10 0)"),
            Some("LINESTRING (10 0, 5 10)"),
            Some("LINESTRING (5 10, 0 0)"),
            Some("LINESTRING (20 0, 30 0)"),
            Some("LINESTRING (30 0, 25 10)"),
            Some("LINESTRING (25 10, 20 0)"),
        ]];

        let result = tester.aggregate_wkt(batches).unwrap();
        assert_scalar_equal_wkb_geometry(
            &result,
            Some("GEOMETRYCOLLECTION (POLYGON ((10 0, 0 0, 5 10, 10 0)), POLYGON ((30 0, 20 0, 25 10, 30 0)))"),
        );
    }

    #[rstest]
    fn polygonize_multiple_batches(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        // Testing merge_batch
        let udf = create_udf();
        let tester = AggregateUdfTester::new(udf.into(), vec![sedona_type.clone()]);

        let batches = vec![
            vec![Some("LINESTRING (0 0, 10 0)")],
            vec![Some("LINESTRING (10 0, 10 10)")],
            vec![Some("LINESTRING (10 10, 0 0)")],
        ];

        let result = tester.aggregate_wkt(batches).unwrap();
        assert_scalar_equal_wkb_geometry(
            &result,
            Some("GEOMETRYCOLLECTION (POLYGON ((10 0, 0 0, 10 10, 10 0)))"),
        );
    }
}
