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
use sedona_expr::aggregate_udf::{SedonaAccumulator, SedonaAccumulatorRef};
use sedona_geometry::wkb_factory::write_wkb_geometrycollection_header;
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};
use wkb::reader::read_wkb;

/// ST_Polygonize_Agg() aggregate implementation using GEOS
pub fn st_polygonize_agg_impl() -> SedonaAccumulatorRef {
    Arc::new(STPolygonizeAgg {})
}

#[derive(Debug)]
struct STPolygonizeAgg {}

impl SedonaAccumulator for STPolygonizeAgg {
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
        Ok(vec![
            Arc::new(Field::new("count", DataType::UInt64, false)),
            Arc::new(Field::new("item", DataType::Binary, true)),
        ])
    }
}

#[derive(Debug)]
struct PolygonizeAccumulator {
    input_type: SedonaType,
    item: Option<Vec<u8>>,
    count: usize,
}

const WKB_HEADER_SIZE: usize = 1 + 4 + 4;

impl PolygonizeAccumulator {
    pub fn new(input_type: SedonaType) -> Self {
        let mut item = Vec::new();
        write_wkb_geometrycollection_header(&mut item, Dimensions::Xy, 0)
            .expect("Failed to write initial GeometryCollection header");

        Self {
            input_type,
            item: Some(item),
            count: 0,
        }
    }

    fn make_wkb_result(&mut self) -> Result<Option<Vec<u8>>> {
        if self.count == 0 {
            return Ok(None);
        }

        let collection_wkb = self.item.as_mut().unwrap();
        let mut header = Vec::new();
        write_wkb_geometrycollection_header(&mut header, Dimensions::Xy, self.count)
            .map_err(|e| DataFusionError::Execution(format!("Failed to write header: {e}")))?;
        collection_wkb[0..WKB_HEADER_SIZE].copy_from_slice(&header);

        let wkb = read_wkb(collection_wkb)
            .map_err(|e| DataFusionError::Execution(format!("Failed to read WKB: {e}")))?;

        let factory = GEOSWkbFactory::new();
        let collection = factory.create(&wkb).map_err(|e| {
            DataFusionError::Execution(format!("Failed to create geometry from WKB: {e}"))
        })?;

        let num_geoms = collection.get_num_geometries().map_err(|e| {
            DataFusionError::Execution(format!("Failed to get number of geometries: {e}"))
        })?;

        let mut geos_geoms = Vec::with_capacity(num_geoms);
        for i in 0..num_geoms {
            let geom = collection.get_geometry_n(i).map_err(|e| {
                DataFusionError::Execution(format!("Failed to get geometry {}: {e}", i))
            })?;
            geos_geoms.push(geom.clone());
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
            return sedona_internal_err!("No input arrays provided to accumulator in update_batch");
        }

        let arg_types = std::slice::from_ref(&self.input_type);
        let args = [ColumnarValue::Array(values[0].clone())];
        let executor = sedona_functions::executor::WkbExecutor::new(arg_types, &args);

        let item_ref = self.item.as_mut().unwrap();
        executor.execute_wkb_void(|maybe_item| {
            if let Some(item) = maybe_item {
                item_ref.extend_from_slice(item.buf());
                self.count += 1;
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
        let item_capacity = self.item.as_ref().map(|e| e.capacity()).unwrap_or(0);
        size_of::<PolygonizeAccumulator>() + item_capacity
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let serialized_count = ScalarValue::UInt64(Some(self.count as u64));
        let serialized_item = ScalarValue::Binary(self.item.take());

        let mut item = Vec::new();
        write_wkb_geometrycollection_header(&mut item, Dimensions::Xy, 0)
            .expect("Failed to write initial GeometryCollection header");
        self.item = Some(item);
        self.count = 0;

        Ok(vec![serialized_count, serialized_item])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        if states.len() != 2 {
            return sedona_internal_err!(
                "Unexpected number of state fields for st_polygonize() (expected 2, got {})",
                states.len()
            );
        }

        let item_ref = match self.item.as_mut() {
            Some(item) => item,
            None => return sedona_internal_err!("Unexpected internal state in ST_Polygonize()"),
        };

        let count_array = states[0].as_primitive::<UInt64Type>();
        let item_array = as_binary_array(&states[1])?;

        for i in 0..count_array.len() {
            let count = count_array.value(i) as usize;
            if count > 0 && !item_array.is_null(i) {
                let item = item_array.value(i);
                // Skip the header and append the geometry data
                item_ref.extend_from_slice(&item[WKB_HEADER_SIZE..]);
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
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_VIEW_GEOMETRY};
    use sedona_testing::{compare::assert_scalar_equal_wkb_geometry, testers::AggregateUdfTester};

    use super::*;

    fn create_udf() -> SedonaAggregateUDF {
        SedonaAggregateUDF::new(
            "st_polygonize_agg",
            vec![st_polygonize_agg_impl()],
            datafusion_expr::Volatility::Immutable,
            None,
        )
    }

    #[test]
    fn udf_metadata() {
        let udf = create_udf();
        let aggregate_udf: AggregateUDF = udf.into();
        assert_eq!(aggregate_udf.name(), "st_polygonize_agg");
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

    #[rstest]
    fn polygonize_single_polygon(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let udf = create_udf();
        let tester = AggregateUdfTester::new(udf.into(), vec![sedona_type.clone()]);

        let batches = vec![vec![Some("POLYGON ((10 0, 0 0, 10 10, 10 0))")]];

        let result = tester.aggregate_wkt(batches).unwrap();
        assert_scalar_equal_wkb_geometry(
            &result,
            Some("GEOMETRYCOLLECTION (POLYGON ((10 0, 0 0, 10 10, 10 0)))"),
        );
    }

    #[rstest]
    fn polygonize_multipolygon(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = create_udf();
        let tester = AggregateUdfTester::new(udf.into(), vec![sedona_type.clone()]);

        let batches = vec![vec![Some(
            "MULTIPOLYGON (((0 0, 1 0, 0 1, 0 0)), ((10 10, 11 10, 10 11, 10 10)))",
        )]];

        let result = tester.aggregate_wkt(batches).unwrap();
        assert_scalar_equal_wkb_geometry(
            &result,
            Some("GEOMETRYCOLLECTION (POLYGON ((0 0, 0 1, 1 0, 0 0)), POLYGON ((10 10, 10 11, 11 10, 10 10)))"),
        );
    }

    #[rstest]
    fn polygonize_closed_ring_linestring(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let udf = create_udf();
        let tester = AggregateUdfTester::new(udf.into(), vec![sedona_type.clone()]);

        let batches = vec![vec![Some("LINESTRING (0 0, 0 1, 1 1, 1 0, 0 0)")]];

        let result = tester.aggregate_wkt(batches).unwrap();
        assert_scalar_equal_wkb_geometry(
            &result,
            Some("GEOMETRYCOLLECTION (POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0)))"),
        );
    }

    #[rstest]
    fn polygonize_point_returns_empty(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let udf = create_udf();
        let tester = AggregateUdfTester::new(udf.into(), vec![sedona_type.clone()]);

        let batches = vec![vec![Some("POINT (0 0)")]];

        let result = tester.aggregate_wkt(batches).unwrap();
        assert_scalar_equal_wkb_geometry(&result, Some("GEOMETRYCOLLECTION EMPTY"));
    }

    #[rstest]
    fn polygonize_multipoint_returns_empty(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let udf = create_udf();
        let tester = AggregateUdfTester::new(udf.into(), vec![sedona_type.clone()]);

        let batches = vec![vec![Some("MULTIPOINT ((0 0), (1 1))")]];

        let result = tester.aggregate_wkt(batches).unwrap();
        assert_scalar_equal_wkb_geometry(&result, Some("GEOMETRYCOLLECTION EMPTY"));
    }

    #[rstest]
    fn polygonize_multilinestring_returns_empty(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let udf = create_udf();
        let tester = AggregateUdfTester::new(udf.into(), vec![sedona_type.clone()]);

        let batches = vec![vec![Some("MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))")]];

        let result = tester.aggregate_wkt(batches).unwrap();
        assert_scalar_equal_wkb_geometry(&result, Some("GEOMETRYCOLLECTION EMPTY"));
    }

    #[rstest]
    fn polygonize_geometrycollection_returns_empty(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let udf = create_udf();
        let tester = AggregateUdfTester::new(udf.into(), vec![sedona_type.clone()]);

        let batches = vec![vec![Some(
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 1))",
        )]];

        let result = tester.aggregate_wkt(batches).unwrap();
        assert_scalar_equal_wkb_geometry(&result, Some("GEOMETRYCOLLECTION EMPTY"));
    }

    #[rstest]
    fn polygonize_empty_linestring_returns_empty(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let udf = create_udf();
        let tester = AggregateUdfTester::new(udf.into(), vec![sedona_type.clone()]);

        let batches = vec![vec![Some("LINESTRING EMPTY")]];

        let result = tester.aggregate_wkt(batches).unwrap();
        assert_scalar_equal_wkb_geometry(&result, Some("GEOMETRYCOLLECTION EMPTY"));
    }
}
