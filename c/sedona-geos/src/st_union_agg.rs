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

use crate::{geos_to_wkb::write_geos_geometry, wkb_to_geos::GEOSWkbFactory};
use arrow_array::{
    cast::AsArray,
    types::{UInt32Type, UInt64Type},
    Array, ArrayRef,
};
use arrow_schema::{DataType, Field, FieldRef};
use datafusion_common::{
    cast::as_binary_array, error::Result, exec_err, DataFusionError, ScalarValue,
};
use datafusion_expr::{Accumulator, ColumnarValue};
use geo_traits::{
    Dimensions, GeometryCollectionTrait, GeometryTrait, GeometryType, LineStringTrait,
    MultiLineStringTrait, MultiPointTrait, MultiPolygonTrait, PointTrait, PolygonTrait,
};
use geos::Geom;
use sedona_common::sedona_internal_err;
use sedona_expr::{
    aggregate_udf::{SedonaAccumulator, SedonaAccumulatorRef},
    item_crs::ItemCrsSedonaAccumulator,
};
use sedona_geometry::{
    types::{GeometryTypeAndDimensions, GeometryTypeAndDimensionsSet, GeometryTypeId},
    wkb_factory::{
        write_wkb_empty_point, write_wkb_geometrycollection_header, write_wkb_linestring_header,
        write_wkb_multilinestring_header, write_wkb_multipoint_header,
        write_wkb_multipolygon_header, write_wkb_polygon_header,
    },
};
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
            Arc::new(Field::new("types_and_dims", DataType::UInt32, false)),
            Arc::new(Field::new("empty_types", DataType::UInt32, false)),
            Arc::new(Field::new("count", DataType::UInt64, false)),
            Arc::new(Field::new("item", DataType::Binary, true)),
        ])
    }
}

#[derive(Debug)]
struct UnionAccumulator {
    input_type: SedonaType,
    types_and_dims: GeometryTypeAndDimensionsSet,
    empty_types: GeometryTypeAndDimensionsSet,
    item: Vec<u8>,
    count: usize,
}

const WKB_HEADER_SIZE: usize = 1 + 4 + 4;

fn is_empty_recursive(geometry: &impl GeometryTrait<T = f64>) -> bool {
    match geometry.as_type() {
        GeometryType::Point(point) => point.coord().is_none(),
        GeometryType::LineString(line_string) => line_string.num_coords() == 0,
        GeometryType::Polygon(polygon) => is_polygon_empty(polygon),
        GeometryType::MultiPoint(multi_point) => {
            multi_point.points().all(|point| point.coord().is_none())
        }
        GeometryType::MultiLineString(multi_line_string) => multi_line_string
            .line_strings()
            .all(|line_string| line_string.num_coords() == 0),
        GeometryType::MultiPolygon(multi_polygon) => multi_polygon
            .polygons()
            .all(|polygon| is_polygon_empty(&polygon)),
        GeometryType::GeometryCollection(collection) => collection
            .geometries()
            .all(|member| is_empty_recursive(&member)),
        _ => false,
    }
}

fn is_polygon_empty(polygon: &impl PolygonTrait<T = f64>) -> bool {
    polygon.num_interiors() == 0
        && polygon
            .exterior()
            .is_none_or(|exterior| exterior.num_coords() == 0)
}

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
            types_and_dims: GeometryTypeAndDimensionsSet::new(),
            empty_types: GeometryTypeAndDimensionsSet::new(),
            item: empty_collection(),
            count: 0,
        }
    }

    fn make_wkb_result(&mut self) -> Result<Option<Vec<u8>>> {
        if self.count == 0 {
            return self.make_empty_result();
        }

        let unique_dimensions = self.types_and_dims.dimensions();
        if unique_dimensions.len() != 1 {
            return exec_err!("Can't ST_Union_Agg() mixed dimension geometries");
        }

        let mut header = Vec::new();
        write_wkb_geometrycollection_header(&mut header, unique_dimensions[0], self.count)
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
        let mut wkb = Vec::new();
        write_geos_geometry(&union, &mut wkb)?;

        Ok(Some(wkb))
    }

    fn make_empty_result(&self) -> Result<Option<Vec<u8>>> {
        let Some(geometry_type) = self
            .empty_types
            .geometry_types()
            .into_iter()
            .max_by_key(|geometry_type| geometry_type.wkb_id())
        else {
            return Ok(None);
        };

        let mut wkb = Vec::new();
        let dims = Dimensions::Xy;
        match geometry_type {
            GeometryTypeId::Point => write_wkb_empty_point(&mut wkb, dims),
            GeometryTypeId::LineString => write_wkb_linestring_header(&mut wkb, dims, 0),
            GeometryTypeId::Polygon => write_wkb_polygon_header(&mut wkb, dims, 0),
            GeometryTypeId::MultiPoint => write_wkb_multipoint_header(&mut wkb, dims, 0),
            GeometryTypeId::MultiLineString => write_wkb_multilinestring_header(&mut wkb, dims, 0),
            GeometryTypeId::MultiPolygon => write_wkb_multipolygon_header(&mut wkb, dims, 0),
            GeometryTypeId::GeometryCollection | GeometryTypeId::Geometry => {
                write_wkb_geometrycollection_header(&mut wkb, dims, 0)
            }
        }
        .map_err(|e| DataFusionError::External(Box::new(e)))?;

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
                let types_and_dims = GeometryTypeAndDimensions::try_from_geom(&item)
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                if is_empty_recursive(&item) {
                    self.empty_types
                        .insert(&types_and_dims)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                } else {
                    self.types_and_dims
                        .insert(&types_and_dims)
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
                    self.item.extend_from_slice(item.buf());
                    self.count += 1;
                }
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
        let serialized_types_and_dims = ScalarValue::UInt32(Some(self.types_and_dims.bits()));
        let serialized_empty_types = ScalarValue::UInt32(Some(self.empty_types.bits()));
        let serialized_count = ScalarValue::UInt64(Some(self.count as u64));
        let serialized_item =
            ScalarValue::Binary(Some(std::mem::replace(&mut self.item, empty_collection())));
        self.types_and_dims.clear();
        self.empty_types.clear();
        self.count = 0;
        Ok(vec![
            serialized_types_and_dims,
            serialized_empty_types,
            serialized_count,
            serialized_item,
        ])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        if states.len() != 4 {
            return sedona_internal_err!(
                "Unexpected number of state fields for st_union_agg() (expected 4, got {})",
                states.len()
            );
        }

        let types_and_dims_array = states[0].as_primitive::<UInt32Type>();
        let empty_types_array = states[1].as_primitive::<UInt32Type>();
        let count_array = states[2].as_primitive::<UInt64Type>();
        let item_array = as_binary_array(&states[3])?;

        for i in 0..count_array.len() {
            self.types_and_dims
                .merge(&GeometryTypeAndDimensionsSet::from_bits(
                    types_and_dims_array.value(i),
                ));
            self.empty_types
                .merge(&GeometryTypeAndDimensionsSet::from_bits(
                    empty_types_array.value(i),
                ));
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
    use sedona_testing::{
        compare::assert_scalar_equal, create::create_scalar, testers::AggregateUdfTester,
    };

    use super::*;

    fn normalized(value: &ScalarValue) -> ScalarValue {
        match value {
            ScalarValue::Binary(Some(wkb)) => {
                let mut geometry = geos::Geometry::new_from_wkb(wkb).unwrap();
                geometry.normalize().unwrap();
                let mut normalized = Vec::new();
                write_geos_geometry(&geometry, &mut normalized).unwrap();
                ScalarValue::Binary(Some(normalized))
            }
            other => other.clone(),
        }
    }

    fn assert_union_equals(actual: &ScalarValue, expected_wkt: Option<&str>) {
        let expected = create_scalar(expected_wkt, &WKB_GEOMETRY);
        assert_scalar_equal(&normalized(actual), &normalized(&expected));
    }

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

        assert_union_equals(
            &tester
                .aggregate_wkt(vec![
                    vec![Some("POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))")],
                    vec![Some("POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))")],
                ])
                .unwrap(),
            Some("POLYGON ((2 1, 2 0, 0 0, 0 2, 1 2, 1 3, 3 3, 3 1, 2 1))"),
        );

        assert_union_equals(
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
            assert_union_equals(&tester.aggregate_wkt(batches).unwrap(), expected);
        }

        assert_union_equals(
            &tester
                .aggregate_wkt(vec![vec![
                    Some("LINESTRING (0 0, 1 1)"),
                    Some("LINESTRING (2 2, 3 3)"),
                ]])
                .unwrap(),
            Some("MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))"),
        );
    }

    #[rstest]
    fn no_input(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = AggregateUdfTester::new(create_udf().into(), vec![sedona_type]);
        assert_union_equals(&tester.aggregate(&vec![]).unwrap(), None);
        assert_union_equals(&tester.aggregate_wkt(vec![vec![None, None]]).unwrap(), None);
    }

    #[test]
    fn item_crs() {
        let tester =
            AggregateUdfTester::new(create_udf().into(), vec![WKB_GEOMETRY_ITEM_CRS.clone()]);
        assert_eq!(tester.return_type().unwrap(), WKB_GEOMETRY_ITEM_CRS.clone());
    }

    #[rstest]
    fn keeps_z_and_m(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = AggregateUdfTester::new(create_udf().into(), vec![sedona_type]);

        assert_union_equals(
            &tester
                .aggregate_wkt(vec![vec![Some("POINT Z (1 2 3)")]])
                .unwrap(),
            Some("POINT Z (1 2 3)"),
        );
        assert_union_equals(
            &tester
                .aggregate_wkt(vec![
                    vec![Some("POINT ZM (1 2 3 4)")],
                    vec![Some("POINT ZM (5 6 7 8)")],
                ])
                .unwrap(),
            Some("MULTIPOINT ZM ((1 2 3 4), (5 6 7 8))"),
        );
        assert_union_equals(
            &tester
                .aggregate_wkt(vec![vec![Some("POINT M (1 2 4)")]])
                .unwrap(),
            Some("POINT M (1 2 4)"),
        );
        assert_union_equals(
            &tester
                .aggregate_wkt(vec![vec![
                    Some("POLYGON Z ((0 0 1, 1 0 1, 1 1 1, 0 1 1, 0 0 1))"),
                    Some("POLYGON Z ((2 0 2, 3 0 2, 3 1 2, 2 1 2, 2 0 2))"),
                ]])
                .unwrap(),
            Some("MULTIPOLYGON Z (((0 0 1, 1 0 1, 1 1 1, 0 1 1, 0 0 1)), ((2 0 2, 3 0 2, 3 1 2, 2 1 2, 2 0 2)))"),
        );
    }

    #[rstest]
    fn mixed_dimensions(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = AggregateUdfTester::new(create_udf().into(), vec![sedona_type]);

        for batches in [
            vec![vec![Some("POINT Z (1 2 3)"), Some("POINT (5 6)")]],
            vec![
                vec![Some("POINT Z (1 2 3)")],
                vec![Some("POINT ZM (5 6 7 8)")],
            ],
        ] {
            let err = tester.aggregate_wkt(batches).unwrap_err();
            assert!(
                err.message().contains("mixed dimension geometries"),
                "unexpected error: {err}"
            );
        }
    }

    #[rstest]
    fn empty_input(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester = AggregateUdfTester::new(create_udf().into(), vec![sedona_type]);

        for (batches, expected) in [
            (vec![vec![Some("POINT ZM EMPTY")]], "POINT EMPTY"),
            (
                vec![vec![Some("POLYGON Z EMPTY")], vec![Some("POINT Z EMPTY")]],
                "POLYGON EMPTY",
            ),
            (
                vec![vec![Some("MULTIPOINT EMPTY"), Some("LINESTRING EMPTY")]],
                "MULTIPOINT EMPTY",
            ),
            (
                vec![
                    vec![Some("POINT EMPTY")],
                    vec![Some("GEOMETRYCOLLECTION EMPTY")],
                ],
                "GEOMETRYCOLLECTION EMPTY",
            ),
            (
                vec![vec![Some("POINT EMPTY")], vec![Some("POINT Z (1 2 3)")]],
                "POINT Z (1 2 3)",
            ),
            (
                vec![vec![Some("POINT ZM EMPTY"), Some("POINT ZM (1 2 3 4)")]],
                "POINT ZM (1 2 3 4)",
            ),
            (
                vec![
                    vec![Some("GEOMETRYCOLLECTION Z (POINT Z EMPTY)")],
                    vec![Some("POINT (1 2)")],
                ],
                "POINT (1 2)",
            ),
            (
                vec![vec![
                    Some("GEOMETRYCOLLECTION (GEOMETRYCOLLECTION (POINT EMPTY))"),
                    Some("POINT Z (1 2 3)"),
                ]],
                "POINT Z (1 2 3)",
            ),
            (
                vec![vec![Some("GEOMETRYCOLLECTION Z (POINT Z EMPTY)")]],
                "GEOMETRYCOLLECTION EMPTY",
            ),
        ] {
            assert_union_equals(&tester.aggregate_wkt(batches).unwrap(), Some(expected));
        }
    }
}
