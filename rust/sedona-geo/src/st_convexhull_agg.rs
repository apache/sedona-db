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

use arrow_array::builder::BinaryBuilder;
use arrow_array::{Array, ArrayRef, BooleanArray};
use arrow_schema::FieldRef;
use datafusion_common::{cast::as_binary_array, error::Result, DataFusionError, ScalarValue};
use datafusion_expr::{Accumulator, ColumnarValue, EmitTo, GroupsAccumulator};
use geo::{algorithm::convex_hull::quick_hull, Coord};
use geo_traits::{Dimensions, GeometryTrait};
use sedona_common::sedona_internal_err;
use sedona_expr::{
    aggregate_udf::{SedonaAccumulator, SedonaAccumulatorRef},
    item_crs::ItemCrsSedonaAccumulator,
};
use sedona_functions::executor::WkbExecutor;
use sedona_geometry::bounds::visit_xy_coords;
use sedona_geometry::wkb_factory::{
    write_wkb_geometrycollection_header, write_wkb_linestring, write_wkb_point, write_wkb_polygon,
};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};
use wkb::reader::read_wkb;

/// ST_ConvexHull_Agg() implementation
pub fn st_convexhull_agg_impl() -> Vec<SedonaAccumulatorRef> {
    ItemCrsSedonaAccumulator::wrap_impl(STConvexHullAgg {})
}

#[derive(Debug)]
struct STConvexHullAgg {}

impl SedonaAccumulator for STConvexHullAgg {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOMETRY);
        matcher.match_args(args)
    }

    fn accumulator(
        &self,
        args: &[SedonaType],
        _output_type: &SedonaType,
    ) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(ConvexHullAccumulator::new(args[0].clone())))
    }

    fn groups_accumulator_supported(&self, _args: &[SedonaType]) -> bool {
        true
    }

    fn groups_accumulator(
        &self,
        args: &[SedonaType],
        _output_type: &SedonaType,
    ) -> Result<Box<dyn GroupsAccumulator>> {
        Ok(Box::new(ConvexHullGroupsAccumulator::new(args[0].clone())))
    }

    fn state_fields(&self, _args: &[SedonaType]) -> Result<Vec<FieldRef>> {
        Ok(vec![Arc::new(WKB_GEOMETRY.to_storage_field("hull", true)?)])
    }
}

fn push_hull_coords(geom: impl GeometryTrait<T = f64>, out: &mut Vec<Coord>) -> Result<()> {
    visit_xy_coords(&geom, false, &mut |x, y| out.push((x, y).into()))
        .map_err(|e| DataFusionError::Execution(format!("ST_ConvexHull_Agg(): {e}")))
}

fn filter_keep(filter: Option<&BooleanArray>, i: usize) -> bool {
    filter.is_none_or(|filter| filter.is_valid(i) && filter.value(i))
}

fn normalize_zero(v: f64) -> f64 {
    if v == 0.0 {
        0.0
    } else {
        v
    }
}

fn coord_cmp(a: &Coord, b: &Coord) -> std::cmp::Ordering {
    normalize_zero(a.y)
        .total_cmp(&normalize_zero(b.y))
        .then(normalize_zero(a.x).total_cmp(&normalize_zero(b.x)))
}

fn compute_hull_vertices(coords: &mut [Coord]) -> Vec<Coord> {
    if coords.is_empty() {
        return Vec::new();
    }

    // geo 0.31's quick_hull always returns a closed ring
    let mut vertices = quick_hull(coords).0;
    vertices.pop();

    match vertices.len() {
        0 | 1 => vertices,
        2 => {
            vertices.sort_unstable_by(coord_cmp);
            vertices
        }
        _ => {
            vertices.reverse();
            let start = vertices
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| coord_cmp(a, b))
                .map(|(i, _)| i)
                .unwrap();
            vertices.rotate_left(start);
            vertices.push(vertices[0]);
            vertices
        }
    }
}

fn hull_wkb_size(vertices: &[Coord]) -> usize {
    match vertices.len() {
        0 => 9,
        1 => 21,
        2 => 41,
        n => 13 + n * 16,
    }
}

fn write_hull_vertices(vertices: &[Coord], writer: &mut impl std::io::Write) -> Result<()> {
    match vertices.len() {
        0 => write_wkb_geometrycollection_header(writer, Dimensions::Xy, 0),
        1 => write_wkb_point(writer, (vertices[0].x, vertices[0].y)),
        2 => write_wkb_linestring(writer, vertices.iter().map(|c| (c.x, c.y))),
        _ => write_wkb_polygon(writer, vertices.iter().map(|c| (c.x, c.y))),
    }
    .map_err(|e| DataFusionError::Execution(format!("Failed to write hull: {e}")))
}

fn write_hull(coords: &mut [Coord], writer: &mut impl std::io::Write) -> Result<()> {
    write_hull_vertices(&compute_hull_vertices(coords), writer)
}

fn push_state_coords(hulls: &dyn Array, index: usize, out: &mut Vec<Coord>) -> Result<bool> {
    let hulls = as_binary_array(hulls)?;
    if hulls.is_null(index) {
        return Ok(false);
    }

    let hull = read_wkb(hulls.value(index))
        .map_err(|e| DataFusionError::Execution(format!("Failed to read WKB: {e}")))?;
    push_hull_coords(&hull, out)?;
    Ok(true)
}

#[derive(Debug)]
struct ConvexHullAccumulator {
    input_type: SedonaType,
    coords: Vec<Coord>,
    has_input: bool,
}

impl ConvexHullAccumulator {
    pub fn new(input_type: SedonaType) -> Self {
        Self {
            input_type,
            coords: Vec::new(),
            has_input: false,
        }
    }

    fn make_wkb_result(&mut self) -> Result<Option<Vec<u8>>> {
        if !self.has_input {
            return Ok(None);
        }

        let mut wkb = Vec::new();
        write_hull(&mut self.coords, &mut wkb)?;
        Ok(Some(wkb))
    }
}

impl Accumulator for ConvexHullAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.is_empty() {
            return sedona_internal_err!("No input arrays provided to accumulator in update_batch");
        }

        let arg_types = [self.input_type.clone()];
        let args = [ColumnarValue::Array(values[0].clone())];
        let executor = WkbExecutor::new(&arg_types, &args);
        executor.execute_wkb_void(|maybe_item| {
            if let Some(item) = maybe_item {
                self.has_input = true;
                push_hull_coords(item, &mut self.coords)?;
            }
            Ok(())
        })?;

        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        Ok(ScalarValue::Binary(self.make_wkb_result()?))
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![ScalarValue::Binary(self.make_wkb_result()?)])
    }

    fn size(&self) -> usize {
        size_of::<ConvexHullAccumulator>() + self.coords.capacity() * size_of::<Coord>()
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        if states.len() != 1 {
            return sedona_internal_err!(
                "Unexpected number of state fields for st_convexhull_agg() (expected 1, got {})",
                states.len()
            );
        }

        for i in 0..states[0].len() {
            self.has_input |= push_state_coords(&states[0], i, &mut self.coords)?;
        }

        Ok(())
    }
}

#[derive(Debug)]
struct ConvexHullGroupsAccumulator {
    input_type: SedonaType,
    coords: Vec<Vec<Coord>>,
    has_input: Vec<bool>,
}

impl ConvexHullGroupsAccumulator {
    pub fn new(input_type: SedonaType) -> Self {
        Self {
            input_type,
            coords: Vec::new(),
            has_input: Vec::new(),
        }
    }

    fn execute_update(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        // Check some of our assumptions about how this will be called
        debug_assert_eq!(values.len(), 1);
        debug_assert_eq!(values[0].len(), group_indices.len());
        if let Some(filter) = opt_filter {
            debug_assert_eq!(values[0].len(), filter.len());
        }

        let arg_types = [self.input_type.clone()];
        let args = [ColumnarValue::Array(values[0].clone())];
        let executor = WkbExecutor::new(&arg_types, &args);
        self.coords.resize_with(total_num_groups, Default::default);
        self.has_input.resize(total_num_groups, false);
        let mut i = 0;

        executor.execute_wkb_void(|maybe_item| {
            let keep = filter_keep(opt_filter, i);
            let group_id = group_indices[i];
            i += 1;

            if keep {
                if let Some(item) = maybe_item {
                    self.has_input[group_id] = true;
                    push_hull_coords(item, &mut self.coords[group_id])?;
                }
            }

            Ok(())
        })?;

        Ok(())
    }

    fn emit_wkb_result(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        let emit_size = match emit_to {
            EmitTo::All => self.coords.len(),
            EmitTo::First(n) => n,
        };

        // Draining keeps indices in lockstep with DataFusion's post-emit group renumbering
        let emitted_coords = self.coords.drain(0..emit_size);
        let emitted_has_input = self.has_input.drain(0..emit_size);

        let mut hulls = Vec::with_capacity(emit_size);
        let mut total_bytes = 0usize;
        for (mut group_coords, has_input) in emitted_coords.zip(emitted_has_input) {
            if has_input {
                let vertices = compute_hull_vertices(&mut group_coords);
                total_bytes += hull_wkb_size(&vertices);
                hulls.push(Some(vertices));
            } else {
                hulls.push(None);
            }
        }

        let mut hull_builder = BinaryBuilder::with_capacity(emit_size, total_bytes);
        for hull in hulls {
            match hull {
                Some(vertices) => {
                    write_hull_vertices(&vertices, &mut hull_builder)?;
                    hull_builder.append_value([]);
                }
                None => hull_builder.append_null(),
            }
        }

        Ok(Arc::new(hull_builder.finish()))
    }

    fn merge_state(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        debug_assert_eq!(values.len(), 1);

        self.coords.resize_with(total_num_groups, Default::default);
        self.has_input.resize(total_num_groups, false);

        for (i, &group_id) in group_indices.iter().enumerate() {
            if !filter_keep(opt_filter, i) {
                continue;
            }

            self.has_input[group_id] |=
                push_state_coords(&values[0], i, &mut self.coords[group_id])?;
        }

        Ok(())
    }
}

impl GroupsAccumulator for ConvexHullGroupsAccumulator {
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        self.execute_update(values, group_indices, opt_filter, total_num_groups)
    }

    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        Ok(vec![self.emit_wkb_result(emit_to)?])
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        self.merge_state(values, group_indices, opt_filter, total_num_groups)
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        self.emit_wkb_result(emit_to)
    }

    fn size(&self) -> usize {
        size_of::<ConvexHullGroupsAccumulator>()
            + self.has_input.capacity()
            + self
                .coords
                .iter()
                .map(|coords| coords.capacity() * size_of::<Coord>())
                .sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use arrow_buffer::{BooleanBuffer, NullBuffer};
    use datafusion_expr::AggregateUDF;
    use rstest::rstest;
    use sedona_expr::aggregate_udf::SedonaAggregateUDF;
    use sedona_schema::datatypes::{WKB_GEOMETRY, WKB_GEOMETRY_ITEM_CRS, WKB_VIEW_GEOMETRY};
    use sedona_testing::{
        compare::{assert_array_equal, assert_scalar_equal, assert_scalar_equal_wkb_geometry},
        create::{create_array, create_scalar_item_crs},
        testers::AggregateUdfTester,
    };

    use super::*;

    fn create_udf() -> SedonaAggregateUDF {
        SedonaAggregateUDF::new(
            "st_convexhull_agg",
            st_convexhull_agg_impl(),
            datafusion_expr::Volatility::Immutable,
        )
    }

    #[test]
    fn udf_metadata() {
        let udf = create_udf();
        let aggregate_udf: AggregateUDF = udf.into();
        assert_eq!(aggregate_udf.name(), "st_convexhull_agg");
    }

    #[rstest]
    fn convex_hull_with_nulls(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = create_udf();
        let tester = AggregateUdfTester::new(udf.into(), vec![sedona_type.clone()]);
        assert_eq!(tester.return_type().unwrap(), WKB_GEOMETRY);

        let batches = vec![vec![
            Some("POINT (0 0)"),
            None,
            Some("POINT (1 1)"),
            None,
            Some("POINT (1 0)"),
            Some("POINT (0 1)"),
        ]];

        let result = tester.aggregate_wkt(batches).unwrap();
        assert_scalar_equal_wkb_geometry(&result, Some("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"));
    }

    #[rstest]
    fn convex_hull_multiple_batches(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        // Testing merge_batch
        let udf = create_udf();
        let tester = AggregateUdfTester::new(udf.into(), vec![sedona_type.clone()]);

        let batches = vec![
            vec![Some("POINT (0 0)"), Some("POINT (1 1)")],
            vec![Some("POINT (1 0)")],
            vec![Some("POINT (0 1)")],
        ];

        let result = tester.aggregate_wkt(batches).unwrap();
        assert_scalar_equal_wkb_geometry(&result, Some("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"));
    }

    #[rstest]
    fn convex_hull_empty_input(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = create_udf();
        let tester = AggregateUdfTester::new(udf.into(), vec![sedona_type.clone()]);

        let batches: Vec<Vec<Option<&str>>> = vec![];
        assert_scalar_equal_wkb_geometry(&tester.aggregate_wkt(batches).unwrap(), None);
    }

    #[rstest]
    fn convex_hull_all_coordinates_empty(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let udf = create_udf();
        let tester = AggregateUdfTester::new(udf.into(), vec![sedona_type.clone()]);

        let batches = vec![vec![Some("POINT EMPTY"), Some("MULTIPOINT EMPTY")]];
        assert_scalar_equal_wkb_geometry(
            &tester.aggregate_wkt(batches).unwrap(),
            Some("GEOMETRYCOLLECTION EMPTY"),
        );
    }

    #[rstest]
    fn convex_hull_all_null(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = create_udf();
        let tester = AggregateUdfTester::new(udf.into(), vec![sedona_type.clone()]);

        let batches = vec![vec![None, None]];
        assert_scalar_equal_wkb_geometry(&tester.aggregate_wkt(batches).unwrap(), None);
    }

    #[rstest]
    fn convex_hull_degenerate_point(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let udf = create_udf();
        let tester = AggregateUdfTester::new(udf.into(), vec![sedona_type.clone()]);

        let batches = vec![vec![Some("POINT (6 7)"), Some("POINT (6 7)")]];
        assert_scalar_equal_wkb_geometry(
            &tester.aggregate_wkt(batches).unwrap(),
            Some("POINT (6 7)"),
        );
    }

    #[rstest]
    fn convex_hull_degenerate_collinear(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let udf = create_udf();
        let tester = AggregateUdfTester::new(udf.into(), vec![sedona_type.clone()]);

        let batches = vec![vec![Some("MULTIPOINT (0 1, 2 3, 4 5, 6 7)")]];
        assert_scalar_equal_wkb_geometry(
            &tester.aggregate_wkt(batches).unwrap(),
            Some("LINESTRING (0 1, 6 7)"),
        );
    }

    #[rstest]
    fn convex_hull_line_string_vertices(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let udf = create_udf();
        let tester = AggregateUdfTester::new(udf.into(), vec![sedona_type.clone()]);

        let batches = vec![vec![Some("LINESTRING (0 0, 10 0, 10 10)")]];
        assert_scalar_equal_wkb_geometry(
            &tester.aggregate_wkt(batches).unwrap(),
            Some("POLYGON ((0 0, 10 10, 10 0, 0 0))"),
        );
    }

    #[rstest]
    fn convex_hull_ring_starts_at_lowest_point(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let udf = create_udf();
        let tester = AggregateUdfTester::new(udf.into(), vec![sedona_type.clone()]);

        let batches = vec![vec![Some("MULTIPOINT (0 5, 5 0, 10 5, 5 10)")]];
        assert_scalar_equal_wkb_geometry(
            &tester.aggregate_wkt(batches).unwrap(),
            Some("POLYGON ((5 0, 0 5, 5 10, 10 5, 5 0))"),
        );

        let batches = vec![vec![Some("MULTIPOINT (5 0, 1 0, 3 4)")]];
        assert_scalar_equal_wkb_geometry(
            &tester.aggregate_wkt(batches).unwrap(),
            Some("POLYGON ((1 0, 3 4, 5 0, 1 0))"),
        );
    }

    #[rstest]
    fn convex_hull_polygon_ignores_interiors(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let udf = create_udf();
        let tester = AggregateUdfTester::new(udf.into(), vec![sedona_type.clone()]);

        let batches = vec![vec![Some(
            "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 2 1, 2 2, 1 2, 1 1))",
        )]];
        assert_scalar_equal_wkb_geometry(
            &tester.aggregate_wkt(batches).unwrap(),
            Some("POLYGON ((0 0, 0 4, 4 4, 4 0, 0 0))"),
        );
    }

    #[rstest]
    fn convex_hull_geometry_collection(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let udf = create_udf();
        let tester = AggregateUdfTester::new(udf.into(), vec![sedona_type.clone()]);

        let batches = vec![vec![Some(
            "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (5 0, 5 5))",
        )]];
        assert_scalar_equal_wkb_geometry(
            &tester.aggregate_wkt(batches).unwrap(),
            Some("POLYGON ((0 0, 5 5, 5 0, 0 0))"),
        );
    }

    #[rstest]
    fn convex_hull_consistent_z_drops_z(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let udf = create_udf();
        let tester = AggregateUdfTester::new(udf.into(), vec![sedona_type.clone()]);

        let batches = vec![vec![
            Some("POINT Z (0 0 5)"),
            Some("POINT Z (1 1 9)"),
            Some("POINT Z (1 0 3)"),
            Some("POINT Z (0 1 7)"),
        ]];
        assert_scalar_equal_wkb_geometry(
            &tester.aggregate_wkt(batches).unwrap(),
            Some("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))"),
        );
    }

    #[rstest]
    fn convex_hull_mixed_dimensions_drops_z(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let udf = create_udf();
        let tester = AggregateUdfTester::new(udf.into(), vec![sedona_type.clone()]);

        let batches = vec![vec![Some("POINT (0 0)"), Some("POINT Z (1 1 5)")]];
        assert_scalar_equal_wkb_geometry(
            &tester.aggregate_wkt(batches).unwrap(),
            Some("LINESTRING (0 0, 1 1)"),
        );
    }

    #[test]
    fn udf_invoke_item_crs() {
        let sedona_type = WKB_GEOMETRY_ITEM_CRS.clone();
        let udf = create_udf();
        let tester = AggregateUdfTester::new(udf.into(), vec![sedona_type.clone()]);
        assert_eq!(tester.return_type().unwrap(), sedona_type.clone());

        let batches = vec![vec![Some("POINT (0 1)"), None, Some("POINT (4 5)")]];
        let expected = create_scalar_item_crs(Some("LINESTRING (0 1, 4 5)"), None, &WKB_GEOMETRY);

        assert_scalar_equal(&tester.aggregate_wkt(batches).unwrap(), &expected);
    }

    #[rstest]
    fn udf_grouped_accumulate(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let udf = create_udf();
        let tester = AggregateUdfTester::new(udf.into(), vec![sedona_type.clone()]);
        assert_eq!(tester.return_type().unwrap(), WKB_GEOMETRY);

        // Six elements, four groups, with one all null group and one partially null group
        let group_indices = vec![0, 3, 1, 1, 0, 2];
        let array0 = create_array(
            &[Some("POINT (0 1)"), None, Some("POINT (2 3)")],
            &sedona_type,
        );
        let array1 = create_array(
            &[Some("POINT (4 5)"), None, Some("POINT (6 7)")],
            &sedona_type,
        );
        let batches = vec![array0, array1];

        let expected = create_array(
            &[
                Some("POINT (0 1)"),
                Some("LINESTRING (2 3, 4 5)"),
                Some("POINT (6 7)"),
                None,
            ],
            &WKB_GEOMETRY,
        );
        let result = tester
            .aggregate_groups(&batches, group_indices.clone(), None, vec![])
            .unwrap();
        assert_array_equal(&result, &expected);

        let result = tester
            .aggregate_groups(&batches, group_indices.clone(), None, vec![1, 1, 1, 1])
            .unwrap();
        assert_array_equal(&result, &expected);

        let filter = vec![false, false, true, true, false, false];
        let expected = create_array(
            &[None, Some("LINESTRING (2 3, 4 5)"), None, None],
            &WKB_GEOMETRY,
        );

        let result = tester
            .aggregate_groups(&batches, group_indices.clone(), Some(&filter), vec![])
            .unwrap();
        assert_array_equal(&result, &expected);
    }

    #[rstest]
    fn udf_grouped_mixed_dimensions_drops_z(
        #[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType,
    ) {
        let udf = create_udf();
        let tester = AggregateUdfTester::new(udf.into(), vec![sedona_type.clone()]);

        let group_indices = vec![0, 0, 1];
        let array0 = create_array(
            &[
                Some("POINT (0 0)"),
                Some("POINT Z (1 1 5)"),
                Some("POINT (5 5)"),
            ],
            &sedona_type,
        );
        let batches = vec![array0];

        let result = tester
            .aggregate_groups(&batches, group_indices, None, vec![])
            .unwrap();
        let expected = create_array(
            &[Some("LINESTRING (0 0, 1 1)"), Some("POINT (5 5)")],
            &WKB_GEOMETRY,
        );
        assert_array_equal(&result, &expected);
    }

    #[test]
    fn emit_first_then_update_does_not_corrupt_remaining_groups() {
        // Regression test for the EmitTo::First(n) offset bug
        let mut acc = ConvexHullGroupsAccumulator::new(WKB_GEOMETRY);

        let batch1 = create_array(&[Some("POINT (0 0)"), Some("POINT (10 10)")], &WKB_GEOMETRY);
        acc.update_batch(&[batch1], &[0, 1], None, 2).unwrap();

        let emitted_first = acc.state(EmitTo::First(1)).unwrap();
        assert_array_equal(
            &emitted_first[0],
            &create_array(&[Some("POINT (0 0)")], &WKB_GEOMETRY),
        );

        // old group 1 is now group 0, group 1 here is brand new
        let batch2 = create_array(
            &[Some("POINT (11 11)"), Some("POINT (99 99)")],
            &WKB_GEOMETRY,
        );
        acc.update_batch(&[batch2], &[0, 1], None, 2).unwrap();

        let result = acc.evaluate(EmitTo::All).unwrap();
        let expected = create_array(
            &[Some("LINESTRING (10 10, 11 11)"), Some("POINT (99 99)")],
            &WKB_GEOMETRY,
        );
        assert_array_equal(&result, &expected);
    }

    #[test]
    fn execute_update_ignores_stale_true_bit_at_null_filter_slot() {
        // Regression test for reading the filter's raw value bit without checking validity
        let mut acc = ConvexHullGroupsAccumulator::new(WKB_GEOMETRY);

        let values = BooleanBuffer::from(vec![true, true, true]);
        let nulls = NullBuffer::from(vec![true, false, true]);
        let filter = BooleanArray::new(values, Some(nulls));

        let batch = create_array(
            &[
                Some("POINT (0 0)"),
                Some("POINT (99 99)"),
                Some("POINT (5 5)"),
            ],
            &WKB_GEOMETRY,
        );
        acc.update_batch(&[batch], &[0, 0, 0], Some(&filter), 1)
            .unwrap();

        let result = acc.evaluate(EmitTo::All).unwrap();
        let expected = create_array(&[Some("LINESTRING (0 0, 5 5)")], &WKB_GEOMETRY);
        assert_array_equal(&result, &expected);
    }

    #[test]
    fn coord_cmp_treats_negative_zero_as_equal_to_zero() {
        let ordering = coord_cmp(&Coord { x: -0.0, y: 5.0 }, &Coord { x: 0.0, y: 5.0 });
        assert_eq!(ordering, std::cmp::Ordering::Equal);

        let ordering = coord_cmp(&Coord { x: 1.0, y: -0.0 }, &Coord { x: 1.0, y: 0.0 });
        assert_eq!(ordering, std::cmp::Ordering::Equal);
    }

    #[test]
    fn hull_wkb_size_matches_actual_bytes_written() {
        let empty: Vec<Coord> = vec![];
        let point = vec![Coord { x: 1.0, y: 2.0 }];
        let line = vec![Coord { x: 1.0, y: 2.0 }, Coord { x: 3.0, y: 4.0 }];
        let polygon = vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 1.0, y: 0.0 },
            Coord { x: 1.0, y: 1.0 },
            Coord { x: 0.0, y: 0.0 },
        ];

        for vertices in [empty, point, line, polygon] {
            let mut buf = Vec::new();
            write_hull_vertices(&vertices, &mut buf).unwrap();
            assert_eq!(hull_wkb_size(&vertices), buf.len());
        }
    }
}
