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

use std::iter::zip;
use std::sync::Arc;

use arrow_array::builder::BinaryBuilder;
use arrow_array::{Array, ArrayRef, BooleanArray, Float64Array};
use arrow_schema::{DataType, Field, FieldRef};
use datafusion_common::{error::Result, exec_datafusion_err, DataFusionError, ScalarValue};
use datafusion_expr::{Accumulator, ColumnarValue, EmitTo, GroupsAccumulator};
use sedona_common::sedona_internal_err;
use sedona_expr::{
    aggregate_udf::{SedonaAccumulator, SedonaAccumulatorRef},
    item_crs::ItemCrsSedonaAccumulator,
};
use sedona_functions::executor::WkbBytesExecutor;
use sedona_geometry::interval::{Interval, IntervalTrait, WraparoundInterval};
use sedona_geometry::wkb_factory::{
    write_wkb_linestring, write_wkb_multipolygon, write_wkb_point, write_wkb_polygon,
    WKB_MIN_PROBABLE_BYTES,
};
use sedona_schema::{datatypes::SedonaType, matchers::ArgMatcher};

use crate::geography::{Geography, GeographyFactory};
use crate::rect_bounder::RectBounder;

// Re-export WKB_GEOMETRY for use in state
use sedona_schema::datatypes::WKB_GEOMETRY;

/// ST_Envelope_Agg() aggregate implementation for geography types
///
/// This uses geodesic calculations via S2 to compute the bounding rectangle,
/// which may return a MULTIPOLYGON for geographies that wrap around the antimeridian.
pub fn st_envelope_agg_impl() -> Vec<SedonaAccumulatorRef> {
    ItemCrsSedonaAccumulator::wrap_impl(vec![Arc::new(STEnvelopeAgg {}) as SedonaAccumulatorRef])
}

#[derive(Debug)]
struct STEnvelopeAgg {}

impl SedonaAccumulator for STEnvelopeAgg {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        // ST_Envelope_Agg() always returns geometry, even for geography input
        let matcher = ArgMatcher::new(vec![ArgMatcher::is_geography()], WKB_GEOMETRY);
        matcher.match_args(args)
    }

    fn groups_accumulator_supported(&self, _args: &[SedonaType]) -> bool {
        true
    }

    fn groups_accumulator(
        &self,
        args: &[SedonaType],
        _output_type: &SedonaType,
    ) -> Result<Box<dyn GroupsAccumulator>> {
        Ok(Box::new(BoundsGroupsAccumulator::new(args[0].clone())))
    }

    fn accumulator(
        &self,
        args: &[SedonaType],
        _output_type: &SedonaType,
    ) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(BoundsAccumulator::new(args[0].clone())))
    }

    fn state_fields(&self, _args: &[SedonaType]) -> Result<Vec<FieldRef>> {
        // State is stored as 4 Float64 values: xmin, ymin, xmax, ymax
        // This allows proper interval merging without re-computing geodesic bounds
        Ok(vec![
            Arc::new(Field::new("xmin", DataType::Float64, true)),
            Arc::new(Field::new("ymin", DataType::Float64, true)),
            Arc::new(Field::new("xmax", DataType::Float64, true)),
            Arc::new(Field::new("ymax", DataType::Float64, true)),
        ])
    }
}

/// Write the envelope WKB for the given intervals
///
/// Returns true if the envelope was written, false if the intervals are empty.
fn write_envelope_from_intervals(
    x: &WraparoundInterval,
    y: &Interval,
    out: &mut impl std::io::Write,
) -> Result<bool> {
    if x.is_empty() || y.is_empty() {
        return Ok(false);
    }

    // Check for wraparound case (lo > hi means crossing the antimeridian)
    if x.is_wraparound() {
        // Wraparound case: MULTIPOLYGON with two polygons
        let poly1 = vec![
            (x.lo(), y.lo()),
            (x.lo(), y.hi()),
            (180.0, y.hi()),
            (180.0, y.lo()),
            (x.lo(), y.lo()),
        ];
        let poly2 = vec![
            (-180.0, y.lo()),
            (-180.0, y.hi()),
            (x.hi(), y.hi()),
            (x.hi(), y.lo()),
            (-180.0, y.lo()),
        ];
        write_wkb_multipolygon(out, [poly1, poly2].into_iter())
            .map_err(|e| DataFusionError::External(e.into()))?;
    } else {
        // Check for degenerate cases
        match (x.width() > 0.0, y.width() > 0.0) {
            // Full polygon
            (true, true) => {
                write_wkb_polygon(
                    out,
                    [
                        (x.lo(), y.lo()),
                        (x.lo(), y.hi()),
                        (x.hi(), y.hi()),
                        (x.hi(), y.lo()),
                        (x.lo(), y.lo()),
                    ]
                    .into_iter(),
                )
                .map_err(|e| DataFusionError::External(e.into()))?;
            }
            // Vertical or horizontal line
            (false, true) | (true, false) => {
                write_wkb_linestring(out, [(x.lo(), y.lo()), (x.hi(), y.hi())].into_iter())
                    .map_err(|e| DataFusionError::External(e.into()))?;
            }
            // Point
            (false, false) => {
                write_wkb_point(out, (x.lo(), y.lo()))
                    .map_err(|e| DataFusionError::External(e.into()))?;
            }
        }
    }
    Ok(true)
}

#[derive(Debug)]
struct BoundsAccumulator {
    input_type: SedonaType,
    x: WraparoundInterval,
    y: Interval,
}

impl BoundsAccumulator {
    pub fn new(input_type: SedonaType) -> Self {
        Self {
            input_type,
            x: WraparoundInterval::empty(),
            y: Interval::empty(),
        }
    }

    fn make_wkb_result(&self) -> Result<Option<Vec<u8>>> {
        let mut wkb = Vec::new();
        if write_envelope_from_intervals(&self.x, &self.y, &mut wkb)? {
            Ok(Some(wkb))
        } else {
            Ok(None)
        }
    }

    fn check_update_input_len(input: &[ArrayRef], expected: usize, context: &str) -> Result<()> {
        if input.is_empty() {
            return sedona_internal_err!("No input arrays provided to accumulator in {context}");
        }
        if input.len() != expected {
            return sedona_internal_err!(
                "Unexpected input length in {} (expected {}, got {})",
                context,
                expected,
                input.len()
            );
        }
        Ok(())
    }

    /// Update from geography WKB values using RectBounder for geodesic bounds
    fn execute_update_geography(&mut self, args: &[ColumnarValue]) -> Result<()> {
        let arg_types = [self.input_type.clone()];
        let executor = WkbBytesExecutor::new(&arg_types, args);

        // Create factory, geog, and bounder for this update batch
        let mut factory = GeographyFactory::new();
        let mut geog = Geography::new();
        let mut bounder = RectBounder::new();

        executor.execute_wkb_void(|maybe_wkb| {
            if let Some(wkb) = maybe_wkb {
                factory
                    .init_from_wkb(wkb, &mut geog)
                    .map_err(|e| exec_datafusion_err!("Error parsing geography: {e}"))?;

                // Clear and compute bounds for this single geography
                bounder.clear();
                bounder
                    .bound(&geog)
                    .map_err(|e| exec_datafusion_err!("Error bounding geography: {e}"))?;

                // Get bounds and merge into our intervals
                if let Some((xmin, ymin, xmax, ymax)) = bounder
                    .finish()
                    .map_err(|e| exec_datafusion_err!("Error finishing bounds: {e}"))?
                {
                    let new_x = WraparoundInterval::new(xmin, xmax);
                    let new_y = Interval::new(ymin, ymax);
                    self.x = self.x.merge_interval(&new_x);
                    self.y = self.y.merge_interval(&new_y);
                }
            }
            Ok(())
        })?;

        Ok(())
    }
}

impl Accumulator for BoundsAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        Self::check_update_input_len(values, 1, "update_batch")?;
        let args = [ColumnarValue::Array(values[0].clone())];
        self.execute_update_geography(&args)
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let wkb = self.make_wkb_result()?;
        Ok(ScalarValue::Binary(wkb))
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        // Return 4 Float64 values: xmin, ymin, xmax, ymax
        // Use None if the interval is empty
        if self.x.is_empty() || self.y.is_empty() {
            Ok(vec![
                ScalarValue::Float64(None),
                ScalarValue::Float64(None),
                ScalarValue::Float64(None),
                ScalarValue::Float64(None),
            ])
        } else {
            Ok(vec![
                ScalarValue::Float64(Some(self.x.lo())),
                ScalarValue::Float64(Some(self.y.lo())),
                ScalarValue::Float64(Some(self.x.hi())),
                ScalarValue::Float64(Some(self.y.hi())),
            ])
        }
    }

    fn size(&self) -> usize {
        std::mem::size_of::<BoundsAccumulator>()
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        Self::check_update_input_len(states, 4, "merge_batch")?;

        // States are 4 Float64 arrays: xmin, ymin, xmax, ymax
        let xmin_arr = states[0]
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| exec_datafusion_err!("Expected Float64Array for xmin state"))?;
        let ymin_arr = states[1]
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| exec_datafusion_err!("Expected Float64Array for ymin state"))?;
        let xmax_arr = states[2]
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| exec_datafusion_err!("Expected Float64Array for xmax state"))?;
        let ymax_arr = states[3]
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| exec_datafusion_err!("Expected Float64Array for ymax state"))?;

        for i in 0..xmin_arr.len() {
            if !xmin_arr.is_null(i) {
                let xmin = xmin_arr.value(i);
                let ymin = ymin_arr.value(i);
                let xmax = xmax_arr.value(i);
                let ymax = ymax_arr.value(i);

                let new_x = WraparoundInterval::new(xmin, xmax);
                let new_y = Interval::new(ymin, ymax);
                self.x = self.x.merge_interval(&new_x);
                self.y = self.y.merge_interval(&new_y);
            }
        }

        Ok(())
    }
}

#[derive(Debug)]
struct BoundsGroupsAccumulator {
    input_type: SedonaType,
    xs: Vec<WraparoundInterval>,
    ys: Vec<Interval>,
    offset: usize,
}

impl BoundsGroupsAccumulator {
    pub fn new(input_type: SedonaType) -> Self {
        Self {
            input_type,
            xs: Vec::new(),
            ys: Vec::new(),
            offset: 0,
        }
    }

    /// Update from geography WKB values
    fn execute_update_geography(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        debug_assert_eq!(self.offset, 0);
        debug_assert_eq!(values.len(), 1);
        debug_assert_eq!(values[0].len(), group_indices.len());
        if let Some(filter) = opt_filter {
            debug_assert_eq!(values[0].len(), filter.len());
        }

        // Ensure we have enough intervals
        self.xs
            .resize(total_num_groups, WraparoundInterval::empty());
        self.ys.resize(total_num_groups, Interval::empty());

        let arg_types = [self.input_type.clone()];
        let args = [ColumnarValue::Array(values[0].clone())];
        let executor = WkbBytesExecutor::new(&arg_types, &args);

        let mut factory = GeographyFactory::new();
        let mut geog = Geography::new();
        let mut bounder = RectBounder::new();
        let mut i = 0;

        if let Some(filter) = opt_filter {
            let mut filter_iter = filter.iter();
            executor.execute_wkb_void(|maybe_wkb| {
                let should_include = filter_iter.next().unwrap().unwrap_or(false);
                if should_include {
                    let group_id = group_indices[i];
                    if let Some(wkb) = maybe_wkb {
                        factory
                            .init_from_wkb(wkb, &mut geog)
                            .map_err(|e| exec_datafusion_err!("Error parsing geography: {e}"))?;

                        bounder.clear();
                        bounder
                            .bound(&geog)
                            .map_err(|e| exec_datafusion_err!("Error bounding geography: {e}"))?;

                        if let Some((xmin, ymin, xmax, ymax)) = bounder
                            .finish()
                            .map_err(|e| exec_datafusion_err!("Error finishing bounds: {e}"))?
                        {
                            let new_x = WraparoundInterval::new(xmin, xmax);
                            let new_y = Interval::new(ymin, ymax);
                            self.xs[group_id] = self.xs[group_id].merge_interval(&new_x);
                            self.ys[group_id] = self.ys[group_id].merge_interval(&new_y);
                        }
                    }
                }
                i += 1;
                Ok(())
            })?;
        } else {
            executor.execute_wkb_void(|maybe_wkb| {
                let group_id = group_indices[i];
                if let Some(wkb) = maybe_wkb {
                    factory
                        .init_from_wkb(wkb, &mut geog)
                        .map_err(|e| exec_datafusion_err!("Error parsing geography: {e}"))?;

                    bounder.clear();
                    bounder
                        .bound(&geog)
                        .map_err(|e| exec_datafusion_err!("Error bounding geography: {e}"))?;

                    if let Some((xmin, ymin, xmax, ymax)) = bounder
                        .finish()
                        .map_err(|e| exec_datafusion_err!("Error finishing bounds: {e}"))?
                    {
                        let new_x = WraparoundInterval::new(xmin, xmax);
                        let new_y = Interval::new(ymin, ymax);
                        self.xs[group_id] = self.xs[group_id].merge_interval(&new_x);
                        self.ys[group_id] = self.ys[group_id].merge_interval(&new_y);
                    }
                }
                i += 1;
                Ok(())
            })?;
        }

        Ok(())
    }

    /// Merge from state arrays (4 Float64 arrays)
    fn execute_merge_states(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        debug_assert_eq!(self.offset, 0);
        debug_assert_eq!(values.len(), 4);

        // Ensure we have enough intervals
        self.xs
            .resize(total_num_groups, WraparoundInterval::empty());
        self.ys.resize(total_num_groups, Interval::empty());

        let xmin_arr = values[0]
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| exec_datafusion_err!("Expected Float64Array for xmin state"))?;
        let ymin_arr = values[1]
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| exec_datafusion_err!("Expected Float64Array for ymin state"))?;
        let xmax_arr = values[2]
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| exec_datafusion_err!("Expected Float64Array for xmax state"))?;
        let ymax_arr = values[3]
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| exec_datafusion_err!("Expected Float64Array for ymax state"))?;

        for i in 0..xmin_arr.len() {
            let should_include = opt_filter.map(|f| f.value(i)).unwrap_or(true);

            if should_include && !xmin_arr.is_null(i) {
                let group_id = group_indices[i];
                let xmin = xmin_arr.value(i);
                let ymin = ymin_arr.value(i);
                let xmax = xmax_arr.value(i);
                let ymax = ymax_arr.value(i);

                let new_x = WraparoundInterval::new(xmin, xmax);
                let new_y = Interval::new(ymin, ymax);
                self.xs[group_id] = self.xs[group_id].merge_interval(&new_x);
                self.ys[group_id] = self.ys[group_id].merge_interval(&new_y);
            }
        }

        Ok(())
    }

    fn emit_wkb_result(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        let emit_size = match emit_to {
            EmitTo::All => self.xs.len(),
            EmitTo::First(n) => n,
        };

        let mut builder =
            BinaryBuilder::with_capacity(emit_size, emit_size * WKB_MIN_PROBABLE_BYTES);

        let emit_range = self.offset..(self.offset + emit_size);
        for (x, y) in zip(&self.xs[emit_range.clone()], &self.ys[emit_range.clone()]) {
            let written = write_envelope_from_intervals(x, y, &mut builder)?;
            if written {
                builder.append_value([]);
            } else {
                builder.append_null();
            }
        }

        match emit_to {
            EmitTo::All => {
                self.xs.clear();
                self.ys.clear();
                self.offset = 0;
            }
            EmitTo::First(n) => {
                self.offset += n;
            }
        }

        Ok(Arc::new(builder.finish()))
    }

    fn emit_state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        let emit_size = match emit_to {
            EmitTo::All => self.xs.len(),
            EmitTo::First(n) => n,
        };

        let mut xmin_builder = arrow_array::builder::Float64Builder::with_capacity(emit_size);
        let mut ymin_builder = arrow_array::builder::Float64Builder::with_capacity(emit_size);
        let mut xmax_builder = arrow_array::builder::Float64Builder::with_capacity(emit_size);
        let mut ymax_builder = arrow_array::builder::Float64Builder::with_capacity(emit_size);

        let emit_range = self.offset..(self.offset + emit_size);
        for (x, y) in zip(&self.xs[emit_range.clone()], &self.ys[emit_range.clone()]) {
            if x.is_empty() || y.is_empty() {
                xmin_builder.append_null();
                ymin_builder.append_null();
                xmax_builder.append_null();
                ymax_builder.append_null();
            } else {
                xmin_builder.append_value(x.lo());
                ymin_builder.append_value(y.lo());
                xmax_builder.append_value(x.hi());
                ymax_builder.append_value(y.hi());
            }
        }

        match emit_to {
            EmitTo::All => {
                self.xs.clear();
                self.ys.clear();
                self.offset = 0;
            }
            EmitTo::First(n) => {
                self.offset += n;
            }
        }

        Ok(vec![
            Arc::new(xmin_builder.finish()),
            Arc::new(ymin_builder.finish()),
            Arc::new(xmax_builder.finish()),
            Arc::new(ymax_builder.finish()),
        ])
    }
}

impl GroupsAccumulator for BoundsGroupsAccumulator {
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        self.execute_update_geography(values, group_indices, opt_filter, total_num_groups)
    }

    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        self.emit_state(emit_to)
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        self.execute_merge_states(values, group_indices, opt_filter, total_num_groups)
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        self.emit_wkb_result(emit_to)
    }

    fn size(&self) -> usize {
        std::mem::size_of::<BoundsGroupsAccumulator>()
            + self.xs.capacity() * std::mem::size_of::<WraparoundInterval>()
            + self.ys.capacity() * std::mem::size_of::<Interval>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_expr::AggregateUDF;
    use rstest::rstest;
    use sedona_expr::aggregate_udf::SedonaAggregateUDF;
    use sedona_geometry::{bounds::wkb_bounds_xy, interval::IntervalTrait};
    use sedona_schema::datatypes::{
        WKB_GEOGRAPHY, WKB_GEOGRAPHY_ITEM_CRS, WKB_GEOMETRY_ITEM_CRS, WKB_VIEW_GEOGRAPHY,
    };
    use sedona_testing::{create::create_array, testers::AggregateUdfTester};

    fn create_udf() -> SedonaAggregateUDF {
        let impls = st_envelope_agg_impl();
        SedonaAggregateUDF::new(
            "st_envelope_agg",
            impls,
            datafusion_expr::Volatility::Immutable,
        )
    }

    /// Helper to extract WKB bytes from a scalar result
    fn get_wkb_bytes(result: &ScalarValue) -> Option<&[u8]> {
        match result {
            ScalarValue::Binary(Some(bytes)) | ScalarValue::LargeBinary(Some(bytes)) => Some(bytes),
            ScalarValue::Binary(None) | ScalarValue::LargeBinary(None) => None,
            _ => panic!("Expected binary, got {result:?}"),
        }
    }

    /// Helper to assert bounds are approximately equal
    /// Uses a small tolerance to account for geodesic numerical precision
    fn assert_bounds_approx(
        actual_bounds: &sedona_geometry::bounding_box::BoundingBox,
        expected_xmin: f64,
        expected_ymin: f64,
        expected_xmax: f64,
        expected_ymax: f64,
    ) {
        const TOLERANCE: f64 = 1e-10;

        let actual_xmin = actual_bounds.x().lo();
        let actual_ymin = actual_bounds.y().lo();
        let actual_xmax = actual_bounds.x().hi();
        let actual_ymax = actual_bounds.y().hi();

        assert!(
            (actual_xmin - expected_xmin).abs() < TOLERANCE,
            "xmin: expected {expected_xmin}, got {actual_xmin}"
        );
        assert!(
            (actual_ymin - expected_ymin).abs() < TOLERANCE,
            "ymin: expected {expected_ymin}, got {actual_ymin}"
        );
        assert!(
            (actual_xmax - expected_xmax).abs() < TOLERANCE,
            "xmax: expected {expected_xmax}, got {actual_xmax}"
        );
        assert!(
            (actual_ymax - expected_ymax).abs() < TOLERANCE,
            "ymax: expected {expected_ymax}, got {actual_ymax}"
        );
    }

    #[test]
    fn udf_metadata() {
        let udf: AggregateUDF = create_udf().into();
        assert_eq!(udf.name(), "st_envelope_agg");
    }

    #[rstest]
    fn udf_aggregate(#[values(WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY)] sedona_type: SedonaType) {
        let tester = AggregateUdfTester::new(create_udf().into(), vec![sedona_type.clone()]);
        assert_eq!(tester.return_type().unwrap(), WKB_GEOMETRY);

        // Multiple points across batches
        let batches = vec![
            vec![Some("POINT (0 1)"), None, Some("POINT (2 3)")],
            vec![Some("POINT (4 5)"), None, Some("POINT (6 7)")],
        ];
        let result = tester.aggregate_wkt(batches).unwrap();
        let wkb_bytes = get_wkb_bytes(&result).expect("Expected non-null result");
        let bounds = wkb_bounds_xy(wkb_bytes).expect("Failed to get bounds");
        // Each point is bounded individually, then intervals are merged
        // For points, the bounds are exact (no geodesic expansion for a single point)
        assert_bounds_approx(&bounds, 0.0, 1.0, 6.0, 7.0);

        // Empty input
        let result = tester.aggregate_wkt(vec![]).unwrap();
        assert!(get_wkb_bytes(&result).is_none());

        // All null input
        let batches = vec![vec![None, None]];
        let result = tester.aggregate_wkt(batches).unwrap();
        assert!(get_wkb_bytes(&result).is_none());
    }

    #[rstest]
    fn udf_invoke_item_crs(#[values(WKB_GEOGRAPHY_ITEM_CRS.clone())] sedona_type: SedonaType) {
        let tester = AggregateUdfTester::new(create_udf().into(), vec![sedona_type.clone()]);
        // ST_Envelope_Agg returns geometry ItemCrs for geography ItemCrs input
        assert_eq!(tester.return_type().unwrap(), WKB_GEOMETRY_ITEM_CRS.clone());

        let batches = vec![vec![Some("POINT (0 1)"), Some("POINT (2 3)")]];
        let result = tester.aggregate_wkt(batches).unwrap();
        // Check bounds instead of exact WKT comparison
        let wkb_bytes = match &result {
            ScalarValue::Struct(arr) => {
                // For ItemCrs, extract the 'item' field
                let item_arr = arr.column_by_name("item").expect("Missing 'item' field");
                let binary_arr = item_arr
                    .as_any()
                    .downcast_ref::<arrow_array::BinaryArray>()
                    .expect("item should be binary");
                binary_arr.value(0)
            }
            _ => panic!("Expected struct, got {result:?}"),
        };
        let bounds = wkb_bounds_xy(wkb_bytes).expect("Failed to get bounds");
        // Each point is bounded individually, then intervals are merged
        assert_bounds_approx(&bounds, 0.0, 1.0, 2.0, 3.0);
    }

    #[rstest]
    fn udf_grouped_accumulate(
        #[values(WKB_GEOGRAPHY, WKB_VIEW_GEOGRAPHY)] sedona_type: SedonaType,
    ) {
        let tester = AggregateUdfTester::new(create_udf().into(), vec![sedona_type.clone()]);
        assert_eq!(tester.return_type().unwrap(), WKB_GEOMETRY);

        // Six elements, four groups
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

        let result = tester
            .aggregate_groups(&batches, group_indices.clone(), None, vec![])
            .unwrap();

        // Group 0: POINT (0 1), POINT (4 5) -> envelope
        // Group 1: POINT (2 3), POINT (4 5) -> envelope
        // Group 2: POINT (6 7) -> point
        // Group 3: None -> null
        assert_eq!(result.len(), 4);
        assert!(!result.is_null(0)); // Group 0 has data
        assert!(!result.is_null(1)); // Group 1 has data
        assert!(!result.is_null(2)); // Group 2 has data
        assert!(result.is_null(3)); // Group 3 is all null
    }
}
