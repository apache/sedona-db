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
use std::{marker::PhantomData, sync::Arc, vec};

use crate::executor::WkbBytesExecutor;
use crate::st_envelope::write_envelope;
use arrow_array::{builder::BinaryBuilder, Array, ArrayRef, BooleanArray};
use arrow_schema::{DataType, Field, FieldRef};
use datafusion_common::exec_datafusion_err;
use datafusion_common::{
    cast::as_float64_array,
    error::{DataFusionError, Result},
    ScalarValue,
};
use datafusion_expr::{Accumulator, ColumnarValue, EmitTo, GroupsAccumulator, Volatility};
use sedona_common::sedona_internal_err;
use sedona_expr::{
    aggregate_udf::{SedonaAccumulator, SedonaAggregateUDF},
    item_crs::ItemCrsSedonaAccumulator,
};
use sedona_geometry::bounds::WkbGeometryBounder;
use sedona_geometry::interval::WraparoundInterval;
use sedona_geometry::{
    bounds::WkbBounder2D,
    interval::{Interval, IntervalTrait},
    wkb_factory::WKB_MIN_PROBABLE_BYTES,
};
use sedona_schema::{
    datatypes::{SedonaType, WKB_GEOMETRY},
    matchers::ArgMatcher,
};

/// ST_Envelope_Agg() aggregate UDF implementation
///
/// An implementation of envelope (bounding shape) calculation.
pub fn st_envelope_agg_udf() -> SedonaAggregateUDF {
    SedonaAggregateUDF::new(
        "st_envelope_agg",
        ItemCrsSedonaAccumulator::wrap_impl(vec![Arc::new(
            STEnvelopeAgg::<WkbGeometryBounder>::new(ArgMatcher::new(
                vec![ArgMatcher::is_geometry()],
                WKB_GEOMETRY,
            )),
        )]),
        Volatility::Immutable,
    )
}

/// Generic ST_Envelope_Agg accumulator that works with any WkbBounder2D implementation
#[derive(Debug)]
pub struct STEnvelopeAgg<T> {
    matcher: ArgMatcher,
    _phantom: PhantomData<T>,
}

impl<T> STEnvelopeAgg<T> {
    /// Create a new STEnvelopeAgg with a specific ArgMatcher
    pub fn new(matcher: ArgMatcher) -> Self {
        Self {
            matcher,
            _phantom: Default::default(),
        }
    }
}

impl<T: WkbBounder2D + Default + std::fmt::Debug + 'static> SedonaAccumulator for STEnvelopeAgg<T> {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        self.matcher.match_args(args)
    }

    fn groups_accumulator_supported(&self, _args: &[SedonaType]) -> bool {
        true
    }

    fn groups_accumulator(
        &self,
        args: &[SedonaType],
        _output_type: &SedonaType,
    ) -> Result<Box<dyn GroupsAccumulator>> {
        Ok(Box::new(BoundsGroupsAccumulator2D::<T>::new(
            args[0].clone(),
        )))
    }

    fn accumulator(
        &self,
        args: &[SedonaType],
        _output_type: &SedonaType,
    ) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(BoundsAccumulator2D::<T>::new(args[0].clone())))
    }

    fn state_fields(&self, _args: &[SedonaType]) -> Result<Vec<FieldRef>> {
        // State is stored as 4 Float64 values: xmin, ymin, xmax, ymax
        Ok(vec![
            Arc::new(Field::new("xmin", DataType::Float64, true)),
            Arc::new(Field::new("ymin", DataType::Float64, true)),
            Arc::new(Field::new("xmax", DataType::Float64, true)),
            Arc::new(Field::new("ymax", DataType::Float64, true)),
        ])
    }
}

/// A generic accumulator for computing 2D bounds using any WkbBounder2D implementation
#[derive(Debug)]
pub struct BoundsAccumulator2D<T: std::fmt::Debug> {
    input_type: SedonaType,
    bounder: T,
}

impl<T: std::fmt::Debug + WkbBounder2D + Default> BoundsAccumulator2D<T> {
    /// Create a new BoundsAccumulator2D with the given input type
    pub fn new(input_type: SedonaType) -> Self {
        Self {
            input_type,
            bounder: T::default(),
        }
    }

    // Create a WKB result based on the current state of the accumulator.
    fn make_wkb_result(&self) -> Result<Option<Vec<u8>>> {
        let mut wkb = Vec::new();
        let (x, y) = self.bounder.finish();
        let written = write_envelope(&x, &y, &mut wkb)?;
        if written {
            Ok(Some(wkb))
        } else {
            Ok(None)
        }
    }

    // Check the input length for update methods.
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

    // Execute the update operation for the accumulator.
    fn execute_update(&mut self, executor: WkbBytesExecutor) -> Result<(), DataFusionError> {
        executor.execute_wkb_void(|maybe_item| {
            if let Some(item) = maybe_item {
                self.bounder
                    .update_wkb_bytes(item)
                    .map_err(|e| exec_datafusion_err!("Error updating bounds: {e}"))?
            }
            Ok(())
        })?;
        Ok(())
    }
}

impl<T: std::fmt::Debug + WkbBounder2D + Default> Accumulator for BoundsAccumulator2D<T> {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        Self::check_update_input_len(values, 1, "update_batch")?;
        let arg_types = [self.input_type.clone()];
        let args = [ColumnarValue::Array(values[0].clone())];
        let executor = WkbBytesExecutor::new(&arg_types, &args);
        self.execute_update(executor)?;
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let wkb = self.make_wkb_result()?;
        Ok(ScalarValue::Binary(wkb))
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        // Return 4 Float64 values: xmin, ymin, xmax, ymax
        let (x, y) = self.bounder.finish();
        Ok(vec![
            ScalarValue::Float64(Some(x.lo())),
            ScalarValue::Float64(Some(y.lo())),
            ScalarValue::Float64(Some(x.hi())),
            ScalarValue::Float64(Some(y.hi())),
        ])
    }

    fn size(&self) -> usize {
        size_of::<BoundsAccumulator2D<T>>()
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        Self::check_update_input_len(states, 4, "merge_batch")?;

        // States are 4 Float64 arrays: xmin, ymin, xmax, ymax
        let xmin_arr = as_float64_array(&states[0])?;
        let ymin_arr = as_float64_array(&states[1])?;
        let xmax_arr = as_float64_array(&states[2])?;
        let ymax_arr = as_float64_array(&states[3])?;

        for i in 0..xmin_arr.len() {
            if !xmin_arr.is_null(i) {
                let xmin = xmin_arr.value(i);
                let ymin = ymin_arr.value(i);
                let xmax = xmax_arr.value(i);
                let ymax = ymax_arr.value(i);

                let new_x = WraparoundInterval::new(xmin, xmax);
                let new_y = Interval::new(ymin, ymax);
                self.bounder
                    .update_bounds(new_x, new_y)
                    .map_err(|e| exec_datafusion_err!("Failed to update bounder: {e}"))?;
            }
        }

        Ok(())
    }
}

#[derive(Debug)]
struct BoundsGroupsAccumulator2D<T: std::fmt::Debug> {
    input_type: SedonaType,
    bounders: Vec<T>,
    offset: usize,
}

impl<T: WkbBounder2D + Default> BoundsGroupsAccumulator2D<T> {
    pub fn new(input_type: SedonaType) -> Self {
        Self {
            input_type,
            bounders: Vec::new(),
            offset: 0,
        }
    }

    fn execute_update(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
        input_type: SedonaType,
    ) -> Result<()> {
        // Check some of our assumptions about how this will be called
        debug_assert_eq!(self.offset, 0);
        debug_assert_eq!(values.len(), 1);
        debug_assert_eq!(values[0].len(), group_indices.len());
        if let Some(filter) = opt_filter {
            debug_assert_eq!(values[0].len(), filter.len());
        }

        let arg_types = [input_type.clone()];
        let args = [ColumnarValue::Array(values[0].clone())];
        let executor = WkbBytesExecutor::new(&arg_types, &args);
        self.bounders
            .resize_with(total_num_groups, Default::default);
        let mut i = 0;

        if let Some(filter) = opt_filter {
            let mut filter_iter = filter.iter();
            executor.execute_wkb_void(|maybe_item| {
                if filter_iter.next().unwrap().unwrap_or(false) {
                    let group_id = group_indices[i];
                    i += 1;
                    if let Some(item) = maybe_item {
                        self.bounders[group_id]
                            .update_wkb_bytes(item)
                            .map_err(|e| exec_datafusion_err!("Error updating bounder: {e}"))?;
                    }
                } else {
                    i += 1;
                }

                Ok(())
            })?;
        } else {
            executor.execute_wkb_void(|maybe_item| {
                let group_id = group_indices[i];
                i += 1;
                if let Some(item) = maybe_item {
                    self.bounders[group_id]
                        .update_wkb_bytes(item)
                        .map_err(|e| exec_datafusion_err!("Error updating bounder: {e}"))?;
                }

                Ok(())
            })?;
        }

        Ok(())
    }

    fn emit_wkb_result(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        let emit_size = match emit_to {
            EmitTo::All => self.bounders.len(),
            EmitTo::First(n) => n,
        };

        let mut builder =
            BinaryBuilder::with_capacity(emit_size, emit_size * WKB_MIN_PROBABLE_BYTES);

        let emit_range = self.offset..(self.offset + emit_size);
        for bounder in &self.bounders[emit_range.clone()] {
            let (x, y) = bounder.finish();
            let written = write_envelope(&x, &y, &mut builder)?;
            if written {
                builder.append_value([]);
            } else {
                builder.append_null();
            }
        }

        match emit_to {
            EmitTo::All => {
                self.bounders = Vec::new();
                self.offset = 0;
            }
            EmitTo::First(n) => {
                self.offset += n;
            }
        }

        Ok(Arc::new(builder.finish()))
    }

    fn emit_state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        use arrow_array::builder::Float64Builder;

        let emit_size = match emit_to {
            EmitTo::All => self.bounders.len(),
            EmitTo::First(n) => n,
        };

        let mut xmin_builder = Float64Builder::with_capacity(emit_size);
        let mut ymin_builder = Float64Builder::with_capacity(emit_size);
        let mut xmax_builder = Float64Builder::with_capacity(emit_size);
        let mut ymax_builder = Float64Builder::with_capacity(emit_size);

        let emit_range = self.offset..(self.offset + emit_size);
        for bounder in &self.bounders[emit_range] {
            let (x, y) = bounder.finish();
            xmin_builder.append_value(x.lo());
            ymin_builder.append_value(y.lo());
            xmax_builder.append_value(x.hi());
            ymax_builder.append_value(y.hi());
        }

        match emit_to {
            EmitTo::All => {
                self.bounders = Vec::new();
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

    fn merge_state(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        debug_assert_eq!(self.offset, 0);
        debug_assert_eq!(values.len(), 4);

        // State is 4 Float64 arrays: xmin, ymin, xmax, ymax
        let xmin_arr = as_float64_array(&values[0])?;
        let ymin_arr = as_float64_array(&values[1])?;
        let xmax_arr = as_float64_array(&values[2])?;
        let ymax_arr = as_float64_array(&values[3])?;

        self.bounders
            .resize_with(total_num_groups, Default::default);

        for (i, &group_id) in group_indices.iter().enumerate() {
            if opt_filter.is_some_and(|f| !f.value(i)) {
                continue;
            }

            if !xmin_arr.is_null(i) {
                let xmin = xmin_arr.value(i);
                let ymin = ymin_arr.value(i);
                let xmax = xmax_arr.value(i);
                let ymax = ymax_arr.value(i);

                self.bounders[group_id]
                    .update_bounds((xmin, xmax).into(), (ymin, ymax).into())
                    .map_err(|e| exec_datafusion_err!("Error updating bounds state: {e}"))?;
            }
        }

        Ok(())
    }
}

impl<T: WkbBounder2D + Default> GroupsAccumulator for BoundsGroupsAccumulator2D<T> {
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        self.execute_update(
            values,
            group_indices,
            opt_filter,
            total_num_groups,
            self.input_type.clone(),
        )
    }

    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        self.emit_state(emit_to)
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&arrow_array::BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        self.merge_state(values, group_indices, opt_filter, total_num_groups)
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        self.emit_wkb_result(emit_to)
    }

    fn size(&self) -> usize {
        size_of::<BoundsGroupsAccumulator2D<T>>()
            + self.bounders.iter().map(|b| b.mem_used()).sum::<usize>()
    }
}

#[cfg(test)]
mod test {
    use datafusion_expr::AggregateUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{
        WKB_GEOMETRY_ITEM_CRS, WKB_VIEW_GEOMETRY, WKB_VIEW_GEOMETRY_ITEM_CRS,
    };
    use sedona_testing::{
        compare::{assert_array_equal, assert_scalar_equal, assert_scalar_equal_wkb_geometry},
        create::{create_array, create_scalar},
        testers::AggregateUdfTester,
    };

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: AggregateUDF = st_envelope_agg_udf().into();
        assert_eq!(udf.name(), "st_envelope_agg");
    }

    #[rstest]
    fn udf(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester =
            AggregateUdfTester::new(st_envelope_agg_udf().into(), vec![sedona_type.clone()]);
        assert_eq!(tester.return_type().unwrap(), WKB_GEOMETRY);

        // Finite input with nulls
        let batches = vec![
            vec![Some("POINT (0 1)"), None, Some("POINT (2 3)")],
            vec![Some("POINT (4 5)"), None, Some("POINT (6 7)")],
        ];
        assert_scalar_equal_wkb_geometry(
            &tester.aggregate_wkt(batches).unwrap(),
            Some("POLYGON((0 1, 0 7, 6 7, 6 1, 0 1))"),
        );

        // Empty input
        assert_scalar_equal_wkb_geometry(&tester.aggregate_wkt(vec![]).unwrap(), None);

        // All coordinates empty
        assert_scalar_equal_wkb_geometry(
            &tester
                .aggregate_wkt(vec![vec![Some("POINT EMPTY")]])
                .unwrap(),
            None,
        );

        // Degenerate output: point
        assert_scalar_equal_wkb_geometry(
            &tester
                .aggregate_wkt(vec![vec![Some("POINT (0 1)")]])
                .unwrap(),
            Some("POINT (0 1)"),
        );

        // Degenerate output: vertical line
        assert_scalar_equal_wkb_geometry(
            &tester
                .aggregate_wkt(vec![vec![Some("MULTIPOINT (0 2, 0 1)")]])
                .unwrap(),
            Some("LINESTRING (0 1, 0 2)"),
        );

        // Degenerate output: horizontal line
        assert_scalar_equal_wkb_geometry(
            &tester
                .aggregate_wkt(vec![vec![Some("MULTIPOINT (1 1, 0 1)")]])
                .unwrap(),
            Some("LINESTRING (0 1, 1 1)"),
        );
    }

    #[rstest]
    fn udf_invoke_item_crs(
        #[values(WKB_GEOMETRY_ITEM_CRS.clone(), WKB_VIEW_GEOMETRY_ITEM_CRS.clone())]
        sedona_type: SedonaType,
    ) {
        let tester =
            AggregateUdfTester::new(st_envelope_agg_udf().into(), vec![sedona_type.clone()]);
        assert_eq!(tester.return_type().unwrap(), WKB_GEOMETRY_ITEM_CRS.clone());

        let batches = vec![
            vec![Some("POINT (0 1)"), None, Some("POINT (2 3)")],
            vec![Some("POINT (4 5)"), None, Some("POINT (6 7)")],
        ];
        let expected = create_scalar(
            Some("POLYGON((0 1, 0 7, 6 7, 6 1, 0 1))"),
            &WKB_GEOMETRY_ITEM_CRS,
        );

        assert_scalar_equal(&tester.aggregate_wkt(batches).unwrap(), &expected);
    }

    #[rstest]
    fn udf_grouped_accumulate(#[values(WKB_GEOMETRY, WKB_VIEW_GEOMETRY)] sedona_type: SedonaType) {
        let tester =
            AggregateUdfTester::new(st_envelope_agg_udf().into(), vec![sedona_type.clone()]);
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
                // First element only + a null
                Some("POINT (0 1)"),
                // Middle two elements
                Some("POLYGON((2 3, 2 5, 4 5, 4 3, 2 3))"),
                // Last element only
                Some("POINT (6 7)"),
                // Only null
                None,
            ],
            &WKB_GEOMETRY,
        );
        let result = tester
            .aggregate_groups(&batches, group_indices.clone(), None, vec![])
            .unwrap();
        assert_array_equal(&result, &expected);

        // We should get the same answer even with a sequence of partial emits
        let result = tester
            .aggregate_groups(&batches, group_indices.clone(), None, vec![1, 1, 1, 1])
            .unwrap();
        assert_array_equal(&result, &expected);

        // Also check with a filter (in this case, filter out all values except
        // the middle two elements).
        let filter = vec![false, false, true, true, false, false];
        let expected = create_array(
            &[None, Some("POLYGON((2 3, 2 5, 4 5, 4 3, 2 3))"), None, None],
            &WKB_GEOMETRY,
        );

        let result = tester
            .aggregate_groups(&batches, group_indices.clone(), Some(&filter), vec![])
            .unwrap();
        assert_array_equal(&result, &expected);
    }
}
