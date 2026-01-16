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
use std::{iter::zip, sync::Arc, vec};

use crate::executor::WkbExecutor;
use crate::st_envelope::write_envelope;
use arrow_array::{builder::BinaryBuilder, Array, ArrayRef, BooleanArray};
use arrow_schema::FieldRef;
use datafusion_common::{
    error::{DataFusionError, Result},
    ScalarValue,
};
use datafusion_expr::{
    scalar_doc_sections::DOC_SECTION_OTHER, Accumulator, ColumnarValue, Documentation, EmitTo,
    GroupsAccumulator, Volatility,
};
use sedona_common::sedona_internal_err;
use sedona_expr::{
    aggregate_udf::{SedonaAccumulator, SedonaAggregateUDF},
    item_crs::ItemCrsSedonaAccumulator,
};
use sedona_geometry::{
    bounds::geo_traits_update_xy_bounds,
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
        ItemCrsSedonaAccumulator::wrap_impl(vec![Arc::new(STEnvelopeAgg {})]),
        Volatility::Immutable,
        Some(st_envelope_agg_doc()),
    )
}

fn st_envelope_agg_doc() -> Documentation {
    Documentation::builder(
        DOC_SECTION_OTHER,
        "Return the entire envelope boundary of all geometries in geom",
        "ST_Envelope_Agg (geom: Geometry)",
    )
    .with_argument("geom", "geometry: Input geometry or geography")
    .with_sql_example("SELECT ST_Envelope_Agg(ST_GeomFromWKT('MULTIPOINT (0 1, 10 11)'))")
    .build()
}

#[derive(Debug)]
struct STEnvelopeAgg {}

impl SedonaAccumulator for STEnvelopeAgg {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        let matcher = ArgMatcher::new(vec![ArgMatcher::is_geometry()], WKB_GEOMETRY);
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
        Ok(Box::new(BoundsGroupsAccumulator2D::new(args[0].clone())))
    }

    fn accumulator(
        &self,
        args: &[SedonaType],
        _output_type: &SedonaType,
    ) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(BoundsAccumulator2D::new(args[0].clone())))
    }

    fn state_fields(&self, _args: &[SedonaType]) -> Result<Vec<FieldRef>> {
        Ok(vec![Arc::new(
            WKB_GEOMETRY.to_storage_field("envelope", true)?,
        )])
    }
}

#[derive(Debug)]
struct BoundsAccumulator2D {
    input_type: SedonaType,
    x: Interval,
    y: Interval,
}

impl BoundsAccumulator2D {
    pub fn new(input_type: SedonaType) -> Self {
        Self {
            input_type,
            x: Interval::empty(),
            y: Interval::empty(),
        }
    }

    // Create a WKB result based on the current state of the accumulator.
    fn make_wkb_result(&self) -> Result<Option<Vec<u8>>> {
        let mut wkb = Vec::new();
        let written = write_envelope(&self.x.into(), &self.y, &mut wkb)?;
        if written {
            Ok(Some(wkb))
        } else {
            Ok(None)
        }
    }

    // Check the input length for update methods.
    fn check_update_input_len(input: &[ArrayRef], expected: usize, context: &str) -> Result<()> {
        if input.is_empty() {
            return Err(DataFusionError::Internal(format!(
                "No input arrays provided to accumulator in {context}"
            )));
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
    fn execute_update(&mut self, executor: WkbExecutor) -> Result<(), DataFusionError> {
        executor.execute_wkb_void(|maybe_item| {
            if let Some(item) = maybe_item {
                geo_traits_update_xy_bounds(item, &mut self.x, &mut self.y)
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
            }
            Ok(())
        })?;
        Ok(())
    }
}

impl Accumulator for BoundsAccumulator2D {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        Self::check_update_input_len(values, 1, "update_batch")?;
        let arg_types = [self.input_type.clone()];
        let args = [ColumnarValue::Array(values[0].clone())];
        let executor = WkbExecutor::new(&arg_types, &args);
        self.execute_update(executor)?;
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let wkb = self.make_wkb_result()?;
        Ok(ScalarValue::Binary(wkb))
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let wkb = self.make_wkb_result()?;
        Ok(vec![ScalarValue::Binary(wkb)])
    }

    fn size(&self) -> usize {
        size_of::<BoundsAccumulator2D>()
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        Self::check_update_input_len(states, 1, "merge_batch")?;
        let array = &states[0];
        let args = [ColumnarValue::Array(array.clone())];
        let arg_types = [WKB_GEOMETRY.clone()];
        let executor = WkbExecutor::new(&arg_types, &args);
        self.execute_update(executor)?;
        Ok(())
    }
}

#[derive(Debug)]
struct BoundsGroupsAccumulator2D {
    input_type: SedonaType,
    xs: Vec<Interval>,
    ys: Vec<Interval>,
    offset: usize,
}

impl BoundsGroupsAccumulator2D {
    pub fn new(input_type: SedonaType) -> Self {
        Self {
            input_type,
            xs: Vec::new(),
            ys: Vec::new(),
            offset: 0,
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
        debug_assert_eq!(self.offset, 0);
        debug_assert_eq!(values.len(), 1);
        debug_assert_eq!(values[0].len(), group_indices.len());
        if let Some(filter) = opt_filter {
            debug_assert_eq!(values[0].len(), filter.len());
        }

        let arg_types = [self.input_type.clone()];
        let args = [ColumnarValue::Array(values[0].clone())];
        let executor = WkbExecutor::new(&arg_types, &args);
        self.xs.resize(total_num_groups, Interval::empty());
        self.ys.resize(total_num_groups, Interval::empty());
        let mut i = 0;

        if let Some(filter) = opt_filter {
            let mut filter_iter = filter.iter();
            executor.execute_wkb_void(|maybe_item| {
                if filter_iter.next().unwrap().unwrap_or(false) {
                    let group_id = group_indices[i];
                    i += 1;
                    if let Some(item) = maybe_item {
                        geo_traits_update_xy_bounds(
                            item,
                            &mut self.xs[group_id],
                            &mut self.ys[group_id],
                        )
                        .map_err(|e| DataFusionError::External(Box::new(e)))?;
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
                    geo_traits_update_xy_bounds(
                        item,
                        &mut self.xs[group_id],
                        &mut self.ys[group_id],
                    )
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                }

                Ok(())
            })?;
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
            let written = write_envelope(&(*x).into(), y, &mut builder)?;
            if written {
                builder.append_value([]);
            } else {
                builder.append_null();
            }
        }

        match emit_to {
            EmitTo::All => {
                self.xs = Vec::new();
                self.ys = Vec::new();
                self.offset = 0;
            }
            EmitTo::First(n) => {
                self.offset += n;
            }
        }

        Ok(Arc::new(builder.finish()))
    }
}

impl GroupsAccumulator for BoundsGroupsAccumulator2D {
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
        opt_filter: Option<&arrow_array::BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        // In this case, our state is identical to our input values
        self.execute_update(values, group_indices, opt_filter, total_num_groups)
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        self.emit_wkb_result(emit_to)
    }

    fn size(&self) -> usize {
        size_of::<BoundsGroupsAccumulator2D>()
            + self.xs.capacity() * size_of::<Interval>()
            + self.ys.capacity() * size_of::<Interval>()
    }
}

#[cfg(test)]
mod test {
    use datafusion_expr::AggregateUDF;
    use rstest::rstest;
    use sedona_schema::datatypes::{WKB_GEOMETRY_ITEM_CRS, WKB_VIEW_GEOMETRY};
    use sedona_testing::{
        compare::{assert_array_equal, assert_scalar_equal_wkb_geometry},
        create::{create_array, create_scalar},
        testers::AggregateUdfTester,
    };

    use super::*;

    #[test]
    fn udf_metadata() {
        let udf: AggregateUDF = st_envelope_agg_udf().into();
        assert_eq!(udf.name(), "st_envelope_agg");
        assert!(udf.documentation().is_some());
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

    #[test]
    fn udf_invoke_item_crs() {
        let sedona_type = WKB_GEOMETRY_ITEM_CRS.clone();
        let tester =
            AggregateUdfTester::new(st_envelope_agg_udf().into(), vec![sedona_type.clone()]);
        assert_eq!(tester.return_type().unwrap(), sedona_type.clone());

        let batches = vec![
            vec![Some("POINT (0 1)"), None, Some("POINT (2 3)")],
            vec![Some("POINT (4 5)"), None, Some("POINT (6 7)")],
        ];
        let expected = create_scalar(Some("POLYGON((0 1, 0 7, 6 7, 6 1, 0 1))"), &sedona_type);

        assert_scalar_equal(&tester.aggregate_wkt(batches).unwrap(), &expected);
    }

    #[test]
    fn udf_grouped_accumulate() {
        let tester = AggregateUdfTester::new(st_envelope_agg_udf().into(), vec![WKB_GEOMETRY]);
        assert_eq!(tester.return_type().unwrap(), WKB_GEOMETRY);

        // Six elements, four groups, with one all null group and one partially null group
        let group_indices = vec![0, 3, 1, 1, 0, 2];
        let array0 = create_array(
            &[Some("POINT (0 1)"), None, Some("POINT (2 3)")],
            &WKB_GEOMETRY,
        );
        let array1 = create_array(
            &[Some("POINT (4 5)"), None, Some("POINT (6 7)")],
            &WKB_GEOMETRY,
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
