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

//! `RS_Stack_Aggr` — the bands of a column of rasters, ordered by an index
//! column, as one raster.
//!
//! ```text
//! RS_Stack_Aggr(raster, index)  -> Raster
//! ```
//!
//! Sedona Spark 1.9 calls this aggregate `RS_Union_Aggr`. It is the aggregate
//! form of [`RS_Stack`](crate::rs_stack): every band of the raster with the
//! lowest index, then every band of the next, and so on, under the header of
//! the lowest-index raster. The rasters must all be on one grid and their bands
//! are carried over as they are, exactly as `RS_Stack` does.
//!
//! The index orders the rasters, since an aggregate sees its rows in no
//! particular order. As in Sedona Spark, the indexes must be distinct and
//! evenly spaced (1, 2, 3 or 0, 10, 20): a gap usually means a missing raster,
//! which would silently shift every later band. Rows with a NULL raster or a
//! NULL index are skipped, and a group with no other rows gives NULL.

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BooleanArray, Int64Array, ListArray, StructArray, new_empty_array,
};
use arrow_buffer::OffsetBuffer;
use arrow_schema::{DataType, Field, FieldRef};
use datafusion_common::arrow::compute::{cast, concat, filter};
use datafusion_common::cast::as_list_array;
use datafusion_common::{Result, ScalarValue, exec_err, internal_datafusion_err};
use datafusion_expr::{Accumulator, Volatility, utils::AggregateOrderSensitivity};
use sedona_common::sedona_internal_err;
use sedona_expr::aggregate_udf::{SedonaAccumulator, SedonaAggregateUDF};
use sedona_raster::array::RasterStructArray;
use sedona_raster::builder::RasterBuilder;
use sedona_raster::traits::RasterRef;
use sedona_schema::datatypes::SedonaType;
use sedona_schema::matchers::ArgMatcher;

use crate::rs_stack::stack;

const FUNC: &str = "RS_Stack_Aggr";

/// `RS_Stack_Aggr()` aggregate UDF — the bands of a column of rasters as one
/// raster, ordered by an index column.
pub fn rs_stack_aggr_udf() -> SedonaAggregateUDF {
    SedonaAggregateUDF::new(
        "rs_stack_aggr",
        vec![Arc::new(RsStackAggr)],
        Volatility::Immutable,
    )
    // The index, not the row order, decides the band order.
    .with_order_sensitivity(AggregateOrderSensitivity::Insensitive)
}

#[derive(Debug)]
struct RsStackAggr;

impl SedonaAccumulator for RsStackAggr {
    fn return_type(&self, args: &[SedonaType]) -> Result<Option<SedonaType>> {
        ArgMatcher::new(
            vec![ArgMatcher::is_raster(), ArgMatcher::is_integer()],
            SedonaType::Raster,
        )
        .match_args(args)
    }

    fn accumulator(
        &self,
        _args: &[SedonaType],
        _output_type: &SedonaType,
    ) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(StackAccumulator::default()))
    }

    fn state_fields(&self, _args: &[SedonaType]) -> Result<Vec<FieldRef>> {
        Ok(vec![
            Arc::new(Field::new("rasters", DataType::List(raster_item()?), true)),
            Arc::new(Field::new("indexes", DataType::List(index_item()), true)),
        ])
    }
}

/// The element field of the state's raster list
fn raster_item() -> Result<FieldRef> {
    Ok(Arc::new(SedonaType::Raster.to_storage_field("item", true)?))
}

/// The element field of the state's index list
fn index_item() -> FieldRef {
    Arc::new(Field::new("item", DataType::Int64, true))
}

/// Collects the non-NULL (raster, index) rows of a group, as the arrays they
/// arrived in, and stacks them by index on evaluation.
#[derive(Debug, Default)]
struct StackAccumulator {
    rasters: Vec<ArrayRef>,
    indexes: Vec<ArrayRef>,
}

impl StackAccumulator {
    fn push(&mut self, rasters: ArrayRef, indexes: ArrayRef) {
        if !rasters.is_empty() {
            self.rasters.push(rasters);
            self.indexes.push(indexes);
        }
    }

    /// Every collected raster as one array, with its indexes
    fn collected(&self) -> Result<(ArrayRef, ArrayRef)> {
        let rasters = if self.rasters.is_empty() {
            new_empty_array(SedonaType::Raster.storage_type())
        } else {
            let parts: Vec<&dyn Array> = self.rasters.iter().map(|a| a.as_ref()).collect();
            concat(&parts)?
        };
        let indexes = if self.indexes.is_empty() {
            new_empty_array(&DataType::Int64)
        } else {
            let parts: Vec<&dyn Array> = self.indexes.iter().map(|a| a.as_ref()).collect();
            concat(&parts)?
        };
        Ok((rasters, indexes))
    }
}

impl Accumulator for StackAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let [rasters, indexes] = values else {
            return sedona_internal_err!("{FUNC} expects two arguments, got {}", values.len());
        };
        let indexes = cast(indexes, &DataType::Int64)?;
        // Skip a row whose raster or index is NULL, as a two-argument
        // aggregate such as CORR skips a row with either argument NULL.
        let keep = BooleanArray::from_iter(
            (0..rasters.len()).map(|i| Some(rasters.is_valid(i) && indexes.is_valid(i))),
        );
        self.push(filter(rasters, &keep)?, filter(&indexes, &keep)?);
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let (rasters, indexes) = self.collected()?;
        if rasters.is_empty() {
            return ScalarValue::try_from(SedonaType::Raster.storage_type());
        }
        let indexes = indexes
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| internal_datafusion_err!("Expected Int64Array for {FUNC} indexes"))?;
        let rasters = rasters
            .as_any()
            .downcast_ref::<StructArray>()
            .ok_or_else(|| internal_datafusion_err!("Expected StructArray for raster"))?;
        let rasters = RasterStructArray::try_new(rasters)?;

        let mut order: Vec<usize> = (0..indexes.len()).collect();
        order.sort_by_key(|&i| indexes.value(i));
        let sorted: Vec<i64> = order.iter().map(|&i| indexes.value(i)).collect();
        check_indexes(&sorted)?;

        let refs = order
            .iter()
            .map(|&i| rasters.get(i))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let refs: Vec<&dyn RasterRef> = refs.iter().map(|r| r as &dyn RasterRef).collect();
        let mut builder = RasterBuilder::new(1);
        stack(FUNC, &mut builder, &refs, &|k| {
            format!("the raster with index {}", sorted[k])
        })?;
        ScalarValue::try_from_array(&builder.finish()?, 0)
    }

    fn size(&self) -> usize {
        size_of_val(self)
            + self
                .rasters
                .iter()
                .chain(&self.indexes)
                .map(|a| a.get_array_memory_size())
                .sum::<usize>()
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let (rasters, indexes) = self.collected()?;
        let one_list = |item: FieldRef, values: ArrayRef| {
            let offsets = OffsetBuffer::from_lengths([values.len()]);
            ScalarValue::List(Arc::new(ListArray::new(item, offsets, values, None)))
        };
        Ok(vec![
            one_list(raster_item()?, rasters),
            one_list(index_item(), indexes),
        ])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let [rasters, indexes] = states else {
            return sedona_internal_err!("{FUNC} expects two state fields, got {}", states.len());
        };
        let rasters = as_list_array(rasters)?;
        let indexes = as_list_array(indexes)?;
        for i in 0..rasters.len() {
            if rasters.is_valid(i) {
                self.push(rasters.value(i), indexes.value(i));
            }
        }
        Ok(())
    }
}

/// Error unless the sorted `indexes` are distinct and evenly spaced
fn check_indexes(indexes: &[i64]) -> Result<()> {
    let steps: Vec<(i64, i64)> = indexes.windows(2).map(|w| (w[0], w[1])).collect();
    if let Some((index, _)) = steps.iter().find(|(a, b)| a == b) {
        return exec_err!(
            "{FUNC}: index {index} is given to more than one raster; each raster needs its own \
             index"
        );
    }
    let step = |(a, b): (i64, i64)| i128::from(b) - i128::from(a);
    if let Some(&first) = steps.first()
        && let Some(&(a, b)) = steps.iter().find(|&&pair| step(pair) != step(first))
    {
        return exec_err!(
            "{FUNC}: indexes must be evenly spaced, but {a} to {b} is a step of {} where {} to {} \
             is a step of {}; is a raster missing?",
            step((a, b)),
            first.0,
            first.1,
            step(first)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_expr::AggregateUDF;
    use sedona_schema::datatypes::RASTER;
    use sedona_testing::raster_spec::{RasterSpec, assert_raster_scalar_equals, raster_array};
    use sedona_testing::testers::AggregateUdfTester;

    #[test]
    fn udf_metadata() {
        let udf: AggregateUDF = rs_stack_aggr_udf().into();
        assert_eq!(udf.name(), "rs_stack_aggr");
        let tester = AggregateUdfTester::new(udf, vec![RASTER, SedonaType::Arrow(DataType::Int64)]);
        assert_eq!(tester.return_type().unwrap(), SedonaType::Raster);
    }

    #[test]
    fn stacks_bands_in_index_order_across_batches() {
        // Rows arrive out of order and split across batches (and so across
        // partial states); the index alone decides the band order.
        let tester = AggregateUdfTester::new(
            rs_stack_aggr_udf().into(),
            vec![RASTER, SedonaType::Arrow(DataType::Int64)],
        );
        let first = RasterSpec::d2(3, 2).band_values(&[0u8, 1, 2, 3, 4, 5]);
        let second = RasterSpec::d2(3, 2).band_values(&[10u8, 11, 12, 13, 14, 15]);
        let third = RasterSpec::d2(3, 2).band_values(&[20u8, 21, 22, 23, 24, 25]);
        let result = tester
            .aggregate_columns(&[
                vec![
                    Arc::new(raster_array(vec![Some(third), Some(first)])) as ArrayRef,
                    Arc::new(Int64Array::from(vec![3, 1])) as ArrayRef,
                ],
                vec![
                    Arc::new(raster_array(vec![Some(second)])) as ArrayRef,
                    Arc::new(Int64Array::from(vec![2])) as ArrayRef,
                ],
            ])
            .unwrap();
        let expected = RasterSpec::d2(3, 2)
            .band_values(&[0u8, 1, 2, 3, 4, 5])
            .band_values(&[10u8, 11, 12, 13, 14, 15])
            .band_values(&[20u8, 21, 22, 23, 24, 25]);
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn keeps_every_band_of_each_raster() {
        let tester = AggregateUdfTester::new(
            rs_stack_aggr_udf().into(),
            vec![RASTER, SedonaType::Arrow(DataType::Int64)],
        );
        let two_bands = RasterSpec::d2(3, 2)
            .band_values(&[0u8, 1, 2, 3, 4, 5])
            .band_values(&[6u8, 7, 8, 9, 10, 11]);
        let one_band = RasterSpec::d2(3, 2).band_values(&[20u8, 21, 22, 23, 24, 25]);
        let result = tester
            .aggregate_columns(&[vec![
                Arc::new(raster_array(vec![Some(one_band), Some(two_bands)])) as ArrayRef,
                Arc::new(Int64Array::from(vec![1, 0])) as ArrayRef,
            ]])
            .unwrap();
        let expected = RasterSpec::d2(3, 2)
            .band_values(&[0u8, 1, 2, 3, 4, 5])
            .band_values(&[6u8, 7, 8, 9, 10, 11])
            .band_values(&[20u8, 21, 22, 23, 24, 25]);
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn evenly_spaced_indexes_may_step_by_more_than_one() {
        let tester = AggregateUdfTester::new(
            rs_stack_aggr_udf().into(),
            vec![RASTER, SedonaType::Arrow(DataType::Int64)],
        );
        let rasters = vec![
            Some(RasterSpec::d2(3, 2).band_values(&[10u8, 11, 12, 13, 14, 15])),
            Some(RasterSpec::d2(3, 2).band_values(&[0u8, 1, 2, 3, 4, 5])),
            Some(RasterSpec::d2(3, 2).band_values(&[20u8, 21, 22, 23, 24, 25])),
        ];
        let result = tester
            .aggregate_columns(&[vec![
                Arc::new(raster_array(rasters)) as ArrayRef,
                Arc::new(Int64Array::from(vec![10, 0, 20])) as ArrayRef,
            ]])
            .unwrap();
        let expected = RasterSpec::d2(3, 2)
            .band_values(&[0u8, 1, 2, 3, 4, 5])
            .band_values(&[10u8, 11, 12, 13, 14, 15])
            .band_values(&[20u8, 21, 22, 23, 24, 25]);
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn a_null_raster_or_index_skips_the_row() {
        let tester = AggregateUdfTester::new(
            rs_stack_aggr_udf().into(),
            vec![RASTER, SedonaType::Arrow(DataType::Int64)],
        );
        let rasters = vec![
            Some(RasterSpec::d2(3, 2).band_values(&[0u8, 1, 2, 3, 4, 5])),
            None,
            Some(RasterSpec::d2(3, 2).band_values(&[10u8, 11, 12, 13, 14, 15])),
            Some(RasterSpec::d2(3, 2).band_values(&[20u8, 21, 22, 23, 24, 25])),
        ];
        let result = tester
            .aggregate_columns(&[vec![
                Arc::new(raster_array(rasters)) as ArrayRef,
                Arc::new(Int64Array::from(vec![Some(1), Some(2), None, Some(2)])) as ArrayRef,
            ]])
            .unwrap();
        let expected = RasterSpec::d2(3, 2)
            .band_values(&[0u8, 1, 2, 3, 4, 5])
            .band_values(&[20u8, 21, 22, 23, 24, 25]);
        assert_raster_scalar_equals(&result, &expected);
    }

    #[test]
    fn no_rows_gives_null() {
        let tester = AggregateUdfTester::new(
            rs_stack_aggr_udf().into(),
            vec![RASTER, SedonaType::Arrow(DataType::Int64)],
        );
        let rasters = vec![
            None,
            Some(RasterSpec::d2(3, 2).band_values(&[0u8, 1, 2, 3, 4, 5])),
        ];
        let result = tester
            .aggregate_columns(&[vec![
                Arc::new(raster_array(rasters)) as ArrayRef,
                Arc::new(Int64Array::from(vec![Some(1), None])) as ArrayRef,
            ]])
            .unwrap();
        assert!(result.is_null());

        let result = tester
            .aggregate_columns(&[vec![
                Arc::new(raster_array(vec![])) as ArrayRef,
                Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef,
            ]])
            .unwrap();
        assert!(result.is_null());
    }

    #[test]
    fn a_repeated_index_errors() {
        // The two rows meet only when their batches' states are merged.
        let tester = AggregateUdfTester::new(
            rs_stack_aggr_udf().into(),
            vec![RASTER, SedonaType::Arrow(DataType::Int64)],
        );
        let err = tester
            .aggregate_columns(&[
                vec![
                    Arc::new(raster_array(vec![Some(
                        RasterSpec::d2(3, 2).band_values(&[0u8, 1, 2, 3, 4, 5]),
                    )])) as ArrayRef,
                    Arc::new(Int64Array::from(vec![1])) as ArrayRef,
                ],
                vec![
                    Arc::new(raster_array(vec![Some(
                        RasterSpec::d2(3, 2).band_values(&[10u8, 11, 12, 13, 14, 15]),
                    )])) as ArrayRef,
                    Arc::new(Int64Array::from(vec![1])) as ArrayRef,
                ],
            ])
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("index 1 is given to more than one raster"),
            "{err}"
        );
    }

    #[test]
    fn unevenly_spaced_indexes_error() {
        let tester = AggregateUdfTester::new(
            rs_stack_aggr_udf().into(),
            vec![RASTER, SedonaType::Arrow(DataType::Int64)],
        );
        let rasters = vec![
            Some(RasterSpec::d2(3, 2).band_values(&[0u8, 1, 2, 3, 4, 5])),
            Some(RasterSpec::d2(3, 2).band_values(&[10u8, 11, 12, 13, 14, 15])),
            Some(RasterSpec::d2(3, 2).band_values(&[20u8, 21, 22, 23, 24, 25])),
        ];
        let err = tester
            .aggregate_columns(&[vec![
                Arc::new(raster_array(rasters)) as ArrayRef,
                Arc::new(Int64Array::from(vec![1, 2, 4])) as ArrayRef,
            ]])
            .unwrap_err();
        assert!(
            err.to_string().contains(
                "indexes must be evenly spaced, but 2 to 4 is a step of 2 where 1 to 2 is a \
                 step of 1; is a raster missing?"
            ),
            "{err}"
        );
    }

    #[test]
    fn rasters_off_the_lowest_index_grid_error() {
        let tester = AggregateUdfTester::new(
            rs_stack_aggr_udf().into(),
            vec![RASTER, SedonaType::Arrow(DataType::Int64)],
        );
        let moved = vec![
            Some(RasterSpec::d2(3, 2).band_values(&[0u8, 1, 2, 3, 4, 5])),
            Some(
                RasterSpec::d2(3, 2)
                    .transform([5.0, 1.0, 0.0, 0.0, 0.0, -1.0])
                    .band_values(&[10u8, 11, 12, 13, 14, 15]),
            ),
        ];
        let err = tester
            .aggregate_columns(&[vec![
                Arc::new(raster_array(moved)) as ArrayRef,
                Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef,
            ]])
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("RS_Stack_Aggr: the raster with index 2 has geotransform"),
            "{err}"
        );
        assert!(
            err.to_string().contains("but the raster with index 1 has"),
            "{err}"
        );

        let reshaped = vec![
            Some(RasterSpec::d2(3, 2).band_values(&[0u8, 1, 2, 3, 4, 5])),
            Some(RasterSpec::d2(2, 3).band_values(&[0u8; 6])),
        ];
        let err = tester
            .aggregate_columns(&[vec![
                Arc::new(raster_array(reshaped)) as ArrayRef,
                Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef,
            ]])
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("the raster with index 2 is 2 x 3, but the raster with index 1 is 3 x 2"),
            "{err}"
        );
    }

    #[test]
    fn indexes_far_apart_do_not_overflow() {
        let tester = AggregateUdfTester::new(
            rs_stack_aggr_udf().into(),
            vec![RASTER, SedonaType::Arrow(DataType::Int64)],
        );
        let rasters = vec![
            Some(RasterSpec::d2(3, 2).band_values(&[0u8, 1, 2, 3, 4, 5])),
            Some(RasterSpec::d2(3, 2).band_values(&[10u8, 11, 12, 13, 14, 15])),
            Some(RasterSpec::d2(3, 2).band_values(&[20u8, 21, 22, 23, 24, 25])),
        ];
        let err = tester
            .aggregate_columns(&[vec![
                Arc::new(raster_array(rasters)) as ArrayRef,
                Arc::new(Int64Array::from(vec![i64::MIN, 0, i64::MAX])) as ArrayRef,
            ]])
            .unwrap_err();
        assert!(err.to_string().contains("evenly spaced"), "{err}");
    }
}
