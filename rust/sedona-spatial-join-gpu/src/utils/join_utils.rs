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

/// Most of the code in this module are copied from the `datafusion_physical_plan::joins::utils` module.
/// https://github.com/apache/datafusion/blob/48.0.0/datafusion/physical-plan/src/joins/utils.rs
use std::{ops::Range, sync::Arc};

use arrow::array::{
    downcast_array, new_null_array, Array, BooleanBufferBuilder, RecordBatch, RecordBatchOptions,
    UInt32Builder, UInt64Builder,
};
use arrow::compute;
use arrow::datatypes::{ArrowNativeType, Schema, UInt32Type, UInt64Type};
use arrow_array::{ArrowPrimitiveType, NativeAdapter, PrimitiveArray, UInt32Array, UInt64Array};
use datafusion_common::cast::as_boolean_array;
use datafusion_common::{JoinSide, Result};
use datafusion_expr::JoinType;
use datafusion_physical_expr::Partitioning;
use datafusion_physical_plan::execution_plan::Boundedness;
use datafusion_physical_plan::joins::utils::{
    adjust_right_output_partitioning, ColumnIndex, JoinFilter,
};
use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties};

/// Some type `join_type` of join need to maintain the matched indices bit map for the left side, and
/// use the bit map to generate the part of result of the join.
///
/// For example of the `Left` join, in each iteration of right side, can get the matched result, but need
/// to maintain the matched indices bit map to get the unmatched row for the left side.
pub(crate) fn need_produce_result_in_final(join_type: JoinType) -> bool {
    matches!(
        join_type,
        JoinType::Left
            | JoinType::LeftAnti
            | JoinType::LeftSemi
            | JoinType::LeftMark
            | JoinType::Full
    )
}

/// In the end of join execution, need to use bit map of the matched
/// indices to generate the final left and right indices.
///
/// For example:
///
/// 1. left_bit_map: `[true, false, true, true, false]`
/// 2. join_type: `Left`
///
/// The result is: `([1,4], [null, null])`
pub(crate) fn get_final_indices_from_bit_map(
    left_bit_map: &BooleanBufferBuilder,
    join_type: JoinType,
) -> (UInt64Array, UInt32Array) {
    let left_size = left_bit_map.len();
    if join_type == JoinType::LeftMark {
        let left_indices = (0..left_size as u64).collect::<UInt64Array>();
        let right_indices = (0..left_size)
            .map(|idx| left_bit_map.get_bit(idx).then_some(0))
            .collect::<UInt32Array>();
        return (left_indices, right_indices);
    }
    let left_indices = if join_type == JoinType::LeftSemi {
        (0..left_size)
            .filter_map(|idx| (left_bit_map.get_bit(idx)).then_some(idx as u64))
            .collect::<UInt64Array>()
    } else {
        // just for `Left`, `LeftAnti` and `Full` join
        // `LeftAnti`, `Left` and `Full` will produce the unmatched left row finally
        (0..left_size)
            .filter_map(|idx| (!left_bit_map.get_bit(idx)).then_some(idx as u64))
            .collect::<UInt64Array>()
    };
    // right_indices
    // all the element in the right side is None
    let mut builder = UInt32Builder::with_capacity(left_indices.len());
    builder.append_nulls(left_indices.len());
    let right_indices = builder.finish();
    (left_indices, right_indices)
}

pub(crate) fn apply_join_filter_to_indices(
    build_input_buffer: &RecordBatch,
    probe_batch: &RecordBatch,
    build_indices: UInt64Array,
    probe_indices: UInt32Array,
    filter: &JoinFilter,
    build_side: JoinSide,
) -> Result<(UInt64Array, UInt32Array)> {
    if build_indices.is_empty() && probe_indices.is_empty() {
        return Ok((build_indices, probe_indices));
    };

    let intermediate_batch = build_batch_from_indices(
        filter.schema(),
        build_input_buffer,
        probe_batch,
        &build_indices,
        &probe_indices,
        filter.column_indices(),
        build_side,
    )?;
    let filter_result = filter
        .expression()
        .evaluate(&intermediate_batch)?
        .into_array(intermediate_batch.num_rows())?;
    let mask = as_boolean_array(&filter_result)?;

    let left_filtered = compute::filter(&build_indices, mask)?;
    let right_filtered = compute::filter(&probe_indices, mask)?;
    Ok((
        downcast_array(left_filtered.as_ref()),
        downcast_array(right_filtered.as_ref()),
    ))
}

/// Returns a new [RecordBatch] by combining the `left` and `right` according to `indices`.
/// The resulting batch has [Schema] `schema`.
pub(crate) fn build_batch_from_indices(
    schema: &Schema,
    build_input_buffer: &RecordBatch,
    probe_batch: &RecordBatch,
    build_indices: &UInt64Array,
    probe_indices: &UInt32Array,
    column_indices: &[ColumnIndex],
    build_side: JoinSide,
) -> Result<RecordBatch> {
    if schema.fields().is_empty() {
        let options = RecordBatchOptions::new()
            .with_match_field_names(true)
            .with_row_count(Some(build_indices.len()));

        return Ok(RecordBatch::try_new_with_options(
            Arc::new(schema.clone()),
            vec![],
            &options,
        )?);
    }

    // build the columns of the new [RecordBatch]:
    // 1. pick whether the column is from the left or right
    // 2. based on the pick, `take` items from the different RecordBatches
    let mut columns: Vec<Arc<dyn Array>> = Vec::with_capacity(schema.fields().len());

    for column_index in column_indices {
        let array = if column_index.side == JoinSide::None {
            // LeftMark join, the mark column is a true if the indices is not null, otherwise it will be false
            Arc::new(compute::is_not_null(probe_indices)?)
        } else if column_index.side == build_side {
            let array = build_input_buffer.column(column_index.index);
            if array.is_empty() || build_indices.null_count() == build_indices.len() {
                // Outer join would generate a null index when finding no match at our side.
                // Therefore, it's possible we are empty but need to populate an n-length null array,
                // where n is the length of the index array.
                assert_eq!(build_indices.null_count(), build_indices.len());
                new_null_array(array.data_type(), build_indices.len())
            } else {
                compute::take(array.as_ref(), build_indices, None)?
            }
        } else {
            let array = probe_batch.column(column_index.index);
            if array.is_empty() || probe_indices.null_count() == probe_indices.len() {
                assert_eq!(probe_indices.null_count(), probe_indices.len());
                new_null_array(array.data_type(), probe_indices.len())
            } else {
                compute::take(array.as_ref(), probe_indices, None)?
            }
        };
        columns.push(array);
    }
    Ok(RecordBatch::try_new(Arc::new(schema.clone()), columns)?)
}

/// The input is the matched indices for left and right and
/// adjust the indices according to the join type
pub(crate) fn adjust_indices_by_join_type(
    left_indices: UInt64Array,
    right_indices: UInt32Array,
    adjust_range: Range<usize>,
    join_type: JoinType,
    preserve_order_for_right: bool,
) -> Result<(UInt64Array, UInt32Array)> {
    match join_type {
        JoinType::Inner => {
            // matched
            Ok((left_indices, right_indices))
        }
        JoinType::Left => {
            // matched
            Ok((left_indices, right_indices))
            // unmatched left row will be produced in the end of loop, and it has been set in the left visited bitmap
        }
        JoinType::Right => {
            // combine the matched and unmatched right result together
            append_right_indices(
                left_indices,
                right_indices,
                adjust_range,
                preserve_order_for_right,
            )
        }
        JoinType::Full => append_right_indices(left_indices, right_indices, adjust_range, false),
        JoinType::RightSemi => {
            // need to remove the duplicated record in the right side
            let right_indices = get_semi_indices(adjust_range, &right_indices);
            // the left_indices will not be used later for the `right semi` join
            Ok((left_indices, right_indices))
        }
        JoinType::RightAnti => {
            // need to remove the duplicated record in the right side
            // get the anti index for the right side
            let right_indices = get_anti_indices(adjust_range, &right_indices);
            // the left_indices will not be used later for the `right anti` join
            Ok((left_indices, right_indices))
        }
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark | JoinType::RightMark => {
            // matched or unmatched left row will be produced in the end of loop
            // When visit the right batch, we can output the matched left row and don't need to wait the end of loop
            Ok((
                UInt64Array::from_iter_values(vec![]),
                UInt32Array::from_iter_values(vec![]),
            ))
        }
    }
}

/// Appends right indices to left indices based on the specified order mode.
///
/// The function operates in two modes:
/// 1. If `preserve_order_for_right` is true, probe matched and unmatched indices
///    are inserted in order using the `append_probe_indices_in_order()` method.
/// 2. Otherwise, unmatched probe indices are simply appended after matched ones.
///
/// # Parameters
/// - `left_indices`: UInt64Array of left indices.
/// - `right_indices`: UInt32Array of right indices.
/// - `adjust_range`: Range to adjust the right indices.
/// - `preserve_order_for_right`: Boolean flag to determine the mode of operation.
///
/// # Returns
/// A tuple of updated `UInt64Array` and `UInt32Array`.
pub(crate) fn append_right_indices(
    left_indices: UInt64Array,
    right_indices: UInt32Array,
    adjust_range: Range<usize>,
    preserve_order_for_right: bool,
) -> Result<(UInt64Array, UInt32Array)> {
    if preserve_order_for_right {
        Ok(append_probe_indices_in_order(
            left_indices,
            right_indices,
            adjust_range,
        ))
    } else {
        let right_unmatched_indices = get_anti_indices(adjust_range, &right_indices);

        if right_unmatched_indices.is_empty() {
            Ok((left_indices, right_indices))
        } else {
            // `into_builder()` can fail here when there is nothing to be filtered and
            // left_indices or right_indices has the same reference to the cached indices.
            // In that case, we use a slower alternative.

            // the new left indices: left_indices + null array
            let mut new_left_indices_builder =
                left_indices.into_builder().unwrap_or_else(|left_indices| {
                    let mut builder = UInt64Builder::with_capacity(
                        left_indices.len() + right_unmatched_indices.len(),
                    );
                    debug_assert_eq!(
                        left_indices.null_count(),
                        0,
                        "expected left indices to have no nulls"
                    );
                    builder.append_slice(left_indices.values());
                    builder
                });
            new_left_indices_builder.append_nulls(right_unmatched_indices.len());
            let new_left_indices = UInt64Array::from(new_left_indices_builder.finish());

            // the new right indices: right_indices + right_unmatched_indices
            let mut new_right_indices_builder =
                right_indices
                    .into_builder()
                    .unwrap_or_else(|right_indices| {
                        let mut builder = UInt32Builder::with_capacity(
                            right_indices.len() + right_unmatched_indices.len(),
                        );
                        debug_assert_eq!(
                            right_indices.null_count(),
                            0,
                            "expected right indices to have no nulls"
                        );
                        builder.append_slice(right_indices.values());
                        builder
                    });
            debug_assert_eq!(
                right_unmatched_indices.null_count(),
                0,
                "expected right unmatched indices to have no nulls"
            );
            new_right_indices_builder.append_slice(right_unmatched_indices.values());
            let new_right_indices = UInt32Array::from(new_right_indices_builder.finish());

            Ok((new_left_indices, new_right_indices))
        }
    }
}

/// Returns `range` indices which are not present in `input_indices`
pub(crate) fn get_anti_indices<T: ArrowPrimitiveType>(
    range: Range<usize>,
    input_indices: &PrimitiveArray<T>,
) -> PrimitiveArray<T>
where
    NativeAdapter<T>: From<<T as ArrowPrimitiveType>::Native>,
{
    let mut bitmap = BooleanBufferBuilder::new(range.len());
    bitmap.append_n(range.len(), false);
    input_indices
        .iter()
        .flatten()
        .map(|v| v.as_usize())
        .filter(|v| range.contains(v))
        .for_each(|v| {
            bitmap.set_bit(v - range.start, true);
        });

    let offset = range.start;

    // get the anti index
    (range)
        .filter_map(|idx| (!bitmap.get_bit(idx - offset)).then_some(T::Native::from_usize(idx)))
        .collect()
}

/// Returns intersection of `range` and `input_indices` omitting duplicates
pub(crate) fn get_semi_indices<T: ArrowPrimitiveType>(
    range: Range<usize>,
    input_indices: &PrimitiveArray<T>,
) -> PrimitiveArray<T>
where
    NativeAdapter<T>: From<<T as ArrowPrimitiveType>::Native>,
{
    let mut bitmap = BooleanBufferBuilder::new(range.len());
    bitmap.append_n(range.len(), false);
    input_indices
        .iter()
        .flatten()
        .map(|v| v.as_usize())
        .filter(|v| range.contains(v))
        .for_each(|v| {
            bitmap.set_bit(v - range.start, true);
        });

    let offset = range.start;

    // get the semi index
    (range)
        .filter_map(|idx| (bitmap.get_bit(idx - offset)).then_some(T::Native::from_usize(idx)))
        .collect()
}

/// Appends probe indices in order by considering the given build indices.
///
/// This function constructs new build and probe indices by iterating through
/// the provided indices, and appends any missing values between previous and
/// current probe index with a corresponding null build index.
///
/// # Parameters
///
/// - `build_indices`: `PrimitiveArray` of `UInt64Type` containing build indices.
/// - `probe_indices`: `PrimitiveArray` of `UInt32Type` containing probe indices.
/// - `range`: The range of indices to consider.
///
/// # Returns
///
/// A tuple of two arrays:
/// - A `PrimitiveArray` of `UInt64Type` with the newly constructed build indices.
/// - A `PrimitiveArray` of `UInt32Type` with the newly constructed probe indices.
fn append_probe_indices_in_order(
    build_indices: PrimitiveArray<UInt64Type>,
    probe_indices: PrimitiveArray<UInt32Type>,
    range: Range<usize>,
) -> (PrimitiveArray<UInt64Type>, PrimitiveArray<UInt32Type>) {
    // Builders for new indices:
    let mut new_build_indices = UInt64Builder::new();
    let mut new_probe_indices = UInt32Builder::new();
    // Set previous index as the start index for the initial loop:
    let mut prev_index = range.start as u32;
    // Zip the two iterators.
    debug_assert!(build_indices.len() == probe_indices.len());
    for (build_index, probe_index) in build_indices
        .values()
        .into_iter()
        .zip(probe_indices.values().into_iter())
    {
        // Append values between previous and current probe index with null build index:
        for value in prev_index..*probe_index {
            new_probe_indices.append_value(value);
            new_build_indices.append_null();
        }
        // Append current indices:
        new_probe_indices.append_value(*probe_index);
        new_build_indices.append_value(*build_index);
        // Set current probe index as previous for the next iteration:
        prev_index = probe_index + 1;
    }
    // Append remaining probe indices after the last valid probe index with null build index.
    for value in prev_index..range.end as u32 {
        new_probe_indices.append_value(value);
        new_build_indices.append_null();
    }
    // Build arrays and return:
    (new_build_indices.finish(), new_probe_indices.finish())
}

pub(crate) fn asymmetric_join_output_partitioning(
    left: &Arc<dyn ExecutionPlan>,
    right: &Arc<dyn ExecutionPlan>,
    join_type: &JoinType,
) -> Partitioning {
    match join_type {
        JoinType::Inner | JoinType::Right => adjust_right_output_partitioning(
            right.output_partitioning(),
            left.schema().fields().len(),
        )
        .unwrap_or_else(|_| Partitioning::UnknownPartitioning(1)),
        JoinType::RightSemi | JoinType::RightAnti => right.output_partitioning().clone(),
        JoinType::Left
        | JoinType::LeftSemi
        | JoinType::LeftAnti
        | JoinType::Full
        | JoinType::LeftMark
        | JoinType::RightMark => {
            Partitioning::UnknownPartitioning(right.output_partitioning().partition_count())
        }
    }
}

/// This function is copied from
/// [`datafusion_physical_plan::physical_plan::execution_plan::boundedness_from_children`].
/// It is used to determine the boundedness of the join operator based on the boundedness of its children.
pub(crate) fn boundedness_from_children<'a>(
    children: impl IntoIterator<Item = &'a Arc<dyn ExecutionPlan>>,
) -> Boundedness {
    let mut unbounded_with_finite_mem = false;

    for child in children {
        match child.boundedness() {
            Boundedness::Unbounded {
                requires_infinite_memory: true,
            } => {
                return Boundedness::Unbounded {
                    requires_infinite_memory: true,
                }
            }
            Boundedness::Unbounded {
                requires_infinite_memory: false,
            } => {
                unbounded_with_finite_mem = true;
            }
            Boundedness::Bounded => {}
        }
    }

    if unbounded_with_finite_mem {
        Boundedness::Unbounded {
            requires_infinite_memory: false,
        }
    } else {
        Boundedness::Bounded
    }
}
