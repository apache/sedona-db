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
/// https://github.com/apache/datafusion/blob/50.2.0/datafusion/physical-plan/src/joins/utils.rs
use std::{ops::Range, sync::Arc};

use arrow::array::{
    downcast_array, new_null_array, Array, BooleanBufferBuilder, RecordBatch, RecordBatchOptions,
    UInt32Builder, UInt64Builder,
};
use arrow::buffer::NullBuffer;
use arrow::compute::{self, take};
use arrow::datatypes::{ArrowNativeType, Schema, UInt32Type, UInt64Type};
use arrow_array::{ArrowPrimitiveType, NativeAdapter, PrimitiveArray, UInt32Array, UInt64Array};
use arrow_schema::SchemaRef;
use datafusion_common::cast::as_boolean_array;
use datafusion_common::{JoinSide, Result};
use datafusion_expr::JoinType;
use datafusion_physical_expr::Partitioning;
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::joins::utils::{
    adjust_right_output_partitioning, ColumnIndex, JoinFilter,
};
use datafusion_physical_plan::projection::{
    join_allows_pushdown, join_table_borders, new_join_children, physical_to_column_exprs,
    update_join_filter, ProjectionExec,
};
use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties};

use crate::spatial_predicate::SpatialPredicateTrait;
use crate::SpatialPredicate;

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

/// Determines if a bitmap is needed to track matched rows in the probe side's "Multi" partition.
///
/// In a spatial partitioned join, the "Multi" partition of the probe side overlaps with multiple
/// partitions of the build side. Consequently, rows in the probe "Multi" partition are processed
/// against multiple build partitions.
///
/// For `Right`, `RightSemi`, `RightAnti`, and `Full` joins, we must track whether a probe row
/// has been matched across *any* of these interactions to correctly produce results:
/// - **Right/Full Outer**: Emit probe rows that never matched any build partition (checked at the last build partition).
/// - **Right Semi**: Emit a probe row the first time it matches, and suppress subsequent matches (deduplication).
/// - **Right Anti**: Emit probe rows only if they never match any build partition (checked at the last build partition).
pub(crate) fn need_probe_multi_partition_bitmap(join_type: JoinType) -> bool {
    matches!(
        join_type,
        JoinType::Right
            | JoinType::RightAnti
            | JoinType::RightSemi
            | JoinType::RightMark
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

pub(crate) fn adjust_indices_with_visited_info(
    left_indices: UInt64Array,
    right_indices: UInt32Array,
    adjust_range: Range<usize>,
    join_type: JoinType,
    preserve_order_for_right: bool,
    visited_info: Option<(&mut BooleanBufferBuilder, usize)>,
    produce_unmatched_probe_rows: bool,
) -> Result<(UInt64Array, UInt32Array)> {
    let Some((bitmap, offset)) = visited_info else {
        return adjust_indices_by_join_type(
            left_indices,
            right_indices,
            adjust_range,
            join_type,
            preserve_order_for_right,
        );
    };

    // Update the bitmap with the current matches first
    for idx in right_indices.values() {
        bitmap.set_bit(offset + (*idx as usize), true);
    }

    match join_type {
        JoinType::Right | JoinType::Full => {
            if !produce_unmatched_probe_rows {
                Ok((left_indices, right_indices))
            } else {
                let unmatched_count = adjust_range
                    .clone()
                    .filter(|&i| !bitmap.get_bit(i + offset))
                    .count();

                if unmatched_count == 0 {
                    return Ok((left_indices, right_indices));
                }

                let mut unmatched_indices = UInt32Builder::with_capacity(unmatched_count);
                for i in adjust_range.clone() {
                    if !bitmap.get_bit(i + offset) {
                        unmatched_indices.append_value(i as u32);
                    }
                }
                let unmatched_right = unmatched_indices.finish();

                let total_len = left_indices.len() + unmatched_count;
                let mut new_left_builder =
                    left_indices.into_builder().unwrap_or_else(|left_indices| {
                        let mut builder = UInt64Builder::with_capacity(total_len);
                        builder.append_slice(left_indices.values());
                        builder
                    });
                new_left_builder.append_nulls(unmatched_count);

                let mut new_right_builder =
                    right_indices
                        .into_builder()
                        .unwrap_or_else(|right_indices| {
                            let mut builder = UInt32Builder::with_capacity(total_len);
                            builder.append_slice(right_indices.values());
                            builder
                        });
                new_right_builder.append_slice(unmatched_right.values());

                Ok((
                    UInt64Array::from(new_left_builder.finish()),
                    UInt32Array::from(new_right_builder.finish()),
                ))
            }
        }
        JoinType::RightSemi => {
            if !produce_unmatched_probe_rows {
                Ok((
                    UInt64Array::from_iter_values(vec![]),
                    UInt32Array::from_iter_values(vec![]),
                ))
            } else {
                let matched_count = adjust_range
                    .clone()
                    .filter(|&i| bitmap.get_bit(i + offset))
                    .count();

                let mut final_right = UInt32Builder::with_capacity(matched_count);
                for i in adjust_range.clone() {
                    if bitmap.get_bit(i + offset) {
                        final_right.append_value(i as u32);
                    }
                }

                let mut final_left = UInt64Builder::with_capacity(matched_count);
                final_left.append_nulls(matched_count);

                Ok((final_left.finish(), final_right.finish()))
            }
        }
        JoinType::RightAnti => {
            if !produce_unmatched_probe_rows {
                Ok((
                    UInt64Array::from_iter_values(vec![]),
                    UInt32Array::from_iter_values(vec![]),
                ))
            } else {
                let unmatched_count = adjust_range
                    .clone()
                    .filter(|&i| !bitmap.get_bit(i + offset))
                    .count();

                let mut unmatched_indices = UInt32Builder::with_capacity(unmatched_count);
                for i in adjust_range.clone() {
                    if !bitmap.get_bit(i + offset) {
                        unmatched_indices.append_value(i as u32);
                    }
                }

                let mut final_left = UInt64Builder::with_capacity(unmatched_count);
                final_left.append_nulls(unmatched_count);

                Ok((final_left.finish(), unmatched_indices.finish()))
            }
        }
        JoinType::RightMark => {
            if !produce_unmatched_probe_rows {
                Ok((
                    UInt64Array::from_iter_values(vec![]),
                    UInt32Array::from_iter_values(vec![]),
                ))
            } else {
                let range_len = adjust_range.len();
                let mut mark_bitmap = BooleanBufferBuilder::new(range_len);

                for i in adjust_range.clone() {
                    mark_bitmap.append(bitmap.get_bit(i + offset));
                }

                let right_indices = UInt32Array::from_iter_values(adjust_range.map(|i| i as u32));

                let left_indices = PrimitiveArray::new(
                    vec![0; range_len].into(),
                    Some(NullBuffer::new(mark_bitmap.finish())),
                );

                Ok((left_indices, right_indices))
            }
        }
        _ => adjust_indices_by_join_type(
            left_indices,
            right_indices,
            adjust_range,
            join_type,
            preserve_order_for_right,
        ),
    }
}

pub(crate) fn apply_join_filter_to_indices(
    build_input_buffer: &RecordBatch,
    probe_batch: &RecordBatch,
    build_indices: UInt64Array,
    probe_indices: UInt32Array,
    distances: Option<&[f64]>,
    filter: &JoinFilter,
    build_side: JoinSide,
) -> Result<(UInt64Array, UInt32Array, Option<Vec<f64>>)> {
    // Forked from DataFusion 50.2.0 `apply_join_filter_to_indices`.
    // https://github.com/apache/datafusion/blob/50.2.0/datafusion/physical-plan/src/joins/utils.rs
    //
    // Changes vs upstream:
    // - Removes the `max_intermediate_size` parameter and its chunking logic.
    // - Calls our forked `build_batch_from_indices(..., join_type)` (needed for mark-join semantics).
    if build_indices.is_empty() && probe_indices.is_empty() {
        return Ok((build_indices, probe_indices, distances.map(|_| Vec::new())));
    };

    let intermediate_batch = build_batch_from_indices(
        filter.schema(),
        build_input_buffer,
        probe_batch,
        &build_indices,
        &probe_indices,
        filter.column_indices(),
        build_side,
        JoinType::Inner,
    )?;
    let filter_result = filter
        .expression()
        .evaluate(&intermediate_batch)?
        .into_array(intermediate_batch.num_rows())?;
    let mask = as_boolean_array(&filter_result)?;

    let left_filtered = compute::filter(&build_indices, mask)?;
    let right_filtered = compute::filter(&probe_indices, mask)?;

    let filtered_distances = if let Some(distances) = distances {
        debug_assert_eq!(
            distances.len(),
            build_indices.len(),
            "distances length should match indices length"
        );
        let dist_array = arrow_array::Float64Array::from(distances.to_vec());
        let filtered = compute::filter(&dist_array, mask)?;
        let filtered = filtered
            .as_any()
            .downcast_ref::<arrow_array::Float64Array>()
            .expect("filtered distance array should be Float64Array");
        Some(filtered.values().to_vec())
    } else {
        None
    };

    Ok((
        downcast_array(left_filtered.as_ref()),
        downcast_array(right_filtered.as_ref()),
        filtered_distances,
    ))
}

/// Returns a new [RecordBatch] by combining the `left` and `right` according to `indices`.
/// The resulting batch has [Schema] `schema`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_batch_from_indices(
    schema: &Schema,
    build_input_buffer: &RecordBatch,
    probe_batch: &RecordBatch,
    build_indices: &UInt64Array,
    probe_indices: &UInt32Array,
    column_indices: &[ColumnIndex],
    build_side: JoinSide,
    join_type: JoinType,
) -> Result<RecordBatch> {
    // Forked from DataFusion 50.2.0 `build_batch_from_indices`.
    // https://github.com/apache/datafusion/blob/50.2.0/datafusion/physical-plan/src/joins/utils.rs
    //
    // Changes vs upstream:
    // - Adds the `join_type` parameter so we can special-case mark joins.
    // - Fixes `RightMark` mark-column construction: for right-mark joins, the mark column must
    //   reflect match status for the *right* rows, so we build it from `build_indices` (the
    //   build-side indices) rather than `probe_indices`.
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
            // For mark joins, the mark column is a true if the indices is not null, otherwise it will be false
            if join_type == JoinType::RightMark {
                Arc::new(compute::is_not_null(build_indices)?)
            } else {
                Arc::new(compute::is_not_null(probe_indices)?)
            }
        } else if column_index.side == build_side {
            let array = build_input_buffer.column(column_index.index);
            if array.is_empty() || build_indices.null_count() == build_indices.len() {
                // Outer join would generate a null index when finding no match at our side.
                // Therefore, it's possible we are empty but need to populate an n-length null array,
                // where n is the length of the index array.
                assert_eq!(build_indices.null_count(), build_indices.len());
                new_null_array(array.data_type(), build_indices.len())
            } else {
                take(array.as_ref(), build_indices, None)?
            }
        } else {
            let array = probe_batch.column(column_index.index);
            if array.is_empty() || probe_indices.null_count() == probe_indices.len() {
                assert_eq!(probe_indices.null_count(), probe_indices.len());
                new_null_array(array.data_type(), probe_indices.len())
            } else {
                take(array.as_ref(), probe_indices, None)?
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
    // Forked from DataFusion 50.2.0 `adjust_indices_by_join_type`.
    // https://github.com/apache/datafusion/blob/50.2.0/datafusion/physical-plan/src/joins/utils.rs
    //
    // Changes vs upstream:
    // - Fixes `RightMark` handling to match our `SpatialJoinStream` contract:
    //   `right_indices` becomes the probe row indices (`adjust_range`), and `left_indices` is a
    //   mark array (null/non-null) indicating match status.
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
        JoinType::RightMark => {
            let new_left_indices = get_mark_indices(&adjust_range, &right_indices);
            let new_right_indices = adjust_range.map(|i| i as u32).collect();
            Ok((new_left_indices, new_right_indices))
        }
        JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark => {
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
    let bitmap = build_range_bitmap(&range, input_indices);
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
    let bitmap = build_range_bitmap(&range, input_indices);
    let offset = range.start;
    // get the semi index
    (range)
        .filter_map(|idx| (bitmap.get_bit(idx - offset)).then_some(T::Native::from_usize(idx)))
        .collect()
}

/// Returns an array for mark joins consisting of default values (zeros) with null/non-null markers.
///
/// For each index in `range`:
/// - If the index appears in `input_indices`, the value is non-null (0)
/// - If the index does not appear in `input_indices`, the value is null
///
/// This is used in mark joins to indicate which rows had matches.
pub(crate) fn get_mark_indices<T: ArrowPrimitiveType, R: ArrowPrimitiveType>(
    range: &Range<usize>,
    input_indices: &PrimitiveArray<T>,
) -> PrimitiveArray<R>
where
    NativeAdapter<T>: From<<T as ArrowPrimitiveType>::Native>,
{
    // Forked from DataFusion 50.2.0 `get_mark_indices`.
    // https://github.com/apache/datafusion/blob/50.2.0/datafusion/physical-plan/src/joins/utils.rs
    //
    // Changes vs upstream:
    // - Generalizes the output array element type (generic `R`) so we can build mark arrays of
    //   different physical types while still using the null buffer to encode match status.
    let mut bitmap = build_range_bitmap(range, input_indices);
    PrimitiveArray::new(
        vec![R::Native::default(); range.len()].into(),
        Some(NullBuffer::new(bitmap.finish())),
    )
}

fn build_range_bitmap<T: ArrowPrimitiveType>(
    range: &Range<usize>,
    input: &PrimitiveArray<T>,
) -> BooleanBufferBuilder {
    let mut builder = BooleanBufferBuilder::new(range.len());
    builder.append_n(range.len(), false);

    input.iter().flatten().for_each(|v| {
        let idx = v.as_usize();
        if range.contains(&idx) {
            builder.set_bit(idx - range.start, true);
        }
    });

    builder
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
    probe_side: JoinSide,
) -> Result<Partitioning> {
    let result = match join_type {
        JoinType::Inner => {
            if probe_side == JoinSide::Right {
                adjust_right_output_partitioning(
                    right.output_partitioning(),
                    left.schema().fields().len(),
                )?
            } else {
                left.output_partitioning().clone()
            }
        }
        JoinType::Right => {
            if probe_side == JoinSide::Right {
                adjust_right_output_partitioning(
                    right.output_partitioning(),
                    left.schema().fields().len(),
                )?
            } else {
                Partitioning::UnknownPartitioning(left.output_partitioning().partition_count())
            }
        }
        JoinType::RightSemi | JoinType::RightAnti | JoinType::RightMark => {
            if probe_side == JoinSide::Right {
                right.output_partitioning().clone()
            } else {
                Partitioning::UnknownPartitioning(left.output_partitioning().partition_count())
            }
        }
        JoinType::Left | JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark => {
            if probe_side == JoinSide::Left {
                left.output_partitioning().clone()
            } else {
                Partitioning::UnknownPartitioning(right.output_partitioning().partition_count())
            }
        }
        JoinType::Full => {
            if probe_side == JoinSide::Right {
                Partitioning::UnknownPartitioning(right.output_partitioning().partition_count())
            } else {
                Partitioning::UnknownPartitioning(left.output_partitioning().partition_count())
            }
        }
    };
    Ok(result)
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

pub(crate) fn compute_join_emission_type(
    left: &Arc<dyn ExecutionPlan>,
    right: &Arc<dyn ExecutionPlan>,
    join_type: JoinType,
    probe_side: JoinSide,
) -> EmissionType {
    let (build, probe) = if probe_side == JoinSide::Left {
        (right, left)
    } else {
        (left, right)
    };

    if build.boundedness().is_unbounded() {
        return EmissionType::Final;
    }

    if probe.pipeline_behavior() == EmissionType::Incremental {
        match join_type {
            // If we only need to generate matched rows from the probe side,
            // we can emit rows incrementally.
            JoinType::Inner => EmissionType::Incremental,
            JoinType::Right | JoinType::RightSemi | JoinType::RightAnti | JoinType::RightMark => {
                if probe_side == JoinSide::Right {
                    EmissionType::Incremental
                } else {
                    EmissionType::Both
                }
            }
            // If we need to generate unmatched rows from the *build side*,
            // we need to emit them at the end.
            JoinType::Left | JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark => {
                if probe_side == JoinSide::Left {
                    EmissionType::Incremental
                } else {
                    EmissionType::Both
                }
            }
            JoinType::Full => EmissionType::Both,
        }
    } else {
        probe.pipeline_behavior()
    }
}

/// Data required to push down a projection through a spatial join.
/// This is mostly taken from https://github.com/apache/datafusion/blob/51.0.0/datafusion/physical-plan/src/projection.rs
pub(crate) struct JoinPushdownData {
    pub projected_left_child: ProjectionExec,
    pub projected_right_child: ProjectionExec,
    pub join_filter: Option<JoinFilter>,
    pub join_on: SpatialPredicate,
}

/// Push down the given `projection` through the spatial join.
/// This code is adapted from https://github.com/apache/datafusion/blob/51.0.0/datafusion/physical-plan/src/projection.rs
pub(crate) fn try_pushdown_through_join(
    projection: &ProjectionExec,
    join_left: &Arc<dyn ExecutionPlan>,
    join_right: &Arc<dyn ExecutionPlan>,
    join_schema: &SchemaRef,
    join_type: JoinType,
    join_filter: Option<&JoinFilter>,
    join_on: &SpatialPredicate,
) -> Result<Option<JoinPushdownData>> {
    let Some(projection_as_columns) = physical_to_column_exprs(projection.expr()) else {
        return Ok(None);
    };

    // Mark joins produce a synthetic column that does not belong to either child. This synthetic
    // `mark` column will make `new_join_children` fail, so we skip pushdown for such joins.
    // This limitation is inherited from DataFusion's builtin `try_pushdown_through_join`.
    if matches!(join_type, JoinType::LeftMark | JoinType::RightMark) {
        return Ok(None);
    }

    let (far_right_left_col_ind, far_left_right_col_ind) =
        join_table_borders(join_left.schema().fields().len(), &projection_as_columns);

    if !join_allows_pushdown(
        &projection_as_columns,
        join_schema,
        far_right_left_col_ind,
        far_left_right_col_ind,
    ) {
        return Ok(None);
    }

    let (projected_left_child, projected_right_child) = new_join_children(
        &projection_as_columns,
        far_right_left_col_ind,
        far_left_right_col_ind,
        join_left,
        join_right,
    )?;

    let new_filter = if let Some(filter) = join_filter {
        let left_cols = &projection_as_columns[0..=far_right_left_col_ind as usize];
        let right_cols = &projection_as_columns[far_left_right_col_ind as usize..];
        match update_join_filter(
            left_cols,
            right_cols,
            filter,
            join_left.schema().fields().len(),
        ) {
            Some(updated) => Some(updated),
            None => return Ok(None),
        }
    } else {
        None
    };

    let projected_left_exprs = projected_left_child.expr();
    let projected_right_exprs = projected_right_child.expr();
    let Some(new_on) =
        join_on.update_for_child_projections(projected_left_exprs, projected_right_exprs)?
    else {
        return Ok(None);
    };

    Ok(Some(JoinPushdownData {
        projected_left_child,
        projected_right_child,
        join_filter: new_filter,
        join_on: new_on,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::{UInt32Array, UInt64Array};
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::SchemaRef;
    use datafusion_common::ScalarValue;
    use datafusion_expr::JoinType;
    use datafusion_expr::Operator;
    use datafusion_physical_expr::expressions::{BinaryExpr, Column, Literal};
    use datafusion_physical_expr::EquivalenceProperties;
    use datafusion_physical_expr::Partitioning;
    use datafusion_physical_expr::PhysicalExpr;
    use datafusion_physical_plan::empty::EmptyExec;
    use datafusion_physical_plan::projection::ProjectionExpr;
    use datafusion_physical_plan::repartition::RepartitionExec;
    use datafusion_physical_plan::DisplayAs;
    use datafusion_physical_plan::DisplayFormatType;
    use datafusion_physical_plan::PlanProperties;
    use rstest::rstest;

    fn setup_and_run(
        join_type: JoinType,
        offset: usize,
    ) -> (UInt64Array, UInt32Array, BooleanBufferBuilder) {
        // adjust_range: 0..5
        let adjust_range = 0..5;

        // Logical visited info for the range 0..5: [true, false, false, true, false]
        // Indices 0 and 3 are already visited.
        let logical_bitmap = vec![true, false, false, true, false];

        let mut bitmap_builder = BooleanBufferBuilder::new(offset + 5);
        // Prepend offset
        bitmap_builder.append_n(offset, false);
        bitmap_builder.append_slice(&logical_bitmap);

        // Current matches: right_indices: [1, 4]
        // This means index 1 and 4 are matched in this batch.
        // Index 2 remains unmatched.
        let right_indices = UInt32Array::from(vec![1, 4]);
        let left_indices = UInt64Array::from(vec![10, 11]); // Corresponding left indices

        // Case: Last partition (produce_unmatched_probe_rows = true)
        let (l, r) = adjust_indices_with_visited_info(
            left_indices.clone(),
            right_indices.clone(),
            adjust_range.clone(),
            join_type,
            false,
            Some((&mut bitmap_builder, offset)),
            true,
        )
        .unwrap();

        (l, r, bitmap_builder)
    }

    fn verify_bitmap(bitmap_builder: &BooleanBufferBuilder, offset: usize) {
        // Verify Bitmap updates
        // Offset + 1 should be set to true (was false)
        // Offset + 4 should be set to true (was false)
        assert!(bitmap_builder.get_bit(offset + 1));
        assert!(bitmap_builder.get_bit(offset + 4));
        // Offset + 0 was true, stays true
        assert!(bitmap_builder.get_bit(offset));
        // Offset + 3 was true, stays true
        assert!(bitmap_builder.get_bit(offset + 3));
        // Offset + 2 was false, stays false (unmatched)
        assert!(!bitmap_builder.get_bit(offset + 2));
    }

    #[rstest]
    #[case(0)]
    #[case(10)]
    fn test_adjust_indices_with_visited_info_right_outer(#[case] offset: usize) {
        let (l, r, bitmap_builder) = setup_and_run(JoinType::Right, offset);
        verify_bitmap(&bitmap_builder, offset);

        // Expected result:
        // Original: (10, 1), (11, 4)
        // Unmatched: (null, 2) (since 0 and 3 were visited before, 1 and 4 are visited now)
        // Total: 3 rows.
        assert_eq!(l.len(), 3);
        assert_eq!(r.len(), 3);

        // Check values
        // Original parts
        assert_eq!(l.value(0), 10);
        assert_eq!(r.value(0), 1);
        assert_eq!(l.value(1), 11);
        assert_eq!(r.value(1), 4);

        // Unmatched part
        assert!(l.is_null(2));
        assert_eq!(r.value(2), 2);
    }

    #[rstest]
    #[case(0)]
    #[case(10)]
    fn test_adjust_indices_with_visited_info_right_semi(#[case] offset: usize) {
        let (l, r, bitmap_builder) = setup_and_run(JoinType::RightSemi, offset);
        verify_bitmap(&bitmap_builder, offset);

        // Expected: 0, 1, 3, 4 (all visited)
        // 0 (pre-visited), 1 (matched now), 3 (pre-visited), 4 (matched now)
        assert_eq!(r.len(), 4);
        let r_values: Vec<u32> = r.values().iter().copied().collect();
        assert!(r_values.contains(&0));
        assert!(r_values.contains(&1));
        assert!(r_values.contains(&3));
        assert!(r_values.contains(&4));

        // Left side should be all nulls
        assert_eq!(l.null_count(), 4);
    }

    #[rstest]
    #[case(0)]
    #[case(10)]
    fn test_adjust_indices_with_visited_info_right_anti(#[case] offset: usize) {
        let (l, r, bitmap_builder) = setup_and_run(JoinType::RightAnti, offset);
        verify_bitmap(&bitmap_builder, offset);

        // Expected: 2 (unvisited)
        assert_eq!(r.len(), 1);
        assert_eq!(r.value(0), 2);
        assert_eq!(l.null_count(), 1);
    }

    #[rstest]
    #[case(0)]
    #[case(10)]
    fn test_adjust_indices_with_visited_info_right_mark(#[case] offset: usize) {
        let (l, r, bitmap_builder) = setup_and_run(JoinType::RightMark, offset);
        verify_bitmap(&bitmap_builder, offset);

        // Expected left: validity [true, true, false, true, true]
        // 0: visited (T)
        // 1: matched now (T)
        // 2: unvisited (F)
        // 3: visited (T)
        // 4: matched now (T)
        assert_eq!(l.len(), 5);
        assert!(l.is_valid(0));
        assert!(l.is_valid(1));
        assert!(l.is_null(2));
        assert!(l.is_valid(3));
        assert!(l.is_valid(4));

        // Expected right: [0, 1, 2, 3, 4]
        assert_eq!(r.len(), 5);
        for i in 0..5 {
            assert_eq!(r.value(i), i as u32);
        }
    }

    use crate::spatial_predicate::{RelationPredicate, SpatialRelationType};

    fn make_schema(prefix: &str, num_fields: usize) -> SchemaRef {
        Arc::new(Schema::new(
            (0..num_fields)
                .map(|i| Field::new(format!("{prefix}{i}"), DataType::Int32, true))
                .collect::<Vec<_>>(),
        ))
    }

    fn assert_hash_partitioning_column_indices(
        partitioning: &Partitioning,
        expected_indices: &[usize],
        expected_partition_count: usize,
    ) {
        match partitioning {
            Partitioning::Hash(exprs, size) => {
                assert_eq!(*size, expected_partition_count);
                assert_eq!(exprs.len(), expected_indices.len());
                for (expr, expected_idx) in exprs.iter().zip(expected_indices.iter()) {
                    let col = expr
                        .as_any()
                        .downcast_ref::<Column>()
                        .expect("expected Column physical expr");
                    assert_eq!(col.index(), *expected_idx);
                }
            }
            other => panic!("expected Hash partitioning, got {other:?}"),
        }
    }

    fn make_join_schema(left: &SchemaRef, right: &SchemaRef) -> SchemaRef {
        let mut fields = Vec::with_capacity(left.fields().len() + right.fields().len());
        fields.extend(left.fields().iter().cloned());
        fields.extend(right.fields().iter().cloned());
        Arc::new(Schema::new(fields))
    }

    fn make_join_projection(
        join_schema: &SchemaRef,
        indices: &[usize],
        aliases: &[&str],
    ) -> Result<ProjectionExec> {
        assert_eq!(indices.len(), aliases.len());
        let exprs = indices
            .iter()
            .zip(aliases.iter())
            .map(|(index, alias)| {
                let field = join_schema.field(*index);
                ProjectionExpr {
                    expr: Arc::new(Column::new(field.name(), *index)),
                    alias: (*alias).to_string(),
                }
            })
            .collect::<Vec<_>>();
        ProjectionExec::try_new(exprs, Arc::new(EmptyExec::new(Arc::clone(join_schema))))
    }

    fn make_join_filter(
        left_indices: Vec<usize>,
        right_indices: Vec<usize>,
        schema: SchemaRef,
    ) -> JoinFilter {
        let expression: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new(schema.field(0).name(), 0)),
            Operator::Eq,
            Arc::new(Column::new(schema.field(1).name(), 1)),
        ));
        JoinFilter::new(
            expression,
            JoinFilter::build_column_indices(left_indices, right_indices),
            schema,
        )
    }

    fn assert_is_column_expr(expr: &Arc<dyn PhysicalExpr>, name: &str, index: usize) {
        let col = expr
            .as_any()
            .downcast_ref::<Column>()
            .expect("expected Column");
        assert_eq!(col.name(), name);
        assert_eq!(col.index(), index);
    }

    #[derive(Debug, Clone)]
    struct PropertiesOnlyExec {
        schema: SchemaRef,
        properties: PlanProperties,
    }

    impl PropertiesOnlyExec {
        fn new(schema: SchemaRef, boundedness: Boundedness, emission_type: EmissionType) -> Self {
            let schema_ref = Arc::clone(&schema);
            let properties = PlanProperties::new(
                EquivalenceProperties::new(schema),
                Partitioning::UnknownPartitioning(1),
                emission_type,
                boundedness,
            );
            Self {
                schema: schema_ref,
                properties,
            }
        }
    }

    impl DisplayAs for PropertiesOnlyExec {
        fn fmt_as(&self, _t: DisplayFormatType, _f: &mut std::fmt::Formatter) -> std::fmt::Result {
            Ok(())
        }
    }

    impl ExecutionPlan for PropertiesOnlyExec {
        fn name(&self) -> &'static str {
            "PropertiesOnlyExec"
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn schema(&self) -> SchemaRef {
            Arc::clone(&self.schema)
        }

        fn properties(&self) -> &PlanProperties {
            &self.properties
        }

        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            vec![]
        }

        fn with_new_children(
            self: Arc<Self>,
            _children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            Ok(self)
        }

        fn execute(
            &self,
            _partition: usize,
            _context: Arc<datafusion_execution::TaskContext>,
        ) -> Result<datafusion_execution::SendableRecordBatchStream> {
            unimplemented!("PropertiesOnlyExec is for properties tests only")
        }

        fn statistics(&self) -> Result<datafusion_common::Statistics> {
            Ok(datafusion_common::Statistics::new_unknown(
                self.schema().as_ref(),
            ))
        }

        fn partition_statistics(
            &self,
            _partition: Option<usize>,
        ) -> Result<datafusion_common::Statistics> {
            Ok(datafusion_common::Statistics::new_unknown(
                self.schema().as_ref(),
            ))
        }
    }

    #[test]
    fn adjust_right_output_partitioning_offsets_hash_columns() -> Result<()> {
        let right_part = Partitioning::Hash(vec![Arc::new(Column::new("r0", 0))], 8);
        let adjusted = adjust_right_output_partitioning(&right_part, 3)?;
        assert_hash_partitioning_column_indices(&adjusted, &[3], 8);

        let right_part_multi = Partitioning::Hash(
            vec![
                Arc::new(Column::new("r0", 0)),
                Arc::new(Column::new("r2", 2)),
            ],
            16,
        );
        let adjusted_multi = adjust_right_output_partitioning(&right_part_multi, 5)?;
        assert_hash_partitioning_column_indices(&adjusted_multi, &[5, 7], 16);
        Ok(())
    }

    #[test]
    fn adjust_right_output_partitioning_passthrough_non_hash() -> Result<()> {
        let right_part = Partitioning::UnknownPartitioning(4);
        let adjusted = adjust_right_output_partitioning(&right_part, 10)?;
        assert!(matches!(adjusted, Partitioning::UnknownPartitioning(4)));
        Ok(())
    }

    #[test]
    fn asymmetric_join_output_partitioning_all_combinations_hash_keys() -> Result<()> {
        // Left is partitioned by l1, right is partitioned by r0.
        // We validate output partitioning for all (probe_side, join_type) combinations.
        let left_partitions = 3;
        let right_partitions = 5;

        let left_schema = make_schema("l", 2);
        let left_len = left_schema.fields().len();
        let left_input: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(left_schema));
        let left: Arc<dyn ExecutionPlan> = Arc::new(RepartitionExec::try_new(
            left_input,
            Partitioning::Hash(vec![Arc::new(Column::new("l1", 1))], left_partitions),
        )?);

        let right_input: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(make_schema("r", 1)));
        let right: Arc<dyn ExecutionPlan> = Arc::new(RepartitionExec::try_new(
            right_input,
            Partitioning::Hash(vec![Arc::new(Column::new("r0", 0))], right_partitions),
        )?);

        let join_types = [
            JoinType::Inner,
            JoinType::Left,
            JoinType::Right,
            JoinType::Full,
            JoinType::LeftSemi,
            JoinType::LeftAnti,
            JoinType::LeftMark,
            JoinType::RightSemi,
            JoinType::RightAnti,
            JoinType::RightMark,
        ];
        let probe_sides = [JoinSide::Left, JoinSide::Right];

        for join_type in join_types {
            for probe_side in probe_sides {
                let out =
                    asymmetric_join_output_partitioning(&left, &right, &join_type, probe_side)?;

                match (join_type, probe_side) {
                    (JoinType::Inner, JoinSide::Right) => {
                        // join output schema is left + right, so offset right partition key
                        assert_hash_partitioning_column_indices(
                            &out,
                            &[left_len],
                            right_partitions,
                        );
                    }
                    (JoinType::Inner, JoinSide::Left) => {
                        assert_hash_partitioning_column_indices(&out, &[1], left_partitions);
                    }

                    (JoinType::Right, JoinSide::Right) => {
                        assert_hash_partitioning_column_indices(
                            &out,
                            &[left_len],
                            right_partitions,
                        );
                    }
                    (JoinType::Right, JoinSide::Left) => {
                        assert!(matches!(
                            out,
                            Partitioning::UnknownPartitioning(n) if n == left_partitions
                        ));
                    }

                    (
                        JoinType::RightSemi | JoinType::RightAnti | JoinType::RightMark,
                        JoinSide::Right,
                    ) => {
                        // right-only output schema (plus mark column for RightMark), so no offset
                        assert_hash_partitioning_column_indices(&out, &[0], right_partitions);
                    }
                    (
                        JoinType::RightSemi | JoinType::RightAnti | JoinType::RightMark,
                        JoinSide::Left,
                    ) => {
                        assert!(matches!(
                            out,
                            Partitioning::UnknownPartitioning(n) if n == left_partitions
                        ));
                    }

                    (
                        JoinType::Left
                        | JoinType::LeftSemi
                        | JoinType::LeftAnti
                        | JoinType::LeftMark,
                        JoinSide::Left,
                    ) => {
                        assert_hash_partitioning_column_indices(&out, &[1], left_partitions);
                    }
                    (
                        JoinType::Left
                        | JoinType::LeftSemi
                        | JoinType::LeftAnti
                        | JoinType::LeftMark,
                        JoinSide::Right,
                    ) => {
                        assert!(matches!(
                            out,
                            Partitioning::UnknownPartitioning(n) if n == right_partitions
                        ));
                    }

                    (JoinType::Full, JoinSide::Left) => {
                        assert!(matches!(
                            out,
                            Partitioning::UnknownPartitioning(n) if n == left_partitions
                        ));
                    }
                    (JoinType::Full, JoinSide::Right) => {
                        assert!(matches!(
                            out,
                            Partitioning::UnknownPartitioning(n) if n == right_partitions
                        ));
                    }

                    _ => unreachable!("unexpected probe_side: {probe_side:?}"),
                }
            }
        }

        Ok(())
    }

    #[test]
    fn compute_join_emission_type_prefers_final_for_unbounded_build() {
        let schema = make_schema("x", 1);
        let build: Arc<dyn ExecutionPlan> = Arc::new(PropertiesOnlyExec::new(
            Arc::clone(&schema),
            datafusion_physical_plan::execution_plan::Boundedness::Unbounded {
                requires_infinite_memory: false,
            },
            EmissionType::Incremental,
        ));
        let probe: Arc<dyn ExecutionPlan> = Arc::new(PropertiesOnlyExec::new(
            schema,
            datafusion_physical_plan::execution_plan::Boundedness::Bounded,
            EmissionType::Incremental,
        ));

        assert_eq!(
            compute_join_emission_type(&build, &probe, JoinType::Inner, JoinSide::Right),
            EmissionType::Final
        );
        assert_eq!(
            compute_join_emission_type(&probe, &build, JoinType::Inner, JoinSide::Left),
            EmissionType::Final
        );
    }

    #[test]
    fn compute_join_emission_type_uses_probe_behavior_for_inner_join() {
        let schema = make_schema("x", 1);
        let build: Arc<dyn ExecutionPlan> = Arc::new(PropertiesOnlyExec::new(
            Arc::clone(&schema),
            datafusion_physical_plan::execution_plan::Boundedness::Bounded,
            EmissionType::Incremental,
        ));
        for probe_emission_type in [EmissionType::Incremental, EmissionType::Both] {
            let probe: Arc<dyn ExecutionPlan> = Arc::new(PropertiesOnlyExec::new(
                Arc::clone(&schema),
                datafusion_physical_plan::execution_plan::Boundedness::Bounded,
                probe_emission_type,
            ));

            assert_eq!(
                compute_join_emission_type(&build, &probe, JoinType::Inner, JoinSide::Right),
                probe_emission_type
            );
            assert_eq!(
                compute_join_emission_type(&probe, &build, JoinType::Inner, JoinSide::Left),
                probe_emission_type
            );
        }
    }

    #[test]
    fn compute_join_emission_type_incremental_when_join_type_and_probe_side_matches() {
        let schema = make_schema("x", 1);
        let left: Arc<dyn ExecutionPlan> = Arc::new(PropertiesOnlyExec::new(
            Arc::clone(&schema),
            datafusion_physical_plan::execution_plan::Boundedness::Bounded,
            EmissionType::Incremental,
        ));
        let right: Arc<dyn ExecutionPlan> = Arc::new(PropertiesOnlyExec::new(
            schema,
            datafusion_physical_plan::execution_plan::Boundedness::Bounded,
            EmissionType::Incremental,
        ));

        for join_type in [
            JoinType::Right,
            JoinType::RightSemi,
            JoinType::RightAnti,
            JoinType::RightMark,
        ] {
            assert_eq!(
                compute_join_emission_type(&left, &right, join_type, JoinSide::Right),
                EmissionType::Incremental
            );
            assert_eq!(
                compute_join_emission_type(&left, &right, join_type, JoinSide::Left),
                EmissionType::Both
            );
        }

        for join_type in [
            JoinType::Left,
            JoinType::LeftSemi,
            JoinType::LeftAnti,
            JoinType::LeftMark,
        ] {
            assert_eq!(
                compute_join_emission_type(&left, &right, join_type, JoinSide::Left),
                EmissionType::Incremental
            );
            assert_eq!(
                compute_join_emission_type(&left, &right, join_type, JoinSide::Right),
                EmissionType::Both
            );
        }
    }

    #[test]
    fn compute_join_emission_type_always_both_for_full_outer_join() {
        let schema = make_schema("x", 1);
        let left: Arc<dyn ExecutionPlan> = Arc::new(PropertiesOnlyExec::new(
            Arc::clone(&schema),
            datafusion_physical_plan::execution_plan::Boundedness::Bounded,
            EmissionType::Incremental,
        ));
        let right: Arc<dyn ExecutionPlan> = Arc::new(PropertiesOnlyExec::new(
            schema,
            datafusion_physical_plan::execution_plan::Boundedness::Bounded,
            EmissionType::Incremental,
        ));

        assert_eq!(
            compute_join_emission_type(&left, &right, JoinType::Full, JoinSide::Left),
            EmissionType::Both
        );
        assert_eq!(
            compute_join_emission_type(&left, &right, JoinType::Full, JoinSide::Right),
            EmissionType::Both
        );
    }

    #[test]
    fn try_pushdown_through_join_updates_children_filter_and_predicate() -> Result<()> {
        let left_schema = make_schema("l", 2);
        let right_schema = make_schema("r", 2);
        let join_schema = make_join_schema(&left_schema, &right_schema);
        let join_left: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(Arc::clone(&left_schema)));
        let join_right: Arc<dyn ExecutionPlan> =
            Arc::new(EmptyExec::new(Arc::clone(&right_schema)));

        let projection = make_join_projection(&join_schema, &[1, 2], &["l1_out", "r0_out"])?;

        let join_on = SpatialPredicate::Relation(RelationPredicate::new(
            Arc::new(Column::new("l1", 1)),
            Arc::new(Column::new("r0", 0)),
            SpatialRelationType::Intersects,
        ));

        let filter_schema = Arc::new(Schema::new(vec![
            Field::new("l1", DataType::Int32, true),
            Field::new("r0", DataType::Int32, true),
        ]));
        let join_filter = make_join_filter(vec![1], vec![0], filter_schema);

        let pushdown = try_pushdown_through_join(
            &projection,
            &join_left,
            &join_right,
            &join_schema,
            JoinType::Inner,
            Some(&join_filter),
            &join_on,
        )?
        .expect("expected pushdown");

        assert_eq!(pushdown.projected_left_child.expr().len(), 1);
        let left_proj = &pushdown.projected_left_child.expr()[0];
        assert_eq!(left_proj.alias, "l1_out");
        let left_col = left_proj
            .expr
            .as_any()
            .downcast_ref::<Column>()
            .expect("expected Column");
        assert_eq!(left_col.name(), "l1");
        assert_eq!(left_col.index(), 1);

        assert_eq!(pushdown.projected_right_child.expr().len(), 1);
        let right_proj = &pushdown.projected_right_child.expr()[0];
        assert_eq!(right_proj.alias, "r0_out");
        let right_col = right_proj
            .expr
            .as_any()
            .downcast_ref::<Column>()
            .expect("expected Column");
        assert_eq!(right_col.name(), "r0");
        assert_eq!(right_col.index(), 0);

        let updated_filter = pushdown.join_filter.expect("expected updated filter");
        let indices = updated_filter.column_indices();
        assert_eq!(indices.len(), 2);
        assert_eq!(indices[0].side, JoinSide::Left);
        assert_eq!(indices[0].index, 0);
        assert_eq!(indices[1].side, JoinSide::Right);
        assert_eq!(indices[1].index, 0);

        let SpatialPredicate::Relation(updated_on) = pushdown.join_on else {
            unreachable!("expected relation predicate")
        };
        assert_is_column_expr(&updated_on.left, "l1_out", 0);
        assert_is_column_expr(&updated_on.right, "r0_out", 0);

        Ok(())
    }

    #[test]
    fn try_pushdown_through_join_skips_mark_join() -> Result<()> {
        let left_schema = make_schema("l", 1);
        let right_schema = make_schema("r", 1);
        let join_schema = make_join_schema(&left_schema, &right_schema);
        let join_left: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(Arc::clone(&left_schema)));
        let join_right: Arc<dyn ExecutionPlan> =
            Arc::new(EmptyExec::new(Arc::clone(&right_schema)));
        let projection = make_join_projection(&join_schema, &[0, 1], &["l0", "r0"])?;

        let join_on = SpatialPredicate::Relation(RelationPredicate::new(
            Arc::new(Column::new("l0", 0)),
            Arc::new(Column::new("r0", 0)),
            SpatialRelationType::Intersects,
        ));

        let result = try_pushdown_through_join(
            &projection,
            &join_left,
            &join_right,
            &join_schema,
            JoinType::LeftMark,
            None,
            &join_on,
        )?;
        assert!(result.is_none());
        Ok(())
    }

    #[test]
    fn try_pushdown_through_join_requires_column_projection() -> Result<()> {
        let left_schema = make_schema("l", 1);
        let right_schema = make_schema("r", 1);
        let join_schema = make_join_schema(&left_schema, &right_schema);
        let join_left: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(Arc::clone(&left_schema)));
        let join_right: Arc<dyn ExecutionPlan> =
            Arc::new(EmptyExec::new(Arc::clone(&right_schema)));

        let projection = ProjectionExec::try_new(
            vec![ProjectionExpr {
                expr: Arc::new(Literal::new(ScalarValue::Int32(Some(1)))),
                alias: "lit".to_string(),
            }],
            Arc::new(EmptyExec::new(Arc::clone(&join_schema))),
        )?;

        let join_on = SpatialPredicate::Relation(RelationPredicate::new(
            Arc::new(Column::new("l0", 0)),
            Arc::new(Column::new("r0", 0)),
            SpatialRelationType::Intersects,
        ));

        let result = try_pushdown_through_join(
            &projection,
            &join_left,
            &join_right,
            &join_schema,
            JoinType::Inner,
            None,
            &join_on,
        )?;
        assert!(result.is_none());
        Ok(())
    }

    #[test]
    fn try_pushdown_through_join_requires_projection_narrowing() -> Result<()> {
        let left_schema = make_schema("l", 2);
        let right_schema = make_schema("r", 2);
        let join_schema = make_join_schema(&left_schema, &right_schema);
        let join_left: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(Arc::clone(&left_schema)));
        let join_right: Arc<dyn ExecutionPlan> =
            Arc::new(EmptyExec::new(Arc::clone(&right_schema)));

        let projection =
            make_join_projection(&join_schema, &[0, 1, 2, 3], &["l0", "l1", "r0", "r1"])?;

        let join_on = SpatialPredicate::Relation(RelationPredicate::new(
            Arc::new(Column::new("l0", 0)),
            Arc::new(Column::new("r0", 0)),
            SpatialRelationType::Intersects,
        ));

        let result = try_pushdown_through_join(
            &projection,
            &join_left,
            &join_right,
            &join_schema,
            JoinType::Inner,
            None,
            &join_on,
        )?;
        assert!(result.is_none());
        Ok(())
    }

    #[test]
    fn try_pushdown_through_join_fails_when_filter_columns_missing() -> Result<()> {
        let left_schema = make_schema("l", 2);
        let right_schema = make_schema("r", 2);
        let join_schema = make_join_schema(&left_schema, &right_schema);
        let join_left: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(Arc::clone(&left_schema)));
        let join_right: Arc<dyn ExecutionPlan> =
            Arc::new(EmptyExec::new(Arc::clone(&right_schema)));

        let projection = make_join_projection(&join_schema, &[1, 3], &["l1_out", "r1_out"])?;

        let join_on = SpatialPredicate::Relation(RelationPredicate::new(
            Arc::new(Column::new("l1", 1)),
            Arc::new(Column::new("r1", 1)),
            SpatialRelationType::Intersects,
        ));

        let filter_schema = Arc::new(Schema::new(vec![
            Field::new("l1", DataType::Int32, true),
            Field::new("r0", DataType::Int32, true),
        ]));
        let join_filter = make_join_filter(vec![1], vec![0], filter_schema);

        let result = try_pushdown_through_join(
            &projection,
            &join_left,
            &join_right,
            &join_schema,
            JoinType::Inner,
            Some(&join_filter),
            &join_on,
        )?;
        assert!(result.is_none());
        Ok(())
    }

    #[test]
    fn try_pushdown_through_join_fails_when_predicate_columns_missing() -> Result<()> {
        let left_schema = make_schema("l", 2);
        let right_schema = make_schema("r", 2);
        let join_schema = make_join_schema(&left_schema, &right_schema);
        let join_left: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(Arc::clone(&left_schema)));
        let join_right: Arc<dyn ExecutionPlan> =
            Arc::new(EmptyExec::new(Arc::clone(&right_schema)));

        let projection = make_join_projection(&join_schema, &[1, 3], &["l1_out", "r1_out"])?;

        let join_on = SpatialPredicate::Relation(RelationPredicate::new(
            Arc::new(Column::new("l1", 1)),
            Arc::new(Column::new("r0", 0)),
            SpatialRelationType::Intersects,
        ));

        let result = try_pushdown_through_join(
            &projection,
            &join_left,
            &join_right,
            &join_schema,
            JoinType::Inner,
            None,
            &join_on,
        )?;
        assert!(result.is_none());
        Ok(())
    }
}
