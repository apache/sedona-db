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
use datafusion_common::cast::as_boolean_array;
use datafusion_common::{JoinSide, Result};
use datafusion_expr::JoinType;
use datafusion_physical_expr::Partitioning;
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
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
    // Forked from DataFusion 50.2.0 `apply_join_filter_to_indices`.
    // https://github.com/apache/datafusion/blob/50.2.0/datafusion/physical-plan/src/joins/utils.rs
    //
    // Changes vs upstream:
    // - Removes the `max_intermediate_size` parameter and its chunking logic.
    // - Calls our forked `build_batch_from_indices(..., join_type)` (needed for mark-join semantics).
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
        JoinType::Inner,
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::SchemaRef;
    use datafusion_expr::JoinType;
    use datafusion_physical_expr::expressions::Column;
    use datafusion_physical_expr::EquivalenceProperties;
    use datafusion_physical_expr::Partitioning;
    use datafusion_physical_plan::empty::EmptyExec;
    use datafusion_physical_plan::repartition::RepartitionExec;
    use datafusion_physical_plan::DisplayAs;
    use datafusion_physical_plan::DisplayFormatType;
    use datafusion_physical_plan::PlanProperties;

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
}
