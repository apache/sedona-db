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

use core::f64;
use std::ops::Range;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayBuilder, AsArray, Float64Array, ListArray, OffsetBufferBuilder, PrimitiveBuilder,
    RecordBatch, StructArray, UInt64Array,
};
use arrow::buffer::OffsetBuffer;
use arrow::compute::{concat, concat_batches, interleave};
use arrow::datatypes::{DataType, Field, Float64Type, Schema, SchemaRef, UInt64Type};
use arrow_array::ArrayRef;
use arrow_schema::Fields;
use datafusion::config::SpillCompression;
use datafusion_common::Result;
use datafusion_execution::disk_manager::RefCountedTempFile;
use datafusion_execution::runtime_env::RuntimeEnv;
use datafusion_physical_plan::metrics::SpillMetrics;
use sedona_common::sedona_internal_err;

use crate::index::spatial_index::DISTANCE_TOLERANCE;
use crate::utils::spill::{RecordBatchSpillReader, RecordBatchSpillWriter};

/// [UnprocessedKNNResultBatch] represents the KNN results produced by probing the spatial index.
/// An [UnprocessedKNNResultBatch] may include KNN results for multiple probe rows.
///
/// The KNN results are stored in a StructArray, where each row corresponds to a KNN result.
/// The results for the same probe row are stored in contiguous rows, and the offsets to
/// split the results into groups per probe row are stored in the `offsets` field.
///
/// Each probe row has a unique index. The index must be strictly increasing
/// across probe rows. The sequence of index across the entire sequence of ingested
/// [UnprocessedKNNResultBatch] must also be strictly increasing. The index is computed based on
/// the 0-based index of the probe row in this probe partition.
///
/// The KNN results are filtered, meaning that the original KNN results obtained by probing
/// the spatial index may be further filtered based on some predicates. It is also possible that
/// all the KNN results for a probe row are filtered out. However, we still need to keep track of the
/// distances of unfiltered results to correctly compute the top-K distances before filtering. This
/// is critical for correctly merging KNN results from multiple partitions.
///
/// Imagine that a KNN query for a probe row yields the following 5 results (K = 5):
///
/// ```text
/// D0  D1  D2  D3  D4
/// R0      R2  R3
/// ```
///
/// Where Di is the distance of the i-th nearest neighbor, and Ri is the result row index.
/// R1 and R4 are filtered out based on some predicate, so the final results only contain R0, R2, and R3.
/// The core idea is that the filtering is applied AFTER determining the top-K distances, so the number
/// of final results may be less than K.
///
/// However, if we split the object side of KNN join into 2 partitions, and the KNN results from
/// each partition are as follows:
///
/// ```text
/// Partition 0:
/// D1  D3  D5  D6  D7
///     R3      R6  R7
///
/// Partition 1:
/// D0  D2  D4  D8  D9
/// R0  R2      R8
/// ```
///
/// If we blindly merge the filtered results from both partitions and take top-k, we would get:
///
/// ```text
/// D0  D2  D3  D6  D8
/// R0  R2  R3  R6  R8
/// ```
///
/// Which contains more results than single-partitioned KNN join (i.e., 5 results instead of 3). This is
/// incorrect.
///
/// When merging the results from both partitions, we need to consider the distances of all unfiltered
/// results to correctly determine the top-K distances before filtering. In this case, the top-5 distances
/// are D0, D1, D2, D3, and D4. We take D4 as the distance threshold to filter merged results. After filtering,
/// we still get R0, R2, and R3 as the final results.
///
/// Please note that the KNN results for the last probe row in this array may be incomplete,
/// this is due to batch slicing during probe result batch production. We should be cautious
/// and correctly handle the KNN results for each probe row across multiple slices.
///
/// Here is a concrete example: the [UnprocessedKNNResultBatch] may contain KNN results for 3 probe rows:
///
/// ```text
/// [P0, R00]
/// [P0, R01]
/// [P0, R02]
/// [P1, R10]
/// [P1, R11]
/// [P1, R12]
/// [P2, R20]
/// ```
///
/// Where Pi is the i-th probe row, and Rij is the j-th KNN result for probe row Pi.
/// The KNN results for probe row P2 could be incomplete, and the next ingested KNN result batch
/// may contain more results for probe row P2:
///
/// ```text
/// [P2, R21]
/// [P2, R22]
/// [P3, R30]
/// ...
/// ```
///
/// In practice, we process the KNN results or a probe row only when we have seen all its results.
/// The may-be incomplete tail part of an ingested [UnprocessedKNNResultBatch] is sliced and concatenated with
/// the next ingested [UnprocessedKNNResultBatch] to form a complete set of KNN results for that probe row.
/// This slicing and concatenating won't happen frequently in practice (once per ingested batch
/// on average), so the performance impact is minimal.
struct UnprocessedKNNResultBatch {
    row_array: StructArray,
    probe_indices: Vec<usize>,
    distances: Vec<f64>,
    unfiltered_probe_indices: Vec<usize>,
    unfiltered_distances: Vec<f64>,
}

impl UnprocessedKNNResultBatch {
    fn new(
        row_array: StructArray,
        probe_indices: Vec<usize>,
        distances: Vec<f64>,
        unfiltered_probe_indices: Vec<usize>,
        unfiltered_distances: Vec<f64>,
    ) -> Self {
        Self {
            row_array,
            probe_indices,
            distances,
            unfiltered_probe_indices,
            unfiltered_distances,
        }
    }

    /// Create a new [UnprocessedKNNResultBatch] representing the unprocessed tail KNN results
    /// from an unprocessed [KNNProbeResult].
    fn new_unprocessed_tail(tail: KNNProbeResult<'_>, row_array: &StructArray) -> Self {
        let index = tail.probe_row_index;
        let num_rows = tail.row_range.len();
        let num_unfiltered_rows = tail.unfiltered_distances.len();

        let sliced_row_array = row_array.slice(tail.row_range.start, num_rows);
        let probe_indices = vec![index; num_rows];
        let distances = tail.distances.to_vec();
        let unfiltered_probe_indices = vec![index; num_unfiltered_rows];
        let unfiltered_distances = tail.unfiltered_distances.to_vec();

        Self {
            row_array: sliced_row_array,
            probe_indices,
            distances,
            unfiltered_probe_indices,
            unfiltered_distances,
        }
    }

    /// Merge the current [UnprocessedKNNResultBatch] with another one, producing a new
    /// [UnprocessedKNNResultBatch].
    fn merge(self, other: Self) -> Result<Self> {
        let concat_array = concat(&[&self.row_array, &other.row_array])?;
        let mut probe_indices = self.probe_indices;
        probe_indices.extend(other.probe_indices);
        let mut distances = self.distances;
        distances.extend(other.distances);
        let mut unfiltered_probe_indices = self.unfiltered_probe_indices;
        unfiltered_probe_indices.extend(other.unfiltered_probe_indices);
        let mut unfiltered_distances = self.unfiltered_distances;
        unfiltered_distances.extend(other.unfiltered_distances);

        Ok(Self {
            row_array: concat_array.as_struct().clone(),
            probe_indices,
            distances,
            unfiltered_probe_indices,
            unfiltered_distances,
        })
    }
}

/// Reorganize [UnprocessedKNNResultBatch] for easier processing. The main goal is to group KNN results by
/// probe row index. There is an iterator implementation [KNNProbeResultIterator] that yields
/// [KNNProbeResult] for each probe row in order.
struct KNNResultArray {
    /// The KNN result batches produced by probing the spatial index with a probe batch
    array: StructArray,
    /// Distance for each KNN result row
    distances: Vec<f64>,
    /// Index for each probe row, this must be strictly increasing.
    indices: Vec<usize>,
    /// Offsets to split the batches into groups per probe row. It is always of length
    /// `indices.len() + 1`.
    offsets: Vec<usize>,
    /// Indices for each unfiltered probe row, This is a superset of `indices`.
    /// This must be strictly increasing.
    unfiltered_indices: Vec<usize>,
    /// Distances for each unfiltered KNN result row. This is a superset of `distances`.
    unfiltered_distances: Vec<f64>,
    /// Offsets to split the unfiltered distances into groups per probe row. It is always of length
    /// `unfiltered_indices.len() + 1`.
    unfiltered_offsets: Vec<usize>,
}

impl KNNResultArray {
    fn new(unprocessed_batch: UnprocessedKNNResultBatch) -> Self {
        let UnprocessedKNNResultBatch {
            row_array,
            probe_indices,
            distances,
            unfiltered_probe_indices,
            unfiltered_distances,
            ..
        } = unprocessed_batch;

        assert_eq!(row_array.len(), probe_indices.len());
        assert_eq!(probe_indices.len(), distances.len());
        assert_eq!(unfiltered_probe_indices.len(), unfiltered_distances.len());
        assert!(probe_indices.len() <= unfiltered_probe_indices.len());

        let (offsets, indices) = compute_range_encoding(probe_indices);
        let (unfiltered_offsets, unfiltered_indices) =
            compute_range_encoding(unfiltered_probe_indices);

        // The iterator implementation relies on `indices` being an in-order subsequence
        // of `unfiltered_indices`.
        debug_assert!({
            let mut j = 0;
            let mut ok = true;
            for &g in &indices {
                while j < unfiltered_indices.len() && unfiltered_indices[j] < g {
                    j += 1;
                }
                if j >= unfiltered_indices.len() || unfiltered_indices[j] != g {
                    ok = false;
                    break;
                }
            }
            ok
        });
        Self {
            array: row_array,
            distances,
            indices,
            offsets,
            unfiltered_indices,
            unfiltered_distances,
            unfiltered_offsets,
        }
    }
}

/// Compute range-encoding offsets and unique indices for a non-decreasing index list.
///
/// This collapses repeated indices while producing offsets that mark the contiguous
/// ranges for each unique index. The returned offsets always have length
/// `unique_indices.len() + 1`.
///
/// Example:
/// - Input indices: `[0, 0, 2, 2, 2, 5]`
/// - Output offsets: `[0, 2, 5, 6]`
/// - Output unique indices: `[0, 2, 5]`
fn compute_range_encoding(mut indices: Vec<usize>) -> (Vec<usize>, Vec<usize>) {
    let mut offsets = Vec::with_capacity(indices.len() + 1);
    offsets.push(0);
    if indices.is_empty() {
        return (offsets, Vec::new());
    }

    let mut prev = indices[0];
    let mut pos = 1;
    for i in 1..indices.len() {
        if indices[i] != prev {
            assert!(indices[i] > prev, "indices must be non-decreasing");
            offsets.push(i);
            indices[pos] = indices[i];
            pos += 1;
        }
        prev = indices[i];
    }
    offsets.push(indices.len());
    indices.truncate(pos);
    (offsets, indices)
}

/// KNNProbeResult represents a unified view for the KNN results for a single probe row.
/// The KNN results can be from a spilled batch or an ingested batch. This intermediate
/// data structure is for working with both spilled and ingested KNN results uniformly.
///
/// KNNProbeResult can also be used to represent KNN results for a probe row that has
/// no filtered results. In this case, the `row_range` will be an empty range, and the
/// `distances` will be an empty slice.
struct KNNProbeResult<'a> {
    /// Index of the probe row
    probe_row_index: usize,
    /// Range of KNN result rows in the implicitly referenced StructArray. The referenced
    /// StructArray only contains filtered results.
    row_range: Range<usize>,
    /// Distances for each KNN result row
    distances: &'a [f64],
    /// Distances for each unfiltered result row. Some of the results were filtered so they
    /// do not appear in the StructArray, but we still need the distances of all unfiltered
    /// results to correctly compute the top-K distances before the filtering.
    unfiltered_distances: &'a [f64],
}

impl<'a> KNNProbeResult<'a> {
    fn new(
        probe_row_index: usize,
        row_range: Range<usize>,
        distances: &'a [f64],
        unfiltered_distances: &'a [f64],
    ) -> Self {
        assert_eq!(row_range.len(), distances.len());
        // Please note that we don't have `unfiltered_distances.len() >= distances.len()` here.
        // We may have ties in `distances`, which may exceed K even after filtering.
        // `unfiltered_distances` does not include distances that are tied with the K-th distance.
        Self {
            probe_row_index,
            row_range,
            distances,
            unfiltered_distances,
        }
    }
}

/// Iterator over [KNNProbeResult] in a [KNNResultArray]
struct KNNProbeResultIterator<'a> {
    array: &'a KNNResultArray,
    unfiltered_pos: usize,
    pos: usize,
}

impl KNNProbeResultIterator<'_> {
    fn new(array: &KNNResultArray) -> KNNProbeResultIterator<'_> {
        KNNProbeResultIterator {
            array,
            unfiltered_pos: 0,
            pos: 0,
        }
    }
}

impl<'a> Iterator for KNNProbeResultIterator<'a> {
    type Item = KNNProbeResult<'a>;

    /// This iterator yields KNNProbeResult for each probe row in the [KNNResultArray].
    /// Given that the [KNNResultArray::indices] is strictly increasing,
    /// The [KNNProbeResult] it yields has strictly increasing [KNNProbeResult::probe_row_index].
    fn next(&mut self) -> Option<Self::Item> {
        if self.unfiltered_pos >= self.array.unfiltered_indices.len() {
            return None;
        }

        let unfiltered_start = self.array.unfiltered_offsets[self.unfiltered_pos];
        let unfiltered_end = self.array.unfiltered_offsets[self.unfiltered_pos + 1];
        let unfiltered_index = self.array.unfiltered_indices[self.unfiltered_pos];

        let start = self.array.offsets[self.pos];
        let index = if self.pos >= self.array.indices.len() {
            // All filtered results have been consumed.
            usize::MAX
        } else {
            self.array.indices[self.pos]
        };

        assert!(index >= unfiltered_index);

        let result = if index == unfiltered_index {
            // This probe row has filtered results
            let end = self.array.offsets[self.pos + 1];
            let row_range = start..end;
            let distances = &self.array.distances[start..end];
            let unfiltered_distances =
                &self.array.unfiltered_distances[unfiltered_start..unfiltered_end];
            self.pos += 1;
            KNNProbeResult::new(index, row_range, distances, unfiltered_distances)
        } else {
            // This probe row has no filtered results
            KNNProbeResult::new(
                unfiltered_index,
                start..start,
                &[],
                &self.array.unfiltered_distances[unfiltered_start..unfiltered_end],
            )
        };

        self.unfiltered_pos += 1;
        Some(result)
    }
}

/// Access arrays in a spilled KNN result batch. Provides easy access to KNN results of
/// probe rows as [KNNProbeResult].
struct SpilledBatchArrays {
    indices: SpilledBatchIndexArray,
    distances: Float64Array,
    unfiltered_distances: Float64Array,
    offsets: OffsetBuffer<i32>,
    unfiltered_offsets: OffsetBuffer<i32>,
    rows: StructArray,
}

impl SpilledBatchArrays {
    fn new(batch: &RecordBatch) -> Self {
        let index_col = batch
            .column(0)
            .as_primitive::<arrow::datatypes::UInt64Type>();

        let unfiltered_dist_list_array = batch.column(2).as_list::<i32>();
        let unfiltered_offset = unfiltered_dist_list_array.offsets();
        let unfiltered_distances = unfiltered_dist_list_array
            .values()
            .as_primitive::<Float64Type>();

        let row_and_dist_list_array = batch.column(1).as_list::<i32>();
        let offsets = row_and_dist_list_array.offsets();
        let row_and_dist_array = row_and_dist_list_array.values().as_struct();
        let dist_array = row_and_dist_array.column(1).as_primitive::<Float64Type>();

        let rows = row_and_dist_array.column(0).as_struct();

        Self {
            indices: SpilledBatchIndexArray::new(index_col.clone()),
            distances: dist_array.clone(),
            unfiltered_distances: unfiltered_distances.clone(),
            offsets: offsets.clone(),
            unfiltered_offsets: unfiltered_offset.clone(),
            rows: rows.clone(),
        }
    }

    /// Get [KNNProbeResult] for the given probe row index inside the spilled batch.
    /// The `row_idx` must be within the range of indices in this spilled batch.
    fn get_probe_result(&self, row_idx: usize) -> KNNProbeResult<'_> {
        let indices = self.indices.array.values().as_ref();
        let unfiltered_offsets = self.unfiltered_offsets.as_ref();
        let unfiltered_start = unfiltered_offsets[row_idx] as usize;
        let unfiltered_end = unfiltered_offsets[row_idx + 1] as usize;
        let unfiltered_distances = self.unfiltered_distances.values().as_ref();
        let offsets = self.offsets.as_ref();
        let start = offsets[row_idx] as usize;
        let end = offsets[row_idx + 1] as usize;
        let distances = self.distances.values().as_ref();
        KNNProbeResult::new(
            indices[row_idx] as usize,
            start..end,
            &distances[start..end],
            &unfiltered_distances[unfiltered_start..unfiltered_end],
        )
    }
}

/// Index array with a cursor for keeping track of the progress of iterating over a
/// spilled batch.
struct SpilledBatchIndexArray {
    array: UInt64Array,
    pos: usize,
}

struct AdvanceToResult {
    skipped_range: Range<usize>,
    found_target: HasFoundIndex,
}

enum HasFoundIndex {
    Found,
    NotFound { should_load_next_batch: bool },
}

impl SpilledBatchIndexArray {
    fn new(array: UInt64Array) -> Self {
        // Values in the index array should be strictly increasing.
        let values = array.values().as_ref();
        for i in 1..values.len() {
            assert!(values[i] > values[i - 1]);
        }

        Self { array, pos: 0 }
    }

    /// Advance the cursor to target index. The `target` is expected to be monotonically increasing
    /// across calls. We still tolerate the case where `target` is smaller than the current position,
    /// in which case we simply return [HasFoundIndex::NotFound].
    ///
    /// Please note that once a `target` is found, the cursor is advanced to the next position.
    /// Advancing to the same `target` again will yield [HasFoundIndex::NotFound].
    fn advance_to(&mut self, target: usize) -> AdvanceToResult {
        let values = self.array.values().as_ref();
        let begin_pos = self.pos;

        // Directly jump to the end if target is larger than the last value, and signal the
        // caller that we should load the next batch.
        if values.last().is_none_or(|last| (*last as usize) < target) {
            self.pos = values.len();
            return AdvanceToResult {
                skipped_range: begin_pos..self.pos,
                found_target: HasFoundIndex::NotFound {
                    should_load_next_batch: true,
                },
            };
        }

        // Iterate over the array from current position, until we hit or exceed target.
        while self.pos < values.len() {
            let value = values[self.pos] as usize;
            if value <= target {
                self.pos += 1;
                if value == target {
                    return AdvanceToResult {
                        skipped_range: begin_pos..self.pos,
                        found_target: HasFoundIndex::Found,
                    };
                }
            } else {
                return AdvanceToResult {
                    skipped_range: begin_pos..self.pos,
                    found_target: HasFoundIndex::NotFound {
                        should_load_next_batch: false,
                    },
                };
            }
        }

        // Reached the end without finding target.
        AdvanceToResult {
            skipped_range: begin_pos..self.pos,
            found_target: HasFoundIndex::NotFound {
                should_load_next_batch: false,
            },
        }
    }
}

/// KNNResultsMerger handles the merging of KNN "nearest so far" results from multiple partitions.
/// It maintains spill files to store intermediate results.
pub struct KNNResultsMerger {
    k: usize,
    include_tie_breaker: bool,
    /// Schema for the intermediate spill files
    spill_schema: SchemaRef,
    /// Runtime env
    runtime_env: Arc<RuntimeEnv>,
    /// Spill compression
    spill_compression: SpillCompression,
    /// Spill metrics
    spill_metrics: SpillMetrics,
    /// Internal state
    state: MergerState,
}

struct MergerState {
    /// File containing results from previous (0..N-1) partitions
    previous_file: Option<RefCountedTempFile>,
    /// Reader for previous file
    previous_reader: Option<RecordBatchSpillReader>,
    /// Spill writer for current (0..N) partitions
    current_writer: Option<RecordBatchSpillWriter>,
    /// Spilled batches loaded from previous file
    spilled_batches: Vec<SpilledBatchArrays>,
    /// Builder for merged KNN result batches or spilled batches
    batch_builder: KNNResultBatchBuilder,
    /// Unprocessed tail KNN results from the last ingested batch
    unprocessed_tail: Option<UnprocessedKNNResultBatch>,
}

impl KNNResultsMerger {
    pub fn try_new(
        k: usize,
        include_tie_breaker: bool,
        target_batch_size: usize,
        runtime_env: Arc<RuntimeEnv>,
        spill_compression: SpillCompression,
        result_schema: SchemaRef,
        spill_metrics: SpillMetrics,
    ) -> Result<Self> {
        let spill_schema = create_spill_schema(Arc::clone(&result_schema));
        let batch_builder =
            KNNResultBatchBuilder::new(Arc::clone(&result_schema), target_batch_size);

        let writer = RecordBatchSpillWriter::try_new(
            runtime_env.clone(),
            spill_schema.clone(),
            "knn_spill",
            spill_compression,
            spill_metrics.clone(),
            None,
        )?;
        Ok(Self {
            k,
            include_tie_breaker,
            spill_schema,
            runtime_env,
            spill_compression,
            spill_metrics,
            state: MergerState {
                previous_file: None,
                previous_reader: None,
                current_writer: Some(writer),
                spilled_batches: Vec::new(),
                batch_builder,
                unprocessed_tail: None,
            },
        })
    }

    pub fn rotate(&mut self, probing_last_index: bool) -> Result<()> {
        self.state.previous_file = self
            .state
            .current_writer
            .take()
            .map(|w| w.finish())
            .transpose()?;
        self.state.previous_reader = None;
        assert!(self.state.unprocessed_tail.is_none());
        assert!(self.state.batch_builder.is_empty());
        self.state.spilled_batches.clear();

        if let Some(file) = &self.state.previous_file {
            self.state.previous_reader = Some(RecordBatchSpillReader::try_new(file)?);
        }

        if !probing_last_index {
            self.state.current_writer = Some(RecordBatchSpillWriter::try_new(
                self.runtime_env.clone(),
                self.spill_schema.clone(),
                "knn_spill",
                self.spill_compression,
                self.spill_metrics.clone(),
                None,
            )?);
        }

        Ok(())
    }

    pub fn ingest(
        &mut self,
        batch: RecordBatch,
        probe_indices: Vec<usize>,
        distances: Vec<f64>,
        unfiltered_probe_indices: Vec<usize>,
        unfiltered_distances: Vec<f64>,
    ) -> Result<Option<RecordBatch>> {
        let row_array = StructArray::from(batch);
        let ingested_batch = UnprocessedKNNResultBatch::new(
            row_array,
            probe_indices,
            distances,
            unfiltered_probe_indices,
            unfiltered_distances,
        );
        let unprocessed_batch = if let Some(tail) = self.state.unprocessed_tail.take() {
            tail.merge(ingested_batch)?
        } else {
            ingested_batch
        };

        let knn_result_array = KNNResultArray::new(unprocessed_batch);
        let knn_query_result_iterator = KNNProbeResultIterator::new(&knn_result_array);

        let mut prev_result_opt: Option<KNNProbeResult<'_>> = None;
        for result in knn_query_result_iterator {
            // Only the previous result is guaranteed to be complete.
            if let Some(result) = prev_result_opt {
                self.merge_and_append_result(&result)?;
            }

            prev_result_opt = Some(result);
        }

        // Assemble this batch. Write to spill file or produce output batch.
        let result_batch_opt = self.flush_merged_batch(Some(&knn_result_array))?;

        // Prepare for ingesting the next batch
        if let Some(unprocessed_result) = prev_result_opt {
            self.state.unprocessed_tail = Some(UnprocessedKNNResultBatch::new_unprocessed_tail(
                unprocessed_result,
                &knn_result_array.array,
            ));
        }

        Ok(result_batch_opt)
    }

    /// Flushes any pending buffered probe index at the end of a probe batch iterator.
    ///
    /// This is used to emit the final probe index that may have been kept buffered because
    /// it could continue in the next produced slice.
    ///
    /// Returns `Ok(Some(batch))` at most once per pending buffered index; if there is nothing
    /// pending (or results are being spilled to disk for non-final indexed partitions), returns
    /// `Ok(None)`.
    pub fn produce_batch_until(
        &mut self,
        end_index_exclusive: usize,
    ) -> Result<Option<RecordBatch>> {
        // Consume and process any unprocessed tail from previous ingested batch
        let tail_batch_opt = if let Some(tail) = self.state.unprocessed_tail.take() {
            let knn_result_array = KNNResultArray::new(tail);
            let knn_query_result_iterator = KNNProbeResultIterator::new(&knn_result_array);
            for result in knn_query_result_iterator {
                self.merge_and_append_result(&result)?;
            }
            self.flush_merged_batch(Some(&knn_result_array))?
        } else {
            None
        };

        // Load spilled batches up to end_index_exclusive, if there's any.
        let spilled_batch_opt = if end_index_exclusive > 0 {
            let end_target_idx = end_index_exclusive - 1;
            // `end_target_idx` might have already been loaded before, but that's fine. The following operation
            // will be a no-op in that case.
            if let Some((batch_idx, row_idx)) = self.load_spilled_batches_up_to(end_target_idx)? {
                let loaded_range = row_idx..(row_idx + 1);
                self.append_spilled_results_in_range(batch_idx, &loaded_range);
            }
            self.flush_merged_batch(None)?
        } else {
            None
        };

        match (tail_batch_opt, spilled_batch_opt) {
            (Some(batch), None) | (None, Some(batch)) => Ok(Some(batch)),
            (None, None) => Ok(None),
            (Some(tail_batch), Some(spilled_batch)) => {
                let result_batch =
                    concat_batches(tail_batch.schema_ref(), [&tail_batch, &spilled_batch])?;
                Ok(Some(result_batch))
            }
        }
    }

    fn merge_and_append_result(&mut self, result: &KNNProbeResult<'_>) -> Result<()> {
        if let Some((spilled_batch_idx, row_idx)) =
            self.load_spilled_batches_up_to(result.probe_row_index)?
        {
            let spilled_batch_array = &self.state.spilled_batches[spilled_batch_idx];
            let spilled_result = spilled_batch_array.get_probe_result(row_idx);
            self.state.batch_builder.merge_and_append(
                &spilled_result,
                spilled_batch_idx,
                result,
                self.k,
                self.include_tie_breaker,
            );
        } else {
            // No spilled results for this index
            self.state
                .batch_builder
                .append(result, RowSelector::FromIngested { row_idx: 0 });
        }
        Ok(())
    }

    /// Load spilled batches until we find the target index, or exhaust all spilled batches.
    /// Returns the (batch_idx, row_idx) of the found target index within the spilled batches,
    /// or None if the target index is not found in any spilled batch.
    fn load_spilled_batches_up_to(&mut self, target_idx: usize) -> Result<Option<(usize, usize)>> {
        loop {
            if !self.state.spilled_batches.is_empty() {
                let batch_idx = self.state.spilled_batches.len() - 1;
                let spilled_batch = &mut self.state.spilled_batches[batch_idx];

                let res = spilled_batch.indices.advance_to(target_idx);

                match res.found_target {
                    HasFoundIndex::Found => {
                        // Found within current batch
                        let row_idx = res.skipped_range.end - 1;
                        self.append_spilled_results_in_range(
                            batch_idx,
                            &(res.skipped_range.start..row_idx),
                        );
                        return Ok(Some((batch_idx, row_idx)));
                    }
                    HasFoundIndex::NotFound {
                        should_load_next_batch,
                    } => {
                        self.append_spilled_results_in_range(batch_idx, &res.skipped_range);
                        if !should_load_next_batch {
                            // Not found, but no need to load the next batch
                            return Ok(None);
                        }
                    }
                }
            }

            // Load next batch
            let Some(prev_reader) = self.state.previous_reader.as_mut() else {
                return Ok(None);
            };
            let Some(batch) = prev_reader.next_batch() else {
                return Ok(None);
            };
            let batch = batch?;
            self.state
                .spilled_batches
                .push(SpilledBatchArrays::new(&batch));
        }
    }

    fn append_spilled_results_in_range(&mut self, batch_idx: usize, row_range: &Range<usize>) {
        let spilled_batch_array = &self.state.spilled_batches[batch_idx];
        for row_idx in row_range.clone() {
            let spilled_result = spilled_batch_array.get_probe_result(row_idx);
            self.state.batch_builder.append(
                &spilled_result,
                RowSelector::FromSpilled {
                    batch_idx,
                    row_idx: 0,
                },
            );
        }
    }

    fn flush_merged_batch(
        &mut self,
        knn_result_array: Option<&KNNResultArray>,
    ) -> Result<Option<RecordBatch>> {
        let spilled_batches = self
            .state
            .spilled_batches
            .iter()
            .map(|b| &b.rows)
            .collect::<Vec<_>>();
        let ingested_array = knn_result_array.map(|a| &a.array);
        let batch_opt = match &mut self.state.current_writer {
            Some(writer) => {
                // Write to spill file
                if let Some(spilled_batch) = self
                    .state
                    .batch_builder
                    .build_spilled_batch(ingested_array, &spilled_batches)?
                {
                    writer.write_batch(spilled_batch)?;
                }
                None
            }
            None => {
                // Produce output batch
                self.state
                    .batch_builder
                    .build_result_batch(ingested_array, &spilled_batches)?
            }
        };

        // Keep only the last spilled batch, since we don't need earlier ones anymore.
        let num_batches = self.state.spilled_batches.len();
        if num_batches > 1 {
            self.state.spilled_batches.drain(0..num_batches - 1);
        }

        Ok(batch_opt)
    }
}

/// Builders for KNN merged result batches or spilled batches.
struct KNNResultBatchBuilder {
    spill_schema: SchemaRef,
    rows_inner_fields: Fields,
    capacity: usize,
    unfiltered_dist_array_builder: PrimitiveBuilder<Float64Type>,
    unfiltered_dist_offsets_builder: OffsetBufferBuilder<i32>,
    index_array_builder: PrimitiveBuilder<UInt64Type>,
    dist_array_builder: PrimitiveBuilder<Float64Type>,
    row_array_offsets_builder: OffsetBufferBuilder<i32>,
    rows_selector: Vec<RowSelector>,
    /// Scratch space for merging top-k distances
    top_k_distances: Vec<f64>,
    /// Scratch space for sorting row selectors by distance when merging KNN results
    row_selector_with_distance: Vec<(RowSelector, f64)>,
}

/// The source of a merged row in the final KNN result. It can be from either a spilled batch
/// or an ingested batch.
#[derive(Copy, Clone)]
enum RowSelector {
    FromSpilled { batch_idx: usize, row_idx: usize },
    FromIngested { row_idx: usize },
}

impl RowSelector {
    fn with_row_idx(&self, row_idx: usize) -> Self {
        match self {
            RowSelector::FromSpilled { batch_idx, .. } => RowSelector::FromSpilled {
                batch_idx: *batch_idx,
                row_idx,
            },
            RowSelector::FromIngested { .. } => RowSelector::FromIngested { row_idx },
        }
    }
}

impl KNNResultBatchBuilder {
    fn new(result_schema: SchemaRef, capacity: usize) -> Self {
        let spill_schema = create_spill_schema(Arc::clone(&result_schema));
        let rows_inner_fields = create_rows_inner_fields(&result_schema);
        let unfiltered_dist_array_builder = Float64Array::builder(capacity);
        let unfiltered_dist_offsets_builder = OffsetBufferBuilder::<i32>::new(capacity);
        let index_array_builder = UInt64Array::builder(capacity);
        let dist_array_builder = Float64Array::builder(capacity);
        let row_array_offsets_builder = OffsetBufferBuilder::<i32>::new(capacity);

        Self {
            spill_schema,
            rows_inner_fields,
            capacity,
            unfiltered_dist_array_builder,
            unfiltered_dist_offsets_builder,
            index_array_builder,
            dist_array_builder,
            row_array_offsets_builder,
            rows_selector: Vec::with_capacity(capacity),
            top_k_distances: Vec::new(),
            row_selector_with_distance: Vec::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.index_array_builder.is_empty()
    }

    fn append(&mut self, results: &KNNProbeResult<'_>, row_selector_template: RowSelector) {
        for (row_idx, dist) in results.row_range.clone().zip(results.distances.iter()) {
            self.rows_selector
                .push(row_selector_template.with_row_idx(row_idx));
            self.dist_array_builder.append_value(*dist);
        }

        self.row_array_offsets_builder
            .push_length(results.row_range.len());
        self.unfiltered_dist_array_builder
            .append_slice(results.unfiltered_distances);
        self.unfiltered_dist_offsets_builder
            .push_length(results.unfiltered_distances.len());
        self.index_array_builder
            .append_value(results.probe_row_index as u64);
    }

    fn merge_and_append(
        &mut self,
        spilled_results: &KNNProbeResult<'_>,
        spilled_batch_idx: usize,
        ingested_results: &KNNProbeResult<'_>,
        k: usize,
        include_tie_breaker: bool,
    ) {
        assert_eq!(
            spilled_results.probe_row_index,
            ingested_results.probe_row_index
        );

        merge_unfiltered_topk(
            k,
            spilled_results.unfiltered_distances,
            ingested_results.unfiltered_distances,
            &mut self.top_k_distances,
        );

        let num_kept_rows = self.append_merged_knn_probe_results(
            spilled_batch_idx,
            spilled_results,
            ingested_results,
            k,
            include_tie_breaker,
        );

        self.row_array_offsets_builder.push_length(num_kept_rows);
        self.unfiltered_dist_array_builder
            .append_slice(&self.top_k_distances);
        self.unfiltered_dist_offsets_builder
            .push_length(self.top_k_distances.len());
        self.index_array_builder
            .append_value(spilled_results.probe_row_index as u64);
    }

    /// Append top K row selectors and distances from `spillled_results` and `ingested_results` that are
    /// within `distance_threshold`.
    /// Returns the number of values inserted into the [`KNNResultArrayBuilders::rows_selector`] and
    /// [`KNNResultArrayBuilders::dist_array_builder`].
    fn append_merged_knn_probe_results(
        &mut self,
        spilled_batch_idx: usize,
        spilled_results: &KNNProbeResult<'_>,
        ingested_results: &KNNProbeResult<'_>,
        k: usize,
        include_tie_breaker: bool,
    ) -> usize {
        // Sort all distances from both spilled and ingested results
        let row_dists = &mut self.row_selector_with_distance;
        row_dists.clear();
        row_dists.reserve(spilled_results.distances.len() + ingested_results.distances.len());

        for (row_idx, dist) in spilled_results
            .row_range
            .clone()
            .zip(spilled_results.distances.iter())
        {
            row_dists.push((
                RowSelector::FromSpilled {
                    batch_idx: spilled_batch_idx,
                    row_idx,
                },
                *dist,
            ));
        }
        for (row_idx, dist) in ingested_results
            .row_range
            .clone()
            .zip(ingested_results.distances.iter())
        {
            row_dists.push((RowSelector::FromIngested { row_idx }, *dist));
        }

        truncate_row_selectors_to_top_k(row_dists, &self.top_k_distances, k, include_tie_breaker);
        for (row_selector, dist) in row_dists.iter() {
            self.rows_selector.push(*row_selector);
            self.dist_array_builder.append_value(*dist);
        }
        row_dists.len()
    }

    fn build_spilled_batch(
        &mut self,
        ingested_results: Option<&StructArray>,
        spilled_results: &[&StructArray],
    ) -> Result<Option<RecordBatch>> {
        if self.index_array_builder.is_empty() {
            return Ok(None);
        }

        // index column: UInt64
        let index_array = Arc::new(self.index_array_builder.finish());

        // rows column: List<Struct<row: Struct<...>, dist: Float64>>
        let rows_array = interleave_spill_and_ingested_rows(
            ingested_results,
            spilled_results,
            &self.rows_selector,
        )?;
        self.rows_selector.clear();
        let dist_array = Arc::new(self.dist_array_builder.finish());
        let row_array_offsets_builder = std::mem::replace(
            &mut self.row_array_offsets_builder,
            OffsetBufferBuilder::<i32>::new(self.capacity),
        );
        let row_offsets = row_array_offsets_builder.finish();
        let row_dist_array = StructArray::try_new(
            self.rows_inner_fields.clone(),
            vec![rows_array, dist_array],
            None,
        )?;
        let row_dist_item_field = Arc::new(Field::new(
            "item",
            DataType::Struct(self.rows_inner_fields.clone()),
            false,
        ));
        let rows_list_array = ListArray::try_new(
            row_dist_item_field,
            row_offsets,
            Arc::new(row_dist_array),
            None,
        )?;

        // unfiltered_dists column: List<Float64>
        let unfiltered_dist_array = Arc::new(self.unfiltered_dist_array_builder.finish());
        let unfiltered_dist_offsets_builder = std::mem::replace(
            &mut self.unfiltered_dist_offsets_builder,
            OffsetBufferBuilder::<i32>::new(self.capacity),
        );
        let unfiltered_offsets = unfiltered_dist_offsets_builder.finish();
        let unfiltered_field = Arc::new(Field::new("item", DataType::Float64, false));
        let unfiltered_list_array = ListArray::try_new(
            unfiltered_field,
            unfiltered_offsets,
            unfiltered_dist_array,
            None,
        )?;

        Ok(Some(RecordBatch::try_new(
            self.spill_schema.clone(),
            vec![
                index_array,
                Arc::new(rows_list_array),
                Arc::new(unfiltered_list_array),
            ],
        )?))
    }

    fn build_result_batch(
        &mut self,
        ingested_results: Option<&StructArray>,
        spilled_results: &[&StructArray],
    ) -> Result<Option<RecordBatch>> {
        if self.index_array_builder.is_empty() {
            return Ok(None);
        }

        // Reset builders for building columns required by spilled batches. Building these columns seems to be wasted work
        // when we only need to produce result batches, but it simplifies the code significantly and the performance impact is minimal.
        self.index_array_builder = UInt64Array::builder(self.capacity);
        self.dist_array_builder = Float64Array::builder(self.capacity);
        self.row_array_offsets_builder = OffsetBufferBuilder::<i32>::new(self.capacity);
        self.unfiltered_dist_array_builder = Float64Array::builder(self.capacity);
        self.unfiltered_dist_offsets_builder = OffsetBufferBuilder::<i32>::new(self.capacity);

        // Build rows StructArray based on rows_selector
        if self.rows_selector.is_empty() {
            return Ok(None);
        }
        let rows_array = interleave_spill_and_ingested_rows(
            ingested_results,
            spilled_results,
            &self.rows_selector,
        )?;
        self.rows_selector.clear();

        let struct_array = rows_array.as_struct();

        Ok(Some(RecordBatch::from(struct_array.clone())))
    }
}

/// Create schema for spilled intermediate KNN results. The schema includes:
/// - index: UInt64
/// - rows: List<Struct<row: Struct<...>, dist: Float64>>
/// - unfiltered_dists: List<Float64> (top-K unfiltered distances so far)
fn create_spill_schema(result_schema: SchemaRef) -> SchemaRef {
    let index_field = Field::new("index", DataType::UInt64, false);
    let rows_inner_fields = create_rows_inner_fields(&result_schema);
    let row_dist_item_field = Field::new("item", DataType::Struct(rows_inner_fields), false);
    let rows_field = Field::new("rows", DataType::List(Arc::new(row_dist_item_field)), false);
    let unfiltered_dists_field = Field::new(
        "unfiltered_dists",
        DataType::List(Arc::new(Field::new("item", DataType::Float64, false))),
        false,
    );
    Arc::new(Schema::new(vec![
        index_field,
        rows_field,
        unfiltered_dists_field,
    ]))
}

fn create_rows_inner_fields(result_schema: &Schema) -> Fields {
    let row_field = Field::new(
        "row",
        DataType::Struct(result_schema.fields().clone()),
        false,
    );
    let dist_field = Field::new("dist", DataType::Float64, false);
    vec![row_field, dist_field].into()
}

fn interleave_spill_and_ingested_rows(
    ingested_results: Option<&StructArray>,
    spilled_results: &[&StructArray],
    rows_selector: &[RowSelector],
) -> Result<ArrayRef> {
    // Build rows StructArray based on rows_selector
    let ingested_array_index = spilled_results.len();
    let mut indices = Vec::with_capacity(rows_selector.len());
    for selector in rows_selector {
        match selector {
            RowSelector::FromSpilled { batch_idx, row_idx } => {
                indices.push((*batch_idx, *row_idx));
            }
            RowSelector::FromIngested { row_idx } => {
                if ingested_results.is_none() {
                    return sedona_internal_err!(
                        "Ingested results array is None when trying to access ingested rows"
                    );
                }
                indices.push((ingested_array_index, *row_idx));
            }
        }
    }

    let mut results_arrays: Vec<&dyn Array> = Vec::with_capacity(ingested_array_index + 1);
    for spilled_array in spilled_results {
        results_arrays.push(spilled_array);
    }
    if let Some(ingested_results) = ingested_results {
        results_arrays.push(ingested_results);
    }
    let rows_array = interleave(&results_arrays, &indices)?;
    Ok(rows_array)
}

fn merge_unfiltered_topk(k: usize, prev: &[f64], new: &[f64], top_k: &mut Vec<f64>) {
    top_k.clear();
    if k == 0 {
        return;
    }
    top_k.reserve(prev.len() + new.len());
    top_k.extend_from_slice(prev);
    top_k.extend_from_slice(new);

    // Keep only the K smallest distances, sorted.
    if top_k.len() > k {
        let kth = k - 1;
        top_k.select_nth_unstable_by(kth, |a, b| a.total_cmp(b));
        top_k.truncate(k);
    }
    top_k.sort_by(|a, b| a.total_cmp(b));
}

fn truncate_row_selectors_to_top_k(
    row_dist_vec: &mut Vec<(RowSelector, f64)>,
    top_k_distances: &[f64],
    k: usize,
    include_tie_breaker: bool,
) {
    let Some(kth_distance) = top_k_distances.last() else {
        row_dist_vec.clear();
        return;
    };

    let distance_threshold = if include_tie_breaker {
        // The distance threshold is slightly looser when including tie breakers, please
        // refer to `SpatialIndex::query_knn` for more details.
        *kth_distance + DISTANCE_TOLERANCE
    } else {
        *kth_distance
    };

    row_dist_vec.sort_unstable_by(|(_, l_dist), (_, r_dist)| l_dist.total_cmp(r_dist));

    // Keep only the row selectors within distance_threshold
    let mut kept_rows = 0;
    for (_, dist) in row_dist_vec.iter() {
        if kept_rows >= k && !include_tie_breaker {
            break;
        }
        if *dist <= distance_threshold {
            kept_rows += 1;
        } else {
            break;
        }
    }

    row_dist_vec.truncate(kept_rows);

    // If the last distance D in top_k_distances has N ties, and include_tie_breaker is false, we
    // need to make sure that the kept rows with distance D should not exceed N, otherwise we'll
    // incorrectly have extra rows kept.
    // To fix this, we need to count how many rows have distance equal to the last distance,
    // and make sure we only keep that many rows among the kept rows with that distance.
    if !include_tie_breaker {
        let last_distance = *kth_distance;
        let num_ties_in_topk = top_k_distances
            .iter()
            .rev()
            .take_while(|d| **d == last_distance)
            .count();

        let num_ties_in_kept = row_dist_vec
            .iter()
            .rev()
            .take_while(|(_, d)| *d == last_distance)
            .count();

        if num_ties_in_kept > num_ties_in_topk {
            let to_remove = num_ties_in_kept - num_ties_in_topk;
            let new_len = row_dist_vec.len() - to_remove;
            row_dist_vec.truncate(new_len);
        }
    }
}

#[cfg(test)]
mod test {
    use arrow::compute::take_record_batch;
    use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
    use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
    use rstest::rstest;

    use super::*;

    #[test]
    fn test_knn_results_array_iterator() {
        // KNNResultArray with 4 probe rows: P1000, P1001, P1002, P1004.
        // P1002 has no filtered results.
        let array = KNNResultArray::new(UnprocessedKNNResultBatch::new(
            StructArray::new_empty_fields(7, None),
            vec![1000, 1000, 1001, 1001, 1001, 1004, 1004],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            vec![
                1000, 1000, 1000, 1001, 1001, 1001, 1002, 1002, 1002, 1004, 1004, 1004,
            ],
            vec![1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 7.0, 8.0, 9.0, 6.0, 7.0, 8.0],
        ));

        let mut iter = KNNProbeResultIterator::new(&array);

        let res0 = iter.next().unwrap();
        assert_eq!(res0.probe_row_index, 1000);
        assert_eq!(res0.row_range, 0..2);
        assert_eq!(res0.distances, &[1.0, 2.0]);
        assert_eq!(res0.unfiltered_distances, &[1.0, 2.0, 3.0]);

        let res1 = iter.next().unwrap();
        assert_eq!(res1.probe_row_index, 1001);
        assert_eq!(res1.row_range, 2..5);
        assert_eq!(res1.distances, &[3.0, 4.0, 5.0]);
        assert_eq!(res1.unfiltered_distances, &[3.0, 4.0, 5.0]);

        let res2 = iter.next().unwrap();
        assert_eq!(res2.probe_row_index, 1002);
        assert_eq!(res2.row_range, 5..5);
        assert!(res2.distances.is_empty());
        assert_eq!(res2.unfiltered_distances, &[7.0, 8.0, 9.0]);

        let res3 = iter.next().unwrap();
        assert_eq!(res3.probe_row_index, 1004);
        assert_eq!(res3.row_range, 5..7);
        assert_eq!(res3.distances, &[6.0, 7.0]);
        assert_eq!(res3.unfiltered_distances, &[6.0, 7.0, 8.0]);

        assert!(iter.next().is_none());
    }

    #[test]
    fn test_knn_results_array_iterator_empty() {
        let array = KNNResultArray::new(UnprocessedKNNResultBatch::new(
            StructArray::new_empty_fields(0, None),
            vec![],
            vec![],
            vec![],
            vec![],
        ));

        let mut iter = KNNProbeResultIterator::new(&array);
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_knn_results_array_iterator_no_filtered() {
        let array = KNNResultArray::new(UnprocessedKNNResultBatch::new(
            StructArray::new_empty_fields(0, None),
            vec![],
            vec![],
            vec![0, 0, 0, 3, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        ));

        let mut iter = KNNProbeResultIterator::new(&array);

        let res0 = iter.next().unwrap();
        assert_eq!(res0.probe_row_index, 0);
        assert_eq!(res0.row_range, 0..0);
        assert!(res0.distances.is_empty());
        assert_eq!(res0.unfiltered_distances, &[1.0, 2.0, 3.0]);

        let res1 = iter.next().unwrap();
        assert_eq!(res1.probe_row_index, 3);
        assert_eq!(res1.row_range, 0..0);
        assert!(res1.distances.is_empty());
        assert_eq!(res1.unfiltered_distances, &[4.0, 5.0]);

        assert!(iter.next().is_none());
    }

    #[test]
    fn test_knn_results_array_iterator_all_kept() {
        let array = KNNResultArray::new(UnprocessedKNNResultBatch::new(
            StructArray::new_empty_fields(5, None),
            vec![0, 0, 0, 3, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![0, 0, 0, 3, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        ));

        let mut iter = KNNProbeResultIterator::new(&array);
        let res0 = iter.next().unwrap();
        assert_eq!(res0.probe_row_index, 0);
        assert_eq!(res0.row_range, 0..3);
        assert_eq!(res0.distances, &[1.0, 2.0, 3.0]);
        assert_eq!(res0.unfiltered_distances, &[1.0, 2.0, 3.0]);

        let res1 = iter.next().unwrap();
        assert_eq!(res1.probe_row_index, 3);
        assert_eq!(res1.row_range, 3..5);
        assert_eq!(res1.distances, &[4.0, 5.0]);
        assert_eq!(res1.unfiltered_distances, &[4.0, 5.0]);
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_knn_results_array_iterator_no_dup() {
        let indices = vec![0, 1, 3, 4, 6];
        let array = KNNResultArray::new(UnprocessedKNNResultBatch::new(
            StructArray::new_empty_fields(5, None),
            indices.clone(),
            vec![0.0, 1.0, 3.0, 4.0, 6.0],
            vec![0, 1, 2, 3, 4, 5, 6],
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ));

        let mut iter = KNNProbeResultIterator::new(&array);
        for k in 0..7 {
            let res0 = iter.next().unwrap();
            assert_eq!(res0.probe_row_index, k);
            assert_eq!(res0.unfiltered_distances, &[k as f64]);

            if let Ok(pos) = indices.binary_search(&k) {
                assert_eq!(res0.row_range, pos..(pos + 1));
                assert_eq!(res0.distances, &[k as f64]);
            } else {
                assert!(res0.row_range.is_empty());
                assert!(res0.distances.is_empty());
            }
        }

        assert!(iter.next().is_none());
    }

    #[test]
    fn test_spill_index_array_advance_to() {
        let mut arr = SpilledBatchIndexArray::new(UInt64Array::from(vec![1, 2, 3, 6, 8, 10]));

        let res = arr.advance_to(0);
        assert_eq!(res.skipped_range, 0..0);
        assert!(matches!(
            res.found_target,
            HasFoundIndex::NotFound {
                should_load_next_batch: false
            }
        ));

        let res = arr.advance_to(1);
        assert_eq!(res.skipped_range, 0..1);
        assert!(matches!(res.found_target, HasFoundIndex::Found));

        // Repeatedly advance to the same target won't move the cursor, and will return NotFound.
        let res = arr.advance_to(1);
        assert_eq!(res.skipped_range, 1..1);
        assert!(matches!(
            res.found_target,
            HasFoundIndex::NotFound {
                should_load_next_batch: false
            }
        ));

        let res = arr.advance_to(2);
        assert_eq!(res.skipped_range, 1..2);
        assert!(matches!(res.found_target, HasFoundIndex::Found));

        // Advance to a missing target within the array, indexes less than the target are skipped.
        // The cursor stops at the first index greater than the target.
        let res = arr.advance_to(4);
        assert_eq!(res.skipped_range, 2..3);
        assert!(matches!(
            res.found_target,
            HasFoundIndex::NotFound {
                should_load_next_batch: false
            }
        ));

        let res = arr.advance_to(6);
        assert_eq!(res.skipped_range, 3..4);
        assert!(matches!(res.found_target, HasFoundIndex::Found));

        let res = arr.advance_to(10);
        assert_eq!(res.skipped_range, 4..6);
        assert!(matches!(res.found_target, HasFoundIndex::Found));

        // Advance to a target larger than the last index, the cursor moves to the end,
        // and signals to load the next batch.
        let res = arr.advance_to(11);
        assert_eq!(res.skipped_range, 6..6);
        assert!(matches!(
            res.found_target,
            HasFoundIndex::NotFound {
                should_load_next_batch: true
            }
        ));
    }

    #[test]
    fn test_spill_index_array_advance_to_skip_all() {
        let mut arr = SpilledBatchIndexArray::new(UInt64Array::from(vec![1, 2, 3, 6, 8, 10]));

        let res = arr.advance_to(100);
        assert_eq!(res.skipped_range, 0..6);
        assert!(matches!(
            res.found_target,
            HasFoundIndex::NotFound {
                should_load_next_batch: true
            }
        ));
    }

    #[test]
    fn test_spill_index_array_advance_to_end() {
        let mut arr = SpilledBatchIndexArray::new(UInt64Array::from(vec![1, 2, 3, 6, 8, 10]));

        let res = arr.advance_to(3);
        assert_eq!(res.skipped_range, 0..3);
        assert!(matches!(res.found_target, HasFoundIndex::Found));

        // Advance to the end by specifying usize::MAX as target.
        let res = arr.advance_to(usize::MAX);
        assert_eq!(res.skipped_range, 3..6);
        assert!(matches!(
            res.found_target,
            HasFoundIndex::NotFound {
                should_load_next_batch: true
            }
        ));
    }

    #[test]
    fn test_spill_index_array_advance_empty() {
        let mut arr = SpilledBatchIndexArray::new(UInt64Array::from(Vec::<u64>::new()));

        let res = arr.advance_to(0);
        assert_eq!(res.skipped_range, 0..0);
        assert!(matches!(
            res.found_target,
            HasFoundIndex::NotFound {
                should_load_next_batch: true
            }
        ));
    }

    #[test]
    fn test_merge_unfiltered_topk() {
        let mut top_k = Vec::new();

        // Normal cases
        merge_unfiltered_topk(3, &[1.0, 3.0, 5.0], &[2.0, 4.0, 6.0], &mut top_k);
        assert_eq!(top_k, vec![1.0, 2.0, 3.0]);
        merge_unfiltered_topk(3, &[5.0, 3.0, 1.0], &[2.0, 6.0, 4.0], &mut top_k);
        assert_eq!(top_k, vec![1.0, 2.0, 3.0]);
        merge_unfiltered_topk(5, &[1.0, 3.0], &[2.0, 4.0, 5.0, 6.0], &mut top_k);
        assert_eq!(top_k, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        merge_unfiltered_topk(5, &[5.0, 3.0, 1.0], &[2.0, 6.0, 4.0], &mut top_k);
        assert_eq!(top_k, vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        // k equals total number of distances
        merge_unfiltered_topk(6, &[5.0, 3.0, 1.0], &[2.0, 6.0, 4.0], &mut top_k);
        assert_eq!(top_k, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // k larger than total number of distances
        merge_unfiltered_topk(10, &[5.0, 3.0, 1.0], &[2.0, 6.0, 4.0], &mut top_k);
        assert_eq!(top_k, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // k is zero (usually this does not happen in practice)
        merge_unfiltered_topk(0, &[1.0, 3.0], &[2.0, 4.0], &mut top_k);
        assert_eq!(top_k, Vec::<f64>::new());
        merge_unfiltered_topk(0, &[], &[], &mut top_k);
        assert_eq!(top_k, Vec::<f64>::new());

        // one side is empty
        merge_unfiltered_topk(2, &[], &[2.0, 1.0], &mut top_k);
        assert_eq!(top_k, vec![1.0, 2.0]);
        merge_unfiltered_topk(2, &[2.0, 1.0], &[], &mut top_k);
        assert_eq!(top_k, vec![1.0, 2.0]);
    }

    fn create_dummy_row_selectors(dists: &[f64]) -> Vec<(RowSelector, f64)> {
        dists
            .iter()
            .enumerate()
            .map(|(i, d)| (RowSelector::FromIngested { row_idx: i }, *d))
            .collect()
    }

    fn count_dist(v: &[(RowSelector, f64)], target: f64) -> usize {
        v.iter().filter(|(_, d)| *d == target).count()
    }

    #[test]
    fn test_truncate_row_selectors_for_empty_unfiltered_top_k() {
        let mut row_dist_vec = create_dummy_row_selectors(&[1.0, 2.0, 3.0]);
        truncate_row_selectors_to_top_k(&mut row_dist_vec, &[], 3, false);
        assert!(row_dist_vec.is_empty());
    }

    #[test]
    fn test_truncate_row_selectors_no_dup() {
        // Keep at most K rows within distance threshold.
        let k = 3;
        let top_k_distances = vec![1.0, 2.0, 3.0];
        let mut row_dist_vec = create_dummy_row_selectors(&[3.0, 2.0, 4.0, 1.0, 5.0]);

        truncate_row_selectors_to_top_k(&mut row_dist_vec, &top_k_distances, k, false);

        assert_eq!(row_dist_vec.len(), 3);
        assert!(row_dist_vec.iter().all(|(_, d)| *d <= 3.0));
        assert_eq!(count_dist(&row_dist_vec, 3.0), 1);
    }

    #[test]
    fn test_truncate_row_selectors_handle_last_ties() {
        // top_k_distances has last distance 4.0 with only 1 tie.
        // Filtered results can contain more 4.0 rows than allowed; we must trim them.
        let k = 5;
        let top_k_distances = vec![1.0, 2.0, 3.0, 3.0, 4.0];
        let mut row_dist_vec = create_dummy_row_selectors(&[4.0, 1.0, 4.0, 2.0, 4.0, 10.0]);

        truncate_row_selectors_to_top_k(&mut row_dist_vec, &top_k_distances, k, false);

        assert!(row_dist_vec.iter().all(|(_, d)| *d <= 4.0));
        assert!(row_dist_vec.len() <= k);
        assert_eq!(count_dist(&row_dist_vec, 4.0), 1);
        assert_eq!(count_dist(&row_dist_vec, 1.0), 1);
        assert_eq!(count_dist(&row_dist_vec, 2.0), 1);

        // top_k_distances has last distance 4.0 with 2 ties.
        // If we keep more than 2 rows with 4.0, we must discard some from the tail.
        let k = 4;
        let top_k_distances = vec![1.0, 2.0, 4.0, 4.0];
        let mut row_dist_vec = create_dummy_row_selectors(&[4.0, 4.0, 4.0, 1.0]);

        truncate_row_selectors_to_top_k(&mut row_dist_vec, &top_k_distances, k, false);

        assert!(row_dist_vec.iter().all(|(_, d)| *d <= 4.0));
        assert!(row_dist_vec.len() <= k);
        assert_eq!(count_dist(&row_dist_vec, 4.0), 2);
        assert_eq!(count_dist(&row_dist_vec, 1.0), 1);

        // Keep fewer ties than in top_k_distances should not trigger any trimming.
        let k = 5;
        let top_k_distances = vec![1.0, 2.0, 3.0, 5.0, 5.0];
        let mut row_dist_vec = create_dummy_row_selectors(&[5.0, 1.0]);

        truncate_row_selectors_to_top_k(&mut row_dist_vec, &top_k_distances, k, false);

        assert_eq!(row_dist_vec.len(), 2);
        assert_eq!(count_dist(&row_dist_vec, 5.0), 1);
        assert_eq!(count_dist(&row_dist_vec, 1.0), 1);
    }

    #[test]
    fn test_truncate_row_selectors_include_tie_breakers() {
        let k = 3;
        let top_k_distances = vec![1.0, 2.0, 3.0];
        let tol_half = DISTANCE_TOLERANCE / 2.0;

        let mut row_dist_vec =
            create_dummy_row_selectors(&[3.0, 1.0, 3.0, 2.0, 3.0 + tol_half, 4.0]);
        truncate_row_selectors_to_top_k(&mut row_dist_vec, &top_k_distances, k, true);

        // Should keep all <= 3.0 + DISTANCE_TOLERANCE (i.e. not limited by k).
        assert!(row_dist_vec.len() > k);
        assert!(row_dist_vec
            .iter()
            .all(|(_, d)| *d <= 3.0 + DISTANCE_TOLERANCE));
        assert_eq!(count_dist(&row_dist_vec, 1.0), 1);
        assert_eq!(count_dist(&row_dist_vec, 2.0), 1);
        assert_eq!(count_dist(&row_dist_vec, 3.0), 2);
        assert_eq!(count_dist(&row_dist_vec, 3.0 + tol_half), 1);
    }

    #[derive(Clone, PartialEq, Debug)]
    struct FuzzTestKNNResult {
        query_id: usize,
        knn_objects: Vec<FuzzKNNResultObject>,
    }

    #[derive(Clone, PartialEq, Debug)]
    struct FuzzKNNResultObject {
        object_id: usize,
        distance: f64,
        is_kept: bool,
    }

    fn create_fuzz_test_data_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("query_id", DataType::UInt64, false),
            Field::new("object_id", DataType::UInt64, false),
        ]))
    }

    fn create_fuzz_test_data(
        k: usize,
        num_rows: usize,
        kept_prob: f64,
        rng: &mut StdRng,
    ) -> Vec<FuzzTestKNNResult> {
        let mut test_data = Vec::with_capacity(num_rows);
        let mut next_object_id = 0;
        for query_id in 0..num_rows {
            // Generate K objects
            let knn_objects = (next_object_id..next_object_id + k)
                .map(|object_id| FuzzKNNResultObject {
                    object_id,
                    distance: rng.random_range(1.0..10.0),
                    is_kept: rng.random_bool(kept_prob),
                })
                .collect::<Vec<FuzzKNNResultObject>>();
            next_object_id += k;

            test_data.push(FuzzTestKNNResult {
                query_id,
                knn_objects,
            });
        }
        test_data
    }

    fn partition_fuzz_test_data(
        test_data: &[FuzzTestKNNResult],
        num_partitions: usize,
        kept_prob: f64,
        rng: &mut StdRng,
    ) -> Vec<Vec<FuzzTestKNNResult>> {
        let mut partitions: Vec<Vec<FuzzTestKNNResult>> = vec![Vec::new(); num_partitions];
        let mut next_object_id = test_data
            .iter()
            .flat_map(|r| r.knn_objects.iter())
            .map(|o| o.object_id)
            .max()
            .unwrap_or(0)
            + 1;
        for result in test_data.iter() {
            // Split the knn_objects into partitions, randomly mix in some objects with large distances
            let distance_threshold = result
                .knn_objects
                .iter()
                .map(|o| o.distance)
                .reduce(f64::max)
                .unwrap_or(0.0);

            let k = result.knn_objects.len();
            if k == 0 {
                for partition in partitions.iter_mut() {
                    partition.push(FuzzTestKNNResult {
                        query_id: result.query_id,
                        knn_objects: Vec::new(),
                    });
                }
                continue;
            }

            let mut extended_knn_objects = result.knn_objects.clone();
            for _ in 0..((num_partitions - 1) * k) {
                extended_knn_objects.push(FuzzKNNResultObject {
                    object_id: next_object_id,
                    distance: distance_threshold + rng.random_range(1..10) as f64,
                    is_kept: rng.random_bool(kept_prob),
                });
                next_object_id += 1;
            }
            extended_knn_objects.shuffle(rng);

            for (part_idx, chunk) in extended_knn_objects.chunks(k).enumerate() {
                partitions[part_idx].push(FuzzTestKNNResult {
                    query_id: result.query_id,
                    knn_objects: chunk.to_vec(),
                });
            }
        }
        partitions
    }

    fn merge_partitioned_test_data(
        partitioned_data: &[Vec<FuzzTestKNNResult>],
    ) -> Vec<FuzzTestKNNResult> {
        let num_queries = partitioned_data[0].len();
        let mut merged_results = Vec::with_capacity(num_queries);
        for query_idx in 0..num_queries {
            let mut knn_objects = Vec::new();
            for partition in partitioned_data.iter() {
                knn_objects.extend_from_slice(&partition[query_idx].knn_objects);
            }
            merged_results.push(FuzzTestKNNResult {
                query_id: partitioned_data[0][query_idx].query_id,
                knn_objects,
            });
        }
        merged_results
    }

    fn compute_expected_results(
        test_data: &[FuzzTestKNNResult],
        k: usize,
        include_tie_breaker: bool,
    ) -> Vec<(usize, Vec<FuzzKNNResultObject>)> {
        let mut expected_results = Vec::with_capacity(test_data.len());
        for result in test_data.iter() {
            let mut knn_objects = result.knn_objects.clone();

            // Take top K objects first
            knn_objects.sort_by(|a, b| {
                a.distance
                    .total_cmp(&b.distance)
                    .then(a.object_id.cmp(&b.object_id))
            });
            if let Some(kth_distance) = knn_objects.get(k.saturating_sub(1)).map(|o| o.distance) {
                if include_tie_breaker {
                    let distance_threshold = kth_distance + DISTANCE_TOLERANCE;
                    knn_objects.retain(|o| o.distance <= distance_threshold);
                } else {
                    knn_objects.truncate(k);
                }
            } else {
                knn_objects.clear();
            }

            // Filter the results to only kept objects
            let kept_objects = knn_objects.into_iter().filter(|o| o.is_kept).collect();

            expected_results.push((result.query_id, kept_objects));
        }
        expected_results
    }

    fn is_fuzz_test_data_equivalent(
        test_data: &[FuzzTestKNNResult],
        partitioned_test_data: &[Vec<FuzzTestKNNResult>],
        k: usize,
        include_tie_breaker: bool,
    ) -> bool {
        let merged_partitioned_test_data = merge_partitioned_test_data(partitioned_test_data);
        let expected_results = compute_expected_results(test_data, k, include_tie_breaker);
        let partitioned_results =
            compute_expected_results(&merged_partitioned_test_data, k, include_tie_breaker);
        expected_results == partitioned_results
    }

    fn ingest_partitioned_fuzz_test_data(
        knn_result_spiller: &mut KNNResultsMerger,
        partitioned_test_data: &[Vec<FuzzTestKNNResult>],
        query_group_size: usize,
        batch_size: usize,
    ) -> Result<Vec<RecordBatch>> {
        let mut merged_record_batches = Vec::new();
        for (i_partition, partition) in partitioned_test_data.iter().enumerate() {
            if i_partition != 0 {
                let is_last_partition = i_partition == partitioned_test_data.len() - 1;
                knn_result_spiller.rotate(is_last_partition)?;
            }

            let mut start_offset = 0;
            for partition_chunk in partition.chunks(query_group_size) {
                let res_batches = ingest_fuzz_test_data_segment(
                    knn_result_spiller,
                    partition_chunk,
                    start_offset,
                    batch_size,
                )?;
                merged_record_batches.extend(res_batches);
                start_offset += partition_chunk.len();
            }

            if let Some(batch) = knn_result_spiller.produce_batch_until(start_offset)? {
                merged_record_batches.push(batch);
            }
        }
        Ok(merged_record_batches)
    }

    fn ingest_fuzz_test_data_segment(
        knn_result_spiller: &mut KNNResultsMerger,
        test_data: &[FuzzTestKNNResult],
        start_offset: usize,
        batch_size: usize,
    ) -> Result<Vec<RecordBatch>> {
        // Assemble the test_data into one RecordBatch
        let mut query_id_builder = UInt64Array::builder(test_data.len());
        let mut object_id_builder = UInt64Array::builder(test_data.len());
        let mut indices = Vec::new();
        let mut distances = Vec::new();
        let mut is_kept = Vec::new();
        for (idx, result) in test_data.iter().enumerate() {
            for obj in result.knn_objects.iter() {
                query_id_builder.append_value(result.query_id as u64);
                object_id_builder.append_value(obj.object_id as u64);
                indices.push(idx + start_offset);
                distances.push(obj.distance);
                is_kept.push(obj.is_kept);
            }
        }
        let query_id_array = Arc::new(query_id_builder.finish());
        let object_id_array = Arc::new(object_id_builder.finish());
        let schema = create_fuzz_test_data_schema();
        let knn_result_batch = RecordBatch::try_new(schema, vec![query_id_array, object_id_array])?;

        // Break the record batch into smaller batches and ingest them
        let mut merged_record_batches = Vec::new();
        for start in (0..knn_result_batch.num_rows()).step_by(batch_size) {
            let end = (start + batch_size).min(knn_result_batch.num_rows());
            let batch = knn_result_batch.slice(start, end - start);

            let unfiltered_distances = distances[start..end].to_vec();
            let unfiltered_indices = indices[start..end].to_vec();
            let is_kept_slice = &is_kept[start..end];

            // Find local indices for kept rows
            let kept_indices_within_batch: Vec<usize> = is_kept_slice
                .iter()
                .enumerate()
                .filter_map(|(i, &kept)| if kept { Some(i) } else { None })
                .collect();

            let kept_indices_array = UInt64Array::from(
                kept_indices_within_batch
                    .iter()
                    .map(|&i| i as u64)
                    .collect::<Vec<u64>>(),
            );
            let batch = take_record_batch(&batch, &kept_indices_array).unwrap();
            let filtered_distances: Vec<f64> = kept_indices_within_batch
                .iter()
                .map(|&i| unfiltered_distances[i])
                .collect();
            let filtered_indices = kept_indices_within_batch
                .iter()
                .map(|&i| unfiltered_indices[i])
                .collect::<Vec<usize>>();

            let res = knn_result_spiller.ingest(
                batch,
                filtered_indices,
                filtered_distances,
                unfiltered_indices,
                unfiltered_distances,
            )?;
            if let Some(res_batch) = res {
                merged_record_batches.push(res_batch);
            }
        }

        Ok(merged_record_batches)
    }

    fn assert_merged_knn_result_is_correct(
        batch: &RecordBatch,
        partitioned_test_data: &[Vec<FuzzTestKNNResult>],
        k: usize,
        include_tie_breaker: bool,
    ) {
        let merged_test_data = merge_partitioned_test_data(partitioned_test_data);
        let expected_results = compute_expected_results(&merged_test_data, k, include_tie_breaker);
        let mut expected_results: Vec<(u64, u64)> = expected_results
            .iter()
            .flat_map(|(query_id, objects)| {
                objects
                    .iter()
                    .map(move |obj| (*query_id as u64, obj.object_id as u64))
            })
            .collect();
        expected_results.sort();

        let mut actual_results: Vec<(u64, u64)> = Vec::new();
        let query_id_array = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let object_id_array = batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        for i in 0..batch.num_rows() {
            actual_results.push((query_id_array.value(i), object_id_array.value(i)));
        }
        actual_results.sort();

        assert_eq!(expected_results, actual_results);
    }

    #[allow(clippy::too_many_arguments)]
    fn fuzz_test_knn_results_merger(
        rng: &mut StdRng,
        num_rows: usize,
        num_partitions: usize,
        kept_prob: f64,
        k: usize,
        include_tie_breaker: bool,
        query_group_size: usize,
        target_batch_size: usize,
    ) -> Result<()> {
        assert!(num_partitions > 1);

        for _ in 0..10 {
            let test_data = create_fuzz_test_data(k, num_rows, kept_prob, rng);
            let partitioned_test_data =
                partition_fuzz_test_data(&test_data, num_partitions, kept_prob, rng);
            assert!(is_fuzz_test_data_equivalent(
                &test_data,
                &partitioned_test_data,
                k,
                include_tie_breaker
            ));

            fuzz_test_knn_results_merger_using_partitioned_data(
                &partitioned_test_data,
                k,
                include_tie_breaker,
                query_group_size,
                target_batch_size,
            )?;
        }

        Ok(())
    }

    fn fuzz_test_knn_results_merger_using_partitioned_data(
        partitioned_test_data: &[Vec<FuzzTestKNNResult>],
        k: usize,
        include_tie_breaker: bool,
        query_group_size: usize,
        target_batch_size: usize,
    ) -> Result<()> {
        let test_data_schema = create_fuzz_test_data_schema();
        let runtime_env = Arc::new(RuntimeEnv::default());
        let metrics_set = ExecutionPlanMetricsSet::new();
        let spill_metrics = SpillMetrics::new(&metrics_set, 0);
        let mut knn_results_merger = KNNResultsMerger::try_new(
            k,
            include_tie_breaker,
            target_batch_size,
            runtime_env,
            SpillCompression::Uncompressed,
            Arc::clone(&test_data_schema),
            spill_metrics,
        )?;

        let batches = ingest_partitioned_fuzz_test_data(
            &mut knn_results_merger,
            partitioned_test_data,
            query_group_size,
            target_batch_size,
        )?;
        let batch = concat_batches(&test_data_schema, batches.iter())?;
        assert_merged_knn_result_is_correct(&batch, partitioned_test_data, k, include_tie_breaker);
        Ok(())
    }

    #[rstest]
    fn test_knn_results_merger(
        #[values(1, 10, 13, 50, 51, 1000)] target_batch_size: usize,
        #[values(false, true)] include_tie_breaker: bool,
    ) {
        let mut rng = StdRng::seed_from_u64(target_batch_size as u64);
        fuzz_test_knn_results_merger(
            &mut rng,
            100,
            4,
            0.5,
            5,
            include_tie_breaker,
            30,
            target_batch_size,
        )
        .unwrap();
    }

    #[rstest]
    fn test_knn_results_merger_empty_query_side(#[values(1, 2, 3)] random_seed: u64) {
        let mut rng = StdRng::seed_from_u64(random_seed);
        fuzz_test_knn_results_merger(&mut rng, 0, 3, 1.0, 10, false, 100, 33).unwrap();
    }

    #[rstest]
    fn test_knn_results_merger_all_filtered(#[values(1, 2, 3)] random_seed: u64) {
        let mut rng = StdRng::seed_from_u64(random_seed);
        fuzz_test_knn_results_merger(&mut rng, 100, 3, 0.0, 10, false, 50, 33).unwrap();
    }

    #[test]
    fn test_knn_results_merger_no_knn_results() {
        let empty_test_data = (0..100)
            .map(|query_id| FuzzTestKNNResult {
                query_id,
                knn_objects: Vec::new(),
            })
            .collect::<Vec<FuzzTestKNNResult>>();
        let partitioned_test_data = vec![empty_test_data.clone(); 3];
        fuzz_test_knn_results_merger_using_partitioned_data(
            &partitioned_test_data,
            5,
            false,
            50,
            33,
        )
        .unwrap();
    }

    #[test]
    fn test_knn_results_merger_k_is_zero() {
        let empty_test_data = (0..100)
            .map(|query_id| FuzzTestKNNResult {
                query_id,
                knn_objects: Vec::new(),
            })
            .collect::<Vec<FuzzTestKNNResult>>();
        let partitioned_test_data = vec![empty_test_data.clone(); 3];
        fuzz_test_knn_results_merger_using_partitioned_data(
            &partitioned_test_data,
            0,
            false,
            50,
            33,
        )
        .unwrap();
    }

    #[rstest]
    fn test_knn_result_merger_with_empty_partitions(#[values(1, 2, 3)] random_seed: u64) {
        let k = 5;
        let include_tie_breaker = false;
        let num_rows = 100;
        let num_partitions = 3;
        let kept_prob = 0.5;
        let mut rng = StdRng::seed_from_u64(random_seed);

        let test_data = create_fuzz_test_data(k, num_rows, kept_prob, &mut rng);
        let partitioned_test_data =
            partition_fuzz_test_data(&test_data, num_partitions, kept_prob, &mut rng);
        for i in 0..(num_partitions + 1) {
            let empty_test_data = (0..num_rows)
                .map(|query_id| FuzzTestKNNResult {
                    query_id,
                    knn_objects: Vec::new(),
                })
                .collect::<Vec<FuzzTestKNNResult>>();

            // Insert a partition with no knn results at position i
            let mut test_data_with_empty_partition = partitioned_test_data.clone();
            test_data_with_empty_partition.insert(i, empty_test_data);

            assert!(is_fuzz_test_data_equivalent(
                &test_data,
                &test_data_with_empty_partition,
                k,
                include_tie_breaker
            ));

            let query_group_size = 30;
            let target_batch_size = 33;
            fuzz_test_knn_results_merger_using_partitioned_data(
                &test_data_with_empty_partition,
                k,
                include_tie_breaker,
                query_group_size,
                target_batch_size,
            )
            .unwrap();
        }
    }

    #[rstest]
    fn test_knn_results_merger_with_missing_probe_rows(
        #[values(1, 10, 13, 50, 51, 1000)] target_batch_size: usize,
    ) {
        let k = 5;
        let include_tie_breaker = true;
        let num_rows = 20;
        let num_partitions = 3;
        let kept_prob = 0.5;
        let mut rng = StdRng::seed_from_u64(target_batch_size as u64);

        let mut test_data = create_fuzz_test_data(k, num_rows, kept_prob, &mut rng);

        // Remove the query results of some probe rows randomly
        for result in test_data.iter_mut() {
            if rng.random_bool(0.1) {
                result.knn_objects.clear();
            }
        }

        let mut partitioned_test_data =
            partition_fuzz_test_data(&test_data, num_partitions, kept_prob, &mut rng);
        assert!(is_fuzz_test_data_equivalent(
            &test_data,
            &partitioned_test_data,
            k,
            include_tie_breaker
        ));

        // Randomly remove some probe rows from each partition
        for partition in partitioned_test_data.iter_mut() {
            for result in partition.iter_mut() {
                if rng.random_bool(0.1) {
                    result.knn_objects.clear();
                }
            }
        }

        let query_group_size = 30;
        let target_batch_size = 33;
        fuzz_test_knn_results_merger_using_partitioned_data(
            &partitioned_test_data,
            k,
            include_tie_breaker,
            query_group_size,
            target_batch_size,
        )
        .unwrap();
    }
}
