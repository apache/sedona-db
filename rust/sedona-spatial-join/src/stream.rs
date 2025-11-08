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
use arrow::array::BooleanBufferBuilder;
use arrow::compute::interleave_record_batch;
use arrow_array::{UInt32Array, UInt64Array};
use datafusion_common::{JoinSide, Result};
use datafusion_expr::JoinType;
use datafusion_physical_plan::joins::utils::StatefulStreamResult;
use datafusion_physical_plan::joins::utils::{ColumnIndex, JoinFilter};
use datafusion_physical_plan::metrics::{self, ExecutionPlanMetricsSet, MetricBuilder};
use datafusion_physical_plan::{handle_state, RecordBatchStream, SendableRecordBatchStream};
use futures::stream::StreamExt;
use futures::{ready, task::Poll};
use parking_lot::Mutex;
use sedona_common::sedona_internal_err;
use sedona_functions::st_analyze_aggr::AnalyzeAccumulator;
use sedona_schema::datatypes::WKB_GEOMETRY;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use crate::evaluated_batch::EvaluatedBatch;
use crate::index::SpatialIndex;
use crate::operand_evaluator::{create_operand_evaluator, distance_value_at, OperandEvaluator};
use crate::spatial_predicate::SpatialPredicate;
use crate::utils::join_utils::{
    adjust_indices_by_join_type, apply_join_filter_to_indices, build_batch_from_indices,
    get_final_indices_from_bit_map, need_produce_result_in_final,
};
use crate::utils::once_fut::{OnceAsync, OnceFut};
use arrow::array::RecordBatch;
use arrow::datatypes::{Schema, SchemaRef};
use sedona_common::option::SpatialJoinOptions;

/// Stream for producing spatial join result batches.
pub(crate) struct SpatialJoinStream {
    /// Input schema
    schema: Arc<Schema>,
    /// join filter
    filter: Option<JoinFilter>,
    /// type of the join
    join_type: JoinType,
    /// The stream of the probe side
    probe_stream: SendableRecordBatchStream,
    /// Information of index and left / right placement of columns
    column_indices: Vec<ColumnIndex>,
    /// Maintains the order of the probe side
    probe_side_ordered: bool,
    /// Join execution metrics
    join_metrics: SpatialJoinProbeMetrics,
    /// Current state of the stream
    state: SpatialJoinStreamState,
    /// Options for the spatial join
    #[allow(unused)]
    options: SpatialJoinOptions,
    /// Target output batch size
    target_output_batch_size: usize,
    /// Once future for the spatial index
    once_fut_spatial_index: OnceFut<SpatialIndex>,
    /// Once async for the spatial index, will be manually disposed by the last finished stream
    /// to avoid unnecessary memory usage.
    once_async_spatial_index: Arc<Mutex<Option<OnceAsync<SpatialIndex>>>>,
    /// The spatial index
    spatial_index: Option<Arc<SpatialIndex>>,
    /// The `on` spatial predicate evaluator
    evaluator: Arc<dyn OperandEvaluator>,
    /// The spatial predicate being evaluated
    spatial_predicate: SpatialPredicate,
}

impl SpatialJoinStream {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        schema: Arc<Schema>,
        on: &SpatialPredicate,
        filter: Option<JoinFilter>,
        join_type: JoinType,
        probe_stream: SendableRecordBatchStream,
        column_indices: Vec<ColumnIndex>,
        probe_side_ordered: bool,
        join_metrics: SpatialJoinProbeMetrics,
        options: SpatialJoinOptions,
        target_output_batch_size: usize,
        once_fut_spatial_index: OnceFut<SpatialIndex>,
        once_async_spatial_index: Arc<Mutex<Option<OnceAsync<SpatialIndex>>>>,
    ) -> Self {
        let evaluator = create_operand_evaluator(on, options.clone());
        Self {
            schema,
            filter,
            join_type,
            probe_stream,
            column_indices,
            probe_side_ordered,
            join_metrics,
            state: SpatialJoinStreamState::WaitBuildIndex,
            options,
            target_output_batch_size,
            once_fut_spatial_index,
            once_async_spatial_index,
            spatial_index: None,
            evaluator,
            spatial_predicate: on.clone(),
        }
    }
}

/// Metrics for the probe phase of the spatial join.
#[derive(Clone, Debug, Default)]
pub(crate) struct SpatialJoinProbeMetrics {
    /// Total time for joining probe-side batches to the build-side batches
    pub(crate) join_time: metrics::Time,
    /// Number of batches consumed by probe-side of this operator
    pub(crate) probe_input_batches: metrics::Count,
    /// Number of rows consumed by probe-side this operator
    pub(crate) probe_input_rows: metrics::Count,
    /// Number of batches produced by this operator
    pub(crate) output_batches: metrics::Count,
    /// Number of rows produced by this operator
    pub(crate) output_rows: metrics::Count,
    /// Number of result candidates retrieved by querying the spatial index
    pub(crate) join_result_candidates: metrics::Count,
    /// Number of join results before filtering
    pub(crate) join_result_count: metrics::Count,
    /// Memory usage of the refiner in bytes
    pub(crate) refiner_mem_used: metrics::Gauge,
    /// Execution mode used for executing the spatial join
    pub(crate) execution_mode: metrics::Gauge,
}

impl SpatialJoinProbeMetrics {
    pub fn new(partition: usize, metrics: &ExecutionPlanMetricsSet) -> Self {
        Self {
            join_time: MetricBuilder::new(metrics).subset_time("join_time", partition),
            probe_input_batches: MetricBuilder::new(metrics)
                .counter("probe_input_batches", partition),
            probe_input_rows: MetricBuilder::new(metrics).counter("probe_input_rows", partition),
            output_batches: MetricBuilder::new(metrics).counter("output_batches", partition),
            output_rows: MetricBuilder::new(metrics).output_rows(partition),
            join_result_candidates: MetricBuilder::new(metrics)
                .counter("join_result_candidates", partition),
            join_result_count: MetricBuilder::new(metrics).counter("join_result_count", partition),
            refiner_mem_used: MetricBuilder::new(metrics).gauge("refiner_mem_used", partition),
            execution_mode: MetricBuilder::new(metrics).gauge("execution_mode", partition),
        }
    }
}

/// This enumeration represents various states of the nested loop join algorithm.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum SpatialJoinStreamState {
    /// The initial mode: waiting for the spatial index to be built
    WaitBuildIndex,
    /// Indicates that build-side has been collected, and stream is ready for
    /// fetching probe-side
    FetchProbeBatch,
    /// Indicates that we're processing a probe batch using the batch iterator
    ProcessProbeBatch(SpatialJoinBatchIterator),
    /// Indicates that probe-side has been fully processed
    ExhaustedProbeSide,
    /// Indicates that we're processing unmatched build-side batches using an iterator
    ProcessUnmatchedBuildBatch(UnmatchedBuildBatchIterator),
    /// Indicates that SpatialJoinStream execution is completed
    Completed,
}

impl SpatialJoinStream {
    fn poll_next_impl(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Result<RecordBatch>>> {
        loop {
            return match &mut self.state {
                SpatialJoinStreamState::WaitBuildIndex => {
                    handle_state!(ready!(self.wait_build_index(cx)))
                }
                SpatialJoinStreamState::FetchProbeBatch => {
                    handle_state!(ready!(self.fetch_probe_batch(cx)))
                }
                SpatialJoinStreamState::ProcessProbeBatch(_) => {
                    handle_state!(ready!(self.process_probe_batch()))
                }
                SpatialJoinStreamState::ExhaustedProbeSide => {
                    handle_state!(ready!(self.setup_unmatched_build_batch_processing()))
                }
                SpatialJoinStreamState::ProcessUnmatchedBuildBatch(_) => {
                    handle_state!(ready!(self.process_unmatched_build_batch()))
                }
                SpatialJoinStreamState::Completed => Poll::Ready(None),
            };
        }
    }

    fn wait_build_index(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Result<StatefulStreamResult<Option<RecordBatch>>>> {
        let index = ready!(self.once_fut_spatial_index.get_shared(cx))?;
        self.spatial_index = Some(index);
        self.state = SpatialJoinStreamState::FetchProbeBatch;
        Poll::Ready(Ok(StatefulStreamResult::Continue))
    }

    fn fetch_probe_batch(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Result<StatefulStreamResult<Option<RecordBatch>>>> {
        let result = self.probe_stream.poll_next_unpin(cx);
        match result {
            Poll::Ready(Some(Ok(batch))) => match self.create_spatial_join_iterator(batch) {
                Ok(iterator) => {
                    self.state = SpatialJoinStreamState::ProcessProbeBatch(iterator);
                    Poll::Ready(Ok(StatefulStreamResult::Continue))
                }
                Err(e) => Poll::Ready(Err(e)),
            },
            Poll::Ready(Some(Err(e))) => Poll::Ready(Err(e)),
            Poll::Ready(None) => {
                self.state = SpatialJoinStreamState::ExhaustedProbeSide;
                Poll::Ready(Ok(StatefulStreamResult::Continue))
            }
            Poll::Pending => Poll::Pending,
        }
    }

    fn process_probe_batch(&mut self) -> Poll<Result<StatefulStreamResult<Option<RecordBatch>>>> {
        let timer = self.join_metrics.join_time.timer();

        // Extract the necessary data first to avoid borrowing conflicts
        let (batch_opt, is_complete) = match &mut self.state {
            SpatialJoinStreamState::ProcessProbeBatch(iterator) => {
                // For KNN joins, we swapped build/probe sides, so build_side should be Right
                // For regular joins, build_side is Left
                let build_side = match &self.spatial_predicate {
                    SpatialPredicate::KNearestNeighbors(_) => JoinSide::Right,
                    _ => JoinSide::Left,
                };

                let batch_opt = match iterator.next_batch(
                    &self.schema,
                    self.filter.as_ref(),
                    self.join_type,
                    &self.column_indices,
                    build_side,
                ) {
                    Ok(opt) => opt,
                    Err(e) => {
                        return Poll::Ready(Err(e));
                    }
                };
                let is_complete = iterator.is_complete();
                (batch_opt, is_complete)
            }
            _ => unreachable!(),
        };

        let result = match batch_opt {
            Some(batch) => {
                // Check if iterator is complete
                if is_complete {
                    self.state = SpatialJoinStreamState::FetchProbeBatch;
                }
                batch
            }
            None => {
                // Iterator finished, move to next probe batch
                self.state = SpatialJoinStreamState::FetchProbeBatch;
                return Poll::Ready(Ok(StatefulStreamResult::Continue));
            }
        };

        timer.done();
        Poll::Ready(Ok(StatefulStreamResult::Ready(Some(result))))
    }

    fn setup_unmatched_build_batch_processing(
        &mut self,
    ) -> Poll<Result<StatefulStreamResult<Option<RecordBatch>>>> {
        let Some(spatial_index) = self.spatial_index.as_ref() else {
            return Poll::Ready(sedona_internal_err!(
                "Expected spatial index to be available"
            ));
        };

        let is_last_stream = spatial_index.report_probe_completed();
        if is_last_stream {
            // Update the memory used by refiner and execution mode used to the metrics
            self.join_metrics
                .refiner_mem_used
                .set(spatial_index.get_refiner_mem_usage());
            self.join_metrics
                .execution_mode
                .set(spatial_index.get_actual_execution_mode().to_usize());

            // Drop the once async to avoid holding a long-living reference to the spatial index.
            // The spatial index will be dropped when this stream is dropped.
            let mut once_async = self.once_async_spatial_index.lock();
            once_async.take();
        }

        // Initial setup for processing unmatched build batches
        if need_produce_result_in_final(self.join_type) {
            // Only produce left-outer batches if this is the last partition that finished probing.
            // This mechanism is similar to the one in NestedLoopJoinStream.
            if !is_last_stream {
                self.state = SpatialJoinStreamState::Completed;
                return Poll::Ready(Ok(StatefulStreamResult::Ready(None)));
            }

            let empty_right_batch = RecordBatch::new_empty(self.probe_stream.schema());

            match UnmatchedBuildBatchIterator::new(spatial_index.clone(), empty_right_batch) {
                Ok(iterator) => {
                    self.state = SpatialJoinStreamState::ProcessUnmatchedBuildBatch(iterator);
                    Poll::Ready(Ok(StatefulStreamResult::Continue))
                }
                Err(e) => Poll::Ready(Err(e)),
            }
        } else {
            // end of the join loop
            self.state = SpatialJoinStreamState::Completed;
            Poll::Ready(Ok(StatefulStreamResult::Ready(None)))
        }
    }

    fn process_unmatched_build_batch(
        &mut self,
    ) -> Poll<Result<StatefulStreamResult<Option<RecordBatch>>>> {
        // Extract the iterator from the state to avoid borrowing conflicts
        let (batch_opt, is_complete) = match &mut self.state {
            SpatialJoinStreamState::ProcessUnmatchedBuildBatch(iterator) => {
                let batch_opt = match iterator.next_batch(
                    &self.schema,
                    self.join_type,
                    &self.column_indices,
                    JoinSide::Left,
                ) {
                    Ok(opt) => opt,
                    Err(e) => return Poll::Ready(Err(e)),
                };
                let is_complete = iterator.is_complete();
                (batch_opt, is_complete)
            }
            _ => {
                return Poll::Ready(sedona_internal_err!(
                    "process_unmatched_build_batch called with invalid state"
                ))
            }
        };

        match batch_opt {
            Some(batch) => {
                // Update metrics
                self.join_metrics.output_batches.add(1);
                self.join_metrics.output_rows.add(batch.num_rows());

                // Check if iterator is complete
                if is_complete {
                    self.state = SpatialJoinStreamState::Completed;
                }

                Poll::Ready(Ok(StatefulStreamResult::Ready(Some(batch))))
            }
            None => {
                // Iterator finished, complete the stream
                self.state = SpatialJoinStreamState::Completed;
                Poll::Ready(Ok(StatefulStreamResult::Ready(None)))
            }
        }
    }

    fn create_spatial_join_iterator(
        &self,
        probe_batch: RecordBatch,
    ) -> Result<SpatialJoinBatchIterator> {
        let num_rows = probe_batch.num_rows();
        self.join_metrics.probe_input_batches.add(1);
        self.join_metrics.probe_input_rows.add(num_rows);

        // Get the spatial index
        let spatial_index = self
            .spatial_index
            .as_ref()
            .expect("Spatial index should be available");

        // Evaluate the probe side geometry expression to get geometry array
        let geom_array = self.evaluator.evaluate_probe(&probe_batch)?;

        // Update the probe side statistics, which may help the spatial index to select a better
        // execution mode for evaluating the spatial predicate.
        if spatial_index.need_more_probe_stats() {
            let mut analyzer = AnalyzeAccumulator::new(WKB_GEOMETRY, WKB_GEOMETRY);
            for wkb in geom_array.wkbs().iter().flatten() {
                analyzer.update_statistics(wkb, wkb.buf().len())?;
            }
            let stats = analyzer.finish();
            spatial_index.merge_probe_stats(stats);
        }

        SpatialJoinBatchIterator::new(SpatialJoinBatchIteratorParams {
            spatial_index: spatial_index.clone(),
            probe_evaluated_batch: EvaluatedBatch {
                batch: probe_batch,
                geom_array,
            },
            join_metrics: self.join_metrics.clone(),
            max_batch_size: self.target_output_batch_size,
            probe_side_ordered: self.probe_side_ordered,
            spatial_predicate: self.spatial_predicate.clone(),
            options: self.options.clone(),
        })
    }
}

impl futures::Stream for SpatialJoinStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        self.poll_next_impl(cx)
    }
}

impl RecordBatchStream for SpatialJoinStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

/// A partial build batch is a batch containing rows from build-side records that are
/// needed to produce a result batch in probe phase. It is created by interleaving the
/// build side batches.
struct PartialBuildBatch {
    batch: RecordBatch,
    indices: UInt64Array,
    interleave_indices_map: HashMap<(i32, i32), usize>,
}

/// Iterator that processes spatial join results in configurable batch sizes
pub(crate) struct SpatialJoinBatchIterator {
    /// The spatial index reference
    spatial_index: Arc<SpatialIndex>,
    /// The probe side batch being processed
    probe_evaluated_batch: EvaluatedBatch,
    /// Current probe row index being processed
    current_probe_idx: usize,
    /// Join metrics for tracking performance
    join_metrics: SpatialJoinProbeMetrics,
    /// Maximum batch size before yielding a result
    max_batch_size: usize,
    /// Maintains the order of the probe side
    probe_side_ordered: bool,
    /// Current accumulated build batch positions
    build_batch_positions: Vec<(i32, i32)>,
    /// Current accumulated probe indices
    probe_indices: Vec<u32>,
    /// Whether iteration is complete
    is_complete: bool,
    /// The spatial predicate being evaluated
    spatial_predicate: SpatialPredicate,
    /// The spatial join options
    options: SpatialJoinOptions,
}

/// Parameters for creating a SpatialJoinBatchIterator
pub(crate) struct SpatialJoinBatchIteratorParams {
    pub spatial_index: Arc<SpatialIndex>,
    pub probe_evaluated_batch: EvaluatedBatch,
    pub join_metrics: SpatialJoinProbeMetrics,
    pub max_batch_size: usize,
    pub probe_side_ordered: bool,
    pub spatial_predicate: SpatialPredicate,
    pub options: SpatialJoinOptions,
}

impl SpatialJoinBatchIterator {
    pub(crate) fn new(params: SpatialJoinBatchIteratorParams) -> Result<Self> {
        Ok(Self {
            spatial_index: params.spatial_index,
            probe_evaluated_batch: params.probe_evaluated_batch,
            current_probe_idx: 0,
            join_metrics: params.join_metrics,
            max_batch_size: params.max_batch_size,
            probe_side_ordered: params.probe_side_ordered,
            build_batch_positions: Vec::new(),
            probe_indices: Vec::new(),
            is_complete: false,
            spatial_predicate: params.spatial_predicate,
            options: params.options,
        })
    }

    pub fn next_batch(
        &mut self,
        schema: &Schema,
        filter: Option<&JoinFilter>,
        join_type: JoinType,
        column_indices: &[ColumnIndex],
        build_side: JoinSide,
    ) -> Result<Option<RecordBatch>> {
        // Process probe rows incrementally until we have enough results or finish
        let initial_size = self.build_batch_positions.len();

        let geom_array = &self.probe_evaluated_batch.geom_array;
        let wkbs = geom_array.wkbs();
        let rects = &geom_array.rects;
        let distance = &geom_array.distance;

        let num_rows = wkbs.len();

        let last_probe_idx = self.current_probe_idx;

        // Process from current position until we hit batch size limit or complete
        while self.current_probe_idx < num_rows && !self.is_complete {
            // Get WKB for current probe index
            let wkb_opt = &wkbs[self.current_probe_idx];

            let Some(wkb) = wkb_opt else {
                // Move to next probe index
                self.current_probe_idx += 1;
                continue;
            };

            let dist = match distance {
                Some(dist) => distance_value_at(dist, self.current_probe_idx)?,
                None => None,
            };

            // Handle KNN queries differently from regular spatial joins
            match &self.spatial_predicate {
                SpatialPredicate::KNearestNeighbors(knn_predicate) => {
                    // For KNN, call query_knn only once per probe geometry (not per rect)
                    let k = knn_predicate.k;
                    let use_spheroid = knn_predicate.use_spheroid;
                    let include_tie_breakers = self.options.knn_include_tie_breakers;

                    let join_result_metrics = self.spatial_index.query_knn(
                        wkb,
                        k,
                        use_spheroid,
                        include_tie_breakers,
                        &mut self.build_batch_positions,
                    )?;

                    self.probe_indices.extend(std::iter::repeat_n(
                        self.current_probe_idx as u32,
                        join_result_metrics.count,
                    ));

                    self.join_metrics
                        .join_result_candidates
                        .add(join_result_metrics.candidate_count);
                    self.join_metrics
                        .join_result_count
                        .add(join_result_metrics.count);
                }
                _ => {
                    // Regular spatial join: process all rects for this probe index
                    let rect_opt = &rects[self.current_probe_idx];
                    if let Some(rect) = rect_opt {
                        let join_result_metrics = self.spatial_index.query(
                            wkb,
                            rect,
                            &dist,
                            &mut self.build_batch_positions,
                        )?;

                        self.probe_indices.extend(std::iter::repeat_n(
                            self.current_probe_idx as u32,
                            join_result_metrics.count,
                        ));

                        self.join_metrics
                            .join_result_candidates
                            .add(join_result_metrics.candidate_count);
                        self.join_metrics
                            .join_result_count
                            .add(join_result_metrics.count);
                    }
                }
            }

            self.current_probe_idx += 1;

            // Early exit if we have enough results
            if self.build_batch_positions.len() >= self.max_batch_size {
                break;
            }
        }

        // Check if we've finished processing all probe rows
        if self.current_probe_idx >= num_rows {
            self.is_complete = true;
        }

        // Return accumulated results if we have any new ones or if we're complete
        if self.build_batch_positions.len() > initial_size || self.is_complete {
            // Process the joined indices to create a RecordBatch
            let probe_indices = std::mem::take(&mut self.probe_indices);
            let batch = self.process_joined_indices_to_batch(
                &self.build_batch_positions,
                probe_indices,
                schema,
                filter,
                join_type,
                column_indices,
                build_side,
                last_probe_idx..self.current_probe_idx,
            )?;

            self.build_batch_positions.clear();
            Ok(Some(batch))
        } else {
            Ok(None)
        }
    }

    /// Check if the iterator has finished processing
    pub fn is_complete(&self) -> bool {
        self.is_complete
    }

    /// Process joined indices and create a RecordBatch
    #[allow(clippy::too_many_arguments)]
    fn process_joined_indices_to_batch(
        &self,
        build_indices: &[(i32, i32)],
        probe_indices: Vec<u32>,
        schema: &Schema,
        filter: Option<&JoinFilter>,
        join_type: JoinType,
        column_indices: &[ColumnIndex],
        build_side: JoinSide,
        probe_range: Range<usize>,
    ) -> Result<RecordBatch> {
        let PartialBuildBatch {
            batch: partial_build_batch,
            indices: build_indices,
            interleave_indices_map,
        } = self.assemble_partial_build_batch(build_indices)?;
        let probe_indices = UInt32Array::from(probe_indices);

        let (build_indices, probe_indices) = match filter {
            Some(filter) => apply_join_filter_to_indices(
                &partial_build_batch,
                &self.probe_evaluated_batch.batch,
                build_indices,
                probe_indices,
                filter,
                build_side,
            )?,
            None => (build_indices, probe_indices),
        };

        // set the left bitmap
        if need_produce_result_in_final(join_type) {
            if let Some(visited_bitmaps) = self.spatial_index.visited_left_side() {
                mark_build_side_rows_as_visited(
                    &build_indices,
                    &interleave_indices_map,
                    visited_bitmaps,
                );
            }
        }

        // adjust the two side indices base on the join type
        let (build_indices, probe_indices) = adjust_indices_by_join_type(
            build_indices,
            probe_indices,
            probe_range,
            join_type,
            self.probe_side_ordered,
        )?;

        // Build the final result batch
        let result_batch = build_batch_from_indices(
            schema,
            &partial_build_batch,
            &self.probe_evaluated_batch.batch,
            &build_indices,
            &probe_indices,
            column_indices,
            build_side,
        )?;

        // Update metrics with actual output
        self.join_metrics.output_batches.add(1);
        self.join_metrics.output_rows.add(result_batch.num_rows());

        Ok(result_batch)
    }

    fn assemble_partial_build_batch(
        &self,
        build_indices: &[(i32, i32)],
    ) -> Result<PartialBuildBatch> {
        let schema = self.spatial_index.schema();
        assemble_partial_build_batch(build_indices, schema, |batch_idx| {
            self.spatial_index.get_indexed_batch(batch_idx)
        })
    }
}

fn assemble_partial_build_batch<'a>(
    build_indices: &'a [(i32, i32)],
    schema: SchemaRef,
    batch_getter: impl Fn(usize) -> &'a RecordBatch,
) -> Result<PartialBuildBatch> {
    let num_rows = build_indices.len();
    if num_rows == 0 {
        let empty_batch = RecordBatch::new_empty(schema);
        let empty_build_indices = UInt64Array::from(vec![] as Vec<u64>);
        let empty_map = HashMap::new();
        return Ok(PartialBuildBatch {
            batch: empty_batch,
            indices: empty_build_indices,
            interleave_indices_map: empty_map,
        });
    }

    // Get only the build batches that are actually needed
    let mut needed_build_batches: Vec<&RecordBatch> = Vec::with_capacity(num_rows);

    // Mapping from global batch index to partial batch index for generating this result batch
    let mut needed_batch_index_map: HashMap<i32, i32> = HashMap::with_capacity(num_rows);

    let mut interleave_indices_map: HashMap<(i32, i32), usize> = HashMap::with_capacity(num_rows);
    let mut interleave_indices: Vec<(usize, usize)> = Vec::with_capacity(num_rows);

    // The indices of joined rows from the partial build batches.
    let mut partial_build_indices_builder = UInt64Array::builder(num_rows);

    for (batch_idx, row_idx) in build_indices {
        let local_batch_idx = if let Some(idx) = needed_batch_index_map.get(batch_idx) {
            *idx
        } else {
            let new_idx = needed_build_batches.len() as i32;
            needed_batch_index_map.insert(*batch_idx, new_idx);
            needed_build_batches.push(batch_getter(*batch_idx as usize));
            new_idx
        };

        if let Some(idx) = interleave_indices_map.get(&(*batch_idx, *row_idx)) {
            // We have already seen this row. It will be in the interleaved batch at position `idx`
            partial_build_indices_builder.append_value(*idx as u64);
        } else {
            // The row has not been seen before, we need to interleave it into the partial build batch.
            // The index of the row in the partial build batch will be `interleave_indices.len()`.
            let idx = interleave_indices.len();
            interleave_indices_map.insert((*batch_idx, *row_idx), idx);
            interleave_indices.push((local_batch_idx as usize, *row_idx as usize));
            partial_build_indices_builder.append_value(idx as u64);
        }
    }

    let partial_build_indices = partial_build_indices_builder.finish();

    // Assemble an interleaved batch on build side, so that we can reuse the join indices
    // processing routines in utils.rs (taken verbatimly from datafusion)
    let partial_build_batch = interleave_record_batch(&needed_build_batches, &interleave_indices)?;

    Ok(PartialBuildBatch {
        batch: partial_build_batch,
        indices: partial_build_indices,
        interleave_indices_map,
    })
}

fn mark_build_side_rows_as_visited(
    build_indices: &UInt64Array,
    interleave_indices_map: &HashMap<(i32, i32), usize>,
    visited_bitmaps: &Mutex<Vec<BooleanBufferBuilder>>,
) {
    // invert the interleave_indices_map for easier getting the global batch index and row index
    // from partial batch row index
    let mut inverted_interleave_indices_map: HashMap<usize, (i32, i32)> =
        HashMap::with_capacity(interleave_indices_map.len());
    for ((batch_idx, row_idx), partial_idx) in interleave_indices_map.iter() {
        inverted_interleave_indices_map.insert(*partial_idx, (*batch_idx, *row_idx));
    }

    // Lock the mutex once and iterate over build_indices to set the left bitmap
    let mut bitmaps = visited_bitmaps.lock();
    for partial_batch_row_idx in build_indices.iter() {
        let Some(partial_batch_row_idx) = partial_batch_row_idx else {
            continue;
        };
        let partial_batch_row_idx = partial_batch_row_idx as usize;
        let Some((batch_idx, row_idx)) =
            inverted_interleave_indices_map.get(&partial_batch_row_idx)
        else {
            continue;
        };
        let Some(bitmap) = bitmaps.get_mut(*batch_idx as usize) else {
            continue;
        };
        bitmap.set_bit(*row_idx as usize, true);
    }
}

// Manual Debug implementation for SpatialJoinBatchIterator
impl std::fmt::Debug for SpatialJoinBatchIterator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpatialJoinBatchIterator")
            .field("max_batch_size", &self.max_batch_size)
            .field("current_probe_idx", &self.current_probe_idx)
            .field("is_complete", &self.is_complete)
            .field(
                "build_batch_positions_len",
                &self.build_batch_positions.len(),
            )
            .field("probe_indices_len", &self.probe_indices.len())
            .finish()
    }
}

/// Iterator that processes unmatched build-side batches for outer joins
pub(crate) struct UnmatchedBuildBatchIterator {
    /// The spatial index reference
    spatial_index: Arc<SpatialIndex>,
    /// Current batch index being processed
    current_batch_idx: usize,
    /// Total number of batches to process
    total_batches: usize,
    /// Empty right batch for joining
    empty_right_batch: RecordBatch,
    /// Whether iteration is complete
    is_complete: bool,
}

impl UnmatchedBuildBatchIterator {
    pub(crate) fn new(
        spatial_index: Arc<SpatialIndex>,
        empty_right_batch: RecordBatch,
    ) -> Result<Self> {
        let visited_left_side = spatial_index.visited_left_side();
        let Some(vec_visited_left_side) = visited_left_side else {
            return sedona_internal_err!("The bitmap for visited left side is not created");
        };

        let total_batches = {
            let visited_bitmaps = vec_visited_left_side.lock();
            visited_bitmaps.len()
        };

        Ok(Self {
            spatial_index,
            current_batch_idx: 0,
            total_batches,
            empty_right_batch,
            is_complete: false,
        })
    }

    pub fn next_batch(
        &mut self,
        schema: &Schema,
        join_type: JoinType,
        column_indices: &[ColumnIndex],
        build_side: JoinSide,
    ) -> Result<Option<RecordBatch>> {
        while self.current_batch_idx < self.total_batches && !self.is_complete {
            let visited_left_side = self.spatial_index.visited_left_side();
            let Some(vec_visited_left_side) = visited_left_side else {
                return sedona_internal_err!("The bitmap for visited left side is not created");
            };

            let batch = {
                let visited_bitmaps = vec_visited_left_side.lock();
                let visited_left_side = &visited_bitmaps[self.current_batch_idx];
                let (left_side, right_side) =
                    get_final_indices_from_bit_map(visited_left_side, join_type);

                build_batch_from_indices(
                    schema,
                    self.spatial_index.get_indexed_batch(self.current_batch_idx),
                    &self.empty_right_batch,
                    &left_side,
                    &right_side,
                    column_indices,
                    build_side,
                )?
            };

            self.current_batch_idx += 1;

            // Check if we've finished processing all batches
            if self.current_batch_idx >= self.total_batches {
                self.is_complete = true;
            }

            // Only return non-empty batches
            if batch.num_rows() > 0 {
                return Ok(Some(batch));
            }
            // If batch is empty, continue to next batch
        }

        // No more batches or iteration complete
        Ok(None)
    }

    /// Check if the iterator has finished processing
    pub fn is_complete(&self) -> bool {
        self.is_complete
    }
}

// Manual Debug implementation for UnmatchedBuildBatchIterator
impl std::fmt::Debug for UnmatchedBuildBatchIterator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnmatchedBuildBatchIterator")
            .field("current_batch_idx", &self.current_batch_idx)
            .field("total_batches", &self.total_batches)
            .field("is_complete", &self.is_complete)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow_array::cast::AsArray;

    fn create_test_batches(
        num_batches: usize,
        rows_per_batch: usize,
    ) -> (SchemaRef, Vec<RecordBatch>) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Int32, false),
            Field::new("partition", DataType::Int32, false),
        ]));

        let mut batches = Vec::with_capacity(num_batches);
        let mut id = 0;
        for batch_idx in 0..num_batches {
            let mut id_builder = Int32Array::builder(rows_per_batch);
            let mut value_builder = Int32Array::builder(rows_per_batch);
            let mut partition_builder = Int32Array::builder(rows_per_batch);
            for row_idx in 0..rows_per_batch {
                id_builder.append_value(id);
                value_builder.append_value(row_idx as i32);
                partition_builder.append_value(batch_idx as i32);
                id += 1;
            }
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(id_builder.finish()),
                    Arc::new(value_builder.finish()),
                    Arc::new(partition_builder.finish()),
                ],
            )
            .unwrap();
            batches.push(batch);
        }

        (schema, batches)
    }

    #[test]
    fn test_assemble_partial_build_batch_empty_batch() {
        let (schema, batches) = create_test_batches(0, 0);

        let build_indices = vec![];
        let result =
            assemble_partial_build_batch(&build_indices, schema, |batch_idx| &batches[batch_idx])
                .unwrap();

        // Empty input should produce empty output
        assert_eq!(result.batch.num_rows(), 0);
        assert_eq!(result.indices.len(), 0);
        assert_eq!(result.interleave_indices_map.len(), 0);
    }

    #[test]
    fn test_assemble_partial_build_batch_empty_indices() {
        let (schema, batches) = create_test_batches(2, 3);

        let build_indices = vec![];
        let result =
            assemble_partial_build_batch(&build_indices, schema, |batch_idx| &batches[batch_idx])
                .unwrap();

        // Empty input should produce empty output
        assert_eq!(result.batch.num_rows(), 0);
        assert_eq!(result.indices.len(), 0);
        assert_eq!(result.interleave_indices_map.len(), 0);
    }

    #[test]
    fn test_assemble_partial_build_batch_single_batch() {
        let (schema, batches) = create_test_batches(1, 5);

        // Reference rows (0,1), (0,3), (0,1) from batch 0
        let build_indices = vec![(0, 1), (0, 3), (0, 1)];
        let result =
            assemble_partial_build_batch(&build_indices, schema, |batch_idx| &batches[batch_idx])
                .unwrap();

        // Verify both constraints using utility functions
        verify_constraints(&result, &build_indices, &batches);
    }

    #[test]
    fn test_assemble_partial_build_batch_multiple_batches() {
        let (schema, batches) = create_test_batches(3, 4);

        // Reference rows from different batches
        let build_indices = vec![
            (0, 1), // batch 0, row 1
            (2, 3), // batch 2, row 3
            (1, 0), // batch 1, row 0
            (0, 1), // batch 0, row 1 (duplicate)
            (1, 2), // batch 1, row 2
        ];
        let result =
            assemble_partial_build_batch(&build_indices, schema, |batch_idx| &batches[batch_idx])
                .unwrap();

        // Verify both constraints using utility functions
        verify_constraints(&result, &build_indices, &batches);
    }

    #[test]
    fn test_assemble_partial_build_batch_duplicate_references() {
        let (schema, batches) = create_test_batches(2, 3);

        // Reference the same row multiple times
        let build_indices = vec![
            (0, 1), // batch 0, row 1
            (1, 2), // batch 1, row 2
            (0, 1), // batch 0, row 1 (duplicate)
            (1, 2), // batch 1, row 2 (duplicate)
            (0, 1), // batch 0, row 1 (duplicate again)
        ];
        let result =
            assemble_partial_build_batch(&build_indices, schema, |batch_idx| &batches[batch_idx])
                .unwrap();

        // Verify both constraints using utility functions
        verify_constraints(&result, &build_indices, &batches);
    }

    /// Verify the constraints for the assemble_partial_build_batch function.
    ///
    /// These constraints verify that the assemble_partial_build_batch function produce correct results.
    /// 1. When the assembled batch is taken using the returned indices, it is equivalent
    ///    to interleaving batches using original batch indices and row indices.
    /// 2. The index mapping is correct. You can find equivalent row from the original
    ///    indexed batches by following the inverted map.
    fn verify_constraints(
        result: &PartialBuildBatch,
        build_indices: &[(i32, i32)],
        batches: &[RecordBatch],
    ) {
        assert_eq!(result.indices.len(), build_indices.len());
        verify_assembled_batch(result, build_indices, batches);
        verify_interleave_indices_map(result, build_indices, batches);
    }

    /// Utility function to verify constraint 1: equivalence of assembled batch vs original batches
    fn verify_assembled_batch(
        result: &PartialBuildBatch,
        build_indices: &[(i32, i32)],
        batches: &[RecordBatch],
    ) {
        for (i, &(batch_idx, row_idx)) in build_indices.iter().enumerate() {
            let partial_idx = result.indices.value(i) as usize;
            let original_batch = &batches[batch_idx as usize];

            // Compare data values across all columns
            for col_idx in 0..original_batch.num_columns() {
                let original_value = original_batch.column(col_idx);
                let assembled_value = result.batch.column(col_idx);
                if matches!(
                    original_value.data_type(),
                    arrow::datatypes::DataType::Int32
                ) {
                    let original_val = original_value
                        .as_primitive::<arrow::datatypes::Int32Type>()
                        .value(row_idx as usize);
                    let assembled_val = assembled_value
                        .as_primitive::<arrow::datatypes::Int32Type>()
                        .value(partial_idx);
                    assert_eq!(
                        original_val, assembled_val,
                        "Column {col_idx} mismatch for build_indices[{i}] = ({batch_idx}, {row_idx})"
                    );
                } else {
                    unreachable!("Only int32 columns are supported");
                }
            }
        }
    }

    /// Utility function to verify constraint 2: index mapping correctness
    fn verify_interleave_indices_map(
        result: &PartialBuildBatch,
        build_indices: &[(i32, i32)],
        batches: &[RecordBatch],
    ) {
        // Create inverted map to verify bidirectional consistency
        let mut inverted_map: HashMap<usize, (i32, i32)> = HashMap::new();
        for (&(batch_idx, row_idx), &partial_idx) in result.interleave_indices_map.iter() {
            inverted_map.insert(partial_idx, (batch_idx, row_idx));
        }

        // Check that we can find each original row via the mapping
        for (i, &(batch_idx, row_idx)) in build_indices.iter().enumerate() {
            let partial_idx = result.indices.value(i) as usize;
            let mapped_indices = result
                .interleave_indices_map
                .get(&(batch_idx, row_idx))
                .expect("build_indices entry should exist in interleave_indices_map");
            assert_eq!(
                partial_idx, *mapped_indices,
                "Index mapping mismatch for build_indices[{i}] = ({batch_idx}, {row_idx})"
            );
        }

        // Verify that for each unique row in assembled batch, we can map back to original
        for i in 0..result.batch.num_rows() {
            let (original_batch_idx, original_row_idx) = inverted_map
                .get(&i)
                .expect("Each assembled batch row should have a mapping to original");
            let original_batch = &batches[*original_batch_idx as usize];

            // Compare the first column (id) to verify correctness
            let original_id = original_batch
                .column(0)
                .as_primitive::<arrow::datatypes::Int32Type>()
                .value(*original_row_idx as usize);
            let assembled_id = result
                .batch
                .column(0)
                .as_primitive::<arrow::datatypes::Int32Type>()
                .value(i);
            assert_eq!(original_id, assembled_id,
                "Data mismatch when mapping back from assembled batch row {i} to original batch {original_batch_idx} row {original_row_idx}");
        }
    }
}
