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

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::datatypes::SchemaRef;
use arrow_array::{Array, RecordBatch, UInt32Array, UInt64Array};
use datafusion::error::{DataFusionError, Result};
use datafusion::physical_plan::{RecordBatchStream, SendableRecordBatchStream};
use datafusion_physical_plan::metrics::{self, ExecutionPlanMetricsSet, MetricBuilder};
use futures::stream::Stream;

use crate::evaluated_batch::EvaluatedBatch;
use crate::index::SpatialIndex;
use crate::operand_evaluator::{create_operand_evaluator, OperandEvaluator};
use crate::spatial_predicate::SpatialPredicate;
use crate::utils::join_utils::{
    adjust_indices_by_join_type, apply_join_filter_to_indices, build_batch_from_indices,
    get_final_indices_from_bit_map, need_produce_result_in_final,
};
use crate::utils::once_fut::{OnceAsync, OnceFut};
use arrow_schema::Schema;
use datafusion_common::{JoinSide, JoinType};
use datafusion_physical_plan::handle_state;
use datafusion_physical_plan::joins::utils::{ColumnIndex, JoinFilter, StatefulStreamResult};
use futures::{ready, StreamExt};
use parking_lot::{Mutex, RwLock};
use sedona_common::{sedona_internal_err, SpatialJoinOptions};
use sysinfo::{MemoryRefreshKind, RefreshKind, System};

/// Metrics for GPU spatial join operations
pub(crate) struct GpuSpatialJoinMetrics {
    /// Total time for GPU join execution
    pub(crate) filter_time: metrics::Time,
    pub(crate) refine_time: metrics::Time,
    pub(crate) post_process_time: metrics::Time,
    /// Number of batches produced by this operator
    pub(crate) output_batches: metrics::Count,
    /// Number of rows produced by this operator
    pub(crate) output_rows: metrics::Count,
}

impl GpuSpatialJoinMetrics {
    pub fn new(partition: usize, metrics: &ExecutionPlanMetricsSet) -> Self {
        Self {
            filter_time: MetricBuilder::new(metrics).subset_time("filter_time", partition),
            refine_time: MetricBuilder::new(metrics).subset_time("refine_time", partition),
            post_process_time: MetricBuilder::new(metrics)
                .subset_time("post_process_time", partition),
            output_batches: MetricBuilder::new(metrics).counter("output_batches", partition),
            output_rows: MetricBuilder::new(metrics).counter("output_rows", partition),
        }
    }
}

pub struct GpuSpatialJoinStream {
    partition: usize,
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
    join_metrics: GpuSpatialJoinMetrics,
    /// Current state of the stream
    state: SpatialJoinStreamState,
    /// Options for the spatial join
    #[allow(unused)]
    options: SpatialJoinOptions,
    /// Once future for the spatial index
    once_fut_spatial_index: OnceFut<Arc<RwLock<SpatialIndex>>>,
    /// Once async for the spatial index, will be manually disposed by the last finished stream
    /// to avoid unnecessary memory usage.
    once_async_spatial_index: Arc<Mutex<Option<OnceAsync<Arc<RwLock<SpatialIndex>>>>>>,
    /// The spatial index
    spatial_index: Option<Arc<RwLock<SpatialIndex>>>,
    /// The `on` spatial predicate evaluator
    evaluator: Arc<dyn OperandEvaluator>,
    /// The spatial predicate being evaluated
    spatial_predicate: SpatialPredicate,
}

impl GpuSpatialJoinStream {
    /// Create a new GPU spatial join stream for probe phase
    ///
    /// This constructor is called per output partition and creates a stream that:
    /// 1. Awaits shared left-side build data from once_build_data
    /// 2. Reads the right partition specified by `partition` parameter
    /// 3. Executes GPU join between shared left data and this partition's right data
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        partition: usize,
        schema: Arc<Schema>,
        on: &SpatialPredicate,
        filter: Option<JoinFilter>,
        join_type: JoinType,
        probe_stream: SendableRecordBatchStream,
        column_indices: Vec<ColumnIndex>,
        probe_side_ordered: bool,
        join_metrics: GpuSpatialJoinMetrics,
        options: SpatialJoinOptions,
        once_fut_spatial_index: OnceFut<Arc<RwLock<SpatialIndex>>>,
        once_async_spatial_index: Arc<Mutex<Option<OnceAsync<Arc<RwLock<SpatialIndex>>>>>>,
    ) -> Self {
        let evaluator = create_operand_evaluator(on, options.clone());
        Self {
            partition,
            schema,
            filter,
            join_type,
            probe_stream,
            column_indices,
            probe_side_ordered,
            join_metrics,
            state: SpatialJoinStreamState::WaitBuildIndex,
            options,
            once_fut_spatial_index,
            once_async_spatial_index,
            spatial_index: None,
            evaluator,
            spatial_predicate: on.clone(),
        }
    }
}

/// State machine for GPU spatial join execution
// #[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub(crate) enum SpatialJoinStreamState {
    /// The initial mode: waiting for the spatial index to be built
    WaitBuildIndex,
    /// Indicates that build-side has been collected, and stream is ready for
    /// fetching probe-side
    FetchProbeBatch,
    /// Indicates that we're processing a probe batch using the batch iterator
    ProcessProbeBatch(Arc<EvaluatedBatch>),
    /// Indicates that probe-side has been fully processed
    ExhaustedProbeSide,
    /// Indicates that we're processing unmatched build-side batches using an iterator
    ProcessUnmatchedBuildBatch(UnmatchedBuildBatchIterator),
    /// Indicates that SpatialJoinStream execution is completed
    Completed,
}

impl GpuSpatialJoinStream {
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
        log::debug!("[GPU Join] Probe stream waiting for build index...");
        let index = ready!(self.once_fut_spatial_index.get(cx))?;
        log::debug!("[GPU Join] Spatial index received, starting probe phase");
        self.spatial_index = Some(index.clone());
        self.state = SpatialJoinStreamState::FetchProbeBatch;
        Poll::Ready(Ok(StatefulStreamResult::Continue))
    }

    fn fetch_probe_batch(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Result<StatefulStreamResult<Option<RecordBatch>>>> {
        let result = self.probe_stream.poll_next_unpin(cx);
        match result {
            Poll::Ready(Some(Ok(probe_batch))) => {
                let num_rows = probe_batch.num_rows();
                log::debug!("[GPU Join] Fetched probe batch: {} rows", num_rows);

                match self.evaluator.evaluate_probe(&probe_batch) {
                    Ok(geom_array) => {
                        let eval_batch = Arc::new(EvaluatedBatch {
                            batch: probe_batch,
                            geom_array,
                        });
                        self.state = SpatialJoinStreamState::ProcessProbeBatch(eval_batch);
                        Poll::Ready(Ok(StatefulStreamResult::Continue))
                    }
                    Err(e) => Poll::Ready(Err(e)),
                }
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Err(e)),
            Poll::Ready(None) => {
                log::debug!("[GPU Join] All probe batches processed");
                self.state = SpatialJoinStreamState::ExhaustedProbeSide;
                Poll::Ready(Ok(StatefulStreamResult::Continue))
            }
            Poll::Pending => Poll::Pending,
        }
    }

    fn process_probe_batch(&mut self) -> Poll<Result<StatefulStreamResult<Option<RecordBatch>>>> {
        let (batch_opt, need_load) = {
            match &self.state {
                SpatialJoinStreamState::ProcessProbeBatch(eval_batch) => {
                    let eval_batch = eval_batch.clone();
                    let build_side = match &self.spatial_predicate {
                        SpatialPredicate::KNearestNeighbors(_) => JoinSide::Right,
                        _ => JoinSide::Left,
                    };
                    let spatial_index = self
                        .spatial_index
                        .as_ref()
                        .expect("Spatial index should be available");

                    log::info!(
                        "Partition {} calls GpuSpatial's filtering for batch with {} rects",
                        self.partition,
                        eval_batch.geom_array.rects.len()
                    );

                    let timer = self.join_metrics.filter_time.timer();
                    let (mut build_ids, mut probe_ids) = {
                        match spatial_index.read().filter(&eval_batch.geom_array.rects) {
                            Ok((build_ids, probe_ids)) => (build_ids, probe_ids),
                            Err(e) => {
                                return Poll::Ready(Err(e));
                            }
                        }
                    };
                    timer.done();
                    log::info!("Found {} joined pairs in GPU spatial join", build_ids.len());

                    let geoms = &eval_batch.geom_array.geometry_array;

                    let timer = self.join_metrics.refine_time.timer();
                    log::info!(
                            "Partition {} calls GpuSpatial's refinement for batch with {} geoms with {:?}", self.partition,
                            geoms.len(),
                            self.spatial_predicate
                        );
                    if let Err(e) = spatial_index.read().refine_loaded(
                        geoms,
                        &self.spatial_predicate,
                        &mut build_ids,
                        &mut probe_ids,
                    ) {
                        return Poll::Ready(Err(e));
                    }

                    timer.done();
                    let time: metrics::Time = Default::default();
                    let timer = time.timer();
                    let res = match self.process_joined_indices_to_batch(
                        build_ids,   // These are now owned Vec<u32>
                        probe_ids,   // These are now owned Vec<u32>
                        &eval_batch, // Pass as reference (Arc ref or Struct ref)
                        build_side,
                    ) {
                        Ok((batch, need_load)) => (Some(batch), need_load),
                        Err(e) => {
                            return Poll::Ready(Err(e));
                        }
                    };
                    timer.done();
                    self.join_metrics.post_process_time = time;
                    res
                }
                _ => unreachable!(),
            }
        };

        // if (need_load) {
        //     self.state = SpatialJoinStreamState::WaitLoadingBuildSide;
        // } else {
        self.state = SpatialJoinStreamState::FetchProbeBatch;
        // }

        Poll::Ready(Ok(StatefulStreamResult::Ready(batch_opt)))
    }

    #[allow(clippy::too_many_arguments)]
    fn process_joined_indices_to_batch(
        &mut self,
        build_indices: Vec<u32>,
        probe_indices: Vec<u32>,
        probe_eval_batch: &EvaluatedBatch,
        build_side: JoinSide,
    ) -> Result<(RecordBatch, bool)> {
        let spatial_index = self
            .spatial_index
            .as_ref()
            .expect("spatial_index should be created")
            .read();

        // thread-safe update of visited left side bitmap
        // let visited_bitmap = spatial_index.visited_left_side();
        // for row_idx in build_indices.iter() {
        //     visited_bitmap.set_bit(*row_idx as usize);
        // }
        // let visited_ratio =
        //     visited_bitmap.count() as f32 / spatial_index.build_batch.num_rows() as f32;
        //
        // let need_load_build_side = visited_ratio > 0.01 && !spatial_index.is_build_side_loaded();

        let join_type = self.join_type;
        // set the left bitmap
        if need_produce_result_in_final(join_type) {
            if let Some(visited_bitmap) = spatial_index.visited_left_side() {
                // Lock the mutex once and iterate over build_indices to set the left bitmap
                let mut locked_bitmap = visited_bitmap.lock();

                for row_idx in build_indices.iter() {
                    locked_bitmap.set_bit(*row_idx as usize, true);
                }
            }
        }
        let need_load_build_side = false;

        // log::info!(
        //     "Visited build side ratio: {}, is_build_side loaded {}",
        //     visited_ratio,
        //     spatial_index.is_build_side_loaded()
        // );

        let join_type = self.join_type;

        let filter = self.filter.as_ref();

        let build_indices_array =
            UInt64Array::from_iter_values(build_indices.into_iter().map(|x| x as u64));
        let probe_indices_array = UInt32Array::from(probe_indices);
        let spatial_index = self.spatial_index.as_ref().ok_or_else(|| {
            DataFusionError::Execution("Spatial index should be available".into())
        })?;

        let (build_indices, probe_indices) = match filter {
            Some(filter) => apply_join_filter_to_indices(
                &spatial_index.read().build_batch.batch,
                &probe_eval_batch.batch,
                build_indices_array,
                probe_indices_array,
                filter,
                build_side,
            )?,
            None => (build_indices_array, probe_indices_array),
        };

        let schema = self.schema.as_ref();

        let probe_range = 0..probe_eval_batch.batch.num_rows();
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
            &spatial_index.read().build_batch.batch,
            &probe_eval_batch.batch,
            &build_indices,
            &probe_indices,
            &self.column_indices,
            build_side,
        )?;
        // Update metrics with actual output
        self.join_metrics.output_batches.add(1);
        self.join_metrics.output_rows.add(result_batch.num_rows());

        Ok((result_batch, need_load_build_side))
    }

    fn setup_unmatched_build_batch_processing(
        &mut self,
    ) -> Poll<Result<StatefulStreamResult<Option<RecordBatch>>>> {
        let Some(spatial_index) = self.spatial_index.as_ref() else {
            return Poll::Ready(sedona_internal_err!(
                "Expected spatial index to be available"
            ));
        };

        let is_last_stream = spatial_index.read().report_probe_completed();
        if is_last_stream {
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
        let batch_opt = match &mut self.state {
            SpatialJoinStreamState::ProcessUnmatchedBuildBatch(iterator) => {
                match iterator.next_batch(
                    &self.schema,
                    self.join_type,
                    &self.column_indices,
                    JoinSide::Left,
                ) {
                    Ok(opt) => opt,
                    Err(e) => return Poll::Ready(Err(e)),
                }
            }
            _ => {
                return Poll::Ready(sedona_internal_err!(
                    "process_unmatched_build_batch called with invalid state"
                ))
            }
        };

        match batch_opt {
            Some(batch) => {
                self.state = SpatialJoinStreamState::Completed;

                Poll::Ready(Ok(StatefulStreamResult::Ready(Some(batch))))
            }
            None => {
                // Iterator finished, complete the stream
                self.state = SpatialJoinStreamState::Completed;
                Poll::Ready(Ok(StatefulStreamResult::Ready(None)))
            }
        }
    }
}

impl Stream for GpuSpatialJoinStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.poll_next_impl(cx)
    }
}

impl RecordBatchStream for GpuSpatialJoinStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

/// Iterator that processes unmatched build-side batches for outer joins
pub(crate) struct UnmatchedBuildBatchIterator {
    /// The spatial index reference
    spatial_index: Arc<RwLock<SpatialIndex>>,
    /// Empty right batch for joining
    empty_right_batch: RecordBatch,
}

impl UnmatchedBuildBatchIterator {
    pub(crate) fn new(
        spatial_index: Arc<RwLock<SpatialIndex>>,
        empty_right_batch: RecordBatch,
    ) -> Result<Self> {
        Ok(Self {
            spatial_index,
            empty_right_batch,
        })
    }

    pub fn next_batch(
        &mut self,
        schema: &Schema,
        join_type: JoinType,
        column_indices: &[ColumnIndex],
        build_side: JoinSide,
    ) -> Result<Option<RecordBatch>> {
        let spatial_index = self.spatial_index.as_ref().read();
        let visited_left_side = spatial_index.visited_left_side();
        let Some(vec_visited_left_side) = visited_left_side else {
            return sedona_internal_err!("The bitmap for visited left side is not created");
        };

        let batch = {
            let visited_bitmap = vec_visited_left_side.lock();
            let (left_side, right_side) =
                get_final_indices_from_bit_map(&visited_bitmap, join_type);

            build_batch_from_indices(
                schema,
                &spatial_index.build_batch.batch,
                &self.empty_right_batch,
                &left_side,
                &right_side,
                column_indices,
                build_side,
            )?
        };

        // Only return non-empty batches
        if batch.num_rows() > 0 {
            return Ok(Some(batch));
        }
        // If batch is empty, continue to next batch

        // No more batches or iteration complete
        Ok(None)
    }
}

// Manual Debug implementation for UnmatchedBuildBatchIterator
impl std::fmt::Debug for UnmatchedBuildBatchIterator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnmatchedBuildBatchIterator").finish()
    }
}
