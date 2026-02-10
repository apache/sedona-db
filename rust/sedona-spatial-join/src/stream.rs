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
use datafusion::config::SpillCompression;
use datafusion::prelude::SessionConfig;
use datafusion_common::{JoinSide, Result};
use datafusion_execution::runtime_env::RuntimeEnv;
use datafusion_expr::JoinType;
use datafusion_physical_plan::joins::utils::StatefulStreamResult;
use datafusion_physical_plan::joins::utils::{ColumnIndex, JoinFilter};
use datafusion_physical_plan::metrics::{
    self, ExecutionPlanMetricsSet, MetricBuilder, SpillMetrics,
};
use datafusion_physical_plan::{handle_state, RecordBatchStream, SendableRecordBatchStream};
use futures::future::BoxFuture;
use futures::stream::StreamExt;
use futures::{ready, task::Poll, FutureExt};
use parking_lot::Mutex;
use sedona_common::{sedona_internal_err, SedonaOptions};
use sedona_functions::st_analyze_agg::AnalyzeAccumulator;
use sedona_schema::datatypes::WKB_GEOMETRY;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use crate::evaluated_batch::evaluated_batch_stream::evaluate::create_evaluated_probe_stream;
use crate::evaluated_batch::evaluated_batch_stream::SendableEvaluatedBatchStream;
use crate::evaluated_batch::EvaluatedBatch;
use crate::index::{partitioned_index_provider::PartitionedIndexProvider, SpatialIndex};
use crate::operand_evaluator::create_operand_evaluator;
use crate::partitioning::SpatialPartition;
use crate::prepare::SpatialJoinComponents;
use crate::probe::knn_results_merger::KNNResultsMerger;
use crate::probe::partitioned_stream_provider::PartitionedProbeStreamProvider;
use crate::probe::ProbeStreamMetrics;
use crate::spatial_predicate::SpatialPredicate;
use crate::utils::join_utils::{
    adjust_indices_with_visited_info, apply_join_filter_to_indices, build_batch_from_indices,
    get_final_indices_from_bit_map, need_probe_multi_partition_bitmap,
    need_produce_result_in_final,
};
use crate::utils::once_fut::{OnceAsync, OnceFut};
use arrow::array::RecordBatch;
use arrow::datatypes::{Schema, SchemaRef};
use sedona_common::option::SpatialJoinOptions;

/// Stream for producing spatial join result batches.
pub(crate) struct SpatialJoinStream {
    /// The partition id of the probe side stream
    probe_partition_id: usize,
    /// Schema of joined results
    schema: Arc<Schema>,
    /// join filter
    filter: Option<JoinFilter>,
    /// type of the join
    join_type: JoinType,
    /// The stream of the probe side
    probe_stream: Option<SendableEvaluatedBatchStream>,
    /// The schema of the probe side
    probe_stream_schema: SchemaRef,
    /// Information of index and left / right placement of columns
    column_indices: Vec<ColumnIndex>,
    /// Maintains the order of the probe side
    probe_side_ordered: bool,
    /// Join execution metrics
    join_metrics: SpatialJoinProbeMetrics,
    /// Current state of the stream
    state: SpatialJoinStreamState,
    /// DataFusion runtime environment
    runtime_env: Arc<RuntimeEnv>,
    /// Options for the spatial join
    options: SpatialJoinOptions,
    /// Metrics set
    metrics_set: ExecutionPlanMetricsSet,
    /// Target output batch size
    target_output_batch_size: usize,
    /// Spill compression codec
    spill_compression: SpillCompression,
    /// Once future for the shared partitioned index provider
    once_fut_spatial_join_components: OnceFut<SpatialJoinComponents>,
    /// Once async for the provider, disposed by the last finished stream
    once_async_spatial_join_components: Arc<Mutex<Option<OnceAsync<SpatialJoinComponents>>>>,
    /// Cached index provider reference after it becomes available
    index_provider: Option<Arc<PartitionedIndexProvider>>,
    /// Probe side evaluated batch stream provider
    probe_stream_provider: Option<PartitionedProbeStreamProvider>,
    /// The spatial index for the current partition
    spatial_index: Option<Arc<SpatialIndex>>,
    /// The probe-side evaluated batch stream for the current partition
    probe_evaluated_stream: Option<SendableEvaluatedBatchStream>,
    /// Pending future for building or waiting on a partitioned index
    pending_index_future: Option<BoxFuture<'static, Option<Result<Arc<SpatialIndex>>>>>,
    /// Total number of regular partitions produced by the provider
    num_regular_partitions: Option<u32>,
    /// The spatial predicate being evaluated
    spatial_predicate: SpatialPredicate,
    /// Bitmap for tracking visited rows in the Multi partition of the probe side.
    /// This is used for outer joins to ensure that we only emit unmatched rows from the Multi
    /// partition once, after all regular partitions have been processed.
    visited_multi_probe_side: Option<Arc<Mutex<BooleanBufferBuilder>>>,
    /// KNN results merger. Only used for partitioned KNN join. This value is Some when this spatial join stream
    /// is for KNN join and the number of partitions is greater than 1, except when in the
    /// [SpatialJoinStreamState::ProcessProbeBatch] state. The `knn_results_merger` will be moved into the
    /// [SpatialJoinBatchIterator] when processing a probe batch, and moved back to here when the iterator is
    /// complete.
    knn_results_merger: Option<Box<KNNResultsMerger>>,
    /// Current offset in the probe side partition
    probe_offset: usize,
}

impl SpatialJoinStream {
    #[allow(clippy::too_many_arguments)]
    /// Create a new [`SpatialJoinStream`] for a single probe-side input partition.
    ///
    /// This wraps the incoming probe stream with expression evaluation (geometry extraction,
    /// envelopes, etc.) and initializes the stream state machine to wait for shared
    /// `SpatialJoinComponents` (index provider + probe stream provider).
    pub(crate) fn new(
        probe_partition_id: usize,
        schema: Arc<Schema>,
        on: &SpatialPredicate,
        filter: Option<JoinFilter>,
        join_type: JoinType,
        probe_stream: SendableRecordBatchStream,
        column_indices: Vec<ColumnIndex>,
        probe_side_ordered: bool,
        session_config: &SessionConfig,
        runtime_env: Arc<RuntimeEnv>,
        metrics: &ExecutionPlanMetricsSet,
        once_fut_spatial_join_components: OnceFut<SpatialJoinComponents>,
        once_async_spatial_join_components: Arc<Mutex<Option<OnceAsync<SpatialJoinComponents>>>>,
    ) -> Self {
        let target_output_batch_size = session_config.batch_size();
        let spill_compression = session_config.spill_compression();
        let sedona_options = session_config
            .options()
            .extensions
            .get::<SedonaOptions>()
            .cloned()
            .unwrap_or_default();

        let evaluator = create_operand_evaluator(on, sedona_options.spatial_join.clone());
        let join_metrics = SpatialJoinProbeMetrics::new(probe_partition_id, metrics);
        let probe_stream = create_evaluated_probe_stream(
            probe_stream,
            Arc::clone(&evaluator),
            join_metrics.join_time.clone(),
        );
        let probe_stream_schema = probe_stream.schema();

        Self {
            probe_partition_id,
            schema: schema.clone(),
            filter,
            join_type,
            probe_stream: Some(probe_stream),
            probe_stream_schema,
            column_indices,
            probe_side_ordered,
            join_metrics,
            state: SpatialJoinStreamState::WaitPrepareSpatialJoinComponents,
            runtime_env,
            options: sedona_options.spatial_join,
            metrics_set: metrics.clone(),
            target_output_batch_size,
            spill_compression,
            once_fut_spatial_join_components,
            once_async_spatial_join_components,
            index_provider: None,
            probe_stream_provider: None,
            spatial_index: None,
            probe_evaluated_stream: None,
            pending_index_future: None,
            num_regular_partitions: None,
            spatial_predicate: on.clone(),
            visited_multi_probe_side: None,
            knn_results_merger: None,
            probe_offset: 0,
        }
    }
}

/// Metrics for the probe phase of the spatial join.
#[derive(Clone, Debug)]
pub(crate) struct SpatialJoinProbeMetrics {
    /// Metrics produced while scanning/partitioning the probe stream.
    pub(crate) probe_stream_metrics: ProbeStreamMetrics,
    /// Total time for joining probe-side batches to the build-side batches
    pub(crate) join_time: metrics::Time,
    /// Number of output batches produced by this stream.
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
    /// Create a new set of probe metrics for the given output partition.
    pub fn new(partition: usize, metrics: &ExecutionPlanMetricsSet) -> Self {
        Self {
            probe_stream_metrics: ProbeStreamMetrics::new(partition, metrics),
            join_time: MetricBuilder::new(metrics).subset_time("join_time", partition),
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
#[allow(clippy::large_enum_variant)]
pub(crate) enum SpatialJoinStreamState {
    /// The initial mode: waiting for the spatial join components to become available
    WaitPrepareSpatialJoinComponents,
    /// Wait for a specific partition's index. The boolean denotes whether this stream should kick
    /// off building the index (`true`) or simply wait for someone else to build it (`false`).
    WaitBuildIndex(u32, bool),
    /// Indicates that build-side has been collected, and stream is ready for
    /// fetching probe-side batches
    FetchProbeBatch(PartitionDescriptor),
    /// Indicates that we're processing a probe batch using the batch iterator
    ProcessProbeBatch(
        PartitionDescriptor,
        BoxFuture<'static, (Box<SpatialJoinBatchIterator>, Result<Option<RecordBatch>>)>,
    ),
    /// Indicates that we have exhausted the current probe stream, move to the Multi partition
    /// or prepare for emitting unmatched build batch
    ExhaustedProbeStream(PartitionDescriptor),
    /// Indicates that probe-side has been fully processed, prepare iterator for producing
    /// unmatched build side batches for outer join
    PrepareUnmatchedBuildBatch(PartitionDescriptor),
    /// Indicates that we're processing unmatched build-side batches using an iterator
    ProcessUnmatchedBuildBatch(PartitionDescriptor, UnmatchedBuildBatchIterator),
    /// Prepare for processing the next partition.
    /// If the last partition has been processed, simply transfer to [`SpatialJoinStreamState::Completed`];
    /// If the there's still more partitions to process, then transfer to [`SpatialJoinStreamState::WaitBuildIndex`] state.
    /// If we are the last one finishing processing the current partition, we can safely
    /// drop the current index and kick off the building of the index for the next partition.
    PrepareForNextPartition(u32, bool),
    /// Indicates that SpatialJoinStream execution is completed
    Completed,
}

impl std::fmt::Debug for SpatialJoinStreamState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WaitPrepareSpatialJoinComponents => write!(f, "WaitPrepareSpatialJoinComponents"),
            Self::WaitBuildIndex(id, build) => f
                .debug_tuple("WaitBuildIndex")
                .field(id)
                .field(build)
                .finish(),
            Self::FetchProbeBatch(desc) => f.debug_tuple("FetchProbeBatch").field(desc).finish(),
            Self::ProcessProbeBatch(desc, _) => {
                f.debug_tuple("ProcessProbeBatch").field(desc).finish()
            }
            Self::ExhaustedProbeStream(desc) => {
                f.debug_tuple("ExhaustedProbeStream").field(desc).finish()
            }
            Self::PrepareUnmatchedBuildBatch(desc) => f
                .debug_tuple("PrepareUnmatchedBuildBatch")
                .field(desc)
                .finish(),
            Self::ProcessUnmatchedBuildBatch(desc, iter) => f
                .debug_tuple("ProcessUnmatchedBuildBatch")
                .field(desc)
                .field(iter)
                .finish(),
            Self::PrepareForNextPartition(id, last) => f
                .debug_tuple("PrepareForNextPartition")
                .field(id)
                .field(last)
                .finish(),
            Self::Completed => write!(f, "Completed"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PartitionDescriptor {
    partition_id: u32,
    partition: SpatialPartition,
}

impl PartitionDescriptor {
    fn regular(partition_id: u32) -> Self {
        Self {
            partition_id,
            partition: SpatialPartition::Regular(partition_id),
        }
    }

    fn multi(partition_id: u32) -> Self {
        Self {
            partition_id,
            partition: SpatialPartition::Multi,
        }
    }
}

impl SpatialJoinStream {
    fn poll_next_impl(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Result<RecordBatch>>> {
        loop {
            return match &self.state {
                SpatialJoinStreamState::WaitPrepareSpatialJoinComponents => {
                    handle_state!(ready!(self.wait_create_spatial_join_components(cx)))
                }
                SpatialJoinStreamState::WaitBuildIndex(partition_id, should_build) => {
                    handle_state!(ready!(self.wait_build_index(
                        *partition_id,
                        *should_build,
                        cx
                    )))
                }
                SpatialJoinStreamState::FetchProbeBatch(desc) => {
                    handle_state!(ready!(self.fetch_probe_batch(*desc, cx)))
                }
                SpatialJoinStreamState::ProcessProbeBatch(_, _) => {
                    handle_state!(ready!(self.process_probe_batch(cx)))
                }
                SpatialJoinStreamState::ExhaustedProbeStream(desc) => {
                    self.probe_evaluated_stream = None;
                    match desc.partition {
                        SpatialPartition::Regular(_) => {
                            if self.num_regular_partitions == Some(1) {
                                // Single-partition spatial join does not have to process the Multi partition.
                                self.state =
                                    SpatialJoinStreamState::PrepareUnmatchedBuildBatch(*desc);
                            } else {
                                log::debug!(
                                    "[Partition {}] Start probing the Multi partition",
                                    self.probe_partition_id
                                );
                                self.state = SpatialJoinStreamState::FetchProbeBatch(
                                    PartitionDescriptor::multi(desc.partition_id),
                                );
                            }
                        }
                        SpatialPartition::Multi => {
                            self.state = SpatialJoinStreamState::PrepareUnmatchedBuildBatch(*desc);
                        }
                        _ => unreachable!(),
                    }
                    continue;
                }
                SpatialJoinStreamState::PrepareUnmatchedBuildBatch(desc) => {
                    handle_state!(ready!(self.setup_unmatched_build_batch_processing(*desc)))
                }
                SpatialJoinStreamState::ProcessUnmatchedBuildBatch(_, _) => {
                    handle_state!(ready!(self.process_unmatched_build_batch()))
                }
                SpatialJoinStreamState::PrepareForNextPartition(partition_id, is_last_stream) => {
                    handle_state!(ready!(
                        self.prepare_for_next_partition(*partition_id, *is_last_stream)
                    ))
                }
                SpatialJoinStreamState::Completed => {
                    log::debug!("[Partition {}] Completed", self.probe_partition_id);
                    Poll::Ready(None)
                }
            };
        }
    }

    fn wait_create_spatial_join_components(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Result<StatefulStreamResult<Option<RecordBatch>>>> {
        if self.index_provider.is_none() {
            let spatial_join_components =
                ready!(self.once_fut_spatial_join_components.get_shared(cx))?;
            let provider = Arc::clone(&spatial_join_components.partitioned_index_provider);
            self.num_regular_partitions = Some(provider.num_regular_partitions() as u32);
            self.index_provider = Some(provider);

            match self.probe_stream.take() {
                Some(probe_stream) => {
                    let probe_stream_provider = PartitionedProbeStreamProvider::new(
                        Arc::clone(&self.runtime_env),
                        spatial_join_components.probe_stream_options.clone(),
                        probe_stream,
                        self.join_metrics.probe_stream_metrics.clone(),
                    );
                    self.probe_stream_provider = Some(probe_stream_provider);
                }
                None => {
                    return Poll::Ready(sedona_internal_err!("Probe stream should be available"));
                }
            }
        }

        let num_partitions = self
            .num_regular_partitions
            .expect("num_regular_partitions should be available");
        if num_partitions == 0 {
            // Usually does not happen. The indexed side should have at least 1 partition.
            self.state = SpatialJoinStreamState::Completed;
            return Poll::Ready(Ok(StatefulStreamResult::Continue));
        }

        if num_partitions > 1 {
            if let SpatialPredicate::KNearestNeighbors(knn) = &self.spatial_predicate {
                self.knn_results_merger = Some(Box::new(KNNResultsMerger::try_new(
                    knn.k as usize,
                    self.options.knn_include_tie_breakers,
                    self.target_output_batch_size,
                    Arc::clone(&self.runtime_env),
                    self.spill_compression,
                    self.schema.clone(),
                    SpillMetrics::new(&self.metrics_set, self.probe_partition_id),
                )?));
            }
        }

        self.state = SpatialJoinStreamState::WaitBuildIndex(0, true);
        Poll::Ready(Ok(StatefulStreamResult::Continue))
    }

    fn wait_build_index(
        &mut self,
        partition_id: u32,
        should_build: bool,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Result<StatefulStreamResult<Option<RecordBatch>>>> {
        let num_partitions = self
            .num_regular_partitions
            .expect("num_regular_partitions should be available");
        if partition_id >= num_partitions {
            self.state = SpatialJoinStreamState::Completed;
            return Poll::Ready(Ok(StatefulStreamResult::Continue));
        }

        if self.pending_index_future.is_none() {
            let provider = Arc::clone(
                self.index_provider
                    .as_ref()
                    .expect("Partitioned index provider should be available"),
            );
            let future = if should_build {
                log::debug!(
                    "[Partition {}] Building index for spatial partition {}",
                    self.probe_partition_id,
                    partition_id
                );
                async move { provider.build_or_wait_for_index(partition_id).await }.boxed()
            } else {
                log::debug!(
                    "[Partition {}] Waiting for index for spatial partition {}",
                    self.probe_partition_id,
                    partition_id
                );
                async move { provider.wait_for_index(partition_id).await }.boxed()
            };
            self.pending_index_future = Some(future);
        }

        let future = self
            .pending_index_future
            .as_mut()
            .expect("pending future must exist");

        match future.poll_unpin(cx) {
            Poll::Ready(Some(Ok(index))) => {
                self.pending_index_future = None;
                self.spatial_index = Some(index);
                log::debug!(
                    "[Partition {}] Start probing spatial partition {}",
                    self.probe_partition_id,
                    partition_id
                );
                self.state = SpatialJoinStreamState::FetchProbeBatch(PartitionDescriptor::regular(
                    partition_id,
                ));
                Poll::Ready(Ok(StatefulStreamResult::Continue))
            }
            Poll::Ready(Some(Err(err))) => {
                self.pending_index_future = None;
                Poll::Ready(Err(err))
            }
            Poll::Ready(None) => {
                self.pending_index_future = None;
                self.state = SpatialJoinStreamState::Completed;
                Poll::Ready(Ok(StatefulStreamResult::Continue))
            }
            Poll::Pending => Poll::Pending,
        }
    }

    fn fetch_probe_batch(
        &mut self,
        partition_desc: PartitionDescriptor,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Result<StatefulStreamResult<Option<RecordBatch>>>> {
        if self.probe_evaluated_stream.is_none() {
            let probe_stream_provider = self
                .probe_stream_provider
                .as_mut()
                .expect("Probe stream provider should be available");

            // Initialize visited_multi_probe_side if needed
            if matches!(partition_desc.partition, SpatialPartition::Multi) {
                let need_probe_bitmap = need_probe_multi_partition_bitmap(self.join_type);

                if self.visited_multi_probe_side.is_none() && need_probe_bitmap {
                    let num_rows =
                        probe_stream_provider.get_partition_row_count(partition_desc.partition)?;
                    let mut buffer = BooleanBufferBuilder::new(num_rows);
                    buffer.append_n(num_rows, false);
                    self.visited_multi_probe_side = Some(Arc::new(Mutex::new(buffer)));
                }
            }

            let evaluated_stream = probe_stream_provider.stream_for(partition_desc.partition)?;
            self.probe_evaluated_stream = Some(evaluated_stream);
            self.probe_offset = 0;
        }

        let probe_evaluated_stream = self
            .probe_evaluated_stream
            .as_mut()
            .expect("Probe evaluated stream should be available");

        let result = probe_evaluated_stream.poll_next_unpin(cx);
        match result {
            Poll::Ready(Some(Ok(batch))) => {
                let num_rows = batch.num_rows();
                match self.create_spatial_join_iterator(partition_desc, batch, self.probe_offset) {
                    Ok(mut iterator) => {
                        self.probe_offset += num_rows;
                        let future = async move {
                            let result = iterator.next_batch().await;
                            (iterator, result)
                        }
                        .boxed();
                        self.state =
                            SpatialJoinStreamState::ProcessProbeBatch(partition_desc, future);
                        Poll::Ready(Ok(StatefulStreamResult::Continue))
                    }
                    Err(e) => Poll::Ready(Err(e)),
                }
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Err(e)),
            Poll::Ready(None) => {
                self.state = SpatialJoinStreamState::ExhaustedProbeStream(partition_desc);
                Poll::Ready(Ok(StatefulStreamResult::Continue))
            }
            Poll::Pending => Poll::Pending,
        }
    }

    fn process_probe_batch(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Result<StatefulStreamResult<Option<RecordBatch>>>> {
        let _timer = self.join_metrics.join_time.timer();

        // Extract the necessary data first to avoid borrowing conflicts
        let (partition_desc, mut iterator, batch_opt) = match &mut self.state {
            SpatialJoinStreamState::ProcessProbeBatch(desc, future) => {
                match future.poll_unpin(cx) {
                    Poll::Ready((iterator, result)) => {
                        let batch_opt = match result {
                            Ok(opt) => opt,
                            Err(e) => {
                                return Poll::Ready(Err(e));
                            }
                        };
                        (*desc, iterator, batch_opt)
                    }
                    Poll::Pending => return Poll::Pending,
                }
            }
            _ => unreachable!(),
        };

        match batch_opt {
            Some(batch) => {
                self.join_metrics.output_batches.add(1);
                self.join_metrics.output_rows.add(batch.num_rows());

                // Check if iterator is complete
                if iterator.is_complete() {
                    self.knn_results_merger = iterator.take_knn_results_merger();
                    self.state = SpatialJoinStreamState::FetchProbeBatch(partition_desc);
                } else {
                    // Iterator is not complete, continue processing the current probe batch
                    let future = async move {
                        let result = iterator.next_batch().await;
                        (iterator, result)
                    }
                    .boxed();
                    self.state = SpatialJoinStreamState::ProcessProbeBatch(partition_desc, future);
                }
                Poll::Ready(Ok(StatefulStreamResult::Ready(Some(batch))))
            }
            None => {
                // Iterator finished, move to the next probe batch
                self.knn_results_merger = iterator.take_knn_results_merger();
                self.state = SpatialJoinStreamState::FetchProbeBatch(partition_desc);
                Poll::Ready(Ok(StatefulStreamResult::Continue))
            }
        }
    }

    fn setup_unmatched_build_batch_processing(
        &mut self,
        partition_desc: PartitionDescriptor,
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
                .set_max(spatial_index.get_refiner_mem_usage());
            self.join_metrics
                .execution_mode
                .set(spatial_index.get_actual_execution_mode().to_usize());
        }

        // Initial setup for processing unmatched build batches
        if need_produce_result_in_final(self.join_type) {
            // Only produce left-outer batches if this is the last partition that finished probing.
            // This mechanism is similar to the one in NestedLoopJoinStream.
            if !is_last_stream {
                self.state = SpatialJoinStreamState::PrepareForNextPartition(
                    partition_desc.partition_id,
                    is_last_stream,
                );
                return Poll::Ready(Ok(StatefulStreamResult::Continue));
            }

            log::debug!(
                "[Partition {}] Producing unmatched build side for spatial partition {}",
                self.probe_partition_id,
                partition_desc.partition_id
            );

            let empty_right_batch = RecordBatch::new_empty(Arc::clone(&self.probe_stream_schema));

            match UnmatchedBuildBatchIterator::try_new(spatial_index.clone(), empty_right_batch) {
                Ok(iterator) => {
                    self.state = SpatialJoinStreamState::ProcessUnmatchedBuildBatch(
                        partition_desc,
                        iterator,
                    );
                    Poll::Ready(Ok(StatefulStreamResult::Continue))
                }
                Err(e) => Poll::Ready(Err(e)),
            }
        } else {
            // end of the join loop
            self.state = SpatialJoinStreamState::PrepareForNextPartition(
                partition_desc.partition_id,
                is_last_stream,
            );
            Poll::Ready(Ok(StatefulStreamResult::Continue))
        }
    }

    fn process_unmatched_build_batch(
        &mut self,
    ) -> Poll<Result<StatefulStreamResult<Option<RecordBatch>>>> {
        let (partition_desc, batch_opt, is_complete) = match &mut self.state {
            SpatialJoinStreamState::ProcessUnmatchedBuildBatch(desc, iterator) => {
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
                (*desc, batch_opt, is_complete)
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
                    self.state = SpatialJoinStreamState::PrepareForNextPartition(
                        partition_desc.partition_id,
                        true,
                    );
                }

                Poll::Ready(Ok(StatefulStreamResult::Ready(Some(batch))))
            }
            None => {
                // Iterator finished, advance to the next spatial partition
                self.state = SpatialJoinStreamState::PrepareForNextPartition(
                    partition_desc.partition_id,
                    true,
                );
                Poll::Ready(Ok(StatefulStreamResult::Continue))
            }
        }
    }

    fn prepare_for_next_partition(
        &mut self,
        current_partition_id: u32,
        is_last_stream: bool,
    ) -> Poll<Result<StatefulStreamResult<Option<RecordBatch>>>> {
        self.spatial_index = None;
        if is_last_stream {
            if let Some(provider) = self.index_provider.as_ref() {
                provider.dispose_index(current_partition_id);
                assert!(provider.num_loaded_indexes() == 0);
            }
        }

        let num_regular_partitions = self
            .num_regular_partitions
            .expect("num_regular_partitions should be available");

        let next_partition_id = current_partition_id + 1;

        if let Some(merger) = self.knn_results_merger.as_deref_mut() {
            if next_partition_id < num_regular_partitions {
                merger.rotate(next_partition_id == num_regular_partitions - 1)?;
            }
        }

        if next_partition_id >= num_regular_partitions {
            if is_last_stream {
                let mut once_async = self.once_async_spatial_join_components.lock();
                once_async.take();
            }

            self.state = SpatialJoinStreamState::Completed;
            Poll::Ready(Ok(StatefulStreamResult::Continue))
        } else {
            self.state = SpatialJoinStreamState::WaitBuildIndex(next_partition_id, is_last_stream);
            Poll::Ready(Ok(StatefulStreamResult::Continue))
        }
    }

    fn create_spatial_join_iterator(
        &mut self,
        partition_desc: PartitionDescriptor,
        probe_evaluated_batch: EvaluatedBatch,
        probe_offset: usize,
    ) -> Result<Box<SpatialJoinBatchIterator>> {
        // Get the spatial index
        let spatial_index = self
            .spatial_index
            .as_ref()
            .expect("Spatial index should be available");

        // Update the probe side statistics, which may help the spatial index to select a better
        // execution mode for evaluating the spatial predicate.
        if spatial_index.need_more_probe_stats() {
            let mut analyzer = AnalyzeAccumulator::new(WKB_GEOMETRY, WKB_GEOMETRY);
            let geom_array = &probe_evaluated_batch.geom_array;
            for wkb in geom_array.wkbs().iter().flatten() {
                analyzer.update_statistics(wkb)?;
            }
            let stats = analyzer.finish();
            spatial_index.merge_probe_stats(stats);
        }

        let visited_probe_side = if matches!(partition_desc.partition, SpatialPartition::Multi) {
            self.visited_multi_probe_side.clone()
        } else {
            None
        };

        // Produce unmatched probe rows when processing the last build partition's Multi partition
        let num_regular_partitions = self
            .num_regular_partitions
            .expect("num_regular_partitions should be available");
        let is_last_build_partition = matches!(partition_desc.partition, SpatialPartition::Multi)
            && (partition_desc.partition_id + 1) == num_regular_partitions;

        // For KNN joins, we may have swapped build/probe sides, so build_side might be Right;
        // For regular joins, build_side is always Left.
        let build_side = match &self.spatial_predicate {
            SpatialPredicate::KNearestNeighbors(knn) => knn.probe_side.negate(),
            _ => JoinSide::Left,
        };

        // Move out the knn_results_merger to the iterator, we'll move it back when the iterator is complete
        // by calling `SpatialJoinBatchIterator::take_knn_results_merger`.
        let knn_results_merger = std::mem::take(&mut self.knn_results_merger);

        let iterator = SpatialJoinBatchIterator::new(SpatialJoinBatchIteratorParams {
            schema: self.schema.clone(),
            filter: self.filter.clone(),
            join_type: self.join_type,
            column_indices: self.column_indices.clone(),
            build_side,
            spatial_index: spatial_index.clone(),
            join_metrics: self.join_metrics.clone(),
            max_batch_size: self.target_output_batch_size,
            probe_side_ordered: self.probe_side_ordered,
            spatial_predicate: self.spatial_predicate.clone(),
            options: self.options.clone(),
            visited_probe_side,
            probe_offset,
            produce_unmatched_probe_rows: is_last_build_partition,
            probe_evaluated_batch: Arc::new(probe_evaluated_batch),
            knn_results_merger,
        })?;
        Ok(Box::new(iterator))
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

/// Iterator that produces spatial join results for a single probe batch.
///
/// This iterator incrementally probes a [`SpatialIndex`] with rows from an evaluated probe batch,
/// accumulates matched build/probe indices, applies an optional join filter, and finally builds
/// joined [`RecordBatch`]es of up to `max_batch_size` rows.
pub(crate) struct SpatialJoinBatchIterator {
    /// Schema of the output record batches
    schema: SchemaRef,
    /// Optional join filter to be applied to the join results
    filter: Option<JoinFilter>,
    /// Type of the join operation
    join_type: JoinType,
    /// Information of index and left / right placement of columns
    column_indices: Vec<ColumnIndex>,
    /// The side of the build stream, either Left or Right
    build_side: JoinSide,
    /// The spatial index reference
    spatial_index: Arc<SpatialIndex>,
    /// The probe side batch being processed
    probe_evaluated_batch: Arc<EvaluatedBatch>,
    /// Join metrics for tracking performance
    join_metrics: SpatialJoinProbeMetrics,
    /// Maximum batch size before yielding a result
    max_batch_size: usize,
    /// Maintains the order of the probe side
    probe_side_ordered: bool,
    /// The spatial predicate being evaluated
    spatial_predicate: SpatialPredicate,
    /// The spatial join options
    options: SpatialJoinOptions,
    /// Bitmap for tracking visited rows in the probe side. This will only be `Some`
    /// when processing the Multi partition for spatial-partitioned right outer joins.
    visited_probe_side: Option<Arc<Mutex<BooleanBufferBuilder>>>,
    /// Offset of the probe batch in the partition
    offset_in_partition: usize,
    /// Whether produce unmatched probe rows for right outer joins. This is only effective
    /// when [Self::visited_probe_side] is `Some`.
    produce_unmatched_probe_rows: bool,
    /// Progress of probing
    progress: Option<ProbeProgress>,
}

struct ProbeProgress {
    /// Index of the probe row to be probed by [SpatialJoinBatchIterator::probe_range] or
    /// [SpatialJoinBatchIterator::probe_knn].
    current_probe_idx: usize,
    /// Index of the lastly produced probe row. This field uses `-1` as a sentinel value
    /// to represent "nothing produced yet" and is stored as `i64` instead of
    /// `Option<usize>` to keep the layout compact and avoid extra branching and
    /// wrapping/unwrapping in the hot probe loop. There are three cases:
    /// - `-1` means nothing was produced yet.
    /// - `>= num_rows` means we have produced all probe rows. The iterator is complete.
    /// - within `[0, num_rows)` means we have produced up to this probe index (inclusive).
    ///   The value is the largest probe row index that has matching build rows so far.
    last_produced_probe_idx: i64,
    /// Current accumulated build batch positions
    build_batch_positions: Vec<(i32, i32)>,
    /// Current accumulated probe indices. Should have the same length as `build_batch_positions`
    probe_indices: Vec<u32>,
    /// Cursor of the position in the `build_batch_positions` and `probe_indices` vectors
    /// for tracking the progress of producing joined batches
    pos: usize,
    /// KNN-specific progress. Only used for KNN join.
    knn: Option<KNNProbeProgress>,
}

struct KNNProbeProgress {
    /// Accumulated comparable (e.g. squared) distances of the KNN results.
    /// Should have the same length as `build_batch_positions`.
    distances: Vec<f64>,
    /// KNN results merger.
    knn_results_merger: Box<KNNResultsMerger>,
}

/// Type alias for a tuple of build and probe indices slices
type BuildAndProbeIndices<'a> = (&'a [(i32, i32)], &'a [u32], Option<&'a [f64]>);

impl ProbeProgress {
    fn indices_for_next_batch(
        &mut self,
        build_side: JoinSide,
        join_type: JoinType,
        max_batch_size: usize,
    ) -> Option<BuildAndProbeIndices<'_>> {
        let end = self.probe_indices.len();

        // Advance the produced probe end index to skip already hit probe side rows
        // when running probe-semi, probe-anti or probe-mark joins. This is because
        // semi/anti/mark joins only care about whether a probe row has matches,
        // and we don't want to produce duplicate unmatched probe rows when the same
        // probe row P has multiple matches and we split probe_indices range into
        // multiple pieces containing P.
        let should_skip_lastly_produced_probe_rows = matches!(
            (build_side, join_type),
            (
                JoinSide::Left,
                JoinType::RightSemi | JoinType::RightAnti | JoinType::RightMark
            ) | (
                JoinSide::Right,
                JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark
            )
        );
        if should_skip_lastly_produced_probe_rows {
            while self.pos < end
                && self.probe_indices[self.pos] as i64 == self.last_produced_probe_idx
            {
                self.pos += 1;
            }
        }

        if self.pos >= end {
            // No more results to produce. Should switch to Probing or Complete state.
            return None;
        }

        // Take a slice of the accumulated results to produce
        let slice_end = (self.pos + max_batch_size).min(end);
        let build_indices = &self.build_batch_positions[self.pos..slice_end];
        let probe_indices = &self.probe_indices[self.pos..slice_end];
        let distances = self
            .knn
            .as_ref()
            .map(|knn| &knn.distances[self.pos..slice_end]);
        self.pos = slice_end;

        Some((build_indices, probe_indices, distances))
    }

    fn next_probe_range(&mut self, probe_indices: &[u32]) -> Range<usize> {
        let last_produced_probe_idx = self.last_produced_probe_idx;
        let start_probe_idx = if probe_indices[0] as i64 == last_produced_probe_idx {
            last_produced_probe_idx as usize
        } else {
            (last_produced_probe_idx + 1) as usize
        };
        let end_probe_idx = {
            let last_probe_idx = probe_indices[probe_indices.len() - 1] as usize;
            self.last_produced_probe_idx = last_probe_idx as i64;
            last_probe_idx + 1
        };
        start_probe_idx..end_probe_idx
    }

    fn last_probe_range(&mut self, num_rows: usize) -> Option<Range<usize>> {
        // Check if we have already produced all probe rows. There are 2 cases:
        // 1. The last produced probe index is at the end (the last row had matches)
        // 2. We have already called produce_last_result_batch before. Ignore this call.
        if self.last_produced_probe_idx + 1 >= num_rows as i64 {
            self.last_produced_probe_idx = num_rows as i64;
            return None;
        }

        let start_probe_idx = (self.last_produced_probe_idx + 1) as usize;
        let end_probe_idx = num_rows;
        self.last_produced_probe_idx = end_probe_idx as i64;
        Some(start_probe_idx..end_probe_idx)
    }
}

/// Parameters for creating a [`SpatialJoinBatchIterator`].
pub(crate) struct SpatialJoinBatchIteratorParams {
    /// Output schema of the joined record batches.
    pub schema: SchemaRef,
    /// Optional post-join filter to apply.
    pub filter: Option<JoinFilter>,
    /// Join type (inner/outer/semi/anti/mark).
    pub join_type: JoinType,
    /// Column placement mapping for assembling output rows.
    pub column_indices: Vec<ColumnIndex>,
    /// Which input side is treated as the build side.
    pub build_side: JoinSide,
    /// Spatial index for the build side.
    pub spatial_index: Arc<SpatialIndex>,
    /// Probe-side batch with any required pre-evaluated columns.
    pub probe_evaluated_batch: Arc<EvaluatedBatch>,
    /// Metrics instance used to report probe work.
    pub join_metrics: SpatialJoinProbeMetrics,
    /// Maximum output batch size produced per iterator step.
    pub max_batch_size: usize,
    /// Whether to preserve probe-side row ordering when producing output.
    pub probe_side_ordered: bool,
    /// Spatial predicate used to query the index.
    pub spatial_predicate: SpatialPredicate,
    /// Spatial join options affecting behavior (KNN tie-breakers, execution mode, etc.).
    pub options: SpatialJoinOptions,
    /// Optional visited bitmap for multi-partition probe side outer join bookkeeping.
    pub visited_probe_side: Option<Arc<Mutex<BooleanBufferBuilder>>>,
    /// Offset of this probe batch within its partition.
    pub probe_offset: usize,
    /// Whether to emit unmatched probe rows (used for right outer joins).
    pub produce_unmatched_probe_rows: bool,
    /// The KNN result merger for merging KNN results across partitions.
    /// Only available when running KNN join with multiple build partitions.
    pub knn_results_merger: Option<Box<KNNResultsMerger>>,
}

impl SpatialJoinBatchIterator {
    pub(crate) fn new(params: SpatialJoinBatchIteratorParams) -> Result<Self> {
        Ok(Self {
            schema: params.schema,
            filter: params.filter,
            join_type: params.join_type,
            column_indices: params.column_indices,
            build_side: params.build_side,
            spatial_index: params.spatial_index,
            probe_evaluated_batch: params.probe_evaluated_batch,
            join_metrics: params.join_metrics,
            max_batch_size: params.max_batch_size,
            probe_side_ordered: params.probe_side_ordered,
            spatial_predicate: params.spatial_predicate,
            options: params.options,
            visited_probe_side: params.visited_probe_side,
            offset_in_partition: params.probe_offset,
            produce_unmatched_probe_rows: params.produce_unmatched_probe_rows,
            progress: Some(ProbeProgress {
                current_probe_idx: 0,
                last_produced_probe_idx: -1,
                build_batch_positions: Vec::new(),
                probe_indices: Vec::new(),
                pos: 0,
                knn: params.knn_results_merger.map(|merger| KNNProbeProgress {
                    distances: Vec::new(),
                    knn_results_merger: merger,
                }),
            }),
        })
    }

    pub async fn next_batch(&mut self) -> Result<Option<RecordBatch>> {
        let progress_opt = std::mem::take(&mut self.progress);
        let mut progress = progress_opt.expect("Progress should be available");
        let res = self.next_batch_inner(&mut progress).await;
        self.progress = Some(progress);
        res
    }

    async fn next_batch_inner(&self, progress: &mut ProbeProgress) -> Result<Option<RecordBatch>> {
        let num_rows = self.probe_evaluated_batch.num_rows();
        loop {
            // Check if we have produced results for the entire probe batch
            if self.is_complete_inner(progress) {
                return Ok(None);
            }

            // Check if we need to probe more rows
            if progress.current_probe_idx < num_rows
                && progress.probe_indices.len() < self.max_batch_size
            {
                match &self.spatial_predicate {
                    SpatialPredicate::KNearestNeighbors(_) => self.probe_knn(progress)?,
                    _ => self.probe_range(progress).await?,
                }
            }

            // Produce result batch from accumulated results
            let joined_batch_opt = if progress.pos < progress.probe_indices.len() {
                let joined_batch_opt = self.produce_result_batch(progress)?;
                if progress.probe_indices.len() - progress.pos < self.max_batch_size {
                    // Drain produced portion of probe_indices to make it shorter, so that we can
                    // probe more rows using self.probe() in the next iteration.
                    self.drain_produced_indices(progress);
                }
                joined_batch_opt
            } else {
                // No more accumulated results even after probing, we must have reached the end
                self.produce_last_result_batch(progress)?
            };

            if let Some(batch) = joined_batch_opt {
                return Ok(Some(batch));
            }
        }
    }

    async fn probe_range(&self, progress: &mut ProbeProgress) -> Result<()> {
        let num_rows = self.probe_evaluated_batch.num_rows();
        let range = progress.current_probe_idx..num_rows;

        // Calculate remaining capacity in the progress buffer to respect max_batch_size
        let max_result_size = self
            .max_batch_size
            .saturating_sub(progress.probe_indices.len());

        let (metrics, next_row_idx) = self
            .spatial_index
            .query_batch(
                &self.probe_evaluated_batch,
                range,
                max_result_size,
                &mut progress.build_batch_positions,
                &mut progress.probe_indices,
            )
            .await?;

        progress.current_probe_idx = next_row_idx;

        self.join_metrics
            .join_result_candidates
            .add(metrics.candidate_count);
        self.join_metrics.join_result_count.add(metrics.count);

        assert!(
            progress.probe_indices.len() == progress.build_batch_positions.len(),
            "Probe indices and build batch positions length should match"
        );

        Ok(())
    }

    /// Process more probe rows and fill in the build_batch_positions and probe_indices
    /// until we have filled in enough results or processed all probe rows.
    fn probe_knn(&self, progress: &mut ProbeProgress) -> Result<()> {
        let geom_array = &self.probe_evaluated_batch.geom_array;
        let wkbs = geom_array.wkbs();

        // Process from current position until we hit batch size limit or complete
        let num_rows = wkbs.len();
        while progress.current_probe_idx < num_rows {
            // Get WKB for current probe index
            let wkb_opt = &wkbs[progress.current_probe_idx];

            let Some(wkb) = wkb_opt else {
                // Move to next probe index
                progress.current_probe_idx += 1;
                continue;
            };

            // Handle KNN queries differently from regular spatial joins
            if let SpatialPredicate::KNearestNeighbors(knn_predicate) = &self.spatial_predicate {
                // For KNN, call query_knn only once per probe geometry (not per rect)
                let k = knn_predicate.k;
                let use_spheroid = knn_predicate.use_spheroid;
                let include_tie_breakers = self.options.knn_include_tie_breakers;

                let join_result_metrics = self.spatial_index.query_knn(
                    wkb,
                    k,
                    use_spheroid,
                    include_tie_breakers,
                    &mut progress.build_batch_positions,
                    progress.knn.as_mut().map(|knn| &mut knn.distances),
                )?;

                progress.probe_indices.extend(std::iter::repeat_n(
                    progress.current_probe_idx as u32,
                    join_result_metrics.count,
                ));

                self.join_metrics
                    .join_result_candidates
                    .add(join_result_metrics.candidate_count);
                self.join_metrics
                    .join_result_count
                    .add(join_result_metrics.count);
            } else {
                unreachable!("probe_knn called for non-KNN predicate");
            }

            assert!(
                progress.probe_indices.len() == progress.build_batch_positions.len(),
                "Probe indices and build batch positions length should match"
            );
            if let Some(knn) = &progress.knn {
                assert!(
                    knn.distances.len() == progress.probe_indices.len(),
                    "Probe indices and distances length should match"
                );
            }
            progress.current_probe_idx += 1;

            // Early exit if we have enough results
            if progress.build_batch_positions.len() >= self.max_batch_size {
                break;
            }
        }

        Ok(())
    }

    fn produce_result_batch(&self, progress: &mut ProbeProgress) -> Result<Option<RecordBatch>> {
        let need_merge_knn_results = progress.knn.is_some();

        let Some((build_indices, probe_indices, distances)) =
            progress.indices_for_next_batch(self.build_side, self.join_type, self.max_batch_size)
        else {
            // No more results to produce
            return Ok(None);
        };

        let (build_partial_batch, build_indices_array, probe_indices_array, filtered_distances) =
            self.produce_filtered_indices(build_indices, probe_indices.to_vec(), distances)?;

        // Prepare unfiltered indices and distances for KNN joins. This has to be done before calling
        // progress.next_probe_range to make the borrow checker happy.
        let (unfiltered_probe_indices, unfiltered_distances) = if need_merge_knn_results {
            (probe_indices.to_vec(), distances.map(|v| v.to_vec()))
        } else {
            (Vec::new(), None)
        };

        // Produce the final joined batch
        let batch = if !probe_indices_array.is_empty() {
            let probe_indices = probe_indices_array.values().as_ref();
            let probe_range = progress.next_probe_range(probe_indices);
            self.build_joined_batch(
                &build_partial_batch,
                build_indices_array,
                probe_indices_array.clone(),
                probe_range,
            )?
        } else if need_merge_knn_results {
            // For KNN joins, it's possible that after filtering there is no matched result.
            // In this case, we still need to call merge.ingest to update the K-nearest-so-far distances.
            RecordBatch::new_empty(self.schema.clone())
        } else {
            return Ok(None);
        };

        let batch_opt = if let Some(knn) = progress.knn.as_mut() {
            let probe_indices_slice = probe_indices_array.values().as_ref();
            let unfiltered_distances = unfiltered_distances.unwrap_or(Vec::new());
            knn.knn_results_merger.ingest(
                batch,
                probe_indices_slice
                    .iter()
                    .map(|i| (*i as usize) + self.offset_in_partition)
                    .collect(),
                filtered_distances.unwrap_or(Vec::new()),
                unfiltered_probe_indices
                    .iter()
                    .map(|i| (*i as usize) + self.offset_in_partition)
                    .collect(),
                unfiltered_distances,
            )?
        } else {
            Some(batch)
        };

        if batch_opt.iter().any(|b| b.num_rows() > 0) {
            Ok(batch_opt)
        } else {
            Ok(None)
        }
    }

    /// There might be unmatched results at the tail of the probe row range that has not been produced,
    /// even after all matched build/probe row indices have been produced. This function produces
    /// those unmatched results as a final batch.
    fn produce_last_result_batch(
        &self,
        progress: &mut ProbeProgress,
    ) -> Result<Option<RecordBatch>> {
        // Ensure all probe rows have been probed, and all pending results have been produced
        let num_rows = self.probe_evaluated_batch.num_rows();
        assert_eq!(progress.current_probe_idx, num_rows);
        assert_eq!(progress.pos, progress.probe_indices.len());

        // For partitioned KNN joins, flush any pending buffered probe index first.
        // If this produces a batch, return it and let the caller poll again.
        if let Some(knn) = progress.knn.as_mut() {
            let end_offset_in_partition = self.offset_in_partition + num_rows;
            if let Some(batch) = knn
                .knn_results_merger
                .produce_batch_until(end_offset_in_partition)?
            {
                if batch.num_rows() > 0 {
                    return Ok(Some(batch));
                }
            }
        }

        let Some(probe_range) = progress.last_probe_range(num_rows) else {
            return Ok(None);
        };

        // Produce unmatched results in range [last_produced_probe_idx + 1, num_rows)
        let build_schema = self.spatial_index.schema();
        let build_empty_batch = RecordBatch::new_empty(build_schema);
        let build_indices_array = UInt64Array::from(Vec::<u64>::new());
        let probe_indices_array = UInt32Array::from(Vec::<u32>::new());
        let batch = self.build_joined_batch(
            &build_empty_batch,
            build_indices_array,
            probe_indices_array,
            probe_range,
        )?;
        Ok(Some(batch))
    }

    fn drain_produced_indices(&self, progress: &mut ProbeProgress) {
        // Move everything after `pos` to the front
        progress.build_batch_positions.drain(0..progress.pos);
        progress.probe_indices.drain(0..progress.pos);
        if let Some(knn) = &mut progress.knn {
            knn.distances.drain(0..progress.pos);
        }
        progress.pos = 0;
    }

    /// Check if the iterator has finished processing
    pub fn is_complete(&self) -> bool {
        let progress = self
            .progress
            .as_ref()
            .expect("Progress should be available");
        self.is_complete_inner(progress)
    }

    pub fn take_knn_results_merger(&mut self) -> Option<Box<KNNResultsMerger>> {
        assert!(self.is_complete(), "Iterator should be complete");
        let progress = self
            .progress
            .as_mut()
            .expect("Progress should be available");
        progress.knn.take().map(|knn| knn.knn_results_merger)
    }

    fn is_complete_inner(&self, progress: &ProbeProgress) -> bool {
        progress.last_produced_probe_idx >= self.probe_evaluated_batch.batch.num_rows() as i64
    }

    fn produce_filtered_indices(
        &self,
        build_indices: &[(i32, i32)],
        probe_indices: Vec<u32>,
        distances: Option<&[f64]>,
    ) -> Result<(RecordBatch, UInt64Array, UInt32Array, Option<Vec<f64>>)> {
        let PartialBuildBatch {
            batch: partial_build_batch,
            indices: build_indices,
            interleave_indices_map,
        } = self.assemble_partial_build_batch(build_indices)?;
        let probe_indices = UInt32Array::from(probe_indices);

        let (build_indices, probe_indices, filtered_distances) = match &self.filter {
            Some(filter) => apply_join_filter_to_indices(
                &partial_build_batch,
                &self.probe_evaluated_batch.batch,
                build_indices,
                probe_indices,
                distances,
                filter,
                self.build_side,
            )?,
            None => (build_indices, probe_indices, distances.map(|d| d.to_vec())),
        };

        // set the build side bitmap
        if need_produce_result_in_final(self.join_type) {
            if let Some(visited_bitmaps) = self.spatial_index.visited_build_side() {
                mark_build_side_rows_as_visited(
                    &build_indices,
                    &interleave_indices_map,
                    visited_bitmaps,
                );
            }
        }

        Ok((
            partial_build_batch,
            build_indices,
            probe_indices,
            filtered_distances,
        ))
    }

    fn build_joined_batch(
        &self,
        partial_build_batch: &RecordBatch,
        build_indices: UInt64Array,
        probe_indices: UInt32Array,
        probe_range: Range<usize>,
    ) -> Result<RecordBatch> {
        // adjust the two side indices based on the join type
        let (build_indices, probe_indices) = {
            let mut visited_probe_side_guard = self.visited_probe_side.as_ref().map(|v| v.lock());
            let visited_info = visited_probe_side_guard
                .as_mut()
                .map(|buffer| (&mut **buffer, self.offset_in_partition));

            adjust_indices_with_visited_info(
                build_indices,
                probe_indices,
                probe_range,
                self.join_type,
                self.probe_side_ordered,
                visited_info,
                self.produce_unmatched_probe_rows,
            )?
        };

        // Build the final result batch
        build_batch_from_indices(
            &self.schema,
            partial_build_batch,
            &self.probe_evaluated_batch.batch,
            &build_indices,
            &probe_indices,
            &self.column_indices,
            self.build_side,
            self.join_type,
        )
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
            .finish()
    }
}

/// Iterator that produces unmatched build-side rows for outer joins.
///
/// This consumes the build-side visited bitmap(s) attached to the [`SpatialIndex`] and emits
/// rows that were never matched by any probe row.
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
    /// Create an iterator for emitting unmatched build-side rows.
    ///
    /// Fails if the spatial index has not been configured to track visited build-side rows.
    pub(crate) fn try_new(
        spatial_index: Arc<SpatialIndex>,
        empty_right_batch: RecordBatch,
    ) -> Result<Self> {
        let visited_left_side = spatial_index.visited_build_side();
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

    /// Produce the next batch of unmatched build-side rows.
    ///
    /// Returns `Ok(None)` when all build-side batches have been scanned.
    pub fn next_batch(
        &mut self,
        schema: &Schema,
        join_type: JoinType,
        column_indices: &[ColumnIndex],
        build_side: JoinSide,
    ) -> Result<Option<RecordBatch>> {
        while self.current_batch_idx < self.total_batches && !self.is_complete {
            let visited_left_side = self.spatial_index.visited_build_side();
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
                    join_type,
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
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

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

    #[test]
    fn test_produce_joined_indices() {
        for max_batch_size in 1..20 {
            verify_produce_probe_indices(&[], 0, max_batch_size);
            verify_produce_probe_indices(&[0, 0, 0, 0], 1, max_batch_size);
            verify_produce_probe_indices(&[0, 0, 0, 0], 10, max_batch_size);
            verify_produce_probe_indices(&[3, 3, 3], 10, max_batch_size);
            verify_produce_probe_indices(&[0, 0, 3, 3, 3, 6, 7], 10, max_batch_size);
            verify_produce_probe_indices(&[0, 3, 3, 3, 4, 5, 5, 9], 10, max_batch_size);
            verify_produce_probe_indices(&[0, 3, 3, 4, 5, 5, 9, 9], 10, max_batch_size);
        }
    }

    #[test]
    fn test_fuzz_produce_probe_indices() {
        let num_rows_range = 0..100;
        let max_batch_size_range = 1..100;
        let match_probability = 0.5;
        let num_matches_range = 1..100;
        for seed in 0..1000 {
            fuzz_produce_probe_indices(
                num_rows_range.clone(),
                max_batch_size_range.clone(),
                match_probability,
                num_matches_range.clone(),
                seed,
            );
        }
    }

    fn fuzz_produce_probe_indices(
        num_rows_range: Range<usize>,
        max_batch_size_range: Range<usize>,
        match_probability: f64,
        num_matches_range: Range<usize>,
        seed: u64,
    ) {
        let mut rng = StdRng::seed_from_u64(seed);
        let num_rows = rng.random_range(num_rows_range);
        let max_batch_size = rng.random_range(max_batch_size_range);
        let mut probe_indices = Vec::with_capacity(num_rows);
        for row in 0..num_rows {
            let has_matches = rng.random_bool(match_probability);
            if has_matches {
                let num_matches = rng.random_range(num_matches_range.clone());
                probe_indices.extend(std::iter::repeat_n(row as u32, num_matches));
            }
        }
        verify_produce_probe_indices(&probe_indices, num_rows, max_batch_size);
    }

    fn verify_produce_probe_indices(probe_indices: &[u32], num_rows: usize, max_batch_size: usize) {
        for join_type in [
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
        ] {
            let expected_probe_indices =
                produce_probe_indices_once(probe_indices, num_rows, join_type);
            let produced_probe_indices = produce_probe_indices_incrementally(
                probe_indices,
                num_rows,
                max_batch_size,
                join_type,
            );
            assert_eq!(
                expected_probe_indices, produced_probe_indices,
                "Fuzz test failed for num_rows: {}, max_batch_size: {}, probe_indices: {:?}",
                num_rows, max_batch_size, probe_indices
            );
        }
    }

    fn produce_probe_indices_once(
        probe_indices: &[u32],
        num_rows: usize,
        join_type: JoinType,
    ) -> Vec<u32> {
        let build_indices = UInt64Array::from(vec![0; probe_indices.len()]);
        let probe_indices_array = UInt32Array::from(probe_indices.to_vec());
        let probe_range = 0..num_rows;
        let (_, result_probe_indices) = adjust_indices_with_visited_info(
            build_indices,
            probe_indices_array,
            probe_range,
            join_type,
            false,
            None,
            true,
        )
        .unwrap();
        let mut expected_probe_indices = result_probe_indices.values().to_vec();
        expected_probe_indices.sort();
        expected_probe_indices
    }

    fn produce_probe_indices_incrementally(
        probe_indices: &[u32],
        num_rows: usize,
        max_batch_size: usize,
        join_type: JoinType,
    ) -> Vec<u32> {
        let build_batch_positions = vec![(0, 0); probe_indices.len()];
        let mut progress = ProbeProgress {
            current_probe_idx: 0,
            last_produced_probe_idx: -1,
            build_batch_positions,
            probe_indices: probe_indices.to_vec(),
            pos: 0,
            knn: None,
        };
        let mut produced_probe_indices: Vec<u32> = Vec::new();
        while let Some((_, probe_indices, _)) =
            progress.indices_for_next_batch(JoinSide::Left, join_type, max_batch_size)
        {
            let probe_indices = probe_indices.to_vec();
            let adjust_range = progress.next_probe_range(&probe_indices);
            let build_indices = UInt64Array::from(vec![0; probe_indices.len()]);
            let probe_indices = UInt32Array::from(probe_indices);
            let (_, result_probe_indices) = adjust_indices_with_visited_info(
                build_indices,
                probe_indices,
                adjust_range,
                join_type,
                false,
                None,
                true,
            )
            .unwrap();
            produced_probe_indices.extend(result_probe_indices.values().as_ref());
        }
        if let Some(last_range) = progress.last_probe_range(num_rows) {
            let build_indices = UInt64Array::from(Vec::<u64>::new());
            let probe_indices = UInt32Array::from(Vec::<u32>::new());
            let (_, result_probe_indices) = adjust_indices_with_visited_info(
                build_indices,
                probe_indices,
                last_range,
                join_type,
                false,
                None,
                true,
            )
            .unwrap();
            produced_probe_indices.extend(result_probe_indices.values().as_ref());
        }

        produced_probe_indices.sort();
        produced_probe_indices
    }
}
