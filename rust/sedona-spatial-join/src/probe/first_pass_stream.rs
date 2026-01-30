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

use std::{
    collections::VecDeque,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use datafusion_common::{DataFusionError, Result};
use futures::{Stream, StreamExt};
use sedona_common::sedona_internal_err;

use crate::probe::ProbeStreamMetrics;
use crate::{
    evaluated_batch::{
        evaluated_batch_stream::{EvaluatedBatchStream, SendableEvaluatedBatchStream},
        EvaluatedBatch,
    },
    partitioning::{
        stream_repartitioner::{
            assign_rows, interleave_evaluated_batch, SpilledPartitions, StreamRepartitioner,
        },
        PartitionedSide, SpatialPartition, SpatialPartitioner,
    },
};

pub(crate) trait FirstPassStreamCallback {
    fn call(self, result: Result<SpilledPartitions>) -> Result<()>;
}

impl<F: FnOnce(Result<SpilledPartitions>) -> Result<()>> FirstPassStreamCallback for F {
    fn call(self, result: Result<SpilledPartitions>) -> Result<()> {
        self(result)
    }
}

pub(crate) struct FirstPassStream<C: FirstPassStreamCallback> {
    source: SendableEvaluatedBatchStream,
    repartitioner: Option<StreamRepartitioner>,
    partitioner: Arc<dyn SpatialPartitioner>,
    pending_output: VecDeque<Result<EvaluatedBatch>>,
    metrics: ProbeStreamMetrics,
    callback: Option<C>,
}

impl<C: FirstPassStreamCallback> FirstPassStream<C> {
    pub fn new(
        source: SendableEvaluatedBatchStream,
        repartitioner: StreamRepartitioner,
        partitioner: Arc<dyn SpatialPartitioner>,
        metrics: ProbeStreamMetrics,
        callback: C,
    ) -> Self {
        Self {
            source,
            repartitioner: Some(repartitioner),
            partitioner,
            pending_output: VecDeque::new(),
            metrics,
            callback: Some(callback),
        }
    }

    fn finish_first_pass(&mut self) -> Result<()> {
        let repartitioner = self.repartitioner.take().ok_or_else(|| {
            DataFusionError::Internal("First pass repartitioner already finished".into())
        })?;
        let parts = repartitioner.finish()?;
        let callback_opt = self.callback.take();
        match callback_opt {
            Some(callback) => callback.call(Ok(parts)),
            None => sedona_internal_err!("Callback has already been called"),
        }
    }

    fn transition_to_failed(&mut self, err: DataFusionError) -> DataFusionError {
        let err_arc = Arc::new(err);
        let callback_opt = self.callback.take();
        if let Some(callback) = callback_opt {
            callback
                .call(Err(DataFusionError::Shared(err_arc.clone())))
                .ok();
        }
        DataFusionError::Shared(err_arc)
    }
}

impl<C: FirstPassStreamCallback + Unpin> EvaluatedBatchStream for FirstPassStream<C> {
    fn is_external(&self) -> bool {
        false
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        self.source.schema()
    }
}

impl<C: FirstPassStreamCallback + Unpin> Stream for FirstPassStream<C> {
    type Item = Result<EvaluatedBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        loop {
            if let Some(result) = this.pending_output.pop_front() {
                return Poll::Ready(Some(result));
            }

            match this.source.poll_next_unpin(cx) {
                Poll::Ready(Some(Ok(batch))) => {
                    this.metrics.probe_input_batches.add(1);
                    this.metrics.probe_input_rows.add(batch.num_rows());

                    let first_pass_split_result = {
                        let _timer = this.metrics.repartition_time.timer();
                        split_for_first_pass(batch, this.partitioner.as_ref())
                    };

                    let split = match first_pass_split_result {
                        Ok(split) => split,
                        Err(err) => {
                            let err = this.transition_to_failed(err);
                            return Poll::Ready(Some(Err(err)));
                        }
                    };

                    if let Some(batch) = split.produced {
                        this.pending_output.push_back(Ok(batch));
                    }

                    if let Some((spill_batch, assignments)) = split.spilled {
                        if let Some(repartitioner) = this.repartitioner.as_mut() {
                            if let Err(err) =
                                repartitioner.insert_repartitioned_batch(spill_batch, &assignments)
                            {
                                let err = this.transition_to_failed(err);
                                return Poll::Ready(Some(Err(err)));
                            }
                        }
                    }
                }
                Poll::Ready(Some(Err(e))) => {
                    let err = this.transition_to_failed(e);
                    return Poll::Ready(Some(Err(err)));
                }
                Poll::Ready(None) => {
                    if let Err(err) = this.finish_first_pass() {
                        let err = this.transition_to_failed(err);
                        return Poll::Ready(Some(Err(err)));
                    }
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// The result of splitting an evaluated batch for producing batches for the first pass.
/// Some rows are produced directly, while others are spilled for further processing.
struct FirstPassSplit {
    /// Rows assigned to Regular(0) or None partition are produced directly
    produced: Option<EvaluatedBatch>,
    /// Rows assigned to other partitions are spilled for further passes
    spilled: Option<(EvaluatedBatch, Vec<SpatialPartition>)>,
}

fn split_for_first_pass(
    batch: EvaluatedBatch,
    partitioner: &dyn SpatialPartitioner,
) -> Result<FirstPassSplit> {
    let mut assignments = Vec::new();
    assign_rows(
        &batch,
        partitioner,
        PartitionedSide::ProbeSide,
        &mut assignments,
    )?;

    let record_batches = vec![&batch.batch];
    let geom_arrays = vec![&batch.geom_array];

    let mut regular_assignments = Vec::new();
    let mut spill_assignments = Vec::new();
    let mut spill_partitions = Vec::new();

    for (row_idx, partition) in assignments.into_iter().enumerate() {
        match partition {
            SpatialPartition::Regular(0) | SpatialPartition::None => {
                regular_assignments.push((0, row_idx))
            }
            other => {
                spill_assignments.push((0, row_idx));
                spill_partitions.push(other);
            }
        }
    }

    let regular0 = if regular_assignments.is_empty() {
        None
    } else {
        Some(interleave_evaluated_batch(
            &record_batches,
            &geom_arrays,
            &regular_assignments,
        )?)
    };

    let spill_batch = if spill_assignments.is_empty() {
        None
    } else {
        let batch = interleave_evaluated_batch(&record_batches, &geom_arrays, &spill_assignments)?;
        Some((batch, spill_partitions))
    };

    Ok(FirstPassSplit {
        produced: regular0,
        spilled: spill_batch,
    })
}
