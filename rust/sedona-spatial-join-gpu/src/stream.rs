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

use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::datatypes::SchemaRef;
use arrow_array::RecordBatch;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::context::TaskContext;
use datafusion::physical_plan::{ExecutionPlan, RecordBatchStream, SendableRecordBatchStream};
use datafusion_physical_plan::metrics::{self, ExecutionPlanMetricsSet, MetricBuilder};
use futures::stream::Stream;

use crate::gpu_backend::GpuBackend;
use std::time::Instant;

/// Stream that executes GPU spatial join
///
/// This stream manages the entire GPU spatial join lifecycle:
/// 1. Initialize GPU context
/// 2. Read data from left child stream
/// 3. Read data from right child stream
/// 4. Execute GPU spatial join
/// 5. Emit result batches
///    Metrics for GPU spatial join operations
pub(crate) struct GpuSpatialJoinMetrics {
    /// Total time for GPU join execution
    pub(crate) join_time: metrics::Time,
    /// Time for batch concatenation
    pub(crate) concat_time: metrics::Time,
    /// Time for GPU kernel execution
    pub(crate) gpu_kernel_time: metrics::Time,
    /// Number of batches produced by this operator
    pub(crate) output_batches: metrics::Count,
    /// Number of rows produced by this operator
    pub(crate) output_rows: metrics::Count,
}

impl GpuSpatialJoinMetrics {
    pub fn new(partition: usize, metrics: &ExecutionPlanMetricsSet) -> Self {
        Self {
            join_time: MetricBuilder::new(metrics).subset_time("join_time", partition),
            concat_time: MetricBuilder::new(metrics).subset_time("concat_time", partition),
            gpu_kernel_time: MetricBuilder::new(metrics).subset_time("gpu_kernel_time", partition),
            output_batches: MetricBuilder::new(metrics).counter("output_batches", partition),
            output_rows: MetricBuilder::new(metrics).counter("output_rows", partition),
        }
    }
}

pub(crate) struct GpuSpatialJoinStream {
    /// Right child execution plan (probe side)
    right: Arc<dyn ExecutionPlan>,

    /// Output schema
    schema: SchemaRef,

    /// Task context
    context: Arc<TaskContext>,

    /// GPU backend for spatial operations
    gpu_backend: Option<GpuBackend>,

    /// Current state of the stream
    state: GpuJoinState,

    /// Result batches to emit
    result_batches: VecDeque<RecordBatch>,

    /// Right side batches (accumulated before GPU transfer)
    right_batches: Vec<RecordBatch>,

    /// Right child stream
    right_stream: Option<SendableRecordBatchStream>,

    /// Partition number to execute
    partition: usize,

    /// Metrics for this join operation
    join_metrics: GpuSpatialJoinMetrics,

    /// Shared build data (left side) from build phase
    once_build_data: crate::once_fut::OnceFut<crate::build_data::GpuBuildData>,
}

/// State machine for GPU spatial join execution
#[derive(Debug)]
enum GpuJoinState {
    /// Initialize GPU context
    Init,

    /// Initialize right child stream
    InitRightStream,

    /// Reading batches from right stream
    ReadRightStream,

    /// Execute GPU spatial join (awaits left-side build data)
    ExecuteGpuJoin,

    /// Emit result batches
    EmitResults,

    /// All results emitted, stream complete
    Done,

    /// Error occurred, stream failed
    Failed(String),
}

impl GpuSpatialJoinStream {
    /// Create a new GPU spatial join stream for probe phase
    ///
    /// This constructor is called per output partition and creates a stream that:
    /// 1. Awaits shared left-side build data from once_build_data
    /// 2. Reads the right partition specified by `partition` parameter
    /// 3. Executes GPU join between shared left data and this partition's right data
    pub fn new_probe(
        once_build_data: crate::once_fut::OnceFut<crate::build_data::GpuBuildData>,
        right: Arc<dyn ExecutionPlan>,
        schema: SchemaRef,
        context: Arc<TaskContext>,
        partition: usize,
        metrics: &ExecutionPlanMetricsSet,
    ) -> Result<Self> {
        Ok(Self {
            right,
            schema,
            context,
            gpu_backend: None,
            state: GpuJoinState::Init,
            result_batches: VecDeque::new(),
            right_batches: Vec::new(),
            right_stream: None,
            partition,
            join_metrics: GpuSpatialJoinMetrics::new(partition, metrics),
            once_build_data,
        })
    }

    /// Poll the stream for next batch
    fn poll_next_impl(&mut self, _cx: &mut Context<'_>) -> Poll<Option<Result<RecordBatch>>> {
        loop {
            match &self.state {
                GpuJoinState::Init => {
                    log::info!("Initializing GPU backend for spatial join");
                    match self.initialize_gpu() {
                        Ok(()) => {
                            log::debug!("GPU backend initialized successfully");
                            self.state = GpuJoinState::InitRightStream;
                        }
                        Err(e) => {
                            // Note: fallback_to_cpu config is in GpuBuildData, will be checked in ExecuteGpuJoin
                            log::error!("GPU initialization failed: {}", e);
                            self.state = GpuJoinState::Failed(e.to_string());
                            return Poll::Ready(Some(Err(e)));
                        }
                    }
                }

                GpuJoinState::InitRightStream => {
                    log::debug!(
                        "Initializing right child stream for partition {}",
                        self.partition
                    );
                    match self.right.execute(self.partition, self.context.clone()) {
                        Ok(stream) => {
                            self.right_stream = Some(stream);
                            self.state = GpuJoinState::ReadRightStream;
                        }
                        Err(e) => {
                            log::error!("Failed to execute right child: {}", e);
                            self.state = GpuJoinState::Failed(e.to_string());
                            return Poll::Ready(Some(Err(e)));
                        }
                    }
                }

                GpuJoinState::ReadRightStream => {
                    if let Some(stream) = &mut self.right_stream {
                        match Pin::new(stream).poll_next(_cx) {
                            Poll::Ready(Some(Ok(batch))) => {
                                log::debug!("Received right batch with {} rows", batch.num_rows());
                                self.right_batches.push(batch);
                                // Continue reading more batches
                                continue;
                            }
                            Poll::Ready(Some(Err(e))) => {
                                log::error!("Error reading right stream: {}", e);
                                self.state = GpuJoinState::Failed(e.to_string());
                                return Poll::Ready(Some(Err(e)));
                            }
                            Poll::Ready(None) => {
                                // Right stream complete for this partition
                                let total_right_rows: usize =
                                    self.right_batches.iter().map(|b| b.num_rows()).sum();
                                log::debug!(
                                    "Read {} right batches with total {} rows from partition {}",
                                    self.right_batches.len(),
                                    total_right_rows,
                                    self.partition
                                );
                                // Move to execute GPU join with this partition's right data
                                self.state = GpuJoinState::ExecuteGpuJoin;
                            }
                            Poll::Pending => {
                                return Poll::Pending;
                            }
                        }
                    } else {
                        self.state = GpuJoinState::Failed("Right stream not initialized".into());
                        return Poll::Ready(Some(Err(DataFusionError::Execution(
                            "Right stream not initialized".into(),
                        ))));
                    }
                }

                GpuJoinState::ExecuteGpuJoin => {
                    log::info!("Awaiting build data and executing GPU spatial join");

                    // Poll the shared build data future
                    let build_data = match futures::ready!(self.once_build_data.get_shared(_cx)) {
                        Ok(data) => data,
                        Err(e) => {
                            log::error!("Failed to get build data: {}", e);
                            self.state = GpuJoinState::Failed(e.to_string());
                            return Poll::Ready(Some(Err(e)));
                        }
                    };

                    log::debug!(
                        "Build data received: {} left rows",
                        build_data.left_row_count
                    );

                    // Execute GPU join with build data
                    match self.execute_gpu_join_with_build_data(&build_data) {
                        Ok(()) => {
                            log::info!(
                                "GPU join completed, produced {} result batches",
                                self.result_batches.len()
                            );
                            self.state = GpuJoinState::EmitResults;
                        }
                        Err(e) => {
                            log::error!("GPU spatial join failed: {}", e);
                            self.state = GpuJoinState::Failed(e.to_string());
                            return Poll::Ready(Some(Err(e)));
                        }
                    }
                }

                GpuJoinState::EmitResults => {
                    if let Some(batch) = self.result_batches.pop_front() {
                        log::debug!("Emitting result batch with {} rows", batch.num_rows());
                        return Poll::Ready(Some(Ok(batch)));
                    }
                    log::debug!("All results emitted, stream complete");
                    self.state = GpuJoinState::Done;
                }

                GpuJoinState::Done => {
                    return Poll::Ready(None);
                }

                GpuJoinState::Failed(msg) => {
                    return Poll::Ready(Some(Err(DataFusionError::Execution(format!(
                        "GPU spatial join failed: {}",
                        msg
                    )))));
                }
            }
        }
    }

    /// Initialize GPU backend
    fn initialize_gpu(&mut self) -> Result<()> {
        // Use device 0 by default - actual device config is in GpuBuildData
        // but we need to initialize GPU context early in the Init state
        let mut backend = GpuBackend::new(0).map_err(|e| {
            DataFusionError::Execution(format!("GPU backend creation failed: {}", e))
        })?;
        backend
            .init()
            .map_err(|e| DataFusionError::Execution(format!("GPU initialization failed: {}", e)))?;
        self.gpu_backend = Some(backend);
        Ok(())
    }

    /// Execute GPU spatial join with build data
    fn execute_gpu_join_with_build_data(
        &mut self,
        build_data: &crate::build_data::GpuBuildData,
    ) -> Result<()> {
        let gpu_backend = self
            .gpu_backend
            .as_mut()
            .ok_or_else(|| DataFusionError::Execution("GPU backend not initialized".into()))?;

        let left_batch = build_data.left_batch();
        let config = build_data.config();

        // Check if we have data to join
        if left_batch.num_rows() == 0 || self.right_batches.is_empty() {
            log::warn!(
                "No data to join (left: {} rows, right: {} batches)",
                left_batch.num_rows(),
                self.right_batches.len()
            );
            // Create empty result with correct schema
            let empty_batch = RecordBatch::new_empty(self.schema.clone());
            self.result_batches.push_back(empty_batch);
            return Ok(());
        }

        let _join_timer = self.join_metrics.join_time.timer();

        log::info!(
            "Processing GPU join with {} left rows and {} right batches",
            left_batch.num_rows(),
            self.right_batches.len()
        );

        // Concatenate all right batches into one batch
        let _concat_timer = self.join_metrics.concat_time.timer();
        let right_batch = if self.right_batches.len() == 1 {
            self.right_batches[0].clone()
        } else {
            let schema = self.right_batches[0].schema();
            let result =
                arrow::compute::concat_batches(&schema, &self.right_batches).map_err(|e| {
                    DataFusionError::Execution(format!(
                        "Failed to concatenate right batches: {}",
                        e
                    ))
                })?;
            result
        };

        log::info!(
            "Using build data: {} left rows, {} right rows",
            left_batch.num_rows(),
            right_batch.num_rows()
        );

        // Concatenation time is tracked by concat_time timer

        // Execute GPU spatial join on concatenated batches
        let _gpu_kernel_timer = self.join_metrics.gpu_kernel_time.timer();
        let result_batch = gpu_backend
            .spatial_join(
                left_batch,
                &right_batch,
                config.left_geom_column.index,
                config.right_geom_column.index,
                config.predicate.into(),
            )
            .map_err(|e| {
                if config.fallback_to_cpu {
                    log::warn!("GPU join failed: {}, should fallback to CPU", e);
                }
                DataFusionError::Execution(format!("GPU spatial join execution failed: {}", e))
            })?;

        log::info!("GPU join produced {} rows", result_batch.num_rows());

        // Only add non-empty result batch
        if result_batch.num_rows() > 0 {
            self.join_metrics.output_batches.add(1);
            self.join_metrics.output_rows.add(result_batch.num_rows());
            self.result_batches.push_back(result_batch);
        }

        Ok(())
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

// Convert GpuSpatialPredicate to libgpuspatial SpatialPredicate
impl From<crate::config::GpuSpatialPredicate> for sedona_libgpuspatial::SpatialPredicate {
    fn from(pred: crate::config::GpuSpatialPredicate) -> Self {
        match pred {
            crate::config::GpuSpatialPredicate::Relation(p) => p,
        }
    }
}
