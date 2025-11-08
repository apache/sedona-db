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

use std::sync::Arc;

use datafusion_common::Result;
use datafusion_common_runtime::JoinSet;
use datafusion_execution::{memory_pool::MemoryReservation, SendableRecordBatchStream};
use datafusion_physical_plan::metrics::{self, ExecutionPlanMetricsSet, MetricBuilder};
use futures::StreamExt;
use sedona_expr::statistics::GeoStatistics;
use sedona_functions::st_analyze_aggr::AnalyzeAccumulator;
use sedona_schema::datatypes::WKB_GEOMETRY;

use crate::{
    evaluated_batch::{
        evaluated_batch_stream::{
            in_mem::InMemoryEvaluatedBatchStream, SendableEvaluatedBatchStream,
        },
        EvaluatedBatch,
    },
    operand_evaluator::OperandEvaluator,
};

pub(crate) struct BuildPartition {
    pub build_side_batch_stream: SendableEvaluatedBatchStream,
    pub geo_statistics: GeoStatistics,

    /// Memory reservation for tracking the memory usage of the build partition
    /// Cleared on `BuildPartition` drop
    pub reservation: MemoryReservation,
}

/// A collector for evaluating the spatial expression on build side batches and collect
/// them as asynchronous streams with additional statistics. The asynchronous streams
/// could then be fed into the spatial index builder to build an in-memory or external
/// spatial index, depending on the statistics collected by the collector.
#[derive(Clone)]
pub(crate) struct BuildSideBatchesCollector {
    evaluator: Arc<dyn OperandEvaluator>,
}

pub(crate) struct CollectBuildSideMetrics {
    /// Number of batches collected
    num_batches: metrics::Count,
    /// Number of rows collected
    num_rows: metrics::Count,
    /// Total in-memory size of batches collected. If the batches were spilled, this size is the
    /// in-memory size if we load all batches into memory. This does not represent the in-memory size
    /// of the resulting BuildPartition.
    total_size_bytes: metrics::Gauge,
    /// Total time taken to collect and process the build side batches. This does not include the time awaiting
    /// for batches from the input stream.
    time_taken: metrics::Time,
}

impl CollectBuildSideMetrics {
    pub fn new(partition: usize, metrics: &ExecutionPlanMetricsSet) -> Self {
        Self {
            num_batches: MetricBuilder::new(metrics).counter("build_input_batches", partition),
            num_rows: MetricBuilder::new(metrics).counter("build_input_rows", partition),
            total_size_bytes: MetricBuilder::new(metrics)
                .gauge("build_input_total_size_bytes", partition),
            time_taken: MetricBuilder::new(metrics)
                .subset_time("build_input_collection_time", partition),
        }
    }
}

impl BuildSideBatchesCollector {
    pub fn new(evaluator: Arc<dyn OperandEvaluator>) -> Self {
        BuildSideBatchesCollector { evaluator }
    }

    pub async fn collect(
        &self,
        mut stream: SendableRecordBatchStream,
        mut reservation: MemoryReservation,
        metrics: &CollectBuildSideMetrics,
    ) -> Result<BuildPartition> {
        let evaluator = self.evaluator.as_ref();
        let mut in_mem_batches: Vec<EvaluatedBatch> = Vec::new();
        let mut analyzer = AnalyzeAccumulator::new(WKB_GEOMETRY, WKB_GEOMETRY);

        while let Some(record_batch) = stream.next().await {
            let record_batch = record_batch?;
            let _timer = metrics.time_taken.timer();

            // Process the record batch and create a BuildSideBatch
            let geom_array = evaluator.evaluate_build(&record_batch)?;

            for wkb in geom_array.wkbs().iter().flatten() {
                analyzer.update_statistics(wkb, wkb.buf().len())?;
            }

            let build_side_batch = EvaluatedBatch {
                batch: record_batch,
                geom_array,
            };

            let in_mem_size = build_side_batch.in_mem_size();
            metrics.num_batches.add(1);
            metrics.num_rows.add(build_side_batch.num_rows());
            metrics.total_size_bytes.add(in_mem_size);

            reservation.try_grow(in_mem_size)?;
            in_mem_batches.push(build_side_batch);
        }

        Ok(BuildPartition {
            build_side_batch_stream: Box::pin(InMemoryEvaluatedBatchStream::new(in_mem_batches)),
            geo_statistics: analyzer.finish(),
            reservation,
        })
    }

    pub async fn collect_all(
        &self,
        streams: Vec<SendableRecordBatchStream>,
        reservations: Vec<MemoryReservation>,
        metrics_vec: Vec<CollectBuildSideMetrics>,
    ) -> Result<Vec<BuildPartition>> {
        if streams.is_empty() {
            return Ok(vec![]);
        }

        // Spawn all tasks to scan all build streams concurrently
        let mut join_set = JoinSet::new();
        for (partition_id, ((stream, metrics), reservation)) in streams
            .into_iter()
            .zip(metrics_vec)
            .zip(reservations)
            .enumerate()
        {
            let collector = self.clone();
            join_set.spawn(async move {
                let result = collector.collect(stream, reservation, &metrics).await;
                (partition_id, result)
            });
        }

        // Wait for all async tasks to finish. Results may be returned in arbitrary order,
        // so we need to reorder them by partition_id later.
        let results = join_set.join_all().await;

        // Reorder results according to partition ids
        let mut partitions: Vec<Option<BuildPartition>> = Vec::with_capacity(results.len());
        partitions.resize_with(results.len(), || None);
        for result in results {
            let (partition_id, partition_result) = result;
            let partition = partition_result?;
            partitions[partition_id] = Some(partition);
        }

        Ok(partitions.into_iter().map(|v| v.unwrap()).collect())
    }
}
