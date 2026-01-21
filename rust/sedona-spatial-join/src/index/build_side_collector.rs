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

use datafusion_common::{DataFusionError, Result};
use datafusion_common_runtime::JoinSet;
use datafusion_execution::{memory_pool::MemoryReservation, SendableRecordBatchStream};
use datafusion_physical_plan::metrics::{self, ExecutionPlanMetricsSet, MetricBuilder};
use futures::StreamExt;
use sedona_common::SpatialJoinOptions;
use sedona_expr::statistics::GeoStatistics;
use sedona_functions::st_analyze_agg::AnalyzeAccumulator;
use sedona_schema::datatypes::WKB_GEOMETRY;

use crate::{
    evaluated_batch::{
        evaluated_batch_stream::{
            evaluate::create_evaluated_build_stream, in_mem::InMemoryEvaluatedBatchStream,
            SendableEvaluatedBatchStream,
        },
        EvaluatedBatch,
    },
    index::SpatialIndexBuilder,
    operand_evaluator::{create_operand_evaluator, OperandEvaluator},
    spatial_predicate::SpatialPredicate,
    utils::bbox_sampler::{BoundingBoxSampler, BoundingBoxSamples},
};

/// Safety buffer applied when pre-growing build-side reservations to leave headroom for
/// auxiliary structures beyond the build batches themselves.
/// 20% was chosen as a conservative margin.
const BUILD_SIDE_RESERVATION_BUFFER_RATIO: f64 = 0.20;

pub(crate) struct BuildPartition {
    pub num_rows: usize,
    pub build_side_batch_stream: SendableEvaluatedBatchStream,
    pub geo_statistics: GeoStatistics,

    /// Subset of build-side bounding boxes kept for building partitioners (e.g. KDB partitioner)
    /// when the indexed data cannot be fully loaded into memory.
    pub bbox_samples: BoundingBoxSamples,

    /// The estimated memory usage of building spatial index from all the data
    /// collected in this partition. The estimated memory used by the global
    /// spatial index will be the sum of these per-partition estimation.
    pub estimated_spatial_index_memory_usage: usize,

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
    spatial_predicate: SpatialPredicate,
    spatial_join_options: SpatialJoinOptions,
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
    pub fn new(
        spatial_predicate: SpatialPredicate,
        spatial_join_options: SpatialJoinOptions,
    ) -> Self {
        let evaluator = create_operand_evaluator(&spatial_predicate, spatial_join_options.clone());
        BuildSideBatchesCollector {
            spatial_predicate,
            spatial_join_options,
            evaluator,
        }
    }

    /// Collect build-side batches from the stream into a `BuildPartition`.
    ///
    /// This method grows the given memory reservation as if an in-memory spatial
    /// index will be built for all collected batches. If the reservation cannot
    /// be grown, batches are spilled to disk and the reservation is left at its
    /// peak value.
    ///
    /// The reservation represents memory available for loading the spatial index.
    /// Across all partitions, the sum of their reservations forms a soft memory
    /// cap for subsequent spatial join operations. Reservations grown here are
    /// not released until the spatial join operator completes.
    pub async fn collect(
        &self,
        mut stream: SendableEvaluatedBatchStream,
        mut reservation: MemoryReservation,
        mut bbox_sampler: BoundingBoxSampler,
        metrics: &CollectBuildSideMetrics,
    ) -> Result<BuildPartition> {
        let mut in_mem_batches: Vec<EvaluatedBatch> = Vec::new();
        let mut total_num_rows = 0;
        let mut total_size_bytes = 0;
        let mut analyzer = AnalyzeAccumulator::new(WKB_GEOMETRY, WKB_GEOMETRY);

        // Reserve memory for holding bbox samples. This should be a small reservation.
        // We simply return error if the reservation cannot be fulfilled, since there's
        // too little memory for the collector and proceeding will risk overshooting the
        // memory limit.
        reservation.try_grow(bbox_sampler.estimate_maximum_memory_usage())?;

        while let Some(evaluated_batch) = stream.next().await {
            let build_side_batch = evaluated_batch?;
            let _timer = metrics.time_taken.timer();

            let geom_array = &build_side_batch.geom_array;
            for wkb in geom_array.wkbs().iter().flatten() {
                let summary = sedona_geometry::analyze::analyze_geometry(wkb)
                    .map_err(|e| DataFusionError::External(Box::new(e)))?;
                if !summary.bbox.is_empty() {
                    bbox_sampler.add_bbox(&summary.bbox);
                }
                analyzer.ingest_geometry_summary(&summary);
            }

            let num_rows = build_side_batch.num_rows();
            let in_mem_size = build_side_batch.in_mem_size()?;
            total_num_rows += num_rows;
            total_size_bytes += in_mem_size;

            metrics.num_batches.add(1);
            metrics.num_rows.add(num_rows);
            metrics.total_size_bytes.add(in_mem_size);

            in_mem_batches.push(build_side_batch);
            reservation.try_grow(in_mem_size)?;
        }

        let geo_statistics = analyzer.finish();
        let extra_mem = SpatialIndexBuilder::estimate_extra_memory_usage(
            &geo_statistics,
            &self.spatial_predicate,
            &self.spatial_join_options,
        );

        // Try to grow the reservation with a safety buffer to leave room for additional data structures
        let buffer_bytes = ((extra_mem + reservation.size()) as f64
            * BUILD_SIDE_RESERVATION_BUFFER_RATIO)
            .ceil() as usize;
        let additional_reservation = extra_mem + buffer_bytes;
        reservation.try_grow(additional_reservation)?;

        let build_side_batch_stream: SendableEvaluatedBatchStream = {
            let schema = stream.schema();
            Box::pin(InMemoryEvaluatedBatchStream::new(schema, in_mem_batches))
        };

        let estimated_spatial_index_memory_usage = total_size_bytes + extra_mem;

        Ok(BuildPartition {
            num_rows: total_num_rows,
            build_side_batch_stream,
            geo_statistics,
            bbox_samples: bbox_sampler.into_samples(),
            estimated_spatial_index_memory_usage,
            reservation,
        })
    }

    pub async fn collect_all(
        &self,
        streams: Vec<SendableRecordBatchStream>,
        reservations: Vec<MemoryReservation>,
        metrics_vec: Vec<CollectBuildSideMetrics>,
        concurrent: bool,
        seed: u64,
    ) -> Result<Vec<BuildPartition>> {
        if streams.is_empty() {
            return Ok(vec![]);
        }

        assert_eq!(
            streams.len(),
            reservations.len(),
            "each build stream must have a reservation"
        );
        assert_eq!(
            streams.len(),
            metrics_vec.len(),
            "each build stream must have a metrics collector"
        );

        if concurrent {
            self.collect_all_concurrently(streams, reservations, metrics_vec, seed)
                .await
        } else {
            self.collect_all_sequentially(streams, reservations, metrics_vec, seed)
                .await
        }
    }

    async fn collect_all_concurrently(
        &self,
        streams: Vec<SendableRecordBatchStream>,
        reservations: Vec<MemoryReservation>,
        metrics_vec: Vec<CollectBuildSideMetrics>,
        seed: u64,
    ) -> Result<Vec<BuildPartition>> {
        // Spawn task for each stream to scan all streams concurrently
        let mut join_set = JoinSet::new();
        for (partition_id, ((stream, metrics), reservation)) in streams
            .into_iter()
            .zip(metrics_vec)
            .zip(reservations)
            .enumerate()
        {
            let collector = self.clone();
            let evaluator = Arc::clone(&self.evaluator);
            let bbox_sampler = BoundingBoxSampler::try_new(
                self.spatial_join_options.min_index_side_bbox_samples,
                self.spatial_join_options.max_index_side_bbox_samples,
                self.spatial_join_options
                    .target_index_side_bbox_sampling_rate,
                seed.wrapping_add(partition_id as u64),
            )?;
            join_set.spawn(async move {
                let evaluated_stream =
                    create_evaluated_build_stream(stream, evaluator, metrics.time_taken.clone());
                let result = collector
                    .collect(evaluated_stream, reservation, bbox_sampler, &metrics)
                    .await;
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

    async fn collect_all_sequentially(
        &self,
        streams: Vec<SendableRecordBatchStream>,
        reservations: Vec<MemoryReservation>,
        metrics_vec: Vec<CollectBuildSideMetrics>,
        seed: u64,
    ) -> Result<Vec<BuildPartition>> {
        // Collect partitions sequentially (for JNI/embedded contexts)
        let mut results = Vec::with_capacity(streams.len());
        for (partition_id, ((stream, metrics), reservation)) in streams
            .into_iter()
            .zip(metrics_vec)
            .zip(reservations)
            .enumerate()
        {
            let evaluator = Arc::clone(&self.evaluator);
            let bbox_sampler = BoundingBoxSampler::try_new(
                self.spatial_join_options.min_index_side_bbox_samples,
                self.spatial_join_options.max_index_side_bbox_samples,
                self.spatial_join_options
                    .target_index_side_bbox_sampling_rate,
                seed.wrapping_add(partition_id as u64),
            )?;

            let evaluated_stream =
                create_evaluated_build_stream(stream, evaluator, metrics.time_taken.clone());
            let result = self
                .collect(evaluated_stream, reservation, bbox_sampler, &metrics)
                .await?;
            results.push(result);
        }
        Ok(results)
    }
}
