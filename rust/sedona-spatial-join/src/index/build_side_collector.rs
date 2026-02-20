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

use datafusion::config::SpillCompression;
use datafusion_common::{DataFusionError, Result};
use datafusion_common_runtime::JoinSet;
use datafusion_execution::{
    memory_pool::MemoryReservation, runtime_env::RuntimeEnv, SendableRecordBatchStream,
};
use datafusion_physical_plan::metrics::{
    self, ExecutionPlanMetricsSet, MetricBuilder, SpillMetrics,
};
use futures::StreamExt;
use sedona_common::{sedona_internal_err, SpatialJoinOptions};
use sedona_expr::statistics::GeoStatistics;
use sedona_functions::st_analyze_agg::AnalyzeAccumulator;
use sedona_schema::datatypes::WKB_GEOMETRY;

use crate::index::spatial_index_builder::SpatialIndexBuilder;
use crate::index::DefaultSpatialIndexBuilder;
use crate::{
    evaluated_batch::{
        evaluated_batch_stream::{
            evaluate::create_evaluated_build_stream, external::ExternalEvaluatedBatchStream,
            in_mem::InMemoryEvaluatedBatchStream, SendableEvaluatedBatchStream,
        },
        spill::EvaluatedBatchSpillWriter,
        EvaluatedBatch,
    },
    operand_evaluator::{create_operand_evaluator, OperandEvaluator},
    spatial_predicate::SpatialPredicate,
    utils::bbox_sampler::{BoundingBoxSampler, BoundingBoxSamples},
};

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

    /// Memory reservation for tracking the maximum memory usage when collecting
    /// the build side. This reservation won't be freed even when spilling is
    /// triggered. We deliberately only grow the memory reservation to probe
    /// the amount of memory available for loading spatial index into memory.
    /// The size of this reservation will be used to determine the maximum size of
    /// each spatial partition, as well as how many spatial partitions to create.
    pub reservation: MemoryReservation,

    /// Metrics collected during the build side collection phase
    pub metrics: CollectBuildSideMetrics,
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
    runtime_env: Arc<RuntimeEnv>,
    spill_compression: SpillCompression,
}

#[derive(Clone)]
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
    /// Spill metrics of build partitions collecting phase
    spill_metrics: SpillMetrics,
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
            spill_metrics: SpillMetrics::new(metrics, partition),
        }
    }

    pub fn spill_metrics(&self) -> SpillMetrics {
        self.spill_metrics.clone()
    }
}

impl BuildSideBatchesCollector {
    pub fn new(
        spatial_predicate: SpatialPredicate,
        spatial_join_options: SpatialJoinOptions,
        runtime_env: Arc<RuntimeEnv>,
        spill_compression: SpillCompression,
    ) -> Self {
        let evaluator = create_operand_evaluator(&spatial_predicate, spatial_join_options.clone());
        BuildSideBatchesCollector {
            spatial_predicate,
            spatial_join_options,
            evaluator,
            runtime_env,
            spill_compression,
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
        metrics: CollectBuildSideMetrics,
    ) -> Result<BuildPartition> {
        let mut spill_writer_opt = None;
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

            match &mut spill_writer_opt {
                None => {
                    // Collected batches are in memory, no spilling happened for this partition before. We'll try
                    // storing this batch in memory first, and switch to writing everything to disk if we fail
                    // to grow the reservation.
                    in_mem_batches.push(build_side_batch);
                    if let Err(e) = reservation.try_grow(in_mem_size) {
                        log::debug!(
                            "Failed to grow reservation by {} bytes. Current reservation: {} bytes. \
                            num rows: {}, reason: {:?}, Spilling...",
                            in_mem_size,
                            reservation.size(),
                            num_rows,
                            e,
                        );
                        spill_writer_opt =
                            self.spill_in_mem_batches(&mut in_mem_batches, &metrics)?;
                    }
                }
                Some(spill_writer) => {
                    spill_writer.append(&build_side_batch)?;
                }
            }
        }

        let geo_statistics = analyzer.finish();
        let extra_mem = DefaultSpatialIndexBuilder::estimate_extra_memory_usage(
            &geo_statistics,
            &self.spatial_predicate,
            &self.spatial_join_options,
        );

        // Try to grow the reservation a bit more to account for any underestimation of
        // memory usage. We proceed even when the growth fails.
        let additional_reservation = extra_mem + (extra_mem + reservation.size()) / 5;
        if let Err(e) = reservation.try_grow(additional_reservation) {
            log::debug!(
                "Failed to grow reservation by {} bytes to account for spatial index building memory usage. \
                Current reservation: {} bytes. reason: {:?}",
                additional_reservation,
                reservation.size(),
                e,
            );
        }

        // If force spill is enabled, flush everything to disk regardless of whether the memory
        // is enough or not.
        if self.spatial_join_options.debug.force_spill && spill_writer_opt.is_none() {
            log::debug!(
                "Force spilling enabled. Spilling {} in-memory batches to disk.",
                in_mem_batches.len()
            );
            spill_writer_opt = self.spill_in_mem_batches(&mut in_mem_batches, &metrics)?;
        }

        let build_side_batch_stream: SendableEvaluatedBatchStream = match spill_writer_opt {
            Some(spill_writer) => {
                let spill_file = spill_writer.finish()?;
                if !in_mem_batches.is_empty() {
                    return sedona_internal_err!(
                        "In-memory batches should have been spilled when spill file exists"
                    );
                }
                Box::pin(ExternalEvaluatedBatchStream::try_from_spill_file(
                    Arc::new(spill_file),
                )?)
            }
            None => {
                let schema = stream.schema();
                Box::pin(InMemoryEvaluatedBatchStream::new(schema, in_mem_batches))
            }
        };

        let estimated_spatial_index_memory_usage = total_size_bytes + extra_mem;

        Ok(BuildPartition {
            num_rows: total_num_rows,
            build_side_batch_stream,
            geo_statistics,
            bbox_samples: bbox_sampler.into_samples(),
            estimated_spatial_index_memory_usage,
            reservation,
            metrics,
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
                    .collect(evaluated_stream, reservation, bbox_sampler, metrics)
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
                .collect(evaluated_stream, reservation, bbox_sampler, metrics)
                .await?;
            results.push(result);
        }
        Ok(results)
    }

    fn spill_in_mem_batches(
        &self,
        in_mem_batches: &mut Vec<EvaluatedBatch>,
        metrics: &CollectBuildSideMetrics,
    ) -> Result<Option<EvaluatedBatchSpillWriter>> {
        if in_mem_batches.is_empty() {
            return Ok(None);
        }

        let build_side_batch = &in_mem_batches[0];

        let schema = build_side_batch.schema();
        let sedona_type = &build_side_batch.geom_array.sedona_type;
        let mut spill_writer = EvaluatedBatchSpillWriter::try_new(
            Arc::clone(&self.runtime_env),
            schema,
            sedona_type,
            "spilling build side batches",
            self.spill_compression,
            metrics.spill_metrics.clone(),
            if self
                .spatial_join_options
                .spilled_batch_in_memory_size_threshold
                == 0
            {
                None
            } else {
                Some(
                    self.spatial_join_options
                        .spilled_batch_in_memory_size_threshold,
                )
            },
        )?;

        for in_mem_batch in in_mem_batches.iter() {
            spill_writer.append(in_mem_batch)?;
        }

        in_mem_batches.clear();
        Ok(Some(spill_writer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        operand_evaluator::EvaluatedGeometryArray,
        spatial_predicate::{RelationPredicate, SpatialRelationType},
    };
    use arrow_array::{ArrayRef, BinaryArray, Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion_common::ScalarValue;
    use datafusion_execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};
    use datafusion_physical_expr::expressions::Literal;
    use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
    use futures::TryStreamExt;
    use sedona_common::SpatialJoinOptions;
    use sedona_geometry::wkb_factory::wkb_point;
    use sedona_schema::datatypes::WKB_GEOMETRY;

    fn test_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]))
    }

    fn sample_batch(ids: &[i32], wkbs: Vec<Option<Vec<u8>>>) -> Result<EvaluatedBatch> {
        assert_eq!(ids.len(), wkbs.len());
        let id_array = Arc::new(Int32Array::from(ids.to_vec())) as ArrayRef;
        let batch = RecordBatch::try_new(test_schema(), vec![id_array])?;
        let geom_values: Vec<Option<&[u8]>> = wkbs
            .iter()
            .map(|wkb_opt| wkb_opt.as_ref().map(|wkb| wkb.as_slice()))
            .collect();
        let geom_array: ArrayRef = Arc::new(BinaryArray::from(geom_values));
        let geom = EvaluatedGeometryArray::try_new(geom_array, &WKB_GEOMETRY)?;
        Ok(EvaluatedBatch {
            batch,
            geom_array: geom,
        })
    }

    fn build_collector() -> BuildSideBatchesCollector {
        let expr: Arc<dyn datafusion_physical_expr::PhysicalExpr> =
            Arc::new(Literal::new(ScalarValue::Null));
        let predicate = SpatialPredicate::Relation(RelationPredicate::new(
            Arc::clone(&expr),
            expr,
            SpatialRelationType::Intersects,
        ));
        BuildSideBatchesCollector::new(
            predicate,
            SpatialJoinOptions::default(),
            Arc::new(RuntimeEnv::default()),
            SpillCompression::Uncompressed,
        )
    }

    fn memory_reservation(limit: usize) -> (MemoryReservation, Arc<dyn MemoryPool>) {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(limit));
        let consumer = MemoryConsumer::new("build-side-test").with_can_spill(true);
        let reservation = consumer.register(&pool);
        (reservation, pool)
    }

    fn build_stream(batches: Vec<EvaluatedBatch>) -> SendableEvaluatedBatchStream {
        let schema = batches
            .first()
            .map(|batch| batch.schema())
            .unwrap_or_else(test_schema);
        Box::pin(InMemoryEvaluatedBatchStream::new(schema, batches))
    }

    fn collect_ids(batches: &[EvaluatedBatch]) -> Vec<i32> {
        let mut ids = Vec::new();
        for batch in batches {
            let array = batch
                .batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            for i in 0..array.len() {
                ids.push(array.value(i));
            }
        }
        ids
    }

    #[tokio::test]
    async fn collect_keeps_batches_in_memory_when_capacity_suffices() -> Result<()> {
        let collector = build_collector();
        let (reservation, _pool) = memory_reservation(10 * 1024 * 1024);
        let sampler = BoundingBoxSampler::try_new(1, 4, 1.0, 7)?;
        let batch_a = sample_batch(
            &[0, 1],
            vec![
                Some(wkb_point((0.0, 0.0)).unwrap()),
                Some(wkb_point((1.0, 1.0)).unwrap()),
            ],
        )?;
        let batch_b = sample_batch(&[2], vec![Some(wkb_point((2.0, 2.0)).unwrap())])?;
        let stream = build_stream(vec![batch_a, batch_b]);
        let metrics_set = ExecutionPlanMetricsSet::new();
        let metrics = CollectBuildSideMetrics::new(0, &metrics_set);

        let partition = collector
            .collect(stream, reservation, sampler, metrics)
            .await?;
        let stream = partition.build_side_batch_stream;
        let is_external = stream.is_external();
        let batches: Vec<EvaluatedBatch> = stream.try_collect().await?;
        let metrics = &partition.metrics;
        assert!(!is_external, "Expected in-memory batches");
        assert_eq!(collect_ids(&batches), vec![0, 1, 2]);
        assert_eq!(partition.num_rows, 3);
        assert_eq!(metrics.num_batches.value(), 2);
        assert_eq!(metrics.num_rows.value(), 3);
        assert!(metrics.total_size_bytes.value() > 0);
        Ok(())
    }

    #[tokio::test]
    async fn collect_spills_when_reservation_cannot_grow() -> Result<()> {
        let collector = build_collector();
        let sampler = BoundingBoxSampler::try_new(1, 2, 1.0, 13)?;
        let bbox_mem = sampler.estimate_maximum_memory_usage();
        let (reservation, _pool) = memory_reservation(bbox_mem + 1);
        let batch_a = sample_batch(
            &[10, 11],
            vec![
                Some(wkb_point((5.0, 5.0)).unwrap()),
                Some(wkb_point((6.0, 6.0)).unwrap()),
            ],
        )?;
        let batch_b = sample_batch(&[12], vec![Some(wkb_point((7.0, 7.0)).unwrap())])?;
        let stream = build_stream(vec![batch_a, batch_b]);
        let metrics_set = ExecutionPlanMetricsSet::new();
        let metrics = CollectBuildSideMetrics::new(0, &metrics_set);

        let partition = collector
            .collect(stream, reservation, sampler, metrics)
            .await?;
        let stream = partition.build_side_batch_stream;
        let is_external = stream.is_external();
        let batches: Vec<EvaluatedBatch> = stream.try_collect().await?;
        let metrics = &partition.metrics;
        assert!(is_external, "Expected batches to spill to disk");
        assert_eq!(collect_ids(&batches), vec![10, 11, 12]);
        let spill_metrics = metrics.spill_metrics();
        assert!(spill_metrics.spill_file_count.value() >= 1);
        assert!(spill_metrics.spilled_rows.value() >= 1);
        Ok(())
    }

    #[tokio::test]
    async fn collect_handles_empty_stream() -> Result<()> {
        let collector = build_collector();
        let (reservation, _pool) = memory_reservation(1024);
        let sampler = BoundingBoxSampler::try_new(1, 2, 1.0, 19)?;
        let stream = build_stream(Vec::new());
        let metrics_set = ExecutionPlanMetricsSet::new();
        let metrics = CollectBuildSideMetrics::new(0, &metrics_set);

        let partition = collector
            .collect(stream, reservation, sampler, metrics)
            .await?;
        assert_eq!(partition.num_rows, 0);
        let stream = partition.build_side_batch_stream;
        let is_external = stream.is_external();
        let batches: Vec<EvaluatedBatch> = stream.try_collect().await?;
        let metrics = &partition.metrics;
        assert!(!is_external);
        assert!(batches.is_empty());
        assert_eq!(metrics.num_batches.value(), 0);
        assert_eq!(metrics.num_rows.value(), 0);
        Ok(())
    }
}
