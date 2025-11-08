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
use arrow_schema::SchemaRef;
use datafusion_physical_plan::metrics::{self, ExecutionPlanMetricsSet, MetricBuilder};
use sedona_common::SpatialJoinOptions;
use sedona_expr::statistics::GeoStatistics;

use datafusion_common::{utils::proxy::VecAllocExt, Result};
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryPool, MemoryReservation};
use datafusion_expr::JoinType;
use futures::StreamExt;
use geo_index::rtree::{sort::HilbertSort, RTree, RTreeBuilder};
use parking_lot::Mutex;
use std::sync::{atomic::AtomicUsize, Arc};

use crate::{
    evaluated_batch::EvaluatedBatch,
    index::{knn_adapter::KnnComponents, spatial_index::SpatialIndex, BuildPartition},
    operand_evaluator::create_operand_evaluator,
    refine::create_refiner,
    spatial_predicate::SpatialPredicate,
    utils::{
        concurrent_reservation::ConcurrentReservation, join_utils::need_produce_result_in_final,
    },
};

// Type aliases for better readability
type SpatialRTree = RTree<f32>;
type DataIdToBatchPos = Vec<(i32, i32)>;
type RTreeBuildResult = (SpatialRTree, DataIdToBatchPos);

/// Rough estimate for in-memory size of the rtree per rect in bytes
const RTREE_MEMORY_ESTIMATE_PER_RECT: usize = 60;

/// The prealloc size for the refiner reservation. This is used to reduce the frequency of growing
/// the reservation when updating the refiner memory reservation.
const REFINER_RESERVATION_PREALLOC_SIZE: usize = 10 * 1024 * 1024; // 10MB

/// Builder for constructing a SpatialIndex from geometry batches.
///
/// This builder handles:
/// 1. Accumulating geometry batches to be indexed
/// 2. Building the spatial R-tree index
/// 3. Setting up memory tracking and visited bitmaps
/// 4. Configuring prepared geometries based on execution mode
pub(crate) struct SpatialIndexBuilder {
    schema: SchemaRef,
    spatial_predicate: SpatialPredicate,
    options: SpatialJoinOptions,
    join_type: JoinType,
    probe_threads_count: usize,
    metrics: SpatialJoinBuildMetrics,

    /// Batches to be indexed
    indexed_batches: Vec<EvaluatedBatch>,
    /// Memory reservation for tracking the memory usage of the spatial index
    reservation: MemoryReservation,

    /// Statistics for indexed geometries
    stats: GeoStatistics,

    /// Memory pool for managing the memory usage of the spatial index
    memory_pool: Arc<dyn MemoryPool>,
}

/// Metrics for the build phase of the spatial join.
#[derive(Clone, Debug, Default)]
pub(crate) struct SpatialJoinBuildMetrics {
    /// Total time for collecting build-side of join
    pub(crate) build_time: metrics::Time,
    /// Memory used by the spatial-index in bytes
    pub(crate) build_mem_used: metrics::Gauge,
}

impl SpatialJoinBuildMetrics {
    pub fn new(partition: usize, metrics: &ExecutionPlanMetricsSet) -> Self {
        Self {
            build_time: MetricBuilder::new(metrics).subset_time("build_time", partition),
            build_mem_used: MetricBuilder::new(metrics).gauge("build_mem_used", partition),
        }
    }
}

impl SpatialIndexBuilder {
    /// Create a new builder with the given configuration.
    pub fn new(
        schema: SchemaRef,
        spatial_predicate: SpatialPredicate,
        options: SpatialJoinOptions,
        join_type: JoinType,
        probe_threads_count: usize,
        memory_pool: Arc<dyn MemoryPool>,
        metrics: SpatialJoinBuildMetrics,
    ) -> Result<Self> {
        let consumer = MemoryConsumer::new("SpatialJoinIndex");
        let reservation = consumer.register(&memory_pool);

        Ok(Self {
            schema,
            spatial_predicate,
            options,
            join_type,
            probe_threads_count,
            metrics,
            indexed_batches: Vec::new(),
            reservation,
            stats: GeoStatistics::empty(),
            memory_pool,
        })
    }

    /// Add a geometry batch to be indexed.
    ///
    /// This method accumulates geometry batches that will be used to build the spatial index.
    /// Each batch contains processed geometry data along with memory usage information.
    pub fn add_batch(&mut self, indexed_batch: EvaluatedBatch) {
        let in_mem_size = indexed_batch.in_mem_size();
        self.indexed_batches.push(indexed_batch);
        self.reservation.grow(in_mem_size);
        self.metrics.build_mem_used.add(in_mem_size);
    }

    pub fn merge_stats(&mut self, stats: GeoStatistics) -> &mut Self {
        self.stats.merge(&stats);
        self
    }

    /// Build the spatial R-tree index from collected geometry batches.
    fn build_rtree(&mut self) -> Result<RTreeBuildResult> {
        let build_timer = self.metrics.build_time.timer();

        let num_rects = self
            .indexed_batches
            .iter()
            .map(|batch| batch.rects().iter().flatten().count())
            .sum::<usize>();

        let mut rtree_builder = RTreeBuilder::<f32>::new(num_rects as u32);
        let mut batch_pos_vec = vec![(-1, -1); num_rects];
        let rtree_mem_estimate = num_rects * RTREE_MEMORY_ESTIMATE_PER_RECT;

        self.reservation
            .grow(batch_pos_vec.allocated_size() + rtree_mem_estimate);

        for (batch_idx, batch) in self.indexed_batches.iter().enumerate() {
            let rects = batch.rects();
            for (idx, rect_opt) in rects.iter().enumerate() {
                let Some(rect) = rect_opt else {
                    continue;
                };
                let min = rect.min();
                let max = rect.max();
                let data_idx = rtree_builder.add(min.x, min.y, max.x, max.y);
                batch_pos_vec[data_idx as usize] = (batch_idx as i32, idx as i32);
            }
        }

        let rtree = rtree_builder.finish::<HilbertSort>();
        build_timer.done();

        self.metrics.build_mem_used.add(self.reservation.size());

        Ok((rtree, batch_pos_vec))
    }

    /// Build visited bitmaps for tracking left-side indices in outer joins.
    fn build_visited_bitmaps(&mut self) -> Result<Option<Mutex<Vec<BooleanBufferBuilder>>>> {
        if !need_produce_result_in_final(self.join_type) {
            return Ok(None);
        }

        let mut bitmaps = Vec::with_capacity(self.indexed_batches.len());
        let mut total_buffer_size = 0;

        for batch in &self.indexed_batches {
            let batch_rows = batch.batch.num_rows();
            let buffer_size = batch_rows.div_ceil(8);
            total_buffer_size += buffer_size;

            let mut bitmap = BooleanBufferBuilder::new(batch_rows);
            bitmap.append_n(batch_rows, false);
            bitmaps.push(bitmap);
        }

        self.reservation.try_grow(total_buffer_size)?;
        self.metrics.build_mem_used.add(total_buffer_size);

        Ok(Some(Mutex::new(bitmaps)))
    }

    /// Create an rtree data index to consecutive index mapping.
    fn build_geom_idx_vec(&mut self, batch_pos_vec: &Vec<(i32, i32)>) -> Vec<usize> {
        let mut num_geometries = 0;
        let mut batch_idx_offset = Vec::with_capacity(self.indexed_batches.len() + 1);
        batch_idx_offset.push(0);
        for batch in &self.indexed_batches {
            num_geometries += batch.batch.num_rows();
            batch_idx_offset.push(num_geometries);
        }

        let mut geom_idx_vec = Vec::with_capacity(batch_pos_vec.len());
        self.reservation.grow(geom_idx_vec.allocated_size());
        for (batch_idx, row_idx) in batch_pos_vec {
            // Convert (batch_idx, row_idx) to a linear, sequential index
            let batch_offset = batch_idx_offset[*batch_idx as usize];
            let prepared_idx = batch_offset + *row_idx as usize;
            geom_idx_vec.push(prepared_idx);
        }

        geom_idx_vec
    }

    /// Finish building and return the completed SpatialIndex.
    pub fn finish(mut self) -> Result<SpatialIndex> {
        if self.indexed_batches.is_empty() {
            return Ok(SpatialIndex::empty(
                self.spatial_predicate,
                self.schema,
                self.options,
                AtomicUsize::new(self.probe_threads_count),
                self.reservation,
                self.memory_pool.clone(),
            ));
        }

        let evaluator = create_operand_evaluator(&self.spatial_predicate, self.options.clone());
        let num_geoms = self
            .indexed_batches
            .iter()
            .map(|batch| batch.batch.num_rows())
            .sum::<usize>();

        let (rtree, batch_pos_vec) = self.build_rtree()?;
        let geom_idx_vec = self.build_geom_idx_vec(&batch_pos_vec);
        let visited_left_side = self.build_visited_bitmaps()?;

        let refiner = create_refiner(
            self.options.spatial_library,
            &self.spatial_predicate,
            self.options.clone(),
            num_geoms,
            self.stats,
        );
        let consumer = MemoryConsumer::new("SpatialJoinRefiner");
        let refiner_reservation = consumer.register(&self.memory_pool);
        let refiner_reservation =
            ConcurrentReservation::try_new(REFINER_RESERVATION_PREALLOC_SIZE, refiner_reservation)
                .unwrap();

        let cache_size = batch_pos_vec.len();
        let knn_components =
            KnnComponents::new(cache_size, &self.indexed_batches, self.memory_pool.clone())?;

        Ok(SpatialIndex {
            schema: self.schema,
            evaluator,
            refiner,
            refiner_reservation,
            rtree,
            data_id_to_batch_pos: batch_pos_vec,
            indexed_batches: self.indexed_batches,
            geom_idx_vec,
            visited_left_side,
            probe_threads_counter: AtomicUsize::new(self.probe_threads_count),
            knn_components,
            reservation: self.reservation,
        })
    }

    pub async fn add_partitions(&mut self, partitions: Vec<BuildPartition>) -> Result<()> {
        for partition in partitions {
            self.add_partition(partition).await?;
        }
        Ok(())
    }

    pub async fn add_partition(&mut self, mut partition: BuildPartition) -> Result<()> {
        let mut stream = partition.build_side_batch_stream;
        while let Some(batch) = stream.next().await {
            let indexed_batch = batch?;
            self.add_batch(indexed_batch);
        }
        self.merge_stats(partition.geo_statistics);
        let mem_bytes = partition.reservation.free();
        self.reservation.try_grow(mem_bytes)?;
        Ok(())
    }
}
