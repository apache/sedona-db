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

use crate::utils::join_utils::need_produce_result_in_final;
use crate::{
    evaluated_batch::EvaluatedBatch,
    index::{spatial_index::SpatialIndex, BuildPartition},
    operand_evaluator::create_operand_evaluator,
    spatial_predicate::SpatialPredicate,
};
use arrow::array::BooleanBufferBuilder;
use arrow::compute::concat;
use arrow_array::RecordBatch;
use datafusion_common::Result;
use datafusion_common::{DataFusionError, JoinType};
use datafusion_physical_plan::metrics;
use datafusion_physical_plan::metrics::{ExecutionPlanMetricsSet, MetricBuilder};
use futures::StreamExt;
use parking_lot::Mutex;
use sedona_common::SpatialJoinOptions;
use sedona_libgpuspatial::GpuSpatial;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

pub struct SpatialIndexBuilder {
    spatial_predicate: SpatialPredicate,
    options: SpatialJoinOptions,
    join_type: JoinType,
    probe_threads_count: usize,
    metrics: SpatialJoinBuildMetrics,
    build_batch: EvaluatedBatch,
}

#[derive(Clone, Debug, Default)]
pub struct SpatialJoinBuildMetrics {
    // Total time for concatenating build-side batches
    pub(crate) concat_time: metrics::Time,
    /// Total time for loading build-side geometries to GPU
    pub(crate) load_time: metrics::Time,
    /// Total time for collecting build-side of join
    pub(crate) build_time: metrics::Time,
}

impl SpatialJoinBuildMetrics {
    pub fn new(partition: usize, metrics: &ExecutionPlanMetricsSet) -> Self {
        Self {
            concat_time: MetricBuilder::new(metrics).subset_time("concat_time", partition),
            load_time: MetricBuilder::new(metrics).subset_time("load_time", partition),
            build_time: MetricBuilder::new(metrics).subset_time("build_time", partition),
        }
    }
}

impl SpatialIndexBuilder {
    pub fn new(
        spatial_predicate: SpatialPredicate,
        options: SpatialJoinOptions,
        join_type: JoinType,
        probe_threads_count: usize,
        metrics: SpatialJoinBuildMetrics,
    ) -> Self {
        Self {
            spatial_predicate,
            options,
            join_type,
            probe_threads_count,
            metrics,
            build_batch: EvaluatedBatch::default(),
        }
    }
    /// Build visited bitmaps for tracking left-side indices in outer joins.
    fn build_visited_bitmap(&mut self) -> Result<Option<Mutex<BooleanBufferBuilder>>> {
        if !need_produce_result_in_final(self.join_type) {
            return Ok(None);
        }

        let total_rows = self.build_batch.batch.num_rows();

        let mut bitmap = BooleanBufferBuilder::new(total_rows);
        bitmap.append_n(total_rows, false);

        Ok(Some(Mutex::new(bitmap)))
    }

    pub fn finish(mut self) -> Result<SpatialIndex> {
        if self.build_batch.batch.num_rows() == 0 {
            return SpatialIndex::new_empty(
                EvaluatedBatch::default(),
                self.spatial_predicate,
                self.options,
                AtomicUsize::new(self.probe_threads_count),
            );
        }

        let mut gs = GpuSpatial::new()
            .and_then(|mut gs| {
                gs.init(
                    self.probe_threads_count as u32,
                    self.options.gpu.device_id as i32,
                )?;
                gs.clear()?;
                Ok(gs)
            })
            .map_err(|e| {
                DataFusionError::Execution(format!("Failed to initialize GPU context {e:?}"))
            })?;

        let build_timer = self.metrics.build_time.timer();
        // Ensure the spatial index is clear before building
        gs.clear().map_err(|e| {
            DataFusionError::Execution(format!("Failed to clear GPU spatial index {e:?}"))
        })?;
        // Add rectangles from build side to the spatial index
        gs.push_build(&self.build_batch.geom_array.rects)
            .map_err(|e| {
                DataFusionError::Execution(format!(
                    "Failed to add geometries to GPU spatial index {e:?}"
                ))
            })?;
        gs.finish_building().map_err(|e| {
            DataFusionError::Execution(format!("Failed to build spatial index on GPU {e:?}"))
        })?;
        build_timer.done();

        let num_rows = self.build_batch.batch.num_rows();

        log::info!("Total build side rows: {}", num_rows);

        let geom_array = self.build_batch.geom_array.geometry_array.clone();

        let load_timer = self.metrics.load_time.timer();
        gs.load_build_array(&geom_array).map_err(|e| {
            DataFusionError::Execution(format!("GPU spatial query failed: {:?}", e))
        })?;
        load_timer.done();

        let visited_left_side = self.build_visited_bitmap()?;
        // Build index for rectangle queries
        Ok(SpatialIndex::new(
            create_operand_evaluator(&self.spatial_predicate, self.options.clone()),
            self.build_batch,
            visited_left_side,
            Arc::new(gs),
            AtomicUsize::new(self.probe_threads_count),
        ))
    }

    pub async fn add_partitions(&mut self, partitions: Vec<BuildPartition>) -> Result<()> {
        let mut indexed_batches: Vec<EvaluatedBatch> = Vec::new();
        for partition in partitions {
            let mut stream = partition.build_side_batch_stream;
            while let Some(batch) = stream.next().await {
                indexed_batches.push(batch?)
            }
        }

        let concat_timer = self.metrics.concat_time.timer();
        let all_record_batches: Vec<&RecordBatch> =
            indexed_batches.iter().map(|batch| &batch.batch).collect();

        if all_record_batches.is_empty() {
            return Err(DataFusionError::Internal(
                "Build side has no batches".into(),
            ));
        }

        // 2. Extract the schema from the first batch
        let schema = all_record_batches[0].schema();

        // 3. Pass the slice of references (&[&RecordBatch])
        self.build_batch.batch = arrow::compute::concat_batches(&schema, all_record_batches)
            .map_err(|e| {
                DataFusionError::Execution(format!("Failed to concatenate left batches: {}", e))
            })?;

        let references: Vec<&dyn arrow::array::Array> = indexed_batches
            .iter()
            .map(|batch| batch.geom_array.geometry_array.as_ref())
            .collect();

        let concat_array = concat(&references)?;

        self.build_batch.geom_array.geometry_array = concat_array;

        self.build_batch.geom_array.rects = indexed_batches
            .iter()
            .flat_map(|batch| batch.geom_array.rects.iter().cloned())
            .collect();
        concat_timer.done();
        Ok(())
    }
}
