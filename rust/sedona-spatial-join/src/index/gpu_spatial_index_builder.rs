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

use crate::index::gpu_spatial_index::GPUSpatialIndex;
use crate::index::spatial_index::{SpatialIndexRef, SpatialJoinBuildMetrics};
use crate::operand_evaluator::EvaluatedGeometryArray;
use crate::utils::join_utils::need_produce_result_in_final;
use crate::{
    evaluated_batch::EvaluatedBatch, index::BuildPartition,
    operand_evaluator::create_operand_evaluator, spatial_predicate::SpatialPredicate,
};
use arrow::array::BooleanBufferBuilder;
use arrow::compute::concat;
use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion_common::Result;
use datafusion_common::{DataFusionError, JoinType};
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryPool, MemoryReservation};
use futures::StreamExt;
use geo_types::{coord, Rect};
use parking_lot::Mutex;
use sedona_common::SpatialJoinOptions;
use sedona_libgpuspatial::GpuSpatial;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

pub struct GPUSpatialIndexBuilder {
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
}

impl GPUSpatialIndexBuilder {
    pub fn new(
        schema: SchemaRef,
        spatial_predicate: SpatialPredicate,
        options: SpatialJoinOptions,
        join_type: JoinType,
        probe_threads_count: usize,
        memory_pool: Arc<dyn MemoryPool>,
        metrics: SpatialJoinBuildMetrics,
    ) -> Self {
        let consumer = MemoryConsumer::new("SpatialJoinIndex");
        let reservation = consumer.register(&memory_pool);

        Self {
            schema,
            spatial_predicate,
            options,
            join_type,
            probe_threads_count,
            metrics,
            indexed_batches: vec![],
            reservation,
        }
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

    pub fn finish(mut self) -> Result<SpatialIndexRef> {
        if self.indexed_batches.is_empty() {
            return Ok(Arc::new(GPUSpatialIndex::empty(
                self.spatial_predicate,
                self.schema,
                self.options,
                AtomicUsize::new(self.probe_threads_count),
                self.reservation,
            )?));
        }

        let mut gs = GpuSpatial::new()
            .and_then(|mut gs| {
                gs.init(
                    self.probe_threads_count as u32,
                    self.options.gpu.device_id as i32,
                )?;
                Ok(gs)
            })
            .map_err(|e| {
                DataFusionError::Execution(format!("Failed to initialize GPU context {e:?}"))
            })?;

        let build_timer = self.metrics.build_time.timer();

        // Concat indexed batches into a single batch to reduce build time
        if self.options.gpu.concat_build {
            let all_record_batches: Vec<&RecordBatch> = self
                .indexed_batches
                .iter()
                .map(|batch| &batch.batch)
                .collect();
            let schema = all_record_batches[0].schema();
            let batch =
                arrow::compute::concat_batches(&schema, all_record_batches).map_err(|e| {
                    DataFusionError::Execution(format!("Failed to concatenate left batches: {}", e))
                })?;

            let references: Vec<&dyn arrow::array::Array> = self
                .indexed_batches
                .iter()
                .map(|batch| batch.geom_array.geometry_array.as_ref())
                .collect();

            let concat_array = concat(&references)?;
            let rects = self
                .indexed_batches
                .iter()
                .flat_map(|batch| batch.geom_array.rects.iter().cloned())
                .collect();
            let eval_batch = EvaluatedBatch {
                batch,
                geom_array: EvaluatedGeometryArray {
                    geometry_array: Arc::new(concat_array),
                    rects,
                    distance: None,
                    wkbs: vec![],
                },
            };
            self.indexed_batches.clear();
            self.indexed_batches.push(eval_batch);
        }

        let mut data_id_to_batch_pos: Vec<(i32, i32)> = Vec::with_capacity(
            self.indexed_batches
                .iter()
                .map(|x| x.batch.num_rows())
                .sum(),
        );
        let empty_rect = Rect::new(
            coord!(x: f32::NAN, y: f32::NAN),
            coord!(x: f32::NAN, y: f32::NAN),
        );
        for (batch_idx, batch) in self.indexed_batches.iter().enumerate() {
            let rects = batch.rects();
            let mut native_rects = Vec::new();
            for (idx, rect_opt) in rects.iter().enumerate() {
                if let Some(rect) = rect_opt {
                    native_rects.push(*rect);
                } else {
                    native_rects.push(empty_rect);
                }
                data_id_to_batch_pos.push((batch_idx as i32, idx as i32));
            }
            // Add rectangles from build side to the spatial index
            gs.index_push_build(&native_rects).map_err(|e| {
                DataFusionError::Execution(format!(
                    "Failed to push rectangles to GPU spatial index {e:?}"
                ))
            })?;
            gs.refiner_push_build(&batch.geom_array.geometry_array)
                .map_err(|e| {
                    DataFusionError::Execution(format!(
                        "Failed to add geometries to GPU refiner {e:?}"
                    ))
                })?;
        }

        gs.index_finish_building().map_err(|e| {
            DataFusionError::Execution(format!("Failed to build spatial index on GPU {e:?}"))
        })?;
        gs.refiner_finish_building().map_err(|e| {
            DataFusionError::Execution(format!("Failed to build spatial refiner on GPU {e:?}"))
        })?;
        build_timer.done();
        let visited_left_side = self.build_visited_bitmaps()?;
        let evaluator = create_operand_evaluator(&self.spatial_predicate, self.options.clone());
        // Build index for rectangle queries
        Ok(Arc::new(GPUSpatialIndex::new(
            self.spatial_predicate,
            self.schema,
            self.options,
            evaluator,
            Arc::new(gs),
            self.indexed_batches,
            data_id_to_batch_pos,
            visited_left_side,
            AtomicUsize::new(self.probe_threads_count),
            self.reservation,
        )?))
    }

    pub fn add_batch(&mut self, indexed_batch: EvaluatedBatch) -> Result<()> {
        let in_mem_size = indexed_batch.in_mem_size()?;
        self.indexed_batches.push(indexed_batch);
        self.reservation.grow(in_mem_size);
        self.metrics.build_mem_used.add(in_mem_size);
        Ok(())
    }
    pub async fn add_partition(&mut self, mut partition: BuildPartition) -> Result<()> {
        let mut stream = partition.build_side_batch_stream;
        while let Some(batch) = stream.next().await {
            let indexed_batch = batch?;
            self.add_batch(indexed_batch)?;
        }
        let mem_bytes = partition.reservation.free();
        self.reservation.try_grow(mem_bytes)?;
        Ok(())
    }

    pub async fn add_partitions(&mut self, partitions: Vec<BuildPartition>) -> Result<()> {
        for partition in partitions {
            self.add_partition(partition).await?;
        }
        Ok(())
    }
}
