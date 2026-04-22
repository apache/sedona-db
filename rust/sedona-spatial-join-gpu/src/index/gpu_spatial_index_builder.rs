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
use crate::options::GpuOptions;
use arrow::array::BooleanBufferBuilder;
use arrow::compute::concat_batches;
use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion_common::Result;
use datafusion_common::{DataFusionError, JoinType};
use futures::StreamExt;
use geo_types::{coord, Rect};
use parking_lot::Mutex;
use sedona_common::{sedona_internal_err, SpatialJoinOptions};
use sedona_expr::statistics::GeoStatistics;
use sedona_libgpuspatial::{GpuSpatialIndex, GpuSpatialOptions, GpuSpatialRefiner};
use sedona_spatial_join::evaluated_batch::evaluated_batch_stream::SendableEvaluatedBatchStream;
use sedona_spatial_join::evaluated_batch::EvaluatedBatch;
use sedona_spatial_join::index::spatial_index::SpatialIndexRef;
use sedona_spatial_join::index::spatial_index_builder::{
    SpatialIndexBuilder, SpatialJoinBuildMetrics,
};
use sedona_spatial_join::operand_evaluator::EvaluatedGeometryArray;
use sedona_spatial_join::utils::join_utils::need_produce_result_in_final;
use sedona_spatial_join::SpatialPredicate;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

pub(crate) struct GPUSpatialIndexBuilder {
    schema: SchemaRef,
    spatial_predicate: SpatialPredicate,
    gpu_options: GpuOptions,
    join_type: JoinType,
    probe_threads_count: usize,
    metrics: SpatialJoinBuildMetrics,
    /// Batches to be indexed
    indexed_batches: Vec<EvaluatedBatch>,
    /// Statistics for indexed geometries
    memory_used: usize,
}

impl GPUSpatialIndexBuilder {
    pub fn new(
        schema: SchemaRef,
        spatial_predicate: SpatialPredicate,
        gpu_options: GpuOptions,
        join_type: JoinType,
        probe_threads_count: usize,
        metrics: SpatialJoinBuildMetrics,
    ) -> Self {
        Self {
            schema,
            spatial_predicate,
            gpu_options,
            join_type,
            probe_threads_count,
            metrics,
            indexed_batches: vec![],
            memory_used: 0,
        }
    }

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

        self.record_memory_usage(total_buffer_size);

        Ok(Some(Mutex::new(bitmaps)))
    }

    fn record_memory_usage(&mut self, bytes: usize) {
        self.memory_used += bytes;
        self.metrics.build_mem_used.set_max(self.memory_used);
    }
    /// Add a geometry batch to be indexed.
    ///
    /// This method accumulates geometry batches that will be used to build the spatial index.
    /// Each batch contains processed geometry data along with memory usage information.
    fn add_batch(&mut self, indexed_batch: EvaluatedBatch) -> Result<()> {
        let in_mem_size = indexed_batch.in_mem_size()?;
        self.indexed_batches.push(indexed_batch);
        self.record_memory_usage(in_mem_size);
        Ok(())
    }

    pub(crate) fn estimate_extra_memory_usage(
        geo_stats: &GeoStatistics,
        _spatial_predicate: &SpatialPredicate,
        _options: &SpatialJoinOptions,
    ) -> usize {
        let num_geoms = geo_stats.total_geometries().unwrap_or(0) as usize;
        // This line estimates the size of temporary memory space for an array of bounding boxes,
        // which will be fed to the GPU library to build an on-device index.
        // The memory is allocated until the finishing of a spatial join.
        // Each geometry requires 4 f32 values to store the bounding rectangle (min_x, min_y, max_x, max_y)
        num_geoms * (4 * 4)
    }
}
fn concat_evaluated_batches(batches: &[EvaluatedBatch]) -> Result<EvaluatedBatch> {
    if batches.is_empty() {
        return sedona_internal_err!("Cannot concatenate empty list of EvaluatedBatches");
    }

    // Concatenate the underlying Arrow RecordBatches
    let schema = batches[0].schema();
    let record_batches: Vec<&RecordBatch> = batches.iter().map(|b| &b.batch).collect();
    let concatenated_batch = concat_batches(&schema, record_batches)?;
    // Concatenate Geometry Arrays using the interleave method
    let geom_arrays: Vec<&EvaluatedGeometryArray> = batches.iter().map(|b| &b.geom_array).collect();
    let concatenated_geom_array = EvaluatedGeometryArray::concat(&geom_arrays)?;

    Ok(EvaluatedBatch {
        batch: concatenated_batch,
        geom_array: concatenated_geom_array,
    })
}
#[async_trait]
impl SpatialIndexBuilder for GPUSpatialIndexBuilder {
    /// Finish building and return the completed SpatialIndex.
    fn finish(&mut self) -> Result<SpatialIndexRef> {
        if self.indexed_batches.is_empty() {
            let visited_build_side = self.build_visited_bitmaps()?;
            return Ok(Arc::new(GPUSpatialIndex::empty(
                self.spatial_predicate.clone(),
                self.schema.clone(),
                self.gpu_options.clone(),
                visited_build_side,
                AtomicUsize::new(self.probe_threads_count),
            )?));
        }
        let build_timer = self.metrics.build_time.timer();
        let gpu_options = GpuSpatialOptions {
            cuda_use_memory_pool: self.gpu_options.use_memory_pool,
            cuda_memory_pool_init_percent: self.gpu_options.memory_pool_init_percentage as i32,
            concurrency: self.probe_threads_count as u32,
            device_id: self.gpu_options.device_id as i32,
            compress_bvh: self.gpu_options.compress_bvh,
            pipeline_batches: self.gpu_options.pipeline_batches as u32,
        };

        let mut index = GpuSpatialIndex::try_new(&gpu_options).map_err(|e| {
            DataFusionError::Execution(format!("Failed to initialize GPU Index: {e:?}"))
        })?;
        let mut refiner = GpuSpatialRefiner::try_new(&gpu_options).map_err(|e| {
            DataFusionError::Execution(format!("Failed to initialize GPU Refiner: {e:?}"))
        })?;

        // Concat indexed batches into a single batch to reduce build time
        let total_rows: usize = self
            .indexed_batches
            .iter()
            .map(|batch| batch.batch.num_rows())
            .sum();

        let sedona_type = self.indexed_batches[0].geom_array.sedona_type().clone();

        if self.gpu_options.concat_build {
            let concat_batch = concat_evaluated_batches(&self.indexed_batches)?;
            self.indexed_batches.clear();
            self.indexed_batches.push(concat_batch);
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

        refiner
            .init_build_schema(sedona_type.storage_type())
            .map_err(|e| {
                DataFusionError::Execution(format!("Failed to init schema for refiner {e:?}"))
            })?;

        let mut native_rects = Vec::with_capacity(total_rows);

        for (batch_idx, batch) in self.indexed_batches.iter().enumerate() {
            let rects = batch.geom_array.rects();

            for (idx, rect_opt) in rects.iter().enumerate() {
                if let Some(rect) = rect_opt {
                    native_rects.push(*rect);
                } else {
                    native_rects.push(empty_rect);
                }
                data_id_to_batch_pos.push((batch_idx as i32, idx as i32));
            }
            refiner
                .push_build(batch.geom_array.geometry_array())
                .map_err(|e| {
                    DataFusionError::Execution(format!(
                        "Failed to add geometries to GPU refiner {e:?}"
                    ))
                })?;
        }

        refiner.finish_building().map_err(|e| {
            DataFusionError::Execution(format!("Failed to build spatial refiner on GPU {e:?}"))
        })?;

        // Add rectangles from build side to the spatial index
        index.push_build(&native_rects).map_err(|e| {
            DataFusionError::Execution(format!(
                "Failed to push rectangles to GPU spatial index {e:?}"
            ))
        })?;

        index.finish_building().map_err(|e| {
            DataFusionError::Execution(format!("Failed to build spatial index on GPU {e:?}"))
        })?;
        build_timer.done();
        let visited_build_side = self.build_visited_bitmaps()?;
        // Build index for rectangle queries
        Ok(Arc::new(GPUSpatialIndex {
            schema: self.schema.clone(),
            index: Arc::new(index),
            refiner: Arc::new(refiner),
            spatial_predicate: self.spatial_predicate.clone(),
            indexed_batches: self
                .indexed_batches
                .drain(0..self.indexed_batches.len())
                .collect(),
            data_id_to_batch_pos,
            visited_build_side,
            probe_threads_counter: AtomicUsize::new(self.probe_threads_count),
        }))
    }

    async fn add_stream(
        &mut self,
        mut stream: SendableEvaluatedBatchStream,
        _geo_statistics: GeoStatistics,
    ) -> Result<()> {
        while let Some(batch) = stream.next().await {
            let indexed_batch = batch?;
            self.add_batch(indexed_batch)?;
        }
        Ok(())
    }
}
