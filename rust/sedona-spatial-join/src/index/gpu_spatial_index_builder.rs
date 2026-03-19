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

use crate::evaluated_batch::evaluated_batch_stream::SendableEvaluatedBatchStream;
use crate::index::gpu_spatial_index::GPUSpatialIndex;
use crate::index::spatial_index::SpatialIndexRef;
use crate::index::spatial_index_builder::{SpatialIndexBuilder, SpatialJoinBuildMetrics};
use crate::operand_evaluator::EvaluatedGeometryArray;
use crate::spatial_predicate::SpatialRelationType;
use crate::utils::join_utils::need_produce_result_in_final;
use crate::{
    evaluated_batch::EvaluatedBatch, operand_evaluator::create_operand_evaluator,
    spatial_predicate::SpatialPredicate,
};
use arrow::array::BooleanBufferBuilder;
use arrow::compute::concat;
use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion_common::Result;
use datafusion_common::{DataFusionError, JoinType};
use futures::StreamExt;
use geo_types::{coord, Rect};
use parking_lot::Mutex;
use sedona_common::SpatialJoinOptions;
use sedona_expr::statistics::GeoStatistics;
use sedona_libgpuspatial::{GpuSpatialIndex, GpuSpatialOptions, GpuSpatialRefiner};
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
    /// Statistics for indexed geometries
    memory_used: usize,
}

impl GPUSpatialIndexBuilder {
    pub(crate) fn is_using_gpu(
        spatial_predicate: &SpatialPredicate,
        join_opts: &SpatialJoinOptions,
    ) -> Result<bool> {
        if join_opts.gpu.enable {
            if Self::is_spatial_predicate_supported_on_gpu(spatial_predicate) {
                return Ok(true);
            } else if join_opts.gpu.fallback_to_cpu {
                log::warn!("Falling back to CPU spatial join as the spatial predicate is not supported on GPU");
                return Ok(false);
            } else {
                return Err(DataFusionError::Execution("GPU spatial join is enabled, but the spatial predicate is not supported on GPU".into()));
            }
        }
        Ok(false)
    }

    fn is_spatial_predicate_supported_on_gpu(spatial_predicate: &SpatialPredicate) -> bool {
        match spatial_predicate {
            SpatialPredicate::Relation(rel) => match rel.relation_type {
                SpatialRelationType::Intersects => true,
                SpatialRelationType::Contains => true,
                SpatialRelationType::Within => true,
                SpatialRelationType::Covers => true,
                SpatialRelationType::CoveredBy => true,
                SpatialRelationType::Touches => true,
                SpatialRelationType::Crosses => false,
                SpatialRelationType::Overlaps => false,
                SpatialRelationType::Equals => true,
            },
            SpatialPredicate::Distance(_) => false,
            SpatialPredicate::KNearestNeighbors(_) => false,
        }
    }
    pub fn new(
        schema: SchemaRef,
        spatial_predicate: SpatialPredicate,
        options: SpatialJoinOptions,
        join_type: JoinType,
        probe_threads_count: usize,
        metrics: SpatialJoinBuildMetrics,
    ) -> Self {
        Self {
            schema,
            spatial_predicate,
            options,
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
}

#[async_trait]
impl SpatialIndexBuilder for GPUSpatialIndexBuilder {
    fn estimate_extra_memory_usage(
        &self,
        geo_stats: &GeoStatistics,
        _spatial_predicate: &SpatialPredicate,
        _options: &SpatialJoinOptions,
    ) -> usize {
        let num_geoms = geo_stats.total_geometries().unwrap_or(0) as usize;
        // Each geometry requires 4 f32 values to store the bounding rectangle (min_x, min_y, max_x, max_y)
        num_geoms * (4 * 4)
    }
    /// Finish building and return the completed SpatialIndex.
    fn finish(mut self) -> Result<SpatialIndexRef> {
        if self.indexed_batches.is_empty() {
            return Ok(Arc::new(GPUSpatialIndex::empty(
                self.spatial_predicate,
                self.schema,
                self.options,
                AtomicUsize::new(self.probe_threads_count),
            )?));
        }
        let build_timer = self.metrics.build_time.timer();
        let gpu_options = GpuSpatialOptions {
            cuda_use_memory_pool: self.options.gpu.use_memory_pool,
            cuda_memory_pool_init_percent: self.options.gpu.memory_pool_init_percentage as i32,
            concurrency: self.probe_threads_count as u32,
            device_id: self.options.gpu.device_id as i32,
            compress_bvh: self.options.gpu.compress_bvh,
            pipeline_batches: self.options.gpu.pipeline_batches as u32,
        };

        let mut index = GpuSpatialIndex::try_new(&gpu_options).map_err(|e| {
            DataFusionError::Execution(format!("Failed to initialize GPU context {e:?}"))
        })?;
        let mut refiner = GpuSpatialRefiner::try_new(&gpu_options).map_err(|e| {
            DataFusionError::Execution(format!("Failed to initialize GPU context {e:?}"))
        })?;

        // Concat indexed batches into a single batch to reduce build time
        let total_rows: usize = self
            .indexed_batches
            .iter()
            .map(|batch| batch.batch.num_rows())
            .sum();

        let sedona_type = self.indexed_batches[0].geom_array.sedona_type.clone();

        if self.options.gpu.concat_build {
            let all_record_batches: Vec<&RecordBatch> = self
                .indexed_batches
                .iter()
                .map(|batch| &batch.batch)
                .collect();
            let schema = all_record_batches[0].schema();
            let batch = arrow::compute::concat_batches(&schema, all_record_batches)?;
            let references: Vec<&dyn arrow::array::Array> = self
                .indexed_batches
                .iter()
                .map(|batch| batch.geom_array.geometry_array.as_ref())
                .collect();
            let concat_array = concat(&references)?;

            // rects have been computed by EvaluatedGeometryArray::try_new, we just need to concatenate them
            let rects = self
                .indexed_batches
                .iter()
                .flat_map(|batch| batch.geom_array.rects.iter().cloned())
                .collect();
            let eval_batch = EvaluatedBatch {
                batch,
                geom_array: EvaluatedGeometryArray {
                    sedona_type: sedona_type.clone(),
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

        refiner
            .init_build_schema(sedona_type.storage_type())
            .map_err(|e| {
                DataFusionError::Execution(format!("Failed to init schema for refiner {e:?}"))
            })?;

        let mut native_rects = Vec::with_capacity(total_rows);

        for (batch_idx, batch) in self.indexed_batches.iter().enumerate() {
            let rects = batch.rects();

            for (idx, rect_opt) in rects.iter().enumerate() {
                if let Some(rect) = rect_opt {
                    native_rects.push(*rect);
                } else {
                    native_rects.push(empty_rect);
                }
                data_id_to_batch_pos.push((batch_idx as i32, idx as i32));
            }
            refiner
                .push_build(&batch.geom_array.geometry_array)
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
        let evaluator = create_operand_evaluator(&self.spatial_predicate, self.options.clone());
        // Build index for rectangle queries
        Ok(Arc::new(GPUSpatialIndex::new(
            self.spatial_predicate,
            self.schema,
            self.options,
            evaluator,
            Arc::new(index),
            Arc::new(refiner),
            self.indexed_batches,
            data_id_to_batch_pos,
            visited_build_side,
            AtomicUsize::new(self.probe_threads_count),
        )?))
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
