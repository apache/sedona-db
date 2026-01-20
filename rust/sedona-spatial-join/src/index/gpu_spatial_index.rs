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

use crate::evaluated_batch::EvaluatedBatch;
use crate::index::spatial_index::SpatialIndex;
use crate::index::QueryResultMetrics;
use crate::operand_evaluator::OperandEvaluator;
use crate::spatial_predicate::SpatialRelationType;
use crate::{operand_evaluator::create_operand_evaluator, spatial_predicate::SpatialPredicate};
use arrow::array::BooleanBufferBuilder;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::memory_pool::MemoryReservation;
use geo_types::{coord, Rect};
use parking_lot::Mutex;
use sedona_common::{ExecutionMode, SpatialJoinOptions};
use sedona_expr::statistics::GeoStatistics;
use sedona_libgpuspatial::{GpuSpatial, GpuSpatialRelationPredicate};
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use wkb::reader::Wkb;

pub struct GPUSpatialIndex {
    pub(crate) schema: SchemaRef,
    pub(crate) _options: SpatialJoinOptions,
    /// The spatial predicate evaluator for the spatial predicate.
    #[allow(dead_code)] // reserved for GPU-based distance evaluation
    pub(crate) evaluator: Arc<dyn OperandEvaluator>,
    /// GPU spatial object for performing GPU-accelerated spatial queries
    pub(crate) gpu_spatial: Arc<GpuSpatial>,
    pub(crate) spatial_predicate: SpatialPredicate,
    /// Indexed batches containing evaluated geometry arrays. It contains the original record
    /// batches and geometry arrays obtained by evaluating the geometry expression on the build side.
    pub(crate) indexed_batches: Vec<EvaluatedBatch>,
    /// An array for translating data index to geometry batch index and row index
    pub(crate) data_id_to_batch_pos: Vec<(i32, i32)>,
    /// Shared bitmap builders for visited left indices, one per batch
    pub(crate) visited_left_side: Option<Mutex<Vec<BooleanBufferBuilder>>>,
    /// Counter of running probe-threads, potentially able to update `bitmap`.
    /// Each time a probe thread finished probing the index, it will decrement the counter.
    /// The last finished probe thread will produce the extra output batches for unmatched
    /// build side when running left-outer joins. See also [`report_probe_completed`].
    pub(crate) probe_threads_counter: AtomicUsize,
    /// Memory reservation for tracking the memory usage of the spatial index
    /// Cleared on `SpatialIndex` drop
    #[expect(dead_code)]
    pub(crate) reservation: MemoryReservation,
}
impl GPUSpatialIndex {
    pub fn empty(
        spatial_predicate: SpatialPredicate,
        schema: SchemaRef,
        options: SpatialJoinOptions,
        probe_threads_counter: AtomicUsize,
        reservation: MemoryReservation,
    ) -> Result<Self> {
        let evaluator = create_operand_evaluator(&spatial_predicate, options.clone());

        Ok(Self {
            schema,
            _options: options,
            evaluator,
            spatial_predicate,
            gpu_spatial: Arc::new(
                GpuSpatial::new().map_err(|e| DataFusionError::Execution(e.to_string()))?,
            ),
            indexed_batches: vec![],
            data_id_to_batch_pos: vec![],
            visited_left_side: None,
            probe_threads_counter,
            reservation,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        spatial_predicate: SpatialPredicate,
        schema: SchemaRef,
        options: SpatialJoinOptions,
        evaluator: Arc<dyn OperandEvaluator>,
        gpu_spatial: Arc<GpuSpatial>,
        indexed_batches: Vec<EvaluatedBatch>,
        data_id_to_batch_pos: Vec<(i32, i32)>,
        visited_left_side: Option<Mutex<Vec<BooleanBufferBuilder>>>,
        probe_threads_counter: AtomicUsize,
        reservation: MemoryReservation,
    ) -> Result<Self> {
        Ok(Self {
            schema,
            _options: options,
            evaluator,
            spatial_predicate,
            gpu_spatial,
            indexed_batches,
            data_id_to_batch_pos,
            visited_left_side,
            probe_threads_counter,
            reservation,
        })
    }

    fn refine_loaded(
        &self,
        probe_geoms: &ArrayRef,
        predicate: &SpatialPredicate,
        build_indices: &mut Vec<u32>,
        probe_indices: &mut Vec<u32>,
    ) -> Result<()> {
        match predicate {
            SpatialPredicate::Relation(rel_p) => {
                self.gpu_spatial
                    .refine_loaded(
                        probe_geoms,
                        Self::convert_relation_type(&rel_p.relation_type)?,
                        build_indices,
                        probe_indices,
                    )
                    .map_err(|e| {
                        DataFusionError::Execution(format!(
                            "GPU spatial refinement failed: {:?}",
                            e
                        ))
                    })?;
                Ok(())
            }
            _ => Err(DataFusionError::NotImplemented(
                "Only Relation predicate is supported for GPU spatial query".to_string(),
            )),
        }
    }
    // Translate Sedona SpatialRelationType to GpuSpatialRelationPredicate
    fn convert_relation_type(t: &SpatialRelationType) -> Result<GpuSpatialRelationPredicate> {
        match t {
            SpatialRelationType::Equals => Ok(GpuSpatialRelationPredicate::Equals),
            SpatialRelationType::Touches => Ok(GpuSpatialRelationPredicate::Touches),
            SpatialRelationType::Contains => Ok(GpuSpatialRelationPredicate::Contains),
            SpatialRelationType::Covers => Ok(GpuSpatialRelationPredicate::Covers),
            SpatialRelationType::Intersects => Ok(GpuSpatialRelationPredicate::Intersects),
            SpatialRelationType::Within => Ok(GpuSpatialRelationPredicate::Within),
            SpatialRelationType::CoveredBy => Ok(GpuSpatialRelationPredicate::CoveredBy),
            _ => {
                // This should not happen as we check for supported predicates earlier
                Err(DataFusionError::Execution(format!(
                    "Unsupported spatial relation type for GPU: {:?}",
                    t
                )))
            }
        }
    }
}

#[async_trait]
impl SpatialIndex for GPUSpatialIndex {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
    fn get_num_indexed_batches(&self) -> usize {
        self.indexed_batches.len()
    }
    fn get_indexed_batch(&self, batch_idx: usize) -> &RecordBatch {
        &self.indexed_batches[batch_idx].batch
    }
    async fn query_batch(
        &self,
        evaluated_batch: &Arc<EvaluatedBatch>,
        range: Range<usize>,
        _max_result_size: usize,
        build_batch_positions: &mut Vec<(i32, i32)>,
        probe_indices: &mut Vec<u32>,
    ) -> Result<(QueryResultMetrics, usize)> {
        if range.is_empty() {
            return Ok((
                QueryResultMetrics {
                    count: 0,
                    candidate_count: 0,
                },
                range.start,
            ));
        }
        let gs = &self.gpu_spatial.as_ref();

        let empty_rect = Rect::new(
            coord!(x: f32::NAN, y: f32::NAN),
            coord!(x: f32::NAN, y: f32::NAN),
        );
        let rects: Vec<_> = range
            .clone()
            .map(|row_idx| evaluated_batch.geom_array.rects[row_idx].unwrap_or(empty_rect))
            .collect();

        let (mut gpu_build_indices, mut gpu_probe_indices) =
            gs.probe(rects.as_ref()).map_err(|e| {
                DataFusionError::Execution(format!("GPU spatial query failed: {:?}", e))
            })?;

        assert_eq!(gpu_build_indices.len(), gpu_probe_indices.len());

        let candidate_count = gpu_build_indices.len();

        self.refine_loaded(
            &evaluated_batch.geom_array.geometry_array,
            &self.spatial_predicate,
            &mut gpu_build_indices,
            &mut gpu_probe_indices,
        )?;

        assert_eq!(gpu_build_indices.len(), gpu_probe_indices.len());

        let total_count = gpu_build_indices.len();

        for (build_idx, probe_idx) in gpu_build_indices.iter().zip(gpu_probe_indices.iter()) {
            let data_id = *build_idx as usize;
            let (batch_idx, row_idx) = self.data_id_to_batch_pos[data_id];
            build_batch_positions.push((batch_idx, row_idx));
            probe_indices.push(range.start as u32 + probe_idx);
        }
        Ok((
            QueryResultMetrics {
                count: total_count,
                candidate_count,
            },
            range.end,
        ))
    }
    fn need_more_probe_stats(&self) -> bool {
        false
    }

    fn merge_probe_stats(&self, stats: GeoStatistics) {
        let _ = stats;
    }

    fn visited_left_side(&self) -> Option<&Mutex<Vec<BooleanBufferBuilder>>> {
        self.visited_left_side.as_ref()
    }

    fn report_probe_completed(&self) -> bool {
        self.probe_threads_counter.fetch_sub(1, Ordering::Relaxed) == 1
    }

    fn get_refiner_mem_usage(&self) -> usize {
        0
    }

    fn get_actual_execution_mode(&self) -> ExecutionMode {
        ExecutionMode::PrepareBuild // GPU-based spatial index is always on PrepareBuild mode
    }
    #[allow(unused)]
    fn query(
        &self,
        probe_wkb: &Wkb,
        probe_rect: &Rect<f32>,
        distance: &Option<f64>,
        build_batch_positions: &mut Vec<(i32, i32)>,
    ) -> Result<QueryResultMetrics> {
        let _ = (probe_wkb, probe_rect, distance, build_batch_positions);
        Err(DataFusionError::NotImplemented(
            "Serial query is not implemented for GPU spatial index".to_string(),
        ))
    }

    fn query_knn(
        &self,
        probe_wkb: &Wkb,
        k: u32,
        use_spheroid: bool,
        include_tie_breakers: bool,
        build_batch_positions: &mut Vec<(i32, i32)>,
    ) -> Result<QueryResultMetrics> {
        let _ = (
            probe_wkb,
            k,
            use_spheroid,
            include_tie_breakers,
            build_batch_positions,
        );
        Err(DataFusionError::NotImplemented(
            "KNN query is not implemented for GPU spatial index".to_string(),
        ))
    }
}
