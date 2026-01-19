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

use std::{
    ops::Range,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion_common::{DataFusionError, Result};
use datafusion_common_runtime::JoinSet;
use datafusion_execution::memory_pool::{MemoryPool, MemoryReservation};
use float_next_after::NextAfter;
use geo::BoundingRect;
use geo_index::rtree::{
    distance::{DistanceMetric, GeometryAccessor},
    util::f64_box_to_f32,
};
use geo_index::rtree::{sort::HilbertSort, RTree, RTreeBuilder, RTreeIndex};
use geo_index::IndexableNum;
use geo_types::Rect;
use parking_lot::Mutex;
use sedona_expr::statistics::GeoStatistics;
use sedona_geo::to_geo::item_to_geometry;
use wkb::reader::Wkb;

use crate::index::spatial_index::{SpatialIndex, SpatialIndexInternal};
use crate::{
    evaluated_batch::EvaluatedBatch,
    index::{
        knn_adapter::{KnnComponents, SedonaKnnAdapter},
        IndexQueryResult, QueryResultMetrics,
    },
    operand_evaluator::{create_operand_evaluator, distance_value_at, OperandEvaluator},
    refine::{create_refiner, IndexQueryResultRefiner},
    spatial_predicate::SpatialPredicate,
    utils::concurrent_reservation::ConcurrentReservation,
};
use arrow::array::BooleanBufferBuilder;
use async_trait::async_trait;
use sedona_common::{option::SpatialJoinOptions, sedona_internal_err, ExecutionMode};

struct CPUSpatialIndexInner {
    pub(crate) schema: SchemaRef,
    pub(crate) options: SpatialJoinOptions,

    /// The spatial predicate evaluator for the spatial predicate.
    pub(crate) evaluator: Arc<dyn OperandEvaluator>,

    /// The refiner for refining the index query results.
    pub(crate) refiner: Arc<dyn IndexQueryResultRefiner>,

    /// Memory reservation for tracking the memory usage of the refiner
    pub(crate) refiner_reservation: ConcurrentReservation,

    /// R-tree index for the geometry batches. It takes MBRs as query windows and returns
    /// data indexes. These data indexes should be translated using `data_id_to_batch_pos` to get
    /// the original geometry batch index and row index, or translated using `prepared_geom_idx_vec`
    /// to get the prepared geometries array index.
    pub(crate) rtree: RTree<f32>,

    /// Indexed batches containing evaluated geometry arrays. It contains the original record
    /// batches and geometry arrays obtained by evaluating the geometry expression on the build side.
    pub(crate) indexed_batches: Vec<EvaluatedBatch>,
    /// An array for translating rtree data index to geometry batch index and row index
    pub(crate) data_id_to_batch_pos: Vec<(i32, i32)>,

    /// An array for translating rtree data index to consecutive index. Each geometry may be indexed by
    /// multiple boxes, so there could be multiple data indexes for the same geometry. A mapping for
    /// squashing the index makes it easier for persisting per-geometry auxiliary data for evaluating
    /// the spatial predicate. This is extensively used by the spatial predicate evaluators for storing
    /// prepared geometries.
    pub(crate) geom_idx_vec: Vec<usize>,

    /// Shared bitmap builders for visited left indices, one per batch
    pub(crate) visited_left_side: Option<Mutex<Vec<BooleanBufferBuilder>>>,

    /// Counter of running probe-threads, potentially able to update `bitmap`.
    /// Each time a probe thread finished probing the index, it will decrement the counter.
    /// The last finished probe thread will produce the extra output batches for unmatched
    /// build side when running left-outer joins. See also [`report_probe_completed`].
    pub(crate) probe_threads_counter: AtomicUsize,

    /// Shared KNN components (distance metrics and geometry cache) for efficient KNN queries
    pub(crate) knn_components: Option<KnnComponents>,

    /// Memory reservation for tracking the memory usage of the spatial index
    /// Cleared on `SpatialIndex` drop
    #[expect(dead_code)]
    pub(crate) reservation: MemoryReservation,
}

#[derive(Clone)]
pub struct CPUSpatialIndex {
    inner: Arc<CPUSpatialIndexInner>,
}

impl CPUSpatialIndex {
    pub fn empty(
        spatial_predicate: SpatialPredicate,
        schema: SchemaRef,
        options: SpatialJoinOptions,
        probe_threads_counter: AtomicUsize,
        mut reservation: MemoryReservation,
        memory_pool: Arc<dyn MemoryPool>,
    ) -> Self {
        let evaluator = create_operand_evaluator(&spatial_predicate, options.clone());
        let refiner = create_refiner(
            options.spatial_library,
            &spatial_predicate,
            options.clone(),
            0,
            GeoStatistics::empty(),
        );
        let refiner_reservation = reservation.split(0);
        let refiner_reservation = ConcurrentReservation::try_new(0, refiner_reservation).unwrap();
        let rtree = RTreeBuilder::<f32>::new(0).finish::<HilbertSort>();
        let knn_components = matches!(spatial_predicate, SpatialPredicate::KNearestNeighbors(_))
            .then(|| KnnComponents::new(0, &[], memory_pool.clone()).unwrap());
        Self {
            inner: Arc::new(CPUSpatialIndexInner {
                schema,
                options,
                evaluator,
                refiner,
                refiner_reservation,
                rtree,
                data_id_to_batch_pos: Vec::new(),
                indexed_batches: Vec::new(),
                geom_idx_vec: Vec::new(),
                visited_left_side: None,
                probe_threads_counter,
                knn_components,
                reservation,
            }),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        schema: SchemaRef,
        options: SpatialJoinOptions,
        evaluator: Arc<dyn OperandEvaluator>,
        refiner: Arc<dyn IndexQueryResultRefiner>,
        refiner_reservation: ConcurrentReservation,
        rtree: RTree<f32>,
        indexed_batches: Vec<EvaluatedBatch>,
        data_id_to_batch_pos: Vec<(i32, i32)>,
        geom_idx_vec: Vec<usize>,
        visited_left_side: Option<Mutex<Vec<BooleanBufferBuilder>>>,
        probe_threads_counter: AtomicUsize,
        knn_components: Option<KnnComponents>,
        reservation: MemoryReservation,
    ) -> Self {
        Self {
            inner: Arc::new(CPUSpatialIndexInner {
                schema,
                options,
                evaluator,
                refiner,
                refiner_reservation,
                rtree,
                data_id_to_batch_pos,
                indexed_batches,
                geom_idx_vec,
                visited_left_side,
                probe_threads_counter,
                knn_components,
                reservation,
            }),
        }
    }
    /// Create a KNN geometry accessor for accessing geometries with caching
    fn create_knn_accessor(&self) -> Result<SedonaKnnAdapter<'_>> {
        let Some(knn_components) = self.inner.knn_components.as_ref() else {
            return sedona_internal_err!("knn_components is not initialized when running KNN join");
        };
        Ok(SedonaKnnAdapter::new(
            &self.inner.indexed_batches,
            &self.inner.data_id_to_batch_pos,
            knn_components,
        ))
    }
    async fn refine_concurrently(
        &self,
        evaluated_batch: &Arc<EvaluatedBatch>,
        row_idx: usize,
        candidates: &[u32],
        distance: Option<f64>,
        refine_chunk_size: usize,
    ) -> Result<(QueryResultMetrics, Vec<(i32, i32)>)> {
        let mut join_set = JoinSet::new();
        for (i, chunk) in candidates.chunks(refine_chunk_size).enumerate() {
            let cloned_evaluated_batch = Arc::clone(evaluated_batch);
            let chunk = chunk.to_vec();
            let index_owned = self.clone();
            join_set.spawn(async move {
                let Some(probe_wkb) = cloned_evaluated_batch.wkb(row_idx) else {
                    return (
                        i,
                        sedona_internal_err!(
                            "Failed to get WKB for row {} in evaluated batch",
                            row_idx
                        ),
                    );
                };
                let mut local_positions: Vec<(i32, i32)> = Vec::with_capacity(chunk.len());
                let res = index_owned.refine(probe_wkb, &chunk, &distance, &mut local_positions);
                (i, res.map(|r| (r, local_positions)))
            });
        }

        // Collect the results in order
        let mut refine_results = Vec::with_capacity(join_set.len());
        refine_results.resize_with(join_set.len(), || None);
        while let Some(res) = join_set.join_next().await {
            let (chunk_idx, refine_res) =
                res.map_err(|e| DataFusionError::External(Box::new(e)))?;
            let (metrics, positions) = refine_res?;
            refine_results[chunk_idx] = Some((metrics, positions));
        }

        let mut total_metrics = QueryResultMetrics {
            count: 0,
            candidate_count: 0,
        };
        let mut all_positions = Vec::with_capacity(candidates.len());
        for res in refine_results {
            let (metrics, positions) = res.expect("All chunks should be processed");
            total_metrics.count += metrics.count;
            total_metrics.candidate_count += metrics.candidate_count;
            all_positions.extend(positions);
        }

        Ok((total_metrics, all_positions))
    }

    fn refine(
        &self,
        probe_wkb: &Wkb,
        candidates: &[u32],
        distance: &Option<f64>,
        build_batch_positions: &mut Vec<(i32, i32)>,
    ) -> Result<QueryResultMetrics> {
        let candidate_count = candidates.len();

        let mut index_query_results = Vec::with_capacity(candidate_count);
        for data_idx in candidates {
            let pos = self.inner.data_id_to_batch_pos[*data_idx as usize];
            let (batch_idx, row_idx) = pos;
            let indexed_batch = &self.inner.indexed_batches[batch_idx as usize];
            let build_wkb = indexed_batch.wkb(row_idx as usize);
            let Some(build_wkb) = build_wkb else {
                continue;
            };
            let distance = self.inner.evaluator.resolve_distance(
                indexed_batch.distance(),
                row_idx as usize,
                distance,
            )?;
            let geom_idx = self.inner.geom_idx_vec[*data_idx as usize];
            index_query_results.push(IndexQueryResult {
                wkb: build_wkb,
                distance,
                geom_idx,
                position: pos,
            });
        }

        if index_query_results.is_empty() {
            return Ok(QueryResultMetrics {
                count: 0,
                candidate_count,
            });
        }

        let results = self.inner.refiner.refine(probe_wkb, &index_query_results)?;
        let num_results = results.len();
        build_batch_positions.extend(results);

        // Update refiner memory reservation
        self.inner
            .refiner_reservation
            .resize(self.inner.refiner.mem_usage())?;

        Ok(QueryResultMetrics {
            count: num_results,
            candidate_count,
        })
    }
}

#[async_trait]
impl SpatialIndexInternal for CPUSpatialIndex {
    fn schema(&self) -> SchemaRef {
        self.inner.schema.clone()
    }

    fn get_num_indexed_batches(&self) -> usize {
        self.inner.indexed_batches.len()
    }

    fn get_indexed_batch(&self, batch_idx: usize) -> &RecordBatch {
        &self.inner.indexed_batches[batch_idx].batch
    }

    fn need_more_probe_stats(&self) -> bool {
        self.inner.refiner.need_more_probe_stats()
    }

    fn merge_probe_stats(&self, stats: GeoStatistics) {
        self.inner.refiner.merge_probe_stats(stats);
    }

    fn visited_left_side(&self) -> Option<&Mutex<Vec<BooleanBufferBuilder>>> {
        self.inner.visited_left_side.as_ref()
    }

    fn report_probe_completed(&self) -> bool {
        self.inner
            .probe_threads_counter
            .fetch_sub(1, Ordering::Relaxed)
            == 1
    }

    fn get_refiner_mem_usage(&self) -> usize {
        self.inner.refiner.mem_usage()
    }

    fn get_actual_execution_mode(&self) -> ExecutionMode {
        self.inner.refiner.actual_execution_mode()
    }

    async fn query_batch(
        &self,
        evaluated_batch: &Arc<EvaluatedBatch>,
        range: Range<usize>,
        max_result_size: usize,
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

        let rects = evaluated_batch.rects();
        let dist = evaluated_batch.distance();
        let mut total_candidates_count = 0;
        let mut total_count = 0;
        let mut current_row_idx = range.start;
        for row_idx in range {
            current_row_idx = row_idx;
            let Some(probe_rect) = rects[row_idx] else {
                continue;
            };

            let min = probe_rect.min();
            let max = probe_rect.max();
            let mut candidates = self.inner.rtree.search(min.x, min.y, max.x, max.y);
            if candidates.is_empty() {
                continue;
            }

            let Some(probe_wkb) = evaluated_batch.wkb(row_idx) else {
                return sedona_internal_err!(
                    "Failed to get WKB for row {} in evaluated batch",
                    row_idx
                );
            };

            // Sort and dedup candidates to avoid duplicate results when we index one geometry
            // using several boxes.
            candidates.sort_unstable();
            candidates.dedup();

            let distance = match dist {
                Some(dist_array) => distance_value_at(dist_array, row_idx)?,
                None => None,
            };

            // Refine the candidates retrieved from the r-tree index by evaluating the actual spatial predicate
            let refine_chunk_size = self.inner.options.parallel_refinement_chunk_size;
            if refine_chunk_size == 0 || candidates.len() < refine_chunk_size * 2 {
                // For small candidate sets, use refine synchronously
                let metrics =
                    self.refine(probe_wkb, &candidates, &distance, build_batch_positions)?;
                probe_indices.extend(std::iter::repeat_n(row_idx as u32, metrics.count));
                total_count += metrics.count;
                total_candidates_count += metrics.candidate_count;
            } else {
                // For large candidate sets, spawn several tasks to parallelize refinement
                let (metrics, positions) = self
                    .refine_concurrently(
                        evaluated_batch,
                        row_idx,
                        &candidates,
                        distance,
                        refine_chunk_size,
                    )
                    .await?;
                build_batch_positions.extend(positions);
                probe_indices.extend(std::iter::repeat_n(row_idx as u32, metrics.count));
                total_count += metrics.count;
                total_candidates_count += metrics.candidate_count;
            }

            if total_count >= max_result_size {
                break;
            }
        }

        let end_idx = current_row_idx + 1;
        Ok((
            QueryResultMetrics {
                count: total_count,
                candidate_count: total_candidates_count,
            },
            end_idx,
        ))
    }
}

impl SpatialIndex for CPUSpatialIndex {
    #[allow(unused)]
    fn query(
        &self,
        probe_wkb: &Wkb,
        probe_rect: &Rect<f32>,
        distance: &Option<f64>,
        build_batch_positions: &mut Vec<(i32, i32)>,
    ) -> Result<QueryResultMetrics> {
        let min = probe_rect.min();
        let max = probe_rect.max();
        let mut candidates = self.inner.rtree.search(min.x, min.y, max.x, max.y);
        if candidates.is_empty() {
            return Ok(QueryResultMetrics {
                count: 0,
                candidate_count: 0,
            });
        }

        // Sort and dedup candidates to avoid duplicate results when we index one geometry
        // using several boxes.
        candidates.sort_unstable();
        candidates.dedup();

        // Refine the candidates retrieved from the r-tree index by evaluating the actual spatial predicate
        self.refine(probe_wkb, &candidates, distance, build_batch_positions)
    }

    fn query_knn(
        &self,
        probe_wkb: &Wkb,
        k: u32,
        use_spheroid: bool,
        include_tie_breakers: bool,
        build_batch_positions: &mut Vec<(i32, i32)>,
    ) -> Result<QueryResultMetrics> {
        if k == 0 {
            return Ok(QueryResultMetrics {
                count: 0,
                candidate_count: 0,
            });
        }

        // Check if index is empty
        if self.inner.indexed_batches.is_empty() || self.inner.data_id_to_batch_pos.is_empty() {
            return Ok(QueryResultMetrics {
                count: 0,
                candidate_count: 0,
            });
        }

        // Convert probe WKB to geo::Geometry
        let probe_geom = match item_to_geometry(probe_wkb) {
            Ok(geom) => geom,
            Err(_) => {
                // Empty or unsupported geometries (e.g., POINT EMPTY) return empty results
                return Ok(QueryResultMetrics {
                    count: 0,
                    candidate_count: 0,
                });
            }
        };

        // Select the appropriate distance metric
        let distance_metric: &dyn DistanceMetric<f32> = {
            let Some(knn_components) = self.inner.knn_components.as_ref() else {
                return sedona_internal_err!(
                    "knn_components is not initialized when running KNN join"
                );
            };
            if use_spheroid {
                &knn_components.haversine_metric
            } else {
                &knn_components.euclidean_metric
            }
        };

        // Create geometry accessor for on-demand WKB decoding and caching
        let geometry_accessor = self.create_knn_accessor()?;

        // Use neighbors_geometry to find k nearest neighbors
        let initial_results = self.inner.rtree.neighbors_geometry(
            &probe_geom,
            Some(k as usize),
            None, // no max_distance filter
            distance_metric,
            &geometry_accessor,
        );

        if initial_results.is_empty() {
            return Ok(QueryResultMetrics {
                count: 0,
                candidate_count: 0,
            });
        }

        let mut final_results = initial_results;
        let mut candidate_count = final_results.len();

        // Handle tie-breakers if enabled
        if include_tie_breakers && !final_results.is_empty() && k > 0 {
            // Calculate distances for the initial k results to find the k-th distance
            let mut distances_with_indices: Vec<(f64, u32)> = Vec::new();

            for &result_idx in &final_results {
                if (result_idx as usize) < self.inner.data_id_to_batch_pos.len() {
                    if let Some(item_geom) = geometry_accessor.get_geometry(result_idx as usize) {
                        let distance = distance_metric.distance_to_geometry(&probe_geom, item_geom);
                        if let Some(distance_f64) = distance.to_f64() {
                            distances_with_indices.push((distance_f64, result_idx));
                        }
                    }
                }
            }

            // Sort by distance
            distances_with_indices
                .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            // Find the k-th distance (if we have at least k results)
            if distances_with_indices.len() >= k as usize {
                let k_idx = (k as usize)
                    .min(distances_with_indices.len())
                    .saturating_sub(1);
                let max_distance = distances_with_indices[k_idx].0;

                // For tie-breakers, create spatial envelope around probe centroid and use rtree.search()

                // Create envelope bounds by expanding the probe bounding box by max_distance
                let Some(rect) = probe_geom.bounding_rect() else {
                    // If bounding rectangle cannot be computed, return empty results
                    return Ok(QueryResultMetrics {
                        count: 0,
                        candidate_count: 0,
                    });
                };

                let min = rect.min();
                let max = rect.max();
                let (min_x, min_y, max_x, max_y) = f64_box_to_f32(min.x, min.y, max.x, max.y);
                let mut distance_f32 = max_distance as f32;
                if (distance_f32 as f64) < max_distance {
                    distance_f32 = distance_f32.next_after(f32::INFINITY);
                }
                let (min_x, min_y, max_x, max_y) = (
                    min_x - distance_f32,
                    min_y - distance_f32,
                    max_x + distance_f32,
                    max_y + distance_f32,
                );

                // Use rtree.search() with envelope bounds (like the old code)
                let expanded_results = self.inner.rtree.search(min_x, min_y, max_x, max_y);

                candidate_count = expanded_results.len();

                // Calculate distances for all results and find ties
                let mut all_distances_with_indices: Vec<(f64, u32)> = Vec::new();

                for &result_idx in &expanded_results {
                    if (result_idx as usize) < self.inner.data_id_to_batch_pos.len() {
                        if let Some(item_geom) = geometry_accessor.get_geometry(result_idx as usize)
                        {
                            let distance =
                                distance_metric.distance_to_geometry(&probe_geom, item_geom);
                            if let Some(distance_f64) = distance.to_f64() {
                                all_distances_with_indices.push((distance_f64, result_idx));
                            }
                        }
                    }
                }

                // Sort by distance
                all_distances_with_indices
                    .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                // Include all results up to and including those with the same distance as the k-th result
                const DISTANCE_TOLERANCE: f64 = 1e-9;
                let mut tie_breaker_results: Vec<u32> = Vec::new();

                for (i, &(distance, result_idx)) in all_distances_with_indices.iter().enumerate() {
                    if i < k as usize {
                        // Include the first k results
                        tie_breaker_results.push(result_idx);
                    } else if (distance - max_distance).abs() <= DISTANCE_TOLERANCE {
                        // Include tie-breakers (same distance as k-th result)
                        tie_breaker_results.push(result_idx);
                    } else {
                        // No more ties, stop
                        break;
                    }
                }

                final_results = tie_breaker_results;
            }
        } else {
            // When tie-breakers are disabled, limit results to exactly k
            if final_results.len() > k as usize {
                final_results.truncate(k as usize);
            }
        }

        // Convert results to build_batch_positions using existing data_id_to_batch_pos mapping
        for &result_idx in &final_results {
            if (result_idx as usize) < self.inner.data_id_to_batch_pos.len() {
                build_batch_positions.push(self.inner.data_id_to_batch_pos[result_idx as usize]);
            }
        }

        Ok(QueryResultMetrics {
            count: final_results.len(),
            candidate_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        index::{CPUSpatialIndexBuilder as SpatialIndexBuilder, SpatialJoinBuildMetrics},
        operand_evaluator::EvaluatedGeometryArray,
        spatial_predicate::{KNNPredicate, RelationPredicate},
    };

    use super::*;
    use crate::index::spatial_index::SpatialIndexRef;
    use arrow_array::RecordBatch;
    use arrow_schema::{DataType, Field};
    use datafusion_common::JoinSide;
    use datafusion_execution::memory_pool::GreedyMemoryPool;
    use datafusion_expr::JoinType;
    use datafusion_physical_expr::expressions::Column;
    use geo_traits::Dimensions;
    use sedona_common::option::{ExecutionMode, SpatialJoinOptions};
    use sedona_geometry::spatial_relation::SpatialRelationType;
    use sedona_geometry::wkb_factory::write_wkb_empty_point;
    use sedona_schema::datatypes::WKB_GEOMETRY;
    use sedona_testing::create::create_array;

    #[test]
    fn test_spatial_index_builder_empty() {
        let memory_pool = Arc::new(GreedyMemoryPool::new(1024 * 1024));
        let options = SpatialJoinOptions {
            execution_mode: ExecutionMode::PrepareBuild,
            ..Default::default()
        };
        let metrics = SpatialJoinBuildMetrics::default();
        let schema = Arc::new(arrow_schema::Schema::empty());
        let spatial_predicate = SpatialPredicate::Relation(RelationPredicate::new(
            Arc::new(Column::new("geom", 0)),
            Arc::new(Column::new("geom", 1)),
            SpatialRelationType::Intersects,
        ));

        let builder = SpatialIndexBuilder::new(
            schema.clone(),
            spatial_predicate,
            options,
            JoinType::Inner,
            4,
            memory_pool,
            metrics,
        )
        .unwrap();

        // Test finishing with empty data
        let index = builder.finish().unwrap();
        assert_eq!(index.schema(), schema);
        assert_eq!(index.get_num_indexed_batches(), 0);
    }

    #[test]
    fn test_spatial_index_builder_add_batch() {
        let memory_pool = Arc::new(GreedyMemoryPool::new(1024 * 1024));
        let options = SpatialJoinOptions {
            execution_mode: ExecutionMode::PrepareBuild,
            ..Default::default()
        };
        let metrics = SpatialJoinBuildMetrics::default();

        let spatial_predicate = SpatialPredicate::Relation(RelationPredicate::new(
            Arc::new(Column::new("geom", 0)),
            Arc::new(Column::new("geom", 1)),
            SpatialRelationType::Intersects,
        ));

        // Create a simple test geometry batch
        let schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "geom",
            DataType::Binary,
            true,
        )]));

        let mut builder = SpatialIndexBuilder::new(
            schema.clone(),
            spatial_predicate,
            options,
            JoinType::Inner,
            4,
            memory_pool,
            metrics,
        )
        .unwrap();

        let batch = RecordBatch::new_empty(schema.clone());
        let geom_batch = create_array(
            &[
                Some("POINT (0.25 0.25)"),
                Some("POINT (10 10)"),
                None,
                Some("POINT (0.25 0.25)"),
            ],
            &WKB_GEOMETRY,
        );
        let indexed_batch = EvaluatedBatch {
            batch,
            geom_array: EvaluatedGeometryArray::try_new(geom_batch, &WKB_GEOMETRY).unwrap(),
        };
        builder.add_batch(indexed_batch).unwrap();

        let index = builder.finish().unwrap();
        assert_eq!(index.schema(), schema);
        assert_eq!(index.get_num_indexed_batches(), 1);
    }

    #[test]
    fn test_knn_query_execution_with_sample_data() {
        // Create a spatial index with sample geometry data
        let memory_pool = Arc::new(GreedyMemoryPool::new(1024 * 1024));
        let options = SpatialJoinOptions {
            execution_mode: ExecutionMode::PrepareBuild,
            ..Default::default()
        };
        let metrics = SpatialJoinBuildMetrics::default();

        let spatial_predicate = SpatialPredicate::KNearestNeighbors(KNNPredicate::new(
            Arc::new(Column::new("geom", 0)),
            Arc::new(Column::new("geom", 1)),
            5,
            false,
            JoinSide::Left,
        ));

        // Create sample geometry data - points at known locations
        let schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "geom",
            DataType::Binary,
            true,
        )]));

        let mut builder = SpatialIndexBuilder::new(
            schema.clone(),
            spatial_predicate,
            options,
            JoinType::Inner,
            4,
            memory_pool,
            metrics,
        )
        .unwrap();

        let batch = RecordBatch::new_empty(schema.clone());

        // Create geometries at different distances from the query point (0, 0)
        let geom_batch = create_array(
            &[
                Some("POINT (1 0)"), // Distance: 1.0
                Some("POINT (0 2)"), // Distance: 2.0
                Some("POINT (3 0)"), // Distance: 3.0
                Some("POINT (0 4)"), // Distance: 4.0
                Some("POINT (5 0)"), // Distance: 5.0
                Some("POINT (2 2)"), // Distance: ~2.83
                Some("POINT (1 1)"), // Distance: ~1.41
            ],
            &WKB_GEOMETRY,
        );

        let indexed_batch = EvaluatedBatch {
            batch,
            geom_array: EvaluatedGeometryArray::try_new(geom_batch, &WKB_GEOMETRY).unwrap(),
        };
        builder.add_batch(indexed_batch).unwrap();

        let index = builder.finish().unwrap();

        // Create a query geometry at origin (0, 0)
        let query_geom = create_array(&[Some("POINT (0 0)")], &WKB_GEOMETRY);
        let query_array = EvaluatedGeometryArray::try_new(query_geom, &WKB_GEOMETRY).unwrap();
        let query_wkb = &query_array.wkbs()[0].as_ref().unwrap();

        // Test KNN query with k=3
        let mut build_positions = Vec::new();
        let result = index
            .query_knn(
                query_wkb,
                3,     // k=3
                false, // use_spheroid=false
                false, // include_tie_breakers=false
                &mut build_positions,
            )
            .unwrap();

        // Verify we got 3 results
        assert_eq!(build_positions.len(), 3);
        assert_eq!(result.count, 3);
        assert!(result.candidate_count >= 3);

        // Create a mapping of positions to verify correct ordering
        // We expect the 3 closest points: (1,0), (1,1), (0,2)
        let expected_closest_indices = vec![0, 6, 1]; // Based on our sample data ordering
        let mut found_indices = Vec::new();

        for (_batch_idx, row_idx) in &build_positions {
            found_indices.push(*row_idx as usize);
        }

        // Sort to compare sets (order might vary due to implementation)
        found_indices.sort();
        let mut expected_sorted = expected_closest_indices;
        expected_sorted.sort();

        assert_eq!(found_indices, expected_sorted);
    }

    #[test]
    fn test_knn_query_execution_with_different_k_values() {
        // Create spatial index with more data points
        let memory_pool = Arc::new(GreedyMemoryPool::new(1024 * 1024));
        let options = SpatialJoinOptions {
            execution_mode: ExecutionMode::PrepareBuild,
            ..Default::default()
        };
        let metrics = SpatialJoinBuildMetrics::default();

        let spatial_predicate = SpatialPredicate::KNearestNeighbors(KNNPredicate::new(
            Arc::new(Column::new("geom", 0)),
            Arc::new(Column::new("geom", 1)),
            5,
            false,
            JoinSide::Left,
        ));

        let schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "geom",
            DataType::Binary,
            true,
        )]));

        let mut builder = SpatialIndexBuilder::new(
            schema.clone(),
            spatial_predicate,
            options,
            JoinType::Inner,
            4,
            memory_pool,
            metrics,
        )
        .unwrap();

        let batch = RecordBatch::new_empty(schema.clone());

        // Create 10 points at regular intervals
        let geom_batch = create_array(
            &[
                Some("POINT (1 0)"),  // 0: Distance 1
                Some("POINT (2 0)"),  // 1: Distance 2
                Some("POINT (3 0)"),  // 2: Distance 3
                Some("POINT (4 0)"),  // 3: Distance 4
                Some("POINT (5 0)"),  // 4: Distance 5
                Some("POINT (6 0)"),  // 5: Distance 6
                Some("POINT (7 0)"),  // 6: Distance 7
                Some("POINT (8 0)"),  // 7: Distance 8
                Some("POINT (9 0)"),  // 8: Distance 9
                Some("POINT (10 0)"), // 9: Distance 10
            ],
            &WKB_GEOMETRY,
        );

        let indexed_batch = EvaluatedBatch {
            batch,
            geom_array: EvaluatedGeometryArray::try_new(geom_batch, &WKB_GEOMETRY).unwrap(),
        };
        builder.add_batch(indexed_batch).unwrap();

        let index = builder.finish().unwrap();

        // Query point at origin
        let query_geom = create_array(&[Some("POINT (0 0)")], &WKB_GEOMETRY);
        let query_array = EvaluatedGeometryArray::try_new(query_geom, &WKB_GEOMETRY).unwrap();
        let query_wkb = &query_array.wkbs()[0].as_ref().unwrap();

        // Test different k values
        for k in [1, 3, 5, 7, 10] {
            let mut build_positions = Vec::new();
            let result = index
                .query_knn(query_wkb, k, false, false, &mut build_positions)
                .unwrap();

            // Verify we got exactly k results (or all available if k > total)
            let expected_results = std::cmp::min(k as usize, 10);
            assert_eq!(build_positions.len(), expected_results);
            assert_eq!(result.count, expected_results);

            // Verify the results are the k closest points
            let mut row_indices: Vec<usize> = build_positions
                .iter()
                .map(|(_, row_idx)| *row_idx as usize)
                .collect();
            row_indices.sort();

            let expected_indices: Vec<usize> = (0..expected_results).collect();
            assert_eq!(row_indices, expected_indices);
        }
    }

    #[test]
    fn test_knn_query_execution_with_spheroid_distance() {
        // Create spatial index
        let memory_pool = Arc::new(GreedyMemoryPool::new(1024 * 1024));
        let options = SpatialJoinOptions {
            execution_mode: ExecutionMode::PrepareBuild,
            ..Default::default()
        };
        let metrics = SpatialJoinBuildMetrics::default();

        let spatial_predicate = SpatialPredicate::KNearestNeighbors(KNNPredicate::new(
            Arc::new(Column::new("geom", 0)),
            Arc::new(Column::new("geom", 1)),
            5,
            true,
            JoinSide::Left,
        ));

        let schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "geom",
            DataType::Binary,
            true,
        )]));

        let mut builder = SpatialIndexBuilder::new(
            schema.clone(),
            spatial_predicate,
            options,
            JoinType::Inner,
            4,
            memory_pool,
            metrics,
        )
        .unwrap();

        let batch = RecordBatch::new_empty(schema.clone());

        // Create points with geographic coordinates (longitude, latitude)
        let geom_batch = create_array(
            &[
                Some("POINT (-74.0 40.7)"), // NYC area
                Some("POINT (-73.9 40.7)"), // Slightly east
                Some("POINT (-74.1 40.7)"), // Slightly west
                Some("POINT (-74.0 40.8)"), // Slightly north
                Some("POINT (-74.0 40.6)"), // Slightly south
            ],
            &WKB_GEOMETRY,
        );

        let indexed_batch = EvaluatedBatch {
            batch,
            geom_array: EvaluatedGeometryArray::try_new(geom_batch, &WKB_GEOMETRY).unwrap(),
        };
        builder.add_batch(indexed_batch).unwrap();

        let index = builder.finish().unwrap();

        // Query point at NYC
        let query_geom = create_array(&[Some("POINT (-74.0 40.7)")], &WKB_GEOMETRY);
        let query_array = EvaluatedGeometryArray::try_new(query_geom, &WKB_GEOMETRY).unwrap();
        let query_wkb = &query_array.wkbs()[0].as_ref().unwrap();

        // Test with planar distance (spheroid distance is not supported)
        let mut build_positions = Vec::new();
        let result = index
            .query_knn(
                query_wkb,
                3,     // k=3
                false, // use_spheroid=false (only supported option)
                false,
                &mut build_positions,
            )
            .unwrap();

        // Should find results with planar distance calculation
        assert!(!build_positions.is_empty()); // At least the exact match
        assert!(result.count >= 1);
        assert!(result.candidate_count >= 1);

        // Test that spheroid distance now works with Haversine metric
        let mut build_positions_spheroid = Vec::new();
        let result_spheroid = index.query_knn(
            query_wkb,
            3,    // k=3
            true, // use_spheroid=true (now supported with Haversine)
            false,
            &mut build_positions_spheroid,
        );

        // Should succeed and return results
        assert!(result_spheroid.is_ok());
        let result_spheroid = result_spheroid.unwrap();
        assert!(!build_positions_spheroid.is_empty());
        assert!(result_spheroid.count >= 1);
        assert!(result_spheroid.candidate_count >= 1);
    }

    #[test]
    fn test_knn_query_execution_edge_cases() {
        // Create spatial index
        let memory_pool = Arc::new(GreedyMemoryPool::new(1024 * 1024));
        let options = SpatialJoinOptions {
            execution_mode: ExecutionMode::PrepareBuild,
            ..Default::default()
        };
        let metrics = SpatialJoinBuildMetrics::default();

        let spatial_predicate = SpatialPredicate::KNearestNeighbors(KNNPredicate::new(
            Arc::new(Column::new("geom", 0)),
            Arc::new(Column::new("geom", 1)),
            5,
            false,
            JoinSide::Left,
        ));

        let schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "geom",
            DataType::Binary,
            true,
        )]));

        let mut builder = SpatialIndexBuilder::new(
            schema.clone(),
            spatial_predicate,
            options,
            JoinType::Inner,
            4,
            memory_pool,
            metrics,
        )
        .unwrap();

        let batch = RecordBatch::new_empty(schema.clone());

        // Create sample data with some edge cases
        let geom_batch = create_array(
            &[
                Some("POINT (1 1)"),
                Some("POINT (2 2)"),
                None, // NULL geometry
                Some("POINT (3 3)"),
            ],
            &WKB_GEOMETRY,
        );

        let indexed_batch = EvaluatedBatch {
            batch,
            geom_array: EvaluatedGeometryArray::try_new(geom_batch, &WKB_GEOMETRY).unwrap(),
        };
        builder.add_batch(indexed_batch).unwrap();

        let index = builder.finish().unwrap();

        let query_geom = create_array(&[Some("POINT (0 0)")], &WKB_GEOMETRY);
        let query_array = EvaluatedGeometryArray::try_new(query_geom, &WKB_GEOMETRY).unwrap();
        let query_wkb = &query_array.wkbs()[0].as_ref().unwrap();

        // Test k=0 (should return no results)
        let mut build_positions = Vec::new();
        let result = index
            .query_knn(
                query_wkb,
                0, // k=0
                false,
                false,
                &mut build_positions,
            )
            .unwrap();

        assert_eq!(build_positions.len(), 0);
        assert_eq!(result.count, 0);
        assert_eq!(result.candidate_count, 0);

        // Test k > available geometries
        let mut build_positions = Vec::new();
        let result = index
            .query_knn(
                query_wkb,
                10, // k=10, but only 3 valid geometries available
                false,
                false,
                &mut build_positions,
            )
            .unwrap();

        // Should return all available valid geometries (excluding NULL)
        assert_eq!(build_positions.len(), 3);
        assert_eq!(result.count, 3);
    }

    #[test]
    fn test_knn_query_execution_empty_index() {
        // Create empty spatial index
        let memory_pool = Arc::new(GreedyMemoryPool::new(1024 * 1024));
        let options = SpatialJoinOptions {
            execution_mode: ExecutionMode::PrepareBuild,
            ..Default::default()
        };
        let metrics = SpatialJoinBuildMetrics::default();
        let schema = Arc::new(arrow_schema::Schema::empty());

        let spatial_predicate = SpatialPredicate::KNearestNeighbors(KNNPredicate::new(
            Arc::new(Column::new("geom", 0)),
            Arc::new(Column::new("geom", 1)),
            5,
            false,
            JoinSide::Left,
        ));

        let builder = SpatialIndexBuilder::new(
            schema.clone(),
            spatial_predicate,
            options,
            JoinType::Inner,
            4,
            memory_pool,
            metrics,
        )
        .unwrap();

        let index = builder.finish().unwrap();

        // Try to query empty index
        let query_geom = create_array(&[Some("POINT (0 0)")], &WKB_GEOMETRY);
        let query_array = EvaluatedGeometryArray::try_new(query_geom, &WKB_GEOMETRY).unwrap();
        let query_wkb = &query_array.wkbs()[0].as_ref().unwrap();

        let mut build_positions = Vec::new();
        let result = index
            .query_knn(query_wkb, 5, false, false, &mut build_positions)
            .unwrap();

        // Should return no results for empty index
        assert_eq!(build_positions.len(), 0);
        assert_eq!(result.count, 0);
        assert_eq!(result.candidate_count, 0);
    }

    #[test]
    fn test_knn_query_execution_with_tie_breakers() {
        // Create a spatial index with sample geometry data
        let memory_pool = Arc::new(GreedyMemoryPool::new(1024 * 1024));
        let options = SpatialJoinOptions {
            execution_mode: ExecutionMode::PrepareBuild,
            ..Default::default()
        };
        let metrics = SpatialJoinBuildMetrics::default();

        let spatial_predicate = SpatialPredicate::KNearestNeighbors(KNNPredicate::new(
            Arc::new(Column::new("geom", 0)),
            Arc::new(Column::new("geom", 1)),
            5,
            false,
            JoinSide::Left,
        ));

        let schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "geom",
            DataType::Binary,
            true,
        )]));

        let mut builder = SpatialIndexBuilder::new(
            schema.clone(),
            spatial_predicate,
            options,
            JoinType::Inner,
            1, // probe_threads_count
            memory_pool.clone(),
            metrics,
        )
        .unwrap();

        let batch = RecordBatch::new_empty(schema.clone());

        // Create points where we have more ties at the k-th distance
        // Query point is at (0.0, 0.0)
        // We'll create a scenario with k=2 where there are 3 points at the same distance
        // This ensures the tie-breaker logic has work to do
        let geom_batch = create_array(
            &[
                Some("POINT (1.0 0.0)"),  // Squared distance 1.0
                Some("POINT (0.0 1.0)"),  // Squared distance 1.0 (tie!)
                Some("POINT (-1.0 0.0)"), // Squared distance 1.0 (tie!)
                Some("POINT (0.0 -1.0)"), // Squared distance 1.0 (tie!)
                Some("POINT (2.0 0.0)"),  // Squared distance 4.0
                Some("POINT (0.0 2.0)"),  // Squared distance 4.0
            ],
            &WKB_GEOMETRY,
        );

        let indexed_batch = EvaluatedBatch {
            batch,
            geom_array: EvaluatedGeometryArray::try_new(geom_batch, &WKB_GEOMETRY).unwrap(),
        };
        builder.add_batch(indexed_batch).unwrap();

        let index = builder.finish().unwrap();

        // Query point at the origin (0.0, 0.0)
        let query_geom = create_array(&[Some("POINT (0.0 0.0)")], &WKB_GEOMETRY);
        let query_array = EvaluatedGeometryArray::try_new(query_geom, &WKB_GEOMETRY).unwrap();
        let query_wkb = &query_array.wkbs()[0].as_ref().unwrap();

        // Test without tie-breakers: should return exactly k=2 results
        let mut build_positions = Vec::new();
        let result = index
            .query_knn(
                query_wkb,
                2,     // k=2
                false, // use_spheroid
                false, // include_tie_breakers
                &mut build_positions,
            )
            .unwrap();

        // Should return exactly 2 results (the closest point + 1 of the tied points)
        assert_eq!(result.count, 2);
        assert_eq!(build_positions.len(), 2);

        // Test with tie-breakers: should return k=2 plus all ties
        let mut build_positions_with_ties = Vec::new();
        let result_with_ties = index
            .query_knn(
                query_wkb,
                2,     // k=2
                false, // use_spheroid
                true,  // include_tie_breakers
                &mut build_positions_with_ties,
            )
            .unwrap();

        // Should return more than 2 results because of ties
        // We have 4 points at squared distance 1.0 (all tied for closest)
        // With k=2 and tie-breakers:
        // - Initial neighbors query returns 2 of the 4 tied points
        // - Tie-breaker logic should find the other 2 tied points
        // - Total should be 4 results (all points at distance 1.0)

        // With 4 points all at the same distance and k=2:
        // - Without tie-breakers: should return exactly 2
        // - With tie-breakers: should return all 4 tied points
        assert_eq!(
            result.count, 2,
            "Without tie-breakers should return exactly k=2"
        );
        assert_eq!(
            result_with_ties.count, 4,
            "With tie-breakers should return all 4 tied points"
        );
        assert_eq!(build_positions_with_ties.len(), 4);
    }

    #[test]
    fn test_query_knn_with_geometry_distance() {
        // Create a spatial index with sample geometry data
        let memory_pool = Arc::new(GreedyMemoryPool::new(1024 * 1024));
        let options = SpatialJoinOptions {
            execution_mode: ExecutionMode::PrepareBuild,
            ..Default::default()
        };
        let metrics = SpatialJoinBuildMetrics::default();

        let spatial_predicate = SpatialPredicate::KNearestNeighbors(KNNPredicate::new(
            Arc::new(Column::new("geom", 0)),
            Arc::new(Column::new("geom", 1)),
            5,
            false,
            JoinSide::Left,
        ));

        // Create sample geometry data - points at known locations
        let schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "geom",
            DataType::Binary,
            true,
        )]));

        let mut builder = SpatialIndexBuilder::new(
            schema.clone(),
            spatial_predicate,
            options,
            JoinType::Inner,
            4,
            memory_pool,
            metrics,
        )
        .unwrap();

        let batch = RecordBatch::new_empty(schema.clone());

        // Create geometries at different distances from the query point (0, 0)
        let geom_batch = create_array(
            &[
                Some("POINT (1 0)"), // Distance: 1.0
                Some("POINT (0 2)"), // Distance: 2.0
                Some("POINT (3 0)"), // Distance: 3.0
                Some("POINT (0 4)"), // Distance: 4.0
                Some("POINT (5 0)"), // Distance: 5.0
                Some("POINT (2 2)"), // Distance: ~2.83
                Some("POINT (1 1)"), // Distance: ~1.41
            ],
            &WKB_GEOMETRY,
        );

        let indexed_batch = EvaluatedBatch {
            batch,
            geom_array: EvaluatedGeometryArray::try_new(geom_batch, &WKB_GEOMETRY).unwrap(),
        };
        builder.add_batch(indexed_batch).unwrap();

        let index = builder.finish().unwrap();

        // Create a query geometry at origin (0, 0)
        let query_geom = create_array(&[Some("POINT (0 0)")], &WKB_GEOMETRY);
        let query_array = EvaluatedGeometryArray::try_new(query_geom, &WKB_GEOMETRY).unwrap();
        let query_wkb = &query_array.wkbs()[0].as_ref().unwrap();

        // Test the geometry-based query_knn method with k=3
        let mut build_positions = Vec::new();
        let result = index
            .query_knn(
                query_wkb,
                3,     // k=3
                false, // use_spheroid=false
                false, // include_tie_breakers=false
                &mut build_positions,
            )
            .unwrap();

        // Verify we got results (should be 3 or less)
        assert!(!build_positions.is_empty());
        assert!(build_positions.len() <= 3);
        assert!(result.count > 0);
        assert!(result.count <= 3);
    }

    #[test]
    fn test_query_knn_with_mixed_geometries() {
        // Create a spatial index with complex geometries where geometry-based
        // distance should differ from centroid-based distance
        let memory_pool = Arc::new(GreedyMemoryPool::new(1024 * 1024));
        let options = SpatialJoinOptions {
            execution_mode: ExecutionMode::PrepareBuild,
            ..Default::default()
        };
        let metrics = SpatialJoinBuildMetrics::default();

        let spatial_predicate = SpatialPredicate::KNearestNeighbors(KNNPredicate::new(
            Arc::new(Column::new("geom", 0)),
            Arc::new(Column::new("geom", 1)),
            5,
            false,
            JoinSide::Left,
        ));

        // Create different geometry types
        let schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "geom",
            DataType::Binary,
            true,
        )]));

        let mut builder = SpatialIndexBuilder::new(
            schema.clone(),
            spatial_predicate,
            options,
            JoinType::Inner,
            4,
            memory_pool,
            metrics,
        )
        .unwrap();

        let batch = RecordBatch::new_empty(schema.clone());

        // Mix of points and linestrings
        let geom_batch = create_array(
            &[
                Some("POINT (1 1)"),               // Simple point
                Some("LINESTRING (2 0, 2 4)"),     // Vertical line - closest point should be (2, 1)
                Some("LINESTRING (10 10, 10 20)"), // Far away line
                Some("POINT (5 5)"),               // Far point
            ],
            &WKB_GEOMETRY,
        );

        let indexed_batch = EvaluatedBatch {
            batch,
            geom_array: EvaluatedGeometryArray::try_new(geom_batch, &WKB_GEOMETRY).unwrap(),
        };
        builder.add_batch(indexed_batch).unwrap();

        let index = builder.finish().unwrap();

        // Query point close to the linestring
        let query_geom = create_array(&[Some("POINT (2.1 1.0)")], &WKB_GEOMETRY);
        let query_array = EvaluatedGeometryArray::try_new(query_geom, &WKB_GEOMETRY).unwrap();
        let query_wkb = &query_array.wkbs()[0].as_ref().unwrap();

        // Test the geometry-based KNN method with mixed geometry types
        let mut build_positions = Vec::new();

        let result = index
            .query_knn(
                query_wkb,
                2,     // k=2
                false, // use_spheroid=false
                false, // include_tie_breakers=false
                &mut build_positions,
            )
            .unwrap();

        // Should return results
        assert!(!build_positions.is_empty());

        // Should work with mixed geometry types
        assert!(result.count > 0);
    }

    #[test]
    fn test_query_knn_with_tie_breakers_geometry_distance() {
        // Create a spatial index with geometries that have identical distances for tie-breaker testing
        let memory_pool = Arc::new(GreedyMemoryPool::new(1024 * 1024));
        let options = SpatialJoinOptions {
            execution_mode: ExecutionMode::PrepareBuild,
            ..Default::default()
        };
        let metrics = SpatialJoinBuildMetrics::default();

        let spatial_predicate = SpatialPredicate::KNearestNeighbors(KNNPredicate::new(
            Arc::new(Column::new("geom", 0)),
            Arc::new(Column::new("geom", 1)),
            5,
            false,
            JoinSide::Left,
        ));

        let schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "geom",
            DataType::Binary,
            true,
        )]));

        let mut builder = SpatialIndexBuilder::new(
            schema.clone(),
            spatial_predicate,
            options,
            JoinType::Inner,
            4,
            memory_pool,
            metrics,
        )
        .unwrap();

        let batch = RecordBatch::new_empty(schema.clone());

        // Create points where we have multiple points at the same distance from the query point
        // Query point will be at (0, 0), and we'll have 4 points all at distance sqrt(2) ≈ 1.414
        let geom_batch = create_array(
            &[
                Some("POINT (1.0 1.0)"),   // Distance: sqrt(2)
                Some("POINT (1.0 -1.0)"),  // Distance: sqrt(2) - tied with above
                Some("POINT (-1.0 1.0)"),  // Distance: sqrt(2) - tied with above
                Some("POINT (-1.0 -1.0)"), // Distance: sqrt(2) - tied with above
                Some("POINT (2.0 0.0)"),   // Distance: 2.0 - farther away
                Some("POINT (0.0 2.0)"),   // Distance: 2.0 - farther away
            ],
            &WKB_GEOMETRY,
        );

        let indexed_batch = EvaluatedBatch {
            batch,
            geom_array: EvaluatedGeometryArray::try_new(geom_batch, &WKB_GEOMETRY).unwrap(),
        };
        builder.add_batch(indexed_batch).unwrap();

        let index = builder.finish().unwrap();

        // Query point at the origin (0.0, 0.0)
        let query_geom = create_array(&[Some("POINT (0.0 0.0)")], &WKB_GEOMETRY);
        let query_array = EvaluatedGeometryArray::try_new(query_geom, &WKB_GEOMETRY).unwrap();
        let query_wkb = &query_array.wkbs()[0].as_ref().unwrap();

        // Test without tie-breakers: should return exactly k=2 results
        let mut build_positions = Vec::new();
        let result = index
            .query_knn(
                query_wkb,
                2,     // k=2
                false, // use_spheroid
                false, // include_tie_breakers=false
                &mut build_positions,
            )
            .unwrap();

        // Should return exactly 2 results
        assert_eq!(result.count, 2);
        assert_eq!(build_positions.len(), 2);

        // Test with tie-breakers: should return all tied points
        let mut build_positions_with_ties = Vec::new();
        let result_with_ties = index
            .query_knn(
                query_wkb,
                2,     // k=2
                false, // use_spheroid
                true,  // include_tie_breakers=true
                &mut build_positions_with_ties,
            )
            .unwrap();

        // Should return 4 results because of ties (all 4 points at distance sqrt(2))
        assert!(result_with_ties.count == 4);

        // Query using a box centered at the origin
        let query_geom = create_array(
            &[Some(
                "POLYGON ((-0.5 -0.5, -0.5 0.5, 0.5 0.5, 0.5 -0.5, -0.5 -0.5))",
            )],
            &WKB_GEOMETRY,
        );
        let query_array = EvaluatedGeometryArray::try_new(query_geom, &WKB_GEOMETRY).unwrap();
        let query_wkb = &query_array.wkbs()[0].as_ref().unwrap();

        // This query should return 4 points
        let mut build_positions_with_ties = Vec::new();
        let result_with_ties = index
            .query_knn(
                query_wkb,
                2,     // k=2
                false, // use_spheroid
                true,  // include_tie_breakers=true
                &mut build_positions_with_ties,
            )
            .unwrap();

        // Should return 4 results because of ties (all 4 points at distance sqrt(2))
        assert!(result_with_ties.count == 4);
    }

    #[test]
    fn test_knn_query_with_empty_geometry() {
        // Create a spatial index with sample geometry data like other tests
        let memory_pool = Arc::new(GreedyMemoryPool::new(1024 * 1024));
        let options = SpatialJoinOptions {
            execution_mode: ExecutionMode::PrepareBuild,
            ..Default::default()
        };
        let metrics = SpatialJoinBuildMetrics::default();

        let spatial_predicate = SpatialPredicate::KNearestNeighbors(KNNPredicate::new(
            Arc::new(Column::new("geom", 0)),
            Arc::new(Column::new("geom", 1)),
            5,
            false,
            JoinSide::Left,
        ));

        // Create geometry batch using the same pattern as other tests
        let schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "geom",
            DataType::Binary,
            true,
        )]));

        let mut builder = SpatialIndexBuilder::new(
            schema.clone(),
            spatial_predicate,
            options,
            JoinType::Inner,
            1, // probe_threads_count
            memory_pool.clone(),
            metrics,
        )
        .unwrap();

        let batch = RecordBatch::new_empty(schema.clone());

        let geom_batch = create_array(
            &[
                Some("POINT (0 0)"),
                Some("POINT (1 1)"),
                Some("POINT (2 2)"),
            ],
            &WKB_GEOMETRY,
        );
        let indexed_batch = EvaluatedBatch {
            batch,
            geom_array: EvaluatedGeometryArray::try_new(geom_batch, &WKB_GEOMETRY).unwrap(),
        };
        builder.add_batch(indexed_batch).unwrap();

        let index = builder.finish().unwrap();

        // Create an empty point WKB
        let mut empty_point_wkb = Vec::new();
        write_wkb_empty_point(&mut empty_point_wkb, Dimensions::Xy).unwrap();

        // Query with the empty point
        let mut build_positions = Vec::new();
        let result = index
            .query_knn(
                &wkb::reader::read_wkb(&empty_point_wkb).unwrap(),
                2,     // k=2
                false, // use_spheroid
                false, // include_tie_breakers
                &mut build_positions,
            )
            .unwrap();

        // Should return empty results for empty geometry
        assert_eq!(result.count, 0);
        assert_eq!(result.candidate_count, 0);
        assert!(build_positions.is_empty());
    }

    async fn setup_index_for_batch_test(
        build_geoms: &[Option<&str>],
        options: SpatialJoinOptions,
    ) -> SpatialIndexRef {
        let memory_pool = Arc::new(GreedyMemoryPool::new(100 * 1024 * 1024));
        let metrics = SpatialJoinBuildMetrics::default();
        let spatial_predicate = SpatialPredicate::Relation(RelationPredicate::new(
            Arc::new(Column::new("left", 0)),
            Arc::new(Column::new("right", 0)),
            SpatialRelationType::Intersects,
        ));
        let schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "geom",
            DataType::Binary,
            true,
        )]));

        let mut builder = SpatialIndexBuilder::new(
            schema,
            spatial_predicate,
            options,
            JoinType::Inner,
            1,
            memory_pool,
            metrics,
        )
        .unwrap();

        let geom_array = create_array(build_geoms, &WKB_GEOMETRY);
        let batch = RecordBatch::try_new(
            Arc::new(arrow_schema::Schema::new(vec![Field::new(
                "geom",
                DataType::Binary,
                true,
            )])),
            vec![Arc::new(geom_array.clone())],
        )
        .unwrap();
        let evaluated_batch = EvaluatedBatch {
            batch,
            geom_array: EvaluatedGeometryArray::try_new(geom_array, &WKB_GEOMETRY).unwrap(),
        };

        builder.add_batch(evaluated_batch).unwrap();
        builder.finish().unwrap()
    }

    fn create_probe_batch(probe_geoms: &[Option<&str>]) -> Arc<EvaluatedBatch> {
        let geom_array = create_array(probe_geoms, &WKB_GEOMETRY);
        let batch = RecordBatch::try_new(
            Arc::new(arrow_schema::Schema::new(vec![Field::new(
                "geom",
                DataType::Binary,
                true,
            )])),
            vec![Arc::new(geom_array.clone())],
        )
        .unwrap();
        Arc::new(EvaluatedBatch {
            batch,
            geom_array: EvaluatedGeometryArray::try_new(geom_array, &WKB_GEOMETRY).unwrap(),
        })
    }

    #[tokio::test]
    async fn test_query_batch_empty_results() {
        let build_geoms = &[Some("POINT (0 0)"), Some("POINT (1 1)")];
        let index = setup_index_for_batch_test(build_geoms, SpatialJoinOptions::default()).await;

        // Probe with geometries that don't intersect
        let probe_geoms = &[Some("POINT (10 10)"), Some("POINT (20 20)")];
        let probe_batch = create_probe_batch(probe_geoms);

        let mut build_batch_positions = Vec::new();
        let mut probe_indices = Vec::new();
        let (metrics, next_idx) = index
            .query_batch(
                &probe_batch,
                0..2,
                usize::MAX,
                &mut build_batch_positions,
                &mut probe_indices,
            )
            .await
            .unwrap();

        assert_eq!(metrics.count, 0);
        assert_eq!(build_batch_positions.len(), 0);
        assert_eq!(probe_indices.len(), 0);
        assert_eq!(next_idx, 2);
    }

    #[tokio::test]
    async fn test_query_batch_max_result_size() {
        let build_geoms = &[
            Some("POINT (0 0)"),
            Some("POINT (0 0)"),
            Some("POINT (0 0)"),
        ];
        let index = setup_index_for_batch_test(build_geoms, SpatialJoinOptions::default()).await;

        // Probe with geometry that intersects all 3
        let probe_geoms = &[Some("POINT (0 0)"), Some("POINT (0 0)")];
        let probe_batch = create_probe_batch(probe_geoms);

        // Case 1: Max result size is large enough
        let mut build_batch_positions = Vec::new();
        let mut probe_indices = Vec::new();
        let (metrics, next_idx) = index
            .query_batch(
                &probe_batch,
                0..2,
                10,
                &mut build_batch_positions,
                &mut probe_indices,
            )
            .await
            .unwrap();
        assert_eq!(metrics.count, 6); // 2 probes * 3 matches
        assert_eq!(next_idx, 2);
        assert_eq!(probe_indices, vec![0, 0, 0, 1, 1, 1]);

        // Case 2: Max result size is small (stops after first probe)
        let mut build_batch_positions = Vec::new();
        let mut probe_indices = Vec::new();
        let (metrics, next_idx) = index
            .query_batch(
                &probe_batch,
                0..2,
                2, // Stop after 2 results
                &mut build_batch_positions,
                &mut probe_indices,
            )
            .await
            .unwrap();

        // It should process the first probe, find 3 matches.
        // Since 3 >= 2, it should stop.
        assert_eq!(metrics.count, 3);
        assert_eq!(next_idx, 1); // Only processed 1 probe
        assert_eq!(probe_indices, vec![0, 0, 0]);
    }

    #[tokio::test]
    async fn test_query_batch_parallel_refinement() {
        // Create enough build geometries to trigger parallel refinement
        // We need candidates.len() >= chunk_size * 2
        // Let's set chunk_size = 2, so we need >= 4 candidates.
        let build_geoms = vec![Some("POINT (0 0)"); 10];
        let options = SpatialJoinOptions {
            parallel_refinement_chunk_size: 2,
            ..Default::default()
        };

        let index = setup_index_for_batch_test(&build_geoms, options).await;

        // Probe with a geometry that intersects all build geometries
        let probe_geoms = &[Some("POLYGON ((-1 -1, 1 -1, 1 1, -1 1, -1 -1))")];
        let probe_batch = create_probe_batch(probe_geoms);

        let mut build_batch_positions = Vec::new();
        let mut probe_indices = Vec::new();
        let (metrics, next_idx) = index
            .query_batch(
                &probe_batch,
                0..1,
                usize::MAX,
                &mut build_batch_positions,
                &mut probe_indices,
            )
            .await
            .unwrap();

        assert_eq!(metrics.count, 10);
        assert_eq!(build_batch_positions.len(), 10);
        assert_eq!(probe_indices, vec![0; 10]);
        assert_eq!(next_idx, 1);
    }

    #[tokio::test]
    async fn test_query_batch_empty_range() {
        let build_geoms = &[Some("POINT (0 0)")];
        let index = setup_index_for_batch_test(build_geoms, SpatialJoinOptions::default()).await;
        let probe_geoms = &[Some("POINT (0 0)"), Some("POINT (0 0)")];
        let probe_batch = create_probe_batch(probe_geoms);

        let mut build_batch_positions = Vec::new();
        let mut probe_indices = Vec::new();

        // Query with empty range
        for empty_ranges in [0..0, 1..1, 2..2] {
            let (metrics, next_idx) = index
                .query_batch(
                    &probe_batch,
                    empty_ranges.clone(),
                    usize::MAX,
                    &mut build_batch_positions,
                    &mut probe_indices,
                )
                .await
                .unwrap();

            assert_eq!(metrics.count, 0);
            assert_eq!(next_idx, empty_ranges.end);
        }
    }

    #[tokio::test]
    async fn test_query_batch_range_offset() {
        let build_geoms = &[Some("POINT (0 0)"), Some("POINT (1 1)")];
        let index = setup_index_for_batch_test(build_geoms, SpatialJoinOptions::default()).await;

        // Probe with 3 geometries:
        // 0: POINT (0 0) - matches build[0] (should be skipped)
        // 1: POINT (0 0) - matches build[0]
        // 2: POINT (1 1) - matches build[1]
        let probe_geoms = &[
            Some("POINT (0 0)"),
            Some("POINT (0 0)"),
            Some("POINT (1 1)"),
        ];
        let probe_batch = create_probe_batch(probe_geoms);

        let mut build_batch_positions = Vec::new();
        let mut probe_indices = Vec::new();

        // Query with range 1..3 (skipping the first probe)
        let (metrics, next_idx) = index
            .query_batch(
                &probe_batch,
                1..3,
                usize::MAX,
                &mut build_batch_positions,
                &mut probe_indices,
            )
            .await
            .unwrap();

        assert_eq!(metrics.count, 2);
        assert_eq!(next_idx, 3);

        // probe_indices should contain indices relative to the batch start (1 and 2)
        assert_eq!(probe_indices, vec![1, 2]);

        // build_batch_positions should contain matches for probe 1 and probe 2
        // probe 1 matches build 0 (0, 0)
        // probe 2 matches build 1 (0, 1)
        // Note: build_batch_positions contains (batch_idx, row_idx)
        // Since we have 1 batch, batch_idx is 0.
        assert_eq!(build_batch_positions, vec![(0, 0), (0, 1)]);
    }

    #[tokio::test]
    async fn test_query_batch_zero_parallel_refinement_chunk_size() {
        let build_geoms = &[
            Some("POINT (0 0)"),
            Some("POINT (0 0)"),
            Some("POINT (0 0)"),
        ];
        let options = SpatialJoinOptions {
            // force synchronous refinement
            parallel_refinement_chunk_size: 0,
            ..Default::default()
        };

        let index = setup_index_for_batch_test(build_geoms, options).await;
        let probe_geoms = &[Some("POINT (0 0)")];
        let probe_batch = create_probe_batch(probe_geoms);

        let mut build_batch_positions = Vec::new();
        let mut probe_indices = Vec::new();

        let result = index
            .query_batch(
                &probe_batch,
                0..1,
                10,
                &mut build_batch_positions,
                &mut probe_indices,
            )
            .await;

        assert!(result.is_ok());
        let (metrics, _) = result.unwrap();
        assert_eq!(metrics.count, 3);
    }
}
