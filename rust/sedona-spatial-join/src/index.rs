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
use once_cell::sync::OnceCell;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion_common::{utils::proxy::VecAllocExt, DataFusionError, Result};
use datafusion_common_runtime::JoinSet;
use datafusion_execution::{
    memory_pool::{MemoryConsumer, MemoryPool, MemoryReservation},
    SendableRecordBatchStream,
};
use datafusion_expr::{ColumnarValue, JoinType};
use datafusion_physical_plan::metrics::{self, ExecutionPlanMetricsSet, MetricBuilder};
use futures::StreamExt;
use geo_index::rtree::distance::{
    DistanceMetric, EuclideanDistance, GeometryAccessor, HaversineDistance,
};
use geo_index::rtree::{sort::HilbertSort, RTree, RTreeBuilder, RTreeIndex};
use geo_index::IndexableNum;
use geo_types::{Geometry, Point, Rect};
use parking_lot::Mutex;
use sedona_expr::statistics::GeoStatistics;
use sedona_functions::st_analyze_aggr::AnalyzeAccumulator;
use sedona_geo::to_geo::item_to_geometry;
use sedona_geo_generic_alg::algorithm::Centroid;
use sedona_schema::datatypes::WKB_GEOMETRY;
use wkb::reader::Wkb;

use crate::{
    concurrent_reservation::ConcurrentReservation,
    operand_evaluator::{create_operand_evaluator, EvaluatedGeometryArray, OperandEvaluator},
    refine::{create_refiner, IndexQueryResultRefiner},
    spatial_predicate::SpatialPredicate,
    utils::need_produce_result_in_final,
};
use arrow::array::BooleanBufferBuilder;
use sedona_common::{option::SpatialJoinOptions, ExecutionMode};

// Type aliases for better readability
type SpatialRTree = RTree<f32>;
type DataIdToBatchPos = Vec<(i32, i32)>;
type RTreeBuildResult = (SpatialRTree, DataIdToBatchPos);

/// The prealloc size for the refiner reservation. This is used to reduce the frequency of growing
/// the reservation when updating the refiner memory reservation.
const REFINER_RESERVATION_PREALLOC_SIZE: usize = 10 * 1024 * 1024; // 10MB

/// Metrics for the build phase of the spatial join.
#[derive(Clone, Debug, Default)]
pub(crate) struct SpatialJoinBuildMetrics {
    /// Total time for collecting build-side of join
    pub(crate) build_time: metrics::Time,
    /// Number of batches consumed by build-side
    pub(crate) build_input_batches: metrics::Count,
    /// Number of rows consumed by build-side
    pub(crate) build_input_rows: metrics::Count,
    /// Memory used by build-side in bytes
    pub(crate) build_mem_used: metrics::Gauge,
}

impl SpatialJoinBuildMetrics {
    pub fn new(partition: usize, metrics: &ExecutionPlanMetricsSet) -> Self {
        Self {
            build_time: MetricBuilder::new(metrics).subset_time("build_time", partition),
            build_input_batches: MetricBuilder::new(metrics)
                .counter("build_input_batches", partition),
            build_input_rows: MetricBuilder::new(metrics).counter("build_input_rows", partition),
            build_mem_used: MetricBuilder::new(metrics).gauge("build_mem_used", partition),
        }
    }
}

/// Builder for constructing a SpatialIndex from geometry batches.
///
/// This builder handles:
/// 1. Accumulating geometry batches to be indexed
/// 2. Building the spatial R-tree index
/// 3. Setting up memory tracking and visited bitmaps
/// 4. Configuring prepared geometries based on execution mode
pub(crate) struct SpatialIndexBuilder {
    spatial_predicate: SpatialPredicate,
    options: SpatialJoinOptions,
    join_type: JoinType,
    probe_threads_count: usize,
    metrics: SpatialJoinBuildMetrics,

    /// Batches to be indexed
    indexed_batches: Vec<IndexedBatch>,
    /// Memory reservation for tracking the memory usage of the spatial index
    reservation: MemoryReservation,

    /// Statistics for indexed geometries
    stats: GeoStatistics,

    /// Memory pool for managing the memory usage of the spatial index
    memory_pool: Arc<dyn MemoryPool>,
}

impl SpatialIndexBuilder {
    /// Create a new builder with the given configuration.
    pub fn new(
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
    pub fn add_batch(&mut self, indexed_batch: IndexedBatch) {
        let in_mem_size = indexed_batch.in_mem_size();
        self.indexed_batches.push(indexed_batch);
        self.reservation.grow(in_mem_size);
        self.metrics.build_mem_used.add(in_mem_size);
    }

    pub fn with_stats(&mut self, stats: GeoStatistics) -> &mut Self {
        self.stats.merge(&stats);
        self
    }

    /// Build the spatial R-tree index from collected geometry batches.
    fn build_rtree(&mut self) -> Result<RTreeBuildResult> {
        let build_timer = self.metrics.build_time.timer();

        let num_rects = self
            .indexed_batches
            .iter()
            .map(|batch| batch.rects().len())
            .sum::<usize>();

        let mut rtree_builder = RTreeBuilder::<f32>::new(num_rects as u32);
        let mut batch_pos_vec = vec![(0, 0); num_rects];
        let rtree_mem_estimate = num_rects * RTREE_MEMORY_ESTIMATE_PER_RECT;

        self.reservation
            .grow(batch_pos_vec.allocated_size() + rtree_mem_estimate);

        for (batch_idx, batch) in self.indexed_batches.iter().enumerate() {
            let rects = batch.rects();
            for (idx, rect) in rects {
                let min = rect.min();
                let max = rect.max();
                let data_idx = rtree_builder.add(min.x, min.y, max.x, max.y);
                batch_pos_vec[data_idx as usize] = (batch_idx as i32, *idx as i32);
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
    pub fn finish(mut self, schema: SchemaRef) -> Result<SpatialIndex> {
        if self.indexed_batches.is_empty() {
            return Ok(SpatialIndex::empty(
                self.spatial_predicate,
                schema,
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
            schema,
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
}

pub(crate) struct SpatialIndex {
    schema: SchemaRef,

    /// The spatial predicate evaluator for the spatial predicate.
    evaluator: Arc<dyn OperandEvaluator>,

    /// The refiner for refining the index query results.
    refiner: Arc<dyn IndexQueryResultRefiner>,

    /// Memory reservation for tracking the memory usage of the refiner
    refiner_reservation: ConcurrentReservation,

    /// R-tree index for the geometry batches. It takes MBRs as query windows and returns
    /// data indexes. These data indexes should be translated using `data_id_to_batch_pos` to get
    /// the original geometry batch index and row index, or translated using `prepared_geom_idx_vec`
    /// to get the prepared geometries array index.
    rtree: RTree<f32>,

    /// Indexed batches containing evaluated geometry arrays. It contains the original record
    /// batches and geometry arrays obtained by evaluating the geometry expression on the build side.
    indexed_batches: Vec<IndexedBatch>,
    /// An array for translating rtree data index to geometry batch index and row index
    data_id_to_batch_pos: Vec<(i32, i32)>,

    /// An array for translating rtree data index to consecutive index. Each geometry may be indexed by
    /// multiple boxes, so there could be multiple data indexes for the same geometry. A mapping for
    /// squashing the index makes it easier for persisting per-geometry auxiliary data for evaluating
    /// the spatial predicate. This is extensively used by the spatial predicate evaluators for storing
    /// prepared geometries.
    geom_idx_vec: Vec<usize>,

    /// Shared bitmap builders for visited left indices, one per batch
    visited_left_side: Option<Mutex<Vec<BooleanBufferBuilder>>>,

    /// Counter of running probe-threads, potentially able to update `bitmap`.
    /// Each time a probe thread finished probing the index, it will decrement the counter.
    /// The last finished probe thread will produce the extra output batches for unmatched
    /// build side when running left-outer joins. See also [`report_probe_completed`].
    probe_threads_counter: AtomicUsize,

    /// Shared KNN components (distance metrics and geometry cache) for efficient KNN queries
    knn_components: KnnComponents,

    /// Memory reservation for tracking the memory usage of the spatial index
    /// Cleared on `SpatialIndex` drop
    #[expect(dead_code)]
    reservation: MemoryReservation,
}

/// Indexed batch containing the original record batch and the evaluated geometry array.
pub(crate) struct IndexedBatch {
    batch: RecordBatch,
    geom_array: EvaluatedGeometryArray,
}

impl IndexedBatch {
    pub fn in_mem_size(&self) -> usize {
        // NOTE: sometimes `geom_array` will reuse the memory of `batch`, especially when
        // the expression for evaluating the geometry is a simple column reference. In this case,
        // the in_mem_size will be overestimated.
        self.batch.get_array_memory_size() + self.geom_array.in_mem_size()
    }

    pub fn wkb(&self, idx: usize) -> Option<&Wkb<'_>> {
        let wkbs = self.geom_array.wkbs();
        wkbs[idx].as_ref()
    }

    pub fn rects(&self) -> &Vec<(usize, Rect<f32>)> {
        &self.geom_array.rects
    }

    pub fn distance(&self) -> &Option<ColumnarValue> {
        &self.geom_array.distance
    }
}

#[derive(Debug)]
pub struct JoinResultMetrics {
    pub count: usize,
    pub candidate_count: usize,
}

impl SpatialIndex {
    fn empty(
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
        Self {
            schema,
            evaluator,
            refiner,
            refiner_reservation,
            rtree,
            data_id_to_batch_pos: Vec::new(),
            indexed_batches: Vec::new(),
            geom_idx_vec: Vec::new(),
            visited_left_side: None,
            probe_threads_counter,
            knn_components: KnnComponents::new(0, &[], memory_pool.clone()).unwrap(), // Empty index has no cache
            reservation,
        }
    }

    pub(crate) fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    /// Create a KNN geometry accessor for accessing geometries with caching
    fn create_knn_accessor(&self) -> SedonaKnnAdapter<'_> {
        SedonaKnnAdapter::new(
            &self.indexed_batches,
            &self.data_id_to_batch_pos,
            &self.knn_components,
        )
    }

    /// Get the batch at the given index.
    pub(crate) fn get_indexed_batch(&self, batch_idx: usize) -> &RecordBatch {
        &self.indexed_batches[batch_idx].batch
    }

    /// Query the spatial index with a probe geometry to find matching build-side geometries.
    ///
    /// This method implements a two-phase spatial join query:
    /// 1. **Filter phase**: Uses the R-tree index with the probe geometry's bounding rectangle
    ///    to quickly identify candidate geometries that might satisfy the spatial predicate
    /// 2. **Refinement phase**: Evaluates the exact spatial predicate on candidates to determine
    ///    actual matches
    ///
    /// # Arguments
    /// * `probe_wkb` - The probe geometry in WKB format
    /// * `probe_rect` - The minimum bounding rectangle of the probe geometry
    /// * `distance` - Optional distance parameter for distance-based spatial predicates
    /// * `build_batch_positions` - Output vector that will be populated with (batch_idx, row_idx)
    ///   pairs for each matching build-side geometry
    ///
    /// # Returns
    /// * `JoinResultMetrics` containing the number of actual matches (`count`) and the number
    ///   of candidates from the filter phase (`candidate_count`)
    pub(crate) fn query(
        &self,
        probe_wkb: &Wkb,
        probe_rect: &Rect<f32>,
        distance: &Option<f64>,
        build_batch_positions: &mut Vec<(i32, i32)>,
    ) -> Result<JoinResultMetrics> {
        let min = probe_rect.min();
        let max = probe_rect.max();
        let mut candidates = self.rtree.search(min.x, min.y, max.x, max.y);
        if candidates.is_empty() {
            return Ok(JoinResultMetrics {
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

    /// Query the spatial index for k nearest neighbors of a given geometry.
    ///
    /// This method finds the k nearest neighbors to the probe geometry using:
    /// 1. R-tree's built-in neighbors() method for efficient KNN search
    /// 2. Distance refinement using actual geometry calculations
    /// 3. Tie-breaker handling when enabled
    ///
    /// # Arguments
    ///
    /// * `probe_wkb` - WKB representation of the probe geometry
    /// * `k` - Number of nearest neighbors to find
    /// * `use_spheroid` - Whether to use spheroid distance calculation
    /// * `include_tie_breakers` - Whether to include additional results with same distance as kth neighbor
    /// * `build_batch_positions` - Output vector for matched positions
    ///
    /// # Returns
    ///
    /// * `JoinResultMetrics` containing the number of actual matches and candidates processed
    pub(crate) fn query_knn(
        &self,
        probe_wkb: &Wkb,
        k: u32,
        use_spheroid: bool,
        include_tie_breakers: bool,
        build_batch_positions: &mut Vec<(i32, i32)>,
    ) -> Result<JoinResultMetrics> {
        if k == 0 {
            return Ok(JoinResultMetrics {
                count: 0,
                candidate_count: 0,
            });
        }

        // Check if index is empty
        if self.indexed_batches.is_empty() || self.data_id_to_batch_pos.is_empty() {
            return Ok(JoinResultMetrics {
                count: 0,
                candidate_count: 0,
            });
        }

        // Convert probe WKB to geo::Geometry
        let probe_geom = match item_to_geometry(probe_wkb) {
            Ok(geom) => geom,
            Err(_) => {
                // Empty or unsupported geometries (e.g., POINT EMPTY) return empty results
                return Ok(JoinResultMetrics {
                    count: 0,
                    candidate_count: 0,
                });
            }
        };

        // Select the appropriate distance metric
        let distance_metric: &dyn DistanceMetric<f32> = if use_spheroid {
            &self.knn_components.haversine_metric
        } else {
            &self.knn_components.euclidean_metric
        };

        // Create geometry accessor for on-demand WKB decoding and caching
        let geometry_accessor = self.create_knn_accessor();

        // Use neighbors_geometry to find k nearest neighbors
        let initial_results = self.rtree.neighbors_geometry(
            &probe_geom,
            Some(k as usize),
            None, // no max_distance filter
            distance_metric,
            &geometry_accessor,
        );

        if initial_results.is_empty() {
            return Ok(JoinResultMetrics {
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
                if (result_idx as usize) < self.data_id_to_batch_pos.len() {
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

                let probe_centroid = probe_geom.centroid().unwrap_or(Point::new(0.0, 0.0));
                let probe_x = probe_centroid.x() as f32;
                let probe_y = probe_centroid.y() as f32;
                let max_distance_f32 = match f32::from_f64(max_distance) {
                    Some(val) => val,
                    None => {
                        // If conversion fails, return empty results for this probe
                        return Ok(JoinResultMetrics {
                            count: 0,
                            candidate_count: 0,
                        });
                    }
                };

                // Create envelope bounds around probe centroid
                let min_x = probe_x - max_distance_f32;
                let min_y = probe_y - max_distance_f32;
                let max_x = probe_x + max_distance_f32;
                let max_y = probe_y + max_distance_f32;

                // Use rtree.search() with envelope bounds (like the old code)
                let expanded_results = self.rtree.search(min_x, min_y, max_x, max_y);

                candidate_count = expanded_results.len();

                // Calculate distances for all results and find ties
                let mut all_distances_with_indices: Vec<(f64, u32)> = Vec::new();

                for &result_idx in &expanded_results {
                    if (result_idx as usize) < self.data_id_to_batch_pos.len() {
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
            if (result_idx as usize) < self.data_id_to_batch_pos.len() {
                build_batch_positions.push(self.data_id_to_batch_pos[result_idx as usize]);
            }
        }

        Ok(JoinResultMetrics {
            count: final_results.len(),
            candidate_count,
        })
    }

    fn refine(
        &self,
        probe_wkb: &Wkb,
        candidates: &[u32],
        distance: &Option<f64>,
        build_batch_positions: &mut Vec<(i32, i32)>,
    ) -> Result<JoinResultMetrics> {
        let candidate_count = candidates.len();

        let mut index_query_results = Vec::with_capacity(candidate_count);
        for data_idx in candidates {
            let pos = self.data_id_to_batch_pos[*data_idx as usize];
            let (batch_idx, row_idx) = pos;
            let indexed_batch = &self.indexed_batches[batch_idx as usize];
            let build_wkb = indexed_batch.wkb(row_idx as usize);
            let Some(build_wkb) = build_wkb else {
                continue;
            };
            let distance = self.evaluator.resolve_distance(
                indexed_batch.distance(),
                row_idx as usize,
                distance,
            )?;
            let geom_idx = self.geom_idx_vec[*data_idx as usize];
            index_query_results.push(IndexQueryResult {
                wkb: build_wkb,
                distance,
                geom_idx,
                position: pos,
            });
        }

        if index_query_results.is_empty() {
            return Ok(JoinResultMetrics {
                count: 0,
                candidate_count,
            });
        }

        let results = self.refiner.refine(probe_wkb, &index_query_results)?;
        let num_results = results.len();
        build_batch_positions.extend(results);

        // Update refiner memory reservation
        self.refiner_reservation.resize(self.refiner.mem_usage())?;

        Ok(JoinResultMetrics {
            count: num_results,
            candidate_count,
        })
    }

    /// Check if the index needs more probe statistics to determine the optimal execution mode.
    ///
    /// # Returns
    /// * `bool` - `true` if the index needs more probe statistics, `false` otherwise.
    pub(crate) fn need_more_probe_stats(&self) -> bool {
        self.refiner.need_more_probe_stats()
    }

    /// Merge the probe statistics into the index.
    ///
    /// # Arguments
    /// * `stats` - The probe statistics to merge.
    pub(crate) fn merge_probe_stats(&self, stats: GeoStatistics) {
        self.refiner.merge_probe_stats(stats);
    }

    /// Get the bitmaps for tracking visited left-side indices. The bitmaps will be updated
    /// by the spatial join stream when producing output batches during index probing phase.
    pub(crate) fn visited_left_side(&self) -> Option<&Mutex<Vec<BooleanBufferBuilder>>> {
        self.visited_left_side.as_ref()
    }

    /// Decrements counter of running threads, and returns `true`
    /// if caller is the last running thread
    pub(crate) fn report_probe_completed(&self) -> bool {
        self.probe_threads_counter.fetch_sub(1, Ordering::Relaxed) == 1
    }

    /// Get the memory usage of the refiner in bytes.
    pub(crate) fn get_refiner_mem_usage(&self) -> usize {
        self.refiner.mem_usage()
    }

    /// Get the actual execution mode used by the refiner
    pub(crate) fn get_actual_execution_mode(&self) -> ExecutionMode {
        self.refiner.actual_execution_mode()
    }
}

pub struct IndexQueryResult<'a, 'b> {
    pub wkb: &'b Wkb<'a>,
    pub distance: Option<f64>,
    pub geom_idx: usize,
    pub position: (i32, i32),
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn build_index(
    mut build_schema: SchemaRef,
    build_streams: Vec<SendableRecordBatchStream>,
    spatial_predicate: SpatialPredicate,
    options: SpatialJoinOptions,
    metrics_vec: Vec<SpatialJoinBuildMetrics>,
    memory_pool: Arc<dyn MemoryPool>,
    join_type: JoinType,
    probe_threads_count: usize,
) -> Result<SpatialIndex> {
    // Handle empty streams case
    if build_streams.is_empty() {
        let consumer = MemoryConsumer::new("SpatialJoinIndex");
        let reservation = consumer.register(&memory_pool);
        return Ok(SpatialIndex::empty(
            spatial_predicate,
            build_schema,
            options,
            AtomicUsize::new(probe_threads_count),
            reservation,
            memory_pool,
        ));
    }

    // Update schema from the first stream
    build_schema = build_streams.first().unwrap().schema();
    let metrics = metrics_vec.first().unwrap().clone();
    let evaluator = create_operand_evaluator(&spatial_predicate, options.clone());

    // Spawn all tasks to scan all build streams concurrently
    let mut join_set = JoinSet::new();
    let collect_statistics = matches!(options.execution_mode, ExecutionMode::Speculative(_));
    for (partition, (stream, metrics)) in build_streams.into_iter().zip(metrics_vec).enumerate() {
        let per_task_evaluator = Arc::clone(&evaluator);
        let consumer = MemoryConsumer::new(format!("SpatialJoinFetchBuild[{partition}]"));
        let reservation = consumer.register(&memory_pool);
        join_set.spawn(async move {
            collect_build_partition(
                stream,
                per_task_evaluator.as_ref(),
                &metrics,
                reservation,
                collect_statistics,
            )
            .await
        });
    }

    // Process each task as it completes and add batches to builder
    let results = join_set.join_all().await;

    // Create the builder to build the index
    let mut builder = SpatialIndexBuilder::new(
        spatial_predicate,
        options,
        join_type,
        probe_threads_count,
        memory_pool.clone(),
        metrics,
    )?;
    for result in results {
        let build_partition =
            result.map_err(|e| DataFusionError::Execution(format!("Task join error: {e}")))?;

        // Add each geometry batch to the builder
        for indexed_batch in build_partition.batches {
            builder.add_batch(indexed_batch);
        }
        builder.with_stats(build_partition.stats);
        // build_partition.reservation will be dropped here.
    }

    // Finish building the index
    builder.finish(build_schema)
}

struct BuildPartition {
    batches: Vec<IndexedBatch>,
    stats: GeoStatistics,

    /// Memory reservation for tracking the memory usage of the build partition
    /// Cleared on `BuildPartition` drop
    #[allow(dead_code)]
    reservation: MemoryReservation,
}

async fn collect_build_partition(
    mut stream: SendableRecordBatchStream,
    evaluator: &dyn OperandEvaluator,
    metrics: &SpatialJoinBuildMetrics,
    mut reservation: MemoryReservation,
    collect_statistics: bool,
) -> Result<BuildPartition> {
    let mut batches = Vec::new();
    let mut analyzer = AnalyzeAccumulator::new(WKB_GEOMETRY, WKB_GEOMETRY);

    while let Some(batch) = stream.next().await {
        let build_timer = metrics.build_time.timer();
        let batch = batch?;

        metrics.build_input_rows.add(batch.num_rows());
        metrics.build_input_batches.add(1);

        let geom_array = evaluator.evaluate_build(&batch)?;
        let indexed_batch = IndexedBatch { batch, geom_array };

        // Update statistics for each geometry in the batch
        if collect_statistics {
            for wkb in indexed_batch.geom_array.wkbs().iter().flatten() {
                analyzer.update_statistics(wkb, wkb.buf().len())?;
            }
        }

        let in_mem_size = indexed_batch.in_mem_size();
        batches.push(indexed_batch);

        reservation.grow(in_mem_size);
        metrics.build_mem_used.add(in_mem_size);
        build_timer.done();
    }

    Ok(BuildPartition {
        batches,
        stats: analyzer.finish(),
        reservation,
    })
}

/// Rough estimate for in-memory size of the rtree per rect in bytes
const RTREE_MEMORY_ESTIMATE_PER_RECT: usize = 60;

/// Shared KNN components that can be reused across queries
struct KnnComponents {
    euclidean_metric: EuclideanDistance,
    haversine_metric: HaversineDistance,
    /// Pre-allocated vector for geometry cache - lock-free access
    /// Indexed by rtree data index for O(1) access
    geometry_cache: Vec<OnceCell<Geometry<f64>>>,
    /// Memory reservation to track geometry cache memory usage
    _reservation: MemoryReservation,
}

impl KnnComponents {
    fn new(
        cache_size: usize,
        indexed_batches: &[IndexedBatch],
        memory_pool: Arc<dyn MemoryPool>,
    ) -> datafusion_common::Result<Self> {
        // Create memory consumer and reservation for geometry cache
        let consumer = MemoryConsumer::new("SpatialJoinKnnGeometryCache");
        let mut reservation = consumer.register(&memory_pool);

        // Estimate maximum possible memory usage based on WKB sizes
        let estimated_memory = Self::estimate_max_memory_usage(indexed_batches);
        reservation.try_grow(estimated_memory)?;

        // Pre-allocate OnceCell vector
        let geometry_cache = (0..cache_size).map(|_| OnceCell::new()).collect();

        Ok(Self {
            euclidean_metric: EuclideanDistance,
            haversine_metric: HaversineDistance::default(),
            geometry_cache,
            _reservation: reservation,
        })
    }

    /// Estimate the maximum memory usage for decoded geometries based on WKB sizes
    fn estimate_max_memory_usage(indexed_batches: &[IndexedBatch]) -> usize {
        let mut total_wkb_size = 0;

        for batch in indexed_batches {
            for wkb in batch.geom_array.wkbs().iter().flatten() {
                total_wkb_size += wkb.buf().len();
            }
        }
        total_wkb_size
    }
}

/// Geometry accessor for SedonaDB KNN queries.
/// This accessor provides on-demand WKB decoding and geometry caching for efficient
/// KNN queries with support for both Euclidean and Haversine distance metrics.
struct SedonaKnnAdapter<'a> {
    indexed_batches: &'a [IndexedBatch],
    data_id_to_batch_pos: &'a [(i32, i32)],
    // Reference to KNN components for cache and memory tracking
    knn_components: &'a KnnComponents,
}

impl<'a> SedonaKnnAdapter<'a> {
    /// Create a new adapter
    fn new(
        indexed_batches: &'a [IndexedBatch],
        data_id_to_batch_pos: &'a [(i32, i32)],
        knn_components: &'a KnnComponents,
    ) -> Self {
        Self {
            indexed_batches,
            data_id_to_batch_pos,
            knn_components,
        }
    }
}

impl<'a> GeometryAccessor for SedonaKnnAdapter<'a> {
    /// Get geometry for the given item index with lock-free caching
    fn get_geometry(&self, item_index: usize) -> Option<&Geometry<f64>> {
        let geometry_cache = &self.knn_components.geometry_cache;

        // Bounds check
        if item_index >= geometry_cache.len() || item_index >= self.data_id_to_batch_pos.len() {
            return None;
        }

        // Try to get from cache first
        if let Some(geom) = geometry_cache[item_index].get() {
            return Some(geom);
        }

        // Cache miss - decode from WKB
        let (batch_idx, row_idx) = self.data_id_to_batch_pos[item_index];
        let indexed_batch = &self.indexed_batches[batch_idx as usize];

        if let Some(wkb) = indexed_batch.wkb(row_idx as usize) {
            if let Ok(geom) = item_to_geometry(wkb) {
                // Try to store in cache - if another thread got there first, we just use theirs
                let _ = geometry_cache[item_index].set(geom);
                // Return reference to the cached geometry
                return geometry_cache[item_index].get();
            }
        }

        // Failed to decode - don't cache invalid results
        None
    }
}

#[cfg(test)]
mod tests {
    use crate::spatial_predicate::{RelationPredicate, SpatialRelationType};

    use super::*;
    use arrow_array::RecordBatch;
    use arrow_schema::{DataType, Field};
    use datafusion_execution::memory_pool::GreedyMemoryPool;
    use datafusion_physical_expr::expressions::Column;
    use geo_traits::Dimensions;
    use sedona_common::option::{ExecutionMode, SpatialJoinOptions};
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
            spatial_predicate,
            options,
            JoinType::Inner,
            4,
            memory_pool,
            metrics,
        )
        .unwrap();

        // Test finishing with empty data
        let index = builder.finish(schema.clone()).unwrap();
        assert_eq!(index.schema(), schema);
        assert_eq!(index.indexed_batches.len(), 0);
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

        let mut builder = SpatialIndexBuilder::new(
            spatial_predicate,
            options,
            JoinType::Inner,
            4,
            memory_pool,
            metrics,
        )
        .unwrap();

        // Create a simple test geometry batch
        let schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "geom",
            DataType::Binary,
            true,
        )]));
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
        let indexed_batch = IndexedBatch {
            batch,
            geom_array: EvaluatedGeometryArray::try_new(geom_batch, &WKB_GEOMETRY).unwrap(),
        };
        builder.add_batch(indexed_batch);
        assert_eq!(builder.indexed_batches.len(), 1);

        let index = builder.finish(schema.clone()).unwrap();
        assert_eq!(index.schema(), schema);
        assert_eq!(index.indexed_batches.len(), 1);
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

        let spatial_predicate = SpatialPredicate::Relation(RelationPredicate::new(
            Arc::new(Column::new("geom", 0)),
            Arc::new(Column::new("geom", 1)),
            SpatialRelationType::Intersects,
        ));

        let mut builder = SpatialIndexBuilder::new(
            spatial_predicate,
            options,
            JoinType::Inner,
            4,
            memory_pool,
            metrics,
        )
        .unwrap();

        // Create sample geometry data - points at known locations
        let schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "geom",
            DataType::Binary,
            true,
        )]));
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

        let indexed_batch = IndexedBatch {
            batch,
            geom_array: EvaluatedGeometryArray::try_new(geom_batch, &WKB_GEOMETRY).unwrap(),
        };
        builder.add_batch(indexed_batch);

        let index = builder.finish(schema).unwrap();

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

        let spatial_predicate = SpatialPredicate::Relation(RelationPredicate::new(
            Arc::new(Column::new("geom", 0)),
            Arc::new(Column::new("geom", 1)),
            SpatialRelationType::Intersects,
        ));

        let mut builder = SpatialIndexBuilder::new(
            spatial_predicate,
            options,
            JoinType::Inner,
            4,
            memory_pool,
            metrics,
        )
        .unwrap();

        let schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "geom",
            DataType::Binary,
            true,
        )]));
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

        let indexed_batch = IndexedBatch {
            batch,
            geom_array: EvaluatedGeometryArray::try_new(geom_batch, &WKB_GEOMETRY).unwrap(),
        };
        builder.add_batch(indexed_batch);

        let index = builder.finish(schema).unwrap();

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

        let spatial_predicate = SpatialPredicate::Relation(RelationPredicate::new(
            Arc::new(Column::new("geom", 0)),
            Arc::new(Column::new("geom", 1)),
            SpatialRelationType::Intersects,
        ));

        let mut builder = SpatialIndexBuilder::new(
            spatial_predicate,
            options,
            JoinType::Inner,
            4,
            memory_pool,
            metrics,
        )
        .unwrap();

        let schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "geom",
            DataType::Binary,
            true,
        )]));
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

        let indexed_batch = IndexedBatch {
            batch,
            geom_array: EvaluatedGeometryArray::try_new(geom_batch, &WKB_GEOMETRY).unwrap(),
        };
        builder.add_batch(indexed_batch);

        let index = builder.finish(schema).unwrap();

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

        let spatial_predicate = SpatialPredicate::Relation(RelationPredicate::new(
            Arc::new(Column::new("geom", 0)),
            Arc::new(Column::new("geom", 1)),
            SpatialRelationType::Intersects,
        ));

        let mut builder = SpatialIndexBuilder::new(
            spatial_predicate,
            options,
            JoinType::Inner,
            4,
            memory_pool,
            metrics,
        )
        .unwrap();

        let schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "geom",
            DataType::Binary,
            true,
        )]));
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

        let indexed_batch = IndexedBatch {
            batch,
            geom_array: EvaluatedGeometryArray::try_new(geom_batch, &WKB_GEOMETRY).unwrap(),
        };
        builder.add_batch(indexed_batch);

        let index = builder.finish(schema).unwrap();

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

        let spatial_predicate = SpatialPredicate::Relation(RelationPredicate::new(
            Arc::new(Column::new("geom", 0)),
            Arc::new(Column::new("geom", 1)),
            SpatialRelationType::Intersects,
        ));

        let builder = SpatialIndexBuilder::new(
            spatial_predicate,
            options,
            JoinType::Inner,
            4,
            memory_pool,
            metrics,
        )
        .unwrap();

        let index = builder.finish(schema).unwrap();

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

        let spatial_predicate = SpatialPredicate::Relation(RelationPredicate::new(
            Arc::new(Column::new("geom", 0)),
            Arc::new(Column::new("geom", 1)),
            SpatialRelationType::Intersects,
        ));

        let mut builder = SpatialIndexBuilder::new(
            spatial_predicate,
            options,
            JoinType::Inner,
            1, // probe_threads_count
            memory_pool.clone(),
            metrics,
        )
        .unwrap();

        let schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "geom",
            DataType::Binary,
            true,
        )]));
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

        let indexed_batch = IndexedBatch {
            batch,
            geom_array: EvaluatedGeometryArray::try_new(geom_batch, &WKB_GEOMETRY).unwrap(),
        };
        builder.add_batch(indexed_batch);

        let index = builder.finish(schema).unwrap();

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

        let spatial_predicate = SpatialPredicate::Relation(RelationPredicate::new(
            Arc::new(Column::new("geom", 0)),
            Arc::new(Column::new("geom", 1)),
            SpatialRelationType::Intersects,
        ));

        let mut builder = SpatialIndexBuilder::new(
            spatial_predicate,
            options,
            JoinType::Inner,
            4,
            memory_pool,
            metrics,
        )
        .unwrap();

        // Create sample geometry data - points at known locations
        let schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "geom",
            DataType::Binary,
            true,
        )]));
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

        let indexed_batch = IndexedBatch {
            batch,
            geom_array: EvaluatedGeometryArray::try_new(geom_batch, &WKB_GEOMETRY).unwrap(),
        };
        builder.add_batch(indexed_batch);

        let index = builder.finish(schema).unwrap();

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

        println!("KNN Geometry test - found {} results", result.count);
        println!("Result positions: {build_positions:?}");
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

        let spatial_predicate = SpatialPredicate::Relation(RelationPredicate::new(
            Arc::new(Column::new("geom", 0)),
            Arc::new(Column::new("geom", 1)),
            SpatialRelationType::Intersects,
        ));

        let mut builder = SpatialIndexBuilder::new(
            spatial_predicate,
            options,
            JoinType::Inner,
            4,
            memory_pool,
            metrics,
        )
        .unwrap();

        // Create different geometry types
        let schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "geom",
            DataType::Binary,
            true,
        )]));
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

        let indexed_batch = IndexedBatch {
            batch,
            geom_array: EvaluatedGeometryArray::try_new(geom_batch, &WKB_GEOMETRY).unwrap(),
        };
        builder.add_batch(indexed_batch);

        let index = builder.finish(schema).unwrap();

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

        println!("KNN with mixed geometries: {build_positions:?}");

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

        let spatial_predicate = SpatialPredicate::Relation(RelationPredicate::new(
            Arc::new(Column::new("geom", 0)),
            Arc::new(Column::new("geom", 1)),
            SpatialRelationType::Intersects,
        ));

        let mut builder = SpatialIndexBuilder::new(
            spatial_predicate,
            options,
            JoinType::Inner,
            4,
            memory_pool,
            metrics,
        )
        .unwrap();

        let schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "geom",
            DataType::Binary,
            true,
        )]));
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

        let indexed_batch = IndexedBatch {
            batch,
            geom_array: EvaluatedGeometryArray::try_new(geom_batch, &WKB_GEOMETRY).unwrap(),
        };
        builder.add_batch(indexed_batch);

        let index = builder.finish(schema).unwrap();

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

        // Should return more than 2 results because of ties (all 4 points at distance sqrt(2))
        assert!(result_with_ties.count >= 2);
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

        let spatial_predicate = SpatialPredicate::Relation(RelationPredicate::new(
            Arc::new(Column::new("geom", 0)),
            Arc::new(Column::new("geom", 1)),
            SpatialRelationType::Intersects,
        ));

        let mut builder = SpatialIndexBuilder::new(
            spatial_predicate,
            options,
            JoinType::Inner,
            1, // probe_threads_count
            memory_pool.clone(),
            metrics,
        )
        .unwrap();

        // Create geometry batch using the same pattern as other tests
        let schema = Arc::new(arrow_schema::Schema::new(vec![Field::new(
            "geom",
            DataType::Binary,
            true,
        )]));
        let batch = RecordBatch::new_empty(schema.clone());

        let geom_batch = create_array(
            &[
                Some("POINT (0 0)"),
                Some("POINT (1 1)"),
                Some("POINT (2 2)"),
            ],
            &WKB_GEOMETRY,
        );
        let indexed_batch = IndexedBatch {
            batch,
            geom_array: EvaluatedGeometryArray::try_new(geom_batch, &WKB_GEOMETRY).unwrap(),
        };
        builder.add_batch(indexed_batch);

        let index = builder.finish(schema).unwrap();

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
}
