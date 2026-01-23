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

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion_common::Result;
use geo_types::Rect;
use parking_lot::Mutex;
use sedona_expr::statistics::GeoStatistics;
use std::{ops::Range, sync::Arc};
use wkb::reader::Wkb;

use crate::{evaluated_batch::EvaluatedBatch, index::QueryResultMetrics};
use arrow::array::BooleanBufferBuilder;
use datafusion_physical_plan::metrics;
use datafusion_physical_plan::metrics::{ExecutionPlanMetricsSet, MetricBuilder};
use sedona_common::ExecutionMode;

/// Metrics for the build phase of the spatial join.
#[derive(Clone, Debug, Default)]
pub struct SpatialJoinBuildMetrics {
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

#[async_trait]
pub(crate) trait SpatialIndex {
    fn schema(&self) -> SchemaRef;

    /// Get all the indexed batches.
    #[allow(dead_code)] // used in tests
    fn get_num_indexed_batches(&self) -> usize;

    /// Get the batch at the given index.
    fn get_indexed_batch(&self, batch_idx: usize) -> &RecordBatch;
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
    #[allow(dead_code)] // for future use
    fn query(
        &self,
        probe_wkb: &Wkb,
        probe_rect: &Rect<f32>,
        distance: &Option<f64>,
        build_batch_positions: &mut Vec<(i32, i32)>,
    ) -> Result<QueryResultMetrics>;

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
    fn query_knn(
        &self,
        probe_wkb: &Wkb,
        k: u32,
        use_spheroid: bool,
        include_tie_breakers: bool,
        build_batch_positions: &mut Vec<(i32, i32)>,
    ) -> Result<QueryResultMetrics>;

    /// Query the spatial index with a batch of probe geometries to find matching build-side geometries.
    ///
    /// This method iterates over the probe geometries in the given range of the evaluated batch.
    /// For each probe geometry, it performs the two-phase spatial join query:
    /// 1. **Filter phase**: Uses the R-tree index with the probe geometry's bounding rectangle
    ///    to quickly identify candidate geometries.
    /// 2. **Refinement phase**: Evaluates the exact spatial predicate on candidates to determine
    ///    actual matches.
    ///
    /// # Arguments
    /// * `evaluated_batch` - The batch containing probe geometries and their bounding rectangles
    /// * `range` - The range of rows in the evaluated batch to process.
    /// * `max_result_size` - The maximum number of results to collect before stopping. If the
    ///   number of results exceeds this limit, the method returns early.
    /// * `build_batch_positions` - Output vector that will be populated with (batch_idx, row_idx)
    ///   pairs for each matching build-side geometry.
    /// * `probe_indices` - Output vector that will be populated with the probe row index (in
    ///   `evaluated_batch`) for each match appended to `build_batch_positions`.
    ///   This means the probe index is repeated `N` times when a probe geometry produces `N` matches,
    ///   keeping `probe_indices.len()` in sync with `build_batch_positions.len()`. `probe_indices` should be sorted in **ascending order**.
    ///
    /// # Returns
    /// * A tuple containing:
    ///   - `QueryResultMetrics`: Aggregated metrics (total matches and candidates) for the processed rows
    ///   - `usize`: The index of the next row to process (exclusive end of the processed range)
    async fn query_batch(
        &self,
        evaluated_batch: &Arc<EvaluatedBatch>,
        range: Range<usize>,
        max_result_size: usize,
        build_batch_positions: &mut Vec<(i32, i32)>,
        probe_indices: &mut Vec<u32>,
    ) -> Result<(QueryResultMetrics, usize)>;

    /// Check if the index needs more probe statistics to determine the optimal execution mode.
    ///
    /// # Returns
    /// * `bool` - `true` if the index needs more probe statistics, `false` otherwise.
    fn need_more_probe_stats(&self) -> bool;
    /// Merge the probe statistics into the index.
    ///
    /// # Arguments
    /// * `stats` - The probe statistics to merge.
    fn merge_probe_stats(&self, stats: GeoStatistics);

    /// Get the bitmaps for tracking visited build-side indices. The bitmaps will be updated
    /// by the spatial join stream when producing output batches during index probing phase.
    fn visited_build_side(&self) -> Option<&Mutex<Vec<BooleanBufferBuilder>>>;
    /// Decrements counter of running threads, and returns `true`
    /// if caller is the last running thread
    fn report_probe_completed(&self) -> bool;
    /// Get the memory usage of the refiner in bytes.
    fn get_refiner_mem_usage(&self) -> usize;
    /// Get the actual execution mode used by the refiner
    fn get_actual_execution_mode(&self) -> ExecutionMode;
}

pub type SpatialIndexRef = Arc<dyn SpatialIndex + Send + Sync>;
