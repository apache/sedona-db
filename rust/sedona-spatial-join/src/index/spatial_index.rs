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
use crate::index::QueryResultMetrics;
use arrow::array::BooleanBufferBuilder;
use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion_common::Result;
use parking_lot::Mutex;
use sedona_common::ExecutionMode;
use sedona_expr::statistics::GeoStatistics;
use std::ops::Range;
use std::sync::Arc;
use wkb::reader::Wkb;

pub const DISTANCE_TOLERANCE: f64 = 1e-9;

/// The `SpatialIndex` trait defines the interface for spatial indexes used in spatial join operations.
/// It provides methods for querying the index with spatial predicates,
/// as well as methods for managing probe statistics and tracking visited build side batches.
/// The trait is designed to be implemented by various spatial index structures
#[async_trait]
pub(crate) trait SpatialIndex {
    /// Returns the schema of the indexed data.
    fn schema(&self) -> SchemaRef;
    /// Returns the number of batches that have been indexed.
    #[cfg(test)] // This is used for tests
    fn num_indexed_batches(&self) -> usize;
    /// Get the batch at the given index.
    fn get_indexed_batch(&self, batch_idx: usize) -> &RecordBatch;
    /// Query the spatial index for k nearest neighbors of a given geometry.
    /// # Arguments
    ///
    /// * `probe_wkb` - WKB representation of the probe geometry
    /// * `k` - Number of nearest neighbors to find
    /// * `use_spheroid` - Whether to use spheroid distance calculation
    /// * `include_tie_breakers` - Whether to include additional results with same distance as kth neighbor
    /// * `build_batch_positions` - Output vector for matched positions
    /// * `distances` - Optional output vector for distances to matched neighbors, aligned with `build_batch_positions`
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
        distances: Option<&mut Vec<f64>>,
    ) -> Result<QueryResultMetrics>;
    /// Query the spatial index with a batch of probe geometries to find matching build-side geometries.
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
    ///   keeping `probe_indices.len()` in sync with `build_batch_positions.len()`.
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
    /// Get the bitmaps for tracking visited left-side indices. The bitmaps will be updated
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
