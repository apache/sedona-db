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

//! Geography-aware spatial index wrapper.
//!
//! This module provides a spatial index implementation that wraps the default
//! spatial index and applies geography-specific refinement using s2geography.

use std::ops::Range;
use std::sync::Arc;

use arrow::array::BooleanBufferBuilder;
use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion_common::Result;
use parking_lot::Mutex;
use sedona_common::ExecutionMode;
use sedona_expr::statistics::GeoStatistics;
use sedona_spatial_join::{
    evaluated_batch::EvaluatedBatch,
    index::{QueryResultMetrics, SpatialIndex, SpatialIndexRef},
};
use wkb::reader::Wkb;

use crate::refiner::GeographyRefinerRef;

/// A spatial index wrapper that applies geography-specific refinement.
///
/// This wrapper delegates most operations to an inner spatial index while
/// intercepting query operations to apply geography-specific predicate evaluation.
///
/// # Architecture
///
/// The wrapper follows a filter-and-refine pattern:
/// 1. The inner spatial index provides initial filtering using its R-tree
/// 2. The geography refiner applies exact spatial predicate evaluation using s2geography
///
/// # Current Status
///
/// The wrapper stores a geography-specific refiner that implements `IndexQueryResultRefiner`.
/// Currently, the query operations still delegate to the inner index (which applies Cartesian
/// refinement). The next step is to intercept the R-tree candidates and apply the geography
/// refiner instead.
#[derive(Clone)]
pub struct GeographySpatialIndex {
    /// The wrapped spatial index
    inner: SpatialIndexRef,
    /// Geography-specific refiner for predicate evaluation
    refiner: GeographyRefinerRef,
}

impl GeographySpatialIndex {
    /// Create a new geography spatial index wrapper.
    ///
    /// # Arguments
    /// * `inner` - The underlying spatial index to wrap
    /// * `refiner` - The geography refiner for predicate evaluation
    pub fn new(inner: SpatialIndexRef, refiner: GeographyRefinerRef) -> Self {
        Self { inner, refiner }
    }

    /// Get a reference to the wrapped spatial index.
    pub fn inner(&self) -> &SpatialIndexRef {
        &self.inner
    }

    /// Get a reference to the geography refiner.
    pub fn refiner(&self) -> &GeographyRefinerRef {
        &self.refiner
    }
}

#[async_trait]
impl SpatialIndex for GeographySpatialIndex {
    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }

    fn num_indexed_batches(&self) -> usize {
        self.inner.num_indexed_batches()
    }

    fn get_indexed_batch(&self, batch_idx: usize) -> &RecordBatch {
        self.inner.get_indexed_batch(batch_idx)
    }

    fn query_knn(
        &self,
        probe_wkb: &Wkb,
        k: u32,
        use_spheroid: bool,
        include_tie_breakers: bool,
        build_batch_positions: &mut Vec<(i32, i32)>,
        distances: Option<&mut Vec<f64>>,
    ) -> Result<QueryResultMetrics> {
        // TODO: Implement geography-specific KNN using s2geography distance calculations
        //
        // The proper implementation should:
        // 1. Use the R-tree to get initial candidates
        // 2. Calculate spherical distances using s2geography
        // 3. Return the k nearest neighbors based on spherical distance
        //
        // For now, we delegate to the inner index which uses Cartesian distance.
        // This is incorrect for geography types but serves as a placeholder.
        log::warn!("GeographySpatialIndex::query_knn currently uses Cartesian distance - geography distance not yet implemented");
        self.inner.query_knn(
            probe_wkb,
            k,
            use_spheroid,
            include_tie_breakers,
            build_batch_positions,
            distances,
        )
    }

    async fn query_batch(
        &self,
        evaluated_batch: &Arc<EvaluatedBatch>,
        range: Range<usize>,
        max_result_size: usize,
        build_batch_positions: &mut Vec<(i32, i32)>,
        probe_indices: &mut Vec<u32>,
    ) -> Result<(QueryResultMetrics, usize)> {
        // TODO: Implement geography-specific refinement
        //
        // The proper implementation should:
        // 1. Get R-tree candidates from the inner index (without refinement)
        // 2. Apply geography-specific predicate evaluation using s2geography
        //
        // Currently, the inner index applies its own (Cartesian) refinement.
        // This may produce incorrect results for geography types because:
        // - Cartesian predicates don't account for spherical geometry
        // - Some valid geography matches might be incorrectly filtered out
        //
        // To properly implement this, we need either:
        // - The IndexQueryResultRefiner trait to be publicly exposed
        // - A way to configure the inner index to skip refinement
        //
        // For now, we delegate to the inner index as a placeholder.
        log::warn!("GeographySpatialIndex::query_batch currently uses Cartesian refinement - geography refinement not yet implemented");
        self.inner
            .query_batch(
                evaluated_batch,
                range,
                max_result_size,
                build_batch_positions,
                probe_indices,
            )
            .await
    }

    fn need_more_probe_stats(&self) -> bool {
        // Use our refiner's stats needs
        self.refiner.need_more_probe_stats() || self.inner.need_more_probe_stats()
    }

    fn merge_probe_stats(&self, stats: GeoStatistics) {
        self.refiner.merge_probe_stats(stats.clone());
        self.inner.merge_probe_stats(stats);
    }

    fn visited_build_side(&self) -> Option<&Mutex<Vec<BooleanBufferBuilder>>> {
        self.inner.visited_build_side()
    }

    fn report_probe_completed(&self) -> bool {
        self.inner.report_probe_completed()
    }

    fn get_refiner_mem_usage(&self) -> usize {
        // Report our geography refiner's memory usage
        self.refiner.mem_usage() + self.inner.get_refiner_mem_usage()
    }

    fn get_actual_execution_mode(&self) -> ExecutionMode {
        // Return our refiner's execution mode
        self.refiner.actual_execution_mode()
    }
}
