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
use std::sync::Arc;

use datafusion_execution::memory_pool::{MemoryConsumer, MemoryPool, MemoryReservation};
use geo_index::rtree::{distance::GeometryAccessor, EuclideanDistance, HaversineDistance};
use geo_types::Geometry;
use sedona_geo::to_geo::item_to_geometry;

use crate::evaluated_batch::EvaluatedBatch;

/// Shared KNN components that can be reused across queries
pub(crate) struct KnnComponents {
    pub euclidean_metric: EuclideanDistance,
    pub haversine_metric: HaversineDistance,
    /// Pre-allocated vector for geometry cache - lock-free access
    /// Indexed by rtree data index for O(1) access
    geometry_cache: Vec<OnceCell<Geometry<f64>>>,
    /// Memory reservation to track geometry cache memory usage
    _reservation: MemoryReservation,
}

impl KnnComponents {
    pub fn new(
        cache_size: usize,
        indexed_batches: &[EvaluatedBatch],
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
    pub fn estimate_max_memory_usage(indexed_batches: &[EvaluatedBatch]) -> usize {
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
pub(crate) struct SedonaKnnAdapter<'a> {
    indexed_batches: &'a [EvaluatedBatch],
    data_id_to_batch_pos: &'a [(i32, i32)],
    // Reference to KNN components for cache and memory tracking
    knn_components: &'a KnnComponents,
}

impl<'a> SedonaKnnAdapter<'a> {
    /// Create a new adapter
    pub fn new(
        indexed_batches: &'a [EvaluatedBatch],
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
