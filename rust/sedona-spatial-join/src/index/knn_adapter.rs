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

use geo::{Centroid, Distance, Euclidean, Haversine};
use geo_types::Geometry;
use sedona_expr::statistics::GeoStatistics;
use sedona_geo::to_geo::item_to_geometry;

use crate::evaluated_batch::EvaluatedBatch;

/// Shared KNN components that can be reused across queries
pub(crate) struct KnnComponents {
    /// Pre-allocated vector for geometry cache - lock-free access
    /// Indexed by rtree data index for O(1) access
    geometry_cache: Vec<OnceCell<Geometry<f64>>>,
    /// Estimated memory usage for decoded geometries
    estimated_memory_usage: usize,
}

/// Heap bytes of the eagerly-allocated `geometry_cache` backbone: one `OnceCell`
/// slot per indexed geometry, allocated up front in [`KnnComponents::new`]. Each
/// slot reserves inline space for the decoded `Geometry`, so this term is exact
/// and independent of how many geometries are later decoded on demand. The
/// nested coordinate buffers of decoded geometries are allocated lazily and are
/// not counted here.
fn cache_backbone_bytes(cache_size: usize) -> usize {
    cache_size * std::mem::size_of::<OnceCell<Geometry<f64>>>()
}

impl KnnComponents {
    pub fn new(
        cache_size: usize,
        indexed_batches: &[EvaluatedBatch],
    ) -> datafusion_common::Result<Self> {
        // Pre-allocate OnceCell vector
        let geometry_cache = (0..cache_size).map(|_| OnceCell::new()).collect();
        let mut total_wkb_size = 0;
        for batch in indexed_batches {
            for wkb in batch.geom_array.wkbs().iter().flatten() {
                total_wkb_size += wkb.buf().len();
            }
        }

        Ok(Self {
            geometry_cache,
            estimated_memory_usage: total_wkb_size + cache_backbone_bytes(cache_size),
        })
    }

    /// Estimate the maximum memory usage for decoded geometries based on statistics
    pub fn estimate_max_memory_usage(build_stats: &GeoStatistics) -> usize {
        let geom_count = build_stats.total_geometries().unwrap_or(0) as usize;
        build_stats.total_size_bytes().unwrap_or(0) as usize + cache_backbone_bytes(geom_count)
    }

    pub fn estimated_memory_usage(&self) -> usize {
        self.estimated_memory_usage
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

    /// Get geometry for the given item index with lock-free caching
    pub fn get_geometry(&self, item_index: usize) -> Option<&Geometry<f64>> {
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

        if let Some(wkb) = indexed_batch.geom_array.wkb(row_idx as usize)
            && let Ok(geom) = item_to_geometry(wkb)
        {
            // Try to store in cache - if another thread got there first, we just use theirs
            let _ = geometry_cache[item_index].set(geom);
            // Return reference to the cached geometry
            return geometry_cache[item_index].get();
        }

        // Failed to decode - don't cache invalid results
        None
    }

    pub fn distance(
        &self,
        probe: &Geometry<f64>,
        item_index: usize,
        use_spheroid: bool,
    ) -> Option<f64> {
        let item = self.get_geometry(item_index)?;
        if use_spheroid {
            let probe_centroid = probe.centroid()?;
            let item_centroid = item.centroid()?;
            Some(Haversine.distance(probe_centroid, item_centroid))
        } else {
            Some(Euclidean.distance(probe, item))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimated_memory_usage_includes_cache_backbone() {
        let slot = std::mem::size_of::<OnceCell<Geometry<f64>>>();

        // Empty cache: no slots and no decoded geometries.
        let empty = KnnComponents::new(0, &[]).unwrap();
        assert_eq!(empty.estimated_memory_usage(), 0);

        // With N slots and no build batches, the estimate is exactly the eager
        // backbone (one OnceCell slot per indexed geometry), with no WKB bytes.
        let n = 1000;
        let components = KnnComponents::new(n, &[]).unwrap();
        assert_eq!(components.estimated_memory_usage(), n * slot);
    }

    #[test]
    fn estimate_max_memory_usage_includes_cache_backbone() {
        let slot = std::mem::size_of::<OnceCell<Geometry<f64>>>();
        let stats = GeoStatistics::empty()
            .with_total_geometries(1000)
            .with_total_size_bytes(4096);
        assert_eq!(
            KnnComponents::estimate_max_memory_usage(&stats),
            4096 + 1000 * slot
        );
    }
}
