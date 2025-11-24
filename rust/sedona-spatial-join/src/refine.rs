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
use std::sync::Arc;

use datafusion_common::Result;
use sedona_common::{ExecutionMode, SpatialJoinOptions, SpatialLibrary};
use sedona_expr::statistics::GeoStatistics;
use wkb::reader::Wkb;

use crate::{index::IndexQueryResult, spatial_predicate::SpatialPredicate};

/// Trait for refining spatial index query results by evaluating exact geometric predicates.
///
/// This trait represents the second phase of the two-phase spatial join algorithm:
/// 1. **Filter phase**: R-tree index identifies candidate geometries based on bounding rectangles
/// 2. **Refinement phase**: This trait evaluates exact spatial predicates on candidates
///
/// The refinement phase eliminates false positives from the filter phase, ensuring that only
/// geometries that truly satisfy the spatial predicate are returned. Different spatial libraries
/// (Geo, GEOS, TG) provide their own implementations with varying performance characteristics
/// and geometric predicate support.
pub(crate) trait IndexQueryResultRefiner: Send + Sync {
    /// Refine index query results by evaluating the exact spatial predicate.
    ///
    /// Takes a probe geometry and a list of candidate build-side geometries from the R-tree
    /// index, then evaluates the configured spatial predicate (e.g., intersects, contains,
    /// within, distance) to determine which candidates are actual matches.
    ///
    /// # Arguments
    /// * `probe` - The probe geometry in WKB format to test against candidates
    /// * `index_query_results` - Candidate geometries from the R-tree filter phase, containing
    ///   WKB data, optional distance parameters, and position information
    ///
    /// # Returns
    /// * `Vec<(i32, i32)>` - Vector of (batch_index, row_index) pairs for geometries that
    ///   satisfy the spatial predicate
    ///
    /// # Performance
    /// This method may use prepared geometries or other optimizations based on the execution mode.
    /// The implementation should handle empty geometries gracefully by skipping them rather than
    /// failing.
    fn refine(
        &self,
        probe: &Wkb<'_>,
        index_query_results: &[IndexQueryResult],
    ) -> Result<Vec<(i32, i32)>>;

    /// Get the current memory usage of the refiner in bytes.
    ///
    /// Used for memory tracking and reservation management. Implementations should account for
    /// prepared geometry caches, internal data structures, and temporary computation buffers.
    ///
    /// # Returns
    /// * `usize` - Current memory usage in bytes
    fn mem_usage(&self) -> usize;

    /// Get the actual execution mode used by the refiner.
    ///
    /// # Returns
    /// * `ExecutionMode` - The actual execution mode used by the refiner
    fn actual_execution_mode(&self) -> ExecutionMode;

    /// Check if the refiner needs more probe statistics to determine the optimal execution mode.
    ///
    /// # Returns
    /// * `bool` - `true` if the refiner needs more probe statistics, `false` otherwise.
    fn need_more_probe_stats(&self) -> bool;

    /// Merge the probe statistics into the refiner.
    ///
    /// # Arguments
    /// * `stats` - The probe statistics to merge.
    fn merge_probe_stats(&self, stats: GeoStatistics);
}

mod exec_mode_selector;
pub mod geo;
pub mod geos;
pub mod tg;

/// Create a spatial predicate refiner for the specified geometry library.
///
/// This factory function instantiates the appropriate refiner implementation based on the
/// selected spatial library backend. Each library provides different trade-offs in terms
/// of performance, memory usage, and geometric predicate support.
///
/// # Arguments
/// * `library` - The spatial library backend to use for geometric computations
/// * `predicate` - The spatial predicate to evaluate (e.g., intersects, contains, distance)
/// * `options` - Configuration options including execution mode and optimization settings
/// * `num_build_geoms` - Total number of build-side geometries, used to size prepared geometry
///   caches when using preparation-based execution modes
/// * `build_stats` - Statistics for the build-side geometries, used to optimize the refine
///   process.
///
/// # Returns
/// * `Arc<dyn IndexQueryResultRefiner>` - Thread-safe refiner implementation for the specified library
pub(crate) fn create_refiner(
    library: SpatialLibrary,
    predicate: &SpatialPredicate,
    options: SpatialJoinOptions,
    num_build_geoms: usize,
    build_stats: GeoStatistics,
) -> Arc<dyn IndexQueryResultRefiner> {
    match library {
        SpatialLibrary::Geo => Arc::new(geo::GeoRefiner::new(predicate, options, build_stats)),
        SpatialLibrary::Geos => Arc::new(geos::GeosRefiner::new(
            predicate,
            options,
            num_build_geoms,
            build_stats,
        )),
        SpatialLibrary::Tg => {
            match tg::TgRefiner::try_new(
                predicate,
                options.clone(),
                num_build_geoms,
                build_stats.clone(),
            ) {
                Ok(refiner) => Arc::new(refiner),
                Err(_) => {
                    // TG does not support all spatial predicates. Fallback to Geo if TG fails to initialize
                    Arc::new(geo::GeoRefiner::new(predicate, options, build_stats))
                }
            }
        }
    }
}
