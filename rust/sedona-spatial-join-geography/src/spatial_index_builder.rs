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

//! Geography-aware spatial index builder.
//!
//! This module provides a spatial index builder that wraps the default builder
//! and produces geography-aware spatial indexes.

use std::sync::Arc;

use async_trait::async_trait;
use datafusion_common::Result;
use sedona_expr::statistics::GeoStatistics;
use sedona_spatial_join::{
    evaluated_batch::evaluated_batch_stream::SendableEvaluatedBatchStream,
    index::{SpatialIndexBuilder, SpatialIndexRef},
    SpatialPredicate,
};

use crate::refiner::GeographyRefiner;
use crate::spatial_index::GeographySpatialIndex;

/// Builder for creating geography-aware spatial indexes.
///
/// This builder wraps a default spatial index builder and produces
/// [`GeographySpatialIndex`] instances that apply geography-specific
/// refinement during spatial join queries.
///
/// # Architecture
///
/// The builder follows a delegation pattern:
/// 1. `add_stream()` is delegated to the inner builder to collect geometry batches
/// 2. `finish()` creates the inner spatial index, then wraps it with a geography refiner
///
/// # Usage
///
/// ```ignore
/// let builder = GeographySpatialIndexBuilder::new(
///     inner_builder,
///     predicate,
/// )?;
///
/// builder.add_stream(stream, stats).await?;
/// let index = builder.finish()?;
/// ```
pub struct GeographySpatialIndexBuilder {
    /// The wrapped default spatial index builder
    inner: Box<dyn SpatialIndexBuilder>,
    /// The spatial predicate for this join
    predicate: SpatialPredicate,
    /// Accumulated build-side statistics
    build_stats: GeoStatistics,
}

impl GeographySpatialIndexBuilder {
    /// Create a new geography spatial index builder.
    ///
    /// # Arguments
    /// * `inner` - The underlying spatial index builder to wrap
    /// * `predicate` - The spatial predicate for this join
    pub fn new(inner: Box<dyn SpatialIndexBuilder>, predicate: SpatialPredicate) -> Self {
        Self {
            inner,
            predicate,
            build_stats: GeoStatistics::empty(),
        }
    }

    /// Get a reference to the spatial predicate.
    pub fn predicate(&self) -> &SpatialPredicate {
        &self.predicate
    }
}

#[async_trait]
impl SpatialIndexBuilder for GeographySpatialIndexBuilder {
    async fn add_stream(
        &mut self,
        stream: SendableEvaluatedBatchStream,
        geo_statistics: GeoStatistics,
    ) -> Result<()> {
        // Accumulate statistics for the geography refiner
        self.build_stats.merge(&geo_statistics);
        // Delegate batch collection to the inner builder
        self.inner.add_stream(stream, geo_statistics).await
    }

    fn finish(&mut self) -> Result<SpatialIndexRef> {
        // Build the inner spatial index
        let inner_index = self.inner.finish()?;

        // Create the geography refiner
        let refiner = Arc::new(GeographyRefiner::new(
            self.predicate.clone(),
            self.build_stats.clone(),
        ));

        // Wrap the inner index with geography-specific refinement
        let geography_index = GeographySpatialIndex::new(inner_index, refiner);

        Ok(Arc::new(geography_index))
    }
}
