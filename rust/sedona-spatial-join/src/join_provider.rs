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

use arrow_schema::SchemaRef;
use datafusion_common::Result;
use datafusion_expr::JoinType;
use sedona_common::SpatialJoinOptions;
use sedona_expr::statistics::GeoStatistics;

use crate::{
    index::{
        spatial_index_builder::{SpatialIndexBuilder, SpatialJoinBuildMetrics},
        DefaultSpatialIndexBuilder,
    },
    operand_evaluator::{DefaultGeometryArrayFactory, EvaluatedGeometryArrayFactory},
    refine::DefaultIndexQueryResultRefinerFactory,
    SpatialPredicate,
};

/// Provider for join internals
///
/// This trait provides an extension point for overriding the evaluation
/// details of a spatial join. In particular it allows plugging in a custom
/// index for accelerated joins on specific hardware (e.g., GPU) and a custom
/// bounder for specific data types (e.g., geography).
pub trait SpatialJoinProvider: std::fmt::Debug + Send + Sync {
    /// Create a new [SpatialIndexBuilder]
    fn try_new_spatial_index_builder(
        &self,
        schema: SchemaRef,
        spatial_predicate: SpatialPredicate,
        options: SpatialJoinOptions,
        join_type: JoinType,
        probe_threads_count: usize,
        metrics: SpatialJoinBuildMetrics,
    ) -> Result<Box<dyn SpatialIndexBuilder>>;

    /// Estimate the amount of memory required by the index and refiner evaluating spatial predicates
    ///
    /// The estimated memory usage should not include the memory required for holding the build side
    /// batches.
    fn estimate_extra_memory_usage(
        &self,
        geo_stats: &GeoStatistics,
        spatial_predicate: &SpatialPredicate,
        options: &SpatialJoinOptions,
    ) -> usize;

    fn evaluated_array_factory(&self) -> Arc<dyn EvaluatedGeometryArrayFactory>;
}

/// Default implementation of the [SpatialJoinProvider]
#[derive(Debug)]
pub struct DefaultSpatialJoinProvider;

impl SpatialJoinProvider for DefaultSpatialJoinProvider {
    fn try_new_spatial_index_builder(
        &self,
        schema: SchemaRef,
        spatial_predicate: SpatialPredicate,
        options: SpatialJoinOptions,
        join_type: JoinType,
        probe_threads_count: usize,
        metrics: SpatialJoinBuildMetrics,
    ) -> Result<Box<dyn SpatialIndexBuilder>> {
        let builder = DefaultSpatialIndexBuilder::new(
            schema,
            spatial_predicate,
            options,
            join_type,
            probe_threads_count,
            metrics,
        )?;
        Ok(Box::new(builder))
    }

    fn estimate_extra_memory_usage(
        &self,
        geo_stats: &GeoStatistics,
        spatial_predicate: &SpatialPredicate,
        options: &SpatialJoinOptions,
    ) -> usize {
        DefaultSpatialIndexBuilder::estimate_extra_memory_usage(
            geo_stats,
            spatial_predicate,
            options,
            Arc::new(DefaultIndexQueryResultRefinerFactory),
        )
    }

    fn evaluated_array_factory(&self) -> Arc<dyn EvaluatedGeometryArrayFactory> {
        Arc::new(DefaultGeometryArrayFactory)
    }
}
