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

use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion_common::{JoinType, Result};
use sedona_common::SpatialJoinOptions;
use sedona_expr::statistics::GeoStatistics;
use sedona_spatial_join::{
    evaluated_batch::evaluated_batch_stream::SendableEvaluatedBatchStream,
    index::{
        default_spatial_index_builder::DefaultSpatialIndexBuilder,
        spatial_index_builder::SpatialJoinBuildMetrics, SpatialIndexBuilder, SpatialIndexRef,
    },
    SpatialPredicate,
};

use crate::refiner::GeographyRefinerFactory;

pub(crate) struct GeographySpatialIndexBuilder {
    inner: DefaultSpatialIndexBuilder,
}

impl GeographySpatialIndexBuilder {
    pub fn new(
        schema: SchemaRef,
        spatial_predicate: SpatialPredicate,
        options: SpatialJoinOptions,
        join_type: JoinType,
        probe_threads_count: usize,
        metrics: SpatialJoinBuildMetrics,
    ) -> Result<Self> {
        let inner = DefaultSpatialIndexBuilder::new(
            schema,
            spatial_predicate,
            options,
            join_type,
            probe_threads_count,
            metrics,
        )?;
        Ok(Self {
            inner: inner
                .with_refiner_factory(Arc::new(GeographyRefinerFactory))
                .with_wraparound((-180.0, 180.0)),
        })
    }
}

#[async_trait]
impl SpatialIndexBuilder for GeographySpatialIndexBuilder {
    async fn add_stream(
        &mut self,
        stream: SendableEvaluatedBatchStream,
        geo_statistics: GeoStatistics,
    ) -> Result<()> {
        self.inner.add_stream(stream, geo_statistics).await
    }

    fn finish(&mut self) -> Result<SpatialIndexRef> {
        self.inner.finish()
    }
}
