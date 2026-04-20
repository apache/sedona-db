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

use arrow_array::ArrayRef;
use datafusion_common::{JoinType, Result};
use sedona_expr::statistics::GeoStatistics;
use sedona_schema::datatypes::SedonaType;
use sedona_spatial_join::{
    index::{spatial_index_builder::SpatialJoinBuildMetrics, SpatialIndexBuilder},
    join_provider::{DefaultSpatialJoinProvider, SpatialJoinProvider},
    operand_evaluator::{EvaluatedGeometryArray, EvaluatedGeometryArrayFactory},
    SpatialJoinOptions, SpatialPredicate,
};

#[derive(Debug)]
pub struct GeographyJoinProvider {
    inner: DefaultSpatialJoinProvider,
}

impl GeographyJoinProvider {
    pub fn new() -> Self {
        GeographyJoinProvider {
            inner: DefaultSpatialJoinProvider,
        }
    }
}

impl Default for GeographyJoinProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl SpatialJoinProvider for GeographyJoinProvider {
    fn try_new_spatial_index_builder(
        &self,
        schema: arrow_schema::SchemaRef,
        spatial_predicate: SpatialPredicate,
        options: SpatialJoinOptions,
        join_type: JoinType,
        probe_threads_count: usize,
        metrics: SpatialJoinBuildMetrics,
    ) -> Result<Box<dyn SpatialIndexBuilder>> {
        self.inner.try_new_spatial_index_builder(
            schema,
            spatial_predicate,
            options,
            join_type,
            probe_threads_count,
            metrics,
        )
    }

    fn estimate_extra_memory_usage(
        &self,
        _geo_stats: &GeoStatistics,
        _spatial_predicate: &SpatialPredicate,
        _options: &SpatialJoinOptions,
    ) -> usize {
        // TODO: calculate
        0
    }

    fn evaluated_array_factory(&self) -> Arc<dyn EvaluatedGeometryArrayFactory> {
        Arc::new(GeographyEvaluatedArrayFactory)
    }
}

#[derive(Debug)]
struct GeographyEvaluatedArrayFactory;

impl EvaluatedGeometryArrayFactory for GeographyEvaluatedArrayFactory {
    fn try_new_evaluated_array(
        &self,
        geometry_array: ArrayRef,
        sedona_type: &SedonaType,
    ) -> Result<EvaluatedGeometryArray> {
        // compute rectangles from wkb using the RectBounder from s2geography

        EvaluatedGeometryArray::try_new(geometry_array, sedona_type)
    }
}
