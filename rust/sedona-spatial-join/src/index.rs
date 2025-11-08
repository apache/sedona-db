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

pub(crate) mod build_side_collector;
mod knn_adapter;
pub(crate) mod spatial_index;
pub(crate) mod spatial_index_builder;

pub(crate) use build_side_collector::{
    BuildPartition, BuildSideBatchesCollector, CollectBuildSideMetrics,
};
pub(crate) use spatial_index::SpatialIndex;
pub(crate) use spatial_index_builder::{SpatialIndexBuilder, SpatialJoinBuildMetrics};
use wkb::reader::Wkb;

/// The result of a spatial index query
pub(crate) struct IndexQueryResult<'a, 'b> {
    pub wkb: &'b Wkb<'a>,
    pub distance: Option<f64>,
    pub geom_idx: usize,
    pub position: (i32, i32),
}

/// The metrics for a spatial index query
#[derive(Debug)]
pub(crate) struct QueryResultMetrics {
    pub count: usize,
    pub candidate_count: usize,
}
