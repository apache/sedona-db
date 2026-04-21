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

use datafusion_physical_plan::metrics::{self, ExecutionPlanMetricsSet, MetricBuilder};
use sedona_expr::statistics::GeoStatistics;

use crate::evaluated_batch::evaluated_batch_stream::SendableEvaluatedBatchStream;
use crate::index::spatial_index::SpatialIndexRef;
use crate::IndexQueryResultRefiner;
use async_trait::async_trait;
use datafusion_common::Result;

/// Builder for constructing a SpatialIndex from geometry batches.
#[async_trait]
pub trait SpatialIndexBuilder: Send + Sync {
    /// Add a stream to this builder
    async fn add_stream(
        &mut self,
        stream: SendableEvaluatedBatchStream,
        geo_statistics: GeoStatistics,
    ) -> Result<()>;

    fn create_refiner(&self) -> Result<Option<Arc<dyn IndexQueryResultRefiner>>> {
        Ok(None)
    }

    /// Finish building and return the completed SpatialIndex.
    fn finish(&mut self) -> Result<SpatialIndexRef>;
}

/// Metrics for the build phase of the spatial join.
#[derive(Clone, Debug, Default)]
pub struct SpatialJoinBuildMetrics {
    /// Total time for collecting build-side of join
    pub build_time: metrics::Time,
    /// Memory used by the spatial-index in bytes
    pub build_mem_used: metrics::Gauge,
}

impl SpatialJoinBuildMetrics {
    pub fn new(partition: usize, metrics: &ExecutionPlanMetricsSet) -> Self {
        Self {
            build_time: MetricBuilder::new(metrics).subset_time("build_time", partition),
            build_mem_used: MetricBuilder::new(metrics).gauge("build_mem_used", partition),
        }
    }
}
