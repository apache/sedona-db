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
use crate::config::GpuSpatialJoinConfig;
use arrow_array::RecordBatch;

/// Shared build-side data for GPU spatial join
#[derive(Clone)]
pub(crate) struct GpuBuildData {
    /// All left-side data concatenated into single batch
    pub(crate) left_batch: RecordBatch,

    /// Configuration (includes geometry column indices, predicate, etc)
    pub(crate) config: GpuSpatialJoinConfig,

    /// Total rows in left batch
    pub(crate) left_row_count: usize,
}

impl GpuBuildData {
    pub fn new(left_batch: RecordBatch, config: GpuSpatialJoinConfig) -> Self {
        let left_row_count = left_batch.num_rows();
        Self {
            left_batch,
            config,
            left_row_count,
        }
    }

    pub fn left_batch(&self) -> &RecordBatch {
        &self.left_batch
    }

    pub fn config(&self) -> &GpuSpatialJoinConfig {
        &self.config
    }
}
