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
// Module declarations
mod build_data;
pub mod config;
pub mod exec;
pub mod gpu_backend;
pub(crate) mod once_fut;
pub mod stream;

// Re-exports for convenience
pub use config::{GeometryColumnInfo, GpuSpatialJoinConfig, GpuSpatialPredicate};
pub use datafusion::logical_expr::JoinType;
pub use exec::GpuSpatialJoinExec;
pub use sedona_libgpuspatial::SpatialPredicate;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("GPU initialization error: {0}")]
    GpuInit(String),

    #[error("DataFusion error: {0}")]
    DataFusion(#[from] datafusion::error::DataFusionError),

    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("GPU spatial operation error: {0}")]
    GpuSpatial(String),
}

pub type Result<T> = std::result::Result<T, Error>;
