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

use datafusion_common::{config::ConfigExtension, extensions_options};

extensions_options! {
    /// Configuration options for GPU-accelerated spatial joins.
    pub struct GpuOptions {
        /// Enable GPU-accelerated spatial joins (requires CUDA and GPU feature flag)
        pub enable: bool, default = false

        /// Concatenate all build-side geometries into one buffer for GPU processing
        pub concat_build: bool, default = true

        /// GPU device ID to use (0 = first GPU, 1 = second, etc.)
        pub device_id: usize, default = 0

        /// Fall back to CPU if GPU initialization or execution fails
        pub fallback_to_cpu: bool, default = true

        /// Use CUDA memory pool for GPU memory management
        pub use_memory_pool: bool, default = true

        /// Percentage of total GPU memory for initializing CUDA memory pool (0-100)
        pub memory_pool_init_percentage: usize, default = 50

        /// Overlap parsing and refinement by pipelining multiple batches; 1 means no pipelining
        pub pipeline_batches: usize, default = 1

        /// Compress BVH to reduce memory usage (may reduce performance)
        pub compress_bvh: bool, default = false
    }
}

impl ConfigExtension for GpuOptions {
    const PREFIX: &'static str = "gpu";
}
