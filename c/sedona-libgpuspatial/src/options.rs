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

/// Options for GPU-accelerated index and refiner.
pub struct GpuSpatialOptions {
    /// Whether to use CUDA memory pool for allocations
    pub cuda_use_memory_pool: bool,
    /// Ratio of initial memory pool size to total GPU memory, between 0 and 100
    pub cuda_memory_pool_init_percent: i32,
    /// How many threads will concurrently use the library
    pub concurrency: u32,
    /// The device id to use
    pub device_id: i32,
    /// Whether to build a compressed BVH, which can reduce memory usage, but may increase build time
    pub compress_bvh: bool,
    /// The number of batches for pipelined refinement that overlaps the WKB loading and refinement. Setting 1 effectively disables pipelining.
    pub pipeline_batches: u32,
}
