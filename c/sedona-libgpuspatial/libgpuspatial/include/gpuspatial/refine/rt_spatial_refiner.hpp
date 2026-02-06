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
#pragma once

#include "gpuspatial/refine/spatial_refiner.hpp"
#include "gpuspatial/rt/rt_engine.hpp"

#include <memory>

namespace gpuspatial {
/** Configuration for RTSpatialRefiner for initializing a RTSpatialRefiner instance
 */
struct RTSpatialRefinerConfig {
  // The ray-tracing engine to use
  std::shared_ptr<RTEngine> rt_engine;
  // Prefer fast build the BVH
  bool prefer_fast_build = false;
  // Compress the BVH to save memory
  bool compact = true;
  // Loader configurations
  // How many threads to use for parsing WKBs
  uint32_t parsing_threads = std::thread::hardware_concurrency();
  // How many threads are allowed to call PushStream concurrently
  uint32_t concurrency = 1;
  // Overlapping parsing and refinement by pipelining multiple batches; 1 means no
  // pipelining
  uint32_t pipeline_batches = 1;
  // the host memory quota for WKB parser compared to the available memory
  float wkb_parser_memory_quota = 0.8;
  // the device memory quota for relate engine compared to the available memory
  float relate_engine_memory_quota = 0.8;
  // this value determines RELATE_MAX_DEPTH
  size_t stack_size_bytes = 3 * 1024;
  bool sort_probe_indices = true;  // Sedona's spatial-join may require ordered output
};

std::unique_ptr<SpatialRefiner> CreateRTSpatialRefiner(
    const RTSpatialRefinerConfig& config);

}  // namespace gpuspatial
