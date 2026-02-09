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

#include "gpuspatial/index/spatial_index.hpp"
#include "gpuspatial/rt/rt_engine.hpp"

#include <memory>
#include <thread>

namespace gpuspatial {
/** Configuration for RTSpatialIndex
 */
struct RTSpatialIndexConfig {
  // The ray tracing engine to use
  std::shared_ptr<RTEngine> rt_engine;
  // Prefer fast build the BVH
  bool prefer_fast_build = false;
  // Compress the BVH to save memory
  bool compact = false;
  // How many threads are allowed to call PushProbe concurrently
  uint32_t concurrency = 1;
  // number of points to represent an AABB when doing point-point queries
  uint32_t n_points_per_aabb = 8;
  RTSpatialIndexConfig() : prefer_fast_build(false), compact(false) {
    concurrency = std::thread::hardware_concurrency();
  }
};

template <typename SCALAR_T, int N_DIM>
std::unique_ptr<SpatialIndex<SCALAR_T, N_DIM>> CreateRTSpatialIndex(
    const RTSpatialIndexConfig& config);

}  // namespace gpuspatial
