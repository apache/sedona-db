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

#include "gpuspatial/index/rt_spatial_index.hpp"
#include "gpuspatial/index/spatial_index.hpp"
#include "gpuspatial/rt/rt_engine.hpp"
#include "gpuspatial/utils/gpu_timer.hpp"
#include "gpuspatial/utils/queue.h"

#include "rmm/cuda_stream_pool.hpp"
#include "rmm/cuda_stream_view.hpp"
#include "rmm/device_uvector.hpp"
#define GPUSPATIAL_PROFILING
namespace gpuspatial {

template <typename SCALAR_T, int N_DIM>
class RTSpatialIndex : public SpatialIndex<SCALAR_T, N_DIM> {
  using point_t = typename SpatialIndex<SCALAR_T, N_DIM>::point_t;
  using box_t = typename SpatialIndex<SCALAR_T, N_DIM>::box_t;
  using scalar_t = typename point_t::scalar_t;
  static constexpr int n_dim = point_t::n_dim;

  using index_t = uint32_t;  // type of the index to represent geometries
  struct SpatialIndexContext {
    rmm::cuda_stream_view stream;
    std::string shader_id;
    rmm::device_buffer bvh_buffer{0, rmm::cuda_stream_default};
    OptixTraversableHandle handle;
    std::vector<char> h_launch_params_buffer;
    rmm::device_buffer launch_params_buffer{0, rmm::cuda_stream_default};
    std::unique_ptr<rmm::device_scalar<uint32_t>> counter;
    // output
    Queue<index_t> build_indices;
    rmm::device_uvector<index_t> probe_indices{0, rmm::cuda_stream_default};
#ifdef GPUSPATIAL_PROFILING
    GPUTimer timer;
    // counters
    double alloc_ms = 0.0;
    double bvh_build_ms = 0.0;
    double rt_ms = 0.0;
    double copy_res_ms = 0.0;
#endif
  };

 public:
  RTSpatialIndex() = default;

  void Init(const typename SpatialIndex<SCALAR_T, N_DIM>::Config* config);

  void Clear() override;

  void PushBuild(const box_t* rects, uint32_t n_rects) override;

  void FinishBuilding() override;

  void Probe(const box_t* rects, uint32_t n_rects, std::vector<uint32_t>* build_indices,
             std::vector<uint32_t>* probe_indices) override;

 private:
  RTSpatialIndexConfig<scalar_t, n_dim> config_;
  std::unique_ptr<rmm::cuda_stream_pool> stream_pool_;
  bool indexing_points_;
  // The rectangles being indexed or the MBRs of grouped points
  rmm::device_uvector<box_t> rects_{0, rmm::cuda_stream_default};
  // Data structures for indexing points
  rmm::device_uvector<index_t> point_ranges_{0, rmm::cuda_stream_default};
  rmm::device_uvector<index_t> reordered_point_indices_{0, rmm::cuda_stream_default};
  rmm::device_uvector<point_t> points_{0, rmm::cuda_stream_default};
  rmm::device_buffer bvh_buffer_{0, rmm::cuda_stream_default};
  OptixTraversableHandle handle_;
  int device_;

  void allocateResultBuffer(SpatialIndexContext& ctx, uint32_t capacity) const;

  void handleBuildPoint(SpatialIndexContext& ctx, ArrayView<point_t> points,
                        bool counting) const;

  void handleBuildPoint(SpatialIndexContext& ctx, ArrayView<box_t> rects,
                        bool counting) const;

  void handleBuildBox(SpatialIndexContext& ctx, ArrayView<point_t> points,
                      bool counting) const;

  void handleBuildBox(SpatialIndexContext& ctx, ArrayView<box_t> rects,
                      bool counting) const;

  void prepareLaunchParamsBoxQuery(SpatialIndexContext& ctx, ArrayView<box_t> probe_rects,
                                   bool forward, bool counting) const;

  void filter(SpatialIndexContext& ctx, uint32_t dim_x) const;

  size_t numGeometries() const {
    return indexing_points_ ? points_.size() : rects_.size();
  }
};
}  // namespace gpuspatial
