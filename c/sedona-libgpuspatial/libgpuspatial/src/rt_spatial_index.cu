
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
#include "gpuspatial/index/rt_spatial_index.cuh"
#include "gpuspatial/rt/launch_parameters.h"
#include "gpuspatial/utils/launcher.h"
#include "gpuspatial/utils/logger.hpp"
#include "gpuspatial/utils/morton_code.h"
#include "gpuspatial/utils/stopwatch.h"

#include "rt/shaders/shader_id.hpp"

#include "rmm/exec_policy.hpp"

#include <thrust/logical.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#define OPTIX_MAX_RAYS (1lu << 30)

namespace gpuspatial {
namespace detail {

template <typename POINT_T>
static rmm::device_uvector<OptixAabb> ComputeAABBs(rmm::cuda_stream_view stream,
                                                   const ArrayView<Box<POINT_T>>& mbrs) {
  rmm::device_uvector<OptixAabb> aabbs(mbrs.size(), stream);

  thrust::transform(rmm::exec_policy_nosync(stream), mbrs.begin(), mbrs.end(),
                    aabbs.begin(), [] __device__(const Box<POINT_T>& mbr) {
                      // handle empty boxes
                      if (mbr.get_min().empty() || mbr.get_max().empty()) {
                        // empty box
                        OptixAabb empty_aabb;
                        empty_aabb.minX = empty_aabb.minY = empty_aabb.minZ = 0.0f;
                        empty_aabb.maxX = empty_aabb.maxY = empty_aabb.maxZ = -1.0f;
                        return empty_aabb;
                      }
                      return mbr.ToOptixAabb();
                    });
  return std::move(aabbs);
}

template <typename POINT_T, typename INDEX_T>
rmm::device_uvector<OptixAabb> ComputeAABBs(
    rmm::cuda_stream_view stream, rmm::device_uvector<POINT_T>& points,
    rmm::device_uvector<INDEX_T>& prefix_sum,
    rmm::device_uvector<INDEX_T>& reordered_indices, int group_size,
    rmm::device_uvector<Box<POINT_T>>& mbrs) {
  using scalar_t = typename POINT_T::scalar_t;
  using box_t = Box<POINT_T>;
  constexpr int n_dim = POINT_T::n_dim;
  static_assert(n_dim == 2 || n_dim == 3, "Only 2D and 3D points are supported");
  POINT_T min_world_corner, max_world_corner;

  min_world_corner.set_max();
  max_world_corner.set_min();

  for (int dim = 0; dim < n_dim; dim++) {
    auto min_val = thrust::transform_reduce(
        rmm::exec_policy_nosync(stream), points.begin(), points.end(),
        [=] __device__(const POINT_T& p) -> scalar_t { return p.get_coordinate(dim); },
        std::numeric_limits<scalar_t>::max(), thrust::minimum<scalar_t>());
    auto max_val = thrust::transform_reduce(
        rmm::exec_policy_nosync(stream), points.begin(), points.end(),
        [=] __device__(const POINT_T& p) -> scalar_t { return p.get_coordinate(dim); },
        std::numeric_limits<scalar_t>::lowest(), thrust::maximum<scalar_t>());
    min_world_corner.set_coordinate(dim, min_val);
    max_world_corner.set_coordinate(dim, max_val);
  }

  auto np = points.size();
  rmm::device_uvector<uint32_t> morton_codes(np, stream);
  // compute morton codes and reorder indices
  thrust::transform(rmm::exec_policy_nosync(stream), points.begin(), points.end(),
                    morton_codes.begin(), [=] __device__(const POINT_T& p) {
                      POINT_T norm_p;

                      for (int dim = 0; dim < n_dim; dim++) {
                        auto min_val = min_world_corner.get_coordinate(dim);
                        auto max_val = max_world_corner.get_coordinate(dim);
                        auto extent = min_val == max_val ? 1 : max_val - min_val;
                        auto norm_val = (p.get_coordinate(dim) - min_val) / extent;
                        norm_p.set_coordinate(dim, norm_val);
                      }
                      return detail::morton_code(norm_p.get_vec());
                    });
  reordered_indices.resize(np, stream);
  thrust::sequence(rmm::exec_policy_nosync(stream), reordered_indices.begin(),
                   reordered_indices.end());
  thrust::sort_by_key(rmm::exec_policy_nosync(stream), morton_codes.begin(),
                      morton_codes.end(), reordered_indices.begin());
  auto n_aabbs = (np + group_size - 1) / group_size;
  mbrs.resize(n_aabbs, stream);
  rmm::device_uvector<OptixAabb> aabbs(n_aabbs, stream);
  rmm::device_uvector<INDEX_T> np_per_aabb(n_aabbs, stream);

  auto* p_reordered_indices = reordered_indices.data();
  auto* p_aabbs = aabbs.data();
  auto* p_np_per_aabb = np_per_aabb.data();
  ArrayView<POINT_T> v_points(points);
  ArrayView<box_t> v_mbrs(mbrs);
  // each warp takes an AABB and processes points_per_aabb points
  LaunchKernel(stream, [=] __device__() mutable {
    typedef cub::WarpReduce<scalar_t> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage[MAX_BLOCK_SIZE / 32];
    auto warp_id = threadIdx.x / 32;
    auto lane_id = threadIdx.x % 32;
    auto global_warp_id = TID_1D / 32;
    auto n_warps = TOTAL_THREADS_1D / 32;

    for (uint32_t aabb_id = global_warp_id; aabb_id < n_aabbs; aabb_id += n_warps) {
      POINT_T min_corner, max_corner;
      size_t idx_begin = aabb_id * group_size;
      size_t idx_end = std::min(np, idx_begin + group_size);
      size_t idx_end_rup = (idx_end + 31) / 32;

      idx_end_rup *= 32;  // round up to the next multiple of 32
      p_np_per_aabb[aabb_id] = idx_end - idx_begin;

      for (auto idx = idx_begin + lane_id; idx < idx_end_rup; idx += 32) {
        POINT_T p;
        auto warp_begin = idx - lane_id;
        auto warp_end = std::min(warp_begin + 32, idx_end);
        auto n_valid = warp_end - warp_begin;

        if (idx < idx_end) {
          auto point_idx = p_reordered_indices[idx];
          p = v_points[point_idx];
        } else {
          p.set_empty();
        }

        if (!p.empty()) {
          for (int dim = 0; dim < n_dim; dim++) {
            auto min_val =
                WarpReduce(temp_storage[warp_id])
                    .Reduce(p.get_coordinate(dim), thrust::minimum<scalar_t>(), n_valid);
            if (lane_id == 0) {
              min_corner.set_coordinate(dim, min_val);
            }
            auto max_val =
                WarpReduce(temp_storage[warp_id])
                    .Reduce(p.get_coordinate(dim), thrust::maximum<scalar_t>(), n_valid);
            if (lane_id == 0) {
              max_corner.set_coordinate(dim, max_val);
            }
          }
        }
      }

      if (lane_id == 0) {
        if (min_corner.empty() || max_corner.empty()) {
          OptixAabb empty_aabb;
          empty_aabb.minX = empty_aabb.minY = empty_aabb.minZ = 0.0f;
          empty_aabb.maxX = empty_aabb.maxY = empty_aabb.maxZ = -1.0f;
          v_mbrs[aabb_id] = box_t();  // empty box
          p_aabbs[aabb_id] = empty_aabb;
        } else {
          box_t ext_mbr(min_corner, max_corner);

          v_mbrs[aabb_id] = ext_mbr;
          p_aabbs[aabb_id] = ext_mbr.ToOptixAabb();
        }
      }
    }
  });
  prefix_sum.resize(n_aabbs + 1, stream);
  prefix_sum.set_element_to_zero_async(0, stream);
  thrust::inclusive_scan(rmm::exec_policy_nosync(stream), np_per_aabb.begin(),
                         np_per_aabb.end(), prefix_sum.begin() + 1);
#ifndef NDEBUG
  auto* p_prefix_sum = prefix_sum.data();

  thrust::for_each(rmm::exec_policy_nosync(stream), thrust::counting_iterator<size_t>(0),
                   thrust::counting_iterator<size_t>(aabbs.size()),
                   [=] __device__(size_t aabb_idx) {
                     auto begin = p_prefix_sum[aabb_idx];
                     auto end = p_prefix_sum[aabb_idx + 1];
                     const auto& aabb = p_aabbs[aabb_idx];

                     for (auto i = begin; i < end; i++) {
                       auto point_idx = p_reordered_indices[i];
                       const auto& p = v_points[point_idx];
                       for (int dim = 0; dim < n_dim; dim++) {
                         auto coord = p.get_coordinate(dim);
                         assert(coord >= (&aabb.minX)[dim] && coord <= (&aabb.maxX)[dim]);
                         assert(v_mbrs[aabb_idx].covers(p));
                       }
                     }
                   });
#endif
  return std::move(aabbs);
}

template <typename POINT_T, typename INDEX_T>
void RefineExactPoints(rmm::cuda_stream_view stream, ArrayView<POINT_T> build_points,
                       ArrayView<POINT_T> probe_points, ArrayView<INDEX_T> prefix_sum,
                       ArrayView<INDEX_T> reordered_indices, ArrayView<INDEX_T> rect_ids,
                       ArrayView<INDEX_T> point_ids, Queue<INDEX_T>& build_indices,
                       ArrayView<INDEX_T> probe_indices) {
  auto d_queue = build_indices.DeviceObject();

  LaunchKernel(stream, [=] __device__() mutable {
    auto lane_id = threadIdx.x % 32;
    auto global_warp_id = TID_1D / 32;
    auto n_warps = TOTAL_THREADS_1D / 32;

    for (uint32_t i = global_warp_id; i < rect_ids.size(); i += n_warps) {
      auto rect_id = rect_ids[i];
      auto point_id = point_ids[i];
      auto build_point_begin = prefix_sum[rect_id];
      auto build_point_end = prefix_sum[rect_id + 1];

      for (uint32_t j = lane_id + build_point_begin; j < build_point_end;
           j += WARP_SIZE) {
        auto build_point_id = reordered_indices[j];
        const auto& build_point = build_points[build_point_id];
        const auto& probe_point = probe_points[point_id];
        if (build_point == probe_point) {
          auto tail = d_queue.Append(build_point_id);
          probe_indices[tail] = point_id;
        }
      }
    }
  });
}
}  // namespace detail

template <typename SCALAR_T, int N_DIM>
RTSpatialIndex<SCALAR_T, N_DIM>::RTSpatialIndex(const RTSpatialIndexConfig& config)
    : config_(config),
      stream_pool_(std::make_unique<rmm::cuda_stream_pool>(config_.concurrency)),
      indexing_points_(false),
      handle_(0) {}

template <typename SCALAR_T, int N_DIM>
void RTSpatialIndex<SCALAR_T, N_DIM>::Clear() {
  GPUSPATIAL_LOG_INFO("RTSpatialIndex %p (Free %zu MB), Clear", this,
                      rmm::available_device_memory().first / 1024 / 1024);
  auto stream = rmm::cuda_stream_default;
  bvh_buffer_.resize(0, stream);
  bvh_buffer_.shrink_to_fit(stream);
  rects_.resize(0, stream);
  rects_.shrink_to_fit(stream);
  points_.resize(0, stream);
  points_.shrink_to_fit(stream);
  stream.synchronize();
}

template <typename SCALAR_T, int N_DIM>
void RTSpatialIndex<SCALAR_T, N_DIM>::PushBuild(const box_t* rects, uint32_t n_rects) {
  GPUSPATIAL_LOG_INFO("RTSpatialIndex %p (Free %zu MB), PushBuild, rectangles %zu", this,
                      rmm::available_device_memory().first / 1024 / 1024, n_rects);
  if (n_rects == 0) return;
  auto stream = rmm::cuda_stream_default;
  auto prev_size = rects_.size();

  rects_.resize(rects_.size() + n_rects, stream);
  CUDA_CHECK(cudaMemcpyAsync(rects_.data() + prev_size, rects, sizeof(box_t) * n_rects,
                             cudaMemcpyHostToDevice, stream));
}

template <typename SCALAR_T, int N_DIM>
void RTSpatialIndex<SCALAR_T, N_DIM>::FinishBuilding() {
  auto stream = rmm::cuda_stream_default;

  indexing_points_ = thrust::all_of(rmm::exec_policy_nosync(stream), rects_.begin(),
                                    rects_.end(), [] __device__(const box_t& box) {
                                      bool is_point = true;
                                      for (int dim = 0; dim < n_dim; dim++) {
                                        is_point &= box.get_min(dim) == box.get_max(dim);
                                      }
                                      return is_point;
                                    });

  rmm::device_uvector<OptixAabb> aabbs{0, stream};
  if (indexing_points_) {
    points_.resize(rects_.size(), stream);
    thrust::transform(rmm::exec_policy_nosync(stream), rects_.begin(), rects_.end(),
                      points_.begin(),
                      [] __device__(const box_t& box) { return box.get_min(); });
    aabbs = std::move(detail::ComputeAABBs(stream, points_, point_ranges_,
                                           reordered_point_indices_,
                                           config_.n_points_per_aabb, rects_));
  } else {
    aabbs = std::move(detail::ComputeAABBs(stream, ArrayView<box_t>(rects_)));
  }

  handle_ = config_.rt_engine->BuildAccelCustom(stream, ArrayView<OptixAabb>(aabbs),
                                                bvh_buffer_, config_.prefer_fast_build,
                                                config_.compact);

  GPUSPATIAL_LOG_INFO(
      "RTSpatialIndex %p (Free %zu MB), FinishBuilding Index on %s, Total geoms: %zu",
      this, rmm::available_device_memory().first / 1024 / 1024,
      indexing_points_ ? "Points" : "Rectangles", numGeometries());
}

template <typename SCALAR_T, int N_DIM>
void RTSpatialIndex<SCALAR_T, N_DIM>::Probe(const box_t* rects, uint32_t n_rects,
                                            std::vector<uint32_t>* build_indices,
                                            std::vector<uint32_t>* probe_indices) {
  if (n_rects == 0) return;
  SpatialIndexContext ctx;
  auto stream = stream_pool_->get_stream();
  rmm::device_uvector<box_t> d_rects(n_rects, stream);
  rmm::device_uvector<point_t> d_points{0, stream};

  CUDA_CHECK(cudaMemcpyAsync(d_rects.data(), rects, sizeof(box_t) * n_rects,
                             cudaMemcpyHostToDevice, stream));

  bool probe_points = thrust::all_of(rmm::exec_policy_nosync(stream), d_rects.begin(),
                                     d_rects.end(), [] __device__(const box_t& box) {
                                       bool is_point = true;
                                       for (int dim = 0; dim < n_dim; dim++) {
                                         is_point &= box.get_min(dim) == box.get_max(dim);
                                       }
                                       return is_point;
                                     });

  if (probe_points) {
    d_points.resize(d_rects.size(), stream);
    thrust::transform(rmm::exec_policy_nosync(stream), d_rects.begin(), d_rects.end(),
                      d_points.begin(),
                      [] __device__(const box_t& box) { return box.get_min(); });
    d_rects.resize(0, stream);
    d_rects.shrink_to_fit(stream);

  } else {
    // Build a BVH over the MBRs of the stream geometries
#ifdef GPUSPATIAL_PROFILING
    ctx.timer.start(stream);
#endif
    rmm::device_uvector<OptixAabb> aabbs(n_rects, stream);
    thrust::transform(rmm::exec_policy_nosync(stream), d_rects.begin(), d_rects.end(),
                      aabbs.begin(),
                      [] __device__(const box_t& mbr) { return mbr.ToOptixAabb(); });
    ctx.handle = config_.rt_engine->BuildAccelCustom(
        stream, ArrayView<OptixAabb>(aabbs), ctx.bvh_buffer, config_.prefer_fast_build,
        config_.compact);
#ifdef GPUSPATIAL_PROFILING
    ctx.bvh_build_ms = ctx.timer.stop(stream);
#endif
  }

  ctx.counter = std::make_unique<rmm::device_scalar<uint32_t>>(0, stream);

  bool swap_ids = false;

  auto query = [&](bool counting) {
#ifdef GPUSPATIAL_PROFILING
    ctx.timer.start(stream);
#endif
    if (indexing_points_) {
      if (probe_points) {
        handleBuildPoint(ctx, ArrayView<point_t>(d_points), counting);
      } else {
        handleBuildPoint(ctx, ArrayView<box_t>(d_rects), counting);
        swap_ids = true;
      }
    } else {
      if (probe_points) {
        handleBuildBox(ctx, ArrayView<point_t>(d_points), counting);
      } else {
        handleBuildBox(ctx, ArrayView<box_t>(d_rects), counting);
      }
    }
#ifdef GPUSPATIAL_PROFILING
    ctx.rt_ms += ctx.timer.stop(stream);
#endif
  };

  // first pass: counting
  query(true /* counting */);

  auto cap = ctx.counter->value(stream);
  if (cap == 0) {
    return;
  }
  allocateResultBuffer(ctx, cap);
  // second pass: retrieve results
  query(false /* counting */);

  auto result_size = ctx.build_indices.size(stream);
  ArrayView<index_t> v_build_indices(ctx.build_indices.data(), result_size);
  ArrayView<index_t> v_probe_indices(ctx.probe_indices.data(), result_size);

  if (swap_ids) {
    // IMPORTANT: In this case, the BVH is built on probe side and points are
    // cast on the build side, so the result pairs are (probe_id, build_id) instead of
    // (build_id, probe_id). We need to swap the output buffers to correct this.
    std::swap(v_build_indices, v_probe_indices);
  }

#ifdef GPUSPATIAL_PROFILING
  Stopwatch sw;
  sw.start();
#endif
  build_indices->resize(result_size);
  CUDA_CHECK(cudaMemcpyAsync(build_indices->data(), v_build_indices.data(),
                             sizeof(index_t) * result_size, cudaMemcpyDeviceToHost,
                             stream));

  probe_indices->resize(result_size);
  CUDA_CHECK(cudaMemcpyAsync(probe_indices->data(), v_probe_indices.data(),
                             sizeof(index_t) * result_size, cudaMemcpyDeviceToHost,
                             stream));
  stream.synchronize();
#ifdef GPUSPATIAL_PROFILING
  sw.stop();
  ctx.copy_res_ms = sw.ms();
  GPUSPATIAL_LOG_INFO(
      "RTSpatialIndex %p (Free %zu MB), Probe %s, Size: %zu, Results: %zu, Alloc: %.2f ms, BVH Build: %.2f ms, RT: %.2f ms, Copy res: %.2f ms",
      this, rmm::available_device_memory().first / 1024 / 1024,
      probe_points ? "Points" : "Rectangles",
      probe_points ? d_points.size() : d_rects.size(), build_indices->size(),
      ctx.alloc_ms, ctx.bvh_build_ms, ctx.rt_ms, ctx.copy_res_ms);
#endif
}

template <typename SCALAR_T, int N_DIM>
void RTSpatialIndex<SCALAR_T, N_DIM>::handleBuildPoint(SpatialIndexContext& ctx,
                                                       ArrayView<point_t> points,
                                                       bool counting) const {
  using launch_params_t = detail::LaunchParamsPointQuery<point_t>;

  ctx.shader_id = GetPointQueryShaderId<point_t>();
  ctx.launch_params_buffer.resize(sizeof(launch_params_t), ctx.stream);
  ctx.h_launch_params_buffer.resize(sizeof(launch_params_t));
  auto& launch_params =
      *reinterpret_cast<launch_params_t*>(ctx.h_launch_params_buffer.data());

  launch_params.rects = ArrayView<box_t>(rects_);
  launch_params.points = points;
  launch_params.handle = handle_;

  uint32_t dim_x = std::min(OPTIX_MAX_RAYS, points.size());

  if (counting) {
    launch_params.count = ctx.counter->data();

    CUDA_CHECK(cudaMemcpyAsync(ctx.launch_params_buffer.data(), &launch_params,
                               sizeof(launch_params_t), cudaMemcpyHostToDevice,
                               ctx.stream));

    filter(ctx, dim_x);
  } else {
    auto cap = ctx.build_indices.capacity();
    Queue<index_t> rect_ids;
    rmm::device_uvector<index_t> point_ids(cap, ctx.stream);

    rect_ids.Init(ctx.stream, cap);

    launch_params.count = nullptr;
    launch_params.rect_ids = rect_ids.DeviceObject();
    launch_params.point_ids = ArrayView<index_t>(point_ids);

    CUDA_CHECK(cudaMemcpyAsync(ctx.launch_params_buffer.data(), &launch_params,
                               sizeof(launch_params_t), cudaMemcpyHostToDevice,
                               ctx.stream));

    filter(ctx, dim_x);

    detail::RefineExactPoints<point_t, index_t>(
        ctx.stream, ArrayView<point_t>(points_), points,
        ArrayView<index_t>(point_ranges_), ArrayView<index_t>(reordered_point_indices_),
        ArrayView<index_t>(rect_ids.data(), rect_ids.size(ctx.stream)),
        ArrayView<index_t>(point_ids), ctx.build_indices,
        ArrayView<index_t>(ctx.probe_indices));
  }
}

template <typename SCALAR_T, int N_DIM>
void RTSpatialIndex<SCALAR_T, N_DIM>::handleBuildPoint(SpatialIndexContext& ctx,
                                                       ArrayView<box_t> rects,
                                                       bool counting) const {
  using launch_params_t = detail::LaunchParamsPointQuery<point_t>;

  ctx.shader_id = GetPointQueryShaderId<point_t>();
  ctx.launch_params_buffer.resize(sizeof(launch_params_t), ctx.stream);
  ctx.h_launch_params_buffer.resize(sizeof(launch_params_t));
  auto& launch_params =
      *reinterpret_cast<launch_params_t*>(ctx.h_launch_params_buffer.data());

  launch_params.rects = rects;
  launch_params.points = ArrayView<point_t>(points_);
  launch_params.handle = ctx.handle;
  if (counting) {
    launch_params.count = ctx.counter->data();
  } else {
    launch_params.count = nullptr;
    launch_params.rect_ids = ctx.build_indices.DeviceObject();
    launch_params.point_ids = ArrayView<index_t>(ctx.probe_indices);
  }

  CUDA_CHECK(cudaMemcpyAsync(ctx.launch_params_buffer.data(), &launch_params,
                             sizeof(launch_params_t), cudaMemcpyHostToDevice,
                             ctx.stream));

  uint32_t dim_x = std::min(OPTIX_MAX_RAYS, points_.size());

  filter(ctx, dim_x);
}

template <typename SCALAR_T, int N_DIM>
void RTSpatialIndex<SCALAR_T, N_DIM>::handleBuildBox(SpatialIndexContext& ctx,
                                                     ArrayView<point_t> points,
                                                     bool counting) const {
  using launch_params_t = detail::LaunchParamsPointQuery<point_t>;

  ctx.shader_id = GetPointQueryShaderId<point_t>();
  ctx.launch_params_buffer.resize(sizeof(launch_params_t), ctx.stream);
  ctx.h_launch_params_buffer.resize(sizeof(launch_params_t));
  auto& launch_params =
      *reinterpret_cast<launch_params_t*>(ctx.h_launch_params_buffer.data());

  launch_params.rects = ArrayView<box_t>(rects_);
  launch_params.points = points;
  launch_params.handle = handle_;
  if (counting) {
    launch_params.count = ctx.counter->data();
  } else {
    launch_params.count = nullptr;
    launch_params.rect_ids = ctx.build_indices.DeviceObject();
    launch_params.point_ids =
        ArrayView<index_t>(ctx.probe_indices.data(), ctx.probe_indices.size());
  }

  CUDA_CHECK(cudaMemcpyAsync(ctx.launch_params_buffer.data(), &launch_params,
                             sizeof(launch_params_t), cudaMemcpyHostToDevice,
                             ctx.stream));

  uint32_t dim_x = std::min(OPTIX_MAX_RAYS, points.size());

  filter(ctx, dim_x);
}

template <typename SCALAR_T, int N_DIM>
void RTSpatialIndex<SCALAR_T, N_DIM>::handleBuildBox(SpatialIndexContext& ctx,
                                                     ArrayView<box_t> rects,
                                                     bool counting) const {
  // forward cast: cast rays from stream geometries with the BVH of build geometries
  {
    auto dim_x = std::min(OPTIX_MAX_RAYS, rects.size());

    prepareLaunchParamsBoxQuery(ctx, rects, true /* forward */, counting);
    filter(ctx, dim_x);
  }
  // backward cast: cast rays from the build geometries with the BVH of stream geometries
  {
    auto dim_x = std::min(OPTIX_MAX_RAYS, rects_.size());

    prepareLaunchParamsBoxQuery(ctx, rects, false /* forward */, counting);
    filter(ctx, dim_x);
  }
}

template <typename SCALAR_T, int N_DIM>
void RTSpatialIndex<SCALAR_T, N_DIM>::allocateResultBuffer(SpatialIndexContext& ctx,
                                                           uint32_t capacity) const {
#ifdef GPUSPATIAL_PROFILING
  ctx.timer.start(ctx.stream);
#endif

  GPUSPATIAL_LOG_INFO(
      "RTSpatialIndex %p (Free %zu MB), Allocate result buffer, memory consumption %zu MB, capacity %u",
      this, rmm::available_device_memory().first / 1024 / 1024,
      (uint64_t)capacity * 2 * sizeof(index_t) / 1024 / 1024, capacity);

  ctx.build_indices.Init(ctx.stream, capacity);
  ctx.probe_indices.resize(capacity, ctx.stream);
#ifdef GPUSPATIAL_PROFILING
  ctx.alloc_ms += ctx.timer.stop(ctx.stream);
#endif
}

template <typename SCALAR_T, int N_DIM>
void RTSpatialIndex<SCALAR_T, N_DIM>::prepareLaunchParamsBoxQuery(
    SpatialIndexContext& ctx, ArrayView<box_t> probe_rects, bool forward,
    bool counting) const {
  using launch_params_t = detail::LaunchParamsBoxQuery<point_t>;
  ctx.launch_params_buffer.resize(sizeof(launch_params_t), ctx.stream);
  ctx.h_launch_params_buffer.resize(sizeof(launch_params_t));
  auto& launch_params =
      *reinterpret_cast<launch_params_t*>(ctx.h_launch_params_buffer.data());

  launch_params.rects1 = ArrayView<box_t>(rects_);
  launch_params.rects2 = probe_rects;

  if (forward) {
    launch_params.handle = handle_;
    ctx.shader_id = GetBoxQueryForwardShaderId<point_t>();
  } else {
    launch_params.handle = ctx.handle;
    ctx.shader_id = GetBoxQueryBackwardShaderId<point_t>();
  }

  if (counting) {
    launch_params.count = ctx.counter->data();
  } else {
    launch_params.count = nullptr;
    launch_params.rect1_ids = ctx.build_indices.DeviceObject();
    launch_params.rect2_ids = ArrayView<index_t>(ctx.probe_indices);
  }

  CUDA_CHECK(cudaMemcpyAsync(ctx.launch_params_buffer.data(), &launch_params,
                             sizeof(launch_params_t), cudaMemcpyHostToDevice,
                             ctx.stream));
}

template <typename SCALAR_T, int N_DIM>
void RTSpatialIndex<SCALAR_T, N_DIM>::filter(SpatialIndexContext& ctx,
                                             uint32_t dim_x) const {
#ifdef GPUSPATIAL_PROFILING
  ctx.timer.start(ctx.stream);
#endif
  if (dim_x > 0) {
    config_.rt_engine->Render(ctx.stream, ctx.shader_id, dim3{dim_x, 1, 1},
                              ArrayView<char>((char*)ctx.launch_params_buffer.data(),
                                              ctx.launch_params_buffer.size()));
  }
#ifdef GPUSPATIAL_PROFILING
  ctx.rt_ms += ctx.timer.stop(ctx.stream);
#endif
}

template <typename SCALAR_T, int N_DIM>
std::unique_ptr<SpatialIndex<SCALAR_T, N_DIM>> CreateRTSpatialIndex(
    const RTSpatialIndexConfig& config) {
  auto index = std::make_unique<RTSpatialIndex<SCALAR_T, N_DIM>>(config);
  GPUSPATIAL_LOG_INFO(
      "Create RTSpatialIndex %p, fast_build = %d, compact = %d, concurrency = %d",
      index.get(), config.prefer_fast_build, config.compact, config.concurrency);
  return std::move(index);
}

template std::unique_ptr<SpatialIndex<float, 2>> CreateRTSpatialIndex(
    const RTSpatialIndexConfig& config);
template std::unique_ptr<SpatialIndex<float, 3>> CreateRTSpatialIndex(
    const RTSpatialIndexConfig& config);
template std::unique_ptr<SpatialIndex<double, 2>> CreateRTSpatialIndex(
    const RTSpatialIndexConfig& config);
template std::unique_ptr<SpatialIndex<double, 3>> CreateRTSpatialIndex(
    const RTSpatialIndexConfig& config);
}  // namespace gpuspatial
