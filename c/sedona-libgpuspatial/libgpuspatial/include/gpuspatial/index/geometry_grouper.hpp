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
#include "gpuspatial/geom/box.cuh"
#include "gpuspatial/loader/device_geometries.cuh"
#include "gpuspatial/utils/launcher.h"
#include "gpuspatial/utils/morton_code.h"

#include "rmm/cuda_stream_view.hpp"
#include "rmm/device_uvector.hpp"
#include "rmm/exec_policy.hpp"

#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

#include <memory>

namespace gpuspatial {
template <typename POINT_T, typename INDEX_T>
class GeometryGrouper {
  using box_t = Box<POINT_T>;
  static constexpr int n_dim = POINT_T::n_dim;
  using scalar_t = typename POINT_T::scalar_t;

 public:
  void Group(const rmm::cuda_stream_view& stream,
             const DeviceGeometries<POINT_T, INDEX_T>& geometries,
             uint32_t geoms_per_aabb) {
    switch (geometries.get_geometry_type()) {
      case GeometryType::kPoint: {
        Group(
            stream,
            geometries.template GetGeometryArrayView<PointArrayView<POINT_T, INDEX_T>>(),
            geoms_per_aabb);
        break;
      }
      case GeometryType::kMultiPoint: {
        Group(stream,
              geometries
                  .template GetGeometryArrayView<MultiPointArrayView<POINT_T, INDEX_T>>(),
              geoms_per_aabb);
        break;
      }
      case GeometryType::kLineString: {
        Group(stream,
              geometries
                  .template GetGeometryArrayView<LineStringArrayView<POINT_T, INDEX_T>>(),
              geoms_per_aabb);
        break;
      }
      case GeometryType::kMultiLineString: {
        Group(stream,
              geometries.template GetGeometryArrayView<
                  MultiLineStringArrayView<POINT_T, INDEX_T>>(),
              geoms_per_aabb);
        break;
      }
      case GeometryType::kPolygon: {
        Group(stream,
              geometries
                  .template GetGeometryArrayView<PolygonArrayView<POINT_T, INDEX_T>>(),
              geoms_per_aabb);
        break;
      }
      case GeometryType::kMultiPolygon: {
        Group(
            stream,
            geometries
                .template GetGeometryArrayView<MultiPolygonArrayView<POINT_T, INDEX_T>>(),
            geoms_per_aabb);
        break;
      }
      case GeometryType::kBox: {
        Group(stream,
              geometries.template GetGeometryArrayView<BoxArrayView<POINT_T, INDEX_T>>(),
              geoms_per_aabb);
        break;
      }
      default:
        assert(false);
    }
  }

  template <typename GEOMETRY_ARRAY_T>
  void Group(const rmm::cuda_stream_view& stream, const GEOMETRY_ARRAY_T& geometries,
             uint32_t geoms_per_aabb) {
    rmm::device_uvector<INDEX_T> morton_codes(geometries.size(), stream);
    POINT_T min_world_corner, max_world_corner;

    min_world_corner.set_max();
    max_world_corner.set_min();

    for (int dim = 0; dim < n_dim; dim++) {
      auto min_val = thrust::transform_reduce(
          rmm::exec_policy_nosync(stream), thrust::make_counting_iterator<INDEX_T>(0),
          thrust::make_counting_iterator<INDEX_T>(geometries.size()),
          [=] __host__ __device__(INDEX_T i) {
            const auto& geom = geometries[i];
            const auto& mbr = geom.get_mbr();

            return mbr.get_min(dim);
          },
          std::numeric_limits<scalar_t>::max(), thrust::minimum<scalar_t>());

      auto max_val = thrust::transform_reduce(
          rmm::exec_policy_nosync(stream), thrust::make_counting_iterator<INDEX_T>(0),
          thrust::make_counting_iterator<INDEX_T>(geometries.size()),
          [=] __host__ __device__(INDEX_T i) {
            const auto& geom = geometries[i];
            const auto& mbr = geom.get_mbr();

            return mbr.get_max(dim);
          },
          std::numeric_limits<scalar_t>::lowest(), thrust::maximum<scalar_t>());
      min_world_corner.set_coordinate(dim, min_val);
      max_world_corner.set_coordinate(dim, max_val);
    }

    // compute morton codes and reorder indices
    thrust::transform(rmm::exec_policy_nosync(stream),
                      thrust::make_counting_iterator<INDEX_T>(0),
                      thrust::make_counting_iterator<INDEX_T>(geometries.size()),
                      morton_codes.begin(), [=] __device__(INDEX_T i) {
                        const auto& geom = geometries[i];
                        const auto& mbr = geom.get_mbr();
                        auto p = mbr.centroid();
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
    reordered_indices_ =
        std::make_unique<rmm::device_uvector<INDEX_T>>(geometries.size(), stream);
    thrust::sequence(rmm::exec_policy_nosync(stream), reordered_indices_->begin(),
                     reordered_indices_->end());
    thrust::sort_by_key(rmm::exec_policy_nosync(stream), morton_codes.begin(),
                        morton_codes.end(), reordered_indices_->begin());

    auto n_aabbs = (geometries.size() + geoms_per_aabb - 1) / geoms_per_aabb;
    aabbs_ = std::make_unique<rmm::device_uvector<OptixAabb>>(n_aabbs, stream);
    OptixAabb empty_aabb;

    if (n_dim == 2) {
      empty_aabb = OptixAabb{
          std::numeric_limits<float>::max(),    std::numeric_limits<float>::max(),    0,
          std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), 0};
    } else if (n_dim == 3) {
      empty_aabb = OptixAabb{
          std::numeric_limits<float>::max(),    std::numeric_limits<float>::max(),
          std::numeric_limits<float>::max(),    std::numeric_limits<float>::lowest(),
          std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()};
    }

    thrust::fill(rmm::exec_policy_nosync(stream), aabbs_->begin(), aabbs_->end(),
                 empty_aabb);

    auto* p_aabbs = aabbs_->data();

    rmm::device_uvector<INDEX_T> n_geoms_per_aabb(n_aabbs, stream);

    auto* p_reordered_indices = reordered_indices_->data();
    auto* p_n_geoms_per_aabb = n_geoms_per_aabb.data();

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
        size_t idx_begin = aabb_id * geoms_per_aabb;
        size_t idx_end = std::min((size_t)geometries.size(), idx_begin + geoms_per_aabb);
        size_t idx_end_rup = (idx_end + 31) / 32;
        idx_end_rup *= 32;  // round up to the next multiple of 32

        p_n_geoms_per_aabb[aabb_id] = idx_end - idx_begin;

        for (auto idx = idx_begin + lane_id; idx < idx_end_rup; idx += 32) {
          Box<Point<float, POINT_T::n_dim>> mbr;

          auto warp_begin = idx - lane_id;
          auto warp_end = std::min(warp_begin + 32, idx_end);
          auto n_valid = warp_end - warp_begin;

          if (idx < idx_end) {
            auto geom_idx = p_reordered_indices[idx];
            mbr = geometries[geom_idx].get_mbr();
          }

          for (int dim = 0; dim < n_dim; dim++) {
            auto min_val =
                WarpReduce(temp_storage[warp_id])
                    .Reduce(mbr.get_min(dim), thrust::minimum<scalar_t>(), n_valid);
            if (lane_id == 0) {
              min_corner.set_coordinate(dim, min_val);
            }
            auto max_val =
                WarpReduce(temp_storage[warp_id])
                    .Reduce(mbr.get_max(dim), thrust::maximum<scalar_t>(), n_valid);
            if (lane_id == 0) {
              max_corner.set_coordinate(dim, max_val);
            }
          }
        }

        if (lane_id == 0) {
          box_t ext_mbr(min_corner, max_corner);
          p_aabbs[aabb_id] = ext_mbr.ToOptixAabb();
        }
      }
    });

    prefix_sum_ = std::make_unique<rmm::device_uvector<INDEX_T>>(n_aabbs + 1, stream);
    prefix_sum_->set_element_to_zero_async(0, stream);
    thrust::inclusive_scan(rmm::exec_policy_nosync(stream), n_geoms_per_aabb.begin(),
                           n_geoms_per_aabb.end(), prefix_sum_->begin() + 1);
#ifndef NDEBUG
    auto* p_prefix_sum = prefix_sum_->data();

    thrust::for_each(rmm::exec_policy_nosync(stream),
                     thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator<size_t>(aabbs_->size()),
                     [=] __device__(size_t aabb_idx) {
                       auto begin = p_prefix_sum[aabb_idx];
                       auto end = p_prefix_sum[aabb_idx + 1];
                       const auto& aabb = p_aabbs[aabb_idx];

                       for (auto i = begin; i < end; i++) {
                         auto geom_idx = p_reordered_indices[i];
                         auto mbr = geometries[geom_idx].get_mbr();
                         assert(mbr.covered_by(aabb));
                       }
                     });
#endif
  }

  ArrayView<OptixAabb> get_aabbs() const {
    if (aabbs_ != nullptr) {
      return ArrayView<OptixAabb>(aabbs_->data(), aabbs_->size());
    }
    return {};
  }

  ArrayView<INDEX_T> get_prefix_sum() const {
    if (prefix_sum_ != nullptr) {
      return ArrayView<INDEX_T>(prefix_sum_->data(), prefix_sum_->size());
    }
    return {};
  }

  ArrayView<INDEX_T> get_reordered_indices() const {
    if (reordered_indices_ != nullptr) {
      return ArrayView<INDEX_T>(reordered_indices_->data(), reordered_indices_->size());
    }
    return {};
  }

  void Clear() {
    aabbs_ = nullptr;
    prefix_sum_ = nullptr;
    reordered_indices_ = nullptr;
  }

 private:
  std::unique_ptr<rmm::device_uvector<OptixAabb>> aabbs_;
  std::unique_ptr<rmm::device_uvector<INDEX_T>> prefix_sum_;
  std::unique_ptr<rmm::device_uvector<INDEX_T>> reordered_indices_;
};
}  // namespace gpuspatial
