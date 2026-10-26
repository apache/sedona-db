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
#include "gpuspatial/relate/relate_engine.cuh"
#include "gpuspatial/utils/array_view.hpp"
#include "gpuspatial/utils/helpers.cuh"
#include "gpuspatial/utils/launcher.hpp"

#include "rmm/cuda_stream_view.hpp"
#include "rmm/device_buffer.hpp"
#include "rmm/device_scalar.hpp"
#include "rmm/device_uvector.hpp"
#include "rmm/exec_policy.hpp"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <cub/cub.cuh>

#include <cstdint>
#include <limits>

namespace gpuspatial {
namespace detail {

template <typename POINT_T, typename INDEX_T>
uint32_t ComputeNumAabbs(const rmm::cuda_stream_view& stream,
                         const PolygonArrayView<POINT_T, INDEX_T>& polygons,
                         ArrayView<uint32_t> polygon_ids, int segs_per_aabb) {
  auto n_polygons = polygon_ids.size();

  rmm::device_uvector<uint32_t> n_aabbs(n_polygons, stream);
  auto* p_n_aabbs = n_aabbs.data();

  LaunchKernel(stream, [=] __device__() {
    using WarpReduce = cub::WarpReduce<uint32_t>;
    __shared__ WarpReduce::TempStorage temp_storage[MAX_BLOCK_SIZE / 32];
    auto lane = threadIdx.x % 32;
    auto warp_id = threadIdx.x / 32;
    auto global_warp_id = TID_1D / 32;
    auto n_warps = TOTAL_THREADS_1D / 32;

    for (auto i = global_warp_id; i < n_polygons; i += n_warps) {
      auto id = polygon_ids[i];
      const auto& polygon = polygons[id];
      uint32_t total_segs = 0;

      for (auto ring = lane; ring < polygon.num_rings(); ring += 32) {
        total_segs +=
            (polygon.get_ring(ring).num_segments() + segs_per_aabb - 1) / segs_per_aabb;
      }
      total_segs = WarpReduce(temp_storage[warp_id]).Sum(total_segs);
      if (lane == 0) {
        p_n_aabbs[i] = total_segs;
      }
    }
  });
  return thrust::reduce(rmm::exec_policy_nosync(stream), n_aabbs.begin(), n_aabbs.end());
}

template <typename POINT_T, typename INDEX_T>
uint32_t ComputeNumAabbs(const rmm::cuda_stream_view& stream,
                         const MultiPolygonArrayView<POINT_T, INDEX_T>& multi_polygons,
                         ArrayView<uint32_t> multi_polygon_ids, int segs_per_aabb) {
  auto n_multi_polygons = multi_polygon_ids.size();
  rmm::device_uvector<uint32_t> n_aabbs(n_multi_polygons, stream);
  auto* p_n_aabbs = n_aabbs.data();

  LaunchKernel(stream, [=] __device__() {
    using WarpReduce = cub::WarpReduce<uint32_t>;
    __shared__ WarpReduce::TempStorage temp_storage[MAX_BLOCK_SIZE / 32];
    auto lane = threadIdx.x % 32;
    auto warp_id = threadIdx.x / 32;
    auto global_warp_id = TID_1D / 32;
    auto n_warps = TOTAL_THREADS_1D / 32;

    for (auto i = global_warp_id; i < n_multi_polygons; i += n_warps) {
      auto id = multi_polygon_ids[i];
      const auto& multi_polygon = multi_polygons[id];

      uint32_t multipoly_aabb_count = 0;

      for (int part_idx = 0; part_idx < multi_polygon.num_polygons(); part_idx++) {
        auto polygon = multi_polygon.get_polygon(part_idx);

        // Local accumulator for this thread
        uint32_t thread_aabb_count = 0;

        for (auto ring = lane; ring < polygon.num_rings(); ring += 32) {
          auto n_segs = polygon.get_ring(ring).num_segments();

          thread_aabb_count += (n_segs + segs_per_aabb - 1) / segs_per_aabb;
        }

        // Reduce across the warp to get total AABBs for this polygon (part)
        uint32_t part_total = WarpReduce(temp_storage[warp_id]).Sum(thread_aabb_count);

        // Add this part's total to the multi-polygon accumulator
        if (lane == 0) {
          multipoly_aabb_count += part_total;
        }
      }

      if (lane == 0) {
        p_n_aabbs[i] = multipoly_aabb_count;
      }
    }
  });
  return thrust::reduce(rmm::exec_policy_nosync(stream), n_aabbs.begin(), n_aabbs.end());
}
}  // namespace detail

template <typename POINT_T, typename INDEX_T>
size_t RelateEngine<POINT_T, INDEX_T>::EstimateBVHSize(
    const rmm::cuda_stream_view& stream, const PolygonArrayView<POINT_T, INDEX_T>& polys,
    ArrayView<uint32_t> poly_ids, int segs_per_aabb) {
  auto num_aabbs = detail::ComputeNumAabbs(stream, polys, poly_ids, segs_per_aabb);
  if (num_aabbs == 0) {
    return 0;
  }

  // temporary but still needed to consider this part of memory
  auto aabb_size = num_aabbs * sizeof(OptixAabb);
  auto bvh_bytes = rt_engine_->EstimateMemoryUsageForAABB(
      num_aabbs, config_.bvh_fast_build, config_.bvh_compact);
  // BVH size and aabb_poly_ids, aabb_ring_ids, aabb_vertex_offsets
  return aabb_size + bvh_bytes + 4 * sizeof(INDEX_T) * num_aabbs;
}

template <typename POINT_T, typename INDEX_T>
size_t RelateEngine<POINT_T, INDEX_T>::EstimateBVHSize(
    const rmm::cuda_stream_view& stream,
    const MultiPolygonArrayView<POINT_T, INDEX_T>& multi_polys,
    ArrayView<uint32_t> multi_poly_ids, int segs_per_aabb) {
  auto num_aabbs =
      detail::ComputeNumAabbs(stream, multi_polys, multi_poly_ids, segs_per_aabb);

  // temporary but still needed to consider this part of memory
  auto aabb_size = num_aabbs * sizeof(OptixAabb);
  auto bvh_bytes = rt_engine_->EstimateMemoryUsageForAABB(
      num_aabbs, config_.bvh_fast_build, config_.bvh_compact);
  // BVH size and aabb_multi_poly_ids, aabb_part_ids, aabb_ring_ids, aabb_vertex_offsets
  return aabb_size + bvh_bytes + 5 * sizeof(INDEX_T) * num_aabbs;
}

template <typename POINT_T, typename INDEX_T>
OptixTraversableHandle RelateEngine<POINT_T, INDEX_T>::BuildBVH(
    const rmm::cuda_stream_view& stream,
    const PolygonArrayView<POINT_T, INDEX_T>& polygons, ArrayView<uint32_t> polygon_ids,
    int segs_per_aabb, rmm::device_buffer& buffer,
    rmm::device_uvector<INDEX_T>& aabb_poly_ids,
    rmm::device_uvector<INDEX_T>& aabb_ring_ids,
    rmm::device_uvector<thrust::pair<INDEX_T, INDEX_T>>& aabb_vertex_offsets) {
  auto n_polygons = polygon_ids.size();
  auto num_aabbs = detail::ComputeNumAabbs(stream, polygons, polygon_ids, segs_per_aabb);
  aabb_poly_ids = std::move(rmm::device_uvector<INDEX_T>(num_aabbs, stream));
  aabb_ring_ids = std::move(rmm::device_uvector<INDEX_T>(num_aabbs, stream));
  aabb_vertex_offsets =
      std::move(rmm::device_uvector<thrust::pair<INDEX_T, INDEX_T>>(num_aabbs, stream));

  auto* p_aabb_poly_ids = aabb_poly_ids.data();
  auto* p_aabb_ring_ids = aabb_ring_ids.data();
  auto* p_aabb_vertex_offsets = aabb_vertex_offsets.data();

  rmm::device_scalar<uint32_t> d_tail(0, stream);
  auto* p_tail = d_tail.data();

  LaunchKernel(stream.value(), [=] __device__() {
    auto lane = threadIdx.x % 32;
    auto global_warp_id = TID_1D / 32;
    auto n_warps = TOTAL_THREADS_1D / 32;

    // each warp takes a polygon
    for (auto i = global_warp_id; i < n_polygons; i += n_warps) {
      auto poly_id = polygon_ids[i];
      const auto& polygon = polygons[poly_id];

      // entire warp sequentially visit each ring
      for (uint32_t ring_idx = 0; ring_idx < polygon.num_rings(); ring_idx++) {
        auto ring = polygon.get_ring(ring_idx);
        auto aabbs_per_ring = (ring.num_segments() + segs_per_aabb - 1) / segs_per_aabb;
        // e.g., num segs = 3, segs_per_aabb = 2
        // The first aabb covers seg 0,1, with vertex id (0,1,2)
        // The second aabb covers seg 2, with vertex id (2,3)
        // each lane takes an aabb
        for (auto aabb_idx = lane; aabb_idx < aabbs_per_ring; aabb_idx += 32) {
          INDEX_T local_vertex_begin = aabb_idx * segs_per_aabb;
          INDEX_T local_vertex_end =
              std::min((INDEX_T)(local_vertex_begin + segs_per_aabb),
                       (INDEX_T)ring.num_segments());

          auto tail = atomicAdd(p_tail, 1);

          assert(tail < num_aabbs);
          p_aabb_poly_ids[tail] = poly_id;
          p_aabb_ring_ids[tail] = ring_idx;
          p_aabb_vertex_offsets[tail] =
              thrust::make_pair(local_vertex_begin, local_vertex_end);
        }
      }
    }
  });
  rmm::device_uvector<OptixAabb> aabbs(num_aabbs, stream);

  // Fill AABBs
  thrust::transform(rmm::exec_policy_nosync(stream),
                    thrust::make_counting_iterator<uint32_t>(0),
                    thrust::make_counting_iterator<uint32_t>(num_aabbs), aabbs.begin(),
                    [=] __device__(const uint32_t& aabb_idx) {
                      OptixAabb aabb;
                      aabb.minX = std::numeric_limits<scalar_t>::max();
                      aabb.minY = std::numeric_limits<scalar_t>::max();
                      aabb.maxX = std::numeric_limits<scalar_t>::lowest();
                      aabb.maxY = std::numeric_limits<scalar_t>::lowest();

                      auto poly_id = p_aabb_poly_ids[aabb_idx];
                      auto ring_id = p_aabb_ring_ids[aabb_idx];
                      auto vertex_offset_pair = p_aabb_vertex_offsets[aabb_idx];
                      const auto& polygon = polygons[poly_id];
                      const auto& ring = polygon.get_ring(ring_id);

                      for (auto vidx = vertex_offset_pair.first;
                           vidx <= vertex_offset_pair.second; vidx++) {
                        const auto& v = ring.get_point(vidx);
                        float x = v.x();
                        float y = v.y();

                        aabb.minX = fminf(aabb.minX, x);
                        aabb.maxX = fmaxf(aabb.maxX, x);
                        aabb.minY = fminf(aabb.minY, y);
                        aabb.maxY = fmaxf(aabb.maxY, y);
                      }

                      if (std::is_same_v<scalar_t, double>) {
                        aabb.minX = next_float_from_double(aabb.minX, -1, 2);
                        aabb.maxX = next_float_from_double(aabb.maxX, 1, 2);
                        aabb.minY = next_float_from_double(aabb.minY, -1, 2);
                        aabb.maxY = next_float_from_double(aabb.maxY, 1, 2);
                      }
                      // Using minZ/maxZ to store polygon id for better filtering
                      // Refer to polygon_point_query.cu
                      aabb.minZ = aabb.maxZ = poly_id;
                      return aabb;
                    });

  assert(rt_engine_ != nullptr);
  return rt_engine_->BuildAccelCustom(stream.value(), ArrayView<OptixAabb>(aabbs), buffer,
                                      config_.bvh_fast_build, config_.bvh_compact);
}

template <typename POINT_T, typename INDEX_T>
OptixTraversableHandle RelateEngine<POINT_T, INDEX_T>::BuildBVH(
    const rmm::cuda_stream_view& stream,
    const MultiPolygonArrayView<POINT_T, INDEX_T>& multi_polys,
    ArrayView<uint32_t> multi_poly_ids, int segs_per_aabb, rmm::device_buffer& buffer,
    rmm::device_uvector<INDEX_T>& aabb_multi_poly_ids,
    rmm::device_uvector<INDEX_T>& aabb_part_ids,
    rmm::device_uvector<INDEX_T>& aabb_ring_ids,
    rmm::device_uvector<thrust::pair<INDEX_T, INDEX_T>>& aabb_vertex_offsets,
    rmm::device_uvector<INDEX_T>& part_begins) {
  auto n_mult_polygons = multi_poly_ids.size();

  auto num_aabbs =
      detail::ComputeNumAabbs(stream, multi_polys, multi_poly_ids, segs_per_aabb);
  if (num_aabbs == 0) {
    return 0;
  }

  aabb_multi_poly_ids = std::move(rmm::device_uvector<INDEX_T>(num_aabbs, stream));
  aabb_part_ids = std::move(rmm::device_uvector<uint32_t>(num_aabbs, stream));
  aabb_ring_ids = std::move(rmm::device_uvector<uint32_t>(num_aabbs, stream));
  aabb_vertex_offsets =
      std::move(rmm::device_uvector<thrust::pair<INDEX_T, INDEX_T>>(num_aabbs, stream));
  rmm::device_uvector<INDEX_T> aabb_seq_ids(num_aabbs, stream);

  auto* p_aabb_multi_poly_ids = aabb_multi_poly_ids.data();
  auto* p_aabb_part_ids = aabb_part_ids.data();
  auto* p_aabb_ring_ids = aabb_ring_ids.data();
  auto* p_aabb_vertex_offsets = aabb_vertex_offsets.data();
  auto* p_aabb_seq_ids = aabb_seq_ids.data();

  rmm::device_scalar<uint32_t> d_tail(0, stream);
  auto* p_tail = d_tail.data();

  LaunchKernel(stream.value(), [=] __device__() {
    auto lane = threadIdx.x % 32;
    auto global_warp_id = TID_1D / 32;
    auto n_warps = TOTAL_THREADS_1D / 32;

    // each warp takes a polygon
    for (auto i = global_warp_id; i < n_mult_polygons; i += n_warps) {
      auto multi_poly_id = multi_poly_ids[i];
      const auto& multi_polygon = multi_polys[multi_poly_id];

      for (uint32_t part_idx = 0; part_idx < multi_polygon.num_polygons(); part_idx++) {
        auto polygon = multi_polygon.get_polygon(part_idx);
        // entire warp sequentially visit each ring
        for (uint32_t ring_idx = 0; ring_idx < polygon.num_rings(); ring_idx++) {
          auto ring = polygon.get_ring(ring_idx);
          auto aabbs_per_ring = (ring.num_segments() + segs_per_aabb - 1) / segs_per_aabb;
          // e.g., num segs = 3, segs_per_aabb = 2
          // The first aabb covers seg 0,1, with vertex id (0,1,2)
          // The second aabb covers seg 2, with vertex id (2,3)
          // each lane takes an aabb
          for (auto aabb_idx = lane; aabb_idx < aabbs_per_ring; aabb_idx += 32) {
            INDEX_T local_vertex_begin = aabb_idx * segs_per_aabb;
            INDEX_T local_vertex_end =
                std::min((INDEX_T)(local_vertex_begin + segs_per_aabb),
                         (INDEX_T)ring.num_segments());

            auto tail = atomicAdd(p_tail, 1);

            assert(tail < num_aabbs);
            p_aabb_multi_poly_ids[tail] = multi_poly_id;
            p_aabb_part_ids[tail] = part_idx;
            p_aabb_ring_ids[tail] = ring_idx;
            p_aabb_vertex_offsets[tail] =
                thrust::make_pair(local_vertex_begin, local_vertex_end);
            p_aabb_seq_ids[tail] = i;
          }
        }
      }
    }
  });

  rmm::device_uvector<OptixAabb> aabbs(num_aabbs, stream);
  part_begins = std::move(rmm::device_uvector<uint32_t>(n_mult_polygons + 1, stream));
  auto* p_part_begins = part_begins.data();
  part_begins.set_element_to_zero_async(0, stream);
  rmm::device_uvector<uint32_t> num_parts(n_mult_polygons, stream);

  thrust::transform(rmm::exec_policy_nosync(stream), multi_poly_ids.begin(),
                    multi_poly_ids.end(), num_parts.begin(), [=] __device__(uint32_t id) {
                      const auto& multi_polygon = multi_polys[id];
                      return multi_polygon.num_polygons();
                    });

  thrust::inclusive_scan(rmm::exec_policy_nosync(stream), num_parts.begin(),
                         num_parts.end(), part_begins.begin() + 1);
  num_parts.resize(0, stream);
  num_parts.shrink_to_fit(stream);

  // Fill AABBs
  thrust::transform(rmm::exec_policy_nosync(stream),
                    thrust::make_counting_iterator<uint32_t>(0),
                    thrust::make_counting_iterator<uint32_t>(num_aabbs), aabbs.begin(),
                    [=] __device__(const uint32_t& aabb_idx) {
                      OptixAabb aabb;
                      aabb.minX = std::numeric_limits<scalar_t>::max();
                      aabb.minY = std::numeric_limits<scalar_t>::max();
                      aabb.maxX = std::numeric_limits<scalar_t>::lowest();
                      aabb.maxY = std::numeric_limits<scalar_t>::lowest();

                      auto multi_poly_id = p_aabb_multi_poly_ids[aabb_idx];
                      auto part_id = p_aabb_part_ids[aabb_idx];
                      auto ring_id = p_aabb_ring_ids[aabb_idx];
                      auto vertex_offset_pair = p_aabb_vertex_offsets[aabb_idx];
                      auto seq_id = p_aabb_seq_ids[aabb_idx];
                      auto multi_polygon = multi_polys[multi_poly_id];
                      const auto& polygon = multi_polygon.get_polygon(part_id);
                      const auto& ring = polygon.get_ring(ring_id);

                      for (auto vidx = vertex_offset_pair.first;
                           vidx <= vertex_offset_pair.second; vidx++) {
                        const auto& v = ring.get_point(vidx);
                        float x = v.x();
                        float y = v.y();

                        aabb.minX = fminf(aabb.minX, x);
                        aabb.maxX = fmaxf(aabb.maxX, x);
                        aabb.minY = fminf(aabb.minY, y);
                        aabb.maxY = fmaxf(aabb.maxY, y);
                      }

                      if (std::is_same_v<scalar_t, double>) {
                        aabb.minX = next_float_from_double(aabb.minX, -1, 2);
                        aabb.maxX = next_float_from_double(aabb.maxX, 1, 2);
                        aabb.minY = next_float_from_double(aabb.minY, -1, 2);
                        aabb.maxY = next_float_from_double(aabb.maxY, 1, 2);
                      }

                      aabb.minZ = aabb.maxZ = p_part_begins[seq_id] + part_id;
                      return aabb;
                    });
  assert(rt_engine_ != nullptr);

  return rt_engine_->BuildAccelCustom(stream.value(), ArrayView<OptixAabb>(aabbs), buffer,
                                      config_.bvh_fast_build, config_.bvh_compact);
}

// Explicit template instantiations
template size_t RelateEngine<Point<double, 2>, uint32_t>::EstimateBVHSize(
    const rmm::cuda_stream_view& stream,
    const PolygonArrayView<Point<double, 2>, uint32_t>& polys,
    ArrayView<uint32_t> poly_ids, int segs_per_aabb);

template size_t RelateEngine<Point<double, 2>, uint32_t>::EstimateBVHSize(
    const rmm::cuda_stream_view& stream,
    const MultiPolygonArrayView<Point<double, 2>, uint32_t>& multi_polys,
    ArrayView<uint32_t> multi_poly_ids, int segs_per_aabb);

template OptixTraversableHandle RelateEngine<Point<double, 2>, uint32_t>::BuildBVH(
    const rmm::cuda_stream_view& stream,
    const PolygonArrayView<Point<double, 2>, uint32_t>& polygons,
    ArrayView<uint32_t> polygon_ids, int segs_per_aabb, rmm::device_buffer& buffer,
    rmm::device_uvector<uint32_t>& aabb_poly_ids,
    rmm::device_uvector<uint32_t>& aabb_ring_ids,
    rmm::device_uvector<thrust::pair<uint32_t, uint32_t>>& aabb_vertex_offsets);

template OptixTraversableHandle RelateEngine<Point<double, 2>, uint32_t>::BuildBVH(
    const rmm::cuda_stream_view& stream,
    const MultiPolygonArrayView<Point<double, 2>, uint32_t>& multi_polys,
    ArrayView<uint32_t> multi_poly_ids, int segs_per_aabb, rmm::device_buffer& buffer,
    rmm::device_uvector<uint32_t>& aabb_multi_poly_ids,
    rmm::device_uvector<uint32_t>& aabb_part_ids,
    rmm::device_uvector<uint32_t>& aabb_ring_ids,
    rmm::device_uvector<thrust::pair<uint32_t, uint32_t>>& aabb_vertex_offsets,
    rmm::device_uvector<uint32_t>& part_begins);

}  // namespace gpuspatial
