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
#include "gpuspatial/relate/predicate.cuh"
#include "gpuspatial/relate/relate.cuh"
#include "gpuspatial/relate/relate_engine.cuh"
#include "gpuspatial/rt/launch_parameters.h"
#include "gpuspatial/utils/array_view.h"
#include "gpuspatial/utils/helpers.h"
#include "gpuspatial/utils/launcher.h"
#include "gpuspatial/utils/logger.hpp"
#include "rt/shaders/shader_id.hpp"

#include "rmm/cuda_stream_view.hpp"
#include "rmm/exec_policy.hpp"

#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "gpuspatial/utils/stopwatch.h"

namespace gpuspatial {
namespace detail {
DEV_HOST_INLINE bool EvaluatePredicate(Predicate p, int32_t im) {
  switch (p) {
    case Predicate::kEquals: {
      return (im & IntersectionMatrix::INTER_INTER_2D) != 0 &&
             (im & IntersectionMatrix::INTER_EXTER_2D) == 0 &&
             (im & IntersectionMatrix::BOUND_EXTER_2D) == 0 &&
             (im & IntersectionMatrix::EXTER_INTER_2D) == 0 &&
             (im & IntersectionMatrix::EXTER_BOUND_2D) == 0;
    }
    case Predicate::kDisjoint: {
      return (im & IntersectionMatrix::INTER_INTER_2D) == 0 &&
             (im & IntersectionMatrix::INTER_BOUND_2D) == 0 &&
             (im & IntersectionMatrix::BOUND_INTER_2D) == 0 &&
             (im & IntersectionMatrix::BOUND_BOUND_2D) == 0;
    }
    case Predicate::kTouches: {
      return (im & IntersectionMatrix::INTER_INTER_2D) == 0 &&
             ((im & IntersectionMatrix::INTER_BOUND_2D) != 0 ||
              (im & IntersectionMatrix::BOUND_INTER_2D) != 0 ||
              (im & IntersectionMatrix::BOUND_BOUND_2D) != 0);
    }
    case Predicate::kContains: {
      return (im & IntersectionMatrix::INTER_INTER_2D) != 0 &&
             (im & IntersectionMatrix::EXTER_INTER_2D) == 0 &&
             (im & IntersectionMatrix::EXTER_BOUND_2D) == 0;
    }
    case Predicate::kCovers: {
      return (im & IntersectionMatrix::EXTER_INTER_2D) == 0 &&
             (im & IntersectionMatrix::EXTER_BOUND_2D) == 0 &&
             ((im & IntersectionMatrix::INTER_INTER_2D) != 0 ||
              (im & IntersectionMatrix::INTER_BOUND_2D) != 0 ||
              (im & IntersectionMatrix::BOUND_INTER_2D) != 0 ||
              (im & IntersectionMatrix::BOUND_BOUND_2D) != 0);
    }
    case Predicate::kIntersects: {
      return (im & IntersectionMatrix::INTER_INTER_2D) != 0 ||
             (im & IntersectionMatrix::INTER_BOUND_2D) != 0 ||
             (im & IntersectionMatrix::BOUND_INTER_2D) != 0 ||
             (im & IntersectionMatrix::BOUND_BOUND_2D) != 0;
    }
    case Predicate::kWithin: {
      return (im & IntersectionMatrix::INTER_INTER_2D) != 0 &&
             (im & IntersectionMatrix::INTER_EXTER_2D) == 0 &&
             (im & IntersectionMatrix::BOUND_EXTER_2D) == 0;
    }
    case Predicate::kCoveredBy: {
      return (im & IntersectionMatrix::INTER_EXTER_2D) == 0 &&
             (im & IntersectionMatrix::BOUND_EXTER_2D) == 0 &&
             ((im & IntersectionMatrix::INTER_INTER_2D) != 0 ||
              (im & IntersectionMatrix::INTER_BOUND_2D) != 0 ||
              (im & IntersectionMatrix::BOUND_INTER_2D) != 0 ||
              (im & IntersectionMatrix::BOUND_BOUND_2D) != 0);
    }
    default:
      assert(false);
  }
  return false;
}

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
RelateEngine<POINT_T, INDEX_T>::RelateEngine(
    const DeviceGeometries<POINT_T, INDEX_T>* geoms1)
    : geoms1_(geoms1) {}

template <typename POINT_T, typename INDEX_T>
RelateEngine<POINT_T, INDEX_T>::RelateEngine(
    const DeviceGeometries<POINT_T, INDEX_T>* geoms1, const RTEngine* rt_engine)
    : geoms1_(geoms1), rt_engine_(rt_engine) {}

template <typename POINT_T, typename INDEX_T>
void RelateEngine<POINT_T, INDEX_T>::Evaluate(
    const rmm::cuda_stream_view& stream, const DeviceGeometries<POINT_T, INDEX_T>& geoms2,
    Predicate predicate, rmm::device_uvector<INDEX_T>& ids1,
    rmm::device_uvector<INDEX_T>& ids2) {
  switch (geoms2.get_geometry_type()) {
    case GeometryType::kPoint: {
      using geom2_array_view_t = PointArrayView<POINT_T, INDEX_T>;
      Evaluate(stream, geoms2.template GetGeometryArrayView<geom2_array_view_t>(),
               predicate, ids1, ids2);
      break;
    }
    case GeometryType::kMultiPoint: {
      using geom2_array_view_t = MultiPointArrayView<POINT_T, INDEX_T>;
      Evaluate(stream, geoms2.template GetGeometryArrayView<geom2_array_view_t>(),
               predicate, ids1, ids2);
      break;
    }
    case GeometryType::kLineString: {
      using geom2_array_view_t = LineStringArrayView<POINT_T, INDEX_T>;
      Evaluate(stream, geoms2.template GetGeometryArrayView<geom2_array_view_t>(),
               predicate, ids1, ids2);
      break;
    }
    case GeometryType::kMultiLineString: {
      using geom2_array_view_t = MultiLineStringArrayView<POINT_T, INDEX_T>;
      Evaluate(stream, geoms2.template GetGeometryArrayView<geom2_array_view_t>(),
               predicate, ids1, ids2);
      break;
    }
    case GeometryType::kPolygon: {
      using geom2_array_view_t = PolygonArrayView<POINT_T, INDEX_T>;
      Evaluate(stream, geoms2.template GetGeometryArrayView<geom2_array_view_t>(),
               predicate, ids1, ids2);
      break;
    }

    case GeometryType::kMultiPolygon: {
      using geom2_array_view_t = MultiPolygonArrayView<POINT_T, INDEX_T>;
      Evaluate(stream, geoms2.template GetGeometryArrayView<geom2_array_view_t>(),
               predicate, ids1, ids2);
      break;
    }
    default:
      assert(false);
  }
}

template <typename POINT_T, typename INDEX_T>
template <typename GEOM2_ARRAY_VIEW_T>
void RelateEngine<POINT_T, INDEX_T>::Evaluate(const rmm::cuda_stream_view& stream,
                                              const GEOM2_ARRAY_VIEW_T& geom_array2,
                                              Predicate predicate,
                                              rmm::device_uvector<INDEX_T>& ids1,
                                              rmm::device_uvector<INDEX_T>& ids2) {
  switch (geoms1_->get_geometry_type()) {
    case GeometryType::kPoint: {
      using geom1_array_view_t = PointArrayView<POINT_T, INDEX_T>;
      Evaluate(stream, geoms1_->template GetGeometryArrayView<geom1_array_view_t>(),
               geom_array2, predicate, ids1, ids2);
      break;
    }
    case GeometryType::kMultiPoint: {
      using geom1_array_view_t = MultiPointArrayView<POINT_T, INDEX_T>;
      Evaluate(stream, geoms1_->template GetGeometryArrayView<geom1_array_view_t>(),
               geom_array2, predicate, ids1, ids2);
      break;
    }
    case GeometryType::kLineString: {
      using geom1_array_view_t = LineStringArrayView<POINT_T, INDEX_T>;
      Evaluate(stream, geoms1_->template GetGeometryArrayView<geom1_array_view_t>(),
               geom_array2, predicate, ids1, ids2);
      break;
    }
    case GeometryType::kMultiLineString: {
      using geom1_array_view_t = MultiLineStringArrayView<POINT_T, INDEX_T>;
      Evaluate(stream, geoms1_->template GetGeometryArrayView<geom1_array_view_t>(),
               geom_array2, predicate, ids1, ids2);
      break;
    }
    case GeometryType::kPolygon: {
      using geom1_array_view_t = PolygonArrayView<POINT_T, INDEX_T>;
      Evaluate(stream, geoms1_->template GetGeometryArrayView<geom1_array_view_t>(),
               geom_array2, predicate, ids1, ids2);
      break;
    }

    case GeometryType::kMultiPolygon: {
      using geom1_array_view_t = MultiPolygonArrayView<POINT_T, INDEX_T>;
      Evaluate(stream, geoms1_->template GetGeometryArrayView<geom1_array_view_t>(),
               geom_array2, predicate, ids1, ids2);
      break;
    }
    default:
      assert(false);
  }
}

template <typename POINT_T, typename INDEX_T>
template <typename GEOM1_ARRAY_VIEW_T, typename GEOM2_ARRAY_VIEW_T>
void RelateEngine<POINT_T, INDEX_T>::Evaluate(const rmm::cuda_stream_view& stream,
                                              const GEOM1_ARRAY_VIEW_T& geom_array1,
                                              const GEOM2_ARRAY_VIEW_T& geom_array2,
                                              Predicate predicate,
                                              rmm::device_uvector<INDEX_T>& ids1,
                                              rmm::device_uvector<INDEX_T>& ids2) {
  assert(ids1.size() == ids2.size());
  size_t ids_size = ids1.size();
  GPUSPATIAL_LOG_INFO(
      "Refine with generic kernel, geom1 %zu, geom2 %zu, predicate %s, result size %zu",
      geom_array1.size(), geom_array2.size(), PredicateToString(predicate), ids_size);
  if (std::is_same_v<GEOM1_ARRAY_VIEW_T, PolygonArrayView<POINT_T, INDEX_T>> &&
          std::is_same_v<GEOM2_ARRAY_VIEW_T, PolygonArrayView<POINT_T, INDEX_T>> ||
      std::is_same_v<GEOM1_ARRAY_VIEW_T, PolygonArrayView<POINT_T, INDEX_T>> &&
          std::is_same_v<GEOM2_ARRAY_VIEW_T, MultiPolygonArrayView<POINT_T, INDEX_T>> ||
      std::is_same_v<GEOM1_ARRAY_VIEW_T, MultiPolygonArrayView<POINT_T, INDEX_T>> &&
          std::is_same_v<GEOM2_ARRAY_VIEW_T, PolygonArrayView<POINT_T, INDEX_T>> ||
      std::is_same_v<GEOM1_ARRAY_VIEW_T, MultiPolygonArrayView<POINT_T, INDEX_T>> &&
          std::is_same_v<GEOM2_ARRAY_VIEW_T, MultiPolygonArrayView<POINT_T, INDEX_T>>) {
    GPUSPATIAL_LOG_WARN(
        "Evaluate Polygon-Polygon relate with the GPU, which is not well-tested and the performance may be poor.");
  }
  auto zip_begin =
      thrust::make_zip_iterator(thrust::make_tuple(ids1.begin(), ids2.begin()));
  auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(ids1.end(), ids2.end()));

  auto end =
      thrust::remove_if(rmm::exec_policy_nosync(stream), zip_begin, zip_end,
                        [=] __device__(const thrust::tuple<INDEX_T, INDEX_T>& tuple) {
                          auto geom1_id = thrust::get<0>(tuple);
                          auto geom2_id = thrust::get<1>(tuple);
                          const auto& geom1 = geom_array1[geom1_id];
                          const auto& geom2 = geom_array2[geom2_id];

                          auto IM = relate(geom1, geom2);
                          return !detail::EvaluatePredicate(predicate, IM);
                        });
  size_t new_size = thrust::distance(zip_begin, end);
  ids1.resize(new_size, stream);
  ids2.resize(new_size, stream);
}

template <typename POINT_T, typename INDEX_T>
void RelateEngine<POINT_T, INDEX_T>::Evaluate(
    const rmm::cuda_stream_view& stream,
    const PointArrayView<POINT_T, INDEX_T>& geom_array1,
    const PolygonArrayView<POINT_T, INDEX_T>& geom_array2, Predicate predicate,
    rmm::device_uvector<INDEX_T>& ids1, rmm::device_uvector<INDEX_T>& ids2) {
  EvaluateImpl(stream, geom_array1, MultiPointArrayView<POINT_T, INDEX_T>(), geom_array2,
               predicate, ids1, ids2, false /*inverse IM*/);
}

template <typename POINT_T, typename INDEX_T>
void RelateEngine<POINT_T, INDEX_T>::Evaluate(
    const rmm::cuda_stream_view& stream,
    const MultiPointArrayView<POINT_T, INDEX_T>& geom_array1,
    const PolygonArrayView<POINT_T, INDEX_T>& geom_array2, Predicate predicate,
    rmm::device_uvector<INDEX_T>& ids1, rmm::device_uvector<INDEX_T>& ids2) {
  EvaluateImpl(stream, PointArrayView<POINT_T, INDEX_T>(), geom_array1, geom_array2,
               predicate, ids1, ids2, false /*inverse IM*/);
}

template <typename POINT_T, typename INDEX_T>
void RelateEngine<POINT_T, INDEX_T>::Evaluate(
    const rmm::cuda_stream_view& stream,
    const PolygonArrayView<POINT_T, INDEX_T>& geom_array1,
    const PointArrayView<POINT_T, INDEX_T>& geom_array2, Predicate predicate,
    rmm::device_uvector<INDEX_T>& ids1, rmm::device_uvector<INDEX_T>& ids2) {
  EvaluateImpl(stream, geom_array2, MultiPointArrayView<POINT_T, INDEX_T>(), geom_array1,
               predicate, ids2, ids1, true /*inverse IM*/);
}

template <typename POINT_T, typename INDEX_T>
void RelateEngine<POINT_T, INDEX_T>::Evaluate(
    const rmm::cuda_stream_view& stream,
    const PolygonArrayView<POINT_T, INDEX_T>& geom_array1,
    const MultiPointArrayView<POINT_T, INDEX_T>& geom_array2, Predicate predicate,
    rmm::device_uvector<INDEX_T>& ids1, rmm::device_uvector<INDEX_T>& ids2) {
  EvaluateImpl(stream, PointArrayView<POINT_T, INDEX_T>(), geom_array2, geom_array1,
               predicate, ids2, ids1, true /*inverse IM*/);
}

template <typename POINT_T, typename INDEX_T>
void RelateEngine<POINT_T, INDEX_T>::Evaluate(
    const rmm::cuda_stream_view& stream,
    const PointArrayView<POINT_T, INDEX_T>& geom_array1,
    const MultiPolygonArrayView<POINT_T, INDEX_T>& geom_array2, Predicate predicate,
    rmm::device_uvector<INDEX_T>& ids1, rmm::device_uvector<INDEX_T>& ids2) {
  EvaluateImpl(stream, geom_array1, MultiPointArrayView<POINT_T, INDEX_T>(), geom_array2,
               predicate, ids1, ids2, false /*inverse IM*/);
}

template <typename POINT_T, typename INDEX_T>
void RelateEngine<POINT_T, INDEX_T>::Evaluate(
    const rmm::cuda_stream_view& stream,
    const MultiPointArrayView<POINT_T, INDEX_T>& geom_array1,
    const MultiPolygonArrayView<POINT_T, INDEX_T>& geom_array2, Predicate predicate,
    rmm::device_uvector<INDEX_T>& ids1, rmm::device_uvector<INDEX_T>& ids2) {
  EvaluateImpl(stream, PointArrayView<POINT_T, INDEX_T>(), geom_array1, geom_array2,
               predicate, ids1, ids2, false /*inverse IM*/);
}

template <typename POINT_T, typename INDEX_T>
void RelateEngine<POINT_T, INDEX_T>::Evaluate(
    const rmm::cuda_stream_view& stream,
    const MultiPolygonArrayView<POINT_T, INDEX_T>& geom_array1,
    const PointArrayView<POINT_T, INDEX_T>& geom_array2, Predicate predicate,
    rmm::device_uvector<INDEX_T>& ids1, rmm::device_uvector<INDEX_T>& ids2) {
  EvaluateImpl(stream, geom_array2, MultiPointArrayView<POINT_T, INDEX_T>(), geom_array1,
               predicate, ids2, ids1, true /*inverse IM*/);
}

template <typename POINT_T, typename INDEX_T>
void RelateEngine<POINT_T, INDEX_T>::Evaluate(
    const rmm::cuda_stream_view& stream,
    const MultiPolygonArrayView<POINT_T, INDEX_T>& geom_array1,
    const MultiPointArrayView<POINT_T, INDEX_T>& geom_array2, Predicate predicate,
    rmm::device_uvector<INDEX_T>& ids1, rmm::device_uvector<INDEX_T>& ids2) {
  EvaluateImpl(stream, PointArrayView<POINT_T, INDEX_T>(), geom_array2, geom_array1,
               predicate, ids2, ids1, true /*inverse IM*/);
}

template <typename POINT_T, typename INDEX_T>
void RelateEngine<POINT_T, INDEX_T>::EvaluateImpl(
    const rmm::cuda_stream_view& stream,
    const PointArrayView<POINT_T, INDEX_T>& point_array,
    const MultiPointArrayView<POINT_T, INDEX_T>& multi_point_array,
    const PolygonArrayView<POINT_T, INDEX_T>& poly_array, Predicate predicate,
    rmm::device_uvector<INDEX_T>& point_ids, rmm::device_uvector<INDEX_T>& poly_ids,
    bool inverse) {
  using params_t = detail::LaunchParamsPolygonPointQuery<POINT_T, INDEX_T>;
  assert(point_array.empty() || multi_point_array.empty());
  assert(point_ids.size() == poly_ids.size());
  size_t ids_size = point_ids.size();
  GPUSPATIAL_LOG_INFO(
      "Refine with ray-tracing, (multi-)point %zu, polygon %zu, predicate %s, result size %zu, inverse %d",
      !point_array.empty() ? point_array.size() : multi_point_array.size(),
      poly_array.size(), PredicateToString(predicate), ids_size, inverse);

  if (ids_size == 0) {
    return;
  }

  auto zip_begin =
      thrust::make_zip_iterator(thrust::make_tuple(point_ids.begin(), poly_ids.begin()));
  auto zip_end =
      thrust::make_zip_iterator(thrust::make_tuple(point_ids.end(), poly_ids.end()));
  auto invalid_tuple = thrust::make_tuple(std::numeric_limits<INDEX_T>::max(),
                                          std::numeric_limits<INDEX_T>::max());

  // Sort by polygon id
  thrust::sort(rmm::exec_policy_nosync(stream), zip_begin, zip_end,
               [] __device__(const thrust::tuple<INDEX_T, INDEX_T>& tu1,
                             const thrust::tuple<INDEX_T, INDEX_T>& tu2) {
                 return thrust::get<1>(tu1) < thrust::get<1>(tu2);
               });

  rmm::device_uvector<INDEX_T> uniq_poly_ids(ids_size, stream);

  thrust::copy(rmm::exec_policy_nosync(stream), poly_ids.begin(), poly_ids.end(),
               uniq_poly_ids.begin());

  // Collect uniq polygon ids to estimate total BVH memory usage
  auto uniq_poly_ids_end = thrust::unique(rmm::exec_policy_nosync(stream),
                                          uniq_poly_ids.begin(), uniq_poly_ids.end());
  uniq_poly_ids.resize(thrust::distance(uniq_poly_ids.begin(), uniq_poly_ids_end),
                       stream);
  uniq_poly_ids.shrink_to_fit(stream);

  auto bvh_bytes = EstimateBVHSize(stream, poly_array, ArrayView<uint32_t>(uniq_poly_ids),
                                   config_.segs_per_aabb);
  size_t avail_bytes = rmm::available_device_memory().first * config_.memory_quota;
  auto n_batches = bvh_bytes / avail_bytes + 1;
  auto batch_size = (ids_size + n_batches - 1) / n_batches;

  GPUSPATIAL_LOG_INFO(
      "Unique polygons %zu, memory quota %zu MB, estimated BVH size %zu MB",
      uniq_poly_ids.size(), avail_bytes / (1024 * 1024), bvh_bytes / (1024 * 1024));

  for (int batch = 0; batch < n_batches; batch++) {
    auto ids_begin = batch * batch_size;
    auto ids_end = std::min(ids_begin + batch_size, ids_size);
    auto ids_size_batch = ids_end - ids_begin;

    // Extract unique polygon IDs in this batch
    uniq_poly_ids.resize(ids_size_batch, stream);
    thrust::copy(rmm::exec_policy_nosync(stream), poly_ids.begin() + ids_begin,
                 poly_ids.begin() + ids_end, uniq_poly_ids.begin());

    // poly ids are sorted
    uniq_poly_ids_end = thrust::unique(rmm::exec_policy_nosync(stream),
                                       uniq_poly_ids.begin(), uniq_poly_ids.end());

    uniq_poly_ids.resize(thrust::distance(uniq_poly_ids.begin(), uniq_poly_ids_end),
                         stream);
    uniq_poly_ids.shrink_to_fit(stream);

    rmm::device_uvector<int> IMs(ids_size_batch, stream);
    rmm::device_uvector<PointLocation> locations(ids_size_batch, stream);
    rmm::device_buffer bvh_buffer(0, stream);
    rmm::device_uvector<INDEX_T> aabb_poly_ids(0, stream), aabb_ring_ids(0, stream);
    rmm::device_uvector<thrust::pair<INDEX_T, INDEX_T>> aabb_vertex_offsets(0, stream);

    // aabb id -> vertex begin[polygon] + ith point in this polygon
    auto handle = BuildBVH(stream, poly_array, ArrayView<INDEX_T>(uniq_poly_ids),
                           config_.segs_per_aabb, bvh_buffer, aabb_poly_ids,
                           aabb_ring_ids, aabb_vertex_offsets);

    params_t params;

    params.points = point_array;
    params.multi_points = multi_point_array;
    params.polygons = poly_array;
    params.uniq_polygon_ids = ArrayView<INDEX_T>(uniq_poly_ids);
    params.query_point_ids = point_ids.data() + ids_begin;
    params.query_polygon_ids = poly_ids.data() + ids_begin;
    params.query_size = ids_size_batch;
    params.IMs = ArrayView<int>(IMs);
    params.handle = handle;
    params.aabb_poly_ids = ArrayView<INDEX_T>(aabb_poly_ids);
    params.aabb_ring_ids = ArrayView<INDEX_T>(aabb_ring_ids);
    params.aabb_vertex_offsets =
        ArrayView<thrust::pair<INDEX_T, INDEX_T>>(aabb_vertex_offsets);

    rmm::device_buffer params_buffer(sizeof(params_t), stream);

    CUDA_CHECK(cudaMemcpyAsync(params_buffer.data(), &params, sizeof(params_t),
                               cudaMemcpyHostToDevice, stream.value()));

    rt_engine_->Render(
        stream, GetPolygonPointQueryShaderId<POINT_T>(),
        dim3{static_cast<unsigned int>(ids_size_batch), 1, 1},
        ArrayView<char>((char*)params_buffer.data(), params_buffer.size()));

    thrust::transform(
        rmm::exec_policy_nosync(stream),
        thrust::make_zip_iterator(thrust::make_tuple(
            point_ids.begin() + ids_begin, poly_ids.begin() + ids_begin, IMs.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(
            point_ids.begin() + ids_end, poly_ids.begin() + ids_end, IMs.end())),
        thrust::make_zip_iterator(thrust::make_tuple(point_ids.begin() + ids_begin,
                                                     poly_ids.begin() + ids_begin)),
        [=] __device__(const thrust::tuple<INDEX_T, INDEX_T, int>& t) {
          auto res = thrust::make_tuple(thrust::get<0>(t), thrust::get<1>(t));
          auto IM = thrust::get<2>(t);

          if (inverse) {
            IM = IntersectionMatrix::Transpose(IM);
          }

          return detail::EvaluatePredicate(predicate, IM) ? res : invalid_tuple;
        });
  }
  auto end = thrust::remove_if(rmm::exec_policy_nosync(stream), zip_begin, zip_end,
                               [=] __device__(const thrust::tuple<INDEX_T, INDEX_T>& tu) {
                                 return tu == invalid_tuple;
                               });
  size_t new_size = thrust::distance(zip_begin, end);
  point_ids.resize(new_size, stream);
  poly_ids.resize(new_size, stream);
}

template <typename POINT_T, typename INDEX_T>
void RelateEngine<POINT_T, INDEX_T>::EvaluateImpl(
    const rmm::cuda_stream_view& stream,
    const PointArrayView<POINT_T, INDEX_T>& point_array,
    const MultiPointArrayView<POINT_T, INDEX_T>& multi_point_array,
    const MultiPolygonArrayView<POINT_T, INDEX_T>& multi_poly_array, Predicate predicate,
    rmm::device_uvector<INDEX_T>& point_ids, rmm::device_uvector<INDEX_T>& multi_poly_ids,
    bool inverse) {
  using params_t = detail::LaunchParamsPointMultiPolygonQuery<POINT_T, INDEX_T>;
  assert(point_array.empty() || multi_point_array.empty());
  assert(point_ids.size() == multi_poly_ids.size());
  size_t ids_size = point_ids.size();
  GPUSPATIAL_LOG_INFO(
      "Refine with ray-tracing, (multi-)point %zu, multi-polygon %zu, predicate %s, result size %zu, inverse %d",
      !point_array.empty() ? point_array.size() : multi_point_array.size(),
      multi_poly_array.size(), PredicateToString(predicate), ids_size, inverse);

  if (ids_size == 0) {
    return;
  }
  auto zip_begin = thrust::make_zip_iterator(
      thrust::make_tuple(point_ids.begin(), multi_poly_ids.begin()));
  auto zip_end = thrust::make_zip_iterator(
      thrust::make_tuple(point_ids.end(), multi_poly_ids.end()));
  auto invalid_tuple = thrust::make_tuple(std::numeric_limits<INDEX_T>::max(),
                                          std::numeric_limits<INDEX_T>::max());

  // Sort by polygon id
  thrust::sort(rmm::exec_policy_nosync(stream), zip_begin, zip_end,
               [] __device__(const thrust::tuple<INDEX_T, INDEX_T>& tu1,
                             const thrust::tuple<INDEX_T, INDEX_T>& tu2) {
                 return thrust::get<1>(tu1) < thrust::get<1>(tu2);
               });

  rmm::device_uvector<uint32_t> uniq_multi_poly_ids(ids_size, stream);

  thrust::copy(rmm::exec_policy_nosync(stream), multi_poly_ids.begin(),
               multi_poly_ids.end(), uniq_multi_poly_ids.begin());

  // Collect uniq polygon ids to estimate total BVH memory usage
  auto uniq_multi_poly_ids_end =
      thrust::unique(rmm::exec_policy_nosync(stream), uniq_multi_poly_ids.begin(),
                     uniq_multi_poly_ids.end());
  uniq_multi_poly_ids.resize(
      thrust::distance(uniq_multi_poly_ids.begin(), uniq_multi_poly_ids_end), stream);
  uniq_multi_poly_ids.shrink_to_fit(stream);

  auto bvh_bytes =
      EstimateBVHSize(stream, multi_poly_array, ArrayView<uint32_t>(uniq_multi_poly_ids),
                      config_.segs_per_aabb);
  size_t avail_bytes = rmm::available_device_memory().first * config_.memory_quota;
  auto n_batches = bvh_bytes / avail_bytes + 1;
  auto batch_size = (ids_size + n_batches - 1) / n_batches;

  GPUSPATIAL_LOG_INFO(
      "Unique multi-polygons %zu, memory quota %zu MB, estimated BVH size %zu MB",
      uniq_multi_poly_ids.size(), avail_bytes / (1024 * 1024), bvh_bytes / (1024 * 1024));
  double t_init = 0, t_compute_aabb = 0, t_build_bvh = 0, t_trace = 0, t_evaluate = 0;

  Stopwatch sw;

  for (int batch = 0; batch < n_batches; batch++) {
    auto ids_begin = batch * batch_size;
    auto ids_end = std::min(ids_begin + batch_size, ids_size);
    auto ids_size_batch = ids_end - ids_begin;

    sw.start();
    // Extract multi polygon IDs in this batch
    uniq_multi_poly_ids.resize(ids_size_batch, stream);

    thrust::copy(rmm::exec_policy_nosync(stream), multi_poly_ids.begin() + ids_begin,
                 multi_poly_ids.begin() + ids_end, uniq_multi_poly_ids.begin());

    // multi polygon ids have been sorted before
    uniq_multi_poly_ids_end =
        thrust::unique(rmm::exec_policy_nosync(stream), uniq_multi_poly_ids.begin(),
                       uniq_multi_poly_ids.end());
    uniq_multi_poly_ids.resize(
        thrust::distance(uniq_multi_poly_ids.begin(), uniq_multi_poly_ids_end), stream);
    uniq_multi_poly_ids.shrink_to_fit(stream);

    rmm::device_uvector<int> IMs(ids_size_batch, stream);
    rmm::device_buffer bvh_buffer(0, stream);
    rmm::device_uvector<INDEX_T> aabb_multi_poly_ids(0, stream), aabb_part_ids(0, stream),
        aabb_ring_ids(0, stream);
    rmm::device_uvector<thrust::pair<INDEX_T, INDEX_T>> aabb_vertex_offsets(0, stream);
    rmm::device_uvector<INDEX_T> uniq_part_begins(0, stream);
    stream.synchronize();
    sw.stop();
    t_init += sw.ms();

    auto handle =
        BuildBVH(stream, multi_poly_array, ArrayView<INDEX_T>(uniq_multi_poly_ids),
                 config_.segs_per_aabb, bvh_buffer, aabb_multi_poly_ids, aabb_part_ids,
                 aabb_ring_ids, aabb_vertex_offsets, uniq_part_begins, t_compute_aabb,
                 t_build_bvh);
    sw.start();

    params_t params;

    params.points = point_array;
    params.multi_points = multi_point_array;
    params.multi_polygons = multi_poly_array;
    params.uniq_multi_polygon_ids = ArrayView<INDEX_T>(uniq_multi_poly_ids);
    params.query_point_ids = point_ids.data() + ids_begin;
    params.query_multi_polygon_ids = multi_poly_ids.data() + ids_begin;
    params.query_size = ids_size_batch;
    params.uniq_part_begins = ArrayView<INDEX_T>(uniq_part_begins);
    params.IMs = ArrayView<int>(IMs);
    params.handle = handle;
    params.aabb_multi_poly_ids = ArrayView<INDEX_T>(aabb_multi_poly_ids);
    params.aabb_part_ids = ArrayView<INDEX_T>(aabb_part_ids);
    params.aabb_ring_ids = ArrayView<INDEX_T>(aabb_ring_ids);
    params.aabb_vertex_offsets =
        ArrayView<thrust::pair<INDEX_T, INDEX_T>>(aabb_vertex_offsets);

    rmm::device_buffer params_buffer(sizeof(params_t), stream);

    CUDA_CHECK(cudaMemcpyAsync(params_buffer.data(), &params, sizeof(params_t),
                               cudaMemcpyHostToDevice, stream.value()));

    rt_engine_->Render(
        stream, GetMultiPolygonPointQueryShaderId<POINT_T>(),
        dim3{static_cast<unsigned int>(ids_size_batch), 1, 1},
        ArrayView<char>((char*)params_buffer.data(), params_buffer.size()));
    stream.synchronize();
    sw.stop();
    t_trace += sw.ms();
    sw.start();
    thrust::transform(
        rmm::exec_policy_nosync(stream),
        thrust::make_zip_iterator(thrust::make_tuple(point_ids.begin() + ids_begin,
                                                     multi_poly_ids.begin() + ids_begin,
                                                     IMs.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(
            point_ids.begin() + ids_end, multi_poly_ids.begin() + ids_end, IMs.end())),
        thrust::make_zip_iterator(thrust::make_tuple(point_ids.begin() + ids_begin,
                                                     multi_poly_ids.begin() + ids_begin)),
        [=] __device__(const thrust::tuple<INDEX_T, INDEX_T, int>& t) {
          auto res = thrust::make_tuple(thrust::get<0>(t), thrust::get<1>(t));
          auto IM = thrust::get<2>(t);

          if (inverse) {
            IM = IntersectionMatrix::Transpose(IM);
          }

          return detail::EvaluatePredicate(predicate, IM) ? res : invalid_tuple;
        });
    stream.synchronize();
    sw.stop();
    t_evaluate += sw.ms();
  }
  GPUSPATIAL_LOG_INFO(
      "init time: %.3f ms, compute_aabb: %.3f ms, build_bvh: %.3f ms, trace_time: %.3f ms, evaluate_time: %.3f ms",
      t_init, t_compute_aabb, t_build_bvh, t_trace, t_evaluate);
  auto end = thrust::remove_if(rmm::exec_policy_nosync(stream), zip_begin, zip_end,
                               [=] __device__(const thrust::tuple<INDEX_T, INDEX_T>& tu) {
                                 return tu == invalid_tuple;
                               });
  size_t new_size = thrust::distance(zip_begin, end);
  point_ids.resize(new_size, stream);
  multi_poly_ids.resize(new_size, stream);
}

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
      num_aabbs, config_.bvh_fast_build, config_.bvh_fast_compact);
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
      num_aabbs, config_.bvh_fast_build, config_.bvh_fast_compact);
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
                                      config_.bvh_fast_build, config_.bvh_fast_compact);
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
    rmm::device_uvector<INDEX_T>& part_begins, double& t_compute_aabb,
    double& t_build_bvh) {
  auto n_mult_polygons = multi_poly_ids.size();
  Stopwatch sw;
  sw.start();

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
  stream.synchronize();

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
  stream.synchronize();
  sw.stop();
  t_compute_aabb += sw.ms();
  sw.start();
  assert(rt_engine_ != nullptr);
  auto handle =
      rt_engine_->BuildAccelCustom(stream.value(), ArrayView<OptixAabb>(aabbs), buffer,
                                   config_.bvh_fast_build, config_.bvh_fast_compact);
  stream.synchronize();
  sw.stop();
  t_build_bvh += sw.ms();
  return handle;
}
// Explicitly instantiate the template for specific types
template class RelateEngine<Point<double, 2>, uint32_t>;
}  // namespace gpuspatial
