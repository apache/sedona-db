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
#include "gpuspatial/rt/launch_parameters.cuh"
#include "gpuspatial/utils/array_view.hpp"
#include "gpuspatial/utils/exception.hpp"
#include "gpuspatial/utils/helpers.cuh"
#include "gpuspatial/utils/logger.hpp"
#include "gpuspatial/mem/memory_manager.hpp"
#include "rt/shaders/shader_id.hpp"
#include "relate/relate_predicate.cuh"

#include "rmm/cuda_stream_view.hpp"
#include "rmm/device_buffer.hpp"
#include "rmm/device_uvector.hpp"
#include "rmm/exec_policy.hpp"

#include <thrust/copy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>
#include <cuda/std/iterator>
#include <algorithm>
#include <limits>

namespace gpuspatial {

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
  // Casting short rays from each point to do precise point-in-polygon test
  // Reference: "Geng L, Lee R, Zhang X. Rayjoin: Fast and precise spatial join.
  // InProceedings of the 38th ACM International Conference on Supercomputing 2024"
  using params_t = detail::LaunchParamsPolygonPointQuery<POINT_T, INDEX_T>;
  assert(point_array.empty() || multi_point_array.empty());
  assert(point_ids.size() == poly_ids.size());
  size_t ids_size = point_ids.size();
  GPUSPATIAL_LOG_INFO(
      "Refine with ray-tracing, (multi-)point %zu, polygon %zu, predicate %s, result size "
      "%zu, inverse %d",
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
  uniq_poly_ids.resize(cuda::std::distance(uniq_poly_ids.begin(), uniq_poly_ids_end),
                       stream);
  uniq_poly_ids.shrink_to_fit(stream);

  auto bvh_bytes = EstimateBVHSize(stream, poly_array, ArrayView<uint32_t>(uniq_poly_ids),
                                   config_.segs_per_aabb);
  size_t avail_bytes =
      MemoryManager::instance().get_available_device_memory() * config_.memory_quota;
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

    uniq_poly_ids.resize(cuda::std::distance(uniq_poly_ids.begin(), uniq_poly_ids_end),
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
  size_t new_size = cuda::std::distance(zip_begin, end);
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
      "Refine with ray-tracing, (multi-)point %zu, multi-polygon %zu, predicate %s, result "
      "size %zu, inverse %d",
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
      cuda::std::distance(uniq_multi_poly_ids.begin(), uniq_multi_poly_ids_end), stream);
  uniq_multi_poly_ids.shrink_to_fit(stream);

  auto bvh_bytes =
      EstimateBVHSize(stream, multi_poly_array, ArrayView<uint32_t>(uniq_multi_poly_ids),
                      config_.segs_per_aabb);
  size_t avail_bytes =
      MemoryManager::instance().get_available_device_memory() * config_.memory_quota;
  auto n_batches = bvh_bytes / avail_bytes + 1;
  auto batch_size = (ids_size + n_batches - 1) / n_batches;

  GPUSPATIAL_LOG_INFO(
      "Unique multi-polygons %zu, memory quota %zu MB, estimated BVH size %zu MB",
      uniq_multi_poly_ids.size(), avail_bytes / (1024 * 1024), bvh_bytes / (1024 * 1024));

  for (int batch = 0; batch < n_batches; batch++) {
    auto ids_begin = batch * batch_size;
    auto ids_end = std::min(ids_begin + batch_size, ids_size);
    auto ids_size_batch = ids_end - ids_begin;

    // Extract multi polygon IDs in this batch
    uniq_multi_poly_ids.resize(ids_size_batch, stream);

    thrust::copy(rmm::exec_policy_nosync(stream), multi_poly_ids.begin() + ids_begin,
                 multi_poly_ids.begin() + ids_end, uniq_multi_poly_ids.begin());

    // multi polygon ids have been sorted before
    uniq_multi_poly_ids_end =
        thrust::unique(rmm::exec_policy_nosync(stream), uniq_multi_poly_ids.begin(),
                       uniq_multi_poly_ids.end());
    uniq_multi_poly_ids.resize(
        cuda::std::distance(uniq_multi_poly_ids.begin(), uniq_multi_poly_ids_end), stream);
    uniq_multi_poly_ids.shrink_to_fit(stream);

    rmm::device_uvector<int> IMs(ids_size_batch, stream);
    rmm::device_buffer bvh_buffer(0, stream);
    rmm::device_uvector<INDEX_T> aabb_multi_poly_ids(0, stream), aabb_part_ids(0, stream),
        aabb_ring_ids(0, stream);
    rmm::device_uvector<thrust::pair<INDEX_T, INDEX_T>> aabb_vertex_offsets(0, stream);
    rmm::device_uvector<INDEX_T> uniq_part_begins(0, stream);

    auto handle =
        BuildBVH(stream, multi_poly_array, ArrayView<INDEX_T>(uniq_multi_poly_ids),
                 config_.segs_per_aabb, bvh_buffer, aabb_multi_poly_ids, aabb_part_ids,
                 aabb_ring_ids, aabb_vertex_offsets, uniq_part_begins);

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
  }
  auto end = thrust::remove_if(rmm::exec_policy_nosync(stream), zip_begin, zip_end,
                               [=] __device__(const thrust::tuple<INDEX_T, INDEX_T>& tu) {
                                 return tu == invalid_tuple;
                               });
  size_t new_size = cuda::std::distance(zip_begin, end);
  point_ids.resize(new_size, stream);
  multi_poly_ids.resize(new_size, stream);
}

// Explicit instantiation of RelateEngine class template
template class RelateEngine<Point<double, 2>, uint32_t>;

}  // namespace gpuspatial
