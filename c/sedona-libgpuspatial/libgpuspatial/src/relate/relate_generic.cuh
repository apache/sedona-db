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

#include "relate/relate_predicate.cuh"

#include "gpuspatial/relate/predicate.hpp"
#include "gpuspatial/relate/relate.hpp"
#include "gpuspatial/utils/logger.hpp"

#include "rmm/cuda_stream_view.hpp"
#include "rmm/device_uvector.hpp"
#include "rmm/exec_policy.hpp"

#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/tuple.h>
#include <cuda/std/iterator>
#include <type_traits>

namespace gpuspatial {
namespace detail {

template <typename POINT_T, typename INDEX_T, typename GEOM1_ARRAY_VIEW_T,
          typename GEOM2_ARRAY_VIEW_T>
void EvaluateGeneric(const rmm::cuda_stream_view& stream,
                     const GEOM1_ARRAY_VIEW_T& geom_array1,
                     const GEOM2_ARRAY_VIEW_T& geom_array2, Predicate predicate,
                     rmm::device_uvector<INDEX_T>& ids1,
                     rmm::device_uvector<INDEX_T>& ids2) {
  assert(ids1.size() == ids2.size());
  size_t ids_size = ids1.size();
  GPUSPATIAL_LOG_INFO(
      "Refine with generic kernel, geom1 %zu, geom2 %zu, predicate %s, result size %zu",
      geom_array1.size(), geom_array2.size(), PredicateToString(predicate), ids_size);
  if ((std::is_same_v<GEOM1_ARRAY_VIEW_T, PolygonArrayView<POINT_T, INDEX_T>> &&
       std::is_same_v<GEOM2_ARRAY_VIEW_T, PolygonArrayView<POINT_T, INDEX_T>>) ||
      (std::is_same_v<GEOM1_ARRAY_VIEW_T, PolygonArrayView<POINT_T, INDEX_T>> &&
       std::is_same_v<GEOM2_ARRAY_VIEW_T, MultiPolygonArrayView<POINT_T, INDEX_T>>) ||
      (std::is_same_v<GEOM1_ARRAY_VIEW_T, MultiPolygonArrayView<POINT_T, INDEX_T>> &&
       std::is_same_v<GEOM2_ARRAY_VIEW_T, PolygonArrayView<POINT_T, INDEX_T>>) ||
      (std::is_same_v<GEOM1_ARRAY_VIEW_T, MultiPolygonArrayView<POINT_T, INDEX_T>> &&
       std::is_same_v<GEOM2_ARRAY_VIEW_T, MultiPolygonArrayView<POINT_T, INDEX_T>>)) {
    GPUSPATIAL_LOG_WARN(
        "Evaluate Polygon-Polygon relate with the GPU, which is not well-tested and the "
        "performance may be poor.");
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
  size_t new_size = cuda::std::distance(zip_begin, end);
  ids1.resize(new_size, stream);
  ids2.resize(new_size, stream);
}

}  // namespace detail
}  // namespace gpuspatial
