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
#include "relate/relate_generic.cuh"

namespace gpuspatial {

template <typename POINT_T, typename INDEX_T>
template <typename GEOM1_ARRAY_VIEW_T, typename GEOM2_ARRAY_VIEW_T>
void RelateEngine<POINT_T, INDEX_T>::Evaluate(const rmm::cuda_stream_view& stream,
                                              const GEOM1_ARRAY_VIEW_T& geom_array1,
                                              const GEOM2_ARRAY_VIEW_T& geom_array2,
                                              Predicate predicate,
                                              rmm::device_uvector<INDEX_T>& ids1,
                                              rmm::device_uvector<INDEX_T>& ids2) {
  detail::EvaluateGeneric<POINT_T, INDEX_T>(stream, geom_array1, geom_array2, predicate,
                                           ids1, ids2);
}

#define INSTANTIATE_POLYGON_EVALUATE(GEOM1_T, GEOM2_T)                          \
  template void RelateEngine<Point<double, 2>, uint32_t>::Evaluate(            \
      const rmm::cuda_stream_view& stream,                                     \
      const GEOM1_T<Point<double, 2>, uint32_t>& geom_array1,                   \
      const GEOM2_T<Point<double, 2>, uint32_t>& geom_array2,                   \
      Predicate predicate, rmm::device_uvector<uint32_t>& ids1,                \
      rmm::device_uvector<uint32_t>& ids2);

INSTANTIATE_POLYGON_EVALUATE(PolygonArrayView, LineStringArrayView)
INSTANTIATE_POLYGON_EVALUATE(PolygonArrayView, MultiLineStringArrayView)
INSTANTIATE_POLYGON_EVALUATE(PolygonArrayView, PolygonArrayView)
INSTANTIATE_POLYGON_EVALUATE(PolygonArrayView, MultiPolygonArrayView)

INSTANTIATE_POLYGON_EVALUATE(MultiPolygonArrayView, LineStringArrayView)
INSTANTIATE_POLYGON_EVALUATE(MultiPolygonArrayView, MultiLineStringArrayView)
INSTANTIATE_POLYGON_EVALUATE(MultiPolygonArrayView, PolygonArrayView)
INSTANTIATE_POLYGON_EVALUATE(MultiPolygonArrayView, MultiPolygonArrayView)

}  // namespace gpuspatial
