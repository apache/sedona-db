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

#include <cassert>

namespace gpuspatial {

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
    case GeometryType::kGeometryCollection: {
      using geom2_array_view_t = GeometryCollectionArrayView<POINT_T, INDEX_T>;
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
    case GeometryType::kGeometryCollection: {
      using geom1_array_view_t = GeometryCollectionArrayView<POINT_T, INDEX_T>;
      Evaluate(stream, geoms1_->template GetGeometryArrayView<geom1_array_view_t>(),
               geom_array2, predicate, ids1, ids2);
      break;
    }
    default:
      assert(false);
  }
}

// Explicit instantiation of RelateEngine class template
template class RelateEngine<Point<double, 2>, uint32_t>;

#define INSTANTIATE_EVALUATE_GEOM2(GEOM2_T)                                      \
  template void RelateEngine<Point<double, 2>, uint32_t>::Evaluate(              \
      const rmm::cuda_stream_view& stream,                                       \
      const GEOM2_T<Point<double, 2>, uint32_t>& geom_array2,                   \
      Predicate predicate, rmm::device_uvector<uint32_t>& ids1,                  \
      rmm::device_uvector<uint32_t>& ids2);

INSTANTIATE_EVALUATE_GEOM2(PointArrayView)
INSTANTIATE_EVALUATE_GEOM2(MultiPointArrayView)
INSTANTIATE_EVALUATE_GEOM2(LineStringArrayView)
INSTANTIATE_EVALUATE_GEOM2(MultiLineStringArrayView)
INSTANTIATE_EVALUATE_GEOM2(PolygonArrayView)
INSTANTIATE_EVALUATE_GEOM2(MultiPolygonArrayView)
INSTANTIATE_EVALUATE_GEOM2(GeometryCollectionArrayView)

}  // namespace gpuspatial
