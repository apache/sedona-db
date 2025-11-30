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
#include "gpuspatial/index/detail/rt_engine.hpp"
#include "gpuspatial/loader/device_geometries.cuh"
#include "gpuspatial/relate/predicate.cuh"
#include "gpuspatial/utils/queue.h"

#include "rmm/cuda_stream_view.hpp"

namespace gpuspatial {

template <typename POINT_T, typename INDEX_T>
class RelateEngine {
  using scalar_t = typename POINT_T::scalar_t;

 public:
  struct Config {
    bool bvh_fast_build = false;
    bool bvh_fast_compact = true;
    float memory_quota = 0.8;
  };

  RelateEngine() = default;

  RelateEngine(const DeviceGeometries<POINT_T, INDEX_T>* geoms1);

  RelateEngine(const DeviceGeometries<POINT_T, INDEX_T>* geoms1,
               const details::RTEngine* rt_engine);

  void set_config(const Config& config) { config_ = config; }

  void Evaluate(const rmm::cuda_stream_view& stream,
                const DeviceGeometries<POINT_T, INDEX_T>& geoms2, Predicate predicate,
                Queue<thrust::pair<uint32_t, uint32_t>>& ids);

  template <typename GEOM2_ARRAY_VIEW_T>
  void Evaluate(const rmm::cuda_stream_view& stream,
                const GEOM2_ARRAY_VIEW_T& geom_array2, Predicate predicate,
                Queue<thrust::pair<uint32_t, uint32_t>>& ids);

  // This is a generic version that can accept any two geometry array views
  template <typename GEOM1_ARRAY_VIEW_T, typename GEOM2_ARRAY_VIEW_T>
  void Evaluate(const rmm::cuda_stream_view& stream,
                const GEOM1_ARRAY_VIEW_T& geom_array1,
                const GEOM2_ARRAY_VIEW_T& geom_array2, Predicate predicate,
                Queue<thrust::pair<uint32_t, uint32_t>>& ids);

  // These are the specific overloads for RT-accelerated PIP queries
  void Evaluate(const rmm::cuda_stream_view& stream,
                const PointArrayView<POINT_T, INDEX_T>& geom_array1,
                const PolygonArrayView<POINT_T, INDEX_T>& geom_array2,
                Predicate predicate, Queue<thrust::pair<uint32_t, uint32_t>>& ids);

  void Evaluate(const rmm::cuda_stream_view& stream,
                const MultiPointArrayView<POINT_T, INDEX_T>& geom_array1,
                const PolygonArrayView<POINT_T, INDEX_T>& geom_array2,
                Predicate predicate, Queue<thrust::pair<uint32_t, uint32_t>>& ids);

  void Evaluate(const rmm::cuda_stream_view& stream,
                const PolygonArrayView<POINT_T, INDEX_T>& geom_array1,
                const PointArrayView<POINT_T, INDEX_T>& geom_array2, Predicate predicate,
                Queue<thrust::pair<uint32_t, uint32_t>>& ids);

  void Evaluate(const rmm::cuda_stream_view& stream,
                const PolygonArrayView<POINT_T, INDEX_T>& geom_array1,
                const MultiPointArrayView<POINT_T, INDEX_T>& geom_array2,
                Predicate predicate, Queue<thrust::pair<uint32_t, uint32_t>>& ids);

  void Evaluate(const rmm::cuda_stream_view& stream,
                const PointArrayView<POINT_T, INDEX_T>& geom_array1,
                const MultiPolygonArrayView<POINT_T, INDEX_T>& geom_array2,
                Predicate predicate, Queue<thrust::pair<uint32_t, uint32_t>>& ids);

  void Evaluate(const rmm::cuda_stream_view& stream,
                const MultiPointArrayView<POINT_T, INDEX_T>& geom_array1,
                const MultiPolygonArrayView<POINT_T, INDEX_T>& geom_array2,
                Predicate predicate, Queue<thrust::pair<uint32_t, uint32_t>>& ids);

  void Evaluate(const rmm::cuda_stream_view& stream,
                const MultiPolygonArrayView<POINT_T, INDEX_T>& geom_array1,
                const PointArrayView<POINT_T, INDEX_T>& geom_array2, Predicate predicate,
                Queue<thrust::pair<uint32_t, uint32_t>>& ids);

  void Evaluate(const rmm::cuda_stream_view& stream,
                const MultiPolygonArrayView<POINT_T, INDEX_T>& geom_array1,
                const MultiPointArrayView<POINT_T, INDEX_T>& geom_array2,
                Predicate predicate, Queue<thrust::pair<uint32_t, uint32_t>>& ids);

  void EvaluateImpl(const rmm::cuda_stream_view& stream,
                    const PointArrayView<POINT_T, INDEX_T>& point_array,
                    const MultiPointArrayView<POINT_T, INDEX_T>& multi_point_array,
                    const PolygonArrayView<POINT_T, INDEX_T>& poly_array,
                    Predicate predicate, Queue<thrust::pair<uint32_t, uint32_t>>& ids,
                    bool inverse = false);

  void EvaluateImpl(const rmm::cuda_stream_view& stream,
                    const PointArrayView<POINT_T, INDEX_T>& point_array,
                    const MultiPointArrayView<POINT_T, INDEX_T>& multi_point_array,
                    const MultiPolygonArrayView<POINT_T, INDEX_T>& multi_poly_array,
                    Predicate predicate, Queue<thrust::pair<uint32_t, uint32_t>>& ids,
                    bool inverse);

  /**
   * Build BVH for a subset of polygons
   * @param stream
   * @param polygons
   * @param polygon_ids
   * @param buffer
   */
  OptixTraversableHandle BuildBVH(const rmm::cuda_stream_view& stream,
                                  const PolygonArrayView<POINT_T, INDEX_T>& polygons,
                                  ArrayView<uint32_t> polygon_ids,
                                  rmm::device_uvector<INDEX_T>& seg_begins,
                                  rmm::device_buffer& buffer,
                                  rmm::device_uvector<INDEX_T>& aabb_poly_ids,
                                  rmm::device_uvector<INDEX_T>& aabb_ring_ids);

  OptixTraversableHandle BuildBVH(
      const rmm::cuda_stream_view& stream,
      const MultiPolygonArrayView<POINT_T, INDEX_T>& multi_polys,
      ArrayView<uint32_t> multi_poly_ids, rmm::device_uvector<INDEX_T>& seg_begins,
      rmm::device_uvector<INDEX_T>& part_begins, rmm::device_buffer& buffer,
      rmm::device_uvector<INDEX_T>& aabb_multi_poly_ids,
      rmm::device_uvector<INDEX_T>& aabb_part_ids,
      rmm::device_uvector<INDEX_T>& aabb_ring_ids);

  size_t EstimateBVHSize(const rmm::cuda_stream_view& stream,
                         const PolygonArrayView<POINT_T, INDEX_T>& polys,
                         ArrayView<uint32_t> poly_ids);

  size_t EstimateBVHSize(const rmm::cuda_stream_view& stream,
                         const MultiPolygonArrayView<POINT_T, INDEX_T>& multi_polys,
                         ArrayView<uint32_t> multi_poly_ids);

 private:
  Config config_;
  const DeviceGeometries<POINT_T, INDEX_T>* geoms1_;
  const details::RTEngine* rt_engine_;
};
}  // namespace gpuspatial
