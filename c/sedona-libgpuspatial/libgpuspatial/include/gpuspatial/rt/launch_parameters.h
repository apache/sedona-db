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
#include "gpuspatial/geom/multi_point.cuh"
#include "gpuspatial/geom/multi_polygon.cuh"
#include "gpuspatial/geom/point.cuh"
#include "gpuspatial/geom/polygon.cuh"
#include "gpuspatial/utils/array_view.h"
#include "gpuspatial/utils/queue_view.h"

#include <thrust/pair.h>

namespace gpuspatial {
namespace detail {

template <typename POINT_T>
struct LaunchParamsPointQuery {
  using box_t = Box<POINT_T>;
  // Input
  ArrayView<box_t> rects;
  ArrayView<POINT_T> points;
  OptixTraversableHandle handle;
  uint32_t* count;
  // Output
  QueueView<uint32_t> rect_ids;
  ArrayView<uint32_t> point_ids;
};

template <typename POINT_T>
struct LaunchParamsBoxQuery {
  using box_t = Box<POINT_T>;
  // Input
  ArrayView<box_t> rects1;
  ArrayView<box_t> rects2;
  // can be either geometries 1 or 2
  OptixTraversableHandle handle;
  uint32_t* count;
  // Output
  QueueView<uint32_t> rect1_ids;
  ArrayView<uint32_t> rect2_ids;
};

/**
 * This query is compatible with both MultiPoint-MultiPolygon and Point-MultiPolygon
 */
template <typename POINT_T, typename INDEX_T>
struct LaunchParamsPolygonPointQuery {
  using point_t = POINT_T;
  using index_t = INDEX_T;
  // Either MultiPointArrayView or PointArrayView will be used
  MultiPointArrayView<point_t, index_t> multi_points;
  PointArrayView<point_t, index_t> points;
  PolygonArrayView<point_t, index_t> polygons;
  ArrayView<index_t> uniq_polygon_ids;  // sorted
  index_t* query_point_ids;
  index_t* query_polygon_ids;
  size_t query_size;
  ArrayView<index_t> seg_begins;
  ArrayView<int> IMs;  // intersection matrices
  OptixTraversableHandle handle;
  ArrayView<index_t> aabb_poly_ids, aabb_ring_ids;
  ArrayView<thrust::pair<index_t, index_t>> aabb_vertex_offsets;
};

/**
 * This query is compatible with both MultiPoint-MultiPolygon and Point-MultiPolygon
 */
template <typename POINT_T, typename INDEX_T>
struct LaunchParamsPointMultiPolygonQuery {
  using point_t = POINT_T;
  using index_t = INDEX_T;
  using scalar_t = typename POINT_T::scalar_t;
  MultiPolygonArrayView<point_t, index_t> multi_polygons;
  // Either MultiPointArrayView or PointArrayView will be used
  MultiPointArrayView<point_t, index_t> multi_points;
  PointArrayView<point_t, index_t> points;
  ArrayView<index_t> uniq_multi_polygon_ids;  // sorted
  index_t* query_point_ids;
  index_t* query_multi_polygon_ids;
  size_t query_size;
  ArrayView<index_t> uniq_part_begins;  // used to calculate z-index for parts
  // each query point has n elements of part_min_y and part_locations, n is # of parts
  ArrayView<int> IMs;  // intersection matrices
  OptixTraversableHandle handle;
  ArrayView<index_t> aabb_multi_poly_ids, aabb_part_ids, aabb_ring_ids;
  ArrayView<thrust::pair<index_t, index_t>> aabb_vertex_offsets;
};

}  // namespace detail

}  // namespace gpuspatial
