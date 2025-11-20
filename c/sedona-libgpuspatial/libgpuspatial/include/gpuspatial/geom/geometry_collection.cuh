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
#include "gpuspatial/geom/geometry_type.cuh"
#include "gpuspatial/geom/line_string.cuh"
#include "gpuspatial/geom/multi_line_string.cuh"
#include "gpuspatial/geom/multi_point.cuh"
#include "gpuspatial/geom/multi_polygon.cuh"
#include "gpuspatial/geom/point.cuh"
#include "gpuspatial/geom/polygon.cuh"
#include "gpuspatial/utils/array_view.h"

namespace gpuspatial {

template <typename POINT_T, typename INDEX_T>
class GeometryCollection {
 public:
  using point_t = POINT_T;
  using line_segments_view_t = LineString<point_t>;
  using box_t = Box<Point<float, point_t::n_dim>>;

  GeometryCollection() = default;

  DEV_HOST GeometryCollection(const ArrayView<GeometryType>& feature_types,
                              const ArrayView<INDEX_T>& ps_num_parts,
                              const ArrayView<INDEX_T>& ps_num_rings,
                              const ArrayView<INDEX_T>& ps_num_points,
                              const ArrayView<point_t>& vertices, const box_t& mbr)
      : feature_types_(feature_types),
        ps_num_parts_(ps_num_parts),
        ps_num_rings_(ps_num_rings),
        ps_num_points_(ps_num_points),
        vertices_(vertices),
        mbr_(mbr) {}

  // DEV_HOST_INLINE bool empty() const {
  //   for (size_t i = 0; i < num_polygons(); i++) {
  //     if (!get_polygon(i).empty()) {
  //       return false;
  //     }
  //   }
  //   return true;
  // }

  DEV_HOST_INLINE INDEX_T num_geometries() const { return feature_types_.size(); }

  DEV_HOST_INLINE GeometryType get_type(INDEX_T geometry_idx) const {
    return feature_types_[geometry_idx];
  }

  DEV_HOST_INLINE POINT_T get_point(INDEX_T geometry_idx) const {
    assert(feature_types_[geometry_idx] == GeometryType::kPoint);
    auto part_begin = ps_num_parts_[geometry_idx];
    auto ring_begin = ps_num_rings_[part_begin];
    auto point_begin = ps_num_points_[ring_begin];
    return vertices_[point_begin];
  }

  DEV_HOST_INLINE LineString<POINT_T> get_line_string(INDEX_T geometry_idx) const {
    assert(feature_types_[geometry_idx] == GeometryType::kLineString);
    auto part_begin = ps_num_parts_[geometry_idx];
    auto ring_begin = ps_num_rings_[part_begin];
    auto point_begin = ps_num_points_[ring_begin];
    auto point_end = ps_num_points_[ring_begin + 1];
    ArrayView<point_t> vertices(const_cast<POINT_T*>(vertices_.data()) + point_begin,
                                point_end - point_begin);

    return {vertices, mbr_};
  }

  DEV_HOST_INLINE Polygon<POINT_T, INDEX_T> get_polygon(INDEX_T geometry_idx) const {
    assert(feature_types_[geometry_idx] == GeometryType::kPolygon);
    auto part_begin = ps_num_parts_[geometry_idx];
    auto part_end = ps_num_parts_[geometry_idx + 1];
    if (part_begin == part_end) return {};
    auto ring_begin = ps_num_rings_[part_begin];
    auto ring_end = ps_num_rings_[part_begin + 1];
    ArrayView<INDEX_T> ps_num_points(
        const_cast<INDEX_T*>(ps_num_points_.data()) + ring_begin,
        ring_end - ring_begin + 1);
    return {ps_num_points, vertices_, mbr_};
  }

  DEV_HOST_INLINE MultiPoint<POINT_T> get_multi_point(INDEX_T geometry_idx) const {
    assert(feature_types_[geometry_idx] == GeometryType::kMultiPoint);
    auto part_begin = ps_num_parts_[geometry_idx];
    auto part_end = ps_num_parts_[geometry_idx + 1];
    if (part_begin == part_end) return {};
    auto ring_begin = ps_num_rings_[part_begin];
    auto point_begin = ps_num_points_[ring_begin];
    auto point_end = ps_num_points_[ring_begin + 1];
    ArrayView<POINT_T> vertices(const_cast<POINT_T*>(vertices_.data()) + point_begin,
                                point_end - point_begin);
    return {vertices, mbr_};
  }

  DEV_HOST_INLINE MultiLineString<POINT_T, INDEX_T> get_multi_linestring(
      INDEX_T geometry_idx) const {
    assert(feature_types_[geometry_idx] == GeometryType::kMultiLineString);
    auto part_begin = ps_num_parts_[geometry_idx];
    auto part_end = ps_num_parts_[geometry_idx + 1];
    if (part_begin == part_end) return {};
    auto ring_begin = ps_num_rings_[part_begin];
    auto ring_end = ps_num_rings_[part_begin + 1];
    ArrayView<INDEX_T> ps_num_points(
        const_cast<INDEX_T*>(ps_num_points_.data()) + ring_begin,
        ring_end - ring_begin + 1);

    return {ps_num_points, vertices_, mbr_};
  }

  DEV_HOST_INLINE MultiPolygon<POINT_T, INDEX_T> get_multi_polygon(
      INDEX_T geometry_idx) const {
    assert(feature_types_[geometry_idx] == GeometryType::kMultiPolygon);
    auto part_begin = ps_num_parts_[geometry_idx];
    auto part_end = ps_num_parts_[geometry_idx + 1];
    ArrayView<INDEX_T> ps_num_rings(
        const_cast<INDEX_T*>(ps_num_rings_.data()) + part_begin,
        part_end - part_begin + 1);
    return {ps_num_rings, ps_num_points_, vertices_, mbr_};
  }

  DEV_HOST_INLINE const box_t& get_mbr() const { return mbr_; }

 private:
  ArrayView<GeometryType> feature_types_;
  ArrayView<INDEX_T> ps_num_parts_;
  ArrayView<INDEX_T> ps_num_rings_;
  ArrayView<INDEX_T> ps_num_points_;
  ArrayView<POINT_T> vertices_;
  box_t mbr_;
};

/**
 * This class can represent an array of polygons or multi-polygons
 * @tparam POINT_T
 */
template <typename POINT_T, typename INDEX_T>
class GeometryCollectionArrayView {
 public:
  using point_t = POINT_T;
  using box_t = Box<Point<float, point_t::n_dim>>;
  using geometry_t = MultiPolygon<point_t, INDEX_T>;
  GeometryCollectionArrayView() = default;

  DEV_HOST GeometryCollectionArrayView(const ArrayView<GeometryType>& feature_types,
                                       const ArrayView<INDEX_T>& ps_num_geoms,
                                       const ArrayView<INDEX_T>& ps_num_parts,
                                       const ArrayView<INDEX_T>& ps_num_rings,
                                       const ArrayView<INDEX_T>& ps_num_points,
                                       const ArrayView<point_t>& vertices,
                                       const ArrayView<box_t>& mbrs)
      : feature_types_(feature_types),
        ps_num_geoms_(ps_num_geoms),
        ps_num_parts_(ps_num_parts),
        ps_num_rings_(ps_num_rings),
        ps_num_points_(ps_num_points),
        vertices_(vertices),
        mbrs_(mbrs) {}

  DEV_HOST_INLINE size_t size() const {
    return ps_num_geoms_.empty() ? 0 : ps_num_geoms_.size() - 1;
  }

  DEV_HOST_INLINE bool empty() const { return size() == 0; }

  DEV_HOST_INLINE GeometryCollection<point_t, INDEX_T> operator[](size_t i) {
    auto geom_begin = ps_num_geoms_[i];
    auto geom_end = ps_num_geoms_[i + 1];

    ArrayView<GeometryType> feature_types(feature_types_.data() + geom_begin,
                                          geom_end - geom_begin);
    ArrayView<INDEX_T> ps_num_parts(ps_num_parts_.data() + geom_begin,
                                    geom_end - geom_begin + 1);

    return {feature_types,  ps_num_parts, ps_num_rings_,
            ps_num_points_, vertices_,    mbrs_[i]};
  }

  DEV_HOST_INLINE GeometryCollection<point_t, INDEX_T> operator[](size_t i) const {
    auto geom_begin = ps_num_geoms_[i];
    auto geom_end = ps_num_geoms_[i + 1];

    ArrayView<GeometryType> feature_types(
        const_cast<GeometryType*>(feature_types_.data()) + geom_begin,
        geom_end - geom_begin);
    ArrayView<INDEX_T> ps_num_parts(
        const_cast<INDEX_T*>(ps_num_parts_.data()) + geom_begin,
        geom_end - geom_begin + 1);

    return {feature_types,  ps_num_parts, ps_num_rings_,
            ps_num_points_, vertices_,    mbrs_[i]};
  }

  DEV_HOST_INLINE ArrayView<INDEX_T> get_prefix_sum_num_geoms() const {
    return ps_num_geoms_;
  }

  DEV_HOST_INLINE ArrayView<INDEX_T> get_prefix_sum_num_parts() const {
    return ps_num_parts_;
  }

  DEV_HOST_INLINE ArrayView<INDEX_T> get_prefix_sum_num_rings() const {
    return ps_num_rings_;
  }

  DEV_HOST_INLINE ArrayView<INDEX_T> get_prefix_sum_num_points() const {
    return ps_num_points_;
  }
  DEV_HOST_INLINE ArrayView<point_t> get_vertices() const { return vertices_; }

  DEV_HOST_INLINE ArrayView<box_t> get_mbrs() const { return mbrs_; }

 private:
  ArrayView<GeometryType> feature_types_;
  ArrayView<INDEX_T> ps_num_geoms_;
  ArrayView<INDEX_T> ps_num_parts_;
  ArrayView<INDEX_T> ps_num_rings_;
  ArrayView<INDEX_T> ps_num_points_;
  ArrayView<POINT_T> vertices_;
  ArrayView<box_t> mbrs_;
};

}  // namespace gpuspatial
