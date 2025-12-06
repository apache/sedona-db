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
#include "gpuspatial/geom/polygon.cuh"

namespace gpuspatial {
template <typename POINT_T, typename INDEX_T>
class MultiPolygon {
 public:
  using point_t = POINT_T;
  using line_segments_view_t = LineString<point_t>;
  using box_t = Box<Point<float, point_t::n_dim>>;

  MultiPolygon() = default;

  DEV_HOST MultiPolygon(const ArrayView<INDEX_T>& prefix_sum_parts,
                        const ArrayView<INDEX_T>& prefix_sum_rings,
                        const ArrayView<point_t>& vertices, const box_t& mbr)
      : prefix_sum_parts_(prefix_sum_parts),
        prefix_sum_rings_(prefix_sum_rings),
        vertices_(vertices),
        mbr_(mbr) {}

  DEV_HOST_INLINE bool empty() const {
    for (size_t i = 0; i < num_polygons(); i++) {
      if (!get_polygon(i).empty()) {
        return false;
      }
    }
    return true;
  }

  DEV_HOST_INLINE INDEX_T num_polygons() const {
    return prefix_sum_parts_.empty() ? 0 : prefix_sum_parts_.size() - 1;
  }

  DEV_HOST_INLINE Polygon<POINT_T, INDEX_T> get_polygon(INDEX_T i) const {
    auto ring_begin = prefix_sum_parts_[i];
    auto ring_end = prefix_sum_parts_[i + 1];
    ArrayView<INDEX_T> prefix_sum_rings(
        const_cast<INDEX_T*>(prefix_sum_rings_.data()) + ring_begin,
        ring_end - ring_begin + 1);
    return {prefix_sum_rings, vertices_, mbr_};
  }

  DEV_HOST_INLINE const ArrayView<INDEX_T>& get_prefix_sum_parts() const {
    return prefix_sum_parts_;
  }

  DEV_HOST_INLINE const ArrayView<INDEX_T>& get_prefix_sum_rings() const {
    return prefix_sum_rings_;
  }

  DEV_HOST_INLINE const ArrayView<POINT_T>& get_vertices() const { return vertices_; }

  DEV_HOST_INLINE const box_t& get_mbr() const { return mbr_; }

  DEV_HOST_INLINE uint32_t num_vertices() const {
    uint32_t nv = 0;
    for (int i = 0; i < num_polygons(); i++) {
      const auto& poly = get_polygon(i);
      nv += poly.num_vertices();
    }
    return nv;
  }

 private:
  ArrayView<INDEX_T> prefix_sum_parts_;
  ArrayView<INDEX_T> prefix_sum_rings_;
  ArrayView<POINT_T> vertices_;
  box_t mbr_;
};

/**
 * This class can represent an array of polygons or multi-polygons
 * @tparam POINT_T
 */
template <typename POINT_T, typename INDEX_T>
class MultiPolygonArrayView {
 public:
  using point_t = POINT_T;
  using box_t = Box<Point<float, point_t::n_dim>>;
  using geometry_t = MultiPolygon<point_t, INDEX_T>;
  MultiPolygonArrayView() = default;

  DEV_HOST MultiPolygonArrayView(const ArrayView<INDEX_T>& prefix_sum_geoms,
                                 const ArrayView<INDEX_T>& prefix_sum_parts,
                                 const ArrayView<INDEX_T>& prefix_sum_rings,
                                 const ArrayView<point_t>& vertices,
                                 const ArrayView<box_t>& mbrs)
      : prefix_sum_geoms_(prefix_sum_geoms),
        prefix_sum_parts_(prefix_sum_parts),
        prefix_sum_rings_(prefix_sum_rings),
        vertices_(vertices),
        mbrs_(mbrs) {}

  DEV_HOST_INLINE size_t size() const {
    return prefix_sum_geoms_.empty() ? 0 : prefix_sum_geoms_.size() - 1;
  }

  DEV_HOST_INLINE bool empty() const { return size() == 0; }

  DEV_HOST_INLINE MultiPolygon<point_t, INDEX_T> operator[](size_t i) {
    auto part_begin = prefix_sum_geoms_[i];
    auto part_end = prefix_sum_geoms_[i + 1];
    ArrayView<INDEX_T> prefix_sum_parts(prefix_sum_parts_.data() + part_begin,
                                        part_end - part_begin + 1);

    return {prefix_sum_parts, prefix_sum_rings_, vertices_, mbrs_[i]};
  }

  DEV_HOST_INLINE MultiPolygon<point_t, INDEX_T> operator[](size_t i) const {
    auto part_begin = prefix_sum_geoms_[i];
    auto part_end = prefix_sum_geoms_[i + 1];
    ArrayView<INDEX_T> prefix_sum_parts(
        const_cast<INDEX_T*>(prefix_sum_parts_.data()) + part_begin,
        part_end - part_begin + 1);

    return {prefix_sum_parts, prefix_sum_rings_, vertices_, mbrs_[i]};
  }

  DEV_HOST_INLINE ArrayView<INDEX_T> get_prefix_sum_geoms() const {
    return prefix_sum_geoms_;
  }

  DEV_HOST_INLINE ArrayView<INDEX_T> get_prefix_sum_parts() const {
    return prefix_sum_parts_;
  }

  DEV_HOST_INLINE ArrayView<INDEX_T> get_prefix_sum_rings() const {
    return prefix_sum_rings_;
  }

  DEV_HOST_INLINE ArrayView<point_t> get_vertices() const { return vertices_; }

  DEV_HOST_INLINE ArrayView<box_t> get_mbrs() const { return mbrs_; }

  DEV_HOST_INLINE bool locate_vertex(uint32_t vertex_idx, uint32_t& geom_idx,
                                     uint32_t& part_idx, uint32_t& ring_idx) const {
    auto it_ring = thrust::upper_bound(thrust::seq, prefix_sum_rings_.begin(),
                                       prefix_sum_rings_.end(), vertex_idx);

    if (it_ring != prefix_sum_rings_.end()) {
      // which ring the vertex belongs to
      auto ring_offset = thrust::distance(prefix_sum_rings_.begin(), it_ring) - 1;
      auto it_part = thrust::upper_bound(thrust::seq, prefix_sum_parts_.begin(),
                                         prefix_sum_parts_.end(), ring_offset);
      if (it_part != prefix_sum_parts_.end()) {
        // which polygon the vertex belongs to
        auto part_offset = thrust::distance(prefix_sum_parts_.begin(), it_part) - 1;
        auto it_geom = thrust::upper_bound(thrust::seq, prefix_sum_geoms_.begin(),
                                           prefix_sum_geoms_.end(), part_offset);

        if (it_geom != prefix_sum_geoms_.end()) {
          geom_idx = thrust::distance(prefix_sum_geoms_.begin(), it_geom) - 1;
          part_idx = part_offset - prefix_sum_geoms_[geom_idx];
          ring_idx = ring_offset - prefix_sum_parts_[part_offset];
          return true;
        }
      }
    }
    return false;
  }

 private:
  ArrayView<INDEX_T> prefix_sum_geoms_;
  ArrayView<INDEX_T> prefix_sum_parts_;
  ArrayView<INDEX_T> prefix_sum_rings_;
  ArrayView<point_t> vertices_;
  ArrayView<box_t> mbrs_;
};
}  // namespace gpuspatial
