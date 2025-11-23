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
#include "gpuspatial/geom/line_string.cuh"
#include "gpuspatial/utils/array_view.h"
#include "gpuspatial/utils/cuda_utils.h"
#include "gpuspatial/utils/floating_point.h"

#include <cub/block/block_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>

#include <thrust/binary_search.h>

namespace gpuspatial {

template <typename POINT_T>
class LinearRing {
  using point_t = POINT_T;
  using line_segment_t = LineSegment<point_t>;

 public:
  LinearRing() = default;

  DEV_HOST LinearRing(const ArrayView<point_t>& vertices) : vertices_(vertices) {}

  DEV_HOST_INLINE line_segment_t get_line_segment(size_t i) const {
    assert(i + 1 < vertices_.size());
    return line_segment_t(vertices_[i], vertices_[i + 1]);
  }

  DEV_HOST_INLINE const point_t& get_point(size_t i) const { return vertices_[i]; }

  DEV_HOST_INLINE size_t num_points() const { return vertices_.size(); }

  DEV_HOST_INLINE size_t num_segments() const {
    return vertices_.empty() ? 0 : vertices_.size() - 1;
  }

  DEV_HOST_INLINE bool empty() const { return num_segments() == 0; }

  DEV_HOST_INLINE bool is_valid() const {
    if (vertices_.empty()) {
      return true;
    }
    if (!is_closed()) {
      return false;
    }
    return vertices_.size() >= 3;
  }

  DEV_HOST_INLINE PointLocation locate_point(const point_t& p) const {
    int wn = 0;

    for (int i = 0; i < num_points() - 1; i++) {
      const auto& p1 = get_point(i);
      const auto& p2 = get_point(i + 1);
      /* zero length segments are ignored. */
      if (p1 == p2) continue;
      LineSegment<point_t> seg(p1, p2);

      auto side = seg.orientation(p);
      if (side == 0) {
        if (seg.get_mbr().covers(p)) return PointLocation::kBoundary; /* on boundary */
      }

      bool is_rising = (p1.y() <= p.y()) && (p.y() < p2.y()) && (side == 1);
      bool is_falling = (p2.y() <= p.y()) && (p.y() < p1.y()) && (side == -1);
      // Add 1 if rising, subtract 1 if falling, add 0 otherwise.
      // The boolean values will be implicitly cast to 0 or 1.
      wn += is_rising - is_falling;
    }
    if (wn == 0) return PointLocation::kOutside;
    return PointLocation::kInside;
  }

  // Locate a point in the ring using a warp. Only lane0 returns the answer.
  DEV_INLINE PointLocation
  locate_point(const point_t& p, cub::WarpReduce<int>::TempStorage* temp_storage) const {
    /* see, point_in_ring */
    int wn = 0;
    auto lane_id = threadIdx.x % 32;
    bool on_boundary = false;

    // TODO: We could use shared memory to cache the points in the ring
    for (auto i = lane_id; i < num_points() - 1; i += 32) {
      const auto& p1 = get_point(i);
      const auto& p2 = get_point(i + 1);

      /* zero length segments are ignored. */
      if (p1 == p2) continue;

      LineSegment<point_t> seg(p1, p2);
      auto side = seg.orientation(p);

      if (side == 0) {
        if (seg.get_mbr().covers(p)) {
          on_boundary = true;
          break;
        }
      }

      bool is_rising = (p1.y() <= p.y()) && (p.y() < p2.y()) && (side == 1);
      bool is_falling = (p2.y() <= p.y()) && (p.y() < p1.y()) && (side == -1);
      // Add 1 if rising, subtract 1 if falling, add 0 otherwise.
      // The boolean values will be implicitly cast to 0 or 1.
      wn += is_rising - is_falling;
    }

    if (__any_sync(0xffffffff, on_boundary)) {
      return PointLocation::kBoundary;
    }

    auto total_wn = cub::WarpReduce<int>(*temp_storage).Sum(wn);
    if (lane_id == 0) {
      if (total_wn == 0) return PointLocation::kOutside;
      return PointLocation::kInside;
    }

    return PointLocation::kError;
  }

  DEV_INLINE PointLocation
  locate_point(const point_t& p,
               cub::BlockReduce<int, MAX_BLOCK_SIZE>::TempStorage* temp_storage) const {
    int wn = 0;
    bool on_boundary = false;

    for (int i = threadIdx.x; i < num_points() - 1; i += blockDim.x) {
      const auto& p1 = get_point(i);
      const auto& p2 = get_point(i + 1);
      /* zero length segments are ignored. */
      if (p1 == p2) continue;
      LineSegment<point_t> seg(p1, p2);

      auto side = seg.orientation(p);
      if (side == 0) {
        if (seg.get_mbr().covers(p)) {
          on_boundary = true;
          break;
        }
      }

      bool is_rising = (p1.y() <= p.y()) && (p.y() < p2.y()) && (side == 1);
      bool is_falling = (p2.y() <= p.y()) && (p.y() < p1.y()) && (side == -1);
      // Add 1 if rising, subtract 1 if falling, add 0 otherwise.
      // The boolean values will be implicitly cast to 0 or 1.
      wn += is_rising - is_falling;
    }

    auto& s_on_boundary = *reinterpret_cast<bool*>(temp_storage);

    if (threadIdx.x == 0) {
      s_on_boundary = false;
    }
    __syncthreads();
    if (on_boundary) {
      s_on_boundary = true;
    }
    __syncthreads();
    if (s_on_boundary) {
      return PointLocation::kBoundary;
    }
    auto total_wn =
        cub::BlockReduce<int, MAX_BLOCK_SIZE>(*temp_storage).Sum(wn, blockDim.x);
    __syncthreads();
    auto& s_total_wn = *reinterpret_cast<int*>(temp_storage);
    if (threadIdx.x == 0) {
      s_total_wn = total_wn;
    }
    __syncthreads();

    if (s_total_wn == 0) {
      return PointLocation::kOutside;
    }
    return PointLocation::kInside;
  }

 private:
  ArrayView<point_t> vertices_;

  DEV_HOST_INLINE bool is_closed() const {
    if (vertices_.empty()) {
      return false;
    }
    return vertices_[0] == vertices_[vertices_.size() - 1];
  }
};

template <typename POINT_T, typename INDEX_T>
class Polygon {
 public:
  using point_t = POINT_T;
  using index_t = INDEX_T;
  using ring_t = LinearRing<point_t>;
  using box_t = Box<Point<float, point_t::n_dim>>;
  using scalar_t = typename point_t::scalar_t;

  Polygon() = default;

  DEV_HOST Polygon(const ArrayView<index_t>& prefix_sum_rings,
                   const ArrayView<point_t>& vertices, const box_t& mbr)
      : prefix_sum_rings_(prefix_sum_rings), vertices_(vertices), mbr_(mbr) {}

  DEV_HOST_INLINE bool empty() const {
    for (size_t i = 0; i < num_rings(); i++) {
      if (!get_ring(i).empty()) {
        return false;
      }
    }
    return true;
  }

  DEV_HOST_INLINE INDEX_T num_rings() const {
    return prefix_sum_rings_.empty() ? 0 : prefix_sum_rings_.size() - 1;
  }

  DEV_HOST_INLINE ring_t get_ring(size_t i) const {
    auto begin_point = prefix_sum_rings_[i];
    auto end_point = prefix_sum_rings_[i + 1];
    return {ArrayView<point_t>(const_cast<point_t*>(vertices_.data()) + begin_point,
                               end_point - begin_point)};
  }

  template <typename TEST_POINT_T>
  DEV_HOST_INLINE typename std::enable_if<TEST_POINT_T::n_dim == 2, bool>::type Contains(
      const TEST_POINT_T& test_point) {
    bool point_is_within = false;
    bool point_on_edge = false;
    // https://web.archive.org/web/20250309050004/https://wrfranklin.org/Research/Short_Notes/pnpoly.html
    // https://github.com/rapidsai/cuspatial/blob/branch-25.08/cpp/include/cuspatial/detail/algorithm/is_point_in_polygon.cuh
    for (int i = 0; i < num_rings(); i++) {
      auto ring = get_ring(i);
      // last point
      auto b = ring.get_point(ring.num_points() - 1);
      bool y0_flag = b.get_coordinate(1) > test_point.get_coordinate(1);
      bool y1_flag;
      for (size_t j = 0; j < ring.num_points(); j++) {
        const auto& a = ring.get_point(j);
        // for each line segment, including the segment between the last and first vertex
        auto run = b.get_coordinate(0) - a.get_coordinate(0);
        auto rise = b.get_coordinate(1) - a.get_coordinate(1);

        // Points on the line segment are the same, so intersection is impossible.
        // This is possible because we allow closed or unclosed polygons.
        scalar_t constexpr zero = 0.0;
        if (float_equal(run, zero) && float_equal(rise, zero)) continue;

        auto rise_to_point = test_point.get_coordinate(1) - a.get_coordinate(1);
        auto run_to_point = test_point.get_coordinate(0) - a.get_coordinate(0);

        // point-on-edge test
        bool is_collinear = float_equal(run * rise_to_point, run_to_point * rise);

        if (is_collinear) {
          auto min_x = a.get_coordinate(0);
          auto max_x = b.get_coordinate(0);
          auto min_y = a.get_coordinate(1);
          auto max_y = b.get_coordinate(1);

          if (min_x > max_x) thrust::swap(min_x, max_x);
          if (min_y > max_y) thrust::swap(min_y, max_y);
          if (min_x <= test_point.get_coordinate(0) &&
              test_point.get_coordinate(0) <= max_x &&
              min_y <= test_point.get_coordinate(1) &&
              test_point.get_coordinate(1) <= max_y) {
            point_on_edge = true;
            break;
          }
        }

        y1_flag = a.get_coordinate(1) > test_point.get_coordinate(1);
        if (y1_flag != y0_flag) {
          // Transform the following inequality to avoid division
          //  test_point.x < (run / rise) * rise_to_point + a.x
          auto lhs = (test_point.get_coordinate(0) - a.get_coordinate(0)) * rise;
          auto rhs = run * rise_to_point;
          if (lhs < rhs != y1_flag) {
            point_is_within = not point_is_within;
          }
        }
        b = a;
        y0_flag = y1_flag;
      }
      if (point_on_edge) {
        point_is_within = false;
        break;
      }
    }

    return point_is_within;
  }

  template <typename TEST_POINT_T>
  DEV_HOST_INLINE typename std::enable_if<TEST_POINT_T::n_dim == 2, PointLocation>::type
  locate_point(const TEST_POINT_T& test_point) const {
    auto rloc = PointLocation::kOutside;

    for (int i = 0; i < num_rings(); i++) {
      auto ring = get_ring(i);
      auto loc = ring.locate_point(test_point);

      if (i == 0) {
        if (loc == PointLocation::kOutside) {
          return PointLocation::kOutside;
        }
        rloc = loc;
      } else {
        if (loc == PointLocation::kInside) {
          return PointLocation::kOutside;
        }
        if (loc == PointLocation::kBoundary) {
          return PointLocation::kBoundary;
        }
      }
    }
    return rloc;
  }

  template <typename TEST_POINT_T>
  DEV_INLINE typename std::enable_if<TEST_POINT_T::n_dim == 2, PointLocation>::type
  locate_point(const TEST_POINT_T& test_point,
               cub::WarpReduce<int>::TempStorage* temp_storage) const {
    auto rloc = PointLocation::kOutside;

    for (int i = 0; i < num_rings(); i++) {
      auto ring = get_ring(i);
      auto loc = ring.locate_point(test_point, temp_storage);
      loc = (PointLocation)__shfl_sync(0xFFFFFFFF, (int)loc, 0);

      if (i == 0) {
        if (loc == PointLocation::kOutside) {
          return PointLocation::kOutside;
        }
        rloc = loc;
      } else {
        if (loc == PointLocation::kInside) {
          return PointLocation::kOutside;
        }
        if (loc == PointLocation::kBoundary) {
          return PointLocation::kBoundary;
        }
      }
    }
    return rloc;
  }

  template <typename TEST_POINT_T>
  DEV_INLINE typename std::enable_if<TEST_POINT_T::n_dim == 2, PointLocation>::type
  locate_point(const TEST_POINT_T& test_point,
               cub::BlockReduce<int, MAX_BLOCK_SIZE>::TempStorage* temp_storage) const {
    auto rloc = PointLocation::kOutside;

    for (int i = 0; i < num_rings(); i++) {
      auto ring = get_ring(i);
      auto loc = ring.locate_point(test_point, temp_storage);

      if (i == 0) {
        if (loc == PointLocation::kOutside) {
          return PointLocation::kOutside;
        }
        rloc = loc;
      } else {
        if (loc == PointLocation::kInside) {
          return PointLocation::kOutside;
        }
        if (loc == PointLocation::kBoundary) {
          return PointLocation::kBoundary;
        }
      }
    }
    return rloc;
  }

  DEV_HOST_INLINE const ArrayView<INDEX_T>& get_prefix_sum_rings() const {
    return prefix_sum_rings_;
  }

  DEV_HOST_INLINE const ArrayView<point_t>& get_vertices() const { return vertices_; }

  DEV_HOST_INLINE uint32_t num_vertices() const {
    uint32_t nv = 0;
    for (int i = 0; i < num_rings(); i++) {
      nv += prefix_sum_rings_[i + 1] - prefix_sum_rings_[i];
    }
    return nv;
  }

  DEV_HOST_INLINE const box_t& get_mbr() const { return mbr_; }

 private:
  ArrayView<INDEX_T> prefix_sum_rings_;
  ArrayView<point_t> vertices_;
  box_t mbr_;
};

/**
 * This class can represent an array of polygons
 * @tparam POINT_T
 */
template <typename POINT_T, typename INDEX_T>
class PolygonArrayView {
  using index_t = INDEX_T;

 public:
  using point_t = POINT_T;
  using box_t = Box<Point<float, point_t::n_dim>>;
  using geometry_t = Polygon<POINT_T, INDEX_T>;
  PolygonArrayView() = default;

  DEV_HOST PolygonArrayView(const ArrayView<index_t>& prefix_sum_polygons,
                            const ArrayView<index_t>& prefix_sum_rings,
                            const ArrayView<point_t>& vertices,
                            const ArrayView<box_t>& mbrs)
      : prefix_sum_polygons_(prefix_sum_polygons),
        prefix_sum_rings_(prefix_sum_rings),
        vertices_(vertices),
        mbrs_(mbrs) {}

  DEV_HOST_INLINE size_t size() const {
    return prefix_sum_polygons_.empty() ? 0 : prefix_sum_polygons_.size() - 1;
  }

  DEV_HOST_INLINE bool empty() const { return size() == 0; }

  DEV_HOST_INLINE Polygon<point_t, index_t> operator[](size_t i) {
    auto ring_begin = prefix_sum_polygons_[i];
    auto ring_end = prefix_sum_polygons_[i + 1];
    auto n_rings = ring_end - ring_begin;

    ArrayView<index_t> prefix_sum_rings(prefix_sum_rings_.data() + ring_begin,
                                        n_rings + 1);
    return Polygon<point_t, index_t>(prefix_sum_rings, vertices_, mbrs_[i]);
  }

  DEV_HOST_INLINE Polygon<point_t, index_t> operator[](size_t i) const {
    auto ring_begin = prefix_sum_polygons_[i];
    auto ring_end = prefix_sum_polygons_[i + 1];
    auto n_rings = ring_end - ring_begin;

    ArrayView<index_t> prefix_sum_rings(
        const_cast<index_t*>(prefix_sum_rings_.data()) + ring_begin, n_rings + 1);
    return Polygon<point_t, index_t>(prefix_sum_rings, vertices_, mbrs_[i]);
  }

  DEV_HOST_INLINE ArrayView<index_t> get_prefix_sum_polygons() const {
    return prefix_sum_polygons_;
  }

  DEV_HOST_INLINE ArrayView<index_t> get_prefix_sum_rings() const {
    return prefix_sum_rings_;
  }

  DEV_HOST_INLINE ArrayView<point_t> get_vertices() const { return vertices_; }

  DEV_HOST_INLINE ArrayView<box_t> mbrs() const { return mbrs_; }

  DEV_HOST_INLINE bool locate_vertex(index_t global_vertex_idx, index_t& polygon_idx,
                                     index_t& ring_idx) const {
    auto it_ring = thrust::upper_bound(thrust::seq, prefix_sum_rings_.begin(),
                                       prefix_sum_rings_.end(), global_vertex_idx);

    if (it_ring != prefix_sum_rings_.end()) {
      // which ring the vertex belongs to
      auto ring_offset = thrust::distance(prefix_sum_rings_.begin(), it_ring) - 1;
      auto it_polygon = thrust::upper_bound(thrust::seq, prefix_sum_polygons_.begin(),
                                            prefix_sum_polygons_.end(), ring_offset);
      if (it_polygon != prefix_sum_polygons_.end()) {
        // which polygon the vertex belongs to
        polygon_idx = thrust::distance(prefix_sum_polygons_.begin(), it_polygon) - 1;
        // which ring of this polygon the vertex belongs to
        ring_idx = ring_offset - prefix_sum_polygons_[polygon_idx];
        return true;
      }
    }
    return false;
  }

 private:
  ArrayView<index_t> prefix_sum_polygons_;
  ArrayView<index_t> prefix_sum_rings_;
  ArrayView<point_t> vertices_;
  ArrayView<box_t> mbrs_;
};

}  // namespace gpuspatial
