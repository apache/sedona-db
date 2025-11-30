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
#include "gpuspatial/geom/point.cuh"
#include "gpuspatial/utils/cuda_utils.h"
#include "gpuspatial/utils/floating_point.h"

namespace gpuspatial {
template <typename POINT_T>
class LineSegment {
  using point_t = POINT_T;
  using scalar_t = typename point_t::scalar_t;
  static constexpr int n_dim = point_t::n_dim;
  using box_t = Box<point_t>;

 public:
  LineSegment() = default;
  DEV_HOST LineSegment(const point_t& p1, const point_t& p2) : p1_(p1), p2_(p2) {}

  DEV_HOST_INLINE const point_t& get_p1() const { return p1_; }

  DEV_HOST_INLINE const point_t& get_p2() const { return p2_; }

  DEV_HOST_INLINE point_t centroid() const {
    point_t c;
    for (int i = 0; i < n_dim; i++) {
      c.set_coordinate(i, (p1_.get_coordinate(i) + p2_.get_coordinate(i)) / 2.0);
    }
    return c;
  }

  DEV_HOST_INLINE int orientation(const point_t& q) const {
    auto d_x = (q.x() - p1_.x());
    auto d_y = (q.y() - p1_.y());
    typename point_t::scalar_t constexpr zero = 0.0;

    if (float_equal(d_x, zero) && float_equal(d_y, zero)) {
      return 0;
    }
    auto v1 = d_x * (p2_.y() - p1_.y());
    auto v2 = (p2_.x() - p1_.x()) * d_y;

    if (float_equal(v1, v2)) {
      return 0;
    }
    auto side = v1 - v2;
    return side < 0 ? -1 : 1;
  }

  DEV_HOST_INLINE box_t get_mbr() const {
    point_t min_p, max_p;
    for (int dim = 0; dim < n_dim; dim++) {
      min_p.set_coordinate(dim, std::numeric_limits<scalar_t>::max());
      max_p.set_coordinate(dim, std::numeric_limits<scalar_t>::lowest());
    }

    for (int dim = 0; dim < n_dim; dim++) {
      auto v1 = p1_.get_coordinate(dim);
      auto v2 = p2_.get_coordinate(dim);
      auto min_v = std::min(v1, v2);
      auto max_v = std::max(v1, v2);
      min_p.set_coordinate(dim, std::min(min_p.get_coordinate(dim), min_v));
      max_p.set_coordinate(dim, std::max(max_p.get_coordinate(dim), max_v));
    }
    return box_t(min_p, max_p);
  }

  template <typename point_type = POINT_T,
            typename std::enable_if<point_type::n_dim == 2, bool>::type = true>
  DEV_HOST_INLINE bool covers(const point_type& q) const {
    auto side = ((q.x() - p1_.x()) * (p2_.y() - p1_.y()) -
                 (p2_.x() - p1_.x()) * (q.y() - p1_.y()));

    if (side == 0) {
      return (p1_.x() <= q.x() && q.x() <= p2_.x()) ||
             (p1_.x() >= q.x() && q.x() >= p2_.x()) ||
             (p1_.y() <= q.y() && q.y() <= p2_.y()) ||
             (p1_.y() >= q.y() && q.y() >= p2_.y());
    }
    return false;
  }

  template <typename point_type = POINT_T,
            typename std::enable_if<point_type::n_dim == 2, bool>::type = true>
  DEV_HOST_INLINE PointLocation locate_point(const point_t& q) const {
    if (orientation(q) == 0) {
      if (((p1_.x() <= q.x() && q.x() <= p2_.x()) ||
           (p2_.x() <= q.x() && q.x() <= p1_.x())) &&
          ((p1_.y() <= q.y() && q.y() <= p2_.y()) ||
           (p2_.y() <= q.y() && q.y() <= p1_.y()))) {
        if ((p1_.x() == q.x() && p1_.y() == q.y()) ||
            (p2_.x() == q.x() && p2_.y() == q.y()))
          return PointLocation::kBoundary;
        return PointLocation::kInside;
      }
    }

    return PointLocation::kOutside;
  }

 private:
  point_t p1_, p2_;
};

}  // namespace gpuspatial
