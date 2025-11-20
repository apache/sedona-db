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
#include "gpuspatial/geom/point.cuh"
#include "gpuspatial/utils/cuda_utils.h"
#include "gpuspatial/utils/doubledouble.h"

namespace gpuspatial {

/**
 * The RayCrossingCounter simulates a ray casting from a point toward the positive y-axis
 * and counts the number of intersections. The intersection status are stored with two
 * uint32_t numbers, so that RayCrossingCounter can be packed/unpacked to be used in OptiX
 */
class RayCrossingCounter {
  enum { RIGHT = -1, LEFT = 1, STRAIGHT = 0, FAILURE = 2 };
  uint32_t crossing_count_;
  // true if the test point lies on an input segment
  uint32_t point_on_segment_;

 public:
  DEV_HOST_INLINE uint32_t& get_crossing_count() { return crossing_count_; }
  DEV_HOST_INLINE uint32_t& get_point_on_segment() { return point_on_segment_; }

  RayCrossingCounter() = default;

  DEV_HOST RayCrossingCounter(uint32_t crossing_count, uint32_t point_on_segment)
      : crossing_count_(crossing_count), point_on_segment_(point_on_segment) {}

  DEV_HOST_INLINE void Init() {
    crossing_count_ = 0;
    point_on_segment_ = 0;
  }

  /** \brief
   * Counts a segment
   * @param point test point
   * @param p1 an endpoint of the segment
   * @param p2 another endpoint of the segment
   */
  template <typename POINT_T>
  DEV_HOST_INLINE void countSegment(const POINT_T& point, const POINT_T& p1,
                                    const POINT_T& p2) {
    auto max_x = fmax(p1.x(), p2.x());
    if (max_x < point.x()) {
      return;
    }
    int current_crossing_count = 0;
    int is_on_segment = 0;

    is_on_segment = point.x() == p2.x() && point.y() == p2.y();
    const bool is_horizontal_on_ray = p1.y() == point.y() && p2.y() == point.y();

    if (is_horizontal_on_ray) {
      auto minx = fmin(p1.x(), p2.x());
      const int is_on_horizontal = point.x() >= minx && point.x() <= max_x;

      is_on_segment = is_on_segment || is_on_horizontal;
    }

    if (!is_horizontal_on_ray) {
      const bool crosses_ray_y = (p1.y() > point.y() && p2.y() <= point.y()) ||
                                 (p2.y() > point.y() && p1.y() <= point.y());

      if (crosses_ray_y) {
        int sign = orientation(p1, p2, point);

        is_on_segment = is_on_segment || sign == 0;

        if (sign != 0) {
          sign = p2.y() < p1.y() ? -sign : sign;
          current_crossing_count = sign > 0;
        }
      }
    }
    if (is_on_segment) {
      point_on_segment_ = 1;
    }

    if (point_on_segment_ == 0) {
      crossing_count_ += current_crossing_count;
    }
  }

  DEV_HOST_INLINE PointLocation location() const {
    if (point_on_segment_ == 1) {
      return PointLocation::kBoundary;
    }

    return (crossing_count_ % 2) == 1 ? PointLocation::kInside : PointLocation::kOutside;
  }

 private:
  DEV_HOST_INLINE static int orientation(double x) {
    return (x < 0.0) ? RIGHT : ((x > 0.0) ? LEFT : STRAIGHT);
  }

  DEV_HOST_INLINE static int orientation(const DoubleDouble& x) {
    DoubleDouble const zero(0.0);
    return (x < zero) ? RIGHT : ((x > zero) ? LEFT : STRAIGHT);
  }

  template <typename POINT_T>
  DEV_HOST_INLINE static int orientation(const POINT_T& p1, const POINT_T& p2,
                                         const POINT_T& q) {
    using scalar_t = typename POINT_T::scalar_t;
    auto det_left = (p1.x() - q.x()) * (p2.y() - q.y());
    auto det_right = (p1.y() - q.y()) * (p2.x() - q.x());
    auto det = det_left - det_right;
    scalar_t zero = 0.0;
    // This is a rewrite of GEOS's orientation algorithm for the GPU to reduce branches

    // Check for the "safe" orientation cases first.
    // The quick exit conditions are when det_left and det_right have opposite signs,
    // or when one of them is zero (including det_left = 0).

    // Condition for safe return: sign(det_left) != sign(det_right) OR det_left == 0.
    // (det_left > 0 and det_right <= 0) OR (det_left < 0 and det_right >= 0) OR (det_left
    // == 0)

    // Combine the two opposite-sign conditions:
    // (det_left * det_right) <= zero covers all cases where signs are opposite or one is
    // zero.
    if (det_left * det_right <= zero) {
      return orientation(det);
    }

    // If we reach here, it means det_left and det_right have the same sign (and are
    // non-zero).
    assert(det_left * det_right > 0);
    // We must calculate det_sum: det_sum = |det_left| + |det_right|

    // Since they have the same sign (or are both zero), this is always true:
    // |det_left| + |det_right| == |det_left + det_right| OR -|det_left + det_right|
    // A safer way is to use the absolute value function:
    auto det_sum = fabs(det_left) + fabs(det_right);

    // OR, since they have the same sign, we can use:
    // det_sum = fabs(det_left + det_right); // This is mathematically equivalent
    // OR, even simpler given the C++ context:
    // det_sum = (det_left > 0) ? (det_left + det_right) : (-det_left - det_right);

    double constexpr DP_SAFE_EPSILON = 1e-15;
    double const err_bound = DP_SAFE_EPSILON * det_sum;
    if (det >= err_bound || -det >= err_bound) {
      return orientation(det);
    }
    // Cannot determine with double, using double double then
    DoubleDouble dx1 = DoubleDouble(p2.x()) - DoubleDouble(p1.x());
    DoubleDouble dy1 = DoubleDouble(p2.y()) - DoubleDouble(p1.y());
    DoubleDouble dx2 = DoubleDouble(q.x()) - DoubleDouble(p2.x());
    DoubleDouble dy2 = DoubleDouble(q.y()) - DoubleDouble(p2.y());

    // cross product
    DoubleDouble d = DoubleDouble(dx1 * dy2) - DoubleDouble(dy1 * dx2);
    return orientation(d);
  }
};

}  // namespace gpuspatial
