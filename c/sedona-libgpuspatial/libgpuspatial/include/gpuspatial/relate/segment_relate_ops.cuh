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

#include "gpuspatial/geom/line_segment.hpp"
#include "gpuspatial/geom/point.hpp"
#include "gpuspatial/geom/polygon.hpp"
#include "gpuspatial/utils/cuda_utils.hpp"
#include "gpuspatial/utils/doubledouble.hpp"
#include "gpuspatial/utils/floating_point.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

namespace gpuspatial {
namespace detail {

enum : uint32_t {
  RAY_MODE_PIP = 0u,
  RAY_MODE_SEG = 1u << 31,

  // Payload 3 bitfield layout for PIP (Point-In-Polygon) ray queries:
  // - Bit 31: Flag indicating whether the test point lies directly on a polygon boundary segment.
  // - Bits 0..30: Accumulated ray-crossing counter across BVH leaf AABBs.
  PIP_POINT_ON_SEG_SHIFT = 31u,
  PIP_POINT_ON_SEG_FLAG = 1u << PIP_POINT_ON_SEG_SHIFT,
  PIP_CROSSING_COUNT_MASK = 0x7fffffffu,

  SEG_HIT_ANY = 1u << 28,
  SEG_HIT_0D = 1u << 29,
  SEG_HIT_1D = 1u << 30,
  SEG_PROPER_CROSS = 1u << 31,
  BVH_GEOM_ID_MASK = 0x0fffffffu,
  SEG_FLAGS_MASK = 0xf0000000u,

  // Maximum number of progression iterations when sweeping along a single probe segment.
  // Guarantees termination even in degenerate geometries or floating-point clustering.
  MAX_SEGMENT_SWEEP_ITERATIONS = 128
};

DEV_HOST_INLINE uint32_t PackPIPCrossingPayload(uint32_t crossing_count,
                                               uint32_t point_on_seg) {
  return ((point_on_seg & 1u) << PIP_POINT_ON_SEG_SHIFT) |
         (crossing_count & PIP_CROSSING_COUNT_MASK);
}

DEV_HOST_INLINE uint32_t UnpackPIPCrossingCount(uint32_t payload) {
  return payload & PIP_CROSSING_COUNT_MASK;
}

DEV_HOST_INLINE uint32_t UnpackPIPPointOnSegment(uint32_t payload) {
  return (payload >> PIP_POINT_ON_SEG_SHIFT) & 1u;
}

DEV_HOST_INLINE void PackDouble(double val, uint32_t& hi, uint32_t& lo) {
#ifdef __CUDA_ARCH__
  hi = __double2hiint(val);
  lo = __double2loint(val);
#else
  uint64_t bits;
  std::memcpy(&bits, &val, sizeof(double));
  hi = static_cast<uint32_t>(bits >> 32);
  lo = static_cast<uint32_t>(bits & 0xffffffffu);
#endif
}

DEV_HOST_INLINE double UnpackDouble(uint32_t hi, uint32_t lo) {
#ifdef __CUDA_ARCH__
  return __hiloint2double(static_cast<int>(hi), static_cast<int>(lo));
#else
  uint64_t bits = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
  double val;
  std::memcpy(&val, &bits, sizeof(double));
  return val;
#endif
}

/**
 * Parametric coordinate tolerance for normalized segment parameters (t, u in [0, 1]).
 *
 * In 2D computational geometry operations such as projecting a point onto a line segment
 * or computing ray-segment intersections, division and subtraction in IEEE 754 floating-point
 * arithmetic incur roundoff error. Exact mathematical values like 0.0 or 1.0 can evaluate
 * to -1e-16 or 1.0000000000000002.
 *
 * - For double (IEEE 754 binary64, 53 mantissa bits, machine epsilon ~ 2.22e-16):
 *   1e-12 provides ~10^4 ULPs of headroom above machine precision to absorb accumulated
 *   rounding errors across cross-products and projections, while remaining 10^12 times
 *   smaller than the segment length, preventing false-positive endpoint classifications.
 *
 * - For float (IEEE 754 binary32, 24 mantissa bits, machine epsilon ~ 1.19e-7):
 *   1e-5f provides headroom above single-precision roundoff noise.
 */
template <typename scalar_t>
DEV_HOST_INLINE constexpr scalar_t ParamTolerance() {
  if constexpr (std::is_same_v<scalar_t, float>) {
    return 1e-5f;
  } else {
    return 1e-12;
  }
}

/**
 * Robust 2D orientation test using filter + DoubleDouble fallback (matching GEOS CGAlgorithmsDD).
 * Returns +1 if q is to the left of p1->p2, -1 if to the right, 0 if collinear.
 */
template <typename POINT_T>
DEV_HOST_INLINE int RobustOrientation(const POINT_T& p1, const POINT_T& p2,
                                      const POINT_T& q) {
  double det_left =
      (static_cast<double>(p1.x()) - static_cast<double>(q.x())) *
      (static_cast<double>(p2.y()) - static_cast<double>(q.y()));
  double det_right =
      (static_cast<double>(p1.y()) - static_cast<double>(q.y())) *
      (static_cast<double>(p2.x()) - static_cast<double>(q.x()));
  double det = det_left - det_right;

  if (det_left * det_right <= 0.0) {
    return (det < 0.0) ? -1 : ((det > 0.0) ? 1 : 0);
  }
  double det_sum = fabs(det_left) + fabs(det_right);
  // Floating-point filter bound (matching Shewchuk 1997 and GEOS CGAlgorithmsDD):
  // For 3 floating-point operations (two multiplications and one subtraction), accumulated
  // roundoff error is bounded by gamma_3 * det_sum ~= 3 * eps_mach * det_sum.
  // With IEEE 754 double precision machine epsilon ~ 2.22e-16, kDpSafeEpsilon = 1e-15
  // (~4.5 * eps_mach) guarantees that if |det| >= err_bound, the sign cannot have flipped
  // due to rounding, so we can safely return the result without DoubleDouble fallback.
  constexpr double kDpSafeEpsilon = 1e-15;
  double err_bound = kDpSafeEpsilon * det_sum;
  if (det >= err_bound || -det >= err_bound) {
    return (det < 0.0) ? -1 : ((det > 0.0) ? 1 : 0);
  }
  DoubleDouble dx1 = DoubleDouble(p2.x()) - DoubleDouble(p1.x());
  DoubleDouble dy1 = DoubleDouble(p2.y()) - DoubleDouble(p1.y());
  DoubleDouble dx2 = DoubleDouble(q.x()) - DoubleDouble(p2.x());
  DoubleDouble dy2 = DoubleDouble(q.y()) - DoubleDouble(p2.y());
  DoubleDouble d = DoubleDouble(dx1 * dy2) - DoubleDouble(dy1 * dx2);
  DoubleDouble const zero(0.0);
  return (d < zero) ? -1 : ((d > zero) ? 1 : 0);
}

template <typename POINT_T>
DEV_HOST_INLINE bool PointInSegmentEnvelope(const POINT_T& p1, const POINT_T& p2,
                                            const POINT_T& q) {
  return ((p1.x() <= q.x() && q.x() <= p2.x()) ||
          (p2.x() <= q.x() && q.x() <= p1.x())) &&
         ((p1.y() <= q.y() && q.y() <= p2.y()) ||
          (p2.y() <= q.y() && q.y() <= p1.y()));
}

/**
 * Project a point Q onto the line defined by segment V1 -> V2, returning parameter t.
 * Assumes V1 != V2.
 */
template <typename POINT_T>
DEV_HOST_INLINE double ProjectPointOnSegment(const POINT_T& v1, const POINT_T& v2,
                                             const POINT_T& q) {
  if (q == v1) return 0.0;
  if (q == v2) return 1.0;
  double dx = static_cast<double>(v2.x()) - static_cast<double>(v1.x());
  double dy = static_cast<double>(v2.y()) - static_cast<double>(v1.y());
  if ( fabs(dx) >= fabs(dy) ) {
    return (static_cast<double>(q.x()) - static_cast<double>(v1.x())) / dx;
  } else {
    return (static_cast<double>(q.y()) - static_cast<double>(v1.y())) / dy;
  }
}

/**
 * Analyze intersection between probe segment V1 -> V2 (at parameter cursor t_cur)
 * and BVH segment Q1 -> Q2.
 * Updates bb_flags (SEG_HIT_ANY, SEG_HIT_0D, SEG_HIT_1D, SEG_PROPER_CROSS),
 * t_next (smallest interior intersection parameter > t_cur), and
 * t_collinear_end (largest collinear cover end starting at or before t_cur).
 */
template <typename POINT_T>
DEV_HOST_INLINE void AnalyzeSegmentIntersection(const POINT_T& v1, const POINT_T& v2,
                                                const POINT_T& q1, const POINT_T& q2,
                                                double t_cur, uint32_t& bb_flags,
                                                double& t_next,
                                                double& t_collinear_end) {
  using scalar_t = typename POINT_T::scalar_t;
  if (q1 == q2) return;

  // Fast bounding-box rejection
  if (fmax(v1.x(), v2.x()) < fmin(q1.x(), q2.x()) ||
      fmin(v1.x(), v2.x()) > fmax(q1.x(), q2.x()) ||
      fmax(v1.y(), v2.y()) < fmin(q1.y(), q2.y()) ||
      fmin(v1.y(), v2.y()) > fmax(q1.y(), q2.y())) {
    return;
  }

  constexpr double eps = static_cast<double>(ParamTolerance<scalar_t>());

  int qp1 = RobustOrientation(q1, q2, v1);
  int qp2 = RobustOrientation(q1, q2, v2);
  int pq1 = RobustOrientation(v1, v2, q1);
  int pq2 = RobustOrientation(v1, v2, q2);

  // Case 1: Collinear segments
  if ((qp1 == 0 && qp2 == 0) || (pq1 == 0 && pq2 == 0)) {
    double tq1 = ProjectPointOnSegment(v1, v2, q1);
    double tq2 = ProjectPointOnSegment(v1, v2, q2);
    double t_lo = fmax(0.0, fmin(tq1, tq2));
    double t_hi = fmin(1.0, fmax(tq1, tq2));

    if (t_hi < t_lo - eps) {
      return;
    }
    if (t_hi <= t_lo + eps) {
      // Collinear segments touch at a single endpoint
      bb_flags |= (SEG_HIT_ANY | SEG_HIT_0D);
      return;
    }

    // 1D collinear overlap on [t_lo, t_hi]
    bb_flags |= (SEG_HIT_ANY | SEG_HIT_1D);
    if (t_lo <= t_cur + eps && t_hi > t_cur + eps) {
      t_collinear_end = fmax(t_collinear_end, t_hi);
    } else if (t_lo > t_cur + eps) {
      t_next = fmin(t_next, t_lo);
    }
    return;
  }

  // Case 2: Non-collinear segments
  // Quick rejection if V1, V2 are strictly on the same side of Q1-Q2
  // or Q1, Q2 are strictly on the same side of V1-V2
  if ((qp1 > 0 && qp2 > 0) || (qp1 < 0 && qp2 < 0) ||
      (pq1 > 0 && pq2 > 0) || (pq1 < 0 && pq2 < 0)) {
    return;
  }

  // Subcase 2a: V1 lies on Q1-Q2 (t = 0.0)
  if (qp1 == 0 && PointInSegmentEnvelope(q1, q2, v1)) {
    bb_flags |= (SEG_HIT_ANY | SEG_HIT_0D);
    return;
  }

  // Subcase 2b: V2 lies on Q1-Q2 (t = 1.0)
  if (qp2 == 0 && PointInSegmentEnvelope(q1, q2, v2)) {
    bb_flags |= (SEG_HIT_ANY | SEG_HIT_0D);
    return;
  }

  // Subcase 2c: Q1 lies on V1-V2
  if (pq1 == 0 && PointInSegmentEnvelope(v1, v2, q1)) {
    bb_flags |= (SEG_HIT_ANY | SEG_HIT_0D);
    double t = ProjectPointOnSegment(v1, v2, q1);
    if (t > t_cur + eps && t < 1.0 - eps) {
      t_next = fmin(t_next, t);
    }
    return;
  }

  // Subcase 2d: Q2 lies on V1-V2
  if (pq2 == 0 && PointInSegmentEnvelope(v1, v2, q2)) {
    bb_flags |= (SEG_HIT_ANY | SEG_HIT_0D);
    double t = ProjectPointOnSegment(v1, v2, q2);
    if (t > t_cur + eps && t < 1.0 - eps) {
      t_next = fmin(t_next, t);
    }
    return;
  }

  // Subcase 2e: Proper crossing between interior of V1-V2 and interior of Q1-Q2
  if (((qp1 > 0 && qp2 < 0) || (qp1 < 0 && qp2 > 0)) &&
      ((pq1 > 0 && pq2 < 0) || (pq1 < 0 && pq2 > 0))) {
    double dx_p = static_cast<double>(v2.x()) - static_cast<double>(v1.x());
    double dy_p = static_cast<double>(v2.y()) - static_cast<double>(v1.y());
    double dx_q = static_cast<double>(q2.x()) - static_cast<double>(q1.x());
    double dy_q = static_cast<double>(q2.y()) - static_cast<double>(q1.y());
    double denom = dx_p * dy_q - dy_p * dx_q;
    if (denom != 0.0) {
      double t = (dy_q * (static_cast<double>(q1.x()) - static_cast<double>(v1.x())) -
                  dx_q * (static_cast<double>(q1.y()) - static_cast<double>(v1.y()))) /
                 denom;
      t = fmax(0.0, fmin(1.0, t));
      bb_flags |= (SEG_HIT_ANY | SEG_HIT_0D | SEG_PROPER_CROSS);
      if (t > t_cur) {
        t_next = fmin(t_next, t);
      }
    }
  }
}

/**
 * Find a guaranteed interior point of a valid non-empty Polygon.
 * Used only in the rare case where 100% of the polygon's boundary is collinearly
 * covered by the other geometry's boundary (so neither BOUND_INTER nor BOUND_EXTER
 * was triggered).
 *
 * Works by taking the midpoint M of the first non-degenerate segment of the exterior
 * ring, determining the inward normal direction from the exterior ring's signed area,
 * and finding the closest ray-segment intersection s_min > 0 with any ring of the
 * polygon. The midpoint M + 0.5 * s_min * n is guaranteed to lie strictly in I(poly).
 */
template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE bool FindPolygonInteriorPoint(const Polygon<POINT_T, INDEX_T>& poly,
                                              POINT_T& out_pt) {
  using scalar_t = typename POINT_T::scalar_t;
  if (poly.empty()) return false;
  auto ext_ring = poly.get_ring(0);
  if (ext_ring.num_segments() < 3) return false;

  // 1. Compute twice the signed area of the exterior ring using shoelace formula
  //    (translated relative to first vertex for numerical stability)
  const auto& p0 = ext_ring.get_point(0);
  double signed_area2 = 0.0;
  for (size_t i = 0; i < ext_ring.num_segments(); ++i) {
    const auto& a = ext_ring.get_point(i);
    const auto& b = ext_ring.get_point(i + 1);
    double ax = static_cast<double>(a.x()) - static_cast<double>(p0.x());
    double ay = static_cast<double>(a.y()) - static_cast<double>(p0.y());
    double bx = static_cast<double>(b.x()) - static_cast<double>(p0.x());
    double by = static_cast<double>(b.y()) - static_cast<double>(p0.y());
    signed_area2 += (ax * by - bx * ay);
  }

  // Try each segment of the exterior ring until a clean inward ray hit is found
  for (size_t seg_idx = 0; seg_idx < ext_ring.num_segments(); ++seg_idx) {
    const auto& v1 = ext_ring.get_point(seg_idx);
    const auto& v2 = ext_ring.get_point(seg_idx + 1);
    if (v1 == v2) continue;

    double dx = static_cast<double>(v2.x()) - static_cast<double>(v1.x());
    double dy = static_cast<double>(v2.y()) - static_cast<double>(v1.y());
    // For CCW ring (signed_area2 > 0), interior is to the left: (-dy, dx).
    // For CW ring (signed_area2 < 0), interior is to the right: (dy, -dx).
    double nx = (signed_area2 >= 0.0) ? -dy : dy;
    double ny = (signed_area2 >= 0.0) ? dx : -dx;

    double mx = 0.5 * (static_cast<double>(v1.x()) + static_cast<double>(v2.x()));
    double my = 0.5 * (static_cast<double>(v1.y()) + static_cast<double>(v2.y()));

    double s_min = std::numeric_limits<double>::max();
    for (size_t r = 0; r < poly.num_rings(); ++r) {
      auto ring = poly.get_ring(r);
      for (size_t j = 0; j < ring.num_segments(); ++j) {
        if (r == 0 && j == seg_idx) continue;
        const auto& q1 = ring.get_point(j);
        const auto& q2 = ring.get_point(j + 1);
        if (q1 == q2) continue;

        double qx = static_cast<double>(q2.x()) - static_cast<double>(q1.x());
        double qy = static_cast<double>(q2.y()) - static_cast<double>(q1.y());
        double denom = nx * qy - ny * qx;
        // denom is the 2D cross product of ray normal and segment direction (nx * qy - ny * qx).
        // Mathematically, denom == 0 when the ray and segment are parallel or collinear.
        // In IEEE 754 double precision, rounding can yield ~1e-17 instead of exact 0.0.
        // kParallelEpsilon = 1e-15 (~4.5 * eps_mach) guards against division-by-zero or subnormal
        // overflow (which produces NaN/Inf) when testing against parallel or collinear edges.
        constexpr double kParallelEpsilon = 1e-15;
        if (fabs(denom) < kParallelEpsilon) continue;

        double diff_x = static_cast<double>(q1.x()) - mx;
        double diff_y = static_cast<double>(q1.y()) - my;
        double s = (diff_x * qy - diff_y * qx) / denom;
        double u = (diff_x * ny - diff_y * nx) / denom;

        // Ray equation: R(s) = M + s * n (s > 0 is inward into the polygon).
        // Segment equation: Q(u) = Q1 + u * (Q2 - Q1) for u in [0, 1].
        //
        // eps (ParamTolerance<double>):
        // - s > eps: Avoids false self-intersections with the ray origin M on the
        //            boundary due to floating-point imprecision on adjacent segments.
        // - u >= -eps && u <= 1.0 + eps: Accommodates floating-point roundoff when the
        //                                inward ray strikes an exact vertex of a ring.
        constexpr double eps = static_cast<double>(ParamTolerance<double>());
        if (s > eps && u >= -eps && u <= 1.0 + eps) {
          if (s < s_min) {
            s_min = s;
          }
        }
      }
    }

    if (s_min < std::numeric_limits<double>::max()) {
      POINT_T candidate(static_cast<scalar_t>(mx + 0.5 * s_min * nx),
                        static_cast<scalar_t>(my + 0.5 * s_min * ny));
      if (poly.locate_point(candidate) == PointLocation::kInside) {
        out_pt = candidate;
        return true;
      }
    }
  }
  return false;
}

}  // namespace detail
}  // namespace gpuspatial
