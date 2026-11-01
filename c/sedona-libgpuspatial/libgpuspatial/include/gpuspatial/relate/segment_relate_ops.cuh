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
  SEG_FLAGS_MASK = 0xf0000000u
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
      if (t > t_cur + eps) {
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
 * Implements GEOS's InteriorPointArea scanline algorithm (InteriorPointArea.cpp):
 * 1. Find a scanline Y that bisects the polygon's Y extent without coinciding with
 *    any vertex Y coordinate.
 * 2. Intersect the horizontal scanline with all non-horizontal segments of all rings.
 * 3. Sort the intersection X coordinates (which occur in pairs by Jordan Curve Theorem)
 *    and choose the midpoint of the widest interior interval.
 *
 * This method is completely scale-free (works identically on nanometer and planetary
 * coordinates) and singularity-free because the scanline is guaranteed not to touch
 * any vertex.
 */
template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE bool FindPolygonInteriorPointScanline(const Polygon<POINT_T, INDEX_T>& poly,
                                                      POINT_T& out_pt) {
  using scalar_t = typename POINT_T::scalar_t;
  if (poly.empty()) return false;
  auto ext_ring = poly.get_ring(0);
  if (ext_ring.num_segments() < 3) return false;

  const auto& mbr = poly.get_mbr();
  double min_y = static_cast<double>(mbr.get_min().y());
  double max_y = static_cast<double>(mbr.get_max().y());
  if (min_y >= max_y) return false;

  double centre_y = 0.5 * (min_y + max_y);
  double lo_y = min_y;
  double hi_y = max_y;

  // Find vertex Y ordinates closest to centre_y from below and above
  for (size_t r = 0; r < poly.num_rings(); ++r) {
    auto ring = poly.get_ring(r);
    for (size_t i = 0; i < ring.num_points(); ++i) {
      double y = static_cast<double>(ring.get_point(i).y());
      if (y <= centre_y) {
        if (y > lo_y) lo_y = y;
      } else {
        if (y < hi_y) hi_y = y;
      }
    }
  }

  // Guaranteed: no vertex lies strictly between lo_y and hi_y.
  // Hence scan_y intersects only interior segment bodies, never vertices or horizontal edges.
  double scan_y = 0.5 * (lo_y + hi_y);

  // Collect and sort crossing X coordinates on the scanline using insertion sort
  // (registers/local memory, no dynamic heap allocation)
  constexpr int kMaxCrossings = 64;
  double crossings[kMaxCrossings];
  int num_crossings = 0;

  for (size_t r = 0; r < poly.num_rings(); ++r) {
    auto ring = poly.get_ring(r);
    for (size_t i = 0; i < ring.num_segments(); ++i) {
      const auto& p0 = ring.get_point(i);
      const auto& p1 = ring.get_point(i + 1);
      double y0 = static_cast<double>(p0.y());
      double y1 = static_cast<double>(p1.y());

      // Skip segments completely above or below the scanline
      if ((y0 > scan_y && y1 > scan_y) || (y0 < scan_y && y1 < scan_y)) continue;
      // Skip horizontal segments
      if (y0 == y1) continue;

      double x0 = static_cast<double>(p0.x());
      double x1 = static_cast<double>(p1.x());
      double x_int = (x0 == x1) ? x0 : (x0 + (scan_y - y0) * (x1 - x0) / (y1 - y0));

      // Insertion sort into fixed-capacity buffer
      if (num_crossings < kMaxCrossings) {
        int idx = num_crossings - 1;
        while (idx >= 0 && crossings[idx] > x_int) {
          crossings[idx + 1] = crossings[idx];
          --idx;
        }
        crossings[idx + 1] = x_int;
        ++num_crossings;
      } else if (x_int < crossings[kMaxCrossings - 1]) {
        int idx = kMaxCrossings - 2;
        while (idx >= 0 && crossings[idx] > x_int) {
          crossings[idx + 1] = crossings[idx];
          --idx;
        }
        crossings[idx + 1] = x_int;
      }
    }
  }

  if (num_crossings < 2) return false;

  // Find widest interior interval [crossings[2k], crossings[2k+1]]
  double max_width = 0.0;
  double best_mid_x = 0.0;
  bool found = false;

  for (int i = 0; i + 1 < num_crossings; i += 2) {
    double width = crossings[i + 1] - crossings[i];
    if (width > max_width || !found) {
      max_width = width;
      best_mid_x = 0.5 * (crossings[i] + crossings[i + 1]);
      found = true;
    }
  }

  if (found) {
    POINT_T candidate(static_cast<scalar_t>(best_mid_x), static_cast<scalar_t>(scan_y));
    if (poly.locate_point(candidate) == PointLocation::kInside) {
      out_pt = candidate;
      return true;
    }
  }

  return false;
}

/**
 * Bulletproof Fallback: Scale-Normalized Extremal-Convex-Vertex Angle Bisector.
 *
 * Used if the horizontal scanline bisector could not yield a confirmed interior point
 * (e.g. extremely complex geometry where crossings exceeded kMaxCrossings = 64
 * and the preserved slices were thin slivers, or an extreme degenerate aspect ratio).
 *
 * Computational Geometry Guarantee:
 * 1. The vertex V_min with the minimum X coordinate (and minimum Y if tied) lies
 *    strictly on the 2D convex hull of the polygon.
 * 2. Therefore, V_min is strictly convex (interior angle < 180 deg) and there are
 *    zero polygon vertices or edges to the left of V_min.
 * 3. The angle bisector vector B of the two edges meeting at V_min points strictly
 *    into the interior cone of the polygon.
 * 4. By ray-marching along B against all other polygon segments using scale-normalized
 *    vector cross products, we determine the distance s_min > 0 to the closest
 *    boundary intersection.
 * 5. The midpoint V_min + 0.5 * s_min * B is mathematically guaranteed to lie
 *    strictly in I(poly), requiring O(1) registers and zero buffer allocations.
 */
template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE bool FindPolygonInteriorPointExtremalVertex(const Polygon<POINT_T, INDEX_T>& poly,
                                                           POINT_T& out_pt) {
  using scalar_t = typename POINT_T::scalar_t;
  if (poly.empty()) return false;
  auto ext_ring = poly.get_ring(0);
  size_t n_pts = ext_ring.num_points();
  if (n_pts < 4) return false;
  size_t n_segs = ext_ring.num_segments();

  size_t min_idx = 0;
  double min_x = static_cast<double>(ext_ring.get_point(0).x());
  double min_y_coord = static_cast<double>(ext_ring.get_point(0).y());

  for (size_t i = 1; i < n_segs; ++i) {
    const auto& pt = ext_ring.get_point(i);
    double px = static_cast<double>(pt.x());
    double py = static_cast<double>(pt.y());
    if (px < min_x || (px == min_x && py < min_y_coord)) {
      min_x = px;
      min_y_coord = py;
      min_idx = i;
    }
  }

  const auto& v_curr = ext_ring.get_point(min_idx);
  const auto& v_prev = ext_ring.get_point((min_idx == 0) ? (n_segs - 1) : (min_idx - 1));
  const auto& v_next = ext_ring.get_point(min_idx + 1);

  double e1_x = static_cast<double>(v_prev.x()) - static_cast<double>(v_curr.x());
  double e1_y = static_cast<double>(v_prev.y()) - static_cast<double>(v_curr.y());
  double e2_x = static_cast<double>(v_next.x()) - static_cast<double>(v_curr.x());
  double e2_y = static_cast<double>(v_next.y()) - static_cast<double>(v_curr.y());

  double l1 = sqrt(e1_x * e1_x + e1_y * e1_y);
  double l2 = sqrt(e2_x * e2_x + e2_y * e2_y);
  if (l1 <= 0.0 || l2 <= 0.0) return false;

  double u1_x = e1_x / l1;
  double u1_y = e1_y / l1;
  double u2_x = e2_x / l2;
  double u2_y = e2_y / l2;

  // Normalized angle bisector vector
  double bx = u1_x + u2_x;
  double by = u1_y + u2_y;
  double blen = sqrt(bx * bx + by * by);
  if (blen <= 1e-15) {
    // 180-degree flat/spike edge: perpendicular normal pointing right (+X)
    bx = -u1_y;
    by = u1_x;
    blen = sqrt(bx * bx + by * by);
    if (blen <= 1e-15) return false;
  }
  bx /= blen;
  by /= blen;

  // Ray equation: R(s) = v_curr + s * B.
  // Initial upper bound for s: half of the shortest adjacent edge length
  double s_min = 0.5 * fmin(l1, l2);

  for (size_t r = 0; r < poly.num_rings(); ++r) {
    auto ring = poly.get_ring(r);
    for (size_t j = 0; j < ring.num_segments(); ++j) {
      if (r == 0 && (j == min_idx || (min_idx > 0 && j == min_idx - 1) ||
                     (min_idx == 0 && j == n_segs - 1))) {
        continue;
      }
      const auto& q1 = ring.get_point(j);
      const auto& q2 = ring.get_point(j + 1);
      double qx = static_cast<double>(q2.x()) - static_cast<double>(q1.x());
      double qy = static_cast<double>(q2.y()) - static_cast<double>(q1.y());
      double qlen = sqrt(qx * qx + qy * qy);
      if (qlen <= 0.0) continue;

      // Scale-normalized cross product denominator (sin theta of ray and segment)
      double denom = bx * qy - by * qx;
      // Dimensionless threshold: avoid division by near-zero if nearly parallel
      if (fabs(denom) < 1e-12 * qlen) continue;

      double diff_x = static_cast<double>(q1.x()) - static_cast<double>(v_curr.x());
      double diff_y = static_cast<double>(q1.y()) - static_cast<double>(v_curr.y());
      double s = (diff_x * qy - diff_y * qx) / denom;
      double u = (diff_x * by - diff_y * bx) / denom;

      constexpr double eps = static_cast<double>(ParamTolerance<double>());
      if (s > eps && u >= -eps && u <= 1.0 + eps) {
        if (s < s_min) {
          s_min = s;
        }
      }
    }
  }

  if (s_min > 0.0) {
    POINT_T candidate(
        static_cast<scalar_t>(static_cast<double>(v_curr.x()) + 0.5 * s_min * bx),
        static_cast<scalar_t>(static_cast<double>(v_curr.y()) + 0.5 * s_min * by));
    if (poly.locate_point(candidate) == PointLocation::kInside) {
      out_pt = candidate;
      return true;
    }
  }

  return false;
}

/**
 * Finds a point strictly in the interior of a polygon (point in I(poly)).
 *
 * Tries the primary GEOS-style scanline bisector algorithm first. If it cannot find
 * or confirm an interior point, falls back to the extremal-convex-vertex angle bisector.
 */
template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE bool FindPolygonInteriorPoint(const Polygon<POINT_T, INDEX_T>& poly,
                                              POINT_T& out_pt) {
  if (FindPolygonInteriorPointScanline(poly, out_pt)) {
    return true;
  }
  return FindPolygonInteriorPointExtremalVertex(poly, out_pt);
}

}  // namespace detail
}  // namespace gpuspatial
