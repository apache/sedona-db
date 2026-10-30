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

#include "gpuspatial/geom/ray_crossing_counter.hpp"
#include "gpuspatial/relate/intersection_matrix.hpp"
#include "gpuspatial/relate/segment_relate_ops.cuh"
#include "gpuspatial/utils/cuda_utils.hpp"
#include "gpuspatial/utils/floating_point.hpp"
#include "gpuspatial/utils/helpers.cuh"
#include "shader_config.hpp"

#include <optix_device.h>
#include <utility>

#ifndef DEV_INLINE
#define DEV_INLINE __device__ __forceinline__
#endif

namespace gpuspatial {
namespace detail {

/**
 * Fast-path check for empty geometries in DE-9IM relate queries.
 * Sets the appropriate disjoint/empty intersection matrix and returns true if either
 * geometry is empty.
 */
template <typename PARAMS_T>
DEV_INLINE bool HandleEmptyGeometries(const PARAMS_T& params, uint32_t pair_idx,
                                      uint32_t thread_idx, bool probe_empty,
                                      bool bvh_empty) {
  if (probe_empty || bvh_empty) {
    if (thread_idx == 0) {
      int im = IntersectionMatrix::EXTER_EXTER_2D;
      if (!probe_empty) {
        im |= IntersectionMatrix::INTER_EXTER_2D | IntersectionMatrix::BOUND_EXTER_1D;
      }
      if (!bvh_empty) {
        im |= IntersectionMatrix::EXTER_INTER_2D | IntersectionMatrix::EXTER_BOUND_1D;
      }
      atomicOr(const_cast<int*>(&params.IMs[pair_idx]),
               params.transpose ? IntersectionMatrix::Transpose(im) : im);
    }
    return true;
  }
  return false;
}

/**
 * Checks whether the current accumulated DE-9IM matrix has reached a state where
 * all key boundary-interior / boundary-exterior interactions are established,
 * permitting an early exit from segment evaluation.
 */
DEV_INLINE bool ShouldEarlyExitRelate(int cur_im, bool transpose) {
  if (transpose) {
    return (cur_im & IntersectionMatrix::INTER_BOUND_2D) &&
           (cur_im & IntersectionMatrix::EXTER_BOUND_2D);
  } else {
    return (cur_im & IntersectionMatrix::BOUND_INTER_2D) &&
           (cur_im & IntersectionMatrix::BOUND_EXTER_2D) &&
           (cur_im & IntersectionMatrix::BOUND_BOUND_1D);
  }
}

/**
 * Evaluates boundary interactions in an OptiX leaf AABB for Ray Mode SEG.
 * Unpacks the parametric state, checks segments against the BVH ring, updates
 * event parameters, and packs results back into OptiX payloads.
 */
template <typename POINT_T, typename RING_T, typename OFFSETS_T>
DEV_INLINE void ProcessSegmentIntersectionLeaf(
    const POINT_T& v1, const POINT_T& v2, const RING_T& ring,
    const OFFSETS_T& vertex_offsets, uint32_t target_geom_id) {
  uint32_t bb_flags = optixGetPayload_1() & detail::SEG_FLAGS_MASK;
  double t_cur = detail::UnpackDouble(optixGetPayload_2(), optixGetPayload_3());
  double t_next = detail::UnpackDouble(optixGetPayload_4(), optixGetPayload_5());
  double t_collinear_end =
      detail::UnpackDouble(optixGetPayload_6(), optixGetPayload_7());

  for (auto v_off = vertex_offsets.first; v_off < vertex_offsets.second; ++v_off) {
    const auto& q1 = ring.get_point(v_off);
    const auto& q2 = ring.get_point(v_off + 1);
    detail::AnalyzeSegmentIntersection(v1, v2, q1, q2, t_cur, bb_flags, t_next,
                                       t_collinear_end);
  }

  optixSetPayload_1((bb_flags & detail::SEG_FLAGS_MASK) |
                    (target_geom_id & detail::BVH_GEOM_ID_MASK));
  uint32_t next_hi, next_lo, col_hi, col_lo;
  detail::PackDouble(t_next, next_hi, next_lo);
  detail::PackDouble(t_collinear_end, col_hi, col_lo);
  optixSetPayload_4(next_hi);
  optixSetPayload_5(next_lo);
  optixSetPayload_6(col_hi);
  optixSetPayload_7(col_lo);
}

/**
 * Sweeps along a single probe segment [v1, v2] from t = 0 to t = 1, advancing from
 * event to event and testing intermediate open spans for point location against the BVH.
 * Guaranteed to terminate without infinite loops via the stagnation guard (t_next <= t_cur + eps).
 */
template <typename POINT_T, typename TRACE_SEG_FN, typename LOCATE_PT_FN>
DEV_INLINE int SweepSegment(const POINT_T& v1, const POINT_T& v2,
                            uint32_t abs_v_idx,
                            TRACE_SEG_FN&& trace_segment_fn,
                            LOCATE_PT_FN&& locate_point_fn) {
  using scalar_t = typename POINT_T::scalar_t;
  double dx = static_cast<double>(v2.x()) - static_cast<double>(v1.x());
  double dy = static_cast<double>(v2.y()) - static_cast<double>(v1.y());
  double L = fmax(fabs(dx), fabs(dy));
  float3 seg_dir =
      (L > 0.0) ? float3{static_cast<float>(dx / L), static_cast<float>(dy / L), 0.0f}
                : float3{1.0f, 0.0f, 0.0f};
  float seg_tmin = 0.0f;
  float seg_tmax = (L > 0.0) ? next_float_from_double(L, 1, 4) : 0.0f;

  double t_cur = 0.0;
  constexpr double eps = static_cast<double>(detail::ParamTolerance<scalar_t>());
  int seg_im = 0;

  while (t_cur < 1.0 - eps) {
    uint32_t bb_flags = 0;
    double t_next = 1.0;
    double t_collinear_end = t_cur;
    trace_segment_fn(seg_dir, seg_tmin, seg_tmax, abs_v_idx, t_cur, bb_flags,
                     t_next, t_collinear_end);

    if (bb_flags & detail::SEG_HIT_1D) {
      seg_im |= IntersectionMatrix::BOUND_BOUND_1D;
    } else if (bb_flags & detail::SEG_HIT_0D) {
      seg_im |= IntersectionMatrix::BOUND_BOUND_0D;
    }
    if (bb_flags & detail::SEG_PROPER_CROSS) {
      seg_im |=
          IntersectionMatrix::BOUND_INTER_1D | IntersectionMatrix::INTER_INTER_2D |
          IntersectionMatrix::EXTER_INTER_2D | IntersectionMatrix::BOUND_EXTER_1D |
          IntersectionMatrix::INTER_EXTER_2D | IntersectionMatrix::EXTER_EXTER_2D;
    }

    if ((bb_flags & detail::SEG_HIT_ANY) == 0 && t_cur == 0.0) {
      POINT_T mid(static_cast<scalar_t>(0.5 * (v1.x() + v2.x())),
                  static_cast<scalar_t>(0.5 * (v1.y() + v2.y())));
      PointLocation loc = locate_point_fn(mid);
      if (loc == PointLocation::kInside) {
        seg_im |= IntersectionMatrix::BOUND_INTER_1D |
                  IntersectionMatrix::INTER_INTER_2D |
                  IntersectionMatrix::EXTER_INTER_2D;
      } else if (loc == PointLocation::kOutside) {
        seg_im |= IntersectionMatrix::BOUND_EXTER_1D |
                  IntersectionMatrix::INTER_EXTER_2D |
                  IntersectionMatrix::EXTER_EXTER_2D;
      }
      break;
    }

    if (t_collinear_end > t_cur + eps) {
      t_cur = t_collinear_end;
      continue;
    }

    // Stagnation guard: if t_next cannot make forward progress,
    // break to avoid infinite loops due to floating-point precision.
    if (t_next <= t_cur + eps) {
      break;
    }

    double t_mid = 0.5 * (t_cur + t_next);
    POINT_T mid(static_cast<scalar_t>(static_cast<double>(v1.x()) + t_mid * dx),
                static_cast<scalar_t>(static_cast<double>(v1.y()) + t_mid * dy));
    PointLocation loc = locate_point_fn(mid);
    if (loc == PointLocation::kInside) {
      seg_im |= IntersectionMatrix::BOUND_INTER_1D |
                IntersectionMatrix::INTER_INTER_2D | IntersectionMatrix::EXTER_INTER_2D;
    } else if (loc == PointLocation::kOutside) {
      seg_im |= IntersectionMatrix::BOUND_EXTER_1D |
                IntersectionMatrix::INTER_EXTER_2D | IntersectionMatrix::EXTER_EXTER_2D;
    }
    t_cur = t_next;
  }

  return seg_im;
}

/**
 * Distributes segments of a probe polygon or multipolygon across cooperative threads
 * using round-robin striding, evaluating each segment via eval_seg_fn.
 */
template <typename PARAMS_T, typename EVAL_SEG_FN>
DEV_INLINE void ProcessProbeSegments(const PARAMS_T& params, uint32_t probe_idx,
                                     uint32_t pair_idx, uint32_t thread_idx,
                                     uint32_t num_threads,
                                     EVAL_SEG_FN&& eval_seg_fn) {
  using point_t = ShaderPointType;
  uint32_t seg_counter = 0;
  bool early_exited = false;

  auto process_poly = [&](const Polygon<point_t, uint32_t>& poly) {
    if (early_exited) return;
    for (uint32_t ring_idx = 0; ring_idx < poly.num_rings(); ++ring_idx) {
      if (early_exited) return;
      auto ring = poly.get_ring(ring_idx);
      if (ring.empty()) continue;
      uint32_t ring_v_begin = poly.get_prefix_sum_rings()[ring_idx];
      uint32_t n_segs = ring.num_segments();
      for (uint32_t s = 0; s < n_segs; ++s, ++seg_counter) {
        if (seg_counter % num_threads != thread_idx) continue;
        if (ShouldEarlyExitRelate(params.IMs[pair_idx], params.transpose)) {
          early_exited = true;
          return;
        }
        eval_seg_fn(ring.get_point(s), ring.get_point(s + 1), ring_v_begin + s);
      }
    }
  };

  if (!params.probe_polygons.empty()) {
    process_poly(params.probe_polygons[probe_idx]);
  } else {
    const auto& mp = params.probe_multi_polygons[probe_idx];
    for (uint32_t part = 0; part < mp.num_polygons(); ++part) {
      if (early_exited) break;
      process_poly(mp.get_polygon(part));
    }
  }
}

/**
 * Final thread reduction and interior point resolution.
 * If all boundary segments were collinearly covered (neither boundary-in-interior nor
 * boundary-in-exterior was recorded), finds a guaranteed interior point in the probe
 * and locates it within the BVH geometry to distinguish equality/containment from disjointness.
 */
template <typename PARAMS_T, typename LOCATE_PT_FN>
DEV_INLINE void FinalizePolygonRelate(const PARAMS_T& params, uint32_t pair_idx,
                                      uint32_t probe_idx, uint32_t num_threads,
                                      LOCATE_PT_FN&& locate_point_fn) {
  using point_t = ShaderPointType;
  uint32_t finished = atomicAdd(&params.finished_count[pair_idx], 1);
  if (finished == num_threads - 1) {
    int cur = params.IMs[pair_idx];
    bool has_b_in_i, has_b_in_e;
    if (params.transpose) {
      has_b_in_i = (cur & IntersectionMatrix::INTER_BOUND_2D) != 0;
      has_b_in_e = (cur & IntersectionMatrix::EXTER_BOUND_2D) != 0;
    } else {
      has_b_in_i = (cur & IntersectionMatrix::BOUND_INTER_2D) != 0;
      has_b_in_e = (cur & IntersectionMatrix::BOUND_EXTER_2D) != 0;
    }

    if (!has_b_in_i && !has_b_in_e) {
      point_t interior_pt;
      bool found = false;
      if (!params.probe_polygons.empty()) {
        found = detail::FindPolygonInteriorPoint(params.probe_polygons[probe_idx],
                                                 interior_pt);
      } else {
        const auto& mp = params.probe_multi_polygons[probe_idx];
        if (mp.num_polygons() > 0) {
          found = detail::FindPolygonInteriorPoint(mp.get_polygon(0), interior_pt);
        }
      }
      if (found) {
        PointLocation loc = locate_point_fn(interior_pt);
        int int_im = (loc == PointLocation::kInside) ? IntersectionMatrix::INTER_INTER_2D
                                                     : IntersectionMatrix::INTER_EXTER_2D;
        atomicOr(const_cast<int*>(&params.IMs[pair_idx]),
                 params.transpose ? IntersectionMatrix::Transpose(int_im) : int_im);
      }
    }
  }
}

}  // namespace detail
}  // namespace gpuspatial
