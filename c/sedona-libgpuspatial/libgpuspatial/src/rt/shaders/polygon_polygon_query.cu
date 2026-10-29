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
#include "gpuspatial/geom/ray_crossing_counter.hpp"
#include "gpuspatial/relate/intersection_matrix.hpp"
#include "gpuspatial/relate/segment_relate_ops.cuh"
#include "gpuspatial/rt/launch_parameters.cuh"
#include "gpuspatial/utils/helpers.cuh"
#include "shader_config.hpp"

#include <cuda_runtime.h>
#include <optix_device.h>
#include <cfloat>

//==============================================================================
// Polygon-Polygon Relate Ray-Tracing Shaders
//
// Basic Idea:
// Evaluates 2D polygon-polygon relate and DE-9IM matrices via OptiX:
// 1. Ray Mode SEG (Segment sweep):
//    Casts a short ray along a probe polygon edge to find the nearest
//    boundary intersection or collinear overlap with BVH polygon rings.
// 2. Ray Mode PIP (Point-In-Polygon):
//    Casts horizontal +x rays to count edge crossings (Jordan Curve Theorem),
//    determining whether test points (e.g. segment midpoints or interior points)
//    lie inside, on the boundary, or outside the BVH polygon.
//==============================================================================

enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };

extern "C" __constant__ gpuspatial::detail::LaunchParamsPolygonPolygonQuery<
    gpuspatial::ShaderPointType, uint32_t>
    params;

extern "C" __global__ void __intersection__gpuspatial() {
  using namespace gpuspatial;
  using point_t = gpuspatial::ShaderPointType;
  using scalar_t = typename point_t::scalar_t;

  auto aabb_id = optixGetPrimitiveIndex();
  uint32_t p0 = optixGetPayload_0();
  uint32_t p1 = optixGetPayload_1();

  bool is_seg_mode = (p0 & detail::RAY_MODE_SEG) != 0;
  auto hit_poly_idx = params.aabb_poly_ids[aabb_id];
  auto hit_ring_idx = params.aabb_ring_ids[aabb_id];
  const auto& vertex_offsets = params.aabb_vertex_offsets[aabb_id];

  if (!is_seg_mode) {
    uint32_t target_ring_idx = p0 & ~detail::RAY_MODE_SEG;
    uint32_t target_poly_idx = p1;
    if (hit_poly_idx != target_poly_idx || hit_ring_idx != target_ring_idx) {
      return;
    }

    uint32_t p3 = optixGetPayload_3();
    uint32_t crossing_count = detail::UnpackPIPCrossingCount(p3);
    uint32_t point_on_seg = detail::UnpackPIPPointOnSegment(p3);
    double px = detail::UnpackDouble(optixGetPayload_4(), optixGetPayload_5());
    double py = detail::UnpackDouble(optixGetPayload_6(), optixGetPayload_7());
    point_t q(static_cast<scalar_t>(px), static_cast<scalar_t>(py));

    auto ring = params.bvh_polygons[target_poly_idx].get_ring(target_ring_idx);
    RayCrossingCounter locator(crossing_count, point_on_seg);

    for (auto v_off = vertex_offsets.first; v_off < vertex_offsets.second; ++v_off) {
      const auto& v1 = ring.get_point(v_off);
      const auto& v2 = ring.get_point(v_off + 1);
      locator.countSegment(q, v1, v2);
    }

    uint32_t new_p3 = detail::PackPIPCrossingPayload(
        locator.get_crossing_count(), locator.get_point_on_segment());
    optixSetPayload_3(new_p3);
  } else {
    uint32_t abs_v_idx = p0 & ~detail::RAY_MODE_SEG;
    uint32_t target_poly_idx = p1 & detail::BVH_GEOM_ID_MASK;
    if (hit_poly_idx != target_poly_idx) {
      return;
    }

    uint32_t bb_flags = p1 & detail::SEG_FLAGS_MASK;
    double t_cur = detail::UnpackDouble(optixGetPayload_2(), optixGetPayload_3());
    double t_next = detail::UnpackDouble(optixGetPayload_4(), optixGetPayload_5());
    double t_collinear_end =
        detail::UnpackDouble(optixGetPayload_6(), optixGetPayload_7());

    const auto& probe_verts = !params.probe_polygons.empty()
                                  ? params.probe_polygons.get_vertices()
                                  : params.probe_multi_polygons.get_vertices();
    const auto& v1 = probe_verts[abs_v_idx];
    const auto& v2 = probe_verts[abs_v_idx + 1];

    auto ring = params.bvh_polygons[target_poly_idx].get_ring(hit_ring_idx);
    for (auto v_off = vertex_offsets.first; v_off < vertex_offsets.second; ++v_off) {
      const auto& q1 = ring.get_point(v_off);
      const auto& q2 = ring.get_point(v_off + 1);
      detail::AnalyzeSegmentIntersection(v1, v2, q1, q2, t_cur, bb_flags, t_next,
                                         t_collinear_end);
    }

    optixSetPayload_1((bb_flags & detail::SEG_FLAGS_MASK) |
                      (target_poly_idx & detail::BVH_GEOM_ID_MASK));
    uint32_t next_hi, next_lo, col_hi, col_lo;
    detail::PackDouble(t_next, next_hi, next_lo);
    detail::PackDouble(t_collinear_end, col_hi, col_lo);
    optixSetPayload_4(next_hi);
    optixSetPayload_5(next_lo);
    optixSetPayload_6(col_hi);
    optixSetPayload_7(col_lo);
  }
}

extern "C" __global__ void __raygen__gpuspatial() {
  using namespace gpuspatial;
  using point_t = gpuspatial::ShaderPointType;
  using scalar_t = typename point_t::scalar_t;

  for (uint32_t i = optixGetLaunchIndex().x; i < params.query_size;
       i += optixGetLaunchDimensions().x) {
    auto probe_idx = params.query_probe_ids[i];
    auto bvh_poly_idx = params.query_bvh_ids[i];

    const auto& bvh_poly = params.bvh_polygons[bvh_poly_idx];
    bool probe_empty = !params.probe_polygons.empty()
                           ? params.probe_polygons[probe_idx].empty()
                           : params.probe_multi_polygons[probe_idx].empty();
    bool bvh_empty = bvh_poly.empty();

    if (probe_empty || bvh_empty) {
      int im = IntersectionMatrix::EXTER_EXTER_2D;
      if (!probe_empty) {
        im |= IntersectionMatrix::INTER_EXTER_2D | IntersectionMatrix::BOUND_EXTER_1D;
      }
      if (!bvh_empty) {
        im |= IntersectionMatrix::EXTER_INTER_2D | IntersectionMatrix::EXTER_BOUND_1D;
      }
      if (params.transpose) {
        params.IMs[i] |= IntersectionMatrix::Transpose(im);
      } else {
        params.IMs[i] = im;
      }
      continue;
    }

    const auto& probe_mbr = !params.probe_polygons.empty()
                                ? params.probe_polygons[probe_idx].get_mbr()
                                : params.probe_multi_polygons[probe_idx].get_mbr();
    const auto& bvh_mbr = bvh_poly.get_mbr();

    if (!probe_mbr.intersects(bvh_mbr)) {
      int im = IntersectionMatrix::INTER_EXTER_2D | IntersectionMatrix::BOUND_EXTER_1D |
               IntersectionMatrix::EXTER_INTER_2D | IntersectionMatrix::EXTER_BOUND_1D |
               IntersectionMatrix::EXTER_EXTER_2D;
      if (params.transpose) {
        params.IMs[i] |= IntersectionMatrix::Transpose(im);
      } else {
        params.IMs[i] = im;
      }
      continue;
    }

    float bvh_min_x = next_float_from_double(bvh_mbr.get_min().x(), -1, 2);
    float bvh_max_x = next_float_from_double(bvh_mbr.get_max().x(), 1, 2);
    float bvh_min_y = next_float_from_double(bvh_mbr.get_min().y(), -1, 2);
    float bvh_max_y = next_float_from_double(bvh_mbr.get_max().y(), 1, 2);

    // Outer shell ring index is always 0 in OGC Simple Feature polygons; holes start at index 1.
    constexpr uint32_t kOuterShellRingIdx = 0u;
    // OptiX visibility mask: 0xFF makes all BVH geometry instances visible to the ray.
    constexpr OptixVisibilityMask kAllVisibleMask = OptixVisibilityMask(0xFF);

    auto locate_point_rt = [&](const point_t& p) -> PointLocation {
      float px_f = static_cast<float>(p.x());
      float py_f = static_cast<float>(p.y());
      if (px_f < bvh_min_x || px_f > bvh_max_x || py_f < bvh_min_y || py_f > bvh_max_y) {
        return PointLocation::kOutside;
      }

      float3 origin = {px_f, py_f, static_cast<float>(bvh_poly_idx)};
      // Cast a horizontal ray in the +x direction to count boundary edge crossings (Jordan Curve Theorem).
      float3 dir = {1.0f, 0.0f, 0.0f};
      float tmin = 0.0f;
      float tmax = next_float_from_double(
          static_cast<double>(bvh_max_x) - static_cast<double>(px_f), 1, 4);

      uint32_t px_hi, px_lo, py_hi, py_lo;
      detail::PackDouble(static_cast<double>(p.x()), px_hi, px_lo);
      detail::PackDouble(static_cast<double>(p.y()), py_hi, py_lo);

      uint32_t p0 = detail::RAY_MODE_PIP | kOuterShellRingIdx;
      uint32_t p1 = bvh_poly_idx;
      uint32_t p2 = 0u;
      uint32_t p3 = 0u;

      optixTrace(params.handle, origin, dir, tmin, tmax, 0.0f, kAllVisibleMask,
                 OPTIX_RAY_FLAG_NONE, SURFACE_RAY_TYPE, RAY_TYPE_COUNT, SURFACE_RAY_TYPE,
                 p0, p1, p2, p3, px_hi, px_lo, py_hi, py_lo);

      RayCrossingCounter locator(detail::UnpackPIPCrossingCount(p3),
                                 detail::UnpackPIPPointOnSegment(p3));
      auto loc = locator.location();
      if (loc != PointLocation::kInside) {
        return loc;
      }

      for (uint32_t ring = 1; ring < bvh_poly.num_rings(); ++ring) {
        p0 = detail::RAY_MODE_PIP | ring;
        p1 = bvh_poly_idx;
        p2 = 0u;
        p3 = 0u;
        optixTrace(params.handle, origin, dir, tmin, tmax, 0.0f,
                   kAllVisibleMask, OPTIX_RAY_FLAG_NONE, SURFACE_RAY_TYPE,
                   RAY_TYPE_COUNT, SURFACE_RAY_TYPE, p0, p1, p2, p3, px_hi, px_lo, py_hi,
                   py_lo);
        RayCrossingCounter hole_locator(detail::UnpackPIPCrossingCount(p3),
                                       detail::UnpackPIPPointOnSegment(p3));
        auto hole_loc = hole_locator.location();
        if (hole_loc == PointLocation::kBoundary) {
          return PointLocation::kBoundary;
        } else if (hole_loc == PointLocation::kInside) {
          return PointLocation::kOutside;
        }
      }
      return PointLocation::kInside;
    };

    auto trace_segment_rt = [&](const float3& origin, const float3& dir, float tmin,
                                float tmax, uint32_t abs_v_idx, double t_cur,
                                uint32_t& bb_flags, double& t_next,
                                double& t_collinear_end) {
      uint32_t p0 = detail::RAY_MODE_SEG | (abs_v_idx & ~detail::RAY_MODE_SEG);
      uint32_t p1 = (bb_flags & detail::SEG_FLAGS_MASK) |
                    (bvh_poly_idx & detail::BVH_GEOM_ID_MASK);
      uint32_t cur_hi, cur_lo, next_hi, next_lo, col_hi, col_lo;
      detail::PackDouble(t_cur, cur_hi, cur_lo);
      detail::PackDouble(t_next, next_hi, next_lo);
      detail::PackDouble(t_collinear_end, col_hi, col_lo);

      optixTrace(params.handle, origin, dir, tmin, tmax, 0.0f, kAllVisibleMask,
                 OPTIX_RAY_FLAG_NONE, SURFACE_RAY_TYPE, RAY_TYPE_COUNT, SURFACE_RAY_TYPE,
                 p0, p1, cur_hi, cur_lo, next_hi, next_lo, col_hi, col_lo);

      bb_flags = p1 & detail::SEG_FLAGS_MASK;
      t_next = detail::UnpackDouble(next_hi, next_lo);
      t_collinear_end = detail::UnpackDouble(col_hi, col_lo);
    };

    int IM = IntersectionMatrix::EXTER_EXTER_2D;
    int global_bb = 0;
    bool global_has_B_in_I = false;
    bool global_has_B_in_E = false;

    auto process_probe_polygon = [&](const Polygon<point_t, uint32_t>& probe_poly) {
      if (probe_poly.empty()) return;
      bool part_has_B_in_I = false;
      bool part_has_B_in_E = false;

      for (uint32_t ring_idx = 0; ring_idx < probe_poly.num_rings(); ++ring_idx) {
        auto ring = probe_poly.get_ring(ring_idx);
        if (ring.empty()) continue;
        uint32_t ring_v_begin = probe_poly.get_prefix_sum_rings()[ring_idx];
        uint32_t n_segs = ring.num_segments();
        PointLocation prev_v2_loc = PointLocation::kError;

        for (uint32_t seg_idx = 0; seg_idx < n_segs; ++seg_idx) {
          if (params.transpose && part_has_B_in_I && part_has_B_in_E) {
            break;
          }
          if (!params.transpose && part_has_B_in_I && part_has_B_in_E &&
              global_bb == IntersectionMatrix::BOUND_BOUND_1D) {
            break;
          }

          uint32_t abs_v_idx = ring_v_begin + seg_idx;
          const auto& v1 = ring.get_point(seg_idx);
          const auto& v2 = ring.get_point(seg_idx + 1);
          if (v1 == v2) continue;

          float seg_min_x = fminf(static_cast<float>(v1.x()), static_cast<float>(v2.x()));
          float seg_max_x = fmaxf(static_cast<float>(v1.x()), static_cast<float>(v2.x()));
          float seg_min_y = fminf(static_cast<float>(v1.y()), static_cast<float>(v2.y()));
          float seg_max_y = fmaxf(static_cast<float>(v1.y()), static_cast<float>(v2.y()));
          if (seg_max_x < bvh_min_x || seg_min_x > bvh_max_x || seg_max_y < bvh_min_y ||
              seg_min_y > bvh_max_y) {
            part_has_B_in_E = true;
            prev_v2_loc = PointLocation::kOutside;
            continue;
          }

          double dx = static_cast<double>(v2.x()) - static_cast<double>(v1.x());
          double dy = static_cast<double>(v2.y()) - static_cast<double>(v1.y());
          double L = fmax(fabs(dx), fabs(dy));
          float3 seg_origin = {static_cast<float>(v1.x()), static_cast<float>(v1.y()),
                               static_cast<float>(bvh_poly_idx)};
          float3 seg_dir = (L > 0.0) ? float3{static_cast<float>(dx / L),
                                              static_cast<float>(dy / L), 0.0f}
                                     : float3{1.0f, 0.0f, 0.0f};
          float seg_tmin = 0.0f;
          float seg_tmax = (L > 0.0) ? next_float_from_double(L, 1, 4) : 0.0f;

          if (part_has_B_in_I && part_has_B_in_E) {
            uint32_t bb_flags = 0;
            double t_next = 1.0;
            double t_collinear_end = 0.0;
            trace_segment_rt(seg_origin, seg_dir, seg_tmin, seg_tmax, abs_v_idx, 0.0,
                             bb_flags, t_next, t_collinear_end);
            if (bb_flags & detail::SEG_HIT_1D) {
              global_bb = IntersectionMatrix::BOUND_BOUND_1D;
            } else if ((bb_flags & detail::SEG_HIT_0D) && global_bb == 0) {
              global_bb = IntersectionMatrix::BOUND_BOUND_0D;
            }
            if (bb_flags & detail::SEG_PROPER_CROSS) {
              part_has_B_in_I = true;
              part_has_B_in_E = true;
            }
            continue;
          }

          double t_cur = 0.0;
          constexpr double eps = static_cast<double>(detail::ParamTolerance<scalar_t>());
          int iter = 0;
          // Step along probe segment v1 -> v2 from parameter t_cur to 1.0.
          // In each iteration, trace_segment_rt tests against BVH rings:
          // - Finds the nearest intersection node t_next along [t_cur, 1.0].
          // - Finds any collinear boundary overlap [t_cur, t_collinear_end].
          // - Updates boundary intersection flags (SEG_HIT_0D, SEG_HIT_1D, SEG_PROPER_CROSS).
          // Capped at MAX_SEGMENT_SWEEP_ITERATIONS (128) to guarantee termination even in degenerate cases.
          while (t_cur < 1.0 - eps && iter < detail::MAX_SEGMENT_SWEEP_ITERATIONS) {
            ++iter;
            uint32_t bb_flags = 0;
            double t_next = 1.0;
            double t_collinear_end = t_cur;
            trace_segment_rt(seg_origin, seg_dir, seg_tmin, seg_tmax, abs_v_idx, t_cur,
                             bb_flags, t_next, t_collinear_end);

            if (bb_flags & detail::SEG_HIT_1D) {
              global_bb = IntersectionMatrix::BOUND_BOUND_1D;
            } else if ((bb_flags & detail::SEG_HIT_0D) && global_bb == 0) {
              global_bb = IntersectionMatrix::BOUND_BOUND_0D;
            }
            if (bb_flags & detail::SEG_PROPER_CROSS) {
              part_has_B_in_I = true;
              part_has_B_in_E = true;
            }

            if ((bb_flags & detail::SEG_HIT_ANY) == 0 && t_cur == 0.0) {
              PointLocation loc = prev_v2_loc;
              if (loc != PointLocation::kInside && loc != PointLocation::kOutside) {
                point_t mid(static_cast<scalar_t>(0.5 * (v1.x() + v2.x())),
                            static_cast<scalar_t>(0.5 * (v1.y() + v2.y())));
                loc = locate_point_rt(mid);
              }
              if (loc == PointLocation::kInside) {
                part_has_B_in_I = true;
              } else if (loc == PointLocation::kOutside) {
                part_has_B_in_E = true;
              }
              prev_v2_loc = loc;
              break;
            }

            prev_v2_loc = PointLocation::kError;

            if (t_collinear_end > t_cur + eps) {
              t_cur = t_collinear_end;
              continue;
            }

            if (!part_has_B_in_I || !part_has_B_in_E) {
              double t_mid = 0.5 * (t_cur + t_next);
              point_t mid(
                  static_cast<scalar_t>(static_cast<double>(v1.x()) + t_mid * dx),
                  static_cast<scalar_t>(static_cast<double>(v1.y()) + t_mid * dy));
              PointLocation loc = locate_point_rt(mid);
              if (loc == PointLocation::kInside) {
                part_has_B_in_I = true;
              } else if (loc == PointLocation::kOutside) {
                part_has_B_in_E = true;
              }
            }
            t_cur = t_next;
          }
        }
      }

      if (part_has_B_in_I) {
        global_has_B_in_I = true;
        IM |= IntersectionMatrix::BOUND_INTER_1D | IntersectionMatrix::INTER_INTER_2D |
              IntersectionMatrix::EXTER_INTER_2D;
      }
      if (part_has_B_in_E) {
        global_has_B_in_E = true;
        IM |= IntersectionMatrix::BOUND_EXTER_1D | IntersectionMatrix::INTER_EXTER_2D |
              IntersectionMatrix::EXTER_EXTER_2D;
      }
      if (!part_has_B_in_I && !part_has_B_in_E) {
        // 100% of probe_poly's boundary is collinearly covered by bvh_poly's boundary
        point_t interior_pt;
        if (detail::FindPolygonInteriorPoint(probe_poly, interior_pt)) {
          PointLocation loc = locate_point_rt(interior_pt);
          if (loc == PointLocation::kInside) {
            IM |= IntersectionMatrix::INTER_INTER_2D;
          } else {
            IM |= IntersectionMatrix::INTER_EXTER_2D;
          }
        }
      }
    };

    if (!params.probe_polygons.empty()) {
      process_probe_polygon(params.probe_polygons[probe_idx]);
    } else {
      const auto& mp = params.probe_multi_polygons[probe_idx];
      for (uint32_t part = 0; part < mp.num_polygons(); ++part) {
        if (params.transpose && global_has_B_in_I && global_has_B_in_E) {
          break;
        }
        process_probe_polygon(mp.get_polygon(part));
      }
    }

    IM |= global_bb;
    if (params.transpose) {
      params.IMs[i] |= IntersectionMatrix::Transpose(IM);
    } else {
      params.IMs[i] = IM;
    }
  }
}
