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
#include "gpuspatial/rt/launch_parameters.cuh"
#include "gpuspatial/utils/helpers.cuh"
#include "polygon_relate_common.cuh"
#include "shader_config.hpp"

#include <cuda_runtime.h>
#include <optix_device.h>
#include <cfloat>

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

    uint32_t new_p3 = detail::PackPIPCrossingPayload(locator.get_crossing_count(),
                                                     locator.get_point_on_segment());
    optixSetPayload_3(new_p3);
  } else {
    uint32_t abs_v_idx = p0 & ~detail::RAY_MODE_SEG;
    uint32_t target_poly_idx = p1 & detail::BVH_GEOM_ID_MASK;
    if (hit_poly_idx != target_poly_idx) {
      return;
    }

    const auto& probe_verts = !params.probe_polygons.empty()
                                  ? params.probe_polygons.get_vertices()
                                  : params.probe_multi_polygons.get_vertices();
    auto ring = params.bvh_polygons[target_poly_idx].get_ring(hit_ring_idx);
    detail::ProcessSegmentIntersectionLeaf(probe_verts[abs_v_idx],
                                           probe_verts[abs_v_idx + 1], ring,
                                           vertex_offsets, target_poly_idx);
  }
}

extern "C" __global__ void __raygen__gpuspatial() {
  using namespace gpuspatial;
  using point_t = gpuspatial::ShaderPointType;

  uint32_t pair_idx = optixGetLaunchIndex().x;
  uint32_t thread_idx = optixGetLaunchIndex().y;
  uint32_t num_threads = optixGetLaunchDimensions().y;

  if (pair_idx >= params.query_size) {
    return;
  }

  auto probe_idx = params.query_probe_ids[pair_idx];
  auto bvh_poly_idx = params.query_bvh_ids[pair_idx];

  const auto& bvh_poly = params.bvh_polygons[bvh_poly_idx];
  bool probe_empty = !params.probe_polygons.empty()
                         ? params.probe_polygons[probe_idx].empty()
                         : params.probe_multi_polygons[probe_idx].empty();
  bool bvh_empty = bvh_poly.empty();

  if (detail::HandleEmptyGeometries(params, pair_idx, thread_idx, probe_empty,
                                    bvh_empty)) {
    return;
  }

  const auto& probe_mbr = !params.probe_polygons.empty()
                              ? params.probe_polygons[probe_idx].get_mbr()
                              : params.probe_multi_polygons[probe_idx].get_mbr();
  const auto& bvh_mbr = bvh_poly.get_mbr();

  if (!probe_mbr.intersects(bvh_mbr)) {
    if (thread_idx == 0) {
      int im = IntersectionMatrix::INTER_EXTER_2D | IntersectionMatrix::BOUND_EXTER_1D |
               IntersectionMatrix::EXTER_INTER_2D | IntersectionMatrix::EXTER_BOUND_1D |
               IntersectionMatrix::EXTER_EXTER_2D;
      atomicOr(&params.IMs[pair_idx],
               params.transpose ? IntersectionMatrix::Transpose(im) : im);
    }
    return;
  }

  if (thread_idx == 0) {
    int base_im = IntersectionMatrix::EXTER_EXTER_2D;
    atomicOr(&params.IMs[pair_idx],
             params.transpose ? IntersectionMatrix::Transpose(base_im) : base_im);
  }

  float bvh_min_x = next_float_from_double(bvh_mbr.get_min().x(), -1, 2);
  float bvh_max_x = next_float_from_double(bvh_mbr.get_max().x(), 1, 2);
  float bvh_min_y = next_float_from_double(bvh_mbr.get_min().y(), -1, 2);
  float bvh_max_y = next_float_from_double(bvh_mbr.get_max().y(), 1, 2);

  constexpr uint32_t kOuterShellRingIdx = 0u;
  constexpr OptixVisibilityMask kAllVisibleMask = OptixVisibilityMask(0xFF);

  auto locate_point_rt = [&](const point_t& p) -> PointLocation {
    float px_f = static_cast<float>(p.x());
    float py_f = static_cast<float>(p.y());
    if (px_f < bvh_min_x || px_f > bvh_max_x || py_f < bvh_min_y || py_f > bvh_max_y) {
      return PointLocation::kOutside;
    }

    float3 origin = {px_f, py_f, static_cast<float>(bvh_poly_idx)};
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
      optixTrace(params.handle, origin, dir, tmin, tmax, 0.0f, kAllVisibleMask,
                 OPTIX_RAY_FLAG_NONE, SURFACE_RAY_TYPE, RAY_TYPE_COUNT, SURFACE_RAY_TYPE,
                 p0, p1, p2, p3, px_hi, px_lo, py_hi, py_lo);
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
    uint32_t p1 =
        (bb_flags & detail::SEG_FLAGS_MASK) | (bvh_poly_idx & detail::BVH_GEOM_ID_MASK);
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

  int local_im = 0;

  auto evaluate_segment = [&](const point_t& v1, const point_t& v2, uint32_t abs_v_idx) {
    if (v1 == v2) return;

    float seg_min_x = fminf(static_cast<float>(v1.x()), static_cast<float>(v2.x()));
    float seg_max_x = fmaxf(static_cast<float>(v1.x()), static_cast<float>(v2.x()));
    float seg_min_y = fminf(static_cast<float>(v1.y()), static_cast<float>(v2.y()));
    float seg_max_y = fmaxf(static_cast<float>(v1.y()), static_cast<float>(v2.y()));
    if (seg_max_x < bvh_min_x || seg_min_x > bvh_max_x || seg_max_y < bvh_min_y ||
        seg_min_y > bvh_max_y) {
      local_im |= IntersectionMatrix::BOUND_EXTER_1D |
                  IntersectionMatrix::INTER_EXTER_2D | IntersectionMatrix::EXTER_EXTER_2D;
      return;
    }

    int seg_im = detail::SweepSegment(
        v1, v2, abs_v_idx,
        [&](const float3& dir, float tmin, float tmax, uint32_t v_idx, double t_cur,
            uint32_t& bb_flags, double& t_next, double& t_collinear_end) {
          float3 origin = {static_cast<float>(v1.x()), static_cast<float>(v1.y()),
                           static_cast<float>(bvh_poly_idx)};
          trace_segment_rt(origin, dir, tmin, tmax, v_idx, t_cur, bb_flags, t_next,
                           t_collinear_end);
        },
        locate_point_rt);

    if (seg_im != 0) {
      local_im |= seg_im;
      if (seg_im &
          (IntersectionMatrix::BOUND_INTER_1D | IntersectionMatrix::BOUND_BOUND_1D)) {
        atomicOr(&params.IMs[pair_idx],
                 params.transpose ? IntersectionMatrix::Transpose(seg_im) : seg_im);
      }
    }
  };

  detail::ProcessProbeSegments(params, probe_idx, pair_idx, thread_idx, num_threads,
                               evaluate_segment);

  if (local_im != 0) {
    atomicOr(&params.IMs[pair_idx],
             params.transpose ? IntersectionMatrix::Transpose(local_im) : local_im);
  }

  detail::FinalizePolygonRelate(params, pair_idx, probe_idx, num_threads, locate_point_rt);
}
