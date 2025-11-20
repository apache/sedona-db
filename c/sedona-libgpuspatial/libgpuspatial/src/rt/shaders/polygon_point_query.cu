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
#include "gpuspatial/geom/line_segment.cuh"
#include "gpuspatial/geom/ray_crossing_counter.cuh"
#include "gpuspatial/index/detail/launch_parameters.h"
#include "gpuspatial/relate/relate.cuh"
#include "shader_config.h"

#include <cuda_runtime.h>
#include <optix_device.h>
#include <cfloat>

enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };
// FLOAT_TYPE is defined by CMakeLists.txt
extern "C" __constant__ gpuspatial::detail::LaunchParamsPolygonPointQuery<
    gpuspatial::ShaderPointType, uint32_t>
    params;

extern "C" __global__ void __intersection__gpuspatial() {
  using namespace gpuspatial;
  auto aabb_id = optixGetPrimitiveIndex();
  auto query_idx = optixGetPayload_0();
  auto reordered_polygon_idx = optixGetPayload_1();
  uint32_t v_offset = optixGetPayload_2();
  auto ring_idx = optixGetPayload_3();
  auto crossing_count = optixGetPayload_4();
  auto point_on_seg = optixGetPayload_5();
  auto point_part_id = optixGetPayload_6();
  const auto& polygons = params.polygons;
  auto point_idx = params.ids[query_idx].first;
  auto polygon_idx = params.ids[query_idx].second;
  auto hit_polygon_idx = params.aabb_poly_ids[aabb_id];
  auto hit_ring_idx = params.aabb_ring_ids[aabb_id];
  // the seg being hit is not from the query polygon
  if (hit_polygon_idx != polygon_idx || hit_ring_idx != ring_idx) {
    return;
  }

  uint32_t local_v1_idx = aabb_id - params.seg_begins[reordered_polygon_idx];
  uint32_t global_v1_idx = v_offset + local_v1_idx;
  uint32_t global_v2_idx = global_v1_idx + 1;

  auto vertices = polygons.get_vertices();
  // segment being hit
  const auto& v1 = vertices[global_v1_idx];
  const auto& v2 = vertices[global_v2_idx];

  RayCrossingCounter locator(crossing_count, point_on_seg);
  if (!params.points.empty()) {
    const auto& p = params.points[point_idx];
    locator.countSegment(p, v1, v2);
  } else if (!params.multi_points.empty()) {
    const auto& p = params.multi_points[point_idx].get_point(point_part_id);
    locator.countSegment(p, v1, v2);
  }
  optixSetPayload_4(locator.get_crossing_count());
  optixSetPayload_5(locator.get_point_on_segment());
}

extern "C" __global__ void __raygen__gpuspatial() {
  using namespace gpuspatial;
  using point_t = gpuspatial::ShaderPointType;
  const auto& ids = params.ids;
  const auto& polygons = params.polygons;

  for (uint32_t i = optixGetLaunchIndex().x; i < ids.size();
       i += optixGetLaunchDimensions().x) {
    auto point_idx = ids[i].first;
    auto polygon_idx = ids[i].second;

    auto it = thrust::lower_bound(thrust::seq, params.polygon_ids.begin(),
                                  params.polygon_ids.end(), polygon_idx);
    assert(it != params.polygon_ids.end());
    uint32_t reordered_polygon_idx = thrust::distance(params.polygon_ids.begin(), it);
    assert(params.polygon_ids[reordered_polygon_idx] == polygon_idx);

    auto handle_point = [&](const point_t& p, uint32_t point_part_id, int& IM) {
      float3 origin;
      // each polygon takes a z-plane
      origin.x = p.x();
      origin.y = p.y();
      // cast ray toward positive x-axis
      float3 dir = {1, 0, 0};
      const auto& polygon = polygons[polygon_idx];
      const auto& mbr = polygon.get_mbr();
      auto width = mbr.get_max().x() - mbr.get_min().x();
      float tmin = 0;
      float tmax = width;

      // first polygon offset
      uint32_t ring_offset = polygons.get_prefix_sum_polygons()[polygon_idx];
      // first vertex offset of the ring
      uint32_t v_offset = polygons.get_prefix_sum_rings()[ring_offset];

      bool matched = false;

      if (polygon.empty()) {
        IM = IM__INTER_EXTER_0D | IM__EXTER_EXTER_2D;
      } else {
        IM = IM__EXTER_EXTER_2D;
      }
      RayCrossingCounter locator;

      if (polygon.empty()) return matched;
      IM |= IM__EXTER_INTER_2D | IM__EXTER_BOUND_1D;
      uint32_t ring = 0;
      locator.Init();
      origin.z = reordered_polygon_idx;
      // test exterior
      optixTrace(params.handle, origin, dir, tmin, tmax, 0, OptixVisibilityMask(255),
                 OPTIX_RAY_FLAG_NONE,             // OPTIX_RAY_FLAG_NONE,
                 SURFACE_RAY_TYPE,                // SBT offset
                 RAY_TYPE_COUNT,                  // SBT stride
                 SURFACE_RAY_TYPE,                // missSBTIndex
                 i,                               // 0
                 reordered_polygon_idx,           // 1
                 v_offset,                        // 2
                 ring,                            // 3
                 locator.get_crossing_count(),    // 4
                 locator.get_point_on_segment(),  // 5
                 point_part_id                    // 6
      );

      auto location = locator.location();
      PointLocation final_location = PointLocation::kError;
      if (location == PointLocation::kInside) {
        final_location = location;
        // test interior
        for (ring = 1; ring < polygon.num_rings(); ring++) {
          locator.Init();
          optixTrace(params.handle, origin, dir, tmin, tmax, 0, OptixVisibilityMask(255),
                     OPTIX_RAY_FLAG_NONE,             // OPTIX_RAY_FLAG_NONE,
                     SURFACE_RAY_TYPE,                // SBT offset
                     RAY_TYPE_COUNT,                  // SBT stride
                     SURFACE_RAY_TYPE,                // missSBTIndex
                     i,                               // 0
                     reordered_polygon_idx,           // 1
                     v_offset,                        // 2
                     ring,                            // 3
                     locator.get_crossing_count(),    // 4
                     locator.get_point_on_segment(),  // 5
                     point_part_id                    // 6
          );
          location = locator.location();
          if (location == PointLocation::kBoundary) {
            final_location = PointLocation::kBoundary;
            break;
          } else if (location == PointLocation::kInside) {
            final_location = PointLocation::kOutside;
            break;
          }
        }
      } else {
        // outside or boundary
        final_location = location;
      }
      assert(final_location != PointLocation::kError);

      switch (final_location) {
        case PointLocation::kInside: {
          matched = true;
          IM |= IM__INTER_INTER_0D;
          break;
        }
        case PointLocation::kBoundary: {
          matched = true;
          IM |= IM__INTER_BOUND_0D;
          break;
        }
        case PointLocation::kOutside: {
          break;
        }
        default:
          assert(false);
      }
#ifndef NDEBUG
      auto ref_loc = polygon.locate_point(params.points[point_idx]);
      if (ref_loc != final_location) {
        printf(
            "reorder %u, poly %u, point %u (%lf, %lf), num rings %u, point %u, loc %d, ref loc %d\n",
            reordered_polygon_idx, polygon_idx, point_idx, p.x(), p.y(),
            polygon.num_rings(), point_idx, (int)final_location, (int)ref_loc);
        assert(false);
      }
#endif
      if (!matched) IM |= IM__INTER_EXTER_0D;
      return matched;
    };

    int IM = IM__EXTER_EXTER_2D;

    if (!params.points.empty()) {
      handle_point(params.points[point_idx], 0 /*unused*/, IM);
    } else if (!params.multi_points.empty()) {
      auto mp = params.multi_points[point_idx];
      for (uint32_t j = 0; j < mp.num_points(); j++) {
        const auto& p = mp.get_point(j);
        if (handle_point(p, j, IM)) {
          // IM will not be changed anymore
          break;
        }
      }
    } else {
      assert(false);
    }

    params.IMs[i] = IM;
  }
}
