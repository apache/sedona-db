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
#include "gpuspatial/rt/launch_parameters.h"
#include "shader_config.h"

#include <cuda_runtime.h>
#include <optix_device.h>
#include <cfloat>

enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };
// FLOAT_TYPE is defined by CMakeLists.txt
extern "C" __constant__
    gpuspatial::detail::LaunchParamsPointQuery<gpuspatial::ShaderPointType>
        params;

extern "C" __global__ void __intersection__gpuspatial() {
  auto aabb_id = optixGetPrimitiveIndex();
  auto point_id = optixGetPayload_0();
  const auto& point = params.points[point_id];
  const auto& rect = params.rects[aabb_id];

  if (rect.covers(point)) {
    if (params.count == nullptr) {
      auto tail = params.rect_ids.Append(aabb_id);
      params.point_ids[tail] = point_id;
    } else {
      atomicAdd(params.count, 1);
    }
  }
}

extern "C" __global__ void __raygen__gpuspatial() {
  using point_t = gpuspatial::ShaderPointType;
  constexpr int n_dim = point_t::n_dim;
  float tmin = 0;
  float tmax = FLT_MIN;

  for (uint32_t i = optixGetLaunchIndex().x; i < params.points.size();
       i += optixGetLaunchDimensions().x) {
    const auto& p = params.points[i];

    float3 origin{0, 0, 0};

    for (int dim = 0; dim < n_dim; dim++) {
      (&origin.x)[dim] = p.get_coordinate(dim);
    }
    float3 dir = {0, 0, 1};

    optixTrace(params.handle, origin, dir, tmin, tmax, 0, OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_NONE,  // OPTIX_RAY_FLAG_NONE,
               SURFACE_RAY_TYPE,     // SBT offset
               RAY_TYPE_COUNT,       // SBT stride
               SURFACE_RAY_TYPE,     // missSBTIndex
               i);
  }
}
