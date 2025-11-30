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
#include "gpuspatial/index/detail/launch_parameters.h"
#include "gpuspatial/relate/relate.cuh"
#include "ray_params.h"
#include "shader_config.h"

#include <cuda_runtime.h>
#include <optix_device.h>
#include <cfloat>
enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };
// FLOAT_TYPE is defined by CMakeLists.txt
extern "C" __constant__
    gpuspatial::detail::LaunchParamsBoxQuery<gpuspatial::ShaderPointType>
        params;

extern "C" __global__ void __intersection__gpuspatial() {
  using point_t = gpuspatial::ShaderPointType;
  constexpr int n_dim = point_t::n_dim;
  using ray_params_t = gpuspatial::detail::RayParams<n_dim>;
  auto geom1_id = optixGetPayload_0();
  auto geom2_id = optixGetPrimitiveIndex();
  const auto& mbr1 = params.mbrs1[geom1_id];
  const auto& mbr2 = params.mbrs2[geom2_id];
  const auto& aabb1 = mbr1.ToOptixAabb();
  const auto aabb2 = mbr2.ToOptixAabb();
  ray_params_t ray_params(aabb1, false);

  if (ray_params.IsHit(aabb2)) {
    if (mbr1.intersects(mbr2)) {
      params.ids.Append(thrust::make_pair(geom1_id, geom2_id));
    }
  }
}

// this is called backward pass in the LibRTS paper
// BVH is built over boxes2
extern "C" __global__ void __raygen__gpuspatial() {
  using point_t = gpuspatial::ShaderPointType;
  constexpr int n_dim = point_t::n_dim;

  for (uint32_t i = optixGetLaunchIndex().x; i < params.mbrs1.size();
       i += optixGetLaunchDimensions().x) {
    const auto& mbr1 = params.mbrs1[i];
    auto aabb1 = mbr1.ToOptixAabb();
    gpuspatial::detail::RayParams<n_dim> ray_params(aabb1, false);
    float3 origin, dir;

    origin.x = ray_params.o.x;
    origin.y = ray_params.o.y;
    origin.z = 0;

    dir.x = ray_params.d.x;
    dir.y = ray_params.d.y;
    dir.z = 0;

    float tmin = 0;
    float tmax = 1;

    optixTrace(params.handle, origin, dir, tmin, tmax, 0, OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_NONE,  // OPTIX_RAY_FLAG_NONE,
               SURFACE_RAY_TYPE,     // SBT offset
               RAY_TYPE_COUNT,       // SBT stride
               SURFACE_RAY_TYPE,     // missSBTIndex
               i);
  }
}
