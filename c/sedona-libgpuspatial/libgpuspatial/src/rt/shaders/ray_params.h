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

#include <optix.h>
#include <thrust/swap.h>

#include <cfloat>

#define FLT_GAMMA(N) (((N) * FLT_EPSILON) / (1 - (N) * FLT_EPSILON))
#define DBL_GAMMA(N) (((N) * DBL_EPSILON) / (1 - (N) * DBL_EPSILON))
namespace gpuspatial {
namespace detail {

template <int N_DIMS>
struct RayParams {};

template <>
struct RayParams<2> {
  float2 o;  // ray origin
  float2 d;  // ray direction

  DEV_HOST RayParams(const OptixAabb& aabb, bool diagonal) {
    float2 p1{aabb.minX, aabb.minY};
    float2 p2{aabb.maxX, aabb.maxY};

    if (diagonal) {
      p1.x = aabb.maxX;
      p1.y = aabb.minY;
      p2.x = aabb.minX;
      p2.y = aabb.maxY;
    }

    o = p1;
    d = {p2.x - p1.x, p2.y - p1.y};
  }

  DEV_HOST_INLINE void PrintParams(const char* prefix) const {
    printf("%s, o: (%.6f, %.6f), d: (%.6f, %.6f)\n", prefix, o.x, o.y, d.x, d.y);
  }

  DEV_HOST_INLINE bool IsHit(const OptixAabb& aabb) const {
    float t0 = 0, t1 = 1.0;
    const auto* p_min = reinterpret_cast<const float*>(&aabb.minX);
    const auto* p_max = reinterpret_cast<const float*>(&aabb.maxX);
    // This is call slab-method, from https://github.com/mmp/pbrt-v4
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      float inv_ray_dir = 1 / reinterpret_cast<const float*>(&d)[i];
      float t_near = (p_min[i] - reinterpret_cast<const float*>(&o)[i]) * inv_ray_dir;
      float t_far = (p_max[i] - reinterpret_cast<const float*>(&o)[i]) * inv_ray_dir;

      if (t_near > t_far) {
        thrust::swap(t_near, t_far);
      }

      t_far *= 1 + 2 * FLT_GAMMA(3);
      t0 = t_near > t0 ? t_near : t0;
      t1 = t_far < t1 ? t_far : t1;

      if (t0 > t1) return false;
    }
    return true;
  }
};

}  // namespace detail
}  // namespace gpuspatial
