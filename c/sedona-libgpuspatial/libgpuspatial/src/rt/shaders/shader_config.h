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

namespace gpuspatial {

#define SHADER_FUNCTION_SUFFIX "gpuspatial"
// TODO: Set separated parameters
#define SHADER_NUM_PAYLOADS (8)

#if defined(SHADER_POINT_FLOAT_2D)
using ShaderPointType = Point<float, 2>;
#define SHADER_POINT_TYPE_ID "SHADER_POINT_FLOAT_2D"
#elif defined(SHADER_POINT_DOUBLE_2D)
using ShaderPointType = Point<double, 2>;
#define SHADER_POINT_TYPE_ID "SHADER_POINT_DOUBLE_2D"
#endif

#if defined(SHADER_INDEX_UINT32)
using ShaderIndexType = uint32_t;
#define SHADER_INDEX_TYPE_ID "SHADER_INDEX_UINT32"
#elif defined(SHADER_INDEX_UINT64)
using ShaderIndexType = uint64_t;
#define SHADER_INDEX_TYPE_ID "SHADER_INDEX_UINT64"
#endif

}  // namespace gpuspatial
