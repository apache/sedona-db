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
#include <vector_types.h>

namespace gpuspatial {
template <typename SCALA_T, int N_DIM>
struct cuda_vec {};

template <>
struct cuda_vec<float, 2> {
  using type = float2;
};

template <>
struct cuda_vec<float, 3> {
  using type = float3;
};

template <>
struct cuda_vec<double, 2> {
  using type = double2;
};

template <>
struct cuda_vec<double, 3> {
  using type = double3;
};

template <typename CUDA_VEC_T>
struct cuda_vec_info {};

template <>
struct cuda_vec_info<float2> {
  using scalar_type = float;
  static constexpr int n_dim = 2;
};

template <>
struct cuda_vec_info<float3> {
  using scalar_type = float;
  static constexpr int n_dim = 3;
};

template <>
struct cuda_vec_info<double2> {
  using scalar_type = double;
  static constexpr int n_dim = 2;
};

template <>
struct cuda_vec_info<double3> {
  using scalar_type = double;
  static constexpr int n_dim = 3;
};
}  // namespace gpuspatial
