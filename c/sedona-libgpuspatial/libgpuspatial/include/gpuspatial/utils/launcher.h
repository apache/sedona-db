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
#include "gpuspatial/utils/cuda_utils.h"
#include "gpuspatial/utils/exception.h"

#include "rmm/cuda_stream_view.hpp"

namespace gpuspatial {
template <typename F, typename... Args>
__global__ void KernelWrapper(F f, Args... args) {
  f(args...);
}

template <typename F, typename... Args>
void LaunchKernel(const rmm::cuda_stream_view& stream, F f, Args&&... args) {
  int grid_size, block_size;

  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                                KernelWrapper<F, Args...>, 0,
                                                reinterpret_cast<int>(MAX_BLOCK_SIZE)));

  KernelWrapper<<<grid_size, block_size, 0, stream>>>(f, std::forward<Args>(args)...);
}

}  // namespace gpuspatial
