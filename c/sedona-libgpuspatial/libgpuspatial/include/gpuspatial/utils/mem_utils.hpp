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
#include "gpuspatial/utils/exception.h"

#include "rmm/cuda_stream_view.hpp"

#include <cuda_runtime.h>
namespace gpuspatial {
namespace detail {
template <typename T>
void async_copy_h2d(const rmm::cuda_stream_view& stream, const T* src, T* dst,
                    size_t count) {
  if (count == 0) return;
  // Calculate the total size in bytes from the element count
  size_t size_in_bytes = count * sizeof(T);
  // Issue the asynchronous copy command to the specified stream
  CUDA_CHECK(cudaMemcpyAsync(dst, src, size_in_bytes, cudaMemcpyHostToDevice, stream));
}
template <typename T>
void async_copy_d2h(const rmm::cuda_stream_view& stream, const T* src, T* dst,
                    size_t count) {
  if (count == 0) return;
  // Calculate the total size in bytes from the element count
  size_t size_in_bytes = count * sizeof(T);

  // Issue the asynchronous copy command to the specified stream
  CUDA_CHECK(cudaMemcpyAsync(dst, src, size_in_bytes, cudaMemcpyDeviceToHost, stream));
}
}  // namespace detail
}  // namespace gpuspatial
