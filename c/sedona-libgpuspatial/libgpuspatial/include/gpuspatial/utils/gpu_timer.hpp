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

#include <cuda_runtime.h>
namespace gpuspatial {
// A simple utility class for timing CUDA kernels.
class GPUTimer {
 public:
  // Constructor creates the start and stop events.
  GPUTimer() {
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
  }

  // Destructor destroys the events.
  ~GPUTimer() {
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));
  }

  // Records the start event in the specified stream.
  void start(cudaStream_t stream = 0) {
    CUDA_CHECK(cudaEventRecord(start_event, stream));
  }

  // Records the stop event and returns the elapsed time in milliseconds.
  float stop(cudaStream_t stream = 0) {
    CUDA_CHECK(cudaEventRecord(stop_event, stream));
    float elapsed_time_ms = 0.0f;
    // The following call will block the CPU thread until the stop event has been
    // recorded.
    CUDA_CHECK(cudaEventSynchronize(stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time_ms, start_event, stop_event));
    return elapsed_time_ms;
  }

 private:
  cudaEvent_t start_event;
  cudaEvent_t stop_event;
};
}  // namespace gpuspatial
