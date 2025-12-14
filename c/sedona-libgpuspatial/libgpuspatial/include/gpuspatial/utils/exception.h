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
#include "gpuspatial/utils/logger.hpp"

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <optix.h>
#include <optix_stubs.h>

#include <sstream>
#include <stdexcept>
#include <string>

#define OPTIX_CHECK(call) ::gpuspatial::optixCheck(call, #call, __FILE__, __LINE__)

#define CUDA_CHECK(call) ::gpuspatial::cudaCheck(call, #call, __FILE__, __LINE__)

namespace gpuspatial {

class GPUException : public std::runtime_error {
 public:
  GPUException(const char* msg) : std::runtime_error(msg) {}

  GPUException(OptixResult res, const char* msg)
      : std::runtime_error(createMessage(res, msg).c_str()) {}

 private:
  std::string createMessage(OptixResult res, const char* msg) {
    std::ostringstream out;
    out << optixGetErrorName(res) << ": " << msg;
    return out.str();
  }
};

inline void optixCheck(OptixResult res, const char* call, const char* file,
                       unsigned int line) {
  if (res != OPTIX_SUCCESS) {
    std::stringstream ss;
    ss << "OptiX API call (" << call << ") failed with error " << optixGetErrorName(res)
       << " (" << file << ":" << line << ")";
    GPUSPATIAL_LOG_ERROR("Optix API error: {}", ss.str());
    throw GPUException(res, ss.str().c_str());
  }
}

inline void cudaCheck(cudaError_t error, const char* call, const char* file,
                      unsigned int line) {
  if (error != cudaSuccess) {
    std::stringstream ss;
    ss << "CUDA API call (" << call << ") failed with error " << cudaGetErrorString(error)
       << " (" << file << ":" << line << ")";
    GPUSPATIAL_LOG_ERROR("CUDA API error: {}", ss.str());
    throw GPUException(ss.str().c_str());
  }
}

}  // namespace gpuspatial
