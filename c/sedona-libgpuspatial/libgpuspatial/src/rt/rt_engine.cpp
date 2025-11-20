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
#include "gpuspatial/index/detail/rt_engine.hpp"
#include "gpuspatial/utils/cuda_utils.h"
#include "gpuspatial/utils/exception.h"
#include "gpuspatial/utils/logger.hpp"

#include "rt/shaders/shader_config.h"

#include "rmm/device_scalar.hpp"

// this header provides OPTIX_FUNCTION_TABLE_SYMBOL
// Only included once in the compilation unit
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {
// OptiX log callback function
void context_log_cb(unsigned int level, const char* tag, const char* message, void*) {
  switch (level) {
    case 1:
      GPUSPATIAL_LOG_CRITICAL("OptiX [%s]: %s", tag, message);
      break;
    case 2:
      GPUSPATIAL_LOG_ERROR("OptiX [%s]: %s", tag, message);
      break;
    case 3:
      GPUSPATIAL_LOG_WARN("OptiX [%s]: %s", tag, message);
      break;
    case 4:
      GPUSPATIAL_LOG_INFO("OptiX [%s]: %s", tag, message);
      break;
  }
}
}  // namespace

namespace gpuspatial {
namespace details {

// --- RTConfig Method Definitions ---

void RTConfig::AddModule(const Module& mod) {
  if (access(mod.get_program_path().c_str(), R_OK) != 0) {
    GPUSPATIAL_LOG_CRITICAL("Cannot open %s", mod.get_program_path().c_str());
    throw std::runtime_error("Cannot open shader file " + mod.get_program_path());
  }
  modules[mod.get_id()] = mod;
}

// --- Free Function Definitions ---

RTConfig get_default_rt_config(const std::string& ptx_root) {
  RTConfig config;
  const std::filesystem::path folder_path{ptx_root};

  for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
    if (entry.is_regular_file() && entry.path().extension() == ".ptx") {
      auto shader_id = entry.path().filename().string();
      Module mod(shader_id);
      mod.set_program_path(entry.path().string());
      mod.set_function_suffix(SHADER_FUNCTION_SUFFIX);
      mod.set_n_payload(SHADER_NUM_PAYLOADS);
      mod.EnableIsIntersection();
      config.AddModule(mod);
    }
  }

#ifndef NDEBUG
  config.opt_level = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
  config.dbg_level = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
  config.opt_level = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
  config.dbg_level = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif

  return config;
}

// --- RTEngine Method Definitions ---

RTEngine::RTEngine() : initialized_(false) {}

RTEngine::~RTEngine() {
  if (initialized_) {
    releaseOptixResources();
  }
}

void RTEngine::Init(const RTConfig& config) {
  if (initialized_) {
    releaseOptixResources();
  }
  initOptix(config);
  createContext();
  createModule(config);
  createRaygenPrograms(config);
  createMissPrograms(config);
  createHitgroupPrograms(config);
  createPipeline(config);
  buildSBT(config);
  initialized_ = true;
}

OptixTraversableHandle RTEngine::BuildAccelCustom(cudaStream_t cuda_stream,
                                                  ArrayView<OptixAabb> aabbs,
                                                  rmm::device_buffer& out_buf,
                                                  bool prefer_fast_build,
                                                  bool compact) const {
  OptixTraversableHandle traversable;
  OptixBuildInput build_input = {};
  CUdeviceptr d_aabb = THRUST_TO_CUPTR(aabbs.data());
  uint32_t build_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
  uint32_t num_prims = aabbs.size();

  assert(reinterpret_cast<uint64_t>(aabbs.data()) % OPTIX_AABB_BUFFER_BYTE_ALIGNMENT ==
         0);

  build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
  build_input.customPrimitiveArray.aabbBuffers = &d_aabb;
  build_input.customPrimitiveArray.flags = build_input_flags;
  build_input.customPrimitiveArray.numSbtRecords = 1;
  build_input.customPrimitiveArray.numPrimitives = num_prims;
  build_input.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
  build_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
  build_input.customPrimitiveArray.primitiveIndexOffset = 0;

  OptixAccelBuildOptions accelOptions = {};

  if (prefer_fast_build) {
    accelOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
  } else {
    accelOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
  }
  if (compact) {
    accelOptions.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  }
  accelOptions.motionOptions.numKeys = 1;
  accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes blas_buffer_sizes;
  OPTIX_CHECK(optixAccelComputeMemoryUsage(optix_context_, &accelOptions, &build_input, 1,
                                           &blas_buffer_sizes));

  GPUSPATIAL_LOG_INFO(
      "ComputeBVHMemoryUsage, AABB count: %u, temp size: %zu MB, output size: %zu MB",
      num_prims, blas_buffer_sizes.tempSizeInBytes / 1024 / 1024,
      blas_buffer_sizes.outputSizeInBytes / 1024 / 1024);

  rmm::device_buffer temp_buf(blas_buffer_sizes.tempSizeInBytes, cuda_stream);
  out_buf.resize(blas_buffer_sizes.outputSizeInBytes, cuda_stream);

  if (compact) {
    rmm::device_scalar<uint64_t> compacted_size(cuda_stream);
    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = reinterpret_cast<CUdeviceptr>(compacted_size.data());

    OPTIX_CHECK(optixAccelBuild(
        optix_context_, cuda_stream, &accelOptions, &build_input, 1,
        reinterpret_cast<CUdeviceptr>(temp_buf.data()), blas_buffer_sizes.tempSizeInBytes,
        reinterpret_cast<CUdeviceptr>(out_buf.data()),
        blas_buffer_sizes.outputSizeInBytes, &traversable, &emitDesc, 1));

    auto size = compacted_size.value(cuda_stream);
    out_buf.resize(size, cuda_stream);
    OPTIX_CHECK(optixAccelCompact(optix_context_, cuda_stream, traversable,
                                  reinterpret_cast<CUdeviceptr>(out_buf.data()), size,
                                  &traversable));
  } else {
    OPTIX_CHECK(optixAccelBuild(
        optix_context_, cuda_stream, &accelOptions, &build_input, 1,
        reinterpret_cast<CUdeviceptr>(temp_buf.data()), blas_buffer_sizes.tempSizeInBytes,
        reinterpret_cast<CUdeviceptr>(out_buf.data()),
        blas_buffer_sizes.outputSizeInBytes, &traversable, nullptr, 0));
  }

  return traversable;
}

void RTEngine::Render(cudaStream_t cuda_stream, const std::string& id, dim3 dim,
                      const ArrayView<char>& params) const {
  OPTIX_CHECK(optixLaunch(resources_.at(id).pipeline, cuda_stream,
                          reinterpret_cast<CUdeviceptr>(params.data()), params.size(),
                          &resources_.at(id).sbt, dim.x, dim.y, dim.z));
}

OptixDeviceContext RTEngine::get_context() const { return optix_context_; }

size_t RTEngine::EstimateMemoryUsageForAABB(size_t num_aabbs, bool prefer_fast_build,
                                            bool compact) const {
  OptixBuildInput build_input = {};
  uint32_t build_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};

  build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
  build_input.customPrimitiveArray.aabbBuffers = nullptr;
  build_input.customPrimitiveArray.flags = build_input_flags;
  build_input.customPrimitiveArray.numSbtRecords = 1;
  build_input.customPrimitiveArray.numPrimitives = num_aabbs;
  build_input.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
  build_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
  build_input.customPrimitiveArray.primitiveIndexOffset = 0;

  OptixAccelBuildOptions accelOptions = {};
  if (prefer_fast_build) {
    accelOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
  } else {
    accelOptions.buildFlags |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
  }
  if (compact) {
    accelOptions.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  }
  accelOptions.motionOptions.numKeys = 1;
  accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes blas_buffer_sizes;
  OPTIX_CHECK(optixAccelComputeMemoryUsage(optix_context_, &accelOptions, &build_input, 1,
                                           &blas_buffer_sizes));
  return blas_buffer_sizes.outputSizeInBytes + blas_buffer_sizes.tempSizeInBytes;
}

// --- Private Methods ---

void RTEngine::initOptix(const RTConfig& config) {
  cudaFree(0);
  int numDevices;
  cudaGetDeviceCount(&numDevices);
  if (numDevices == 0)
    throw std::runtime_error("RTEngine: no CUDA capable devices found!");

  OPTIX_CHECK(optixInit());
}

void RTEngine::createContext() {
  CUresult cu_res = cuCtxGetCurrent(&cuda_context_);
  if (cu_res != CUDA_SUCCESS) {
    GPUSPATIAL_LOG_CRITICAL("Error querying current context: error code %d\n", cu_res);
    throw std::runtime_error("Error querying current context");
  }
  OptixDeviceContextOptions options = {};
  options.logCallbackFunction = context_log_cb;
  options.logCallbackData = nullptr;

#ifndef NDEBUG
  options.logCallbackLevel = 4;
  options.validationMode =
      OptixDeviceContextValidationMode::OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#else
  options.logCallbackLevel = 2;
#endif
  OPTIX_CHECK(optixDeviceContextCreate(cuda_context_, &options, &optix_context_));
}

void RTEngine::createModule(const RTConfig& config) {
  module_compile_options_.maxRegisterCount = config.max_reg_count;
  module_compile_options_.optLevel = config.opt_level;
  module_compile_options_.debugLevel = config.dbg_level;

  pipeline_link_options_.maxTraceDepth = config.max_trace_depth;

  for (const auto& [id, module] : config.modules) {
    std::vector<char> programData = readData(module.get_program_path());
    auto pipeline_compile_options = module.get_pipeline_compile_options();
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixModuleCreate(optix_context_, &module_compile_options_,
                                  &pipeline_compile_options, programData.data(),
                                  programData.size(), log, &sizeof_log,
                                  &resources_[id].module));
#ifndef NDEBUG
    if (sizeof_log > 1) {
      GPUSPATIAL_LOG_INFO("CreateModule %s", log);
    }
#endif
  }
}

void RTEngine::createRaygenPrograms(const RTConfig& config) {
  for (auto const& [id, module] : config.modules) {
    auto f_name = "__raygen__" + module.get_function_suffix();
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = resources_.at(id).module;
    pgDesc.raygen.entryFunctionName = f_name.c_str();

    char log[2048];
    size_t sizeof_log = sizeof(log);

    OPTIX_CHECK(optixProgramGroupCreate(optix_context_, &pgDesc, 1, &pgOptions, log,
                                        &sizeof_log, &resources_[id].raygen_pg));
#ifndef NDEBUG
    if (sizeof_log > 1) {
      GPUSPATIAL_LOG_INFO("CreateRaygenPrograms %s", log);
    }
#endif
  }
}

void RTEngine::createMissPrograms(const RTConfig& config) {
  for (auto const& [id, module] : config.modules) {
    auto f_name = "__miss__" + module.get_function_suffix();
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module = nullptr;
    pgDesc.miss.entryFunctionName = nullptr;

    if (module.IsMissEnable()) {
      pgDesc.miss.module = resources_.at(id).module;
      pgDesc.miss.entryFunctionName = f_name.c_str();
    }

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optix_context_, &pgDesc, 1, &pgOptions, log,
                                        &sizeof_log, &resources_[id].miss_pg));
#ifndef NDEBUG
    if (sizeof_log > 1) {
      GPUSPATIAL_LOG_INFO("CreateMissPrograms %s", log);
    }
#endif
  }
}

void RTEngine::createHitgroupPrograms(const RTConfig& config) {
  for (auto const& [id, module] : config.modules) {
    auto f_name_anythit = "__anyhit__" + module.get_function_suffix();
    auto f_name_intersect = "__intersection__" + module.get_function_suffix();
    auto f_name_closesthit = "__closesthit__" + module.get_function_suffix();
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pg_desc = {};
    pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pg_desc.hitgroup.moduleIS = nullptr;
    pg_desc.hitgroup.entryFunctionNameIS = nullptr;
    pg_desc.hitgroup.moduleAH = nullptr;
    pg_desc.hitgroup.entryFunctionNameAH = nullptr;
    pg_desc.hitgroup.moduleCH = nullptr;
    pg_desc.hitgroup.entryFunctionNameCH = nullptr;

    if (module.IsIsIntersectionEnabled()) {
      pg_desc.hitgroup.moduleIS = resources_.at(id).module;
      pg_desc.hitgroup.entryFunctionNameIS = f_name_intersect.c_str();
    }
    if (module.IsAnyHitEnable()) {
      pg_desc.hitgroup.moduleAH = resources_.at(id).module;
      pg_desc.hitgroup.entryFunctionNameAH = f_name_anythit.c_str();
    }
    if (module.IsClosestHitEnable()) {
      pg_desc.hitgroup.moduleCH = resources_.at(id).module;
      pg_desc.hitgroup.entryFunctionNameCH = f_name_closesthit.c_str();
    }

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optix_context_, &pg_desc, 1, &pgOptions, log,
                                        &sizeof_log, &resources_[id].hitgroup_pg));
#ifndef NDEBUG
    if (sizeof_log > 1) {
      GPUSPATIAL_LOG_INFO("CreateHitgroupPrograms %s", log);
    }
#endif
  }
}

void RTEngine::createPipeline(const RTConfig& config) {
  for (const auto& [id, module] : config.modules) {
    std::vector<OptixProgramGroup> program_groups;
    program_groups.push_back(resources_.at(id).raygen_pg);
    program_groups.push_back(resources_.at(id).miss_pg);
    program_groups.push_back(resources_.at(id).hitgroup_pg);
    auto options = module.get_pipeline_compile_options();
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(optix_context_, &options, &pipeline_link_options_,
                                    program_groups.data(), (int)program_groups.size(),
                                    log, &sizeof_log, &resources_[id].pipeline));
#ifndef NDEBUG
    if (sizeof_log > 1) {
      GPUSPATIAL_LOG_INFO("CreatePipeline %s", log);
    }
#endif
    OptixStackSizes stack_sizes = {};
    for (auto& prog_group : program_groups) {
      OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes,
                                                resources_.at(id).pipeline));
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, config.max_trace_depth, 0, 0,
                                           &direct_callable_stack_size_from_traversal,
                                           &direct_callable_stack_size_from_state,
                                           &continuation_stack_size));
    OPTIX_CHECK(optixPipelineSetStackSize(
        resources_.at(id).pipeline, direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state, continuation_stack_size,
        config.max_traversable_depth));
  }
}

void RTEngine::buildSBT(const RTConfig& config) {
  for (const auto& [id, module] : config.modules) {
    auto& res = resources_[id];
    auto& sbt = res.sbt;
    std::vector<RaygenRecord> raygenRecords;
    {
      RaygenRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(res.raygen_pg, &rec));
      rec.data = nullptr;
      raygenRecords.push_back(rec);
    }
    res.raygen_records = raygenRecords;
    sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(
        thrust::raw_pointer_cast(res.raygen_records.data()));

    std::vector<MissRecord> missRecords;
    {
      MissRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(res.miss_pg, &rec));
      rec.data = nullptr;
      missRecords.push_back(rec);
    }
    res.miss_records = missRecords;
    sbt.missRecordBase =
        reinterpret_cast<CUdeviceptr>(thrust::raw_pointer_cast(res.miss_records.data()));
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount = (int)missRecords.size();
    sbt.callablesRecordBase = 0;

    std::vector<HitgroupRecord> hitgroupRecords;
    {
      HitgroupRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(res.hitgroup_pg, &rec));
      rec.data = nullptr;
      hitgroupRecords.push_back(rec);
    }
    res.hitgroup_records = hitgroupRecords;
    sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(
        thrust::raw_pointer_cast(res.hitgroup_records.data()));
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
  }
}

size_t RTEngine::getAccelAlignedSize(size_t size) {
  if (size % OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT == 0) {
    return size;
  }
  return size - size % OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT +
         OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT;
}

std::vector<char> RTEngine::readData(const std::string& filename) {
  std::ifstream inputData(filename, std::ios::binary);
  if (inputData.fail()) {
    GPUSPATIAL_LOG_ERROR("readData() Failed to open file %s", filename);
    return {};
  }
  std::vector<char> data(std::istreambuf_iterator<char>(inputData), {});
  if (inputData.fail()) {
    GPUSPATIAL_LOG_ERROR("readData() Failed to read file %s", filename);
    return {};
  }
  return data;
}

void RTEngine::releaseOptixResources() {
  for (auto& [id, res] : resources_) {
    optixPipelineDestroy(res.pipeline);
    optixProgramGroupDestroy(res.raygen_pg);
    optixProgramGroupDestroy(res.miss_pg);
    optixProgramGroupDestroy(res.hitgroup_pg);
    optixModuleDestroy(res.module);
  }
  optixDeviceContextDestroy(optix_context_);
}

}  // namespace details
}  // namespace gpuspatial
