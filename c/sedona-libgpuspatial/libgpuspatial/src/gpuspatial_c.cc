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

#include "gpuspatial/gpuspatial_c.h"
#include "gpuspatial/index/rt_spatial_index.hpp"
#include "gpuspatial/index/spatial_index.hpp"
#include "gpuspatial/refine/rt_spatial_refiner.hpp"
#include "gpuspatial/rt/rt_engine.hpp"
#include "gpuspatial/utils/exception.h"

#include <threads.h>
#include <algorithm>
#include <cstring>
#include <memory>

// -----------------------------------------------------------------------------
// INTERNAL HELPERS
// -----------------------------------------------------------------------------
// This is what the private_data points to for the public C interfaces
template <typename T>
struct GpuSpatialWrapper {
  T payload;
  std::string last_error;  // Pointer to std::string to store last error message
};

// The unified error handling wrapper
// Func: The lambda containing the logic
template <typename T, typename Func>
int SafeExecute(GpuSpatialWrapper<T>* wrapper, Func&& func) {
  try {
    func();
    wrapper->last_error.clear();
    return 0;
  } catch (const std::exception& e) {
    wrapper->last_error = std::string(e.what());
    return EINVAL;
  } catch (...) {
    wrapper->last_error = "Unknown internal error";
    return EINVAL;
  }
}

// -----------------------------------------------------------------------------
// IMPLEMENTATION
// -----------------------------------------------------------------------------

struct GpuSpatialRTEngineExporter {
  using private_data_t = GpuSpatialWrapper<std::shared_ptr<gpuspatial::RTEngine>>;
  static void Export(private_data_t* private_data, struct GpuSpatialRTEngine* out) {
    out->init = CInit;
    out->release = CRelease;
    out->get_last_error = CGetLastError;
    out->private_data = private_data;
  }

  static int CInit(GpuSpatialRTEngine* self, GpuSpatialRTEngineConfig* config) {
    return SafeExecute(static_cast<private_data_t*>(self->private_data), [&] {
      std::string ptx_root(config->ptx_root);
      auto rt_config = gpuspatial::get_default_rt_config(ptx_root);

      CUDA_CHECK(cudaSetDevice(config->device_id));
      static_cast<private_data_t*>(self->private_data)->payload->Init(rt_config);
    });
  }

  static void CRelease(GpuSpatialRTEngine* self) {
    delete static_cast<private_data_t*>(self->private_data);
    self->private_data = nullptr;
  }

  static const char* CGetLastError(GpuSpatialRTEngine* self) {
    auto* private_data = static_cast<private_data_t*>(self->private_data);
    return private_data->last_error.c_str();
  }
};

int GpuSpatialRTEngineCreate(struct GpuSpatialRTEngine* instance) {
  try {
    auto rt_engine = std::make_shared<gpuspatial::RTEngine>();
    GpuSpatialRTEngineExporter::Export(
        new GpuSpatialWrapper<std::shared_ptr<gpuspatial::RTEngine>>{rt_engine},
        instance);
  } catch (std::exception& e) {
    GpuSpatialRTEngineExporter::Export(
        new GpuSpatialWrapper<std::shared_ptr<gpuspatial::RTEngine>>{nullptr, e.what()},
        instance);
    return EINVAL;
  } catch (...) {
    GpuSpatialRTEngineExporter::Export(
        new GpuSpatialWrapper<std::shared_ptr<gpuspatial::RTEngine>>{nullptr,
                                                                     "Unknown error"},
        instance);
    return EINVAL;
  }
  return 0;
}

struct GpuSpatialIndexFloat2DExporter {
  using scalar_t = float;
  static constexpr int n_dim = 2;
  using self_t = SedonaFloatIndex2D;
  using spatial_index_t = gpuspatial::SpatialIndex<scalar_t, n_dim>;

  struct Payload {
    std::unique_ptr<spatial_index_t> index;
    int device_id;
  };

  struct ResultBuffer {
    std::vector<uint32_t> build_indices;
    std::vector<uint32_t> probe_indices;
    ResultBuffer() = default;

    ResultBuffer(const ResultBuffer&) = delete;
    ResultBuffer& operator=(const ResultBuffer&) = delete;

    ResultBuffer(ResultBuffer&&) = default;
    ResultBuffer& operator=(ResultBuffer&&) = default;
  };

  using private_data_t = GpuSpatialWrapper<Payload>;
  using context_t = GpuSpatialWrapper<ResultBuffer>;

  static void Export(std::unique_ptr<spatial_index_t> index, int device_id,
                     const std::string& last_error, struct SedonaFloatIndex2D* out) {
    out->clear = &CClear;
    out->create_context = &CCreateContext;
    out->destroy_context = &CDestroyContext;
    out->push_build = &CPushBuild;
    out->finish_building = &CFinishBuilding;
    out->probe = &CProbe;
    out->get_build_indices_buffer = &CGetBuildIndicesBuffer;
    out->get_probe_indices_buffer = &CGetProbeIndicesBuffer;
    out->get_last_error = &CGetLastError;
    out->context_get_last_error = &CContextGetLastError;
    out->release = &CRelease;
    out->private_data =
        new private_data_t{Payload{std::move(index), device_id}, last_error};
  }

  static void CCreateContext(struct SedonaSpatialIndexContext* context) {
    context->private_data = new context_t();
  }

  static void CDestroyContext(struct SedonaSpatialIndexContext* context) {
    delete static_cast<context_t*>(context->private_data);
    context->private_data = nullptr;
  }

  static int CClear(self_t* self) {
    return SafeExecute(static_cast<private_data_t*>(self->private_data),
                       [=] { use_index(self).Clear(); });
  }

  static int CPushBuild(self_t* self, const float* buf, uint32_t n_rects) {
    return SafeExecute(static_cast<private_data_t*>(self->private_data), [&] {
      auto* rects = reinterpret_cast<const spatial_index_t::box_t*>(buf);
      use_index(self).PushBuild(rects, n_rects);
    });
  }

  static int CFinishBuilding(self_t* self) {
    return SafeExecute(static_cast<private_data_t*>(self->private_data),
                       [&] { use_index(self).FinishBuilding(); });
  }

  static int CProbe(self_t* self, SedonaSpatialIndexContext* context, const float* buf,
                    uint32_t n_rects) {
    return SafeExecute(static_cast<context_t*>(context->private_data), [&] {
      auto* rects = reinterpret_cast<const spatial_index_t::box_t*>(buf);
      auto& buff = static_cast<context_t*>(context->private_data)->payload;
      use_index(self).Probe(rects, n_rects, &buff.build_indices, &buff.probe_indices);
    });
  }

  static void CGetBuildIndicesBuffer(struct SedonaSpatialIndexContext* context,
                                     uint32_t** build_indices,
                                     uint32_t* build_indices_length) {
    auto* ctx = static_cast<context_t*>(context->private_data);
    *build_indices = ctx->payload.build_indices.data();
    *build_indices_length = ctx->payload.build_indices.size();
  }

  static void CGetProbeIndicesBuffer(struct SedonaSpatialIndexContext* context,
                                     uint32_t** probe_indices,
                                     uint32_t* probe_indices_length) {
    auto* ctx = static_cast<context_t*>(context->private_data);
    *probe_indices = ctx->payload.probe_indices.data();
    *probe_indices_length = ctx->payload.probe_indices.size();
  }

  static const char* CGetLastError(self_t* self) {
    auto* private_data = static_cast<private_data_t*>(self->private_data);
    return private_data->last_error.c_str();
  }

  static const char* CContextGetLastError(SedonaSpatialIndexContext* self) {
    auto* private_data = static_cast<context_t*>(self->private_data);
    return private_data->last_error.c_str();
  }

  static void CRelease(self_t* self) {
    delete static_cast<private_data_t*>(self->private_data);
    self->private_data = nullptr;
  }

  static spatial_index_t& use_index(self_t* self) {
    auto* private_data = static_cast<private_data_t*>(self->private_data);

    CUDA_CHECK(cudaSetDevice(private_data->payload.device_id));
    if (private_data->payload.index == nullptr) {
      throw std::runtime_error("SpatialIndex is not initialized");
    }
    return *(private_data->payload.index);
  }
};

int GpuSpatialIndexFloat2DCreate(struct SedonaFloatIndex2D* index,
                                 const struct GpuSpatialIndexConfig* config) {
  gpuspatial::RTSpatialIndexConfig rt_index_config;
  auto rt_engine = static_cast<GpuSpatialWrapper<std::shared_ptr<gpuspatial::RTEngine>>*>(
                       config->rt_engine->private_data)
                       ->payload;
  rt_index_config.rt_engine = rt_engine;
  rt_index_config.concurrency = config->concurrency;
  try {
    if (rt_index_config.rt_engine == nullptr) {
      throw std::runtime_error("RTEngine is not initialized");
    }
    // Create SpatialIndex may involve GPU operations, set device here
    CUDA_CHECK(cudaSetDevice(config->device_id));

    auto uniq_index = gpuspatial::CreateRTSpatialIndex<float, 2>(rt_index_config);
    GpuSpatialIndexFloat2DExporter::Export(std::move(uniq_index), config->device_id, "",
                                           index);
  } catch (std::exception& e) {
    GpuSpatialIndexFloat2DExporter::Export(nullptr, config->device_id, e.what(), index);
    return EINVAL;
  }
  return 0;
}

struct GpuSpatialRefinerExporter {
  struct Payload {
    std::unique_ptr<gpuspatial::SpatialRefiner> refiner;
    int device_id;
  };
  using private_data_t = GpuSpatialWrapper<Payload>;

  static void Export(std::unique_ptr<gpuspatial::SpatialRefiner> refiner, int device_id,
                     const std::string& last_error, struct SedonaSpatialRefiner* out) {
    out->clear = &CClear;
    out->push_build = &CPushBuild;
    out->finish_building = &CFinishBuilding;
    out->refine_loaded = &CRefineLoaded;
    out->refine = &CRefine;
    out->get_last_error = &CGetLastError;
    out->release = &CRelease;
    out->private_data =
        new private_data_t{Payload{std::move(refiner), device_id}, last_error};
  }

  static int CClear(SedonaSpatialRefiner* self) {
    return SafeExecute(static_cast<private_data_t*>(self->private_data),
                       [&] { use_refiner(self).Clear(); });
  }

  static int CPushBuild(SedonaSpatialRefiner* self, const ArrowSchema* build_schema,
                        const ArrowArray* build_array) {
    return SafeExecute(static_cast<private_data_t*>(self->private_data),
                       [&] { use_refiner(self).PushBuild(build_schema, build_array); });
  }

  static int CFinishBuilding(SedonaSpatialRefiner* self) {
    return SafeExecute(static_cast<private_data_t*>(self->private_data),
                       [&] { use_refiner(self).FinishBuilding(); });
  }

  static int CRefineLoaded(SedonaSpatialRefiner* self, const ArrowSchema* probe_schema,
                           const ArrowArray* probe_array,
                           SedonaSpatialRelationPredicate predicate,
                           uint32_t* build_indices, uint32_t* probe_indices,
                           uint32_t indices_size, uint32_t* new_indices_size) {
    return SafeExecute(static_cast<private_data_t*>(self->private_data), [&] {
      *new_indices_size = use_refiner(self).Refine(
          probe_schema, probe_array, static_cast<gpuspatial::Predicate>(predicate),
          build_indices, probe_indices, indices_size);
    });
  }

  static int CRefine(SedonaSpatialRefiner* self, const ArrowSchema* schema1,
                     const ArrowArray* array1, const ArrowSchema* schema2,
                     const ArrowArray* array2, SedonaSpatialRelationPredicate predicate,
                     uint32_t* indices1, uint32_t* indices2, uint32_t indices_size,
                     uint32_t* new_indices_size) {
    return SafeExecute(static_cast<private_data_t*>(self->private_data), [&] {
      *new_indices_size = use_refiner(self).Refine(
          schema1, array1, schema2, array2, static_cast<gpuspatial::Predicate>(predicate),
          indices1, indices2, indices_size);
    });
  }

  static const char* CGetLastError(SedonaSpatialRefiner* self) {
    auto* private_data = static_cast<private_data_t*>(self->private_data);
    return private_data->last_error.c_str();
  }

  static void CRelease(SedonaSpatialRefiner* self) {
    delete static_cast<private_data_t*>(self->private_data);
    self->private_data = nullptr;
  }

  static gpuspatial::SpatialRefiner& use_refiner(SedonaSpatialRefiner* self) {
    auto* private_data = static_cast<private_data_t*>(self->private_data);

    CUDA_CHECK(cudaSetDevice(private_data->payload.device_id));
    if (private_data->payload.refiner == nullptr) {
      throw std::runtime_error("SpatialRefiner is not initialized");
    }
    return *(private_data->payload.refiner);
  }
};

int GpuSpatialRefinerCreate(SedonaSpatialRefiner* refiner,
                            const GpuSpatialRefinerConfig* config) {
  gpuspatial::RTSpatialRefinerConfig rt_refiner_config;
  auto rt_engine = static_cast<GpuSpatialWrapper<std::shared_ptr<gpuspatial::RTEngine>>*>(
                       config->rt_engine->private_data)
                       ->payload;

  rt_refiner_config.rt_engine = rt_engine;
  rt_refiner_config.concurrency = config->concurrency;
  rt_refiner_config.compact = config->compress_bvh;
  rt_refiner_config.pipeline_batches = config->pipeline_batches;

  try {
    if (rt_refiner_config.rt_engine == nullptr) {
      throw std::runtime_error("RTEngine is not initialized");
    }
    // Create Refinner may involve GPU operations, set device here
    CUDA_CHECK(cudaSetDevice(config->device_id));

    auto uniq_refiner = gpuspatial::CreateRTSpatialRefiner(rt_refiner_config);
    GpuSpatialRefinerExporter::Export(std::move(uniq_refiner), config->device_id, "",
                                      refiner);
  } catch (std::exception& e) {
    GpuSpatialRefinerExporter::Export(nullptr, config->device_id, e.what(), refiner);
    return EINVAL;
  }
  return 0;
}
