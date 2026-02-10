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
#include "gpuspatial/mem/memory_manager.hpp"
#include "gpuspatial/refine/rt_spatial_refiner.hpp"
#include "gpuspatial/rt/rt_engine.hpp"
#include "gpuspatial/utils/exception.hpp"

#include "nanoarrow/nanoarrow.hpp"

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

struct GpuSpatialRuntimeExporter {
  struct Payload {
    std::shared_ptr<gpuspatial::RTEngine> rt_engine;
    int device_id;
  };

  using private_data_t = GpuSpatialWrapper<Payload>;
  static void Export(struct GpuSpatialRuntime* out) {
    private_data_t* private_data =
        new private_data_t{Payload{std::make_shared<gpuspatial::RTEngine>()}, ""};
    out->init = CInit;
    out->release = CRelease;
    out->get_last_error = CGetLastError;
    out->private_data = private_data;
  }

  static int CInit(GpuSpatialRuntime* self, GpuSpatialRuntimeConfig* config) {
    return SafeExecute(static_cast<private_data_t*>(self->private_data), [&] {
      std::string ptx_root(config->ptx_root);
      auto rt_config = gpuspatial::get_default_rt_config(ptx_root);

      GPUSPATIAL_LOG_INFO("Initializing GpuSpatialRuntime on device %d, PTX root %s",
                          config->device_id, config->ptx_root);

      CUDA_CHECK(cudaSetDevice(config->device_id));

      gpuspatial::MemoryManager::instance().Init(config->use_cuda_memory_pool,
                                                 config->cuda_memory_pool_init_precent);

      static_cast<private_data_t*>(self->private_data)
          ->payload.rt_engine->Init(rt_config);
    });
  }

  static void CRelease(GpuSpatialRuntime* self) {
    gpuspatial::MemoryManager::instance().Shutdown();
    delete static_cast<private_data_t*>(self->private_data);
    self->private_data = nullptr;
  }

  static const char* CGetLastError(GpuSpatialRuntime* self) {
    auto* private_data = static_cast<private_data_t*>(self->private_data);
    return private_data->last_error.c_str();
  }
};

void GpuSpatialRuntimeCreate(struct GpuSpatialRuntime* runtime) {
  GpuSpatialRuntimeExporter::Export(runtime);
}

using runtime_data_t = GpuSpatialRuntimeExporter::private_data_t;

struct GpuSpatialIndexFloat2DExporter {
  using scalar_t = float;
  static constexpr int n_dim = 2;
  using self_t = SedonaFloatIndex2D;
  using spatial_index_t = gpuspatial::SpatialIndex<scalar_t, n_dim>;

  struct Payload {
    std::unique_ptr<spatial_index_t> index;
    runtime_data_t* rdata;
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

  static void Export(const struct GpuSpatialIndexConfig* config,
                     struct SedonaFloatIndex2D* out) {
    auto* rdata = static_cast<runtime_data_t*>(config->runtime->private_data);

    gpuspatial::RTSpatialIndexConfig index_config;

    index_config.rt_engine = rdata->payload.rt_engine;
    index_config.concurrency = config->concurrency;

    // Create SpatialIndex may involve GPU operations, set device here
    CUDA_CHECK(cudaSetDevice(rdata->payload.device_id));

    auto uniq_index = gpuspatial::CreateRTSpatialIndex<float, 2>(index_config);

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
    out->private_data = new private_data_t{Payload{std::move(uniq_index), rdata}, ""};
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
    auto* r_data = private_data->payload.rdata;

    CUDA_CHECK(cudaSetDevice(r_data->payload.device_id));
    return *(private_data->payload.index);
  }
};

int GpuSpatialIndexFloat2DCreate(struct SedonaFloatIndex2D* index,
                                 const struct GpuSpatialIndexConfig* config) {
  try {
    GpuSpatialIndexFloat2DExporter::Export(config, index);
  } catch (std::exception& e) {
    GPUSPATIAL_LOG_ERROR("Failed to create GpuSpatialIndexFloat2D: %s", e.what());
    return EINVAL;
  }
  return 0;
}

struct GpuSpatialRefinerExporter {
  struct Payload {
    std::unique_ptr<gpuspatial::SpatialRefiner> refiner;
    nanoarrow::UniqueArrayView build_array_view;
    nanoarrow::UniqueArrayView probe_array_view;
    runtime_data_t* rdata;
  };
  using private_data_t = GpuSpatialWrapper<Payload>;

  static void Export(const GpuSpatialRefinerConfig* config,
                     struct SedonaSpatialRefiner* out) {
    auto* rdata = static_cast<runtime_data_t*>(config->runtime->private_data);

    gpuspatial::RTSpatialRefinerConfig refiner_config;

    refiner_config.rt_engine = rdata->payload.rt_engine;
    refiner_config.concurrency = config->concurrency;
    refiner_config.compact = config->compress_bvh;
    refiner_config.pipeline_batches = config->pipeline_batches;

    // Create Refinner may involve GPU operations, set device here
    CUDA_CHECK(cudaSetDevice(rdata->payload.device_id));

    auto refiner = gpuspatial::CreateRTSpatialRefiner(refiner_config);

    out->clear = &CClear;
    out->init_schema = &CInitSchema;
    out->push_build = &CPushBuild;
    out->finish_building = &CFinishBuilding;
    out->refine = &CRefine;
    out->get_last_error = &CGetLastError;
    out->release = &CRelease;
    out->private_data =
        new private_data_t{Payload{std::move(refiner), nanoarrow::UniqueArrayView(),
                                   nanoarrow::UniqueArrayView(), rdata},
                           ""};
  }

  static int CClear(SedonaSpatialRefiner* self) {
    return SafeExecute(static_cast<private_data_t*>(self->private_data),
                       [&] { use_refiner(self).Clear(); });
  }

  static int CInitSchema(SedonaSpatialRefiner* self, const ArrowSchema* build_schema,
                         const ArrowSchema* probe_schema) {
    return SafeExecute(static_cast<private_data_t*>(self->private_data), [&] {
      auto* private_data = static_cast<private_data_t*>(self->private_data);
      ArrowError arrow_error;
      if (ArrowArrayViewInitFromSchema(private_data->payload.build_array_view.get(),
                                       build_schema, &arrow_error) != NANOARROW_OK) {
        throw std::runtime_error("ArrowArrayViewInitFromSchema error " +
                                 std::string(arrow_error.message));
      }
      if (ArrowArrayViewInitFromSchema(private_data->payload.probe_array_view.get(),
                                       probe_schema, &arrow_error) != NANOARROW_OK) {
        throw std::runtime_error("ArrowArrayViewInitFromSchema error " +
                                 std::string(arrow_error.message));
      }
    });
  }

  static int CPushBuild(SedonaSpatialRefiner* self, const ArrowArray* build_array) {
    return SafeExecute(static_cast<private_data_t*>(self->private_data), [&] {
      auto* private_data = static_cast<private_data_t*>(self->private_data);
      auto* array_view = private_data->payload.build_array_view.get();
      ArrowError arrow_error;

      if (ArrowArrayViewSetArray(array_view, build_array, &arrow_error) != NANOARROW_OK) {
        throw std::runtime_error("ArrowArrayViewSetArray error " +
                                 std::string(arrow_error.message));
      }

      use_refiner(self).PushBuild(array_view);
    });
  }

  static int CFinishBuilding(SedonaSpatialRefiner* self) {
    return SafeExecute(static_cast<private_data_t*>(self->private_data),
                       [&] { use_refiner(self).FinishBuilding(); });
  }

  static int CRefine(SedonaSpatialRefiner* self, const ArrowArray* probe_array,
                     SedonaSpatialRelationPredicate predicate, uint32_t* build_indices,
                     uint32_t* probe_indices, uint32_t indices_size,
                     uint32_t* new_indices_size) {
    return SafeExecute(static_cast<private_data_t*>(self->private_data), [&] {
      auto* private_data = static_cast<private_data_t*>(self->private_data);
      auto* array_view = private_data->payload.build_array_view.get();
      ArrowError arrow_error;

      if (ArrowArrayViewSetArray(array_view, probe_array, &arrow_error) != NANOARROW_OK) {
        throw std::runtime_error("ArrowArrayViewSetArray error " +
                                 std::string(arrow_error.message));
      }

      *new_indices_size = use_refiner(self).Refine(
          array_view, static_cast<gpuspatial::Predicate>(predicate), build_indices,
          probe_indices, indices_size);
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
    auto* r_data = private_data->payload.rdata;

    CUDA_CHECK(cudaSetDevice(r_data->payload.device_id));
    return *(private_data->payload.refiner);
  }
};

int GpuSpatialRefinerCreate(SedonaSpatialRefiner* refiner,
                            const GpuSpatialRefinerConfig* config) {
  try {
    GpuSpatialRefinerExporter::Export(config, refiner);
  } catch (std::exception& e) {
    GPUSPATIAL_LOG_ERROR("Failed to create GpuSpatialRefiner: %s", e.what());
    return EINVAL;
  }
  return 0;
}
