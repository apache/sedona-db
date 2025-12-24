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
#include <memory>

#define GPUSPATIAL_ERROR_MSG_BUFFER_SIZE (1024)

struct GpuSpatialRTEngineExporter {
  static void Export(std::shared_ptr<gpuspatial::RTEngine> rt_engine,
                     struct GpuSpatialRTEngine* out) {
    out->init = CInit;
    out->release = CRelease;
    out->private_data = new std::shared_ptr<gpuspatial::RTEngine>(rt_engine);
    out->last_error = new char[GPUSPATIAL_ERROR_MSG_BUFFER_SIZE];
  }

  static int CInit(GpuSpatialRTEngine* self, GpuSpatialRTEngineConfig* config) {
    int err = 0;
    auto rt_engine = (std::shared_ptr<gpuspatial::RTEngine>*)self->private_data;
    std::string ptx_root(config->ptx_root);
    auto rt_config = gpuspatial::get_default_rt_config(ptx_root);
    try {
      CUDA_CHECK(cudaSetDevice(config->device_id));
      rt_engine->get()->Init(rt_config);
    } catch (const std::exception& e) {
      int len =
          std::min(strlen(e.what()), (size_t)(GPUSPATIAL_ERROR_MSG_BUFFER_SIZE - 1));
      auto* last_error = const_cast<char*>(self->last_error);
      strncpy(last_error, e.what(), len);
      last_error[len] = '\0';
      err = EINVAL;
    }
    return err;
  }

  static void CRelease(GpuSpatialRTEngine* self) {
    delete static_cast<std::shared_ptr<gpuspatial::RTEngine>*>(self->private_data);
    delete[] self->last_error;
    self->private_data = nullptr;
    self->last_error = nullptr;
  }
};

void GpuSpatialRTEngineCreate(struct GpuSpatialRTEngine* instance) {
  auto rt_engine = std::make_shared<gpuspatial::RTEngine>();
  GpuSpatialRTEngineExporter::Export(rt_engine, instance);
}

struct GpuSpatialIndexFloat2DExporter {
  using scalar_t = float;
  static constexpr int n_dim = 2;
  using self_t = GpuSpatialIndexFloat2D;
  using spatial_index_t = gpuspatial::SpatialIndex<scalar_t, n_dim>;
  static void Export(std::unique_ptr<spatial_index_t>& idx,
                     struct GpuSpatialIndexFloat2D* out) {
    out->init = &CInit;
    out->clear = &CClear;
    out->create_context = &CCreateContext;
    out->destroy_context = &CDestroyContext;
    out->push_build = &CPushBuild;
    out->finish_building = &CFinishBuilding;
    out->probe = &CProbe;
    out->get_build_indices_buffer = &CGetBuildIndicesBuffer;
    out->get_probe_indices_buffer = &CGetProbeIndicesBuffer;
    out->release = &CRelease;
    out->private_data = idx.release();
    out->last_error = new char[GPUSPATIAL_ERROR_MSG_BUFFER_SIZE];
  }

  static int CInit(self_t* self, GpuSpatialIndexConfig* config) {
    int err = 0;
    auto* index = static_cast<spatial_index_t*>(self->private_data);
    try {
      gpuspatial::RTSpatialIndexConfig<scalar_t, n_dim> index_config;

      auto rt_engine =
          (std::shared_ptr<gpuspatial::RTEngine>*)config->rt_engine->private_data;
      index_config.rt_engine = *rt_engine;
      index_config.concurrency = config->concurrency;

      CUDA_CHECK(cudaSetDevice(config->device_id));
      index->Init(&index_config);
    } catch (const std::exception& e) {
      int len =
          std::min(strlen(e.what()), (size_t)(GPUSPATIAL_ERROR_MSG_BUFFER_SIZE - 1));
      auto* last_error = const_cast<char*>(self->last_error);
      strncpy(last_error, e.what(), len);
      last_error[len] = '\0';
      err = EINVAL;
    }
    return err;
  }

  static void CCreateContext(self_t* self, struct GpuSpatialIndexContext* context) {
    auto* index = static_cast<spatial_index_t*>(self->private_data);
    context->last_error = new char[GPUSPATIAL_ERROR_MSG_BUFFER_SIZE];
    context->build_indices = new std::vector<uint32_t>();
    context->probe_indices = new std::vector<uint32_t>();
  }

  static void CDestroyContext(struct GpuSpatialIndexContext* context) {
    delete[] context->last_error;
    delete (std::vector<uint32_t>*)context->build_indices;
    delete (std::vector<uint32_t>*)context->probe_indices;
    context->last_error = nullptr;
    context->build_indices = nullptr;
    context->probe_indices = nullptr;
  }

  static void CClear(self_t* self) {
    auto* index = static_cast<spatial_index_t*>(self->private_data);
    index->Clear();
  }

  static int CPushBuild(self_t* self, const float* buf, uint32_t n_rects) {
    auto* index = static_cast<spatial_index_t*>(self->private_data);
    int err = 0;
    try {
      auto* rects = reinterpret_cast<const spatial_index_t::box_t*>(buf);

      index->PushBuild(rects, n_rects);
    } catch (const std::exception& e) {
      int len =
          std::min(strlen(e.what()), (size_t)(GPUSPATIAL_ERROR_MSG_BUFFER_SIZE - 1));
      auto* last_error = const_cast<char*>(self->last_error);
      strncpy(last_error, e.what(), len);
      last_error[len] = '\0';
      err = EINVAL;
    }
    return err;
  }

  static int CFinishBuilding(self_t* self) {
    auto* index = static_cast<spatial_index_t*>(self->private_data);
    int err = 0;
    try {
      index->FinishBuilding();
    } catch (const std::exception& e) {
      int len =
          std::min(strlen(e.what()), (size_t)(GPUSPATIAL_ERROR_MSG_BUFFER_SIZE - 1));
      auto* last_error = const_cast<char*>(self->last_error);
      strncpy(last_error, e.what(), len);
      last_error[len] = '\0';
      err = EINVAL;
    }
    return err;
  }

  static int CProbe(self_t* self, GpuSpatialIndexContext* context, const float* buf,
                    uint32_t n_rects) {
    auto* index = static_cast<spatial_index_t*>(self->private_data);
    auto* rects = reinterpret_cast<const spatial_index_t::box_t*>(buf);
    int err = 0;
    try {
      index->Probe(rects, n_rects,
                   static_cast<std::vector<uint32_t>*>(context->build_indices),
                   static_cast<std::vector<uint32_t>*>(context->probe_indices));
    } catch (const std::exception& e) {
      int len =
          std::min(strlen(e.what()), (size_t)(GPUSPATIAL_ERROR_MSG_BUFFER_SIZE - 1));
      strncpy((char*)context->last_error, e.what(), len);
      ((char*)context->last_error)[len] = '\0';
      err = EINVAL;
    }
    return err;
  }

  static void CGetBuildIndicesBuffer(struct GpuSpatialIndexContext* context,
                                     void** build_indices,
                                     uint32_t* build_indices_length) {
    auto* vec = static_cast<std::vector<uint32_t>*>(context->build_indices);

    *build_indices = vec->data();
    *build_indices_length = vec->size();
  }

  static void CGetProbeIndicesBuffer(struct GpuSpatialIndexContext* context,
                                     void** probe_indices,
                                     uint32_t* probe_indices_length) {
    auto* vec = static_cast<std::vector<uint32_t>*>(context->probe_indices);

    *probe_indices = vec->data();
    *probe_indices_length = vec->size();
  }

  static void CRelease(self_t* self) {
    delete[] self->last_error;
    delete static_cast<spatial_index_t*>(self->private_data);
    self->private_data = nullptr;
    self->last_error = nullptr;
  }
};

void GpuSpatialIndexFloat2DCreate(struct GpuSpatialIndexFloat2D* index) {
  auto uniq_index = gpuspatial::CreateRTSpatialIndex<float, 2>();
  GpuSpatialIndexFloat2DExporter::Export(uniq_index, index);
}

struct GpuSpatialRefinerExporter {
  static void Export(std::unique_ptr<gpuspatial::SpatialRefiner>& refiner,
                     struct GpuSpatialRefiner* out) {
    out->private_data = refiner.release();
    out->init = &CInit;
    out->load_build_array = &CLoadBuildArray;
    out->refine_loaded = &CRefineLoaded;
    out->refine = &CRefine;
    out->release = &CRelease;
    out->last_error = new char[GPUSPATIAL_ERROR_MSG_BUFFER_SIZE];
  }

  static int CInit(GpuSpatialRefiner* self, GpuSpatialRefinerConfig* config) {
    int err = 0;
    auto* refiner = static_cast<gpuspatial::SpatialRefiner*>(self->private_data);
    try {
      gpuspatial::RTSpatialRefinerConfig refiner_config;

      auto rt_engine =
          (std::shared_ptr<gpuspatial::RTEngine>*)config->rt_engine->private_data;
      refiner_config.rt_engine = *rt_engine;
      refiner_config.concurrency = config->concurrency;
      CUDA_CHECK(cudaSetDevice(config->device_id));
      refiner->Init(&refiner_config);
    } catch (const std::exception& e) {
      int len =
          std::min(strlen(e.what()), (size_t)(GPUSPATIAL_ERROR_MSG_BUFFER_SIZE - 1));
      auto* last_error = const_cast<char*>(self->last_error);
      strncpy(last_error, e.what(), len);
      last_error[len] = '\0';
      err = EINVAL;
    }
    return err;
  }

  static int CLoadBuildArray(GpuSpatialRefiner* self, const ArrowSchema* build_schema,
                             const ArrowArray* build_array) {
    int err = 0;
    auto* refiner = static_cast<gpuspatial::SpatialRefiner*>(self->private_data);
    try {
      refiner->Clear();
      refiner->LoadBuildArray(build_schema, build_array);
    } catch (const std::exception& e) {
      int len =
          std::min(strlen(e.what()), (size_t)(GPUSPATIAL_ERROR_MSG_BUFFER_SIZE - 1));
      auto* last_error = const_cast<char*>(self->last_error);
      strncpy(last_error, e.what(), len);
      last_error[len] = '\0';
      err = EINVAL;
    }
    return err;
  }

  static int CRefineLoaded(GpuSpatialRefiner* self, const ArrowSchema* probe_schema,
                           const ArrowArray* probe_array,
                           GpuSpatialRelationPredicate predicate, uint32_t* build_indices,
                           uint32_t* probe_indices, uint32_t indices_size,
                           uint32_t* new_indices_size) {
    auto* refiner = static_cast<gpuspatial::SpatialRefiner*>(self->private_data);
    int err = 0;
    try {
      *new_indices_size = refiner->Refine(probe_schema, probe_array,
                                          static_cast<gpuspatial::Predicate>(predicate),
                                          build_indices, probe_indices, indices_size);
    } catch (const std::exception& e) {
      int len =
          std::min(strlen(e.what()), (size_t)(GPUSPATIAL_ERROR_MSG_BUFFER_SIZE - 1));
      auto* last_error = const_cast<char*>(self->last_error);
      strncpy(last_error, e.what(), len);
      last_error[len] = '\0';
      err = EINVAL;
    }
    return err;
  }

  static int CRefine(GpuSpatialRefiner* self, const ArrowSchema* schema1,
                     const ArrowArray* array1, const ArrowSchema* schema2,
                     const ArrowArray* array2, GpuSpatialRelationPredicate predicate,
                     uint32_t* indices1, uint32_t* indices2, uint32_t indices_size,
                     uint32_t* new_indices_size) {
    auto* refiner = static_cast<gpuspatial::SpatialRefiner*>(self->private_data);
    int err = 0;
    try {
      *new_indices_size = refiner->Refine(schema1, array1, schema2, array2,
                                          static_cast<gpuspatial::Predicate>(predicate),
                                          indices1, indices2, indices_size);
    } catch (const std::exception& e) {
      int len =
          std::min(strlen(e.what()), (size_t)(GPUSPATIAL_ERROR_MSG_BUFFER_SIZE - 1));
      auto* last_error = const_cast<char*>(self->last_error);
      strncpy(last_error, e.what(), len);
      last_error[len] = '\0';
      err = EINVAL;
    }
    return err;
  }

  static void CRelease(GpuSpatialRefiner* self) {
    delete[] self->last_error;
    auto* joiner = static_cast<gpuspatial::SpatialRefiner*>(self->private_data);
    delete joiner;
    self->private_data = nullptr;
    self->last_error = nullptr;
  }
};

void GpuSpatialRefinerCreate(GpuSpatialRefiner* refiner) {
  auto uniq_refiner = gpuspatial::CreateRTSpatialRefiner();
  GpuSpatialRefinerExporter::Export(uniq_refiner, refiner);
}
