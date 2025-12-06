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
#include "gpuspatial/index/spatial_joiner.hpp"

#include <threads.h>
#include <memory>
#define GPUSPATIAL_ERROR_MSG_BUFFER_SIZE (1024)

struct GpuSpatialJoinerExporter {
  static void Export(std::unique_ptr<gpuspatial::StreamingJoiner>& idx,
                     struct GpuSpatialJoiner* out) {
    out->private_data = idx.release();
    out->init = &CInit;
    out->clear = &CClear;
    out->push_build = &CPushBuild;
    out->finish_building = &CFinishBuilding;
    out->create_context = &CCreateContext;
    out->destroy_context = &CDestroyContext;
    out->push_stream = &CPushStream;
    out->get_build_indices_buffer = &CGetBuildIndicesBuffer;
    out->get_stream_indices_buffer = &CGetStreamIndicesBuffer;
    out->release = &CRelease;
    out->last_error = new char[GPUSPATIAL_ERROR_MSG_BUFFER_SIZE];
  }

  static int CInit(struct GpuSpatialJoiner* self, struct GpuSpatialJoinerConfig* config) {
    int err = 0;
    auto* joiner = static_cast<gpuspatial::StreamingJoiner*>(self->private_data);
    try {
      gpuspatial::InitSpatialJoiner(joiner, config->ptx_root, config->concurrency);
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

  static void CCreateContext(struct GpuSpatialJoiner* self,
                             struct GpuSpatialJoinerContext* context) {
    auto* joiner = static_cast<gpuspatial::StreamingJoiner*>(self->private_data);
    context->private_data = new std::shared_ptr(joiner->CreateContext());
    context->last_error = new char[GPUSPATIAL_ERROR_MSG_BUFFER_SIZE];
    context->build_indices = new std::vector<uint32_t>();
    context->stream_indices = new std::vector<uint32_t>();
  }

  static void CDestroyContext(struct GpuSpatialJoinerContext* context) {
    delete (std::shared_ptr<gpuspatial::StreamingJoiner::Context>*)context->private_data;
    delete[] context->last_error;
    delete (std::vector<uint32_t>*)context->build_indices;
    delete (std::vector<uint32_t>*)context->stream_indices;
    context->private_data = nullptr;
    context->last_error = nullptr;
    context->build_indices = nullptr;
    context->stream_indices = nullptr;
  }

  static void CClear(struct GpuSpatialJoiner* self) {
    auto* joiner = static_cast<gpuspatial::StreamingJoiner*>(self->private_data);
    joiner->Clear();
  }

  static int CPushBuild(struct GpuSpatialJoiner* self, const struct ArrowSchema* schema,
                        const struct ArrowArray* array, int64_t offset, int64_t length) {
    auto* joiner = static_cast<gpuspatial::StreamingJoiner*>(self->private_data);
    int err = 0;
    try {
      joiner->PushBuild(schema, array, offset, length);
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

  static int CFinishBuilding(struct GpuSpatialJoiner* self) {
    auto* joiner = static_cast<gpuspatial::StreamingJoiner*>(self->private_data);
    int err = 0;
    try {
      joiner->FinishBuilding();
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

  static int CPushStream(struct GpuSpatialJoiner* self,
                         struct GpuSpatialJoinerContext* context,
                         const struct ArrowSchema* schema, const struct ArrowArray* array,
                         int64_t offset, int64_t length,
                         enum GpuSpatialPredicate predicate, int32_t array_index_offset) {
    auto* joiner = static_cast<gpuspatial::StreamingJoiner*>(self->private_data);
    auto* private_data =
        (std::shared_ptr<gpuspatial::StreamingJoiner::Context>*)context->private_data;
    int err = 0;
    try {
      joiner->PushStream(private_data->get(), schema, array, offset, length,
                         static_cast<gpuspatial::Predicate>(predicate),
                         static_cast<std::vector<uint32_t>*>(context->build_indices),
                         static_cast<std::vector<uint32_t>*>(context->stream_indices),
                         array_index_offset);
    } catch (const std::exception& e) {
      int len =
          std::min(strlen(e.what()), (size_t)(GPUSPATIAL_ERROR_MSG_BUFFER_SIZE - 1));
      strncpy((char*)context->last_error, e.what(), len);
      ((char*)context->last_error)[len] = '\0';
      err = EINVAL;
    }
    return err;
  }

  static void CGetBuildIndicesBuffer(struct GpuSpatialJoinerContext* context,
                                     void** build_indices,
                                     uint32_t* build_indices_length) {
    auto* vec = static_cast<std::vector<uint32_t>*>(context->build_indices);

    *build_indices = vec->data();
    *build_indices_length = vec->size();
  }

  static void CGetStreamIndicesBuffer(struct GpuSpatialJoinerContext* context,
                                      void** stream_indices,
                                      uint32_t* stream_indices_length) {
    auto* vec = static_cast<std::vector<uint32_t>*>(context->stream_indices);

    *stream_indices = vec->data();
    *stream_indices_length = vec->size();
  }

  static void CRelease(struct GpuSpatialJoiner* self) {
    delete[] self->last_error;
    auto* joiner = static_cast<gpuspatial::StreamingJoiner*>(self->private_data);
    delete joiner;
    self->private_data = nullptr;
    self->last_error = nullptr;
  }
};

void GpuSpatialJoinerCreate(struct GpuSpatialJoiner* joiner) {
  auto idx = gpuspatial::CreateSpatialJoiner();
  GpuSpatialJoinerExporter::Export(idx, joiner);
}
