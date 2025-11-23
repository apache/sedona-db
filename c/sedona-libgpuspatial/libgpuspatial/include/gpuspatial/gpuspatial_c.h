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
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct GpuSpatialJoinerConfig {
  uint32_t concurrency;
  const char* ptx_root;
};

struct GpuSpatialJoinerContext {
  const char* last_error;  // Pointer to std::string to store last error message
  void* private_data;      // GPUSpatial context
  void* build_indices;     // Pointer to std::vector<uint32_t> to store results
  void* stream_indices;
};

enum GpuSpatialPredicate {
  GpuSpatialPredicateEquals = 0,
  GpuSpatialPredicateDisjoint,
  GpuSpatialPredicateTouches,
  GpuSpatialPredicateContains,
  GpuSpatialPredicateCovers,
  GpuSpatialPredicateIntersects,
  GpuSpatialPredicateWithin,
  GpuSpatialPredicateCoveredBy
};

struct GpuSpatialJoiner {
  int (*init)(struct GpuSpatialJoiner* self, struct GpuSpatialJoinerConfig* config);
  void (*clear)(struct GpuSpatialJoiner* self);
  void (*create_context)(struct GpuSpatialJoiner* self,
                         struct GpuSpatialJoinerContext* context);
  void (*destroy_context)(struct GpuSpatialJoinerContext* context);
  int (*push_build)(struct GpuSpatialJoiner* self, const struct ArrowSchema* schema,
                    const struct ArrowArray* array, int64_t offset, int64_t length);
  int (*finish_building)(struct GpuSpatialJoiner* self);
  int (*push_stream)(struct GpuSpatialJoiner* self,
                     struct GpuSpatialJoinerContext* context,
                     const struct ArrowSchema* schema, const struct ArrowArray* array,
                     int64_t offset, int64_t length, enum GpuSpatialPredicate predicate,
                     int32_t array_index_offset);
  void (*get_build_indices_buffer)(struct GpuSpatialJoinerContext* context,
                                   void** build_indices, uint32_t* build_indices_length);
  void (*get_stream_indices_buffer)(struct GpuSpatialJoinerContext* context,
                                    void** stream_indices,
                                    uint32_t* stream_indices_length);
  void (*release)(struct GpuSpatialJoiner* self);
  void* private_data;
  const char* last_error;
};

void GpuSpatialJoinerCreate(struct GpuSpatialJoiner* index);
#ifdef __cplusplus
}
#endif
