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
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ArrowSchema;
struct ArrowArray;

// Interfaces for ray-tracing engine (OptiX)
struct GpuSpatialRuntimeConfig {
  /** Path to PTX files */
  const char* ptx_root;
  /** Device ID to use, 0 is the first GPU */
  int device_id;
  /** Ratio of initial memory pool size to total GPU memory, between 0.0 and 1.0; zero is
   * effectively disable async memory allocation and using cudaMalloc */
  float cuda_init_memory_pool_ratio;
};

struct GpuSpatialRuntime {
  /** Initialize the runtime (OptiX) with the given configuration
   * @return 0 on success, non-zero on failure
   */
  int (*init)(struct GpuSpatialRuntime* self, struct GpuSpatialRuntimeConfig* config);
  void (*release)(struct GpuSpatialRuntime* self);
  const char* (*get_last_error)(struct GpuSpatialRuntime* self);
  void* private_data;
};

/** Create an instance of GpuSpatialRuntime */
void GpuSpatialRuntimeCreate(struct GpuSpatialRuntime* runtime);

struct GpuSpatialIndexConfig {
  /** Pointer to an initialized GpuSpatialRuntime struct */
  struct GpuSpatialRuntime* runtime;
  /** How many threads will concurrently call Probe method */
  uint32_t concurrency;
};

// An opaque context for concurrent probing
struct SedonaSpatialIndexContext {
  void* private_data;
};

struct SedonaFloatIndex2D {
  /** Clear the spatial index, removing all built data */
  int (*clear)(struct SedonaFloatIndex2D* self);
  /** Create a new context for concurrent probing */
  void (*create_context)(struct SedonaSpatialIndexContext* context);
  /** Destroy a previously created context */
  void (*destroy_context)(struct SedonaSpatialIndexContext* context);
  /** Push rectangles for building the spatial index, each rectangle is represented by 4
   * floats: [min_x, min_y, max_x, max_y] Points can also be indexed by providing [x, y,
   * x, y] but points and rectangles cannot be mixed
   *
   * @return 0 on success, non-zero on failure
   */
  int (*push_build)(struct SedonaFloatIndex2D* self, const float* buf, uint32_t n_rects);
  /**
   * Finish building the spatial index after all rectangles have been pushed
   *
   * @return 0 on success, non-zero on failure
   */
  int (*finish_building)(struct SedonaFloatIndex2D* self);
  /**
   * Probe the spatial index with the given rectangles, each rectangle is represented by 4
   * floats: [min_x, min_y, max_x, max_y] Points can also be probed by providing [x, y, x,
   * y] but points and rectangles cannot be mixed in one Probe call. The results of the
   * probe will be stored in the context.
   *
   * @return 0 on success, non-zero on failure
   */
  int (*probe)(struct SedonaFloatIndex2D* self, struct SedonaSpatialIndexContext* context,
               const float* buf, uint32_t n_rects);
  /** Get the build indices buffer from the context
   *
   * @return A pointer to the buffer and its length
   */
  void (*get_build_indices_buffer)(struct SedonaSpatialIndexContext* context,
                                   uint32_t** build_indices,
                                   uint32_t* build_indices_length);
  /** Get the probe indices buffer from the context
   *
   * @return A pointer to the buffer and its length
   */
  void (*get_probe_indices_buffer)(struct SedonaSpatialIndexContext* context,
                                   uint32_t** probe_indices,
                                   uint32_t* probe_indices_length);
  const char* (*get_last_error)(struct SedonaFloatIndex2D* self);
  const char* (*context_get_last_error)(struct SedonaSpatialIndexContext* context);
  /** Release the spatial index and free all resources */
  void (*release)(struct SedonaFloatIndex2D* self);
  void* private_data;
};

int GpuSpatialIndexFloat2DCreate(struct SedonaFloatIndex2D* index,
                                 const struct GpuSpatialIndexConfig* config);

struct GpuSpatialRefinerConfig {
  /** Pointer to an initialized GpuSpatialRuntime struct */
  struct GpuSpatialRuntime* runtime;
  /** How many threads will concurrently call Probe method */
  uint32_t concurrency;
  /** Whether to compress the BVH structures to save memory */
  bool compress_bvh;
  /** Number of batches to pipeline for parsing and refinement; setting to 1 disables
   * pipelining */
  uint32_t pipeline_batches;
};

enum SedonaSpatialRelationPredicate {
  SedonaSpatialPredicateEquals = 0,
  SedonaSpatialPredicateDisjoint,
  SedonaSpatialPredicateTouches,
  SedonaSpatialPredicateContains,
  SedonaSpatialPredicateCovers,
  SedonaSpatialPredicateIntersects,
  SedonaSpatialPredicateWithin,
  SedonaSpatialPredicateCoveredBy
};

struct SedonaSpatialRefiner {
  int (*clear)(struct SedonaSpatialRefiner* self);

  int (*push_build)(struct SedonaSpatialRefiner* self,
                    const struct ArrowSchema* build_schema,
                    const struct ArrowArray* build_array);

  int (*finish_building)(struct SedonaSpatialRefiner* self);

  int (*refine_loaded)(struct SedonaSpatialRefiner* self,
                       const struct ArrowSchema* probe_schema,
                       const struct ArrowArray* probe_array,
                       enum SedonaSpatialRelationPredicate predicate,
                       uint32_t* build_indices, uint32_t* probe_indices,
                       uint32_t indices_size, uint32_t* new_indices_size);

  int (*refine)(struct SedonaSpatialRefiner* self, const struct ArrowSchema* schema1,
                const struct ArrowArray* array1, const struct ArrowSchema* schema2,
                const struct ArrowArray* array2,
                enum SedonaSpatialRelationPredicate predicate, uint32_t* indices1,
                uint32_t* indices2, uint32_t indices_size, uint32_t* new_indices_size);
  const char* (*get_last_error)(struct SedonaSpatialRefiner* self);
  void (*release)(struct SedonaSpatialRefiner* self);
  void* private_data;
};

int GpuSpatialRefinerCreate(struct SedonaSpatialRefiner* refiner,
                            const struct GpuSpatialRefinerConfig* config);
#ifdef __cplusplus
}
#endif
