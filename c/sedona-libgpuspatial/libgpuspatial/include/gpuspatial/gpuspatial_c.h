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
  /** Whether to use CUDA memory pool for allocations */
  bool use_cuda_memory_pool;
  /** Ratio of initial memory pool size to total GPU memory, between 0 and 100 */
  int cuda_memory_pool_init_precent;
};

/** Opaque runtime for GPU spatial operations
 * Each process should have exactly one instance of GpuSpatialRuntime
 */
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
  const struct GpuSpatialRuntime* runtime;
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
   * floats: [min_x, min_y, max_x, max_y].
   * Points can also be indexed by providing degenerated rectangles [x, y, x, y].
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
   * probe will be stored in the context. The callback function will be called for each
   * batch of results, with the build and probe indices of the candidate pairs in the
   * batch. The user_data pointer will be passed to the callback
   *
   * @return 0 on success, non-zero on failure
   */
  int (*probe)(struct SedonaFloatIndex2D* self, struct SedonaSpatialIndexContext* context,
               const float* buf, uint32_t n_rects,
               void (*callback)(const uint32_t* build_indices,
                                const uint32_t* probe_indices, uint32_t length,
                                void* user_data),
               void* user_data);
  /** Get the last error message from either the index
   *
   * @return A pointer to the error message string
   */
  const char* (*get_last_error)(struct SedonaFloatIndex2D* self);
  /** Get the last error message from the context
   *
   * @return A pointer to the error message string
   */
  const char* (*context_get_last_error)(struct SedonaSpatialIndexContext* context);
  /** Release the spatial index and free all resources */
  void (*release)(struct SedonaFloatIndex2D* self);
  void* private_data;
};

/** Create an instance of GpuSpatialIndex for 2D float rectangles/points
 *  @return 0 on success, non-zero on failure
 */
int GpuSpatialIndexFloat2DCreate(struct SedonaFloatIndex2D* index,
                                 const struct GpuSpatialIndexConfig* config);

struct GpuSpatialRefinerConfig {
  /** Pointer to an initialized GpuSpatialRuntime struct */
  const struct GpuSpatialRuntime* runtime;
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

/** An opaque spatial refiner that can refine candidate pairs of geometries */
struct SedonaSpatialRefiner {
  /** Clear all built geometries from the refiner */
  int (*clear)(struct SedonaSpatialRefiner* self);

  int (*init_schema)(struct SedonaSpatialRefiner* self,
                     const struct ArrowSchema* build_schema,
                     const struct ArrowSchema* probe_schema);

  /** Push geometries for building the spatial refiner
   *
   * @param build_array The Arrow array of the build geometries
   * @return 0 on success, non-zero on failure
   */
  int (*push_build)(struct SedonaSpatialRefiner* self,
                    const struct ArrowArray* build_array);
  /**
   * Finish building the spatial refiner after all geometries have been pushed
   *
   * @return 0 on success, non-zero on failure
   */
  int (*finish_building)(struct SedonaSpatialRefiner* self);

  /**
   * Refine candidate pairs of geometries
   *
   * @param probe_array The Arrow array of the probe geometries
   * @param predicate The spatial relation predicate to evaluate
   * @param build_indices An array of build-side indices corresponding to candidate pairs.
   * This is a global index from 0 to N-1, where N is the total number of build geometries
   * pushed.
   * @param probe_indices An array of probe-side indices corresponding to candidate pairs.
   * This is a local index from 0 to M - 1, where M is the number of geometries in the
   * probe_array.
   * @param indices_size The number of candidate pairs
   * @param new_indices_size Output parameter to store the number of refined pairs
   * @return 0 on success, non-zero on failure
   */
  int (*refine)(struct SedonaSpatialRefiner* self, const struct ArrowArray* probe_array,
                enum SedonaSpatialRelationPredicate predicate, uint32_t* build_indices,
                uint32_t* probe_indices, uint32_t indices_size,
                uint32_t* new_indices_size);

  /** Get the last error message
   *
   * @return A pointer to the error message string
   */
  const char* (*get_last_error)(struct SedonaSpatialRefiner* self);

  /** Release the spatial refiner and free all resources */
  void (*release)(struct SedonaSpatialRefiner* self);
  void* private_data;
};

/** Create an instance of GpuSpatialRefiner
 * @return 0 on success, non-zero on failure
 */
int GpuSpatialRefinerCreate(struct SedonaSpatialRefiner* refiner,
                            const struct GpuSpatialRefinerConfig* config);
#ifdef __cplusplus
}
#endif
