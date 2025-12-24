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

struct GpuSpatialRTEngineConfig {
  /** Path to PTX files */
  const char* ptx_root;
  /** Device ID to use, 0 is the first GPU */
  int device_id;
};

struct GpuSpatialRTEngine {
  /** Initialize the ray-tracing engine (OptiX) with the given configuration
   * @return 0 on success, non-zero on failure
   */
  int (*init)(struct GpuSpatialRTEngine* self, struct GpuSpatialRTEngineConfig* config);
  void (*release)(struct GpuSpatialRTEngine* self);
  void* private_data;
  const char* last_error;
};

/** Create an instance of GpuSpatialRTEngine */
void GpuSpatialRTEngineCreate(struct GpuSpatialRTEngine* instance);

struct GpuSpatialIndexConfig {
  /** Pointer to an initialized GpuSpatialRTEngine struct */
  struct GpuSpatialRTEngine* rt_engine;
  /** How many threads will concurrently call Probe method */
  uint32_t concurrency;
  /** Device ID to use, 0 is the first GPU */
  int device_id;
};

struct GpuSpatialIndexContext {
  const char* last_error;  // Pointer to std::string to store last error message
  void* build_indices;     // Pointer to std::vector<uint32_t> to store results
  void* probe_indices;
};

struct GpuSpatialIndexFloat2D {
  /** Initialize the spatial index with the given configuration
   *
   * @return 0 on success, non-zero on failure
   */
  int (*init)(struct GpuSpatialIndexFloat2D* self, struct GpuSpatialIndexConfig* config);
  /** Clear the spatial index, removing all built data */
  void (*clear)(struct GpuSpatialIndexFloat2D* self);
  /** Create a new context for concurrent probing */
  void (*create_context)(struct GpuSpatialIndexFloat2D* self,
                         struct GpuSpatialIndexContext* context);
  /** Destroy a previously created context */
  void (*destroy_context)(struct GpuSpatialIndexContext* context);
  /** Push rectangles for building the spatial index, each rectangle is represented by 4
   * floats: [min_x, min_y, max_x, max_y] Points can also be indexed by providing [x, y,
   * x, y] but points and rectangles cannot be mixed
   *
   * @return 0 on success, non-zero on failure
   */
  int (*push_build)(struct GpuSpatialIndexFloat2D* self, const float* buf,
                    uint32_t n_rects);
  /**
   * Finish building the spatial index after all rectangles have been pushed
   *
   * @return 0 on success, non-zero on failure
   */
  int (*finish_building)(struct GpuSpatialIndexFloat2D* self);
  /**
   * Probe the spatial index with the given rectangles, each rectangle is represented by 4
   * floats: [min_x, min_y, max_x, max_y] Points can also be probed by providing [x, y, x,
   * y] but points and rectangles cannot be mixed in one Probe call. The results of the
   * probe will be stored in the context.
   *
   * @return 0 on success, non-zero on failure
   */
  int (*probe)(struct GpuSpatialIndexFloat2D* self,
               struct GpuSpatialIndexContext* context, const float* buf,
               uint32_t n_rects);
  /** Get the build indices buffer from the context
   *
   * @return A pointer to the buffer and its length
   */
  void (*get_build_indices_buffer)(struct GpuSpatialIndexContext* context,
                                   void** build_indices, uint32_t* build_indices_length);
  /** Get the probe indices buffer from the context
   *
   * @return A pointer to the buffer and its length
   */
  void (*get_probe_indices_buffer)(struct GpuSpatialIndexContext* context,
                                   void** probe_indices, uint32_t* probe_indices_length);
  /** Release the spatial index and free all resources */
  void (*release)(struct GpuSpatialIndexFloat2D* self);
  void* private_data;
  const char* last_error;
};

void GpuSpatialIndexFloat2DCreate(struct GpuSpatialIndexFloat2D* index);

struct GpuSpatialRefinerConfig {
  /** Pointer to an initialized GpuSpatialRTEngine struct */
  struct GpuSpatialRTEngine* rt_engine;
  /** How many threads will concurrently call Probe method */
  uint32_t concurrency;
  /** Device ID to use, 0 is the first GPU */
  int device_id;
};

enum GpuSpatialRelationPredicate {
  GpuSpatialPredicateEquals = 0,
  GpuSpatialPredicateDisjoint,
  GpuSpatialPredicateTouches,
  GpuSpatialPredicateContains,
  GpuSpatialPredicateCovers,
  GpuSpatialPredicateIntersects,
  GpuSpatialPredicateWithin,
  GpuSpatialPredicateCoveredBy
};

struct GpuSpatialRefiner {
  int (*init)(struct GpuSpatialRefiner* self, struct GpuSpatialRefinerConfig* config);

  int (*load_build_array)(struct GpuSpatialRefiner* self,
                          const struct ArrowSchema* schema1,
                          const struct ArrowArray* array1);

  int (*refine_loaded)(struct GpuSpatialRefiner* self,
                       const struct ArrowSchema* probe_schema,
                       const struct ArrowArray* probe_array,
                       enum GpuSpatialRelationPredicate predicate,
                       uint32_t* build_indices, uint32_t* probe_indices,
                       uint32_t indices_size, uint32_t* new_indices_size);

  int (*refine)(struct GpuSpatialRefiner* self, const struct ArrowSchema* schema1,
                const struct ArrowArray* array1, const struct ArrowSchema* schema2,
                const struct ArrowArray* array2,
                enum GpuSpatialRelationPredicate predicate, uint32_t* indices1,
                uint32_t* indices2, uint32_t indices_size, uint32_t* new_indices_size);
  void (*release)(struct GpuSpatialRefiner* self);
  void* private_data;
  const char* last_error;
};

void GpuSpatialRefinerCreate(struct GpuSpatialRefiner* refiner);
#ifdef __cplusplus
}
#endif
