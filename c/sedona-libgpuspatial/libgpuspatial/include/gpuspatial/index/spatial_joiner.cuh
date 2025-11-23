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
#include "geoarrow/geoarrow_type.h"
#include "gpuspatial/geom/box.cuh"
#include "gpuspatial/geom/point.cuh"
#include "gpuspatial/index/detail/rt_engine.hpp"
#include "gpuspatial/index/geometry_grouper.hpp"
#include "gpuspatial/index/object_pool.hpp"
#include "gpuspatial/index/relate_engine.cuh"
#include "gpuspatial/index/streaming_joiner.hpp"
#include "gpuspatial/loader/device_geometries.cuh"
#include "gpuspatial/loader/parallel_wkb_loader.h"
#include "gpuspatial/utils/gpu_timer.hpp"
#include "gpuspatial/utils/queue.h"
#include "gpuspatial/utils/thread_pool.h"

#include "rmm/cuda_stream_pool.hpp"
#include "rmm/cuda_stream_view.hpp"
#include "rmm/device_uvector.hpp"

#include <fstream>
#include <thread>


// #define GPUSPATIAL_PROFILING
namespace gpuspatial {

class SpatialJoiner : public StreamingJoiner {
  // TODO: Assuming every thing is 2D in double for now
  using scalar_t = double;
  static constexpr int n_dim = 2;
  using index_t = uint32_t;  // type of the index to represent geometries
  // geometry types
  using point_t = Point<scalar_t, n_dim>;
  using multi_point_t = MultiPoint<point_t>;
  using line_string_t = LineString<point_t>;
  using multi_line_string_t = MultiLineString<point_t, index_t>;
  using polygon_t = Polygon<point_t, index_t>;
  using multi_polygon_t = MultiPolygon<point_t, index_t>;
  // geometry array types
  using point_array_t = PointArrayView<point_t, index_t>;
  using multi_point_array_t = MultiPointArrayView<point_t, index_t>;
  using line_string_array_t = LineStringArrayView<point_t, index_t>;
  using multi_line_string_array_t = MultiLineStringArrayView<point_t, index_t>;
  using polygon_array_t = PolygonArrayView<point_t, index_t>;
  using multi_polygon_array_t = MultiPolygonArrayView<point_t, index_t>;

  using dev_geometries_t = DeviceGeometries<point_t, index_t>;
  using box_t = Box<Point<float, n_dim>>;
  using loader_t = ParallelWkbLoader<point_t, index_t>;

 public:
  struct SpatialJoinerConfig : Config {
    const char* ptx_root;
    // Prefer fast build the BVH
    bool prefer_fast_build = false;
    // Compress the BVH to save memory
    bool compact = true;
    // Loader configurations
    // How many threads to use for parsing WKBs
    uint32_t parsing_threads = std::thread::hardware_concurrency();
    // How many threads are allowed to call PushStream concurrently
    uint32_t concurrency = 1;
    // number of points to represent an AABB when doing point-point queries
    uint32_t n_points_per_aabb = 8;
    // reserve a ratio of available memory for result sets
    float result_buffer_memory_reserve_ratio = 0.2;
    // the memory quota for relate engine compared to the available memory
    float relate_engine_memory_quota = 0.8;
    // this value determines RELATE_MAX_DEPTH
    size_t stack_size_bytes = 3 * 1024;
    SpatialJoinerConfig() : ptx_root(nullptr), prefer_fast_build(false), compact(false) {
      concurrency = std::thread::hardware_concurrency();
    }
  };

  struct SpatialJoinerContext : Context {
    rmm::cuda_stream_view cuda_stream;
    std::string shader_id;
    std::unique_ptr<loader_t> stream_loader;
    dev_geometries_t stream_geometries;
    std::unique_ptr<rmm::device_buffer> bvh_buffer;
    OptixTraversableHandle handle;
    std::vector<char> h_launch_params_buffer;
    std::unique_ptr<rmm::device_buffer> launch_params_buffer;
    // output
    Queue<thrust::pair<index_t, index_t>> results;
    int32_t array_index_offset;
#ifdef GPUSPATIAL_PROFILING
    GPUTimer timer;
    // counters
    double parse_ms = 0.0;
    double alloc_ms = 0.0;
    double filter_ms = 0.0;
    double refine_ms = 0.0;
    double copy_res_ms = 0.0;
#endif
  };

  SpatialJoiner() = default;

  ~SpatialJoiner() = default;

  void Init(const Config* config) override;

  void Clear() override;

  void PushBuild(const ArrowSchema* schema, const ArrowArray* array, int64_t offset,
                 int64_t length) override;

  void FinishBuilding() override;

  std::shared_ptr<Context> CreateContext() override { return ctx_pool_->take(); }

  void PushStream(Context* ctx, const ArrowSchema* schema, const ArrowArray* array,
                  int64_t offset, int64_t length, Predicate predicate,
                  std::vector<uint32_t>* build_indices,
                  std::vector<uint32_t>* stream_indices,
                  int32_t array_index_offset) override;

  // Internal method but has to be public for the CUDA kernel to access
  void handleBuildPointStreamPoint(SpatialJoinerContext* ctx, Predicate predicate,
                                   std::vector<uint32_t>* build_indices,
                                   std::vector<uint32_t>* stream_indices);

  void handleBuildBoxStreamPoint(SpatialJoinerContext* ctx, Predicate predicate,
                                 std::vector<uint32_t>* build_indices,
                                 std::vector<uint32_t>* stream_indices);

  void handleBuildPointStreamBox(SpatialJoinerContext* ctx, Predicate predicate,
                                 std::vector<uint32_t>* build_indices,
                                 std::vector<uint32_t>* stream_indices);

  void handleBuildBoxStreamBox(SpatialJoinerContext* ctx, Predicate predicate,
                               std::vector<uint32_t>* build_indices,
                               std::vector<uint32_t>* stream_indices);

  void filter(SpatialJoinerContext* ctx, uint32_t dim_x, bool swap_id = false);

  void refine(SpatialJoinerContext* ctx, Predicate predicate,
              std::vector<uint32_t>* build_indices,
              std::vector<uint32_t>* stream_indices);

 private:
  SpatialJoinerConfig config_;
  std::unique_ptr<rmm::cuda_stream_pool> stream_pool_;
  std::shared_ptr<ThreadPool> thread_pool_;
  details::RTEngine rt_engine_;
  std::unique_ptr<rmm::device_buffer> bvh_buffer_;
  std::unique_ptr<loader_t> build_loader_;

  DeviceGeometries<point_t, index_t> build_geometries_;
  // For grouping points with space-filing curve
  GeometryGrouper<point_t, index_t> geometry_grouper_;
  RelateEngine<point_t, index_t> relate_engine_;
  OptixTraversableHandle handle_;

  std::shared_ptr<ObjectPool<SpatialJoinerContext>> ctx_pool_;

  OptixTraversableHandle buildBVH(const rmm::cuda_stream_view& stream,
                                  const ArrayView<OptixAabb>& aabbs,
                                  std::unique_ptr<rmm::device_buffer>& buffer);

  void allocateResultBuffer(SpatialJoinerContext* ctx);

  void prepareLaunchParamsBoxQuery(SpatialJoinerContext* ctx, bool forward);
};

}  // namespace gpuspatial
