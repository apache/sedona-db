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
#include "gpuspatial/geom/box.hpp"
#include "gpuspatial/geom/point.hpp"
#include "gpuspatial/loader/device_geometries.hpp"
#include "gpuspatial/loader/parallel_wkb_loader.hpp"
#include "gpuspatial/refine/rt_spatial_refiner.hpp"
#include "gpuspatial/refine/spatial_refiner.hpp"
#include "gpuspatial/relate/relate_engine.cuh"
#include "gpuspatial/rt/rt_engine.hpp"
#include "gpuspatial/utils/gpu_timer.hpp"
#include "gpuspatial/utils/thread_pool.hpp"

#include "geoarrow/geoarrow_type.h"
#include "nanoarrow/nanoarrow.h"

#include "rmm/cuda_stream_pool.hpp"
#include "rmm/cuda_stream_view.hpp"

namespace gpuspatial {

class RTSpatialRefiner : public SpatialRefiner {
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

  static_assert(sizeof(Box<Point<float, 2>>) == sizeof(box_t),
                "Box<Point<float, 2>> size mismatch!");

 public:
  struct IndicesMap {
    // Sorted unique original indices
    std::vector<uint32_t> h_uniq_indices;
    rmm::device_uvector<uint32_t> d_uniq_indices{0, rmm::cuda_stream_default};
    // Mapping from original indices to consecutive zero-based indices
    rmm::device_uvector<uint32_t> d_reordered_indices{0, rmm::cuda_stream_default};
  };
  struct SpatialRefinerContext {
    rmm::cuda_stream_view cuda_stream;
#ifdef GPUSPATIAL_PROFILING
    GPUTimer timer;
    // counters
    double parse_ms = 0.0;
    double alloc_ms = 0.0;
    double refine_ms = 0.0;
    double copy_res_ms = 0.0;
#endif
  };

  RTSpatialRefiner() = default;

  RTSpatialRefiner(const RTSpatialRefinerConfig& config);

  ~RTSpatialRefiner() = default;

  void Clear() override;

  void PushBuild(const ArrowArrayView* build_array) override;

  void FinishBuilding() override;

  uint32_t Refine(const ArrowArrayView* probe_array, Predicate predicate,
                  uint32_t* build_indices, uint32_t* probe_indices,
                  uint32_t len) override;

  uint32_t RefinePipelined(const ArrowArrayView* probe_array, Predicate predicate,
                           uint32_t* build_indices, uint32_t* probe_indices,
                           uint32_t len);

 private:
  RTSpatialRefinerConfig config_;
  std::unique_ptr<rmm::cuda_stream_pool> stream_pool_;
  std::shared_ptr<ThreadPool> thread_pool_;
  std::unique_ptr<ParallelWkbLoader<point_t, index_t>> wkb_loader_;
  dev_geometries_t build_geometries_;

  template <typename INDEX_IT>
  void buildIndicesMap(rmm::cuda_stream_view stream, INDEX_IT index_begin,
                       INDEX_IT index_end, IndicesMap& indices_map) const;
};

}  // namespace gpuspatial
