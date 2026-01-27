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

#include "test_common.hpp"

#include "gpuspatial/gpuspatial_c.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>
#include "array_stream.hpp"
#include "geoarrow_geos/geoarrow_geos.hpp"
#include "nanoarrow/nanoarrow.hpp"

TEST(RuntimeTest, InitializeRuntime) {
  GpuSpatialRuntime runtime;
  GpuSpatialRuntimeCreate(&runtime);
  GpuSpatialRuntimeConfig config;

  std::string ptx_root = TestUtils::GetTestShaderPath();
  config.ptx_root = ptx_root.c_str();
  config.device_id = 0;
  config.cuda_init_memory_pool_ratio = 0;
  ASSERT_EQ(runtime.init(&runtime, &config), 0);

  runtime.release(&runtime);
}

TEST(RuntimeTest, ErrorTest) {
  GpuSpatialRuntime runtime;
  GpuSpatialRuntimeCreate(&runtime);
  GpuSpatialRuntimeConfig runtime_config;

  runtime_config.ptx_root = "/invalid/path/to/ptx";
  runtime_config.device_id = 0;
  runtime_config.cuda_init_memory_pool_ratio = 0;

  EXPECT_NE(runtime.init(&runtime, &runtime_config), 0);

  const char* raw_error = runtime.get_last_error(&runtime);
  printf("Error received: %s\n", raw_error);

  std::string error_msg(raw_error);

  EXPECT_NE(error_msg.find("No such file or directory"), std::string::npos)
      << "Error message was corrupted or incorrect. Got: " << error_msg;

  runtime.release(&runtime);
}

TEST(SpatialIndexTest, InitializeIndex) {
  GpuSpatialRuntime runtime;
  GpuSpatialRuntimeCreate(&runtime);
  GpuSpatialRuntimeConfig runtime_config;

  std::string ptx_root = TestUtils::GetTestShaderPath();
  runtime_config.ptx_root = ptx_root.c_str();
  runtime_config.device_id = 0;
  runtime_config.cuda_init_memory_pool_ratio = 0.1;
  ASSERT_EQ(runtime.init(&runtime, &runtime_config), 0);

  SedonaFloatIndex2D index;
  GpuSpatialIndexConfig index_config;

  index_config.runtime = &runtime;
  index_config.concurrency = 1;

  ASSERT_EQ(GpuSpatialIndexFloat2DCreate(&index, &index_config), 0);

  index.release(&index);
  runtime.release(&runtime);
}

TEST(RefinerTest, InitializeRefiner) {
  GpuSpatialRuntime runtime;
  GpuSpatialRuntimeCreate(&runtime);
  GpuSpatialRuntimeConfig runtime_config;

  std::string ptx_root = TestUtils::GetTestShaderPath();
  runtime_config.ptx_root = ptx_root.c_str();
  runtime_config.device_id = 0;
  runtime_config.cuda_init_memory_pool_ratio = 0.1;
  ASSERT_EQ(runtime.init(&runtime, &runtime_config), 0);

  SedonaSpatialRefiner refiner;
  GpuSpatialRefinerConfig refiner_config;

  refiner_config.runtime = &runtime;
  refiner_config.concurrency = 1;

  ASSERT_EQ(GpuSpatialRefinerCreate(&refiner, &refiner_config), 0);

  refiner.release(&refiner);
  runtime.release(&runtime);
}

class CWrapperTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::string ptx_root = TestUtils::GetTestShaderPath();

    GpuSpatialRuntimeCreate(&runtime_);
    GpuSpatialRuntimeConfig runtime_config;

    runtime_config.ptx_root = ptx_root.c_str();
    runtime_config.device_id = 0;
    runtime_config.cuda_init_memory_pool_ratio = 0.1;
    ASSERT_EQ(runtime_.init(&runtime_, &runtime_config), 0);

    GpuSpatialIndexConfig index_config;

    index_config.runtime = &runtime_;
    index_config.concurrency = 1;

    ASSERT_EQ(GpuSpatialIndexFloat2DCreate(&index_, &index_config), 0);

    GpuSpatialRefinerConfig refiner_config;

    refiner_config.runtime = &runtime_;
    refiner_config.concurrency = 1;

    ASSERT_EQ(GpuSpatialRefinerCreate(&refiner_, &refiner_config), 0);
  }

  void TearDown() override {
    refiner_.release(&refiner_);
    index_.release(&index_);
    runtime_.release(&runtime_);
  }
  GpuSpatialRuntime runtime_;
  SedonaFloatIndex2D index_;
  SedonaSpatialRefiner refiner_;
};

TEST_F(CWrapperTest, InitializeJoiner) {
  using fpoint_t = gpuspatial::Point<float, 2>;
  using box_t = gpuspatial::Box<fpoint_t>;
  // Test if the joiner initializes correctly

  auto poly_path = TestUtils::GetTestDataPath("arrowipc/test_polygons.arrows");
  auto point_path = TestUtils::GetTestDataPath("arrowipc/test_points.arrows");
  nanoarrow::UniqueArrayStream poly_stream, point_stream;

  gpuspatial::ArrayStreamFromIpc(poly_path, "geometry", poly_stream.get());
  gpuspatial::ArrayStreamFromIpc(point_path, "geometry", point_stream.get());

  nanoarrow::UniqueSchema build_schema, stream_schema;
  nanoarrow::UniqueArray build_array, stream_array;
  ArrowError error;
  ArrowErrorSet(&error, "");

  int n_row_groups = 100;

  geoarrow::geos::ArrayReader reader;

  for (int i = 0; i < n_row_groups; i++) {
    ASSERT_EQ(ArrowArrayStreamGetNext(poly_stream.get(), build_array.get(), &error),
              NANOARROW_OK);
    ASSERT_EQ(ArrowArrayStreamGetSchema(poly_stream.get(), build_schema.get(), &error),
              NANOARROW_OK);

    ASSERT_EQ(ArrowArrayStreamGetNext(point_stream.get(), stream_array.get(), &error),
              NANOARROW_OK);
    ASSERT_EQ(ArrowArrayStreamGetSchema(point_stream.get(), stream_schema.get(), &error),
              NANOARROW_OK);

    class GEOSCppHandle {
     public:
      GEOSContextHandle_t handle;

      GEOSCppHandle() { handle = GEOS_init_r(); }

      ~GEOSCppHandle() { GEOS_finish_r(handle); }
    };
    GEOSCppHandle handle;

    reader.InitFromEncoding(handle.handle, GEOARROW_GEOS_ENCODING_WKB);

    geoarrow::geos::GeometryVector geom_build(handle.handle);

    geom_build.resize(build_array->length);
    size_t n_build;

    ASSERT_EQ(reader.Read(build_array.get(), 0, build_array->length,
                          geom_build.mutable_data(), &n_build),
              GEOARROW_GEOS_OK);
    auto* tree = GEOSSTRtree_create_r(handle.handle, 10);
    std::vector<box_t> rects;

    for (size_t build_idx = 0; build_idx < build_array->length; build_idx++) {
      auto* geom = geom_build.borrow(build_idx);
      auto* box = GEOSEnvelope_r(handle.handle, geom);

      double xmin, ymin, xmax, ymax;
      int result = GEOSGeom_getExtent_r(handle.handle, box, &xmin, &ymin, &xmax, &ymax);
      ASSERT_EQ(result, 1);
      box_t bbox(fpoint_t((float)xmin, (float)ymin), fpoint_t((float)xmax, (float)ymax));

      rects.push_back(bbox);

      GEOSGeom_setUserData_r(handle.handle, (GEOSGeometry*)geom, (void*)build_idx);
      GEOSSTRtree_insert_r(handle.handle, tree, box, (void*)geom);
      GEOSGeom_destroy_r(handle.handle, box);
    }

    index_.clear(&index_);
    ASSERT_EQ(index_.push_build(&index_, (float*)rects.data(), rects.size()), 0);
    ASSERT_EQ(index_.finish_building(&index_), 0);

    geoarrow::geos::GeometryVector geom_stream(handle.handle);
    size_t n_stream;
    geom_stream.resize(stream_array->length);

    ASSERT_EQ(reader.Read(stream_array.get(), 0, stream_array->length,
                          geom_stream.mutable_data(), &n_stream),
              GEOARROW_GEOS_OK);

    std::vector<box_t> queries;

    for (size_t stream_idx = 0; stream_idx < stream_array->length; stream_idx++) {
      auto* geom = geom_stream.borrow(stream_idx);
      double xmin, ymin, xmax, ymax;
      int result = GEOSGeom_getExtent_r(handle.handle, geom, &xmin, &ymin, &xmax, &ymax);
      ASSERT_EQ(result, 1);
      box_t bbox(fpoint_t((float)xmin, (float)ymin), fpoint_t((float)xmax, (float)ymax));
      queries.push_back(bbox);
    }

    SedonaSpatialIndexContext idx_ctx;
    index_.create_context(&idx_ctx);

    index_.probe(&index_, &idx_ctx, (float*)queries.data(), queries.size());

    uint32_t* build_indices_ptr;
    uint32_t* probe_indices_ptr;
    uint32_t build_indices_length;
    uint32_t probe_indices_length;

    index_.get_build_indices_buffer(&idx_ctx, &build_indices_ptr, &build_indices_length);
    index_.get_probe_indices_buffer(&idx_ctx, &probe_indices_ptr, &probe_indices_length);

    uint32_t new_len;
    ASSERT_EQ(
        refiner_.refine(&refiner_, build_schema.get(), build_array.get(),
                        stream_schema.get(), stream_array.get(),
                        SedonaSpatialRelationPredicate::SedonaSpatialPredicateContains,
                        (uint32_t*)build_indices_ptr, (uint32_t*)probe_indices_ptr,
                        build_indices_length, &new_len),
        0);

    std::vector<uint32_t> build_indices((uint32_t*)build_indices_ptr,
                                        (uint32_t*)build_indices_ptr + new_len);
    std::vector<uint32_t> probe_indices((uint32_t*)probe_indices_ptr,
                                        (uint32_t*)probe_indices_ptr + new_len);

    struct Payload {
      GEOSContextHandle_t handle;
      const GEOSGeometry* geom;
      std::vector<uint32_t> build_indices;
      std::vector<uint32_t> stream_indices;
      SedonaSpatialRelationPredicate predicate;
    };

    Payload payload;
    payload.predicate = SedonaSpatialRelationPredicate::SedonaSpatialPredicateContains;
    payload.handle = handle.handle;

    for (size_t offset = 0; offset < n_stream; offset++) {
      auto* geom = geom_stream.borrow(offset);
      GEOSGeom_setUserData_r(handle.handle, (GEOSGeometry*)geom, (void*)offset);
      payload.geom = geom;

      GEOSSTRtree_query_r(
          handle.handle, tree, geom,
          [](void* item, void* data) {
            auto* geom_build = (GEOSGeometry*)item;
            auto* payload = (Payload*)data;
            auto* geom_stream = payload->geom;

            if (GEOSContains_r(payload->handle, geom_build, geom_stream) == 1) {
              auto build_id = (size_t)GEOSGeom_getUserData_r(payload->handle, geom_build);
              auto stream_id =
                  (size_t)GEOSGeom_getUserData_r(payload->handle, geom_stream);
              payload->build_indices.push_back(build_id);
              payload->stream_indices.push_back(stream_id);
            }
          },
          (void*)&payload);
    }

    ASSERT_EQ(payload.build_indices.size(), build_indices.size());
    ASSERT_EQ(payload.stream_indices.size(), probe_indices.size());
    TestUtils::sort_vectors_by_index(payload.build_indices, payload.stream_indices);
    TestUtils::sort_vectors_by_index(build_indices, probe_indices);
    for (size_t j = 0; j < build_indices.size(); j++) {
      ASSERT_EQ(payload.build_indices[j], build_indices[j]);
      ASSERT_EQ(payload.stream_indices[j], probe_indices[j]);
    }
    index_.destroy_context(&idx_ctx);
  }
}
