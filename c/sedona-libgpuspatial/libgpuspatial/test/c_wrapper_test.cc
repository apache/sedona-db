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

#include "array_stream.hpp"
#include "test_common.hpp"

#include "gpuspatial/gpuspatial_c.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "nanoarrow/nanoarrow.hpp"

#include <random>
#include <vector>

TEST(RuntimeTest, InitializeRuntime) {
  GpuSpatialRuntime runtime;
  GpuSpatialRuntimeCreate(&runtime);
  GpuSpatialRuntimeConfig config;

  std::string ptx_root = TestUtils::GetTestShaderPath();
  config.ptx_root = ptx_root.c_str();
  config.device_id = 0;
  config.use_cuda_memory_pool = false;
  ASSERT_EQ(runtime.init(&runtime, &config), 0);

  runtime.release(&runtime);
}

TEST(RuntimeTest, ErrorTest) {
  GpuSpatialRuntime runtime;
  GpuSpatialRuntimeCreate(&runtime);
  GpuSpatialRuntimeConfig runtime_config;

  runtime_config.ptx_root = "/invalid/path/to/ptx";
  runtime_config.device_id = 0;
  runtime_config.use_cuda_memory_pool = false;

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
  runtime_config.use_cuda_memory_pool = true;
  runtime_config.cuda_memory_pool_init_precent = 10;
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
  runtime_config.use_cuda_memory_pool = true;
  runtime_config.cuda_memory_pool_init_precent = 10;
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
    runtime_config.use_cuda_memory_pool = true;
    runtime_config.cuda_memory_pool_init_precent = 10;
    ASSERT_EQ(runtime_.init(&runtime_, &runtime_config), 0);

    GpuSpatialIndexConfig index_config;

    index_config.runtime = &runtime_;
    index_config.concurrency = 1;

    ASSERT_EQ(GpuSpatialIndexFloat2DCreate(&index_, &index_config), 0);

    GpuSpatialRefinerConfig refiner_config;

    refiner_config.runtime = &runtime_;
    refiner_config.concurrency = 1;
    refiner_config.compress_bvh = false;
    refiner_config.pipeline_batches = 1;

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
  // Define types matching the CWrapper expectation (float for GPU)
  using coord_t = float;
  using fpoint_t = gpuspatial::Point<coord_t, 2>;
  using box_t = gpuspatial::Box<fpoint_t>;

  // 1. Load Data using ReadParquet
  auto poly_path = TestUtils::GetTestDataPath("synthetic_pip/polygons.parquet");
  auto point_path = TestUtils::GetTestDataPath("synthetic_pip/points.parquet");

  // ReadParquet loads the file into batches (std::vector<std::shared_ptr<arrow::Array>>)
  // Assuming batch_size=1000 or similar default inside ReadParquet
  auto poly_arrays = gpuspatial::ReadParquet(poly_path);
  auto point_arrays = gpuspatial::ReadParquet(point_path);

  // 2. Setup GEOS C++ Objects
  auto factory = geos::geom::GeometryFactory::create();
  geos::io::WKBReader wkb_reader(*factory);

  // Iterate over batches (replacing the stream loop)
  size_t num_batches = std::min(poly_arrays.size(), point_arrays.size());

  for (size_t i = 0; i < num_batches; i++) {
    auto build_arrow_arr = poly_arrays[i];
    auto probe_arrow_arr = point_arrays[i];

    // Export to C-style ArrowArrays for the CWrapper (SUT)
    nanoarrow::UniqueArray build_array, probe_array;
    nanoarrow::UniqueSchema build_schema, probe_schema;

    ARROW_THROW_NOT_OK(
        arrow::ExportArray(*build_arrow_arr, build_array.get(), build_schema.get()));
    ARROW_THROW_NOT_OK(
        arrow::ExportArray(*probe_arrow_arr, probe_array.get(), probe_schema.get()));

    // --- Build Phase ---

    // Boxes for the GPU Index (SUT)
    std::vector<box_t> rects;

    // View for parsing WKB
    nanoarrow::UniqueArrayView build_view;
    ArrowError error;

    ASSERT_EQ(ArrowArrayViewInitFromSchema(build_view.get(), build_schema.get(), &error),
              NANOARROW_OK)
        << error.message;
    ASSERT_EQ(ArrowArrayViewSetArray(build_view.get(), build_array.get(), &error),
              NANOARROW_OK)
        << error.message;

    for (int64_t j = 0; j < build_array->length; j++) {
      // Parse WKB
      ArrowStringView wkb = ArrowArrayViewGetStringUnsafe(build_view.get(), j);
      auto geom = wkb_reader.read(reinterpret_cast<const unsigned char*>(wkb.data),
                                  wkb.size_bytes);

      // Extract Envelope for GPU
      const auto* env = geom->getEnvelopeInternal();
      double xmin = 0, ymin = 0, xmax = -1, ymax = -1;
      if (!env->isNull()) {
        xmin = env->getMinX();
        ymin = env->getMinY();
        xmax = env->getMaxX();
        ymax = env->getMaxY();
      }

      // Add to GPU Build Data
      rects.emplace_back(fpoint_t((float)xmin, (float)ymin),
                         fpoint_t((float)xmax, (float)ymax));
    }

    // Initialize SUT (CWrapper Index)
    index_.clear(&index_);
    ASSERT_EQ(index_.push_build(&index_, (float*)rects.data(), rects.size()), 0);
    ASSERT_EQ(index_.finish_building(&index_), 0);

    // --- Probe Phase ---
    std::vector<box_t> queries;

    nanoarrow::UniqueArrayView probe_view;

    ASSERT_EQ(ArrowArrayViewInitFromSchema(probe_view.get(), probe_schema.get(), &error),
              NANOARROW_OK)
        << error.message;
    ASSERT_EQ(ArrowArrayViewSetArray(probe_view.get(), probe_array.get(), &error),
              NANOARROW_OK)
        << error.message;

    for (int64_t j = 0; j < probe_array->length; j++) {
      ArrowStringView wkb = ArrowArrayViewGetStringUnsafe(probe_view.get(), j);
      auto geom = wkb_reader.read(reinterpret_cast<const unsigned char*>(wkb.data),
                                  wkb.size_bytes);

      const auto* env = geom->getEnvelopeInternal();
      double xmin = 0, ymin = 0, xmax = -1, ymax = -1;
      if (!env->isNull()) {
        xmin = env->getMinX();
        ymin = env->getMinY();
        xmax = env->getMaxX();
        ymax = env->getMaxY();
      }

      queries.emplace_back(fpoint_t((float)xmin, (float)ymin),
                           fpoint_t((float)xmax, (float)ymax));
    }

    // Run SUT Probe
    SedonaSpatialIndexContext idx_ctx;
    index_.create_context(&idx_ctx);
    index_.probe(&index_, &idx_ctx, (float*)queries.data(), queries.size());

    // Retrieve SUT Results
    uint32_t* build_indices_ptr;
    uint32_t* probe_indices_ptr;
    uint32_t build_indices_length;
    uint32_t probe_indices_length;

    index_.get_build_indices_buffer(&idx_ctx, &build_indices_ptr, &build_indices_length);
    index_.get_probe_indices_buffer(&idx_ctx, &probe_indices_ptr, &probe_indices_length);

    refiner_.clear(&refiner_);
    ASSERT_EQ(refiner_.push_build(&refiner_, build_schema.get(), build_array.get()), 0);
    ASSERT_EQ(refiner_.finish_building(&refiner_), 0);

    uint32_t new_len;
    ASSERT_EQ(refiner_.refine(
                  &refiner_, probe_schema.get(), probe_array.get(),
                  SedonaSpatialRelationPredicate::SedonaSpatialPredicateContains,
                  build_indices_ptr, probe_indices_ptr, build_indices_length, &new_len),
              0);

    std::vector<uint32_t> sut_build_indices(build_indices_ptr,
                                            build_indices_ptr + new_len);
    std::vector<uint32_t> sut_stream_indices(probe_indices_ptr,
                                             probe_indices_ptr + new_len);

    index_.destroy_context(&idx_ctx);

    std::vector<uint32_t> ref_build_indices;
    std::vector<uint32_t> ref_stream_indices;

    // Wrap single arrays in vectors to match signature
    std::vector<ArrowArray*> build_inputs = {build_array.get()};
    std::vector<ArrowArray*> probe_inputs = {probe_array.get()};

    TestUtils::ComputeGeosJoin(build_schema.get(), build_inputs, probe_schema.get(),
                               probe_inputs, gpuspatial::Predicate::kContains,
                               ref_build_indices, ref_stream_indices);

    // Compare Results
    ASSERT_EQ(ref_build_indices.size(), sut_build_indices.size());
    ASSERT_EQ(ref_stream_indices.size(), sut_stream_indices.size());

    TestUtils::sort_vectors_by_index(ref_build_indices, ref_stream_indices);
    TestUtils::sort_vectors_by_index(sut_build_indices, sut_stream_indices);

    for (size_t j = 0; j < sut_build_indices.size(); j++) {
      ASSERT_EQ(ref_build_indices[j], sut_build_indices[j]);
      ASSERT_EQ(ref_stream_indices[j], sut_stream_indices[j]);
    }
  }
}
