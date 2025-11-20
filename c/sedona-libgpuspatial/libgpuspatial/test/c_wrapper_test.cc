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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>
#include "array_stream.hpp"
#include "nanoarrow/nanoarrow.hpp"

#include "../include/gpuspatial/gpuspatial_c.h"
namespace TestUtils {
std::string GetTestDataPath(const std::string& relative_path_to_file);
}

class CWrapperTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize the GpuSpatialJoiner
    GpuSpatialJoinerCreate(&joiner_);
    struct GpuSpatialJoinerConfig config_;
    std::string ptx_root = TestUtils::GetTestDataPath("shaders_ptx");

    // Set up the configuration
    config_.concurrency = 2;  // Example concurrency level
    config_.ptx_root = ptx_root.c_str();

    ASSERT_EQ(joiner_.init(&joiner_, &config_), 0);
    // Initialize the context
  }

  void TearDown() override {
    // Clean up
    joiner_.release(&joiner_);
  }

  struct GpuSpatialJoiner joiner_;
};

TEST_F(CWrapperTest, InitializeJoiner) {
  // Test if the joiner initializes correctly
  struct GpuSpatialJoinerContext context_;
  joiner_.create_context(&joiner_, &context_);

  auto poly_path = TestUtils::GetTestDataPath("../test_data/test_polygons.arrows");
  auto point_path = TestUtils::GetTestDataPath("../test_data/test_points.arrows");
  nanoarrow::UniqueArrayStream poly_stream, point_stream;

  gpuspatial::ArrayStreamFromIpc(poly_path, "geometry", poly_stream.get());
  gpuspatial::ArrayStreamFromIpc(point_path, "geometry", point_stream.get());

  nanoarrow::UniqueSchema build_schema, stream_schema;
  nanoarrow::UniqueArray build_array, stream_array;
  ArrowError error;
  ArrowErrorSet(&error, "");

  int n_row_groups = 100;

  for (int i = 0; i < n_row_groups; i++) {
    ASSERT_EQ(ArrowArrayStreamGetNext(poly_stream.get(), build_array.get(), &error),
              NANOARROW_OK);
    ASSERT_EQ(ArrowArrayStreamGetSchema(poly_stream.get(), build_schema.get(), &error),
              NANOARROW_OK);

    ASSERT_EQ(ArrowArrayStreamGetNext(point_stream.get(), stream_array.get(), &error),
              NANOARROW_OK);
    ASSERT_EQ(ArrowArrayStreamGetSchema(point_stream.get(), stream_schema.get(), &error),
              NANOARROW_OK);

    joiner_.push_build(&joiner_, build_schema.get(), build_array.get(), 0,
                       build_array->length);
    joiner_.finish_building(&joiner_);

    joiner_.push_stream(&joiner_, &context_, stream_schema.get(), stream_array.get(), 0,
                        stream_array->length, GpuSpatialPredicateContains, 0);

    void* build_indices_ptr;
    void* stream_indices_ptr;
    uint32_t build_indices_length;
    uint32_t stream_indices_length;

    joiner_.get_build_indices_buffer(&context_, (void**)&build_indices_ptr,
                                     &build_indices_length);
    joiner_.get_stream_indices_buffer(&context_, (void**)&stream_indices_ptr,
                                      &stream_indices_length);
  }

  joiner_.destroy_context(&context_);
}
