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

#include "gpuspatial/geom/point.cuh"
#include "gpuspatial/utils/array_view.h"
#include "gpuspatial/utils/pinned_vector.h"

#include "gtest/gtest.h"
#include "rmm/cuda_stream_view.hpp"
#include "rmm/device_uvector.hpp"
#include "rmm/exec_policy.hpp"

#include "arrow/api.h"
#include "arrow/c/bridge.h"
#include "arrow/filesystem/api.h"
#include "arrow/record_batch.h"
#include "arrow/util/macros.h"
#include "parquet/arrow/reader.h"

#include <thrust/copy.h>

#include <filesystem>

#define ARROW_THROW_NOT_OK(status_expr)       \
  do {                                        \
    arrow::Status _s = (status_expr);         \
    if (!_s.ok()) {                           \
      throw std::runtime_error(_s.message()); \
    }                                         \
  } while (0)

namespace TestUtils {
using PointTypes =
    ::testing::Types<gpuspatial::Point<float, 2>, gpuspatial::Point<double, 2>>;
using PointIndexTypePairs =
    ::testing::Types<std::pair<gpuspatial::Point<float, 2>, uint32_t>,
                     std::pair<gpuspatial::Point<double, 2>, uint32_t>,
                     std::pair<gpuspatial::Point<float, 2>, uint64_t>,
                     std::pair<gpuspatial::Point<double, 2>, uint64_t>>;

std::string GetTestDataPath(const std::string& relative_path_to_file);
std::string GetTestShaderPath();

template <typename T>
gpuspatial::PinnedVector<T> ToVector(const rmm::cuda_stream_view& stream,
                                     const rmm::device_uvector<T>& d_vec) {
  gpuspatial::PinnedVector<T> vec(d_vec.size());

  thrust::copy(rmm::exec_policy_nosync(stream), d_vec.begin(), d_vec.end(), vec.begin());
  return vec;
}
template <typename T>
gpuspatial::PinnedVector<T> ToVector(const rmm::cuda_stream_view& stream,
                                     const gpuspatial::ArrayView<T>& arr) {
  gpuspatial::PinnedVector<T> vec(arr.size());

  thrust::copy(rmm::exec_policy_nosync(stream), arr.begin(), arr.end(), vec.begin());
  return vec;
}

// Function to convert a relative path string to an absolute path string
std::string GetCanonicalPath(const std::string& relative_path_str) {
  try {
    // 1. Create a path object from the relative string
    std::filesystem::path relative_path = relative_path_str;

    // 2. Resolve it against the current working directory (CWD)
    std::filesystem::path absolute_path = std::filesystem::absolute(relative_path);
    std::filesystem::path canonical_path = std::filesystem::canonical(absolute_path);

    // 3. Return the absolute path as a string
    return canonical_path.string();
  } catch (const std::filesystem::filesystem_error& e) {
    std::cerr << "Filesystem Error: " << e.what() << std::endl;
    return "";  // Return an empty string on error
  }
}

template <typename KeyType, typename ValueType>
void sort_vectors_by_index(std::vector<KeyType>& keys, std::vector<ValueType>& values) {
  // 1. Create an index vector {0, 1, 2, ...}
  std::vector<size_t> indices(keys.size());
  // Fills 'indices' with 0, 1, 2, ..., N-1
  std::iota(indices.begin(), indices.end(), 0);

  // 2. Sort the indices based on the values in the 'keys' vector
  // The lambda compares the key elements at two different indices
  std::sort(indices.begin(), indices.end(), [&keys, &values](size_t i, size_t j) {
    return keys[i] < keys[j] || keys[i] == keys[j] && values[i] < values[j];
  });

  // 3. Create new, sorted vectors
  std::vector<KeyType> sorted_keys;
  std::vector<ValueType> sorted_values;

  for (size_t i : indices) {
    sorted_keys.push_back(keys[i]);
    sorted_values.push_back(values[i]);
  }

  // Replace the original vectors with the sorted ones
  keys = std::move(sorted_keys);
  values = std::move(sorted_values);
}

}  // namespace TestUtils
