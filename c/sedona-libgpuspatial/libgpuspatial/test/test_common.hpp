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
// Helper function to check if a string ends with a specific suffix
static bool HasSuffix(const std::string& str, const std::string& suffix) {
  if (str.length() >= suffix.length()) {
    return (0 == str.compare(str.length() - suffix.length(), suffix.length(), suffix));
  }
  return false;
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

arrow::Status ReadParquetFromFolder(
    arrow::fs::FileSystem* fs, const std::string& folder, int64_t batch_size,
    const char* column_name, std::vector<std::shared_ptr<arrow::Array>>& record_batches) {
  arrow::fs::FileSelector selector;
  selector.base_dir = folder;
  selector.recursive = true;

  ARROW_ASSIGN_OR_RAISE(auto file_infos, fs->GetFileInfo(selector));
  std::cout << "Found " << file_infos.size() << " total objects in " << folder
            << std::endl;

  // 4. Iterate through files, filter for Parquet, and read them
  for (const auto& file_info : file_infos) {
    // Skip directories (which are just prefixes in S3)
    if (file_info.type() != arrow::fs::FileType::File) {
      continue;
    }

    const std::string& path = file_info.path();

    // Optional: Filter for files with a .parquet extension
    if (!HasSuffix(path, ".parquet")) {
      std::cout << "  - Skipping non-parquet file: " << path << std::endl;
      continue;
    }
    std::cout << "--- Processing Parquet file: " << path << " ---" << std::endl;

    auto input_file = fs->OpenInputFile(file_info);

    auto arrow_reader =
        parquet::arrow::OpenFile(input_file.ValueOrDie(), arrow::default_memory_pool())
            .ValueOrDie();

    arrow_reader->set_batch_size(batch_size);

    auto rb_reader = arrow_reader->GetRecordBatchReader().ValueOrDie();
    while (true) {
      std::shared_ptr<arrow::RecordBatch> batch;
      ARROW_THROW_NOT_OK(rb_reader->ReadNext(&batch));
      if (!batch) {
        break;
      }
      record_batches.push_back(batch->GetColumnByName(column_name));
    }
  }

  return arrow::Status::OK();
}

// Function to read a single Parquet file and extract a column.
arrow::Status ReadParquetFromFile(
    arrow::fs::FileSystem* fs,     // 1. Filesystem pointer (e.g., LocalFileSystem)
    const std::string& file_path,  // 2. Single file path instead of a folder
    int64_t batch_size, const char* column_name,
    std::vector<std::shared_ptr<arrow::Array>>& out_arrays) {
  // 1. Get FileInfo for the single path
  ARROW_ASSIGN_OR_RAISE(auto file_info, fs->GetFileInfo(file_path));

  // Check if the path points to a file
  if (file_info.type() != arrow::fs::FileType::File) {
    return arrow::Status::Invalid("Path is not a file: ", file_path);
  }

  std::cout << "--- Processing Parquet file: " << file_path << " ---" << std::endl;

  // 2. Open the input file
  ARROW_ASSIGN_OR_RAISE(auto input_file, fs->OpenInputFile(file_info));

  // 3. Open the Parquet file and create an Arrow reader
  ARROW_ASSIGN_OR_RAISE(auto arrow_reader, parquet::arrow::OpenFile(
                                               input_file, arrow::default_memory_pool()));

  // 4. Set the batch size
  arrow_reader->set_batch_size(batch_size);

  // 5. Get the RecordBatchReader
  auto rb_reader = arrow_reader->GetRecordBatchReader().ValueOrDie();
  // 6. Read all record batches and extract the column
  while (true) {
    std::shared_ptr<arrow::RecordBatch> batch;

    // Read the next batch
    ARROW_THROW_NOT_OK(rb_reader->ReadNext(&batch));

    // Check for end of stream
    if (!batch) {
      break;
    }

    // Extract the specified column and add to the output vector
    std::shared_ptr<arrow::Array> column_array = batch->GetColumnByName(column_name);
    if (!column_array) {
      return arrow::Status::Invalid("Column not found: ", column_name);
    }
    out_arrays.push_back(column_array);
  }

  std::cout << "Finished reading. Total arrays extracted: " << out_arrays.size()
            << std::endl;
  return arrow::Status::OK();
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
