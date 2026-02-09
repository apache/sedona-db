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

#include "gpuspatial/geom/point.hpp"
#include "gpuspatial/relate/predicate.hpp"
#include "gpuspatial/utils/array_view.hpp"
#include "gpuspatial/utils/pinned_vector.hpp"

#include "gtest/gtest.h"
#include "rmm/cuda_stream_view.hpp"
#include "rmm/device_uvector.hpp"
#include "rmm/exec_policy.hpp"

#include <geos/geom/Envelope.h>
#include <geos/geom/Geometry.h>
#include <geos/geom/GeometryFactory.h>
#include <geos/index/ItemVisitor.h>
#include <geos/index/strtree/STRtree.h>
#include <geos/io/WKBReader.h>

#include "nanoarrow/nanoarrow.h"
#include "nanoarrow/nanoarrow.hpp"

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
inline std::string GetCanonicalPath(const std::string& relative_path_str) {
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

// Helper to evaluate predicates using GEOS C++ API
static bool EvaluateGeosPredicate(gpuspatial::Predicate predicate,
                                  const geos::geom::Geometry* geom1,
                                  const geos::geom::Geometry* geom2) {
  switch (predicate) {
    case gpuspatial::Predicate::kContains:
      return geom1->contains(geom2);
    case gpuspatial::Predicate::kIntersects:
      return geom1->intersects(geom2);
    case gpuspatial::Predicate::kWithin:
      return geom1->within(geom2);
    case gpuspatial::Predicate::kEquals:
      return geom1->equals(geom2);
    case gpuspatial::Predicate::kTouches:
      return geom1->touches(geom2);
    default:
      throw std::out_of_range("Unsupported GEOS predicate enumeration value.");
  }
}

// Helper structure to keep visitor context
struct JoinVisitorContext {
  const geos::geom::Geometry* probe_geom;
  std::vector<uint32_t>* build_indices;
  std::vector<uint32_t>* probe_indices;
  size_t current_probe_index;
  gpuspatial::Predicate predicate;
};

// GEOS Visitor Implementation
class JoinVisitor : public geos::index::ItemVisitor {
 public:
  JoinVisitorContext* ctx;
  explicit JoinVisitor(JoinVisitorContext* c) : ctx(c) {}

  void visitItem(void* item) override {
    const auto* build_geom = static_cast<const geos::geom::Geometry*>(item);

    // Use the existing predicate evaluator from TestUtils
    if (EvaluateGeosPredicate(ctx->predicate, build_geom, ctx->probe_geom)) {
      size_t build_idx = (size_t)build_geom->getUserData();

      ctx->build_indices->push_back(static_cast<uint32_t>(build_idx));
      ctx->probe_indices->push_back(static_cast<uint32_t>(ctx->current_probe_index));
    }
  }
};

inline void ComputeGeosJoin(ArrowSchema* build_schema,
                            const std::vector<ArrowArray*>& build_arrays,
                            ArrowSchema* probe_schema,
                            const std::vector<ArrowArray*>& probe_arrays,
                            gpuspatial::Predicate predicate,
                            std::vector<uint32_t>& out_build_indices,
                            std::vector<uint32_t>& out_probe_indices) {
  // Initialize GEOS components
  auto factory = geos::geom::GeometryFactory::create();
  geos::io::WKBReader wkb_reader(*factory);
  geos::index::strtree::STRtree tree(10);

  // Storage to keep geometries alive during the operation
  std::vector<std::unique_ptr<geos::geom::Geometry>> build_geoms_storage;
  ArrowError error;

  // --- Build Phase ---
  size_t global_build_offset = 0;

  for (auto* array : build_arrays) {
    nanoarrow::UniqueArrayView array_view;
    if (ArrowArrayViewInitFromSchema(array_view.get(), build_schema, &error) !=
        NANOARROW_OK) {
      throw std::runtime_error("GEOS Build: Failed to init view: " +
                               std::string(error.message));
    }
    if (ArrowArrayViewSetArray(array_view.get(), array, &error) != NANOARROW_OK) {
      throw std::runtime_error("GEOS Build: Failed to set array: " +
                               std::string(error.message));
    }

    for (int64_t i = 0; i < array->length; i++) {
      // Parse WKB
      ArrowStringView wkb_view = ArrowArrayViewGetStringUnsafe(array_view.get(), i);
      auto geom = wkb_reader.read(reinterpret_cast<const unsigned char*>(wkb_view.data),
                                  wkb_view.size_bytes);

      // Set global index as user data
      size_t current_idx = global_build_offset + i;
      geom->setUserData((void*)current_idx);

      // Insert into Index
      tree.insert(geom->getEnvelopeInternal(), geom.get());

      // Transfer ownership
      build_geoms_storage.push_back(std::move(geom));
    }
    global_build_offset += array->length;
  }

  // --- Probe Phase ---
  size_t global_probe_offset = 0;
  JoinVisitorContext ctx;
  ctx.build_indices = &out_build_indices;
  ctx.probe_indices = &out_probe_indices;
  ctx.predicate = predicate;
  JoinVisitor visitor(&ctx);

  for (auto* array : probe_arrays) {
    nanoarrow::UniqueArrayView array_view;
    if (ArrowArrayViewInitFromSchema(array_view.get(), probe_schema, &error) !=
        NANOARROW_OK) {
      throw std::runtime_error("GEOS Probe: Failed to init view: " +
                               std::string(error.message));
    }
    if (ArrowArrayViewSetArray(array_view.get(), array, &error) != NANOARROW_OK) {
      throw std::runtime_error("GEOS Probe: Failed to set array: " +
                               std::string(error.message));
    }

    for (int64_t i = 0; i < array->length; i++) {
      ArrowStringView wkb_view = ArrowArrayViewGetStringUnsafe(array_view.get(), i);
      auto geom = wkb_reader.read(reinterpret_cast<const unsigned char*>(wkb_view.data),
                                  wkb_view.size_bytes);

      ctx.probe_geom = geom.get();
      ctx.current_probe_index = global_probe_offset + i;

      // Query the tree
      tree.query(geom->getEnvelopeInternal(), visitor);
    }
    global_probe_offset += array->length;
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
