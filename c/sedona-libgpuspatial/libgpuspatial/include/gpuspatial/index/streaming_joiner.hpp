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
#include "gpuspatial/relate/predicate.cuh"

#include "nanoarrow/nanoarrow.hpp"

#include <memory>
#include <stdexcept>
#include <vector>
namespace gpuspatial {

class StreamingJoiner {
 public:
  struct Context {
    virtual ~Context() = default;
  };

  struct Config {
    virtual ~Config() = default;
  };

  virtual ~StreamingJoiner() = default;

  /**
   * Initialize the index with the given configuration. This method should be called only
   * once before using the index.
   * @param config
   */
  virtual void Init(const Config* config) = 0;

  /**
   * Provide an array of geometries to build the index.
   * @param array ArrowArray that contains the geometries in WKB format.
   * @param offset starting index of the ArrowArray
   * @param length length of the ArrowArray to read.
   */
  virtual void PushBuild(const ArrowSchema* schema, const ArrowArray* array,
                         int64_t offset, int64_t length) = 0;

  /**
   * Waiting the index to be built.
   * This method should be called after all geometries have been pushed.
   */
  virtual void FinishBuilding() = 0;

  /**
   * Remove all geometries from the index, so the index can reused.
   */
  virtual void Clear() = 0;

  /**
   * Query the index with an array of geometries in WKB format and return the indices of
   * the geometries in stream and the index that satisfy a given predicate. This method is
   * thread-safe.
   * @param context A context object that can be used to store intermediate results.
   * @param array ArrowArray that contains the geometries in WKB format.
   * @param offset starting index of the ArrowArray
   * @param length length of the ArrowArray to read.
   * @param predicate A predicate to filter the query results.
   * @param build_indices A vector to store the indices of the geometries in the index
   * that have a spatial overlap with the geometries in the stream.
   * @param stream_indices A vector to store the indices of the geometries in the stream
   * that have a spatial overlap with the geometries in the index.
   * @param stream_index_offset An offset to be added to stream_indices
   */
  virtual void PushStream(Context* context, const ArrowSchema* schema,
                          const ArrowArray* array, int64_t offset, int64_t length,
                          Predicate predicate, std::vector<uint32_t>* build_indices,
                          std::vector<uint32_t>* stream_indices,
                          int32_t stream_index_offset) {
    throw std::runtime_error("Not implemented");
  }

  /**
   * Create a context object for issuing queries against the index.
   * @return A context object that is used to store intermediate results.
   */
  virtual std::shared_ptr<Context> CreateContext() {
    throw std::runtime_error("Not implemented");
  }
};

}  // namespace gpuspatial
