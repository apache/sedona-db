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

#include <memory>
#include <stdexcept>
#include <vector>

namespace gpuspatial {
template <typename SCALAR_T, int N_DIM>
class SpatialIndex {
 public:
  using point_t = Point<SCALAR_T, N_DIM>;
  using box_t = Box<point_t>;

  virtual ~SpatialIndex() = default;

  /**
   * Provide an array of geometries to build the index.
   * @param rects An array of rectangles to be indexed.
   */
  virtual void PushBuild(const box_t* rects, uint32_t n_rects) = 0;

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
   * Query the index with an array of rectangles and return the indices of
   * the rectangles. This method is thread-safe.
   * @param build_indices A vector to store the indices of the geometries in the index
   * that have a spatial overlap with the geometries in the stream.
   * @param stream_indices A vector to store the indices of the geometries in the stream
   * that have a spatial overlap with the geometries in the index.
   */
  virtual void Probe(const box_t* rects, uint32_t n_rects,
                     std::vector<uint32_t>* build_indices,
                     std::vector<uint32_t>* stream_indices) {
    throw std::runtime_error("Not implemented");
  }
};

}  // namespace gpuspatial
