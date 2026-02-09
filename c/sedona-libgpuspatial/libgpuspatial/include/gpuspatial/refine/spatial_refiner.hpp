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
#include "gpuspatial/relate/predicate.hpp"

#include "nanoarrow/nanoarrow.h"

namespace gpuspatial {
/** This class refines candidate pairs of geometries based on a spatial predicate.
 *
 * The SpatialRefiner is initialized by pushing build-side geometries via PushBuild(),
 * followed by a call to FinishBuilding(). After that, the Refine() method can be called
 * multiple times with probe-side geometries and candidate index pairs to filter out
 * non-matching pairs based on the specified spatial predicate.
 */
class SpatialRefiner {
 public:
  virtual ~SpatialRefiner() = default;

  /** Clear the internal state of the refiner, allowing it to be reused.
   */
  virtual void Clear() = 0;

  /** Push build-side geometries to the refiner.
   *
   * @param build_array An ArrowArrayView containing the build-side geometries.
   */
  virtual void PushBuild(const ArrowArrayView* build_array) = 0;

  /** Finalize the build-side geometries after all have been pushed. The Refine function
   * can only be used after this call.
   */
  virtual void FinishBuilding() = 0;

  /** Refine candidate pairs of geometries based on a spatial predicate.
   *
   * @param probe_array An ArrowArrayView containing the probe-side geometries.
   * @param predicate The spatial predicate to use for refinement.
   * @param build_indices An array of build-side indices corresponding to candidate pairs.
   * This is a global index from 0 to N-1, where N is the total number of build geometries
   * pushed.
   * @param probe_indices An array of probe-side indices corresponding to candidate pairs.
   * This is a local index from 0 to M - 1, where M is the number of geometries in the
   * probe_array.
   * @param len The length of the build_indices and probe_indices arrays.
   * @return The number of candidate pairs that satisfy the spatial predicate after
   * refinement.
   */
  virtual uint32_t Refine(const ArrowArrayView* probe_array, Predicate predicate,
                          uint32_t* build_indices, uint32_t* probe_indices,
                          uint32_t len) = 0;
};

}  // namespace gpuspatial
