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

namespace gpuspatial {

enum class Predicate {
  kEquals,
  kDisjoint,
  kTouches,
  kContains,
  kCovers,
  kIntersects,
  kWithin,
  kCoveredBy
};

/**
 * @brief Converts a Predicate enum class value to its string representation.
 *
 * @param predicate The Predicate value to convert.
 * @return const char* A string literal corresponding to the enum value.
 * Returns "Unknown Predicate" if the value is not recognized.
 */
inline const char* PredicateToString(Predicate predicate) {
  switch (predicate) {
    case Predicate::kEquals:
      return "Equals";
    case Predicate::kDisjoint:
      return "Disjoint";
    case Predicate::kTouches:
      return "Touches";
    case Predicate::kContains:
      return "Contains";
    case Predicate::kCovers:
      return "Covers";
    case Predicate::kIntersects:
      return "Intersects";
    case Predicate::kWithin:
      return "Within";
    case Predicate::kCoveredBy:
      return "CoveredBy";
    default:
      // Handle any unexpected values safely
      return "Unknown Predicate";
  }
}
}  // namespace gpuspatial
