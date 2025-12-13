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

#include "gpuspatial/utils/array_view.h"
#include "gpuspatial/utils/cuda_utils.h"
#include "gpuspatial/utils/helpers.h"

#include <optix_types.h>

namespace gpuspatial {
template <typename POINT_T>
class Box {
  using point_t = POINT_T;
  using scalar_t = typename point_t::scalar_t;
  constexpr static int n_dim = point_t::n_dim;

 public:
  DEV_HOST Box() { set_empty(); }

  DEV_HOST Box(const point_t& min, const point_t& max) : min_(min), max_(max) {}

  DEV_HOST_INLINE bool covers(const point_t& p) const {
    bool covers = true;

    for (int dim = 0; covers && dim < n_dim; dim++) {
      auto val = p.get_coordinate(dim);
      covers &= min_.get_coordinate(dim) <= val && max_.get_coordinate(dim) >= val;
    }
    return covers;
  }

  DEV_HOST_INLINE bool covers(const Box& other) const {
    bool covers = true;

    for (int dim = 0; covers && dim < n_dim; dim++) {
      covers &= other.min_.get_coordinate(dim) >= min_.get_coordinate(dim) &&
                other.max_.get_coordinate(dim) <= max_.get_coordinate(dim);
    }
    return covers;
  }

  DEV_HOST_INLINE bool contains(const point_t& p) const {
    bool contains = true;
    for (int dim = 0; contains && dim < n_dim; dim++) {
      auto val = p.get_coordinate(dim);
      contains &= min_.get_coordinate(dim) < val && max_.get_coordinate(dim) > val;
    }
    return contains;
  }

  DEV_HOST_INLINE bool contains(const Box& other) const {
    bool contains = true;

    for (int dim = 0; contains && dim < n_dim; dim++) {
      contains &= other.min_.get_coordinate(dim) > min_.get_coordinate(dim) &&
                  other.max_.get_coordinate(dim) < max_.get_coordinate(dim);
    }
    return contains;
  }

  DEV_HOST_INLINE bool intersects(const point_t& p) const { return covers(p); }

  DEV_HOST_INLINE bool intersects(const Box& other) const {
    bool intersects = true;

    for (int dim = 0; dim < n_dim && intersects; dim++) {
      intersects &= other.min_.get_coordinate(dim) <= max_.get_coordinate(dim) &&
                    other.max_.get_coordinate(dim) >= min_.get_coordinate(dim);
    }
    return intersects;
  }

  DEV_HOST_INLINE OptixAabb ToOptixAabb() const {
    OptixAabb aabb;

    memset(&aabb, 0, sizeof(OptixAabb));
    if (sizeof(scalar_t) == sizeof(float)) {
      for (int dim = 0; dim < n_dim; dim++) {
        reinterpret_cast<float*>(&aabb.minX)[dim] = min_.get_coordinate(dim);
        reinterpret_cast<float*>(&aabb.maxX)[dim] = max_.get_coordinate(dim);
      }
    } else {
      for (int dim = 0; dim < n_dim; dim++) {
        auto min_val = min_.get_coordinate(dim);
        auto max_val = max_.get_coordinate(dim);

        reinterpret_cast<float*>(&aabb.minX)[dim] =
            next_float_from_double(min_val, -1, 2);
        reinterpret_cast<float*>(&aabb.maxX)[dim] = next_float_from_double(max_val, 1, 2);
      }
    }
    return aabb;
  }

  DEV_HOST_INLINE bool covered_by(const OptixAabb& aabb) const {
    bool covered = true;
    for (int dim = 0; dim < n_dim && covered; dim++) {
      auto min_val = reinterpret_cast<const float*>(&aabb.minX)[dim];
      auto max_val = reinterpret_cast<const float*>(&aabb.maxX)[dim];

      covered &= min_val <= get_min(dim) && max_val >= get_max(dim);
    }
    return covered;
  }

  DEV_HOST_INLINE bool intersects(const OptixAabb& aabb) const {
    bool intersects = true;
    for (int dim = 0; dim < n_dim && intersects; dim++) {
      auto min_val = reinterpret_cast<const float*>(&aabb.minX)[dim];
      auto max_val = reinterpret_cast<const float*>(&aabb.maxX)[dim];

      intersects &= min_val <= get_max(dim) && max_val >= get_min(dim);
    }
    return intersects;
  }

  DEV_HOST_INLINE void set_min(const point_t& min) { min_ = min; }

  DEV_HOST_INLINE void set_max(const point_t& max) { max_ = max; }

  DEV_HOST_INLINE const point_t& get_min() const { return min_; }

  DEV_HOST_INLINE scalar_t get_min(int dim) const { return min_.get_coordinate(dim); }

  DEV_HOST_INLINE const point_t& get_max() const { return max_; }

  DEV_HOST_INLINE scalar_t get_max(int dim) const { return max_.get_coordinate(dim); }

  DEV_HOST_INLINE point_t centroid() const {
    point_t c;
    for (int dim = 0; dim < n_dim; dim++) {
      auto val = (min_.get_coordinate(dim) + max_.get_coordinate(dim)) / 2;

      c.set_coordinate(dim, val);
    }
    return c;
  }

  DEV_HOST_INLINE void Expand(const point_t& p) {
    auto* p_min = min_.get_data();
    auto* p_max = max_.get_data();

    for (int dim = 0; dim < n_dim; dim++) {
      auto val = p.get_coordinate(dim);

      p_min[dim] = std::min(p_min[dim], val);
      p_max[dim] = std::max(p_max[dim], val);
    }
  }

  DEV_HOST_INLINE void set_empty() {
    for (int dim = 0; dim < n_dim; dim++) {
      min_.set_coordinate(dim, std::numeric_limits<scalar_t>::max());
      max_.set_coordinate(dim, std::numeric_limits<scalar_t>::lowest());
    }
  }

  DEV_HOST_INLINE bool is_empty() const { return min_.x() > max_.x(); }

  // exposed these methods to GeometryGrouper
  DEV_HOST_INLINE Box& get_mbr() { return *this; }

  DEV_HOST_INLINE const Box& get_mbr() const { return *this; }

#if defined(__CUDA_ARCH__)
  DEV_INLINE void ExpandAtomic(const point_t& p) {
    auto* p_min = min_.get_data();
    auto* p_max = max_.get_data();

    for (int dim = 0; dim < n_dim; dim++) {
      auto val = p.get_coordinate(dim);

      atomicMin(&p_min[dim], val);
      atomicMax(&p_max[dim], val);
    }
  }
#endif

 private:
  point_t min_, max_;
};
template <typename SCALAR_T, int N_DIM>
class Point;

template <typename POINT_T, typename INDEX_T>
class BoxArrayView {
  using box_t = Box<Point<float, POINT_T::n_dim>>;

 public:
  using point_t = POINT_T;
  using geometry_t = box_t;

  BoxArrayView() = default;

  DEV_HOST BoxArrayView(const ArrayView<box_t>& boxes) : boxes_(boxes) {}

  DEV_HOST_INLINE size_t size() const { return boxes_.size(); }

  DEV_HOST_INLINE box_t& operator[](size_t i) { return boxes_[i]; }

  DEV_HOST_INLINE const box_t& operator[](size_t i) const { return boxes_[i]; }

 private:
  ArrayView<box_t> boxes_;
};

}  // namespace gpuspatial
