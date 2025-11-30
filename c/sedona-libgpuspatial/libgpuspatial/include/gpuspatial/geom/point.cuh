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
#include "gpuspatial/geom/box.cuh"
#include "gpuspatial/utils/array_view.h"
#include "gpuspatial/utils/cuda_utils.h"
#include "gpuspatial/utils/floating_point.h"
#include "gpuspatial/utils/type_traits.h"

namespace gpuspatial {
enum class PointLocation {
  kOutside,
  kInside,
  kBoundary,
  kError,
};

template <typename SCALA_T, int N_DIM>
class Point {
 public:
  using point_t = Point<SCALA_T, N_DIM>;
  using scalar_t = SCALA_T;
  using vec_t = typename cuda_vec<SCALA_T, N_DIM>::type;
  static constexpr int n_dim = N_DIM;
  static_assert(n_dim >= 2, "N_DIM should be at least 2");

  Point() = default;

  DEV_HOST Point(const vec_t& data) : data_(data) {}

  // Only enabled if SCALA_T is double.
  template <typename... Args>
  DEV_HOST Point(Args... args) : data_{args...} {
    // Ensure the correct number of arguments are passed
    static_assert(sizeof...(args) == N_DIM, "Incorrect number of initializers for Point");

    // Ensure all arguments are convertible to the point's scalar type
    static_assert((std::is_convertible_v<Args, scalar_t> && ...),
                  "All initializers must be convertible to the Point's scalar type");
  }

  DEV_HOST_INLINE SCALA_T& get_coordinate(int dim) {
    return reinterpret_cast<SCALA_T*>(&data_.x)[dim];
  }

  DEV_HOST_INLINE const SCALA_T& get_coordinate(int dim) const {
    return reinterpret_cast<const SCALA_T*>(&data_.x)[dim];
  }

  DEV_HOST_INLINE void set_coordinate(int dim, SCALA_T coordinate) {
    reinterpret_cast<SCALA_T*>(&data_.x)[dim] = coordinate;
  }

  DEV_HOST_INLINE vec_t& get_vec() { return data_; }

  DEV_HOST_INLINE const vec_t& get_vec() const { return data_; }

  DEV_HOST_INLINE scalar_t* get_data() { return &data_.x; }

  DEV_HOST_INLINE const scalar_t* get_data() const { return &data_.x; }

  DEV_HOST_INLINE bool empty() const { return std::isnan(data_.x); }

  DEV_HOST_INLINE void set_empty() {
    for (int dim = 0; dim < n_dim; dim++) {
      set_coordinate(dim, std::numeric_limits<scalar_t>::quiet_NaN());
    }
  }

  DEV_HOST_INLINE void set_min() {
    for (int dim = 0; dim < n_dim; dim++) {
      set_coordinate(dim, std::numeric_limits<scalar_t>::lowest());
    }
  }

  DEV_HOST_INLINE void set_max() {
    for (int dim = 0; dim < n_dim; dim++) {
      set_coordinate(dim, std::numeric_limits<scalar_t>::max());
    }
  }
  /**
   * @brief Provides access to the x-coordinate.
   * This method is only available if N_DIM >= 1.
   */
  DEV_HOST_INLINE scalar_t& x() { return data_.x; }

  /**
   * @brief Provides const access to the x-coordinate.
   * This method is only available if N_DIM >= 1.
   */
  DEV_HOST_INLINE const scalar_t& x() const {
    if constexpr (N_DIM >= 1) {
      return data_.x;
    }
  }

  /**
   * @brief Provides access to the y-coordinate.
   * This method is only available if N_DIM >= 2.
   */
  DEV_HOST_INLINE scalar_t& y() { return data_.y; }

  /**
   * @brief Provides const access to the y-coordinate.
   * This method is only available if N_DIM >= 2.
   */
  DEV_HOST_INLINE const scalar_t& y() const { return data_.y; }

  template <int D = N_DIM>
  DEV_HOST_INLINE typename std::enable_if<D >= 3, scalar_t&>::type z() {
    return data_.z;
  }

  /**
   * @brief Provides const access to the z-coordinate.
   * This method is only available if N_DIM >= 3, enabled via std::enable_if.
   */
  template <int D = N_DIM>
  DEV_HOST_INLINE typename std::enable_if<D >= 3, const scalar_t&>::type z() const {
    return data_.z;
  }

  DEV_HOST_INLINE bool operator==(const Point& other) const {
    for (int dim = 0; dim < N_DIM; dim++) {
      if (!float_equal(get_coordinate(dim), other.get_coordinate(dim))) {
        return false;
      }
    }
    return true;
  }

  DEV_HOST_INLINE bool operator!=(const Point& other) const {
    for (int dim = 0; dim < N_DIM; dim++) {
      if (!float_equal(get_coordinate(dim), other.get_coordinate(dim))) {
        return true;
      }
    }
    return false;
  }

  DEV_HOST_INLINE Point operator+(const Point& other) const {
    Point result;
    for (int dim = 0; dim < N_DIM; dim++) {
      result.set_coordinate(dim, get_coordinate(dim) + other.get_coordinate(dim));
    }
    return result;
  }

  DEV_HOST_INLINE Point operator-(const Point& other) const {
    Point result;
    for (int dim = 0; dim < N_DIM; dim++) {
      result.set_coordinate(dim, get_coordinate(dim) - other.get_coordinate(dim));
    }
    return result;
  }

  DEV_HOST_INLINE Point operator/(const Point& other) const {
    Point result;
    for (int dim = 0; dim < N_DIM; dim++) {
      result.set_coordinate(dim, get_coordinate(dim) / other.get_coordinate(dim));
    }
    return result;
  }

  DEV_HOST_INLINE scalar_t& operator[](int dim) { return (&data_.x)[dim]; }

  DEV_HOST_INLINE const scalar_t& operator[](int dim) const { return (&data_.x)[dim]; }

  DEV_HOST_INLINE Box<Point<float, N_DIM>> get_mbr() const {
    Point<float, N_DIM> min_corner, max_corner;
    for (int dim = 0; dim < N_DIM; dim++) {
      auto val = get_coordinate(dim);
      auto min_val = next_float_from_double(val, -1, 1);
      auto max_val = next_float_from_double(val, 1, 1);
      min_corner.set_coordinate(dim, min_val);
      max_corner.set_coordinate(dim, max_val);
    }

    return {min_corner, max_corner};
  }

  DEV_HOST_INLINE bool covered_by(const OptixAabb& aabb) const {
    bool covered = true;
    for (int dim = 0; dim < n_dim && covered; dim++) {
      auto min_val = reinterpret_cast<const float*>(&aabb.minX)[dim];
      auto max_val = reinterpret_cast<const float*>(&aabb.maxX)[dim];
      auto val = get_coordinate(dim);

      covered &= min_val <= val && max_val >= val;
    }
    return covered;
  }

  // For being called by templated methods
  DEV_HOST_INLINE uint32_t num_vertices() const { return 1; }

  DEV_HOST_INLINE Point<float, N_DIM> as_float() const {
    Point<float, N_DIM> result;
    for (int dim = 0; dim < N_DIM; dim++) {
      result.set_coordinate(dim, static_cast<float>(get_coordinate(dim)));
    }
    return result;
  }

 private:
  vec_t data_;
};

template <typename POINT_T, typename INDEX_T>
class PointArrayView {
 public:
  using point_t = POINT_T;
  using geometry_t = point_t;

  PointArrayView() = default;

  DEV_HOST PointArrayView(const ArrayView<POINT_T>& points) : points_(points) {}

  DEV_HOST_INLINE INDEX_T size() const { return points_.size(); }

  DEV_HOST_INLINE bool empty() const { return size() == 0; }

  DEV_HOST_INLINE POINT_T& operator[](INDEX_T i) { return points_[i]; }

  DEV_HOST_INLINE const POINT_T& operator[](INDEX_T i) const { return points_[i]; }

  DEV_HOST_INLINE ArrayView<POINT_T> get_points() const { return points_; }

 private:
  ArrayView<POINT_T> points_;
};
}  // namespace gpuspatial
