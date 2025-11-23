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

namespace gpuspatial {

template <typename POINT_T>
class MultiPoint {
 public:
  using point_t = POINT_T;
  using box_t = Box<Point<float, point_t::n_dim>>;

  MultiPoint() = default;

  DEV_HOST MultiPoint(const ArrayView<POINT_T>& points, const box_t& mbr)
      : points_(points), mbr_(mbr) {}

  DEV_HOST_INLINE const POINT_T& get_point(size_t i) const { return points_[i]; }

  DEV_HOST_INLINE size_t num_points() const { return points_.size(); }

  DEV_HOST_INLINE bool empty() const {
    for (size_t i = 0; i < num_points(); i++) {
      if (!get_point(i).empty()) {
        return false;
      }
    }
    return true;
  }

  DEV_HOST_INLINE const box_t& get_mbr() const { return mbr_; }

 private:
  ArrayView<POINT_T> points_;
  box_t mbr_;
};

template <typename POINT_T, typename INDEX_T>
class MultiPointArrayView {
 public:
  using point_t = POINT_T;
  using box_t = Box<Point<float, point_t::n_dim>>;
  using geometry_t = MultiPoint<POINT_T>;

  MultiPointArrayView() = default;

  DEV_HOST MultiPointArrayView(const ArrayView<INDEX_T>& prefix_sum,
                               const ArrayView<POINT_T>& points,
                               const ArrayView<box_t>& mbrs)
      : prefix_sum_(prefix_sum), points_(points), mbrs_(mbrs) {}

  DEV_HOST_INLINE size_t size() const {
    return prefix_sum_.empty() ? 0 : prefix_sum_.size() - 1;
  }

  DEV_HOST_INLINE bool empty() const { return size() == 0; }

  DEV_HOST_INLINE MultiPoint<POINT_T> operator[](size_t i) {
    auto begin = prefix_sum_[i];
    auto end = prefix_sum_[i + 1];
    return {ArrayView<POINT_T>(points_.data() + begin, end - begin), mbrs_[i]};
  }

  DEV_HOST_INLINE MultiPoint<POINT_T> operator[](size_t i) const {
    auto begin = prefix_sum_[i];
    auto end = prefix_sum_[i + 1];

    return {ArrayView<POINT_T>(const_cast<POINT_T*>(points_.data()) + begin, end - begin),
            mbrs_[i]};
  }

  DEV_HOST_INLINE ArrayView<INDEX_T> get_prefix_sum() const { return prefix_sum_; }

  DEV_HOST_INLINE ArrayView<POINT_T> get_points() const { return points_; }

  DEV_HOST_INLINE ArrayView<box_t> get_mbrs() const { return mbrs_; }

 private:
  ArrayView<INDEX_T> prefix_sum_;
  ArrayView<POINT_T> points_;
  ArrayView<box_t> mbrs_;
};

}  // namespace gpuspatial
