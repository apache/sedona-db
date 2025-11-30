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
#include "gpuspatial/geom/line_segment.cuh"
#include "gpuspatial/utils/array_view.h"
#include "gpuspatial/utils/cuda_utils.h"

namespace gpuspatial {
template <typename POINT_T>
class LineString {
 public:
  using point_t = POINT_T;
  using line_segment_t = LineSegment<point_t>;
  using box_t = Box<Point<float, point_t::n_dim>>;

  LineString() = default;

  DEV_HOST LineString(const ArrayView<point_t>& vertices, const box_t& mbr)
      : vertices_(vertices), mbr_(mbr) {}

  DEV_HOST_INLINE line_segment_t get_line_segment(size_t i) const {
    assert(i + 1 < vertices_.size());
    return line_segment_t(vertices_[i], vertices_[i + 1]);
  }

  DEV_HOST_INLINE const point_t& get_point(size_t i) const { return vertices_[i]; }

  DEV_HOST_INLINE size_t num_points() const { return vertices_.size(); }

  DEV_HOST_INLINE size_t num_segments() const {
    return vertices_.empty() ? 0 : vertices_.size() - 1;
  }

  DEV_HOST_INLINE ArrayView<point_t> get_vertices() const { return vertices_; }

  DEV_HOST_INLINE bool is_zero_length() const {
    if (vertices_.size() >= 2) {
      auto first = vertices_[0];
      for (size_t i = 1; i < vertices_.size(); ++i) {
        if (first != vertices_[i]) {
          return false;  // Found a point that is not equal to the first
        }
      }
    }
    return true;
  }

  DEV_HOST_INLINE bool is_closed() const {
    if (num_segments() == 0) {
      return false;
    }
    return vertices_[0] == vertices_[vertices_.size() - 1];
  }

  DEV_HOST_INLINE bool empty() const { return num_segments() == 0; }

  DEV_HOST_INLINE const box_t& get_mbr() const { return mbr_; }

 private:
  ArrayView<point_t> vertices_;
  box_t mbr_;
};

template <typename POINT_T, typename INDEX_T>
class LineStringArrayView {
 public:
  using point_t = POINT_T;
  using box_t = Box<Point<float, point_t::n_dim>>;
  using geometry_t = LineString<POINT_T>;

  LineStringArrayView() = default;

  DEV_HOST LineStringArrayView(const ArrayView<INDEX_T>& prefix_sum,
                               const ArrayView<POINT_T>& vertices,
                               const ArrayView<box_t>& mbrs)
      : prefix_sum_(prefix_sum), vertices_(vertices), mbrs_(mbrs) {}

  DEV_HOST_INLINE size_t size() const {
    return prefix_sum_.empty() ? 0 : prefix_sum_.size() - 1;
  }

  DEV_HOST_INLINE bool empty() const { return size() == 0; }

  DEV_HOST_INLINE LineString<POINT_T> operator[](size_t i) {
    auto begin = prefix_sum_[i];
    auto end = prefix_sum_[i + 1];
    return {ArrayView<POINT_T>(vertices_.data() + begin, end - begin), mbrs_[i]};
  }

  DEV_HOST_INLINE LineString<POINT_T> operator[](size_t i) const {
    auto begin = prefix_sum_[i];
    auto end = prefix_sum_[i + 1];
    return {
        ArrayView<POINT_T>(const_cast<POINT_T*>(vertices_.data()) + begin, end - begin),
        mbrs_[i]};
  }

  DEV_HOST_INLINE ArrayView<INDEX_T> get_prefix_sum() const { return prefix_sum_; }

  DEV_HOST_INLINE ArrayView<POINT_T> get_vertices() const { return vertices_; }

  DEV_HOST_INLINE ArrayView<box_t> get_mbrs() const { return mbrs_; }

 private:
  ArrayView<INDEX_T> prefix_sum_;
  ArrayView<POINT_T> vertices_;
  ArrayView<box_t> mbrs_;
};

}  // namespace gpuspatial
