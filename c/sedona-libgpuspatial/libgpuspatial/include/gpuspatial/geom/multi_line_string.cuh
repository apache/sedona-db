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
#include "gpuspatial/geom/line_string.cuh"
#include "gpuspatial/utils/array_view.h"
#include "gpuspatial/utils/cuda_utils.h"

namespace gpuspatial {
template <typename POINT_T, typename INDEX_T>
class MultiLineString {
 public:
  using point_t = POINT_T;
  using box_t = Box<Point<float, point_t::n_dim>>;

  MultiLineString() = default;

  DEV_HOST MultiLineString(const ArrayView<INDEX_T>& prefix_sum_part,
                           const ArrayView<point_t>& vertices, const box_t& mbr)
      : prefix_sum_part_(prefix_sum_part), vertices_(vertices), mbr_(mbr) {}

  DEV_HOST_INLINE LineString<POINT_T> get_line_string(size_t i) const {
    auto begin = prefix_sum_part_[i];
    auto end = prefix_sum_part_[i + 1];
    return {
        ArrayView<POINT_T>(const_cast<point_t*>(vertices_.data()) + begin, end - begin),
        mbr_};
  }

  DEV_HOST_INLINE size_t num_line_strings() const {
    return prefix_sum_part_.empty() ? 0 : prefix_sum_part_.size() - 1;
  }

  DEV_HOST_INLINE bool empty() const {
    for (size_t i = 0; i < num_line_strings(); i++) {
      if (!get_line_string(i).empty()) {
        return false;
      }
    }
    return true;
  }
  DEV_HOST_INLINE const box_t& get_mbr() const { return mbr_; }

 private:
  ArrayView<INDEX_T> prefix_sum_part_;
  ArrayView<point_t> vertices_;
  box_t mbr_;
};

template <typename POINT_T, typename INDEX_T>
class MultiLineStringArrayView {
 public:
  using point_t = POINT_T;
  using box_t = Box<Point<float, point_t::n_dim>>;
  using geometry_t = MultiLineString<POINT_T, INDEX_T>;

  MultiLineStringArrayView() = default;

  DEV_HOST MultiLineStringArrayView(const ArrayView<INDEX_T>& prefix_sum_geoms,
                                    const ArrayView<INDEX_T>& prefix_sum_parts,
                                    const ArrayView<POINT_T>& vertices,
                                    const ArrayView<box_t>& mbrs)
      : prefix_sum_geoms_(prefix_sum_geoms),
        prefix_sum_parts_(prefix_sum_parts),
        vertices_(vertices),
        mbrs_(mbrs) {}

  DEV_HOST_INLINE size_t size() const {
    return prefix_sum_geoms_.empty() ? 0 : prefix_sum_geoms_.size() - 1;
  }

  DEV_HOST_INLINE bool empty() const { return size() == 0; }

  DEV_HOST_INLINE MultiLineString<POINT_T, INDEX_T> operator[](size_t i) {
    auto begin = prefix_sum_geoms_[i];
    auto end = prefix_sum_geoms_[i + 1];
    return {ArrayView<INDEX_T>(prefix_sum_parts_.data() + begin, end - begin + 1),
            vertices_, mbrs_[i]};
  }

  DEV_HOST_INLINE MultiLineString<POINT_T, INDEX_T> operator[](size_t i) const {
    auto begin = prefix_sum_geoms_[i];
    auto end = prefix_sum_geoms_[i + 1];
    return {ArrayView<INDEX_T>(const_cast<INDEX_T*>(prefix_sum_parts_.data()) + begin,
                               end - begin + 1),
            vertices_, mbrs_[i]};
  }

  DEV_HOST_INLINE ArrayView<INDEX_T> get_prefix_sum_geoms() const {
    return prefix_sum_geoms_;
  }

  DEV_HOST_INLINE ArrayView<INDEX_T> get_prefix_sum_parts() const {
    return prefix_sum_parts_;
  }

  DEV_HOST_INLINE ArrayView<POINT_T> get_vertices() const { return vertices_; }

  DEV_HOST_INLINE ArrayView<box_t> get_mbrs() const { return mbrs_; }

 private:
  ArrayView<INDEX_T> prefix_sum_geoms_;
  ArrayView<INDEX_T> prefix_sum_parts_;
  ArrayView<POINT_T> vertices_;
  ArrayView<box_t> mbrs_;
};

}  // namespace gpuspatial
