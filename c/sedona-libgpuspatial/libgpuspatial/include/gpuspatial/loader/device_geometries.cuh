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
#include "gpuspatial/geom/geometry_type.cuh"
#include "gpuspatial/geom/multi_line_string.cuh"
#include "gpuspatial/geom/multi_point.cuh"
#include "gpuspatial/geom/multi_polygon.cuh"
#include "gpuspatial/geom/polygon.cuh"
#include "gpuspatial/utils/array_view.h"

#include "rmm/device_uvector.hpp"

namespace gpuspatial {
template <typename POINT_T>
class PointSegment;

template <typename POINT_T, typename INDEX_T>
class MultiPointSegment;

template <typename POINT_T, typename INDEX_T>
class LineStringSegment;

template <typename POINT_T, typename INDEX_T>
class MultiLineStringSegment;

template <typename POINT_T, typename INDEX_T>
class PolygonSegment;

template <typename POINT_T, typename INDEX_T>
class MultiPolygonSegment;

template <typename POINT_T, typename INDEX_T>
class BoxSegment;

template <typename POINT_T, typename INDEX_T>
class ParallelWkbLoader;

template <typename POINT_T, typename INDEX_T>
struct DeviceGeometries {
  using point_t = POINT_T;
  using box_t = Box<Point<float, point_t::n_dim>>;

  DeviceGeometries() : type_(GeometryType::kNumGeometryTypes) {}

  template <typename GeometryArrayView_T>
  GeometryArrayView_T GetGeometryArrayView() const {
    // The const version has identical logic
    if constexpr (std::is_same_v<GeometryArrayView_T, PointArrayView<POINT_T, INDEX_T>>) {
      return {ArrayView<POINT_T>(points_)};
    } else if constexpr (std::is_same_v<GeometryArrayView_T,
                                        MultiPointArrayView<POINT_T, INDEX_T>>) {
      return {ArrayView<INDEX_T>(offsets_.multi_point_offsets.ps_num_points),
              ArrayView<POINT_T>(points_), ArrayView<box_t>(mbrs_)};
    } else if constexpr (std::is_same_v<GeometryArrayView_T,
                                        LineStringArrayView<POINT_T, INDEX_T>>) {
      return {ArrayView<INDEX_T>(offsets_.line_string_offsets.ps_num_points),
              ArrayView<POINT_T>(points_), ArrayView<box_t>(mbrs_)};
    } else if constexpr (std::is_same_v<GeometryArrayView_T,
                                        MultiLineStringArrayView<POINT_T, INDEX_T>>) {
      return {ArrayView<INDEX_T>(offsets_.multi_line_string_offsets.ps_num_parts),
              ArrayView<INDEX_T>(offsets_.multi_line_string_offsets.ps_num_points),
              ArrayView<POINT_T>(points_), ArrayView<box_t>(mbrs_)};
    } else if constexpr (std::is_same_v<GeometryArrayView_T,
                                        PolygonArrayView<point_t, INDEX_T>>) {
      return {ArrayView<INDEX_T>(offsets_.polygon_offsets.ps_num_rings),
              ArrayView<INDEX_T>(offsets_.polygon_offsets.ps_num_points),
              ArrayView<POINT_T>(points_), ArrayView<box_t>(mbrs_)};
    } else if constexpr (std::is_same_v<GeometryArrayView_T,
                                        MultiPolygonArrayView<point_t, INDEX_T>>) {
      return {ArrayView<INDEX_T>(offsets_.multi_polygon_offsets.ps_num_parts),
              ArrayView<INDEX_T>(offsets_.multi_polygon_offsets.ps_num_rings),
              ArrayView<INDEX_T>(offsets_.multi_polygon_offsets.ps_num_points),
              ArrayView<POINT_T>(points_), ArrayView<box_t>(mbrs_)};
    } else if constexpr (std::is_same_v<GeometryArrayView_T,
                                        BoxArrayView<point_t, INDEX_T>>) {
      return {ArrayView<box_t>(mbrs_)};
    } else {
      static_assert(sizeof(GeometryArrayView_T) == 0,
                    "Unsupported GeometryView type requested.");
    }
    return {};
  }

  struct MultiPointOffsets {
    // content is the index to points_
    rmm::device_uvector<INDEX_T> ps_num_points{0, rmm::cuda_stream_default};
  };

  struct LineStringOffsets {
    // content is the index to points
    rmm::device_uvector<INDEX_T> ps_num_points{0, rmm::cuda_stream_default};
  };

  struct MultiLineStringOffsets {
    // content is the index to prefix_sum_parts
    rmm::device_uvector<INDEX_T> ps_num_parts{0, rmm::cuda_stream_default};
    // content is the index to points
    rmm::device_uvector<INDEX_T> ps_num_points{0, rmm::cuda_stream_default};
  };

  struct PolygonOffsets {
    // content is the index to prefix_sum_rings
    rmm::device_uvector<INDEX_T> ps_num_rings{0, rmm::cuda_stream_default};
    // content is the index to points
    rmm::device_uvector<INDEX_T> ps_num_points{0, rmm::cuda_stream_default};
  };

  struct MultiPolygonOffsets {
    // content is the index to prefix_sum_parts
    rmm::device_uvector<INDEX_T> ps_num_parts{0, rmm::cuda_stream_default};
    // content is the index to prefix_sum_rings
    rmm::device_uvector<INDEX_T> ps_num_rings{0, rmm::cuda_stream_default};
    // content is the index to points
    rmm::device_uvector<INDEX_T> ps_num_points{0, rmm::cuda_stream_default};
  };

  struct GeometryCollectionOffsets {
    rmm::device_uvector<GeometryType> feature_types{0, rmm::cuda_stream_default};
    rmm::device_uvector<INDEX_T> ps_num_geoms{0, rmm::cuda_stream_default};
    // content is the index to prefix_sum_parts
    rmm::device_uvector<INDEX_T> ps_num_parts{0, rmm::cuda_stream_default};
    // content is the index to prefix_sum_rings
    rmm::device_uvector<INDEX_T> ps_num_rings{0, rmm::cuda_stream_default};
    // content is the index to points
    rmm::device_uvector<INDEX_T> ps_num_points{0, rmm::cuda_stream_default};
  };

  struct Offsets {
    LineStringOffsets line_string_offsets;
    PolygonOffsets polygon_offsets;
    MultiPointOffsets multi_point_offsets;
    MultiLineStringOffsets multi_line_string_offsets;
    MultiPolygonOffsets multi_polygon_offsets;
    GeometryCollectionOffsets geom_collection_offsets;
  };

  ArrayView<box_t> get_mbrs() const { return ArrayView<box_t>(mbrs_); }

  ArrayView<point_t> get_points() const {
    return ArrayView<point_t>(const_cast<point_t*>(points_.data()), points_.size());
  }

  Offsets& get_offsets() { return offsets_; }

  const Offsets& get_offsets() const { return offsets_; }

  GeometryType get_geometry_type() const { return type_; }

  size_t num_features() const {
    return mbrs_.size() == 0 ? points_.size() : mbrs_.size();
  }

  void Clear(rmm::cuda_stream_view stream) {
    type_ = GeometryType::kNumGeometryTypes;
    free(stream, points_);
    free(stream, mbrs_);
    free(stream, offsets_.line_string_offsets.ps_num_points);
    free(stream, offsets_.polygon_offsets.ps_num_rings);
    free(stream, offsets_.polygon_offsets.ps_num_points);
    free(stream, offsets_.multi_point_offsets.ps_num_points);
    free(stream, offsets_.multi_line_string_offsets.ps_num_parts);
    free(stream, offsets_.multi_line_string_offsets.ps_num_points);
    free(stream, offsets_.multi_polygon_offsets.ps_num_parts);
    free(stream, offsets_.multi_polygon_offsets.ps_num_rings);
    free(stream, offsets_.multi_polygon_offsets.ps_num_points);
    free(stream, offsets_.geom_collection_offsets.feature_types);
    free(stream, offsets_.geom_collection_offsets.ps_num_geoms);
    free(stream, offsets_.geom_collection_offsets.ps_num_parts);
    free(stream, offsets_.geom_collection_offsets.ps_num_rings);
    free(stream, offsets_.geom_collection_offsets.ps_num_points);
  }

 private:
  friend class PointSegment<POINT_T>;
  friend class MultiPointSegment<POINT_T, INDEX_T>;
  friend class LineStringSegment<POINT_T, INDEX_T>;
  friend class MultiLineStringSegment<POINT_T, INDEX_T>;
  friend class PolygonSegment<POINT_T, INDEX_T>;
  friend class MultiPolygonSegment<POINT_T, INDEX_T>;
  friend class BoxSegment<POINT_T, INDEX_T>;
  friend class ParallelWkbLoader<POINT_T, INDEX_T>;

  // a type for all geometries in this collection
  GeometryType type_;
  rmm::device_uvector<point_t> points_{0, rmm::cuda_stream_default};
  Offsets offsets_;
  // This should be empty if type_ is Point
  // Otherwise, each feature should have a corresponding MBR
  rmm::device_uvector<box_t> mbrs_{0, rmm::cuda_stream_default};

  template <typename T>
  void free(rmm::cuda_stream_view stream, rmm::device_uvector<T>& vec) {
    vec.resize(0, stream);
    vec.shrink_to_fit(stream);
  }
};

}  // namespace gpuspatial
