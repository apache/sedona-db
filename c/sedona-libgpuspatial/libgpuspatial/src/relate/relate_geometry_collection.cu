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
#include "gpuspatial/relate/relate_engine.cuh"
#include "gpuspatial/utils/array_view.hpp"
#include "gpuspatial/utils/helpers.cuh"

#include "rmm/cuda_stream_view.hpp"
#include "rmm/device_uvector.hpp"
#include "rmm/exec_policy.hpp"

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>
#include <cuda/std/iterator>

namespace gpuspatial {

template <typename POINT_T, typename INDEX_T>
template <bool GC_IS_FIRST, typename OTHER_ARRAY_VIEW_T>
void RelateEngine<POINT_T, INDEX_T>::EvaluateGeometryCollection(
    const rmm::cuda_stream_view& stream,
    const GeometryCollectionArrayView<POINT_T, INDEX_T>& gc_array,
    const OTHER_ARRAY_VIEW_T& other_array, Predicate predicate,
    rmm::device_uvector<INDEX_T>& gc_ids, rmm::device_uvector<INDEX_T>& other_ids) {
  assert(gc_ids.size() == other_ids.size());
  size_t ids_size = gc_ids.size();
  if (ids_size == 0 || gc_array.empty()) {
    gc_ids.resize(0, stream);
    other_ids.resize(0, stream);
    return;
  }

  auto ps_geoms = gc_array.get_prefix_sum_num_geoms();
  auto ps_parts = gc_array.get_prefix_sum_num_parts();
  auto ps_rings = gc_array.get_prefix_sum_num_rings();
  auto ps_points = gc_array.get_prefix_sum_num_points();
  auto feature_types = gc_array.get_feature_types();
  auto mbrs = gc_array.get_mbrs();

  rmm::device_uvector<INDEX_T> counts(ids_size, stream);
  thrust::transform(rmm::exec_policy_nosync(stream), gc_ids.begin(), gc_ids.end(),
                    counts.begin(), [=] __device__(INDEX_T gc_id) {
                      return ps_geoms[gc_id + 1] - ps_geoms[gc_id];
                    });

  bool all_single_geom = thrust::all_of(
      rmm::exec_policy_nosync(stream), counts.begin(), counts.end(),
      [] __device__(INDEX_T c) { return c == 1; });

  rmm::device_uvector<INDEX_T> geom_ids(0, stream);
  rmm::device_uvector<INDEX_T> exp_other_ids(0, stream);
  bool has_multi_geoms = false;

  if (all_single_geom) {
    counts.resize(0, stream);
    counts.shrink_to_fit(stream);
    geom_ids.resize(ids_size, stream);
    exp_other_ids.resize(ids_size, stream);
    thrust::transform(rmm::exec_policy_nosync(stream), gc_ids.begin(), gc_ids.end(),
                      geom_ids.begin(),
                      [=] __device__(INDEX_T gc_id) { return ps_geoms[gc_id]; });
    thrust::copy(rmm::exec_policy_nosync(stream), other_ids.begin(), other_ids.end(),
                 exp_other_ids.begin());
  } else {
    has_multi_geoms = true;
    rmm::device_uvector<INDEX_T> ps_counts(ids_size + 1, stream);
    ps_counts.set_element_to_zero_async(0, stream);
    thrust::inclusive_scan(rmm::exec_policy_nosync(stream), counts.begin(), counts.end(),
                           ps_counts.begin() + 1);
    counts.resize(0, stream);
    counts.shrink_to_fit(stream);

    INDEX_T total_expanded = ps_counts.back_element(stream);
    stream.synchronize();
    geom_ids.resize(total_expanded, stream);
    exp_other_ids.resize(total_expanded, stream);
    if (total_expanded > 0) {
      auto ps_counts_view = ArrayView<INDEX_T>(ps_counts);
      auto gc_ids_view = ArrayView<INDEX_T>(gc_ids);
      auto other_ids_view = ArrayView<INDEX_T>(other_ids);
      auto out_zip = thrust::make_zip_iterator(
          thrust::make_tuple(geom_ids.begin(), exp_other_ids.begin()));
      thrust::transform(
          rmm::exec_policy_nosync(stream),
          thrust::make_counting_iterator<INDEX_T>(0),
          thrust::make_counting_iterator<INDEX_T>(total_expanded), out_zip,
          [=] __device__(INDEX_T idx) {
            auto it = thrust::upper_bound(thrust::seq, ps_counts_view.begin(),
                                          ps_counts_view.end(), idx);
            // The index into the original candidate pair arrays (gc_ids, other_ids)
            INDEX_T pair_idx = cuda::std::distance(ps_counts_view.begin(), it) - 1;
            INDEX_T local_geom = idx - ps_counts_view[pair_idx];
            INDEX_T gc_id = gc_ids_view[pair_idx];
            return thrust::make_tuple(ps_geoms[gc_id] + local_geom,
                                      other_ids_view[pair_idx]);
          });
    }
  }

  size_t total_cand = geom_ids.size();
  if (total_cand == 0) {
    gc_ids.resize(0, stream);
    other_ids.resize(0, stream);
    return;
  }

  rmm::device_uvector<INDEX_T> out_gc_ids(0, stream);
  rmm::device_uvector<INDEX_T> out_other_ids(0, stream);
  out_gc_ids.reserve(total_cand, stream);
  out_other_ids.reserve(total_cand, stream);

  auto call_evaluate = [&](const auto& sub_view, rmm::device_uvector<INDEX_T>& sub_gc_ids,
                           rmm::device_uvector<INDEX_T>& sub_other_ids) {
    if constexpr (GC_IS_FIRST) {
      Evaluate(stream, sub_view, other_array, predicate, sub_gc_ids, sub_other_ids);
    } else {
      Evaluate(stream, other_array, sub_view, predicate, sub_other_ids, sub_gc_ids);
    }
  };

  auto append_results = [&](const rmm::device_uvector<INDEX_T>& sub_gc_ids,
                            const rmm::device_uvector<INDEX_T>& sub_other_ids) {
    if (sub_gc_ids.is_empty()) return;
    size_t prev_size = out_gc_ids.size();
    size_t add_size = sub_gc_ids.size();
    out_gc_ids.resize(prev_size + add_size, stream);
    out_other_ids.resize(prev_size + add_size, stream);
    thrust::copy(rmm::exec_policy_nosync(stream), sub_gc_ids.begin(), sub_gc_ids.end(),
                 out_gc_ids.begin() + prev_size);
    thrust::copy(rmm::exec_policy_nosync(stream), sub_other_ids.begin(),
                 sub_other_ids.end(), out_other_ids.begin() + prev_size);
  };

  rmm::device_uvector<INDEX_T> sub_gc_ids(0, stream);
  rmm::device_uvector<INDEX_T> sub_other_ids(0, stream);
  auto cand_zip_begin = thrust::make_zip_iterator(
      thrust::make_tuple(geom_ids.begin(), exp_other_ids.begin()));
  auto cand_zip_end =
      thrust::make_zip_iterator(thrust::make_tuple(geom_ids.end(), exp_other_ids.end()));

  auto extract_by_type = [&](GeometryType target_type) -> size_t {
    sub_gc_ids.resize(total_cand, stream);
    sub_other_ids.resize(total_cand, stream);
    auto sub_zip_begin = thrust::make_zip_iterator(
        thrust::make_tuple(sub_gc_ids.begin(), sub_other_ids.begin()));
    auto sub_zip_end = thrust::copy_if(
        rmm::exec_policy_nosync(stream), cand_zip_begin, cand_zip_end, sub_zip_begin,
        [=] __device__(const thrust::tuple<INDEX_T, INDEX_T>& t) {
          return feature_types[thrust::get<0>(t)] == target_type;
        });
    size_t n = cuda::std::distance(sub_zip_begin, sub_zip_end);
    sub_gc_ids.resize(n, stream);
    sub_other_ids.resize(n, stream);
    return n;
  };

  // 1. Point sub-array
  if (extract_by_type(GeometryType::kPoint) > 0) {
    thrust::transform(rmm::exec_policy_nosync(stream), sub_gc_ids.begin(),
                      sub_gc_ids.end(), sub_gc_ids.begin(),
                      [=] __device__(INDEX_T geom_idx) {
                        return ps_points[ps_rings[ps_parts[geom_idx]]];
                      });
    call_evaluate(gc_array.get_point_array_view(), sub_gc_ids, sub_other_ids);
    thrust::transform(rmm::exec_policy_nosync(stream), sub_gc_ids.begin(),
                      sub_gc_ids.end(), sub_gc_ids.begin(),
                      [=] __device__(INDEX_T point_idx) {
                        return gc_array.locate_vertex(point_idx);
                      });
    append_results(sub_gc_ids, sub_other_ids);
  }

  // 2. MultiPoint and LineString sub-arrays (indexed by ring_idx)
  using box_t = typename GeometryCollectionArrayView<POINT_T, INDEX_T>::box_t;
  rmm::device_uvector<box_t> ring_mbrs(0, stream);
  auto get_ring_mbrs = [&]() -> ArrayView<box_t> {
    size_t num_rings = ps_points.empty() ? 0 : ps_points.size() - 1;
    if (ring_mbrs.is_empty() && num_rings > 0) {
      ring_mbrs.resize(num_rings, stream);
      thrust::transform(rmm::exec_policy_nosync(stream),
                        thrust::make_counting_iterator<INDEX_T>(0),
                        thrust::make_counting_iterator<INDEX_T>(num_rings),
                        ring_mbrs.begin(), [=] __device__(INDEX_T ring_idx) {
                          return mbrs[gc_array.locate_ring(ring_idx)];
                        });
    }
    return ArrayView<box_t>(ring_mbrs);
  };

  if (extract_by_type(GeometryType::kMultiPoint) > 0) {
    thrust::transform(rmm::exec_policy_nosync(stream), sub_gc_ids.begin(),
                      sub_gc_ids.end(), sub_gc_ids.begin(),
                      [=] __device__(INDEX_T geom_idx) {
                        return ps_rings[ps_parts[geom_idx]];
                      });
    call_evaluate(gc_array.get_multi_point_array_view(get_ring_mbrs()), sub_gc_ids,
                  sub_other_ids);
    thrust::transform(rmm::exec_policy_nosync(stream), sub_gc_ids.begin(),
                      sub_gc_ids.end(), sub_gc_ids.begin(),
                      [=] __device__(INDEX_T ring_idx) {
                        return gc_array.locate_ring(ring_idx);
                      });
    append_results(sub_gc_ids, sub_other_ids);
  }

  if (extract_by_type(GeometryType::kLineString) > 0) {
    thrust::transform(rmm::exec_policy_nosync(stream), sub_gc_ids.begin(),
                      sub_gc_ids.end(), sub_gc_ids.begin(),
                      [=] __device__(INDEX_T geom_idx) {
                        return ps_rings[ps_parts[geom_idx]];
                      });
    call_evaluate(gc_array.get_line_string_array_view(get_ring_mbrs()), sub_gc_ids,
                  sub_other_ids);
    thrust::transform(rmm::exec_policy_nosync(stream), sub_gc_ids.begin(),
                      sub_gc_ids.end(), sub_gc_ids.begin(),
                      [=] __device__(INDEX_T ring_idx) {
                        return gc_array.locate_ring(ring_idx);
                      });
    append_results(sub_gc_ids, sub_other_ids);
  }

  // 3. MultiLineString and Polygon sub-arrays (indexed by part_idx)
  rmm::device_uvector<box_t> part_mbrs(0, stream);
  auto get_part_mbrs = [&]() -> ArrayView<box_t> {
    size_t num_parts = ps_rings.empty() ? 0 : ps_rings.size() - 1;
    if (part_mbrs.is_empty() && num_parts > 0) {
      part_mbrs.resize(num_parts, stream);
      thrust::transform(rmm::exec_policy_nosync(stream),
                        thrust::make_counting_iterator<INDEX_T>(0),
                        thrust::make_counting_iterator<INDEX_T>(num_parts),
                        part_mbrs.begin(), [=] __device__(INDEX_T part_idx) {
                          return mbrs[gc_array.locate_part(part_idx)];
                        });
    }
    return ArrayView<box_t>(part_mbrs);
  };

  if (extract_by_type(GeometryType::kMultiLineString) > 0) {
    thrust::transform(rmm::exec_policy_nosync(stream), sub_gc_ids.begin(),
                      sub_gc_ids.end(), sub_gc_ids.begin(),
                      [=] __device__(INDEX_T geom_idx) { return ps_parts[geom_idx]; });
    call_evaluate(gc_array.get_multi_line_string_array_view(get_part_mbrs()), sub_gc_ids,
                  sub_other_ids);
    thrust::transform(rmm::exec_policy_nosync(stream), sub_gc_ids.begin(),
                      sub_gc_ids.end(), sub_gc_ids.begin(),
                      [=] __device__(INDEX_T part_idx) {
                        return gc_array.locate_part(part_idx);
                      });
    append_results(sub_gc_ids, sub_other_ids);
  }

  if (extract_by_type(GeometryType::kPolygon) > 0) {
    thrust::transform(rmm::exec_policy_nosync(stream), sub_gc_ids.begin(),
                      sub_gc_ids.end(), sub_gc_ids.begin(),
                      [=] __device__(INDEX_T geom_idx) { return ps_parts[geom_idx]; });
    call_evaluate(gc_array.get_polygon_array_view(get_part_mbrs()), sub_gc_ids,
                  sub_other_ids);
    thrust::transform(rmm::exec_policy_nosync(stream), sub_gc_ids.begin(),
                      sub_gc_ids.end(), sub_gc_ids.begin(),
                      [=] __device__(INDEX_T part_idx) {
                        return gc_array.locate_part(part_idx);
                      });
    append_results(sub_gc_ids, sub_other_ids);
  }

  // 4. MultiPolygon sub-array (indexed by geom_idx)
  rmm::device_uvector<box_t> geom_mbrs(0, stream);
  auto get_geom_mbrs = [&]() -> ArrayView<box_t> {
    size_t num_geoms = feature_types.size();
    if (!has_multi_geoms && num_geoms == gc_array.size()) {
      return mbrs;
    }
    if (geom_mbrs.is_empty() && num_geoms > 0) {
      geom_mbrs.resize(num_geoms, stream);
      thrust::transform(rmm::exec_policy_nosync(stream),
                        thrust::make_counting_iterator<INDEX_T>(0),
                        thrust::make_counting_iterator<INDEX_T>(num_geoms),
                        geom_mbrs.begin(), [=] __device__(INDEX_T geom_idx) {
                          return mbrs[gc_array.locate_geom(geom_idx)];
                        });
    }
    return ArrayView<box_t>(geom_mbrs);
  };

  if (extract_by_type(GeometryType::kMultiPolygon) > 0) {
    call_evaluate(gc_array.get_multi_polygon_array_view(get_geom_mbrs()), sub_gc_ids,
                  sub_other_ids);
    thrust::transform(rmm::exec_policy_nosync(stream), sub_gc_ids.begin(),
                      sub_gc_ids.end(), sub_gc_ids.begin(),
                      [=] __device__(INDEX_T geom_idx) {
                        return gc_array.locate_geom(geom_idx);
                      });
    append_results(sub_gc_ids, sub_other_ids);
  }

  if (has_multi_geoms && out_gc_ids.size() > 1) {
    auto out_zip_begin = thrust::make_zip_iterator(
        thrust::make_tuple(out_gc_ids.begin(), out_other_ids.begin()));
    auto out_zip_end = thrust::make_zip_iterator(
        thrust::make_tuple(out_gc_ids.end(), out_other_ids.end()));
    thrust::sort(rmm::exec_policy_nosync(stream), out_zip_begin, out_zip_end);
    auto uniq_end =
        thrust::unique(rmm::exec_policy_nosync(stream), out_zip_begin, out_zip_end);
    size_t uniq_size = cuda::std::distance(out_zip_begin, uniq_end);
    out_gc_ids.resize(uniq_size, stream);
    out_other_ids.resize(uniq_size, stream);
  }

  gc_ids = std::move(out_gc_ids);
  other_ids = std::move(out_other_ids);
}

template <typename POINT_T, typename INDEX_T>
template <typename GEOM2_ARRAY_VIEW_T>
void RelateEngine<POINT_T, INDEX_T>::Evaluate(
    const rmm::cuda_stream_view& stream,
    const GeometryCollectionArrayView<POINT_T, INDEX_T>& geom_array1,
    const GEOM2_ARRAY_VIEW_T& geom_array2, Predicate predicate,
    rmm::device_uvector<INDEX_T>& ids1, rmm::device_uvector<INDEX_T>& ids2) {
  EvaluateGeometryCollection<true>(stream, geom_array1, geom_array2, predicate, ids1,
                                   ids2);
}

template <typename POINT_T, typename INDEX_T>
template <typename GEOM1_ARRAY_VIEW_T>
void RelateEngine<POINT_T, INDEX_T>::Evaluate(
    const rmm::cuda_stream_view& stream, const GEOM1_ARRAY_VIEW_T& geom_array1,
    const GeometryCollectionArrayView<POINT_T, INDEX_T>& geom_array2, Predicate predicate,
    rmm::device_uvector<INDEX_T>& ids1, rmm::device_uvector<INDEX_T>& ids2) {
  EvaluateGeometryCollection<false>(stream, geom_array2, geom_array1, predicate, ids2,
                                    ids1);
}

template <typename POINT_T, typename INDEX_T>
void RelateEngine<POINT_T, INDEX_T>::Evaluate(
    const rmm::cuda_stream_view& stream,
    const GeometryCollectionArrayView<POINT_T, INDEX_T>& geom_array1,
    const GeometryCollectionArrayView<POINT_T, INDEX_T>& geom_array2, Predicate predicate,
    rmm::device_uvector<INDEX_T>& ids1, rmm::device_uvector<INDEX_T>& ids2) {
  EvaluateGeometryCollection<true>(stream, geom_array1, geom_array2, predicate, ids1,
                                   ids2);
}

// Explicit template instantiations for Evaluate(GC, other) and Evaluate(other, GC)
#define INSTANTIATE_GC_EVALUATE(OTHER_VIEW_T)                                            \
  template void RelateEngine<Point<double, 2>, uint32_t>::Evaluate<                       \
      OTHER_VIEW_T<Point<double, 2>, uint32_t>>(                                         \
      const rmm::cuda_stream_view& stream,                                               \
      const GeometryCollectionArrayView<Point<double, 2>, uint32_t>& geom_array1,         \
      const OTHER_VIEW_T<Point<double, 2>, uint32_t>& geom_array2, Predicate predicate,  \
      rmm::device_uvector<uint32_t>& ids1, rmm::device_uvector<uint32_t>& ids2);        \
                                                                                         \
  template void RelateEngine<Point<double, 2>, uint32_t>::Evaluate<                       \
      OTHER_VIEW_T<Point<double, 2>, uint32_t>>(                                         \
      const rmm::cuda_stream_view& stream,                                               \
      const OTHER_VIEW_T<Point<double, 2>, uint32_t>& geom_array1,                      \
      const GeometryCollectionArrayView<Point<double, 2>, uint32_t>& geom_array2,         \
      Predicate predicate, rmm::device_uvector<uint32_t>& ids1,                         \
      rmm::device_uvector<uint32_t>& ids2);

INSTANTIATE_GC_EVALUATE(PointArrayView)
INSTANTIATE_GC_EVALUATE(MultiPointArrayView)
INSTANTIATE_GC_EVALUATE(LineStringArrayView)
INSTANTIATE_GC_EVALUATE(MultiLineStringArrayView)
INSTANTIATE_GC_EVALUATE(PolygonArrayView)
INSTANTIATE_GC_EVALUATE(MultiPolygonArrayView)

// Instantiate non-template member functions of RelateEngine
template class RelateEngine<Point<double, 2>, uint32_t>;

}  // namespace gpuspatial
