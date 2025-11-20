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

#include "gpuspatial/geom/geometry_type.cuh"
#include "gpuspatial/loader/device_geometries.cuh"
#include "gpuspatial/utils/logger.hpp"
#include "gpuspatial/utils/mem_utils.hpp"
#include "gpuspatial/utils/stopwatch.h"
#include "gpuspatial/utils/thread_pool.h"

#include "nanoarrow/nanoarrow.h"

#include "rmm/cuda_stream_view.hpp"
#include "rmm/device_uvector.hpp"
#include "rmm/exec_policy.hpp"

#include <thrust/scan.h>

#include <thread>
#include <unordered_set>

#include <sys/sysinfo.h>
#include <unistd.h>

namespace gpuspatial {
namespace detail {

inline long long get_free_physical_memory_linux() {
  struct sysinfo info;
  if (sysinfo(&info) == 0) {
    // info.freeram is in bytes (or unit defined by info.mem_unit)
    // Use info.freeram * info.mem_unit for total free bytes
    return (long long)info.freeram * (long long)info.mem_unit;
  }
  return 0;  // Error
}

// Copied from GeoArrow, it is faster than using GeoArrowWKBReaderRead
struct WKBReaderPrivate {
  const uint8_t* data;
  int64_t size_bytes;
  const uint8_t* data0;
  int need_swapping;
  GeoArrowGeometry geom;
};

static int WKBReaderReadEndian(struct WKBReaderPrivate* s, struct GeoArrowError* error) {
  if (s->size_bytes > 0) {
    s->need_swapping = s->data[0] != GEOARROW_NATIVE_ENDIAN;
    s->data++;
    s->size_bytes--;
    return GEOARROW_OK;
  } else {
    GeoArrowErrorSet(error, "Expected endian byte but found end of buffer at byte %ld",
                     (long)(s->data - s->data0));
    return EINVAL;
  }
}

static int WKBReaderReadUInt32(struct WKBReaderPrivate* s, uint32_t* out,
                               struct GeoArrowError* error) {
  if (s->size_bytes >= 4) {
    memcpy(out, s->data, sizeof(uint32_t));
    s->data += sizeof(uint32_t);
    s->size_bytes -= sizeof(uint32_t);
    if (s->need_swapping) {
      *out = __builtin_bswap32(*out);
    }
    return GEOARROW_OK;
  } else {
    GeoArrowErrorSet(error, "Expected uint32 but found end of buffer at byte %ld",
                     (long)(s->data - s->data0));
    return EINVAL;
  }
}

/**
 * @brief This is a general structure to hold parsed geometries on host side
 * There are three modes: Single geometry type, Multi geometry type, GeometryCollection
 * Point: using vertices only
 * LineString: using num_points and vertices
 * Polygon: using num_rings, num_points and vertices
 * MultiPoint: using num_points
 * MultiLineString: using num_parts, num_points and vertices
 * MultiPolygon: using num_parts, num_rings, num_points and vertices
 * GeometryCollection: using all vectors. Empty geometry are treated at the last level
 * with num_points = 0 but still having one entry in num_geoms, num_parts and num_rings
 */
template <typename POINT_T, typename INDEX_T>
struct HostParsedGeometries {
  constexpr static int n_dim = POINT_T::n_dim;
  using mbr_t = Box<Point<float, n_dim>>;
  // each feature should have only one type except GeometryCollection
  std::vector<GeometryType> feature_types;
  // This number should be one except GeometryCollection, which should be unnested # of
  // geometries
  // the size of this vector is equal to number of features
  std::vector<INDEX_T> num_geoms;
  std::vector<INDEX_T> num_parts;
  std::vector<INDEX_T> num_rings;
  std::vector<INDEX_T> num_points;
  std::vector<POINT_T> vertices;
  std::vector<mbr_t> mbrs;
  bool multi = false;
  bool has_geometry_collection = false;
  bool create_mbr = false;

  HostParsedGeometries(bool multi_, bool has_geometry_collection_, bool create_mbr_) {
    // Multi and GeometryCollection are mutually exclusive
    assert(!(multi_ && has_geometry_collection_));
    multi = multi_;
    has_geometry_collection = has_geometry_collection_;
    create_mbr = create_mbr_;
  }

  void AddGeometry(const GeoArrowGeometryView* geom) {
    if (geom == nullptr) {
      throw std::runtime_error("Null geometry not supported yet");
      return;
    }

    auto root = geom->root;
    const GeoArrowGeometryNode* finish = nullptr;
    // All should be one except for GeometryCollection
    uint32_t ngeoms =
        root->geometry_type == GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION ? 0 : 1;
    mbr_t mbr;
    mbr.set_empty();
    mbr_t* p_mbr = create_mbr ? &mbr : nullptr;

    switch (root->geometry_type) {
      case GEOARROW_GEOMETRY_TYPE_POINT: {
        finish = addPoint(root, p_mbr);
        break;
      }
      case GEOARROW_GEOMETRY_TYPE_LINESTRING: {
        finish = addLineString(root, p_mbr);
        break;
      }
      case GEOARROW_GEOMETRY_TYPE_POLYGON: {
        finish = addPolygon(root, p_mbr);
        break;
      }
      case GEOARROW_GEOMETRY_TYPE_MULTIPOINT: {
        finish = addMultiPoint(root, p_mbr);
        break;
      }
      case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING: {
        finish = addMultiLineString(root, p_mbr);
        break;
      }
      case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON: {
        finish = addMultiPolygon(root, p_mbr);
        break;
      }
      case GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION: {
        assert(has_geometry_collection);
        finish = addGeometryCollection(root, p_mbr, ngeoms);
        break;
      }
      default:
        throw std::runtime_error("Unsupported geometry type in GeoArrowGeometryView");
    }
    assert(finish == root + geom->size_nodes);
    if (has_geometry_collection) {
      num_geoms.push_back(ngeoms);
    }
    if (create_mbr) {
      mbrs.push_back(mbr);
    }
  }

 private:
  const GeoArrowGeometryNode* addPoint(const GeoArrowGeometryNode* node, mbr_t* mbr) {
    assert(node->geometry_type == GEOARROW_GEOMETRY_TYPE_POINT);
    auto point = readPoint(node);
    if (has_geometry_collection) {
      feature_types.push_back(GeometryType::kPoint);
      num_parts.push_back(1);
      num_rings.push_back(1);
      num_points.push_back(1);
    } else if (multi) {
      num_points.push_back(1);
    }
    vertices.push_back(point);
    if (mbr != nullptr) {
      mbr->Expand(point.as_float());
    }
    return node + 1;
  }

  const GeoArrowGeometryNode* addMultiPoint(const GeoArrowGeometryNode* node,
                                            mbr_t* mbr) {
    assert(node->geometry_type == GEOARROW_GEOMETRY_TYPE_MULTIPOINT);
    auto np = node->size;
    if (has_geometry_collection) {
      feature_types.push_back(GeometryType::kMultiPoint);
      num_parts.push_back(1);
      num_rings.push_back(1);
      num_points.push_back(np);
    } else {
      num_points.push_back(np);
    }

    for (uint32_t i = 0; i < node->size; i++) {
      auto point_node = node + i + 1;
      auto point = readPoint(point_node);
      vertices.push_back(point);
      if (mbr != nullptr) {
        mbr->Expand(point.as_float());
      }
    }
    return node + node->size + 1;
  }

  const GeoArrowGeometryNode* addLineString(const GeoArrowGeometryNode* node,
                                            mbr_t* mbr) {
    assert(node->geometry_type == GEOARROW_GEOMETRY_TYPE_LINESTRING);
    if (has_geometry_collection) {
      feature_types.push_back(GeometryType::kLineString);
      num_parts.push_back(1);
      num_rings.push_back(1);
    } else if (multi) {
      num_parts.push_back(1);
    }
    // push_back to num_points and vertices
    return processLineString(node, mbr);
  }

  const GeoArrowGeometryNode* addMultiLineString(const GeoArrowGeometryNode* node,
                                                 mbr_t* mbr) {
    assert(node->geometry_type == GEOARROW_GEOMETRY_TYPE_MULTILINESTRING);
    if (has_geometry_collection) {
      feature_types.push_back(GeometryType::kMultiLineString);
      // Treat the whole MultiLineString as one part, where each linestring is a ring
      num_parts.push_back(1);
      num_rings.push_back(node->size);
    } else {
      num_parts.push_back(node->size);
    }
    const GeoArrowGeometryNode* end = node + 1;
    for (uint32_t i = 0; i < node->size; i++) {
      auto* part_node = node + i + 1;
      // push_back to num_points and vertices
      end = processLineString(part_node, mbr);
    }
    return end;
  }

  const GeoArrowGeometryNode* addPolygon(const GeoArrowGeometryNode* node, mbr_t* mbr) {
    assert(node->geometry_type == GEOARROW_GEOMETRY_TYPE_POLYGON);
    if (has_geometry_collection) {
      feature_types.push_back(GeometryType::kPolygon);
      num_parts.push_back(1);
      num_rings.push_back(node->size);
    } else if (multi) {
      num_parts.push_back(1);
      num_rings.push_back(node->size);
    } else {
      num_rings.push_back(node->size);
    }

    auto ring_node = node + 1;
    // visit rings
    for (uint32_t i = 0; i < node->size; i++) {
      // push_back to num_points and vertices
      ring_node = processLineString(ring_node, mbr);
    }
    return ring_node;
  }

  const GeoArrowGeometryNode* addMultiPolygon(const GeoArrowGeometryNode* begin,
                                              mbr_t* mbr) {
    assert(begin->geometry_type == GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON);
    if (has_geometry_collection) {
      feature_types.push_back(GeometryType::kMultiPolygon);
    }
    num_parts.push_back(begin->size);
    auto* polygon_node = begin + 1;
    // for each polygon
    for (auto i = 0; i < begin->size; i++) {
      num_rings.push_back(polygon_node->size);
      auto* ring_node = polygon_node + 1;
      // visit rings
      for (int j = 0; j < polygon_node->size; j++) {
        ring_node = processLineString(ring_node, mbr);
      }
      polygon_node = ring_node;
    }
    return polygon_node;
  }

  const GeoArrowGeometryNode* addGeometryCollection(const GeoArrowGeometryNode* begin,
                                                    mbr_t* mbr, uint32_t& ngeoms) {
    assert(begin->geometry_type == GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION);

    auto curr_node = begin + 1;
    for (int i = 0; i < begin->size; i++) {
      if (curr_node->geometry_type != GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION) {
        ngeoms++;
      }
      switch (curr_node->geometry_type) {
        case GEOARROW_GEOMETRY_TYPE_POINT: {
          curr_node = addPoint(curr_node, mbr);
          break;
        }
        case GEOARROW_GEOMETRY_TYPE_LINESTRING: {
          curr_node = addLineString(curr_node, mbr);
          break;
        }
        case GEOARROW_GEOMETRY_TYPE_POLYGON: {
          curr_node = addPolygon(curr_node, mbr);
          break;
        }
        case GEOARROW_GEOMETRY_TYPE_MULTIPOINT: {
          curr_node = addMultiPoint(curr_node, mbr);
          break;
        }
        case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING: {
          curr_node = addMultiLineString(curr_node, mbr);
          break;
        }
        case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON: {
          curr_node = addMultiPolygon(curr_node, mbr);
          break;
        }
        case GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION: {
          curr_node = addGeometryCollection(curr_node, mbr, ngeoms);
          break;
        }
      }
    }
    return curr_node;
  }

  POINT_T readPoint(const GeoArrowGeometryNode* point_node) {
    assert(point_node->geometry_type == GEOARROW_GEOMETRY_TYPE_POINT);
    bool swap_endian = (point_node->flags & GEOARROW_GEOMETRY_NODE_FLAG_SWAP_ENDIAN);
    POINT_T point;

    for (int dim = 0; dim < POINT_T::n_dim; ++dim) {
      uint64_t coord_int;
      memcpy(&coord_int, point_node->coords[dim], sizeof(uint64_t));

      if (swap_endian) {
        coord_int = __builtin_bswap64(coord_int);
      }

      double coord_double;
      memcpy(&coord_double, &coord_int, sizeof(double));

      point.set_coordinate(dim, coord_double);
    }
    return point;
  }

  const GeoArrowGeometryNode* processLineString(const GeoArrowGeometryNode* node,
                                                mbr_t* mbr) {
    assert(node->geometry_type == GEOARROW_GEOMETRY_TYPE_LINESTRING);
    const uint8_t* p_coord[n_dim];
    int32_t d_coord[n_dim];

    for (int dim = 0; dim < n_dim; dim++) {
      p_coord[dim] = node->coords[dim];
      d_coord[dim] = node->coord_stride[dim];
    }

    num_points.push_back(node->size);

    for (uint32_t j = 0; j < node->size; j++) {
      POINT_T point;

      for (int dim = 0; dim < n_dim; dim++) {
        auto* coord = p_coord[dim];
        uint64_t coord_int;
        double coord_double;

        coord_int = *reinterpret_cast<const uint64_t*>(coord);
        if (node->flags & GEOARROW_GEOMETRY_NODE_FLAG_SWAP_ENDIAN) {
          coord_int = __builtin_bswap64(coord_int);
        }
        coord_double = *reinterpret_cast<double*>(&coord_int);
        point.set_coordinate(dim, coord_double);
        p_coord[dim] += d_coord[dim];
      }
      vertices.push_back(point);
      if (mbr != nullptr) {
        mbr->Expand(point.as_float());
      }
    }
    return node + 1;
  }
};

template <typename POINT_T, typename INDEX_T>
struct DeviceParsedGeometries {
  constexpr static int n_dim = POINT_T::n_dim;
  using mbr_t = Box<Point<float, n_dim>>;
  // will be moved to DeviceGeometries
  rmm::device_uvector<GeometryType> feature_types{0, rmm::cuda_stream_default};
  // These are temp vectors during parsing, which will be used to calculate offsets
  rmm::device_uvector<INDEX_T> num_geoms{0, rmm::cuda_stream_default};
  rmm::device_uvector<INDEX_T> num_parts{0, rmm::cuda_stream_default};
  rmm::device_uvector<INDEX_T> num_rings{0, rmm::cuda_stream_default};
  rmm::device_uvector<INDEX_T> num_points{0, rmm::cuda_stream_default};
  // will be moved to DeviceGeometries
  rmm::device_uvector<POINT_T> vertices{0, rmm::cuda_stream_default};
  rmm::device_uvector<mbr_t> mbrs{0, rmm::cuda_stream_default};

  void Clear(rmm::cuda_stream_view stream, bool free_memory = true) {
    feature_types.resize(0, stream);
    num_geoms.resize(0, stream);
    num_parts.resize(0, stream);
    num_rings.resize(0, stream);
    num_points.resize(0, stream);
    vertices.resize(0, stream);
    mbrs.resize(0, stream);
    if (free_memory) {
      feature_types.shrink_to_fit(stream);
      num_geoms.shrink_to_fit(stream);
      num_parts.shrink_to_fit(stream);
      num_rings.shrink_to_fit(stream);
      num_points.shrink_to_fit(stream);
      vertices.shrink_to_fit(stream);
      mbrs.shrink_to_fit(stream);
    }
  }

  void Append(rmm::cuda_stream_view stream,
              const std::vector<HostParsedGeometries<POINT_T, INDEX_T>>& host_geoms) {
    size_t sz_feature_types = 0;
    size_t sz_num_geoms = 0;
    size_t sz_num_parts = 0;
    size_t sz_num_rings = 0;
    size_t sz_num_points = 0;
    size_t sz_vertices = 0;
    size_t sz_mbrs = 0;

    for (auto& geoms : host_geoms) {
      sz_feature_types += geoms.feature_types.size();
      sz_num_geoms += geoms.num_geoms.size();
      sz_num_parts += geoms.num_parts.size();
      sz_num_rings += geoms.num_rings.size();
      sz_num_points += geoms.num_points.size();
      sz_vertices += geoms.vertices.size();
      sz_mbrs += geoms.mbrs.size();
    }
    size_t prev_sz_feature_types = feature_types.size();
    size_t prev_sz_num_geoms = num_geoms.size();
    size_t prev_sz_num_parts = num_parts.size();
    size_t prev_sz_num_rings = num_rings.size();
    size_t prev_sz_num_points = num_points.size();
    size_t prev_sz_vertices = vertices.size();
    size_t prev_sz_mbrs = mbrs.size();

    GPUSPATIAL_LOG_DEBUG(
        "Available %lu MB, num parts %lu MB (new %lu MB), num rings %lu MB (new %lu MB), num points %lu MB (new %lu MB), vertices %lu MB (new %lu MB), mbrs %lu MB (new %lu MB)",
        rmm::available_device_memory().first / 1024 / 1024,
        prev_sz_num_parts * sizeof(INDEX_T) / 1024 / 1024,
        sz_num_parts * sizeof(INDEX_T) / 1024 / 1024,
        prev_sz_num_rings * sizeof(INDEX_T) / 1024 / 1024,
        sz_num_rings * sizeof(INDEX_T) / 1024 / 1024,
        prev_sz_num_points * sizeof(INDEX_T) / 1024 / 1024,
        sz_num_points * sizeof(INDEX_T) / 1024 / 1024,
        prev_sz_vertices * sizeof(POINT_T) / 1024 / 1024,
        sz_vertices * sizeof(POINT_T) / 1024 / 1024,
        prev_sz_mbrs * sizeof(mbr_t) / 1024 / 1024,
        sz_mbrs * sizeof(mbr_t) / 1024 / 1024);

    feature_types.resize(feature_types.size() + sz_feature_types, stream);
    num_geoms.resize(num_geoms.size() + sz_num_geoms, stream);
    num_parts.resize(num_parts.size() + sz_num_parts, stream);
    num_rings.resize(num_rings.size() + sz_num_rings, stream);
    num_points.resize(num_points.size() + sz_num_points, stream);
    vertices.resize(vertices.size() + sz_vertices, stream);
    mbrs.resize(mbrs.size() + sz_mbrs, stream);

    for (auto& geoms : host_geoms) {
      detail::async_copy_h2d(stream, geoms.feature_types.data(),
                             feature_types.data() + prev_sz_feature_types,
                             geoms.feature_types.size());
      detail::async_copy_h2d(stream, geoms.num_geoms.data(),
                             num_geoms.data() + prev_sz_num_geoms,
                             geoms.num_geoms.size());
      detail::async_copy_h2d(stream, geoms.num_parts.data(),
                             num_parts.data() + prev_sz_num_parts,
                             geoms.num_parts.size());
      detail::async_copy_h2d(stream, geoms.num_rings.data(),
                             num_rings.data() + prev_sz_num_rings,
                             geoms.num_rings.size());
      detail::async_copy_h2d(stream, geoms.num_points.data(),
                             num_points.data() + prev_sz_num_points,
                             geoms.num_points.size());
      detail::async_copy_h2d(stream, geoms.vertices.data(),
                             vertices.data() + prev_sz_vertices, geoms.vertices.size());
      detail::async_copy_h2d(stream, geoms.mbrs.data(), mbrs.data() + prev_sz_mbrs,
                             geoms.mbrs.size());
      prev_sz_feature_types += geoms.feature_types.size();
      prev_sz_num_geoms += geoms.num_geoms.size();
      prev_sz_num_parts += geoms.num_parts.size();
      prev_sz_num_rings += geoms.num_rings.size();
      prev_sz_num_points += geoms.num_points.size();
      prev_sz_vertices += geoms.vertices.size();
      prev_sz_mbrs += geoms.mbrs.size();
    }
  }
};
}  // namespace detail

template <typename POINT_T, typename INDEX_T>
class ParallelWkbLoader {
  constexpr static int n_dim = POINT_T::n_dim;
  using scalar_t = typename POINT_T::scalar_t;
  // using low precision for memory saving
  using mbr_t = Box<Point<float, n_dim>>;

 public:
  struct Config {
    // How many rows of WKBs to process in one chunk
    // This value affects the peak memory usage and overheads
    int chunk_size = 16 * 1024;
  };

  ParallelWkbLoader()
      : thread_pool_(std::make_shared<ThreadPool>(std::thread::hardware_concurrency())) {}

  ParallelWkbLoader(const std::shared_ptr<ThreadPool>& thread_pool)
      : thread_pool_(thread_pool) {}

  void Init(const Config& config = Config()) {
    ArrowArrayViewInitFromType(&array_view_, NANOARROW_TYPE_BINARY);
    config_ = config;
    geometry_type_ = GeometryType::kNull;
  }

  void Clear(rmm::cuda_stream_view stream) {
    geometry_type_ = GeometryType::kNull;
    geoms_.Clear(stream);
  }

  void Parse(rmm::cuda_stream_view stream, const ArrowArray* array, int64_t offset,
             int64_t length) {
    using host_geometries_t = detail::HostParsedGeometries<POINT_T, INDEX_T>;
    ArrowError arrow_error;
    if (ArrowArrayViewSetArray(&array_view_, array, &arrow_error) != NANOARROW_OK) {
      throw std::runtime_error("ArrowArrayViewSetArray error " +
                               std::string(arrow_error.message));
    }
    auto parallelism = thread_pool_->num_threads();
    auto est_bytes = estimateTotalBytes(array, offset, length);
    auto free_memory = detail::get_free_physical_memory_linux();
    uint32_t est_n_chunks = est_bytes / free_memory + 1;
    uint32_t chunk_size = (length + est_n_chunks - 1) / est_n_chunks;

    GPUSPATIAL_LOG_INFO(
        "Parsing %ld rows, est arrow size %ld MB, free memory %lld, chunk size %u\n",
        length, est_bytes / 1024 / 1024, free_memory / 1024 / 1024, chunk_size);

    auto n_chunks = (length + chunk_size - 1) / chunk_size;
    Stopwatch sw;
    double t_fetch_type = 0, t_parse = 0, t_copy = 0;

    sw.start();
    updateGeometryType(offset, length);
    sw.stop();
    t_fetch_type = sw.ms();

    bool multi = geometry_type_ == GeometryType::kMultiPoint ||
                 geometry_type_ == GeometryType::kMultiLineString ||
                 geometry_type_ == GeometryType::kMultiPolygon;
    bool has_geometry_collection = geometry_type_ == GeometryType::kGeometryCollection;
    bool create_mbr = geometry_type_ != GeometryType::kPoint;

    // reserve space
    geoms_.vertices.reserve(est_bytes / sizeof(POINT_T), stream);
    if (create_mbr) geoms_.mbrs.reserve(array->length, stream);

    // Batch processing to reduce the peak memory usage
    for (int64_t chunk = 0; chunk < n_chunks; chunk++) {
      auto chunk_start = chunk * chunk_size;
      auto chunk_end = std::min(length, (chunk + 1) * chunk_size);
      auto work_size = chunk_end - chunk_start;

      std::vector<std::future<host_geometries_t>> pending_local_geoms;
      auto thread_work_size = (work_size + parallelism - 1) / parallelism;
      sw.start();
      // Each thread will parse in parallel and store results sequentially
      for (int thread_idx = 0; thread_idx < parallelism; thread_idx++) {
        auto run = [&](int tid) {
          // FIXME: SetDevice
          auto thread_work_start = chunk_start + tid * thread_work_size;
          auto thread_work_end =
              std::min(chunk_end, thread_work_start + thread_work_size);
          host_geometries_t local_geoms(multi, has_geometry_collection, create_mbr);
          GeoArrowWKBReader reader;
          GeoArrowError error;
          GEOARROW_THROW_NOT_OK(nullptr, GeoArrowWKBReaderInit(&reader));

          for (uint32_t work_offset = thread_work_start; work_offset < thread_work_end;
               work_offset++) {
            auto arrow_offset = work_offset + offset;
            // handle null value
            if (ArrowArrayViewIsNull(&array_view_, arrow_offset)) {
              local_geoms.AddGeometry(nullptr);
            } else {
              auto item = ArrowArrayViewGetBytesUnsafe(&array_view_, arrow_offset);
              GeoArrowGeometryView geom;

              GEOARROW_THROW_NOT_OK(
                  &error,
                  GeoArrowWKBReaderRead(&reader, {item.data.as_uint8, item.size_bytes},
                                        &geom, &error));
              local_geoms.AddGeometry(&geom);
            }
          }

          return std::move(local_geoms);
        };
        pending_local_geoms.push_back(std::move(thread_pool_->enqueue(run, thread_idx)));
      }

      std::vector<host_geometries_t> local_geoms;
      for (auto& fu : pending_local_geoms) {
        local_geoms.push_back(std::move(fu.get()));
      }
      sw.stop();
      t_parse += sw.ms();
      sw.start();
      geoms_.Append(stream, local_geoms);
      stream.synchronize();
      sw.stop();
      t_copy += sw.ms();
    }
    GPUSPATIAL_LOG_INFO(
        "ParallelWkbLoader::Parse: fetched type in %.3f ms, parsed in %.3f ms, copied in "
        "%.3f ms",
        t_fetch_type, t_parse, t_copy);
  }

  DeviceGeometries<POINT_T, INDEX_T> Finish(rmm::cuda_stream_view stream) {
    Stopwatch sw;
    GPUSPATIAL_LOG_INFO(
        "Finish building, type %s, num parts %lu, num rings %lu, num points %lu, vertices %lu",
        GeometryTypeToString(geometry_type_), geoms_.num_parts.size(),
        geoms_.num_rings.size(), geoms_.num_points.size(), geoms_.vertices.size());

    sw.start();
    // Calculate one by one to reduce peak memory
    rmm::device_uvector<INDEX_T> ps_num_geoms(0, stream);
    calcPrefixSum(stream, geoms_.num_geoms, ps_num_geoms);

    rmm::device_uvector<INDEX_T> ps_num_parts(0, stream);
    calcPrefixSum(stream, geoms_.num_parts, ps_num_parts);

    rmm::device_uvector<INDEX_T> ps_num_rings(0, stream);
    calcPrefixSum(stream, geoms_.num_rings, ps_num_rings);

    rmm::device_uvector<INDEX_T> ps_num_points(0, stream);
    calcPrefixSum(stream, geoms_.num_points, ps_num_points);

    DeviceGeometries<POINT_T, INDEX_T> device_geometries;

    if constexpr (std::is_same_v<scalar_t, double>) {
      thrust::transform(rmm::exec_policy_nosync(stream), geoms_.mbrs.begin(),
                        geoms_.mbrs.end(), geoms_.mbrs.begin(),
                        [] __device__(const mbr_t& mbr) -> mbr_t {
                          Point<float, n_dim> min_corner, max_corner;
                          for (int dim = 0; dim < n_dim; dim++) {
                            auto min_val = mbr.get_min(dim);
                            auto max_val = mbr.get_max(dim);
                            // Two rounds of next_float to ensure the MBR fully covers the
                            // original geometry, refer to RayJoin paper
                            min_corner[dim] = next_float_from_double(min_val, -1, 2);
                            max_corner[dim] = next_float_from_double(max_val, 1, 2);
                          }
                          return {min_corner, max_corner};
                        });
    }
    device_geometries.mbrs_ = std::move(geoms_.mbrs);
    device_geometries.type_ = geometry_type_;
    device_geometries.points_ = std::move(geoms_.vertices);

    // move type specific data
    switch (geometry_type_) {
      case GeometryType::kPoint: {
        // Do nothing, all points have been moved
        break;
      }
      case GeometryType::kLineString: {
        device_geometries.offsets_.line_string_offsets.ps_num_points =
            std::move(ps_num_points);
        break;
      }
      case GeometryType::kPolygon: {
        device_geometries.offsets_.polygon_offsets.ps_num_rings = std::move(ps_num_rings);
        device_geometries.offsets_.polygon_offsets.ps_num_points =
            std::move(ps_num_points);
        break;
      }
      case GeometryType::kMultiPoint: {
        device_geometries.offsets_.multi_point_offsets.ps_num_points =
            std::move(ps_num_points);
        break;
      }
      case GeometryType::kMultiLineString: {
        device_geometries.offsets_.multi_line_string_offsets.ps_num_parts =
            std::move(ps_num_parts);
        device_geometries.offsets_.multi_line_string_offsets.ps_num_points =
            std::move(ps_num_points);
        break;
      }
      case GeometryType::kMultiPolygon: {
        device_geometries.offsets_.multi_polygon_offsets.ps_num_parts =
            std::move(ps_num_parts);
        device_geometries.offsets_.multi_polygon_offsets.ps_num_rings =
            std::move(ps_num_rings);
        device_geometries.offsets_.multi_polygon_offsets.ps_num_points =
            std::move(ps_num_points);
        break;
      }
      case GeometryType::kGeometryCollection: {
        device_geometries.offsets_.geom_collection_offsets.feature_types =
            std::move(geoms_.feature_types);
        device_geometries.offsets_.geom_collection_offsets.ps_num_geoms =
            std::move(ps_num_geoms);
        device_geometries.offsets_.geom_collection_offsets.ps_num_parts =
            std::move(ps_num_parts);
        device_geometries.offsets_.geom_collection_offsets.ps_num_rings =
            std::move(ps_num_rings);
        device_geometries.offsets_.geom_collection_offsets.ps_num_points =
            std::move(ps_num_points);
        break;
      }
    }
    Clear(stream);
    stream.synchronize();
    sw.stop();
    GPUSPATIAL_LOG_INFO("Finish building DeviceGeometries in %.3f ms", sw.ms());
    return std::move(device_geometries);
  }

 private:
  Config config_;
  ArrowArrayView array_view_;
  GeometryType geometry_type_;
  detail::DeviceParsedGeometries<POINT_T, INDEX_T> geoms_;
  std::shared_ptr<ThreadPool> thread_pool_;

  void updateGeometryType(int64_t offset, int64_t length) {
    if (geometry_type_ == GeometryType::kGeometryCollection) {
      // it's already the most generic type
      return;
    }

    std::vector<bool> type_flags(8 /*WKB types*/, false);
    std::vector<std::thread> workers;
    auto parallelism = thread_pool_->num_threads();
    auto thread_work_size = (length + parallelism - 1) / parallelism;
    std::vector<std::future<void>> futures;

    for (int thread_idx = 0; thread_idx < parallelism; thread_idx++) {
      auto run = [&](int tid) {
        auto thread_work_start = tid * thread_work_size;
        auto thread_work_end = std::min(length, thread_work_start + thread_work_size);
        GeoArrowWKBReader reader;
        GeoArrowError error;
        GEOARROW_THROW_NOT_OK(nullptr, GeoArrowWKBReaderInit(&reader));

        for (uint32_t work_offset = thread_work_start; work_offset < thread_work_end;
             work_offset++) {
          auto arrow_offset = work_offset + offset;
          // handle null value
          if (ArrowArrayViewIsNull(&array_view_, arrow_offset)) {
            continue;
          }
          auto item = ArrowArrayViewGetBytesUnsafe(&array_view_, arrow_offset);
          auto* s = (struct detail::WKBReaderPrivate*)reader.private_data;

          s->data = item.data.as_uint8;
          s->data0 = s->data;
          s->size_bytes = item.size_bytes;

          NANOARROW_THROW_NOT_OK(detail::WKBReaderReadEndian(s, &error));
          uint32_t geometry_type;
          NANOARROW_THROW_NOT_OK(detail::WKBReaderReadUInt32(s, &geometry_type, &error));
          if (geometry_type > 7) {
            throw std::runtime_error(
                "Extended WKB types are not currently supported, type = " +
                std::to_string(geometry_type));
          }
          assert(geometry_type < type_flags.size());
          type_flags[geometry_type] = true;
        }
      };
      futures.push_back(std::move(thread_pool_->enqueue(run, thread_idx)));
    }
    for (auto& fu : futures) {
      fu.get();
    }

    std::unordered_set<GeometryType> types;
    // include existing geometry type
    if (geometry_type_ != GeometryType::kNull) {
      types.insert(geometry_type_);
    }

    for (int i = 1; i <= 7; i++) {
      if (type_flags[i]) {
        types.insert(static_cast<GeometryType>(i));
      }
    }

    GeometryType final_type;
    // Infer a generic type that can represent the current and previous types
    switch (types.size()) {
      case 0:
        final_type = GeometryType::kNull;
        break;
      case 1:
        final_type = *types.begin();
        break;
      case 2: {
        if (types.count(GeometryType::kPoint) && types.count(GeometryType::kMultiPoint)) {
          final_type = GeometryType::kMultiPoint;
        } else if (types.count(GeometryType::kLineString) &&
                   types.count(GeometryType::kMultiLineString)) {
          final_type = GeometryType::kMultiLineString;
        } else if (types.count(GeometryType::kPolygon) &&
                   types.count(GeometryType::kMultiPolygon)) {
          final_type = GeometryType::kMultiPolygon;
        } else {
          final_type = GeometryType::kGeometryCollection;
        }
        break;
      }
      default:
        final_type = GeometryType::kGeometryCollection;
    }
    geometry_type_ = final_type;
  }

  template <typename T>
  void appendVector(rmm::cuda_stream_view stream, rmm::device_uvector<T>& d_vec,
                    const std::vector<T>& h_vec) {
    if (h_vec.empty()) return;
    auto prev_size = d_vec.size();
    d_vec.resize(prev_size + h_vec.size(), stream);
    detail::async_copy_h2d(stream, h_vec.data(), d_vec.data() + prev_size, h_vec.size());
  }

  template <typename T>
  void calcPrefixSum(rmm::cuda_stream_view stream, rmm::device_uvector<T>& nums,
                     rmm::device_uvector<T>& ps) {
    if (nums.size() == 0) return;
    ps.resize(nums.size() + 1, stream);
    ps.set_element_to_zero_async(0, stream);
    thrust::inclusive_scan(rmm::exec_policy_nosync(stream), nums.begin(), nums.end(),
                           ps.begin() + 1);
    nums.resize(0, stream);
    nums.shrink_to_fit(stream);
  }

  size_t estimateTotalBytes(const ArrowArray* array, int64_t offset, int64_t length) {
    ArrowError arrow_error;
    if (ArrowArrayViewSetArray(&array_view_, array, &arrow_error) != NANOARROW_OK) {
      throw std::runtime_error("ArrowArrayViewSetArray error " +
                               std::string(arrow_error.message));
    }
    size_t total_bytes = 0;
    for (int64_t i = 0; i < length; i++) {
      if (!ArrowArrayViewIsNull(&array_view_, offset + i)) {
        auto item = ArrowArrayViewGetBytesUnsafe(&array_view_, offset + i);
        total_bytes += item.size_bytes - 1      // byte order
                       - 2 * sizeof(uint32_t);  // type + size
      }
    }
    return total_bytes;
  }
};
}  // namespace gpuspatial
