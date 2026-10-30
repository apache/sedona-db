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

#define GPUSPATIAL_RELATE_TEST
#include "../src/rt/shaders/shader_id.hpp"
#include "gpuspatial/loader/device_geometries.hpp"
#include "gpuspatial/loader/parallel_wkb_loader.hpp"
#include "gpuspatial/relate/relate_engine.cuh"
#include "gpuspatial/rt/launch_parameters.cuh"
#include "gpuspatial/rt/rt_engine.hpp"
#include "test_common.hpp"

#include <geos/geom/Geometry.h>
#include <geos/geom/IntersectionMatrix.h>
#include <geos/io/WKTReader.h>
#include <gtest/gtest.h>
#include <thrust/sequence.h>

#include <memory>
#include <string>
#include <vector>

namespace gpuspatial {
namespace {

using point_t = Point<double, 2>;
using index_t = uint32_t;
using loader_t = ParallelWkbLoader<point_t, index_t>;

std::shared_ptr<RTEngine> GetTestRTEngine() {
  static auto rt_engine = [] {
    auto engine = std::make_shared<RTEngine>();
    std::string ptx_root = TestUtils::GetTestShaderPath();
    auto config = get_default_rt_config(ptx_root);
    engine->Init(config);
    return engine;
  }();
  return rt_engine;
}

DeviceGeometries<point_t, index_t> LoadDeviceGeoms(const std::vector<std::string>& wkts) {
  nanoarrow::UniqueArrayStream stream;
  ArrayStreamFromWKT({wkts}, GEOARROW_TYPE_WKB, stream.get());
  nanoarrow::UniqueArray array;
  nanoarrow::UniqueSchema schema;
  nanoarrow::UniqueArrayView array_view;
  ArrowError error;

  if (ArrowArrayStreamGetSchema(stream.get(), schema.get(), &error) != NANOARROW_OK ||
      ArrowArrayStreamGetNext(stream.get(), array.get(), &error) != NANOARROW_OK ||
      ArrowArrayViewInitFromSchema(array_view.get(), schema.get(), &error) !=
          NANOARROW_OK ||
      ArrowArrayViewSetArray(array_view.get(), array.get(), &error) != NANOARROW_OK) {
    throw std::runtime_error(error.message);
  }

  loader_t loader;
  auto cuda_stream = rmm::cuda_stream_default;
  loader.Init();
  loader.Parse(cuda_stream, array_view.get(), 0, array->length);
  auto dev_geoms = loader.Finish(cuda_stream);
  cuda_stream.synchronize();
  return dev_geoms;
}

void RunPolygonPass(const rmm::cuda_stream_view& cuda_stream,
                    const DeviceGeometries<point_t, index_t>& probe_geoms,
                    const DeviceGeometries<point_t, index_t>& bvh_geoms,
                    rmm::device_uvector<index_t>& d_probe_ids,
                    rmm::device_uvector<index_t>& d_bvh_ids,
                    rmm::device_uvector<int>& d_ims,
                    rmm::device_uvector<uint32_t>& d_finished_count, uint32_t query_size,
                    uint32_t threads_per_pair, bool transpose) {
  auto rt_engine = GetTestRTEngine();
  RelateEngine<point_t, index_t> engine(&bvh_geoms, rt_engine.get());
  auto bvh_poly_view =
      bvh_geoms.GetGeometryArrayView<PolygonArrayView<point_t, index_t>>();
  size_t n_bvh = bvh_poly_view.size();
  rmm::device_uvector<uint32_t> uniq_bvh_ids(n_bvh, cuda_stream);
  thrust::sequence(rmm::exec_policy_nosync(cuda_stream), uniq_bvh_ids.begin(),
                   uniq_bvh_ids.end(), 0);

  rmm::device_buffer bvh_buffer(0, cuda_stream);
  rmm::device_uvector<index_t> aabb_poly_ids(0, cuda_stream);
  rmm::device_uvector<index_t> aabb_ring_ids(0, cuda_stream);
  rmm::device_uvector<thrust::pair<index_t, index_t>> aabb_vertex_offsets(0, cuda_stream);

  auto handle =
      engine.BuildBVH(cuda_stream, bvh_poly_view, ArrayView<uint32_t>(uniq_bvh_ids), 32,
                      bvh_buffer, aabb_poly_ids, aabb_ring_ids, aabb_vertex_offsets);

  CUDA_CHECK(cudaMemsetAsync(d_finished_count.data(), 0, query_size * sizeof(uint32_t),
                             cuda_stream.value()));

  detail::LaunchParamsPolygonPolygonQuery<point_t, index_t> params;
  if (probe_geoms.get_geometry_type() == GeometryType::kPolygon) {
    params.probe_polygons =
        probe_geoms.GetGeometryArrayView<PolygonArrayView<point_t, index_t>>();
  } else {
    params.probe_multi_polygons =
        probe_geoms.GetGeometryArrayView<MultiPolygonArrayView<point_t, index_t>>();
  }
  params.bvh_polygons = bvh_poly_view;
  params.uniq_bvh_ids = ArrayView<index_t>(uniq_bvh_ids);
  params.query_probe_ids = d_probe_ids.data();
  params.query_bvh_ids = d_bvh_ids.data();
  params.query_size = query_size;
  params.IMs = ArrayView<int>(d_ims);
  params.finished_count = d_finished_count.data();
  params.transpose = transpose;
  params.handle = handle;
  params.aabb_poly_ids = ArrayView<index_t>(aabb_poly_ids);
  params.aabb_ring_ids = ArrayView<index_t>(aabb_ring_ids);
  params.aabb_vertex_offsets =
      ArrayView<thrust::pair<index_t, index_t>>(aabb_vertex_offsets);

  rmm::device_buffer params_buffer(sizeof(params), cuda_stream);
  CUDA_CHECK(cudaMemcpyAsync(params_buffer.data(), &params, sizeof(params),
                             cudaMemcpyHostToDevice, cuda_stream.value()));

  dim3 launch_dim(query_size, threads_per_pair, 1);
  rt_engine->Render(cuda_stream, GetPolygonPolygonQueryShaderId<point_t>(), launch_dim,
                    ArrayView<char>(reinterpret_cast<char*>(params_buffer.data()),
                                    params_buffer.size()));
  cuda_stream.synchronize();
}

void RunMultiPolygonPass(const rmm::cuda_stream_view& cuda_stream,
                         const DeviceGeometries<point_t, index_t>& probe_geoms,
                         const DeviceGeometries<point_t, index_t>& bvh_geoms,
                         rmm::device_uvector<index_t>& d_probe_ids,
                         rmm::device_uvector<index_t>& d_bvh_ids,
                         rmm::device_uvector<int>& d_ims,
                         rmm::device_uvector<uint32_t>& d_finished_count,
                         uint32_t query_size, uint32_t threads_per_pair, bool transpose) {
  auto rt_engine = GetTestRTEngine();
  RelateEngine<point_t, index_t> engine(&bvh_geoms, rt_engine.get());
  auto bvh_mp_view =
      bvh_geoms.GetGeometryArrayView<MultiPolygonArrayView<point_t, index_t>>();
  size_t n_bvh = bvh_mp_view.size();
  rmm::device_uvector<uint32_t> uniq_bvh_ids(n_bvh, cuda_stream);
  thrust::sequence(rmm::exec_policy_nosync(cuda_stream), uniq_bvh_ids.begin(),
                   uniq_bvh_ids.end(), 0);

  rmm::device_buffer bvh_buffer(0, cuda_stream);
  rmm::device_uvector<index_t> aabb_multi_poly_ids(0, cuda_stream);
  rmm::device_uvector<index_t> aabb_part_ids(0, cuda_stream);
  rmm::device_uvector<index_t> aabb_ring_ids(0, cuda_stream);
  rmm::device_uvector<thrust::pair<index_t, index_t>> aabb_vertex_offsets(0, cuda_stream);
  rmm::device_uvector<index_t> uniq_part_begins(0, cuda_stream);

  auto handle =
      engine.BuildBVH(cuda_stream, bvh_mp_view, ArrayView<uint32_t>(uniq_bvh_ids), 32,
                      bvh_buffer, aabb_multi_poly_ids, aabb_part_ids, aabb_ring_ids,
                      aabb_vertex_offsets, uniq_part_begins);

  CUDA_CHECK(cudaMemsetAsync(d_finished_count.data(), 0, query_size * sizeof(uint32_t),
                             cuda_stream.value()));

  detail::LaunchParamsMultiPolygonPolygonQuery<point_t, index_t> params;
  if (probe_geoms.get_geometry_type() == GeometryType::kPolygon) {
    params.probe_polygons =
        probe_geoms.GetGeometryArrayView<PolygonArrayView<point_t, index_t>>();
  } else {
    params.probe_multi_polygons =
        probe_geoms.GetGeometryArrayView<MultiPolygonArrayView<point_t, index_t>>();
  }
  params.bvh_multi_polygons = bvh_mp_view;
  params.uniq_bvh_ids = ArrayView<index_t>(uniq_bvh_ids);
  params.query_probe_ids = d_probe_ids.data();
  params.query_bvh_ids = d_bvh_ids.data();
  params.query_size = query_size;
  params.uniq_part_begins = ArrayView<index_t>(uniq_part_begins);
  params.IMs = ArrayView<int>(d_ims);
  params.finished_count = d_finished_count.data();
  params.transpose = transpose;
  params.handle = handle;
  params.aabb_multi_poly_ids = ArrayView<index_t>(aabb_multi_poly_ids);
  params.aabb_part_ids = ArrayView<index_t>(aabb_part_ids);
  params.aabb_ring_ids = ArrayView<index_t>(aabb_ring_ids);
  params.aabb_vertex_offsets =
      ArrayView<thrust::pair<index_t, index_t>>(aabb_vertex_offsets);

  rmm::device_buffer params_buffer(sizeof(params), cuda_stream);
  CUDA_CHECK(cudaMemcpyAsync(params_buffer.data(), &params, sizeof(params),
                             cudaMemcpyHostToDevice, cuda_stream.value()));

  dim3 launch_dim(query_size, threads_per_pair, 1);
  rt_engine->Render(cuda_stream, GetMultiPolygonPolygonQueryShaderId<point_t>(),
                    launch_dim,
                    ArrayView<char>(reinterpret_cast<char*>(params_buffer.data()),
                                    params_buffer.size()));
  cuda_stream.synchronize();
}

void TestPolygonPolygonShader(const std::string& probe_wkt, const std::string& bvh_wkt,
                              uint32_t threads_per_pair) {
  auto cuda_stream = rmm::cuda_stream_default;
  auto probe_geoms = LoadDeviceGeoms({probe_wkt});
  auto bvh_geoms = LoadDeviceGeoms({bvh_wkt});

  rmm::device_uvector<index_t> d_probe_ids(1, cuda_stream);
  rmm::device_uvector<index_t> d_bvh_ids(1, cuda_stream);
  rmm::device_uvector<int> d_ims(1, cuda_stream);
  rmm::device_uvector<uint32_t> d_finished_count(1, cuda_stream);

  d_probe_ids.set_element_to_zero_async(0, cuda_stream);
  d_bvh_ids.set_element_to_zero_async(0, cuda_stream);
  d_ims.set_element_to_zero_async(0, cuda_stream);

  // Pass 1: probe = A, BVH = B, transpose = false
  RunPolygonPass(cuda_stream, probe_geoms, bvh_geoms, d_probe_ids, d_bvh_ids, d_ims,
                 d_finished_count, 1, threads_per_pair, false);

  // Pass 2: probe = B, BVH = A, transpose = true (accumulate transposed IM)
  RunPolygonPass(cuda_stream, bvh_geoms, probe_geoms, d_bvh_ids, d_probe_ids, d_ims,
                 d_finished_count, 1, threads_per_pair, true);

  int gpu_im = d_ims.element(0, cuda_stream);
  char gpu_im_str[10];
  IntersectionMatrix::ToString(gpu_im, gpu_im_str);

  geos::io::WKTReader reader;
  auto geos_probe = reader.read(probe_wkt);
  auto geos_bvh = reader.read(bvh_wkt);
  auto matrix = geos_probe->relate(geos_bvh.get());
  std::string expected_im = matrix->toString();

  EXPECT_STREQ(expected_im.c_str(), gpu_im_str)
      << "Mismatch for probe: " << probe_wkt << "\nBVH: " << bvh_wkt
      << "\nthreads_per_pair: " << threads_per_pair;
}

void TestMultiPolygonPolygonShader(const std::string& geom_a_wkt,
                                   const std::string& geom_b_wkt,
                                   uint32_t threads_per_pair) {
  auto cuda_stream = rmm::cuda_stream_default;
  auto geoms_a = LoadDeviceGeoms({geom_a_wkt});
  auto geoms_b = LoadDeviceGeoms({geom_b_wkt});

  rmm::device_uvector<index_t> d_ids_a(1, cuda_stream);
  rmm::device_uvector<index_t> d_ids_b(1, cuda_stream);
  rmm::device_uvector<int> d_ims(1, cuda_stream);
  rmm::device_uvector<uint32_t> d_finished_count(1, cuda_stream);

  d_ids_a.set_element_to_zero_async(0, cuda_stream);
  d_ids_b.set_element_to_zero_async(0, cuda_stream);
  d_ims.set_element_to_zero_async(0, cuda_stream);

  bool a_is_poly = geoms_a.get_geometry_type() == GeometryType::kPolygon;
  bool b_is_poly = geoms_b.get_geometry_type() == GeometryType::kPolygon;

  // Pass 1: probe = A, BVH = B, transpose = false
  if (b_is_poly) {
    RunPolygonPass(cuda_stream, geoms_a, geoms_b, d_ids_a, d_ids_b, d_ims,
                   d_finished_count, 1, threads_per_pair, false);
  } else {
    RunMultiPolygonPass(cuda_stream, geoms_a, geoms_b, d_ids_a, d_ids_b, d_ims,
                        d_finished_count, 1, threads_per_pair, false);
  }

  // Pass 2: probe = B, BVH = A, transpose = true
  if (a_is_poly) {
    RunPolygonPass(cuda_stream, geoms_b, geoms_a, d_ids_b, d_ids_a, d_ims,
                   d_finished_count, 1, threads_per_pair, true);
  } else {
    RunMultiPolygonPass(cuda_stream, geoms_b, geoms_a, d_ids_b, d_ids_a, d_ims,
                        d_finished_count, 1, threads_per_pair, true);
  }

  int gpu_im = d_ims.element(0, cuda_stream);
  char gpu_im_str[10];
  IntersectionMatrix::ToString(gpu_im, gpu_im_str);

  geos::io::WKTReader reader;
  auto geos_a = reader.read(geom_a_wkt);
  auto geos_b = reader.read(geom_b_wkt);
  auto matrix = geos_a->relate(geos_b.get());
  std::string expected_im = matrix->toString();

  EXPECT_STREQ(expected_im.c_str(), gpu_im_str)
      << "Mismatch for A: " << geom_a_wkt << "\nB: " << geom_b_wkt
      << "\nthreads_per_pair: " << threads_per_pair;
}

}  // namespace

// =============================================================================
// Polygon-Polygon Shader Unit Tests
// =============================================================================

TEST(PolygonPolygonShaderTest, DisjointMBR) {
  std::string a = "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))";
  std::string b = "POLYGON ((20 20, 30 20, 30 30, 20 30, 20 20))";
  for (uint32_t threads : {1u, 4u, 32u}) {
    TestPolygonPolygonShader(a, b, threads);
  }
}

TEST(PolygonPolygonShaderTest, DisjointIntersectingMBR) {
  std::string a = "POLYGON ((0 0, 10 0, 0 10, 0 0))";
  std::string b = "POLYGON ((10 10, 1 10, 10 1, 10 10))";
  for (uint32_t threads : {1u, 4u, 32u}) {
    TestPolygonPolygonShader(a, b, threads);
  }
}

TEST(PolygonPolygonShaderTest, VertexTouch) {
  std::string a = "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))";
  std::string b = "POLYGON ((10 10, 20 10, 20 20, 10 20, 10 10))";
  for (uint32_t threads : {1u, 4u, 32u}) {
    TestPolygonPolygonShader(a, b, threads);
  }
}

TEST(PolygonPolygonShaderTest, CollinearEdgeTouch) {
  std::string a = "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))";
  std::string b = "POLYGON ((10 0, 20 0, 20 10, 10 10, 10 0))";
  for (uint32_t threads : {1u, 4u, 32u}) {
    TestPolygonPolygonShader(a, b, threads);
  }
}

TEST(PolygonPolygonShaderTest, OverlappingIntersection) {
  std::string a = "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))";
  std::string b = "POLYGON ((5 5, 15 5, 15 15, 5 15, 5 5))";
  for (uint32_t threads : {1u, 4u, 32u}) {
    TestPolygonPolygonShader(a, b, threads);
  }
}

TEST(PolygonPolygonShaderTest, InteriorContainment) {
  std::string outer = "POLYGON ((0 0, 20 0, 20 20, 0 20, 0 0))";
  std::string inner = "POLYGON ((5 5, 15 5, 15 15, 5 15, 5 5))";
  for (uint32_t threads : {1u, 4u, 32u}) {
    TestPolygonPolygonShader(inner, outer, threads);
    TestPolygonPolygonShader(outer, inner, threads);
  }
}

TEST(PolygonPolygonShaderTest, PolygonWithHole) {
  std::string with_hole =
      "POLYGON ((0 0, 30 0, 30 30, 0 30, 0 0), (10 10, 20 10, 20 20, 10 20, 10 10))";
  std::string inside_hole = "POLYGON ((12 12, 18 12, 18 18, 12 18, 12 12))";
  std::string in_solid = "POLYGON ((2 2, 8 2, 8 8, 2 8, 2 2))";

  for (uint32_t threads : {1u, 4u, 32u}) {
    TestPolygonPolygonShader(inside_hole, with_hole, threads);
    TestPolygonPolygonShader(in_solid, with_hole, threads);
  }
}

TEST(PolygonPolygonShaderTest, IdenticalPolygonsCollinearReduction) {
  std::string poly = "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))";
  for (uint32_t threads : {1u, 2u, 4u, 8u, 16u, 32u}) {
    TestPolygonPolygonShader(poly, poly, threads);
  }
}

TEST(PolygonPolygonShaderTest, MultiVertexPolygon) {
  std::string large_poly =
      "POLYGON ((0 0, 2 1, 4 0, 6 1, 8 0, 10 1, 10 10, 8 9, 6 10, 4 9, 2 10, 0 10, "
      "0 0))";
  std::string small_poly = "POLYGON ((3 2, 7 2, 7 8, 3 8, 3 2))";
  for (uint32_t threads : {1u, 4u, 16u, 32u}) {
    TestPolygonPolygonShader(small_poly, large_poly, threads);
  }
}

// =============================================================================
// MultiPolygon-Polygon Shader Unit Tests
// =============================================================================

TEST(MultiPolygonPolygonShaderTest, MultiPolygonBvhWithPolygonProbe) {
  std::string mp_bvh =
      "MULTIPOLYGON (((0 0, 10 0, 10 10, 0 10, 0 0)), "
      "((20 20, 30 20, 30 30, 20 30, 20 20)))";
  std::string probe_overlaps_part1 = "POLYGON ((5 5, 15 5, 15 15, 5 15, 5 5))";
  std::string probe_inside_part2 = "POLYGON ((22 22, 28 22, 28 28, 22 28, 22 22))";
  std::string probe_disjoint = "POLYGON ((40 40, 50 40, 50 50, 40 50, 40 40))";

  for (uint32_t threads : {1u, 4u, 16u}) {
    TestMultiPolygonPolygonShader(probe_overlaps_part1, mp_bvh, threads);
    TestMultiPolygonPolygonShader(probe_inside_part2, mp_bvh, threads);
    TestMultiPolygonPolygonShader(probe_disjoint, mp_bvh, threads);
  }
}

TEST(MultiPolygonPolygonShaderTest, MultiPolygonBvhWithMultiPolygonProbe) {
  std::string mp1 =
      "MULTIPOLYGON (((0 0, 10 0, 10 10, 0 10, 0 0)), "
      "((30 30, 40 30, 40 40, 30 40, 30 30)))";
  std::string mp2 =
      "MULTIPOLYGON (((5 5, 15 5, 15 15, 5 15, 5 5)), "
      "((50 50, 60 50, 60 60, 50 60, 50 50)))";

  for (uint32_t threads : {1u, 4u, 16u}) {
    TestMultiPolygonPolygonShader(mp1, mp2, threads);
  }
}

TEST(MultiPolygonPolygonShaderTest, MultiPartCulling) {
  std::string mp =
      "MULTIPOLYGON ("
      "((0 0, 5 0, 5 5, 0 5, 0 0)), "
      "((10 10, 15 10, 15 15, 10 15, 10 10)), "
      "((20 20, 25 20, 25 25, 20 25, 20 20)), "
      "((30 30, 35 30, 35 35, 30 35, 30 30))"
      ")";
  std::string poly = "POLYGON ((22 22, 27 22, 27 27, 22 27, 22 22))";

  for (uint32_t threads : {1u, 4u, 16u}) {
    TestMultiPolygonPolygonShader(poly, mp, threads);
  }
}

// =============================================================================
// Batch Evaluation Test
// =============================================================================

TEST(PolygonPolygonShaderTest, BatchMultiPairEvaluation) {
  std::vector<std::string> probe_wkts = {
      "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
      "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
      "POLYGON ((5 5, 15 5, 15 15, 5 15, 5 5))",
      "POLYGON ((2 2, 8 2, 8 8, 2 8, 2 2))",
  };
  std::vector<std::string> bvh_wkts = {
      "POLYGON ((20 20, 30 20, 30 30, 20 30, 20 20))",  // disjoint
      "POLYGON ((10 0, 20 0, 20 10, 10 10, 10 0))",     // touch edge
      "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",        // overlap
      "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",        // within
  };

  size_t query_size = probe_wkts.size();
  auto cuda_stream = rmm::cuda_stream_default;

  auto probe_geoms = LoadDeviceGeoms(probe_wkts);
  auto bvh_geoms = LoadDeviceGeoms(bvh_wkts);

  rmm::device_uvector<index_t> d_probe_ids(query_size, cuda_stream);
  rmm::device_uvector<index_t> d_bvh_ids(query_size, cuda_stream);
  thrust::sequence(rmm::exec_policy_nosync(cuda_stream), d_probe_ids.begin(),
                   d_probe_ids.end(), 0);
  thrust::sequence(rmm::exec_policy_nosync(cuda_stream), d_bvh_ids.begin(),
                   d_bvh_ids.end(), 0);

  rmm::device_uvector<int> d_ims(query_size, cuda_stream);
  rmm::device_uvector<uint32_t> d_finished_count(query_size, cuda_stream);
  CUDA_CHECK(
      cudaMemsetAsync(d_ims.data(), 0, query_size * sizeof(int), cuda_stream.value()));

  // Pass 1: probe = probe_geoms, BVH = bvh_geoms, transpose = false
  RunPolygonPass(cuda_stream, probe_geoms, bvh_geoms, d_probe_ids, d_bvh_ids, d_ims,
                 d_finished_count, query_size, 8, false);

  // Pass 2: probe = bvh_geoms, BVH = probe_geoms, transpose = true
  RunPolygonPass(cuda_stream, bvh_geoms, probe_geoms, d_bvh_ids, d_probe_ids, d_ims,
                 d_finished_count, query_size, 8, true);

  std::vector<int> h_ims(query_size);
  CUDA_CHECK(cudaMemcpy(h_ims.data(), d_ims.data(), query_size * sizeof(int),
                        cudaMemcpyDeviceToHost));

  geos::io::WKTReader reader;
  for (size_t i = 0; i < query_size; ++i) {
    char gpu_im_str[10];
    IntersectionMatrix::ToString(h_ims[i], gpu_im_str);

    auto geos_probe = reader.read(probe_wkts[i]);
    auto geos_bvh = reader.read(bvh_wkts[i]);
    auto matrix = geos_probe->relate(geos_bvh.get());
    std::string expected_im = matrix->toString();

    EXPECT_STREQ(expected_im.c_str(), gpu_im_str)
        << "Batch pair " << i << " mismatch for probe: " << probe_wkts[i]
        << " BVH: " << bvh_wkts[i];
  }
}

TEST(PolygonPolygonShaderTest, MicroPolygonEqualsSelf) {
  // Test scale-free FindPolygonInteriorPoint on a polygon with 1e-8 edge length
  std::string micro_wkt =
      "POLYGON ((0 0, 0.00000001 0, 0.00000001 0.00000001, 0 0.00000001, 0 0))";
  TestPolygonPolygonShader(micro_wkt, micro_wkt, 1);
  TestPolygonPolygonShader(micro_wkt, micro_wkt, 4);
}

TEST(PolygonPolygonShaderTest, SubdividedBoundaryNotch) {
  // Test sweep progress beyond initial segment steps and verify tail gap handling:
  // a: 1 long edge from (0,0) to (300,0)
  // b: 130 collinear unit segments, followed by a notch into the exterior at [150..160]
  std::string wkt_a = "POLYGON ((0 0, 300 0, 300 10, 0 10, 0 0))";

  std::string wkt_b = "POLYGON ((";
  for (int i = 0; i < 130; ++i) {
    wkt_b += std::to_string(i) + " 0, ";
  }
  wkt_b += "150 0, 150 1, 160 1, 160 0, 300 0, 300 10, 0 10, 0 0))";

  TestPolygonPolygonShader(wkt_a, wkt_b, 1);
  TestPolygonPolygonShader(wkt_a, wkt_b, 4);
  TestPolygonPolygonShader(wkt_b, wkt_a, 1);
  TestPolygonPolygonShader(wkt_b, wkt_a, 4);
}

TEST(PolygonPolygonShaderTest, ManySubdivisionsComb500Teeth) {
  // Verifies that a segment crossing hundreds of teeth (1000 crossings) converges
  // naturally without an arbitrary iteration cap and matches GEOS DE-9IM.
  // Polygon a: rectangle crossing all teeth horizontally: [0, 1000] x [4, 6]
  std::string wkt_a = "POLYGON ((0 4, 1000 4, 1000 6, 0 6, 0 4))";

  // Polygon b: comb with 500 vertical teeth of height 10
  std::string wkt_b = "POLYGON ((0 0, ";
  for (int i = 0; i < 500; ++i) {
    wkt_b += std::to_string(2 * i) + " 10, " + std::to_string(2 * i + 1) + " 10, " +
             std::to_string(2 * i + 1) + " 0, ";
  }
  wkt_b += "1000 0, 1000 -2, 0 -2, 0 0))";

  TestPolygonPolygonShader(wkt_a, wkt_b, 1);
  TestPolygonPolygonShader(wkt_a, wkt_b, 4);
}

TEST(PolygonPolygonShaderTest, PolygonWithHoleOverlappingPolygon) {
  std::string wkt1 =
      "POLYGON ((40 60, 420 60, 420 320, 40 320, 40 60), (200 140, 160 220, 260 200, 200 140))";
  std::string wkt2 = "POLYGON ((80 100, 360 100, 360 280, 80 280, 80 100))";

  TestPolygonPolygonShader(wkt1, wkt2, 1);
  TestPolygonPolygonShader(wkt1, wkt2, 4);
  TestPolygonPolygonShader(wkt2, wkt1, 1);
  TestPolygonPolygonShader(wkt2, wkt1, 4);
}

}  // namespace gpuspatial
