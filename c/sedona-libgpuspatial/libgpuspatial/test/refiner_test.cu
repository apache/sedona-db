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
#include "array_stream.hpp"
#include "test_common.hpp"

#include "gpuspatial/index/rt_spatial_index.hpp"
#include "gpuspatial/loader/device_geometries.hpp"
#include "gpuspatial/refine/rt_spatial_refiner.hpp"

#include "nanoarrow/nanoarrow.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "gpuspatial/index/rt_spatial_index.cuh"
#include "gpuspatial/refine/rt_spatial_refiner.cuh"

namespace gpuspatial {

void TestJoiner(ArrowSchema* build_schema, std::vector<ArrowArray*>& build_arrays,
                ArrowSchema* probe_schema, std::vector<ArrowArray*>& probe_arrays,
                Predicate predicate, bool pipelined = false) {
  using namespace TestUtils;
  using coord_t = double;
  using fpoint_t = Point<coord_t, 2>;
  using box_t = Box<fpoint_t>;

  auto rt_engine = std::make_shared<RTEngine>();
  {
    std::string ptx_root = TestUtils::GetTestShaderPath();
    auto config = get_default_rt_config(ptx_root);
    rt_engine->Init(config);
  }

  RTSpatialIndexConfig idx_config;
  idx_config.rt_engine = rt_engine;
  auto rt_index = CreateRTSpatialIndex<coord_t, 2>(idx_config);

  RTSpatialRefinerConfig refiner_config;
  refiner_config.rt_engine = rt_engine;
  if (pipelined) {
    refiner_config.pipeline_batches = 10;
  }
  auto rt_refiner = CreateRTSpatialRefiner(refiner_config);

  // Initialize GEOS C++ components
  auto geos_factory = geos::geom::GeometryFactory::create();
  geos::io::WKBReader wkb_reader(*geos_factory);
  geos::index::strtree::STRtree tree(10);

  // Storage for GEOS geometries to ensure they outlive the tree
  // The STRtree stores raw pointers, so we must own the objects
  std::vector<std::unique_ptr<geos::geom::Geometry>> build_geoms_storage;
  size_t total_build_length = 0;
  for (auto& array : build_arrays) {
    total_build_length += array->length;
  }
  build_geoms_storage.reserve(total_build_length);

  size_t tail_build = 0;
  ArrowError error;

  // --- Build Phase ---
  for (auto& array : build_arrays) {
    nanoarrow::UniqueArrayView array_view;
    ASSERT_EQ(ArrowArrayViewInitFromSchema(array_view.get(), build_schema, &error),
              NANOARROW_OK)
        << error.message;
    ASSERT_EQ(ArrowArrayViewSetArray(array_view.get(), array, &error), NANOARROW_OK)
        << error.message;

    std::vector<box_t> rects;
    rects.reserve(array->length);

    for (int64_t i = 0; i < array->length; i++) {
      // Parse WKB
      ArrowStringView wkb_view = ArrowArrayViewGetStringUnsafe(array_view.get(), i);
      // Copy the view to a buffer because WKBReader reads from istream or byte array
      // We can cast directly if the underlying type allows
      std::stringstream iss;
      auto geom = wkb_reader.read(reinterpret_cast<const unsigned char*>(wkb_view.data),
                                  wkb_view.size_bytes);

      // Calculate Envelope for GPU Index
      const geos::geom::Envelope* env = geom->getEnvelopeInternal();

      double xmin = 0, ymin = 0, xmax = -1, ymax = -1;
      if (!env->isNull()) {
        xmin = env->getMinX();
        ymin = env->getMinY();
        xmax = env->getMaxX();
        ymax = env->getMaxY();
      }

      box_t bbox(fpoint_t((float)xmin, (float)ymin), fpoint_t((float)xmax, (float)ymax));
      rects.push_back(bbox);

      // Store User Data (global offset)
      size_t global_offset = tail_build + i;
      geom->setUserData((void*)global_offset);

      // Insert into GEOS STRtree
      tree.insert(env, geom.get());

      // Transfer ownership to storage vector
      build_geoms_storage.push_back(std::move(geom));
    }

    rt_index->PushBuild(rects.data(), rects.size());
    tail_build += array->length;

    rt_refiner->PushBuild(array_view.get());
  }

  rt_index->FinishBuilding();
  rt_refiner->FinishBuilding();

  // --- Probe Phase ---
  for (auto& probe_array : probe_arrays) {
    nanoarrow::UniqueArrayView probe_view;
    ASSERT_EQ(ArrowArrayViewInitFromSchema(probe_view.get(), probe_schema, &error),
              NANOARROW_OK)
        << error.message;
    ASSERT_EQ(ArrowArrayViewSetArray(probe_view.get(), probe_array, &error), NANOARROW_OK)
        << error.message;

    std::vector<box_t> queries;
    std::vector<std::unique_ptr<geos::geom::Geometry>> probe_geoms;
    probe_geoms.reserve(probe_array->length);

    for (int64_t i = 0; i < probe_array->length; i++) {
      ArrowBufferView wkb_view = ArrowArrayViewGetBytesUnsafe(probe_view.get(), i);
      auto geom = wkb_reader.read(reinterpret_cast<const unsigned char*>(wkb_view.data),
                                  wkb_view.size_bytes);

      const geos::geom::Envelope* env = geom->getEnvelopeInternal();

      double xmin = 0, ymin = 0, xmax = -1, ymax = -1;
      if (!env->isNull()) {
        xmin = env->getMinX();
        ymin = env->getMinY();
        xmax = env->getMaxX();
        ymax = env->getMaxY();
      }

      box_t bbox(fpoint_t((float)xmin, (float)ymin), fpoint_t((float)xmax, (float)ymax));
      queries.push_back(bbox);

      // Store user data as local offset for verification logic
      geom->setUserData((void*)i);
      probe_geoms.push_back(std::move(geom));
    }

    std::vector<uint32_t> build_indices, stream_indices;

    // GPU Probe
    rt_index->Probe(queries.data(), queries.size(), &build_indices, &stream_indices);

    // GPU Refine
    auto new_size = rt_refiner->Refine(probe_view.get(), predicate, build_indices.data(),
                                       stream_indices.data(), build_indices.size());

    build_indices.resize(new_size);
    stream_indices.resize(new_size);

    // --- CPU Verification (GEOS C++) ---
    std::vector<uint32_t> expected_build_indices;
    std::vector<uint32_t> expected_stream_indices;

    ComputeGeosJoin(build_schema, build_arrays, probe_schema,
                    std::vector<ArrowArray*>{probe_array}, predicate,
                    expected_build_indices, expected_stream_indices);

    // Assertions
    ASSERT_EQ(expected_build_indices.size(), build_indices.size());
    ASSERT_EQ(expected_stream_indices.size(), stream_indices.size());

    TestUtils::sort_vectors_by_index(expected_build_indices, expected_stream_indices);
    TestUtils::sort_vectors_by_index(build_indices, stream_indices);

    for (size_t j = 0; j < build_indices.size(); j++) {
      ASSERT_EQ(expected_build_indices[j], build_indices[j]);
      ASSERT_EQ(expected_stream_indices[j], stream_indices[j]);
    }
  }
}

TEST(JoinerTest, PIPContains) {
  using namespace TestUtils;
  auto fs = std::make_shared<arrow::fs::LocalFileSystem>();

  std::vector<std::string> polys{
      GetTestDataPath("synthetic_pip/polygons.parquet"),
      GetTestDataPath("countries/natural-earth_countries_geo.parquet")};
  std::vector<std::string> points{GetTestDataPath("synthetic_pip/points.parquet"),
                                  GetTestDataPath("countries/generated_points.parquet")};

  for (int i = 0; i < polys.size(); i++) {
    auto poly_path = TestUtils::GetTestDataPath(polys[i]);
    auto point_path = TestUtils::GetCanonicalPath(points[i]);
    auto poly_arrays = ReadParquet(poly_path, 1000);
    auto point_arrays = ReadParquet(point_path, 1000);
    std::vector<nanoarrow::UniqueArray> poly_uniq_arrays, point_uniq_arrays;
    std::vector<nanoarrow::UniqueSchema> poly_uniq_schema, point_uniq_schema;

    for (auto& arr : poly_arrays) {
      ARROW_THROW_NOT_OK(arrow::ExportArray(*arr, poly_uniq_arrays.emplace_back().get(),
                                            poly_uniq_schema.emplace_back().get()));
    }
    for (auto& arr : point_arrays) {
      ARROW_THROW_NOT_OK(arrow::ExportArray(*arr, point_uniq_arrays.emplace_back().get(),
                                            point_uniq_schema.emplace_back().get()));
    }

    std::vector<ArrowArray*> poly_c_arrays, point_c_arrays;
    for (auto& arr : poly_uniq_arrays) {
      poly_c_arrays.push_back(arr.get());
    }
    for (auto& arr : point_uniq_arrays) {
      point_c_arrays.push_back(arr.get());
    }
    TestJoiner(poly_uniq_schema[0].get(), poly_c_arrays, point_uniq_schema[0].get(),
               point_c_arrays, Predicate::kContains);
  }
}

TEST(JoinerTest, PIPContainsPipelined) {
  using namespace TestUtils;
  auto fs = std::make_shared<arrow::fs::LocalFileSystem>();

  std::vector<std::string> polys{
      GetTestDataPath("synthetic_pip/polygons.parquet"),
      GetTestDataPath("countries/natural-earth_countries_geo.parquet")};
  std::vector<std::string> points{GetTestDataPath("synthetic_pip/points.parquet"),
                                  GetTestDataPath("countries/generated_points.parquet")};

  for (int i = 0; i < polys.size(); i++) {
    auto poly_path = TestUtils::GetTestDataPath(polys[i]);
    auto point_path = TestUtils::GetCanonicalPath(points[i]);
    auto poly_arrays = ReadParquet(poly_path, 1000);
    auto point_arrays = ReadParquet(point_path, 1000);
    std::vector<nanoarrow::UniqueArray> poly_uniq_arrays, point_uniq_arrays;
    std::vector<nanoarrow::UniqueSchema> poly_uniq_schema, point_uniq_schema;

    for (auto& arr : poly_arrays) {
      ARROW_THROW_NOT_OK(arrow::ExportArray(*arr, poly_uniq_arrays.emplace_back().get(),
                                            poly_uniq_schema.emplace_back().get()));
    }
    for (auto& arr : point_arrays) {
      ARROW_THROW_NOT_OK(arrow::ExportArray(*arr, point_uniq_arrays.emplace_back().get(),
                                            point_uniq_schema.emplace_back().get()));
    }

    std::vector<ArrowArray*> poly_c_arrays, point_c_arrays;
    for (auto& arr : poly_uniq_arrays) {
      poly_c_arrays.push_back(arr.get());
    }
    for (auto& arr : point_uniq_arrays) {
      point_c_arrays.push_back(arr.get());
    }
    TestJoiner(poly_uniq_schema[0].get(), poly_c_arrays, point_uniq_schema[0].get(),
               point_c_arrays, Predicate::kContains, true);
  }
}

TEST(JoinerTest, PIPWithin) {
  using namespace TestUtils;
  auto fs = std::make_shared<arrow::fs::LocalFileSystem>();

  std::vector<std::string> polys{
      GetTestDataPath("synthetic_pip/polygons.parquet"),
      GetTestDataPath("countries/natural-earth_countries_geo.parquet")};
  std::vector<std::string> points{GetTestDataPath("synthetic_pip/points.parquet"),
                                  GetTestDataPath("countries/generated_points.parquet")};

  for (int i = 0; i < polys.size(); i++) {
    auto poly_path = TestUtils::GetTestDataPath(polys[i]);
    auto point_path = TestUtils::GetCanonicalPath(points[i]);
    auto poly_arrays = ReadParquet(poly_path, 1000);
    auto point_arrays = ReadParquet(point_path, 1000);
    std::vector<nanoarrow::UniqueArray> poly_uniq_arrays, point_uniq_arrays;
    std::vector<nanoarrow::UniqueSchema> poly_uniq_schema, point_uniq_schema;

    for (auto& arr : poly_arrays) {
      ARROW_THROW_NOT_OK(arrow::ExportArray(*arr, poly_uniq_arrays.emplace_back().get(),
                                            poly_uniq_schema.emplace_back().get()));
    }
    for (auto& arr : point_arrays) {
      ARROW_THROW_NOT_OK(arrow::ExportArray(*arr, point_uniq_arrays.emplace_back().get(),
                                            point_uniq_schema.emplace_back().get()));
    }

    std::vector<ArrowArray*> poly_c_arrays, point_c_arrays;
    for (auto& arr : poly_uniq_arrays) {
      poly_c_arrays.push_back(arr.get());
    }
    for (auto& arr : point_uniq_arrays) {
      point_c_arrays.push_back(arr.get());
    }
    TestJoiner(point_uniq_schema[0].get(), point_c_arrays, poly_uniq_schema[0].get(),
               poly_c_arrays, Predicate::kWithin);
  }
}

TEST(JoinerTest, PolygonPolygonIntersects) {
  using namespace TestUtils;
  auto fs = std::make_shared<arrow::fs::LocalFileSystem>();

  std::vector<std::string> polys1{GetTestDataPath("synthetic_poly/polygons1.parquet")};
  std::vector<std::string> polys2{GetTestDataPath("synthetic_poly/polygons2.parquet")};

  for (int i = 0; i < polys1.size(); i++) {
    auto poly1_path = TestUtils::GetTestDataPath(polys1[i]);
    auto poly2_path = TestUtils::GetCanonicalPath(polys2[i]);
    auto poly1_arrays = ReadParquet(poly1_path, 1000);
    auto point2_arrays = ReadParquet(poly2_path, 1000);
    std::vector<nanoarrow::UniqueArray> poly1_uniq_arrays, poly2_uniq_arrays;
    std::vector<nanoarrow::UniqueSchema> poly1_uniq_schema, poly2_uniq_schema;

    for (auto& arr : poly1_arrays) {
      ARROW_THROW_NOT_OK(arrow::ExportArray(*arr, poly1_uniq_arrays.emplace_back().get(),
                                            poly1_uniq_schema.emplace_back().get()));
    }
    for (auto& arr : point2_arrays) {
      ARROW_THROW_NOT_OK(arrow::ExportArray(*arr, poly2_uniq_arrays.emplace_back().get(),
                                            poly2_uniq_schema.emplace_back().get()));
    }

    std::vector<ArrowArray*> poly1_c_arrays, poly2_c_arrays;
    for (auto& arr : poly1_uniq_arrays) {
      poly1_c_arrays.push_back(arr.get());
    }
    for (auto& arr : poly2_uniq_arrays) {
      poly2_c_arrays.push_back(arr.get());
    }
    TestJoiner(poly1_uniq_schema[0].get(), poly1_c_arrays, poly2_uniq_schema[0].get(),
               poly2_c_arrays, Predicate::kIntersects);
  }
}
}  // namespace gpuspatial
