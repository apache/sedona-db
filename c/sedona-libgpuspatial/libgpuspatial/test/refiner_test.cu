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
      auto geom = wkb_reader.read(wkb_view.data.as_uint8, wkb_view.size_bytes);

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

void TestJoinerWKT(const std::vector<std::vector<std::string>>& build_wkt_batches,
                   const std::vector<std::vector<std::string>>& probe_wkt_batches,
                   Predicate predicate, bool pipelined = false) {
  nanoarrow::UniqueArrayStream build_stream, probe_stream;
  ArrayStreamFromWKT(build_wkt_batches, GEOARROW_TYPE_WKB, build_stream.get());
  ArrayStreamFromWKT(probe_wkt_batches, GEOARROW_TYPE_WKB, probe_stream.get());

  nanoarrow::UniqueSchema build_schema, probe_schema;
  ArrowError error;
  ASSERT_EQ(ArrowArrayStreamGetSchema(build_stream.get(), build_schema.get(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(ArrowArrayStreamGetSchema(probe_stream.get(), probe_schema.get(), &error),
            NANOARROW_OK)
      << error.message;

  std::vector<nanoarrow::UniqueArray> build_uniq_arrays, probe_uniq_arrays;
  while (true) {
    nanoarrow::UniqueArray arr;
    ASSERT_EQ(ArrowArrayStreamGetNext(build_stream.get(), arr.get(), &error),
              NANOARROW_OK)
        << error.message;
    if (arr->release == nullptr) break;
    build_uniq_arrays.push_back(std::move(arr));
  }
  while (true) {
    nanoarrow::UniqueArray arr;
    ASSERT_EQ(ArrowArrayStreamGetNext(probe_stream.get(), arr.get(), &error),
              NANOARROW_OK)
        << error.message;
    if (arr->release == nullptr) break;
    probe_uniq_arrays.push_back(std::move(arr));
  }

  std::vector<ArrowArray*> build_c_arrays, probe_c_arrays;
  for (auto& arr : build_uniq_arrays) {
    build_c_arrays.push_back(arr.get());
  }
  for (auto& arr : probe_uniq_arrays) {
    probe_c_arrays.push_back(arr.get());
  }

  TestJoiner(build_schema.get(), build_c_arrays, probe_schema.get(), probe_c_arrays,
             predicate, pipelined);
}

TEST(JoinerTest, MixedGeometryProbeIntersects) {
  // Build side has concave / L-shaped polygons and multipolygons where MBR != geometry
  std::vector<std::vector<std::string>> build_batches = {{
      // L-shaped polygon: MBR is [0, 0, 10, 10], but upper-right [2, 2, 10, 10] is empty!
      "POLYGON ((0 0, 10 0, 10 2, 2 2, 2 10, 0 10, 0 0))",
      // Polygon with hole: MBR is [20, 20, 30, 30], hole is [22, 22, 28, 28]
      "POLYGON ((20 20, 30 20, 30 30, 20 30, 20 20), (22 22, 28 22, 28 28, 22 28, 22 22))",
      // MultiPolygon with two disjoint parts: MBR is [40, 40, 60, 60], gap in middle
      "MULTIPOLYGON (((40 40, 45 40, 45 45, 40 45, 40 40)), ((55 55, 60 55, 60 60, 55 60, 55 55)))",
  }};

  // Probe side mixes Point, MultiPoint, LineString, MultiLineString, Polygon, MultiPolygon
  // causing ParallelWkbLoader to upcast the probe column to GeometryCollection.
  std::vector<std::vector<std::string>> probe_batches = {{
      // Points inside MBR but outside actual polygon (false positive if only MBR-checked)
      "POINT (5 5)",
      "POINT (8 8)",
      "POINT (25 25)",
      "POINT (50 50)",
      // Points truly inside polygon
      "POINT (1 1)",
      "POINT (21 21)",
      "POINT (42 42)",
      "POINT (57 57)",
      // MultiPoints: one outside (in MBR gap), one partially inside, one fully inside
      "MULTIPOINT ((5 5), (8 8))",
      "MULTIPOINT ((5 5), (1 1))",
      "MULTIPOINT ((1 1), (1 5))",
      // LineStrings: one inside MBR gap, one crossing polygon
      "LINESTRING (4 4, 9 9)",
      "LINESTRING (0 5, 5 5)",
      // MultiLineStrings: one in hole, one crossing hole boundary
      "MULTILINESTRING ((23 23, 27 27), (24 24, 26 26))",
      "MULTILINESTRING ((23 23, 27 27), (19 25, 21 25))",
      // Polygons: one inside L-shape gap, one overlapping L-shape, one properly contained
      "POLYGON ((4 4, 9 4, 9 9, 4 9, 4 4))",
      "POLYGON ((0.5 0.5, 5 0.5, 5 5, 0.5 5, 0.5 0.5))",
      "POLYGON ((0.2 0.2, 1.8 0.2, 1.8 1.8, 0.2 1.8, 0.2 0.2))",
      // MultiPolygons: one inside MultiPolygon gap, one overlapping
      "MULTIPOLYGON (((47 47, 52 47, 52 52, 47 52, 47 47)))",
      "MULTIPOLYGON (((44 44, 48 44, 48 48, 44 48, 44 44)))",
  }};

  TestJoinerWKT(build_batches, probe_batches, Predicate::kIntersects);
  TestJoinerWKT(build_batches, probe_batches, Predicate::kContains);

  // Also test ST_Intersects with nested multi-part GeometryCollections
  auto probe_with_nested_gc = probe_batches;
  probe_with_nested_gc[0].push_back("GEOMETRYCOLLECTION (POINT (5 5), POINT (1 1))");
  probe_with_nested_gc[0].push_back("GEOMETRYCOLLECTION (POINT (5 5), LINESTRING (4 4, 9 9))");
  TestJoinerWKT(build_batches, probe_with_nested_gc, Predicate::kIntersects);
}

TEST(JoinerTest, MixedGeometryBuildAndBothSidesIntersects) {
  std::vector<std::vector<std::string>> mixed_side_1 = {{
      "POLYGON ((0 0, 10 0, 10 2, 2 2, 2 10, 0 10, 0 0))",
      "POINT (5 5)",
      "POINT (1 1)",
      "LINESTRING (4 4, 9 9)",
      "MULTIPOLYGON (((40 40, 45 40, 45 45, 40 45, 40 40)), ((55 55, 60 55, 60 60, 55 60, 55 55)))",
  }};

  std::vector<std::vector<std::string>> points_only = {{
      "POINT (5 5)",
      "POINT (1 1)",
      "POINT (42 42)",
      "POINT (50 50)",
  }};

  std::vector<std::vector<std::string>> mixed_side_2 = {{
      "POINT (5 5)",
      "POINT (1 1)",
      "POINT (50 50)",
      "POINT (42 42)",
      "POLYGON ((4 4, 9 4, 9 9, 4 9, 4 4))",
      "POLYGON ((0 0, 10 0, 10 2, 2 2, 2 10, 0 10, 0 0))",
      "LINESTRING (0 5, 5 5)",
  }};

  // Build side is mixed (GeometryCollection), probe side is homogeneous (Point)
  TestJoinerWKT(mixed_side_1, points_only, Predicate::kIntersects);
  TestJoinerWKT(mixed_side_1, points_only, Predicate::kContains);

  // Both build and probe sides are mixed (GeometryCollection vs GeometryCollection)
  TestJoinerWKT(mixed_side_1, mixed_side_2, Predicate::kIntersects);
}
}  // namespace gpuspatial
