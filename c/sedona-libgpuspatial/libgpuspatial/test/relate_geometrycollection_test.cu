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
#include "test_common.hpp"

#include "gpuspatial/relate/relate_engine.cuh"
#include "gpuspatial/rt/rt_engine.hpp"

#include <geos/geom/Geometry.h>
#include <geos/io/WKTReader.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

namespace gpuspatial {

namespace {

std::shared_ptr<RTEngine> GetSharedRTEngine() {
  static std::shared_ptr<RTEngine> rt_engine = []() {
    auto engine = std::make_shared<RTEngine>();
    std::string ptx_root = TestUtils::GetTestShaderPath();
    auto config = get_default_rt_config(ptx_root);
    engine->Init(config);
    return engine;
  }();
  return rt_engine;
}

DeviceGeometries<point_t, index_t> LoadDeviceGeometriesFromWKT(
    const std::vector<std::string>& wkts) {
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

void TestRelateGC(const std::string& wkt1, const std::string& wkt2,
                  const std::vector<Predicate>& predicates) {
  geos::io::WKTReader r;
  auto geos_a = r.read(wkt1);
  auto geos_b = r.read(wkt2);

  auto cuda_stream = rmm::cuda_stream_default;
  auto dev_geom1 = LoadDeviceGeometriesFromWKT({wkt1});
  auto dev_geom2 = LoadDeviceGeometriesFromWKT({wkt2});
  auto rt_engine = GetSharedRTEngine();

  for (Predicate p : predicates) {
    // 1. Forward: dev_geom1 vs dev_geom2
    {
      RelateEngine<point_t, index_t> engine(&dev_geom1, rt_engine.get());
      rmm::device_uvector<index_t> ids1(1, cuda_stream);
      rmm::device_uvector<index_t> ids2(1, cuda_stream);
      ids1.set_element_to_zero_async(0, cuda_stream);
      ids2.set_element_to_zero_async(0, cuda_stream);

      engine.Evaluate(cuda_stream, dev_geom2, p, ids1, ids2);
      cuda_stream.synchronize();
      bool rt_res = !ids1.is_empty();
      bool geos_res = TestUtils::EvaluateGeosPredicate(p, geos_a.get(), geos_b.get());

      EXPECT_EQ(rt_res, geos_res)
          << "Predicate mismatch for " << PredicateToString(p) << ":\n"
          << "  A = " << wkt1 << "\n  B = " << wkt2
          << "\n  RT = " << rt_res << ", GEOS = " << geos_res;
    }

    // 2. Backward: dev_geom2 vs dev_geom1
    {
      RelateEngine<point_t, index_t> engine(&dev_geom2, rt_engine.get());
      rmm::device_uvector<index_t> ids1(1, cuda_stream);
      rmm::device_uvector<index_t> ids2(1, cuda_stream);
      ids1.set_element_to_zero_async(0, cuda_stream);
      ids2.set_element_to_zero_async(0, cuda_stream);

      engine.Evaluate(cuda_stream, dev_geom1, p, ids1, ids2);
      cuda_stream.synchronize();
      bool rt_res = !ids1.is_empty();
      bool geos_res = TestUtils::EvaluateGeosPredicate(p, geos_b.get(), geos_a.get());

      EXPECT_EQ(rt_res, geos_res)
          << "Predicate mismatch for " << PredicateToString(p) << ":\n"
          << "  A = " << wkt2 << "\n  B = " << wkt1
          << "\n  RT = " << rt_res << ", GEOS = " << geos_res;
    }
  }
}

void TestRelateGCBatches(const std::vector<std::string>& wkts1,
                         const std::vector<std::string>& wkts2,
                         Predicate predicate) {
  auto cuda_stream = rmm::cuda_stream_default;
  auto dev_geom1 = LoadDeviceGeometriesFromWKT(wkts1);
  auto dev_geom2 = LoadDeviceGeometriesFromWKT(wkts2);
  auto rt_engine = GetSharedRTEngine();
  RelateEngine<point_t, index_t> engine(&dev_geom1, rt_engine.get());

  geos::io::WKTReader reader;
  std::vector<std::unique_ptr<geos::geom::Geometry>> geos1, geos2;
  for (const auto& w : wkts1) geos1.push_back(reader.read(w));
  for (const auto& w : wkts2) geos2.push_back(reader.read(w));

  std::vector<index_t> h_ids1, h_ids2;
  std::vector<std::pair<index_t, index_t>> expected_pairs;

  for (index_t i = 0; i < wkts1.size(); ++i) {
    for (index_t j = 0; j < wkts2.size(); ++j) {
      if (geos1[i]->getEnvelopeInternal()->intersects(geos2[j]->getEnvelopeInternal())) {
        h_ids1.push_back(i);
        h_ids2.push_back(j);
        if (TestUtils::EvaluateGeosPredicate(predicate, geos1[i].get(), geos2[j].get())) {
          expected_pairs.emplace_back(i, j);
        }
      }
    }
  }

  if (h_ids1.empty()) {
    EXPECT_TRUE(expected_pairs.empty());
    return;
  }

  rmm::device_uvector<index_t> d_ids1(h_ids1.size(), cuda_stream);
  rmm::device_uvector<index_t> d_ids2(h_ids2.size(), cuda_stream);
  CUDA_CHECK(cudaMemcpyAsync(d_ids1.data(), h_ids1.data(),
                             h_ids1.size() * sizeof(index_t), cudaMemcpyHostToDevice,
                             cuda_stream.value()));
  CUDA_CHECK(cudaMemcpyAsync(d_ids2.data(), h_ids2.data(),
                             h_ids2.size() * sizeof(index_t), cudaMemcpyHostToDevice,
                             cuda_stream.value()));

  engine.Evaluate(cuda_stream, dev_geom2, predicate, d_ids1, d_ids2);
  auto out_ids1 = TestUtils::ToVector(cuda_stream, d_ids1);
  auto out_ids2 = TestUtils::ToVector(cuda_stream, d_ids2);
  cuda_stream.synchronize();

  std::vector<std::pair<index_t, index_t>> actual_pairs;
  for (size_t k = 0; k < out_ids1.size(); ++k) {
    actual_pairs.emplace_back(out_ids1[k], out_ids2[k]);
  }
  std::sort(expected_pairs.begin(), expected_pairs.end());
  std::sort(actual_pairs.begin(), actual_pairs.end());
  EXPECT_EQ(actual_pairs, expected_pairs)
      << "Batch mismatch on predicate " << PredicateToString(predicate);
}

}  // namespace

TEST(RelateGeometryCollectionTest, GeometryCollectionChildPolygonInPass2) {
  std::string poly = "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))";
  std::string gc =
      "GEOMETRYCOLLECTION (POLYGON ((1 1, 2 1, 2 2, 1 2, 1 1)), POINT (100 100))";
  TestRelateGC(poly, gc, {Predicate::kIntersects});
}

TEST(RelateGeometryCollectionTest, GeosGeometryCollectionTestSuite) {
  struct TestCase {
    std::string name;
    std::string wkt_a;
    std::string wkt_b;
  };

  // Test cases adapted from GEOS RelateNGGC test suite and GeometryCollection edge cases
  std::vector<TestCase> cases = {
      {"testMP_GLP_GEOS1027",
       "MULTIPOLYGON (((0 0, 3 0, 3 3, 0 3, 0 0)))",
       "GEOMETRYCOLLECTION (LINESTRING (1 2, 1 1), POINT (0 0))"},
      {"testGPL_A",
       "GEOMETRYCOLLECTION (POINT (7 1), LINESTRING (6 5, 6 4))",
       "POLYGON ((7 1, 1 3, 3 9, 7 1))"},
      {"testP_GPL",
       "POINT (0 0)",
       "GEOMETRYCOLLECTION (POINT (0 0), LINESTRING (0 0, 1 0))"},
      {"testLineInOverlappingPolygonsTouchingInteriorEdge",
       "LINESTRING (3 7, 7 3)",
       "GEOMETRYCOLLECTION (POLYGON ((1 9, 7 9, 7 3, 1 3, 1 9)), POLYGON ((9 1, 3 1, 3 7, 9 7, 9 1)))"},
      {"testLineInOverlappingPolygonsCrossingInteriorEdgeAtVertex",
       "LINESTRING (2 2, 8 8)",
       "GEOMETRYCOLLECTION (POLYGON ((1 1, 1 7, 7 7, 7 1, 1 1)), POLYGON ((9 9, 9 3, 3 3, 3 9, 9 9)))"},
      {"testLineInOverlappingPolygonsCrossingInteriorEdgeProper",
       "LINESTRING (2 4, 6 8)",
       "GEOMETRYCOLLECTION (POLYGON ((1 1, 1 7, 7 7, 7 1, 1 1)), POLYGON ((9 9, 9 3, 3 3, 3 9, 9 9)))"},
      {"testPolygonInOverlappingPolygonsTouchingBoundaries",
       "GEOMETRYCOLLECTION (POLYGON ((1 9, 6 9, 6 4, 1 4, 1 9)), POLYGON ((9 1, 4 1, 4 6, 9 6, 9 1)))",
       "POLYGON ((2 6, 6 2, 8 4, 4 8, 2 6))"},
      {"testLineInOverlappingPolygonsBoundaries",
       "LINESTRING (1 6, 9 6, 9 1, 1 1, 1 6)",
       "GEOMETRYCOLLECTION (POLYGON ((1 1, 1 6, 6 6, 6 1, 1 1)), POLYGON ((9 1, 4 1, 4 6, 9 6, 9 1)))"},
      {"testAdjacentPolygonsContainedInAdjacentPolygons",
       "GEOMETRYCOLLECTION (POLYGON ((2 2, 2 5, 4 5, 4 2, 2 2)), POLYGON ((8 2, 4 3, 4 4, 8 5, 8 2)))",
       "GEOMETRYCOLLECTION (POLYGON ((1 1, 1 6, 4 6, 4 1, 1 1)), POLYGON ((9 1, 4 1, 4 6, 9 6, 9 1)))"},
      {"testGCMultiPolygonIntersectsPolygon",
       "POLYGON ((2 5, 3 5, 3 3, 2 3, 2 5))",
       "GEOMETRYCOLLECTION (MULTIPOLYGON (((1 4, 4 4, 4 1, 1 1, 1 4)), ((5 4, 8 4, 8 1, 5 1, 5 4))))"},
      {"testPolygonContainsGCMultiPolygonElement",
       "POLYGON ((0 5, 4 5, 4 1, 0 1, 0 5))",
       "GEOMETRYCOLLECTION (MULTIPOLYGON (((1 4, 3 4, 3 2, 1 2, 1 4)), ((6 4, 8 4, 8 2, 6 2, 6 4))))"},
      {"testPolygonOverlappingGCPolygon",
       "GEOMETRYCOLLECTION (POLYGON ((18.6 40.8, 16.8825 39.618567, 16.9319 39.5461, 17.10985 39.485133, 16.6143 38.4302, 16.43145 38.313267, 16.2 37.5, 14.8 37.8, 14.96475 40.474933, 18.6 40.8)))",
       "POLYGON ((16.3649953125 38.37219358064516, 16.3649953125 39.545924774193544, 17.949465625000002 39.545924774193544, 17.949465625000002 38.37219358064516, 16.3649953125 38.37219358064516))"},
      {"testAdjPolygonsCoverPolygonWithEndpointInside",
       "GEOMETRYCOLLECTION (POLYGON ((5 5, 2 9, 9 9, 9 5, 5 5)), POLYGON ((3 1, 5 5, 9 5, 9 1, 3 1)), POLYGON ((1 9, 2 9, 5 5, 3 1, 1 1, 1 9)))",
       "POLYGON ((3 7, 7 7, 7 3, 3 3, 3 7))"},
      {"testAdjPolygonsCoverPointAtNode",
       "GEOMETRYCOLLECTION (POLYGON ((5 5, 2 9, 9 9, 9 5, 5 5)), POLYGON ((3 1, 5 5, 9 5, 9 1, 3 1)), POLYGON ((1 9, 2 9, 5 5, 3 1, 1 1, 1 9)))",
       "POINT (5 5)"},
      {"testAdjPolygonsCoverPointOnEdge",
       "GEOMETRYCOLLECTION (POLYGON ((5 5, 2 9, 9 9, 9 5, 5 5)), POLYGON ((3 1, 5 5, 9 5, 9 1, 3 1)), POLYGON ((1 9, 2 9, 5 5, 3 1, 1 1, 1 9)))",
       "POINT (7 5)"},
      {"testAdjPolygonsContainingPolygonTouchingInteriorEndpoint",
       "GEOMETRYCOLLECTION (POLYGON ((5 5, 2 9, 9 9, 9 5, 5 5)), POLYGON ((3 1, 5 5, 9 5, 9 1, 3 1)), POLYGON ((1 9, 2 9, 5 5, 3 1, 1 1, 1 9)))",
       "POLYGON ((5 5, 7 5, 7 3, 5 3, 5 5))"},
      {"testAdjPolygonsOverlappedByPolygonWithHole",
       "GEOMETRYCOLLECTION (POLYGON ((5 5, 2 9, 9 9, 9 5, 5 5)), POLYGON ((3 1, 5 5, 9 5, 9 1, 3 1)), POLYGON ((1 9, 2 9, 5 5, 3 1, 1 1, 1 9)))",
       "POLYGON ((0 10, 10 10, 10 0, 0 0, 0 10), (2 8, 8 8, 8 2, 2 2, 2 8))"},
      {"testAdjPolygonsContainingLine",
       "GEOMETRYCOLLECTION (POLYGON ((5 5, 2 9, 9 9, 9 5, 5 5)), POLYGON ((3 1, 5 5, 9 5, 9 1, 3 1)), POLYGON ((1 9, 2 9, 5 5, 3 1, 1 1, 1 9)))",
       "LINESTRING (5 5, 7 7)"},
      {"testAdjPolygonsContainingLineAndPoint",
       "GEOMETRYCOLLECTION (POLYGON ((5 5, 2 9, 9 9, 9 5, 5 5)), POLYGON ((3 1, 5 5, 9 5, 9 1, 3 1)), POLYGON ((1 9, 2 9, 5 5, 3 1, 1 1, 1 9)))",
       "GEOMETRYCOLLECTION (POINT (5 5), LINESTRING (5 7, 7 7))"},
      {"testPolygonContainingPointsInBoundary",
       "POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))",
       "GEOMETRYCOLLECTION (POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0)), MULTIPOINT ((0 2), (0 5)))"},
      {"testPolygonContainingLineInBoundary",
       "POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))",
       "GEOMETRYCOLLECTION (POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0)), LINESTRING (0 2, 0 5))"},
      {"testPolygonContainingBentLineInBoundary",
       "POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))",
       "GEOMETRYCOLLECTION (POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0)), LINESTRING (0 2, 0 5, 5 5))"},
      {"testGCContainedPolygonWithDistantPoint",
       "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
       "GEOMETRYCOLLECTION (POLYGON ((1 1, 2 1, 2 2, 1 2, 1 1)), POINT (100 100))"},
      {"testGCContainedMultiPolygonWithDistantPoint",
       "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
       "GEOMETRYCOLLECTION (MULTIPOLYGON (((1 1, 2 1, 2 2, 1 2, 1 1)), ((3 3, 4 3, 4 4, 3 4, 3 3))), POINT (100 100))"},
      {"testGCContainedLineWithDistantPoint",
       "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
       "GEOMETRYCOLLECTION (LINESTRING (1 1, 2 2), POINT (100 100))"},
      {"testTwoGCsWithDistantNonIntersectingAndContainedComponents",
       "GEOMETRYCOLLECTION (POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0)), POINT (-100 -100))",
       "GEOMETRYCOLLECTION (POLYGON ((1 1, 2 1, 2 2, 1 2, 1 1)), POINT (100 100))"},
  };

  for (const auto& tc : cases) {
    SCOPED_TRACE(tc.name);
    TestRelateGC(tc.wkt_a, tc.wkt_b, {Predicate::kIntersects});
  }
}

TEST(RelateGeometryCollectionTest, MixedGeometryProbeIntersects) {
  // Build side has concave / L-shaped polygons and multipolygons where MBR != geometry
  std::vector<std::string> build_wkts = {
      // L-shaped polygon: MBR is [0, 0, 10, 10], but upper-right [2, 2, 10, 10] is empty!
      "POLYGON ((0 0, 10 0, 10 2, 2 2, 2 10, 0 10, 0 0))",
      // Polygon with hole: MBR is [20, 20, 30, 30], hole is [22, 22, 28, 28]
      "POLYGON ((20 20, 30 20, 30 30, 20 30, 20 20), (22 22, 28 22, 28 28, 22 28, 22 22))",
      // MultiPolygon with two disjoint parts: MBR is [40, 40, 60, 60], gap in middle
      "MULTIPOLYGON (((40 40, 45 40, 45 45, 40 45, 40 40)), ((55 55, 60 55, 60 60, 55 60, 55 55)))",
  };

  // Probe side mixes Point, MultiPoint, LineString, MultiLineString, Polygon, MultiPolygon
  // causing ParallelWkbLoader to upcast the probe column to GeometryCollection.
  std::vector<std::string> probe_wkts = {
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
  };

  TestRelateGCBatches(build_wkts, probe_wkts, Predicate::kIntersects);
  TestRelateGCBatches(build_wkts, probe_wkts, Predicate::kContains);

  // Also test ST_Intersects with nested multi-part GeometryCollections
  auto probe_with_nested_gc = probe_wkts;
  probe_with_nested_gc.push_back("GEOMETRYCOLLECTION (POINT (5 5), POINT (1 1))");
  probe_with_nested_gc.push_back("GEOMETRYCOLLECTION (POINT (5 5), LINESTRING (4 4, 9 9))");
  TestRelateGCBatches(build_wkts, probe_with_nested_gc, Predicate::kIntersects);
}

TEST(RelateGeometryCollectionTest, MixedGeometryBuildAndBothSidesIntersects) {
  // Contains all 6 geometry types (Point, MultiPoint, LineString, MultiLineString,
  // Polygon, MultiPolygon) so all 6 sub-array branches in EvaluateGeometryCollection
  // are exercised on both build and probe sides.
  std::vector<std::string> mixed_all_6_types_1 = {
      // 1. Point (inside L-shape gap vs inside L-shape)
      "POINT (5 5)",
      "POINT (1 1)",
      // 2. MultiPoint (in gap vs partially inside L-shape)
      "MULTIPOINT ((5 5), (8 8))",
      "MULTIPOINT ((5 5), (1 1))",
      // 3. LineString (in gap vs crossing L-shape)
      "LINESTRING (4 4, 9 9)",
      "LINESTRING (0 5, 5 5)",
      // 4. MultiLineString (in hole vs crossing hole boundary)
      "MULTILINESTRING ((23 23, 27 27), (24 24, 26 26))",
      "MULTILINESTRING ((23 23, 27 27), (19 25, 21 25))",
      // 5. Polygon (L-shaped polygon and polygon with hole)
      "POLYGON ((0 0, 10 0, 10 2, 2 2, 2 10, 0 10, 0 0))",
      "POLYGON ((20 20, 30 20, 30 30, 20 30, 20 20), (22 22, 28 22, 28 28, 22 28, 22 22))",
      // 6. MultiPolygon (two disjoint parts with gap in [45, 55])
      "MULTIPOLYGON (((40 40, 45 40, 45 45, 40 45, 40 40)), ((55 55, 60 55, 60 60, 55 60, 55 55)))",
  };

  std::vector<std::string> mixed_all_6_types_2 = {
      // 1. Point
      "POINT (5 5)",
      "POINT (1 1)",
      "POINT (25 25)",
      "POINT (21 21)",
      "POINT (50 50)",
      "POINT (42 42)",
      // 2. MultiPoint
      "MULTIPOINT ((5 5), (8 8))",
      "MULTIPOINT ((1 1), (5 5))",
      "MULTIPOINT ((42 42), (50 50))",
      // 3. LineString
      "LINESTRING (4 4, 9 9)",
      "LINESTRING (0 5, 5 5)",
      "LINESTRING (46 46, 54 54)",
      // 4. MultiLineString
      "MULTILINESTRING ((4 4, 9 9), (6 6, 8 8))",
      "MULTILINESTRING ((0 1, 5 1), (46 46, 54 54))",
      // 5. Polygon
      "POLYGON ((4 4, 9 4, 9 9, 4 9, 4 4))",
      "POLYGON ((0.5 0.5, 5 0.5, 5 5, 0.5 5, 0.5 0.5))",
      // 6. MultiPolygon
      "MULTIPOLYGON (((47 47, 52 47, 52 52, 47 52, 47 47)))",
      "MULTIPOLYGON (((44 44, 48 44, 48 48, 44 48, 44 44)))",
  };

  // Homogeneous batches for each of the 6 geometry types
  std::vector<std::string> points_only = {
      "POINT (5 5)", "POINT (1 1)", "POINT (21 21)", "POINT (25 25)", "POINT (42 42)", "POINT (50 50)",
  };
  std::vector<std::string> multipoints_only = {
      "MULTIPOINT ((5 5), (8 8))", "MULTIPOINT ((1 1), (5 5))", "MULTIPOINT ((42 42), (50 50))",
  };
  std::vector<std::string> linestrings_only = {
      "LINESTRING (4 4, 9 9)", "LINESTRING (0 5, 5 5)", "LINESTRING (23 23, 27 27)",
  };
  std::vector<std::string> multilinestrings_only = {
      "MULTILINESTRING ((4 4, 9 9), (6 6, 8 8))", "MULTILINESTRING ((0 5, 5 5), (20 20, 21 21))",
  };
  std::vector<std::string> polygons_only = {
      "POLYGON ((4 4, 9 4, 9 9, 4 9, 4 4))",
      "POLYGON ((0.5 0.5, 5 0.5, 5 5, 0.5 5, 0.5 0.5))",
      "POLYGON ((23 23, 27 23, 27 27, 23 27, 23 23))",
  };
  std::vector<std::string> multipolygons_only = {
      "MULTIPOLYGON (((47 47, 52 47, 52 52, 47 52, 47 47)))",
      "MULTIPOLYGON (((44 44, 48 44, 48 48, 44 48, 44 44)))",
  };

  // 1. Build side is GeometryCollection (all 6 types, single-geom per row) vs each homogeneous probe type
  for (const auto& probe_batch : {points_only, multipoints_only, linestrings_only,
                                  multilinestrings_only, polygons_only, multipolygons_only}) {
    TestRelateGCBatches(mixed_all_6_types_1, probe_batch, Predicate::kIntersects);
    TestRelateGCBatches(probe_batch, mixed_all_6_types_1, Predicate::kIntersects);
  }
  for (const auto& probe_batch :
       {points_only, multipoints_only, linestrings_only, multilinestrings_only}) {
    TestRelateGCBatches(mixed_all_6_types_1, probe_batch, Predicate::kContains);
    TestRelateGCBatches(probe_batch, mixed_all_6_types_1, Predicate::kWithin);
  }

  // 2. Both build and probe sides are GeometryCollection (all 6 types, single-geom fast path)
  TestRelateGCBatches(mixed_all_6_types_1, mixed_all_6_types_2, Predicate::kIntersects);

  // 3. Multi-element GEOMETRYCOLLECTIONs covering all 6 child geometry types + POINT EMPTY + GC EMPTY
  // (exercises the general expansion + deduplication path in EvaluateGeometryCollection)
  auto multi_gc_side_1 = mixed_all_6_types_1;
  multi_gc_side_1.push_back("GEOMETRYCOLLECTION EMPTY");
  multi_gc_side_1.push_back("GEOMETRYCOLLECTION (POINT EMPTY, POINT (1 1))");
  multi_gc_side_1.push_back("GEOMETRYCOLLECTION (POINT EMPTY, POINT (5 5))");
  multi_gc_side_1.push_back(
      "GEOMETRYCOLLECTION ("
      "POINT (1 1), "
      "MULTIPOINT ((5 5), (1 1)), "
      "LINESTRING (0 5, 5 5), "
      "MULTILINESTRING ((23 23, 27 27), (19 25, 21 25)), "
      "POLYGON ((0 0, 10 0, 10 2, 2 2, 2 10, 0 10, 0 0)), "
      "MULTIPOLYGON (((40 40, 45 40, 45 45, 40 45, 40 40)), ((55 55, 60 55, 60 60, 55 60, 55 55)))"
      ")");

  auto multi_gc_side_2 = mixed_all_6_types_2;
  multi_gc_side_2.push_back("GEOMETRYCOLLECTION EMPTY");
  multi_gc_side_2.push_back("GEOMETRYCOLLECTION (POINT EMPTY, POINT (5 5))");
  multi_gc_side_2.push_back(
      "GEOMETRYCOLLECTION ("
      "POINT (5 5), "
      "MULTIPOINT ((5 5), (8 8)), "
      "LINESTRING (4 4, 9 9), "
      "MULTILINESTRING ((4 4, 9 9), (6 6, 8 8)), "
      "POLYGON ((4 4, 9 4, 9 9, 4 9, 4 4)), "
      "MULTIPOLYGON (((47 47, 52 47, 52 52, 47 52, 47 47)))"
      ")");

  TestRelateGCBatches(multi_gc_side_1, multi_gc_side_2, Predicate::kIntersects);
}

TEST(RelateGeometryCollectionTest, EmptyAndNestedGeometryCollection) {
  // Empty GeometryCollection does not intersect any geometry
  std::string empty_gc = "GEOMETRYCOLLECTION EMPTY";
  std::string pt = "POINT (1 1)";
  std::string poly = "POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))";

  TestRelateGC(empty_gc, pt, {Predicate::kIntersects});
  TestRelateGC(empty_gc, poly, {Predicate::kIntersects});
  TestRelateGC(empty_gc, empty_gc, {Predicate::kIntersects});

  // Nested GeometryCollection
  std::string nested_gc = "GEOMETRYCOLLECTION (GEOMETRYCOLLECTION (POINT (1 1)))";
  TestRelateGC(nested_gc, pt, {Predicate::kIntersects});
  TestRelateGC(nested_gc, poly, {Predicate::kIntersects});
}

TEST(RelateGeometryCollectionTest, SharedBoundaryAndJunctions) {
  // Two adjacent polygons sharing the vertical segment from (10, 0) to (10, 10)
  std::string adjacent_polys =
      "GEOMETRYCOLLECTION ("
      "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0)), "
      "POLYGON ((10 0, 20 0, 20 10, 10 10, 10 0))"
      ")";

  // 1. Point on the shared interior boundary between the two polygons
  TestRelateGC(adjacent_polys, "POINT (10 5)", {Predicate::kIntersects});

  // 2. Point at junction vertex of the shared boundary
  TestRelateGC(adjacent_polys, "POINT (10 10)", {Predicate::kIntersects});
  TestRelateGC(adjacent_polys, "POINT (10 0)", {Predicate::kIntersects});

  // 3. Point strictly inside one polygon
  TestRelateGC(adjacent_polys, "POINT (5 5)", {Predicate::kIntersects});

  // 4. Point completely outside both polygons
  TestRelateGC(adjacent_polys, "POINT (25 5)", {Predicate::kIntersects});

  // 5. Line segment running collinear along the shared interior boundary
  TestRelateGC(adjacent_polys, "LINESTRING (10 2, 10 8)", {Predicate::kIntersects});

  // 6. Line segment crossing perpendicularly through the shared interior boundary
  TestRelateGC(adjacent_polys, "LINESTRING (5 5, 15 5)", {Predicate::kIntersects});

  // 7. Line segment touching shared boundary only at the junction vertex
  TestRelateGC(adjacent_polys, "LINESTRING (10 10, 15 15)", {Predicate::kIntersects});

  // 8. Line segment running along the outer boundary
  TestRelateGC(adjacent_polys, "LINESTRING (0 2, 0 8)", {Predicate::kIntersects});

  // 9. Line segment completely outside
  TestRelateGC(adjacent_polys, "LINESTRING (30 0, 30 10)", {Predicate::kIntersects});

  // 4 adjacent squares meeting at a 4-way central junction node (10, 10)
  std::string four_quadrants =
      "GEOMETRYCOLLECTION ("
      "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0)), "
      "POLYGON ((10 0, 20 0, 20 10, 10 10, 10 0)), "
      "POLYGON ((10 10, 20 10, 20 20, 10 20, 10 10)), "
      "POLYGON ((0 10, 10 10, 10 20, 0 20, 0 10))"
      ")";

  // Central 4-way junction node
  TestRelateGC(four_quadrants, "POINT (10 10)", {Predicate::kIntersects});

  // Diagonal line passing through the central junction
  TestRelateGC(four_quadrants, "LINESTRING (5 5, 15 15)", {Predicate::kIntersects});

  // Straight line traversing across the junction along the shared boundary
  TestRelateGC(four_quadrants, "LINESTRING (10 5, 10 15)", {Predicate::kIntersects});
  TestRelateGC(four_quadrants, "LINESTRING (5 10, 15 10)", {Predicate::kIntersects});
}

TEST(RelateGeometryCollectionTest, MixedEmptyComponents) {
  // GC with POLYGON EMPTY alongside valid LINESTRING
  std::string gc_empty_poly_and_line =
      "GEOMETRYCOLLECTION (POLYGON EMPTY, LINESTRING (0 0, 10 10))";
  TestRelateGC(gc_empty_poly_and_line, "POINT (5 5)", {Predicate::kIntersects});
  TestRelateGC(gc_empty_poly_and_line, "POINT (0 5)", {Predicate::kIntersects});

  // GC with POINT EMPTY alongside valid POLYGON
  std::string gc_empty_pt_and_poly =
      "GEOMETRYCOLLECTION (POINT EMPTY, POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0)))";
  TestRelateGC(gc_empty_pt_and_poly, "POINT (5 5)", {Predicate::kIntersects});
  TestRelateGC(gc_empty_pt_and_poly, "POINT (20 20)", {Predicate::kIntersects});

  // GC with MULTIPOINT EMPTY and POLYGON EMPTY alongside valid LINESTRING
  std::string gc_multiple_empties =
      "GEOMETRYCOLLECTION (POLYGON EMPTY, MULTIPOINT EMPTY, LINESTRING (2 2, 8 2))";
  TestRelateGC(gc_multiple_empties, "LINESTRING (5 0, 5 10)", {Predicate::kIntersects});
  TestRelateGC(gc_multiple_empties, "LINESTRING (0 5, 10 5)", {Predicate::kIntersects});

  // GC containing exclusively empty components of different types
  std::string all_empty_gc =
      "GEOMETRYCOLLECTION (POINT EMPTY, LINESTRING EMPTY, POLYGON EMPTY)";
  TestRelateGC(all_empty_gc, "POINT (5 5)", {Predicate::kIntersects});
  TestRelateGC(all_empty_gc, "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
               {Predicate::kIntersects});
  TestRelateGC(all_empty_gc, "GEOMETRYCOLLECTION EMPTY", {Predicate::kIntersects});
}

TEST(RelateGeometryCollectionTest, MultiPredicateEvaluation) {
  // Single-geom GC containing a polygon vs point / linestring / polygon probes
  std::string gc_poly = "GEOMETRYCOLLECTION (POLYGON ((0 0, 20 0, 20 20, 0 20, 0 0)))";
  std::string inside_pt = "POINT (10 10)";
  std::string outside_pt = "POINT (30 30)";
  std::string inside_line = "LINESTRING (5 5, 15 15)";
  std::string outside_line = "LINESTRING (25 25, 35 35)";
  std::string crossing_line = "LINESTRING (10 10, 30 10)";
  std::string inside_poly = "POLYGON ((5 5, 15 5, 15 15, 5 15, 5 5))";

  // Tests with Predicate::kIntersects
  TestRelateGC(gc_poly, inside_pt, {Predicate::kIntersects});
  TestRelateGC(gc_poly, outside_pt, {Predicate::kIntersects});
  TestRelateGC(gc_poly, inside_line, {Predicate::kIntersects});
  TestRelateGC(gc_poly, outside_line, {Predicate::kIntersects});
  TestRelateGC(gc_poly, crossing_line, {Predicate::kIntersects});
  TestRelateGC(gc_poly, inside_poly, {Predicate::kIntersects});

  // Tests with Predicate::kContains and Predicate::kWithin for single-geom GC
  TestRelateGC(gc_poly, inside_pt, {Predicate::kContains, Predicate::kWithin});
  TestRelateGC(gc_poly, outside_pt, {Predicate::kContains, Predicate::kWithin});
  TestRelateGC(gc_poly, inside_line, {Predicate::kContains, Predicate::kWithin});
  TestRelateGC(gc_poly, outside_line, {Predicate::kContains, Predicate::kWithin});
  TestRelateGC(gc_poly, crossing_line, {Predicate::kContains, Predicate::kWithin});

  // Both sides are GeometryCollections
  std::string gc_inside_pt = "GEOMETRYCOLLECTION (POINT (10 10))";
  std::string gc_outside_pt = "GEOMETRYCOLLECTION (POINT (30 30))";
  std::string gc_inside_line = "GEOMETRYCOLLECTION (LINESTRING (5 5, 15 15))";
  TestRelateGC(gc_poly, gc_inside_pt,
               {Predicate::kIntersects, Predicate::kContains, Predicate::kWithin});
  TestRelateGC(gc_poly, gc_outside_pt,
               {Predicate::kIntersects, Predicate::kContains, Predicate::kWithin});
  TestRelateGC(gc_poly, gc_inside_line,
               {Predicate::kIntersects, Predicate::kContains, Predicate::kWithin});
}

}  // namespace gpuspatial
