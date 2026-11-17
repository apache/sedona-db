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
#include <geos/operation/relateng/RelateMatrixPredicate.h>
#include <geos/operation/relateng/RelateNG.h>
#include <gtest/gtest.h>

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>

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

}  // namespace

void TestRelatePolygonRT(const char* wkt1, const char* wkt2, bool check_im) {
  geos::io::WKTReader r;
  auto geos_a = r.read(wkt1);
  auto geos_b = r.read(wkt2);

  geos::operation::relateng::RelateMatrixPredicate pred;
  geos::operation::relateng::RelateNG::relate(geos_a.get(), geos_b.get(), pred);
  std::string expected_im_str = pred.getIM()->toString();

  auto cuda_stream = rmm::cuda_stream_default;
  auto dev_geom1 = LoadDeviceGeometriesFromWKT({wkt1});
  auto dev_geom2 = LoadDeviceGeometriesFromWKT({wkt2});
  auto rt_engine = GetSharedRTEngine();
  RelateEngine<point_t, index_t> engine(&dev_geom1, rt_engine.get());

  if (check_im) {
    rmm::device_uvector<index_t> ids1(1, cuda_stream);
    rmm::device_uvector<index_t> ids2(1, cuda_stream);
    ids1.set_element_to_zero_async(0, cuda_stream);
    ids2.set_element_to_zero_async(0, cuda_stream);
    rmm::device_uvector<int> ims(1, cuda_stream);
    ims.set_element_to_zero_async(0, cuda_stream);

    PolygonArrayView<point_t, index_t> empty_poly;
    MultiPolygonArrayView<point_t, index_t> empty_mp;
    bool g1_is_poly = dev_geom1.get_geometry_type() == GeometryType::kPolygon;
    bool g2_is_poly = dev_geom2.get_geometry_type() == GeometryType::kPolygon;

    if (g1_is_poly && g2_is_poly) {
      auto v1 = dev_geom1.GetGeometryArrayView<PolygonArrayView<point_t, index_t>>();
      auto v2 = dev_geom2.GetGeometryArrayView<PolygonArrayView<point_t, index_t>>();
      engine.EvaluateImpl(cuda_stream, v1, empty_mp, v2, ids1, ids2, ims, false);
      engine.EvaluateImpl(cuda_stream, v2, empty_mp, v1, ids2, ids1, ims, true);
    } else if (g1_is_poly && !g2_is_poly) {
      auto v1 = dev_geom1.GetGeometryArrayView<PolygonArrayView<point_t, index_t>>();
      auto v2 = dev_geom2.GetGeometryArrayView<MultiPolygonArrayView<point_t, index_t>>();
      engine.EvaluateImpl(cuda_stream, v1, empty_mp, v2, ids1, ids2, ims, false);
      engine.EvaluateImpl(cuda_stream, empty_poly, v2, v1, ids2, ids1, ims, true);
    } else if (!g1_is_poly && g2_is_poly) {
      auto v1 = dev_geom1.GetGeometryArrayView<MultiPolygonArrayView<point_t, index_t>>();
      auto v2 = dev_geom2.GetGeometryArrayView<PolygonArrayView<point_t, index_t>>();
      engine.EvaluateImpl(cuda_stream, empty_poly, v1, v2, ids1, ids2, ims, false);
      engine.EvaluateImpl(cuda_stream, v2, empty_mp, v1, ids2, ids1, ims, true);
    } else {
      auto v1 = dev_geom1.GetGeometryArrayView<MultiPolygonArrayView<point_t, index_t>>();
      auto v2 = dev_geom2.GetGeometryArrayView<MultiPolygonArrayView<point_t, index_t>>();
      engine.EvaluateImpl(cuda_stream, empty_poly, v1, v2, ids1, ids2, ims, false);
      engine.EvaluateImpl(cuda_stream, empty_poly, v2, v1, ids2, ids1, ims, true);
    }

    int rt_im = ims.element(0, cuda_stream);
    char rt_im_str[10];
    IntersectionMatrix::ToString(rt_im, rt_im_str);
    EXPECT_STREQ(expected_im_str.c_str(), rt_im_str)
        << "RT DE-9IM mismatch for:\n  A = " << wkt1 << "\n  B = " << wkt2;
  }

  const std::vector<Predicate> predicates = {
      Predicate::kEquals,   Predicate::kDisjoint, Predicate::kTouches,
      Predicate::kContains, Predicate::kCovers,   Predicate::kIntersects,
      Predicate::kWithin,   Predicate::kCoveredBy};

  for (Predicate p : predicates) {
    rmm::device_uvector<index_t> ids1(1, cuda_stream);
    rmm::device_uvector<index_t> ids2(1, cuda_stream);
    ids1.set_element_to_zero_async(0, cuda_stream);
    ids2.set_element_to_zero_async(0, cuda_stream);

    engine.Evaluate(cuda_stream, dev_geom2, p, ids1, ids2);
    cuda_stream.synchronize();
    bool rt_res = !ids1.is_empty();
    bool geos_res = TestUtils::EvaluateGeosPredicate(p, geos_a.get(), geos_b.get());

    EXPECT_EQ(rt_res, geos_res)
        << "Predicate " << PredicateToString(p) << " mismatch:\n  A = " << wkt1
        << "\n  B = " << wkt2;
  }
}

//======= Polygon Relate Tests =============

//======= A/P  =============

TEST(RelatePolygonTest, PolygonPointInside) {
  Polygon<point_t, index_t> poly1;
  point_t p2;
  std::string wkt1 = "POLYGON ((0 10, 10 10, 10 0, 0 0, 0 10))";
  std::string wkt2 = "POINT (1 1)";
  Context<point_t, index_t> ctx1;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPoint(wkt2.c_str(), p2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, p2);
}

TEST(RelatePolygonTest, PolygonPointOutside) {
  Polygon<point_t, index_t> poly1;
  point_t p2;
  std::string wkt1 = "POLYGON ((10 0, 0 0, 0 10, 10 0))";
  std::string wkt2 = "POINT (8 8)";
  Context<point_t, index_t> ctx1;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPoint(wkt2.c_str(), p2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, p2);
}

TEST(RelatePolygonTest, PolygonPointBoundary) {
  Polygon<point_t, index_t> poly1;
  point_t p2;
  std::string wkt1 = "POLYGON ((10 0, 0 0, 0 10, 10 0))";
  std::string wkt2 = "POINT (1 0)";
  Context<point_t, index_t> ctx1;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPoint(wkt2.c_str(), p2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, p2);
}

TEST(RelatePolygonTest, PolygonPointExterior) {
  Polygon<point_t, index_t> poly1;
  point_t p2;
  std::string wkt1 = "POLYGON ((1 5, 5 5, 5 1, 1 1, 1 5))";
  std::string wkt2 = "POINT (7 7)";
  Context<point_t, index_t> ctx1;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPoint(wkt2.c_str(), p2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, p2);
}

//======= A/L  =============

TEST(RelatePolygonTest, PolygonLineStringContainedAtLineVertex) {
  Polygon<point_t, index_t> poly1;
  LineString<point_t> ls2;
  std::string wkt1 = "POLYGON ((1 5, 5 5, 5 1, 1 1, 1 5))";
  std::string wkt2 = "LINESTRING (2 3, 3 5, 4 3)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, ls2);
}

TEST(RelatePolygonTest, PolygonLineStringTouchAtLineVertex) {
  Polygon<point_t, index_t> poly1;
  LineString<point_t> ls2;
  std::string wkt1 = "POLYGON ((1 5, 5 5, 5 1, 1 1, 1 5))";
  std::string wkt2 = "LINESTRING (1 8, 3 5, 5 8)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, ls2);
}

TEST(RelatePolygonTest, PolygonLineStringInside) {
  Polygon<point_t, index_t> poly1;
  LineString<point_t> ls2;
  std::string wkt1 = "POLYGON ((0 10, 10 10, 10 0, 0 0, 0 10))";
  std::string wkt2 = "LINESTRING (1 8, 3 5, 5 8)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, ls2);
}

TEST(RelatePolygonTest, PolygonLineStringOutside) {
  Polygon<point_t, index_t> poly1;
  LineString<point_t> ls2;
  std::string wkt1 = "POLYGON ((10 0, 0 0, 0 10, 10 0))";
  std::string wkt2 = "LINESTRING (4 8, 9 3)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, ls2);
}

TEST(RelatePolygonTest, PolygonLineStringInBoundary) {
  Polygon<point_t, index_t> poly1;
  LineString<point_t> ls2;
  std::string wkt1 = "POLYGON ((10 0, 0 0, 0 10, 10 0))";
  std::string wkt2 = "LINESTRING (1 0, 9 0)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, ls2);
}

TEST(RelatePolygonTest, MultiPolygonLineStringCrossingContained) {
  MultiPolygon<point_t, index_t> m_poly1;
  LineString<point_t> ls2;
  std::string wkt1 =
      "MULTIPOLYGON (((20 80, 180 80, 100 0, 20 80)), ((20 160, 180 160, 100 80, 20 160)))";
  std::string wkt2 = "LINESTRING (100 140, 100 40)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTMultiPolygon(ctx1, wkt1.c_str(), m_poly1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), m_poly1, ls2);
}

TEST(RelatePolygonTest, LineStringPolygonRelateLA_220) {
  LineString<point_t> ls1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "LINESTRING (90 210, 210 90)";
  std::string wkt2 = "POLYGON ((150 150, 410 150, 280 20, 20 20, 150 150))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, poly2);
}

TEST(RelatePolygonTest, LineCrossingPolygonAtShellHolePoint) {
  LineString<point_t> ls1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "LINESTRING (60 160, 150 70)";
  std::string wkt2 =
      "POLYGON ((190 190, 360 20, 20 20, 190 190), (110 110, 250 100, 140 30, 110 110))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, poly2);
}

TEST(RelatePolygonTest, LineCrossingPolygonAtNonVertex) {
  LineString<point_t> ls1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "LINESTRING (20 60, 150 60)";
  std::string wkt2 = "POLYGON ((150 150, 410 150, 280 20, 20 20, 150 150))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, poly2);
}

TEST(RelatePolygonTest, PolygonLinesContainedCollinearEdge) {
  Polygon<point_t, index_t> poly1;
  MultiLineString<point_t, index_t> m2;
  std::string wkt1 = "POLYGON ((110 110, 200 20, 20 20, 110 110))";
  std::string wkt2 =
      "MULTILINESTRING ((110 110, 60 40, 70 20, 150 20, 170 40), (180 30, 40 30, 110 80))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTMultiLineString(ctx2, wkt2.c_str(), m2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, m2);
}

//======= A/A  =============

TEST(RelatePolygonTest, PolygonsEdgeAdjacent) {
  Polygon<point_t, index_t> poly1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "POLYGON ((1 3, 3 3, 3 1, 1 1, 1 3))";
  std::string wkt2 = "POLYGON ((5 3, 5 1, 3 1, 3 3, 5 3))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, poly2);
}

TEST(RelatePolygonTest, PolygonsEdgeAdjacent2) {
  Polygon<point_t, index_t> poly1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "POLYGON ((1 3, 4 3, 3 0, 1 1, 1 3))";
  std::string wkt2 = "POLYGON ((5 3, 5 1, 3 0, 4 3, 5 3))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, poly2);
}

TEST(RelatePolygonTest, PolygonsNested) {
  Polygon<point_t, index_t> poly1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "POLYGON ((1 9, 9 9, 9 1, 1 1, 1 9))";
  std::string wkt2 = "POLYGON ((2 8, 8 8, 8 2, 2 2, 2 8))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, poly2);
}

TEST(RelatePolygonTest, PolygonsOverlapProper) {
  Polygon<point_t, index_t> poly1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "POLYGON ((1 1, 1 7, 7 7, 7 1, 1 1))";
  std::string wkt2 = "POLYGON ((2 8, 8 8, 8 2, 2 2, 2 8))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, poly2);
}

TEST(RelatePolygonTest, PolygonsOverlapAtNodes) {
  Polygon<point_t, index_t> poly1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "POLYGON ((1 5, 5 5, 5 1, 1 1, 1 5))";
  std::string wkt2 = "POLYGON ((7 3, 5 1, 3 3, 5 5, 7 3))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, poly2);
}

TEST(RelatePolygonTest, PolygonsContainedAtNodes) {
  Polygon<point_t, index_t> poly1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "POLYGON ((1 5, 5 5, 6 2, 1 1, 1 5))";
  std::string wkt2 = "POLYGON ((1 1, 5 5, 6 2, 1 1))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, poly2);
}

TEST(RelatePolygonTest, PolygonsOverlappingWithBoundaryInside) {
  Polygon<point_t, index_t> poly1;
  MultiPolygon<point_t, index_t> m2;
  std::string wkt1 = "POLYGON ((100 60, 140 100, 100 140, 60 100, 100 60))";
  std::string wkt2 =
      "MULTIPOLYGON (((80 40, 120 40, 120 80, 80 80, 80 40)), ((120 80, 160 80, 160 120, 120 120, 120 80)), ((80 120, 120 120, 120 160, 80 160, 80 120)), ((40 80, 80 80, 80 120, 40 120, 40 80)))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTMultiPolygon(ctx2, wkt2.c_str(), m2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, m2);
}

TEST(RelatePolygonTest, PolygonsOverlapVeryNarrow) {
  Polygon<point_t, index_t> poly1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "POLYGON ((120 100, 120 200, 200 200, 200 100, 120 100))";
  std::string wkt2 = "POLYGON ((100 100, 100000 110, 100000 100, 100 100))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, poly2);
}

TEST(RelatePolygonTest, ValidateRelateAA_86) {
  Polygon<point_t, index_t> poly1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "POLYGON ((170 120, 300 120, 250 70, 120 70, 170 120))";
  std::string wkt2 =
      "POLYGON ((150 150, 410 150, 280 20, 20 20, 150 150), (170 120, 330 120, 260 50, 100 50, 170 120))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, poly2);
}

TEST(RelatePolygonTest, ValidateRelateAA_97) {
  Polygon<point_t, index_t> poly1;
  MultiPolygon<point_t, index_t> m2;
  std::string wkt1 = "POLYGON ((330 150, 200 110, 150 150, 280 190, 330 150))";
  std::string wkt2 =
      "MULTIPOLYGON (((140 110, 260 110, 170 20, 50 20, 140 110)), ((300 270, 420 270, 340 190, 220 190, 300 270)))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTMultiPolygon(ctx2, wkt2.c_str(), m2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, m2);
}

TEST(RelatePolygonTest, AdjacentPolygons) {
  Polygon<point_t, index_t> poly1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "POLYGON ((1 9, 6 9, 6 1, 1 1, 1 9))";
  std::string wkt2 = "POLYGON ((9 9, 9 4, 6 4, 6 9, 9 9))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, poly2);
}

TEST(RelatePolygonTest, AdjacentPolygonsTouchingAtPoint) {
  Polygon<point_t, index_t> poly1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "POLYGON ((1 9, 6 9, 6 1, 1 1, 1 9))";
  std::string wkt2 = "POLYGON ((9 9, 9 4, 6 4, 7 9, 9 9))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, poly2);
}

TEST(RelatePolygonTest, AdjacentPolygonsOverlappping) {
  Polygon<point_t, index_t> poly1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "POLYGON ((1 9, 6 9, 6 1, 1 1, 1 9))";
  std::string wkt2 = "POLYGON ((9 9, 9 4, 6 4, 5 9, 9 9))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, poly2);
}

TEST(RelatePolygonTest, ContainsProperlyPolygonContained) {
  Polygon<point_t, index_t> poly1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "POLYGON ((1 9, 9 9, 9 1, 1 1, 1 9))";
  std::string wkt2 = "POLYGON ((2 8, 5 8, 5 5, 2 5, 2 8))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, poly2);
}

TEST(RelatePolygonTest, ContainsProperlyPolygonTouching) {
  Polygon<point_t, index_t> poly1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "POLYGON ((1 9, 9 9, 9 1, 1 1, 1 9))";
  std::string wkt2 = "POLYGON ((9 1, 5 1, 5 5, 9 5, 9 1))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, poly2);
}

TEST(RelatePolygonTest, ContainsProperlyPolygonsOverlapping) {
  MultiPolygon<point_t, index_t> m1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "MULTIPOLYGON (((1 9, 6 9, 6 4, 1 4, 1 9)), ((2 4, 6 7, 9 1, 2 4)))";
  std::string wkt2 = "POLYGON ((5 5, 6 5, 6 4, 5 4, 5 5))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTMultiPolygon(ctx1, wkt1.c_str(), m1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), m1, poly2);
}


TEST(RelatePolygonTest, RepeatedPointAA) {
  Polygon<point_t, index_t> poly1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "POLYGON ((1 9, 9 7, 9 1, 1 3, 1 9))";
  std::string wkt2 = "POLYGON ((1 3, 1 3, 1 3, 3 7, 9 7, 9 7, 1 3))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, poly2);
}


TEST(RelatePolygonTest, PointEmpty_vs_Polygon) {
  point_t g1;
  Polygon<point_t, index_t> g2;
  Context<point_t, index_t> ctx2;
  ParseWKTPoint(E_POINT.c_str(), g1);
  ParseWKTPolygon(ctx2, NE_POLY.c_str(), g2);
  TestRelate(E_POINT.c_str(), NE_POLY.c_str(), g1, g2);
}


TEST(RelatePolygonTest, LineStringEmpty_vs_Polygon) {
  LineString<point_t> g1;
  Polygon<point_t, index_t> g2;
  Context<point_t, index_t> ctx1, ctx2;
  ParseWKTLineString(ctx1, E_LINE.c_str(), g1);
  ParseWKTPolygon(ctx2, NE_POLY.c_str(), g2);
  TestRelate(E_LINE.c_str(), NE_POLY.c_str(), g1, g2);
}

// --- POLYGON EMPTY vs Non-Empty ---

TEST(RelatePolygonTest, PolygonEmpty_vs_Point) {
  Polygon<point_t, index_t> g1;
  point_t g2;
  Context<point_t, index_t> ctx1;
  ParseWKTPolygon(ctx1, E_POLY.c_str(), g1);
  ParseWKTPoint(NE_POINT.c_str(), g2);
  TestRelate(E_POLY.c_str(), NE_POINT.c_str(), g1, g2);
}

TEST(RelatePolygonTest, PolygonEmpty_vs_LineString) {
  Polygon<point_t, index_t> g1;
  LineString<point_t> g2;
  Context<point_t, index_t> ctx1, ctx2;
  ParseWKTPolygon(ctx1, E_POLY.c_str(), g1);
  ParseWKTLineString(ctx2, NE_LINE.c_str(), g2);
  TestRelate(E_POLY.c_str(), NE_LINE.c_str(), g1, g2);
}

TEST(RelatePolygonTest, PolygonEmpty_vs_Polygon) {
  Polygon<point_t, index_t> g1, g2;
  Context<point_t, index_t> ctx1, ctx2;
  ParseWKTPolygon(ctx1, E_POLY.c_str(), g1);
  ParseWKTPolygon(ctx2, NE_POLY.c_str(), g2);
  TestRelate(E_POLY.c_str(), NE_POLY.c_str(), g1, g2);
}


TEST(RelatePolygonTest, MultiPointEmpty_vs_Polygon) {
  MultiPoint<point_t> g1;
  Polygon<point_t, index_t> g2;
  Context<point_t, index_t> ctx1, ctx2;
  ParseWKTMultiPoint(ctx1, E_MPOINT.c_str(), g1);
  ParseWKTPolygon(ctx2, NE_POLY.c_str(), g2);
  TestRelate(E_MPOINT.c_str(), NE_POLY.c_str(), g1, g2);
}

TEST(RelatePolygonTest, MultiLineStringEmpty_vs_Polygon) {
  MultiLineString<point_t, index_t> g1;
  Polygon<point_t, index_t> g2;
  Context<point_t, index_t> ctx1, ctx2;
  ParseWKTMultiLineString(ctx1, E_MLINE.c_str(), g1);
  ParseWKTPolygon(ctx2, NE_POLY.c_str(), g2);
  TestRelate(E_MLINE.c_str(), NE_POLY.c_str(), g1, g2);
}

// --- MULTIPOLYGON EMPTY vs Non-Empty ---

TEST(RelatePolygonTest, MultiPolygonEmpty_vs_Point) {
  MultiPolygon<point_t, index_t> g1;
  point_t g2;
  Context<point_t, index_t> ctx1;
  ParseWKTMultiPolygon(ctx1, E_MPOLY.c_str(), g1);
  ParseWKTPoint(NE_POINT.c_str(), g2);
  TestRelate(E_MPOLY.c_str(), NE_POINT.c_str(), g1, g2);
}

TEST(RelatePolygonTest, MultiPolygonEmpty_vs_LineString) {
  MultiPolygon<point_t, index_t> g1;
  LineString<point_t> g2;
  Context<point_t, index_t> ctx1, ctx2;
  ParseWKTMultiPolygon(ctx1, E_MPOLY.c_str(), g1);
  ParseWKTLineString(ctx2, NE_LINE.c_str(), g2);
  TestRelate(E_MPOLY.c_str(), NE_LINE.c_str(), g1, g2);
}

TEST(RelatePolygonTest, MultiPolygonEmpty_vs_Polygon) {
  MultiPolygon<point_t, index_t> g1;
  Polygon<point_t, index_t> g2;
  Context<point_t, index_t> ctx1, ctx2;
  ParseWKTMultiPolygon(ctx1, E_MPOLY.c_str(), g1);
  ParseWKTPolygon(ctx2, NE_POLY.c_str(), g2);
  TestRelate(E_MPOLY.c_str(), NE_POLY.c_str(), g1, g2);
}

TEST(RelatePolygonTest, Point_vs_PolygonEmpty) {
  point_t g1;
  Polygon<point_t, index_t> g2;
  Context<point_t, index_t> ctx2;
  ParseWKTPoint(NE_POINT.c_str(), g1);
  ParseWKTPolygon(ctx2, E_POLY.c_str(), g2);
  TestRelate(NE_POINT.c_str(), E_POLY.c_str(), g1, g2);
}


TEST(RelatePolygonTest, LineString_vs_PolygonEmpty) {
  LineString<point_t> g1;
  Polygon<point_t, index_t> g2;
  Context<point_t, index_t> ctx1, ctx2;
  ParseWKTLineString(ctx1, NE_LINE.c_str(), g1);
  ParseWKTPolygon(ctx2, E_POLY.c_str(), g2);
  TestRelate(NE_LINE.c_str(), E_POLY.c_str(), g1, g2);
}

// --- Non-Empty POLYGON vs Empty ---

TEST(RelatePolygonTest, Polygon_vs_PointEmpty) {
  Polygon<point_t, index_t> g1;
  point_t g2;
  Context<point_t, index_t> ctx1;
  ParseWKTPolygon(ctx1, NE_POLY.c_str(), g1);
  ParseWKTPoint(E_POINT.c_str(), g2);
  TestRelate(NE_POLY.c_str(), E_POINT.c_str(), g1, g2);
}

TEST(RelatePolygonTest, Polygon_vs_LineStringEmpty) {
  Polygon<point_t, index_t> g1;
  LineString<point_t> g2;
  Context<point_t, index_t> ctx1, ctx2;
  ParseWKTPolygon(ctx1, NE_POLY.c_str(), g1);
  ParseWKTLineString(ctx2, E_LINE.c_str(), g2);
  TestRelate(NE_POLY.c_str(), E_LINE.c_str(), g1, g2);
}

TEST(RelatePolygonTest, Polygon_vs_PolygonEmpty) {
  Polygon<point_t, index_t> g1, g2;
  Context<point_t, index_t> ctx1, ctx2;
  ParseWKTPolygon(ctx1, NE_POLY.c_str(), g1);
  ParseWKTPolygon(ctx2, E_POLY.c_str(), g2);
  TestRelate(NE_POLY.c_str(), E_POLY.c_str(), g1, g2);
}

TEST(RelatePolygonTest, PreparedTest) {
  Polygon<point_t, index_t> poly1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))";
  std::string wkt2 = "POLYGON((0.5 0.5, 1.5 0.5, 1.5 1.5, 0.5 1.5, 0.5 0.5))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, poly2);
}


//======= Polygon RT Specific Tests =============

TEST(RelatePolygonRTTest, EqualsAndShiftedRingVertices) {
  // Identical polygons with shifted starting vertices and reversed ring orientation
  TestRelatePolygonRT("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
                      "POLYGON ((10 0, 10 10, 0 10, 0 0, 10 0))");
  TestRelatePolygonRT("POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
                      "POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))");
  // Identical polygons with holes
  TestRelatePolygonRT(
      "POLYGON ((0 0, 20 0, 20 20, 0 20, 0 0), (5 5, 15 5, 15 15, 5 15, 5 5))",
      "POLYGON ((20 0, 20 20, 0 20, 0 0, 20 0), (15 5, 15 15, 5 15, 5 5, 15 5))");
}

TEST(RelatePolygonRTTest, HoleAnd100PercentCollinearBoundary) {
  // Polygon A has a hole [5,5]-[15,15].
  std::string poly_with_hole =
      "POLYGON ((0 0, 20 0, 20 20, 0 20, 0 0), (5 5, 15 5, 15 15, 5 15, 5 5))";
  // Polygon B is strictly inside the hole -> Disjoint (FF2FF1212)
  TestRelatePolygonRT(poly_with_hole.c_str(),
                      "POLYGON ((7 7, 13 7, 13 13, 7 13, 7 7))");
  // Polygon B exactly fills the hole -> 100% of B's boundary is collinear with A's hole
  // boundary, while B's interior is in A's exterior! Exercises FindPolygonInteriorPoint.
  TestRelatePolygonRT(poly_with_hole.c_str(),
                      "POLYGON ((5 5, 15 5, 15 15, 5 15, 5 5))");
  // Polygon B has the exact same outer shell as A, while A has a hole -> B covers/contains
  // A, and 100% of B's boundary is collinear with A's outer shell!
  TestRelatePolygonRT(poly_with_hole.c_str(),
                      "POLYGON ((0 0, 20 0, 20 20, 0 20, 0 0))");
}

TEST(RelatePolygonRTTest, MultipleCollinearSegmentsOnSingleEdge) {
  // A has a single long bottom edge from (0,0) to (100,0).
  // B is a MultiPolygon / comb-shaped polygon whose top boundary touches A's bottom edge
  // across multiple disjoint collinear intervals and gaps, exercising the t_cur cursor loop.
  std::string a = "POLYGON ((0 0, 100 0, 100 20, 0 20, 0 0))";
  std::string b_touch =
      "POLYGON ((10 0, 20 0, 20 -5, 30 -5, 30 0, 40 0, 40 -5, 50 -5, 50 0, 60 0, "
      "60 -10, 10 -10, 10 0))";
  TestRelatePolygonRT(a.c_str(), b_touch.c_str());

  // Same comb-shaped polygon, but one tooth pokes into A's interior (y = +2)
  std::string b_overlap =
      "POLYGON ((10 0, 20 0, 20 -5, 30 -5, 30 2, 40 2, 40 -5, 50 -5, 50 0, 60 0, "
      "60 -10, 10 -10, 10 0))";
  TestRelatePolygonRT(a.c_str(), b_overlap.c_str());
}

TEST(RelatePolygonRTTest, MultiPolygonAllCombinationsAndPredicates) {
  std::string mp1 =
      "MULTIPOLYGON (((0 0, 10 0, 10 10, 0 10, 0 0)), ((20 0, 30 0, 30 10, 20 10, 20 0)))";
  std::string mp2_equal =
      "MULTIPOLYGON (((20 0, 30 0, 30 10, 20 10, 20 0)), ((0 0, 10 0, 10 10, 0 10, 0 0)))";
  std::string mp2_within =
      "MULTIPOLYGON (((1 1, 9 1, 9 9, 1 9, 1 1)), ((22 2, 28 2, 28 8, 22 8, 22 2)))";
  std::string mp2_covered =
      "MULTIPOLYGON (((0 0, 5 0, 5 5, 0 5, 0 0)), ((20 0, 25 0, 25 5, 20 5, 20 0)))";
  std::string mp2_touch_edge =
      "MULTIPOLYGON (((10 0, 20 0, 20 10, 10 10, 10 0)))";
  std::string mp2_touch_vertex =
      "MULTIPOLYGON (((10 10, 20 10, 20 20, 10 20, 10 10)))";
  std::string mp2_disjoint_in_gap =
      "MULTIPOLYGON (((12 2, 18 2, 18 8, 12 8, 12 2)))";

  TestRelatePolygonRT(mp1.c_str(), mp2_equal.c_str());
  TestRelatePolygonRT(mp1.c_str(), mp2_within.c_str());
  TestRelatePolygonRT(mp1.c_str(), mp2_covered.c_str());
  TestRelatePolygonRT(mp1.c_str(), mp2_touch_edge.c_str());
  TestRelatePolygonRT(mp1.c_str(), mp2_touch_vertex.c_str());
  TestRelatePolygonRT(mp1.c_str(), mp2_disjoint_in_gap.c_str());
}

TEST(RelatePolygonRTTest, BatchMultiPairSortingAndAlignment) {
  // Verify that when evaluating many candidate pairs out-of-order across Pass 1 (sorted by
  // id2) and Pass 2 (re-sorted by id1 with IMs), every pair's IM is properly aligned.
  std::vector<std::string> wkts1 = {
      "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
      "POLYGON ((20 20, 30 20, 30 30, 20 30, 20 20))",
      "POLYGON ((0 0, 50 0, 50 50, 0 50, 0 0))",
      "POLYGON ((100 100, 110 100, 110 110, 100 110, 100 100))",
  };
  std::vector<std::string> wkts2 = {
      "POLYGON ((5 5, 15 5, 15 15, 5 15, 5 5))",
      "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
      "POLYGON ((10 0, 20 0, 20 10, 10 10, 10 0))",
      "POLYGON ((22 22, 28 22, 28 28, 22 28, 22 22))",
      "POLYGON ((200 200, 210 200, 210 210, 200 210, 200 200))",
  };

  auto cuda_stream = rmm::cuda_stream_default;
  auto dev_geom1 = LoadDeviceGeometriesFromWKT(wkts1);
  auto dev_geom2 = LoadDeviceGeometriesFromWKT(wkts2);
  auto rt_engine = GetSharedRTEngine();
  RelateEngine<point_t, index_t> engine(&dev_geom1, rt_engine.get());

  geos::io::WKTReader reader;
  std::vector<std::unique_ptr<geos::geom::Geometry>> geos1, geos2;
  for (const auto& w : wkts1) geos1.push_back(reader.read(w));
  for (const auto& w : wkts2) geos2.push_back(reader.read(w));

  const std::vector<Predicate> predicates = {
      Predicate::kEquals,   Predicate::kDisjoint, Predicate::kTouches,
      Predicate::kContains, Predicate::kCovers,   Predicate::kIntersects,
      Predicate::kWithin,   Predicate::kCoveredBy};

  for (Predicate p : predicates) {
    std::vector<index_t> h_ids1, h_ids2;
    std::vector<std::pair<index_t, index_t>> expected_pairs;
    for (index_t i = 0; i < wkts1.size(); ++i) {
      for (index_t j = 0; j < wkts2.size(); ++j) {
        h_ids1.push_back(i);
        h_ids2.push_back(j);
        if (TestUtils::EvaluateGeosPredicate(p, geos1[i].get(), geos2[j].get())) {
          expected_pairs.emplace_back(i, j);
        }
      }
    }

    rmm::device_uvector<index_t> d_ids1(h_ids1.size(), cuda_stream);
    rmm::device_uvector<index_t> d_ids2(h_ids2.size(), cuda_stream);
    CUDA_CHECK(cudaMemcpyAsync(d_ids1.data(), h_ids1.data(),
                               h_ids1.size() * sizeof(index_t), cudaMemcpyHostToDevice,
                               cuda_stream.value()));
    CUDA_CHECK(cudaMemcpyAsync(d_ids2.data(), h_ids2.data(),
                               h_ids2.size() * sizeof(index_t), cudaMemcpyHostToDevice,
                               cuda_stream.value()));

    engine.Evaluate(cuda_stream, dev_geom2, p, d_ids1, d_ids2);
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
        << "Batch mismatch on predicate " << PredicateToString(p);
  }
}


TEST(RelatePolygonRTTest, GeosDisparityCases) {
  // Case 1: Empty Polygon vs Non-Empty Polygon
  TestRelatePolygonRT("POLYGON EMPTY", "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))");

  // Case 2: Polygon with hole touching outer shell at a single vertex (valid OGC polygon)
  TestRelatePolygonRT(
      "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (5 0, 8 5, 2 5, 5 0))",
      "POLYGON ((2 5, 8 5, 5 9, 2 5))", false);
}

struct GeosPolyPolyCase {
  const char* name;
  const char* wkt_a;
  const char* wkt_b;
};

TEST(RelatePolygonRTTest, AllGeosPolygonPolygonTestSuite) {
  static const std::vector<GeosPolyPolyCase> cases = {
    {"RelateNGTest::test<42> PolygonsEdgeAdjacent", "POLYGON ((1 3, 3 3, 3 1, 1 1, 1 3))", "POLYGON ((5 3, 5 1, 3 1, 3 3, 5 3))"},
    {"RelateNGTest::test<43> PolygonsEdgeAdjacent2", "POLYGON ((1 3, 4 3, 3 0, 1 1, 1 3))", "POLYGON ((5 3, 5 1, 3 0, 4 3, 5 3))"},
    {"RelateNGTest::test<44> PolygonsNested", "POLYGON ((1 9, 9 9, 9 1, 1 1, 1 9))", "POLYGON ((2 8, 8 8, 8 2, 2 2, 2 8))"},
    {"RelateNGTest::test<45> PolygonsOverlapProper", "POLYGON ((1 1, 1 7, 7 7, 7 1, 1 1))", "POLYGON ((2 8, 8 8, 8 2, 2 2, 2 8))"},
    {"RelateNGTest::test<46> PolygonsOverlapAtNodes", "POLYGON ((1 5, 5 5, 5 1, 1 1, 1 5))", "POLYGON ((7 3, 5 1, 3 3, 5 5, 7 3))"},
    {"RelateNGTest::test<47> PolygonsContainedAtNodes", "POLYGON ((1 5, 5 5, 6 2, 1 1, 1 5))", "POLYGON ((1 1, 5 5, 6 2, 1 1))"},
    {"RelateNGTest::test<48> PolygonsNestedWithHole", "POLYGON ((40 60, 420 60, 420 320, 40 320, 40 60), (200 140, 160 220, 260 200, 200 140))", "POLYGON ((80 100, 360 100, 360 280, 80 280, 80 100))"},
    {"RelateNGTest::test<49> PolygonsOverlappingWithBoundaryInside", "POLYGON ((100 60, 140 100, 100 140, 60 100, 100 60))", "MULTIPOLYGON (((80 40, 120 40, 120 80, 80 80, 80 40)), ((120 80, 160 80, 160 120, 120 120, 120 80)), ((80 120, 120 120, 120 160, 80 160, 80 120)), ((40 80, 80 80, 80 120, 40 120, 40 80)))"},
    {"RelateNGTest::test<50> PolygonsOverlapVeryNarrow", "POLYGON ((120 100, 120 200, 200 200, 200 100, 120 100))", "POLYGON ((100 100, 100000 110, 100000 100, 100 100))"},
    {"RelateNGTest::test<51> ValidateRelateAA_86", "POLYGON ((170 120, 300 120, 250 70, 120 70, 170 120))", "POLYGON ((150 150, 410 150, 280 20, 20 20, 150 150), (170 120, 330 120, 260 50, 100 50, 170 120))"},
    {"RelateNGTest::test<52> ValidateRelateAA_97", "POLYGON ((330 150, 200 110, 150 150, 280 190, 330 150))", "MULTIPOLYGON (((140 110, 260 110, 170 20, 50 20, 140 110)), ((300 270, 420 270, 340 190, 220 190, 300 270)))"},
    {"RelateNGTest::test<53> AdjacentPolygons", "POLYGON ((1 9, 6 9, 6 1, 1 1, 1 9))", "POLYGON ((9 9, 9 4, 6 4, 6 9, 9 9))"},
    {"RelateNGTest::test<54> AdjacentPolygonsTouchingAtPoint", "POLYGON ((1 9, 6 9, 6 1, 1 1, 1 9))", "POLYGON ((9 9, 9 4, 6 4, 7 9, 9 9))"},
    {"RelateNGTest::test<55> AdjacentPolygonsOverlappping", "POLYGON ((1 9, 6 9, 6 1, 1 1, 1 9))", "POLYGON ((9 9, 9 4, 6 4, 5 9, 9 9))"},
    {"RelateNGTest::test<56> ContainsProperlyPolygonContained", "POLYGON ((1 9, 9 9, 9 1, 1 1, 1 9))", "POLYGON ((2 8, 5 8, 5 5, 2 5, 2 8))"},
    {"RelateNGTest::test<57> ContainsProperlyPolygonTouching", "POLYGON ((1 9, 9 9, 9 1, 1 1, 1 9))", "POLYGON ((9 1, 5 1, 5 5, 9 5, 9 1))"},
    {"RelateNGTest::test<58> ContainsProperlyPolygonsOverlapping", "MULTIPOLYGON (((1 9, 6 9, 6 4, 1 4, 1 9)), ((2 4, 6 7, 9 1, 2 4)))", "POLYGON ((5 5, 6 5, 6 4, 5 4, 5 5))"},
    {"RelateNGTest::test<60> RepeatedPointAA", "POLYGON ((1 9, 9 7, 9 1, 1 3, 1 9))", "POLYGON ((1 3, 1 3, 1 3, 3 7, 9 7, 9 7, 1 3))"},
    {"RelateNGTest::test<63> PreparedTest", "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))", "POLYGON((0.5 0.5, 1.5 0.5, 1.5 1.5, 0.5 1.5, 0.5 0.5))"},
    {"RelateNGRobustnessTest::test<13> testGEOS_766", "POLYGON ((26639.240191093646 6039.3615818717535, 26639.240191093646 5889.361620883223, 28000.000095100608 5889.362081553552, 28000.000095100608 6039.361620882992, 28700.00019021402 6039.361620882992, 28700.00019021402 5889.361822800367, 29899.538842431968 5889.362160452064, 32465.59665091549 5889.362882757903, 32969.2837182586 -1313.697771558439, 31715.832811969216 -1489.87008918589, 31681.039836323587 -1242.3030298361555, 32279.3890331618 -1158.210534269224, 32237.63710287376 -861.1301136466199, 32682.89764107368 -802.0828534499739, 32247.445200905553 5439.292852892075, 31797.06861513178 5439.292852892075, 31797.06861513178 5639.36178850523, 29899.538849750803 5639.361268079038, 26167.69458275995 5639.3602445643955, 26379.03654594742 2617.0293071870683, 26778.062167926924 2644.9318977193907, 26792.01346261031 2445.419086759444, 26193.472956813417 2403.5650586598513, 25939.238114175267 6039.361685403233, 26639.240191093646 6039.3615818717535), (32682.89764107368 -802.0828534499738, 32682.89764107378 -802.0828534499669, 32247.445200905655 5439.292852892082, 32247.445200905553 5439.292852892075, 32682.89764107368 -802.0828534499738))", "POLYGON ((32450.100392347143 5889.362314133216, 32050.104955691 5891.272957209961, 32100.021071878822 16341.272221116333, 32500.016508656867 16339.361578039587, 32450.100392347143 5889.362314133216))"},
    {"RelateNGRobustnessTest::test<14> testGEOS_1026", "POLYGON((335645.7810000004 5677846.65,335648.6579999998 5677845.801999999,335650.8630842535 5677845.143617179,335650.77673334075 5677844.7250704905,335642.90299999993 5677847.498,335645.7810000004 5677846.65))", "POLYGON((335642.903 5677847.498,335642.894 5677847.459,335645.92 5677846.69,335647.378 5677852.523,335644.403 5677853.285,335644.374 5677853.293,335642.903 5677847.498))"},
    {"RelateNGRobustnessTest::test<16> testPostGIS_5362", "POLYGON ((-707259.66 -1121493.36, -707205.9 -1121605.808, -707310.5388 -1121540.5446, -707318.8200000001 -1121533.21, -707259.66 -1121493.36))", "POLYGON ((-707356.18 -1121550.69, -707332.82 -1121536.63, -707318.82 -1121533.21, -707321.72 -1121535.08, -707327.4 -1121539.21, -707356.18 -1121550.69))"},
    {"RelateNGRobustnessTest::test<20> testGISSE_484691", "POLYGON ((1.839012980156925 43.169860517728324, 1.838983490127865 43.169860200336274, 1.838898525601717 43.169868281549725, 1.838918565176068 43.1699719478626, 1.838920733577112 43.16998636433192, 1.838978629555589 43.16997979090823, 1.838982586839382 43.169966339940714, 1.838974943184281 43.169918580432174, 1.839020497362873 43.169914572864634, 1.839012980156925 43.169860517728324))", "POLYGON ((1.8391355300979277 43.16987802887805, 1.83913336164737 43.16986361241434, 1.8390129801569248 43.169860517728324, 1.8390790978572837 43.16987292371998, 1.8390909520103162 43.16995581178317, 1.8391377530291442 43.16995091801345, 1.8391293863398452 43.16987796276235, 1.8391355300979277 43.16987802887805))"},
    {"general/TestEqualsExact.xml [13] A - polygon with no holes, permuted", "POLYGON((40 60, 420 60, 420 320, 40 320, 40 60))", "POLYGON((420 60, 420 320, 40 320, 40 60, 420 60))"},
    {"general/TestEqualsExact.xml [17] mA - permuted", "MULTIPOLYGON (((50 100, 100 100, 100 50, 50 50, 50 100)), ((150 100, 200 100, 200 50, 150 50, 150 100)))", "MULTIPOLYGON (((150 100, 200 100, 200 50, 150 50, 150 100)), ((50 100, 100 100, 100 50, 50 50, 50 100)))"},
    {"general/TestPreparedPolygonPredicate.xml [18] A/A - two equal polygons", "POLYGON((20 20, 20 100, 120 100, 140 20, 20 20))", "POLYGON((20 20, 20 100, 120 100, 140 20, 20 20))"},
    {"general/TestRectanglePredicate.xml [1] A disjoint", "POLYGON( (0 0, 80 0, 80 80, 0 80, 0 0))", "POLYGON( (100 200, 100 140, 180 140, 180 200, 100 200))"},
    {"general/TestRectanglePredicate.xml [2] A contained in rectangle", "POLYGON((0 0, 100 0, 100 100, 0 100, 0 0))", "POLYGON((10 10, 10 90, 90 90, 90 10, 10 10))"},
    {"general/TestRectanglePredicate.xml [3] A containing rectangle", "POLYGON((0 0, 100 0, 100 100, 0 100, 0 0))", "POLYGON ((60 180, -100 120, -140 60, -40 20, -100 -80, 40 -20, 140 -100, 140 40, 260 160, 80 120, 60 180))"},
    {"general/TestRectanglePredicate.xml [4] mA containing rectangle", "POLYGON((0 0, 100 0, 100 100, 0 100, 0 0))", "MULTIPOLYGON (((-60 180, -60 -60, 40 -20, 140 -100, 180 120, -20 140, -60 180)), ((20 280, 0 180, 180 160, 200 280, 20 280)))"},
    {"general/TestRelateAA.xml [1] AA disjoint", "POLYGON( (0 0, 80 0, 80 80, 0 80, 0 0))", "POLYGON( (100 200, 100 140, 180 140, 180 200, 100 200))"},
    {"general/TestRelateAA.xml [2] AA equal but opposite orientation", "POLYGON( (0 0, 140 0, 140 140, 0 140, 0 0))", "POLYGON( (140 0, 0 0, 0 140, 140 140, 140 0))"},
    {"general/TestRelateAA.xml [3] AA A-shell contains B-shell", "POLYGON( (40 60, 360 60, 360 300, 40 300, 40 60))", "POLYGON( (120 100, 280 100, 280 240, 120 240, 120 100))"},
    {"general/TestRelateAA.xml [4] AA A-shell contains B-shell contains A-hole", "POLYGON( (40 60, 420 60, 420 320, 40 320, 40 60), (200 140, 160 220, 260 200, 200 140))", "POLYGON( (80 100, 360 100, 360 280, 80 280, 80 100))"},
    {"general/TestRelateAA.xml [5] AA A-shell contains B-shell contains A-hole contains B-hole", "POLYGON( (0 280, 0 0, 260 0, 260 280, 0 280), (220 240, 40 240, 40 40, 220 40, 220 240))", "POLYGON( (20 260, 240 260, 240 20, 20 20, 20 260), (160 180, 80 180, 120 120, 160 180))"},
    {"general/TestRelateAA.xml [6] AA A-shell overlapping B-shell", "POLYGON( (60 80, 200 80, 200 220, 60 220, 60 80))", "POLYGON( (120 140, 260 140, 260 260, 120 260, 120 140))"},
    {"general/TestRelateAA.xml [7] AA A-shell overlapping B-shell at B-vertex", "POLYGON( (60 220, 220 220, 140 140, 60 220))", "POLYGON( (100 180, 180 180, 180 100, 100 100, 100 180))"},
    {"general/TestRelateAA.xml [8] AA A-shell overlapping B-shell at A & B-vertex", "POLYGON( (40 40, 180 40, 180 180, 40 180, 40 40))", "POLYGON( (180 40, 40 180, 160 280, 300 140, 180 40))"},
    {"general/TestRelateAA.xml [9] AmA A-shells overlapping B-shell at A-vertex", "POLYGON( (100 60, 140 100, 100 140, 60 100, 100 60))", "MULTIPOLYGON( ( (80 40, 120 40, 120 80, 80 80, 80 40)), ( (120 80, 160 80, 160 120, 120 120, 120 80)), ( (80 120, 120 120, 120 160, 80 160, 80 120)), ( (40 80, 80 80, 80 120, 40 120, 40 80)))"},
    {"general/TestRelateAA.xml [10] AA A-shell touches B-shell, which contains A-hole", "POLYGON( (40 280, 200 280, 200 100, 40 100, 40 280), (100 220, 120 220, 120 200, 100 180, 100 220))", "POLYGON( (40 280, 180 260, 180 120, 60 120, 40 280))"},
    {"general/TestRelateAA.xml [11] AA - A-hole contains B, boundaries touch in line", "POLYGON( (0 200, 0 0, 200 0, 200 200, 0 200), (20 180, 130 180, 130 30, 20 30, 20 180))", "POLYGON( (60 90, 130 90, 130 30, 60 30, 60 90))"},
    {"general/TestRelateAA.xml [12] AA - A-hole contains B, boundaries touch in points", "POLYGON( (150 150, 410 150, 280 20, 20 20, 150 150), (170 120, 330 120, 260 50, 100 50, 170 120))", "POLYGON( (270 90, 200 50, 150 80, 210 120, 270 90))"},
    {"general/TestRelateAA.xml [13] AA - A contained completely in B", "POLYGON ((0 0, 20 80, 120 80, -20 120, 0 0))", "POLYGON ((60 180, -100 120, -140 60, -40 20, -100 -80, 40 -20, 140 -100, 140 40, 260 160, 80 120, 60 180))"},
    {"general/TestRelateAA.xml [14] A/mA A-shells overlapping B-shell at A-vertex", "POLYGON ((100 60, 140 100, 100 140, 60 100, 100 60))", "MULTIPOLYGON (((80 40, 120 40, 120 80, 80 80, 80 40)), ((120 80, 160 80, 160 120, 120 120, 120 80)), ((80 120, 120 120, 120 160, 80 160, 80 120)), ((40 80, 80 80, 80 120, 40 120, 40 80)))"},
    {"general/TestRelateEmpty.xml [17] A/A - empty polygon VS empty polygon", "POLYGON EMPTY", "POLYGON EMPTY"},
    {"general/TestRelateEmpty.xml [20] A/mP - empty polygon VS empty MultiPolygon", "POLYGON EMPTY", "MULTIPOLYGON EMPTY"},
    {"general/TestRelateEmpty.xml [38] mA/A - empty Multipolygon VS empty polygon", "MULTIPOLYGON EMPTY", "POLYGON EMPTY"},
    {"general/TestRelateEmpty.xml [41] mA/mA - empty Multipolygon VS empty MultiPolygon", "MULTIPOLYGON EMPTY", "MULTIPOLYGON EMPTY"},
    {"issue/issue-geos-716.xml [1] http://trac.osgeo.org/geos/ticket/716 a CW, b CW", "POLYGON((0.04745459120333707 7.998990511152711, 10.222919749090828 34.08521670606894, 10 10, 0.04745459120333707 7.998990511152711))", "POLYGON((15 5, 0.010044793132693013 7.903085268568247, 10.260329547338191 34.181121949106455, 15 5))"},
    {"issue/issue-geos-716.xml [2] http://trac.osgeo.org/geos/ticket/716 a CCW, b CW", "POLYGON((0.04745459120333707 7.998990511152711, 10 10, 10.222919749090828 34.08521670606894, 0.04745459120333707 7.998990511152711))", "POLYGON((15 5, 0.010044793132693013 7.903085268568247, 10.260329547338191 34.181121949106455, 15 5))"},
    {"issue/issue-geos-716.xml [3] http://trac.osgeo.org/geos/ticket/716 a CW, b CCW", "POLYGON((0.04745459120333707 7.998990511152711, 10.222919749090828 34.08521670606894, 10 10, 0.04745459120333707 7.998990511152711))", "POLYGON((15 5, 10.260329547338191 34.181121949106455, 0.010044793132693013 7.903085268568247, 15 5))"},
    {"issue/issue-geos-716.xml [4] http://trac.osgeo.org/geos/ticket/716 a CCW, b CCW", "POLYGON((0.04745459120333707 7.998990511152711, 10 10, 10.222919749090828 34.08521670606894, 0.04745459120333707 7.998990511152711))", "POLYGON((15 5, 10.260329547338191 34.181121949106455, 0.010044793132693013 7.903085268568247, 15 5))"},
    {"misc/InvalidRelates.xml [1] A/A-3-1-8: the closing point of a polygon touching the boundary of another polygon where the closing point touching the boundary at a vertex [dim(0){A.A.Bdy.CP = B.A.Bdy.TP}]", "POLYGON((160 200, 210 70, 120 70, 160 200))", "POLYGON((160 200, 260 40, 70 40, 160 200, 20 20, 310 20, 160 200))"},
    {"misc/InvalidRelates.xml [2] A/A-3-1-9: the closing point of a polygon touching the boundary of another polygon where the closing point intersecting the boundary at a non-vertex [dim(0){A.A.Bdy.CP = B.A.Bdy.TP}]", "POLYGON((110 140, 200 70, 200 160, 110 140))", "POLYGON((110 140, 110 50, 60 50, 60 90, 160 190, 20 110, 20 20, 200 20, 110 140))"},
    {"validate/TestRelateAA-big.xml [1] A/A-6-18: a polygon overlapping a very skinny polygon [dim(2){A.A.Int = B.A.Int}, dim(1){A.A.Bdy.V-EP = B.A.Bdy.NV-EP}, dim(0){A.A.Bdy.CP = B.A.Bdy.CP}, dim(0){A.A.Bdy.NV = B.A.Bdy.NV}]", "POLYGON( (100 100, 100 200, 200 200, 200 100, 100 100))", "POLYGON( (100 100, 1000000000000000 110, 1000000000000000 100, 100 100))"},
    {"validate/TestRelateAA-big.xml [2] A/A-6-24: a polygon overlapping a very skinny polygon [dim(2){A.A.Int = B.A.Int}, dim(1){A.A.Bdy.V-EP = B.A.Bdy.NV-NV}, dim(0){A.A.Bdy.NV = B.A.Bdy.NV}]", "POLYGON( (120 100, 120 200, 200 200, 200 100, 120 100))", "POLYGON( (100 100, 1000000000000000 110, 1000000000000000 100, 100 100))"},
    {"validate/TestRelateAA.xml [1] A/A-1-1: same polygons [dim(2){A.A.Int = B.A.Int}, dim(1){A.A.Bdy.SP-EP = B.A.Bdy.SP-EP}]", "POLYGON( (20 20, 20 100, 120 100, 140 20, 20 20))", "POLYGON( (20 20, 20 100, 120 100, 140 20, 20 20))"},
    {"validate/TestRelateAA.xml [2] A/A-1-2: same polygons with reverse sequence of points [dim(2){A.A.Int = B.A.Int}, dim(1){A.A.Bdy.SP-EP = B.A.Bdy.EP-SP}]", "POLYGON( (20 20, 20 100, 120 100, 140 20, 20 20))", "POLYGON( (20 20, 140 20, 120 100, 20 100, 20 20))"},
    {"validate/TestRelateAA.xml [3] A/A-1-3: same polygons with different sequence of points [dim(2){A.A.Int = B.A.Int}, dim(1){A.A.Bdy.SP-EP = B.A.Bdy.SP-EP}]", "POLYGON( (20 20, 20 100, 120 100, 140 20, 20 20))", "POLYGON( (120 100, 140 20, 20 20, 20 100, 120 100))"},
    {"validate/TestRelateAA.xml [4] A/A-1-4: same polygons with different number of points [dim(2){A.A.Int = B.A.Int}, dim(1){A.A.Bdy.SP-EP = B.A.Bdy.SP-EP}]", "POLYGON( (20 20, 20 100, 120 100, 140 20, 20 20))", "POLYGON( (20 100, 60 100, 120 100, 140 20, 80 20, 20 20, 20 100))"},
    {"validate/TestRelateAA.xml [5] A/A-2: different polygons [dim(2){A.A.Int = B.A.Ext}]", "POLYGON( (0 0, 80 0, 80 80, 0 80, 0 0))", "POLYGON( (100 200, 100 140, 180 140, 180 200, 100 200))"},
    {"validate/TestRelateAA.xml [6] A/A-3-1-1: the closing point of a polygon touching the closing point of another polygon [dim(0){A.A.Bdy.CP = B.A.Bdy.CP}]", "POLYGON( (140 120, 160 20, 20 20, 20 120, 140 120))", "POLYGON( (140 120, 140 200, 240 200, 240 120, 140 120))"},
    {"validate/TestRelateAA.xml [7] A/A-3-1-2: the closing point of a polygon touching the boundary (at a non-vertex) of another polygon [dim(0){A.A.Bdy.CP = B.A.Bdy.NV}]", "POLYGON( (140 120, 160 20, 20 20, 20 120, 140 120))", "POLYGON( (80 180, 140 260, 260 200, 200 60, 80 180))"},
    {"validate/TestRelateAA.xml [8] A/A-3-1-3: the closing point of a polygon touching the boundary (at a vertex) of another polygon [dim(0){A.A.Bdy.CP = B.A.Bdy.V}]", "POLYGON( (140 120, 160 20, 20 20, 20 120, 140 120))", "POLYGON( (240 80, 140 120, 180 240, 280 200, 240 80))"},
    {"validate/TestRelateAA.xml [9] A/A-3-1-4: the boundary (at a non-vertex) of a polygon touching the closing point of another polygon [dim(0){A.A.Bdy.NV = B.A.Bdy.CP}]", "POLYGON( (140 160, 20 20, 270 20, 150 160, 230 40, 60 40, 140 160))", "POLYGON( (140 40, 180 80, 120 100, 140 40))"},
    {"validate/TestRelateAA.xml [10] A/A-3-1-5: the boundary (at a non-vertex) of a polygon touching the boundary (at a vertex) of another polygon [dim(0){A.A.Bdy.NV = B.A.Bdy.V}]", "POLYGON( (140 160, 20 20, 270 20, 150 160, 230 40, 60 40, 140 160))", "POLYGON( (120 100, 180 80, 130 40, 120 100))"},
    {"validate/TestRelateAA.xml [11] A/A-3-1-6: the boundary (at a vertex) of a polygon touching the boundary (at a non-vertex) of another polygon [dim(0){A.A.Bdy.V = B.A.Bdy.NV}]", "POLYGON( (20 20, 180 20, 140 140, 20 140, 20 20))", "POLYGON( (180 100, 80 200, 180 280, 260 200, 180 100))"},
    {"validate/TestRelateAA.xml [12] A/A-3-1-7: the boundary (at a vertex) of a polygon touching the boundary (at a vertex) of another polygon [dim(0){A.A.Bdy.V = B.A.Bdy.V}]", "POLYGON( (140 120, 160 20, 20 20, 20 120, 140 120))", "POLYGON( (140 140, 20 120, 0 220, 120 240, 140 140))"},
    {"validate/TestRelateAA.xml [13] A/A-3-2-1: two polygons touching at multiple points [dim(0){A.A.Bdy.CP = B.A.Bdy.CP}, dim(0){A.A.Bdy.V = B.A.Bdy.V}]", "POLYGON( (20 120, 20 20, 260 20, 260 120, 200 40, 140 120, 80 40, 20 120))", "POLYGON( (20 120, 20 240, 260 240, 260 120, 200 200, 140 120, 80 200, 20 120))"},
    {"validate/TestRelateAA.xml [14] A/A-3-2-2: two polygons touching at multiple points [dim(0){A.A.Bdy.CP = B.A.Bdy.CP}, dim(0){A.A.Bdy.V = B.A.Bdy.NV}]", "POLYGON( (20 120, 20 20, 260 20, 260 120, 180 40, 140 120, 100 40, 20 120))", "POLYGON( (20 120, 300 120, 140 240, 20 120))"},
    {"validate/TestRelateAA.xml [15] A/A-3-2-3: two polygons touching at multiple points [dim(0){A.A.Bdy.CP = B.A.Bdy.NV}, dim(0){A.A.Bdy.V = B.A.Bdy.NV}]", "POLYGON( (20 20, 20 300, 280 300, 280 260, 220 260, 60 100, 60 60, 280 60, 280 20, 20 20))", "POLYGON( (100 140, 160 80, 280 180, 200 240, 220 160, 160 200, 180 120, 100 140))"},
    {"validate/TestRelateAA.xml [16] A/A-3-2-4: two polygons touching at multiple points [dim(0){A.A.Bdy.V = B.A.Bdy.NV}]", "POLYGON( (20 20, 20 300, 280 300, 280 260, 220 260, 60 100, 60 60, 280 60, 280 20, 20 20))", "POLYGON( (260 200, 180 80, 120 160, 200 160, 180 220, 260 200))"},
    {"validate/TestRelateAA.xml [17] A/A-3-2-5: two polygons touching at multiple points [dim(0){A.A.Bdy.V = B.A.Bdy.NV}]", "POLYGON( (20 20, 280 20, 280 140, 220 60, 140 140, 80 60, 20 140, 20 20))", "POLYGON( (0 140, 300 140, 140 240, 0 140))"},
    {"validate/TestRelateAA.xml [18] A/A-3-2-6: two polygons touching at multiple points [dim(0){A.A.Bdy.V = B.A.Bdy.V}, dim(0){A.A.Bdy.V = B.A.Bdy.NV}]", "POLYGON( (20 20, 280 20, 280 140, 220 60, 140 140, 80 60, 20 140, 20 20))", "POLYGON( (20 240, 20 140, 320 140, 180 240, 20 240))"},
    {"validate/TestRelateAA.xml [19] A/A-3-2-7: two polygons touching at multiple points [dim(0){A.A.Bdy.V = B.A.Bdy.V}]", "POLYGON( (20 20, 280 20, 280 140, 220 60, 140 140, 80 60, 20 140, 20 20))", "POLYGON( (20 240, 20 140, 80 180, 140 140, 220 180, 280 140, 280 240, 20 240))"},
    {"validate/TestRelateAA.xml [20] A/A-3-3-1: two polygons touching along a boundary [dim(1){A.A.Bdy.SP-V = B.A.Bdy.SP-NV}]", "POLYGON( (120 120, 180 60, 20 20, 20 120, 120 120))", "POLYGON( (120 120, 220 20, 280 20, 240 160, 120 120))"},
    {"validate/TestRelateAA.xml [21] A/A-3-3-2: two polygons touching along a boundary [dim(1){A.A.Bdy.SP-V = B.A.Bdy.SP-V}]", "POLYGON( (140 120, 160 20, 20 20, 20 120, 140 120))", "POLYGON( (140 120, 160 20, 260 120, 220 200, 140 120))"},
    {"validate/TestRelateAA.xml [22] A/A-3-3-3: two polygons touching along a boundary [dim(1){A.A.Bdy.SP-V = B.A.Bdy.NV-V}]", "POLYGON( (20 140, 120 40, 20 40, 20 140))", "POLYGON( (190 140, 190 20, 140 20, 20 140, 190 140))"},
    {"validate/TestRelateAA.xml [23] A/A-3-3-4: two polygons touching along a boundary [dim(1){A.A.Bdy.SP-V = B.A.Bdy.NV-V}]", "POLYGON( (120 120, 180 60, 20 20, 20 120, 120 120))", "POLYGON( (300 20, 220 20, 120 120, 260 160, 300 20))"},
    {"validate/TestRelateAA.xml [24] A/A-3-3-5: two polygons touching along a boundary [dim(1){A.A.Bdy.SP-V = B.A.Bdy.V-EP}]", "POLYGON( (140 120, 160 20, 20 20, 20 120, 140 120))", "POLYGON( (140 120, 240 160, 280 60, 160 20, 140 120))"},
    {"validate/TestRelateAA.xml [25] A/A-3-3-6: two polygons touching along a boundary [dim(1){A.A.Bdy.SP-V = B.A.Bdy.V-V}]", "POLYGON( (120 120, 180 60, 20 20, 20 120, 120 120))", "POLYGON( (280 60, 180 60, 120 120, 260 180, 280 60))"},
    {"validate/TestRelateAA.xml [26] A/A-3-3-7: two polygons touching along a boundary [dim(1){A.A.Bdy.NV-NV = B.A.Bdy.V-V}]", "POLYGON( (140 120, 160 20, 20 20, 20 120, 140 120))", "POLYGON( (120 200, 120 120, 40 120, 40 200, 120 200))"},
    {"validate/TestRelateAA.xml [27] A/A-3-3-8: two polygons touching along a boundary [dim(1){A.A.Bdy.NV-EP = B.A.Bdy.V-V}]", "POLYGON( (140 120, 160 20, 20 20, 20 120, 140 120))", "POLYGON( (160 220, 140 120, 60 120, 40 220, 160 220))"},
    {"validate/TestRelateAA.xml [28] A/A-3-3-9: two polygons touching along a boundary [dim(1){A.A.Bdy.V-EP = B.A.Bdy.V-SP}]", "POLYGON( (140 120, 160 20, 20 20, 20 120, 140 120))", "POLYGON( (140 120, 20 120, 20 220, 140 220, 140 120))"},
    {"validate/TestRelateAA.xml [29] A/A-3-3-10: two polygons touching along a boundary [dim(1){A.A.Bdy.V-V = B.A.Bdy.NV-NV}]", "POLYGON( (120 120, 180 60, 20 20, 20 120, 120 120))", "POLYGON( (320 20, 220 20, 80 160, 240 140, 320 20))"},
    {"validate/TestRelateAA.xml [30] A/A-5-1: a polygon containing another polygon [dim(2){A.A.Int = B.A.Int}, dim(1){A.A.Bdy.SP-EP = B.A.Int}]", "POLYGON( (20 20, 20 180, 220 180, 220 20, 20 20))", "POLYGON( (60 40, 60 140, 180 140, 180 40, 60 40))"},
    {"validate/TestRelateAA.xml [31] A/A-5-2-1: a polygon containing another polygon [dim(2){A.A.Int = B.A.Int}, dim(0){A.A.Bdy.CP = B.A.Bdy.CP}]", "POLYGON( (20 20, 20 180, 220 180, 220 20, 20 20))", "POLYGON( (20 20, 80 140, 160 60, 20 20))"},
    {"validate/TestRelateAA.xml [32] A/A-5-2-2: a polygon containing another polygon [dim(2){A.A.Int = B.A.Int}, dim(0){A.A.Bdy.CP = B.A.Bdy.V}]", "POLYGON( (20 20, 20 180, 220 180, 220 20, 20 20))", "POLYGON( (160 60, 20 20, 100 140, 160 60))"},
    {"validate/TestRelateAA.xml [33] A/A-5-2-3: a polygon containing another polygon [dim(2){A.A.Int = B.A.Int}, dim(0){A.A.Bdy.NV = B.A.Bdy.CP}]", "POLYGON( (20 20, 20 180, 220 180, 220 20, 20 20))", "POLYGON( (20 100, 140 160, 160 40, 20 100))"},
    {"validate/TestRelateAA.xml [34] A/A-5-2-4: a polygon containing another polygon [dim(2){A.A.Int = B.A.Int}, dim(0){A.A.Bdy.NV = B.A.Bdy.V}]", "POLYGON( (20 20, 20 180, 220 180, 220 20, 20 20))", "POLYGON( (160 40, 20 100, 160 160, 160 40))"},
    {"validate/TestRelateAA.xml [35] A/A-5-2-5: a polygon containing another polygon [dim(2){A.A.Int = B.A.Int}, dim(0){A.A.Bdy.V = B.A.Bdy.CP}]", "POLYGON( (20 20, 20 180, 220 180, 220 20, 20 20))", "POLYGON( (20 180, 180 120, 80 40, 20 180))"},
    {"validate/TestRelateAA.xml [36] A/A-5-2-6: a polygon containing another polygon [dim(2){A.A.Int = B.A.Int}, dim(0){A.A.Bdy.V = B.A.Bdy.V}]", "POLYGON( (20 20, 20 180, 220 180, 220 20, 20 20))", "POLYGON( (180 120, 100 40, 20 180, 180 120))"},
    {"validate/TestRelateAA.xml [37] A/A-5-3-1: a polygon containing another polygon [dim(2){A.A.Int = B.A.Int}, dim(0){A.A.Bdy.CP = B.A.Bdy.CP}, dim(0){A.A.Bdy.NV = B.A.Bdy.V}]", "POLYGON( (20 20, 20 180, 220 180, 220 20, 20 20))", "POLYGON( (20 20, 140 40, 140 120, 20 160, 80 80, 20 20))"},
    {"validate/TestRelateAA.xml [38] A/A-5-3-2: a polygon containing another polygon [dim(2){A.A.Int = B.A.Int}, dim(0){A.A.Bdy.CP = B.A.Bdy.CP}, dim(0){A.A.Bdy.V = B.A.Bdy.V}]", "POLYGON( (20 20, 20 180, 220 180, 220 20, 20 20))", "POLYGON( (20 20, 140 40, 140 140, 20 180, 80 100, 20 20))"},
    {"validate/TestRelateAA.xml [39] A/A-5-3-3: a polygon containing another polygon [dim(2){A.A.Int = B.A.Int}, dim(0){A.A.Bdy.NV = B.A.Bdy.V}, dim(0){A.A.Bdy.NV = B.A.Bdy.V}]", "POLYGON( (20 20, 20 180, 220 180, 220 20, 20 20))", "POLYGON( (40 180, 60 100, 180 100, 200 180, 120 120, 40 180))"},
    {"validate/TestRelateAA.xml [40] A/A-5-3-4: a polygon containing another polygon [dim(2){A.A.Int = B.A.Int}, dim(0){A.A.Bdy.V = B.A.Bdy.CP}, dim(0){A.A.Bdy.V = B.A.Bdy.V}]", "POLYGON( (20 20, 20 180, 220 180, 220 20, 20 20))", "POLYGON( (20 180, 60 80, 180 80, 220 180, 120 120, 20 180))"},
    {"validate/TestRelateAA.xml [41] A/A-5-3-5: a polygon containing another polygon [dim(2){A.A.Int = B.A.Int}, dim(0){A.A.Bdy.V = B.A.Bdy.V}, dim(0){A.A.Bdy.NV = B.A.Bdy.V}, dim(0){A.A.Bdy.NV = B.A.Bdy.V}]", "POLYGON( (20 20, 20 180, 220 180, 220 20, 20 20))", "POLYGON( (40 60, 20 180, 100 100, 140 180, 160 120, 220 100, 140 40, 40 60))"},
    {"validate/TestRelateAA.xml [42] A/A-5-3-6: a polygon containing another polygon [dim(2){A.A.Int = B.A.Int}, dim(0){A.A.Bdy.V = B.A.Bdy.V}, dim(0){A.A.Bdy.V = B.A.Bdy.V}]", "POLYGON( (20 20, 20 180, 220 180, 220 20, 20 20))", "POLYGON( (60 100, 180 100, 220 180, 120 140, 20 180, 60 100))"},
    {"validate/TestRelateAA.xml [43] A/A-5-4-1: a polygon containing another polygon [dim(2){A.A.Int = B.A.Int}, dim(1){A.A.Bdy.SP-NV = B.A.Bdy.SP-V}]", "POLYGON( (20 20, 20 180, 220 180, 220 20, 20 20))", "POLYGON( (20 20, 20 140, 120 120, 120 40, 20 20))"},
    {"validate/TestRelateAA.xml [44] A/A-5-4-2: a polygon containing another polygon [dim(2){A.A.Int = B.A.Int}, dim(1){A.A.Bdy.SP-V = B.A.Bdy.SP-V)}]", "POLYGON( (20 20, 20 180, 220 180, 220 20, 20 20))", "POLYGON( (20 20, 20 180, 140 140, 140 60, 20 20))"},
    {"validate/TestRelateAA.xml [45] A/A-5-4-3: a polygon containing another polygon [dim(2){A.A.Int = B.A.Int}, dim(1){A.A.Bdy.SP-NV = B.A.Bdy.V-EP}]", "POLYGON( (20 20, 20 180, 220 180, 220 20, 20 20))", "POLYGON( (20 20, 120 40, 120 120, 20 140, 20 20))"},
    {"validate/TestRelateAA.xml [46] A/A-5-4-4: a polygon containing another polygon [dim(2){A.A.Int = B.A.Int}, dim(1){A.A.Bdy.SP-NV = B.A.Bdy.V-V}]", "POLYGON( (20 20, 20 180, 220 180, 220 20, 20 20))", "POLYGON( (120 40, 20 20, 20 140, 120 120, 120 40))"},
    {"validate/TestRelateAA.xml [47] A/A-5-4-5: a polygon containing another polygon [dim(2){A.A.Int = B.A.Int}, dim(1){A.A.Bdy.SP-V = B.A.Bdy.V-EP}]", "POLYGON( (20 20, 20 180, 220 180, 220 20, 20 20))", "POLYGON( (20 20, 140 60, 140 140, 20 180, 20 20))"},
    {"validate/TestRelateAA.xml [48] A/A-5-4-6: a polygon containing another polygon [dim(2){A.A.Int = B.A.Int}, dim(1){A.A.Bdy.SP-V = B.A.Bdy.V-V}]", "POLYGON( (20 20, 20 180, 220 180, 220 20, 20 20))", "POLYGON( (140 60, 20 20, 20 180, 140 140, 140 60))"},
    {"validate/TestRelateAA.xml [49] A/A-5-4-7: a polygon containing another polygon [dim(2){A.A.Int = B.A.Int}, dim(1){A.A.Bdy.NV-EP = B.A.Bdy.V-EP}]", "POLYGON( (20 20, 20 180, 220 180, 220 20, 20 20))", "POLYGON( (20 20, 60 120, 140 120, 180 20, 20 20))"},
    {"validate/TestRelateAA.xml [50] A/A-5-4-8: a polygon containing another polygon [dim(2){A.A.Int = B.A.Int}, dim(1){A.A.Bdy.NV-NV = B.A.Bdy.V-EP}]", "POLYGON( (20 20, 20 180, 220 180, 220 20, 20 20))", "POLYGON( (20 40, 120 40, 120 120, 20 140, 20 40))"},
    {"validate/TestRelateAA.xml [51] A/A-5-5-1: a polygon containing another polygon [dim(2){A.A.Int = B.A.Int}, dim(1){A.A.Bdy.SP-V = B.A.Bdy.SP-V}, dim(1){A.A.Bdy.(NV, V) = B.A.Bdy.(V, V)}]", "POLYGON( (20 20, 20 180, 220 180, 220 20, 20 20))", "POLYGON( (20 20, 20 180, 60 120, 100 180, 140 120, 220 180, 200 120, 140 60, 20 20))"},
    {"validate/TestRelateAA.xml [52] A/A-6-1: a polygon overlapping another polygon [dim(2){A.A.Int = B.A.Int}]", "POLYGON( (150 150, 330 150, 250 70, 70 70, 150 150))", "POLYGON( (150 150, 270 150, 140 20, 20 20, 150 150))"},
    {"validate/TestRelateAA.xml [53] A/A-6-2: a polygon overlapping another polygon [dim(2){A.A.Int = B.A.Int}]", "POLYGON( (150 150, 270 150, 330 150, 250 70, 190 70, 70 70, 150 150))", "POLYGON( (150 150, 270 150, 190 70, 140 20, 20 20, 70 70, 150 150))"},
    {"validate/TestRelateAA.xml [54] A/A-6-3: spiky polygons overlapping; boundary <-> boundary intersecting at 0 dimension [dim(2){A.A.Int = B.A.Int}]", "POLYGON( (20 20, 60 50, 20 40, 60 70, 20 60, 60 90, 20 90, 70 110, 20 130, 80 130, 20 150, 80 160, 20 170, 80 180, 20 200, 80 200, 30 240, 80 220, 50 260, 100 220, 100 260, 120 220, 130 260, 140 220, 150 280, 150 190, 160 280, 170 190, 180 280, 190 190, 200 280, 210 190, 220 280, 230 190, 240 260, 250 230, 260 260, 260 220, 290 270, 290 220, 330 260, 300 210, 340 240, 290 180, 340 210, 290 170, 350 170, 240 150, 350 150, 240 140, 350 130, 240 120, 350 120, 240 110, 350 110, 240 100, 350 100, 240 90, 350 90, 240 80, 350 80, 300 70, 340 60, 290 60, 340 40, 300 50, 340 20, 270 60, 310 20, 250 60, 270 20, 230 60, 240 20, 210 60, 210 20, 190 70, 190 20, 180 90, 170 20, 160 90, 150 20, 140 90, 130 20, 120 90, 110 20, 100 90, 100 20, 90 60, 80 20, 70 40, 20 20))", "POLYGON( (190 140, 140 130, 200 160, 130 150, 210 170, 130 170, 210 180, 120 190, 220 200, 120 200, 250 210, 120 210, 250 220, 120 220, 250 230, 120 240, 230 240, 120 250, 240 260, 120 260, 240 270, 120 270, 270 290, 120 290, 230 300, 150 310, 250 310, 180 320, 250 320, 200 360, 260 330, 240 360, 280 320, 290 370, 290 320, 320 360, 310 320, 360 360, 310 310, 380 340, 310 290, 390 330, 310 280, 410 310, 310 270, 420 280, 310 260, 430 250, 300 250, 440 240, 300 240, 450 230, 280 220, 440 220, 280 210, 440 210, 300 200, 430 190, 300 190, 440 180, 330 180, 430 150, 320 180, 420 130, 300 180, 410 120, 280 180, 400 110, 280 170, 390 90, 280 160, 400 70, 270 160, 450 30, 260 160, 420 30, 250 160, 390 30, 240 160, 370 30, 230 160, 360 30, 230 150, 330 50, 240 130, 330 30, 230 130, 310 30, 220 130, 280 30, 230 100, 270 40, 220 110, 250 30, 210 130, 240 30, 210 100, 220 40, 200 90, 200 20, 190 100, 180 30, 20 20, 180 40, 20 30, 180 50, 20 50, 180 60, 30 60, 180 70, 20 70, 170 80, 80 80, 170 90, 20 80, 180 100, 40 100, 200 110, 60 110, 200 120, 120 120, 190 140))"},
    {"validate/TestRelateAA.xml [55] A/A-6-4: spiky polygons overlapping; boundary <-> boundary intersecting at 1 dimension at a few locations [dim(2){A.A.Int = B.A.Int}]", "POLYGON( (70 150, 20 160, 110 160, 20 180, 100 200, 20 200, 190 210, 20 210, 160 220, 20 220, 150 230, 60 240, 180 250, 20 260, 170 260, 60 270, 160 270, 100 310, 170 280, 200 260, 180 230, 210 260, 130 330, 230 250, 210 290, 240 250, 230 210, 260 300, 250 230, 270 300, 270 240, 300 340, 280 250, 320 330, 290 250, 340 350, 290 240, 350 360, 270 190, 350 340, 290 200, 350 330, 300 190, 360 320, 310 190, 360 300, 320 200, 360 280, 330 200, 360 260, 340 200, 370 260, 340 180, 390 290, 340 170, 400 260, 350 170, 400 250, 350 160, 410 240, 350 150, 400 170, 350 140, 310 170, 340 140, 270 180, 330 140, 260 170, 310 140, 240 170, 290 140, 200 190, 270 140, 180 190, 260 140, 170 190, 260 130, 170 180, 250 130, 170 170, 240 120, 170 160, 210 120, 170 150, 210 110, 340 130, 230 110, 420 140, 220 100, 410 130, 220 90, 400 120, 220 80, 390 110, 220 70, 420 110, 240 70, 420 100, 260 70, 420 90, 280 70, 430 80, 230 60, 430 60, 270 50, 450 40, 210 50, 370 40, 260 40, 460 30, 160 40, 210 60, 200 110, 190 60, 190 120, 170 50, 180 130, 150 30, 170 130, 140 20, 160 120, 130 20, 160 150, 120 20, 160 170, 110 20, 160 190, 100 20, 150 190, 90 20, 140 180, 80 20, 120 140, 70 20, 120 150, 60 20, 110 150, 50 20, 100 140, 50 30, 90 130, 40 30, 80 120, 30 30, 80 130, 30 40, 80 140, 20 40, 70 140, 40 90, 60 130, 20 90, 60 140, 20 130, 70 150))", "POLYGON( (190 140, 140 130, 200 160, 130 150, 210 170, 130 170, 210 180, 120 190, 220 200, 120 200, 250 210, 120 210, 250 220, 120 220, 250 230, 120 240, 230 240, 120 250, 240 260, 120 260, 240 270, 120 270, 270 290, 120 290, 230 300, 150 310, 250 310, 180 320, 250 320, 200 360, 260 330, 240 360, 280 320, 290 370, 290 320, 320 360, 310 320, 360 360, 310 310, 380 340, 310 290, 390 330, 310 280, 410 310, 310 270, 420 280, 310 260, 430 250, 300 250, 440 240, 300 240, 450 230, 280 220, 440 220, 280 210, 440 210, 300 200, 430 190, 300 190, 440 180, 330 180, 430 150, 320 180, 420 130, 300 180, 410 120, 280 180, 400 110, 280 170, 390 90, 280 160, 400 70, 270 160, 450 30, 260 160, 420 30, 250 160, 390 30, 240 160, 370 30, 230 160, 360 30, 230 150, 330 50, 240 130, 330 30, 230 130, 310 30, 220 130, 280 30, 230 100, 270 40, 220 110, 250 30, 210 130, 240 30, 210 100, 220 40, 200 90, 200 20, 190 100, 180 30, 20 20, 180 40, 20 30, 180 50, 20 50, 180 60, 30 60, 180 70, 20 70, 170 80, 80 80, 170 90, 20 80, 180 100, 40 100, 200 110, 60 110, 200 120, 120 120, 190 140))"},
    {"validate/TestRelateAA.xml [56] A/A-6-5: a polygon overlapping another polygon [dim(2){A.A.Int = B.A.Int}, dim(0){A.A.Bdy.CP = B.A.Bdy.CP}, dim(0){A.A.Bdy.V = B.A.Bdy.V}]", "POLYGON( (60 160, 220 160, 220 20, 60 20, 60 160))", "POLYGON( (60 160, 20 200, 260 200, 220 160, 140 80, 60 160))"},
    {"validate/TestRelateAA.xml [57] A/A-6-6: a polygon overlapping another polygon [dim(2){A.A.Int = B.A.Int}, dim(0){A.A.Bdy.CP = B.A.Bdy.CP}, dim(0){A.A.Bdy.V = B.A.Bdy.NV}]", "POLYGON( (60 160, 220 160, 220 20, 60 20, 60 160))", "POLYGON( (60 160, 20 200, 260 200, 140 80, 60 160))"},
    {"validate/TestRelateAA.xml [58] A/A-6-7: a polygon overlapping another polygon [dim(2){A.A.Int = B.A.Int}, dim(0){A.A.Bdy.CP = B.A.Bdy.NV}, dim(0){A.A.Bdy.V = B.A.Bdy.NV}]", "POLYGON( (60 160, 220 160, 220 20, 60 20, 60 160))", "POLYGON( (20 200, 140 80, 260 200, 20 200))"},
    {"validate/TestRelateAA.xml [59] A/A-6-8: a polygon overlapping another polygon [dim(2){A.A.Int = B.A.Int}, dim(0){A.A.Bdy.CP = B.A.Bdy.V}, dim(0){A.A.Bdy.V = B.A.Bdy.V}]", "POLYGON( (60 160, 220 160, 220 20, 60 20, 60 160))", "POLYGON( (20 200, 60 160, 140 80, 220 160, 260 200, 20 200))"},
    {"validate/TestRelateAA.xml [60] A/A-6-9: a polygon overlapping another polygon [dim(2){A.A.Int = B.A.Int}, dim(0){A.A.Bdy.CP = B.A.Bdy.V}, dim(0){A.A.Bdy.V = B.A.Bdy.NV}]", "POLYGON( (60 160, 220 160, 220 20, 60 20, 60 160))", "POLYGON( (20 200, 60 160, 140 80, 260 200, 20 200))"},
    {"validate/TestRelateAA.xml [61] A/A-6-10: a polygon overlapping a skinny polygon [dim(2){A.A.Int = B.A.Int}, dim(0){A.A.Bdy.NV = B.A.Bdy.NV}]", "POLYGON( (0 0, 0 200, 200 200, 200 0, 0 0))", "POLYGON( (100 100, 1000000 110, 10000000 100, 100 100))"},
    {"validate/TestRelateAA.xml [62] A/A-6-11: a polygon overlapping a skinny polygon [dim(2){A.A.Int = B.A.Int}, dim(0){A.A.Bdy.NV = B.A.Bdy.CP}, dim(0){A.A.Bdy.NV = B.A.Bdy.NV}]", "POLYGON( (100 0, 100 200, 200 200, 200 0, 100 0))", "POLYGON( (100 100, 1000000 110, 10000000 100, 100 100))"},
    {"validate/TestRelateAA.xml [63] A/A-6-12: a polygon overlapping a skinny polygon [dim(2){A.A.Int = B.A.Int}, dim(0){A.A.Bdy.NV = B.A.Bdy.NV}]", "POLYGON( (120 0, 120 200, 200 200, 200 0, 120 0))", "POLYGON( (100 100, 1000000 110, 10000000 100, 100 100))"},
    {"validate/TestRelateAA.xml [64] A/A-6-13: a polygon overlapping a skinny polygon [dim(2){A.A.Int = B.A.Int}, dim(0){A.A.Bdy.NV = B.A.Bdy.NV}]", "POLYGON( (0 0, 0 200, 110 200, 110 0, 0 0))", "POLYGON( (100 100, 1000000 110, 10000000 100, 100 100))"},
    {"validate/TestRelateAA.xml [65] A/A-6-14: a polygon overlapping a skinny polygon [dim(2){A.A.Int = B.A.Int}, dim(1){A.A.Bdy.V-EP = B.A.Bdy.NV-EP}, dim(0){A.A.Bdy.CP = B.A.Bdy.CP}, dim(0){A.A.Bdy.NV = B.A.Bdy.NV}]", "POLYGON( (100 100, 100 200, 200 200, 200 100, 100 100))", "POLYGON( (100 100, 2100 110, 2100 100, 100 100))"},
    {"validate/TestRelateAA.xml [66] A/A-6-15: a polygon overlapping a skinny polygon [dim(2){A.A.Int = B.A.Int}, dim(1){A.A.Bdy.V-EP = B.A.Bdy.NV-EP}, dim(0){A.A.Bdy.CP = B.A.Bdy.CP}, dim(0){A.A.Bdy.NV = B.A.Bdy.NV}]", "POLYGON( (100 100, 100 200, 200 200, 200 100, 100 100))", "POLYGON( (100 100, 2101 110, 2101 100, 100 100))"},
    {"validate/TestRelateAA.xml [67] A/A-6-16: two skinny polygons overlapping [dim(2){A.A.Int = B.A.Int}, dim(1){A.A.Bdy.V-EP = B.A.Bdy.NV-EP}, dim(0){A.A.Bdy.CP = B.A.Bdy.CP}, dim(0){A.A.Bdy.NV = B.A.Bdy.NV}]", "POLYGON( (100 100, 200 200, 200 100, 100 100))", "POLYGON( (100 100, 2101 110, 2101 100, 100 100))"},
    {"validate/TestRelateAA.xml [68] A/A-6-17: a polygon overlapping a skinny polygon [dim(2){A.A.Int = B.A.Int}, dim(1){A.A.Bdy.V-EP = B.A.Bdy.NV-EP}, dim(0){A.A.Bdy.CP = B.A.Bdy.CP}, dim(0){A.A.Bdy.NV = B.A.Bdy.NV}]", "POLYGON( (100 100, 100 200, 200 200, 200 100, 100 100))", "POLYGON( (100 100, 1000000 110, 1000000 100, 100 100))"},
    {"validate/TestRelateAA.xml [69] A/A-6-19: a polygon overlapping a skinny polygon [dim(2){A.A.Int = B.A.Int}, dim(1){A.A.Bdy.V-EP = B.A.Bdy.NV-NV}, dim(0){A.A.Bdy.NV = B.A.Bdy.NV}]", "POLYGON( (120 100, 120 200, 200 200, 200 100, 120 100))", "POLYGON( (100 100, 500 110, 500 100, 100 100))"},
    {"validate/TestRelateAA.xml [70] A/A-6-20: a polygon overlapping a skinny polygon [dim(2){A.A.Int = B.A.Int}, dim(1){A.A.Bdy.V-EP = B.A.Bdy.NV-NV}, dim(0){A.A.Bdy.NV = B.A.Bdy.NV}]", "POLYGON( (120 100, 120 200, 200 200, 200 100, 120 100))", "POLYGON( (100 100, 501 110, 501 100, 100 100))"},
    {"validate/TestRelateAA.xml [71] A/A-6-21: a polygon overlapping a skinny polygon [dim(2){A.A.Int = B.A.Int}, dim(1){A.A.Bdy.V-EP = B.A.Bdy.NV-NV}, dim(0){A.A.Bdy.NV = B.A.Bdy.NV}]", "POLYGON( (120 100, 130 200, 200 200, 200 100, 120 100))", "POLYGON( (100 100, 501 110, 501 100, 100 100))"},
    {"validate/TestRelateAA.xml [72] A/A-6-22: a polygon overlapping a skinny polygon [dim(2){A.A.Int = B.A.Int}, dim(1){A.A.Bdy.V-EP = B.A.Bdy.NV-NV}, dim(0){A.A.Bdy.NV = B.A.Bdy.NV}]", "POLYGON( (120 100, 17 200, 200 200, 200 100, 120 100))", "POLYGON( (100 100, 501 110, 501 100, 100 100))"},
    {"validate/TestRelateAA.xml [73] A/A-6-23: a polygon overlapping a skinny polygon [dim(2){A.A.Int = B.A.Int}, dim(1){A.A.Bdy.V-EP = B.A.Bdy.NV-NV}, dim(0){A.A.Bdy.NV = B.A.Bdy.NV}]", "POLYGON( (120 100, 120 200, 200 200, 200 100, 120 100))", "POLYGON( (100 100, 1000000 110, 1000000 100, 100 100))"},
    {"validate/TestRelateAA.xml [74] A/A-6-25: two skinny polygons overlapping [dim(2){A.A.Int = B.A.Int}, dim(0){A.A.Bdy.NV = B.A.Bdy.NV}]", "POLYGON( (101 99, 101 1000000, 102 1000000, 101 99))", "POLYGON( (100 100, 1000000 110, 1000000 100, 100 100))"},
    {"validate/TestRelateAA.xml [75] A/A-6-26: two skinny polygons overlapping [dim(2){A.A.Int = B.A.Int}, dim(1){A.A.Bdy.V-EP = B.A.Bdy.NV-EP}, dim(0){A.A.Bdy.CP = B.A.Bdy.CP}, dim(0){A.A.Bdy.NV = B.A.Bdy.NV}]", "POLYGON( (100 100, 200 101, 200 100, 100 100))", "POLYGON( (100 100, 2101 110, 2101 100, 100 100))"},
    {"validate/TestRelateAA.xml [76] A/A-6-26: two polygons overlapping [dim(2){A.A.Int = B.A.Int}, dim(0){A.A.Bdy.NV = B.A.Bdy.NV}]", "POLYGON( (16 319, 150 39, 25 302, 160 20, 265 20, 127 317, 16 319))", "POLYGON( (10 307, 22 307, 153 34, 22 34, 10 307))"},
    {"validate/TestRelateAA.xml [77] A/Ah-3-1: the closing point of a polygon touching the closing points of another polygon and its hole [dim(0){A.A.Bdy.CP = B.A.oBdy.CP}, dim(0){A.A.Bdy.CP = B.A.iBdy.CP}]", "POLYGON( (160 200, 210 70, 120 70, 160 200))", "POLYGON( (160 200, 310 20, 20 20, 160 200), (160 200, 260 40, 70 40, 160 200))"},
    {"validate/TestRelateAA.xml [78] A/Ah-3-2: the boundary of a polygon touching the inner boundary of another polygon at two spots [dim(2){A.A.Int = B.A.Ext.h}, dim(0){A.A.oBdy.SP = B.A.iBdy.SP}, dim(0){A.A.oBdy.V = B.A.iBdy.V}]", "POLYGON( (170 120, 240 100, 260 50, 190 70, 170 120))", "POLYGON( (150 150, 410 150, 280 20, 20 20, 150 150), (170 120, 330 120, 260 50, 100 50, 170 120))"},
    {"validate/TestRelateAA.xml [79] A/Ah-3-3: the boundary of a polygon touching the inner boundary of another polygon at two spots [dim(2){A.A.Int = B.A.Ext.h}, dim(0){A.A.oBdy.SP = B.A.iBdy.SP}, dim(0){A.A.oBdy.V = B.A.iBdy.V}]", "POLYGON( (270 90, 200 50, 150 80, 210 120, 270 90))", "POLYGON( (150 150, 410 150, 280 20, 20 20, 150 150), (170 120, 330 120, 260 50, 100 50, 170 120))"},
    {"validate/TestRelateAA.xml [80] A/Ah-3-4: the boundary of a polygon touching the inner boundary of another polygon at one spot [dim(2){A.A.Int = B.A.Ext.h}, dim(0){A.A.oBdy.SP = B.A.iBdy.SP}]", "POLYGON( (170 120, 260 100, 240 60, 150 80, 170 120))", "POLYGON( (150 150, 410 150, 280 20, 20 20, 150 150), (170 120, 330 120, 260 50, 100 50, 170 120))"},
    {"validate/TestRelateAA.xml [81] A/Ah-3-5: the boundary of a polygon touching the inner boundary of another polygon at one spot [dim(2){A.A.Int = B.A.Ext.h}, dim(0){A.A.oBdy.SP = B.A.iBdy.NV}]", "POLYGON( (220 120, 270 80, 200 60, 160 100, 220 120))", "POLYGON( (150 150, 410 150, 280 20, 20 20, 150 150), (170 120, 330 120, 260 50, 100 50, 170 120))"},
    {"validate/TestRelateAA.xml [82] A/Ah-3-6: the boundary of a polygon touching the inner boundary of another polygon at one spot [dim(2){A.A.Int = B.A.Ext.h}, dim(0){A.A.oBdy.SP = B.A.iBdy.V}]", "POLYGON( (260 50, 180 70, 180 110, 260 90, 260 50))", "POLYGON( (150 150, 410 150, 280 20, 20 20, 150 150), (170 120, 330 120, 260 50, 100 50, 170 120))"},
    {"validate/TestRelateAA.xml [83] A/Ah-3-7: the boundary of a polygon touching the inner boundary of another polygon at two spots [dim(2){A.A.Int = B.A.Ext.h}, dim(0){A.A.oBdy.V = B.A.iBdy.NV}, dim(0){A.A.oBdy.V = B.A.iBdy.NV}]", "POLYGON( (230 110, 290 80, 190 60, 140 90, 230 110))", "POLYGON( (150 150, 410 150, 280 20, 20 20, 150 150), (170 120, 330 120, 260 50, 100 50, 170 120))"},
    {"validate/TestRelateAA.xml [84] A/Ah-3-8: the boundary of a polygon touching the inner boundary of another polygon [dim(2){A.A.Int = B.A.Ext.h}, dim(1){A.A.oBdy.SP-EP = B.A.iBdy.SP-EP}]", "POLYGON( (170 120, 330 120, 260 50, 100 50, 170 120))", "POLYGON( (150 150, 410 150, 280 20, 20 20, 150 150), (170 120, 330 120, 260 50, 100 50, 170 120))"},
    {"validate/TestRelateAA.xml [85] A/Ah-3-9: part of the boundary of a polygon touching part of the inner boundary of another polygon [dim(2){A.A.Int = B.A.Ext.h}, dim(1){A.A.oBdy.SP-V = B.A.iBdy.SP-NV}, dim(1){A.A.oBdy.V-EP = B.A.iBdy.NV-EP}]", "POLYGON( (170 120, 330 120, 280 70, 120 70, 170 120))", "POLYGON( (150 150, 410 150, 280 20, 20 20, 150 150), (170 120, 330 120, 260 50, 100 50, 170 120))"},
    {"validate/TestRelateAA.xml [86] A/Ah-3-10: part of the boundary of a polygon touching part of the inner boundary of another polygon [dim(2){A.A.Int = B.A.Ext.h}, dim(1){A.A.oBdy.SP-V = B.A.iBdy.SP-NV}, dim(1){A.A.oBdy.V-EP = B.A.iBdy.NV-EP}]", "POLYGON( (170 120, 300 120, 250 70, 120 70, 170 120))", "POLYGON( (150 150, 410 150, 280 20, 20 20, 150 150), (170 120, 330 120, 260 50, 100 50, 170 120))"},
    {"validate/TestRelateAA.xml [87] A/Ah-3-11: part of the boundary of a polygon touching part of the inner boundary of another polygon [dim(2){A.A.Int = B.A.Ext.h}, dim(1){A.A.oBdy.V-V-V = B.A.iBdy.NV-V-NV}]", "POLYGON( (190 100, 310 100, 260 50, 140 50, 190 100))", "POLYGON( (150 150, 410 150, 280 20, 20 20, 150 150), (170 120, 330 120, 260 50, 100 50, 170 120))"},
    {"validate/TestRelateAA.xml [88] A/Ah-5-1: an entire polygon within another polygon which has a hole [dim(2){A.A.Ext = B.A.Int}, dim(2){A.A.Int = B.A.Int}]", "POLYGON( (280 130, 360 130, 270 40, 190 40, 280 130))", "POLYGON( (150 150, 410 150, 280 20, 20 20, 150 150), (170 120, 250 120, 180 50, 100 50, 170 120))"},
    {"validate/TestRelateAA.xml [89] A/Ah-5-2: an entire polygon within another polygon which has a hole [dim(2){A.A.Int = B.A.Int}, dim(2){A.A.Ext = B.A.Int}]", "POLYGON( (220 80, 180 40, 80 40, 170 130, 270 130, 230 90, 300 90, 250 30, 280 30, 390 140, 150 140, 40 30, 230 30, 280 80, 220 80))", "POLYGON( (150 150, 410 150, 280 20, 20 20, 150 150), (170 120, 250 120, 180 50, 100 50, 170 120))"},
    {"validate/TestRelateAA.xml [90] A/Ah-5-3: polygon A within polygon B, the boundary of A touching the inner boundary of B [dim(2){A.A.Int = B.A.Int}, dim(2){A.A.Ext = B.A.Int}, dim(1){A.A.Bdy.NV-NV = B.A.iBdy.V-V}]", "POLYGON( (260 130, 360 130, 280 40, 170 40, 260 130))", "POLYGON( (150 150, 410 150, 280 20, 20 20, 150 150), (170 120, 250 120, 180 50, 100 50, 170 120))"},
    {"validate/TestRelateAA.xml [91] A/Ah-5-4: polygon A within polygon B, the boundary of A touching the inner boundary of B [dim(2){A.A.Int = B.A.Int}, dim(2){A.A.Ext = B.A.Int}, dim(1){A.A.Bdy.V-V = B.A.iBdy.NV-NV}]", "POLYGON( (240 110, 340 110, 290 60, 190 60, 240 110))", "POLYGON( (150 150, 410 150, 280 20, 20 20, 150 150), (170 120, 250 120, 180 50, 100 50, 170 120))"},
    {"validate/TestRelateAA.xml [92] A/Ah-5-5: polygon A within polygon B, the boundary of A touching the inner boundary of B [dim(2){A.A.Int = B.A.Int}, dim(2){A.A.Ext = B.A.Int}, dim(1){A.A.Bdy.V-V = B.A.iBdy.V-V}]", "POLYGON( (250 120, 350 120, 280 50, 180 50, 250 120))", "POLYGON( (150 150, 410 150, 280 20, 20 20, 150 150), (170 120, 250 120, 180 50, 100 50, 170 120))"},
    {"validate/TestRelateAA.xml [93] Ah/Ah-1-1: same polygons (with a hole) [dim(2){A.A.Int = B.A.Int}, dim(1){A.A.oBdy.SP-EP = B.A.oBdy.SP-EP}, dim(1){A.A.iBdy.SP-EP = B.A.iBdy.SP-EP}]", "POLYGON( (230 210, 230 20, 20 20, 20 210, 230 210), (120 180, 50 50, 200 50, 120 180))", "POLYGON( (230 210, 230 20, 20 20, 20 210, 230 210), (120 180, 50 50, 200 50, 120 180))"},
    {"validate/TestRelateAA.xml [94] A2h/A2h-1-1: same polygons (with two holes) [dim(2){A.A.Int = B.A.Int}, dim(1){A.A.oBdy.SP-EP = B.A.oBdy.SP-EP}, dim(1){A.A.iBdy.SP-EP = B.A.iBdy.SP-EP}]", "POLYGON( (230 210, 230 20, 20 20, 20 210, 230 210), (140 40, 40 40, 40 170, 140 40), (110 190, 210 190, 210 50, 110 190))", "POLYGON( (230 210, 230 20, 20 20, 20 210, 230 210), (140 40, 40 40, 40 170, 140 40), (110 190, 210 190, 210 50, 110 190))"},
    {"validate/TestRelateAA.xml [95] A/mA-3-1: a polygon touching multipolygon at two points [dim(2){A.A.Int = B.2A.Ext}, dim(0){A.A.oBdy.CP = B.2A2.oBdy.NV}, dim(0){A.A.oBdy.V = B.2A1.oBdy.NV}]", "POLYGON( (280 190, 330 150, 200 110, 150 150, 280 190))", "MULTIPOLYGON( ( (140 110, 260 110, 170 20, 50 20, 140 110)), ( (300 270, 420 270, 340 190, 220 190, 300 270)))"},
    {"validate/TestRelateAA.xml [96] A/mA-3-2: a polygon touching multipolygon at two points [dim(2){A.A.Int = B.2A.Ext}, dim(0){A.A.oBdy.V = B.2A1.oBdy.CP}, dim(0){A.A.oBdy.V = B.2A2.oBdy.V}]", "POLYGON( (80 190, 220 190, 140 110, 0 110, 80 190))", "MULTIPOLYGON( ( (140 110, 260 110, 170 20, 50 20, 140 110)), ( (300 270, 420 270, 340 190, 220 190, 300 270)))"},
    {"validate/TestRelateAA.xml [97] A/mA-3-3: a polygon touching multipolygon at two points [dim(2){A.A.Int = B.2A.Ext}, dim(0){A.A.oBdy.V = B.2A2.oBdy.NV}, dim(0){A.A.oBdy.V = B.2A1.oBdy.NV}]", "POLYGON( (330 150, 200 110, 150 150, 280 190, 330 150))", "MULTIPOLYGON( ( (140 110, 260 110, 170 20, 50 20, 140 110)), ( (300 270, 420 270, 340 190, 220 190, 300 270)))"},
    {"validate/TestRelateAA.xml [98] A/mA-3-4: a polygon touching multipolygon at one spoint [dim(2){A.A.Int = B.2A.Ext}, dim(0){A.A.oBdy.V = B.2A2.oBdy.NV}]", "POLYGON( (290 190, 340 150, 220 120, 170 170, 290 190))", "MULTIPOLYGON( ( (140 110, 260 110, 170 20, 50 20, 140 110)), ( (300 270, 420 270, 340 190, 220 190, 300 270)))"},
    {"validate/TestRelateAA.xml [99] A/mA-3-5: a polygon touching multipolygon along boundaries [dim(2){A.A.Int = B.2A.Ext}, dim(1){A.A.oBdy.SP-V = B.2A2.oBdy.V-V}, dim(1){A.A.oBdy.V-V = B.2A1.oBdy.V-SP}]", "POLYGON( (220 190, 340 190, 260 110, 140 110, 220 190))", "MULTIPOLYGON( ( (140 110, 260 110, 170 20, 50 20, 140 110)), ( (300 270, 420 270, 340 190, 220 190, 300 270)))"},
    {"validate/TestRelateAA.xml [100] A/mA-3-6: a polygon touching multipolygon along boundaries and at a point [dim(2){A.A.Int = B.2A.Ext}, dim(1){A.A.oBdy.V-NV = B.2A1.oBdy.NV-SP}, dim(0){A.A.oBdy.V = B.2A2.oBdy.V}]", "POLYGON( (140 190, 220 190, 100 70, 20 70, 140 190))", "MULTIPOLYGON( ( (140 110, 260 110, 170 20, 50 20, 140 110)), ( (300 270, 420 270, 340 190, 220 190, 300 270)))"},
    {"validate/TestRelateAA.xml [101] A/mA-6-1: a polygon overlapping multipolygon [dim(2){A.A.Int = B.4A.Int}, dim(0){A.A.Bdy.NV = B.A.Bdy.V}, dim(0){A.A.Bdy.NV = B.A.Bdy.CP}, dim(0){A.A.Bdy.NV = B.A.Bdy.V}, dim(0){A.A.Bdy.NV = B.A.Bdy.CP}]", "POLYGON( (140 220, 60 140, 140 60, 220 140, 140 220))", "MULTIPOLYGON( ( (100 20, 180 20, 180 100, 100 100, 100 20)), ( (20 100, 100 100, 100 180, 20 180, 20 100)), ( (100 180, 180 180, 180 260, 100 260, 100 180)), ( (180 100, 260 100, 260 180, 180 180, 180 100)))"},
    {"validate/TestRelateAA.xml [102] mA/mA-3-1: MultiPolygon touching MultiPolygon [dim(0){A.mA.Bdy.TP = B.mA.Bdy.TP}]", "MULTIPOLYGON( ( (110 110, 70 200, 150 200, 110 110)), ( (110 110, 150 20, 70 20, 110 110)))", "MULTIPOLYGON( ( (110 110, 160 160, 210 110, 160 60, 110 110)), ( (110 110, 60 60, 10 110, 60 160, 110 110)))"},
    {"validate/TestRelateAA.xml [103] mAh/mAh-3-1: MultiPolygon touching MultiPolygon [dim(0){A.mA.Bdy.TP = B.mA.Bdy.TP}]", "MULTIPOLYGON( ( (110 110, 70 200, 150 200, 110 110), (110 110, 100 180, 120 180, 110 110)), ( (110 110, 150 20, 70 20, 110 110), (110 110, 120 40, 100 40, 110 110)))", "MULTIPOLYGON( ( (110 110, 160 160, 210 110, 160 60, 110 110), (110 110, 160 130, 160 90, 110 110)), ( (110 110, 60 60, 10 110, 60 160, 110 110), (110 110, 60 90, 60 130, 110 110)))"},
    {"validate/TestRelateAA.xml [104] mAh/mAh-3-2: MultiPolygon touching MultiPolygon [dim(1){A.mA.Bdy.NV-EP = B.mA.Bdy.V-SP}, dim(1){A.mA.Bdy.SP-NV = B.mA.Bdy.EP-V}]", "MULTIPOLYGON( ( (110 110, 70 200, 200 200, 110 110), (110 110, 100 180, 120 180, 110 110)), ( (110 110, 200 20, 70 20, 110 110), (110 110, 120 40, 100 40, 110 110)))", "MULTIPOLYGON( ( (110 110, 160 160, 210 110, 160 60, 110 110), (110 110, 160 130, 160 90, 110 110)), ( (110 110, 60 60, 10 110, 60 160, 110 110), (110 110, 60 90, 60 130, 110 110)))"},
    {"validate/TestRelateAA.xml [105] mAh/mAh-3-3: MultiPolygon touching MultiPolygon [dim(1){A.mA.Bdy.SP-NV = B.mA.Bdy.EP-V}, dim(1){A.mA.Bdy.NV-EP = B.mA.Bdy.V-SP}, dim(1){A.mA.Bdy.NV-EP = B.mA.Bdy.V-SP}, dim(1){A.mA.Bdy.SP-NV = B.mA.Bdy.EP-V}]", "MULTIPOLYGON( ( (110 110, 20 200, 200 200, 110 110), (110 110, 100 180, 120 180, 110 110)), ( (110 110, 200 20, 20 20, 110 110), (110 110, 120 40, 100 40, 110 110)))", "MULTIPOLYGON( ( (110 110, 160 160, 210 110, 160 60, 110 110), (110 110, 160 130, 160 90, 110 110)), ( (110 110, 60 60, 10 110, 60 160, 110 110), (110 110, 60 90, 60 130, 110 110)))"},
    {"validate/TestRelateAA.xml [106] mAh/mAh-6-1: MultiPolygon touching MultiPolygon [dim(2){A.mA.Int = B.mA.Int}]", "MULTIPOLYGON( ( (110 110, 70 200, 200 200, 110 110), (110 110, 100 180, 120 180, 110 110)), ( (110 110, 200 20, 70 20, 110 110), (110 110, 120 40, 100 40, 110 110)))", "MULTIPOLYGON( ( (110 110, 160 160, 210 110, 160 60, 110 110), (110 110, 160 130, 160 90, 110 110)), ( (110 110, 60 60, 10 110, 60 160, 110 110), (110 110, 60 90, 60 130, 110 110)))"},
    {"validate/TestRelateAA.xml [107] mAh/mAh-6-2: MultiPolygon touching MultiPolygon [dim(2){A.mA.Int = B.mA.Int}]", "MULTIPOLYGON( ( (110 110, 70 200, 200 200, 110 110), (110 110, 100 180, 120 180, 110 110)), ( (110 110, 200 20, 70 20, 110 110), (110 110, 120 40, 100 40, 110 110)))", "MULTIPOLYGON( ( (110 110, 70 200, 210 110, 70 20, 110 110), (110 110, 110 140, 150 110, 110 80, 110 110)), ( (110 110, 60 60, 10 110, 60 160, 110 110), (110 110, 60 90, 60 130, 110 110)))"},
    {"validate/TestRelateAC.xml [1] AC A-shells overlapping B-shell at A-vertex", "POLYGON( (100 60, 140 100, 100 140, 60 100, 100 60))", "MULTIPOLYGON( ( (80 40, 120 40, 120 80, 80 80, 80 40)), ( (120 80, 160 80, 160 120, 120 120, 120 80)), ( (80 120, 120 120, 120 160, 80 160, 80 120)), ( (40 80, 80 80, 80 120, 40 120, 40 80)))"},
  };

  auto cuda_stream = rmm::cuda_stream_default;
  auto rt_engine = GetSharedRTEngine();
  geos::io::WKTReader reader;

  const std::vector<Predicate> predicates = {
      Predicate::kEquals,   Predicate::kDisjoint, Predicate::kTouches,
      Predicate::kContains, Predicate::kCovers,   Predicate::kIntersects,
      Predicate::kWithin,   Predicate::kCoveredBy};

  size_t total_cases = cases.size();
  size_t exact_im_matches = 0;
  size_t pred_match_geos_cases = 0;

  for (size_t idx = 0; idx < total_cases; ++idx) {
    const auto& tc = cases[idx];
    auto geos_a = reader.read(tc.wkt_a);
    auto geos_b = reader.read(tc.wkt_b);
    bool a_valid = geos_a->isValid();
    bool b_valid = geos_b->isValid();

    geos::operation::relateng::RelateMatrixPredicate im_pred;
    geos::operation::relateng::RelateNG::relate(geos_a.get(), geos_b.get(), im_pred);
    std::string geos_im_str = im_pred.getIM()->toString();

    auto dev_geom1 = LoadDeviceGeometriesFromWKT({tc.wkt_a});
    auto dev_geom2 = LoadDeviceGeometriesFromWKT({tc.wkt_b});
    RelateEngine<point_t, index_t> engine(&dev_geom1, rt_engine.get());

    // 1. Compute DE-9IM matrix via RT
    rmm::device_uvector<index_t> ids1(1, cuda_stream);
    rmm::device_uvector<index_t> ids2(1, cuda_stream);
    ids1.set_element_to_zero_async(0, cuda_stream);
    ids2.set_element_to_zero_async(0, cuda_stream);
    rmm::device_uvector<int> ims(1, cuda_stream);
    ims.set_element_to_zero_async(0, cuda_stream);

    PolygonArrayView<point_t, index_t> empty_poly;
    MultiPolygonArrayView<point_t, index_t> empty_mp;
    bool g1_is_poly = dev_geom1.get_geometry_type() == GeometryType::kPolygon;
    bool g2_is_poly = dev_geom2.get_geometry_type() == GeometryType::kPolygon;

    if (g1_is_poly && g2_is_poly) {
      auto v1 = dev_geom1.GetGeometryArrayView<PolygonArrayView<point_t, index_t>>();
      auto v2 = dev_geom2.GetGeometryArrayView<PolygonArrayView<point_t, index_t>>();
      engine.EvaluateImpl(cuda_stream, v1, empty_mp, v2, ids1, ids2, ims, false);
      engine.EvaluateImpl(cuda_stream, v2, empty_mp, v1, ids2, ids1, ims, true);
    } else if (g1_is_poly && !g2_is_poly) {
      auto v1 = dev_geom1.GetGeometryArrayView<PolygonArrayView<point_t, index_t>>();
      auto v2 = dev_geom2.GetGeometryArrayView<MultiPolygonArrayView<point_t, index_t>>();
      engine.EvaluateImpl(cuda_stream, v1, empty_mp, v2, ids1, ids2, ims, false);
      engine.EvaluateImpl(cuda_stream, empty_poly, v2, v1, ids2, ids1, ims, true);
    } else if (!g1_is_poly && g2_is_poly) {
      auto v1 = dev_geom1.GetGeometryArrayView<MultiPolygonArrayView<point_t, index_t>>();
      auto v2 = dev_geom2.GetGeometryArrayView<PolygonArrayView<point_t, index_t>>();
      engine.EvaluateImpl(cuda_stream, empty_poly, v1, v2, ids1, ids2, ims, false);
      engine.EvaluateImpl(cuda_stream, v2, empty_mp, v1, ids2, ids1, ims, true);
    } else {
      auto v1 = dev_geom1.GetGeometryArrayView<MultiPolygonArrayView<point_t, index_t>>();
      auto v2 = dev_geom2.GetGeometryArrayView<MultiPolygonArrayView<point_t, index_t>>();
      engine.EvaluateImpl(cuda_stream, empty_poly, v1, v2, ids1, ids2, ims, false);
      engine.EvaluateImpl(cuda_stream, empty_poly, v2, v1, ids2, ids1, ims, true);
    }

    int rt_im = ims.element(0, cuda_stream);
    char rt_im_str[10];
    IntersectionMatrix::ToString(rt_im, rt_im_str);

    if (geos_im_str == rt_im_str) {
      exact_im_matches++;
    } else {
      std::cout << "[IM DIFF] Case #" << idx << " (" << tc.name << ")"
                << " [valid=" << (a_valid && b_valid) << "]: GEOS_IM=" << geos_im_str
                << ", RT_IM=" << rt_im_str << "\n  A = " << tc.wkt_a
                << "\n  B = " << tc.wkt_b << "\n";
    }

    // 2. Evaluate all 8 predicates
    bool all_preds_match_geos = true;

    for (Predicate p : predicates) {
      rmm::device_uvector<index_t> p_ids1(1, cuda_stream);
      rmm::device_uvector<index_t> p_ids2(1, cuda_stream);
      p_ids1.set_element_to_zero_async(0, cuda_stream);
      p_ids2.set_element_to_zero_async(0, cuda_stream);
      engine.Evaluate(cuda_stream, dev_geom2, p, p_ids1, p_ids2);
      cuda_stream.synchronize();

      bool rt_res = !p_ids1.is_empty();
      bool geos_res = TestUtils::EvaluateGeosPredicate(p, geos_a.get(), geos_b.get());

      EXPECT_EQ(rt_res, geos_res)
          << "Case #" << idx << " (" << tc.name << ") " << PredicateToString(p);
      if (rt_res != geos_res) {
        all_preds_match_geos = false;
      }
    }

    if (all_preds_match_geos) pred_match_geos_cases++;
  }

  std::cout << "\n====================================================================\n"
            << "GEOS Polygon-Polygon Test Suite Summary (" << total_cases << " cases)\n"
            << "  Exact 9-cell DE-9IM matches with GEOS: " << exact_im_matches << " / "
            << total_cases << "\n"
            << "  All 8 predicates match GEOS:           " << pred_match_geos_cases
            << " / " << total_cases << "\n"
            << "====================================================================\n";
  EXPECT_EQ(pred_match_geos_cases, total_cases);
}

TEST(RelatePolygonTest, BatchLoopRangeBoundaryWithSmallMemoryQuota) {
  // Reproduce the scenario from PR #1505 review where rounding up batch_size
  // consumes all candidate pairs before the final iteration (n_batches > actual_batches).
  std::vector<std::string> polys1 = {
      "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
      "POLYGON ((1 1, 2 1, 2 2, 1 2, 1 1))",
      "POLYGON ((2 2, 3 2, 3 3, 2 3, 2 2))",
      "POLYGON ((3 3, 4 3, 4 4, 3 4, 3 3))",
      "POLYGON ((4 4, 5 4, 5 5, 4 5, 4 4))",
  };
  std::vector<std::string> polys2 = {
      "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))",
      "POLYGON ((1 1, 2 1, 2 2, 1 2, 1 1))",
      "POLYGON ((2 2, 3 2, 3 3, 2 3, 2 2))",
      "POLYGON ((3 3, 4 3, 4 4, 3 4, 3 3))",
      "POLYGON ((4 4, 5 4, 5 5, 4 5, 4 4))",
  };

  auto cuda_stream = rmm::cuda_stream_default;
  auto dev_geom1 = LoadDeviceGeometriesFromWKT(polys1);
  auto dev_geom2 = LoadDeviceGeometriesFromWKT(polys2);
  auto rt_engine = GetSharedRTEngine();

  RelateEngine<point_t, index_t> engine(&dev_geom1, rt_engine.get());
  RelateEngine<point_t, index_t>::Config re_config;
  // Extremely small memory quota to force n_batches > 1 with a small number of candidate pairs
  re_config.memory_quota = 1e-9f;
  engine.set_config(re_config);

  const size_t num_pairs = 5;
  std::vector<index_t> h_ids = {0, 1, 2, 3, 4};
  rmm::device_uvector<index_t> ids1(num_pairs, cuda_stream);
  rmm::device_uvector<index_t> ids2(num_pairs, cuda_stream);
  cudaMemcpyAsync(ids1.data(), h_ids.data(), num_pairs * sizeof(index_t),
                  cudaMemcpyHostToDevice, cuda_stream.value());
  cudaMemcpyAsync(ids2.data(), h_ids.data(), num_pairs * sizeof(index_t),
                  cudaMemcpyHostToDevice, cuda_stream.value());

  // Without the loop range break check (if ids_begin >= ids_size),
  // this would underflow on the trailing batch and attempt to allocate SIZE_MAX.
  EXPECT_NO_THROW({
    engine.Evaluate(cuda_stream, dev_geom2, Predicate::kIntersects, ids1, ids2);
    cuda_stream.synchronize();
  });
  EXPECT_EQ(ids1.size(), num_pairs);
}

TEST(RelatePolygonTest, ZeroMemoryQuotaDoesNotDivideByZero) {
  std::vector<std::string> polys1 = {"POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"};
  std::vector<std::string> polys2 = {"POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"};

  auto cuda_stream = rmm::cuda_stream_default;
  auto dev_geom1 = LoadDeviceGeometriesFromWKT(polys1);
  auto dev_geom2 = LoadDeviceGeometriesFromWKT(polys2);
  auto rt_engine = GetSharedRTEngine();

  RelateEngine<point_t, index_t> engine(&dev_geom1, rt_engine.get());
  RelateEngine<point_t, index_t>::Config re_config;
  // Zero memory quota produces avail_bytes == 0 without std::max clamp
  re_config.memory_quota = 0.0f;
  engine.set_config(re_config);

  rmm::device_uvector<index_t> ids1(1, cuda_stream);
  rmm::device_uvector<index_t> ids2(1, cuda_stream);
  ids1.set_element_to_zero_async(0, cuda_stream);
  ids2.set_element_to_zero_async(0, cuda_stream);

  EXPECT_NO_THROW({
    engine.Evaluate(cuda_stream, dev_geom2, Predicate::kIntersects, ids1, ids2);
    cuda_stream.synchronize();
  });
  EXPECT_EQ(ids1.size(), 1);
}

}  // namespace gpuspatial
