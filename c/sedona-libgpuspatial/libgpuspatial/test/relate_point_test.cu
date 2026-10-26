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

#include <gtest/gtest.h>

namespace gpuspatial {

//======= P/P =============

TEST(RelatePointTest, PointPointDisjoint) {
  point_t p1, p2;
  std::string wkt1 = "POINT (0 0)";
  std::string wkt2 = "POINT (1 1)";
  ParseWKTPoint(wkt1.c_str(), p1);
  ParseWKTPoint(wkt2.c_str(), p2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), p1, p2);
}

TEST(RelatePointTest, MultiPointMultiPointContained) {
  MultiPoint<point_t> m1, m2;
  std::string wkt1 = "MULTIPOINT (0 0, 1 1, 2 2)";
  std::string wkt2 = "MULTIPOINT (1 1, 2 2)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTMultiPoint(ctx1, wkt1.c_str(), m1);
  ParseWKTMultiPoint(ctx2, wkt2.c_str(), m2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), m1, m2);
}

TEST(RelatePointTest, MultiPointMultiPointEqual) {
  MultiPoint<point_t> m1, m2;
  std::string wkt1 = "MULTIPOINT (0 0, 1 1, 2 2)";
  std::string wkt2 = "MULTIPOINT (0 0, 1 1, 2 2)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTMultiPoint(ctx1, wkt1.c_str(), m1);
  ParseWKTMultiPoint(ctx2, wkt2.c_str(), m2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), m1, m2);
}

TEST(RelatePointTest, MultiPointMultiPointValidateRelatePP_13) {
  MultiPoint<point_t> m1, m2;
  std::string wkt1 = "MULTIPOINT ((80 70), (140 120), (20 20), (200 170))";
  std::string wkt2 = "MULTIPOINT ((80 70), (140 120), (80 170), (200 80))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTMultiPoint(ctx1, wkt1.c_str(), m1);
  ParseWKTMultiPoint(ctx2, wkt2.c_str(), m2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), m1, m2);
}

//======= Empty Point / MultiPoint =============

TEST(RelatePointTest, PointEmpty_vs_Point) {
  point_t g1;
  point_t g2;
  ParseWKTPoint(E_POINT.c_str(), g1);
  ParseWKTPoint(NE_POINT.c_str(), g2);
  TestRelate(E_POINT.c_str(), NE_POINT.c_str(), g1, g2);
}

TEST(RelatePointTest, PointEmpty_vs_LineString) {
  point_t g1;
  LineString<point_t> g2;
  Context<point_t, index_t> ctx;
  ParseWKTPoint(E_POINT.c_str(), g1);
  ParseWKTLineString(ctx, NE_LINE.c_str(), g2);
  TestRelate(E_POINT.c_str(), NE_LINE.c_str(), g1, g2);
}

TEST(RelatePointTest, PointEmpty_vs_Polygon) {
  point_t g1;
  Polygon<point_t, index_t> g2;
  Context<point_t, index_t> ctx2;
  ParseWKTPoint(E_POINT.c_str(), g1);
  ParseWKTPolygon(ctx2, NE_POLY.c_str(), g2);
  TestRelate(E_POINT.c_str(), NE_POLY.c_str(), g1, g2);
}

TEST(RelatePointTest, MultiPointEmpty_vs_Point) {
  MultiPoint<point_t> g1;
  point_t g2;
  Context<point_t, index_t> ctx1;
  ParseWKTMultiPoint(ctx1, E_MPOINT.c_str(), g1);
  ParseWKTPoint(NE_POINT.c_str(), g2);
  TestRelate(E_MPOINT.c_str(), NE_POINT.c_str(), g1, g2);
}

TEST(RelatePointTest, MultiPointEmpty_vs_LineString) {
  MultiPoint<point_t> g1;
  LineString<point_t> g2;
  Context<point_t, index_t> ctx1, ctx2;
  ParseWKTMultiPoint(ctx1, E_MPOINT.c_str(), g1);
  ParseWKTLineString(ctx2, NE_LINE.c_str(), g2);
  TestRelate(E_MPOINT.c_str(), NE_LINE.c_str(), g1, g2);
}

TEST(RelatePointTest, MultiPointEmpty_vs_Polygon) {
  MultiPoint<point_t> g1;
  Polygon<point_t, index_t> g2;
  Context<point_t, index_t> ctx1, ctx2;
  ParseWKTMultiPoint(ctx1, E_MPOINT.c_str(), g1);
  ParseWKTPolygon(ctx2, NE_POLY.c_str(), g2);
  TestRelate(E_MPOINT.c_str(), NE_POLY.c_str(), g1, g2);
}

TEST(RelatePointTest, Point_vs_PointEmpty) {
  point_t g1;
  point_t g2;
  ParseWKTPoint(NE_POINT.c_str(), g1);
  ParseWKTPoint(E_POINT.c_str(), g2);
  TestRelate(NE_POINT.c_str(), E_POINT.c_str(), g1, g2);
}

TEST(RelatePointTest, Point_vs_LineStringEmpty) {
  point_t g1;
  LineString<point_t> g2;
  Context<point_t, index_t> ctx2;
  ParseWKTPoint(NE_POINT.c_str(), g1);
  ParseWKTLineString(ctx2, E_LINE.c_str(), g2);
  TestRelate(NE_POINT.c_str(), E_LINE.c_str(), g1, g2);
}

TEST(RelatePointTest, Point_vs_PolygonEmpty) {
  point_t g1;
  Polygon<point_t, index_t> g2;
  Context<point_t, index_t> ctx2;
  ParseWKTPoint(NE_POINT.c_str(), g1);
  ParseWKTPolygon(ctx2, E_POLY.c_str(), g2);
  TestRelate(NE_POINT.c_str(), E_POLY.c_str(), g1, g2);
}

}  // namespace gpuspatial
