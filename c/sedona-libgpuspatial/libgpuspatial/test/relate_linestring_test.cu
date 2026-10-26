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

//======= L/P =============

TEST(RelateLineStringTest, LineStringMultiPointContains) {
  LineString<point_t> ls1;
  MultiPoint<point_t> m2;
  std::string wkt1 = "LINESTRING (0 0, 1 1, 2 2)";
  std::string wkt2 = "MULTIPOINT (0 0, 1 1, 2 2)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTMultiPoint(ctx2, wkt2.c_str(), m2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, m2);
}

TEST(RelateLineStringTest, LineStringMultiPointOverlaps) {
  LineString<point_t> ls1;
  MultiPoint<point_t> m2;
  std::string wkt1 = "LINESTRING (0 0, 1 1)";
  std::string wkt2 = "MULTIPOINT (0 0, 1 1, 2 2)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTMultiPoint(ctx2, wkt2.c_str(), m2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, m2);
}

TEST(RelateLineStringTest, ZeroLengthLinePoint) {
  LineString<point_t> ls1;
  point_t p2;
  std::string wkt1 = "LINESTRING (0 0, 0 0)";
  std::string wkt2 = "POINT (0 0)";
  Context<point_t, index_t> ctx1;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTPoint(wkt2.c_str(), p2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, p2);
}

TEST(RelateLineStringTest, NonZeroLengthLinePoint) {
  LineString<point_t> ls1;
  point_t p2;
  std::string wkt1 = "LINESTRING (0 0, 0 0, 9 9)";
  std::string wkt2 = "POINT (1 1)";
  Context<point_t, index_t> ctx1;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTPoint(wkt2.c_str(), p2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, p2);
}

TEST(RelateLineStringTest, LinePointIntAndExt) {
  MultiPoint<point_t> m1;
  LineString<point_t> ls2;
  std::string wkt1 = "MULTIPOINT ((60 60), (100 100))";
  std::string wkt2 = "LINESTRING (40 40, 80 80)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTMultiPoint(ctx1, wkt1.c_str(), m1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), m1, ls2);
}

//======= L/L =============

TEST(RelateLineStringTest, ZeroLengthLineLine) {
  LineString<point_t> ls1;
  LineString<point_t> ls2;
  std::string wkt1 = "LINESTRING (10 10, 10 10, 10 10)";
  std::string wkt2 = "LINESTRING (10 10, 10 10)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, ls2);
}

TEST(RelateLineStringTest, LinesCrossProper) {
  LineString<point_t> ls1;
  LineString<point_t> ls2;
  std::string wkt1 = "LINESTRING (0 0, 9 9)";
  std::string wkt2 = "LINESTRING (0 9, 9 0)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, ls2);
}

TEST(RelateLineStringTest, LinesOverlap) {
  LineString<point_t> ls1;
  LineString<point_t> ls2;
  std::string wkt1 = "LINESTRING (0 0, 5 5)";
  std::string wkt2 = "LINESTRING (3 3, 9 9)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, ls2);
}

TEST(RelateLineStringTest, LinesCrossVertex) {
  LineString<point_t> ls1;
  LineString<point_t> ls2;
  std::string wkt1 = "LINESTRING (0 0, 8 8)";
  std::string wkt2 = "LINESTRING (0 8, 4 4, 8 0)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, ls2);
}

TEST(RelateLineStringTest, LinesTouchVertex) {
  LineString<point_t> ls1;
  LineString<point_t> ls2;
  std::string wkt1 = "LINESTRING (0 0, 8 0)";
  std::string wkt2 = "LINESTRING (0 8, 4 0, 8 8)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, ls2);
}

TEST(RelateLineStringTest, LinesDisjointByEnvelope) {
  LineString<point_t> ls1;
  LineString<point_t> ls2;
  std::string wkt1 = "LINESTRING (0 0, 9 9)";
  std::string wkt2 = "LINESTRING (10 19, 19 10)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, ls2);
}

TEST(RelateLineStringTest, LinesDisjoint) {
  LineString<point_t> ls1;
  LineString<point_t> ls2;
  std::string wkt1 = "LINESTRING (0 0, 9 9)";
  std::string wkt2 = "LINESTRING (4 2, 8 6)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, ls2);
}

TEST(RelateLineStringTest, LinesRingTouchAtNode) {
  LineString<point_t> ls1;
  LineString<point_t> ls2;
  std::string wkt1 = "LINESTRING (5 5, 1 8, 1 1, 5 5)";
  std::string wkt2 = "LINESTRING (5 5, 9 5)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, ls2);
}

TEST(RelateLineStringTest, LinesTouchAtBdy) {
  LineString<point_t> ls1;
  LineString<point_t> ls2;
  std::string wkt1 = "LINESTRING (5 5, 1 8)";
  std::string wkt2 = "LINESTRING (5 5, 9 5)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, ls2);
}

TEST(RelateLineStringTest, LinesOverlapWithDisjointLine) {
  LineString<point_t> ls1;
  MultiLineString<point_t, index_t> m_ls2;
  std::string wkt1 = "LINESTRING (1 1, 9 9)";
  std::string wkt2 = "MULTILINESTRING ((2 2, 8 8), (6 2, 8 4))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTMultiLineString(ctx2, wkt2.c_str(), m_ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, m_ls2);
}

TEST(RelateLineStringTest, LinesDisjointOverlappingEnvelopes) {
  LineString<point_t> ls1;
  LineString<point_t> ls2;
  std::string wkt1 = "LINESTRING (60 0, 20 80, 100 80, 80 120, 40 140)";
  std::string wkt2 = "LINESTRING (60 40, 140 40, 140 160, 0 160)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, ls2);
}

TEST(RelateLineStringTest, LinesContained_JTS396) {
  LineString<point_t> ls1;
  LineString<point_t> ls2;
  std::string wkt1 = "LINESTRING (1 0, 0 2, 0 0, 2 2)";
  std::string wkt2 = "LINESTRING (0 0, 2 2)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, ls2);
}

TEST(RelateLineStringTest, LinesContainedWithSelfIntersection) {
  LineString<point_t> ls1;
  LineString<point_t> ls2;
  std::string wkt1 = "LINESTRING (2 0, 0 2, 0 0, 2 2)";
  std::string wkt2 = "LINESTRING (0 0, 2 2)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, ls2);
}

TEST(RelateLineStringTest, LineContainedInRing) {
  LineString<point_t> ls1;
  LineString<point_t> ls2;
  std::string wkt1 = "LINESTRING(60 60, 100 100, 140 60)";
  std::string wkt2 = "LINESTRING(100 100, 180 20, 20 20, 100 100)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, ls2);
}

TEST(RelateLineStringTest, LineLineProperIntersection) {
  MultiLineString<point_t, index_t> m1;
  LineString<point_t> ls2;
  std::string wkt1 = "MULTILINESTRING ((0 0, 1 1), (0.5 0.5, 1 0.1, -1 0.1))";
  std::string wkt2 = "LINESTRING (0 0, 1 1)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTMultiLineString(ctx1, wkt1.c_str(), m1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), m1, ls2);
}

TEST(RelateLineStringTest, LineSelfIntersectionCollinear) {
  LineString<point_t> ls1;
  LineString<point_t> ls2;
  std::string wkt1 = "LINESTRING (9 6, 1 6, 1 0, 5 6, 9 6)";
  std::string wkt2 = "LINESTRING (9 9, 3 1)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, ls2);
}

TEST(RelateLineStringTest, RepeatedPointLL) {
  LineString<point_t> l1;
  LineString<point_t> l2;
  std::string wkt1 = "LINESTRING(0 0, 5 5, 5 5, 5 5, 9 9)";
  std::string wkt2 = "LINESTRING(0 9, 5 5, 5 5, 5 5, 9 0)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), l1);
  ParseWKTLineString(ctx2, wkt2.c_str(), l2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), l1, l2);
}

//======= Empty LineString / MultiLineString =============

TEST(RelateLineStringTest, LineStringEmpty_vs_Point) {
  LineString<point_t> g1;
  point_t g2;
  Context<point_t, index_t> ctx1;
  ParseWKTLineString(ctx1, E_LINE.c_str(), g1);
  ParseWKTPoint(NE_POINT.c_str(), g2);
  TestRelate(E_LINE.c_str(), NE_POINT.c_str(), g1, g2);
}

TEST(RelateLineStringTest, LineStringEmpty_vs_LineString) {
  LineString<point_t> g1, g2;
  Context<point_t, index_t> ctx1, ctx2;
  ParseWKTLineString(ctx1, E_LINE.c_str(), g1);
  ParseWKTLineString(ctx2, NE_LINE.c_str(), g2);
  TestRelate(E_LINE.c_str(), NE_LINE.c_str(), g1, g2);
}

TEST(RelateLineStringTest, LineStringEmpty_vs_Polygon) {
  LineString<point_t> g1;
  Polygon<point_t, index_t> g2;
  Context<point_t, index_t> ctx1, ctx2;
  ParseWKTLineString(ctx1, E_LINE.c_str(), g1);
  ParseWKTPolygon(ctx2, NE_POLY.c_str(), g2);
  TestRelate(E_LINE.c_str(), NE_POLY.c_str(), g1, g2);
}

TEST(RelateLineStringTest, MultiLineStringEmpty_vs_Point) {
  MultiLineString<point_t, index_t> g1;
  point_t g2;
  Context<point_t, index_t> ctx1;
  ParseWKTMultiLineString(ctx1, E_MLINE.c_str(), g1);
  ParseWKTPoint(NE_POINT.c_str(), g2);
  TestRelate(E_MLINE.c_str(), NE_POINT.c_str(), g1, g2);
}

TEST(RelateLineStringTest, MultiLineStringEmpty_vs_LineString) {
  MultiLineString<point_t, index_t> g1;
  LineString<point_t> g2;
  Context<point_t, index_t> ctx1, ctx2;
  ParseWKTMultiLineString(ctx1, E_MLINE.c_str(), g1);
  ParseWKTLineString(ctx2, NE_LINE.c_str(), g2);
  TestRelate(E_MLINE.c_str(), NE_LINE.c_str(), g1, g2);
}

TEST(RelateLineStringTest, MultiLineStringEmpty_vs_Polygon) {
  MultiLineString<point_t, index_t> g1;
  Polygon<point_t, index_t> g2;
  Context<point_t, index_t> ctx1, ctx2;
  ParseWKTMultiLineString(ctx1, E_MLINE.c_str(), g1);
  ParseWKTPolygon(ctx2, NE_POLY.c_str(), g2);
  TestRelate(E_MLINE.c_str(), NE_POLY.c_str(), g1, g2);
}

TEST(RelateLineStringTest, LineString_vs_PointEmpty) {
  LineString<point_t> g1;
  point_t g2;
  Context<point_t, index_t> ctx1;
  ParseWKTLineString(ctx1, NE_LINE.c_str(), g1);
  ParseWKTPoint(E_POINT.c_str(), g2);
  TestRelate(NE_LINE.c_str(), E_POINT.c_str(), g1, g2);
}

TEST(RelateLineStringTest, LineString_vs_LineStringEmpty) {
  LineString<point_t> g1, g2;
  Context<point_t, index_t> ctx1, ctx2;
  ParseWKTLineString(ctx1, NE_LINE.c_str(), g1);
  ParseWKTLineString(ctx2, E_LINE.c_str(), g2);
  TestRelate(NE_LINE.c_str(), E_LINE.c_str(), g1, g2);
}

TEST(RelateLineStringTest, LineString_vs_PolygonEmpty) {
  LineString<point_t> g1;
  Polygon<point_t, index_t> g2;
  Context<point_t, index_t> ctx1, ctx2;
  ParseWKTLineString(ctx1, NE_LINE.c_str(), g1);
  ParseWKTPolygon(ctx2, E_POLY.c_str(), g2);
  TestRelate(NE_LINE.c_str(), E_POLY.c_str(), g1, g2);
}

}  // namespace gpuspatial
