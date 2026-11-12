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
}  // namespace gpuspatial
