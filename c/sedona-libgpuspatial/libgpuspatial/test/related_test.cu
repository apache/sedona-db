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
#include "gpuspatial/loader/parallel_wkb_loader.h"
#include "gpuspatial/relate/relate.cuh"
#include "gpuspatial/utils/pinned_vector.h"

#include "test_common.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <geos/geom/Geometry.h>
#include <geos/io/WKTReader.h>
#include <geos/operation/relateng/RelateGeometry.h>
#include <geos/operation/relateng/RelateMatrixPredicate.h>
#include <geos/operation/relateng/RelateNG.h>
#include <geos/operation/relateng/RelatePredicate.h>
#include <gtest/gtest.h>

using namespace geos::geom;
using namespace geos::operation::relateng;
using geos::io::WKTReader;

// Test cases are from
// https://github.com/libgeos/geos/blob/2d2802d7f7acd7919599b94f3d1530e8cd987aee/tests/unit/operation/relateng/RelateNGTest.cpp

namespace gpuspatial {
using point_t = Point<double, 2>;
using index_t = uint32_t;
using box_t = Box<Point<float, 2>>;
using loader_t = ParallelWkbLoader<point_t, index_t>;

template <typename POINT_T, typename INDEX_T>
struct Context {
  PinnedVector<POINT_T> points;
  PinnedVector<INDEX_T> prefix_sum1;
  PinnedVector<INDEX_T> prefix_sum2;
  PinnedVector<INDEX_T> prefix_sum3;
  PinnedVector<box_t> mbrs;
};

template <typename POINT_T>
void ParseWKTPoint(const char* wkt, POINT_T& point) {
  nanoarrow::UniqueArrayStream stream;
  ArrayStreamFromWKT({{wkt}}, GEOARROW_TYPE_WKB, stream.get());
  nanoarrow::UniqueArray array;
  ArrowError error;
  ArrowErrorSet(&error, "");

  ASSERT_EQ(ArrowArrayStreamGetNext(stream.get(), array.get(), &error), NANOARROW_OK);
  loader_t loader;
  auto cuda_stream = rmm::cuda_stream_default;

  loader.Init();
  loader.Parse(cuda_stream, array.get(), 0, array->length);
  auto device_geometries = loader.Finish(cuda_stream);
  auto h_vec = TestUtils::ToVector(cuda_stream, device_geometries.get_points());
  cuda_stream.synchronize();
  point = h_vec[0];
}

template <typename POINT_T, typename INDEX_T>
void ParseWKTMultiPoint(Context<POINT_T, INDEX_T>& ctx, const char* wkt,
                        MultiPoint<POINT_T>& multi_point) {
  nanoarrow::UniqueArrayStream stream;
  ArrayStreamFromWKT({{wkt}}, GEOARROW_TYPE_WKB, stream.get());
  nanoarrow::UniqueArray array;
  ArrowError error;
  ArrowErrorSet(&error, "");

  ASSERT_EQ(ArrowArrayStreamGetNext(stream.get(), array.get(), &error), NANOARROW_OK);
  loader_t loader;
  auto cuda_stream = rmm::cuda_stream_default;

  loader.Init();
  loader.Parse(cuda_stream, array.get(), 0, array->length);
  auto device_geometries = loader.Finish(cuda_stream);

  ctx.prefix_sum1 = TestUtils::ToVector(
      cuda_stream, device_geometries.get_offsets().multi_point_offsets.ps_num_points);
  ctx.points = TestUtils::ToVector(cuda_stream, device_geometries.get_points());
  ctx.mbrs = TestUtils::ToVector(cuda_stream, device_geometries.get_mbrs());
  cuda_stream.synchronize();
  MultiPointArrayView multi_array_view(
      ArrayView<INDEX_T>(ctx.prefix_sum1.data(), ctx.prefix_sum1.size()),
      ArrayView<POINT_T>(ctx.points.data(), ctx.points.size()),
      ArrayView<box_t>(ctx.mbrs.data(), ctx.mbrs.size()));
  multi_point = multi_array_view[0];
}

template <typename POINT_T, typename INDEX_T>
void ParseWKTLineString(Context<POINT_T, INDEX_T>& ctx, const char* wkt,
                        LineString<POINT_T>& ls) {
  nanoarrow::UniqueArrayStream stream;
  ArrayStreamFromWKT({{wkt}}, GEOARROW_TYPE_WKB, stream.get());
  nanoarrow::UniqueArray array;
  ArrowError error;
  ArrowErrorSet(&error, "");

  ASSERT_EQ(ArrowArrayStreamGetNext(stream.get(), array.get(), &error), NANOARROW_OK);
  loader_t loader;
  auto cuda_stream = rmm::cuda_stream_default;

  loader.Init();
  loader.Parse(cuda_stream, array.get(), 0, array->length);
  auto device_geometries = loader.Finish(cuda_stream);
  ctx.prefix_sum1 = TestUtils::ToVector(
      cuda_stream, device_geometries.get_offsets().line_string_offsets.ps_num_points);
  ctx.points = TestUtils::ToVector(cuda_stream, device_geometries.get_points());
  ctx.mbrs = TestUtils::ToVector(cuda_stream, device_geometries.get_mbrs());
  cuda_stream.synchronize();
  LineStringArrayView<POINT_T, INDEX_T> ls_array_view(
      ArrayView<INDEX_T>(ctx.prefix_sum1.data(), ctx.prefix_sum1.size()),
      ArrayView<POINT_T>(ctx.points.data(), ctx.points.size()),
      ArrayView<box_t>(ctx.mbrs.data(), ctx.mbrs.size()));
  ls = ls_array_view[0];
}

template <typename POINT_T, typename INDEX_T>
void ParseWKTMultiLineString(Context<POINT_T, INDEX_T>& ctx, const char* wkt,
                             MultiLineString<POINT_T, INDEX_T>& m_ls) {
  nanoarrow::UniqueArrayStream stream;
  ArrayStreamFromWKT({{wkt}}, GEOARROW_TYPE_WKB, stream.get());
  nanoarrow::UniqueArray array;
  ArrowError error;
  ArrowErrorSet(&error, "");

  ASSERT_EQ(ArrowArrayStreamGetNext(stream.get(), array.get(), &error), NANOARROW_OK);
  loader_t loader;
  auto cuda_stream = rmm::cuda_stream_default;

  loader.Init();
  loader.Parse(cuda_stream, array.get(), 0, array->length);
  auto device_geometries = loader.Finish(cuda_stream);
  ctx.prefix_sum1 = TestUtils::ToVector(
      cuda_stream,
      device_geometries.get_offsets().multi_line_string_offsets.ps_num_parts);
  ctx.prefix_sum2 = TestUtils::ToVector(
      cuda_stream,
      device_geometries.get_offsets().multi_line_string_offsets.ps_num_points);
  ctx.points = TestUtils::ToVector(cuda_stream, device_geometries.get_points());
  ctx.mbrs = TestUtils::ToVector(cuda_stream, device_geometries.get_mbrs());
  cuda_stream.synchronize();
  MultiLineStringArrayView<POINT_T, INDEX_T> m_ls_array_view(
      ArrayView<INDEX_T>(ctx.prefix_sum1.data(), ctx.prefix_sum1.size()),
      ArrayView<INDEX_T>(ctx.prefix_sum2.data(), ctx.prefix_sum2.size()),
      ArrayView<POINT_T>(ctx.points.data(), ctx.points.size()),
      ArrayView<box_t>(ctx.mbrs.data(), ctx.mbrs.size()));
  m_ls = m_ls_array_view[0];
}

template <typename POINT_T, typename INDEX_T>
void ParseWKTPolygon(Context<POINT_T, INDEX_T>& ctx, const char* wkt,
                     Polygon<POINT_T, INDEX_T>& poly) {
  nanoarrow::UniqueArrayStream stream;
  ArrayStreamFromWKT({{wkt}}, GEOARROW_TYPE_WKB, stream.get());
  nanoarrow::UniqueArray array;
  ArrowError error;
  ArrowErrorSet(&error, "");

  ASSERT_EQ(ArrowArrayStreamGetNext(stream.get(), array.get(), &error), NANOARROW_OK);
  loader_t loader;
  auto cuda_stream = rmm::cuda_stream_default;

  loader.Init();
  loader.Parse(cuda_stream, array.get(), 0, array->length);
  auto device_geometries = loader.Finish(cuda_stream);
  ctx.prefix_sum1 = TestUtils::ToVector(
      cuda_stream, device_geometries.get_offsets().polygon_offsets.ps_num_rings);
  ctx.prefix_sum2 = TestUtils::ToVector(
      cuda_stream, device_geometries.get_offsets().polygon_offsets.ps_num_points);
  ctx.points = TestUtils::ToVector(cuda_stream, device_geometries.get_points());
  ctx.mbrs = TestUtils::ToVector(cuda_stream, device_geometries.get_mbrs());
  cuda_stream.synchronize();
  PolygonArrayView<POINT_T, INDEX_T> poly_array_view(
      ArrayView<INDEX_T>(ctx.prefix_sum1.data(), ctx.prefix_sum1.size()),
      ArrayView<INDEX_T>(ctx.prefix_sum2.data(), ctx.prefix_sum2.size()),
      ArrayView<POINT_T>(ctx.points.data(), ctx.points.size()),
      ArrayView<box_t>(ctx.mbrs.data(), ctx.mbrs.size()));
  poly = poly_array_view[0];
}

template <typename POINT_T, typename INDEX_T>
void ParseWKTMultiPolygon(Context<POINT_T, INDEX_T>& ctx, const char* wkt,
                          MultiPolygon<POINT_T, INDEX_T>& poly) {
  nanoarrow::UniqueArrayStream stream;
  ArrayStreamFromWKT({{wkt}}, GEOARROW_TYPE_WKB, stream.get());
  nanoarrow::UniqueArray array;
  ArrowError error;
  ArrowErrorSet(&error, "");

  ASSERT_EQ(ArrowArrayStreamGetNext(stream.get(), array.get(), &error), NANOARROW_OK);
  loader_t loader;
  auto cuda_stream = rmm::cuda_stream_default;

  loader.Init();
  loader.Parse(cuda_stream, array.get(), 0, array->length);
  auto device_geometries = loader.Finish(cuda_stream);
  ctx.prefix_sum1 = TestUtils::ToVector(
      cuda_stream, device_geometries.get_offsets().multi_polygon_offsets.ps_num_parts);
  ctx.prefix_sum2 = TestUtils::ToVector(
      cuda_stream, device_geometries.get_offsets().multi_polygon_offsets.ps_num_rings);
  ctx.prefix_sum3 = TestUtils::ToVector(
      cuda_stream, device_geometries.get_offsets().multi_polygon_offsets.ps_num_points);
  ctx.points = TestUtils::ToVector(cuda_stream, device_geometries.get_points());
  ctx.mbrs = TestUtils::ToVector(cuda_stream, device_geometries.get_mbrs());
  cuda_stream.synchronize();
  MultiPolygonArrayView<POINT_T, INDEX_T> poly_array_view(
      ArrayView<INDEX_T>(ctx.prefix_sum1.data(), ctx.prefix_sum1.size()),
      ArrayView<INDEX_T>(ctx.prefix_sum2.data(), ctx.prefix_sum2.size()),
      ArrayView<INDEX_T>(ctx.prefix_sum3.data(), ctx.prefix_sum3.size()),
      ArrayView<POINT_T>(ctx.points.data(), ctx.points.size()),
      ArrayView<box_t>(ctx.mbrs.data(), ctx.mbrs.size()));
  poly = poly_array_view[0];
}

template <typename GEOMETRY1_T, typename GEOMETRY2_T>
void TestRelate(const char* wkt1, const char* wkt2, const GEOMETRY1_T& g1,
                const GEOMETRY2_T& g2) {
  WKTReader r;
  auto a = r.read(wkt1);
  auto b = r.read(wkt2);

  RelateMatrixPredicate pred;
  RelateNG::relate(a.get(), b.get(), pred);
  std::string actualVal = pred.getIM()->toString();

  int val = relate(g1, g2);
  char res[10];
  IM__ToString(val, res);
  ASSERT_STREQ(actualVal.c_str(), res);
}

TEST(RelateTest, PointPointDisjoint) {
  point_t p1, p2;

  std::string wkt1 = "POINT (0 0)";
  std::string wkt2 = "POINT (1 1)";
  ParseWKTPoint(wkt1.c_str(), p1);
  ParseWKTPoint(wkt2.c_str(), p2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), p1, p2);
}

TEST(RelateTest, MultiPointMultiPointContained) {
  MultiPoint<point_t> m1, m2;
  std::string wkt1 = "MULTIPOINT (0 0, 1 1, 2 2)";
  std::string wkt2 = "MULTIPOINT (1 1, 2 2)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTMultiPoint(ctx1, wkt1.c_str(), m1);
  ParseWKTMultiPoint(ctx2, wkt2.c_str(), m2);

  TestRelate(wkt1.c_str(), wkt2.c_str(), m1, m2);
}

TEST(RelateTest, MultiPointMultiPointEqual) {
  using point_t = Point<double, 2>;
  MultiPoint<point_t> m1, m2;
  std::string wkt1 = "MULTIPOINT (0 0, 1 1, 2 2)";
  std::string wkt2 = "MULTIPOINT (0 0, 1 1, 2 2)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTMultiPoint(ctx1, wkt1.c_str(), m1);
  ParseWKTMultiPoint(ctx2, wkt2.c_str(), m2);

  TestRelate(wkt1.c_str(), wkt2.c_str(), m1, m2);
}

TEST(RelateTest, MultiPointMultiPointValidateRelatePP_13) {
  MultiPoint<point_t> m1, m2;
  std::string wkt1 = "MULTIPOINT ((80 70), (140 120), (20 20), (200 170))";
  std::string wkt2 = "MULTIPOINT ((80 70), (140 120), (80 170), (200 80))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTMultiPoint(ctx1, wkt1.c_str(), m1);
  ParseWKTMultiPoint(ctx2, wkt2.c_str(), m2);

  TestRelate(wkt1.c_str(), wkt2.c_str(), m1, m2);
}

TEST(RelateTest, LineStringMultiPointContains) {
  LineString<point_t> ls1;
  MultiPoint<point_t> m2;
  std::string wkt1 = "LINESTRING (0 0, 1 1, 2 2)";
  std::string wkt2 = "MULTIPOINT (0 0, 1 1, 2 2)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTMultiPoint(ctx2, wkt2.c_str(), m2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, m2);
}

TEST(RelateTest, LineStringMultiPointOverlaps) {
  LineString<point_t> ls1;
  MultiPoint<point_t> m2;
  std::string wkt1 = "LINESTRING (0 0, 1 1)";
  std::string wkt2 = "MULTIPOINT (0 0, 1 1, 2 2)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTMultiPoint(ctx2, wkt2.c_str(), m2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, m2);
}

TEST(RelateTest, ZeroLengthLinePoint) {
  LineString<point_t> ls1;
  point_t p2;
  std::string wkt1 = "LINESTRING (0 0, 0 0)";
  std::string wkt2 = "POINT (0 0)";
  Context<point_t, index_t> ctx1;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTPoint(wkt2.c_str(), p2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, p2);
}

TEST(RelateTest, ZeroLengthLineLine) {
  LineString<point_t> ls1;
  LineString<point_t> ls2;
  std::string wkt1 = "LINESTRING (10 10, 10 10, 10 10)";
  std::string wkt2 = "LINESTRING (10 10, 10 10)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, ls2);
}

TEST(RelateTest, NonZeroLengthLinePoint) {
  LineString<point_t> ls1;
  point_t p2;
  std::string wkt1 = "LINESTRING (0 0, 0 0, 9 9)";
  std::string wkt2 = "POINT (1 1)";
  Context<point_t, index_t> ctx1;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTPoint(wkt2.c_str(), p2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, p2);
}

TEST(RelateTest, LinePointIntAndExt) {
  MultiPoint<point_t> m1;
  LineString<point_t> ls2;
  std::string wkt1 = "MULTIPOINT ((60 60), (100 100))";
  std::string wkt2 = "LINESTRING (40 40, 80 80)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTMultiPoint(ctx1, wkt1.c_str(), m1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), m1, ls2);
}

TEST(RelateTest, LinesCrossProper) {
  LineString<point_t> ls1;
  LineString<point_t> ls2;
  std::string wkt1 = "LINESTRING (0 0, 9 9)";
  std::string wkt2 = "LINESTRING (0 9, 9 0)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, ls2);
}

TEST(RelateTest, LinesOverlap) {
  LineString<point_t> ls1;
  LineString<point_t> ls2;
  std::string wkt1 = "LINESTRING (0 0, 5 5)";
  std::string wkt2 = "LINESTRING (3 3, 9 9)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, ls2);
}

TEST(RelateTest, LinesCrossVertex) {
  LineString<point_t> ls1;
  LineString<point_t> ls2;
  std::string wkt1 = "LINESTRING (0 0, 8 8)";
  std::string wkt2 = "LINESTRING (0 8, 4 4, 8 0)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, ls2);
}

TEST(RelateTest, LinesTouchVertex) {
  LineString<point_t> ls1;
  LineString<point_t> ls2;
  std::string wkt1 = "LINESTRING (0 0, 8 0)";
  std::string wkt2 = "LINESTRING (0 8, 4 0, 8 8)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, ls2);
}

TEST(RelateTest, LinesDisjointByEnvelope) {
  LineString<point_t> ls1;
  LineString<point_t> ls2;
  std::string wkt1 = "LINESTRING (0 0, 9 9)";
  std::string wkt2 = "LINESTRING (10 19, 19 10)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, ls2);
}

TEST(RelateTest, LinesDisjoint) {
  LineString<point_t> ls1;
  LineString<point_t> ls2;
  std::string wkt1 = "LINESTRING (0 0, 9 9)";
  std::string wkt2 = "LINESTRING (4 2, 8 6)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, ls2);
}

// FIXME: wkt1 is a closed polyline, which has no boundary according to JTS's
// Mod2BoundaryNodeRule We have to implement a similar rule in gpuspatial to handle this
// case correctly TEST(RelateTest, LinesClosedEmpty) {
//   MultiLineString<point_t, index_t> m_ls1;
//   LineString<point_t> ls2;
//   std::string wkt1 = "MULTILINESTRING ((0 0, 0 1), (0 1, 1 1, 1 0, 0 0))";
//   std::string wkt2 = "LINESTRING EMPTY";
//   Context<point_t, index_t> ctx1, ctx2;
//
//   ParseWKTMultiLineString(ctx1, wkt1.c_str(), m_ls1);
//   ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
//   TestRelate(wkt1.c_str(), wkt2.c_str(), m_ls1, ls2);
// }

TEST(RelateTest, LinesRingTouchAtNode) {
  LineString<point_t> ls1;
  LineString<point_t> ls2;
  std::string wkt1 = "LINESTRING (5 5, 1 8, 1 1, 5 5)";
  std::string wkt2 = "LINESTRING (5 5, 9 5)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, ls2);
}

TEST(RelateTest, LinesTouchAtBdy) {
  LineString<point_t> ls1;
  LineString<point_t> ls2;
  std::string wkt1 = "LINESTRING (5 5, 1 8)";
  std::string wkt2 = "LINESTRING (5 5, 9 5)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, ls2);
}

TEST(RelateTest, LinesOverlapWithDisjointLine) {
  LineString<point_t> ls1;
  MultiLineString<point_t, index_t> m_ls2;
  std::string wkt1 = "LINESTRING (1 1, 9 9)";
  std::string wkt2 = "MULTILINESTRING ((2 2, 8 8), (6 2, 8 4))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTMultiLineString(ctx2, wkt2.c_str(), m_ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, m_ls2);
}

TEST(RelateTest, LinesDisjointOverlappingEnvelopes) {
  LineString<point_t> ls1;
  LineString<point_t> ls2;
  std::string wkt1 = "LINESTRING (60 0, 20 80, 100 80, 80 120, 40 140)";
  std::string wkt2 = "LINESTRING (60 40, 140 40, 140 160, 0 160)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, ls2);
}

/**
 * FIXME:
 * Case from https://github.com/locationtech/jts/issues/270
 * Strictly, the lines cross, since their interiors intersect
 * according to the Orientation predicate.
 * However, the computation of the intersection point is
 * non-robust, and reports it as being equal to the endpoint
 * POINT (-10 0.0000000000000012)
 * For consistency the relate algorithm uses the intersection node topology.
 */
// TEST(RelateTest, LineStringLineString10) {
//   LineString<point_t> ls1;
//   LineString<point_t> ls2;
//   std::string wkt1 = "LINESTRING (0 0, -10 0.0000000000000012)";
//   std::string wkt2 = "LINESTRING (-9.999143275740073 -0.1308959557133398, -10
//   0.0000000000001054)"; Context<point_t, index_t> ctx1, ctx2;
//
//   ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
//   ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
//   TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, ls2);
// }

TEST(RelateTest, LinesContained_JTS396) {
  LineString<point_t> ls1;
  LineString<point_t> ls2;
  std::string wkt1 = "LINESTRING (1 0, 0 2, 0 0, 2 2)";
  std::string wkt2 = "LINESTRING (0 0, 2 2)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, ls2);
}

TEST(RelateTest, LinesContainedWithSelfIntersection) {
  LineString<point_t> ls1;
  LineString<point_t> ls2;
  std::string wkt1 = "LINESTRING (2 0, 0 2, 0 0, 2 2)";
  std::string wkt2 = "LINESTRING (0 0, 2 2)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, ls2);
}

TEST(RelateTest, LineContainedInRing) {
  LineString<point_t> ls1;
  LineString<point_t> ls2;
  std::string wkt1 = "LINESTRING(60 60, 100 100, 140 60)";
  std::string wkt2 = "LINESTRING(100 100, 180 20, 20 20, 100 100)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, ls2);
}

TEST(RelateTest, LineLineProperIntersection) {
  MultiLineString<point_t, index_t> m1;
  LineString<point_t> ls2;
  std::string wkt1 = "MULTILINESTRING ((0 0, 1 1), (0.5 0.5, 1 0.1, -1 0.1))";
  std::string wkt2 = "LINESTRING (0 0, 1 1)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTMultiLineString(ctx1, wkt1.c_str(), m1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), m1, ls2);
}

TEST(RelateTest, LineSelfIntersectionCollinear) {
  LineString<point_t> ls1;
  LineString<point_t> ls2;
  std::string wkt1 = "LINESTRING (9 6, 1 6, 1 0, 5 6, 9 6)";
  std::string wkt2 = "LINESTRING (9 9, 3 1)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, ls2);
}

//======= A/P  =============

TEST(RelateTest, PolygonPointInside) {
  Polygon<point_t, index_t> poly1;
  point_t p2;
  std::string wkt1 = "POLYGON ((0 10, 10 10, 10 0, 0 0, 0 10))";
  std::string wkt2 = "POINT (1 1)";
  Context<point_t, index_t> ctx1;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPoint(wkt2.c_str(), p2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, p2);
}

TEST(RelateTest, PolygonPointOutside) {
  Polygon<point_t, index_t> poly1;
  point_t p2;
  std::string wkt1 = "POLYGON ((10 0, 0 0, 0 10, 10 0))";
  std::string wkt2 = "POINT (8 8)";
  Context<point_t, index_t> ctx1;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPoint(wkt2.c_str(), p2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, p2);
}

TEST(RelateTest, PolygonPointBoundary) {
  Polygon<point_t, index_t> poly1;
  point_t p2;
  std::string wkt1 = "POLYGON ((10 0, 0 0, 0 10, 10 0))";
  std::string wkt2 = "POINT (1 0)";
  Context<point_t, index_t> ctx1;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPoint(wkt2.c_str(), p2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, p2);
}

TEST(RelateTest, PolygonPointExterior) {
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

TEST(RelateTest, PolygonLineStringContainedAtLineVertex) {
  Polygon<point_t, index_t> poly1;
  LineString<point_t> ls2;
  std::string wkt1 = "POLYGON ((1 5, 5 5, 5 1, 1 1, 1 5))";
  std::string wkt2 = "LINESTRING (2 3, 3 5, 4 3)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, ls2);
}

TEST(RelateTest, PolygonLineStringTouchAtLineVertex) {
  Polygon<point_t, index_t> poly1;
  LineString<point_t> ls2;
  std::string wkt1 = "POLYGON ((1 5, 5 5, 5 1, 1 1, 1 5))";
  std::string wkt2 = "LINESTRING (1 8, 3 5, 5 8)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, ls2);
}

TEST(RelateTest, PolygonLineStringInside) {
  Polygon<point_t, index_t> poly1;
  LineString<point_t> ls2;
  std::string wkt1 = "POLYGON ((0 10, 10 10, 10 0, 0 0, 0 10))";
  std::string wkt2 = "LINESTRING (1 8, 3 5, 5 8)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, ls2);
}

TEST(RelateTest, PolygonLineStringOutside) {
  Polygon<point_t, index_t> poly1;
  LineString<point_t> ls2;
  std::string wkt1 = "POLYGON ((10 0, 0 0, 0 10, 10 0))";
  std::string wkt2 = "LINESTRING (4 8, 9 3)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, ls2);
}

TEST(RelateTest, PolygonLineStringInBoundary) {
  Polygon<point_t, index_t> poly1;
  LineString<point_t> ls2;
  std::string wkt1 = "POLYGON ((10 0, 0 0, 0 10, 10 0))";
  std::string wkt2 = "LINESTRING (1 0, 9 0)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTLineString(ctx2, wkt2.c_str(), ls2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, ls2);
}

TEST(RelateTest, MultiPolygonLineStringCrossingContained) {
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

TEST(RelateTest, LineStringPolygonRelateLA_220) {
  LineString<point_t> ls1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "LINESTRING (90 210, 210 90)";
  std::string wkt2 = "POLYGON ((150 150, 410 150, 280 20, 20 20, 150 150))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, poly2);
}

TEST(RelateTest, LineCrossingPolygonAtShellHolePoint) {
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

TEST(RelateTest, LineCrossingPolygonAtNonVertex) {
  LineString<point_t> ls1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "LINESTRING (20 60, 150 60)";
  std::string wkt2 = "POLYGON ((150 150, 410 150, 280 20, 20 20, 150 150))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), ls1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), ls1, poly2);
}

TEST(RelateTest, PolygonLinesContainedCollinearEdge) {
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

TEST(RelateTest, PolygonsEdgeAdjacent) {
  Polygon<point_t, index_t> poly1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "POLYGON ((1 3, 3 3, 3 1, 1 1, 1 3))";
  std::string wkt2 = "POLYGON ((5 3, 5 1, 3 1, 3 3, 5 3))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, poly2);
}

TEST(RelateTest, PolygonsEdgeAdjacent2) {
  Polygon<point_t, index_t> poly1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "POLYGON ((1 3, 4 3, 3 0, 1 1, 1 3))";
  std::string wkt2 = "POLYGON ((5 3, 5 1, 3 0, 4 3, 5 3))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, poly2);
}

TEST(RelateTest, PolygonsNested) {
  Polygon<point_t, index_t> poly1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "POLYGON ((1 9, 9 9, 9 1, 1 1, 1 9))";
  std::string wkt2 = "POLYGON ((2 8, 8 8, 8 2, 2 2, 2 8))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, poly2);
}

TEST(RelateTest, PolygonsOverlapProper) {
  Polygon<point_t, index_t> poly1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "POLYGON ((1 1, 1 7, 7 7, 7 1, 1 1))";
  std::string wkt2 = "POLYGON ((2 8, 8 8, 8 2, 2 2, 2 8))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, poly2);
}

TEST(RelateTest, PolygonsOverlapAtNodes) {
  Polygon<point_t, index_t> poly1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "POLYGON ((1 5, 5 5, 5 1, 1 1, 1 5))";
  std::string wkt2 = "POLYGON ((7 3, 5 1, 3 3, 5 5, 7 3))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, poly2);
}

TEST(RelateTest, PolygonsContainedAtNodes) {
  Polygon<point_t, index_t> poly1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "POLYGON ((1 5, 5 5, 6 2, 1 1, 1 5))";
  std::string wkt2 = "POLYGON ((1 1, 5 5, 6 2, 1 1))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, poly2);
}

/*
// FIXME: IM__EXTER_BOUND_1D should not be set
TEST(RelateTest, PolygonsNestedWithHole) {
  Polygon<point_t, index_t> poly1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 =
      "POLYGON ((40 60, 420 60, 420 320, 40 320, 40 60), (200 140, 160 220, 260 200, 200
140))"; std::string wkt2 = "POLYGON ((80 100, 360 100, 360 280, 80 280, 80 100))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, poly2);
}
*/

TEST(RelateTest, PolygonsOverlappingWithBoundaryInside) {
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

TEST(RelateTest, PolygonsOverlapVeryNarrow) {
  Polygon<point_t, index_t> poly1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "POLYGON ((120 100, 120 200, 200 200, 200 100, 120 100))";
  std::string wkt2 = "POLYGON ((100 100, 100000 110, 100000 100, 100 100))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, poly2);
}

TEST(RelateTest, ValidateRelateAA_86) {
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

TEST(RelateTest, ValidateRelateAA_97) {
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

TEST(RelateTest, AdjacentPolygons) {
  Polygon<point_t, index_t> poly1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "POLYGON ((1 9, 6 9, 6 1, 1 1, 1 9))";
  std::string wkt2 = "POLYGON ((9 9, 9 4, 6 4, 6 9, 9 9))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, poly2);
}

TEST(RelateTest, AdjacentPolygonsTouchingAtPoint) {
  Polygon<point_t, index_t> poly1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "POLYGON ((1 9, 6 9, 6 1, 1 1, 1 9))";
  std::string wkt2 = "POLYGON ((9 9, 9 4, 6 4, 7 9, 9 9))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, poly2);
}

TEST(RelateTest, AdjacentPolygonsOverlappping) {
  Polygon<point_t, index_t> poly1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "POLYGON ((1 9, 6 9, 6 1, 1 1, 1 9))";
  std::string wkt2 = "POLYGON ((9 9, 9 4, 6 4, 5 9, 9 9))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, poly2);
}

TEST(RelateTest, ContainsProperlyPolygonContained) {
  Polygon<point_t, index_t> poly1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "POLYGON ((1 9, 9 9, 9 1, 1 1, 1 9))";
  std::string wkt2 = "POLYGON ((2 8, 5 8, 5 5, 2 5, 2 8))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, poly2);
}

TEST(RelateTest, ContainsProperlyPolygonTouching) {
  Polygon<point_t, index_t> poly1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "POLYGON ((1 9, 9 9, 9 1, 1 1, 1 9))";
  std::string wkt2 = "POLYGON ((9 1, 5 1, 5 5, 9 5, 9 1))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, poly2);
}

TEST(RelateTest, ContainsProperlyPolygonsOverlapping) {
  MultiPolygon<point_t, index_t> m1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "MULTIPOLYGON (((1 9, 6 9, 6 4, 1 4, 1 9)), ((2 4, 6 7, 9 1, 2 4)))";
  std::string wkt2 = "POLYGON ((5 5, 6 5, 6 4, 5 4, 5 5))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTMultiPolygon(ctx1, wkt1.c_str(), m1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), m1, poly2);
}

TEST(RelateTest, RepeatedPointLL) {
  LineString<point_t> l1;
  LineString<point_t> l2;
  std::string wkt1 = "LINESTRING(0 0, 5 5, 5 5, 5 5, 9 9)";
  std::string wkt2 = "LINESTRING(0 9, 5 5, 5 5, 5 5, 9 0)";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTLineString(ctx1, wkt1.c_str(), l1);
  ParseWKTLineString(ctx2, wkt2.c_str(), l2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), l1, l2);
}

TEST(RelateTest, RepeatedPointAA) {
  Polygon<point_t, index_t> poly1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "POLYGON ((1 9, 9 7, 9 1, 1 3, 1 9))";
  std::string wkt2 = "POLYGON ((1 3, 1 3, 1 3, 3 7, 9 7, 9 7, 1 3))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, poly2);
}

// Define non-empty WKTs for reuse
const std::string NE_POINT = "POINT (1 1)";
const std::string NE_LINE = "LINESTRING (1 1, 2 2)";
const std::string NE_POLY = "POLYGON ((1 1, 1 2, 2 1, 1 1))";

// Define empty WKTs for reuse
const std::string E_POINT = "POINT EMPTY";
const std::string E_LINE = "LINESTRING EMPTY";
const std::string E_POLY = "POLYGON EMPTY";
const std::string E_MPOINT = "MULTIPOINT EMPTY";
const std::string E_MLINE = "MULTILINESTRING EMPTY";
const std::string E_MPOLY = "MULTIPOLYGON EMPTY";

// Note: POINT EMPTY is parsed as an empty MultiPoint, as the Point parser expects a
// single coordinate.

/******************************************************
 * Tests for Empty Geometries vs. Non-Empty Geometries
 ******************************************************/

// --- POINT EMPTY vs Non-Empty ---

TEST(RelateEmptyTest, PointEmpty_vs_Point) {
  point_t g1;
  point_t g2;
  ParseWKTPoint(E_POINT.c_str(), g1);
  ParseWKTPoint(NE_POINT.c_str(), g2);
  TestRelate(E_POINT.c_str(), NE_POINT.c_str(), g1, g2);
}

TEST(RelateEmptyTest, PointEmpty_vs_LineString) {
  point_t g1;
  LineString<point_t> g2;
  Context<point_t, index_t> ctx;
  ParseWKTPoint(E_POINT.c_str(), g1);
  ParseWKTLineString(ctx, NE_LINE.c_str(), g2);
  TestRelate(E_POINT.c_str(), NE_LINE.c_str(), g1, g2);
}

TEST(RelateEmptyTest, PointEmpty_vs_Polygon) {
  point_t g1;
  Polygon<point_t, index_t> g2;
  Context<point_t, index_t> ctx2;
  ParseWKTPoint(E_POINT.c_str(), g1);
  ParseWKTPolygon(ctx2, NE_POLY.c_str(), g2);
  TestRelate(E_POINT.c_str(), NE_POLY.c_str(), g1, g2);
}

// --- LINESTRING EMPTY vs Non-Empty ---

TEST(RelateEmptyTest, LineStringEmpty_vs_Point) {
  LineString<point_t> g1;
  point_t g2;
  Context<point_t, index_t> ctx1;
  ParseWKTLineString(ctx1, E_LINE.c_str(), g1);
  ParseWKTPoint(NE_POINT.c_str(), g2);
  TestRelate(E_LINE.c_str(), NE_POINT.c_str(), g1, g2);
}

TEST(RelateEmptyTest, LineStringEmpty_vs_LineString) {
  LineString<point_t> g1, g2;
  Context<point_t, index_t> ctx1, ctx2;
  ParseWKTLineString(ctx1, E_LINE.c_str(), g1);
  ParseWKTLineString(ctx2, NE_LINE.c_str(), g2);
  TestRelate(E_LINE.c_str(), NE_LINE.c_str(), g1, g2);
}

TEST(RelateEmptyTest, LineStringEmpty_vs_Polygon) {
  LineString<point_t> g1;
  Polygon<point_t, index_t> g2;
  Context<point_t, index_t> ctx1, ctx2;
  ParseWKTLineString(ctx1, E_LINE.c_str(), g1);
  ParseWKTPolygon(ctx2, NE_POLY.c_str(), g2);
  TestRelate(E_LINE.c_str(), NE_POLY.c_str(), g1, g2);
}

// --- POLYGON EMPTY vs Non-Empty ---

TEST(RelateEmptyTest, PolygonEmpty_vs_Point) {
  Polygon<point_t, index_t> g1;
  point_t g2;
  Context<point_t, index_t> ctx1;
  ParseWKTPolygon(ctx1, E_POLY.c_str(), g1);
  ParseWKTPoint(NE_POINT.c_str(), g2);
  TestRelate(E_POLY.c_str(), NE_POINT.c_str(), g1, g2);
}

TEST(RelateEmptyTest, PolygonEmpty_vs_LineString) {
  Polygon<point_t, index_t> g1;
  LineString<point_t> g2;
  Context<point_t, index_t> ctx1, ctx2;
  ParseWKTPolygon(ctx1, E_POLY.c_str(), g1);
  ParseWKTLineString(ctx2, NE_LINE.c_str(), g2);
  TestRelate(E_POLY.c_str(), NE_LINE.c_str(), g1, g2);
}

TEST(RelateEmptyTest, PolygonEmpty_vs_Polygon) {
  Polygon<point_t, index_t> g1, g2;
  Context<point_t, index_t> ctx1, ctx2;
  ParseWKTPolygon(ctx1, E_POLY.c_str(), g1);
  ParseWKTPolygon(ctx2, NE_POLY.c_str(), g2);
  TestRelate(E_POLY.c_str(), NE_POLY.c_str(), g1, g2);
}

// --- MULTIPOINT EMPTY vs Non-Empty ---

TEST(RelateEmptyTest, MultiPointEmpty_vs_Point) {
  MultiPoint<point_t> g1;
  point_t g2;
  Context<point_t, index_t> ctx1;
  ParseWKTMultiPoint(ctx1, E_MPOINT.c_str(), g1);
  ParseWKTPoint(NE_POINT.c_str(), g2);
  TestRelate(E_MPOINT.c_str(), NE_POINT.c_str(), g1, g2);
}

TEST(RelateEmptyTest, MultiPointEmpty_vs_LineString) {
  MultiPoint<point_t> g1;
  LineString<point_t> g2;
  Context<point_t, index_t> ctx1, ctx2;
  ParseWKTMultiPoint(ctx1, E_MPOINT.c_str(), g1);
  ParseWKTLineString(ctx2, NE_LINE.c_str(), g2);
  TestRelate(E_MPOINT.c_str(), NE_LINE.c_str(), g1, g2);
}

TEST(RelateEmptyTest, MultiPointEmpty_vs_Polygon) {
  MultiPoint<point_t> g1;
  Polygon<point_t, index_t> g2;
  Context<point_t, index_t> ctx1, ctx2;
  ParseWKTMultiPoint(ctx1, E_MPOINT.c_str(), g1);
  ParseWKTPolygon(ctx2, NE_POLY.c_str(), g2);
  TestRelate(E_MPOINT.c_str(), NE_POLY.c_str(), g1, g2);
}

// --- MULTILINESTRING EMPTY vs Non-Empty ---

TEST(RelateEmptyTest, MultiLineStringEmpty_vs_Point) {
  MultiLineString<point_t, index_t> g1;
  point_t g2;
  Context<point_t, index_t> ctx1;
  ParseWKTMultiLineString(ctx1, E_MLINE.c_str(), g1);
  ParseWKTPoint(NE_POINT.c_str(), g2);
  TestRelate(E_MLINE.c_str(), NE_POINT.c_str(), g1, g2);
}

TEST(RelateEmptyTest, MultiLineStringEmpty_vs_LineString) {
  MultiLineString<point_t, index_t> g1;
  LineString<point_t> g2;
  Context<point_t, index_t> ctx1, ctx2;
  ParseWKTMultiLineString(ctx1, E_MLINE.c_str(), g1);
  ParseWKTLineString(ctx2, NE_LINE.c_str(), g2);
  TestRelate(E_MLINE.c_str(), NE_LINE.c_str(), g1, g2);
}

TEST(RelateEmptyTest, MultiLineStringEmpty_vs_Polygon) {
  MultiLineString<point_t, index_t> g1;
  Polygon<point_t, index_t> g2;
  Context<point_t, index_t> ctx1, ctx2;
  ParseWKTMultiLineString(ctx1, E_MLINE.c_str(), g1);
  ParseWKTPolygon(ctx2, NE_POLY.c_str(), g2);
  TestRelate(E_MLINE.c_str(), NE_POLY.c_str(), g1, g2);
}

// --- MULTIPOLYGON EMPTY vs Non-Empty ---

TEST(RelateEmptyTest, MultiPolygonEmpty_vs_Point) {
  MultiPolygon<point_t, index_t> g1;
  point_t g2;
  Context<point_t, index_t> ctx1;
  ParseWKTMultiPolygon(ctx1, E_MPOLY.c_str(), g1);
  ParseWKTPoint(NE_POINT.c_str(), g2);
  TestRelate(E_MPOLY.c_str(), NE_POINT.c_str(), g1, g2);
}

TEST(RelateEmptyTest, MultiPolygonEmpty_vs_LineString) {
  MultiPolygon<point_t, index_t> g1;
  LineString<point_t> g2;
  Context<point_t, index_t> ctx1, ctx2;
  ParseWKTMultiPolygon(ctx1, E_MPOLY.c_str(), g1);
  ParseWKTLineString(ctx2, NE_LINE.c_str(), g2);
  TestRelate(E_MPOLY.c_str(), NE_LINE.c_str(), g1, g2);
}

TEST(RelateEmptyTest, MultiPolygonEmpty_vs_Polygon) {
  MultiPolygon<point_t, index_t> g1;
  Polygon<point_t, index_t> g2;
  Context<point_t, index_t> ctx1, ctx2;
  ParseWKTMultiPolygon(ctx1, E_MPOLY.c_str(), g1);
  ParseWKTPolygon(ctx2, NE_POLY.c_str(), g2);
  TestRelate(E_MPOLY.c_str(), NE_POLY.c_str(), g1, g2);
}

/******************************************************
 * Tests for Non-Empty Geometries vs. Empty Geometries
 ******************************************************/

// --- Non-Empty POINT vs Empty ---

TEST(RelateEmptyTest, Point_vs_PointEmpty) {
  point_t g1;
  point_t g2;
  Context<point_t, index_t> ctx2;
  ParseWKTPoint(NE_POINT.c_str(), g1);
  ParseWKTPoint(E_POINT.c_str(), g2);
  TestRelate(NE_POINT.c_str(), E_POINT.c_str(), g1, g2);
}

TEST(RelateEmptyTest, Point_vs_LineStringEmpty) {
  point_t g1;
  LineString<point_t> g2;
  Context<point_t, index_t> ctx2;
  ParseWKTPoint(NE_POINT.c_str(), g1);
  ParseWKTLineString(ctx2, E_LINE.c_str(), g2);
  TestRelate(NE_POINT.c_str(), E_LINE.c_str(), g1, g2);
}

TEST(RelateEmptyTest, Point_vs_PolygonEmpty) {
  point_t g1;
  Polygon<point_t, index_t> g2;
  Context<point_t, index_t> ctx2;
  ParseWKTPoint(NE_POINT.c_str(), g1);
  ParseWKTPolygon(ctx2, E_POLY.c_str(), g2);
  TestRelate(NE_POINT.c_str(), E_POLY.c_str(), g1, g2);
}

// --- Non-Empty LINESTRING vs Empty ---

TEST(RelateEmptyTest, LineString_vs_PointEmpty) {
  LineString<point_t> g1;
  point_t g2;
  Context<point_t, index_t> ctx1;
  ParseWKTLineString(ctx1, NE_LINE.c_str(), g1);
  ParseWKTPoint(E_POINT.c_str(), g2);
  TestRelate(NE_LINE.c_str(), E_POINT.c_str(), g1, g2);
}

TEST(RelateEmptyTest, LineString_vs_LineStringEmpty) {
  LineString<point_t> g1, g2;
  Context<point_t, index_t> ctx1, ctx2;
  ParseWKTLineString(ctx1, NE_LINE.c_str(), g1);
  ParseWKTLineString(ctx2, E_LINE.c_str(), g2);
  TestRelate(NE_LINE.c_str(), E_LINE.c_str(), g1, g2);
}

TEST(RelateEmptyTest, LineString_vs_PolygonEmpty) {
  LineString<point_t> g1;
  Polygon<point_t, index_t> g2;
  Context<point_t, index_t> ctx1, ctx2;
  ParseWKTLineString(ctx1, NE_LINE.c_str(), g1);
  ParseWKTPolygon(ctx2, E_POLY.c_str(), g2);
  TestRelate(NE_LINE.c_str(), E_POLY.c_str(), g1, g2);
}

// --- Non-Empty POLYGON vs Empty ---

TEST(RelateEmptyTest, Polygon_vs_PointEmpty) {
  Polygon<point_t, index_t> g1;
  point_t g2;
  Context<point_t, index_t> ctx1;
  ParseWKTPolygon(ctx1, NE_POLY.c_str(), g1);
  ParseWKTPoint(E_POINT.c_str(), g2);
  TestRelate(NE_POLY.c_str(), E_POINT.c_str(), g1, g2);
}

TEST(RelateEmptyTest, Polygon_vs_LineStringEmpty) {
  Polygon<point_t, index_t> g1;
  LineString<point_t> g2;
  Context<point_t, index_t> ctx1, ctx2;
  ParseWKTPolygon(ctx1, NE_POLY.c_str(), g1);
  ParseWKTLineString(ctx2, E_LINE.c_str(), g2);
  TestRelate(NE_POLY.c_str(), E_LINE.c_str(), g1, g2);
}

TEST(RelateEmptyTest, Polygon_vs_PolygonEmpty) {
  Polygon<point_t, index_t> g1, g2;
  Context<point_t, index_t> ctx1, ctx2;
  ParseWKTPolygon(ctx1, NE_POLY.c_str(), g1);
  ParseWKTPolygon(ctx2, E_POLY.c_str(), g2);
  TestRelate(NE_POLY.c_str(), E_POLY.c_str(), g1, g2);
}

TEST(RelateTest, PreparedTest) {
  Polygon<point_t, index_t> poly1;
  Polygon<point_t, index_t> poly2;
  std::string wkt1 = "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))";
  std::string wkt2 = "POLYGON((0.5 0.5, 1.5 0.5, 1.5 1.5, 0.5 1.5, 0.5 0.5))";
  Context<point_t, index_t> ctx1, ctx2;

  ParseWKTPolygon(ctx1, wkt1.c_str(), poly1);
  ParseWKTPolygon(ctx2, wkt2.c_str(), poly2);
  TestRelate(wkt1.c_str(), wkt2.c_str(), poly1, poly2);
}

}  // namespace gpuspatial
