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

#include "gpuspatial/relate/segment_relate_ops.cuh"

#include <gtest/gtest.h>
#include <algorithm>
#include <vector>

namespace gpuspatial {

using point_d = Point<double, 2>;
using point_f = Point<float, 2>;

TEST(SegmentRelateOpsTest, ParamToleranceValues) {
  EXPECT_DOUBLE_EQ(detail::ParamTolerance<double>(), 1e-12);
  EXPECT_FLOAT_EQ(detail::ParamTolerance<float>(), 1e-5f);
}

TEST(SegmentRelateOpsTest, RobustOrientationCollinear) {
  point_d p1(0.0, 0.0);
  point_d p2(2.0, 2.0);
  point_d q_collinear(1.0, 1.0);
  EXPECT_EQ(detail::RobustOrientation(p1, p2, q_collinear), 0);

  point_d q_left(0.0, 1.0);
  EXPECT_EQ(detail::RobustOrientation(p1, p2, q_left), 1);

  point_d q_right(1.0, 0.0);
  EXPECT_EQ(detail::RobustOrientation(p1, p2, q_right), -1);
}

TEST(SegmentRelateOpsTest, RobustOrientationNearCollinear) {
  // Points along line y = x with a tiny perturbation that requires high precision
  point_d p1(0.5, 0.5);
  point_d p2(12.0, 12.0);
  point_d q_left(6.0 - 1e-13, 6.0 + 1e-13);
  point_d q_right(6.0 + 1e-13, 6.0 - 1e-13);

  EXPECT_EQ(detail::RobustOrientation(p1, p2, q_left), 1);
  EXPECT_EQ(detail::RobustOrientation(p1, p2, q_right), -1);
}

TEST(SegmentRelateOpsTest, ProjectPointOnSegment) {
  // Horizontal segment
  point_d h1(0.0, 0.0);
  point_d h2(10.0, 0.0);
  EXPECT_NEAR(detail::ProjectPointOnSegment(h1, h2, point_d(0.0, 0.0)), 0.0, 1e-12);
  EXPECT_NEAR(detail::ProjectPointOnSegment(h1, h2, point_d(10.0, 0.0)), 1.0, 1e-12);
  EXPECT_NEAR(detail::ProjectPointOnSegment(h1, h2, point_d(5.0, 0.0)), 0.5, 1e-12);
  EXPECT_NEAR(detail::ProjectPointOnSegment(h1, h2, point_d(-2.0, 0.0)), -0.2, 1e-12);
  EXPECT_NEAR(detail::ProjectPointOnSegment(h1, h2, point_d(15.0, 0.0)), 1.5, 1e-12);

  // Vertical segment
  point_d v1(0.0, 0.0);
  point_d v2(0.0, 20.0);
  EXPECT_NEAR(detail::ProjectPointOnSegment(v1, v2, point_d(0.0, 10.0)), 0.5, 1e-12);
}

TEST(SegmentRelateOpsTest, PointInSegmentEnvelope) {
  point_d p1(0.0, 0.0);
  point_d p2(10.0, 10.0);

  EXPECT_TRUE(detail::PointInSegmentEnvelope(p1, p2, point_d(5.0, 5.0)));
  EXPECT_TRUE(detail::PointInSegmentEnvelope(p1, p2, point_d(0.0, 0.0)));
  EXPECT_TRUE(detail::PointInSegmentEnvelope(p1, p2, point_d(10.0, 10.0)));
  EXPECT_TRUE(detail::PointInSegmentEnvelope(p1, p2, point_d(2.0, 8.0)));
  EXPECT_FALSE(detail::PointInSegmentEnvelope(p1, p2, point_d(11.0, 5.0)));
  EXPECT_FALSE(detail::PointInSegmentEnvelope(p1, p2, point_d(-1.0, 5.0)));
}

TEST(SegmentRelateOpsTest, AnalyzeSegmentProperCrossing) {
  point_d v1(0.0, 0.0);
  point_d v2(10.0, 10.0);
  point_d q1(0.0, 10.0);
  point_d q2(10.0, 0.0);

  uint32_t bb_flags = 0;
  double t_next = 1.0;
  double t_collinear_end = 0.0;

  detail::AnalyzeSegmentIntersection(v1, v2, q1, q2, 0.0, bb_flags, t_next, t_collinear_end);

  EXPECT_TRUE((bb_flags & detail::SEG_HIT_ANY) != 0);
  EXPECT_TRUE((bb_flags & detail::SEG_PROPER_CROSS) != 0);
  EXPECT_NEAR(t_next, 0.5, 1e-12);
}

TEST(SegmentRelateOpsTest, AnalyzeSegmentCollinearOverlap) {
  point_d v1(0.0, 0.0);
  point_d v2(10.0, 0.0);
  point_d q1(4.0, 0.0);
  point_d q2(14.0, 0.0);

  uint32_t bb_flags = 0;
  double t_next = 1.0;
  double t_collinear_end = 0.0;

  detail::AnalyzeSegmentIntersection(v1, v2, q1, q2, 0.0, bb_flags, t_next, t_collinear_end);

  EXPECT_TRUE((bb_flags & detail::SEG_HIT_ANY) != 0);
  EXPECT_TRUE((bb_flags & detail::SEG_HIT_1D) != 0);
  EXPECT_NEAR(t_next, 0.4, 1e-12);
}

TEST(SegmentRelateOpsTest, AnalyzeSegmentTJunctionTouch) {
  point_d v1(0.0, 5.0);
  point_d v2(10.0, 5.0);
  point_d q1(5.0, 0.0);
  point_d q2(5.0, 5.0);  // Touches interior of v1->v2 at (5, 5)

  uint32_t bb_flags = 0;
  double t_next = 1.0;
  double t_collinear_end = 0.0;

  detail::AnalyzeSegmentIntersection(v1, v2, q1, q2, 0.0, bb_flags, t_next, t_collinear_end);

  EXPECT_TRUE((bb_flags & detail::SEG_HIT_ANY) != 0);
  EXPECT_TRUE((bb_flags & detail::SEG_HIT_0D) != 0);
  EXPECT_FALSE((bb_flags & detail::SEG_PROPER_CROSS) != 0);
  EXPECT_NEAR(t_next, 0.5, 1e-12);
}

TEST(SegmentRelateOpsTest, AnalyzeSegmentDisjoint) {
  point_d v1(0.0, 0.0);
  point_d v2(10.0, 0.0);
  point_d q1(0.0, 5.0);
  point_d q2(10.0, 5.0);

  uint32_t bb_flags = 0;
  double t_next = 1.0;
  double t_collinear_end = 0.0;

  detail::AnalyzeSegmentIntersection(v1, v2, q1, q2, 0.0, bb_flags, t_next, t_collinear_end);

  EXPECT_EQ(bb_flags, 0u);
  EXPECT_DOUBLE_EQ(t_next, 1.0);
}

struct TestPolygonHolder {
  std::vector<uint32_t> ring_offsets;
  std::vector<point_d> vertices;
  Box<Point<float, 2>> mbr;

  Polygon<point_d, uint32_t> AsPolygon() const {
    return Polygon<point_d, uint32_t>(
        ArrayView<uint32_t>(const_cast<uint32_t*>(ring_offsets.data()), ring_offsets.size()),
        ArrayView<point_d>(const_cast<point_d*>(vertices.data()), vertices.size()),
        mbr);
  }
};

static TestPolygonHolder MakeTestPolygon(const std::vector<std::vector<point_d>>& rings) {
  TestPolygonHolder result;
  result.ring_offsets.push_back(0);
  for (const auto& ring : rings) {
    for (const auto& pt : ring) {
      result.vertices.push_back(pt);
      result.mbr.Expand(Point<float, 2>(static_cast<float>(pt.x()), static_cast<float>(pt.y())));
    }
    result.ring_offsets.push_back(static_cast<uint32_t>(result.vertices.size()));
  }
  return result;
}

TEST(SegmentRelateOpsTest, FindPolygonInteriorPointConvex) {
  auto poly_holder = MakeTestPolygon({
      {point_d(0.0, 0.0), point_d(10.0, 0.0), point_d(10.0, 10.0), point_d(0.0, 10.0),
       point_d(0.0, 0.0)},
  });
  auto poly = poly_holder.AsPolygon();

  point_d interior_pt;
  bool ok = detail::FindPolygonInteriorPoint(poly, interior_pt);
  ASSERT_TRUE(ok);
  EXPECT_EQ(poly.locate_point(interior_pt), PointLocation::kInside);
}

TEST(SegmentRelateOpsTest, FindPolygonInteriorPointConcaveLShape) {
  // L-shaped polygon where centroid might be close to or outside the boundary
  auto poly_holder = MakeTestPolygon({
      {point_d(0.0, 0.0), point_d(10.0, 0.0), point_d(10.0, 4.0), point_d(4.0, 4.0),
       point_d(4.0, 10.0), point_d(0.0, 10.0), point_d(0.0, 0.0)},
  });
  auto poly = poly_holder.AsPolygon();

  point_d interior_pt;
  bool ok = detail::FindPolygonInteriorPoint(poly, interior_pt);
  ASSERT_TRUE(ok);
  EXPECT_EQ(poly.locate_point(interior_pt), PointLocation::kInside);
}

TEST(SegmentRelateOpsTest, FindPolygonInteriorPointWithHole) {
  // Donut polygon with a 4x4 hole in the center
  auto poly_holder = MakeTestPolygon({
      {point_d(0.0, 0.0), point_d(10.0, 0.0), point_d(10.0, 10.0), point_d(0.0, 10.0),
       point_d(0.0, 0.0)},
      {point_d(3.0, 3.0), point_d(7.0, 3.0), point_d(7.0, 7.0), point_d(3.0, 7.0),
       point_d(3.0, 3.0)},
  });
  auto poly = poly_holder.AsPolygon();

  point_d interior_pt;
  bool ok = detail::FindPolygonInteriorPoint(poly, interior_pt);
  ASSERT_TRUE(ok);
  EXPECT_EQ(poly.locate_point(interior_pt), PointLocation::kInside);
}

TEST(SegmentRelateOpsTest, FindPolygonInteriorPointMicroPolygon) {
  auto poly_holder = MakeTestPolygon({
      {point_d(0.0, 0.0), point_d(1e-8, 0.0), point_d(1e-8, 1e-8), point_d(0.0, 1e-8),
       point_d(0.0, 0.0)},
  });
  auto poly = poly_holder.AsPolygon();

  point_d interior_pt;
  bool ok = detail::FindPolygonInteriorPoint(poly, interior_pt);
  ASSERT_TRUE(ok);
  EXPECT_EQ(poly.locate_point(interior_pt), PointLocation::kInside);
}

TEST(SegmentRelateOpsTest, FindPolygonInteriorPointExceedsMaxCrossingsFallback) {
  // A comb polygon with 45 teeth produces 90 scanline crossings, exceeding kMaxCrossings = 64.
  // Tests the fallback to the extremal-convex-vertex angle bisector.
  std::vector<point_d> ring;
  ring.emplace_back(0.0, 0.0);
  for (int i = 0; i < 45; ++i) {
    ring.emplace_back(2.0 * i, 10.0);
    ring.emplace_back(2.0 * i + 1.0, 10.0);
    ring.emplace_back(2.0 * i + 1.0, 0.0);
  }
  ring.emplace_back(90.0, 0.0);
  ring.emplace_back(90.0, -5.0);
  ring.emplace_back(0.0, -5.0);
  ring.emplace_back(0.0, 0.0);

  auto poly_holder = MakeTestPolygon({ring});
  auto poly = poly_holder.AsPolygon();
  point_d interior_pt;

  // Test the extremal vertex method directly
  bool ok_extremal = detail::FindPolygonInteriorPointExtremalVertex(poly, interior_pt);
  ASSERT_TRUE(ok_extremal);
  EXPECT_EQ(poly.locate_point(interior_pt), PointLocation::kInside);

  // Test that FindPolygonInteriorPoint transparently falls back and succeeds
  bool ok = detail::FindPolygonInteriorPoint(poly, interior_pt);
  ASSERT_TRUE(ok);
  EXPECT_EQ(poly.locate_point(interior_pt), PointLocation::kInside);
}

}  // namespace gpuspatial
