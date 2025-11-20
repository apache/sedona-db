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
#include "gpuspatial/geom/geometry_collection.cuh"
#include "gpuspatial/geom/multi_polygon.cuh"
#include "gpuspatial/loader/device_geometries.cuh"
#include "gpuspatial/utils/pinned_vector.h"
#include "nanoarrow/nanoarrow.hpp"

#include "gpuspatial/geom/multi_point.cuh"
#include "test_common.hpp"

#include <geoarrow/geoarrow.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rmm/cuda_stream.hpp>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>
#include "gpuspatial/loader/parallel_wkb_loader.h"
namespace gpuspatial {

template <typename T>
class WKBLoaderTest : public ::testing::Test {};

TYPED_TEST_SUITE(WKBLoaderTest, TestUtils::PointIndexTypePairs);
TYPED_TEST(WKBLoaderTest, Point) {
  using point_t = typename TypeParam::first_type;
  using index_t = typename TypeParam::second_type;
  nanoarrow::UniqueArrayStream stream;
  ArrayStreamFromWKT({{"POINT (0 0)"},
                      {"POINT (10 20)", "POINT (-5.5 -12.3)"},
                      {"POINT (100 -50)", "POINT (3.1415926535 2.7182818284)",
                       "POINT (0.0001 0.00005)", "POINT (-1234567.89 -9876543.21)"},
                      {"POINT (999999999 1)", "POINT (1 999999999)", "POINT EMPTY"}},
                     GEOARROW_TYPE_WKB, stream.get());

  rmm::cuda_stream cuda_stream;
  ParallelWkbLoader<point_t, index_t> loader;
  typename ParallelWkbLoader<point_t, index_t>::Config config;

  loader.Init(config);

  while (1) {
    nanoarrow::UniqueArray array;
    ArrowError error;
    ArrowErrorSet(&error, "Failed to get next array from stream");
    EXPECT_EQ(ArrowArrayStreamGetNext(stream.get(), array.get(), &error), NANOARROW_OK);
    if (array->length == 0) {
      break;
    }
    loader.Parse(cuda_stream, array.get(), 0, array->length);
  }

  auto geometries = loader.Finish(cuda_stream);
  auto points = TestUtils::ToVector(cuda_stream, geometries.get_points());
  cuda_stream.synchronize();
  EXPECT_EQ(points.size(), 10);
  EXPECT_EQ(points[0], point_t(0, 0));
  EXPECT_EQ(points[1], point_t(10, 20));
  EXPECT_EQ(points[2], point_t(-5.5, -12.3));
  EXPECT_EQ(points[3], point_t(100, -50));
  EXPECT_EQ(points[4], point_t(3.1415926535, 2.7182818284));
  EXPECT_EQ(points[5], point_t(0.0001, 0.00005));
  EXPECT_EQ(points[6], point_t(-1234567.89, -9876543.21));
  EXPECT_EQ(points[7], point_t(999999999, 1));
  EXPECT_EQ(points[8], point_t(1, 999999999));
  EXPECT_TRUE(points[9].empty());
}

TYPED_TEST(WKBLoaderTest, MultiPoint) {
  using point_t = typename TypeParam::first_type;
  using index_t = typename TypeParam::second_type;
  nanoarrow::UniqueArrayStream stream;
  ArrayStreamFromWKT({{"MULTIPOINT ((0 0), (1 1))"},
                      {"MULTIPOINT ((2 2), (3 3), (4 4), EMPTY)"},
                      {"MULTIPOINT ((-1 -1))"},
                      {"MULTIPOINT EMPTY"},
                      {"MULTIPOINT ((5.5 6.6), (7.7 8.8))"}},
                     GEOARROW_TYPE_WKB, stream.get());
  ParallelWkbLoader<point_t, index_t> loader;
  typename ParallelWkbLoader<point_t, index_t>::Config config;
  rmm::cuda_stream cuda_stream;

  loader.Init(config);

  while (1) {
    nanoarrow::UniqueArray array;
    ArrowError error;
    ArrowErrorSet(&error, "Failed to get next array from stream");
    EXPECT_EQ(ArrowArrayStreamGetNext(stream.get(), array.get(), &error), NANOARROW_OK);
    if (array->length == 0) {
      break;
    }
    loader.Parse(cuda_stream, array.get(), 0, array->length);
  }

  auto geometries = loader.Finish(cuda_stream);
  auto offsets = TestUtils::ToVector(
      cuda_stream, geometries.get_offsets().multi_point_offsets.ps_num_points);
  auto points = TestUtils::ToVector(cuda_stream, geometries.get_points());
  auto mbrs = TestUtils::ToVector(cuda_stream, geometries.get_mbrs());
  cuda_stream.synchronize();
  MultiPointArrayView<point_t, index_t> array_view(
      ArrayView<index_t>{offsets}, ArrayView<point_t>{points},
      ArrayView<Box<Point<float, point_t::n_dim>>>{mbrs});
  EXPECT_EQ(array_view.size(), 5);
  EXPECT_EQ(array_view[0].num_points(), 2);
  EXPECT_EQ(array_view[0].get_point(0), point_t(0, 0));
  EXPECT_EQ(array_view[0].get_point(1), point_t(1, 1));

  EXPECT_EQ(array_view[1].num_points(), 4);
  EXPECT_EQ(array_view[1].get_point(0), point_t(2, 2));
  EXPECT_EQ(array_view[1].get_point(1), point_t(3, 3));
  EXPECT_EQ(array_view[1].get_point(2), point_t(4, 4));
  EXPECT_TRUE(array_view[1].get_point(3).empty());

  EXPECT_EQ(array_view[2].num_points(), 1);
  EXPECT_EQ(array_view[2].get_point(0), point_t(-1, -1));

  EXPECT_EQ(array_view[3].num_points(), 0);
  EXPECT_EQ(array_view[4].num_points(), 2);
  EXPECT_EQ(array_view[4].get_point(0), point_t(5.5, 6.6));
  EXPECT_EQ(array_view[4].get_point(1), point_t(7.7, 8.8));
}

TYPED_TEST(WKBLoaderTest, PointMultiPoint) {
  using point_t = typename TypeParam::first_type;
  using index_t = typename TypeParam::second_type;
  nanoarrow::UniqueArrayStream stream;
  ArrayStreamFromWKT({{"POINT (1 2)", "MULTIPOINT ((3 4), (5 6))"},
                      {"POINT (7 8)", "MULTIPOINT ((9 10))"},
                      {"MULTIPOINT EMPTY", "POINT (11 12)"}},
                     GEOARROW_TYPE_WKB, stream.get());
  rmm::cuda_stream cuda_stream;
  ParallelWkbLoader<point_t, index_t> loader;
  typename ParallelWkbLoader<point_t, index_t>::Config config;
  loader.Init(config);

  while (1) {
    nanoarrow::UniqueArray array;
    ArrowError error;
    ArrowErrorSet(&error, "Failed to get next array from stream");
    EXPECT_EQ(ArrowArrayStreamGetNext(stream.get(), array.get(), &error), NANOARROW_OK);
    if (array->length == 0) {
      break;
    }
    loader.Parse(cuda_stream, array.get(), 0, array->length);
  }

  auto geometries = loader.Finish(cuda_stream);
  auto offsets = TestUtils::ToVector(
      cuda_stream, geometries.get_offsets().multi_point_offsets.ps_num_points);
  auto points = TestUtils::ToVector(cuda_stream, geometries.get_points());
  auto mbrs = TestUtils::ToVector(cuda_stream, geometries.get_mbrs());
  cuda_stream.synchronize();
  MultiPointArrayView<point_t, index_t> array_view(
      ArrayView<index_t>{offsets}, ArrayView<point_t>{points},
      ArrayView<Box<Point<float, point_t::n_dim>>>{mbrs});
  EXPECT_EQ(array_view.size(), 6);
  EXPECT_EQ(array_view[0].num_points(), 1);
  EXPECT_EQ(array_view[0].get_point(0), point_t(1, 2));

  EXPECT_EQ(array_view[1].num_points(), 2);
  EXPECT_EQ(array_view[1].get_point(0), point_t(3, 4));
  EXPECT_EQ(array_view[1].get_point(1), point_t(5, 6));

  EXPECT_EQ(array_view[2].num_points(), 1);
  EXPECT_EQ(array_view[2].get_point(0), point_t(7, 8));

  EXPECT_EQ(array_view[3].num_points(), 1);
  EXPECT_EQ(array_view[3].get_point(0), point_t(9, 10));

  EXPECT_EQ(array_view[4].num_points(), 0);

  EXPECT_EQ(array_view[5].num_points(), 1);
  EXPECT_EQ(array_view[5].get_point(0), point_t(11, 12));
}

TYPED_TEST(WKBLoaderTest, PointWKBLoaderArrowIPC) {
  nanoarrow::UniqueArrayStream stream;

  auto path = TestUtils::GetTestDataPath("../test_data/test_points.arrows");

  ArrayStreamFromIpc(path, "geometry", stream.get());

  nanoarrow::UniqueArray array;
  ArrowError error;
  ArrowErrorSet(&error, "Failed to get next array from stream");

  using point_t = typename TypeParam::first_type;
  using index_t = typename TypeParam::second_type;
  rmm::cuda_stream cuda_stream;
  ParallelWkbLoader<point_t, index_t> loader;

  loader.Init();
  for (int i = 0; i < 100; i++) {
    ASSERT_EQ(ArrowArrayStreamGetNext(stream.get(), array.get(), &error), NANOARROW_OK);
    loader.Parse(cuda_stream, array.get(), 0, array->length);
    auto geometries = loader.Finish(cuda_stream);
    ASSERT_EQ(geometries.get_points().size(), 1000);
  }
}

TYPED_TEST(WKBLoaderTest, PolygonWKBLoaderArrowIPC) {
  using point_t = typename TypeParam::first_type;
  using index_t = typename TypeParam::second_type;
  nanoarrow::UniqueArrayStream stream;

  auto path = TestUtils::GetTestDataPath("../test_data/test_polygons.arrows");

  ArrayStreamFromIpc(path, "geometry", stream.get());

  nanoarrow::UniqueArray array;
  ArrowError error;
  ArrowErrorSet(&error, "Failed to get next array from stream");

  double polysize = 0.5;
  int n_row_groups = 100;
  int n_per_row_group = 1000;
  rmm::cuda_stream cuda_stream;

  ParallelWkbLoader<point_t, index_t> loader;

  loader.Init();

  for (int i = 0; i < n_row_groups; i++) {
    ASSERT_EQ(ArrowArrayStreamGetNext(stream.get(), array.get(), &error), NANOARROW_OK);
    loader.Parse(cuda_stream, array.get(), 0, array->length);
  }

  auto geometries = loader.Finish(cuda_stream);
  auto points = TestUtils::ToVector(cuda_stream, geometries.get_points());
  auto& offsets = geometries.get_offsets();

  auto ps_num_rings =
      TestUtils::ToVector(cuda_stream, offsets.polygon_offsets.ps_num_rings);
  auto ps_num_points =
      TestUtils::ToVector(cuda_stream, offsets.polygon_offsets.ps_num_points);
  auto mbrs = TestUtils::ToVector(cuda_stream, geometries.get_mbrs());
  cuda_stream.synchronize();
  ArrayView<index_t> v_ps_num_rings(ps_num_rings);
  ArrayView<index_t> v_ps_num_points(ps_num_points);
  ArrayView<point_t> v_points(points);

  PolygonArrayView<point_t, index_t> polygon_array(
      v_ps_num_rings, v_ps_num_points, v_points,
      ArrayView<Box<Point<float, point_t::n_dim>>>(mbrs));

  ASSERT_EQ(polygon_array.size(), n_row_groups * n_per_row_group);

  for (size_t geom_idx = 0; geom_idx < polygon_array.size(); geom_idx++) {
    auto polygon = polygon_array[geom_idx];

    auto line_string = polygon.get_ring(0);
    assert(line_string.num_segments() <= 9);

    for (size_t point_idx = 0; point_idx < line_string.num_points(); point_idx++) {
      const auto& point = line_string.get_point(point_idx);
      auto x = point.get_coordinate(0);
      auto y = point.get_coordinate(1);
      ASSERT_TRUE(x >= -polysize && x <= 1 + polysize);
      ASSERT_TRUE(y >= -polysize && y <= 1 + polysize);
    }
  }
}

TYPED_TEST(WKBLoaderTest, PolygonWKBLoaderWithHoles) {
  using point_t = typename TypeParam::first_type;
  using index_t = typename TypeParam::second_type;
  nanoarrow::UniqueArrayStream stream;
  ArrayStreamFromWKT(
      {{"POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))",
        "POLYGON ((35 10, 45 45, 15 40, 10 20, 35 10), (20 30, 35 35, 30 20, 20 30))",
        "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 3 2, 3 3, 2 3, 2 2), (6 6, 8 6, 8 8, 6 8, 6 6))",
        "POLYGON ((30 0, 60 20, 50 50, 10 50, 0 20, 30 0), (20 30, 25 40, 15 40, 20 30), (30 30, 35 40, 25 40, 30 30), (40 30, 45 40, 35 40, 40 30))",
        "POLYGON ((40 0, 50 30, 80 20, 90 70, 60 90, 30 80, 20 40, 40 0), (50 20, 65 30, 60 50, 45 40, 50 20), (30 60, 50 70, 45 80, 30 60))"}},
      GEOARROW_TYPE_WKB, stream.get());

  nanoarrow::UniqueArray array;
  ArrowError error;
  ArrowErrorSet(&error, "Failed to get next array from stream");

  ParallelWkbLoader<point_t, index_t> loader;
  rmm::cuda_stream cuda_stream;

  loader.Init();

  ASSERT_EQ(ArrowArrayStreamGetNext(stream.get(), array.get(), &error), NANOARROW_OK);

  loader.Parse(cuda_stream, array.get(), 0, array->length);
  auto geometries = loader.Finish(cuda_stream);

  auto points = TestUtils::ToVector(cuda_stream, geometries.get_points());
  const auto& offsets = geometries.get_offsets();
  auto ps_num_rings =
      TestUtils::ToVector(cuda_stream, offsets.polygon_offsets.ps_num_rings);
  auto ps_num_points =
      TestUtils::ToVector(cuda_stream, offsets.polygon_offsets.ps_num_points);
  auto mbrs = TestUtils::ToVector(cuda_stream, geometries.get_mbrs());
  cuda_stream.synchronize();
  ArrayView<index_t> v_ps_num_rings(ps_num_rings);
  ArrayView<index_t> v_ps_num_points(ps_num_points);
  ArrayView<point_t> v_points(points);
  ArrayView<Box<Point<float, point_t::n_dim>>> v_mbrs(mbrs.data(), mbrs.size());

  PolygonArrayView<point_t, index_t> polygon_array(v_ps_num_rings, v_ps_num_points,
                                                   v_points, v_mbrs);

  ASSERT_EQ(polygon_array.size(), 5);

  auto poly0 = polygon_array[0];
  ASSERT_EQ(poly0.num_rings(), 1);
  ASSERT_EQ(poly0.get_ring(0).num_segments(), 4);
  ASSERT_EQ(poly0.get_ring(0).num_points(), 5);

  ASSERT_TRUE(poly0.Contains(point_t{30, 20}));
  ASSERT_TRUE(poly0.Contains(point_t{22.5, 22.5}));
  ASSERT_FALSE(poly0.Contains(point_t{15, 15}));
  ASSERT_FALSE(poly0.Contains(point_t{40, 15}));

  auto poly1 = polygon_array[1];
  ASSERT_EQ(poly1.num_rings(), 2);
  ASSERT_EQ(poly1.get_ring(0).num_segments(), 4);
  ASSERT_EQ(poly1.get_ring(1).num_segments(), 3);

  ASSERT_TRUE(poly1.Contains(point_t{20, 20}));
  ASSERT_TRUE(poly1.Contains(point_t{35, 20}));
  ASSERT_FALSE(poly1.Contains(point_t{30, 25}));

  auto poly2 = polygon_array[2];

  ASSERT_EQ(poly2.num_rings(), 3);
  ASSERT_EQ(poly2.get_ring(0).num_segments(), 4);
  ASSERT_EQ(poly2.get_ring(1).num_segments(), 4);
  ASSERT_EQ(poly2.get_ring(2).num_segments(), 4);

  ASSERT_TRUE(poly2.Contains(point_t{1, 1}));
  ASSERT_TRUE(poly2.Contains(point_t{6, 4}));

  ASSERT_TRUE(poly2.Contains(point_t{9, 9}));
  ASSERT_FALSE(poly2.Contains(point_t{2.5, 2.5}));
  ASSERT_FALSE(poly2.Contains(point_t{7, 7}));
  ASSERT_FALSE(poly2.Contains(point_t{11, 11}));

  auto poly3 = polygon_array[3];
  ASSERT_EQ(poly3.num_rings(), 4);
  ASSERT_EQ(poly3.get_ring(0).num_segments(), 5);
  ASSERT_EQ(poly3.get_ring(1).num_segments(), 3);
  ASSERT_EQ(poly3.get_ring(2).num_segments(), 3);
  ASSERT_EQ(poly3.get_ring(3).num_segments(), 3);

  ASSERT_TRUE(poly3.Contains(point_t{30, 20}));
  ASSERT_TRUE(poly3.Contains(point_t{50, 40}));
  ASSERT_FALSE(poly3.Contains(point_t{20, 35}));
  ASSERT_FALSE(poly3.Contains(point_t{30, 35}));
  ASSERT_FALSE(poly3.Contains(point_t{40, 35}));

  auto poly4 = polygon_array[4];

  ASSERT_EQ(poly4.num_rings(), 3);
  ASSERT_EQ(poly4.get_ring(0).num_segments(), 7);
  ASSERT_EQ(poly4.get_ring(1).num_segments(), 4);
  ASSERT_EQ(poly4.get_ring(2).num_segments(), 3);

  ASSERT_TRUE(poly4.Contains(point_t{40, 20}));
  ASSERT_TRUE(poly4.Contains(point_t{60, 70}));
  ASSERT_FALSE(poly4.Contains(point_t{45, 70}));
  ASSERT_FALSE(poly4.Contains(point_t{55, 35}));
  // ASSERT_FALSE(poly4.Contains(point_t{52, 23}));

  index_t polygon_idx, ring_idx;
  uint32_t v_idx = 0;
  for (int polygon = 0; polygon < polygon_array.size(); polygon++) {
    for (int ring = 0; ring < polygon_array[polygon].num_rings(); ring++) {
      for (int v = 0; v < polygon_array[polygon].get_ring(ring).num_points(); v++) {
        ASSERT_TRUE(polygon_array.locate_vertex(v_idx++, polygon_idx, ring_idx));
        ASSERT_EQ(polygon_idx, polygon);
        ASSERT_EQ(ring_idx, ring);
      }
    }
  }
}

TYPED_TEST(WKBLoaderTest, PolygonWKBLoaderMultipolygon) {
  using point_t = typename TypeParam::first_type;
  using index_t = typename TypeParam::second_type;
  nanoarrow::UniqueArrayStream stream;
  ArrayStreamFromWKT(
      {{"POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))",
        "POLYGON ((35 10, 45 45, 15 40, 10 20, 35 10), (20 30, 35 35, 30 20, 20 30))",
        "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 3 2, 3 3, 2 3, 2 2), (6 6, 8 6, 8 8, 6 8, 6 6))",
        "MULTIPOLYGON (((0 0, 0 1, 1 1, 1 0, 0 0)), ((2 2, 2 3, 3 3, 3 2, 2 2)))",
        "POLYGON ((30 0, 60 20, 50 50, 10 50, 0 20, 30 0), (20 30, 25 40, 15 40, 20 30), (30 30, 35 40, 25 40, 30 30), (40 30, 45 40, 35 40, 40 30))",
        "MULTIPOLYGON (((40 40, 20 45, 45 30, 40 40)), ((20 35, 10 30, 10 10, 30 5, 45 20, 20 35), (30 20, 20 15, 20 25, 30 20)))",
        "POLYGON ((40 0, 50 30, 80 20, 90 70, 60 90, 30 80, 20 40, 40 0), (50 20, 65 30, 60 50, 45 40, 50 20), (30 60, 50 70, 45 80, 30 60))",
        "MULTIPOLYGON (((-1 0, 0 1, 1 0, 0 -1, -1 0)), ((2 2, 2 3, 3 3, 3 2, 2 2)), ((0 4, 1 5, 2 4, 0 4)))"}},
      GEOARROW_TYPE_WKB, stream.get());

  nanoarrow::UniqueArray array;
  ArrowError error;
  ArrowErrorSet(&error, "Failed to get next array from stream");

  rmm::cuda_stream cuda_stream;

  ASSERT_EQ(ArrowArrayStreamGetNext(stream.get(), array.get(), &error), NANOARROW_OK);

  ParallelWkbLoader<point_t, index_t> loader;

  loader.Init();
  loader.Parse(cuda_stream, array.get(), 0, array->length);

  auto geometries = loader.Finish(cuda_stream);
  const auto& offsets = geometries.get_offsets();
  auto points = TestUtils::ToVector(cuda_stream, geometries.get_points());
  auto prefix_sum_geoms =
      TestUtils::ToVector(cuda_stream, offsets.multi_polygon_offsets.ps_num_parts);
  auto prefix_sum_parts =
      TestUtils::ToVector(cuda_stream, offsets.multi_polygon_offsets.ps_num_rings);
  auto prefix_sum_rings =
      TestUtils::ToVector(cuda_stream, offsets.multi_polygon_offsets.ps_num_points);
  auto mbrs = TestUtils::ToVector(cuda_stream, geometries.get_mbrs());
  cuda_stream.synchronize();

  ArrayView<index_t> v_prefix_sum_geoms(prefix_sum_geoms);
  ArrayView<index_t> v_prefix_sum_parts(prefix_sum_parts);
  ArrayView<index_t> v_prefix_sum_rings(prefix_sum_rings);
  ArrayView<point_t> v_points(points);
  ArrayView<Box<Point<float, point_t::n_dim>>> v_mbrs(mbrs.data(), mbrs.size());

  MultiPolygonArrayView<point_t, index_t> multi_polygon_array(
      v_prefix_sum_geoms, v_prefix_sum_parts, v_prefix_sum_rings, v_points, v_mbrs);

  ASSERT_EQ(multi_polygon_array.size(), 8);

  ASSERT_EQ(multi_polygon_array[0].num_polygons(), 1);
  auto polygon = multi_polygon_array[0].get_polygon(0);
  ASSERT_EQ(polygon.num_rings(), 1);
  ASSERT_EQ(multi_polygon_array[1].num_polygons(), 1);
  polygon = multi_polygon_array[1].get_polygon(0);
  ASSERT_EQ(polygon.num_rings(), 2);
  ASSERT_EQ(multi_polygon_array[2].num_polygons(), 1);
  polygon = multi_polygon_array[2].get_polygon(0);
  ASSERT_EQ(polygon.num_rings(), 3);
  ASSERT_EQ(multi_polygon_array[3].num_polygons(), 2);
  polygon = multi_polygon_array[3].get_polygon(0);
  ASSERT_EQ(polygon.num_rings(), 1);
  polygon = multi_polygon_array[3].get_polygon(1);
  ASSERT_EQ(polygon.num_rings(), 1);
  ASSERT_EQ(multi_polygon_array[4].num_polygons(), 1);

  ASSERT_EQ(multi_polygon_array[5].num_polygons(), 2);
  polygon = multi_polygon_array[5].get_polygon(0);
  ASSERT_EQ(polygon.num_rings(), 1);
  polygon = multi_polygon_array[5].get_polygon(1);
  ASSERT_EQ(polygon.num_rings(), 2);
  ASSERT_EQ(multi_polygon_array[6].num_polygons(), 1);
  polygon = multi_polygon_array[6].get_polygon(0);
  ASSERT_EQ(polygon.num_rings(), 3);
  ASSERT_EQ(multi_polygon_array[7].num_polygons(), 3);
  polygon = multi_polygon_array[7].get_polygon(0);
  ASSERT_EQ(polygon.num_rings(), 1);
  polygon = multi_polygon_array[7].get_polygon(1);
  ASSERT_EQ(polygon.num_rings(), 1);
  polygon = multi_polygon_array[7].get_polygon(2);
  ASSERT_EQ(polygon.num_rings(), 1);

  uint32_t geom_idx, part_idx, ring_idx;
  uint32_t v_idx = 0;
  for (int geom = 0; geom < multi_polygon_array.size(); geom++) {
    const auto& polys = multi_polygon_array[geom];
    for (int part = 0; part < polys.num_polygons(); part++) {
      auto poly = polys.get_polygon(part);
      for (int ring = 0; ring < poly.num_rings(); ring++) {
        for (int v = 0; v < poly.get_ring(ring).num_points(); v++) {
          ASSERT_TRUE(
              multi_polygon_array.locate_vertex(v_idx++, geom_idx, part_idx, ring_idx));
          ASSERT_EQ(geom, geom_idx);
          ASSERT_EQ(part, part_idx);
          ASSERT_EQ(ring, ring_idx);
        }
      }
    }
  }
}

TYPED_TEST(WKBLoaderTest, PolygonWKBLoaderMultipolygonLocate) {
  using point_t = typename TypeParam::first_type;
  using index_t = typename TypeParam::second_type;
  nanoarrow::UniqueArrayStream stream;
  ArrayStreamFromWKT(
      {{"POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))",
        "POLYGON ((35 10, 45 45, 15 40, 10 20, 35 10), (20 30, 35 35, 30 20, 20 30))",
        "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 3 2, 3 3, 2 3, 2 2), (6 6, 8 6, 8 8, 6 8, 6 6))",
        "MULTIPOLYGON (((0 0, 0 1, 1 1, 1 0, 0 0)), EMPTY, ((2 2, 2 3, 3 3, 3 2, 2 2)))",
        "POLYGON ((30 0, 60 20, 50 50, 10 50, 0 20, 30 0), EMPTY, (20 30, 25 40, 15 40, 20 30), (30 30, 35 40, 25 40, 30 30), (40 30, 45 40, 35 40, 40 30))",
        "POLYGON EMPTY",
        "MULTIPOLYGON (((40 40, 20 45, 45 30, 40 40)), EMPTY, ((20 35, 10 30, 10 10, 30 5, 45 20, 20 35), (30 20, 20 15, 20 25, 30 20)))",
        "POLYGON EMPTY",
        "POLYGON ((40 0, 50 30, 80 20, 90 70, 60 90, 30 80, 20 40, 40 0), (50 20, 65 30, 60 50, 45 40, 50 20), (30 60, 50 70, 45 80, 30 60))",
        "MULTIPOLYGON (((-1 0, 0 1, 1 0, 0 -1, -1 0)), ((2 2, 2 3, 3 3, 3 2, 2 2)), ((0 4, 1 5, 2 4, 0 4)))"}},
      GEOARROW_TYPE_WKB, stream.get());

  nanoarrow::UniqueArray array;
  ArrowError error;
  ArrowErrorSet(&error, "Failed to get next array from stream");

  ParallelWkbLoader<point_t, index_t> loader;
  rmm::cuda_stream cuda_stream;

  loader.Init();
  ASSERT_EQ(ArrowArrayStreamGetNext(stream.get(), array.get(), &error), NANOARROW_OK);

  loader.Parse(cuda_stream, array.get(), 0, array->length);

  auto geometries = loader.Finish(cuda_stream);
  const auto& offsets = geometries.get_offsets();
  auto points = TestUtils::ToVector(cuda_stream, geometries.get_points());
  auto prefix_sum_geoms =
      TestUtils::ToVector(cuda_stream, offsets.multi_polygon_offsets.ps_num_parts);
  auto prefix_sum_parts =
      TestUtils::ToVector(cuda_stream, offsets.multi_polygon_offsets.ps_num_rings);
  auto prefix_sum_rings =
      TestUtils::ToVector(cuda_stream, offsets.multi_polygon_offsets.ps_num_points);
  auto mbrs = TestUtils::ToVector(cuda_stream, geometries.get_mbrs());
  cuda_stream.synchronize();

  ArrayView<index_t> v_prefix_sum_geoms(prefix_sum_geoms);
  ArrayView<index_t> v_prefix_sum_parts(prefix_sum_parts);
  ArrayView<index_t> v_prefix_sum_rings(prefix_sum_rings);
  ArrayView<point_t> v_points(points);
  ArrayView<Box<Point<float, point_t::n_dim>>> v_mbrs(mbrs.data(), mbrs.size());

  MultiPolygonArrayView<point_t, index_t> multi_polygon_array(
      v_prefix_sum_geoms, v_prefix_sum_parts, v_prefix_sum_rings, v_points, v_mbrs);

  uint32_t geom_idx, part_idx, ring_idx;
  uint32_t v_idx = 0;
  for (int geom = 0; geom < multi_polygon_array.size(); geom++) {
    const auto& polys = multi_polygon_array[geom];
    for (int part = 0; part < polys.num_polygons(); part++) {
      auto poly = polys.get_polygon(part);
      for (int ring = 0; ring < poly.num_rings(); ring++) {
        for (int v = 0; v < poly.get_ring(ring).num_points(); v++) {
          ASSERT_TRUE(
              multi_polygon_array.locate_vertex(v_idx++, geom_idx, part_idx, ring_idx));
          ASSERT_EQ(geom, geom_idx);
          ASSERT_EQ(part, part_idx);
          ASSERT_EQ(ring, ring_idx);
        }
      }
    }
  }
}

TYPED_TEST(WKBLoaderTest, MixTypes) {
  using point_t = typename TypeParam::first_type;
  using index_t = typename TypeParam::second_type;
  nanoarrow::UniqueArrayStream stream;

  ArrayStreamFromWKT(
      {
          {"POINT (30 10)", "POINT EMPTY", "LINESTRING (30 10, 10 30, 40 40)",
           "LINESTRING EMPTY",
           "POLYGON ((35 10, 45 45, 15 40, 10 20, 35 10), (20 30, 35 35, 30 20, 20 30))",
           "POLYGON EMPTY", "MULTIPOINT (10 40, 40 30, 20 20, 30 10)",
           "MULTILINESTRING ((10 10, 20 20, 10 40), (40 40, 30 30, 40 20, 30 10))",
           "MULTIPOLYGON (((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 20, 15 5)))"},
      },
      GEOARROW_TYPE_WKB, stream.get());
  nanoarrow::UniqueArray array;
  ArrowError error;
  ArrowErrorSet(&error, "Failed to get next array from stream");

  rmm::cuda_stream cuda_stream;

  ASSERT_EQ(ArrowArrayStreamGetNext(stream.get(), array.get(), &error), NANOARROW_OK);

  ParallelWkbLoader<point_t, index_t> loader;

  loader.Init();

  loader.Parse(cuda_stream, array.get(), 0, array->length);
  auto geometries = loader.Finish(cuda_stream);
  const auto& offsets = geometries.get_offsets();

  ASSERT_EQ(geometries.get_geometry_type(), GeometryType::kGeometryCollection);

  auto points = TestUtils::ToVector(cuda_stream, geometries.get_points());
  auto feature_types =
      TestUtils::ToVector(cuda_stream, offsets.geom_collection_offsets.feature_types);
  auto ps_num_geoms =
      TestUtils::ToVector(cuda_stream, offsets.geom_collection_offsets.ps_num_geoms);
  auto ps_num_parts =
      TestUtils::ToVector(cuda_stream, offsets.geom_collection_offsets.ps_num_parts);
  auto ps_num_rings =
      TestUtils::ToVector(cuda_stream, offsets.geom_collection_offsets.ps_num_rings);
  auto ps_num_points =
      TestUtils::ToVector(cuda_stream, offsets.geom_collection_offsets.ps_num_points);
  auto mbrs = TestUtils::ToVector(cuda_stream, geometries.get_mbrs());
  cuda_stream.synchronize();

  ASSERT_EQ(ps_num_geoms.size(), 10);

  ArrayView<GeometryType> v_feature_types(feature_types);
  ArrayView<index_t> v_ps_num_geoms(ps_num_geoms);
  ArrayView<index_t> v_ps_num_parts(ps_num_parts);
  ArrayView<index_t> v_ps_num_rings(ps_num_rings);
  ArrayView<index_t> v_ps_num_points(ps_num_points);
  ArrayView<point_t> v_points(points);
  ArrayView<Box<Point<float, point_t::n_dim>>> v_mbrs(mbrs.data(), mbrs.size());

  GeometryCollectionArrayView<point_t, index_t> geom_collection_array(
      v_feature_types, v_ps_num_geoms, v_ps_num_parts, v_ps_num_rings, v_ps_num_points,
      v_points, v_mbrs);
  ASSERT_EQ(geom_collection_array[0].num_geometries(), 1);
  ASSERT_EQ(geom_collection_array[0].get_type(0), GeometryType::kPoint);
  ASSERT_EQ(geom_collection_array[0].get_point(0), point_t(30, 10));

  ASSERT_EQ(geom_collection_array[1].num_geometries(), 1);
  ASSERT_EQ(geom_collection_array[1].get_type(0), GeometryType::kPoint);

  ASSERT_EQ(geom_collection_array[2].num_geometries(), 1);
  ASSERT_EQ(geom_collection_array[2].get_type(0), GeometryType::kLineString);
  ASSERT_EQ(geom_collection_array[2].get_line_string(0).num_points(), 3);

  ASSERT_EQ(geom_collection_array[3].num_geometries(), 1);
  ASSERT_EQ(geom_collection_array[3].get_type(0), GeometryType::kLineString);
  ASSERT_TRUE(geom_collection_array[3].get_line_string(0).empty());

  ASSERT_EQ(geom_collection_array[4].num_geometries(), 1);
  ASSERT_EQ(geom_collection_array[4].get_type(0), GeometryType::kPolygon);
  ASSERT_EQ(geom_collection_array[4].get_polygon(0).num_rings(), 2);
  ASSERT_EQ(geom_collection_array[4].get_polygon(0).get_ring(0).num_points(), 5);
  ASSERT_EQ(geom_collection_array[4].get_polygon(0).get_ring(1).num_points(), 4);

  ASSERT_EQ(geom_collection_array[5].num_geometries(), 1);
  ASSERT_EQ(geom_collection_array[5].get_type(0), GeometryType::kPolygon);
  ASSERT_TRUE(geom_collection_array[5].get_polygon(0).empty());

  ASSERT_EQ(geom_collection_array[6].num_geometries(), 1);
  ASSERT_EQ(geom_collection_array[6].get_type(0), GeometryType::kMultiPoint);
  ASSERT_EQ(geom_collection_array[6].get_multi_point(0).num_points(), 4);

  ASSERT_EQ(geom_collection_array[7].num_geometries(), 1);
  ASSERT_EQ(geom_collection_array[7].get_type(0), GeometryType::kMultiLineString);
  ASSERT_EQ(geom_collection_array[7].get_multi_linestring(0).num_line_strings(), 2);
  ASSERT_EQ(
      geom_collection_array[7].get_multi_linestring(0).get_line_string(0).num_points(),
      3);
  ASSERT_EQ(
      geom_collection_array[7].get_multi_linestring(0).get_line_string(1).num_points(),
      4);

  ASSERT_EQ(geom_collection_array[8].num_geometries(), 1);
  ASSERT_EQ(geom_collection_array[8].get_type(0), GeometryType::kMultiPolygon);
  ASSERT_EQ(geom_collection_array[8].get_multi_polygon(0).num_polygons(), 2);
  ASSERT_EQ(geom_collection_array[8].get_multi_polygon(0).get_polygon(0).num_rings(), 1);
  ASSERT_EQ(geom_collection_array[8].get_multi_polygon(0).get_polygon(1).num_rings(), 1);
}

TYPED_TEST(WKBLoaderTest, GeomCollection) {
  using point_t = typename TypeParam::first_type;
  using index_t = typename TypeParam::second_type;
  nanoarrow::UniqueArrayStream stream;

  ArrayStreamFromWKT(
      {{"GEOMETRYCOLLECTION ( POINT (10 10), LINESTRING (20 20, 30 30, 40 20), GEOMETRYCOLLECTION ( POLYGON ((50 50, 60 50, 60 60, 50 60, 50 50)), MULTIPOINT (70 70, 80 80) ) )",
        "MULTIPOLYGON(((30 20, 45 40, 10 40, 30 20)), ((15 5, 40 10, 10 30, 15 5), (20 15, 35 15, 35 25, 20 25, 20 15)))"}},
      GEOARROW_TYPE_WKB, stream.get());
  nanoarrow::UniqueArray array;
  ArrowError error;
  ArrowErrorSet(&error, "Failed to get next array from stream");

  rmm::cuda_stream cuda_stream;

  ASSERT_EQ(ArrowArrayStreamGetNext(stream.get(), array.get(), &error), NANOARROW_OK);

  ParallelWkbLoader<point_t, index_t> loader;
  typename ParallelWkbLoader<point_t, index_t>::Config config;

  loader.Init(config);

  loader.Parse(cuda_stream, array.get(), 0, array->length);
  auto geometries = loader.Finish(cuda_stream);

  const auto& offsets = geometries.get_offsets();

  ASSERT_EQ(geometries.get_geometry_type(), GeometryType::kGeometryCollection);

  auto points = TestUtils::ToVector(cuda_stream, geometries.get_points());
  auto feature_types =
      TestUtils::ToVector(cuda_stream, offsets.geom_collection_offsets.feature_types);
  auto ps_num_geoms =
      TestUtils::ToVector(cuda_stream, offsets.geom_collection_offsets.ps_num_geoms);
  auto ps_num_parts =
      TestUtils::ToVector(cuda_stream, offsets.geom_collection_offsets.ps_num_parts);
  auto ps_num_rings =
      TestUtils::ToVector(cuda_stream, offsets.geom_collection_offsets.ps_num_rings);
  auto ps_num_points =
      TestUtils::ToVector(cuda_stream, offsets.geom_collection_offsets.ps_num_points);
  auto mbrs = TestUtils::ToVector(cuda_stream, geometries.get_mbrs());
  cuda_stream.synchronize();
  ASSERT_EQ(ps_num_geoms.size(), 3);

  ArrayView<GeometryType> v_feature_types(feature_types);
  ArrayView<index_t> v_ps_num_geoms(ps_num_geoms);
  ArrayView<index_t> v_ps_num_parts(ps_num_parts);
  ArrayView<index_t> v_ps_num_rings(ps_num_rings);
  ArrayView<index_t> v_ps_num_points(ps_num_points);
  ArrayView<point_t> v_points(points);
  ArrayView<Box<Point<float, point_t::n_dim>>> v_mbrs(mbrs.data(), mbrs.size());

  GeometryCollectionArrayView<point_t, index_t> geom_collection_array(
      v_feature_types, v_ps_num_geoms, v_ps_num_parts, v_ps_num_rings, v_ps_num_points,
      v_points, v_mbrs);

  ASSERT_EQ(geom_collection_array[0].num_geometries(), 4);
  ASSERT_EQ(geom_collection_array[0].get_type(0), GeometryType::kPoint);
  ASSERT_EQ(geom_collection_array[0].get_point(0), point_t(10, 10));
  ASSERT_EQ(geom_collection_array[0].get_type(1), GeometryType::kLineString);
  ASSERT_EQ(geom_collection_array[0].get_line_string(1).num_points(), 3);
  ASSERT_EQ(geom_collection_array[0].get_type(2), GeometryType::kPolygon);
  ASSERT_EQ(geom_collection_array[0].get_polygon(2).num_rings(), 1);
  ASSERT_EQ(geom_collection_array[0].get_type(3), GeometryType::kMultiPoint);
  ASSERT_EQ(geom_collection_array[0].get_multi_point(3).num_points(), 2);
  ASSERT_EQ(geom_collection_array[1].num_geometries(), 1);
  ASSERT_EQ(geom_collection_array[1].get_multi_polygon(0).num_polygons(), 2);
  ASSERT_EQ(geom_collection_array[1].get_multi_polygon(0).get_polygon(0).num_rings(), 1);
  ASSERT_EQ(geom_collection_array[1].get_multi_polygon(0).get_polygon(1).num_rings(), 2);
}

}  // namespace gpuspatial
