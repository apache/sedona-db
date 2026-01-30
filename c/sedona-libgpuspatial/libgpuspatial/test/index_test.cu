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
#include "gpuspatial/index/rt_spatial_index.cuh"
#include "test_common.hpp"

#include <geos/geom/Envelope.h>
#include <geos/index/ItemVisitor.h>
#include <geos/index/strtree/STRtree.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <numeric>  // For std::iota
#include <random>
#include <vector>

namespace gpuspatial {
template <typename T>
struct SpatialIndexTest : public ::testing::Test {
  using index_t = RTSpatialIndex<typename T::scalar_t, T::n_dim>;
  std::shared_ptr<RTEngine> rt_engine;
  index_t index;

  SpatialIndexTest() {
    auto ptx_root = TestUtils::GetTestShaderPath();

    rt_engine = std::make_shared<RTEngine>();
    rt_engine->Init(get_default_rt_config(ptx_root));
    RTSpatialIndexConfig config;
    config.rt_engine = rt_engine;
    index = std::move(index_t(config));
  }
};
using PointTypes = ::testing::Types<Point<float, 2>, Point<double, 2>>;
TYPED_TEST_SUITE(SpatialIndexTest, PointTypes);

template <typename POINT_T>
std::vector<Box<POINT_T>> GeneratePoints(size_t n, std::mt19937& rng) {
  using scalar_t = typename POINT_T::scalar_t;
  std::vector<Box<POINT_T>> rects(n);

  for (size_t i = 0; i < n; i++) {
    POINT_T p;
    for (int dim = 0; dim < POINT_T::n_dim; dim++) {
      std::uniform_real_distribution<scalar_t> dist(-180.0, 180.0);
      p.set_coordinate(dim, dist(rng));
    }
    rects[i] = Box<POINT_T>(p, p);
  }
  return rects;
}

template <typename POINT_T>
std::vector<Box<POINT_T>> GenerateRects(size_t n, std::mt19937& rng) {
  using scalar_t = typename POINT_T::scalar_t;
  std::vector<Box<POINT_T>> rects(n);
  std::uniform_real_distribution<scalar_t> distSize(0.0, 100);

  for (size_t i = 0; i < n; ++i) {
    POINT_T min_pt, max_pt, size_pt;

    for (int dim = 0; dim < POINT_T::n_dim; dim++) {
      std::uniform_real_distribution<scalar_t> dist(-180.0, 180.0);
      min_pt.set_coordinate(dim, dist(rng));
      size_pt.set_coordinate(dim, distSize(rng));
    }
    max_pt = min_pt + size_pt;
    rects[i] = Box<POINT_T>(min_pt, max_pt);
  }
  return rects;
}

template <typename POINT_T>
void ComputeReference(const std::vector<Box<POINT_T>>& build,
                      const std::vector<Box<POINT_T>>& probe,
                      std::vector<uint32_t>& build_indices,
                      std::vector<uint32_t>& probe_indices) {
  geos::index::strtree::STRtree tree;

  // FIX: Create a storage container for envelopes that persists
  // for the lifetime of the tree usage.
  std::vector<geos::geom::Envelope> build_envelopes;
  build_envelopes.reserve(build.size());

  // 2. Build Phase
  for (uint32_t j = 0; j < build.size(); j++) {
    auto min_corner = build[j].get_min();
    auto max_corner = build[j].get_max();

    // Emplace the envelope into our persistent vector
    build_envelopes.emplace_back(min_corner.x(), max_corner.x(), min_corner.y(),
                                 max_corner.y());

    // Pass the address of the element inside the vector
    // Note: We reserved memory above, so pointers shouldn't be invalidated by resizing
    tree.insert(&build_envelopes.back(),
                reinterpret_cast<void*>(static_cast<uintptr_t>(j)));
  }

  tree.build();

  // 3. Define Visitor (No changes needed here)
  class InteractionVisitor : public geos::index::ItemVisitor {
   public:
    const std::vector<Box<POINT_T>>* build;
    const std::vector<Box<POINT_T>>* probe;
    std::vector<uint32_t>* b_indices;
    std::vector<uint32_t>* p_indices;
    uint32_t current_probe_idx;

    void visitItem(void* item) override {
      uintptr_t build_idx_ptr = reinterpret_cast<uintptr_t>(item);
      uint32_t build_idx = static_cast<uint32_t>(build_idx_ptr);

      // Refinement step
      if ((*build)[build_idx].intersects((*probe)[current_probe_idx])) {
        b_indices->push_back(build_idx);
        p_indices->push_back(current_probe_idx);
      }
    }
  };

  InteractionVisitor visitor;
  visitor.build = &build;
  visitor.probe = &probe;
  visitor.b_indices = &build_indices;
  visitor.p_indices = &probe_indices;

  // 4. Probe Phase
  for (uint32_t i = 0; i < probe.size(); i++) {
    auto min_corner = probe[i].get_min();
    auto max_corner = probe[i].get_max();

    // It is safe to create this on the stack here because `query`
    // finishes executing before `search_env` goes out of scope.
    geos::geom::Envelope search_env(min_corner.x(), max_corner.x(), min_corner.y(),
                                    max_corner.y());

    visitor.current_probe_idx = i;
    tree.query(&search_env, visitor);
  }
}

template <typename T, typename U>
void sort_vectors(std::vector<T>& v1, std::vector<U>& v2) {
  if (v1.size() != v2.size()) return;

  // 1. Create indices [0, 1, 2, ..., N-1]
  std::vector<size_t> p(v1.size());
  std::iota(p.begin(), p.end(), 0);

  // 2. Sort indices based on comparing values in v1 and v2
  std::sort(p.begin(), p.end(), [&](size_t i, size_t j) {
    if (v1[i] != v1[j]) return v1[i] < v1[j];  // Primary sort by v1
    return v2[i] < v2[j];                      // Secondary sort by v2
  });

  // 3. Apply permutation (Reorder v1 and v2 based on sorted indices)
  // Note: Doing this in-place with O(1) space is complex;
  // using auxiliary O(N) space is standard.
  std::vector<T> sorted_v1, sorted_v2;
  sorted_v1.reserve(v1.size());
  sorted_v2.reserve(v2.size());

  for (size_t i : p) {
    sorted_v1.push_back(v1[i]);
    sorted_v2.push_back(v2[i]);
  }

  v1 = std::move(sorted_v1);
  v2 = std::move(sorted_v2);
}

TYPED_TEST(SpatialIndexTest, PointPoint) {
  using point_t = TypeParam;
  std::mt19937 gen(0);

  for (int i = 1; i <= 10000; i *= 2) {
    auto points1 = GeneratePoints<point_t>(i, gen);
    this->index.Clear();
    this->index.PushBuild(points1.data(), points1.size());
    this->index.FinishBuilding();

    for (int j = 1; j <= 10000; j *= 2) {
      auto points2 = GeneratePoints<point_t>(j, gen);

      size_t count = static_cast<size_t>(points1.size() * 0.2);

      // 2. Define the starting point (the last 'count' elements)
      auto start_it = points1.end() - count;

      // 3. Append to the second vector
      points2.insert(points2.end(), start_it, points1.end());

      std::vector<uint32_t> build_indices, probe_indices;
      this->index.Probe(points2.data(), points2.size(), &build_indices, &probe_indices);
      sort_vectors(build_indices, probe_indices);

      std::vector<uint32_t> ref_build_indices, ref_probe_indices;
      ComputeReference(points1, points2, ref_build_indices, ref_probe_indices);
      sort_vectors(ref_build_indices, ref_probe_indices);

      ASSERT_EQ(build_indices, ref_build_indices);
      ASSERT_EQ(probe_indices, ref_probe_indices);
    }
  }
}

TYPED_TEST(SpatialIndexTest, BoxPoint) {
  using point_t = TypeParam;
  std::mt19937 gen(0);

  for (int i = 1; i <= 10000; i *= 2) {
    auto rects1 = GenerateRects<point_t>(i, gen);
    this->index.Clear();
    this->index.PushBuild(rects1.data(), rects1.size());
    this->index.FinishBuilding();

    for (int j = 1; j <= 10000; j *= 2) {
      auto points2 = GeneratePoints<point_t>(j, gen);
      std::vector<uint32_t> build_indices, probe_indices;
      this->index.Probe(points2.data(), points2.size(), &build_indices, &probe_indices);
      sort_vectors(build_indices, probe_indices);

      std::vector<uint32_t> ref_build_indices, ref_probe_indices;
      ComputeReference(rects1, points2, ref_build_indices, ref_probe_indices);
      sort_vectors(ref_build_indices, ref_probe_indices);

      ASSERT_EQ(build_indices, ref_build_indices);
      ASSERT_EQ(probe_indices, ref_probe_indices);
    }
  }
}

TYPED_TEST(SpatialIndexTest, PointBox) {
  using point_t = TypeParam;
  std::mt19937 gen(0);

  for (int i = 1; i <= 10000; i *= 2) {
    auto points1 = GeneratePoints<point_t>(i, gen);
    this->index.Clear();
    this->index.PushBuild(points1.data(), points1.size());
    this->index.FinishBuilding();

    for (int j = 1; j <= 10000; j *= 2) {
      auto rects2 = GenerateRects<point_t>(j, gen);
      std::vector<uint32_t> build_indices, probe_indices;
      this->index.Probe(rects2.data(), rects2.size(), &build_indices, &probe_indices);
      sort_vectors(build_indices, probe_indices);

      std::vector<uint32_t> ref_build_indices, ref_probe_indices;
      ComputeReference(points1, rects2, ref_build_indices, ref_probe_indices);
      sort_vectors(ref_build_indices, ref_probe_indices);

      ASSERT_EQ(build_indices, ref_build_indices);
      ASSERT_EQ(probe_indices, ref_probe_indices);
    }
  }
}

TYPED_TEST(SpatialIndexTest, BoxBox) {
  using point_t = TypeParam;
  std::mt19937 gen(0);

  for (int i = 1; i <= 10000; i *= 2) {
    auto rects1 = GenerateRects<point_t>(i, gen);
    this->index.Clear();
    this->index.PushBuild(rects1.data(), rects1.size());
    this->index.FinishBuilding();

    for (int j = 1; j <= 10000; j *= 2) {
      auto rects2 = GenerateRects<point_t>(j, gen);
      std::vector<uint32_t> build_indices, probe_indices;
      this->index.Probe(rects2.data(), rects2.size(), &build_indices, &probe_indices);
      sort_vectors(build_indices, probe_indices);

      std::vector<uint32_t> ref_build_indices, ref_probe_indices;
      ComputeReference(rects1, rects2, ref_build_indices, ref_probe_indices);
      sort_vectors(ref_build_indices, ref_probe_indices);

      ASSERT_EQ(build_indices, ref_build_indices);
      ASSERT_EQ(probe_indices, ref_probe_indices);
    }
  }
}
}  // namespace gpuspatial
