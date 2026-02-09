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
#include "test_common.hpp"

#include <gtest/gtest.h>

#include "array_stream.hpp"

#include "geoarrow/geoarrow.hpp"
#include "nanoarrow/nanoarrow.hpp"

using BoxXY = geoarrow::array_util::BoxXY<double>;

namespace gpuspatial {

TEST(ArrayStream, StreamFromWkt) {
  nanoarrow::UniqueArrayStream stream;
  ArrayStreamFromWKT(
      {{"POINT (0 1)", "POINT (2 3)", "POINT (4 5)"}, {"POINT (6 7)", "POINT (8 9)"}},
      GEOARROW_TYPE_WKB, stream.get());

  struct ArrowError error{};
  nanoarrow::UniqueArray array;
  int64_t n_batches = 0;
  int64_t n_rows = 0;
  testing::WKBBounder bounder;
  while (true) {
    ASSERT_EQ(ArrowArrayStreamGetNext(stream.get(), array.get(), &error), NANOARROW_OK)
        << error.message;
    if (array->release == nullptr) {
      break;
    }

    n_batches += 1;
    n_rows += array->length;
    bounder.Read(array.get());
    array.reset();
  }

  ASSERT_EQ(n_batches, 2);
  ASSERT_EQ(n_rows, 5);

  EXPECT_EQ(bounder.Bounds().xmin(), 0);
  EXPECT_EQ(bounder.Bounds().ymin(), 1);
  EXPECT_EQ(bounder.Bounds().xmax(), 8);
  EXPECT_EQ(bounder.Bounds().ymax(), 9);
}

}  // namespace gpuspatial
