# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

test_that("basic literals can be converted to expressions", {
  expect_identical(
    as_sedonadb_literal(NULL)$debug_string(),
    "Literal(NULL, None)"
  )

  expect_identical(
    as_sedonadb_literal("foofy")$debug_string(),
    'Literal(Utf8("foofy"), None)'
  )

  expect_identical(
    as_sedonadb_literal(1L)$debug_string(),
    "Literal(Int32(1), None)"
  )

  expect_identical(
    as_sedonadb_literal(1.0)$debug_string(),
    "Literal(Float64(1), None)"
  )

  expect_identical(
    as_sedonadb_literal(as.raw(c(1:3)))$debug_string(),
    'Literal(Binary("1,2,3"), None)'
  )
})

test_that("literals can request a type", {
  expect_identical(
    as_sedonadb_literal(1.0, type = nanoarrow::na_float())$debug_string(),
    "Cast(Cast { expr: Literal(Float64(1), None), data_type: Float32 })"
  )
})

test_that("literals with Arrow extension metadata can be converted to literals", {
  expect_snapshot(as_sedonadb_literal(wk::as_wkb("POINT (0 1)")))
})

test_that("non-scalars can't be automatically converted to literals", {
  expect_error(
    as_sedonadb_literal(1:5)$debug_string(),
    "Can't convert non-scalar to sedonadb_expr"
  )
})
