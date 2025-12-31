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
    as_sedonadb_expr("foofy")$debug_string(),
    'Literal(Utf8("foofy"), None)'
  )

  expect_identical(
    as_sedonadb_expr(1L)$debug_string(),
    'Literal(Int32(1), None)'
  )

  expect_identical(
    as_sedonadb_expr(1.0)$debug_string(),
    'Literal(Float64(1), None)'
  )

  expect_identical(
    as_sedonadb_expr(as.raw(c(1:3)))$debug_string(),
    'Literal(Binary("1,2,3"), None)'
  )
})

test_that("non-scalars can't be automatically converted to literals", {
  expect_error(
    as_sedonadb_expr(1:5)$debug_string(),
    "Can't convert non-scalar to sedonadb_expr"
  )
})

test_that("expressions can be printed", {
  expect_snapshot(
    print(as_sedonadb_expr("foofy"))
  )
})

test_that("literal expressions can be translated", {
  expect_snapshot(sd_eval_expr(quote(1L)))
})

test_that("column expressions can be translated", {
  schema <- nanoarrow::na_struct(list(col0 = nanoarrow::na_int32()))
  expr_ctx <- sd_expr_ctx(schema)

  expect_snapshot(sd_eval_expr(quote(col0), expr_ctx))
  expect_snapshot(sd_eval_expr(quote(.data$col0), expr_ctx))
  col_zero <- "col0"
  expect_snapshot(sd_eval_expr(quote(.data[[col_zero]]), expr_ctx))

  expect_error(
    sd_eval_expr(quote(col1), expr_ctx),
    "object 'col1' not found"
  )
})

test_that("function calls containing no SedonaDB expressions can be translated", {
  # Ensure these are evaluated in R (i.e., the resulting expression is a literal)
  expect_snapshot(sd_eval_expr(quote(abs(-1L))))
})

test_that("function calls containing SedonaDB expressions can be translated", {
  schema <- nanoarrow::na_struct(list(col0 = nanoarrow::na_int32()))
  expr_ctx <- sd_expr_ctx(schema)
  expect_snapshot(sd_eval_expr(quote(abs(col0)), expr_ctx))
})
