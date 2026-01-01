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

test_that("basic expression types can be constructed", {
  expect_snapshot(sd_expr_column("foofy"))
  expect_snapshot(sd_expr_literal(1L))
  expect_snapshot(sd_expr_scalar_function("abs", list(1L)))
  expect_snapshot(sd_expr_cast(1L, nanoarrow::na_int64()))
  expect_snapshot(sd_expr_alias(1L, "foofy"))
})

test_that("casts to a type with extension metadata can't be constructed", {
  expect_error(
    sd_expr_cast(1L, geoarrow::geoarrow_wkb()),
    "Can't cast to Arrow extension type 'geoarrow.wkb'"
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

test_that("function calls with a translation become function calls", {
  expect_snapshot(sd_eval_expr(quote(abs(-1L))))
})

test_that("function calls without a translation are evaluated in R", {
  function_without_a_translation <- function(x) x + 1L
  expect_snapshot(sd_eval_expr(quote(function_without_a_translation(1L))))
})
