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

test_that("sd_join_by() captures join conditions", {
  jb <- sd_join_by(x$id == y$id)
  expect_s3_class(jb, "sedonadb_join_by")
  expect_length(jb$exprs, 1)

  # Multiple conditions
  jb2 <- sd_join_by(x$id == y$id, x$date >= y$start_date)
  expect_length(jb2$exprs, 2)
})

test_that("sd_join_by() prints nicely", {
  jb1 <- sd_join_by(x$id == y$id)
  output1 <- capture.output(print(jb1))
  expect_true(any(grepl("<sedonadb_join_by>", output1)))
  expect_true(any(grepl("x\\$id == y\\$id", output1)))

  jb2 <- sd_join_by(x$id == y$id, x$value > y$threshold)
  output2 <- capture.output(print(jb2))
  expect_true(any(grepl("x\\$id == y\\$id", output2)))
  expect_true(any(grepl("x\\$value > y\\$threshold", output2)))
})

test_that("sd_join_by() requires at least one condition", {
  expect_error(sd_join_by(), "requires at least one join condition")
})

test_that("sd_join_expr_ctx() creates qualified column references", {
  x_schema <- nanoarrow::na_struct(list(
    id = nanoarrow::na_int32(),
    x_only = nanoarrow::na_string()
  ))
  y_schema <- nanoarrow::na_struct(list(
    id = nanoarrow::na_int32(),
    y_only = nanoarrow::na_string()
  ))

  ctx <- sd_join_expr_ctx(x_schema, y_schema)

  # Check that x and y table refs are stored in the context
  expect_s3_class(ctx$x_ref, "sedonadb_table_ref")
  expect_s3_class(ctx$y_ref, "sedonadb_table_ref")

  # Check that ambiguous columns are tracked
  expect_equal(ctx$ambiguous_columns, "id")
})

test_that("qualified column references produce correct expressions", {
  x_schema <- nanoarrow::na_struct(list(
    id = nanoarrow::na_int32(),
    name = nanoarrow::na_string()
  ))
  y_schema <- nanoarrow::na_struct(list(
    id = nanoarrow::na_int32(),
    value = nanoarrow::na_double()
  ))

  ctx <- sd_join_expr_ctx(x_schema, y_schema)

  # Access qualified columns via table ref stored in context
  x_id <- ctx$x_ref$id
  y_id <- ctx$y_ref$id

  expect_s3_class(x_id, "SedonaDBExpr")
  expect_s3_class(y_id, "SedonaDBExpr")

  # Check that the column expressions include qualifiers
  expect_snapshot(x_id)
  expect_snapshot(y_id)
})

test_that("unambiguous columns can be referenced without qualifier", {
  x_schema <- nanoarrow::na_struct(list(
    id = nanoarrow::na_int32(),
    x_only = nanoarrow::na_string()
  ))
  y_schema <- nanoarrow::na_struct(list(
    id = nanoarrow::na_int32(),
    y_only = nanoarrow::na_string()
  ))

  ctx <- sd_join_expr_ctx(x_schema, y_schema)

  # x_only is only in x, y_only is only in y - should be accessible without qualifier
  x_only_expr <- rlang::eval_tidy(quote(x_only), data = ctx$data)
  y_only_expr <- rlang::eval_tidy(quote(y_only), data = ctx$data)

  expect_s3_class(x_only_expr, "SedonaDBExpr")
  expect_s3_class(y_only_expr, "SedonaDBExpr")

  # These should have the appropriate qualifiers
  expect_match(x_only_expr$display(), "x.x_only")
  expect_match(y_only_expr$display(), "y.y_only")

  # x and y table refs should also be accessible
  x_ref <- rlang::eval_tidy(quote(x), data = ctx$data)
  y_ref <- rlang::eval_tidy(quote(y), data = ctx$data)

  expect_s3_class(x_ref, "sedonadb_table_ref")
  expect_s3_class(y_ref, "sedonadb_table_ref")
})

test_that("sd_eval_join_conditions() evaluates equality conditions", {
  x_schema <- nanoarrow::na_struct(list(id = nanoarrow::na_int32()))
  y_schema <- nanoarrow::na_struct(list(id = nanoarrow::na_int32()))

  ctx <- sd_join_expr_ctx(x_schema, y_schema)
  jb <- sd_join_by(x$id == y$id)

  conditions <- sd_eval_join_conditions(jb, ctx)

  expect_length(conditions, 1)
  expect_s3_class(conditions[[1]], "SedonaDBExpr")
  expect_snapshot(conditions[[1]])
})

test_that("sd_eval_join_conditions() evaluates inequality conditions", {
  x_schema <- nanoarrow::na_struct(list(value = nanoarrow::na_double()))
  y_schema <- nanoarrow::na_struct(list(threshold = nanoarrow::na_double()))

  ctx <- sd_join_expr_ctx(x_schema, y_schema)
  jb <- sd_join_by(x$value > y$threshold)

  conditions <- sd_eval_join_conditions(jb, ctx)

  expect_length(conditions, 1)
  expect_snapshot(conditions[[1]])
})

test_that("sd_eval_join_conditions() evaluates multiple conditions", {
  x_schema <- nanoarrow::na_struct(list(
    id = nanoarrow::na_int32(),
    date = nanoarrow::na_date32()
  ))
  y_schema <- nanoarrow::na_struct(list(
    id = nanoarrow::na_int32(),
    start_date = nanoarrow::na_date32()
  ))

  ctx <- sd_join_expr_ctx(x_schema, y_schema)
  jb <- sd_join_by(x$id == y$id, x$date >= y$start_date)

  conditions <- sd_eval_join_conditions(jb, ctx)

  expect_length(conditions, 2)
  expect_snapshot(conditions[[1]])
  expect_snapshot(conditions[[2]])
})

test_that("ambiguous column references produce helpful errors", {
  x_schema <- nanoarrow::na_struct(list(id = nanoarrow::na_int32()))
  y_schema <- nanoarrow::na_struct(list(id = nanoarrow::na_int32()))

  ctx <- sd_join_expr_ctx(x_schema, y_schema)

  # Trying to use 'id' without qualifier should error
  jb <- sd_join_by(id == y$id)

  expect_error(
    sd_eval_join_conditions(jb, ctx),
    "Column 'id' is ambiguous"
  )
})

test_that("missing column references produce helpful errors", {
  x_schema <- nanoarrow::na_struct(list(id = nanoarrow::na_int32()))
  y_schema <- nanoarrow::na_struct(list(id = nanoarrow::na_int32()))

  ctx <- sd_join_expr_ctx(x_schema, y_schema)

  # Column that doesn't exist
  jb <- sd_join_by(x$nonexistent == y$id)

  expect_error(
    sd_eval_join_conditions(jb, ctx),
    "Column 'nonexistent' not found in table 'x'"
  )
})

test_that("table ref $ accessor validates column names", {
  x_schema <- nanoarrow::na_struct(list(id = nanoarrow::na_int32()))
  y_schema <- nanoarrow::na_struct(list(id = nanoarrow::na_int32()))

  ctx <- sd_join_expr_ctx(x_schema, y_schema)

  expect_error(ctx$x_ref$nonexistent, "Column 'nonexistent' not found in table 'x'")
  expect_error(ctx$y_ref$nonexistent, "Column 'nonexistent' not found in table 'y'")
})

test_that("sd_join() creates join plan with sd_join_by()", {
  x <- data.frame(id = 1:3, x_val = c("a", "b", "c")) |> as_sedonadb_dataframe()
  y <- data.frame(id = 2:4, y_val = c(10, 20, 30)) |> as_sedonadb_dataframe()

  plan <- sd_join(x, y, sd_join_by(x$id == y$id))

  expect_s3_class(plan, "sedonadb_join_plan")
  expect_equal(plan$how, "inner")
  expect_length(plan$conditions, 1)
})

test_that("sd_join() creates natural join when by is NULL", {
  x <- data.frame(id = 1:3, x_val = c("a", "b", "c")) |> as_sedonadb_dataframe()
  y <- data.frame(id = 2:4, y_val = c(10, 20, 30)) |> as_sedonadb_dataframe()

  plan <- sd_join(x, y)

  expect_s3_class(plan, "sedonadb_join_plan")
  expect_length(plan$conditions, 1)  # Natural join on 'id'
})

test_that("sd_join() validates join type", {
  x <- data.frame(id = 1:3) |> as_sedonadb_dataframe()
  y <- data.frame(id = 2:4) |> as_sedonadb_dataframe()

  expect_s3_class(sd_join(x, y, how = "left"), "sedonadb_join_plan")
  expect_s3_class(sd_join(x, y, how = "right"), "sedonadb_join_plan")
  expect_s3_class(sd_join(x, y, how = "full"), "sedonadb_join_plan")
  expect_error(sd_join(x, y, how = "invalid"))
})

test_that("sd_join_plan prints nicely", {
  x <- data.frame(id = 1:3, x_val = c("a", "b", "c")) |> as_sedonadb_dataframe()
  y <- data.frame(id = 2:4, y_val = c(10, 20, 30)) |> as_sedonadb_dataframe()

  plan <- sd_join(x, y, sd_join_by(x$id == y$id))

  output <- capture.output(print(plan))
  expect_true(any(grepl("<sedonadb_join_plan>", output)))
  expect_true(any(grepl("inner join", output)))
  expect_true(any(grepl("x.id = y.id", output)))
})
