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
  expect_snapshot(print(jb1))

  jb2 <- sd_join_by(x$id == y$id, x$value > y$threshold)
  expect_snapshot(print(jb2))
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
  expect_s3_class(ctx$table_refs$x, "sedonadb_table_ref")
  expect_s3_class(ctx$table_refs$y, "sedonadb_table_ref")

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
  x_id <- ctx$table_refs$x$id
  y_id <- ctx$table_refs$y$id

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

test_that("sd_eval_join_conditions() evaluates symbols as equality conditions", {
  x_schema <- nanoarrow::na_struct(list(id = nanoarrow::na_int32()))
  y_schema <- nanoarrow::na_struct(list(id = nanoarrow::na_int32()))
  ctx <- sd_join_expr_ctx(x_schema, y_schema)
  expect_snapshot(print(sd_eval_join_conditions(sd_join_by(id), ctx)))
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

test_that("table ref accessor validates column names", {
  x_schema <- nanoarrow::na_struct(list(id = nanoarrow::na_int32()))
  y_schema <- nanoarrow::na_struct(list(id = nanoarrow::na_int32()))

  ctx <- sd_join_expr_ctx(x_schema, y_schema)

  expect_error(
    get_from_table_ref(ctx$table_refs$x, "nonexistent"),
    "Column 'nonexistent' not found in table 'x'"
  )
  expect_error(
    get_from_table_ref(ctx$table_refs$y, "nonexistent"),
    "Column 'nonexistent' not found in table 'y'"
  )
})

test_that("sd_build_join_conditions() works with sd_join_by()", {
  x_schema <- nanoarrow::na_struct(list(
    id = nanoarrow::na_int32(),
    x_val = nanoarrow::na_string()
  ))
  y_schema <- nanoarrow::na_struct(list(
    id = nanoarrow::na_int32(),
    y_val = nanoarrow::na_int32()
  ))
  ctx <- sd_join_expr_ctx(x_schema, y_schema)

  conditions <- sd_build_join_conditions(
    ctx,
    sd_join_by(x$id == y$id)
  )

  expect_length(conditions, 1)
  expect_s3_class(conditions[[1]], "SedonaDBExpr")
})

test_that("sd_build_join_conditions() creates natural join when by is NULL", {
  x_schema <- nanoarrow::na_struct(list(
    id = nanoarrow::na_int32(),
    x_val = nanoarrow::na_string()
  ))
  y_schema <- nanoarrow::na_struct(list(
    id = nanoarrow::na_int32(),
    y_val = nanoarrow::na_int32()
  ))
  ctx <- sd_join_expr_ctx(x_schema, y_schema)

  expect_snapshot(print(sd_build_join_conditions(ctx)))
})

test_that("sd_build_join_conditions() creates equijoin conditions for names", {
  x_schema <- nanoarrow::na_struct(list(
    id = nanoarrow::na_int32(),
    x_val = nanoarrow::na_string()
  ))
  y_schema <- nanoarrow::na_struct(list(
    id = nanoarrow::na_int32(),
    y_val = nanoarrow::na_int32()
  ))
  ctx <- sd_join_expr_ctx(x_schema, y_schema)

  expect_snapshot(sd_build_join_conditions(ctx, c("id", "x_val" = "y_val")))
})

test_that("sd_join_select_default() creates default select spec", {
  spec <- sd_join_select_default()
  expect_s3_class(spec, "sedonadb_join_select_default")
  expect_equal(spec$suffix, c(".x", ".y"))

  # Custom suffix
  spec2 <- sd_join_select_default(suffix = c("_left", "_right"))
  expect_equal(spec2$suffix, c("_left", "_right"))
})

test_that("sd_join_select_default() validates suffix argument", {
  expect_error(sd_join_select_default(suffix = ".x"), "length 2")
  expect_error(sd_join_select_default(suffix = c(1, 2)), "character")
})

test_that("sd_join_select_default() prints nicely", {
  spec <- sd_join_select_default()
  expect_snapshot(print(spec))

  spec2 <- sd_join_select_default(suffix = c("_l", "_r"))
  expect_snapshot(print(spec2))
})

test_that("sd_join_select() captures column selections", {
  sel <- sd_join_select(x$id, y$value)
  expect_s3_class(sel, "sedonadb_join_select")
  expect_length(sel$exprs, 2)

  # With renaming
  sel2 <- sd_join_select(out_id = x$id, out_val = y$value)
  expect_length(sel2$exprs, 2)
  expect_equal(names(sel2$exprs), c("out_id", "out_val"))
})

test_that("sd_join_select() prints nicely", {
  sel1 <- sd_join_select(x$id, y$value)
  expect_snapshot(print(sel1))

  sel2 <- sd_join_select(out = x$id, val = y$value)
  expect_snapshot(print(sel2))
})

test_that("sd_eval_join_select_exprs() evaluates column references", {
  x_schema <- nanoarrow::na_struct(list(
    id = nanoarrow::na_int32(),
    name = nanoarrow::na_string()
  ))
  y_schema <- nanoarrow::na_struct(list(
    id = nanoarrow::na_int32(),
    value = nanoarrow::na_double()
  ))
  ctx <- sd_join_expr_ctx(x_schema, y_schema)

  expect_snapshot(sd_eval_join_select_exprs(sd_join_select(x$id, y$value, name), ctx))
})

test_that("sd_eval_join_select_exprs() handles renaming", {
  x_schema <- nanoarrow::na_struct(list(
    id = nanoarrow::na_int32(),
    name = nanoarrow::na_string()
  ))
  y_schema <- nanoarrow::na_struct(list(
    id = nanoarrow::na_int32(),
    value = nanoarrow::na_double()
  ))
  ctx <- sd_join_expr_ctx(x_schema, y_schema)

  expect_snapshot(sd_eval_join_select_exprs(
    sd_join_select(my_id = x$id, my_val = y$value, x_name = name),
    ctx
  ))
})

test_that("sd_eval_join_select_exprs() errors on ambiguous columns", {
  x_schema <- nanoarrow::na_struct(list(id = nanoarrow::na_int32()))
  y_schema <- nanoarrow::na_struct(list(id = nanoarrow::na_int32()))
  ctx <- sd_join_expr_ctx(x_schema, y_schema)

  sel <- sd_join_select(id) # Ambiguous without qualifier
  expect_error(
    sd_eval_join_select_exprs(sel, ctx),
    "Column 'id' is ambiguous"
  )
})

test_that("sd_eval_join_select_exprs() errors on missing columns", {
  x_schema <- nanoarrow::na_struct(list(id = nanoarrow::na_int32()))
  y_schema <- nanoarrow::na_struct(list(id = nanoarrow::na_int32()))
  ctx <- sd_join_expr_ctx(x_schema, y_schema)

  sel <- sd_join_select(x$nonexistent)
  expect_error(
    sd_eval_join_select_exprs(sel, ctx),
    "Column 'nonexistent' not found"
  )
})

test_that("sd_build_default_select() handles equijoin keys", {
  x_schema <- nanoarrow::na_struct(list(
    id_x = nanoarrow::na_int32(),
    x_key = nanoarrow::na_string()
  ))
  y_schema <- nanoarrow::na_struct(list(
    id_y = nanoarrow::na_int32(),
    y_key = nanoarrow::na_double()
  ))
  ctx <- sd_join_expr_ctx(x_schema, y_schema)

  # Build equijoin condition
  conditions <- sd_build_join_conditions(ctx, by = c("x_key" = "y_key"))

  # For an inner or left join, the y join key should be removed
  expect_snapshot(sd_build_default_select(ctx, conditions, join_type = "inner"))
  expect_snapshot(sd_build_default_select(ctx, conditions, join_type = "left"))

  # For a right join, the x join key should be removed
  expect_snapshot(sd_build_default_select(ctx, conditions, join_type = "right"))

  # For filtering joins, we don't need any extra projecting by default so they
  # return NULL
  expect_null(sd_build_default_select(ctx, conditions, join_type = "leftsemi"))
  expect_null(sd_build_default_select(ctx, conditions, join_type = "leftanti"))
  expect_null(sd_build_default_select(ctx, conditions, join_type = "rightsemi"))
  expect_null(sd_build_default_select(ctx, conditions, join_type = "rightanti"))

  # For full joins we coalesce the input keys
  expect_snapshot(sd_build_default_select(ctx, conditions, join_type = "full"))
})

test_that("sd_extract_equijoin_keys() extracts simple equality keys", {
  x_schema <- nanoarrow::na_struct(list(
    id_x = nanoarrow::na_int32(),
    date = nanoarrow::na_date32()
  ))
  y_schema <- nanoarrow::na_struct(list(
    id_y = nanoarrow::na_int32(),
    start = nanoarrow::na_date32()
  ))
  ctx <- sd_join_expr_ctx(x_schema, y_schema)

  # Multiple conditions: one equality, one inequality
  jb <- sd_join_by(x$id_x == y$id_y, x$date >= y$start)
  conditions <- sd_eval_join_conditions(jb, ctx)

  keys <- sd_extract_equijoin_keys(conditions)

  # Only the equality condition should be extracted
  expect_equal(keys$x_cols, "id_x")
  expect_equal(keys$y_cols, "id_y")
})
