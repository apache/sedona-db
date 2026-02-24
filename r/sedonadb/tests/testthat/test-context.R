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

test_that("the global context is never replaced", {
  # Check a few times to make sure this is true
  ctx <- sd_connect(global = TRUE)
  expect_true(rlang::is_reference(ctx, global_ctx$ctx))

  ctx <- sd_connect(global = TRUE)
  expect_true(rlang::is_reference(ctx, global_ctx$ctx))

  expect_snapshot_warning(
    expect_true(
      rlang::is_reference(
        sd_connect(global = TRUE, memory_limit = "5g"),
        global_ctx$ctx
      )
    )
  )
})

test_that("scoped connections can be created", {
  ctx <- sd_connect(
    memory_limit = "1g",
    temp_dir = tempfile(),
    memory_pool_type = "fair",
    unspillable_reserve_ratio = 0.5
  )

  df <- data.frame(x = 1:10)
  sd_to_view(df, "some_name", ctx = ctx, overwrite = TRUE)

  df2 <- data.frame(y = 11:20)
  sd_to_view(df2, "some_name", overwrite = TRUE)

  expect_identical(
    ctx |> sd_ctx_view("some_name") |> sd_collect(),
    df
  )

  expect_identical(
    sd_view("some_name") |> sd_collect(),
    df2
  )
})

test_that("unrecognized options result in a warning", {
  expect_snapshot_warning(sd_connect(not_an_option = "foofy"))
})

test_that("sd_read_parquet() works", {
  path <- system.file("files/natural-earth_cities_geo.parquet", package = "sedonadb")
  expect_identical(sd_count(sd_read_parquet(path)), 243)

  expect_identical(sd_count(sd_read_parquet(c(path, path))), 243 * 2)
})

test_that("views can be created and dropped", {
  df <- sd_sql("SELECT 1 as one")
  expect_true(rlang::is_reference(sd_to_view(df, "foofy"), df))
  expect_identical(
    sd_sql("SELECT * FROM foofy") |> sd_collect(),
    data.frame(one = 1)
  )

  expect_identical(
    sd_view("foofy") |> sd_collect(),
    data.frame(one = 1)
  )

  sd_drop_view("foofy")
  expect_error(sd_sql("SELECT * FROM foofy"), "table '(.*?)' not found")
  expect_error(sd_view("foofy"), "No table named 'foofy'")
})

test_that("scalar udfs can be registered", {
  udf <- ctx()$scalar_udf_xptr("st_envelope")
  expect_s3_class(udf, "datafusion_scalar_udf")

  sd_register_udf(udf)
  df <- sd_sql("SELECT ST_Envelope(ST_Point(0, 1)) as geom") |> sd_collect()
  expect_identical(
    wk::as_wkt(df$geom),
    wk::wkt("POINT (0 1)")
  )
})

test_that("configure_proj() errors for invalid inputs", {
  expect_error(
    sd_configure_proj("not a preset"),
    "Unknown preset"
  )

  expect_error(
    sd_configure_proj(database_path = "file that does not exist"),
    "Invalid database path"
  )

  expect_error(
    sd_configure_proj(search_path = "dir that does not exist"),
    "Invalid search path"
  )
})

test_that(".fns can have its contents listed", {
  expect_contains(names(.fns), "st_intersects")
  expect_contains(.DollarNames(.fns, "st_int"), "st_intersects")
})
