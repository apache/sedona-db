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

test_that("dataframe can be created from data.frame", {
  df <- as_sedonadb_dataframe(data.frame(one = 1, two = "two"))
  expect_s3_class(df, "sedonadb_dataframe")
  expect_identical(sd_collect(df), data.frame(one = 1, two = "two"))

  # Ensure that geo columns with crs are handled
  df <- as_sedonadb_dataframe(
    data.frame(
      geom = wk::as_wkb(wk::wkt("POINT (0 1)", crs = "EPSG:32620"))
    )
  )

  re_df <- sd_collect(df)
  expect_identical(
    wk::as_wkt(re_df$geom),
    wk::wkt("POINT (0 1)", crs = wk::wk_crs_projjson("EPSG:32620"))
  )
})

test_that("dataframe can be created from nanoarrow objects", {
  r_df <- data.frame(geom = wk::as_wkb("POINT (0 1)"))

  array <- nanoarrow::as_nanoarrow_array(r_df)
  df <- as_sedonadb_dataframe(array)
  expect_s3_class(df, "sedonadb_dataframe")
  expect_identical(sd_collect(df, ptype = r_df), r_df)

  stream <- nanoarrow::as_nanoarrow_array_stream(r_df)
  df <- as_sedonadb_dataframe(stream, lazy = TRUE)
  expect_s3_class(df, "sedonadb_dataframe")
  expect_identical(sd_collect(df, ptype = r_df), r_df)

  stream <- nanoarrow::as_nanoarrow_array_stream(r_df)
  df <- as_sedonadb_dataframe(stream, lazy = FALSE)
  expect_s3_class(df, "sedonadb_dataframe")
  expect_identical(sd_collect(df, ptype = r_df), r_df)
})

test_that("dataframe property accessors work", {
  df <- sd_sql("SELECT ST_Point(0, 1) as pt")
  expect_identical(ncol(df), 1L)
  expect_identical(nrow(df), NA_integer_)
  expect_identical(colnames(df), "pt")
})

test_that("dataframe head() works", {
  df <- sd_sql("SELECT 1 as one, 'two' as two")
  expect_identical(
    as.data.frame(head(df, 0)),
    data.frame(one = double(), two = character())
  )
})

test_that("dataframe rows can be counted", {
  df <- sd_sql("SELECT 1 as one, 'two' as two")
  expect_identical(sd_count(df), 1)
})

test_that("dataframe can be computed", {
  df <- sd_sql("SELECT 1 as one, 'two' as two")
  df_computed <- sd_compute(df)
  expect_identical(sd_collect(df), sd_collect(df_computed))
})

test_that("dataframe can be collected", {
  df <- sd_sql("SELECT 1 as one, 'two' as two")
  expect_identical(
    sd_collect(df),
    data.frame(one = 1, two = "two")
  )

  expect_identical(
    sd_collect(df, ptype = data.frame(one = integer(), two = character())),
    data.frame(one = 1L, two = "two")
  )
})

test_that("dataframe can be converted to an R data.frame", {
  df <- sd_sql("SELECT 1 as one, 'two' as two")
  expect_identical(
    as.data.frame(df),
    data.frame(one = 1, two = "two")
  )
})

test_that("dataframe can be converted to an array stream", {
  df <- sd_sql("SELECT 1 as one, 'two' as two")
  stream <- nanoarrow::as_nanoarrow_array_stream(df)
  expect_s3_class(stream, "nanoarrow_array_stream")
  expect_identical(
    as.data.frame(stream),
    data.frame(one = 1, two = "two")
  )
})

test_that("dataframe can be printed", {
  df <- sd_sql("SELECT ST_Point(0, 1) as pt")
  expect_output(expect_identical(print(df), df), "POINT")
})

test_that("dataframe print uses ASCII when requested", {
  df <- sd_sql("SELECT ST_Point(0, 1) as pt")
  withr::with_options(list(cli.unicode = FALSE), {
    expect_output(print(df), "--+")
  })
})

test_that("dataframe print limits max output based on options", {
  df <- sd_sql("SELECT ST_Point(0, 1) as pt")
  withr::with_options(list(pillar.print_max = 0), {
    expect_output(print(df), "Preview of up to 0 row\\(s\\)")
  })
})

test_that("dataframe print limits max output based on options", {
  df <- sd_sql("SELECT 'a really really really really long string' as str")
  withr::with_options(list(width = 10, cli.unicode = FALSE), {
    expect_output(print(df), "| a r... |")
  })
})
