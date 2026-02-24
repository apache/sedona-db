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

test_that("dataframe can be created from an FFI table provider", {
  df <- as_sedonadb_dataframe(data.frame(one = 1, two = "two"))
  provider <- df$df$to_provider()
  df2 <- as_sedonadb_dataframe(provider)
  expect_identical(
    sd_collect(df2),
    data.frame(one = 1, two = "two")
  )
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

test_that("sd_with_params() fills in placeholder values", {
  df <- sd_sql("SELECT $1 + 1 AS two") |> sd_with_params(1)
  expect_identical(sd_collect(df), data.frame(two = 2))

  df <- sd_sql("SELECT $x + 1 AS two") |> sd_with_params(x = 1)
  expect_identical(sd_collect(df), data.frame(two = 2))

  # Check that an error occurs for missing parameters
  expect_snapshot_error(
    sd_sql("SELECT $x + 1 AS two") |> sd_with_params()
  )
  expect_snapshot_error(
    sd_sql("SELECT $1 + 1 AS two") |> sd_with_params()
  )

  # Check error for mixed named/unnamed
  expect_snapshot_error(
    sd_sql("SELECT $x + 1 AS two") |> sd_with_params(x = 1, 2)
  )
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

  expect_error(
    nanoarrow::as_nanoarrow_array_stream(df, schema = nanoarrow::na_int32()),
    "Requested schema is not supported"
  )
})

test_that("dataframe can be printed", {
  df <- sd_sql("SELECT ST_Point(0, 1) as pt")
  expect_snapshot(print(df))

  grouped <- df |> sd_group_by(x = .fns$st_x(pt))
  expect_snapshot(print(grouped))
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

test_that("sd_write_parquet can write simple data", {
  df <- sd_sql(
    "SELECT * FROM (VALUES ('one', 1), ('two', 2), ('three', 3)) AS t(a, b)"
  )

  # Test single file output (path ending with .parquet)
  tmp_parquet_file <- tempfile(fileext = ".parquet")
  on.exit(unlink(tmp_parquet_file))

  result <- sd_write_parquet(df, tmp_parquet_file)
  expect_identical(result, df) # Should return the input invisibly
  expect_true(file.exists(tmp_parquet_file))

  # Read back and verify contents
  df_roundtrip <- sd_read_parquet(tmp_parquet_file)
  expect_identical(
    sd_collect(df_roundtrip),
    data.frame(
      a = c("one", "two", "three"),
      b = c(1, 2, 3)
    )
  )
})

test_that("sd_write_parquet can write to directory", {
  df <- sd_sql(
    "SELECT * FROM (VALUES ('one', 1), ('two', 2), ('three', 3)) AS t(a, b)"
  )

  # Test directory output (path not ending with .parquet)
  tmp_parquet_dir <- tempfile()
  on.exit(unlink(tmp_parquet_dir, recursive = TRUE))

  sd_write_parquet(df, tmp_parquet_dir)
  expect_true(dir.exists(tmp_parquet_dir))

  # Read back and verify contents
  df_roundtrip <- sd_read_parquet(tmp_parquet_dir)
  expect_identical(
    sd_collect(df_roundtrip),
    data.frame(
      a = c("one", "two", "three"),
      b = c(1, 2, 3)
    )
  )
})

test_that("sd_write_parquet can partition data", {
  df <- sd_sql(
    "SELECT * FROM (VALUES ('one', 1), ('two', 2), ('three', 3)) AS t(a, b)"
  )

  tmp_parquet_dir <- tempfile()
  on.exit(unlink(tmp_parquet_dir, recursive = TRUE))

  sd_write_parquet(df, tmp_parquet_dir, partition_by = "a")

  # Read back and verify partitioning worked
  roundtrip_data <- sd_read_parquet(tmp_parquet_dir) |>
    sd_collect()

  # Should have the same data (order might be different due to partitioning)
  expect_setequal(roundtrip_data$b, c(1L, 2L, 3L))
})

test_that("sd_write_parquet can sort data", {
  df <- sd_sql(
    "SELECT * FROM (VALUES ('two', 2), ('one', 1), ('three', 3)) AS t(a, b)"
  )

  tmp_parquet_file <- tempfile(fileext = ".parquet")
  on.exit(unlink(tmp_parquet_file))

  sd_write_parquet(df, tmp_parquet_file, sort_by = "a")

  # Read back and verify sorting
  roundtrip_data <- sd_read_parquet(tmp_parquet_file) |>
    sd_collect()

  expect_identical(
    roundtrip_data,
    data.frame(
      a = c("one", "three", "two"),
      b = c(1, 3, 2)
    )
  )
})

test_that("sd_write_parquet can write geometry data", {
  df <- sd_sql(
    "SELECT ST_Point(1, 2, 4326) as geom, 'test' as name"
  )

  tmp_parquet_file <- tempfile(fileext = ".parquet")
  on.exit(unlink(tmp_parquet_file))

  sd_write_parquet(df, tmp_parquet_file)

  # Read back and verify geometry is preserved
  roundtrip_data <- sd_read_parquet(tmp_parquet_file) |>
    sd_collect()

  expect_identical(
    wk::as_wkt(roundtrip_data$geom),
    wk::wkt("POINT (1 2)", crs = "OGC:CRS84")
  )
})

test_that("sd_write_parquet validates geoparquet_version parameter", {
  df <- sd_sql(
    "SELECT ST_Point(1, 2, 4326) as geom, 'test' as name"
  )
  tmp_parquet_file <- tempfile(fileext = ".parquet")
  on.exit(unlink(tmp_parquet_file))

  # GeoParquet 1.0 shouldn't add any columns
  sd_write_parquet(df, tmp_parquet_file, geoparquet_version = "1.0")
  expect_identical(
    sd_read_parquet(tmp_parquet_file) |> colnames(),
    c("geom", "name")
  )

  # GeoParquet 1.1 should add a geom_bbox column
  sd_write_parquet(df, tmp_parquet_file, geoparquet_version = "1.1")
  expect_identical(
    sd_read_parquet(tmp_parquet_file) |> colnames(),
    c("geom_bbox", "geom", "name")
  )

  # Invalid version should error
  expect_snapshot_error(
    sd_write_parquet(df, tmp_parquet_file, geoparquet_version = "2.0"),
  )
})

test_that("sd_write_parquet accepts max_row_group_size parameter", {
  skip_if_not_installed("arrow")

  tmp_parquet_file <- tempfile(fileext = ".parquet")
  tmp_parquet_file_tiny_groups <- tempfile(fileext = ".parquet")
  on.exit(unlink(c(tmp_parquet_file, tmp_parquet_file_tiny_groups)))

  df <- data.frame(x = 1:1e5)
  sd_write_parquet(df, tmp_parquet_file)
  sd_write_parquet(df, tmp_parquet_file_tiny_groups, max_row_group_size = 100)

  # Check file size (tiny row groups have more metadata)
  expect_gt(
    file.size(tmp_parquet_file_tiny_groups),
    file.size(tmp_parquet_file)
  )
})

test_that("sd_write_parquet accepts compression parameter", {
  tmp_parquet_file <- tempfile(fileext = ".parquet")
  tmp_parquet_file_uncompressed <- tempfile(fileext = ".parquet")
  on.exit(unlink(c(tmp_parquet_file, tmp_parquet_file_uncompressed)))

  df <- data.frame(x = 1:1e5)
  sd_write_parquet(df, tmp_parquet_file)
  sd_write_parquet(df, tmp_parquet_file_uncompressed, compression = "uncompressed")

  expect_gt(
    file.size(tmp_parquet_file_uncompressed),
    file.size(tmp_parquet_file)
  )
})

test_that("sd_write_parquet accepts options parameter", {
  tmp_parquet_file <- tempfile(fileext = ".parquet")
  tmp_parquet_file_uncompressed <- tempfile(fileext = ".parquet")
  on.exit(unlink(c(tmp_parquet_file, tmp_parquet_file_uncompressed)))

  df <- data.frame(x = 1:1e5)
  sd_write_parquet(df, tmp_parquet_file)
  sd_write_parquet(
    df,
    tmp_parquet_file_uncompressed,
    options = list(compression = "uncompressed")
  )

  expect_gt(
    file.size(tmp_parquet_file_uncompressed),
    file.size(tmp_parquet_file)
  )
})

test_that("sd_write_parquet() errors for inappropriately sized options", {
  tmp_parquet_file <- tempfile(fileext = ".parquet")
  on.exit(unlink(tmp_parquet_file))

  df <- data.frame(x = 1:10)
  expect_snapshot_error(
    sd_write_parquet(df, tmp_parquet_file, options = list(foofy = character(0)))
  )

  expect_snapshot_error(
    sd_write_parquet(df, tmp_parquet_file, options = list("option without name"))
  )
})

test_that("sd_select() works with dplyr-like select syntax", {
  skip_if_not_installed("tidyselect")

  df_in <- data.frame(one = 1, two = "two", THREE = 3.0)

  expect_identical(
    df_in |> sd_select(2:3) |> sd_collect(),
    data.frame(two = "two", THREE = 3.0)
  )

  expect_identical(
    df_in |> sd_select(three_renamed = THREE, one) |> sd_collect(),
    data.frame(three_renamed = 3.0, one = 1)
  )

  expect_identical(
    df_in |> sd_select(TWO = two) |> sd_collect(),
    data.frame(TWO = "two")
  )
})

test_that("sd_transmute() works with dplyr-like transmute syntax", {
  df_in <- data.frame(x = 1:10)

  # checks that (1) unnamed inputs like `x` are named `x` in the output
  # and (2) named inputs are given an alias and (3) expressions are
  # translated.
  expect_identical(
    df_in |> sd_transmute(x, y = x + 1L) |> sd_collect(),
    data.frame(x = 1:10, y = 2:11)
  )

  # Check that the calling environment is handled
  integer_one <- 1L
  expect_identical(
    df_in |> sd_transmute(x, y = x + integer_one) |> sd_collect(),
    data.frame(x = 1:10, y = 2:11)
  )
})

test_that("sd_filter() works with dplyr-like filter syntax", {
  df_in <- data.frame(x = 1:10)

  # Zero conditions
  expect_identical(
    df_in |> sd_filter() |> sd_collect(),
    df_in
  )

  # One condition
  expect_identical(
    df_in |> sd_filter(x >= 5) |> sd_collect(),
    data.frame(x = 5:10)
  )

  # Multiple conditions
  expect_identical(
    df_in |> sd_filter(x >= 5, x >= 6) |> sd_collect(),
    data.frame(x = 6:10)
  )

  # Ensure null handling of conditions is dplyr-like (drops nulls)
  expect_identical(
    df_in |> sd_filter(x >= NA_integer_) |> sd_collect(),
    data.frame(x = integer())
  )
})

test_that("sd_arrange() works with dplyr-like arrange syntax", {
  df_in <- data.frame(x = 1:10, y = letters[10:1])

  # Zero conditions
  expect_identical(
    df_in |> sd_filter() |> sd_collect(),
    df_in
  )

  # One condition
  expect_identical(
    df_in |> sd_arrange(x) |> sd_collect(),
    df_in
  )

  # Check descending with desc()
  expect_identical(
    df_in |> sd_arrange(desc(x)) |> sd_collect(),
    data.frame(x = 10:1, y = letters[1:10])
  )

  # Check descending with dplyr::desc()
  expect_identical(
    df_in |> sd_arrange(dplyr::desc(x)) |> sd_collect(),
    data.frame(x = 10:1, y = letters[1:10])
  )
})

test_that("sd_group_by() and sd_ungroup() modify .data$group_by", {
  df_in <- data.frame(letter = letters[1:10], x = 1:10)

  # Check that we can group
  grouped <- df_in |> sd_group_by(letter, x)
  expect_identical(names(grouped$group_by), c("letter", "x"))

  # Check that we can ungroup
  expect_identical(names(sd_ungroup(grouped)$group_by), NULL)

  # Check that we can ungroup using sd_group_by()
  expect_identical(names(sd_group_by(grouped)$group_by), NULL)
})

test_that("sd_summarise() + sd_group_by() works as expected", {
  df_in <- data.frame(letter = c(rep("a", 3), rep("b", 4), rep("c", 3)), x = 1:10)

  expect_identical(
    df_in |>
      sd_group_by(letter) |>
      sd_summarise(x = sum(x), n = n()) |>
      sd_arrange(letter) |>
      sd_collect(),
    data.frame(
      letter = c("a", "b", "c"),
      x = c(1 + 2 + 3, 4 + 5 + 6 + 7, 8 + 9 + 10),
      n = c(3, 4, 3)
    )
  )
})

test_that("sd_summarise() works with dplyr-like summarise syntax", {
  df_in <- data.frame(x = as.double(1:10))

  expect_identical(
    df_in |> sd_summarise(x = sum(x)) |> sd_collect(),
    data.frame(x = sum(as.double(1:10)))
  )
})
