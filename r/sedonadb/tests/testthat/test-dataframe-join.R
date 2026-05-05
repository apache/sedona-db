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

test_that("sd_join() select argument is applied to join results", {
  df1 <- data.frame(common = "from_x", letters_x = letters[1:6], key = 1:6)
  df2 <- data.frame(common = "from_y", key = 10:4, letters_y = LETTERS[1:7])

  # With select = NULL, columns are blindly stacked
  joined <- sd_join(df1, df2, sd_join_by(x$key == y$key), select = NULL)
  expect_identical(
    colnames(joined),
    c(names(df1), names(df2))
  )

  # With select = sd_join_select_default()
  joined <- sd_join(
    df1,
    df2,
    sd_join_by(x$key == y$key),
    select = sd_join_select_default()
  )
  expect_identical(
    colnames(joined),
    c("common.x", "letters_x", "key", "common.y", "letters_y")
  )

  # Check at least one result
  expect_identical(
    as.data.frame(joined |> sd_arrange(key)),
    merge(df1, df2, by = "key")[c(
      "common.x",
      "letters_x",
      "key",
      "common.y",
      "letters_y"
    )]
  )

  # Check that custom suffixes work
  joined <- sd_join(
    df1,
    df2,
    sd_join_by(x$key == y$key),
    select = sd_join_select_default(suffix = c("_custom_x", "_custom_y"))
  )
  expect_identical(
    colnames(joined),
    c("common_custom_x", "letters_x", "key", "common_custom_y", "letters_y")
  )

  # Check that custom selections work
  joined <- sd_join(
    df1,
    df2,
    sd_join_by(x$key == y$key),
    select = sd_join_select(letters_x, key = y$key, common = x$common, y$letters_y)
  )
  expect_identical(
    colnames(joined),
    c("letters_x", "key", "common", "letters_y")
  )
})

test_that("sd_join() join_type argument is applied to join results", {
  df1 <- data.frame(letters_x = letters[1:6], key = 1:6)
  df2 <- data.frame(key = 10:4, letters_y = LETTERS[1:7])

  joined <- df1 |> sd_join(df2, by = "key", join_type = "left")
  expect_identical(
    as.data.frame(joined |> sd_arrange(key)),
    merge(df1, df2, by = "key", all.x = TRUE, all.y = FALSE)[c(
      "letters_x",
      "key",
      "letters_y"
    )]
  )

  joined <- df1 |> sd_join(df2, by = "key", join_type = "right")
  expect_identical(
    as.data.frame(joined |> sd_arrange(key)),
    merge(df1, df2, by = "key", all.x = FALSE, all.y = TRUE)[c(
      "letters_x",
      "key",
      "letters_y"
    )]
  )

  joined <- df1 |> sd_join(df2, by = "key", join_type = "full")
  expect_identical(
    as.data.frame(joined |> sd_arrange(key)),
    merge(df1, df2, by = "key", all.x = TRUE, all.y = TRUE)[c(
      "letters_x",
      "key",
      "letters_y"
    )]
  )

  df1$extra_column <- "foofy"
  joined <- df1 |> sd_join(df2, by = "key", join_type = "full")
  expect_identical(colnames(joined), c("letters_x", "key", "extra_column", "letters_y"))
})

test_that("sd_join() computes the correct columns for spatial predicate joins", {
  cities <- sd_read_parquet(system.file(
    "files/natural-earth_cities_geo.parquet",
    package = "sedonadb"
  ))
  countries <- sd_read_parquet(system.file(
    "files/natural-earth_countries_geo.parquet",
    package = "sedonadb"
  ))

  # left_join: We should get one geometry column from the logical left side (points)
  df <- cities |> sd_left_join(countries, by = sd_join_intersects())
  expect_identical(
    colnames(df),
    c("name.x", "geometry", "name.y", "continent")
  )
  expect_identical(
    df |> sd_transmute(dim = .fns$st_dimension(geometry)) |> head(1) |> as.data.frame(),
    data.frame(dim = 0L)
  )

  # inner_join: We should get one geometry column from the logical left side (points)
  df <- cities |> sd_inner_join(countries, by = sd_join_intersects())
  expect_identical(
    colnames(df),
    c("name.x", "geometry", "name.y", "continent")
  )
  expect_identical(
    df |> sd_transmute(dim = .fns$st_dimension(geometry)) |> head(1) |> as.data.frame(),
    data.frame(dim = 0L)
  )

  # right_join: We should get one geometry column from the logical right side (polygons)
  df <- cities |> sd_right_join(countries, by = sd_join_intersects())
  expect_identical(
    colnames(df),
    c("name.x", "geometry", "name.y", "continent")
  )
  expect_identical(
    df |> sd_transmute(dim = .fns$st_dimension(geometry)) |> head(1) |> as.data.frame(),
    data.frame(dim = 2L)
  )

  # full_join: Keeps both geometry columns by default
  df <- cities |> sd_full_join(countries, by = sd_join_intersects())
  expect_identical(
    colnames(df),
    c("name.x", "geometry.x", "name.y", "continent", "geometry.y")
  )
  expect_identical(
    df |> sd_transmute(dim = .fns$st_dimension(geometry.x)) |> head(1) |> as.data.frame(),
    data.frame(dim = 0L)
  )
  expect_identical(
    df |> sd_transmute(dim = .fns$st_dimension(geometry.y)) |> head(1) |> as.data.frame(),
    data.frame(dim = 2L)
  )

  # Any join: Keeps both geometry columns if keep = TRUE
  df <- cities |> sd_inner_join(countries, by = sd_join_intersects(), keep = TRUE)
  expect_identical(
    colnames(df),
    c("name.x", "geometry.x", "name.y", "continent", "geometry.y")
  )
  expect_identical(
    df |> sd_transmute(dim = .fns$st_dimension(geometry.x)) |> head(1) |> as.data.frame(),
    data.frame(dim = 0L)
  )
  expect_identical(
    df |> sd_transmute(dim = .fns$st_dimension(geometry.y)) |> head(1) |> as.data.frame(),
    data.frame(dim = 2L)
  )
})
