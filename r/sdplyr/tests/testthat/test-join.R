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

test_that("dplyr::*_join() works for sedonadb_dataframe", {
  df1 <- tibble(key = 1:3, val1 = c("a", "b", "c"))
  df2 <- tibble(key = 2:4, val2 = c("x", "y", "z"))

  expect_identical(
    df1 |>
      as_sedonadb_dataframe() |>
      inner_join(as_sedonadb_dataframe(df2), by = "key") |>
      arrange(key) |>
      collect(),
    df1 |> inner_join(df2, by = "key") |> arrange(key)
  )

  expect_identical(
    df1 |>
      as_sedonadb_dataframe() |>
      left_join(as_sedonadb_dataframe(df2), by = "key") |>
      arrange(key) |>
      collect(),
    df1 |> left_join(df2, by = "key") |> arrange(key)
  )

  expect_identical(
    df1 |>
      as_sedonadb_dataframe() |>
      right_join(as_sedonadb_dataframe(df2), by = "key") |>
      arrange(key) |>
      collect(),
    df1 |> right_join(df2, by = "key") |> arrange(key)
  )

  expect_identical(
    df1 |>
      as_sedonadb_dataframe() |>
      full_join(as_sedonadb_dataframe(df2), by = "key") |>
      arrange(key) |>
      collect(),
    df1 |> full_join(df2, by = "key") |> arrange(key)
  )

  expect_identical(
    df1 |>
      as_sedonadb_dataframe() |>
      semi_join(as_sedonadb_dataframe(df2), by = "key") |>
      arrange(key) |>
      collect(),
    df1 |> semi_join(df2, by = "key") |> arrange(key)
  )

  expect_identical(
    df1 |>
      as_sedonadb_dataframe() |>
      anti_join(as_sedonadb_dataframe(df2), by = "key") |>
      arrange(key) |>
      collect(),
    df1 |> anti_join(df2, by = "key") |> arrange(key)
  )

  df3 <- tibble(a = 1:2)
  df4 <- tibble(b = c("x", "y"))

  expect_identical(
    df3 |>
      as_sedonadb_dataframe() |>
      cross_join(as_sedonadb_dataframe(df4)) |>
      arrange(a, b) |>
      collect(),
    df3 |> cross_join(df4) |> arrange(a, b)
  )
})

test_that("dplyr::*_join(keep = TRUE) works for sedonadb_dataframe", {
  df1 <- tibble(key_x = 1:3, val1 = c("a", "b", "c"))
  df2 <- tibble(key_y = 2:4, val2 = c("x", "y", "z"))

  expect_identical(
    df1 |>
      as_sedonadb_dataframe() |>
      inner_join(as_sedonadb_dataframe(df2), by = c("key_x" = "key_y"), keep = TRUE) |>
      arrange(key_x) |>
      collect(),
    df1 |> inner_join(df2, by = c("key_x" = "key_y"), keep = TRUE) |> arrange(key_x)
  )

  expect_identical(
    df1 |>
      as_sedonadb_dataframe() |>
      left_join(as_sedonadb_dataframe(df2), by = c("key_x" = "key_y"), keep = TRUE) |>
      arrange(key_x) |>
      collect(),
    df1 |> left_join(df2, by = c("key_x" = "key_y"), keep = TRUE) |> arrange(key_x)
  )

  expect_identical(
    df1 |>
      as_sedonadb_dataframe() |>
      right_join(as_sedonadb_dataframe(df2), by = c("key_x" = "key_y"), keep = TRUE) |>
      arrange(key_x) |>
      collect(),
    df1 |> right_join(df2, by = c("key_x" = "key_y"), keep = TRUE) |> arrange(key_x)
  )

  expect_identical(
    df1 |>
      as_sedonadb_dataframe() |>
      full_join(as_sedonadb_dataframe(df2), by = c("key_x" = "key_y"), keep = TRUE) |>
      arrange(key_x) |>
      collect(),
    df1 |> full_join(df2, by = c("key_x" = "key_y"), keep = TRUE) |> arrange(key_x)
  )
})
