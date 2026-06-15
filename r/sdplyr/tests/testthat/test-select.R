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

test_that("select() basic usage works for sedonadb_dataframe", {
  df <- tibble(one = 1L, two = "two", THREE = 3.0)

  expect_identical(
    df |> as_sedonadb_dataframe() |> select(2:3) |> collect(),
    df |> select(2:3)
  )

  expect_identical(
    df |> as_sedonadb_dataframe() |> select(three_renamed = THREE, one) |> collect(),
    df |> select(three_renamed = THREE, one)
  )

  expect_identical(
    df |> as_sedonadb_dataframe() |> select(TWO = two) |> collect(),
    df |> select(TWO = two)
  )
})

test_that("select() works with everything()", {
  df <- tibble(a = 1L, b = 2L, c = 3L)
  expect_identical(
    df |> as_sedonadb_dataframe() |> select(everything()) |> collect(),
    df |> select(everything())
  )
})

test_that("select() works with last_col()", {
  df <- tibble(a = 1L, b = 2L, c = 3L)
  expect_identical(
    df |> as_sedonadb_dataframe() |> select(last_col()) |> collect(),
    df |> select(last_col())
  )

  expect_identical(
    df |> as_sedonadb_dataframe() |> select(last_col(offset = 1)) |> collect(),
    df |> select(last_col(offset = 1))
  )
})

test_that("select() works with starts_with()", {
  df <- tibble(val_1 = 1L, val_2 = 2L, other = 3L)
  expect_identical(
    df |> as_sedonadb_dataframe() |> select(starts_with("val")) |> collect(),
    df |> select(starts_with("val"))
  )
})

test_that("select() works with ends_with()", {
  df <- tibble(x_coord = 1.0, y_coord = 2.0, other = 3L)
  expect_identical(
    df |> as_sedonadb_dataframe() |> select(ends_with("coord")) |> collect(),
    df |> select(ends_with("coord"))
  )
})

test_that("select() works with contains()", {
  df <- tibble(score_total = 100L, score_avg = 50L, id = 1L)
  expect_identical(
    df |> as_sedonadb_dataframe() |> select(contains("score")) |> collect(),
    df |> select(contains("score"))
  )
})

test_that("select() works with matches()", {
  df <- tibble(val_1 = 1L, val_2 = 2L, other = 3L)
  expect_identical(
    df |> as_sedonadb_dataframe() |> select(matches("_[0-9]$")) |> collect(),
    df |> select(matches("_[0-9]$"))
  )
})

test_that("select() works with num_range()", {
  df <- tibble(x1 = 1L, x2 = 2L, x3 = 3L, other = 4L)
  expect_identical(
    df |> as_sedonadb_dataframe() |> select(num_range("x", 1:2)) |> collect(),
    df |> select(num_range("x", 1:2))
  )
})

test_that("select() works with all_of()", {
  df <- tibble(a = 1L, b = 2L, c = 3L)
  expect_identical(
    df |> as_sedonadb_dataframe() |> select(all_of(c("a", "c"))) |> collect(),
    df |> select(all_of(c("a", "c")))
  )

  expect_error(
    df |> as_sedonadb_dataframe() |> select(all_of(c("a", "nonexistent"))) |> collect()
  )
})

test_that("select() works with any_of()", {
  df <- tibble(a = 1L, b = 2L)
  expect_identical(
    df |> as_sedonadb_dataframe() |> select(any_of(c("a", "nonexistent"))) |> collect(),
    df |> select(any_of(c("a", "nonexistent")))
  )

  expect_identical(
    df |> as_sedonadb_dataframe() |> select(any_of(c("x", "y"))) |> collect(),
    df |> select(any_of(c("x", "y")))
  )
})

test_that("select() works with where()", {
  df <- tibble(num = 1L, chr = "a")
  expect_identical(
    df |> as_sedonadb_dataframe() |> select(where(is.numeric)) |> collect(),
    df |> select(where(is.numeric))
  )

  expect_identical(
    df |> as_sedonadb_dataframe() |> select(where(is.character)) |> collect(),
    df |> select(where(is.character))
  )
})

test_that("select() works with : range operator", {
  df <- tibble(a = 1L, b = 2L, c = 3L, d = 4L)
  expect_identical(
    df |> as_sedonadb_dataframe() |> select(b:d) |> collect(),
    df |> select(b:d)
  )
})

test_that("select() works with ! negation", {
  df <- tibble(a = 1L, b = 2L, c = 3L)
  expect_identical(
    df |> as_sedonadb_dataframe() |> select(!b) |> collect(),
    df |> select(!b)
  )
})

test_that("select() works with c() for combining selections", {
  df <- tibble(a = 1L, val_1 = 2L, val_2 = 3L)
  expect_identical(
    df |> as_sedonadb_dataframe() |> select(c(a, starts_with("val"))) |> collect(),
    df |> select(c(a, starts_with("val")))
  )
})

test_that("select() works with - for removing columns", {
  df <- tibble(a = 1L, b = 2L, c = 3L)
  expect_identical(
    df |> as_sedonadb_dataframe() |> select(-b) |> collect(),
    df |> select(-b)
  )

  expect_identical(
    df |> as_sedonadb_dataframe() |> select(-c(a, b)) |> collect(),
    df |> select(-c(a, b))
  )
})

test_that("select() works with & (intersection)", {
  df <- tibble(score_avg = 1L, score_total = 2L, other_avg = 3L)
  expect_identical(
    df |>
      as_sedonadb_dataframe() |>
      select(starts_with("score") & ends_with("avg")) |>
      collect(),
    df |> select(starts_with("score") & ends_with("avg"))
  )
})

test_that("select() works with | (union)", {
  df <- tibble(val_1 = 1L, x_coord = 2.0, other = 3L)
  expect_identical(
    df |>
      as_sedonadb_dataframe() |>
      select(starts_with("val") | ends_with("coord")) |>
      collect(),
    df |> select(starts_with("val") | ends_with("coord"))
  )
})
