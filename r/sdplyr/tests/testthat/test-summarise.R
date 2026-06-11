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

test_that("group_by() works for sedonadb_dataframe", {
  df <- tibble(letter = c("a", "a", "b", "b"), x = c(1, 2, 3, 4))

  expect_identical(
    df |>
      as_sedonadb_dataframe() |>
      group_by(letter) |>
      summarise(total = sum(x), .groups = "drop") |>
      arrange(letter) |>
      collect(),
    df |>
      group_by(letter) |>
      summarise(total = sum(x), .groups = "drop") |>
      arrange(letter)
  )
})

test_that("group_by(.add = TRUE) adds to existing groups", {
  df <- tibble(a = c("x", "x", "y", "y"), b = c(1, 2, 1, 2), val = c(1, 2, 3, 4))

  expect_identical(
    df |>
      as_sedonadb_dataframe() |>
      group_by(a) |>
      group_by(b, .add = TRUE) |>
      summarise(total = sum(val), .groups = "drop") |>
      arrange(a, b) |>
      collect(),
    df |>
      group_by(a) |>
      group_by(b, .add = TRUE) |>
      summarise(total = sum(val), .groups = "drop") |>
      arrange(a, b)
  )
})

test_that("ungroup() removes groups", {
  df <- tibble(letter = c("a", "a", "b"), x = c(1, 2, 3))

  expect_identical(
    df |>
      as_sedonadb_dataframe() |>
      group_by(letter) |>
      ungroup() |>
      summarise(total = sum(x), .groups = "drop") |>
      collect(),
    df |>
      group_by(letter) |>
      ungroup() |>
      summarise(total = sum(x), .groups = "drop")
  )
})

test_that("summarise() works without groups", {
  df <- tibble(x = c(1, 2, 3, 4, 5))

  expect_identical(
    df |>
      as_sedonadb_dataframe() |>
      summarise(total = sum(x), .groups = "drop") |>
      collect(),
    df |> summarise(total = sum(x), .groups = "drop")
  )
})

test_that("summarize() alternate spelling works", {
  df <- tibble(x = c(1, 2, 3, 4, 5))

  expect_identical(
    df |>
      as_sedonadb_dataframe() |>
      summarize(total = sum(x), .groups = "drop") |>
      collect(),
    df |> summarize(total = sum(x), .groups = "drop")
  )
})

test_that("summarise() works with multiple aggregations", {
  df <- tibble(letter = c("a", "a", "b", "b", "b"), x = c(1, 2, 3, 4, 5))

  expect_identical(
    df |>
      as_sedonadb_dataframe() |>
      group_by(letter) |>
      summarise(total = sum(x), avg = mean(x), .groups = "drop") |>
      arrange(letter) |>
      collect(),
    df |>
      group_by(letter) |>
      summarise(total = sum(x), avg = mean(x), .groups = "drop") |>
      arrange(letter)
  )
})

test_that("summarise() works with .by argument", {
  df <- tibble(letter = c("a", "a", "b", "b"), x = c(1, 2, 3, 4))

  expect_identical(
    df |>
      as_sedonadb_dataframe() |>
      summarise(total = sum(x), .by = letter) |>
      arrange(letter) |>
      collect(),
    df |>
      summarise(total = sum(x), .by = letter) |>
      arrange(letter)
  )
})

test_that("summarise() works with .by = c(...) for multiple groups", {
  df <- tibble(a = c("x", "x", "y", "y"), b = c(1, 2, 1, 2), val = c(1, 2, 3, 4))

  expect_identical(
    df |>
      as_sedonadb_dataframe() |>
      summarise(total = sum(val), .by = c(a, b)) |>
      arrange(a, b) |>
      collect(),
    df |>
      summarise(total = sum(val), .by = c(a, b)) |>
      arrange(a, b)
  )
})

test_that("summarise() referring to previous args", {
  df <- tibble(letter = c("a", "a", "b", "b"), x = c(1, 2, 3, 4))

  expect_identical(
    df |>
      as_sedonadb_dataframe() |>
      group_by(letter) |>
      summarise(total = sum(x), doubled = total * 2, .groups = "drop") |>
      arrange(letter) |>
      collect(),
    df |>
      group_by(letter) |>
      summarise(total = sum(x), doubled = total * 2, .groups = "drop") |>
      arrange(letter)
  )
})

test_that("summarise() with duplicate column names", {
  df <- tibble(letter = c("a", "a", "b", "b"), x = c(1, 2, 3, 4))

  expect_identical(
    df |>
      as_sedonadb_dataframe() |>
      group_by(letter) |>
      summarise(y = mean(x), y = sum(x), .groups = "drop") |>
      arrange(letter) |>
      collect(),
    df |>
      group_by(letter) |>
      summarise(y = mean(x), y = sum(x), .groups = "drop") |>
      arrange(letter)
  )
})

test_that("summarise() errors on unsupported .groups values", {
  df <- tibble(x = c(1, 2, 3)) |> as_sedonadb_dataframe()

  expect_snapshot_error(
    df |> summarise(total = sum(x), .groups = "keep")
  )

  expect_snapshot_error(
    df |> summarise(total = sum(x), .groups = "keep", .by = x)
  )
})

test_that("summarise() messages for implicit .groups", {
  df <- tibble(x = c(1, 2, 3)) |> as_sedonadb_dataframe()

  expect_snapshot(
    df |> summarise(total = sum(x))
  )
})
