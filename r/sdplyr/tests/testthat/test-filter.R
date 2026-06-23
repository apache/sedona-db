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

test_that("filter() works for sedonadb_dataframe", {
  df <- tibble(x = c(3L, 1L, 2L), y = c("c", "a", "b"))

  expect_identical(
    df |> as_sedonadb_dataframe() |> filter(x > 1L) |> arrange(x) |> collect(),
    df |> filter(x > 1L) |> arrange(x)
  )

  expect_identical(
    df |> as_sedonadb_dataframe() |> filter(x == 2L) |> arrange(x) |> collect(),
    df |> filter(x == 2L) |> arrange(x)
  )

  expect_identical(
    df |> as_sedonadb_dataframe() |> filter(y == "a") |> arrange(x) |> collect(),
    df |> filter(y == "a") |> arrange(x)
  )

  expect_identical(
    df |> as_sedonadb_dataframe() |> filter(x != 1L) |> arrange(x) |> collect(),
    df |> filter(x != 1L) |> arrange(x)
  )
})

test_that("filter() works with multiple conditions", {
  df <- tibble(x = c(1L, 1L, 2L, 2L), y = c(4L, 3L, 2L, 1L))

  expect_identical(
    df |> as_sedonadb_dataframe() |> filter(x == 1L, y > 3L) |> arrange(x) |> collect(),
    df |> filter(x == 1L, y > 3L) |> arrange(x)
  )

  expect_identical(
    df |> as_sedonadb_dataframe() |> filter(x >= 1L, y <= 2L) |> arrange(x) |> collect(),
    df |> filter(x >= 1L, y <= 2L) |> arrange(x)
  )

  expect_identical(
    df |> as_sedonadb_dataframe() |> filter(x == 1L | y == 1L) |> arrange(x) |> collect(),
    df |> filter(x == 1L | y == 1L) |> arrange(x)
  )
})

test_that("filter() handles NA values", {
  df <- tibble(x = c(2L, NA, 1L, NA, 3L))

  # Comparisons with NA return NA (falsy), so rows with NA are filtered out
  expect_identical(
    df |> as_sedonadb_dataframe() |> filter(x > 1L) |> collect(),
    df |> filter(x > 1L)
  )

  expect_identical(
    df |> as_sedonadb_dataframe() |> filter(x == 2L) |> collect(),
    df |> filter(x == 2L)
  )

  expect_identical(
    df |> as_sedonadb_dataframe() |> filter(is.na(x)) |> collect(),
    df |> filter(is.na(x))
  )

  expect_identical(
    df |> as_sedonadb_dataframe() |> filter(!is.na(x)) |> collect(),
    df |> filter(!is.na(x))
  )
})

test_that("filter(..., .by) is unsupported", {
  df <- tibble(x = 1:3, g = c("a", "a", "b"))
  expect_snapshot_error(
    df |> as_sedonadb_dataframe() |> filter(x > 1L, .by = "g")
  )
})

test_that("filter(..., .preserve) is unsupported", {
  df <- tibble(x = 1:3)
  expect_snapshot_error(
    df |> as_sedonadb_dataframe() |> filter(x > 1L, .preserve = TRUE)
  )
})
