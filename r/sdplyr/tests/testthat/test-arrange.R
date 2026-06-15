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

test_that("arrange() works for sedonadb_dataframe", {
  df <- tibble(x = c(3L, 1L, 2L), y = c("c", "a", "b"))

  expect_identical(
    df |> as_sedonadb_dataframe() |> arrange(x) |> collect(),
    df |> arrange(x)
  )

  expect_identical(
    df |> as_sedonadb_dataframe() |> arrange(y) |> collect(),
    df |> arrange(y)
  )

  expect_identical(
    df |> as_sedonadb_dataframe() |> arrange(desc(x)) |> collect(),
    df |> arrange(desc(x))
  )

  expect_identical(
    df |> as_sedonadb_dataframe() |> arrange(desc(y)) |> collect(),
    df |> arrange(desc(y))
  )
})

test_that("arrange() works with multiple columns", {
  df <- tibble(x = c(1L, 1L, 2L, 2L), y = c(4L, 3L, 2L, 1L))

  expect_identical(
    df |> as_sedonadb_dataframe() |> arrange(x, y) |> collect(),
    df |> arrange(x, y)
  )

  expect_identical(
    df |> as_sedonadb_dataframe() |> arrange(x, desc(y)) |> collect(),
    df |> arrange(x, desc(y))
  )

  expect_identical(
    df |> as_sedonadb_dataframe() |> arrange(desc(x), y) |> collect(),
    df |> arrange(desc(x), y)
  )
})

test_that("arrange() handles NA values", {
  df <- tibble(x = c(2L, NA, 1L, NA, 3L))

  expect_identical(
    df |> as_sedonadb_dataframe() |> arrange(x) |> collect(),
    df |> arrange(x)
  )

  expect_identical(
    df |> as_sedonadb_dataframe() |> arrange(desc(x)) |> collect(),
    df |> arrange(desc(x))
  )
})

test_that("arrange(..., .by_group) is unsupported", {
  df <- tibble(x = 1:3)
  expect_snapshot_error(
    df |> as_sedonadb_dataframe() |> arrange(.by_group = TRUE)
  )
})
