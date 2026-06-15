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

test_that("mutate() works for sedonadb_dataframe", {
  df <- tibble(x = 1:3, y = c("a", "b", "c"))

  # Purely additive usage
  expect_identical(
    df |> as_sedonadb_dataframe() |> mutate(z = x + 1L) |> collect(),
    df |> mutate(z = x + 1L)
  )

  # Purely replacement usage
  expect_identical(
    df |> as_sedonadb_dataframe() |> mutate(x = x + 1L) |> collect(),
    df |> mutate(x = x + 1L)
  )

  # Some of both
  expect_identical(
    df |> as_sedonadb_dataframe() |> mutate(x = x + 1L, z = 2L) |> collect(),
    df |> mutate(x = x + 1L, z = 2L)
  )

  # Referring to previous args
  expect_identical(
    df |> as_sedonadb_dataframe() |> mutate(x = x + 1L, z = x + 1L) |> collect(),
    df |> mutate(x = x + 1L, z = x + 1L)
  )

  expect_identical(
    df |>
      as_sedonadb_dataframe() |>
      mutate(x = x + 1L, z = x + 1L, x = z + 2L) |>
      collect(),
    df |> mutate(x = x + 1L, z = x + 1L, x = z + 2L)
  )
})

test_that("transmute() works for sedonadb_dataframe", {
  df <- tibble(x = 1:3, y = c("a", "b", "c"))

  # Check basic transmute()
  expect_identical(
    df |> as_sedonadb_dataframe() |> transmute(z = x + 1L) |> collect(),
    df |> transmute(z = x + 1L)
  )

  # Check transmute with zero args
  expect_identical(
    df |> as_sedonadb_dataframe() |> transmute() |> collect(),
    df |> transmute()
  )

  # Check transmute referring to previous args
  expect_identical(
    df |> as_sedonadb_dataframe() |> transmute(x = x + 1L, z = x + 1L) |> collect(),
    df |> transmute(x = x + 1L, z = x + 1L)
  )

  expect_identical(
    df |>
      as_sedonadb_dataframe() |>
      transmute(x = x + 1L, z = x + 1L, x = z + 2L) |>
      collect(),
    df |> transmute(x = x + 1L, z = x + 1L, x = z + 2L)
  )
})
