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

test_that("select() works for sedonadb_dataframe", {
  skip_if_not_installed("dplyr")

  df <- sd_sql("SELECT 1 as one, 'two' as two, 3.0 as \"THREE\"")

  expect_identical(
    df |> dplyr::select(2:3) |> dplyr::collect(),
    tibble::tibble(two = "two", THREE = 3.0)
  )

  expect_identical(
    df |> dplyr::select(three_renamed = THREE, one) |> dplyr::collect(),
    tibble::tibble(three_renamed = 3.0, one = 1)
  )

  expect_identical(
    df |> dplyr::select(TWO = two) |> dplyr::collect(),
    tibble::tibble(TWO = "two")
  )
})

test_that("sd_join() defaults match dplyr join defaults", {
  df1 <- data.frame(key_x = 1:6, letters = letters[1:6])
  df2 <- data.frame(key_y = 10:4, letters = LETTERS[1:7])

  expect_identical(
    sd_join(df1, df2, by = c("key_x" = "key_y"), join_type = "inner") |>
      sd_arrange(key_x) |>
      as.data.frame(),
    dplyr::inner_join(df1, df2, by = c("key_x" = "key_y")) |>
      dplyr::arrange(key_x) |>
      as.data.frame()
  )

  expect_identical(
    sd_join(df1, df2, by = c("key_x" = "key_y"), join_type = "left") |>
      sd_arrange(key_x) |>
      as.data.frame(),
    dplyr::left_join(df1, df2, by = c("key_x" = "key_y")) |>
      dplyr::arrange(key_x) |>
      as.data.frame()
  )

  expect_identical(
    sd_join(df1, df2, by = c("key_x" = "key_y"), join_type = "right") |>
      sd_arrange(key_x) |>
      as.data.frame(),
    dplyr::right_join(df1, df2, by = c("key_x" = "key_y")) |>
      dplyr::arrange(key_x) |>
      as.data.frame()
  )

  expect_identical(
    sd_join(df1, df2, by = c("key_x" = "key_y"), join_type = "full") |>
      sd_arrange(key_x) |>
      as.data.frame(),
    dplyr::full_join(df1, df2, by = c("key_x" = "key_y")) |>
      dplyr::arrange(key_x) |>
      as.data.frame()
  )

  expect_identical(
    sd_join(df1, df2, by = c("key_x" = "key_y"), join_type = "leftanti") |>
      sd_arrange(key_x) |>
      as.data.frame(),
    dplyr::anti_join(df1, df2, by = c("key_x" = "key_y")) |>
      dplyr::arrange(key_x) |>
      as.data.frame()
  )

  expect_identical(
    sd_join(df1, df2, by = c("key_x" = "key_y"), join_type = "leftsemi") |>
      sd_arrange(key_x) |>
      as.data.frame(),
    dplyr::semi_join(df1, df2, by = c("key_x" = "key_y")) |>
      dplyr::arrange(key_x) |>
      as.data.frame()
  )
})
