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

test_that("sd_join() defaults match dplyr join defaults", {
  df1 <- data.frame(key_x = 1:6, letters = letters[1:6])
  df2 <- data.frame(key_y = 10:4, letters = LETTERS[1:7])

  expect_identical(
    sd_inner_join(df1, df2, by = c("key_x" = "key_y")) |>
      sd_arrange(key_x) |>
      as.data.frame(),
    dplyr::inner_join(df1, df2, by = c("key_x" = "key_y")) |>
      dplyr::arrange(key_x) |>
      as.data.frame()
  )

  expect_identical(
    sd_left_join(df1, df2, by = c("key_x" = "key_y")) |>
      sd_arrange(key_x) |>
      as.data.frame(),
    dplyr::left_join(df1, df2, by = c("key_x" = "key_y")) |>
      dplyr::arrange(key_x) |>
      as.data.frame()
  )

  expect_identical(
    sd_right_join(df1, df2, by = c("key_x" = "key_y")) |>
      sd_arrange(key_x) |>
      as.data.frame(),
    dplyr::right_join(df1, df2, by = c("key_x" = "key_y")) |>
      dplyr::arrange(key_x) |>
      as.data.frame()
  )

  expect_identical(
    sd_full_join(df1, df2, by = c("key_x" = "key_y")) |>
      sd_arrange(key_x) |>
      as.data.frame(),
    dplyr::full_join(df1, df2, by = c("key_x" = "key_y")) |>
      dplyr::arrange(key_x) |>
      as.data.frame()
  )

  expect_identical(
    sd_anti_join(df1, df2, by = c("key_x" = "key_y")) |>
      sd_arrange(key_x) |>
      as.data.frame(),
    dplyr::anti_join(df1, df2, by = c("key_x" = "key_y")) |>
      dplyr::arrange(key_x) |>
      as.data.frame()
  )

  expect_identical(
    sd_semi_join(df1, df2, by = c("key_x" = "key_y")) |>
      sd_arrange(key_x) |>
      as.data.frame(),
    dplyr::semi_join(df1, df2, by = c("key_x" = "key_y")) |>
      dplyr::arrange(key_x) |>
      as.data.frame()
  )

  expect_identical(
    sd_cross_join(df1, df2) |>
      sd_arrange(key_x) |>
      as.data.frame(),
    dplyr::cross_join(df1, df2) |>
      dplyr::arrange(key_x) |>
      as.data.frame()
  )
})

test_that("sd_join(keep = TRUE) behaviour matches dplyr join", {
  df1 <- data.frame(key_x = 1:6, letters = letters[1:6])
  df2 <- data.frame(key_y = 10:4, letters = LETTERS[1:7])

  expect_identical(
    sd_inner_join(df1, df2, by = c("key_x" = "key_y"), keep = TRUE) |>
      sd_arrange(key_x) |>
      as.data.frame(),
    dplyr::inner_join(df1, df2, by = c("key_x" = "key_y"), keep = TRUE) |>
      dplyr::arrange(key_x) |>
      as.data.frame()
  )

  expect_identical(
    sd_left_join(df1, df2, by = c("key_x" = "key_y"), keep = TRUE) |>
      sd_arrange(key_x) |>
      as.data.frame(),
    dplyr::left_join(df1, df2, by = c("key_x" = "key_y"), keep = TRUE) |>
      dplyr::arrange(key_x) |>
      as.data.frame()
  )

  expect_identical(
    sd_right_join(df1, df2, by = c("key_x" = "key_y"), keep = TRUE) |>
      sd_arrange(key_x) |>
      as.data.frame(),
    dplyr::right_join(df1, df2, by = c("key_x" = "key_y"), keep = TRUE) |>
      dplyr::arrange(key_x) |>
      as.data.frame()
  )

  expect_identical(
    sd_full_join(df1, df2, by = c("key_x" = "key_y"), keep = TRUE) |>
      sd_arrange(key_x) |>
      as.data.frame(),
    dplyr::full_join(df1, df2, by = c("key_x" = "key_y"), keep = TRUE) |>
      dplyr::arrange(key_x) |>
      as.data.frame()
  )
})
