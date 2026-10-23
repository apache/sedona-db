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

test_that("unnest generics delegate data frames to tidyr", {
  df <- tibble(id = 1:2, x = list(1:2, integer()))
  expect_identical(
    unnest_longer(df, x, keep_empty = TRUE),
    tidyr::unnest_longer(df, x, keep_empty = TRUE)
  )
  df <- tibble(x = list(list(a = 1, b = 2)))
  expect_identical(
    unnest_wider(df, x, names_sep = "_"),
    tidyr::unnest_wider(df, x, names_sep = "_")
  )
})

test_that("unnest generics dispatch to SedonaDB", {
  df <- sd_sql("SELECT [1, 2] AS x")
  expect_equal(collect(unnest_longer(df, starts_with("x"))), tibble(x = c(1, 2)))
  df <- sd_sql("SELECT named_struct('x', 1) AS s")
  expect_equal(collect(unnest_wider(df, s, names_sep = "_")), tibble(s_x = 1))
  expect_error(unnest_longer(df, s, keep_empty = TRUE), "keep_empty")
  expect_error(unnest_longer(df, s, values_to = "x"), "values_to")
  expect_error(unnest_wider(df, s, simplify = FALSE), "simplify")
  expect_error(unnest_wider(df, s, unknown = TRUE), "must be empty")
})
