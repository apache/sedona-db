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
