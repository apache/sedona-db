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

test_that("collect() returns a collected tibble", {
  df <- tibble(one = 1L, two = "two", THREE = 3.0)
  expect_identical(
    df |> as_sedonadb_dataframe() |> collect(),
    df
  )
})

test_that("compute() returns a collected SedonaDB dataframe", {
  df <- tibble(one = 1L, two = "two", THREE = 3.0)
  computed <- df |> as_sedonadb_dataframe() |> compute()
  expect_s3_class(computed, "sedonadb_dataframe")
  expect_identical(
    computed |> collect(),
    df
  )
})
