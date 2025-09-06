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

test_that("sd_read_parquet() works", {
  path <- system.file("files/natural-earth_cities_geo.parquet", package = "sedonadb")
  expect_identical(sd_count(sd_read_parquet(path)), 243)

  expect_identical(sd_count(sd_read_parquet(c(path, path))), 243 * 2)
})

test_that("views can be created and dropped", {
  df <- sd_sql("SELECT 1 as one")
  expect_true(rlang::is_reference(sd_to_view(df, "foofy"), df))
  expect_identical(
    sd_sql("SELECT * FROM foofy") |> sd_collect(),
    data.frame(one = 1)
  )

  expect_identical(
    sd_view("foofy") |> sd_collect(),
    data.frame(one = 1)
  )

  sd_drop_view("foofy")
  expect_error(sd_sql("SELECT * FROM foofy"), "table '(.*?)' not found")
  expect_error(sd_view("foofy"), "No table named 'foofy'")
})
