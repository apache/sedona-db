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

test_that("sf geometry objects can be converted to SedonaDB literals", {
  skip_if_not_installed("sf")

  objects <- list(
    sf::st_as_sf(sf::st_as_sfc("POINT (0 1)")),
    sf::st_as_sfc("POINT (0 1)"),
    sf::st_point(c(0, 1)),
    sf::st_bbox(sf::st_sfc(sf::st_point(c(0, 1)), sf::st_point(c(2, 3))))
  )

  for (x in objects) {
    df <- sd_sql("SELECT ST_Translate($1, 0, 0) as geom", params = list(x))
    collected <- sd_collect(df)
    expect_identical(sf::st_as_sfc(collected$geom), sf::st_as_sfc(wk::as_wkb(x)))
  }
})

test_that("sf objects can be converted to and from SedonaDB data frames", {
  skip_if_not_installed("sf")

  nc <- sf::read_sf(system.file("shape/nc.shp", package = "sf"))
  df <- as_sedonadb_dataframe(nc)

  # Compare attributes separately
  expect_true(sf::st_as_sf(df) |> sf::st_crs() == nc |> sf::st_crs())
  expect_equal(
    sf::st_as_sf(df) |> sf::st_set_crs(NA) |> as.data.frame(),
    nc |> sf::st_set_crs(NA) |> as.data.frame()
  )
})
