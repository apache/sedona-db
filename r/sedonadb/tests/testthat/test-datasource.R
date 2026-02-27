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

test_that("sd_read_sf() works for layers with named geometry columns", {
  skip_if_not_installed("sf")

  nc_gpkg <- system.file("gpkg/nc.gpkg", package = "sf")

  from_stream <- sf::st_as_sf(sd_read_sf(nc_gpkg))
  from_sf <- sf::st_read(nc_gpkg, quiet = TRUE)

  # Expect identical CRS
  expect_true(sf::st_crs(from_stream) == sf::st_crs(from_sf))

  # Expect identical content without CRS
  expect_equal(
    from_stream |> sf::st_set_crs(NA) |> as.data.frame(),
    from_sf |> sf::st_set_crs(NA) |> as.data.frame()
  )
})

test_that("sd_read_sf() works for layers with unnamed geometry columns", {
  skip_if_not_installed("sf")

  nc_shp <- system.file("shape/nc.shp", package = "sf")

  from_stream <- sf::st_as_sf(sd_read_sf(nc_shp))
  from_sf <- sf::st_read(nc_shp, quiet = TRUE, promote_to_multi = FALSE)

  # Expect identical CRS
  expect_true(sf::st_crs(from_stream) == sf::st_crs(from_sf))

  # The from_stream version has a geometry column named "wkb_geometry" but
  # sf renames this internally to "geometry"
  expect_true("wkb_geometry" %in% names(from_stream))
  colnames(from_stream)[colnames(from_stream) == "wkb_geometry"] <- "geometry"
  sf::st_geometry(from_stream) <- "geometry"

  # Expect identical content without CRS
  expect_equal(
    from_stream |> sf::st_set_crs(NA) |> as.data.frame(),
    from_sf |> sf::st_set_crs(NA) |> as.data.frame()
  )
})

test_that("sd_read_sf() works for database dsns / non-default layers", {
  skip_if_not_installed("sf")

  # Can be tested using docker compose up with
  # postgresql://localhost:5432/postgres?user=postgres&password=password
  test_uri <- Sys.getenv("SEDONADB_POSTGRESQL_TEST_URI", unset = "")
  if (identical(test_uri, "")) {
    skip("SEDONADB_POSTGRESQL_TEST_URI is not set")
  }

  nc_gpkg <- system.file("gpkg/nc.gpkg", package = "sf")
  sf::st_write(
    sf::st_read(nc_gpkg, quiet = TRUE),
    test_uri,
    "test_sf_nc",
    append = FALSE,
    driver = "PostgreSQL",
    quiet = TRUE
  )

  from_stream <- sf::st_as_sf(sd_read_sf(test_uri, "test_sf_nc"))
  from_sf <- sf::st_read(test_uri, "test_sf_nc", quiet = TRUE)

  # Expect identical CRS
  expect_true(sf::st_crs(from_stream) == sf::st_crs(from_sf))

  # Expect identical content without CRS
  expect_equal(
    from_stream |> sf::st_set_crs(NA) |> as.data.frame(),
    from_sf |> sf::st_set_crs(NA) |> as.data.frame()
  )
})

test_that("sd_read_sf() works with filter", {
  skip_if_not_installed("sf")

  nc_gpkg <- system.file("gpkg/nc.gpkg", package = "sf")
  filter <- wk::rct(-77.901, 36.162, -77.075, 36.556, crs = sf::st_crs("NAD27"))

  from_stream <- sf::st_as_sf(sd_read_sf(nc_gpkg, filter = filter))
  from_sf <- sf::st_read(nc_gpkg, quiet = TRUE, wkt_filter = wk::as_wkt(filter))

  # Expect identical CRS
  expect_true(sf::st_crs(from_stream) == sf::st_crs(from_sf))

  # Expect identical content without CRS
  expect_equal(
    from_stream |> sf::st_set_crs(NA) |> as.data.frame(),
    from_sf |> sf::st_set_crs(NA) |> as.data.frame()
  )

  # Check for error if filtered with an invalid CRS
  wk::wk_crs(filter) <- NULL
  expect_snapshot_error(sd_read_sf(nc_gpkg, filter = filter))
})
