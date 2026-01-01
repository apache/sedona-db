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

test_that("sd_parse_crs works for GeoArrow metadata with EPSG", {
  meta <- '{"crs": {"id": {"authority": "EPSG", "code": 5070}, "name": "NAD83 / Conus Albers"}}'
  parsed <- sedonadb:::sd_parse_crs(meta)
  expect_identical(parsed$authority_code, "EPSG:5070")
  expect_identical(parsed$srid, 5070L)
  expect_identical(parsed$name, "NAD83 / Conus Albers")
  # The proj_string is the *unwrapped* and *minified* PROJJSON content
  expect_match(parsed$proj_string, '"authority":"EPSG"', fixed = TRUE)
  expect_match(parsed$proj_string, '"code":5070', fixed = TRUE)
})

test_that("sd_parse_crs works for Engineering CRS (no EPSG ID)", {
  # A realistic example of a local engineering CRS that wouldn't have an EPSG code
  meta <- '{
    "crs": {
      "type": "EngineeringCRS",
      "name": "Construction Site Local Grid",
      "datum": {
        "type": "EngineeringDatum",
        "name": "Local Datum"
      },
      "coordinate_system": {
        "subtype": "Cartesian",
        "axis": [
          {"name": "Northing", "abbreviation": "N", "direction": "north", "unit": "metre"},
          {"name": "Easting", "abbreviation": "E", "direction": "east", "unit": "metre"}
        ]
      }
    }
  }'
  parsed <- sedonadb:::sd_parse_crs(meta)
  expect_null(parsed$authority_code)
  expect_null(parsed$srid)
  expect_identical(parsed$name, "Construction Site Local Grid")
  expect_true(!is.null(parsed$proj_string))
})

test_that("sd_parse_crs returns NULL if crs field is missing", {
  expect_null(sedonadb:::sd_parse_crs('{"something_else": 123}'))
  expect_null(sedonadb:::sd_parse_crs('{}'))
})

test_that("sd_parse_crs handles invalid JSON gracefully", {
  expect_error(
    sedonadb:::sd_parse_crs('invalid json'),
    "Failed to parse metadata JSON"
  )
})

test_that("sd_parse_crs works with plain strings if that's what's in 'crs'", {
  meta <- '{"crs": "EPSG:4326"}'
  parsed <- sedonadb:::sd_parse_crs(meta)
  # Note: PROJ/sedona normalizes EPSG:4326 (lat/lon) to OGC:CRS84 (lon/lat)
  # for consistent axis order in WKT/GeoJSON contexts.
  expect_identical(parsed$authority_code, "OGC:CRS84")
  expect_identical(parsed$srid, 4326L)
  expect_true(!is.null(parsed$proj_string))
})
