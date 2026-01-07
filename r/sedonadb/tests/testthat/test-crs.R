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
  expect_snapshot(sedonadb:::sd_parse_crs(meta))
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
  expect_snapshot(sedonadb:::sd_parse_crs(meta))
})

test_that("sd_parse_crs returns NULL if crs field is missing", {
  expect_snapshot(sedonadb:::sd_parse_crs('{"something_else": 123}'))
  expect_snapshot(sedonadb:::sd_parse_crs('{}'))
})

test_that("sd_parse_crs handles invalid JSON gracefully", {
  expect_snapshot(
    sedonadb:::sd_parse_crs('invalid json'),
    error = TRUE
  )
})

test_that("sd_parse_crs works with plain strings if that's what's in 'crs'", {
  meta <- '{"crs": "EPSG:4326"}'
  expect_snapshot(sedonadb:::sd_parse_crs(meta))
})

# Tests for CRS display in print.sedonadb_dataframe

test_that("print.sedonadb_dataframe shows CRS info for geometry column with EPSG", {
  df <- sd_sql("SELECT ST_SetSRID(ST_Point(1, 2), 4326) as geom")
  expect_snapshot(print(df, n = 0))
})

test_that("print.sedonadb_dataframe shows CRS info with different SRID", {
  df <- sd_sql("SELECT ST_SetSRID(ST_Point(1, 2), 5070) as geom")
  expect_snapshot(print(df, n = 0))
})

test_that("print.sedonadb_dataframe shows multiple geometry columns with CRS", {
  df <- sd_sql(
    "
    SELECT
      ST_SetSRID(ST_Point(1, 2), 4326) as geom1,
      ST_SetSRID(ST_Point(3, 4), 5070) as geom2
  "
  )
  expect_snapshot(print(df, n = 0))
})

test_that("print.sedonadb_dataframe handles geometry without explicit CRS", {
  # ST_Point without ST_SetSRID may not have CRS metadata
  df <- sd_sql("SELECT ST_Point(1, 2) as geom")
  expect_snapshot(print(df, n = 0))
})

test_that("print.sedonadb_dataframe respects width parameter for geometry line", {
  df <- sd_sql(
    "
    SELECT
      ST_SetSRID(ST_Point(1, 2), 4326) as very_long_geometry_column_name_1,
      ST_SetSRID(ST_Point(3, 4), 4326) as very_long_geometry_column_name_2
  "
  )
  # Use a narrow width to trigger truncation
  expect_snapshot(print(df, n = 0, width = 60))
})
