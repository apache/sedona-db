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

# Tests for CRS display in print.sedonadb_dataframe (lines 325-360 of dataframe.R)

test_that("print.sedonadb_dataframe shows CRS info for geometry column with EPSG", {
  df <- sd_sql("SELECT ST_SetSRID(ST_Point(1, 2), 4326) as geom")
  output <- capture.output(print(df, n = 0))

  # Check that the Geometry line is present

  geo_line <- grep("^# Geometry:", output, value = TRUE)
  expect_length(geo_line, 1)

  # Should show CRS information (OGC:CRS84 or EPSG:4326)
  expect_match(geo_line, "geom .*(CRS: OGC:CRS84|CRS: EPSG:4326)")
})

test_that("print.sedonadb_dataframe shows CRS info with different SRID", {
  df <- sd_sql("SELECT ST_SetSRID(ST_Point(1, 2), 5070) as geom")
  output <- capture.output(print(df, n = 0))

  geo_line <- grep("^# Geometry:", output, value = TRUE)
  expect_length(geo_line, 1)
  expect_match(geo_line, "geom .*(CRS: EPSG:5070|CRS:.*5070)")
})

test_that("print.sedonadb_dataframe shows multiple geometry columns with CRS", {
  df <- sd_sql(
    "
    SELECT
      ST_SetSRID(ST_Point(1, 2), 4326) as geom1,
      ST_SetSRID(ST_Point(3, 4), 5070) as geom2
  "
  )
  output <- capture.output(print(df, n = 0))

  geo_line <- grep("^# Geometry:", output, value = TRUE)
  expect_length(geo_line, 1)
  # Should contain both geometry columns
  expect_match(geo_line, "geom1")
  expect_match(geo_line, "geom2")
})

test_that("print.sedonadb_dataframe handles geometry without explicit CRS", {
  # ST_Point without ST_SetSRID may not have CRS metadata
  df <- sd_sql("SELECT ST_Point(1, 2) as geom")
  output <- capture.output(print(df, n = 0))

  # May or may not have a Geometry line depending on extension metadata
  # At least it should not error
  expect_true(any(grepl("sedonadb_dataframe", output)))
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
  output <- capture.output(print(df, n = 0, width = 60))

  geo_line <- grep("^# Geometry:", output, value = TRUE)
  if (length(geo_line) > 0) {
    # Line should be truncated with "..."
    expect_lte(nchar(geo_line), 60)
    expect_match(geo_line, "\\.\\.\\.$")
  }
})
