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

# Additional edge cases for sd_parse_crs

test_that("sd_parse_crs handles NULL input", {
  expect_error(
    sedonadb:::sd_parse_crs(NULL),
    "must be character"
  )
})

test_that("sd_parse_crs handles empty string", {
  expect_snapshot(
    sedonadb:::sd_parse_crs(""),
    error = TRUE
  )
})

test_that("sd_parse_crs handles CRS with only name, no ID", {
  meta <- '{
    "crs": {
      "type": "GeographicCRS",
      "name": "Custom Geographic CRS"
    }
  }'
  expect_snapshot(sedonadb:::sd_parse_crs(meta))
})

test_that("sd_parse_crs handles OGC:CRS84", {
  # Common case in GeoParquet/GeoArrow

  meta <- '{"crs": "OGC:CRS84"}'
  expect_snapshot(sedonadb:::sd_parse_crs(meta))
})

# Explicit tests for Rust wrappers to ensure uppercase casing

test_that("SedonaTypeR$crs_display() uses uppercase authority codes", {
  df <- sd_sql("SELECT ST_SetSRID(ST_Point(1, 2), 4326) as geom")
  schema <- nanoarrow::infer_nanoarrow_schema(df)
  sd_type <- sedonadb:::SedonaTypeR$new(schema$children$geom)
  expect_snapshot(sd_type$crs_display())

  df5070 <- sd_sql("SELECT ST_SetSRID(ST_Point(1, 2), 5070) as geom")
  sd_type5070 <- sedonadb:::SedonaTypeR$new(nanoarrow::infer_nanoarrow_schema(df5070)$children$geom)
  expect_snapshot(sd_type5070$crs_display())
})

test_that("SedonaCrsR$display() uses uppercase authority codes", {
  df <- sd_sql("SELECT ST_SetSRID(ST_Point(1, 2), 4326) as geom")
  sd_type <- sedonadb:::SedonaTypeR$new(nanoarrow::infer_nanoarrow_schema(df)$children$geom)
  crs <- sd_type$crs()
  expect_snapshot(crs$display())
})

# CRS preservation through data creation paths

test_that("CRS is preserved when creating from data.frame with geometry", {
  df <- as_sedonadb_dataframe(
    data.frame(
      geom = wk::as_wkb(wk::wkt("POINT (0 1)", crs = "EPSG:32620"))
    )
  )

  re_df <- sd_collect(df)
  crs <- wk::wk_crs(re_df$geom)
  expect_false(is.null(crs))
  # Check that the CRS contains EPSG:32620 info
  expect_true(
    grepl("32620", as.character(crs)) ||
      grepl("32620", jsonlite::toJSON(crs, auto_unbox = TRUE))
  )
})

test_that("CRS is preserved through nanoarrow stream roundtrip", {
  r_df <- data.frame(
    geom = wk::as_wkb(wk::wkt("POINT (0 1)", crs = "EPSG:4326"))
  )

  stream <- nanoarrow::as_nanoarrow_array_stream(r_df)
  df <- as_sedonadb_dataframe(stream, lazy = FALSE)
  re_df <- sd_collect(df)

  crs <- wk::wk_crs(re_df$geom)
  expect_false(is.null(crs))
})

test_that("Different CRS values are preserved independently", {
  # Create geometry with non-default CRS
  df <- sd_sql(
    "
    SELECT
      ST_SetSRID(ST_Point(1, 2), 4326) as geom_wgs84,
      ST_SetSRID(ST_Point(3, 4), 32632) as geom_utm
  "
  )

  re_df <- sd_collect(df)

  # Both geometries should have CRS metadata
  crs1 <- wk::wk_crs(re_df$geom_wgs84)
  crs2 <- wk::wk_crs(re_df$geom_utm)

  expect_false(is.null(crs1))
  expect_false(is.null(crs2))
})

# Parquet roundtrip with CRS

test_that("CRS is preserved through parquet write/read", {
  df <- sd_sql("SELECT ST_SetSRID(ST_Point(1, 2), 4326) as geom")

  tmp_parquet_file <- tempfile(fileext = ".parquet")
  on.exit(unlink(tmp_parquet_file))

  sd_write_parquet(df, tmp_parquet_file)
  df_roundtrip <- sd_read_parquet(tmp_parquet_file)
  re_df <- sd_collect(df_roundtrip)

  # Verify geometry has CRS
  crs <- wk::wk_crs(re_df$geom)
  expect_false(is.null(crs))
})

test_that("Non-standard CRS is preserved through parquet roundtrip", {
  # Use a less common SRID (NAD83 / Conus Albers)
  df <- sd_sql("SELECT ST_SetSRID(ST_Point(1, 2), 5070) as geom")

  tmp_parquet_file <- tempfile(fileext = ".parquet")
  on.exit(unlink(tmp_parquet_file))

  sd_write_parquet(df, tmp_parquet_file)
  df_roundtrip <- sd_read_parquet(tmp_parquet_file)
  re_df <- sd_collect(df_roundtrip)

  crs <- wk::wk_crs(re_df$geom)
  expect_false(is.null(crs))
  # Check CRS info contains 5070
  crs_str <- jsonlite::toJSON(crs, auto_unbox = TRUE)
  expect_true(grepl("5070", crs_str) || grepl("Albers", crs_str))
})

# Multiple geometry columns with different CRS through operations

test_that("Multiple geometry columns preserve their CRS after operations", {
  df <- sd_sql(
    "
    SELECT
      ST_SetSRID(ST_Point(1, 2), 4326) as point_a,
      ST_SetSRID(ST_Point(3, 4), 5070) as point_b,
      'test' as name
  "
  )

  # Collect and check both CRS are preserved

  re_df <- sd_collect(df)

  crs_a <- wk::wk_crs(re_df$point_a)
  crs_b <- wk::wk_crs(re_df$point_b)

  expect_false(is.null(crs_a))
  expect_false(is.null(crs_b))
  # They should be different
  expect_false(identical(crs_a, crs_b))
})
