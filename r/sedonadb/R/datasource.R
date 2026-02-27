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

read_sf_stream <- function(
  dsn,
  layer = NULL,
  ...,
  query = NA,
  options = NULL,
  drivers = NULL,
  filter = NULL,
  fid_column_name = NULL,
  ctx = NULL,
  lazy = FALSE
) {
  check_dots_empty(..., label = "sd_read_sf_stream()")

  if (is.null(layer)) {
    layer <- character(0)
  } else {
    layer <- enc2utf8(layer)
  }

  if (nchar(dsn) == 0) {
    stop("Expected non-empty value for dsn")
  }

  dsn_exists <- file.exists(dsn)

  # A heuristic to catch common database DSNs so that we don't try to normalize
  # them as file paths
  dsn_isdb <- grepl("^(pg|mssql|pgeo|odbc|postgresql):", tolower(dsn))

  # Normalize (e.g., replace ~) and ensure internal encoding is UTF-8
  if (length(dsn) == 1 && dsn_exists && !dsn_isdb) {
    dsn <- enc2utf8(normalizePath(dsn))
  }

  # Rcpp expects these to be character vectors
  options <- as.character(options)
  drivers <- as.character(drivers)
  fid_column_name <- as.character(fid_column_name)

  if (!is.null(filter)) {
    filter <- wk::as_wkt(filter)
    if (length(filter) != 1) {
      stop("Filter must be a geometry-like object of length one")
    }
  } else {
    filter <- character(0)
  }

  stream <- nanoarrow::nanoarrow_allocate_array_stream()
  info <- sf:::CPL_read_gdal_stream(
    stream,
    dsn,
    layer,
    query,
    options,
    TRUE, # quiet
    drivers,
    filter,
    dsn_exists,
    dsn_isdb,
    fid_column_name,
    getOption("width")
  )

  # Check filter for CRS equality
  if (!identical(filter, character())) {
    filter_crs <- sf::st_crs(filter)

    for (column_crs in info[[2]]) {
      column_crs_sf <- sf::st_crs(column_crs)
      if (filter_crs != column_crs_sf) {
        stop(
          sprintf(
            "filter crs (%s) does not match output CRS (%s)",
            format(filter_crs),
            format(column_crs_sf)
          )
        )
      }
    }
  }

  # sf doesn't currently support GEOMETRY_METADATA_ENCODING=GEOARROW, so we
  # need to post-process the stream to ensure the CRS is set on the output
  geometry_column_names <- info[[1]]
  geometry_column_crses <- vapply(
    info[[2]],
    function(x) wk::wk_crs_projjson(sf::st_crs(x)),
    character(1)
  )

  # The sf implementation assigns the "missing" geometry column name "geometry"
  # where the name in the schema is "wkb_geometry".
  if (
    "geometry" %in% geometry_column_names && !("wkb_geometry" %in% geometry_column_names)
  ) {
    geometry_column_names <- c(geometry_column_names, "wkb_geometry")
    geometry_column_crses <- c(
      geometry_column_crses,
      geometry_column_crses[geometry_column_names == "geometry"]
    )
  }

  stream_out <- nanoarrow::nanoarrow_allocate_array_stream()
  apply_crses_to_sf_stream(
    stream,
    geometry_column_names,
    geometry_column_crses,
    stream_out
  )

  stream_out
}
