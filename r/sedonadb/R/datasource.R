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

#' Read GDAL/OGR via the sf package
#'
#' Uses the ArrowArrayStream interface to GDAL exposed via the sf package
#' to read GDAL/OGR-based data sources.
#'
#' @param ctx A SedonaDB context created using [sd_connect()].
#' @param dsn,layer Description of datasource and layer. See [sf::read_sf()]
#'   for details.
#' @param ... Currently unused and must be empty
#' @param query A SQL query to pass on to GDAL/OGR.
#' @param options A character vector with layer open options in the
#'   form "KEY=VALUE".
#' @param drivers A list of drivers to try if the dsn cannot be guessed.
#' @param filter A spatial object that may be used to filter while reading.
#'   In the future SedonaDB will automatically calculate this value based on
#'   the query. May be any spatial object that can be converted to WKT via
#'   [wk::as_wkt()]. This filter's CRS must match that of the data.
#' @param fid_column_name An optional name for the feature id (FID) column.
#' @param lazy Use `TRUE` to stream the data from the source rather than collect
#'   first. This can be faster for large data sources but can also be confusing
#'   because the data may only be scanned exactly once.
#'
#' @returns A SedonaDB DataFrame.
#' @export
#'
#' @examples
#' nc_gpkg <- system.file("gpkg/nc.gpkg", package = "sf")
#' sd_read_sf(nc_gpkg)
#'
sd_read_sf <- function(
  dsn,
  layer = NULL,
  ...,
  query = NA,
  options = NULL,
  drivers = NULL,
  filter = NULL,
  fid_column_name = NULL,
  lazy = FALSE
) {
  sd_ctx_read_sf(
    ctx(),
    dsn = dsn,
    layer = layer,
    ...,
    query = query,
    options = options,
    drivers = drivers,
    filter = filter,
    fid_column_name = fid_column_name,
    lazy = lazy
  )
}

#' @rdname sd_read_sf
#' @export
sd_ctx_read_sf <- function(
  ctx,
  dsn,
  layer = NULL,
  ...,
  query = NA,
  options = NULL,
  drivers = NULL,
  filter = NULL,
  fid_column_name = NULL,
  lazy = FALSE
) {
  stream <- read_sf_stream(
    dsn = dsn,
    layer = layer,
    ...,
    query = query,
    options = options,
    drivers = drivers,
    filter = filter,
    fid_column_name = fid_column_name
  )

  df <- ctx$data_frame_from_array_stream(stream, collect_now = !lazy)
  new_sedonadb_dataframe(ctx, df)
}


read_sf_stream <- function(
  dsn,
  layer = NULL,
  ...,
  query = NA,
  options = NULL,
  drivers = NULL,
  filter = NULL,
  fid_column_name = NULL
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
  dsn_is_http <- grepl("^https://", dsn)

  # Normalize (e.g., replace ~) and ensure internal encoding is UTF-8
  if (length(dsn) == 1 && dsn_exists && !dsn_isdb && !dsn_is_http) {
    dsn <- enc2utf8(normalizePath(dsn))

    if (endsWith(dsn, ".zip")) {
      dsn <- paste0("/vsizip/", dsn)
    }
  }

  if (dsn_is_http) {
    dsn <- paste0("/vsicurl/", enc2utf8(dsn))

    if (endsWith(dsn, ".zip")) {
      dsn <- paste0("/vsizip/", dsn)
    }
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
  read_fn <- asNamespace("sf")[["CPL_read_gdal_stream"]]
  info <- read_fn(
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
    filter_crs <- wk::wk_crs(filter)

    for (column_crs in info[[2]]) {
      column_crs_sf <- sf::st_crs(column_crs)
      if (!wk::wk_crs_equal(filter_crs, column_crs_sf)) {
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
