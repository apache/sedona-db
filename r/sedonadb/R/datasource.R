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

read_sf_internal <- function(
  dsn,
  layer = NULL,
  ...,
  query = NA,
  options = NULL,
  drivers = character(0),
  filter = NULL,
  fid_column_name = character(0),
  ctx = NULL,
  lazy = FALSE
) {
  check_dots_empty(..., label = "sd_read_sf_stream()")

  layer <- if (is.null(layer)) {
    character(0)
  } else {
    enc2utf8(layer)
  }

  if (nchar(dsn) == 0) {
    stop("Expected non-empty value for dsn")
  }

  dsn_exists <- file.exists(dsn)

  # A heuristic to catch common database DSNs so that we don't try to normalize
  # them as file paths
  dsn_isdb <- grepl("^(pg|mssql|pgeo|odbc):", tolower(dsn))

  # Normalize (e.g., replace ~) and ensure internal encoding is UTF-8
  if (length(dsn) == 1 && dsn_exists && !dsn_isdb) {
    dsn <- enc2utf8(normalizePath(dsn))
  }

  options <- as.character(options)

  if (!is.null(filter)) {
    filter <- wk::as_wkt(filter)
    if (length(filter) != 1){
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

  # sf doesn't currently support GEOMETRY_METADATA_ENCODING=GEOARROW, so we
  # need to post-process the stream

  # Check that the filter's CRS matched the dataset CRS
  # if (!is.null(filter)) {
  #   filter_crs <- wk::wk_crs(filter)
  #   for (crs in info[[2]]) {
  #     if (!wk::wk_crs_equal(sf::st_crs(crs), filter_crs)) {
  #       stop("filter CRS is not equal to geometry column CRS")
  #     }
  #   }
  # }

  # Return the stream. The lifecycle of the underlying GDAL dataset is tied
  # to the lifecycle of the stream (i.e., when the stream id released, the
  # dataset object will be closed)
  stream

  if (is.null(ctx)) {
    ctx <- ctx()
  }

  df <- ctx$data_frame_from_array_stream(stream, collect_now = !lazy)
  new_sedonadb_dataframe(ctx, df)
}
