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

#' Convert an object to a DataFrame
#'
#' @param x An object to convert
#' @param ... Extra arguments passed to/from methods
#' @param schema The requested schema
#'
#' @returns A sedonadb_dataframe
#' @export
#'
#' @examples
#' as_sedonadb_dataframe(data.frame(x = 1:3))
#'
as_sedonadb_dataframe <- function(x, ..., schema = NULL) {
  UseMethod("as_sedonadb_dataframe")
}

#' @export
as_sedonadb_dataframe.sedonadb_dataframe <- function(x, ..., schema = NULL) {
  # In the future, schema can be handled with a cast
  x
}

#' @export
as_sedonadb_dataframe.data.frame <- function(x, ..., schema = NULL) {
  array <- nanoarrow::as_nanoarrow_array(x, schema = schema)
  stream <- nanoarrow::basic_array_stream(list(array))
  ctx <- ctx()
  df <- ctx$data_frame_from_array_stream(stream, collect_now = TRUE)
  new_sedonadb_dataframe(ctx, df)
}

#' @export
as_sedonadb_dataframe.nanoarrow_array <- function(x, ..., schema = NULL) {
  stream <- nanoarrow::as_nanoarrow_array_stream(x, schema = schema)
  ctx <- ctx()
  df <- ctx$data_frame_from_array_stream(stream, collect_now = TRUE)

  # Verify schema is handled
  as_sedonadb_dataframe(new_sedonadb_dataframe(ctx, df), schema = schema)
}

#' @export
as_sedonadb_dataframe.nanoarrow_array_stream <- function(x, ..., schema = NULL,
                                                         lazy = TRUE) {
  stream <- nanoarrow::as_nanoarrow_array_stream(x, schema = schema)
  ctx <- ctx()
  df <- ctx$data_frame_from_array_stream(stream, collect_now = !lazy)

  # Verify schema is handled
  as_sedonadb_dataframe(new_sedonadb_dataframe(ctx, df), schema = schema)
}

#' @export
as_sedonadb_dataframe.datafusion_table_provider <- function(x, ..., schema = NULL) {
  ctx <- ctx()
  df <- ctx$data_frame_from_table_provider(x)
  new_sedonadb_dataframe(ctx, df)
}

#' Count rows in a DataFrame
#'
#' @param .data A sedonadb_dataframe
#'
#' @returns The number of rows after executing the query
#' @export
#'
#' @examples
#' sd_sql("SELECT 1 as one") |> sd_count()
#'
sd_count <- function(.data) {
  .data$df$count()
}

#' Register a DataFrame as a named view
#'
#' This is useful for creating a view that can be referenced in a SQL
#' statement. Use [sd_drop_view()] to remove it.
#'
#' @inheritParams sd_count
#' @inheritParams sd_drop_view
#' @param overwrite Use TRUE to overwrite a view with the same name (if it exists)
#'
#' @returns .data, invisibly
#' @export
#'
#' @examples
#' sd_sql("SELECT 1 as one") |> sd_to_view("foofy")
#' sd_sql("SELECT * FROM foofy")
#'
sd_to_view <- function(.data, table_ref, overwrite = FALSE) {
  .data <- as_sedonadb_dataframe(.data)
  .data$df$to_view(.data$ctx, table_ref, overwrite)
  invisible(.data)
}

#' Collect a DataFrame into memory
#'
#' Use `sd_compute()` to collect and return the result as a DataFrame;
#' use `sd_collect()` to collect and return the result as an R data.frame.
#'
#' @inheritParams sd_count
#' @param ptype The target R object. See [nanoarrow::convert_array_stream].
#'
#' @returns `sd_compute()` returns a sedonadb_dataframe; `sd_collect()` returns
#'   a data.frame (or subclass according to `ptype`).
#' @export
#'
#' @examples
#' sd_sql("SELECT 1 as one") |> sd_compute()
#' sd_sql("SELECT 1 as one") |> sd_collect()
#'
sd_compute <- function(.data) {
  .data <- as_sedonadb_dataframe(.data)
  df <- .data$df$compute(.data$ctx)
  new_sedonadb_dataframe(.data$ctx, df)
}

#' @export
#' @rdname sd_compute
sd_collect <- function(.data, ptype = NULL) {
  .data <- as_sedonadb_dataframe(.data)
  stream <- nanoarrow::nanoarrow_allocate_array_stream()
  size <- .data$df$collect(stream)
  nanoarrow::convert_array_stream(stream, size = size, to = ptype)
}

#' Preview and print the results of running a query
#'
#' This is used to implement `print()` for the sedonadb_dataframe or can
#' be used to explicitly preview if `options(sedonadb.interactive = FALSE)`.
#'
#' @inheritParams sd_count
#' @param n The number of rows to preview. Use `Inf` to preview all rows.
#'   Defaults to `getOption("pillar.print_max")`.
#' @param ascii Use `TRUE` to force ASCII table formatting or `FALSE` to force
#'   unicode formatting. By default, use a heuristic to determine if the output
#'   is unicode-friendly or the value of `getOption("cli.unicode")`.
#' @param width The character width of the output. Defaults to
#'   `getOption("width")`.
#'
#' @returns .data, invisibly
#' @export
#'
#' @examples
#' sd_sql("SELECT 1 as one") |> sd_preview()
#'
sd_preview <- function(.data, n = NULL, ascii = NULL, width = NULL) {
  .data <- as_sedonadb_dataframe(.data)

  if (is.null(width)) {
    width <- getOption("width")
  }

  if (is.null(n)) {
    n <- getOption("pillar.print_max", 6)
  }

  if (is.null(ascii)) {
    ascii <- !is_utf8_output()
  }

  content <- .data$df$show(
    .data$ctx,
    width_chars = as.integer(width),
    limit = as.double(n),
    ascii = ascii
  )

  cat(content)
  cat(paste0("Preview of up to ", n, " row(s)\n"))

  invisible(.data)
}

#' Write DataFrame to (Geo)Parquet files
#'
#' Write this DataFrame to one or more (Geo)Parquet files. For input that contains
#' geometry columns, GeoParquet metadata is written such that suitable readers can
#' recreate Geometry/Geography types when reading the output and potentially read
#' fewer row groups when only a subset of the file is needed for a given query.
#'
#' @inheritParams sd_count
#' @param path A filename or directory to which parquet file(s) should be written
#' @param partition_by A character vector of column names to partition by. If non-empty,
#'   applies hive-style partitioning to the output
#' @param sort_by A character vector of column names to sort by. Currently only
#'   ascending sort is supported
#' @param single_file_output Use TRUE or FALSE to force writing a single Parquet
#'   file vs. writing one file per partition to a directory. By default,
#'   a single file is written if `partition_by` is unspecified and
#'   `path` ends with `.parquet`
#' @param geoparquet_version GeoParquet metadata version to write if output contains
#'   one or more geometry columns. The default ("1.0") is the most widely
#'   supported and will result in geometry columns being recognized in many
#'   readers; however, only includes statistics at the file level.
#'   Use "1.1" to compute an additional bounding box column
#'   for every geometry column in the output: some readers can use these columns
#'   to prune row groups when files contain an effective spatial ordering.
#'   The extra columns will appear just before their geometry column and
#'   will be named "[geom_col_name]_bbox" for all geometry columns except
#'   "geometry", whose bounding box column name is just "bbox"
#' @param overwrite_bbox_columns Use TRUE to overwrite any bounding box columns
#'   that already exist in the input. This is useful in a read -> modify
#'   -> write scenario to ensure these columns are up-to-date. If FALSE
#'   (the default), an error will be raised if a bbox column already exists
#'
#' @returns The input, invisibly
#' @export
#'
#' @examples
#' tmp_parquet <- tempfile(fileext = ".parquet")
#'
#' sd_sql("SELECT ST_SetSRID(ST_Point(1, 2), 4326) as geom") |>
#'   sd_write_parquet(tmp_parquet)
#'
#' sd_read_parquet(tmp_parquet)
#' unlink(tmp_parquet)
#'
sd_write_parquet <- function(.data,
                             path,
                             partition_by = character(0),
                             sort_by = character(0),
                             single_file_output = NULL,
                             geoparquet_version = "1.0",
                             overwrite_bbox_columns = FALSE) {

  # Determine single_file_output default based on path and partition_by
  if (is.null(single_file_output)) {
    single_file_output <- length(partition_by) == 0 && grepl("\\.parquet$", path)
  }

  # Validate geoparquet_version
  if (!is.null(geoparquet_version)) {
    if (!geoparquet_version %in% c("1.0", "1.1")) {
      stop("geoparquet_version must be '1.0' or '1.1'")
    }
  }

  # Call the underlying Rust method
  .data$df$to_parquet(
    ctx = .data$ctx,
    path = path,
    partition_by = partition_by,
    sort_by = sort_by,
    single_file_output = single_file_output,
    overwrite_bbox_columns = overwrite_bbox_columns,
    geoparquet_version = geoparquet_version
  )

  invisible(.data)
}

new_sedonadb_dataframe <- function(ctx, internal_df) {
  structure(list(ctx = ctx, df = internal_df), class = "sedonadb_dataframe")
}

#' @importFrom utils head
#' @export
head.sedonadb_dataframe <- function(x, n = 6L, ...) {
  new_sedonadb_dataframe(x$ctx, x$df$limit(as.double(n)))
}

#' @export
dimnames.sedonadb_dataframe <- function(x, ...) {
  list(NULL, names(infer_nanoarrow_schema(x)$children))
}

#' @export
dim.sedonadb_dataframe <- function(x, ...) {
  c(NA_integer_, length(infer_nanoarrow_schema(x)$children))
}

#' @export
as.data.frame.sedonadb_dataframe <- function(x, ...) {
  stream <- nanoarrow::nanoarrow_allocate_array_stream()
  size <- x$df$collect(stream)
  nanoarrow::convert_array_stream(stream, size = size)
}

#' @importFrom nanoarrow infer_nanoarrow_schema
#' @export
infer_nanoarrow_schema.sedonadb_dataframe <- function(x, ...) {
  schema <- nanoarrow::nanoarrow_allocate_schema()
  x$df$to_arrow_schema(schema)
  schema
}

#' @importFrom nanoarrow as_nanoarrow_array_stream
#' @export
as_nanoarrow_array_stream.sedonadb_dataframe <- function(x, ..., schema = NULL) {
  if (!is.null(schema)) {
    schema <- nanoarrow::as_nanoarrow_schema(schema)
  }

  stream <- nanoarrow::nanoarrow_allocate_array_stream()
  x$df$to_arrow_stream(stream, schema)
  stream
}

#' @export
print.sedonadb_dataframe <- function(x, ..., width = NULL, n = NULL) {
  if (isTRUE(getOption("sedonadb.interactive", TRUE))) {
    sd_preview(x, n = n, width = width)
  } else {
    sd_preview(x, n = 0)
    cat("Use options(sedonadb.interactive = TRUE) or use sd_preview() to print\n")
  }

  invisible(x)
}

# Borrowed from cli but without detecting LaTeX output.
is_utf8_output <- function() {
  opt <- getOption("cli.unicode", NULL)
  if (!is.null(opt)) {
    isTRUE(opt)
  } else {
    l10n_info()$`UTF-8`
  }
}
