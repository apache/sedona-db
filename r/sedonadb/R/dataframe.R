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
#' @param ctx A SedonaDB context. This should always be passed to inner calls
#'   to SedonaDB functions; NULL implies the global context.
#' @param ... Extra arguments passed to/from methods
#' @param schema The requested schema
#'
#' @returns A sedonadb_dataframe
#' @export
#'
#' @examples
#' as_sedonadb_dataframe(data.frame(x = 1:3))
#'
as_sedonadb_dataframe <- function(x, ..., schema = NULL, ctx = NULL) {
  UseMethod("as_sedonadb_dataframe")
}

#' @export
as_sedonadb_dataframe.sedonadb_dataframe <- function(x, ..., schema = NULL, ctx = NULL) {
  # In the future, schema can be handled with a cast
  x
}

#' @export
as_sedonadb_dataframe.data.frame <- function(x, ..., schema = NULL, ctx = NULL) {
  array <- nanoarrow::as_nanoarrow_array(x, schema = schema)
  stream <- nanoarrow::basic_array_stream(list(array))

  if (is.null(ctx)) {
    ctx <- ctx()
  }

  df <- ctx$data_frame_from_array_stream(stream, collect_now = TRUE)
  new_sedonadb_dataframe(ctx, df)
}

#' @export
as_sedonadb_dataframe.nanoarrow_array <- function(x, ..., schema = NULL, ctx = NULL) {
  stream <- nanoarrow::as_nanoarrow_array_stream(x, schema = schema)

  if (is.null(ctx)) {
    ctx <- ctx()
  }

  df <- ctx$data_frame_from_array_stream(stream, collect_now = TRUE)

  # Verify schema is handled
  as_sedonadb_dataframe(new_sedonadb_dataframe(ctx, df), schema = schema)
}

#' @export
as_sedonadb_dataframe.nanoarrow_array_stream <- function(
  x,
  ...,
  schema = NULL,
  lazy = TRUE,
  ctx = NULL
) {
  stream <- nanoarrow::as_nanoarrow_array_stream(x, schema = schema)

  if (is.null(ctx)) {
    ctx <- ctx()
  }

  df <- ctx$data_frame_from_array_stream(stream, collect_now = !lazy)

  # Verify schema is handled
  as_sedonadb_dataframe(new_sedonadb_dataframe(ctx, df), schema = schema)
}

#' @export
as_sedonadb_dataframe.datafusion_table_provider <- function(
  x,
  ...,
  schema = NULL,
  ctx = NULL
) {
  if (is.null(ctx)) {
    ctx <- ctx()
  }

  df <- ctx$data_frame_from_table_provider(x)
  new_sedonadb_dataframe(ctx, df)
}

#' Count rows in a DataFrame
#'
#' @param .data A sedonadb_dataframe or an object that can be coerced to one.
#'
#' @returns The number of rows after executing the query
#' @export
#'
#' @examples
#' sd_sql("SELECT 1 as one") |> sd_count()
#'
sd_count <- function(.data) {
  .data <- as_sedonadb_dataframe(.data)
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
sd_to_view <- function(.data, table_ref, ctx = NULL, overwrite = FALSE) {
  if (is.null(ctx)) {
    ctx <- ctx()
  }

  .data <- as_sedonadb_dataframe(.data, ctx = ctx)
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

  schema <- nanoarrow::infer_nanoarrow_schema(.data)
  if (is.null(.data$group_by)) {
    grouped_label <- ""
    grouped_vars <- ""
  } else {
    grouped_label <- "grouped "
    grouped_vars <- sprintf(
      " | [%s]",
      paste0("`", names(.data$group_by), "`", collapse = ", ")
    )
  }

  cat(
    sprintf(
      "<%ssedonab_dataframe: NA x %d%s>\n",
      grouped_label,
      length(schema$children),
      grouped_vars
    )
  )

  cat(content)
  cat(paste0("Preview of up to ", n, " row(s)\n"))

  invisible(.data)
}

#' Keep or drop columns of a SedonaDB DataFrame
#'
#' @inheritParams sd_count
#' @param ... One or more bare names. Evaluated like [dplyr::select()].
#'
#' @returns An object of class sedonadb_dataframe
#' @export
#'
#' @examples
#' data.frame(x = 1:10, y = letters[1:10]) |> sd_select(x)
#'
sd_select <- function(.data, ...) {
  .data <- as_sedonadb_dataframe(.data)
  if (!is.null(.data$group_by)) {
    stop("sd_select() does not support grouped input")
  }

  schema <- nanoarrow::infer_nanoarrow_schema(.data)
  ptype <- nanoarrow::infer_nanoarrow_ptype(schema)
  loc <- tidyselect::eval_select(rlang::expr(c(...)), data = ptype)

  df <- .data$df$select_indices(names(loc), loc - 1L)
  new_sedonadb_dataframe(.data$ctx, df)
}

#' Create, modify, and delete columns of a SedonaDB DataFrame
#'
#' @inheritParams sd_count
#' @param ... Named expressions for new columns to create. These are evaluated
#'   in the same way as [dplyr::transmute()] except does not support extra
#'   dplyr features such as `across()` or `.by`.
#'
#' @returns An object of class sedonadb_dataframe
#' @export
#'
#' @examples
#' data.frame(x = 1:10) |>
#'   sd_transmute(y = x + 1L)
#'
sd_transmute <- function(.data, ...) {
  .data <- as_sedonadb_dataframe(.data)
  if (!is.null(.data$group_by)) {
    stop("sd_transmute() does not support grouped input")
  }

  expr_quos <- rlang::enquos(...)
  env <- parent.frame()

  expr_ctx <- sd_expr_ctx(infer_nanoarrow_schema(.data), env, ctx = .data$ctx)
  r_exprs <- expr_quos |> rlang::quos_auto_name() |> lapply(rlang::quo_get_expr)
  sd_exprs <- lapply(r_exprs, sd_eval_expr, expr_ctx = expr_ctx, env = env)

  # Ensure inputs are given aliases to account for the expected column name
  exprs_names <- names(r_exprs)
  for (i in seq_along(sd_exprs)) {
    name <- exprs_names[i]
    if (!is.na(name) && name != "") {
      sd_exprs[[i]] <- sd_expr_alias(sd_exprs[[i]], name, expr_ctx$factory)
    }
  }

  df <- .data$df$select(sd_exprs)
  new_sedonadb_dataframe(.data$ctx, df)
}

#' Keep rows of a SedonaDB DataFrame that match a condition
#'
#' @inheritParams sd_count
#' @param ... Unnamed expressions for filter conditions. These are evaluated
#'   in the same way as [dplyr::filter()] except does not support extra
#'   dplyr features such as `across()` or `.by`.
#'
#' @returns An object of class sedonadb_dataframe
#' @export
#'
#' @examples
#' data.frame(x = 1:10) |> sd_filter(x > 5)
#'
sd_filter <- function(.data, ...) {
  .data <- as_sedonadb_dataframe(.data)
  if (!is.null(.data$group_by)) {
    stop("sd_filter() does not support grouped input")
  }

  rlang::check_dots_unnamed()

  expr_quos <- rlang::enquos(...)
  env <- parent.frame()

  expr_ctx <- sd_expr_ctx(infer_nanoarrow_schema(.data), env, ctx = .data$ctx)
  r_exprs <- expr_quos |> lapply(rlang::quo_get_expr)
  sd_exprs <- lapply(r_exprs, sd_eval_expr, expr_ctx = expr_ctx, env = env)

  df <- .data$df$filter(sd_exprs)
  new_sedonadb_dataframe(.data$ctx, df)
}

#' Order rows of a SedonaDB data frame using column values
#'
#' @inheritParams sd_count
#' @param ... Unnamed expressions for arrange expressions. These are evaluated
#'   in the same way as [dplyr::arrange()] except does not support extra
#'   dplyr features such as `across()`, `.by_group`, or `.locale`.
#'
#' @returns An object of class sedonadb_dataframe
#' @export
#'
#' @examples
#' data.frame(x = c(10:1, NA)) |> sd_arrange(x)
#' data.frame(x = c(1:10, NA)) |> sd_arrange(desc(x))
#'
sd_arrange <- function(.data, ...) {
  .data <- as_sedonadb_dataframe(.data)
  if (!is.null(.data$group_by)) {
    stop("sd_arrange() does not support grouped input")
  }

  rlang::check_dots_unnamed()

  expr_quos <- rlang::enquos(...)
  env <- parent.frame()

  expr_ctx <- sd_expr_ctx(infer_nanoarrow_schema(.data), env, ctx = .data$ctx)
  r_exprs <- expr_quos |> lapply(rlang::quo_get_expr)

  # Specifically for sd_arrange(), we need to unwrap desc() calls
  unwrapped <- unwrap_desc(r_exprs)

  sd_exprs <- lapply(
    unwrapped$inner_exprs,
    sd_eval_expr,
    expr_ctx = expr_ctx,
    env = env
  )

  df <- .data$df$arrange(sd_exprs, unwrapped$is_descending)
  new_sedonadb_dataframe(.data$ctx, df)
}

#' Group SedonaDB DataFrames by one or more expressions
#'
#' Note that unlike [dplyr::group_by()], these groups are dropped after
#' any transformations.
#'
#' @inheritParams sd_count
#' @param ... Named expressions whose unique combination will be used as
#'   groups to potentially compute a future aggregate expression. These are
#'   evaluated in the same way as [dplyr::group_by()] except `.add` nor
#'   `.drop` are supported.
#'
#' @returns An object of class sedonadb_dataframe
#' @export
#'
#' @examples
#' data.frame(letter = c(rep("a", 3), rep("b", 4), rep("c", 3)), x = 1:10) |>
#'   sd_group_by(letter) |>
#'   sd_summarise(x = sum(x))
#'
sd_group_by <- function(.data, ...) {
  .data <- as_sedonadb_dataframe(.data)
  expr_quos <- rlang::enquos(...)
  env <- parent.frame()

  expr_ctx <- sd_expr_ctx(infer_nanoarrow_schema(.data), env, ctx = .data$ctx)
  r_exprs <- expr_quos |> rlang::quos_auto_name() |> lapply(rlang::quo_get_expr)
  sd_exprs <- lapply(r_exprs, sd_eval_expr, expr_ctx = expr_ctx, env = env)

  # Ensure inputs are given aliases to account for the expected column name
  exprs_names <- names(r_exprs)
  for (i in seq_along(sd_exprs)) {
    name <- exprs_names[i]
    if (!is.na(name) && name != "") {
      sd_exprs[[i]] <- sd_expr_alias(sd_exprs[[i]], name, expr_ctx$factory)
    }
  }

  new_sedonadb_dataframe(.data$ctx, .data$df, group_by = sd_exprs)
}

#' @rdname sd_group_by
#' @export
sd_ungroup <- function(.data) {
  .data <- as_sedonadb_dataframe(.data)
  .data$group_by <- NULL
  .data
}

#' Aggregate SedonaDB DataFrames to a single row per group
#'
#' @inheritParams sd_count
#' @param ... Aggregate expressions. These are evaluated in the same way as
#'   [dplyr::summarise()] except the outer expression must be an aggregate
#'   expression (e.g., `sum(x) + 1` is not currently possible).
#'
#' @returns An object of class sedonadb_dataframe
#' @export
#'
#' @examples
#' data.frame(x = c(10:1, NA)) |> sd_summarise(x = sum(x, na.rm = TRUE))
#'
sd_summarise <- function(.data, ...) {
  .data <- as_sedonadb_dataframe(.data)

  expr_quos <- rlang::enquos(...)
  env <- parent.frame()

  expr_ctx <- sd_expr_ctx(infer_nanoarrow_schema(.data), env, ctx = .data$ctx)
  r_exprs <- expr_quos |> rlang::quos_auto_name() |> lapply(rlang::quo_get_expr)
  sd_exprs <- lapply(r_exprs, sd_eval_expr, expr_ctx = expr_ctx, env = env)

  # Ensure inputs are given aliases to account for the expected column name
  exprs_names <- names(r_exprs)
  for (i in seq_along(sd_exprs)) {
    name <- exprs_names[i]
    if (!is.na(name) && name != "") {
      sd_exprs[[i]] <- sd_expr_alias(sd_exprs[[i]], name, expr_ctx$factory)
    }
  }

  df <- .data$df$aggregate(as.list(.data$group_by), sd_exprs)
  new_sedonadb_dataframe(.data$ctx, df)
}

#' @rdname sd_summarise
#' @export
sd_summarize <- function(.data, ...) {
  sd_summarise(.data, ...)
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
#' @param options A named list of key/value options to be used when constructing
#'   a parquet writer. Common options are exposed as other arguments to
#'   `sd_write_parquet()`; however, this argument allows setting any DataFusion
#'   Parquet writer option. If an option is specified here and by another
#'   argument to this function, the value specified as an explicit argument
#'   takes precedence.
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
#'   will be named "\[geom_col_name\]_bbox" for all geometry columns except
#'   "geometry", whose bounding box column name is just "bbox"
#' @param overwrite_bbox_columns Use TRUE to overwrite any bounding box columns
#'   that already exist in the input. This is useful in a read -> modify
#'   -> write scenario to ensure these columns are up-to-date. If FALSE
#'   (the default), an error will be raised if a bbox column already exists
#' @param max_row_group_size Target maximum number of rows in each row group.
#'   Defaults to the global configuration value (1M rows).
#' @param compression Sets the Parquet compression codec. Valid values are:
#'   uncompressed, snappy, gzip(level), brotli(level), lz4, zstd(level), and
#'   lz4_raw. Defaults to the global configuration value (zstd(3)).
#'
#' @returns The input, invisibly
#' @export
#'
#' @examples
#' tmp_parquet <- tempfile(fileext = ".parquet")
#'
#' sd_sql("SELECT ST_Point(1, 2, 4326) as geom") |>
#'   sd_write_parquet(tmp_parquet)
#'
#' sd_read_parquet(tmp_parquet)
#' unlink(tmp_parquet)
#'
sd_write_parquet <- function(
  .data,
  path,
  options = NULL,
  partition_by = character(0),
  sort_by = character(0),
  single_file_output = NULL,
  geoparquet_version = "1.0",
  overwrite_bbox_columns = FALSE,
  max_row_group_size = NULL,
  compression = NULL
) {
  .data <- as_sedonadb_dataframe(.data)
  if (!is.null(.data$group_by)) {
    stop("sd_write_parquet() does not support grouped input")
  }

  # Determine single_file_output default based on path and partition_by
  if (is.null(single_file_output)) {
    single_file_output <- length(partition_by) == 0 && grepl("\\.parquet$", path)
  }

  # Build the options list: start with user-provided options, then override
  # with explicitly-specified arguments
  if (is.null(options)) {
    options <- list()
  } else {
    options <- as.list(options)
  }

  if (!is.null(max_row_group_size)) {
    options[["max_row_group_size"]] <- as.character(as.integer(max_row_group_size))
  }

  if (!is.null(compression)) {
    options[["compression"]] <- as.character(compression)
  }

  # Validate and apply geoparquet_version
  if (!is.null(geoparquet_version)) {
    options[["geoparquet_version"]] <- as.character(geoparquet_version)
  }

  options[["overwrite_bbox_columns"]] <- tolower(as.character(overwrite_bbox_columns))

  # Convert options to parallel character vectors for Rust
  option_keys <- names(options)
  option_values <- as.character(unlist(options, use.names = FALSE))

  if (is.null(option_keys) || any(is.na(option_keys)) || any(option_keys == "")) {
    stop("All option values must be named")
  }

  if (length(option_keys) != length(option_values)) {
    stop("All option values must be length 1")
  }

  # Call the underlying Rust method
  .data$df$to_parquet(
    ctx = .data$ctx,
    path = path,
    option_keys = option_keys,
    option_values = option_values,
    partition_by = partition_by,
    sort_by = sort_by,
    single_file_output = single_file_output
  )

  invisible(.data)
}

new_sedonadb_dataframe <- function(ctx, internal_df, ..., group_by = NULL) {
  if (length(group_by) == 0) {
    group_by <- NULL
  }

  structure(
    list(ctx = ctx, df = internal_df, group_by = group_by),
    class = "sedonadb_dataframe"
  )
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
