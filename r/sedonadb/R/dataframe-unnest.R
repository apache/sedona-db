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

#' Expand nested columns of a SedonaDB DataFrame
#'
#' `sd_unnest_longer()` expands list columns by one level into rows, repeating
#' the other columns. Null and empty lists produce no rows. Multiple columns
#' are expanded in parallel, padding shorter lists with nulls (rather than
#' recycling as in tidyr). This corresponds to Python's `DataFrame.unnest()`,
#' except that null lists are dropped.
#'
#' `sd_unnest_wider()` expands struct columns by one level into columns,
#' preserving rows and the position of the selected columns. Nested fields
#' remain nested. Output column names must be unique.
#'
#' @inheritParams sd_count
#' @param col Columns to expand, evaluated using tidyselect. Renaming is not
#'   supported. Longer requires list columns; wider requires struct columns.
#' @param names_sep Optional string separating the original column name and
#'   field name. By default, only field names are used.
#' @returns An object of class sedonadb_dataframe.
#' @export
#' @examples
#' sd_sql("SELECT [1, 2, 3] AS values") |> sd_unnest_longer(values)
#' sd_sql("SELECT named_struct('x', 1, 'y', 2) AS point") |>
#'   sd_unnest_wider(point, names_sep = "_")
sd_unnest_longer <- function(.data, col) {
  .data <- as_sedonadb_dataframe(.data)
  selected <- sd_unnest_select(.data, rlang::enquo(col), "longer")
  df <- .data$df$unnest(names(selected))
  new_sedonadb_dataframe(.data$ctx, df)
}

#' @rdname sd_unnest_longer
#' @export
sd_unnest_wider <- function(.data, col, names_sep = NULL) {
  .data <- as_sedonadb_dataframe(.data)
  selected <- sd_unnest_select(.data, rlang::enquo(col), "wider")
  if (
    !is.null(names_sep) &&
      (!is.character(names_sep) || length(names_sep) != 1L || is.na(names_sep))
  ) {
    stop("`names_sep` must be NULL or a single non-missing string")
  }

  schema <- nanoarrow::infer_nanoarrow_schema(.data)
  exprs <- list()
  output_names <- character()
  for (name in names(schema$children)) {
    if (name %in% names(selected)) {
      fields <- names(schema$children[[name]]$children)
      exprs <- c(
        exprs,
        lapply(fields, function(field) {
          rlang::call2("$", rlang::sym(name), rlang::sym(field))
        })
      )
      output_names <- c(
        output_names,
        if (is.null(names_sep)) {
          fields
        } else {
          paste0(name, names_sep, fields)
        }
      )
    } else {
      exprs <- c(exprs, list(rlang::sym(name)))
      output_names <- c(output_names, name)
    }
  }
  if (anyDuplicated(output_names) || any(!nzchar(output_names))) {
    stop("Output names must be non-empty and unique; use `names_sep` to disambiguate")
  }
  names(exprs) <- output_names
  # Evaluate all projections against the input, avoiding sequential transmute
  # semantics when a field has the same name as another input column.
  expr_ctx <- sd_expr_ctx(schema, parent.frame(), ctx = .data$ctx)
  exprs <- lapply(exprs, sd_eval_expr, expr_ctx = expr_ctx, env = parent.frame())
  exprs <- Map(function(expr, name) expr$alias(name), exprs, output_names)
  new_sedonadb_dataframe(.data$ctx, .data$df$select(exprs))
}

sd_unnest_select <- function(.data, col, direction) {
  if (!is.null(.data$group_by)) {
    stop("Unnesting does not support grouped input")
  }
  schema <- nanoarrow::infer_nanoarrow_schema(.data)
  ptype <- nanoarrow::infer_nanoarrow_ptype(schema)
  selected <- tidyselect::eval_select(col, ptype, allow_rename = FALSE)
  if (!length(selected)) {
    stop("Select at least one column to unnest")
  }
  for (name in names(selected)) {
    format <- schema$children[[name]]$format
    valid <- if (direction == "longer") {
      format %in% c("+l", "+L", "+vl", "+vL") || startsWith(format, "+w:")
    } else {
      format == "+s"
    }
    if (!valid) {
      stop(sprintf(
        "Column '%s' must be a %s column",
        name,
        if (direction == "longer") "list" else "struct"
      ))
    }
  }
  selected
}
