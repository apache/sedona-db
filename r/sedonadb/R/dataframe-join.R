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

#' Join two SedonaDB DataFrames
#'
#' Perform a join operation between two dataframes. Use [sd_join_by()] to
#' specify join conditions using `x$column` and `y$column` syntax to
#' reference columns from the left and right tables respectively.
#'
#' @param x The left dataframe
#' @param y The right dataframe (will use the same context as x)
#' @param by Join specification. One of:
#'   - A `sedonadb_join_by` object from [sd_join_by()]
#'   - A character vector of column names to join on in both tables
#'   - A named character vector mapping left-table column names to
#'     right-table column names, e.g. `c(x_val = "y_val")`
#'   - `NULL` for a natural join on columns with matching names
#' @param join_type The type of join to perform. One of "inner", "left", "right",
#'   "full", "leftsemi", "rightsemi", "leftanti", "rightanti", "leftmark",
#'   or "rightmark".
#' @param select Post-join column selection. One of
#'   - `NULL` for no modification, which may result in duplicate (unqualified)
#'     column names. The column may still be
#'     referred to with a qualifier in advanced usage using [sd_expr_column()].
#'   - [sd_join_select_default()] for dplyr-like behaviour (equi-join keys
#'     removed, intersecting names suffixed)
#'   - [sd_join_select()] for a custom selection
#'
#' @returns An object of class sedonadb_dataframe
#' @export
#'
#' @examples
#' df1 <- data.frame(x = letters[1:10], y = 1:10)
#' df2 <- data.frame(y = 10:1, z = LETTERS[1:10])
#' df1 |> sd_join(df2)
#'
sd_join <- function(
  x,
  y,
  by = NULL,
  join_type = "inner",
  select = sd_join_select_default()
) {
  x <- as_sedonadb_dataframe(x)
  y <- as_sedonadb_dataframe(y, ctx = x$ctx)

  x_schema <- infer_nanoarrow_schema(x)
  y_schema <- infer_nanoarrow_schema(y)
  join_expr_ctx <- sd_join_expr_ctx(x_schema, y_schema, ctx = x$ctx)
  join_conditions <- sd_build_join_conditions(join_expr_ctx, by, ctx = x$ctx)

  df <- x$df$join(y$df, join_conditions, join_type, left_alias = "x", right_alias = "y")
  out <- new_sedonadb_dataframe(x$ctx, df)

  # Apply post-join column selection if needed
  if (is.null(select)) {
    projection <- NULL
  } else if (inherits(select, "sedonadb_join_select_default")) {
    # Default select: remove duplicate equijoin keys, apply suffixes
    projection <- sd_build_default_select(
      join_expr_ctx,
      join_conditions,
      select$suffix,
      join_type
    )
  } else if (inherits(select, "sedonadb_join_select")) {
    # Custom select: evaluate user expressions
    projection <- sd_eval_join_select_exprs(select, join_expr_ctx)
  } else {
    stop(
      "`select` must be NULL, sd_join_select_default(), or sd_join_select()",
      call. = FALSE
    )
  }

  # NULL return from these functions means that no extra projecting is needed
  if (is.null(projection)) {
    out
  } else {
    sd_transmute(out, !!!projection)
  }
}

#' @rdname sd_join
#' @export
sd_left_join <- function(x, y, by = NULL, select = sd_join_select_default()) {
  sd_join(x, y, by = by, select = select, join_type = "left")
}

#' @rdname sd_join
#' @export
sd_right_join <- function(x, y, by = NULL, select = sd_join_select_default()) {
  sd_join(x, y, by = by, select = select, join_type = "right")
}

#' @rdname sd_join
#' @export
sd_inner_join <- function(x, y, by = NULL, select = sd_join_select_default()) {
  sd_join(x, y, by = by, select = select, join_type = "inner")
}

#' @rdname sd_join
#' @export
sd_full_join <- function(x, y, by = NULL, select = sd_join_select_default()) {
  sd_join(x, y, by = by, select = select, join_type = "full")
}

#' @rdname sd_join
#' @export
sd_semi_join <- function(x, y, by = NULL) {
  sd_join(x, y, by = by, join_type = "leftsemi")
}

#' @rdname sd_join
#' @export
sd_anti_join <- function(x, y, by = NULL) {
  sd_join(x, y, by = by, join_type = "leftanti")
}

#' @rdname sd_join
#' @export
sd_cross_join <- function(x, y, by = NULL) {
  sd_join(x, y, by = character(), join_type = "inner")
}
