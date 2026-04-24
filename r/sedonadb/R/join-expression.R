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

#' Specify join conditions
#'
#' Use `sd_join_by()` to specify join conditions for [sd_join()] using
#' expressions that reference columns from both tables. Table references
#' are specified using `x$column` and `y$column` syntax to disambiguate
#' columns from the left and right tables.
#'
#' @param ... Expressions specifying join conditions. These should be
#'   comparison expressions (e.g., `x$id == y$id`, `x$value > y$threshold`).
#'   Multiple conditions are combined with AND.
#'
#' @returns An object of class `sedonadb_join_by` containing the unevaluated
#'   join condition expressions.
#' @export
#'
#' @examples
#' # Equality join on id column
#' sd_join_by(x$id == y$id)
#'
#' # Multiple conditions (combined with AND)
#' sd_join_by(x$id == y$id, x$date >= y$start_date)
#'
#' # Inequality join
#' sd_join_by(x$value > y$threshold)
#'
sd_join_by <- function(...) {
  exprs <- rlang::enquos(...)

  if (length(exprs) == 0) {
    stop("sd_join_by() requires at least one join condition")
  }

  structure(
    list(
      exprs = exprs
    ),
    class = "sedonadb_join_by"
  )
}

#' @export
print.sedonadb_join_by <- function(x, ...) {
  cat("<sedonadb_join_by>\n")
  for (i in seq_along(x$exprs)) {
    cat("  ", rlang::expr_deparse(rlang::quo_get_expr(x$exprs[[i]])), "\n", sep = "")
  }
  invisible(x)
}

#' Expression evaluation context for joins
#'
#' Creates a context for evaluating join conditions that can reference columns
#' from two tables using qualified references (`x$col` and `y$col`).
#'
#' @param x_schema Schema for the left table
#' @param y_schema Schema for the right table
#' @param env The expression environment
#' @param ctx A SedonaDB context
#' @param x_qualifier Qualifier for left table columns (default "x")
#' @param y_qualifier Qualifier for right table columns (default "y")
#'
#' @return An object of class sedonadb_join_expr_ctx
#' @noRd
sd_join_expr_ctx <- function(
  x_schema,
  y_schema,
  env = parent.frame(),
  ctx = NULL
) {
  x_schema <- nanoarrow::as_nanoarrow_schema(x_schema)
  y_schema <- nanoarrow::as_nanoarrow_schema(y_schema)

  x_names <- as.character(names(x_schema$children))
  y_names <- as.character(names(y_schema$children))

  factory <- sd_expr_factory(ctx = ctx)

  # We hard-code these for the purposes of the join expression
  x_qualifier <- "x"
  y_qualifier <- "y"

  # Create qualified column references for both tables
  # These are accessed via x$col and y$col syntax
  x_cols <- lapply(x_names, function(name) {
    sd_expr_column(name, qualifier = x_qualifier, factory = factory)
  })
  names(x_cols) <- x_names

  y_cols <- lapply(y_names, function(name) {
    sd_expr_column(name, qualifier = y_qualifier, factory = factory)
  })
  names(y_cols) <- y_names

  # Create table reference objects that support `$` access
  x_ref <- structure(x_cols, class = "sedonadb_table_ref", qualifier = x_qualifier)
  y_ref <- structure(y_cols, class = "sedonadb_table_ref", qualifier = y_qualifier)

  # The data mask contains x and y as table references
  data <- list(x = x_ref, y = y_ref)

  # Also include unqualified column references for unambiguous columns
  all_names <- unique(c(x_names, y_names))
  ambiguous <- intersect(x_names, y_names)

  for (name in all_names) {
    if (!(name %in% ambiguous)) {
      # Unambiguous column - add to data mask
      if (name %in% x_names) {
        data[[name]] <- x_cols[[name]]
      } else {
        data[[name]] <- y_cols[[name]]
      }
    }
  }

  structure(
    list(
      factory = factory,
      x_schema = x_schema,
      y_schema = y_schema,
      x_qualifier = x_qualifier,
      y_qualifier = y_qualifier,
      x_ref = x_ref,
      y_ref = y_ref,
      ambiguous_columns = ambiguous,
      data = rlang::as_data_mask(data),
      env = env,
      fns = default_fns
    ),
    class = c("sedonadb_join_expr_ctx", "sedonadb_expr_ctx")
  )
}

#' @export
`$.sedonadb_table_ref` <- function(x, name) {
  if (!(name %in% names(x))) {
    qualifier <- attr(x, "qualifier")
    stop(
      sprintf("Column '%s' not found in table '%s'", name, qualifier),
      call. = FALSE
    )
  }
  x[[name]]
}

#' Evaluate join conditions
#'
#' Evaluates join condition expressions captured by [sd_join_by()] into
#' SedonaDB expressions using a join expression context.
#'
#' @param join_by A `sedonadb_join_by` object from [sd_join_by()]
#' @param join_expr_ctx A `sedonadb_join_expr_ctx` from `sd_join_expr_ctx()`
#'
#' @returns A list of `SedonaDBExpr` objects representing the join conditions
#' @noRd
sd_eval_join_conditions <- function(join_by, join_expr_ctx) {
  ensure_translations_registered()

  stopifnot(inherits(join_by, "sedonadb_join_by"))

  lapply(join_by$exprs, function(quo) {
    expr <- rlang::quo_get_expr(quo)
    env <- rlang::quo_get_env(quo)

    rlang::try_fetch(
      {
        result <- sd_eval_join_expr_inner(expr, join_expr_ctx, env)
        as_sd_expr(result, factory = join_expr_ctx$factory)
      },
      error = function(e) {
        rlang::abort(
          sprintf("Error evaluating join condition %s", rlang::expr_label(expr)),
          parent = e
        )
      }
    )
  })
}

sd_eval_join_expr_inner <- function(expr, join_expr_ctx, env) {
  if (rlang::is_call(expr)) {
    # Special handling for x$col and y$col syntax
    if (rlang::is_call(expr, "$")) {
      lhs <- expr[[2]]
      rhs <- expr[[3]]

      # Check if this is x$col or y$col pattern
      if (rlang::is_symbol(lhs) && as.character(lhs) %in% c("x", "y")) {
        table_ref <- rlang::eval_tidy(lhs, data = join_expr_ctx$data, env = env)
        col_name <- as.character(rhs)
        # Use the $ S3 method to get proper error handling for missing columns
        return(`$.sedonadb_table_ref`(table_ref, col_name))
      }
    }

    # Check for ambiguous unqualified column reference
    if (rlang::is_symbol(expr)) {
      name <- as.character(expr)
      if (name %in% join_expr_ctx$ambiguous_columns) {
        stop(
          sprintf("Column '%s' is ambiguous (exists in both tables). ", name),
          sprintf("Use x$%s or y$%s to disambiguate.", name, name),
          call. = FALSE
        )
      }
    }

    # Extract function name
    call_name <- rlang::call_name(expr)

    # If we have a translation, use it (but with join-aware argument evaluation)
    if (!is.null(call_name) && !is.null(join_expr_ctx$fns[[call_name]])) {
      # Evaluate arguments with join context
      evaluated_args <- lapply(
        expr[-1],
        sd_eval_join_expr_inner,
        join_expr_ctx = join_expr_ctx,
        env = env
      )

      # Build and evaluate the translated call
      new_fn_expr <- rlang::call2("$", join_expr_ctx$fns, rlang::sym(call_name))
      new_call <- rlang::call2(new_fn_expr, join_expr_ctx, !!!evaluated_args)
      return(rlang::eval_tidy(new_call, data = join_expr_ctx$data, env = env))
    }

    # Default: evaluate with tidy eval
    rlang::eval_tidy(expr, data = join_expr_ctx$data, env = env)
  } else if (rlang::is_symbol(expr)) {
    # Check for ambiguous column reference
    name <- as.character(expr)
    if (name %in% join_expr_ctx$ambiguous_columns) {
      stop(
        sprintf(
          "Column '%s' is ambiguous (exists in both tables). ",
          name
        ),
        sprintf("Use x$%s or y$%s to disambiguate.", name, name),
        call. = FALSE
      )
    }
    rlang::eval_tidy(expr, data = join_expr_ctx$data, env = env)
  } else {
    # Literal or other expression
    rlang::eval_tidy(expr, data = join_expr_ctx$data, env = env)
  }
}

#' Build join conditions from a `by` specification
#'
#' Evaluates the `by` argument to produce a list of join condition expressions.
#' Supports natural joins (NULL) and explicit conditions via [sd_join_by()].
#'
#' @param join_expr_ctx Object produced by `sd_join_expr_ctx()`
#' @param by A `sedonadb_join_by` object from [sd_join_by()], or `NULL` for
#'   a natural join on columns with matching names.
#' @param ctx A SedonaDB context
#'
#' @returns A list of `SedonaDBExpr` objects representing the join conditions
#' @noRd
sd_build_join_conditions <- function(join_expr_ctx, by = NULL, ctx = NULL) {
  if (is.null(by)) {
    # Natural join: find common column names
    x_names <- names(join_expr_ctx$x_schema$children)
    y_names <- names(join_expr_ctx$y_schema$children)
    common <- intersect(x_names, y_names)

    if (length(common) == 0) {
      stop(
        "No common columns found for natural join. ",
        "Use sd_join_by() to specify join conditions."
      )
    }

    # Build equality conditions for common columns
    join_conditions <- lapply(common, function(col) {
      sd_expr_binary(
        "==",
        sd_expr_column(col, qualifier = "x", factory = join_expr_ctx$factory),
        sd_expr_column(col, qualifier = "y", factory = join_expr_ctx$factory),
        factory = join_expr_ctx$factory
      )
    })
  } else if (inherits(by, "sedonadb_join_by")) {
    join_conditions <- sd_eval_join_conditions(by, join_expr_ctx)
  } else {
    stop("`by` must be NULL (natural join) or a sd_join_by() object")
  }

  join_conditions
}
