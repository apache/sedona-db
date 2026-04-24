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

#' Specify default post-join column selection
#'
#' Use `sd_join_select_default()` to specify that the join result should
#' remove duplicate equijoin key columns (keeping the x-side version) and
#' apply suffixes to any remaining overlapping column names.
#'
#' @param suffix A character vector of length 2 specifying suffixes to add
#'   to overlapping column names from the left (x) and right (y) tables.
#'
#' @returns An object of class `sedonadb_join_select_default` specifying
#'   the default column selection behavior.
#' @export
#'
#' @examples
#' # Default suffixes
#' sd_join_select_default()
#'
#' # Custom suffixes
#' sd_join_select_default(suffix = c("_left", "_right"))
#'
sd_join_select_default <- function(suffix = c(".x", ".y")) {
  if (!is.character(suffix) || length(suffix) != 2) {
    stop("`suffix` must be a character vector of length 2")
  }

  structure(
    list(suffix = suffix),
    class = "sedonadb_join_select_default"
  )
}

#' @export
print.sedonadb_join_select_default <- function(x, ...) {
  cat("<sedonadb_join_select_default>\n")
  cat("  suffix: c(\"", x$suffix[1], "\", \"", x$suffix[2], "\")\n", sep = "")
  invisible(x)
}

#' Specify custom post-join column selection
#'
#' Use `sd_join_select()` to specify which columns to include in the join
#' result and optionally rename them. Columns are referenced using `x$column`
#' and `y$column` syntax to disambiguate columns from the left and right tables.
#'
#' @param ... Named expressions specifying output columns. Each expression
#'   should reference a column using `x$column` or `y$column` syntax.
#'   The name of the argument becomes the output column name. Unnamed
#'   arguments use the original column name (without table prefix).
#'
#' @returns An object of class `sedonadb_join_select` containing the
#'   unevaluated column selection expressions.
#' @export
#'
#' @examples
#' # Select and rename columns
#' sd_join_select(id = x$id, left_value = x$value, right_value = y$value)
#'
#' # Unnamed arguments keep original column name
#' sd_join_select(x$id, x$name, y$value)
#'
sd_join_select <- function(...) {
  exprs <- rlang::enquos(...)

  if (length(exprs) == 0) {
    stop("sd_join_select() requires at least one column selection")
  }

  structure(
    list(exprs = exprs),
    class = "sedonadb_join_select"
  )
}

#' @export
print.sedonadb_join_select <- function(x, ...) {
  cat("<sedonadb_join_select>\n")
  for (i in seq_along(x$exprs)) {
    name <- names(x$exprs)[i]
    expr_str <- rlang::expr_deparse(rlang::quo_get_expr(x$exprs[[i]]))
    if (!is.null(name) && nzchar(name)) {
      cat("  ", name, " = ", expr_str, "\n", sep = "")
    } else {
      cat("  ", expr_str, "\n", sep = "")
    }
  }
  invisible(x)
}

#' Evaluate custom join select expressions
#'
#' Evaluates column selection expressions captured by [sd_join_select()] into
#' a list of output column specifications.
#'
#' @param join_select A `sedonadb_join_select` object from [sd_join_select()]
#' @param join_expr_ctx A `sedonadb_join_expr_ctx` from `sd_join_expr_ctx()`
#'
#' @returns A named list of expressions
#' @noRd
sd_eval_join_select_exprs <- function(join_select, join_expr_ctx) {
  stopifnot(inherits(join_select, "sedonadb_join_select"))

  exprs <- lapply(join_select$exprs, function(quo) {
    expr <- rlang::quo_get_expr(quo)
    env <- rlang::quo_get_env(quo)

    rlang::try_fetch(
      {
        # Evaluate the expression to get a column reference
        sd_eval_join_select_expr_inner(expr, join_expr_ctx, env)
      },
      error = function(e) {
        rlang::abort(
          sprintf(
            "Error evaluating select expression %s",
            rlang::expr_label(expr)
          ),
          parent = e
        )
      }
    )
  })

  is_unnamed <- names(exprs) == ""
  names(exprs)[is_unnamed] <- lapply(exprs[is_unnamed], function(e) e$qualified_name()[2])
  exprs
}

#' Evaluate a single join select expression
#'
#' @param expr An unevaluated R expression
#' @param join_expr_ctx A join expression context
#' @param env The expression environment
#'
#' @returns A `SedonaDBExpr` column expression
#' @noRd
sd_eval_join_select_expr_inner <- function(expr, join_expr_ctx, env) {
  if (rlang::is_call(expr, "$")) {
    # x$col or y$col syntax
    lhs <- expr[[2]]
    rhs <- expr[[3]]

    if (rlang::is_symbol(lhs) && as.character(lhs) %in% c("x", "y")) {
      table_ref <- rlang::eval_tidy(lhs, data = join_expr_ctx$data, env = env)
      col_name <- as.character(rhs)
      return(`$.sedonadb_table_ref`(table_ref, col_name))
    }
  }

  if (rlang::is_symbol(expr)) {
    name <- as.character(expr)
    if (name %in% join_expr_ctx$ambiguous_columns) {
      stop(
        sprintf("Column '%s' is ambiguous (exists in both tables). ", name),
        sprintf("Use x$%s or y$%s to disambiguate.", name, name),
        call. = FALSE
      )
    }
    # Unambiguous column reference
    return(rlang::eval_tidy(expr, data = join_expr_ctx$data, env = env))
  }

  # For select, we only allow column references, not arbitrary expressions

  stop(
    "sd_join_select() expressions must be column references ",
    "(e.g., x$col or y$col), not arbitrary expressions",
    call. = FALSE
  )
}

#' Build default column selection for join result
#'
#' Creates a column selection that:
#' 1. Removes duplicate equijoin key columns (keeps x-side)
#' 2. Applies suffixes to remaining overlapping column names
#'
#' @param join_expr_ctx A `sedonadb_join_expr_ctx` from `sd_join_expr_ctx()`
#' @param join_conditions List of join condition expressions
#' @param suffix Character vector of length 2 for left/right suffixes
#'
#' @returns A named list of expressions
#' @noRd
sd_build_default_select <- function(join_expr_ctx, join_conditions, suffix) {
  x_names <- names(join_expr_ctx$x_schema$children)
  y_names <- names(join_expr_ctx$y_schema$children)

  # Extract equijoin key pairs (simple x$col == y$col conditions)
  # and remove them from the y_names
  equijoin_keys <- sd_extract_equijoin_keys(join_conditions)
  y_names <- setdiff(y_names, equijoin_keys$y_cols)

  # Calculate names that need suffixing
  common_names <- intersect(x_names, y_names)
  x_name_needs_suffix <- x_names %in% common_names
  y_name_needs_suffix <- y_names %in% common_names

  # Apply suffixes to column names that need it, but keep a copy of the input
  # names unchanged since we'll need those to get the original column expr
  x_names_out <- x_names
  x_names_out[x_name_needs_suffix] <- paste0(x_names_out[x_name_needs_suffix], suffix[1])
  y_names_out <- y_names
  y_names_out[y_name_needs_suffix] <- paste0(y_names_out[y_name_needs_suffix], suffix[2])

  # Create the expressions named with the appropriate output name
  exprs <- c(
    lapply(x_names, function(name) {
      sd_expr_column(name, qualifier = "x", factory = join_expr_ctx$factory)
    }),
    lapply(y_names, function(name) {
      sd_expr_column(name, qualifier = "y", factory = join_expr_ctx$factory)
    })
  )
  names(exprs) <- c(x_names_out, y_names_out)
  exprs
}

#' Extract equijoin key column pairs from join conditions
#'
#' Identifies simple equality conditions of the form `x$col == y$col` and
#' returns the column names involved.
#'
#' @param join_conditions List of join condition expressions
#'
#' @returns A list with `x_cols` and `y_cols` character vectors of matching
#'   column names from each side of equijoin conditions.
#' @noRd
sd_extract_equijoin_keys <- function(join_conditions) {
  x_cols <- character()
  y_cols <- character()

  for (cond in join_conditions) {
    if (!inherits(cond, "SedonaDBExpr")) {
      next
    }

    # TODO: this is totally insane, we can do a better job extracting
    # equijoin keys from expressions.

    # Try to parse the display string for equality pattern
    # Format: "x.col = y.col " (may have trailing space)
    display <- trimws(cond$display())

    # Match pattern: "x.something = y.something"
    # Handles column names with underscores, numbers, etc.
    pattern <- "^(x\\.[a-zA-Z_][a-zA-Z0-9_]*)\\s*=\\s*(y\\.[a-zA-Z_][a-zA-Z0-9_]*)$"
    match <- regmatches(display, regexec(pattern, display))[[1]]

    if (length(match) == 3) {
      # Extract column names without qualifier
      x_col <- sub("^x\\.", "", match[2])
      y_col <- sub("^y\\.", "", match[3])
      x_cols <- c(x_cols, x_col)
      y_cols <- c(y_cols, y_col)
    }
  }

  list(x_cols = x_cols, y_cols = y_cols)
}
