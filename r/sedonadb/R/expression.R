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

#' Create SedonaDB logical expressions
#'
#' @param column_name A column name
#' @param x An object to convert to a SedonaDB literal (constant).
#' @param qualifier An optional qualifier (e.g., table reference) that may be
#'   used to disambiguate a specific reference
#' @param function_name The name of the function to call. This name is resolved
#'   from the context associated with `factory`.
#' @param type A destination type into which `expr` should be cast.
#' @param expr A SedonaDBExpr or object coercible to one with [as_sd_expr()].
#' @param alias An alias to apply to `expr`.
#' @param op Operator name for a binary expression. In general these follow
#'   R function names (e.g., `>`, `<`, `+`, `-`).
#' @param lhs,rhs Arguments to a binary expression
#' @param factory A [sd_expr_factory()]. This factory wraps a SedonaDB context
#'   and is used to resolve scalar functions and/or retrieve options.
#' @param ctx A SedonaDB context or NULL to use the default context.
#' @param args A list of SedonaDBExpr or object coercible to one with
#'   [as_sd_expr()].
#' @param na.rm For aggregate expressions, should nulls be ignored? The R
#'   idiom is to respect null; however, the SQL idiom is to drop them. The
#'   default value follows the R idiom (`na.rm = FALSE`).
#' @param distinct For aggregate expressions, use only distinct values.
#' @param ... Reserved for future use
#'
#' @returns An object of class SedonaDBExpr
#' @export
#'
#' @examples
#' sd_expr_column("foofy")
#' sd_expr_literal(1L)
#' sd_expr_scalar_function("abs", list(1L))
#' sd_expr_cast(1L, nanoarrow::na_int64())
#' sd_expr_alias(1L, "foofy")
#'
sd_expr_column <- function(column_name, qualifier = NULL, factory = sd_expr_factory()) {
  factory$column(column_name, qualifier)
}

#' @rdname sd_expr_column
#' @export
sd_expr_literal <- function(x, type = NULL, factory = sd_expr_factory()) {
  as_sedonadb_literal(x, type = type, factory = factory)
}

#' @rdname sd_expr_column
#' @export
sd_expr_binary <- function(op, lhs, rhs, factory = sd_expr_factory()) {
  factory$binary(op, as_sd_expr(lhs), as_sd_expr(rhs))
}

#' @rdname sd_expr_column
#' @export
sd_expr_negative <- function(expr, factory = sd_expr_factory()) {
  as_sd_expr(expr, factory = factory)$negate()
}

#' @rdname sd_expr_column
#' @export
sd_expr_any_function <- function(
  function_name,
  args,
  ...,
  na.rm = NULL, # nolint: object_name_linter
  factory = sd_expr_factory()
) {
  args_as_expr <- lapply(args, as_sd_expr, factory = factory)
  factory$any_function(function_name, args_as_expr, na_rm = na.rm)
}

#' @rdname sd_expr_column
#' @export
sd_expr_scalar_function <- function(function_name, args, factory = sd_expr_factory()) {
  args_as_expr <- lapply(args, as_sd_expr, factory = factory)
  factory$scalar_function(function_name, args_as_expr)
}

#' @rdname sd_expr_column
#' @export
sd_expr_aggregate_function <- function(
  function_name,
  args,
  ...,
  na.rm = FALSE, # nolint: object_name_linter
  distinct = FALSE,
  factory = sd_expr_factory()
) {
  args_as_expr <- lapply(args, as_sd_expr, factory = factory)
  factory$aggregate_function(
    function_name,
    args_as_expr,
    na_rm = na.rm,
    distinct = distinct
  )
}

#' @rdname sd_expr_column
#' @export
sd_expr_cast <- function(expr, type, factory = sd_expr_factory()) {
  expr <- as_sd_expr(expr, factory = factory)
  type <- nanoarrow::as_nanoarrow_schema(type)
  expr$cast(type)
}

#' @rdname sd_expr_column
#' @export
sd_expr_alias <- function(expr, alias, factory = sd_expr_factory()) {
  expr <- as_sd_expr(expr, factory = factory)
  expr$alias(alias)
}

#' @rdname sd_expr_column
#' @export
as_sd_expr <- function(x, factory = sd_expr_factory()) {
  if (inherits(x, "SedonaDBExpr")) {
    x
  } else {
    sd_expr_literal(x, factory = factory)
  }
}

#' @rdname sd_expr_column
#' @export
is_sd_expr <- function(x) {
  inherits(x, "SedonaDBExpr")
}

#' @rdname sd_expr_column
#' @export
sd_expr_factory <- function(ctx = NULL) {
  if (is.null(ctx)) {
    ctx <- ctx()
  }

  SedonaDBExprFactory$new(ctx)
}

#' @export
print.SedonaDBExpr <- function(x, ...) {
  cat("<SedonaDBExpr>\n")
  cat(x$display())
  cat("\n")
  invisible(x)
}

#' SedonaDB Functions
#'
#' This object is an escape hatch for calling SedonaDB/DataFusion functions
#' directly for translations that are not yet registered or are otherwise
#' misbehaving.
#'
#' @export .fns
.fns <- structure(list(), class = "sedonadb_fns")

# For IDE autocomplete
#' @export
names.sedonadb_fns <- function(x) {
  ctx <- ctx()
  ctx$list_functions()
}

# nolint start: object_name_linter
#' @importFrom utils .DollarNames
#' @export
.DollarNames.sedonadb_fns <- function(x, pattern = "") {
  grep(pattern, names(x), value = TRUE)
}
# nolint end

#' Evaluate an R expression into a SedonaDB expression
#'
#' @param expr An R expression (e.g., the result of `quote()`).
#' @param expr_ctx An `sd_expr_ctx()`
#' @param env An evaluation environment. Defaults to the calling environment.
#'
#' @returns A `SedonaDBExpr`
#' @noRd
sd_eval_expr <- function(expr, expr_ctx = sd_expr_ctx(env = env), env = parent.frame()) {
  ensure_translations_registered()

  rlang::try_fetch(
    {
      result <- sd_eval_expr_inner(expr, expr_ctx)
      as_sd_expr(result, factory = expr_ctx$factory)
    },
    error = function(e) {
      rlang::abort(
        sprintf("Error evaluating translated expression %s", rlang::expr_label(expr)),
        parent = e
      )
    }
  )
}

sd_eval_expr_inner <- function(expr, expr_ctx) {
  if (rlang::is_call(expr)) {
    # Special syntax for the escape hatch of "just call a DataFusion function" is
    # the expression .fns$datafusion_fn_name(arg1, arg2)
    if (rlang::is_call(expr[[1]], "$") && rlang::is_symbol(expr[[1]][[2]], ".fns")) {
      fn_key <- as.character(expr[[1]][[3]])
      return(sd_eval_datafusion_fn(fn_key, expr, expr_ctx))
    }

    # Extract `pkg::fun` or `fun` if this is a usual call (e.g., not
    # something fancy like `fun()()`)
    call_name <- rlang::call_name(expr)

    # If this is not a fancy function call and  we have a translation, call it.
    # Individual translations can choose to defer to the R function if all the
    # arguments are R objects and not SedonaDB expressions (or the user can
    # use !! to force R evaluation).
    if (!is.null(call_name) && !is.null(expr_ctx$fns[[call_name]])) {
      sd_eval_translation(call_name, expr, expr_ctx)
    } else {
      sd_eval_default(expr, expr_ctx)
    }
  } else {
    sd_eval_default(expr, expr_ctx)
  }
}

sd_eval_datafusion_fn <- function(fn_key, expr, expr_ctx) {
  # Evaluate arguments
  evaluated_args <- lapply(expr[-1], sd_eval_expr_inner, expr_ctx = expr_ctx)

  na_rm <- evaluated_args$na.rm
  evaluated_args$na.rm <- NULL

  if (any(rlang::have_name(evaluated_args))) {
    stop(
      sprintf(
        "Expected unnamed arguments to SedonaDB SQL function but got %s",
        paste(
          names(evaluated_args)[rlang::have_name(evaluated_args)],
          collapse = ", "
        )
      )
    )
  }

  sd_expr_any_function(fn_key, evaluated_args, na.rm = na_rm, factory = expr_ctx$factory)
}

sd_eval_translation <- function(fn_key, expr, expr_ctx) {
  # Replace the function with the translation in such a way that
  # any error resulting from the call doesn't have an absolute garbage error
  # stack trace
  new_fn_expr <- rlang::call2("$", expr_ctx$fns, rlang::sym(fn_key))

  # Evaluate arguments individually. We may need to allow translations to
  # override this step to have more control over the expression evaluation.
  evaluated_args <- lapply(expr[-1], sd_eval_expr_inner, expr_ctx = expr_ctx)

  # Recreate the call, injecting the context as the first argument
  new_call <- rlang::call2(new_fn_expr, expr_ctx, !!!evaluated_args)

  # ...and evaluate it
  sd_eval_default(new_call, expr_ctx)
}

sd_eval_default <- function(expr, expr_ctx) {
  rlang::eval_tidy(expr, data = expr_ctx$data, env = expr_ctx$env)
}

# Needed for sd_arrange(), as wrapping expression in desc() is how a descending
# sort order is specified. Unwraps desc(inner_expr) to separate the expressions.
unwrap_desc <- function(exprs) {
  inner_exprs <- vector("list", length(exprs))
  is_descending <- vector("logical", length(exprs))
  for (i in seq_along(exprs)) {
    expr <- exprs[[i]]

    if (rlang::is_call(expr, "desc") || rlang::is_call(expr, "desc", ns = "dplyr")) {
      inner_exprs[[i]] <- expr[[2]]
      is_descending[[i]] <- TRUE
    } else {
      inner_exprs[[i]] <- expr
      is_descending[[i]] <- FALSE
    }
  }

  list(inner_exprs = inner_exprs, is_descending = is_descending)
}

#' Expression evaluation context
#'
#' A context to use for evaluating a set of related R expressions into
#' SedonaDB expressions. One expression context may be used to translate
#' multiple expressions (e.g., all arguments to `mutate()`).
#'
#' @param schema A schema-like object coerced using
#'   [nanoarrow::as_nanoarrow_schema()]. This is used to create the data mask
#'   for expressions.
#' @param env The expression environment. This is needed to evaluate expressions.
#' @param ctx A SedonaDB context whose function registry should be used to resolve
#'   functions.
#'
#' @return An object of class sedonadb_expr_ctx
#' @noRd
sd_expr_ctx <- function(schema = NULL, env = parent.frame(), ctx = NULL) {
  if (is.null(schema)) {
    schema <- nanoarrow::na_struct()
  }

  schema <- nanoarrow::as_nanoarrow_schema(schema)
  data_names <- as.character(names(schema$children))
  data <- lapply(data_names, sd_expr_column)
  names(data) <- data_names

  structure(
    list(
      factory = sd_expr_factory(ctx = ctx),
      schema = schema,
      data = rlang::as_data_mask(data),
      env = env,
      fns = default_fns
    ),
    class = "sedonadb_expr_ctx"
  )
}

#' Register an R function translation into a SedonaDB expression
#'
#' @param qualified_name The name of the function in the form `pkg::fun` or
#'   `fun` if the package name is not relevant. This allows translations to
#'   support calls to `fun()` or `pkg::fun()` that appear in an R expression.
#' @param fn A function. The first argument must always be `.ctx`, which
#'   is the instance of `sd_expr_ctx()` that may be used to construct
#'   the required expressions (using `$factory`).
#'
#' @returns fn, invisibly
#' @noRd
sd_register_translation <- function(qualified_name, fn) {
  stopifnot(is.function(fn))

  pieces <- strsplit(qualified_name, "::")[[1]]
  unqualified_name <- pieces[[2]]

  default_fns[[qualified_name]] <- default_fns[[unqualified_name]] <- fn
  invisible(fn)
}

#' Register a translation that always forwards its arguments to DataFusion
#'
#' @param fn_name The name of the function
#' @returns fn, invisibly
#' @noRd
sd_register_datafusion_fn <- function(fn_name) {
  force(fn_name)

  fn <- function(.ctx, ...) {
    evaluated_args <- list(...)
    na_rm <- evaluated_args$na.rm
    evaluated_args$na.rm <- NULL

    if (any(rlang::have_name(evaluated_args))) {
      stop(
        sprintf(
          "Expected unnamed arguments to SedonaDB SQL function but got %s",
          paste(
            names(evaluated_args)[rlang::have_name(evaluated_args)],
            collapse = ", "
          )
        )
      )
    }

    sd_expr_any_function(
      fn_name,
      evaluated_args,
      na.rm = na_rm,
      factory = .ctx$factory
    )
  }

  default_fns[[fn_name]] <- fn
  invisible(fn)
}

default_fns <- new.env(parent = emptyenv())

# Register translations lazily because SQL users don't need them and because
# we need rlang for this and it is currently in Suggests
ensure_translations_registered <- function() {
  if (!is.null(default_fns$abs)) {
    return()
  }

  # Register default translations for our st_, sd_, and rs_ functions
  fn_names <- utils::.DollarNames(.fns, "^(st|rs|sd)_")
  for (fn_name in fn_names) {
    sd_register_datafusion_fn(fn_name)
  }

  sd_register_translation("base::abs", function(.ctx, x) {
    sd_expr_scalar_function("abs", list(x), factory = .ctx$factory)
  })

  # nolint start: object_name_linter
  sd_register_translation("base::sum", function(.ctx, x, ..., na.rm = FALSE) {
    sd_expr_aggregate_function("sum", list(x), na.rm = na.rm, factory = .ctx$factory)
  })
  # nolint end

  sd_register_translation("base::+", function(.ctx, lhs, rhs) {
    if (missing(rhs)) {
      # Use a double negative to ensure this fails for non-numeric types
      sd_expr_negative(
        sd_expr_negative(lhs, factory = .ctx$factory),
        factory = .ctx$factory
      )
    } else {
      sd_expr_binary("+", lhs, rhs, factory = .ctx$factory)
    }
  })

  sd_register_translation("base::-", function(.ctx, lhs, rhs) {
    if (missing(rhs)) {
      sd_expr_negative(lhs, factory = .ctx$factory)
    } else {
      sd_expr_binary("-", lhs, rhs, factory = .ctx$factory)
    }
  })

  for (op in c("==", "!=", ">", ">=", "<", "<=", "*", "/", "&", "|")) {
    sd_register_translation(
      paste0("base::", op),
      rlang::inject(function(.ctx, lhs, rhs) {
        sd_expr_binary(!!op, lhs, rhs, factory = .ctx$factory)
      })
    )
  }

  sd_register_translation("dplyr::n", function(.ctx) {
    sd_expr_aggregate_function("count", list(1L), na.rm = FALSE, factory = .ctx$factory)
  })
}

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
  ctx = NULL,
  x_qualifier = "x",
  y_qualifier = "y"
) {
  x_schema <- nanoarrow::as_nanoarrow_schema(x_schema)
  y_schema <- nanoarrow::as_nanoarrow_schema(y_schema)

  x_names <- as.character(names(x_schema$children))
  y_names <- as.character(names(y_schema$children))

  factory <- sd_expr_factory(ctx = ctx)

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
#' @param join_expr_ctx A `sedonadb_join_expr_ctx` from [sd_join_expr_ctx()]
#'
#' @returns A list of `SedonaDBExpr` objects representing the join conditions
#' @noRd
sd_eval_join_conditions <- function(join_by, join_expr_ctx) {
  ensure_translations_registered()

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
      evaluated_args <- lapply(expr[-1], sd_eval_join_expr_inner,
                               join_expr_ctx = join_expr_ctx, env = env)

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
