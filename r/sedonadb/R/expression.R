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
sd_expr_scalar_function <- function(function_name, args, factory = sd_expr_factory()) {
  args_as_expr <- lapply(args, as_sd_expr, factory = factory)
  # Not sure why we need this exactly (something about savvy)
  args_as_expr_ptr <- lapply(args_as_expr, "[[", ".ptr")
  factory$scalar_function(function_name, args_as_expr_ptr)
}

#' @rdname sd_expr_column
#' @export
sd_expr_aggregate_function <- function(function_name, args, ...,
                                       na.rm = FALSE, distinct = FALSE, factory = sd_expr_factory()) {
  args_as_expr <- lapply(args, as_sd_expr, factory = factory)
  # Not sure why we need this exactly (something about savvy)
  args_as_expr_ptr <- lapply(args_as_expr, "[[", ".ptr")
  factory$aggregate_function(function_name, args_as_expr_ptr, na_rm = na.rm, distinct = distinct)
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
sd_expr_factory <- function() {
  SedonaDBExprFactory$new(ctx())
}

#' @export
print.SedonaDBExpr <- function(x, ...) {
  cat("<SedonaDBExpr>\n")
  cat(x$display())
  cat("\n")
  invisible(x)
}

#' Evaluate an R expression into a SedonaDB expression
#'
#' @param expr An R expression (e.g., the result of `quote()`).
#' @param expr_ctx An `sd_expr_ctx()`
#'
#' @returns A `SedonaDBExpr`
#' @noRd
sd_eval_expr <- function(expr, expr_ctx = sd_expr_ctx(env = env), env = parent.frame()) {
  ensure_translations_registered()

  rlang::try_fetch({
    result <- sd_eval_expr_inner(expr, expr_ctx)
    as_sd_expr(result, factory = factory)
  }, error = function(e) {
    rlang::abort(
      sprintf("Error evaluating translated expression %s", rlang::expr_label(expr)),
      parent = e
    )
  })
}

sd_eval_expr_inner <- function(expr, expr_ctx) {
  if (rlang::is_call(expr)) {
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
#'
#' @return An object of class sedonadb_expr_ctx
#' @noRd
sd_expr_ctx <- function(schema = NULL, env = parent.frame()) {
  if (is.null(schema)) {
    schema <- nanoarrow::na_struct()
  }

  schema <- nanoarrow::as_nanoarrow_schema(schema)
  data_names <- as.character(names(schema$children))
  data <- lapply(data_names, sd_expr_column)
  names(data) <- data_names

  structure(
    list(
      factory = sd_expr_factory(),
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

default_fns <- new.env(parent = emptyenv())

# Register translations lazily because SQL users don't need them and because
# we need rlang for this and it is currently in Suggests
ensure_translations_registered <- function() {
  if (!is.null(default_fns$abs)) {
    return()
  }

  sd_register_translation("base::abs", function(.ctx, x) {
    sd_expr_scalar_function("abs", list(x), factory = .ctx$factory)
  })

  sd_register_translation("base::sum", function(.ctx, x, ..., na.rm = FALSE) {
    sd_expr_aggregate_function("sum", list(x), na.rm = na.rm, factory = .ctx$factory)
  })

  sd_register_translation("base::+", function(.ctx, lhs, rhs) {
    if (missing(rhs)) {
      # Use a double negative to ensure this fails for non-numeric types
      sd_expr_negative(sd_expr_negative(lhs, factory = .ctx$factory), factory = .ctx$factory)
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
    sd_register_translation(paste0("base::", op), rlang::inject(function(.ctx, lhs, rhs) {
      sd_expr_binary(!!op, lhs, rhs, factory = .ctx$factory)
    }))
  }
}
