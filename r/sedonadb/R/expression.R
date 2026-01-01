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

#' Create a SedonaDB Literal Expression
#'
#' @param x An object to convert to a SedonaDB literal
#' @param ... Passed to/from methods
#' @param type An optional data type to request for the output
#' @param factory An expression factory object that should be passed to any
#'   other calls to `as_sedonadb_literal()`.
#'
#' @returns An object of class SedonaDBExpr
#' @export
as_sedonadb_literal <- function(x, ..., type = NULL, factory = NULL) {
  UseMethod("as_sedonadb_literal")
}

#' @export
as_sedonadb_literal.SedonaDBExpr <- function(x, ..., type = NULL) {
  handle_type_request(x, type)
}

#' @export
as_sedonadb_literal.character <- function(x, ..., type = NULL, factory = NULL) {
  as_sedonadb_literal_from_nanoarrow(x, ..., type = type)
}

#' @export
as_sedonadb_literal.integer <- function(x, ..., type = NULL) {
  as_sedonadb_literal_from_nanoarrow(x, ..., type = type)
}

#' @export
as_sedonadb_literal.double <- function(x, ..., type = NULL) {
  as_sedonadb_literal_from_nanoarrow(x, ..., type = type)
}

#' @export
as_sedonadb_literal.raw <- function(x, ..., type = NULL) {
  as_sedonadb_literal_from_nanoarrow(list(x), ..., type = type)
}

as_sedonadb_literal_from_nanoarrow <- function(x, ..., type = NULL) {
  if (length(x) != 1 || is.object(x)) {
    stop("Can't convert non-scalar to sedonadb_expr")
  }

  array <- nanoarrow::as_nanoarrow_array(x)
  as_sedonadb_literal(array, type = type)
}

#' @export
as_sedonadb_literal.nanoarrow_array <- function(x, ..., type = NULL) {
  schema <- nanoarrow::infer_nanoarrow_schema(x)

  array_export <- nanoarrow::nanoarrow_allocate_array()
  nanoarrow::nanoarrow_pointer_export(x, array_export)

  expr <- SedonaDBExprFactory$literal(array_export, schema)
  handle_type_request(expr, type)
}

handle_type_request <- function(x, type) {
  if (!is.null(type)) {
    x$cast(nanoarrow::as_nanoarrow_schema(x))
  } else {
    x
  }
}

#' @export
print.SedonaDBExpr <- function(x, ...) {
  cat("<SedonaDBExpr>\n")
  cat(x$debug_string())
  cat("\n")
  invisible(x)
}

#' Expression evaluation context
#'
#' A context to use for evaluating a set of related R expressions into
#' SedonaDB expressions.
#'
#' @param schema A schema-like object coerced using
#'   [nanoarrow::as_nanoarrow_schema()].
#'
#' @return An object of class sedonadb_expr_ctx
#' @noRd
sd_expr_ctx <- function(schema = NULL, env = parent.frame()) {
  if (is.null(schema)) {
    schema <- nanoarrow::na_struct()
  }

  schema <- nanoarrow::as_nanoarrow_schema(schema)
  data_names <- as.character(names(schema$children))
  data <- lapply(data_names, SedonaDBExprFactory$column)
  names(data) <- data_names

  structure(
    list(
      factory = SedonaDBExprFactory$new(ctx()),
      schema = schema,
      data = rlang::as_data_mask(data),
      data_names = data_names,
      env = env,
      fns = default_fns
    ),
    class = "sedonadb_expr_ctx"
  )
}


#' Evaluate an R expression into a SedonaDB expression
#'
#' @param expr An R expression (e.g., the result of `quote()`).
#' @param expr_ctx An `sd_expr_ctx()`
#'
#' @returns A `SedonaDBExpr`
#' @noRd
sd_eval_expr <- function(expr, expr_ctx = sd_expr_ctx()) {
  if (rlang::is_call(expr)) {
    # If there is no expression anywhere in this call, just evaluate it in R
    # and move on.
    if (!r_expr_contains_sedonadb_expr(expr, expr_ctx)) {
      return(sd_eval_default(expr, expr_ctx))
    }

    # Handle `pkg::fun` or `fun`
    call_name <- rlang::call_name(expr)
    if (!is.null(call_name) && !is.null(expr_ctx$fns[[call_name]])) {
      return(sd_eval_translation(call_name, expr, expr_ctx))
    } else {
      # Otherwise we have an inlined function and we just have to evaluate
      return(sd_eval_default(expr, expr_ctx))
    }
  }

  sd_eval_default(expr, expr_ctx)
}

sd_eval_translation <- function(fn_key, expr, expr_ctx) {
  # Replace the function with the translation in such a way that
  # any error resulting from the call doesn't have an absolute garbage error
  # stack trace
  new_fn_expr <- rlang::call2("$", expr_ctx$fns, rlang::sym(fn_key))

  # Evaluate arguments individually. We may need to allow translations to
  # override this step to have more control over the expression evaluation.
  evaluated_args <- lapply(expr[-1], sd_eval_expr, expr_ctx = expr_ctx)

  # Recreate the call, injecting the factory as the first argument
  new_call <- rlang::call2(new_fn_expr, expr_ctx$factory, !!!evaluated_args)

  # ...and evaluate it
  sd_eval_default(new_call, expr_ctx)
}

sd_eval_default <- function(expr, expr_ctx) {
  rlang::try_fetch({
    r_result <- rlang::eval_tidy(expr, data = expr_ctx$data, env = expr_ctx$env)
    as_sedonadb_literal(r_result)
  }, error = function(e) {
    rlang::abort(
      "SedonaDB R evaluation error",
      parent = e
    )
  })
}

r_expr_contains_sedonadb_expr <- function(expr, expr_ctx) {
  if (rlang::is_call(expr, c("$", "[[")) && rlang::is_symbol(expr[[1]], ".data")) {
    # An attempt to access the .data pronoun will either error or return an
    # SedonaDB expression
    TRUE
  } else if (rlang::is_symbol(expr, expr_ctx$data_names)) {
    TRUE
  } else if (rlang::is_call(expr)) {
    for (i in seq_along(expr)) {
      if (r_expr_contains_sedonadb_expr(expr[[i]], expr_ctx)) {
        return(TRUE)
      }
    }

    FALSE
  } else if(rlang::is_atomic(expr)) {
    inherits(expr, "sedonadb_expr")
  } else {
    FALSE
  }
}

#' Register an R function translation into a SedonaDB expression
#'
#' @param qualified_name The name of the function in the form `pkg::fun` or
#'   `fun` if the package name is not relevant. This allows translations to
#'   support calls to `fun()` or `pkg::fun()` that appear in an R expression.
#' @param fn A function. The first argument should always be `.factory`, which
#'   is the instance of `SedonaDBExprFactory` that may be used to construct
#'   the required expressions.
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

sd_register_translation("base::abs", function(.factory, x) {
  # Not sure why I need $.ptr here
  .factory$scalar_function("abs", list(x$.ptr))
})
