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

#' Create a SedonaDB Logical Expression
#'
#' @param x An object
#' @param ... Passed to/from methods
#' @param type An optional data type to request for the output
#'
#' @returns An object of class SedonaDBExpr
#' @export
as_sedonadb_expr <- function(x, ..., type = NULL) {
  UseMethod("as_sedonadb_expr")
}

#' @export
as_sedonadb_expr.SedonaDBExpr <- function(x, ..., type = NULL) {
  handle_type_request(x, type)
}

#' @export
as_sedonadb_expr.character <- function(x, ..., type = NULL) {
  as_sedonadb_expr_from_nanoarrow(x, ..., type = type)
}

#' @export
as_sedonadb_expr.integer <- function(x, ..., type = NULL) {
  as_sedonadb_expr_from_nanoarrow(x, ..., type = type)
}

#' @export
as_sedonadb_expr.double <- function(x, ..., type = NULL) {
  as_sedonadb_expr_from_nanoarrow(x, ..., type = type)
}

#' @export
as_sedonadb_expr.raw <- function(x, ..., type = NULL) {
  as_sedonadb_expr_from_nanoarrow(list(x), ..., type = type)
}

as_sedonadb_expr_from_nanoarrow <- function(x, ..., type = NULL) {
  if (length(x) != 1 || is.object(x)) {
    stop("Can't convert non-scalar to sedonadb_expr")
  }

  array <- nanoarrow::as_nanoarrow_array(x)
  as_sedonadb_expr(array, type = type)
}

#' @export
as_sedonadb_expr.nanoarrow_array <- function(x, ..., type = NULL) {
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
