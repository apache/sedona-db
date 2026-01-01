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

#' S3 Generic to create a SedonaDB literal expression
#'
#' This generic provides the opportunity for objects to register a mechanism
#' to be understood as literals in the context of a SedonaDB expression.
#' Users constructing expressions directly should use [sd_expr_literal()].
#'
#' @param x An object to convert to a SedonaDB literal
#' @param ... Passed to/from methods
#' @param type An optional data type to request for the output
#' @param factory An `sd_expr_factory()` that should be passed to any
#'   other calls to `as_sedonadb_literal()` if needed
#'
#' @returns An object of class SedonaDBExpr
#' @export
#'
#' @examples
#' as_sedonadb_literal("abcd")
#'
as_sedonadb_literal <- function(x, ..., type = NULL, factory = NULL) {
  UseMethod("as_sedonadb_literal")
}

#' @export
as_sedonadb_literal.NULL <- function(x, ..., type = NULL) {
  na <- nanoarrow::nanoarrow_array_init(nanoarrow::na_na()) |>
    nanoarrow::nanoarrow_array_modify(list(length = 1L, null_count = 1L))
  as_sedonadb_literal_from_nanoarrow(na, ..., type = type)
}

#' @export
as_sedonadb_literal.character <- function(x, ..., type = NULL) {
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
  array <- nanoarrow::as_nanoarrow_array(x)
  if (array$length != 1L) {
    stop("Can't convert non-scalar to sedonadb_expr")
  }

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
