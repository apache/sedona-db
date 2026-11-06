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

#' Expand nested columns
#'
#' These generics delegate ordinary data frames to [tidyr::unnest_longer()]
#' and [tidyr::unnest_wider()]. SedonaDB methods use
#' [sedonadb::sd_unnest_longer()] and [sedonadb::sd_unnest_wider()].
#'
#' For SedonaDB, longer supports `data` and `col`; wider additionally supports
#' `names_sep`. Other arguments are unsupported. Multiple list columns are
#' expanded in parallel with null padding, rather than tidyr's recycling.
#' Empty and null lists are dropped. Wider accepts struct columns only.
#'
#' @param data A data frame or SedonaDB DataFrame.
#' @param col Columns to expand, evaluated using tidyselect.
#' @param ... Additional arguments passed to tidyr for ordinary data frames.
#' @returns A data frame or SedonaDB DataFrame, matching the input.
#' @md
#' @export
unnest_longer <- function(data, col, ...) {
  UseMethod("unnest_longer")
}

#' @export
unnest_longer.default <- function(data, col, ...) {
  tidyr::unnest_longer(data, {{ col }}, ...)
}

#' @export
unnest_longer.sedonadb_dataframe <- function(
  data,
  col,
  values_to = sdplyr_unsupported(),
  indices_to = sdplyr_unsupported(),
  indices_include = sdplyr_unsupported(),
  keep_empty = sdplyr_unsupported(),
  names_repair = sdplyr_unsupported(),
  simplify = sdplyr_unsupported(),
  ptype = sdplyr_unsupported(),
  transform = sdplyr_unsupported(),
  ...
) {
  rlang::check_dots_empty()
  assert_unsupported(
    values_to,
    indices_to,
    indices_include,
    keep_empty,
    names_repair,
    simplify,
    ptype,
    transform
  )
  sedonadb::sd_unnest_longer(data, {{ col }})
}

#' @rdname unnest_longer
#' @export
unnest_wider <- function(data, col, ...) {
  UseMethod("unnest_wider")
}

#' @export
unnest_wider.default <- function(data, col, ...) {
  tidyr::unnest_wider(data, {{ col }}, ...)
}

#' @export
unnest_wider.sedonadb_dataframe <- function(
  data,
  col,
  names_sep = NULL,
  simplify = sdplyr_unsupported(),
  strict = sdplyr_unsupported(),
  names_repair = sdplyr_unsupported(),
  ptype = sdplyr_unsupported(),
  transform = sdplyr_unsupported(),
  ...
) {
  rlang::check_dots_empty()
  assert_unsupported(simplify, strict, names_repair, ptype, transform)
  sedonadb::sd_unnest_wider(data, {{ col }}, names_sep = names_sep)
}
