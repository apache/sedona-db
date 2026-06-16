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

#' @exportS3Method dplyr::group_by
group_by.sedonadb_dataframe <- function(
  .data,
  ...,
  .add = FALSE,
  .drop = sdplyr_unsupported()
) {
  exprs <- rlang::enquos(...)

  if (.add) {
    existing_groups <- unclass(.data)$group_by
    sedonadb::sd_group_by(.data, !!!existing_groups, !!!exprs)
  } else {
    sedonadb::sd_group_by(.data, !!!exprs)
  }
}

#' @exportS3Method dplyr::ungroup
ungroup.sedonadb_dataframe <- function(.data, ...) {
  rlang::check_dots_empty()
  sedonadb::sd_ungroup(.data)
}

#' @exportS3Method dplyr::summarise
summarise.sedonadb_dataframe <- function(.data, ..., .by = NULL, .groups = NULL) {
  exprs <- rlang::enquos(...)
  .by <- rlang::enquo(.by)

  if (!is.null(.groups) && !rlang::quo_is_null(.by)) {
    rlang::abort("Can't supply both .groups and .by")
  } else if (is.null(.groups) && rlang::quo_is_null(.by)) {
    rlang::inform(
      c(
        "In sdplyr, groups are dropped by default on summarise",
        "i" = 'Use .groups = "drop" to suppress this message'
      )
    )
  } else if (!identical(.groups, "drop") && rlang::quo_is_null(.by)) {
    rlang::abort('sdplyr::summarise() only supports .groups = "drop"')
  }

  .data <- if (rlang::quo_is_null(.by)) {
    .data
  } else if (rlang::quo_is_call(.by, "c")) {
    by_args <- rlang::call_args(rlang::quo_get_expr(.by))
    sedonadb::sd_group_by(.data, !!!by_args)
  } else {
    sedonadb::sd_group_by(.data, !!.by)
  }

  sedonadb::sd_summarise(.data, !!!exprs)
}
