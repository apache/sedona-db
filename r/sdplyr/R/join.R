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

#' @exportS3Method dplyr::left_join
left_join.sedonadb_dataframe <- function(
  x,
  y,
  by = NULL,
  copy = FALSE,
  suffix = c(".x", ".y"),
  ...,
  keep = NULL
) {
  rlang::check_dots_empty()
  sedonadb::sd_left_join(x, y, by = by, keep = keep)
}

#' @exportS3Method dplyr::right_join
right_join.sedonadb_dataframe <- function(
  x,
  y,
  by = NULL,
  copy = FALSE,
  suffix = c(".x", ".y"),
  ...,
  keep = NULL
) {
  rlang::check_dots_empty()
  sedonadb::sd_right_join(x, y, by = by, keep = keep)
}

#' @exportS3Method dplyr::inner_join
inner_join.sedonadb_dataframe <- function(
  x,
  y,
  by = NULL,
  copy = FALSE,
  suffix = c(".x", ".y"),
  ...,
  keep = NULL
) {
  rlang::check_dots_empty()
  sedonadb::sd_inner_join(x, y, by = by, keep = keep)
}

#' @exportS3Method dplyr::full_join
full_join.sedonadb_dataframe <- function(
  x,
  y,
  by = NULL,
  copy = FALSE,
  suffix = c(".x", ".y"),
  ...,
  keep = NULL
) {
  rlang::check_dots_empty()
  sedonadb::sd_full_join(x, y, by = by, keep = keep)
}

#' @exportS3Method dplyr::semi_join
semi_join.sedonadb_dataframe <- function(x, y, by = NULL, copy = FALSE, ...) {
  rlang::check_dots_empty()
  sedonadb::sd_semi_join(x, y, by = by)
}

#' @exportS3Method dplyr::anti_join
anti_join.sedonadb_dataframe <- function(x, y, by = NULL, copy = FALSE, ...) {
  rlang::check_dots_empty()
  sedonadb::sd_anti_join(x, y, by = by)
}

#' @exportS3Method dplyr::cross_join
cross_join.sedonadb_dataframe <- function(
  x,
  y,
  copy = FALSE,
  suffix = c(".x", ".y"),
  ...
) {
  rlang::check_dots_empty()
  sedonadb::sd_cross_join(x, y)
}
