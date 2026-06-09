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

# nolint start: object_name_linter

left_join.sedonadb_dataframe <- function(
  x,
  y,
  by = NULL,
  copy = FALSE,
  suffix = c(".x", ".y"),
  ...,
  keep = NULL
) {
  sedonadb::sd_left_join(x, y, by = by, keep = keep)
}

right_join.sedonadb_dataframe <- function(
  x,
  y,
  by = NULL,
  copy = FALSE,
  suffix = c(".x", ".y"),
  ...,
  keep = NULL
) {
  sedonadb::sd_right_join(x, y, by = by, keep = keep)
}

inner_join.sedonadb_dataframe <- function(
  x,
  y,
  by = NULL,
  copy = FALSE,
  suffix = c(".x", ".y"),
  ...,
  keep = NULL
) {
  sedonadb::sd_inner_join(x, y, by = by, keep = keep)
}

full_join.sedonadb_dataframe <- function(
  x,
  y,
  by = NULL,
  copy = FALSE,
  suffix = c(".x", ".y"),
  ...,
  keep = NULL
) {
  sedonadb::sd_full_join(x, y, by = by, keep = keep)
}

semi_join.sedonadb_dataframe <- function(x, y, by = NULL, copy = FALSE, ...) {
  sedonadb::sd_semi_join(x, y, by = by)
}

anti_join.sedonadb_dataframe <- function(x, y, by = NULL, copy = FALSE, ...) {
  sedonadb::sd_anti_join(x, y, by = by)
}

cross_join.sedonadb_dataframe <- function(
  x,
  y,
  copy = FALSE,
  suffix = c(".x", ".y"),
  ...
) {
  sedonadb::sd_cross_join(x, y)
}

# nolint end
